import Mathlib

namespace NUMINAMATH_CALUDE_hearts_then_king_probability_l1458_145844

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)
  (suits : ∀ c ∈ cards, c.1 ∈ Finset.range 4)
  (ranks : ∀ c ∈ cards, c.2 ∈ Finset.range 13)

/-- The probability of drawing a specific sequence of cards from a shuffled deck -/
def draw_probability (d : Deck) (seq : List (Nat × Nat)) : ℚ :=
  sorry

/-- Hearts suit is represented by 0 -/
def hearts : Nat := 0

/-- King rank is represented by 12 (0-indexed) -/
def king : Nat := 12

theorem hearts_then_king_probability :
  ∀ d : Deck, 
    draw_probability d [(hearts, 0), (hearts, 1), (hearts, 2), (hearts, 3), (0, king)] = 286 / 124900 := by
  sorry

end NUMINAMATH_CALUDE_hearts_then_king_probability_l1458_145844


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l1458_145884

theorem sqrt_difference_comparison (m : ℝ) (hm : m > 1) :
  Real.sqrt m - Real.sqrt (m - 1) > Real.sqrt (m + 1) - Real.sqrt m := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l1458_145884


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_l1458_145823

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^6 - 3 * x + 9) + 5 * (x^7 - 3 * x^3 + 2 * x^2 - 4)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_equals_value_at_one : p 1 = 32 := by sorry

/-- The sum of coefficients of the polynomial p is 32 -/
theorem sum_of_coefficients : ∃ (c : ℝ), p c = 32 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_value_at_one_sum_of_coefficients_l1458_145823


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l1458_145841

theorem trig_expression_equals_four : 
  (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) / 
  ((4 * (Real.cos (10 * π / 180))^2 - 2) * Real.sin (10 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l1458_145841


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1458_145801

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℚ) * (55 / 100) - (total_votes : ℚ) * (30 / 100) = 174 →
  total_votes = 696 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1458_145801


namespace NUMINAMATH_CALUDE_log_x2y2_l1458_145826

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16/11 := by
sorry

end NUMINAMATH_CALUDE_log_x2y2_l1458_145826


namespace NUMINAMATH_CALUDE_water_jars_count_l1458_145886

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 4

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 7

/-- Represents the volume of water in quarts -/
def water_in_quarts : ℕ := total_water * 4

theorem water_jars_count :
  num_jars * 3 = 12 ∧
  num_jars * (1 + 2 + 4) = water_in_quarts :=
by sorry

#check water_jars_count

end NUMINAMATH_CALUDE_water_jars_count_l1458_145886


namespace NUMINAMATH_CALUDE_fraction_calculation_l1458_145817

theorem fraction_calculation : (1/4 + 3/8 - 7/12) / (1/24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1458_145817


namespace NUMINAMATH_CALUDE_faucet_filling_time_faucet_filling_time_is_135_l1458_145812

/-- If three faucets can fill a 100-gallon tub in 6 minutes, 
    then four faucets will fill a 50-gallon tub in 135 seconds. -/
theorem faucet_filling_time : ℝ → Prop :=
  fun time_seconds =>
    let three_faucet_volume : ℝ := 100  -- gallons
    let three_faucet_time : ℝ := 6    -- minutes
    let four_faucet_volume : ℝ := 50   -- gallons
    
    let one_faucet_rate : ℝ := three_faucet_volume / (3 * three_faucet_time)
    let four_faucet_rate : ℝ := 4 * one_faucet_rate
    
    time_seconds = (four_faucet_volume / four_faucet_rate) * 60

theorem faucet_filling_time_is_135 : faucet_filling_time 135 := by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_faucet_filling_time_is_135_l1458_145812


namespace NUMINAMATH_CALUDE_triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l1458_145802

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ :=
  sorry

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  sorry

-- Function to find the symmetric point
def symmetricPoint (p : Point) (l : Line) : Point :=
  sorry

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line) : Prop :=
  sorry

-- Theorem 1
theorem triangle_area_is_two :
  let l : Line := { a := 1, b := -1, c := -2 }
  let t : Triangle := { p1 := { x := 0, y := 0 }, p2 := { x := 2, y := 0 }, p3 := { x := 0, y := -2 } }
  triangleArea t = 2 :=
sorry

-- Theorem 2
theorem symmetric_point_correct :
  let l : Line := { a := -1, b := 1, c := 1 }
  let p : Point := { x := 0, y := 2 }
  symmetricPoint p l = { x := 1, y := 1 } :=
sorry

-- Theorem 3
theorem line_equal_intercepts :
  let l : Line := { a := 1, b := 1, c := -2 }
  pointOnLine { x := 1, y := 1 } l ∧ hasEqualIntercepts l :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_symmetric_point_correct_line_equal_intercepts_l1458_145802


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1458_145828

theorem smallest_n_for_terminating_decimal : 
  ∃ (n : ℕ+), n = 24 ∧ 
  (∀ (m : ℕ+), m < n → ¬(∃ (a b : ℕ), (m : ℚ) / (m + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5))) ∧
  (∃ (a b : ℕ), (n : ℚ) / (n + 101 : ℚ) = (a : ℚ) / (b : ℚ) ∧ b ≠ 0 ∧ 
  (∀ (p : ℕ), Prime p → p ∣ b → p = 2 ∨ p = 5)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1458_145828


namespace NUMINAMATH_CALUDE_intersection_equality_l1458_145807

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x) ∧ 1 - x > 0}
def B (m : ℝ) : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + m}

-- State the theorem
theorem intersection_equality (m : ℝ) : A ∩ B m = A ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_l1458_145807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1458_145815

/-- Given an arithmetic sequence {aₙ}, prove that S₂₀₁₀ = 1005 under the given conditions -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) → -- Definition of Sₙ
  (∃ O A B C : ℝ × ℝ, 
    B - O = a 1005 • (A - O) + a 1006 • (C - O) ∧ -- Vector equation
    ∃ t : ℝ, B = t • A + (1 - t) • C ∧ -- Collinearity condition
    t ≠ 0 ∧ t ≠ 1) → -- Line doesn't pass through O
  S 2010 = 1005 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1458_145815


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1458_145880

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1458_145880


namespace NUMINAMATH_CALUDE_youseff_walk_time_l1458_145875

/-- The number of blocks from Youseff's home to his office -/
def distance : ℕ := 12

/-- The time in seconds it takes Youseff to ride his bike one block -/
def bike_time : ℕ := 20

/-- The additional time in minutes it takes Youseff to walk compared to biking -/
def additional_time : ℕ := 8

/-- The time in seconds it takes Youseff to walk one block -/
def walk_time : ℕ := sorry

theorem youseff_walk_time :
  walk_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_youseff_walk_time_l1458_145875


namespace NUMINAMATH_CALUDE_faulty_odometer_distance_l1458_145876

/-- Represents an odometer that skips certain digits --/
structure SkippingOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Calculates the actual distance traveled given a skipping odometer --/
def actual_distance (o : SkippingOdometer) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem faulty_odometer_distance :
  let o : SkippingOdometer := { reading := 3509, skipped_digits := [4, 6] }
  actual_distance o = 2964 :=
sorry

end NUMINAMATH_CALUDE_faulty_odometer_distance_l1458_145876


namespace NUMINAMATH_CALUDE_power_multiplication_specific_power_multiplication_l1458_145836

theorem power_multiplication (a b c : ℕ) : (10 : ℕ) ^ a * (10 : ℕ) ^ b = (10 : ℕ) ^ (a + b) := by
  sorry

theorem specific_power_multiplication : (10 : ℕ) ^ 65 * (10 : ℕ) ^ 64 = (10 : ℕ) ^ 129 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_specific_power_multiplication_l1458_145836


namespace NUMINAMATH_CALUDE_solve_quadratic_coefficients_l1458_145813

-- Define the universal set U
def U : Set ℤ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℤ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the theorem
theorem solve_quadratic_coefficients :
  ∀ b c : ℤ, (U \ A b c = {2}) → (b = -8 ∧ c = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_coefficients_l1458_145813


namespace NUMINAMATH_CALUDE_largest_n_for_equation_solution_exists_for_two_l1458_145849

theorem largest_n_for_equation : 
  ∀ n : ℕ+, n > 2 → 
  ¬∃ x y z : ℕ+, n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

theorem solution_exists_for_two :
  ∃ x y z : ℕ+, 2^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_solution_exists_for_two_l1458_145849


namespace NUMINAMATH_CALUDE_nicole_cookies_l1458_145842

theorem nicole_cookies (N : ℚ) : 
  (((1 - N) * (1 - 3/5)) = 6/25) → N = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_nicole_cookies_l1458_145842


namespace NUMINAMATH_CALUDE_perpendicular_implies_perpendicular_lines_l1458_145829

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_implies_perpendicular_lines 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  parallel α β → perpendicularLines l m :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_perpendicular_lines_l1458_145829


namespace NUMINAMATH_CALUDE_max_cutlery_sets_l1458_145895

theorem max_cutlery_sets (dinner_forks knives soup_spoons teaspoons dessert_forks butter_knives : ℕ) 
  (max_capacity : ℕ) (dinner_fork_weight knife_weight soup_spoon_weight teaspoon_weight dessert_fork_weight butter_knife_weight : ℕ) : 
  dinner_forks = 6 →
  knives = dinner_forks + 9 →
  soup_spoons = 2 * knives →
  teaspoons = dinner_forks / 2 →
  dessert_forks = teaspoons / 3 →
  butter_knives = 2 * dessert_forks →
  max_capacity = 20000 →
  dinner_fork_weight = 80 →
  knife_weight = 100 →
  soup_spoon_weight = 85 →
  teaspoon_weight = 50 →
  dessert_fork_weight = 70 →
  butter_knife_weight = 65 →
  (max_capacity - (dinner_forks * dinner_fork_weight + knives * knife_weight + 
    soup_spoons * soup_spoon_weight + teaspoons * teaspoon_weight + 
    dessert_forks * dessert_fork_weight + butter_knives * butter_knife_weight)) / 
    (dinner_fork_weight + knife_weight) = 84 := by
  sorry

end NUMINAMATH_CALUDE_max_cutlery_sets_l1458_145895


namespace NUMINAMATH_CALUDE_remainder_3125_div_98_l1458_145855

theorem remainder_3125_div_98 : 3125 % 98 = 87 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3125_div_98_l1458_145855


namespace NUMINAMATH_CALUDE_arrangement_problem_l1458_145824

def boys : ℕ := 2
def girls : ℕ := 3
def total : ℕ := boys + girls

-- Helper function for permutations
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

theorem arrangement_problem :
  -- (1)
  permutations total 4 = 120 ∧
  -- (2)
  permutations total 4 = 120 ∧
  -- (3)
  (permutations 3 3) * (permutations 3 3) = 36 ∧
  -- (4)
  (permutations 3 3) * (permutations 4 2) = 72 ∧
  -- (5)
  3 * (permutations 4 4) = 72 ∧
  -- (6)
  Nat.factorial total - 2 * (Nat.factorial 4) + (Nat.factorial 3) = 78 ∧
  -- (7)
  permutations total 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_problem_l1458_145824


namespace NUMINAMATH_CALUDE_car_travel_distance_l1458_145897

/-- Represents the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

theorem car_travel_distance :
  let initial_distance : ℚ := 3
  let initial_time : ℚ := 4
  let total_time : ℚ := 120
  distance_traveled (initial_distance / initial_time) total_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l1458_145897


namespace NUMINAMATH_CALUDE_llama_accessible_area_l1458_145839

/-- Represents a rectangular shed -/
structure Shed :=
  (length : ℝ)
  (width : ℝ)

/-- Calculates the area accessible to a llama tied to the corner of a shed -/
def accessible_area (s : Shed) (leash_length : ℝ) : ℝ :=
  -- Definition to be filled
  sorry

/-- Theorem stating the accessible area for a llama tied to a 2m by 4m shed with a 4m leash -/
theorem llama_accessible_area :
  let s : Shed := ⟨4, 2⟩
  let leash_length : ℝ := 4
  accessible_area s leash_length = 13 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_llama_accessible_area_l1458_145839


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l1458_145854

/-- The 'Twirly Tea Cups' ride problem -/
theorem twirly_tea_cups_capacity 
  (total_capacity : ℕ) 
  (num_teacups : ℕ) 
  (h1 : total_capacity = 63) 
  (h2 : num_teacups = 7) : 
  total_capacity / num_teacups = 9 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l1458_145854


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l1458_145879

/-- A geometric sequence with a_3 = 2 and a_7 = 8 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ (∀ n, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 7 = 8

/-- Theorem: In a geometric sequence where a_3 = 2 and a_7 = 8, a_5 = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l1458_145879


namespace NUMINAMATH_CALUDE_sum_of_two_5cm_cubes_volume_l1458_145861

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s^3

/-- The sum of volumes of two cubes with edge length s -/
def sum_of_two_cube_volumes (s : ℝ) : ℝ := 2 * cube_volume s

theorem sum_of_two_5cm_cubes_volume :
  sum_of_two_cube_volumes 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_5cm_cubes_volume_l1458_145861


namespace NUMINAMATH_CALUDE_square_difference_square_sum_mental_math_strategy_l1458_145888

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) :=
  sorry

theorem square_sum (n : ℕ) : (n + 1)^2 = n^2 + (2*n + 1) :=
  sorry

theorem mental_math_strategy :
  49^2 = 50^2 - 99 ∧ 51^2 = 50^2 + 101 :=
by
  have h1 : 49^2 = 50^2 - 99 := by
    calc
      49^2 = (50 - 1)^2 := by rfl
      _ = 50^2 - (2*50 - 1) := by apply square_difference
      _ = 50^2 - 99 := by ring
  
  have h2 : 51^2 = 50^2 + 101 := by
    calc
      51^2 = (50 + 1)^2 := by rfl
      _ = 50^2 + (2*50 + 1) := by apply square_sum
      _ = 50^2 + 101 := by ring
  
  exact ⟨h1, h2⟩

#check mental_math_strategy

end NUMINAMATH_CALUDE_square_difference_square_sum_mental_math_strategy_l1458_145888


namespace NUMINAMATH_CALUDE_chocolate_chip_cookies_l1458_145850

/-- The number of cookies in each bag -/
def cookies_per_bag : ℕ := 9

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := 41

/-- The number of baggies that can be made with all cookies -/
def total_baggies : ℕ := 6

/-- The theorem stating the number of chocolate chip cookies -/
theorem chocolate_chip_cookies : 
  cookies_per_bag * total_baggies - oatmeal_cookies = 13 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookies_l1458_145850


namespace NUMINAMATH_CALUDE_probability_of_sum_magnitude_at_least_sqrt2_l1458_145891

/-- The roots of z^12 - 1 = 0 -/
def twelfthRootsOfUnity : Finset ℂ := sorry

/-- The condition that v and w are distinct -/
def areDistinct (v w : ℂ) : Prop := v ≠ w

/-- The condition that v and w are roots of z^12 - 1 = 0 -/
def areRoots (v w : ℂ) : Prop := v ∈ twelfthRootsOfUnity ∧ w ∈ twelfthRootsOfUnity

/-- The number of pairs (v, w) satisfying |v + w| ≥ √2 -/
def satisfyingPairs : ℕ := sorry

/-- The total number of distinct pairs (v, w) -/
def totalPairs : ℕ := sorry

theorem probability_of_sum_magnitude_at_least_sqrt2 :
  satisfyingPairs / totalPairs = 10 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_magnitude_at_least_sqrt2_l1458_145891


namespace NUMINAMATH_CALUDE_triangle_shape_l1458_145832

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def has_equal_roots (t : Triangle) : Prop :=
  ∃ x : ℝ, t.b * (x^2 + 1) + t.c * (x^2 - 1) - 2 * t.a * x = 0 ∧
  ∀ y : ℝ, t.b * (y^2 + 1) + t.c * (y^2 - 1) - 2 * t.a * y = 0 → y = x

def angle_condition (t : Triangle) : Prop :=
  Real.sin t.C * Real.cos t.A - Real.cos t.C * Real.sin t.A = 0

-- Define an isosceles right-angled triangle
def is_isosceles_right_angled (t : Triangle) : Prop :=
  t.a = t.b ∧ t.A = t.B ∧ t.C = Real.pi / 2

-- State the theorem
theorem triangle_shape (t : Triangle) :
  has_equal_roots t → angle_condition t → is_isosceles_right_angled t := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1458_145832


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1458_145833

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 < x + 6 ↔ -2 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1458_145833


namespace NUMINAMATH_CALUDE_a_minus_b_value_l1458_145848

theorem a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 2)
  (hab : |a + b| = -(a + b)) :
  a - b = -2 ∨ a - b = -6 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l1458_145848


namespace NUMINAMATH_CALUDE_series_sum_equals_eleven_twentieths_l1458_145899

theorem series_sum_equals_eleven_twentieths : 
  (1 / 3 : ℚ) + (1 / 5 : ℚ) + (1 / 7 : ℚ) + (1 / 9 : ℚ) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_eleven_twentieths_l1458_145899


namespace NUMINAMATH_CALUDE_pucks_not_in_original_position_l1458_145890

/-- Represents the arrangement of three pucks -/
inductive Arrangement
  | ABC
  | ACB
  | BAC
  | BCA
  | CAB
  | CBA

/-- Represents a single swap operation -/
def swap : Arrangement → Arrangement
  | Arrangement.ABC => Arrangement.BAC
  | Arrangement.ACB => Arrangement.CAB
  | Arrangement.BAC => Arrangement.ABC
  | Arrangement.BCA => Arrangement.CBA
  | Arrangement.CAB => Arrangement.ACB
  | Arrangement.CBA => Arrangement.BCA

/-- Applies n swaps to the initial arrangement -/
def applySwaps (n : Nat) (init : Arrangement) : Arrangement :=
  match n with
  | 0 => init
  | n + 1 => swap (applySwaps n init)

/-- Theorem stating that after 25 swaps, the arrangement cannot be the same as the initial one -/
theorem pucks_not_in_original_position (init : Arrangement) : 
  applySwaps 25 init ≠ init :=
sorry

end NUMINAMATH_CALUDE_pucks_not_in_original_position_l1458_145890


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1458_145808

/-- A line with slope 6 passing through (4, -3) and intersecting y = -x + 1 has m + b = -21 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 6 ∧ 
  -3 = 6 * 4 + b ∧ 
  ∃ x y : ℝ, y = 6 * x + b ∧ y = -x + 1 →
  m + b = -21 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1458_145808


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1458_145810

theorem smallest_sum_of_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C : ℚ) / B = 7 / 4 →  -- C/B = 7/4
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' : ℚ) / B' = 7 / 4 → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1458_145810


namespace NUMINAMATH_CALUDE_complex_square_equality_l1458_145870

theorem complex_square_equality (a b : ℕ+) :
  (↑a - Complex.I * ↑b) ^ 2 = 15 - 8 * Complex.I →
  ↑a - Complex.I * ↑b = 4 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1458_145870


namespace NUMINAMATH_CALUDE_fence_perimeter_is_200_l1458_145847

/-- A square field enclosed by evenly spaced triangular posts -/
structure FenceSetup where
  total_posts : ℕ
  post_width : ℝ
  gap_width : ℝ

/-- Calculate the outer perimeter of the fence setup -/
def outer_perimeter (f : FenceSetup) : ℝ :=
  let posts_per_side := f.total_posts / 4
  let gaps_per_side := posts_per_side - 1
  let side_length := posts_per_side * f.post_width + gaps_per_side * f.gap_width
  4 * side_length

/-- Theorem: The outer perimeter of the given fence setup is 200 feet -/
theorem fence_perimeter_is_200 : 
  outer_perimeter ⟨36, 2, 4⟩ = 200 := by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_200_l1458_145847


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l1458_145820

theorem inscribed_circle_square_area (r : ℝ) (h : r = 6) :
  let circle_area := π * r^2
  let square_side := r * (2:ℝ).sqrt
  let quarter_circle_area := (1/4:ℝ) * π * square_side^2
  circle_area - quarter_circle_area = 18 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l1458_145820


namespace NUMINAMATH_CALUDE_tunnel_length_l1458_145878

/-- Calculates the length of a tunnel given train and travel information -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 4 →
  train_speed = 90 →
  (train_speed / 60) * exit_time - train_length = 4 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l1458_145878


namespace NUMINAMATH_CALUDE_coefficients_of_our_equation_l1458_145834

/-- Given a quadratic equation ax^2 + bx + c = 0, 
    returns the coefficients a, b, and c as a triple -/
def quadratic_coefficients (a b c : ℚ) : ℚ × ℚ × ℚ := (a, b, c)

/-- The quadratic equation 3x^2 - 6x - 1 = 0 -/
def our_equation := quadratic_coefficients 3 (-6) (-1)

theorem coefficients_of_our_equation :
  our_equation.2.1 = -6 ∧ our_equation.2.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_our_equation_l1458_145834


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1458_145866

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1458_145866


namespace NUMINAMATH_CALUDE_sara_golf_balls_l1458_145896

theorem sara_golf_balls (x : ℕ) : x = 16 * (3 * 4) → x / 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l1458_145896


namespace NUMINAMATH_CALUDE_line_passes_through_center_l1458_145804

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3*x + 2*y = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (2, -3)

/-- Theorem stating that the line passes through the center of the circle -/
theorem line_passes_through_center : 
  line_equation center.1 center.2 ∧ circle_equation center.1 center.2 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_center_l1458_145804


namespace NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l1458_145821

theorem ratio_sum_squares_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 4725) :
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l1458_145821


namespace NUMINAMATH_CALUDE_parallel_vector_problem_l1458_145858

/-- Given two vectors a and b in ℝ², where a = (-2, 1), |b| = 5, and a is parallel to b,
    prove that b is either (-2√5, √5) or (2√5, -√5). -/
theorem parallel_vector_problem (a b : ℝ × ℝ) : 
  a = (-2, 1) → 
  ‖b‖ = 5 → 
  ∃ (k : ℝ), b = k • a → 
  b = (-2 * Real.sqrt 5, Real.sqrt 5) ∨ b = (2 * Real.sqrt 5, -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vector_problem_l1458_145858


namespace NUMINAMATH_CALUDE_student_calculation_l1458_145859

theorem student_calculation (chosen_number : ℕ) (h : chosen_number = 121) : 
  2 * chosen_number - 138 = 104 := by
  sorry

#check student_calculation

end NUMINAMATH_CALUDE_student_calculation_l1458_145859


namespace NUMINAMATH_CALUDE_student_sums_correct_l1458_145853

theorem student_sums_correct (total : ℕ) (correct : ℕ) (wrong : ℕ) : 
  total = 54 → 
  wrong = 2 * correct → 
  total = correct + wrong → 
  correct = 18 := by
sorry

end NUMINAMATH_CALUDE_student_sums_correct_l1458_145853


namespace NUMINAMATH_CALUDE_complex_expression_1_complex_expression_2_l1458_145811

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_squared : i^2 = -1
axiom i_fourth : i^4 = 1

-- Theorem for the first expression
theorem complex_expression_1 :
  (4 - i^5) * (6 + 2*i^7) + (7 + i^11) * (4 - 3*i) = 57 - 39*i :=
by sorry

-- Theorem for the second expression
theorem complex_expression_2 :
  (5 * (4 + i)^2) / (i * (2 + i)) = -47 - 98*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_1_complex_expression_2_l1458_145811


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1458_145805

/-- Given a cylinder with base area S and lateral surface that unfolds into a square,
    prove that its lateral surface area is 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  circumference * height = 4 * Real.pi * S := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1458_145805


namespace NUMINAMATH_CALUDE_odd_function_property_l1458_145887

/-- Given a function f(x) = x^5 + ax^3 + bx, where a and b are real constants,
    if f(-2) = 10, then f(2) = -10 -/
theorem odd_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 + a*x^3 + b*x
  f (-2) = 10 → f 2 = -10 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1458_145887


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l1458_145830

theorem jelly_bean_ratio : 
  ∀ (total_jelly_beans red_jelly_beans coconut_flavored_red_jelly_beans : ℕ),
    total_jelly_beans = 4000 →
    coconut_flavored_red_jelly_beans = 750 →
    4 * coconut_flavored_red_jelly_beans = red_jelly_beans →
    3 * total_jelly_beans = 4 * red_jelly_beans :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l1458_145830


namespace NUMINAMATH_CALUDE_min_value_theorem_l1458_145889

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1 / (3 * a + 5 * b) + 1 / (3 * b + 5 * c) + 1 / (3 * c + 5 * a) ≥ 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1458_145889


namespace NUMINAMATH_CALUDE_equalizing_amount_is_55_l1458_145814

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The condition when Foma gives Ierema 70 gold coins -/
def condition1 (w : MerchantWealth) : Prop :=
  w.ierema + 70 = w.yuliy

/-- The condition when Foma gives Ierema 40 gold coins -/
def condition2 (w : MerchantWealth) : Prop :=
  w.foma - 40 = w.yuliy

/-- The amount of gold coins Foma should give Ierema to equalize their wealth -/
def equalizingAmount (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

theorem equalizing_amount_is_55 (w : MerchantWealth) 
  (h1 : condition1 w) (h2 : condition2 w) : 
  equalizingAmount w = 55 := by
  sorry

end NUMINAMATH_CALUDE_equalizing_amount_is_55_l1458_145814


namespace NUMINAMATH_CALUDE_integral_abs_x_minus_one_l1458_145872

-- Define the function to be integrated
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem integral_abs_x_minus_one : ∫ x in (-2)..2, f x = 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_abs_x_minus_one_l1458_145872


namespace NUMINAMATH_CALUDE_ordering_proof_l1458_145877

noncomputable def a : ℝ := Real.log 2.6
noncomputable def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem ordering_proof : b > c ∧ c > a := by sorry

end NUMINAMATH_CALUDE_ordering_proof_l1458_145877


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1458_145851

theorem hyperbola_eccentricity (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : p > 0)
  (h_equilateral : b / a = Real.sqrt 3 / 3) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1458_145851


namespace NUMINAMATH_CALUDE_henry_tic_tac_toe_wins_l1458_145838

theorem henry_tic_tac_toe_wins 
  (total_games : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) 
  (h1 : total_games = 14) 
  (h2 : losses = 2) 
  (h3 : draws = 10) : 
  total_games - losses - draws = 2 := by
  sorry

end NUMINAMATH_CALUDE_henry_tic_tac_toe_wins_l1458_145838


namespace NUMINAMATH_CALUDE_brick_height_l1458_145856

/-- The surface area of a rectangular prism given its length, width, and height -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with length 8 cm, width 6 cm, 
    and surface area 152 cm² is 2 cm -/
theorem brick_height : 
  ∃ (h : ℝ), h > 0 ∧ surface_area 8 6 h = 152 → h = 2 := by
sorry

end NUMINAMATH_CALUDE_brick_height_l1458_145856


namespace NUMINAMATH_CALUDE_negation_of_existence_exp_minus_x_minus_one_negation_l1458_145874

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem exp_minus_x_minus_one_negation :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_exp_minus_x_minus_one_negation_l1458_145874


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1458_145894

/-- Given two points P and P' that are symmetric with respect to the origin,
    prove that 2a+b = -3 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (2*a + 1 = -1 ∧ 4 = -(3*b - 1)) → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1458_145894


namespace NUMINAMATH_CALUDE_continuity_at_two_l1458_145843

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / (x^2 - 4)

theorem continuity_at_two :
  ∀ (c : ℝ), ContinuousAt f 2 ↔ c = 7/4 := by sorry

end NUMINAMATH_CALUDE_continuity_at_two_l1458_145843


namespace NUMINAMATH_CALUDE_not_all_electric_implies_some_not_electric_l1458_145869

-- Define the set of all cars in the parking lot
variable (Car : Type)
variable (parking_lot : Set Car)

-- Define a predicate for electric cars
variable (is_electric : Car → Prop)

-- Define the theorem
theorem not_all_electric_implies_some_not_electric
  (h : ¬ ∀ (c : Car), c ∈ parking_lot → is_electric c) :
  ∃ (c : Car), c ∈ parking_lot ∧ ¬ is_electric c :=
by
  sorry

end NUMINAMATH_CALUDE_not_all_electric_implies_some_not_electric_l1458_145869


namespace NUMINAMATH_CALUDE_max_points_scored_l1458_145827

-- Define the variables
def total_shots : ℕ := 50
def three_point_success_rate : ℚ := 3 / 10
def two_point_success_rate : ℚ := 4 / 10

-- Define the function to calculate points
def calculate_points (three_point_shots : ℕ) : ℚ :=
  let two_point_shots : ℕ := total_shots - three_point_shots
  (three_point_success_rate * 3 * three_point_shots) + (two_point_success_rate * 2 * two_point_shots)

-- Theorem statement
theorem max_points_scored :
  ∃ (max_points : ℚ), max_points = 45 ∧
  ∀ (x : ℕ), x ≤ total_shots → calculate_points x ≤ max_points :=
sorry

end NUMINAMATH_CALUDE_max_points_scored_l1458_145827


namespace NUMINAMATH_CALUDE_camera_profit_difference_l1458_145873

/-- Represents the profit calculation for camera sales -/
def camera_profit (num_cameras : ℕ) (buy_price sell_price : ℚ) : ℚ :=
  num_cameras * (sell_price - buy_price)

/-- Represents the problem of calculating the difference in profit between Maddox and Theo -/
theorem camera_profit_difference : 
  let num_cameras : ℕ := 3
  let buy_price : ℚ := 20
  let maddox_sell_price : ℚ := 28
  let theo_sell_price : ℚ := 23
  let maddox_profit := camera_profit num_cameras buy_price maddox_sell_price
  let theo_profit := camera_profit num_cameras buy_price theo_sell_price
  maddox_profit - theo_profit = 15 := by
sorry


end NUMINAMATH_CALUDE_camera_profit_difference_l1458_145873


namespace NUMINAMATH_CALUDE_point_circle_relationship_l1458_145868

theorem point_circle_relationship :
  ∀ θ : ℝ,
  let P : ℝ × ℝ := (5 * Real.cos θ, 4 * Real.sin θ)
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 25}
  P ∈ C ∨ (P.1^2 + P.2^2 < 25) :=
by sorry

end NUMINAMATH_CALUDE_point_circle_relationship_l1458_145868


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angle_l1458_145840

theorem triangle_arithmetic_sequence_angle (α d : ℝ) :
  (α - d) + α + (α + d) = 180 → α = 60 ∨ (α - d) = 60 ∨ (α + d) = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_angle_l1458_145840


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1458_145898

/-- Represents a right triangle with side lengths 6, 8, and 10 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (is_right : a^2 + b^2 = c^2)
  (side_lengths : a = 6 ∧ b = 8 ∧ c = 10)

/-- Represents three mutually externally tangent circles -/
structure TangentCircles :=
  (r₁ : ℝ) (r₂ : ℝ) (r₃ : ℝ)
  (tangent_condition : r₁ + r₂ = 6 ∧ r₁ + r₃ = 8 ∧ r₂ + r₃ = 10)

/-- The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 6-8-10 right triangle is 56π -/
theorem sum_of_circle_areas (t : Triangle) (c : TangentCircles) :
  π * (c.r₁^2 + c.r₂^2 + c.r₃^2) = 56 * π :=
sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1458_145898


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1458_145860

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1458_145860


namespace NUMINAMATH_CALUDE_expression_value_l1458_145883

theorem expression_value : 
  let a : ℕ := 2017
  let b : ℕ := 2016
  let c : ℕ := 2015
  ((a^2 + b^2)^2 - c^2 - 4 * a^2 * b^2) / (a^2 + c - b^2) = 2018 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1458_145883


namespace NUMINAMATH_CALUDE_candy_probability_theorem_l1458_145857

/-- Represents a packet of candies -/
structure CandyPacket where
  blue : ℕ
  total : ℕ
  h_total_pos : total > 0
  h_blue_le_total : blue ≤ total

/-- Represents a box containing two packets of candies -/
structure CandyBox where
  packet1 : CandyPacket
  packet2 : CandyPacket

/-- The probability of drawing a blue candy from the box -/
def blue_probability (box : CandyBox) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

theorem candy_probability_theorem :
  (∃ box : CandyBox, blue_probability box = 5/13) ∧
  (∃ box : CandyBox, blue_probability box = 7/18) ∧
  (∀ box : CandyBox, blue_probability box ≠ 17/40) :=
sorry

end NUMINAMATH_CALUDE_candy_probability_theorem_l1458_145857


namespace NUMINAMATH_CALUDE_reading_time_calculation_l1458_145800

def total_reading_time (total_chapters : ℕ) (reading_time_per_chapter : ℕ) : ℕ :=
  let chapters_read := total_chapters - (total_chapters / 3)
  (chapters_read * reading_time_per_chapter) / 60

theorem reading_time_calculation :
  total_reading_time 31 20 = 7 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l1458_145800


namespace NUMINAMATH_CALUDE_parallelogram_height_l1458_145825

/-- The height of a parallelogram with given base and area -/
theorem parallelogram_height (base area height : ℝ) 
  (h_base : base = 14)
  (h_area : area = 336)
  (h_formula : area = base * height) : height = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1458_145825


namespace NUMINAMATH_CALUDE_bennett_window_screens_l1458_145892

theorem bennett_window_screens (january february march : ℕ) : 
  february = 2 * january →
  february = march / 4 →
  january + february + march = 12100 →
  march = 8800 := by sorry

end NUMINAMATH_CALUDE_bennett_window_screens_l1458_145892


namespace NUMINAMATH_CALUDE_puzzle_completion_time_l1458_145819

/-- Calculates the time to complete puzzles given the number of puzzles, pieces per puzzle, and completion rate. -/
def time_to_complete_puzzles (num_puzzles : ℕ) (pieces_per_puzzle : ℕ) (pieces_per_set : ℕ) (minutes_per_set : ℕ) : ℕ :=
  let total_pieces := num_puzzles * pieces_per_puzzle
  let num_sets := total_pieces / pieces_per_set
  num_sets * minutes_per_set

/-- Theorem stating that completing 2 puzzles of 2000 pieces each, at a rate of 100 pieces per 10 minutes, takes 400 minutes. -/
theorem puzzle_completion_time :
  time_to_complete_puzzles 2 2000 100 10 = 400 := by
  sorry

#eval time_to_complete_puzzles 2 2000 100 10

end NUMINAMATH_CALUDE_puzzle_completion_time_l1458_145819


namespace NUMINAMATH_CALUDE_right_triangle_in_sets_l1458_145846

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2

/-- The sets of side lengths given in the problem --/
def side_length_sets : List (ℕ × ℕ × ℕ) :=
  [(5, 4, 3), (1, 2, 3), (5, 6, 7), (2, 2, 3)]

theorem right_triangle_in_sets :
  ∃! (a b c : ℕ), (a, b, c) ∈ side_length_sets ∧ is_right_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_in_sets_l1458_145846


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1458_145822

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3/5 * x, then a = 5 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3/5 * x) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1458_145822


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1458_145885

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l1458_145885


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1458_145831

-- Part 1: Non-existence of positive integer sequence
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) :=
sorry

-- Part 2: Existence of positive irrational number sequence
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1458_145831


namespace NUMINAMATH_CALUDE_replaced_person_weight_l1458_145837

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (9/2) 101 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l1458_145837


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1458_145893

/-- Represents the average age of a cricket team --/
def average_age : ℝ := 23

/-- Represents the number of team members --/
def team_size : ℕ := 11

/-- Represents the age of the captain --/
def captain_age : ℕ := 25

/-- Represents the age of the wicket keeper --/
def wicket_keeper_age : ℕ := captain_age + 3

/-- Represents the age of the vice-captain --/
def vice_captain_age : ℕ := wicket_keeper_age - 4

/-- Theorem stating that the average age of the cricket team is 23 years --/
theorem cricket_team_average_age :
  average_age * team_size =
    captain_age + wicket_keeper_age + vice_captain_age +
    (team_size - 3) * (average_age - 1) := by
  sorry

#check cricket_team_average_age

end NUMINAMATH_CALUDE_cricket_team_average_age_l1458_145893


namespace NUMINAMATH_CALUDE_slope_angle_of_line_PQ_l1458_145881

theorem slope_angle_of_line_PQ 
  (a b c : ℝ) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) 
  (P : ℝ × ℝ) 
  (hP : P = (b, b + c)) 
  (Q : ℝ × ℝ) 
  (hQ : Q = (a, c + a)) : 
  Real.arctan ((Q.2 - P.2) / (Q.1 - P.1)) = π / 4 := by
  sorry

#check slope_angle_of_line_PQ

end NUMINAMATH_CALUDE_slope_angle_of_line_PQ_l1458_145881


namespace NUMINAMATH_CALUDE_max_value_at_negative_one_l1458_145865

-- Define a monic cubic polynomial
def monic_cubic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^3 + a*x^2 + b*x + c

-- Define the condition that all roots are non-negative
def non_negative_roots (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = 0 → x ≥ 0

-- Main theorem
theorem max_value_at_negative_one (f : ℝ → ℝ) :
  monic_cubic f →
  f 0 = -64 →
  non_negative_roots f →
  ∀ g : ℝ → ℝ, monic_cubic g → g 0 = -64 → non_negative_roots g →
  f (-1) ≤ -125 ∧ (∃ h : ℝ → ℝ, monic_cubic h ∧ h 0 = -64 ∧ non_negative_roots h ∧ h (-1) = -125) :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_negative_one_l1458_145865


namespace NUMINAMATH_CALUDE_no_square_with_seven_lattice_points_l1458_145867

/-- A square in a right-angled coordinate system -/
structure RotatedSquare where
  /-- The center of the square -/
  center : ℝ × ℝ
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle between the sides of the square and the coordinate axes (in radians) -/
  angle : ℝ

/-- A lattice point in the coordinate system -/
def LatticePoint : Type := ℤ × ℤ

/-- Predicate to check if a point is inside a rotated square -/
def is_inside (s : RotatedSquare) (p : ℝ × ℝ) : Prop := sorry

/-- Count the number of lattice points inside a rotated square -/
def count_lattice_points_inside (s : RotatedSquare) : ℕ := sorry

/-- Theorem stating that no rotated square at 45° contains exactly 7 lattice points -/
theorem no_square_with_seven_lattice_points :
  ¬ ∃ (s : RotatedSquare), s.angle = π / 4 ∧ count_lattice_points_inside s = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_square_with_seven_lattice_points_l1458_145867


namespace NUMINAMATH_CALUDE_mikes_initial_cards_l1458_145864

theorem mikes_initial_cards (initial_cards current_cards cards_sold : ℕ) :
  current_cards = 74 →
  cards_sold = 13 →
  initial_cards = current_cards + cards_sold →
  initial_cards = 87 := by
sorry

end NUMINAMATH_CALUDE_mikes_initial_cards_l1458_145864


namespace NUMINAMATH_CALUDE_seventh_term_is_10_4_l1458_145852

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first four terms is 20
  sum_first_four : a + (a + d) + (a + 2*d) + (a + 3*d) = 20
  -- Fifth term is 8
  fifth_term : a + 4*d = 8

/-- The seventh term of the arithmetic sequence is 10.4 -/
theorem seventh_term_is_10_4 (seq : ArithmeticSequence) : 
  seq.a + 6*seq.d = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_10_4_l1458_145852


namespace NUMINAMATH_CALUDE_house_bedrooms_count_l1458_145803

/-- A house with two floors and a certain number of bedrooms on each floor. -/
structure House where
  second_floor_bedrooms : ℕ
  first_floor_bedrooms : ℕ

/-- The total number of bedrooms in a house. -/
def total_bedrooms (h : House) : ℕ :=
  h.second_floor_bedrooms + h.first_floor_bedrooms

/-- Theorem stating that a house with 2 bedrooms on the second floor and 8 on the first floor has 10 bedrooms in total. -/
theorem house_bedrooms_count :
  ∀ (h : House), h.second_floor_bedrooms = 2 → h.first_floor_bedrooms = 8 →
  total_bedrooms h = 10 := by
  sorry

end NUMINAMATH_CALUDE_house_bedrooms_count_l1458_145803


namespace NUMINAMATH_CALUDE_elective_schemes_count_l1458_145845

/-- The number of courses offered by the school -/
def total_courses : ℕ := 10

/-- The number of conflicting courses (A, B, C) -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must elect -/
def courses_to_elect : ℕ := 3

/-- The number of different elective schemes available for a student -/
def elective_schemes : ℕ := Nat.choose (total_courses - conflicting_courses) courses_to_elect +
                             conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_elect - 1)

theorem elective_schemes_count :
  elective_schemes = 98 :=
sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l1458_145845


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1458_145809

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 24 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : distribute_balls 5 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1458_145809


namespace NUMINAMATH_CALUDE_kit_prices_correct_l1458_145882

/-- The price of kit B in yuan -/
def price_B : ℝ := 150

/-- The price of kit A in yuan -/
def price_A : ℝ := 180

/-- The relationship between the prices of kit A and kit B -/
def price_relation : Prop := price_A = 1.2 * price_B

/-- The equation representing the difference in quantities purchased -/
def quantity_difference : Prop :=
  (9900 / price_A) - (7500 / price_B) = 5

theorem kit_prices_correct :
  price_relation ∧ quantity_difference → price_A = 180 ∧ price_B = 150 := by
  sorry

end NUMINAMATH_CALUDE_kit_prices_correct_l1458_145882


namespace NUMINAMATH_CALUDE_prime_sequence_implies_composite_l1458_145806

theorem prime_sequence_implies_composite (p : ℕ) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (3*p + 2))
  (h3 : Nat.Prime (5*p + 4))
  (h4 : Nat.Prime (7*p + 6))
  (h5 : Nat.Prime (9*p + 8))
  (h6 : Nat.Prime (11*p + 10)) :
  ¬(Nat.Prime (6*p + 11)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_implies_composite_l1458_145806


namespace NUMINAMATH_CALUDE_second_denomination_value_l1458_145816

theorem second_denomination_value (total_amount : ℕ) (total_notes : ℕ) : 
  total_amount = 400 →
  total_notes = 75 →
  ∃ (x : ℕ), 
    x > 1 ∧ 
    x < 10 ∧
    (total_notes / 3) * (1 + x + 10) = total_amount →
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_denomination_value_l1458_145816


namespace NUMINAMATH_CALUDE_i_to_2010_eq_neg_one_l1458_145818

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2010_eq_neg_one : i^2010 = -1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2010_eq_neg_one_l1458_145818


namespace NUMINAMATH_CALUDE_bus_passengers_l1458_145871

theorem bus_passengers (initial : ℕ) (got_on : ℕ) (current : ℕ) : 
  got_on = 13 → current = 17 → initial + got_on = current → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1458_145871


namespace NUMINAMATH_CALUDE_exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l1458_145863

/-- The set A defined by the quadratic equation (a-1)x^2 + 3x - 2 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | (a - 1) * x^2 + 3 * x - 2 = 0}

/-- The theorem stating the existence of 'a' values for which A has exactly two subsets -/
theorem exist_a_with_two_subsets :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  (∀ (S : Set ℝ), S ⊆ A a₁ → (S = ∅ ∨ S = A a₁)) ∧
  (∀ (S : Set ℝ), S ⊆ A a₂ → (S = ∅ ∨ S = A a₂)) ∧
  a₁ = 1 ∧ a₂ = -1/8 := by
  sorry

/-- The theorem stating that for a = 1, A = {2/3} -/
theorem a_eq_one_implies_A_eq_two_thirds :
  A 1 = {2/3} := by
  sorry

/-- The theorem stating that for a = -1/8, A = {4/3} -/
theorem a_eq_neg_one_eighth_implies_A_eq_four_thirds :
  A (-1/8) = {4/3} := by
  sorry

end NUMINAMATH_CALUDE_exist_a_with_two_subsets_a_eq_one_implies_A_eq_two_thirds_a_eq_neg_one_eighth_implies_A_eq_four_thirds_l1458_145863


namespace NUMINAMATH_CALUDE_power_four_times_four_equals_square_to_fourth_l1458_145835

theorem power_four_times_four_equals_square_to_fourth (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_four_equals_square_to_fourth_l1458_145835


namespace NUMINAMATH_CALUDE_topsoil_cost_l1458_145862

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards : ℝ := 7

/-- The total cost of topsoil for a given number of cubic yards -/
def total_cost (yards : ℝ) : ℝ :=
  yards * cubic_feet_per_cubic_yard * cost_per_cubic_foot

theorem topsoil_cost : total_cost cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l1458_145862
