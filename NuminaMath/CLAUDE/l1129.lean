import Mathlib

namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1129_112920

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) :
  a = 5 → r = 3/2 → geometric_sequence a r 10 = 98415/512 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1129_112920


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l1129_112932

theorem triangle_cosine_problem (A B C : ℝ) (a b : ℝ) :
  -- Conditions
  A + B + C = Real.pi ∧  -- Sum of angles in a triangle
  B = (A + B + C) / 3 ∧  -- Angles form arithmetic sequence
  a = 8 ∧ 
  b = 7 ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Conclusion
  Real.cos C = -11/14 ∨ Real.cos C = -13/14 := by
sorry


end NUMINAMATH_CALUDE_triangle_cosine_problem_l1129_112932


namespace NUMINAMATH_CALUDE_only_D_opposite_sign_l1129_112994

-- Define the pairs of numbers
def pair_A : ℤ × ℤ := (-(-1), 1)
def pair_B : ℤ × ℤ := ((-1)^2, 1)
def pair_C : ℤ × ℤ := (|(-1)|, 1)
def pair_D : ℤ × ℤ := (-1, 1)

-- Define a function to check if two numbers are opposite in sign
def opposite_sign (a b : ℤ) : Prop := a * b < 0

-- Theorem stating that only pair D contains numbers with opposite signs
theorem only_D_opposite_sign :
  ¬(opposite_sign pair_A.1 pair_A.2) ∧
  ¬(opposite_sign pair_B.1 pair_B.2) ∧
  ¬(opposite_sign pair_C.1 pair_C.2) ∧
  (opposite_sign pair_D.1 pair_D.2) :=
sorry

end NUMINAMATH_CALUDE_only_D_opposite_sign_l1129_112994


namespace NUMINAMATH_CALUDE_divide_subtract_problem_l1129_112999

theorem divide_subtract_problem (x : ℝ) : 
  (990 / x) - 100 = 10 → x = 9 := by sorry

end NUMINAMATH_CALUDE_divide_subtract_problem_l1129_112999


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l1129_112960

/-- The initial number of apples Tessa had -/
def initial_apples : ℝ := sorry

/-- The number of apples Anita gives to Tessa -/
def apples_from_anita : ℝ := 5.0

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℝ := 4.0

/-- The number of apples left after making the pie -/
def apples_left : ℝ := 11

/-- Theorem stating that Tessa initially had 10 apples -/
theorem tessa_initial_apples : 
  initial_apples + apples_from_anita - apples_for_pie = apples_left ∧ 
  initial_apples = 10 := by sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l1129_112960


namespace NUMINAMATH_CALUDE_composite_shape_sum_l1129_112970

/-- Represents a 3D geometric shape with faces, edges, and vertices -/
structure Shape where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The initial triangular prism -/
def triangularPrism : Shape := ⟨5, 9, 6⟩

/-- Attaches a regular pentagonal prism to a quadrilateral face of the given shape -/
def attachPentagonalPrism (s : Shape) : Shape :=
  ⟨s.faces - 1 + 7, s.edges + 10, s.vertices + 5⟩

/-- Adds a pyramid to a pentagonal face of the given shape -/
def addPyramid (s : Shape) : Shape :=
  ⟨s.faces - 1 + 5, s.edges + 5, s.vertices + 1⟩

/-- Calculates the sum of faces, edges, and vertices of a shape -/
def sumFeatures (s : Shape) : ℕ :=
  s.faces + s.edges + s.vertices

/-- Theorem stating that the sum of features of the final composite shape is 51 -/
theorem composite_shape_sum :
  sumFeatures (addPyramid (attachPentagonalPrism triangularPrism)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_sum_l1129_112970


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l1129_112914

theorem candle_flower_groupings :
  let n_candles : ℕ := 4
  let k_candles : ℕ := 2
  let n_flowers : ℕ := 9
  let k_flowers : ℕ := 8
  (n_candles.choose k_candles) * (n_flowers.choose k_flowers) = 27 := by
  sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l1129_112914


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1129_112990

-- Problem 1
theorem problem_1 : (-81) - (1/4 : ℚ) + (-7) - (3/4 : ℚ) - (-22) = -67 := by sorry

-- Problem 2
theorem problem_2 : -(4^2) / ((-2)^3) - 2^2 * (-1/2 : ℚ) = 4 := by sorry

-- Problem 3
theorem problem_3 : -(1^2023) - 24 * ((1/2 : ℚ) - 2/3 + 3/8) = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1129_112990


namespace NUMINAMATH_CALUDE_emily_lives_emily_final_lives_l1129_112935

/-- Calculates the final number of lives in Emily's video game. -/
theorem emily_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) :
  initial ≥ lost →
  initial - lost + gained = initial + gained - lost :=
by
  sorry

/-- Proves that Emily ends up with 41 lives. -/
theorem emily_final_lives : 
  let initial : ℕ := 42
  let lost : ℕ := 25
  let gained : ℕ := 24
  initial ≥ lost →
  initial - lost + gained = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_lives_emily_final_lives_l1129_112935


namespace NUMINAMATH_CALUDE_phone_call_duration_l1129_112942

/-- Calculates the duration of a phone call given the initial card value, cost per minute, and remaining credit. -/
def call_duration (initial_value : ℚ) (cost_per_minute : ℚ) (remaining_credit : ℚ) : ℚ :=
  (initial_value - remaining_credit) / cost_per_minute

/-- Theorem stating that given the specific values from the problem, the call duration is 22 minutes. -/
theorem phone_call_duration :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16/100
  let remaining_credit : ℚ := 2648/100
  call_duration initial_value cost_per_minute remaining_credit = 22 := by
sorry

end NUMINAMATH_CALUDE_phone_call_duration_l1129_112942


namespace NUMINAMATH_CALUDE_on_time_speed_l1129_112923

-- Define the variables
def distance : ℝ → ℝ → ℝ := λ speed time => speed * time

-- Define the conditions
def early_arrival (d : ℝ) (T : ℝ) : Prop := distance 20 (T - 0.5) = d
def late_arrival (d : ℝ) (T : ℝ) : Prop := distance 12 (T + 0.5) = d

-- Define the theorem
theorem on_time_speed (d : ℝ) (T : ℝ) :
  early_arrival d T → late_arrival d T → distance 15 T = d :=
by sorry

end NUMINAMATH_CALUDE_on_time_speed_l1129_112923


namespace NUMINAMATH_CALUDE_circle_inequality_l1129_112924

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + c ≥ 0) ↔ c ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l1129_112924


namespace NUMINAMATH_CALUDE_parabola_properties_l1129_112996

theorem parabola_properties (a b c : ℝ) (h1 : 0 < a) (h2 : a < c) 
  (h3 : a + b + c = 0) : 
  (2 * a + b < 0 ∧ 
   ∃ x : ℝ, x > 1 ∧ (2 * a * x + b ≤ 0) ∧
   (b^2 + 4 * a^2 > 0)) ∧
  ¬(∀ x : ℝ, x > 1 → 2 * a * x + b > 0) := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1129_112996


namespace NUMINAMATH_CALUDE_todays_production_l1129_112959

theorem todays_production (n : ℕ) (past_average : ℝ) (new_average : ℝ) 
  (h1 : n = 9)
  (h2 : past_average = 50)
  (h3 : new_average = 54) :
  (n + 1) * new_average - n * past_average = 90 := by
  sorry

end NUMINAMATH_CALUDE_todays_production_l1129_112959


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1129_112980

theorem system_solution_ratio (k : ℝ) (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 3*z = 0 →
  3*x + k*y - 2*z = 0 →
  2*x + 4*y - 3*z = 0 →
  x*z / (y^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1129_112980


namespace NUMINAMATH_CALUDE_point_on_line_l1129_112951

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line :
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨0, -4⟩
  let p3 : Point := ⟨2, 2⟩
  collinear p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1129_112951


namespace NUMINAMATH_CALUDE_bhupathi_abhinav_fraction_l1129_112984

theorem bhupathi_abhinav_fraction : 
  ∀ (abhinav bhupathi : ℚ),
  abhinav + bhupathi = 1210 →
  bhupathi = 484 →
  ∃ (x : ℚ), (4 / 15) * abhinav = x * bhupathi ∧ x = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_bhupathi_abhinav_fraction_l1129_112984


namespace NUMINAMATH_CALUDE_rug_area_l1129_112952

/-- The area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h1 : floor_length = 12)
  (h2 : floor_width = 10)
  (h3 : strip_width = 3)
  (h4 : floor_length > 0)
  (h5 : floor_width > 0)
  (h6 : strip_width > 0)
  (h7 : 2 * strip_width < floor_length)
  (h8 : 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_l1129_112952


namespace NUMINAMATH_CALUDE_sequence_existence_l1129_112958

theorem sequence_existence (a b : ℤ) (ha : a > 2) (hb : b > 2) :
  ∃ (k : ℕ) (n : ℕ → ℤ), 
    n 1 = a ∧ 
    n k = b ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i < k → (n i + n (i + 1)) ∣ (n i * n (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_sequence_existence_l1129_112958


namespace NUMINAMATH_CALUDE_triangle_angle_and_area_l1129_112916

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the measure of angle B and the area of the triangle. -/
theorem triangle_angle_and_area 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = 6)
  (h2 : b = 5)
  (h3 : Real.cos A = -4/5) :
  B = π/6 ∧ 
  (1/2 * a * b * Real.sin C = (9 * Real.sqrt 3 - 12) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_area_l1129_112916


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l1129_112950

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the intersection points
def x_intercept : ℝ := -1
def y_intercept : ℝ := 2

-- Define the area of the triangle
def triangle_area : ℝ := 1

-- Theorem statement
theorem triangle_area_is_one :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  triangle_area = (1/2) * |x_intercept| * |y_intercept| := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_one_l1129_112950


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1129_112954

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 41 / 20 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1129_112954


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1129_112904

theorem quadratic_equation_roots : ∃ x y : ℝ, x ≠ y ∧ 
  x^2 - 6*x + 1 = 0 ∧ y^2 - 6*y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1129_112904


namespace NUMINAMATH_CALUDE_race_outcomes_count_l1129_112903

/-- The number of participants in the race -/
def num_participants : Nat := 5

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_podium_positions : Nat := 3

/-- Calculate the number of permutations of k items chosen from n items -/
def permutations (n k : Nat) : Nat :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The theorem stating that the number of different 1st-2nd-3rd place outcomes
    in a race with 5 participants and no ties is equal to 60 -/
theorem race_outcomes_count : 
  permutations num_participants num_podium_positions = 60 := by
  sorry


end NUMINAMATH_CALUDE_race_outcomes_count_l1129_112903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1129_112948

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 + a 6 = 8 ∧ 
  a 3 + a 4 = 3

/-- The common difference of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1129_112948


namespace NUMINAMATH_CALUDE_card_probability_theorem_probability_after_10_shuffles_value_l1129_112911

/-- The probability that card 6 is higher than card 3 after n shuffles -/
def p (n : ℕ) : ℚ :=
  (3^n - 2^n) / (2 * 3^n)

/-- The recurrence relation for the probability -/
def recurrence (p_prev : ℚ) : ℚ :=
  (4 * p_prev + 1) / 6

theorem card_probability_theorem (n : ℕ) :
  p n = recurrence (p (n - 1)) ∧ p 0 = 0 :=
by sorry

/-- The probability that card 6 is higher than card 3 after 10 shuffles -/
def probability_after_10_shuffles : ℚ := p 10

theorem probability_after_10_shuffles_value :
  probability_after_10_shuffles = (3^10 - 2^10) / (2 * 3^10) :=
by sorry

end NUMINAMATH_CALUDE_card_probability_theorem_probability_after_10_shuffles_value_l1129_112911


namespace NUMINAMATH_CALUDE_wall_area_l1129_112940

/- Define the types of tiles -/
inductive TileType
| Small
| Regular
| Jumbo

/- Define the properties of the wall and tiles -/
structure WallProperties where
  smallRatio : ℚ
  regularRatio : ℚ
  jumboRatio : ℚ
  smallProportion : ℚ
  regularProportion : ℚ
  jumboProportion : ℚ
  jumboLengthMultiplier : ℚ
  regularArea : ℚ

/- Define the wall properties based on the given conditions -/
def wall : WallProperties :=
  { smallRatio := 2/1
  , regularRatio := 3/1
  , jumboRatio := 3/1
  , smallProportion := 2/5
  , regularProportion := 3/10
  , jumboProportion := 1/4
  , jumboLengthMultiplier := 3
  , regularArea := 90
  }

/- Theorem statement -/
theorem wall_area (w : WallProperties) : 
  w.smallRatio = 2/1 ∧ 
  w.regularRatio = 3/1 ∧ 
  w.jumboRatio = 3/1 ∧
  w.smallProportion = 2/5 ∧ 
  w.regularProportion = 3/10 ∧ 
  w.jumboProportion = 1/4 ∧
  w.jumboLengthMultiplier = 3 ∧
  w.regularArea = 90 →
  (w.regularArea / w.regularProportion : ℚ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_l1129_112940


namespace NUMINAMATH_CALUDE_rectangle_area_is_100_l1129_112941

-- Define the rectangle
def Rectangle (width : ℝ) (length : ℝ) : Type :=
  { w : ℝ // w = width } × { l : ℝ // l = length }

-- Define the properties of the rectangle
def rectangle_properties (r : Rectangle 5 (4 * 5)) : Prop :=
  r.2.1 = 4 * r.1.1

-- Define the area of a rectangle
def area (r : Rectangle 5 (4 * 5)) : ℝ :=
  r.1.1 * r.2.1

-- Theorem statement
theorem rectangle_area_is_100 (r : Rectangle 5 (4 * 5)) 
  (h : rectangle_properties r) : area r = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_100_l1129_112941


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l1129_112972

theorem sum_remainder_zero : (((7283 + 7284 + 7285 + 7286 + 7287) * 2) % 9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l1129_112972


namespace NUMINAMATH_CALUDE_solve_income_problem_l1129_112929

def income_problem (day2 day3 day4 day5 average : ℚ) : Prop :=
  let known_days := [day2, day3, day4, day5]
  let total := 5 * average
  let sum_known := day2 + day3 + day4 + day5
  let day1 := total - sum_known
  (day2 = 150) ∧ (day3 = 750) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 420) →
  day1 = 300

theorem solve_income_problem :
  ∀ day2 day3 day4 day5 average,
  income_problem day2 day3 day4 day5 average :=
by
  sorry

end NUMINAMATH_CALUDE_solve_income_problem_l1129_112929


namespace NUMINAMATH_CALUDE_smallest_prime_divides_infinitely_many_and_all_l1129_112998

def a (n : ℕ) : ℕ := 4^(2*n+1) + 3^(n+2)

def is_divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

def divides_infinitely_many (p : ℕ) : Prop :=
  ∀ N, ∃ n ≥ N, is_divisible_by (a n) p

def divides_all (p : ℕ) : Prop :=
  ∀ n, n ≥ 1 → is_divisible_by (a n) p

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m, 1 < m → m < p → ¬(is_divisible_by p m)

theorem smallest_prime_divides_infinitely_many_and_all :
  ∃ (p q : ℕ),
    is_prime p ∧
    is_prime q ∧
    divides_infinitely_many p ∧
    divides_all q ∧
    (∀ p', is_prime p' → divides_infinitely_many p' → p ≤ p') ∧
    (∀ q', is_prime q' → divides_all q' → q ≤ q') ∧
    p = 5 ∧
    q = 13 :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divides_infinitely_many_and_all_l1129_112998


namespace NUMINAMATH_CALUDE_smallest_positive_integer_form_l1129_112906

theorem smallest_positive_integer_form (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ ∃ (a b : ℤ), k = 1237 * a + 78653 * b ∧ ∀ (l : ℕ), l > 0 → ∃ (c d : ℤ), l = 1237 * c + 78653 * d → k ≤ l :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_form_l1129_112906


namespace NUMINAMATH_CALUDE_gym_income_is_10800_l1129_112908

/-- A gym charges its members twice a month and has a fixed number of members. -/
structure Gym where
  charge_per_half_month : ℕ
  charges_per_month : ℕ
  num_members : ℕ

/-- Calculate the monthly income of the gym -/
def monthly_income (g : Gym) : ℕ :=
  g.charge_per_half_month * g.charges_per_month * g.num_members

/-- Theorem stating that the gym's monthly income is $10,800 -/
theorem gym_income_is_10800 (g : Gym) 
  (h1 : g.charge_per_half_month = 18)
  (h2 : g.charges_per_month = 2)
  (h3 : g.num_members = 300) : 
  monthly_income g = 10800 := by
  sorry

end NUMINAMATH_CALUDE_gym_income_is_10800_l1129_112908


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1129_112905

theorem wire_cut_ratio (p q : ℝ) (h : p > 0 ∧ q > 0) : 
  (p^2 / 16 = π * (q / (2 * π))^2) → p / q = 4 / Real.sqrt π := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1129_112905


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l1129_112919

/-- The ratio of the volume of a cube inscribed in a sphere to the volume of the sphere. -/
theorem cube_sphere_volume_ratio : ℝ :=
  2 * Real.sqrt 3 / Real.pi

/-- Theorem: For a cube inscribed in a sphere, the ratio of the volume of the cube
    to the volume of the sphere is 2√3/π. -/
theorem cube_inscribed_in_sphere_volume_ratio :
  let s : ℝ := cube_side_length -- side length of the cube
  let r : ℝ := sphere_radius -- radius of the sphere
  let cube_volume : ℝ := s^3
  let sphere_volume : ℝ := (4/3) * Real.pi * r^3
  r = (Real.sqrt 3 / 2) * s → -- condition that the cube is inscribed in the sphere
  cube_volume / sphere_volume = cube_sphere_volume_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_cube_inscribed_in_sphere_volume_ratio_l1129_112919


namespace NUMINAMATH_CALUDE_max_goats_from_coconuts_l1129_112979

/-- Represents the trading rates and initial coconut count --/
structure TradingRates :=
  (coconuts_per_crab : ℝ)
  (crabs_per_fish : ℝ)
  (fish_per_goat : ℝ)
  (initial_coconuts : ℕ)

/-- Calculates the maximum number of whole goats obtainable --/
def max_goats (rates : TradingRates) : ℕ :=
  sorry

/-- The theorem stating that given the specific trading rates and 1000 coconuts, 
    Max can obtain 33 goats --/
theorem max_goats_from_coconuts :
  let rates := TradingRates.mk 3.5 (6.25 / 5.5) 7.5 1000
  max_goats rates = 33 := by sorry

end NUMINAMATH_CALUDE_max_goats_from_coconuts_l1129_112979


namespace NUMINAMATH_CALUDE_tomato_count_l1129_112977

theorem tomato_count (plant1 plant2 plant3 plant4 : ℕ) : 
  plant1 = 8 →
  plant2 = plant1 + 4 →
  plant3 = 3 * (plant1 + plant2) →
  plant4 = plant3 →
  plant1 + plant2 + plant3 + plant4 = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_count_l1129_112977


namespace NUMINAMATH_CALUDE_equation_solution_l1129_112933

theorem equation_solution :
  ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1129_112933


namespace NUMINAMATH_CALUDE_sydney_texts_total_l1129_112985

/-- The number of texts Sydney sends to each person on Monday -/
def monday_texts : ℕ := 5

/-- The number of texts Sydney sends to each person on Tuesday -/
def tuesday_texts : ℕ := 15

/-- The number of people Sydney sends texts to -/
def num_recipients : ℕ := 2

/-- The total number of texts Sydney sent over both days -/
def total_texts : ℕ := (monday_texts * num_recipients) + (tuesday_texts * num_recipients)

theorem sydney_texts_total : total_texts = 40 := by
  sorry

end NUMINAMATH_CALUDE_sydney_texts_total_l1129_112985


namespace NUMINAMATH_CALUDE_max_take_home_pay_l1129_112978

/-- Represents the income in thousands of dollars -/
def income (x : ℝ) : ℝ := x + 10

/-- Represents the tax rate as a percentage -/
def taxRate (x : ℝ) : ℝ := x

/-- Calculates the take-home pay given the income parameter x -/
def takeHomePay (x : ℝ) : ℝ := 30250 - 10 * (x - 45)^2

/-- Theorem stating that the income yielding the maximum take-home pay is $55,000 -/
theorem max_take_home_pay :
  ∃ (x : ℝ), (∀ (y : ℝ), takeHomePay y ≤ takeHomePay x) ∧ income x = 55 := by
  sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l1129_112978


namespace NUMINAMATH_CALUDE_hangar_length_proof_l1129_112968

/-- The length of an airplane hangar given the number of planes it can fit and the length of each plane. -/
def hangar_length (num_planes : ℕ) (plane_length : ℕ) : ℕ :=
  num_planes * plane_length

/-- Theorem stating that a hangar fitting 7 planes of 40 feet each is 280 feet long. -/
theorem hangar_length_proof :
  hangar_length 7 40 = 280 := by
  sorry

end NUMINAMATH_CALUDE_hangar_length_proof_l1129_112968


namespace NUMINAMATH_CALUDE_solve_chicken_problem_l1129_112907

/-- Represents the problem of calculating the number of chickens sold -/
def chicken_problem (selling_price feed_cost feed_weight feed_per_chicken total_profit : ℚ) : Prop :=
  let cost_per_chicken := (feed_per_chicken / feed_weight) * feed_cost
  let profit_per_chicken := selling_price - cost_per_chicken
  let num_chickens := total_profit / profit_per_chicken
  num_chickens = 50

/-- Theorem stating the solution to the chicken problem -/
theorem solve_chicken_problem :
  chicken_problem 1.5 2 20 2 65 := by
  sorry

#check solve_chicken_problem

end NUMINAMATH_CALUDE_solve_chicken_problem_l1129_112907


namespace NUMINAMATH_CALUDE_work_completion_time_l1129_112946

/-- The number of days it takes worker A to complete the work -/
def days_A : ℚ := 10

/-- The efficiency ratio of worker B compared to worker A -/
def efficiency_ratio : ℚ := 1.75

/-- The number of days it takes worker B to complete the work -/
def days_B : ℚ := 40 / 7

theorem work_completion_time :
  days_A * efficiency_ratio = days_B :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1129_112946


namespace NUMINAMATH_CALUDE_unique_f_l1129_112975

def is_valid_f (f : Nat → Nat) : Prop :=
  ∀ n m : Nat, n > 1 → m > 1 → n ≠ m → f n * f m = f ((n * m) ^ 2021)

theorem unique_f : 
  ∀ f : Nat → Nat, is_valid_f f → (∀ x : Nat, x > 1 → f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_f_l1129_112975


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1129_112902

/-- Calculates the total cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSqm

/-- Proves that the plastering cost for a tank with given dimensions is 223.2 rupees. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSqm : ℝ := 0.30  -- 30 paise = 0.30 rupees
  plasteringCost length width depth costPerSqm = 223.2 := by
  sorry

#eval plasteringCost 25 12 6 0.30

end NUMINAMATH_CALUDE_tank_plastering_cost_l1129_112902


namespace NUMINAMATH_CALUDE_glenn_total_spent_l1129_112962

/-- The cost of a movie ticket on Monday -/
def monday_price : ℚ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℚ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℚ := 5 * monday_price

/-- The discount rate for Wednesday -/
def discount_rate : ℚ := 1 / 10

/-- The cost of popcorn and drink on Saturday -/
def popcorn_drink_cost : ℚ := 7

/-- The total amount Glenn spends -/
def total_spent : ℚ := wednesday_price * (1 - discount_rate) + saturday_price + popcorn_drink_cost

theorem glenn_total_spent : total_spent = 41 := by
  sorry

end NUMINAMATH_CALUDE_glenn_total_spent_l1129_112962


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1129_112925

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23976 →
  b % 8 = 0 →
  a = b - b / 8 →
  b - a = 1598 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1129_112925


namespace NUMINAMATH_CALUDE_race_distance_proof_l1129_112928

/-- The distance of the race where B beats C by 100 m, given the conditions from the problem. -/
def race_distance : ℝ := 700

theorem race_distance_proof (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 900)
  (h2 : Va / Vc = 600 / 472.5)
  (h3 : Vb / Vc = (race_distance - 100) / race_distance) : 
  race_distance = 700 := by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1129_112928


namespace NUMINAMATH_CALUDE_exponential_comparison_l1129_112976

theorem exponential_comparison (h1 : 1.5 > 1) (h2 : 2.3 < 3.2) :
  1.5^2.3 < 1.5^3.2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_comparison_l1129_112976


namespace NUMINAMATH_CALUDE_inequality_range_l1129_112912

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ a ∈ Set.Icc (-1) 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1129_112912


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1129_112982

/-- Given a line y = mx + b, where the point (2,3) is reflected to (8,7) across this line,
    prove that m + b = 9.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = 3 ∧ x₂ = 8 ∧ y₂ = 7 ∧
    (y₂ - y₁) / (x₂ - x₁) = -1 / m ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) ∈ {(x, y) | y = m * x + b}) →
  m + b = 9.5 := by
sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l1129_112982


namespace NUMINAMATH_CALUDE_problem_solution_l1129_112966

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) : x^7 - 6*x^5 + 5*x^3 - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1129_112966


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_property_l1129_112922

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The property given in the problem -/
def property (n : ℕ) : Prop :=
  binomial (n + 1) 7 - binomial n 7 = binomial n 8

/-- The theorem to be proved -/
theorem smallest_n_satisfying_property : 
  ∀ n : ℕ, n > 0 → (property n ↔ n ≥ 14) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_property_l1129_112922


namespace NUMINAMATH_CALUDE_inequality_proof_l1129_112992

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1129_112992


namespace NUMINAMATH_CALUDE_team_not_lose_prob_l1129_112943

structure PlayerStats where
  cf_rate : ℝ
  winger_rate : ℝ
  am_rate : ℝ
  cf_lose_prob : ℝ
  winger_lose_prob : ℝ
  am_lose_prob : ℝ

def not_lose_prob (stats : PlayerStats) : ℝ :=
  stats.cf_rate * (1 - stats.cf_lose_prob) +
  stats.winger_rate * (1 - stats.winger_lose_prob) +
  stats.am_rate * (1 - stats.am_lose_prob)

theorem team_not_lose_prob (stats : PlayerStats)
  (h1 : stats.cf_rate = 0.2)
  (h2 : stats.winger_rate = 0.5)
  (h3 : stats.am_rate = 0.3)
  (h4 : stats.cf_lose_prob = 0.4)
  (h5 : stats.winger_lose_prob = 0.2)
  (h6 : stats.am_lose_prob = 0.2) :
  not_lose_prob stats = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_team_not_lose_prob_l1129_112943


namespace NUMINAMATH_CALUDE_right_angled_triangle_k_values_l1129_112987

theorem right_angled_triangle_k_values (A B C : ℝ × ℝ) :
  let AB := B - A
  let AC := C - A
  AB = (2, 3) →
  AC = (1, k) →
  (AB.1 * AC.1 + AB.2 * AC.2 = 0 ∨
   AB.1 * (AC.1 - AB.1) + AB.2 * (AC.2 - AB.2) = 0 ∨
   AC.1 * (AB.1 - AC.1) + AC.2 * (AB.2 - AC.2) = 0) →
  k = -2/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2 ∨ k = 11/3 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_k_values_l1129_112987


namespace NUMINAMATH_CALUDE_fraction_ordering_l1129_112918

theorem fraction_ordering : (7 : ℚ) / 29 < 11 / 33 ∧ 11 / 33 < 13 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1129_112918


namespace NUMINAMATH_CALUDE_vacant_seats_l1129_112901

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 60/100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end NUMINAMATH_CALUDE_vacant_seats_l1129_112901


namespace NUMINAMATH_CALUDE_nyc_streetlights_l1129_112974

/-- The number of streetlights bought by the New York City Council -/
theorem nyc_streetlights (num_squares : ℕ) (lights_per_square : ℕ) (unused_lights : ℕ) :
  num_squares = 15 →
  lights_per_square = 12 →
  unused_lights = 20 →
  num_squares * lights_per_square + unused_lights = 200 := by
  sorry


end NUMINAMATH_CALUDE_nyc_streetlights_l1129_112974


namespace NUMINAMATH_CALUDE_point_distance_theorem_l1129_112939

theorem point_distance_theorem (x y : ℝ) (h1 : x > 2) :
  y = 14 ∧ (x - 2)^2 + (y - 8)^2 = 12^2 →
  x^2 + y^2 = 284 := by sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l1129_112939


namespace NUMINAMATH_CALUDE_natalia_documentaries_l1129_112957

/-- The number of documentaries in Natalia's library --/
def documentaries (novels comics albums crates_used crate_capacity : ℕ) : ℕ :=
  crates_used * crate_capacity - (novels + comics + albums)

/-- Theorem stating the number of documentaries in Natalia's library --/
theorem natalia_documentaries :
  documentaries 145 271 209 116 9 = 419 := by
  sorry

end NUMINAMATH_CALUDE_natalia_documentaries_l1129_112957


namespace NUMINAMATH_CALUDE_smallest_cover_l1129_112949

/-- The side length of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := 12

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_cover :
  (∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0) ∧
  square_area % rectangle_area = 0 ∧
  num_rectangles = 12 := by
  sorry

#check smallest_cover

end NUMINAMATH_CALUDE_smallest_cover_l1129_112949


namespace NUMINAMATH_CALUDE_phone_bill_calculation_l1129_112917

def initial_balance : ℚ := 800
def rent_payment : ℚ := 450
def paycheck_deposit : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def final_balance : ℚ := 1563

theorem phone_bill_calculation : 
  initial_balance - rent_payment + paycheck_deposit - electricity_bill - internet_bill - final_balance = 70 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_calculation_l1129_112917


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l1129_112927

theorem fraction_zero_implies_x_neg_two (x : ℝ) :
  (|x| - 2) / (x^2 - x - 2) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_neg_two_l1129_112927


namespace NUMINAMATH_CALUDE_tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l1129_112934

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + (a - 2) / x + 2 - 2 * a

-- State the theorems to be proved
theorem tangent_line_at_2_when_a_1 :
  ∀ x y : ℝ, f 1 2 = 3/2 ∧ (5 * x - 4 * y - 4 = 0 ↔ y - 3/2 = 5/4 * (x - 2)) :=
sorry

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a ≤ 2 → 
    StrictMono (f a) ∧ 
    StrictMonoOn (f a) (Set.Ioi 0)) ∧
  (a > 2 → 
    (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧
      StrictMonoOn (f a) (Set.Iic x₁) ∧
      StrictAntiOn (f a) (Set.Ioc x₁ 0) ∧
      StrictAntiOn (f a) (Set.Ioc 0 x₂) ∧
      StrictMonoOn (f a) (Set.Ioi x₂))) :=
sorry

theorem f_geq_2ln_x_iff (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 2 * Real.log x) ↔ a ≥ 1 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_2_when_a_1_monotonic_intervals_f_geq_2ln_x_iff_l1129_112934


namespace NUMINAMATH_CALUDE_product_as_sum_of_squares_l1129_112993

theorem product_as_sum_of_squares : 85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_squares_l1129_112993


namespace NUMINAMATH_CALUDE_books_together_l1129_112981

/-- The number of books Tim and Sam have together -/
def total_books (tim_books sam_books : ℕ) : ℕ := tim_books + sam_books

/-- Theorem: Tim and Sam have 96 books together -/
theorem books_together : total_books 44 52 = 96 := by sorry

end NUMINAMATH_CALUDE_books_together_l1129_112981


namespace NUMINAMATH_CALUDE_outfits_count_l1129_112913

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
  (pants : ℕ) (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * (green_hats + blue_hats) * pants) +
  (green_shirts * (red_hats + blue_hats) * pants) +
  (blue_shirts * (red_hats + green_hats) * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : 
  num_outfits 6 4 5 7 9 7 6 = 1526 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1129_112913


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1129_112963

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l1129_112963


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1129_112965

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 20) :
  let side_length := perimeter / 4
  let area := side_length * side_length
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1129_112965


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l1129_112930

theorem art_gallery_theorem (total : ℕ) 
  (h1 : total > 0)
  (h2 : (total / 3 : ℚ) = total / 3)  -- Ensures division is exact
  (h3 : ((total / 3) / 6 : ℚ) = (total / 3) / 6)  -- Ensures division is exact
  (h4 : ((2 * total / 3) / 3 : ℚ) = (2 * total / 3) / 3)  -- Ensures division is exact
  (h5 : 2 * (2 * total / 3) / 3 = 1200) :
  total = 2700 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l1129_112930


namespace NUMINAMATH_CALUDE_pie_eaten_after_seven_trips_l1129_112991

def eat_pie (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pie_eaten_after_seven_trips :
  eat_pie 7 = 1093 / 2187 :=
by sorry

end NUMINAMATH_CALUDE_pie_eaten_after_seven_trips_l1129_112991


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l1129_112909

theorem triangle_determinant_zero (A B C : ℝ) (h : A + B + C = π) : 
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.sin A ^ 2, Real.cos A / Real.sin A, Real.sin A],
    ![Real.sin B ^ 2, Real.cos B / Real.sin B, Real.sin B],
    ![Real.sin C ^ 2, Real.cos C / Real.sin C, Real.sin C]
  ]
  Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l1129_112909


namespace NUMINAMATH_CALUDE_circplus_neg_three_eight_l1129_112967

/-- The ⊕ operation for rational numbers -/
def circplus (a b : ℚ) : ℚ := a * b + (a - b)

/-- Theorem stating that (-3) ⊕ 8 = -35 -/
theorem circplus_neg_three_eight : circplus (-3) 8 = -35 := by sorry

end NUMINAMATH_CALUDE_circplus_neg_three_eight_l1129_112967


namespace NUMINAMATH_CALUDE_swimming_pool_width_l1129_112971

/-- Proves that the width of a rectangular swimming pool is 20 feet -/
theorem swimming_pool_width :
  ∀ (length width : ℝ) (water_removed : ℝ) (depth_lowered : ℝ),
    length = 60 →
    water_removed = 4500 →
    depth_lowered = 0.5 →
    water_removed / 7.5 = length * width * depth_lowered →
    width = 20 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_width_l1129_112971


namespace NUMINAMATH_CALUDE_bridget_apples_l1129_112969

theorem bridget_apples : ∃ (x : ℕ), 
  x > 0 ∧ 
  (2 * x) % 3 = 0 ∧ 
  (2 * x) / 3 - 5 = 2 ∧ 
  x = 11 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l1129_112969


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1129_112997

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l1129_112997


namespace NUMINAMATH_CALUDE_prairie_total_area_l1129_112900

/-- The total area of a prairie given the area covered by a dust storm and the area left untouched. -/
theorem prairie_total_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end NUMINAMATH_CALUDE_prairie_total_area_l1129_112900


namespace NUMINAMATH_CALUDE_train_crossing_time_l1129_112995

/- Define the train speed in m/s -/
def train_speed : ℝ := 20

/- Define the time to pass a man on the platform in seconds -/
def time_pass_man : ℝ := 18

/- Define the length of the platform in meters -/
def platform_length : ℝ := 260

/- Calculate the length of the train -/
def train_length : ℝ := train_speed * time_pass_man

/- Calculate the total distance the train needs to travel -/
def total_distance : ℝ := train_length + platform_length

/- Theorem: The time for the train to cross the platform is 31 seconds -/
theorem train_crossing_time : 
  total_distance / train_speed = 31 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1129_112995


namespace NUMINAMATH_CALUDE_equation_solution_l1129_112956

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) := by
  use (-37 : ℚ)
  constructor
  · -- Prove that y = -37 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l1129_112956


namespace NUMINAMATH_CALUDE_ball_game_bill_l1129_112953

theorem ball_game_bill (num_adults num_children : ℕ) 
  (adult_price child_price : ℚ) : 
  num_adults = 10 → 
  num_children = 11 → 
  adult_price = 8 → 
  child_price = 4 → 
  (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price = 124 := by
  sorry

end NUMINAMATH_CALUDE_ball_game_bill_l1129_112953


namespace NUMINAMATH_CALUDE_race_outcomes_five_participants_l1129_112921

/-- The number of different 1st-2nd-3rd place outcomes in a race -/
def raceOutcomes (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 1) * (n - 2)

/-- Theorem: In a race with 5 participants where one must finish first and there are no ties,
    the number of different 1st-2nd-3rd place outcomes is 12. -/
theorem race_outcomes_five_participants :
  raceOutcomes 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_five_participants_l1129_112921


namespace NUMINAMATH_CALUDE_percentage_greater_than_l1129_112910

theorem percentage_greater_than (X Y Z : ℝ) : 
  (X - Y) / (Y + Z) * 100 = 100 * (X - Y) / (Y + Z) :=
by sorry

end NUMINAMATH_CALUDE_percentage_greater_than_l1129_112910


namespace NUMINAMATH_CALUDE_soccer_handshakes_l1129_112915

theorem soccer_handshakes (players_per_team : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  players_per_team = 11 → num_teams = 2 → num_referees = 3 →
  (players_per_team * players_per_team) + (players_per_team * num_teams * num_referees) = 187 :=
by sorry

end NUMINAMATH_CALUDE_soccer_handshakes_l1129_112915


namespace NUMINAMATH_CALUDE_range_of_m_min_distance_to_origin_range_of_slope_l1129_112936

-- Define the circle C
def C (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + m = 0

-- Define point P
def P : ℝ × ℝ := (0, 4)

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x y, C m x y → (P.1 - x)^2 + (P.2 - y)^2 > 0) → 16 < m ∧ m < 25 :=
sorry

-- Theorem 2
theorem min_distance_to_origin (x y : ℝ) :
  C 24 x y → x^2 + y^2 ≥ 16 :=
sorry

-- Theorem 3
theorem range_of_slope (x y : ℝ) :
  C 24 x y → x ≠ 0 → -Real.sqrt 2 / 4 ≤ (y - 4) / x ∧ (y - 4) / x ≤ Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_min_distance_to_origin_range_of_slope_l1129_112936


namespace NUMINAMATH_CALUDE_total_soap_cost_two_years_l1129_112964

/-- Represents the types of soap --/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Returns the price of a given soap type --/
def soapPrice (s : SoapType) : ℚ :=
  match s with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Applies the bulk discount to a given quantity and price --/
def applyDiscount (quantity : ℕ) (price : ℚ) : ℚ :=
  let totalPrice := price * quantity
  if quantity ≥ 10 then totalPrice * (1 - 0.15)
  else if quantity ≥ 7 then totalPrice * (1 - 0.10)
  else if quantity ≥ 4 then totalPrice * (1 - 0.05)
  else totalPrice

/-- Calculates the cost of soap for a given type over 2 years --/
def soapCostTwoYears (s : SoapType) : ℚ :=
  let price := soapPrice s
  applyDiscount 7 price + price

/-- Theorem: The total amount Elias spends on soap in 2 years is $109.50 --/
theorem total_soap_cost_two_years :
  soapCostTwoYears SoapType.Lavender +
  soapCostTwoYears SoapType.Lemon +
  soapCostTwoYears SoapType.Sandalwood = 109.5 := by
  sorry

end NUMINAMATH_CALUDE_total_soap_cost_two_years_l1129_112964


namespace NUMINAMATH_CALUDE_power_function_symmetry_l1129_112947

/-- A function f is a power function if it can be written as f(x) = ax^n for some constant a and real number n. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x, f x = a * x^n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x. -/
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Given that f(x) = (t^2 - t + 1)x^((t+3)/5) is a power function and 
    symmetric about the y-axis, prove that t = 1. -/
theorem power_function_symmetry (t : ℝ) : 
  let f := fun (x : ℝ) ↦ (t^2 - t + 1) * x^((t+3)/5)
  is_power_function f ∧ symmetric_about_y_axis f → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_symmetry_l1129_112947


namespace NUMINAMATH_CALUDE_july_birth_percentage_l1129_112983

/-- The percentage of scientists born in July, given the total number of scientists and the number born in July. -/
theorem july_birth_percentage 
  (total_scientists : ℕ) 
  (july_births : ℕ) 
  (h1 : total_scientists = 200) 
  (h2 : july_births = 17) : 
  (july_births : ℚ) / total_scientists * 100 = 8.5 := by
sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l1129_112983


namespace NUMINAMATH_CALUDE_excellent_scorers_l1129_112944

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

-- Define a function to represent whether a student scores excellent
def scores_excellent : Student → Prop := sorry

-- Define the statements made by each student
def statement_A : Prop := scores_excellent Student.A → scores_excellent Student.B
def statement_B : Prop := scores_excellent Student.B → scores_excellent Student.C
def statement_C : Prop := scores_excellent Student.C → scores_excellent Student.D
def statement_D : Prop := scores_excellent Student.D → scores_excellent Student.E

-- Define a function to count the number of students scoring excellent
def count_excellent : (Student → Prop) → Nat := sorry

-- Theorem statement
theorem excellent_scorers :
  (statement_A ∧ statement_B ∧ statement_C ∧ statement_D) →
  (count_excellent scores_excellent = 3) →
  (scores_excellent Student.C ∧ scores_excellent Student.D ∧ scores_excellent Student.E ∧
   ¬scores_excellent Student.A ∧ ¬scores_excellent Student.B) :=
sorry

end NUMINAMATH_CALUDE_excellent_scorers_l1129_112944


namespace NUMINAMATH_CALUDE_rth_term_of_arithmetic_progression_l1129_112945

def sum_of_n_terms (n : ℕ) : ℕ := 2*n + 3*n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) :
  sum_of_n_terms r - sum_of_n_terms (r - 1) = 3*r^2 + 5*r - 2 :=
by sorry

end NUMINAMATH_CALUDE_rth_term_of_arithmetic_progression_l1129_112945


namespace NUMINAMATH_CALUDE_orange_marbles_count_l1129_112961

theorem orange_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) : 
  total = 24 →
  blue = total / 2 →
  red = 6 →
  orange = total - blue - red →
  orange = 6 := by
sorry

end NUMINAMATH_CALUDE_orange_marbles_count_l1129_112961


namespace NUMINAMATH_CALUDE_pet_calculation_l1129_112986

theorem pet_calculation (taylor_pets : ℕ) (total_pets : ℕ) : 
  taylor_pets = 4 → 
  total_pets = 32 → 
  ∃ (other_friends_pets : ℕ),
    total_pets = taylor_pets + 3 * (2 * taylor_pets) + 2 * other_friends_pets ∧ 
    other_friends_pets = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_calculation_l1129_112986


namespace NUMINAMATH_CALUDE_tony_cheese_purchase_l1129_112937

theorem tony_cheese_purchase (initial_amount : ℕ) (cheese_cost : ℕ) (beef_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 87)
  (h2 : cheese_cost = 7)
  (h3 : beef_cost = 5)
  (h4 : remaining_amount = 61) :
  (initial_amount - remaining_amount - beef_cost) / cheese_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_tony_cheese_purchase_l1129_112937


namespace NUMINAMATH_CALUDE_min_value_bound_l1129_112938

/-- For positive real numbers x and y, (x + 2y)⁺ is always greater than or equal to 9 -/
theorem min_value_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) : (x + 2*y)⁺ ≥ 9 := by
  sorry

#check min_value_bound

end NUMINAMATH_CALUDE_min_value_bound_l1129_112938


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l1129_112931

theorem pyramid_max_volume (a b c h : ℝ) (angle : ℝ) :
  a = 5 ∧ b = 12 ∧ c = 13 →
  a^2 + b^2 = c^2 →
  angle ≥ 30 * π / 180 →
  (∃ (height : ℝ), height > 0 ∧
    (∀ (face_height : ℝ), face_height > 0 →
      Real.cos (Real.arccos (height / face_height)) ≥ Real.cos angle)) →
  (1/3 : ℝ) * (1/2 * a * b) * h ≤ 150 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l1129_112931


namespace NUMINAMATH_CALUDE_curve_intersection_arithmetic_sequence_l1129_112973

/-- Given a curve C: y = 1/x (x > 0) and points A₁(x₁, 0) and A₂(x₂, 0) where x₂ > x₁ > 0,
    perpendicular lines to the x-axis from A₁ and A₂ intersect C at B₁ and B₂.
    The line B₁B₂ intersects the x-axis at A₃(x₃, 0).
    This theorem proves that x₁, x₃/2, x₂ form an arithmetic sequence. -/
theorem curve_intersection_arithmetic_sequence
  (x₁ x₂ : ℝ)
  (h₁ : 0 < x₁)
  (h₂ : x₁ < x₂)
  (x₃ : ℝ)
  (h₃ : x₃ = x₁ + x₂) :
  x₂ - x₃/2 = x₃/2 - x₁ :=
by sorry

end NUMINAMATH_CALUDE_curve_intersection_arithmetic_sequence_l1129_112973


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_as_coefficients_l1129_112955

theorem quadratic_equation_roots_as_coefficients :
  ∀ (A B : ℝ),
  (∀ x : ℝ, x^2 + A*x + B = 0 ↔ x = A ∨ x = B) →
  ((A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_as_coefficients_l1129_112955


namespace NUMINAMATH_CALUDE_joe_paint_usage_l1129_112926

theorem joe_paint_usage (total_paint : ℝ) (second_week_fraction : ℝ) (total_used : ℝ) :
  total_paint = 360 →
  second_week_fraction = 1 / 7 →
  total_used = 128.57 →
  ∃ (first_week_fraction : ℝ),
    first_week_fraction * total_paint +
    second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used ∧
    first_week_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l1129_112926


namespace NUMINAMATH_CALUDE_parabola_vertex_l1129_112989

/-- The vertex of the parabola y = x^2 - 2x + 4 has coordinates (1, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = x^2 - 2*x + 4 → (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    ∀ (x' : ℝ), x'^2 - 2*x' + 4 ≥ k ∧ 
    (x'^2 - 2*x' + 4 = k ↔ x' = h)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1129_112989


namespace NUMINAMATH_CALUDE_triangle_count_l1129_112988

/-- The number of small triangles in the first section -/
def first_section_small : ℕ := 6

/-- The number of small triangles in the additional section -/
def additional_section_small : ℕ := 5

/-- The number of triangles made by combining 2 small triangles in the first section -/
def first_section_combined_2 : ℕ := 4

/-- The number of triangles made by combining 4 small triangles in the first section -/
def first_section_combined_4 : ℕ := 1

/-- The number of combined triangles in the additional section -/
def additional_section_combined : ℕ := 0

/-- The total number of triangles in the figure -/
def total_triangles : ℕ := 16

theorem triangle_count :
  first_section_small + additional_section_small +
  first_section_combined_2 + first_section_combined_4 +
  additional_section_combined = total_triangles := by sorry

end NUMINAMATH_CALUDE_triangle_count_l1129_112988
