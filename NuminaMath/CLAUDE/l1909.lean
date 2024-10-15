import Mathlib

namespace NUMINAMATH_CALUDE_max_popsicles_for_10_dollars_l1909_190981

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize
  | Seven : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3
  | BoxSize.Seven => 4

/-- Returns the number of popsicles in a box given its size -/
def boxPopsicles (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5
  | BoxSize.Seven => 7

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  single : ℕ
  three : ℕ
  five : ℕ
  seven : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * boxCost BoxSize.Single +
  p.three * boxCost BoxSize.Three +
  p.five * boxCost BoxSize.Five +
  p.seven * boxCost BoxSize.Seven

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.single * boxPopsicles BoxSize.Single +
  p.three * boxPopsicles BoxSize.Three +
  p.five * boxPopsicles BoxSize.Five +
  p.seven * boxPopsicles BoxSize.Seven

/-- Theorem: The maximum number of popsicles that can be bought with $10 is 17 -/
theorem max_popsicles_for_10_dollars :
  ∀ p : Purchase, totalCost p ≤ 10 → totalPopsicles p ≤ 17 ∧
  ∃ p : Purchase, totalCost p ≤ 10 ∧ totalPopsicles p = 17 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_for_10_dollars_l1909_190981


namespace NUMINAMATH_CALUDE_production_line_b_units_l1909_190982

theorem production_line_b_units (total_units : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_units = 5000 →
  ratio_a = 1 →
  ratio_b = 2 →
  ratio_c = 2 →
  (total_units * ratio_b) / (ratio_a + ratio_b + ratio_c) = 2000 := by
sorry

end NUMINAMATH_CALUDE_production_line_b_units_l1909_190982


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l1909_190956

/-- Given that z varies inversely as the cube root of x, and z = 2 when x = 8,
    prove that x = 1 when z = 4. -/
theorem inverse_cube_root_relation (z x : ℝ) (h1 : z * x^(1/3) = 2 * 8^(1/3)) :
  z = 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l1909_190956


namespace NUMINAMATH_CALUDE_power_of_81_l1909_190930

theorem power_of_81 : 81^(8/3) = 59049 * (9^(1/3)) := by sorry

end NUMINAMATH_CALUDE_power_of_81_l1909_190930


namespace NUMINAMATH_CALUDE_ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l1909_190985

-- Define a function to check if a number is a difference of two squares
def is_diff_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

-- Theorem stating that 10 cannot be expressed as the difference of two squares
theorem ten_not_diff_of_squares : ¬ is_diff_of_squares 10 :=
sorry

-- Theorems stating that 5, 7, 8, and 9 can be expressed as the difference of two squares
theorem five_is_diff_of_squares : is_diff_of_squares 5 :=
sorry

theorem seven_is_diff_of_squares : is_diff_of_squares 7 :=
sorry

theorem eight_is_diff_of_squares : is_diff_of_squares 8 :=
sorry

theorem nine_is_diff_of_squares : is_diff_of_squares 9 :=
sorry

end NUMINAMATH_CALUDE_ten_not_diff_of_squares_five_is_diff_of_squares_seven_is_diff_of_squares_eight_is_diff_of_squares_nine_is_diff_of_squares_l1909_190985


namespace NUMINAMATH_CALUDE_f_two_zeros_sum_greater_than_two_l1909_190957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (a/2) * x^2 + (a-1) * x

theorem f_two_zeros_sum_greater_than_two (a : ℝ) (h : a > 2) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  f a x₁ = 0 ∧ f a x₂ = 0 ∧
  (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ + x₂ > 2 := by sorry

end NUMINAMATH_CALUDE_f_two_zeros_sum_greater_than_two_l1909_190957


namespace NUMINAMATH_CALUDE_challenge_probabilities_l1909_190989

/-- A challenge with 3 equally difficult questions -/
structure Challenge where
  num_questions : ℕ := 3
  num_chances : ℕ := 3
  correct_prob : ℝ := 0.7

/-- The probability of passing the challenge on the second attempt -/
def prob_pass_second_attempt (c : Challenge) : ℝ :=
  (1 - c.correct_prob) * c.correct_prob

/-- The overall probability of passing the challenge -/
def prob_pass_challenge (c : Challenge) : ℝ :=
  1 - (1 - c.correct_prob) ^ c.num_chances

/-- Theorem stating the probabilities for the given challenge -/
theorem challenge_probabilities (c : Challenge) :
  prob_pass_second_attempt c = 0.21 ∧ prob_pass_challenge c = 0.973 := by
  sorry

#check challenge_probabilities

end NUMINAMATH_CALUDE_challenge_probabilities_l1909_190989


namespace NUMINAMATH_CALUDE_g_values_l1909_190918

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom sum_one : ∀ x, g x + f x = 1
axiom g_odd : ∀ x, g (x + 1) = -g (-x + 1)
axiom f_odd : ∀ x, f (2 - x) = -f (2 + x)

-- Define the theorem
theorem g_values : g 0 = -1 ∧ g 1 = 0 ∧ g 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_values_l1909_190918


namespace NUMINAMATH_CALUDE_homes_cleaned_l1909_190976

theorem homes_cleaned (earning_per_home : ℝ) (total_earned : ℝ) (h1 : earning_per_home = 46.0) (h2 : total_earned = 12696) :
  total_earned / earning_per_home = 276 := by
  sorry

end NUMINAMATH_CALUDE_homes_cleaned_l1909_190976


namespace NUMINAMATH_CALUDE_greatest_a_value_l1909_190979

theorem greatest_a_value (a : ℝ) : 
  (9 * Real.sqrt ((3 * a)^2 + 1^2) - 9 * a^2 - 1) / (Real.sqrt (1 + 3 * a^2) + 2) = 3 →
  a ≤ Real.sqrt (13/3) :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_value_l1909_190979


namespace NUMINAMATH_CALUDE_john_uber_earnings_l1909_190909

/-- Calculates the total money made from Uber before considering depreciation -/
def total_money_before_depreciation (initial_car_value trade_in_value profit_after_depreciation : ℕ) : ℕ :=
  profit_after_depreciation + (initial_car_value - trade_in_value)

/-- Theorem stating that John's total money made from Uber before depreciation is $30,000 -/
theorem john_uber_earnings :
  let initial_car_value : ℕ := 18000
  let trade_in_value : ℕ := 6000
  let profit_after_depreciation : ℕ := 18000
  total_money_before_depreciation initial_car_value trade_in_value profit_after_depreciation = 30000 := by
  sorry

end NUMINAMATH_CALUDE_john_uber_earnings_l1909_190909


namespace NUMINAMATH_CALUDE_polynomial_factor_l1909_190980

theorem polynomial_factor (x : ℝ) :
  ∃ (k : ℝ), (29 * 37 * x^4 + 2 * x^2 + 9) = k * (x^2 - 2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l1909_190980


namespace NUMINAMATH_CALUDE_julia_video_games_fraction_l1909_190900

/-- Given the number of video games owned by Theresa, Julia, and Tory,
    prove that Julia has 1/3 as many video games as Tory. -/
theorem julia_video_games_fraction (theresa julia tory : ℕ) : 
  theresa = 3 * julia + 5 →
  tory = 6 →
  theresa = 11 →
  julia * 3 = tory := by
  sorry

end NUMINAMATH_CALUDE_julia_video_games_fraction_l1909_190900


namespace NUMINAMATH_CALUDE_kim_shoes_problem_l1909_190984

theorem kim_shoes_problem (num_pairs : ℕ) (prob_same_color : ℚ) : 
  num_pairs = 7 →
  prob_same_color = 7692307692307693 / 100000000000000000 →
  (1 : ℚ) / (num_pairs * 2 - 1) = prob_same_color →
  num_pairs * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_kim_shoes_problem_l1909_190984


namespace NUMINAMATH_CALUDE_max_dots_on_surface_l1909_190947

/-- The sum of dots on a standard die -/
def standardDieSum : ℕ := 21

/-- The maximum number of dots visible on a die with 5 visible faces -/
def maxDotsOn5Faces : ℕ := 20

/-- The number of dots visible on a die with 4 visible faces -/
def dotsOn4Faces : ℕ := 14

/-- The maximum number of dots visible on a die with 2 visible faces -/
def maxDotsOn2Faces : ℕ := 11

/-- The number of dice with 5 visible faces -/
def numDice5Faces : ℕ := 6

/-- The number of dice with 4 visible faces -/
def numDice4Faces : ℕ := 5

/-- The number of dice with 2 visible faces -/
def numDice2Faces : ℕ := 2

theorem max_dots_on_surface :
  numDice5Faces * maxDotsOn5Faces +
  numDice4Faces * dotsOn4Faces +
  numDice2Faces * maxDotsOn2Faces = 212 :=
by sorry

end NUMINAMATH_CALUDE_max_dots_on_surface_l1909_190947


namespace NUMINAMATH_CALUDE_friend_bicycles_count_friend_owns_ten_bicycles_l1909_190914

theorem friend_bicycles_count (ignatius_bicycles : ℕ) (tires_per_bicycle : ℕ) 
  (friend_unicycles : ℕ) (friend_tricycles : ℕ) : ℕ :=
  let ignatius_total_tires := ignatius_bicycles * tires_per_bicycle
  let friend_total_tires := 3 * ignatius_total_tires
  let friend_other_tires := friend_unicycles * 1 + friend_tricycles * 3
  let friend_bicycle_tires := friend_total_tires - friend_other_tires
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_ten_bicycles :
  friend_bicycles_count 4 2 1 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_bicycles_count_friend_owns_ten_bicycles_l1909_190914


namespace NUMINAMATH_CALUDE_smallest_k_for_fraction_equation_l1909_190996

theorem smallest_k_for_fraction_equation : 
  (∃ k : ℕ, k > 0 ∧ 
    (∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) ∧ 
  (∀ k : ℕ, k > 0 → k < 1001 → 
    ¬(∃ a b : ℕ, a > 500000 ∧ 
      1 / (a : ℚ) + 1 / ((a + k) : ℚ) = 1 / (b : ℚ))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_fraction_equation_l1909_190996


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1909_190944

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) ≥ 3 + Real.sqrt 3 :=
by sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / b) + (b / c) + (c / a) + Real.sqrt ((a / b)^2 + (b / c)^2 + (c / a)^2) = 3 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1909_190944


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1909_190948

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 4 * Real.sqrt 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  (r = 4 * Real.sqrt 6) ∧ 
  (θ = π / 8) ∧ 
  (r > 0) ∧ 
  (0 ≤ θ) ∧ 
  (θ < 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l1909_190948


namespace NUMINAMATH_CALUDE_units_digit_sum_base9_l1909_190951

/-- The units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := (a + b) % 9

theorem units_digit_sum_base9 :
  unitsDigitBase9 (addBase9 45 76) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_base9_l1909_190951


namespace NUMINAMATH_CALUDE_min_cards_l1909_190921

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_cards : ∃ (n a b c d e : ℕ),
  n = 63 ∧
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧
  (n - a) % 5 = 0 ∧
  (n - a - b) % 3 = 0 ∧
  (n - a - b - c) % 2 = 0 ∧
  n - a - b - c - d = e ∧
  ∀ (m : ℕ), m < n →
    ¬(∃ (a' b' c' d' e' : ℕ),
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧ is_prime e' ∧
      a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > e' ∧
      (m - a') % 5 = 0 ∧
      (m - a' - b') % 3 = 0 ∧
      (m - a' - b' - c') % 2 = 0 ∧
      m - a' - b' - c' - d' = e') :=
by sorry

end NUMINAMATH_CALUDE_min_cards_l1909_190921


namespace NUMINAMATH_CALUDE_union_equality_iff_m_range_l1909_190928

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ 2*m + 1}

theorem union_equality_iff_m_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_m_range_l1909_190928


namespace NUMINAMATH_CALUDE_anya_balloons_count_l1909_190943

def total_balloons : ℕ := 672
def num_colors : ℕ := 4

theorem anya_balloons_count : 
  let balloons_per_color := total_balloons / num_colors
  let anya_balloons := balloons_per_color / 2
  anya_balloons = 84 := by sorry

end NUMINAMATH_CALUDE_anya_balloons_count_l1909_190943


namespace NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1909_190906

theorem boy_scouts_permission_slips 
  (total_permission : Real) 
  (boy_scouts_percentage : Real) 
  (girl_scouts_permission : Real) :
  total_permission = 0.7 →
  boy_scouts_percentage = 0.6 →
  girl_scouts_permission = 0.625 →
  (total_permission - ((1 - boy_scouts_percentage) * girl_scouts_permission)) / boy_scouts_percentage = 0.75 := by
sorry

end NUMINAMATH_CALUDE_boy_scouts_permission_slips_l1909_190906


namespace NUMINAMATH_CALUDE_magnitude_of_a_is_two_l1909_190973

def a (x : ℝ) : Fin 2 → ℝ := ![1, x]
def b (x : ℝ) : Fin 2 → ℝ := ![-1, x]

theorem magnitude_of_a_is_two (x : ℝ) :
  (∀ i : Fin 2, ((2 • a x - b x) • b x) = 0) → 
  Real.sqrt ((a x 0) ^ 2 + (a x 1) ^ 2) = 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_a_is_two_l1909_190973


namespace NUMINAMATH_CALUDE_bennys_work_days_l1909_190931

theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 →
  total_hours = 18 →
  days_worked * hours_per_day = total_hours →
  days_worked = 6 := by
sorry

end NUMINAMATH_CALUDE_bennys_work_days_l1909_190931


namespace NUMINAMATH_CALUDE_remainder_problem_l1909_190972

theorem remainder_problem (n : ℕ) (r₃ r₆ r₉ : ℕ) :
  r₃ < 3 ∧ r₆ < 6 ∧ r₉ < 9 →
  n % 3 = r₃ ∧ n % 6 = r₆ ∧ n % 9 = r₉ →
  r₃ + r₆ + r₉ = 15 →
  n % 18 = 17 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1909_190972


namespace NUMINAMATH_CALUDE_train_crossing_time_l1909_190912

/-- Proves that a train 300 meters long, traveling at 90 km/hr, will take 12 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 300 →
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1909_190912


namespace NUMINAMATH_CALUDE_all_two_digit_numbers_appear_l1909_190974

/-- Represents a sequence of numbers from 1 to 1,000,000 in arbitrary order -/
def ArbitrarySequence := Fin 1000000 → Fin 1000000

/-- Represents a two-digit number (from 10 to 99) -/
def TwoDigitNumber := Fin 90

/-- A function that checks if a given two-digit number appears in the sequence when cut into two-digit pieces -/
def appearsInSequence (seq : ArbitrarySequence) (n : TwoDigitNumber) : Prop :=
  ∃ i : Fin 999999, (seq i).val / 100 % 100 = n.val + 10 ∨ (seq i).val % 100 = n.val + 10

/-- The main theorem statement -/
theorem all_two_digit_numbers_appear (seq : ArbitrarySequence) :
  ∀ n : TwoDigitNumber, appearsInSequence seq n :=
sorry

end NUMINAMATH_CALUDE_all_two_digit_numbers_appear_l1909_190974


namespace NUMINAMATH_CALUDE_quadratic_and_squared_equation_solutions_l1909_190992

theorem quadratic_and_squared_equation_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) ∧ x₁ = 1/2 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, ((y₁ - 2)^2 = (2 * y₁ + 3)^2) ∧ ((y₂ - 2)^2 = (2 * y₂ + 3)^2) ∧ y₁ = -5 ∧ y₂ = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_squared_equation_solutions_l1909_190992


namespace NUMINAMATH_CALUDE_symmetric_point_xoz_plane_l1909_190940

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOz plane in three-dimensional space -/
def xOzPlane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz_plane :
  let A : Point3D := ⟨-1, 2, 3⟩
  let Q : Point3D := symmetricPointXOZ A
  Q = ⟨-1, -2, 3⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoz_plane_l1909_190940


namespace NUMINAMATH_CALUDE_power_of_point_formula_l1909_190937

/-- The power of a point with respect to a circle -/
def power_of_point (d R : ℝ) : ℝ := d^2 - R^2

/-- Theorem: The power of a point with respect to a circle is d^2 - R^2,
    where d is the distance from the point to the center of the circle,
    and R is the radius of the circle. -/
theorem power_of_point_formula (d R : ℝ) :
  power_of_point d R = d^2 - R^2 := by sorry

end NUMINAMATH_CALUDE_power_of_point_formula_l1909_190937


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l1909_190942

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  dist_sum : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def computeEllipseParams (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem: for the given ellipse, the sum of its parameters is 18 -/
theorem ellipse_param_sum :
  let e := Ellipse.mk (4, 2) (10, 2) 10
  let params := computeEllipseParams e
  params.h + params.k + params.a + params.b = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l1909_190942


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_l1909_190936

theorem square_reciprocal_sum (m : ℝ) (h : m + 1/m = 5) : 
  m^2 + 1/m^2 + 4 = 27 := by
sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_l1909_190936


namespace NUMINAMATH_CALUDE_john_soup_vegetables_l1909_190964

/-- Represents the weights of vegetables used in John's soup recipe --/
structure SoupVegetables where
  carrots : ℝ
  potatoes : ℝ
  bell_peppers : ℝ

/-- Calculates the total weight of vegetables used in the soup --/
def total_vegetable_weight (v : SoupVegetables) : ℝ :=
  v.carrots + v.potatoes + v.bell_peppers

/-- Represents John's soup recipe --/
structure SoupRecipe where
  beef_bought : ℝ
  beef_unused : ℝ
  vegetables : SoupVegetables

/-- Theorem stating the correct weights of vegetables in John's soup --/
theorem john_soup_vegetables (recipe : SoupRecipe) : 
  recipe.beef_bought = 4 ∧ 
  recipe.beef_unused = 1 ∧ 
  total_vegetable_weight recipe.vegetables = 2 * (recipe.beef_bought - recipe.beef_unused) ∧
  recipe.vegetables.carrots = recipe.vegetables.potatoes ∧
  recipe.vegetables.bell_peppers = 2 * recipe.vegetables.carrots →
  recipe.vegetables = SoupVegetables.mk 1.5 1.5 3 := by
  sorry


end NUMINAMATH_CALUDE_john_soup_vegetables_l1909_190964


namespace NUMINAMATH_CALUDE_no_consecutive_product_for_nine_power_minus_seven_l1909_190932

theorem no_consecutive_product_for_nine_power_minus_seven :
  ∀ n : ℕ, ¬∃ k : ℕ, 9^n - 7 = k * (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_product_for_nine_power_minus_seven_l1909_190932


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1909_190955

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 2^8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1909_190955


namespace NUMINAMATH_CALUDE_correct_selection_count_l1909_190988

/-- The number of ways to select 4 students from 7, including both boys and girls -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

/-- Theorem stating the correct number of selections -/
theorem correct_selection_count :
  select_students 7 4 3 4 = 34 := by
  sorry

#eval select_students 7 4 3 4

end NUMINAMATH_CALUDE_correct_selection_count_l1909_190988


namespace NUMINAMATH_CALUDE_perimeter_invariant_under_translation_l1909_190934

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a hexagon formed by the intersection of two equilateral triangles -/
structure IntersectionHexagon where
  triangle1 : EquilateralTriangle
  triangle2 : EquilateralTriangle

/-- Calculates the perimeter of the intersection hexagon -/
def perimeter (h : IntersectionHexagon) : ℝ :=
  sorry

/-- Represents a parallel translation of a triangle -/
def parallelTranslation (t : EquilateralTriangle) (v : ℝ × ℝ) : EquilateralTriangle :=
  sorry

/-- The theorem stating that the perimeter remains constant under parallel translation -/
theorem perimeter_invariant_under_translation 
  (h : IntersectionHexagon) 
  (v : ℝ × ℝ) 
  (h' : IntersectionHexagon := ⟨h.triangle1, parallelTranslation h.triangle2 v⟩) : 
  perimeter h = perimeter h' :=
sorry

end NUMINAMATH_CALUDE_perimeter_invariant_under_translation_l1909_190934


namespace NUMINAMATH_CALUDE_inequality_proof_l1909_190950

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1909_190950


namespace NUMINAMATH_CALUDE_water_level_rise_l1909_190960

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise 
  (cube_edge : ℝ) 
  (vessel_length : ℝ) 
  (vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 5) 
  (h_vessel_length : vessel_length = 10) 
  (h_vessel_width : vessel_width = 5) : 
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l1909_190960


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l1909_190903

theorem no_positive_integer_solutions :
  ¬ ∃ (n : ℕ+) (p : ℕ), Prime p ∧ n.val^2 - 47*n.val + 660 = p := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l1909_190903


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_5_256_minus_1_l1909_190952

theorem largest_power_of_two_dividing_5_256_minus_1 :
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) →
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) ∧
  (∀ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n → n = 10) :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_5_256_minus_1_l1909_190952


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l1909_190971

theorem pizza_order_theorem : 
  (1 : ℚ) / 2 + (1 : ℚ) / 3 + (1 : ℚ) / 6 = 1 := by sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l1909_190971


namespace NUMINAMATH_CALUDE_eighth_prime_is_19_l1909_190978

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Theorem: The 8th prime number is 19 -/
theorem eighth_prime_is_19 : nthPrime 8 = 19 := by sorry

end NUMINAMATH_CALUDE_eighth_prime_is_19_l1909_190978


namespace NUMINAMATH_CALUDE_solution_values_l1909_190965

def has_twenty_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 4 * y + z = n) 
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 20

theorem solution_values (n : ℕ+) (h : has_twenty_solutions n) : n = 21 ∨ n = 22 :=
sorry

end NUMINAMATH_CALUDE_solution_values_l1909_190965


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1909_190915

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) → 
  c * b = 20 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1909_190915


namespace NUMINAMATH_CALUDE_stream_speed_l1909_190908

/-- Proves that the speed of the stream is 4 km/hr given the boat's speed in still water and its downstream travel details -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 24 →
  distance = 140 →
  time = 5 →
  (boat_speed + (distance / time - boat_speed)) = 4 := by
sorry


end NUMINAMATH_CALUDE_stream_speed_l1909_190908


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l1909_190967

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

theorem parabola_point_coordinates (p : Parabola) (F : ℝ × ℝ) (A : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  F = (1, 0) →
  (A.x, A.y) • (1 - A.x, -A.y) = -4 →
  (A.x = 1 ∧ A.y = 2) ∨ (A.x = 1 ∧ A.y = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l1909_190967


namespace NUMINAMATH_CALUDE_light_path_in_cube_l1909_190993

theorem light_path_in_cube (cube_side : ℝ) (reflect_point_dist1 : ℝ) (reflect_point_dist2 : ℝ) :
  cube_side = 12 ∧ reflect_point_dist1 = 7 ∧ reflect_point_dist2 = 5 →
  ∃ (m n : ℕ), 
    (m = 12 ∧ n = 218) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ n)) ∧
    (m * Real.sqrt n = 12 * cube_side) :=
by sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l1909_190993


namespace NUMINAMATH_CALUDE_prob_at_least_75_cents_is_correct_l1909_190954

-- Define the coin types and their quantities
structure CoinBox :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

-- Define the function to calculate the total number of coins
def totalCoins (box : CoinBox) : Nat :=
  box.pennies + box.nickels + box.dimes + box.quarters

-- Define the function to calculate the number of ways to choose 7 coins
def waysToChoose7 (box : CoinBox) : Nat :=
  Nat.choose (totalCoins box) 7

-- Define the probability of drawing coins worth at least 75 cents
def probAtLeast75Cents (box : CoinBox) : Rat :=
  2450 / waysToChoose7 box

-- State the theorem
theorem prob_at_least_75_cents_is_correct (box : CoinBox) :
  box.pennies = 4 ∧ box.nickels = 5 ∧ box.dimes = 7 ∧ box.quarters = 3 →
  probAtLeast75Cents box = 2450 / 50388 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_75_cents_is_correct_l1909_190954


namespace NUMINAMATH_CALUDE_expected_digits_20_sided_die_l1909_190938

/-- The expected number of digits when rolling a fair 20-sided die with numbers 1 to 20 -/
theorem expected_digits_20_sided_die : 
  let die_faces : Finset ℕ := Finset.range 20
  let one_digit_count : ℕ := (die_faces.filter (λ n => n < 10)).card
  let two_digit_count : ℕ := (die_faces.filter (λ n => n ≥ 10)).card
  let total_faces : ℕ := die_faces.card
  let expected_value : ℚ := (one_digit_count * 1 + two_digit_count * 2) / total_faces
  expected_value = 31 / 20 := by
sorry

end NUMINAMATH_CALUDE_expected_digits_20_sided_die_l1909_190938


namespace NUMINAMATH_CALUDE_age_difference_l1909_190986

-- Define the ages
def katie_daughter_age : ℕ := 12
def lavinia_daughter_age : ℕ := katie_daughter_age - 10
def lavinia_son_age : ℕ := 2 * katie_daughter_age

-- Theorem statement
theorem age_difference : lavinia_son_age - lavinia_daughter_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1909_190986


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1909_190968

/-- The equation of a potential ellipse -/
def is_potential_ellipse (m n x y : ℝ) : Prop :=
  x^2 / m + y^2 / n = 1

/-- The condition mn > 0 -/
def condition_mn_positive (m n : ℝ) : Prop :=
  m * n > 0

/-- Definition of an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem condition_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → condition_mn_positive m n) ∧
  (∃ m n : ℝ, condition_mn_positive m n ∧ ¬is_ellipse m n) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1909_190968


namespace NUMINAMATH_CALUDE_croissant_price_is_three_l1909_190962

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 50
def num_quiches : ℕ := 2
def price_per_quiche : ℝ := 15
def num_croissants : ℕ := 6
def num_biscuits : ℕ := 6
def price_per_biscuit : ℝ := 2
def final_price : ℝ := 54

def total_cost (price_per_croissant : ℝ) : ℝ :=
  num_quiches * price_per_quiche + 
  num_croissants * price_per_croissant + 
  num_biscuits * price_per_biscuit

theorem croissant_price_is_three :
  ∃ (price_per_croissant : ℝ),
    price_per_croissant = 3 ∧
    total_cost price_per_croissant > discount_threshold ∧
    (1 - discount_rate) * total_cost price_per_croissant = final_price :=
sorry

end NUMINAMATH_CALUDE_croissant_price_is_three_l1909_190962


namespace NUMINAMATH_CALUDE_minimum_apples_in_basket_l1909_190913

theorem minimum_apples_in_basket (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) → n ≥ 62 :=
by sorry

end NUMINAMATH_CALUDE_minimum_apples_in_basket_l1909_190913


namespace NUMINAMATH_CALUDE_brown_children_divisibility_l1909_190935

theorem brown_children_divisibility :
  ∃! n : ℕ, n ∈ Finset.range 10 ∧ ¬(7773 % n = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_brown_children_divisibility_l1909_190935


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_11_l1909_190953

theorem closest_integer_to_sqrt_11 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 11| ≤ |m - Real.sqrt 11| ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_11_l1909_190953


namespace NUMINAMATH_CALUDE_jacob_calorie_consumption_l1909_190924

/-- Jacob's calorie consumption problem -/
theorem jacob_calorie_consumption (planned_max : ℕ) (breakfast lunch dinner : ℕ) 
  (h1 : planned_max < 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : dinner = 1100) :
  breakfast + lunch + dinner - planned_max = 600 :=
by sorry

end NUMINAMATH_CALUDE_jacob_calorie_consumption_l1909_190924


namespace NUMINAMATH_CALUDE_max_automobiles_on_ferry_l1909_190958

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the minimum weight of an automobile in pounds -/
def min_auto_weight : ℝ := 1600

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating the maximum number of automobiles that can be loaded onto the ferry -/
theorem max_automobiles_on_ferry :
  ⌊(ferry_capacity * tons_to_pounds) / min_auto_weight⌋ = 62 := by
  sorry

end NUMINAMATH_CALUDE_max_automobiles_on_ferry_l1909_190958


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1909_190920

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 27 = (x + n)^2 + 3) → 
  b = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1909_190920


namespace NUMINAMATH_CALUDE_distance_to_origin_l1909_190925

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_origin_l1909_190925


namespace NUMINAMATH_CALUDE_samantha_bus_time_l1909_190916

/-- Calculates the time Samantha spends on the bus given her schedule --/
theorem samantha_bus_time :
  let leave_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes
  let return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
  let total_away_time : Nat := return_time - leave_time
  let class_time : Nat := 8 * 45  -- 8 classes of 45 minutes each
  let lunch_time : Nat := 40
  let extracurricular_time : Nat := 90
  let total_school_time : Nat := class_time + lunch_time + extracurricular_time
  let bus_time : Nat := total_away_time - total_school_time
  bus_time = 110 := by sorry

end NUMINAMATH_CALUDE_samantha_bus_time_l1909_190916


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1909_190933

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => x * (2 * x + 4) - (10 + 5 * x)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = 5/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1909_190933


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1909_190991

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1909_190991


namespace NUMINAMATH_CALUDE_square_of_triple_l1909_190904

theorem square_of_triple (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_triple_l1909_190904


namespace NUMINAMATH_CALUDE_mass_of_apples_left_correct_l1909_190975

/-- Calculates the mass of apples left after sales -/
def mass_of_apples_left (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) : ℕ :=
  (kidney_apples + golden_apples + canada_apples) - apples_sold

/-- Proves that the mass of apples left is correct given the initial masses and the mass of apples sold -/
theorem mass_of_apples_left_correct 
  (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) :
  mass_of_apples_left kidney_apples golden_apples canada_apples apples_sold =
  (kidney_apples + golden_apples + canada_apples) - apples_sold :=
by
  sorry

/-- Verifies the specific case in the problem -/
example : mass_of_apples_left 23 37 14 36 = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_mass_of_apples_left_correct_l1909_190975


namespace NUMINAMATH_CALUDE_gas_consumption_reduction_l1909_190946

theorem gas_consumption_reduction (initial_price : ℝ) (initial_consumption : ℝ) 
  (h1 : initial_price > 0) (h2 : initial_consumption > 0) :
  let price_after_increases := initial_price * 1.3 * 1.2
  let new_consumption := initial_consumption * initial_price / price_after_increases
  let reduction_percentage := (initial_consumption - new_consumption) / initial_consumption * 100
  reduction_percentage = (1 - 1 / (1.3 * 1.2)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_gas_consumption_reduction_l1909_190946


namespace NUMINAMATH_CALUDE_smallest_value_of_roots_sum_l1909_190917

/-- 
Given a quadratic equation x^2 - t*x + q with roots α and β,
where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
the smallest possible value of 1/α^2011 + 1/β^2011 is 2.
-/
theorem smallest_value_of_roots_sum (t q α β : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → α^n + β^n = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  (∀ x : ℝ, x^2 - t*x + q = 0 → x = α ∨ x = β) →
  (1 / α^2011 + 1 / β^2011) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_roots_sum_l1909_190917


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l1909_190997

theorem smaller_circle_radius (R : ℝ) (r : ℝ) : 
  R = 12 →  -- Larger circle radius is 12 meters
  4 * (2 * r) = 2 * R →  -- Four smaller circles' diameters equal larger circle's diameter
  r = 3  -- Radius of smaller circle is 3 meters
:= by sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l1909_190997


namespace NUMINAMATH_CALUDE_car_speed_problem_l1909_190923

/-- Given a car traveling for two hours with an average speed of 95 km/h
    and a second hour speed of 70 km/h, prove that the speed in the first hour is 120 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 70 ∧ average_speed = 95 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1909_190923


namespace NUMINAMATH_CALUDE_blank_expression_proof_l1909_190927

theorem blank_expression_proof (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end NUMINAMATH_CALUDE_blank_expression_proof_l1909_190927


namespace NUMINAMATH_CALUDE_solve_for_a_l1909_190990

theorem solve_for_a : ∃ a : ℝ, 
  (2 * 1 - a * (-1) = 3) ∧ (a = 1) := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l1909_190990


namespace NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l1909_190987

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 50 is the smallest positive integer n for which f(n) = 3 -/
theorem smallest_n_with_three_pairs : ∀ k : ℕ, 0 < k → k < 50 → f k ≠ 3 ∧ f 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l1909_190987


namespace NUMINAMATH_CALUDE_sprinkle_cans_remaining_l1909_190922

theorem sprinkle_cans_remaining (initial : ℕ) (final : ℕ) 
  (h1 : initial = 12) 
  (h2 : final = initial / 2 - 3) : 
  final = 3 := by
  sorry

end NUMINAMATH_CALUDE_sprinkle_cans_remaining_l1909_190922


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l1909_190969

theorem cos_20_minus_cos_40 : Real.cos (20 * Real.pi / 180) - Real.cos (40 * Real.pi / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l1909_190969


namespace NUMINAMATH_CALUDE_min_value_theorem_l1909_190939

theorem min_value_theorem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a + 20 * b = 2) (h2 : c + 20 * d = 2) :
  (1 / a + 1 / (b * c * d)) ≥ 441 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1909_190939


namespace NUMINAMATH_CALUDE_original_number_problem_l1909_190905

theorem original_number_problem : ∃ x : ℕ, x / 3 = 42 ∧ x = 126 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l1909_190905


namespace NUMINAMATH_CALUDE_frank_cookie_fraction_l1909_190983

/-- Given the number of cookies for Millie, calculate Mike's cookies -/
def mikeCookies (millieCookies : ℕ) : ℕ := 3 * millieCookies

/-- Calculate the fraction of Frank's cookies compared to Mike's -/
def frankFraction (frankCookies millieCookies : ℕ) : ℚ :=
  frankCookies / (mikeCookies millieCookies)

/-- Theorem: Frank's fraction of cookies compared to Mike's is 1/4 -/
theorem frank_cookie_fraction :
  frankFraction 3 4 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_cookie_fraction_l1909_190983


namespace NUMINAMATH_CALUDE_sum_18_probability_l1909_190966

/-- A fair coin with sides labeled 5 and 15 -/
inductive Coin
| Five : Coin
| Fifteen : Coin

/-- A standard six-sided die -/
inductive Die
| One : Die
| Two : Die
| Three : Die
| Four : Die
| Five : Die
| Six : Die

/-- The probability of getting a sum of 18 when flipping the coin and rolling the die -/
def prob_sum_18 : ℚ :=
  1 / 12

/-- Theorem stating that the probability of getting a sum of 18 is 1/12 -/
theorem sum_18_probability : prob_sum_18 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_18_probability_l1909_190966


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1909_190995

-- Define the radius of the large circle
def R : ℝ := 6

-- Define the centers of the smaller circles
structure Center where
  x : ℝ
  y : ℝ

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : Center
  B : Center
  C : Center
  D : Center

-- Define the condition that the circles touch
def circles_touch (q : Quadrilateral) : Prop := sorry

-- Define the condition that A and C touch at the center of the large circle
def AC_touch_center (q : Quadrilateral) : Prop := sorry

-- Define the area of the quadrilateral
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area (q : Quadrilateral) :
  circles_touch q →
  AC_touch_center q →
  area q = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1909_190995


namespace NUMINAMATH_CALUDE_sum_first_25_odd_numbers_l1909_190907

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  n * n

/-- The 25th odd number -/
def last_odd_number (n : ℕ) : ℕ :=
  2 * n - 1

theorem sum_first_25_odd_numbers :
  sum_odd_numbers 25 = 625 :=
sorry

end NUMINAMATH_CALUDE_sum_first_25_odd_numbers_l1909_190907


namespace NUMINAMATH_CALUDE_abs_diff_given_prod_and_sum_l1909_190901

theorem abs_diff_given_prod_and_sum (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_given_prod_and_sum_l1909_190901


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1909_190945

theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 49 = (x + b)^2 + c) → b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l1909_190945


namespace NUMINAMATH_CALUDE_quadruple_sum_square_l1909_190959

theorem quadruple_sum_square (a b c d m n : ℕ+) : 
  a^2 + b^2 + c^2 + d^2 = 1989 →
  a + b + c + d = m^2 →
  max a (max b (max c d)) = n^2 →
  m = 9 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_quadruple_sum_square_l1909_190959


namespace NUMINAMATH_CALUDE_num_regions_correct_l1909_190999

/-- A structure representing a collection of planes in 3D space -/
structure PlaneCollection where
  n : ℕ
  intersection_of_two : ∀ p q : Fin n, p ≠ q → Line
  intersection_of_three : ∀ p q r : Fin n, p ≠ q ∧ q ≠ r ∧ p ≠ r → Point
  no_four_intersect : ∀ p q r s : Fin n, p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ p ≠ r ∧ p ≠ s ∧ q ≠ s → ¬ Point

/-- The number of non-overlapping regions created by n planes -/
def num_regions (pc : PlaneCollection) : ℕ :=
  (pc.n^3 + 5*pc.n + 6) / 6

/-- Theorem stating that the number of regions is correct -/
theorem num_regions_correct (pc : PlaneCollection) :
  num_regions pc = (pc.n^3 + 5*pc.n + 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_num_regions_correct_l1909_190999


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_quadratic_inequality_condition_l1909_190929

-- Statement 1
theorem necessary_not_sufficient_condition (x : ℝ) :
  (x + |x| > 0 → x ≠ 0) ∧ ¬(x ≠ 0 → x + |x| > 0) := by sorry

-- Statement 2
theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c ≤ 0) ↔ 
  (∀ x : ℝ, a*x^2 + b*x + c ≥ 0) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_quadratic_inequality_condition_l1909_190929


namespace NUMINAMATH_CALUDE_perpendicular_lengths_determine_side_length_l1909_190977

/-- An equilateral triangle with a point inside and perpendiculars to the sides -/
structure EquilateralTriangleWithPoint where
  -- The side length of the equilateral triangle
  side_length : ℝ
  -- The lengths of the perpendiculars from the interior point to the sides
  perp_length_1 : ℝ
  perp_length_2 : ℝ
  perp_length_3 : ℝ
  -- Ensure the triangle is equilateral and the point is inside
  side_length_pos : 0 < side_length
  perp_lengths_pos : 0 < perp_length_1 ∧ 0 < perp_length_2 ∧ 0 < perp_length_3
  perp_sum_bound : perp_length_1 + perp_length_2 + perp_length_3 < side_length * (3 / 2)

/-- The theorem stating the relationship between the perpendicular lengths and the side length -/
theorem perpendicular_lengths_determine_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.perp_length_1 = 2) 
  (h2 : t.perp_length_2 = 3) 
  (h3 : t.perp_length_3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lengths_determine_side_length_l1909_190977


namespace NUMINAMATH_CALUDE_davids_english_marks_l1909_190910

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem stating that David's marks in English are 76 --/
theorem davids_english_marks (marks : Marks) :
  marks.mathematics = 65 →
  marks.physics = 82 →
  marks.chemistry = 67 →
  marks.biology = 85 →
  average [marks.english, marks.mathematics, marks.physics, marks.chemistry, marks.biology] = 75 →
  marks.english = 76 := by
  sorry


end NUMINAMATH_CALUDE_davids_english_marks_l1909_190910


namespace NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1909_190919

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Define what it means for two angles of a triangle to be equal
def twoAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem contrapositive_isosceles_angles (t : Triangle) :
  (¬(isIsosceles t) → ¬(twoAnglesEqual t)) ↔ (twoAnglesEqual t → isIsosceles t) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_isosceles_angles_l1909_190919


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1909_190994

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 84 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1909_190994


namespace NUMINAMATH_CALUDE_unique_right_angle_point_implies_radius_one_l1909_190949

-- Define the circle C
def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define points A and B
def point_A : ℝ × ℝ := (-4, 0)
def point_B : ℝ × ℝ := (4, 0)

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let PA := (P.1 - point_A.1, P.2 - point_A.2)
  let PB := (P.1 - point_B.1, P.2 - point_B.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0

-- Main theorem
theorem unique_right_angle_point_implies_radius_one (r : ℝ) :
  (∃! P, P ∈ circle_C r ∧ right_angle P) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_angle_point_implies_radius_one_l1909_190949


namespace NUMINAMATH_CALUDE_percentage_problem_l1909_190998

theorem percentage_problem (P : ℝ) : 
  (0.20 * 30 = P / 100 * 16 + 2) → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1909_190998


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1909_190902

def setA : Set ℤ := {x | |x| < 3}
def setB : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  setA ∩ setB = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1909_190902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1909_190963

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 2 = -1 →
  a 3 = 4 →
  a 4 + a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1909_190963


namespace NUMINAMATH_CALUDE_base_of_second_term_l1909_190941

theorem base_of_second_term (e : ℕ) (base : ℚ) 
  (h1 : e = 35)
  (h2 : (1/5 : ℚ)^e * base^18 = 1/(2*(10^35))) :
  base = 1/4 := by sorry

end NUMINAMATH_CALUDE_base_of_second_term_l1909_190941


namespace NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1909_190926

theorem sum_and_ratio_implies_difference (x y : ℝ) : 
  x + y = 540 → x / y = 0.75 → y - x = 77.143 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_implies_difference_l1909_190926


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l1909_190970

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

-- Define the number 2102012 in base 3
def number_base_3 : List Nat := [2, 1, 0, 2, 0, 1, 2]

-- Convert the base 3 number to decimal
def number_decimal : Nat := base_3_to_decimal number_base_3

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number_decimal ∧ 
  ∀ (q : Nat), Nat.Prime q → q ∣ number_decimal → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l1909_190970


namespace NUMINAMATH_CALUDE_password_count_l1909_190961

/-- The number of possible values for the last two digits of a birth year. -/
def year_choices : ℕ := 100

/-- The number of possible values for the birth month. -/
def month_choices : ℕ := 12

/-- The number of possible values for the birth date. -/
def day_choices : ℕ := 31

/-- The total number of possible six-digit passwords. -/
def total_passwords : ℕ := year_choices * month_choices * day_choices

theorem password_count : total_passwords = 37200 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l1909_190961


namespace NUMINAMATH_CALUDE_volume_cube_sphere_region_l1909_190911

/-- The volume of the region within a cube of side length 4 cm, outside an inscribed sphere
    tangent to the cube, and closest to one vertex of the cube. -/
theorem volume_cube_sphere_region (π : ℝ) (h : π = Real.pi) :
  let a : ℝ := 4
  let cube_volume : ℝ := a ^ 3
  let sphere_radius : ℝ := a / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  let outside_sphere_volume : ℝ := cube_volume - sphere_volume
  let region_volume : ℝ := (1 / 8) * outside_sphere_volume
  region_volume = 8 * (1 - π / 6) :=
by sorry

end NUMINAMATH_CALUDE_volume_cube_sphere_region_l1909_190911
