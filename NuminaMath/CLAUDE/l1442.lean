import Mathlib

namespace NUMINAMATH_CALUDE_arrangements_eq_two_pow_l1442_144219

/-- The number of arrangements of the sequence 1, 2, ..., n, where each number
    is either strictly greater than all the numbers before it or strictly less
    than all the numbers before it. -/
def arrangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * arrangements (n - 1)

/-- Theorem stating that the number of arrangements for n numbers is 2^(n-1) -/
theorem arrangements_eq_two_pow (n : ℕ) : arrangements n = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_two_pow_l1442_144219


namespace NUMINAMATH_CALUDE_project_choices_l1442_144204

/-- The number of projects available to choose from -/
def num_projects : ℕ := 5

/-- The number of students choosing projects -/
def num_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- The main theorem stating the number of ways students can choose projects -/
theorem project_choices : 
  (choose num_students 2) * (permute num_projects 3) + (permute num_projects num_students) = 480 :=
sorry

end NUMINAMATH_CALUDE_project_choices_l1442_144204


namespace NUMINAMATH_CALUDE_sphere_radius_is_four_l1442_144224

/-- Represents a truncated cone with given dimensions and a tangent sphere -/
structure TruncatedConeWithSphere where
  baseRadius : ℝ
  topRadius : ℝ
  height : ℝ
  sphereRadius : ℝ

/-- Checks if the given dimensions satisfy the conditions for a truncated cone with a tangent sphere -/
def isValidConfiguration (cone : TruncatedConeWithSphere) : Prop :=
  cone.baseRadius > cone.topRadius ∧
  cone.height > 0 ∧
  cone.sphereRadius > 0 ∧
  -- The sphere is tangent to the top, bottom, and lateral surface
  cone.sphereRadius = cone.height - Real.sqrt ((cone.baseRadius - cone.topRadius)^2 + cone.height^2)

/-- Theorem stating that for a truncated cone with given dimensions and a tangent sphere, the radius of the sphere is 4 -/
theorem sphere_radius_is_four :
  ∀ (cone : TruncatedConeWithSphere),
    cone.baseRadius = 24 ∧
    cone.topRadius = 6 ∧
    cone.height = 20 ∧
    isValidConfiguration cone →
    cone.sphereRadius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_four_l1442_144224


namespace NUMINAMATH_CALUDE_square_expression_l1442_144255

theorem square_expression (a b : ℝ) (square : ℝ) :
  square * (2 * a * b) = 4 * a^2 * b → square = 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_square_expression_l1442_144255


namespace NUMINAMATH_CALUDE_mrs_hilt_books_read_l1442_144213

def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

theorem mrs_hilt_books_read :
  total_chapters_read / chapters_per_book = 4 := by sorry

end NUMINAMATH_CALUDE_mrs_hilt_books_read_l1442_144213


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1442_144256

theorem cube_volume_from_space_diagonal :
  ∀ (s : ℝ), s > 0 → s * Real.sqrt 3 = 10 * Real.sqrt 3 → s^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l1442_144256


namespace NUMINAMATH_CALUDE_pats_stickers_l1442_144249

/-- Pat's sticker problem -/
theorem pats_stickers (initial_stickers earned_stickers : ℕ) 
  (h1 : initial_stickers = 39)
  (h2 : earned_stickers = 22) :
  initial_stickers + earned_stickers = 61 :=
by sorry

end NUMINAMATH_CALUDE_pats_stickers_l1442_144249


namespace NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l1442_144253

/-- Proves the minimum speed required for the second person to arrive earlier -/
theorem min_speed_to_arrive_earlier
  (distance : ℝ)
  (speed_A : ℝ)
  (delay : ℝ)
  (h_distance : distance = 180)
  (h_speed_A : speed_A = 30)
  (h_delay : delay = 2) :
  ∀ speed_B : ℝ, speed_B > 45 →
    distance / speed_B + delay < distance / speed_A :=
by sorry

end NUMINAMATH_CALUDE_min_speed_to_arrive_earlier_l1442_144253


namespace NUMINAMATH_CALUDE_max_triples_value_l1442_144275

/-- The size of the square table -/
def n : ℕ := 999

/-- Represents the color of a cell in the table -/
inductive CellColor
| White
| Red

/-- Represents a cell in the table -/
structure Cell where
  row : Fin n
  col : Fin n

/-- Represents the coloring of the table -/
def TableColoring := Fin n → Fin n → CellColor

/-- Counts the number of valid triples for a given table coloring -/
def countTriples (coloring : TableColoring) : ℕ := sorry

/-- The maximum number of valid triples possible -/
def maxTriples : ℕ := (4 * n^4) / 27

/-- Theorem stating that the maximum number of valid triples is (4 * 999⁴) / 27 -/
theorem max_triples_value :
  ∀ (coloring : TableColoring), countTriples coloring ≤ maxTriples :=
by sorry

end NUMINAMATH_CALUDE_max_triples_value_l1442_144275


namespace NUMINAMATH_CALUDE_not_square_sum_ceiling_l1442_144285

theorem not_square_sum_ceiling (a b : ℕ+) : ¬∃ (n : ℕ), (n : ℝ)^2 = (a : ℝ)^2 + ⌈(4 * (a : ℝ)^2) / (b : ℝ)⌉ := by
  sorry

end NUMINAMATH_CALUDE_not_square_sum_ceiling_l1442_144285


namespace NUMINAMATH_CALUDE_expected_coffee_tea_difference_l1442_144203

/-- Represents the outcome of rolling a fair eight-sided die -/
inductive DieRoll
  | one | two | three | four | five | six | seven | eight

/-- Represents the drink Alice chooses based on her die roll -/
inductive Drink
  | coffee | tea | juice

/-- Function that determines the drink based on the die roll -/
def chooseDrink (roll : DieRoll) : Drink :=
  match roll with
  | DieRoll.one => Drink.juice
  | DieRoll.two => Drink.coffee
  | DieRoll.three => Drink.tea
  | DieRoll.four => Drink.coffee
  | DieRoll.five => Drink.tea
  | DieRoll.six => Drink.coffee
  | DieRoll.seven => Drink.tea
  | DieRoll.eight => Drink.coffee

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- Theorem stating the expected difference between coffee and tea days -/
theorem expected_coffee_tea_difference :
  let p_coffee : ℚ := 1/2
  let p_tea : ℚ := 3/8
  let expected_coffee_days : ℚ := p_coffee * daysInYear
  let expected_tea_days : ℚ := p_tea * daysInYear
  let difference : ℚ := expected_coffee_days - expected_tea_days
  ⌊difference⌋ = 45 := by sorry


end NUMINAMATH_CALUDE_expected_coffee_tea_difference_l1442_144203


namespace NUMINAMATH_CALUDE_max_sum_with_digit_constraints_l1442_144292

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is within a specific digit range -/
def is_n_digit (n : ℕ) (lower : ℕ) (upper : ℕ) : Prop :=
  lower ≤ n ∧ n ≤ upper

theorem max_sum_with_digit_constraints :
  ∃ (a b c : ℕ),
    is_n_digit a 10 99 ∧
    is_n_digit b 100 999 ∧
    is_n_digit c 1000 9999 ∧
    sum_of_digits (a + b) = 2 ∧
    sum_of_digits (b + c) = 2 ∧
    ∀ (x y z : ℕ),
      is_n_digit x 10 99 →
      is_n_digit y 100 999 →
      is_n_digit z 1000 9999 →
      sum_of_digits (x + y) = 2 →
      sum_of_digits (y + z) = 2 →
      x + y + z ≤ a + b + c ∧
      a + b + c = 10199 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_digit_constraints_l1442_144292


namespace NUMINAMATH_CALUDE_marcy_spears_count_l1442_144210

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings Marcy has -/
def num_saplings : ℕ := 6

/-- The number of logs Marcy has -/
def num_logs : ℕ := 1

/-- Theorem: Marcy can make 27 spears from 6 saplings and 1 log -/
theorem marcy_spears_count :
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry

end NUMINAMATH_CALUDE_marcy_spears_count_l1442_144210


namespace NUMINAMATH_CALUDE_even_perfect_square_ablab_l1442_144296

theorem even_perfect_square_ablab : 
  ∃! n : ℕ, 
    (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 
      n = 10000 * a + 1000 * b + 100 + 10 * a + b) ∧ 
    (∃ m : ℕ, n = m^2) ∧ 
    (∃ k : ℕ, n = 2 * k) ∧
    n = 76176 :=
by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_ablab_l1442_144296


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1442_144286

/-- The number of ways to arrange n people in a row -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of people to be seated -/
def total_people : ℕ := 8

/-- The number of ways to arrange people with restrictions -/
def seating_arrangements : ℕ :=
  2 * factorial (total_people - 1) - 2 * factorial (total_people - 2) * factorial 2

theorem correct_seating_arrangements :
  seating_arrangements = 7200 := by
  sorry

#eval seating_arrangements

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1442_144286


namespace NUMINAMATH_CALUDE_fruit_group_sizes_l1442_144277

theorem fruit_group_sizes (total_bananas total_oranges total_apples : ℕ)
                          (banana_groups orange_groups apple_groups : ℕ)
                          (h1 : total_bananas = 142)
                          (h2 : total_oranges = 356)
                          (h3 : total_apples = 245)
                          (h4 : banana_groups = 47)
                          (h5 : orange_groups = 178)
                          (h6 : apple_groups = 35) :
  ∃ (B O A : ℕ),
    banana_groups * B = total_bananas ∧
    orange_groups * O = total_oranges ∧
    apple_groups * A = total_apples ∧
    B = 3 ∧ O = 2 ∧ A = 7 := by
  sorry

end NUMINAMATH_CALUDE_fruit_group_sizes_l1442_144277


namespace NUMINAMATH_CALUDE_circle_radius_implies_m_value_l1442_144208

/-- The equation of a circle with center (h, k) and radius r -/
def circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + m = 0

theorem circle_radius_implies_m_value :
  ∀ m : ℝ, 
  (∃ h k : ℝ, ∀ x y : ℝ, given_equation x y m ↔ circle_equation x y h k 3) →
  m = -7 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_implies_m_value_l1442_144208


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l1442_144281

theorem wrapping_paper_usage 
  (total_used : ℚ) 
  (num_presents : ℕ) 
  (h1 : total_used = 4 / 15) 
  (h2 : num_presents = 5) :
  total_used / num_presents = 4 / 75 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l1442_144281


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1442_144220

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, 725*m ≡ 1025*m [ZMOD 40] → n ≤ m) ∧ 
  (725*n ≡ 1025*n [ZMOD 40]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1442_144220


namespace NUMINAMATH_CALUDE_tiffany_treasures_l1442_144248

/-- The number of points each treasure is worth -/
def points_per_treasure : ℕ := 6

/-- The number of treasures Tiffany found on the second level -/
def treasures_second_level : ℕ := 5

/-- Tiffany's total score -/
def total_score : ℕ := 48

/-- The number of treasures Tiffany found on the first level -/
def treasures_first_level : ℕ := (total_score - points_per_treasure * treasures_second_level) / points_per_treasure

theorem tiffany_treasures : treasures_first_level = 3 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_treasures_l1442_144248


namespace NUMINAMATH_CALUDE_power_quotient_nineteen_l1442_144207

theorem power_quotient_nineteen : 19^11 / 19^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_quotient_nineteen_l1442_144207


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l1442_144250

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means for x to be a fixed point of f
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

-- Statement 1
theorem fixed_points_for_specific_values :
  is_fixed_point 1 3 (-2) ∧ is_fixed_point 1 3 (-1) :=
sorry

-- Statement 2
theorem condition_for_two_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) ↔
  (0 < a ∧ a < 1) :=
sorry

-- Statement 3
theorem minimum_b_value (a : ℝ) (h : 0 < a ∧ a < 1) :
  let g (x : ℝ) := -x + (2 * a) / (5 * a^2 - 4 * a + 1)
  ∃ b x y : ℝ, x ≠ y ∧ 
    is_fixed_point a b x ∧ 
    is_fixed_point a b y ∧ 
    g ((x + y) / 2) = (x + y) / 2 ∧
    (∀ b' : ℝ, b' ≥ b) ∧
    b = -2 :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_condition_for_two_fixed_points_minimum_b_value_l1442_144250


namespace NUMINAMATH_CALUDE_base4_multiplication_l1442_144222

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base4_multiplication (a b : List Nat) :
  decimalToBase4 (base4ToDecimal a * base4ToDecimal b) = [3, 2, 1, 3, 3] ↔
  a = [3, 1, 2, 1] ∧ b = [1, 2] :=
sorry

end NUMINAMATH_CALUDE_base4_multiplication_l1442_144222


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l1442_144231

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_factorials :
  let sum := (List.range 50).map (λ i => factorial (i + 1)) |> List.foldl (· + ·) 0
  last_two_digits sum = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_l1442_144231


namespace NUMINAMATH_CALUDE_largest_unexpressible_l1442_144265

/-- The set An for a given n -/
def An (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, k < n ∧ x = 2^n - 2^k}

/-- The property of being expressible as a sum of elements from An -/
def isExpressible (n : ℕ) (m : ℕ) : Prop :=
  ∃ (s : Multiset ℕ), (∀ x ∈ s, x ∈ An n) ∧ (s.sum = m)

/-- The main theorem -/
theorem largest_unexpressible (n : ℕ) (h : n ≥ 2) :
  ∀ m : ℕ, m > (n - 2) * 2^n + 1 → isExpressible n m :=
sorry

end NUMINAMATH_CALUDE_largest_unexpressible_l1442_144265


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1442_144290

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the asymptotes of the hyperbola
def asymptote_pos (x y : ℝ) : Prop := y = Real.sqrt 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -Real.sqrt 2 * x

-- State the theorem
theorem distance_focus_to_asymptotes :
  ∃ (d : ℝ), d = Real.sqrt 6 / 3 ∧
  (∀ (x y : ℝ), asymptote_pos x y →
    d = abs (Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) ∧
  (∀ (x y : ℝ), asymptote_neg x y →
    d = abs (-Real.sqrt 2 * focus.1 - focus.2) / Real.sqrt (1 + 2)) :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1442_144290


namespace NUMINAMATH_CALUDE_cube_volume_and_surface_area_l1442_144272

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of the cube -/
def Cube.sumEdgeLength (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of the cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of the cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

theorem cube_volume_and_surface_area 
  (c : Cube) 
  (h : c.sumEdgeLength = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_and_surface_area_l1442_144272


namespace NUMINAMATH_CALUDE_faster_walking_speed_l1442_144236

/-- Proves that given a person who walked 100 km at 10 km/hr, if they had walked at a faster speed
    for the same amount of time and covered an additional 20 km, their faster speed would be 12 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 100 →
  actual_speed = 10 →
  additional_distance = 20 →
  (actual_distance + additional_distance) / (actual_distance / actual_speed) = 12 :=
by sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l1442_144236


namespace NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l1442_144227

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic :
  is_quadratic_one_var f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l1442_144227


namespace NUMINAMATH_CALUDE_pencil_sharpening_l1442_144235

/-- The length sharpened off a pencil is equal to the difference between its initial and final lengths. -/
theorem pencil_sharpening (initial_length final_length : ℝ) :
  initial_length ≥ final_length →
  initial_length - final_length = initial_length - final_length :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l1442_144235


namespace NUMINAMATH_CALUDE_path_area_calculation_l1442_144230

/-- Calculates the area of a path surrounding a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end NUMINAMATH_CALUDE_path_area_calculation_l1442_144230


namespace NUMINAMATH_CALUDE_probability_log_integer_l1442_144278

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 48

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 24 / 95 := by sorry

end NUMINAMATH_CALUDE_probability_log_integer_l1442_144278


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1442_144205

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1442_144205


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l1442_144282

/-- Given a cubic function f(x) = ax³ + bx + 4 where a and b are non-zero real numbers,
    if f(5) = 10, then f(-5) = -2 -/
theorem cubic_function_symmetry (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := λ x : ℝ => a * x^3 + b * x + 4
  f 5 = 10 → f (-5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l1442_144282


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l1442_144260

-- Part (a)
theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (r.choose m) * (m.choose k) = (r.choose k) * ((r - k).choose (m - k)) := by
  sorry

-- Part (b)
theorem binomial_coefficient_identity_b (n m : ℕ) :
  (n + 1).choose (m + 1) = n.choose m + n.choose (m + 1) := by
  sorry

-- Part (c)
theorem binomial_coefficient_identity_c (n : ℕ) :
  (2 * n).choose n = (Finset.range (n + 1)).sum (λ k => (n.choose k) ^ 2) := by
  sorry

-- Part (d)
theorem binomial_coefficient_identity_d (m n k : ℕ) (h : k ≤ n) :
  (m + n).choose k = (Finset.range (k + 1)).sum (λ p => (n.choose p) * (m.choose (k - p))) := by
  sorry

-- Part (e)
theorem binomial_coefficient_identity_e (n k : ℕ) (h : k ≤ n) :
  n.choose k = (Finset.range (n - k + 1)).sum (λ i => (k + i - 1).choose (k - 1)) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_a_binomial_coefficient_identity_b_binomial_coefficient_identity_c_binomial_coefficient_identity_d_binomial_coefficient_identity_e_l1442_144260


namespace NUMINAMATH_CALUDE_three_students_got_A_l1442_144229

structure Student :=
  (name : String)
  (gotA : Bool)

def Emily : Student := ⟨"Emily", false⟩
def Frank : Student := ⟨"Frank", false⟩
def Grace : Student := ⟨"Grace", false⟩
def Harry : Student := ⟨"Harry", false⟩

def students : List Student := [Emily, Frank, Grace, Harry]

def emilyStatement (s : List Student) : Prop :=
  (Emily.gotA = true) → (Frank.gotA = true)

def frankStatement (s : List Student) : Prop :=
  (Frank.gotA = true) → (Grace.gotA = true)

def graceStatement (s : List Student) : Prop :=
  (Grace.gotA = true) → (Harry.gotA = true)

def harryStatement (s : List Student) : Prop :=
  (Harry.gotA = true) → (Emily.gotA = false)

def exactlyThreeGotA (s : List Student) : Prop :=
  (s.filter (λ x => x.gotA)).length = 3

theorem three_students_got_A :
  ∀ s : List Student,
    s = students →
    emilyStatement s →
    frankStatement s →
    graceStatement s →
    harryStatement s →
    exactlyThreeGotA s →
    (Frank.gotA = true ∧ Grace.gotA = true ∧ Harry.gotA = true ∧ Emily.gotA = false) :=
by sorry


end NUMINAMATH_CALUDE_three_students_got_A_l1442_144229


namespace NUMINAMATH_CALUDE_total_votes_is_82_l1442_144271

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ

/-- Conditions for the baking contest votes -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 12 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + (2 * votes.witch / 5) ∧
  votes.mermaid = votes.dragon - 7 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5

/-- Theorem stating that the total number of votes is 82 -/
theorem total_votes_is_82 (votes : CakeVotes) 
  (h : contestConditions votes) : 
  votes.unicorn + votes.witch + votes.dragon + votes.mermaid + votes.fairy = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_is_82_l1442_144271


namespace NUMINAMATH_CALUDE_expression_evaluation_l1442_144209

theorem expression_evaluation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y) / (1 / x) + 2 = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1442_144209


namespace NUMINAMATH_CALUDE_used_car_selection_l1442_144262

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end NUMINAMATH_CALUDE_used_car_selection_l1442_144262


namespace NUMINAMATH_CALUDE_superinvariant_characterization_l1442_144200

/-- A set S is superinvariant if for any stretching A of S, there exists a translation B
    such that the images of S under A and B agree. -/
def Superinvariant (S : Set ℝ) : Prop :=
  ∀ (x₀ a : ℝ) (h : a > 0),
    ∃ b : ℝ,
      (∀ x ∈ S, ∃ y ∈ S, x₀ + a * (x - x₀) = y + b) ∧
      (∀ t ∈ S, ∃ u ∈ S, t + b = x₀ + a * (u - x₀))

/-- The set of all possible superinvariant sets for a given Γ. -/
def SuperinvariantSets (Γ : ℝ) : Set (Set ℝ) :=
  {∅, {Γ}, Set.Iio Γ, Set.Iic Γ, Set.Ioi Γ, Set.Ici Γ, (Set.Iio Γ) ∪ (Set.Ioi Γ), Set.univ}

/-- Theorem stating that a set is superinvariant if and only if it belongs to
    SuperinvariantSets for some Γ. -/
theorem superinvariant_characterization (S : Set ℝ) :
  Superinvariant S ↔ ∃ Γ : ℝ, S ∈ SuperinvariantSets Γ := by
  sorry

end NUMINAMATH_CALUDE_superinvariant_characterization_l1442_144200


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1442_144232

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 5 →
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1442_144232


namespace NUMINAMATH_CALUDE_casper_candy_problem_l1442_144225

theorem casper_candy_problem (initial_candies : ℚ) : 
  let day1_remaining := (3/4) * initial_candies - 3
  let day2_remaining := (4/5) * day1_remaining - 5
  let day3_remaining := day2_remaining - 10
  day3_remaining = 10 → initial_candies = 224/3 := by
  sorry

end NUMINAMATH_CALUDE_casper_candy_problem_l1442_144225


namespace NUMINAMATH_CALUDE_discounted_milk_price_is_correct_l1442_144295

/-- The discounted price of a gallon of whole milk -/
def discounted_milk_price : ℝ := 2

/-- The normal price of a gallon of whole milk -/
def normal_milk_price : ℝ := 3

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings when buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- The number of gallons of milk bought -/
def milk_quantity : ℕ := 3

/-- The number of boxes of cereal bought -/
def cereal_quantity : ℕ := 5

theorem discounted_milk_price_is_correct :
  (milk_quantity : ℝ) * (normal_milk_price - discounted_milk_price) + 
  (cereal_quantity : ℝ) * cereal_discount = total_savings := by
  sorry

end NUMINAMATH_CALUDE_discounted_milk_price_is_correct_l1442_144295


namespace NUMINAMATH_CALUDE_total_medals_1996_l1442_144252

def gold_medals : ℕ := 16
def silver_medals : ℕ := 22
def bronze_medals : ℕ := 12

theorem total_medals_1996 : gold_medals + silver_medals + bronze_medals = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_medals_1996_l1442_144252


namespace NUMINAMATH_CALUDE_factorial_difference_not_seven_l1442_144289

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_difference_not_seven (a b : ℕ) (h : b > a) :
  ∃ k : ℕ, (factorial b - factorial a) % 10 ≠ 7 :=
sorry

end NUMINAMATH_CALUDE_factorial_difference_not_seven_l1442_144289


namespace NUMINAMATH_CALUDE_find_multiple_of_ages_l1442_144241

/-- Given Hiram's age and Allyson's age, find the multiple M that satisfies the equation. -/
theorem find_multiple_of_ages (hiram_age allyson_age : ℕ) (M : ℚ)
  (h1 : hiram_age = 40)
  (h2 : allyson_age = 28)
  (h3 : hiram_age + 12 = M * allyson_age - 4) :
  M = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_of_ages_l1442_144241


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1442_144283

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 2}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | log10 (x^2 + 2*x + 2) < 1} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1442_144283


namespace NUMINAMATH_CALUDE_lateral_surface_area_is_4S_l1442_144273

/-- A regular quadrilateral pyramid with specific properties -/
structure RegularQuadPyramid where
  -- The dihedral angle at the lateral edge
  dihedral_angle : ℝ
  -- The area of the diagonal section
  diagonal_section_area : ℝ
  -- Condition that the dihedral angle is 120°
  angle_is_120 : dihedral_angle = 120 * π / 180

/-- The lateral surface area of a regular quadrilateral pyramid -/
def lateral_surface_area (p : RegularQuadPyramid) : ℝ := 4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid with a 120° dihedral angle
    at the lateral edge is 4 times the area of its diagonal section -/
theorem lateral_surface_area_is_4S (p : RegularQuadPyramid) :
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_is_4S_l1442_144273


namespace NUMINAMATH_CALUDE_systematic_sample_interval_count_l1442_144212

/-- Calculates the number of sampled individuals within a given interval in a systematic sample. -/
def sampledInInterval (totalPopulation : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let groupDistance := totalPopulation / sampleSize
  (intervalEnd - intervalStart + 1) / groupDistance

/-- Theorem stating that for the given parameters, the number of sampled individuals in the interval [61, 140] is 4. -/
theorem systematic_sample_interval_count :
  sampledInInterval 840 42 61 140 = 4 := by
  sorry

#eval sampledInInterval 840 42 61 140

end NUMINAMATH_CALUDE_systematic_sample_interval_count_l1442_144212


namespace NUMINAMATH_CALUDE_hugo_first_roll_four_given_win_l1442_144251

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given he rolled a 4
def hugo_win_given_four_prob : ℚ := 256 / 1296

-- Theorem to prove
theorem hugo_first_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) 
  (roll_four_prob : ℚ) (hugo_win_given_four_prob : ℚ) :
  num_players = 5 ∧ die_sides = 6 ∧ 
  hugo_win_prob = 1 / num_players ∧
  roll_four_prob = 1 / die_sides ∧
  hugo_win_given_four_prob = 256 / 1296 →
  (roll_four_prob * hugo_win_given_four_prob) / hugo_win_prob = 40 / 243 :=
by sorry

end NUMINAMATH_CALUDE_hugo_first_roll_four_given_win_l1442_144251


namespace NUMINAMATH_CALUDE_sqrt_of_square_negative_eleven_l1442_144214

theorem sqrt_of_square_negative_eleven : Real.sqrt ((-11)^2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_negative_eleven_l1442_144214


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l1442_144266

theorem natural_number_divisibility (a b : ℕ) : 
  (∃ k : ℕ, a = k * (b + 1)) → 
  (∃ m : ℕ, 43 = m * (a + b)) → 
  ((a = 22 ∧ b = 21) ∨ 
   (a = 33 ∧ b = 10) ∨ 
   (a = 40 ∧ b = 3) ∨ 
   (a = 42 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l1442_144266


namespace NUMINAMATH_CALUDE_sample_size_example_l1442_144280

/-- Definition of a sample size in a statistical context -/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem: The sample size for 100 items selected from a population of 5000 is 100 -/
theorem sample_size_example : sample_size 5000 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_example_l1442_144280


namespace NUMINAMATH_CALUDE_max_profit_appliance_business_l1442_144245

/-- Represents the cost and profit structure for small electrical appliances --/
structure ApplianceBusiness where
  cost_a : ℝ  -- Cost of one unit of type A
  cost_b : ℝ  -- Cost of one unit of type B
  profit_a : ℝ  -- Profit from selling one unit of type A
  profit_b : ℝ  -- Profit from selling one unit of type B

/-- Theorem stating the maximum profit for the given business scenario --/
theorem max_profit_appliance_business 
  (business : ApplianceBusiness)
  (h1 : 2 * business.cost_a + 3 * business.cost_b = 90)
  (h2 : 3 * business.cost_a + business.cost_b = 65)
  (h3 : business.profit_a = 3)
  (h4 : business.profit_b = 4)
  (h5 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 50 → 
    2750 ≤ a * business.cost_a + (150 - a) * business.cost_b ∧
    a * business.cost_a + (150 - a) * business.cost_b ≤ 2850)
  (h6 : ∀ a : ℕ, 30 ≤ a ∧ a ≤ 35 → 
    565 ≤ a * business.profit_a + (150 - a) * business.profit_b) :
  ∃ (max_profit : ℝ), 
    max_profit = 30 * business.profit_a + 120 * business.profit_b ∧
    max_profit = 570 ∧
    ∀ (a : ℕ), 30 ≤ a ∧ a ≤ 35 → 
      a * business.profit_a + (150 - a) * business.profit_b ≤ max_profit :=
by sorry


end NUMINAMATH_CALUDE_max_profit_appliance_business_l1442_144245


namespace NUMINAMATH_CALUDE_mile_to_rod_l1442_144242

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversions
axiom mile_to_furlong : mile = 10 * furlong
axiom furlong_to_rod : furlong = 40 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 400 * rod := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l1442_144242


namespace NUMINAMATH_CALUDE_smallest_n_trailing_zeros_l1442_144291

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The smallest integer n ≥ 48 for which the number of trailing zeros in n! is exactly n - 48 -/
theorem smallest_n_trailing_zeros : ∀ n : ℕ, n ≥ 48 → (trailingZeros n = n - 48 → n ≥ 62) ∧ trailingZeros 62 = 62 - 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_trailing_zeros_l1442_144291


namespace NUMINAMATH_CALUDE_log_problem_l1442_144206

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 4 = x) → 
  (Real.log 27 / Real.log 2 = k * x) → 
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l1442_144206


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1442_144216

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with ratio q
  a 1 * a 2 * a 3 = 2 →             -- First condition
  a 2 * a 3 * a 4 = 16 →            -- Second condition
  q = 2 :=                          -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1442_144216


namespace NUMINAMATH_CALUDE_vector_not_parallel_implies_x_value_l1442_144294

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (4, x)

-- Define the condition that vectors are not parallel
def not_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 ≠ v.2 * w.1

-- Theorem statement
theorem vector_not_parallel_implies_x_value :
  ∃ x : ℝ, not_parallel a (b x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_not_parallel_implies_x_value_l1442_144294


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1442_144257

/-- The function g(x) defined as x^2 + bx + 1 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- Theorem stating that -3 is not in the range of g(x) if and only if b is in the open interval (-4, 4) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, g b x ≠ -3) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1442_144257


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1442_144215

/-- The perimeter of a rectangle given its width and height -/
def perimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The width of the large rectangle in terms of small rectangles -/
def large_width : ℕ := 5

/-- The height of the large rectangle in terms of small rectangles -/
def large_height : ℕ := 4

theorem rectangle_perimeter_problem (x y : ℝ) 
  (hA : perimeter (6 * x) y = 56)
  (hB : perimeter (4 * x) (3 * y) = 56) :
  perimeter (2 * x) (3 * y) = 40 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1442_144215


namespace NUMINAMATH_CALUDE_sum_of_digits_properties_l1442_144276

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_properties :
  (∀ n : ℕ, sum_of_digits (2 * n) ≤ 2 * sum_of_digits n) ∧
  (∀ n : ℕ, 2 * sum_of_digits n ≤ 10 * sum_of_digits (2 * n)) ∧
  (∃ k : ℕ, sum_of_digits k = 1996 * sum_of_digits (3 * k)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_properties_l1442_144276


namespace NUMINAMATH_CALUDE_expression_range_l1442_144264

theorem expression_range (x y : ℝ) (h : x^2 + (y - 2)^2 ≤ 1) :
  1 ≤ (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ∧
  (x + Real.sqrt 3 * y) / Real.sqrt (x^2 + y^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_range_l1442_144264


namespace NUMINAMATH_CALUDE_solution_set_theorem_l1442_144299

/-- A function f: ℝ → ℝ is increasing -/
def IsIncreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

/-- The set of x where |f(x)| ≥ 2 -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ := {x : ℝ | |f x| ≥ 2}

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_increasing : IsIncreasing f) 
  (h_f1 : f 1 = -2) 
  (h_f3 : f 3 = 2) : 
  SolutionSet f = Set.Ici 3 ∪ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l1442_144299


namespace NUMINAMATH_CALUDE_equal_after_adjustments_l1442_144270

/-- The number of adjustments needed to equalize the number of boys and girls -/
def num_adjustments : ℕ := 8

/-- The initial number of boys -/
def initial_boys : ℕ := 40

/-- The initial number of girls -/
def initial_girls : ℕ := 0

/-- The number of boys reduced in each adjustment -/
def boys_reduction : ℕ := 3

/-- The number of girls increased in each adjustment -/
def girls_increase : ℕ := 2

/-- Calculates the number of boys after a given number of adjustments -/
def boys_after (n : ℕ) : ℤ :=
  initial_boys - n * boys_reduction

/-- Calculates the number of girls after a given number of adjustments -/
def girls_after (n : ℕ) : ℤ :=
  initial_girls + n * girls_increase

theorem equal_after_adjustments :
  boys_after num_adjustments = girls_after num_adjustments := by
  sorry

end NUMINAMATH_CALUDE_equal_after_adjustments_l1442_144270


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1442_144243

theorem imaginary_part_of_complex_product : Complex.im ((4 - 8 * Complex.I) * Complex.I) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1442_144243


namespace NUMINAMATH_CALUDE_right_angled_isosceles_unique_indivisible_l1442_144228

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- side length
  b : ℝ  -- base length
  ha : a > 0
  hb : b > 0

/-- A right-angled isosceles triangle -/
def RightAngledIsoscelesTriangle (t : IsoscelesTriangle) : Prop :=
  t.a = t.b * Real.sqrt 2 / 2

/-- Predicate for a triangle that can be divided into three isosceles triangles with equal side lengths -/
def CanBeDividedIntoThreeIsosceles (t : IsoscelesTriangle) : Prop :=
  ∃ (t1 t2 t3 : IsoscelesTriangle),
    t1.a = t2.a ∧ t2.a = t3.a ∧
    -- Additional conditions to ensure the three triangles form a partition of t
    sorry

/-- Theorem stating that only right-angled isosceles triangles cannot be divided into three isosceles triangles with equal side lengths -/
theorem right_angled_isosceles_unique_indivisible (t : IsoscelesTriangle) :
  ¬(CanBeDividedIntoThreeIsosceles t) ↔ RightAngledIsoscelesTriangle t :=
sorry

end NUMINAMATH_CALUDE_right_angled_isosceles_unique_indivisible_l1442_144228


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l1442_144268

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem smallest_number_with_conditions : ℕ := by
  let n : ℕ := 2979942
  have h1 : tens_digit n = 4 := by sorry
  have h2 : units_digit n = 2 := by sorry
  have h3 : sum_of_digits n = 42 := by sorry
  have h4 : n % 42 = 0 := by sorry
  have h5 : ∀ m : ℕ, m < n →
    ¬(tens_digit m = 4 ∧ units_digit m = 2 ∧ sum_of_digits m = 42 ∧ m % 42 = 0) := by sorry
  exact n

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l1442_144268


namespace NUMINAMATH_CALUDE_square_fraction_count_l1442_144284

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, ∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    (∀ n : ℤ, n ∉ S → ¬∃ m : ℤ, n / (25 - n) = m^2) ∧ 
    S.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l1442_144284


namespace NUMINAMATH_CALUDE_matts_current_age_matts_age_is_65_l1442_144239

/-- Given that James turned 27 three years ago and in 5 years, Matt will be twice James' age,
    prove that Matt's current age is 65. -/
theorem matts_current_age : ℕ → Prop :=
  fun age_matt : ℕ =>
    let age_james_3_years_ago : ℕ := 27
    let years_since_james_27 : ℕ := 3
    let years_until_matt_twice_james : ℕ := 5
    let age_james : ℕ := age_james_3_years_ago + years_since_james_27
    let age_james_in_5_years : ℕ := age_james + years_until_matt_twice_james
    let age_matt_in_5_years : ℕ := 2 * age_james_in_5_years
    age_matt = age_matt_in_5_years - years_until_matt_twice_james ∧ age_matt = 65

/-- Proof of Matt's current age -/
theorem matts_age_is_65 : matts_current_age 65 := by
  sorry

end NUMINAMATH_CALUDE_matts_current_age_matts_age_is_65_l1442_144239


namespace NUMINAMATH_CALUDE_existence_of_special_integer_l1442_144267

theorem existence_of_special_integer :
  ∃ (A : ℕ), 
    (∃ (n : ℕ), A = n * (n + 1) * (n + 2)) ∧
    (∃ (k : ℕ), (A / 10^k) % 10^99 = 10^99 - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integer_l1442_144267


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l1442_144201

theorem function_satisfies_equation (x b : ℝ) : 
  let y := (b + x) / (1 + b*x)
  let y' := ((1 - b^2) / (1 + b*x)^2)
  y - x * y' = b * (1 + x^2 * y') := by sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l1442_144201


namespace NUMINAMATH_CALUDE_second_chapter_pages_l1442_144258

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  two_chapters : chapter1_pages + chapter2_pages = total_pages

/-- The specific book in the problem -/
def problem_book : Book where
  total_pages := 93
  chapter1_pages := 60
  chapter2_pages := 33
  two_chapters := by sorry

theorem second_chapter_pages (b : Book) 
  (h1 : b.total_pages = 93) 
  (h2 : b.chapter1_pages = 60) : 
  b.chapter2_pages = 33 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_pages_l1442_144258


namespace NUMINAMATH_CALUDE_seventieth_number_is_557_l1442_144226

/-- The nth positive integer that leaves a remainder of 5 when divided by 8 -/
def nth_number (n : ℕ) : ℕ := 8 * (n - 1) + 5

/-- Proposition: The 70th positive integer that leaves a remainder of 5 when divided by 8 is 557 -/
theorem seventieth_number_is_557 : nth_number 70 = 557 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_number_is_557_l1442_144226


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l1442_144246

theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (fuel_a : ℝ), 
    fuel_a = 98 ∧
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l1442_144246


namespace NUMINAMATH_CALUDE_bracelet_price_is_15_l1442_144233

/-- The price of a gold heart necklace in dollars -/
def gold_heart_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def coffee_mug_price : ℕ := 20

/-- The number of bracelets bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces bought -/
def necklaces_bought : ℕ := 2

/-- The number of coffee mugs bought -/
def mugs_bought : ℕ := 1

/-- The amount paid in dollars -/
def amount_paid : ℕ := 100

/-- The change received in dollars -/
def change_received : ℕ := 15

theorem bracelet_price_is_15 :
  ∃ (bracelet_price : ℕ),
    bracelet_price * bracelets_bought +
    gold_heart_price * necklaces_bought +
    coffee_mug_price * mugs_bought =
    amount_paid - change_received ∧
    bracelet_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_price_is_15_l1442_144233


namespace NUMINAMATH_CALUDE_afternoon_evening_difference_is_24_l1442_144274

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 33

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 34

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 10

/-- The difference between the number of campers rowing in the afternoon and evening -/
def afternoon_evening_difference : ℕ := afternoon_campers - evening_campers

theorem afternoon_evening_difference_is_24 : 
  afternoon_evening_difference = 24 := by sorry

end NUMINAMATH_CALUDE_afternoon_evening_difference_is_24_l1442_144274


namespace NUMINAMATH_CALUDE_two_red_more_likely_than_one_four_l1442_144237

/-- The number of red balls in the box -/
def red_balls : ℕ := 4

/-- The number of white balls in the box -/
def white_balls : ℕ := 2

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + white_balls

/-- The number of faces on each die -/
def die_faces : ℕ := 6

/-- The probability of drawing two red balls from the box -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- The probability of rolling at least one 4 with two dice -/
def prob_at_least_one_four : ℚ := 1 - (die_faces - 1)^2 / die_faces^2

/-- Theorem stating that the probability of drawing two red balls is greater than
    the probability of rolling at least one 4 with two dice -/
theorem two_red_more_likely_than_one_four : prob_two_red > prob_at_least_one_four :=
sorry

end NUMINAMATH_CALUDE_two_red_more_likely_than_one_four_l1442_144237


namespace NUMINAMATH_CALUDE_markup_percentages_correct_l1442_144287

/-- Represents an item with its purchase price, overhead percentage, and desired net profit. -/
structure Item where
  purchase_price : ℕ
  overhead_percent : ℕ
  net_profit : ℕ

/-- Calculates the selling price of an item, rounded up to the nearest whole dollar. -/
def selling_price (item : Item) : ℕ :=
  let total_cost := item.purchase_price + (item.purchase_price * item.overhead_percent / 100) + item.net_profit
  (total_cost + 99) / 100 * 100

/-- Calculates the markup percentage for an item, rounded up to the nearest whole percent. -/
def markup_percentage (item : Item) : ℕ :=
  let markup := selling_price item - item.purchase_price
  ((markup * 100 + item.purchase_price - 1) / item.purchase_price)

theorem markup_percentages_correct (item_a item_b item_c : Item) : 
  item_a.purchase_price = 48 ∧ 
  item_a.overhead_percent = 20 ∧ 
  item_a.net_profit = 12 ∧
  item_b.purchase_price = 36 ∧ 
  item_b.overhead_percent = 15 ∧ 
  item_b.net_profit = 8 ∧
  item_c.purchase_price = 60 ∧ 
  item_c.overhead_percent = 25 ∧ 
  item_c.net_profit = 16 →
  markup_percentage item_a = 46 ∧
  markup_percentage item_b = 39 ∧
  markup_percentage item_c = 52 := by
  sorry

end NUMINAMATH_CALUDE_markup_percentages_correct_l1442_144287


namespace NUMINAMATH_CALUDE_weekend_art_class_earnings_l1442_144202

/-- Calculates the total money earned over a weekend of art classes --/
def weekend_earnings (beginner_cost advanced_cost : ℕ)
  (saturday_beginner saturday_advanced : ℕ)
  (sibling_discount : ℕ) (sibling_pairs : ℕ) : ℕ :=
  let saturday_total := beginner_cost * saturday_beginner + advanced_cost * saturday_advanced
  let sunday_total := beginner_cost * (saturday_beginner / 2) + advanced_cost * (saturday_advanced / 2)
  let total_before_discount := saturday_total + sunday_total
  let total_discount := sibling_discount * (2 * sibling_pairs)
  total_before_discount - total_discount

/-- Theorem stating that the total earnings for the weekend is $720.00 --/
theorem weekend_art_class_earnings :
  weekend_earnings 15 20 20 10 3 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_weekend_art_class_earnings_l1442_144202


namespace NUMINAMATH_CALUDE_digit_property_characterization_l1442_144259

def has_property (z : Nat) : Prop :=
  z < 10 ∧ 
  ∀ k : Nat, k ≥ 1 → 
    ∃ n : Nat, n ≥ 1 ∧ 
      ∃ m : Nat, n^9 = m * 10^k + z * ((10^k - 1) / 9)

theorem digit_property_characterization :
  ∀ z : Nat, has_property z ↔ z ∈ ({0, 1, 3, 7, 9} : Set Nat) :=
sorry

end NUMINAMATH_CALUDE_digit_property_characterization_l1442_144259


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l1442_144211

theorem hyperbola_asymptote_tangent_circle (m : ℝ) : 
  m > 0 →
  (∀ x y : ℝ, y^2 - x^2/m^2 = 1 → x^2 + y^2 - 4*y + 3 = 0) →
  m = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l1442_144211


namespace NUMINAMATH_CALUDE_balls_remaining_l1442_144261

def initial_balls : ℕ := 10
def removed_balls : ℕ := 3

theorem balls_remaining : initial_balls - removed_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_balls_remaining_l1442_144261


namespace NUMINAMATH_CALUDE_wage_increase_l1442_144293

theorem wage_increase (original_wage new_wage : ℝ) (increase_percentage : ℝ) :
  new_wage = original_wage * (1 + increase_percentage / 100) →
  increase_percentage = 30 →
  new_wage = 78 →
  original_wage = 60 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l1442_144293


namespace NUMINAMATH_CALUDE_board_number_game_l1442_144279

theorem board_number_game (n : ℕ) (h : n = 2009) : 
  let initial_sum := n * (n + 1) / 2
  let initial_remainder := initial_sum % 13
  ∃ (a : ℕ), a ≤ n ∧ (a + 9 + 999) % 13 = initial_remainder ∧ a = 8 :=
sorry

end NUMINAMATH_CALUDE_board_number_game_l1442_144279


namespace NUMINAMATH_CALUDE_equation_to_lines_l1442_144223

/-- The set of points satisfying 2x^2 + y^2 + 3xy + 3x + y = 2 is equivalent to the set of points on the lines y = -x - 2 and y = -2x + 1 -/
theorem equation_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_to_lines_l1442_144223


namespace NUMINAMATH_CALUDE_alternating_7x7_grid_difference_l1442_144263

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := List Square

/-- Represents the entire grid -/
def Grid := List Row

/-- Generates an alternating row starting with the given square type -/
def alternatingRow (start : Square) (length : Nat) : Row :=
  sorry

/-- Counts the number of dark squares in a row -/
def countDarkInRow (row : Row) : Nat :=
  sorry

/-- Counts the number of light squares in a row -/
def countLightInRow (row : Row) : Nat :=
  sorry

/-- Generates a 7x7 grid with alternating squares, starting with a dark square -/
def generateGrid : Grid :=
  sorry

/-- Counts the total number of dark squares in the grid -/
def countTotalDark (grid : Grid) : Nat :=
  sorry

/-- Counts the total number of light squares in the grid -/
def countTotalLight (grid : Grid) : Nat :=
  sorry

theorem alternating_7x7_grid_difference :
  let grid := generateGrid
  countTotalDark grid = countTotalLight grid + 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_7x7_grid_difference_l1442_144263


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1442_144269

theorem value_of_a_minus_b (a b : ℝ) : (a - 5)^2 + |b^3 - 27| = 0 → a - b = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1442_144269


namespace NUMINAMATH_CALUDE_f_two_range_l1442_144244

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the theorem
theorem f_two_range (a b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   f a b c 1 = 0 ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0) →
  (∀ x : ℝ, f a b c (x + 1) = -f a b c (-x + 1)) →
  0 < f a b c 2 ∧ f a b c 2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_two_range_l1442_144244


namespace NUMINAMATH_CALUDE_part_one_part_two_l1442_144221

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

-- Define the solution set A
def A (a : ℝ) : Set ℝ := {x | inequality a x}

-- Define set B
def B : Set ℝ := Set.Ioo (-2) 2

-- Part 1
theorem part_one : A 2 ∪ B = Set.Ioc (-2) 3 := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1442_144221


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1442_144240

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1442_144240


namespace NUMINAMATH_CALUDE_tims_initial_amount_l1442_144238

/-- Tim's candy bar purchase scenario -/
def candy_bar_purchase (initial_amount paid change : ℕ) : Prop :=
  initial_amount = paid + change

/-- Theorem: Tim's initial amount before buying the candy bar -/
theorem tims_initial_amount : ∃ (initial_amount : ℕ), 
  candy_bar_purchase initial_amount 45 5 ∧ initial_amount = 50 := by
  sorry

end NUMINAMATH_CALUDE_tims_initial_amount_l1442_144238


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l1442_144218

theorem smallest_multiples_sum :
  ∀ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0) → d ≤ y) →
  c + d = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l1442_144218


namespace NUMINAMATH_CALUDE_karlee_grapes_l1442_144234

theorem karlee_grapes (G : ℚ) : 
  (G * 3/5 * 3/5 + G * 3/5) = 96 → G = 100 := by
  sorry

end NUMINAMATH_CALUDE_karlee_grapes_l1442_144234


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1442_144288

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -(59/72) := by sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l1442_144288


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1442_144254

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 20 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes
    with at least one ball in each box -/
theorem seven_balls_four_boxes : distribute_balls 7 4 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1442_144254


namespace NUMINAMATH_CALUDE_unclaimed_books_fraction_l1442_144297

/-- Represents the fraction of books each person takes -/
def take_books (total : ℚ) (fraction : ℚ) (remaining : ℚ) : ℚ :=
  fraction * remaining

/-- The fraction of books that goes unclaimed after all four people take their share -/
def unclaimed_fraction : ℚ :=
  let total := 1
  let al_takes := take_books total (2/5) total
  let bert_takes := take_books total (3/10) (total - al_takes)
  let carl_takes := take_books total (1/5) (total - al_takes - bert_takes)
  let dan_takes := take_books total (1/10) (total - al_takes - bert_takes - carl_takes)
  total - (al_takes + bert_takes + carl_takes + dan_takes)

theorem unclaimed_books_fraction :
  unclaimed_fraction = 1701 / 2500 :=
sorry

end NUMINAMATH_CALUDE_unclaimed_books_fraction_l1442_144297


namespace NUMINAMATH_CALUDE_power_of_three_equality_l1442_144298

theorem power_of_three_equality (x : ℕ) :
  3^x = 3^20 * 3^20 * 3^18 + 3^19 * 3^20 * 3^19 + 3^18 * 3^21 * 3^19 → x = 59 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l1442_144298


namespace NUMINAMATH_CALUDE_y_intercept_after_transformation_l1442_144247

/-- A linear function f(x) = -2x + 3 -/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- The transformed function g(x) after moving f(x) up by 2 units -/
def g (x : ℝ) : ℝ := f x + 2

/-- Theorem: The y-intercept of g(x) is at the point (0, 5) -/
theorem y_intercept_after_transformation :
  g 0 = 5 := by sorry

end NUMINAMATH_CALUDE_y_intercept_after_transformation_l1442_144247


namespace NUMINAMATH_CALUDE_absolute_value_symmetry_l1442_144217

/-- A function f : ℝ → ℝ is symmetric about the line x = c if f(c + x) = f(c - x) for all x ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem absolute_value_symmetry (a : ℝ) :
  SymmetricAbout (fun x ↦ |x - a|) 3 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_symmetry_l1442_144217
