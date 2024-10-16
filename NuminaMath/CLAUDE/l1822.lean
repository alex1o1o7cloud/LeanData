import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l1822_182251

theorem two_digit_average_decimal (m n : ℕ) : 
  10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100 →  -- m and n are 2-digit positive integers
  (m + n) / 2 = m + n / 100 →            -- their average equals the decimal m.n
  min m n = 49 :=                        -- the smaller of m and n is 49
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l1822_182251


namespace NUMINAMATH_CALUDE_aaron_cards_found_l1822_182292

/-- Given that Aaron initially had 5 cards and ended up with 67 cards,
    prove that he found 62 cards. -/
theorem aaron_cards_found :
  let initial_cards : ℕ := 5
  let final_cards : ℕ := 67
  let cards_found := final_cards - initial_cards
  cards_found = 62 := by sorry

end NUMINAMATH_CALUDE_aaron_cards_found_l1822_182292


namespace NUMINAMATH_CALUDE_symmetric_function_zero_l1822_182221

/-- A function f: ℝ → ℝ satisfying specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2*x + 2) = -f (-2*x - 2)) ∧ 
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem stating that for a function with the given symmetry properties, f(4) = 0 -/
theorem symmetric_function_zero (f : ℝ → ℝ) (h : SymmetricFunction f) : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_zero_l1822_182221


namespace NUMINAMATH_CALUDE_intersection_implies_sum_of_slopes_is_five_l1822_182200

/-- Given two sets A and B in R^2, defined by linear equations,
    prove that if their intersection is a single point (2, 5),
    then the sum of their slopes is 5. -/
theorem intersection_implies_sum_of_slopes_is_five 
  (a b : ℝ) 
  (A : Set (ℝ × ℝ)) 
  (B : Set (ℝ × ℝ)) 
  (h1 : A = {p : ℝ × ℝ | p.2 = a * p.1 + 1})
  (h2 : B = {p : ℝ × ℝ | p.2 = p.1 + b})
  (h3 : A ∩ B = {(2, 5)}) : 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_of_slopes_is_five_l1822_182200


namespace NUMINAMATH_CALUDE_negation_of_p_negation_of_q_l1822_182288

-- Define the statement p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 5*x ≥ -25/4

-- Define the statement q
def q : Prop := ∃ n : ℕ, Even n ∧ n % 3 = 0

-- Theorem for the negation of p
theorem negation_of_p : (¬p) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x < -25/4) :=
sorry

-- Theorem for the negation of q
theorem negation_of_q : (¬q) ↔ (∀ n : ℕ, Even n → n % 3 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_p_negation_of_q_l1822_182288


namespace NUMINAMATH_CALUDE_divisor_sum_theorem_l1822_182235

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (i j k : ℕ) : 
  (sum_of_geometric_series 1 2 i) * (sum_of_geometric_series 1 3 j) * (sum_of_geometric_series 1 5 k) = 3600 → 
  i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_theorem_l1822_182235


namespace NUMINAMATH_CALUDE_sum_reciprocal_pairs_bound_l1822_182290

/-- 
Given non-negative real numbers x, y, and z satisfying xy + yz + zx = 1,
the sum 1/(x+y) + 1/(y+z) + 1/(z+x) is greater than or equal to 5/2.
-/
theorem sum_reciprocal_pairs_bound (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum_prod : x*y + y*z + z*x = 1) : 
  1/(x+y) + 1/(y+z) + 1/(z+x) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_pairs_bound_l1822_182290


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1822_182246

/-- Given single-digit integers a and b satisfying certain conditions, prove their sum is 7 --/
theorem digit_sum_problem (a b : ℕ) : 
  a < 10 → b < 10 → (4 * a) % 10 = 6 → 3 * b * 10 + 4 * a = 116 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1822_182246


namespace NUMINAMATH_CALUDE_mike_marbles_l1822_182272

theorem mike_marbles (given_away : ℕ) (remaining : ℕ) : 
  given_away = 4 → remaining = 4 → given_away + remaining = 8 := by
  sorry

end NUMINAMATH_CALUDE_mike_marbles_l1822_182272


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l1822_182227

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Reverse digits of a two-digit integer -/
def reverseDigits (n : ℕ) : ℕ := 
  10 * (n % 10) + (n / 10)

theorem two_digit_reverse_sum (x y m : ℕ) : 
  TwoDigitInt x → 
  TwoDigitInt y → 
  y = reverseDigits x → 
  x^2 - y^2 = 4 * m^2 → 
  0 < m → 
  x + y + m = 105 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l1822_182227


namespace NUMINAMATH_CALUDE_christopher_alexander_difference_l1822_182202

/-- Represents the number of joggers bought by each person -/
structure JoggerPurchases where
  christopher : Nat
  tyson : Nat
  alexander : Nat

/-- The conditions of the jogger purchase problem -/
def jogger_problem (purchases : JoggerPurchases) : Prop :=
  purchases.christopher = 80 ∧
  purchases.christopher = 20 * purchases.tyson ∧
  purchases.alexander = purchases.tyson + 22

/-- The theorem to be proved -/
theorem christopher_alexander_difference 
  (purchases : JoggerPurchases) 
  (h : jogger_problem purchases) : 
  purchases.christopher - purchases.alexander = 54 := by
  sorry

end NUMINAMATH_CALUDE_christopher_alexander_difference_l1822_182202


namespace NUMINAMATH_CALUDE_snacks_at_dawn_l1822_182215

theorem snacks_at_dawn (S : ℕ) : 
  (3 * S / 5 : ℚ) = 180 → S = 300 := by
  sorry

end NUMINAMATH_CALUDE_snacks_at_dawn_l1822_182215


namespace NUMINAMATH_CALUDE_octal_2011_equals_base5_13113_l1822_182276

-- Define a function to convert from octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldr (fun (i, digit) acc => acc + digit * (8 ^ i)) 0

-- Define a function to convert from decimal to base-5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem octal_2011_equals_base5_13113 :
  decimal_to_base5 (octal_to_decimal [1, 1, 0, 2]) = [3, 1, 1, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_octal_2011_equals_base5_13113_l1822_182276


namespace NUMINAMATH_CALUDE_jesses_stamps_l1822_182273

/-- Jesse's stamp collection problem -/
theorem jesses_stamps (european_stamps : ℕ) (asian_stamps : ℕ) : 
  european_stamps = 333 →
  european_stamps = 3 * asian_stamps →
  european_stamps + asian_stamps = 444 := by
sorry

end NUMINAMATH_CALUDE_jesses_stamps_l1822_182273


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l1822_182234

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l1822_182234


namespace NUMINAMATH_CALUDE_prime_pairs_problem_l1822_182299

theorem prime_pairs_problem :
  ∀ p q : ℕ,
    1 < p → p < 100 →
    1 < q → q < 100 →
    Prime p →
    Prime q →
    Prime (p + 6) →
    Prime (p + 10) →
    Prime (q + 4) →
    Prime (q + 10) →
    Prime (p + q + 1) →
    ((p = 7 ∧ q = 3) ∨ (p = 13 ∧ q = 3) ∨ (p = 37 ∧ q = 3) ∨ (p = 97 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pairs_problem_l1822_182299


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1822_182209

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (large_cube_edge : ℝ) (h : large_cube_edge = 12) :
  let sphere_diameter := large_cube_edge
  let small_cube_diagonal := sphere_diameter
  let small_cube_edge := small_cube_diagonal / Real.sqrt 3
  let small_cube_volume := small_cube_edge ^ 3
  small_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1822_182209


namespace NUMINAMATH_CALUDE_irreducible_fractions_l1822_182248

theorem irreducible_fractions (a b m n : ℕ) (h_n : n > 0) :
  (Nat.gcd a b = 1 → Nat.gcd (b - a) b = 1) ∧
  (Nat.gcd (m - n) (m + n) = 1 → Nat.gcd m n = 1) ∧
  (∃ (k : ℕ), (5 * n + 2) = k * (10 * n + 7) → Nat.gcd (5 * n + 2) (10 * n + 7) = 3) :=
by sorry

end NUMINAMATH_CALUDE_irreducible_fractions_l1822_182248


namespace NUMINAMATH_CALUDE_grandpa_xiaoqiang_age_relation_l1822_182266

theorem grandpa_xiaoqiang_age_relation (x : ℕ) : 
  66 - x = 7 * (12 - x) ↔ 
  (∃ (grandpa_age xiaoqiang_age : ℕ), 
    grandpa_age = 66 ∧ 
    xiaoqiang_age = 12 ∧ 
    grandpa_age - x = 7 * (xiaoqiang_age - x)) :=
by sorry

end NUMINAMATH_CALUDE_grandpa_xiaoqiang_age_relation_l1822_182266


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1822_182270

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) (h1 : width = length / 3) (h2 : 2 * (width + length) = 72) :
  width * length = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1822_182270


namespace NUMINAMATH_CALUDE_function_composition_result_l1822_182287

/-- Given a function f(x) = x^2 - 2x, prove that f(f(f(1))) = 3 -/
theorem function_composition_result (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2*x) : f (f (f 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l1822_182287


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1822_182275

theorem modulus_of_complex_number (z : ℂ) (h : z = Complex.I * (2 - Complex.I)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1822_182275


namespace NUMINAMATH_CALUDE_union_of_sets_l1822_182236

theorem union_of_sets : 
  let P : Set Int := {-2, 2}
  let Q : Set Int := {-1, 0, 2, 3}
  P ∪ Q = {-2, -1, 0, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1822_182236


namespace NUMINAMATH_CALUDE_equation_equivalence_l1822_182280

theorem equation_equivalence (x : ℝ) : 
  (2*x + 1) / 3 - (5*x - 3) / 2 = 1 ↔ 2*(2*x + 1) - 3*(5*x - 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1822_182280


namespace NUMINAMATH_CALUDE_square_window_side_length_l1822_182213

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  height : ℝ
  width : ℝ
  ratio : height / width = 5 / 2

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : GlassPane
  border_width : ℝ
  side_length : ℝ

/-- Theorem stating the side length of the square window -/
theorem square_window_side_length 
  (window : SquareWindow)
  (h1 : window.border_width = 2)
  (h2 : window.side_length = 4 * window.pane.width + 5 * window.border_width)
  (h3 : window.side_length = 2 * window.pane.height + 3 * window.border_width) :
  window.side_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_window_side_length_l1822_182213


namespace NUMINAMATH_CALUDE_specific_rectangle_measurements_l1822_182297

/-- A rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculate the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem stating the area and perimeter of a specific rectangle -/
theorem specific_rectangle_measurements :
  let r : Rectangle := { length := 0.5, width := 0.36 }
  area r = 0.18 ∧ perimeter r = 1.72 := by
  sorry

end NUMINAMATH_CALUDE_specific_rectangle_measurements_l1822_182297


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1822_182271

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  ((n + 5) - n = 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1822_182271


namespace NUMINAMATH_CALUDE_alexey_game_max_score_l1822_182289

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem alexey_game_max_score :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

end NUMINAMATH_CALUDE_alexey_game_max_score_l1822_182289


namespace NUMINAMATH_CALUDE_sum_of_products_l1822_182201

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l1822_182201


namespace NUMINAMATH_CALUDE_PQRSTU_volume_l1822_182257

/-- A right pyramid with a regular pentagon base -/
structure RightPentagonalPyramid where
  /-- Side length of the base pentagon -/
  baseSideLength : ℝ
  /-- Length of an edge from apex to base vertex -/
  apexToBaseLength : ℝ

/-- Volume of a right pentagonal pyramid -/
def volume (p : RightPentagonalPyramid) : ℝ :=
  sorry

/-- The specific pyramid PQRSTU -/
def PQRSTU : RightPentagonalPyramid :=
  { baseSideLength := 10
  , apexToBaseLength := 10 }

theorem PQRSTU_volume :
  volume PQRSTU = 625 := by sorry

end NUMINAMATH_CALUDE_PQRSTU_volume_l1822_182257


namespace NUMINAMATH_CALUDE_english_majors_count_l1822_182268

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem english_majors_count (bio_majors : ℕ) (engineers : ℕ) (total_selections : ℕ) :
  bio_majors = 6 →
  engineers = 5 →
  total_selections = 200 →
  ∃ (eng_majors : ℕ), 
    choose eng_majors 3 * choose bio_majors 3 * choose engineers 3 = total_selections ∧
    eng_majors = 3 :=
by sorry

end NUMINAMATH_CALUDE_english_majors_count_l1822_182268


namespace NUMINAMATH_CALUDE_problem_4_l1822_182249

theorem problem_4 (a : ℝ) : (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_4_l1822_182249


namespace NUMINAMATH_CALUDE_sum_of_even_integers_l1822_182269

theorem sum_of_even_integers (n : ℕ) (sum_first_n : ℕ) (first : ℕ) (last : ℕ) :
  n = 50 →
  sum_first_n = 2550 →
  first = 102 →
  last = 200 →
  (n : ℕ) * (2 + 2 * n) = 2 * sum_first_n →
  (last - first) / 2 + 1 = n →
  n / 2 * (first + last) = 7550 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_even_integers_l1822_182269


namespace NUMINAMATH_CALUDE_square_difference_divided_l1822_182211

theorem square_difference_divided : (111^2 - 99^2) / 12 = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l1822_182211


namespace NUMINAMATH_CALUDE_car_travel_time_l1822_182250

/-- Proves that a car traveling 810 km at 162 km/h takes 5 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 810 ∧ speed = 162 → time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l1822_182250


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l1822_182264

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def satisfies_condition (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p % 2 = 1 → p < n → is_prime (n - p)

theorem largest_satisfying_number :
  satisfies_condition 10 ∧ ∀ n : ℕ, n > 10 → ¬(satisfies_condition n) :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l1822_182264


namespace NUMINAMATH_CALUDE_inequality_proof_l1822_182217

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Define the set M
def M : Set ℝ := {x | f x < 4}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  2 * |a + b| < |4 + a * b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1822_182217


namespace NUMINAMATH_CALUDE_square_area_ratio_l1822_182265

theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (6*x)^2) = 1/45 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1822_182265


namespace NUMINAMATH_CALUDE_unique_a_value_l1822_182216

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_value : ∃! a : ℝ, 1 ∈ A a ∧ a = 0 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l1822_182216


namespace NUMINAMATH_CALUDE_min_production_quantity_l1822_182212

def cost_function (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

def selling_price : ℝ := 25

theorem min_production_quantity :
  ∃ (min_x : ℝ), min_x = 150 ∧
  ∀ (x : ℝ), x ∈ Set.Ioo 0 240 →
    (selling_price * x ≥ cost_function x ↔ x ≥ min_x) :=
sorry

end NUMINAMATH_CALUDE_min_production_quantity_l1822_182212


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1822_182279

-- Define the curve C
def C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for a and b
def condition (a b : ℝ) : Prop := a = 2 ∧ b = Real.sqrt 2

-- Theorem stating that the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ a b : ℝ, a * b ≠ 0 →
    (condition a b → C a b (Real.sqrt 2) 1)) ∧
  (∃ a b : ℝ, a * b ≠ 0 ∧ C a b (Real.sqrt 2) 1 ∧ ¬ condition a b) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1822_182279


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l1822_182283

theorem complex_modulus_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + t * Complex.I) = 12 → t = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l1822_182283


namespace NUMINAMATH_CALUDE_mps_45_equals_kmph_162_l1822_182294

/-- Converts a speed from meters per second to kilometers per hour. -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- Theorem stating that 45 meters per second is equal to 162 kilometers per hour. -/
theorem mps_45_equals_kmph_162 :
  mps_to_kmph 45 = 162 := by
  sorry

end NUMINAMATH_CALUDE_mps_45_equals_kmph_162_l1822_182294


namespace NUMINAMATH_CALUDE_video_game_lives_l1822_182203

theorem video_game_lives (initial lives_lost lives_gained : ℕ) :
  initial ≥ lives_lost →
  initial - lives_lost + lives_gained = initial + lives_gained - lives_lost :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l1822_182203


namespace NUMINAMATH_CALUDE_factorization_proof_l1822_182204

theorem factorization_proof (y : ℝ) : 4*y*(y+2) + 9*(y+2) + 2*(y+2) = (y+2)*(4*y+11) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1822_182204


namespace NUMINAMATH_CALUDE_monomial_product_l1822_182224

/-- Given two monomials 4x⁴y² and 3x²y³, prove that their product is 12x⁶y⁵ -/
theorem monomial_product :
  ∀ (x y : ℝ), (4 * x^4 * y^2) * (3 * x^2 * y^3) = 12 * x^6 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_monomial_product_l1822_182224


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1822_182214

theorem quadratic_roots_relation (k l p q : ℕ) : 
  (∃ (a b : ℝ), a^2 - k*a + l = 0 ∧ b^2 - k*b + l = 0 ∧ 
   (a + 1/b)^2 - p*(a + 1/b) + q = 0 ∧ (b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1822_182214


namespace NUMINAMATH_CALUDE_complement_of_M_l1822_182207

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 < 2*x}

theorem complement_of_M : Set.compl M = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1822_182207


namespace NUMINAMATH_CALUDE_ice_cream_cones_sold_l1822_182228

theorem ice_cream_cones_sold (milkshakes : ℕ) (difference : ℕ) : 
  milkshakes = 82 → 
  milkshakes = ice_cream_cones + difference → 
  difference = 15 →
  ice_cream_cones = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cones_sold_l1822_182228


namespace NUMINAMATH_CALUDE_square_difference_of_roots_l1822_182291

theorem square_difference_of_roots (α β : ℝ) : 
  (α^2 - 2*α - 4 = 0) → (β^2 - 2*β - 4 = 0) → (α - β)^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_roots_l1822_182291


namespace NUMINAMATH_CALUDE_number_divided_by_2000_l1822_182206

theorem number_divided_by_2000 : ∃ x : ℝ, x / 2000 = 0.012625 ∧ x = 25.25 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_2000_l1822_182206


namespace NUMINAMATH_CALUDE_non_self_intersecting_chains_count_l1822_182219

/-- Represents a point on a circle -/
structure CirclePoint where
  label : ℕ

/-- Represents a polygonal chain on a circle -/
structure PolygonalChain where
  points : List CirclePoint
  is_non_self_intersecting : Bool

/-- The number of ways to form a non-self-intersecting polygonal chain -/
def count_non_self_intersecting_chains (n : ℕ) : ℕ :=
  n * 2^(n-2)

/-- Theorem stating the number of ways to form a non-self-intersecting polygonal chain -/
theorem non_self_intersecting_chains_count 
  (n : ℕ) 
  (h : n > 1) :
  (∀ (chain : PolygonalChain), 
    chain.points.length = n ∧ 
    chain.is_non_self_intersecting = true) →
  (∃! count : ℕ, count = count_non_self_intersecting_chains n) :=
sorry

end NUMINAMATH_CALUDE_non_self_intersecting_chains_count_l1822_182219


namespace NUMINAMATH_CALUDE_ellipse_focus_m_value_l1822_182278

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    and left focus at (-4, 0), prove that m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ x^2 / 25 + y^2 / m^2 = 1) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_m_value_l1822_182278


namespace NUMINAMATH_CALUDE_birthday_height_calculation_l1822_182267

/-- Given an initial height and a growth rate, calculates the new height -/
def new_height (initial_height : ℝ) (growth_rate : ℝ) : ℝ :=
  initial_height * (1 + growth_rate)

/-- Proves that given an initial height of 119.7 cm and a growth rate of 5%,
    the new height is 125.685 cm -/
theorem birthday_height_calculation :
  new_height 119.7 0.05 = 125.685 := by
  sorry

end NUMINAMATH_CALUDE_birthday_height_calculation_l1822_182267


namespace NUMINAMATH_CALUDE_work_earnings_theorem_l1822_182274

/-- Given the following conditions:
  - I worked t+2 hours
  - I earned 4t-4 dollars per hour
  - Bob worked 4t-6 hours
  - Bob earned t+3 dollars per hour
  - I earned three dollars more than Bob
Prove that t = 7/2 -/
theorem work_earnings_theorem (t : ℚ) : 
  (t + 2) * (4 * t - 4) = (4 * t - 6) * (t + 3) + 3 → t = 7/2 := by
sorry

end NUMINAMATH_CALUDE_work_earnings_theorem_l1822_182274


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1822_182210

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (3 + n) = 8 → n = 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1822_182210


namespace NUMINAMATH_CALUDE_kite_plot_area_l1822_182237

/-- The scale of the map in miles per inch -/
def scale : ℚ := 200 / 2

/-- The length of the first diagonal on the map in inches -/
def diagonal1_map : ℚ := 2

/-- The length of the second diagonal on the map in inches -/
def diagonal2_map : ℚ := 10

/-- The area of a kite given its diagonals -/
def kite_area (d1 d2 : ℚ) : ℚ := (1 / 2) * d1 * d2

/-- The theorem stating that the area of the kite-shaped plot is 100,000 square miles -/
theorem kite_plot_area : 
  kite_area (diagonal1_map * scale) (diagonal2_map * scale) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_kite_plot_area_l1822_182237


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1822_182239

theorem sin_330_degrees : Real.sin (330 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1822_182239


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l1822_182285

theorem square_sum_equals_five (a b c : ℝ) 
  (h : a + b + c + 3 = 2 * (Real.sqrt a + Real.sqrt (b + 1) + Real.sqrt (c - 1))) :
  a^2 + b^2 + c^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l1822_182285


namespace NUMINAMATH_CALUDE_cylinder_base_radius_l1822_182258

/-- Given a cylinder with generatrix length 3 cm and lateral area 12π cm², 
    prove that the radius of the base is 2 cm. -/
theorem cylinder_base_radius 
  (generatrix : ℝ) 
  (lateral_area : ℝ) 
  (h1 : generatrix = 3) 
  (h2 : lateral_area = 12 * Real.pi) : 
  lateral_area / (2 * Real.pi * generatrix) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_radius_l1822_182258


namespace NUMINAMATH_CALUDE_estimate_population_characteristic_l1822_182231

/-- Given a population and a sample, estimate the total number with a certain characteristic -/
theorem estimate_population_characteristic
  (total_population : ℕ)
  (sample_size : ℕ)
  (sample_with_characteristic : ℕ)
  (sample_size_positive : sample_size > 0)
  (sample_size_le_total : sample_size ≤ total_population)
  (sample_with_characteristic_le_sample : sample_with_characteristic ≤ sample_size) :
  let estimated_total := (total_population * sample_with_characteristic) / sample_size
  estimated_total = 6000 ∧ 
  estimated_total ≤ total_population ∧
  (sample_with_characteristic : ℚ) / (sample_size : ℚ) = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_estimate_population_characteristic_l1822_182231


namespace NUMINAMATH_CALUDE_largest_common_term_l1822_182238

def first_sequence (n : ℕ) : ℕ := 3 + 10 * (n - 1)
def second_sequence (n : ℕ) : ℕ := 5 + 8 * (n - 1)

theorem largest_common_term : 
  (∃ (n m : ℕ), first_sequence n = second_sequence m ∧ first_sequence n = 133) ∧
  (∀ (x : ℕ), x > 133 → x ≤ 150 → 
    (∀ (n m : ℕ), first_sequence n ≠ x ∨ second_sequence m ≠ x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1822_182238


namespace NUMINAMATH_CALUDE_box_length_given_cube_fill_l1822_182261

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling the box -/
structure CubeFill where
  sideLength : ℕ
  count : ℕ

/-- Theorem stating the relationship between box dimensions and cube fill -/
theorem box_length_given_cube_fill 
  (box : BoxDimensions) 
  (cube : CubeFill) 
  (h1 : box.width = 20) 
  (h2 : box.depth = 10) 
  (h3 : cube.count = 56) 
  (h4 : box.length * box.width * box.depth = cube.count * cube.sideLength ^ 3) 
  (h5 : cube.sideLength ∣ box.width ∧ cube.sideLength ∣ box.depth) :
  box.length = 280 := by
  sorry

#check box_length_given_cube_fill

end NUMINAMATH_CALUDE_box_length_given_cube_fill_l1822_182261


namespace NUMINAMATH_CALUDE_seaweed_for_livestock_l1822_182208

def total_seaweed : ℝ := 500

def fire_percentage : ℝ := 0.4
def medicinal_percentage : ℝ := 0.2
def food_and_feed_percentage : ℝ := 0.4

def human_consumption_ratio : ℝ := 0.3

theorem seaweed_for_livestock (total : ℝ) (fire_pct : ℝ) (med_pct : ℝ) (food_feed_pct : ℝ) (human_ratio : ℝ) 
    (h1 : total = total_seaweed)
    (h2 : fire_pct = fire_percentage)
    (h3 : med_pct = medicinal_percentage)
    (h4 : food_feed_pct = food_and_feed_percentage)
    (h5 : human_ratio = human_consumption_ratio)
    (h6 : fire_pct + med_pct + food_feed_pct = 1) :
  food_feed_pct * total * (1 - human_ratio) = 140 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_for_livestock_l1822_182208


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l1822_182233

/-- Represents a 3x3 grid with numbers 1, 2, and 3 -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Check if a row contains 1, 2, and 3 -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ col : Fin 3, g row col = n

/-- Check if a column contains 1, 2, and 3 -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ row : Fin 3, g row col = n

/-- Check if the main diagonal contains 1, 2, and 3 -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i i = n

/-- Check if the anti-diagonal contains 1, 2, and 3 -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ n : Fin 3, ∃ i : Fin 3, g i (2 - i) = n

/-- A grid is valid if all rows, columns, and diagonals contain 1, 2, and 3 -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_main_diagonal g ∧
  valid_anti_diagonal g

theorem sum_of_A_and_B (g : Grid) (h : valid_grid g) 
  (h1 : g 0 0 = 2) (h2 : g 1 2 = 3) : 
  g 1 1 + g 2 0 = 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_B_l1822_182233


namespace NUMINAMATH_CALUDE_least_number_of_trees_l1822_182225

theorem least_number_of_trees (n : ℕ) : (n > 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0) → n ≥ 210 :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_trees_l1822_182225


namespace NUMINAMATH_CALUDE_sin_ratio_comparison_l1822_182253

theorem sin_ratio_comparison : (Real.sin (3 * π / 180)) / (Real.sin (4 * π / 180)) > (Real.sin (1 * π / 180)) / (Real.sin (2 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_comparison_l1822_182253


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1822_182230

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / (d : ℚ) = 3 + 834 / 999 ∧
  n + d = 4830 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l1822_182230


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_is_one_fourth_l1822_182222

/-- A regular tetrahedron with painted stripes on its faces -/
structure StripedTetrahedron where
  /-- The number of faces of the tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripe_configs : Nat
  /-- The probability of a continuous stripe pattern given all faces have intersecting stripes -/
  prob_continuous_intersecting : ℚ

/-- The probability of at least one continuous stripe pattern encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : ℚ :=
  2 * (1 / 2) ^ 3

theorem probability_continuous_stripe_is_one_fourth (t : StripedTetrahedron) 
    (h1 : t.num_faces = 4)
    (h2 : t.stripe_configs = 2)
    (h3 : t.prob_continuous_intersecting = 1 / 16) :
  probability_continuous_stripe t = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_is_one_fourth_l1822_182222


namespace NUMINAMATH_CALUDE_senate_subcommittee_combinations_l1822_182263

theorem senate_subcommittee_combinations (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) :
  total_republicans = 8 → 
  total_democrats = 6 → 
  subcommittee_republicans = 3 → 
  subcommittee_democrats = 2 → 
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 840 := by
sorry

end NUMINAMATH_CALUDE_senate_subcommittee_combinations_l1822_182263


namespace NUMINAMATH_CALUDE_find_a_value_l1822_182284

theorem find_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5) ∧ 
  (∃ x : ℝ, x > 3 → (x^2 - 4*x + a) + |x - 3| > 5) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l1822_182284


namespace NUMINAMATH_CALUDE_hassans_orange_trees_l1822_182293

/-- Represents the number of trees in an orchard --/
structure Orchard :=
  (orange : ℕ)
  (apple : ℕ)

/-- The total number of trees in an orchard --/
def Orchard.total (o : Orchard) : ℕ := o.orange + o.apple

theorem hassans_orange_trees :
  ∀ (ahmed hassan : Orchard),
  ahmed.orange = 8 →
  ahmed.apple = 4 * hassan.apple →
  hassan.apple = 1 →
  ahmed.total = hassan.total + 9 →
  hassan.orange = 2 := by
sorry

end NUMINAMATH_CALUDE_hassans_orange_trees_l1822_182293


namespace NUMINAMATH_CALUDE_point_on_extension_line_l1822_182281

theorem point_on_extension_line (θ : ℝ) (M : ℝ × ℝ) :
  (∃ k : ℝ, k > 1 ∧ M = (k * Real.cos θ, k * Real.sin θ)) →
  (M.1^2 + M.2^2 = 4) →
  M = (-2 * Real.cos θ, -2 * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l1822_182281


namespace NUMINAMATH_CALUDE_logo_enlargement_l1822_182255

/-- Calculates the height of a proportionally enlarged logo --/
def enlarged_logo_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Proves that a 3x2 inch logo enlarged to 12 inches wide will be 8 inches tall --/
theorem logo_enlargement :
  enlarged_logo_height 3 2 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_logo_enlargement_l1822_182255


namespace NUMINAMATH_CALUDE_middle_school_soccer_league_l1822_182232

theorem middle_school_soccer_league (n : ℕ) : n = 9 :=
  by
  have total_games : n * (n - 1) / 2 = 36 := by sorry
  have min_games_per_team : n - 1 ≥ 8 := by sorry
  sorry

#check middle_school_soccer_league

end NUMINAMATH_CALUDE_middle_school_soccer_league_l1822_182232


namespace NUMINAMATH_CALUDE_women_reseating_l1822_182260

/-- The number of ways to reseat n women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away. -/
def C : ℕ → ℕ
  | 0 => 0  -- Added for completeness
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | n + 4 => 2 * C (n + 3) + 2 * C (n + 2) + C (n + 1)

/-- The number of ways to reseat 9 women in a circle, where each woman can sit in her original seat,
    an adjacent seat, or two seats away, is equal to 3086. -/
theorem women_reseating : C 9 = 3086 := by
  sorry

end NUMINAMATH_CALUDE_women_reseating_l1822_182260


namespace NUMINAMATH_CALUDE_gianna_savings_l1822_182241

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

/-- Gianna's savings problem -/
theorem gianna_savings :
  arithmetic_sum 365 39 2 = 147095 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_l1822_182241


namespace NUMINAMATH_CALUDE_wage_difference_l1822_182229

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def validSteakhouseWages (w : SteakhouseWages) : Prop :=
  w.manager = 7.5 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.2

/-- The theorem stating the difference between manager's and chef's wages -/
theorem wage_difference (w : SteakhouseWages) (h : validSteakhouseWages w) :
  w.manager - w.chef = 3 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l1822_182229


namespace NUMINAMATH_CALUDE_range_of_a_l1822_182252

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1822_182252


namespace NUMINAMATH_CALUDE_extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l1822_182240

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

-- Theorem for the extreme values when a = -3
theorem extreme_values_when_a_neg_three :
  (∃ x₁ x₂ : ℝ, f (-3) x₁ = 5 ∧ f (-3) x₂ = -6 ∧
    ∀ x : ℝ, f (-3) x ≤ 5 ∧ f (-3) x ≥ -6) :=
sorry

-- Theorem for the intersection with x-axis when a ≥ 1
theorem one_intersection_when_a_ge_one :
  ∀ a : ℝ, a ≥ 1 →
    ∃! x : ℝ, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_neg_three_one_intersection_when_a_ge_one_l1822_182240


namespace NUMINAMATH_CALUDE_total_allocation_schemes_l1822_182298

def num_classes : ℕ := 4
def total_spots : ℕ := 5
def min_spots_class_a : ℕ := 2

def allocation_schemes (n c m : ℕ) : ℕ :=
  -- n: total spots
  -- c: number of classes
  -- m: minimum spots for Class A
  sorry

theorem total_allocation_schemes :
  allocation_schemes total_spots num_classes min_spots_class_a = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_allocation_schemes_l1822_182298


namespace NUMINAMATH_CALUDE_proper_subset_of_A_l1822_182296

def A : Set ℝ := { x | x^2 < 5*x }

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by sorry

end NUMINAMATH_CALUDE_proper_subset_of_A_l1822_182296


namespace NUMINAMATH_CALUDE_rational_polynomial_has_rational_coeffs_l1822_182286

/-- A polynomial that maps rationals to rationals has rational coefficients -/
theorem rational_polynomial_has_rational_coeffs (P : Polynomial ℚ) :
  (∀ q : ℚ, ∃ r : ℚ, P.eval q = r) →
  (∀ q : ℚ, ∃ r : ℚ, (P.eval q : ℚ) = r) →
  ∀ i : ℕ, ∃ q : ℚ, P.coeff i = q :=
sorry

end NUMINAMATH_CALUDE_rational_polynomial_has_rational_coeffs_l1822_182286


namespace NUMINAMATH_CALUDE_money_division_l1822_182256

theorem money_division (alice bond charlie : ℕ) 
  (h1 : charlie = 495)
  (h2 : (alice - 10) * 18 * 24 = (bond - 20) * 11 * 24)
  (h3 : (alice - 10) * 24 * 18 = (charlie - 15) * 11 * 18)
  (h4 : (bond - 20) * 24 * 11 = (charlie - 15) * 18 * 11) :
  alice + bond + charlie = 1105 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1822_182256


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1822_182277

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ (1/2) * (4 * x^2 - 4) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) ∧ x = 20 + Real.sqrt 410 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1822_182277


namespace NUMINAMATH_CALUDE_min_value_fraction_l1822_182295

theorem min_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) :
  (2 * a + b : ℚ) / (a - 2 * b) + (a - 2 * b : ℚ) / (2 * a + b) ≥ 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1822_182295


namespace NUMINAMATH_CALUDE_line_intersection_symmetry_l1822_182247

/-- Given a line y = -x + m intersecting the x-axis at A, prove that when moved 6 units left
    to intersect the x-axis at A', if A' is symmetric to A about the origin, then m = 3. -/
theorem line_intersection_symmetry (m : ℝ) : 
  let A : ℝ × ℝ := (m, 0)
  let A' : ℝ × ℝ := (m - 6, 0)
  (A'.1 = -A.1 ∧ A'.2 = -A.2) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_symmetry_l1822_182247


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1822_182243

open Real

/-- The cyclic sum of a function over five variables -/
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ) (a b c d e : ℝ) : ℝ :=
  f a b c d e + f b c d e a + f c d e a b + f d e a b c + f e a b c d

/-- Theorem: For positive real numbers a, b, c, d, e satisfying abcde = 1,
    the cyclic sum of (a + abc)/(1 + ab + abcd) is greater than or equal to 10/3 -/
theorem cyclic_sum_inequality (a b c d e : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
    (h_prod : a * b * c * d * e = 1) :
    cyclicSum (fun a b c d e => (a + a*b*c)/(1 + a*b + a*b*c*d)) a b c d e ≥ 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1822_182243


namespace NUMINAMATH_CALUDE_product_powers_equality_l1822_182223

theorem product_powers_equality (a : ℝ) : 
  let b := a - 1
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * (a^16 + b^16) * (a^32 + b^32) = a^64 - b^64 := by
  sorry

end NUMINAMATH_CALUDE_product_powers_equality_l1822_182223


namespace NUMINAMATH_CALUDE_probability_specific_draw_l1822_182254

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 4

/-- Represents the number of 4s in a standard deck -/
def FoursInDeck : ℕ := 4

/-- Represents the number of clubs in a standard deck -/
def ClubsInDeck : ℕ := 13

/-- Represents the number of 2s in a standard deck -/
def TwosInDeck : ℕ := 4

/-- Represents the number of hearts in a standard deck -/
def HeartsInDeck : ℕ := 13

/-- The probability of drawing a 4, then a club, then a 2, then a heart from a standard 52-card deck -/
theorem probability_specific_draw : 
  (FoursInDeck : ℚ) / StandardDeck *
  ClubsInDeck / (StandardDeck - 1) *
  TwosInDeck / (StandardDeck - 2) *
  HeartsInDeck / (StandardDeck - 3) = 4 / 10829 := by
  sorry

end NUMINAMATH_CALUDE_probability_specific_draw_l1822_182254


namespace NUMINAMATH_CALUDE_ratio_difference_l1822_182242

theorem ratio_difference (a b c : ℝ) (h1 : a / b = 3 / 5) (h2 : b / c = 5 / 7) (h3 : c = 56) : c - a = 32 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l1822_182242


namespace NUMINAMATH_CALUDE_min_chips_to_capture_all_l1822_182226

/-- Represents a rhombus-shaped game board --/
structure RhombusBoard :=
  (side_divisions : ℕ)
  (angle : ℝ)

/-- Represents a chip placement on the board --/
structure ChipPlacement :=
  (x : ℕ)
  (y : ℕ)

/-- Determines if a cell is captured by a chip --/
def is_captured (board : RhombusBoard) (chip : ChipPlacement) (cell : ChipPlacement) : Prop :=
  sorry

/-- Determines if all cells on the board are captured --/
def all_cells_captured (board : RhombusBoard) (chips : List ChipPlacement) : Prop :=
  sorry

/-- The main theorem --/
theorem min_chips_to_capture_all (board : RhombusBoard) :
  board.side_divisions = 9 →
  board.angle = 60 →
  ∃ (chips : List ChipPlacement),
    chips.length = 6 ∧
    all_cells_captured board chips ∧
    ∀ (other_chips : List ChipPlacement),
      other_chips.length < 6 →
      ¬(all_cells_captured board other_chips) :=
sorry

end NUMINAMATH_CALUDE_min_chips_to_capture_all_l1822_182226


namespace NUMINAMATH_CALUDE_midpoint_count_l1822_182244

theorem midpoint_count (n : ℕ) (h : n ≥ 2) :
  ∃ N : ℕ, (2 * n - 3 ≤ N) ∧ (N ≤ n * (n - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_count_l1822_182244


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1822_182262

/-- A quadratic function with vertex at (-2, 3) passing through (3, -45) has a = -48/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Quadratic function definition
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Vertex condition
  (-45 = a * 3^2 + b * 3 + c) →           -- Point condition
  a = -48/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1822_182262


namespace NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l1822_182259

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonic_and_odd_l1822_182259


namespace NUMINAMATH_CALUDE_fraction_equality_l1822_182205

theorem fraction_equality (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1822_182205


namespace NUMINAMATH_CALUDE_grandmother_rolls_l1822_182282

def total_rolls : ℕ := 12
def uncle_rolls : ℕ := 4
def neighbor_rolls : ℕ := 3
def remaining_rolls : ℕ := 2

theorem grandmother_rolls : 
  total_rolls - (uncle_rolls + neighbor_rolls + remaining_rolls) = 3 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_rolls_l1822_182282


namespace NUMINAMATH_CALUDE_problem_statement_l1822_182245

theorem problem_statement (a b c : Int) (h1 : a = -2) (h2 : b = 3) (h3 : c = -4) :
  a - (b - c) = -9 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1822_182245


namespace NUMINAMATH_CALUDE_sector_central_angle_l1822_182220

theorem sector_central_angle (area : Real) (radius : Real) (h1 : area = 3 / 8 * Real.pi) (h2 : radius = 1) :
  (2 * area) / (radius ^ 2) = 3 / 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1822_182220


namespace NUMINAMATH_CALUDE_unique_solution_system_l1822_182218

theorem unique_solution_system (a b : ℕ+) 
  (h1 : a^(b:ℕ) + 3 = b^(a:ℕ)) 
  (h2 : 3 * a^(b:ℕ) = b^(a:ℕ) + 13) : 
  a = 2 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1822_182218
