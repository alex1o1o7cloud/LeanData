import Mathlib

namespace NUMINAMATH_CALUDE_decreasing_function_positive_range_l82_8262

-- Define a decreasing function f on ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem decreasing_function_positive_range
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_inequality : ∀ x, f x / f' x + x < 1) :
  ∀ x, f x > 0 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_positive_range_l82_8262


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l82_8215

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (hx : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l82_8215


namespace NUMINAMATH_CALUDE_total_fish_count_l82_8235

/-- The number of fish in three tanks given specific conditions -/
def total_fish (goldfish1 guppies1 : ℕ) : ℕ :=
  let tank1 := goldfish1 + guppies1
  let tank2 := 2 * goldfish1 + 3 * guppies1
  let tank3 := 3 * goldfish1 + 2 * guppies1
  tank1 + tank2 + tank3

/-- Theorem stating that the total number of fish is 162 given the specific conditions -/
theorem total_fish_count : total_fish 15 12 = 162 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l82_8235


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l82_8217

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(x-1) + 2/(y-2) ≥ 2 :=
by
  sorry

theorem min_value_achieved (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y ∧ 1/(x-1) + 2/(y-2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l82_8217


namespace NUMINAMATH_CALUDE_betty_blue_beads_l82_8257

/-- Given a ratio of red to blue beads and a number of red beads, calculate the number of blue beads -/
def calculate_blue_beads (red_ratio : ℕ) (blue_ratio : ℕ) (total_red : ℕ) : ℕ :=
  (total_red / red_ratio) * blue_ratio

/-- Theorem: Given Betty's bead ratio and total red beads, prove she has 20 blue beads -/
theorem betty_blue_beads :
  let red_ratio : ℕ := 3
  let blue_ratio : ℕ := 2
  let total_red : ℕ := 30
  calculate_blue_beads red_ratio blue_ratio total_red = 20 := by
  sorry

#eval calculate_blue_beads 3 2 30

end NUMINAMATH_CALUDE_betty_blue_beads_l82_8257


namespace NUMINAMATH_CALUDE_max_profit_l82_8233

noncomputable section

-- Define the cost function G(x)
def G (x : ℝ) : ℝ := 2.8 + x

-- Define the revenue function R(x)
def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x
  else 11

-- Define the profit function f(x)
def f (x : ℝ) : ℝ := R x - G x

-- Theorem stating the maximum profit and the corresponding production quantity
theorem max_profit :
  ∃ (x_max : ℝ), x_max = 4 ∧
  ∀ (x : ℝ), 0 ≤ x → f x ≤ f x_max ∧
  f x_max = 3.6 :=
sorry

end

end NUMINAMATH_CALUDE_max_profit_l82_8233


namespace NUMINAMATH_CALUDE_members_playing_both_l82_8220

/-- The number of members who play both badminton and tennis in a sports club -/
theorem members_playing_both (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_neither : neither = 3) :
  badminton + tennis - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_both_l82_8220


namespace NUMINAMATH_CALUDE_count_white_rhinos_l82_8213

/-- Given information about rhinos and their weights, prove the number of white rhinos --/
theorem count_white_rhinos (white_rhino_weight : ℕ) (black_rhino_count : ℕ) (black_rhino_weight : ℕ) (total_weight : ℕ) : 
  white_rhino_weight = 5100 →
  black_rhino_count = 8 →
  black_rhino_weight = 2000 →
  total_weight = 51700 →
  (total_weight - black_rhino_count * black_rhino_weight) / white_rhino_weight = 7 := by
sorry

end NUMINAMATH_CALUDE_count_white_rhinos_l82_8213


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l82_8270

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l82_8270


namespace NUMINAMATH_CALUDE_certain_number_proof_l82_8218

theorem certain_number_proof : ∃ x : ℝ, 0.60 * x = 0.45 * 30 + 16.5 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l82_8218


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l82_8228

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l82_8228


namespace NUMINAMATH_CALUDE_base_3_12021_equals_142_l82_8246

def base_3_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

theorem base_3_12021_equals_142 :
  base_3_to_10 [1, 2, 0, 2, 1] = 142 := by
  sorry

end NUMINAMATH_CALUDE_base_3_12021_equals_142_l82_8246


namespace NUMINAMATH_CALUDE_no_reciprocal_implies_one_l82_8249

/-- If a number minus 1 does not have a reciprocal, then that number equals 1 -/
theorem no_reciprocal_implies_one (a : ℝ) : (∀ x : ℝ, x * (a - 1) ≠ 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_implies_one_l82_8249


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_18_12_15_l82_8204

theorem least_five_digit_divisible_by_18_12_15 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 10080 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_18_12_15_l82_8204


namespace NUMINAMATH_CALUDE_equal_roots_right_triangle_equilateral_triangle_roots_l82_8286

/-- A triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The quadratic equation associated with the triangle -/
def triangle_quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem equal_roots_right_triangle (t : Triangle) :
  (∃ x : ℝ, (∀ y : ℝ, triangle_quadratic t y = 0 ↔ y = x)) →
  t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, triangle_quadratic t x = 0 ↔ x = 0 ∨ x = -1) :=
sorry

end NUMINAMATH_CALUDE_equal_roots_right_triangle_equilateral_triangle_roots_l82_8286


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l82_8216

/-- An isosceles triangle with two sides of length 8 and one side of length 4 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 → b = 8 → c = 4 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l82_8216


namespace NUMINAMATH_CALUDE_range_of_f_l82_8229

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := { x | 1 ≤ x ∧ x < 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -4 ≤ y ∧ y < 5 } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l82_8229


namespace NUMINAMATH_CALUDE_angle_abc_measure_l82_8226

/-- A configuration with a square inscribed in a regular pentagon sharing a side -/
structure SquareInPentagon where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the square in degrees -/
  square_angle : ℝ
  /-- The angle ABC formed by the vertex of the pentagon adjacent to the shared side
      and the two nearest vertices of the square -/
  angle_abc : ℝ
  /-- The pentagon_angle is 108 degrees -/
  pentagon_angle_eq : pentagon_angle = 108
  /-- The square_angle is 90 degrees -/
  square_angle_eq : square_angle = 90

/-- The angle ABC in a SquareInPentagon configuration is 27 degrees -/
theorem angle_abc_measure (config : SquareInPentagon) : config.angle_abc = 27 :=
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l82_8226


namespace NUMINAMATH_CALUDE_smallest_n_l82_8275

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_n : 
  let n : ℕ := 9075
  ∀ m : ℕ, m > 0 → 
    (is_factor (5^2) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (3^3) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (11^2) (m * (2^5) * (6^2) * (7^3) * (13^4))) →
    m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_l82_8275


namespace NUMINAMATH_CALUDE_messaging_packages_theorem_l82_8273

/-- Represents a messaging package --/
structure Package where
  cost : ℕ
  people : ℕ

/-- Calculates the number of ways to connect n people using given packages --/
def countConnections (n : ℕ) (packages : List Package) : ℕ :=
  sorry

/-- Calculates the minimum cost to connect n people using given packages --/
def minCost (n : ℕ) (packages : List Package) : ℕ :=
  sorry

theorem messaging_packages_theorem :
  let n := 4  -- number of friends
  let packageA := Package.mk 10 3
  let packageB := Package.mk 5 2
  let packages := [packageA, packageB]
  (minCost n packages = 15) ∧
  (countConnections n packages = 28) := by
  sorry

end NUMINAMATH_CALUDE_messaging_packages_theorem_l82_8273


namespace NUMINAMATH_CALUDE_walking_meeting_point_l82_8206

/-- Represents the meeting of two people walking towards each other --/
theorem walking_meeting_point (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (deceleration_a : ℝ) (acceleration_b : ℝ) (h : ℕ) :
  total_distance = 100 ∧ 
  speed_a = 5 ∧ 
  speed_b = 4 ∧ 
  deceleration_a = 0.4 ∧ 
  acceleration_b = 0.5 →
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 + 
  (h : ℝ) * (2 * speed_b + (h - 1) * acceleration_b) / 2 = total_distance ∧ 
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 = 
  total_distance / 2 - 31 := by
  sorry

#check walking_meeting_point

end NUMINAMATH_CALUDE_walking_meeting_point_l82_8206


namespace NUMINAMATH_CALUDE_inequality_implication_l82_8296

theorem inequality_implication (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l82_8296


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l82_8293

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l82_8293


namespace NUMINAMATH_CALUDE_infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l82_8267

theorem infinitely_many_double_numbers_plus_one_square_not_power_of_ten :
  ∀ m : ℕ, ∃ k > m, ∃ N : ℕ,
    Odd k ∧
    ∃ t : ℕ, N * (10^k + 1) + 1 = t^2 ∧
    ¬∃ n : ℕ, N * (10^k + 1) + 1 = 10^n := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_double_numbers_plus_one_square_not_power_of_ten_l82_8267


namespace NUMINAMATH_CALUDE_angle_problem_l82_8253

theorem angle_problem (x : ℝ) :
  (x > 0) →
  (x - 30 > 0) →
  (2 * x + (x - 30) = 360) →
  (x = 130) := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l82_8253


namespace NUMINAMATH_CALUDE_digits_of_multiples_of_3_l82_8260

/-- The number of multiples of 3 from 1 to 100 -/
def multiplesOf3 : ℕ := 33

/-- The number of single-digit multiples of 3 from 1 to 100 -/
def singleDigitMultiples : ℕ := 3

/-- The number of two-digit multiples of 3 from 1 to 100 -/
def twoDigitMultiples : ℕ := multiplesOf3 - singleDigitMultiples

/-- The total number of digits written when listing all multiples of 3 from 1 to 100 -/
def totalDigits : ℕ := singleDigitMultiples * 1 + twoDigitMultiples * 2

theorem digits_of_multiples_of_3 : totalDigits = 63 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_multiples_of_3_l82_8260


namespace NUMINAMATH_CALUDE_paco_initial_cookies_l82_8265

/-- Proves that Paco had 40 cookies initially given the problem conditions -/
theorem paco_initial_cookies :
  ∀ x : ℕ,
  x - 2 + 37 = 75 →
  x = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_paco_initial_cookies_l82_8265


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l82_8234

theorem rectangle_area_reduction (original_area : ℝ) 
  (h1 : original_area = 432) 
  (length_reduction : ℝ) (width_reduction : ℝ)
  (h2 : length_reduction = 0.15)
  (h3 : width_reduction = 0.20) : 
  original_area * (1 - length_reduction) * (1 - width_reduction) = 293.76 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l82_8234


namespace NUMINAMATH_CALUDE_shopping_theorem_l82_8207

def shopping_calculation (initial_amount : ℝ) 
  (baguette_cost : ℝ) (baguette_quantity : ℕ)
  (water_cost : ℝ) (water_quantity : ℕ)
  (chocolate_cost : ℝ) (chocolate_quantity : ℕ)
  (milk_cost : ℝ) (milk_discount : ℝ)
  (chips_cost : ℝ) (chips_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let baguette_total := baguette_cost * baguette_quantity
  let water_total := water_cost * water_quantity
  let chocolate_total := (chocolate_cost * 2) * 0.8 * (1 + sales_tax)
  let milk_total := milk_cost * (1 - milk_discount)
  let chips_total := (chips_cost + chips_cost * chips_discount) * (1 + sales_tax)
  let total_cost := baguette_total + water_total + chocolate_total + milk_total + chips_total
  initial_amount - total_cost

theorem shopping_theorem : 
  shopping_calculation 50 2 2 1 2 1.5 3 3.5 0.1 2.5 0.5 0.08 = 34.208 := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l82_8207


namespace NUMINAMATH_CALUDE_tuesday_children_count_l82_8208

/-- Represents the number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := sorry

/-- Theorem stating that the number of children who went to the zoo on Tuesday is 4 -/
theorem tuesday_children_count : tuesday_children = 4 := by
  have monday_revenue : ℕ := 7 * 3 + 5 * 4
  have tuesday_revenue : ℕ := tuesday_children * 3 + 2 * 4
  have total_revenue : ℕ := 61
  have revenue_equation : monday_revenue + tuesday_revenue = total_revenue := sorry
  sorry

end NUMINAMATH_CALUDE_tuesday_children_count_l82_8208


namespace NUMINAMATH_CALUDE_range_of_a_l82_8264

-- Define the condition that |x-3|+|x+5|>a holds for any x ∈ ℝ
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 3| + |x + 5| > a

-- State the theorem
theorem range_of_a :
  {a : ℝ | condition a} = Set.Iio 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l82_8264


namespace NUMINAMATH_CALUDE_initial_average_weight_l82_8243

theorem initial_average_weight (a b c d e : ℝ) : 
  -- Initial conditions
  (a + b + c) / 3 = (a + b + c) / 3 →
  -- Adding packet d
  (a + b + c + d) / 4 = 80 →
  -- Replacing a with e
  (b + c + d + e) / 4 = 79 →
  -- Relationship between d and e
  e = d + 3 →
  -- Weight of packet a
  a = 75 →
  -- Conclusion: initial average weight
  (a + b + c) / 3 = 84 := by
sorry


end NUMINAMATH_CALUDE_initial_average_weight_l82_8243


namespace NUMINAMATH_CALUDE_expression_value_at_three_l82_8266

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l82_8266


namespace NUMINAMATH_CALUDE_min_value_shifted_l82_8255

/-- A quadratic function f(x) with a minimum value of 2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + 5 - c

/-- The function g(x) which is f(x-2015) -/
def g (c : ℝ) (x : ℝ) : ℝ := f c (x - 2015)

theorem min_value_shifted (c : ℝ) (h : ∃ (m : ℝ), ∀ (x : ℝ), f c x ≥ m ∧ ∃ (x₀ : ℝ), f c x₀ = m) 
  (hmin : ∃ (x₀ : ℝ), f c x₀ = 2) :
  ∃ (m : ℝ), ∀ (x : ℝ), g c x ≥ m ∧ ∃ (x₀ : ℝ), g c x₀ = m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_shifted_l82_8255


namespace NUMINAMATH_CALUDE_no_120_cents_combination_l82_8222

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins --/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem: It's impossible to select 6 coins with a total value of 120 cents --/
theorem no_120_cents_combination :
  ¬ ∃ (selection : CoinSelection), selection.length = 6 ∧ totalValue selection = 120 := by
  sorry

end NUMINAMATH_CALUDE_no_120_cents_combination_l82_8222


namespace NUMINAMATH_CALUDE_intersection_line_equation_l82_8237

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 6*x - 7*y - 4*z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7*y - z - 5 = 0

-- Define the line equation
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 35 = (y - 4/7) / 2 ∧ (y - 4/7) / 2 = z / 49

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → line_equation x y z :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l82_8237


namespace NUMINAMATH_CALUDE_club_officer_selection_l82_8239

/-- The number of ways to choose distinct officers from a group -/
def chooseOfficers (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1).factorial / (n - k).factorial

theorem club_officer_selection :
  chooseOfficers 12 5 = 95040 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l82_8239


namespace NUMINAMATH_CALUDE_vector_properties_l82_8290

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

def M : ℝ × ℝ := (C.1 + 3*c.1, C.2 + 3*c.2)
def N : ℝ × ℝ := (C.1 - 2*b.1, C.2 - 2*b.2)

theorem vector_properties :
  (3*a.1 + b.1 - 3*c.1 = 6 ∧ 3*a.2 + b.2 - 3*c.2 = -42) ∧
  (a = (-b.1 - c.1, -b.2 - c.2)) ∧
  (M = (0, 20) ∧ N = (9, 2) ∧ (M.1 - N.1 = 9 ∧ M.2 - N.2 = -18)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l82_8290


namespace NUMINAMATH_CALUDE_mikes_remaining_cards_l82_8248

/-- Given Mike's initial number of baseball cards and the number of cards Sam bought,
    prove that Mike's remaining number of cards is the difference between his initial number
    and the number Sam bought. -/
theorem mikes_remaining_cards (initial_cards sam_bought : ℕ) :
  initial_cards - sam_bought = initial_cards - sam_bought :=
by sorry

/-- Mike's initial number of baseball cards -/
def mike_initial_cards : ℕ := 87

/-- Number of cards Sam bought from Mike -/
def sam_bought_cards : ℕ := 13

/-- Mike's remaining number of cards -/
def mike_remaining_cards : ℕ := mike_initial_cards - sam_bought_cards

#eval mike_remaining_cards  -- Should output 74

end NUMINAMATH_CALUDE_mikes_remaining_cards_l82_8248


namespace NUMINAMATH_CALUDE_increasing_quadratic_condition_l82_8292

/-- If f(x) = x^2 + 2(a - 1)x + 2 is an increasing function on the interval (4, +∞), then a ≥ -3 -/
theorem increasing_quadratic_condition (a : ℝ) : 
  (∀ x > 4, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) → a ≥ -3 := by
sorry

end NUMINAMATH_CALUDE_increasing_quadratic_condition_l82_8292


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l82_8299

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (7/6 * usual_rate) * (usual_time - 2) = usual_rate * usual_time →
  usual_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l82_8299


namespace NUMINAMATH_CALUDE_pencils_taken_l82_8252

theorem pencils_taken (initial_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 79)
  (h2 : remaining_pencils = 75) :
  initial_pencils - remaining_pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_taken_l82_8252


namespace NUMINAMATH_CALUDE_wifes_raise_is_760_l82_8241

/-- Calculates the raise amount for Don's wife given the conditions of the problem -/
def wifes_raise (dons_raise : ℚ) (income_difference : ℚ) (raise_percentage : ℚ) : ℚ :=
  let dons_income := dons_raise / raise_percentage
  let wifes_income := dons_income - (income_difference / (1 + raise_percentage))
  wifes_income * raise_percentage

/-- Proves that Don's wife's raise is 760 given the problem conditions -/
theorem wifes_raise_is_760 : 
  wifes_raise 800 540 (8/100) = 760 := by
  sorry

#eval wifes_raise 800 540 (8/100)

end NUMINAMATH_CALUDE_wifes_raise_is_760_l82_8241


namespace NUMINAMATH_CALUDE_tan_sqrt_two_identity_l82_8245

theorem tan_sqrt_two_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  1 + Real.sin (2 * α) + (Real.cos α)^2 = (4 + Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sqrt_two_identity_l82_8245


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l82_8289

/-- Represents the speed of a boat in km/hr -/
def boat_speed : ℝ := 6

/-- Represents the distance traveled against the stream in km -/
def distance_against : ℝ := 5

/-- Represents the time of travel in hours -/
def travel_time : ℝ := 1

/-- Calculates the speed of the stream based on the boat's speed and distance traveled against the stream -/
def stream_speed : ℝ := boat_speed - distance_against

/-- Calculates the effective speed of the boat along the stream -/
def effective_speed : ℝ := boat_speed + stream_speed

/-- Theorem: The boat travels 7 km along the stream in one hour -/
theorem boat_distance_along_stream :
  effective_speed * travel_time = 7 := by sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l82_8289


namespace NUMINAMATH_CALUDE_least_cube_divisible_by_17280_l82_8269

theorem least_cube_divisible_by_17280 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(17280 ∣ y^3)) ∧ (17280 ∣ x^3) ↔ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_least_cube_divisible_by_17280_l82_8269


namespace NUMINAMATH_CALUDE_third_motorcyclist_speed_l82_8283

theorem third_motorcyclist_speed 
  (v1 : ℝ) (v2 : ℝ) (v3 : ℝ) (t_delay : ℝ) (t_diff : ℝ) :
  v1 = 80 →
  v2 = 60 →
  t_delay = 0.5 →
  t_diff = 1.25 →
  v3 * (v3 * t_diff / (v3 - v1) - t_delay) = v1 * (v3 * t_diff / (v3 - v1)) →
  v3 * (v3 * t_diff / (v3 - v1) - v3 * t_diff / (v3 - v2) - t_delay) = 
    v2 * (v3 * t_diff / (v3 - v1) - t_delay) →
  v3 = 100 := by
sorry


end NUMINAMATH_CALUDE_third_motorcyclist_speed_l82_8283


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l82_8288

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) (hneq : x ≠ y) :
  1 / x + 1 / y > 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l82_8288


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l82_8244

theorem gasoline_tank_capacity : 
  ∀ (capacity : ℝ),
  (5/6 : ℝ) * capacity - (1/3 : ℝ) * capacity = 20 →
  capacity = 40 := by
sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l82_8244


namespace NUMINAMATH_CALUDE_problem_solution_l82_8230

theorem problem_solution : 
  ∃ x : ℝ, (0.4 * 2 = 0.25 * (0.3 * 15 + x)) ∧ (x = -1.3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l82_8230


namespace NUMINAMATH_CALUDE_max_value_abc_l82_8221

theorem max_value_abc (a b c : ℝ) (h : a + 3*b + c = 6) :
  ∃ m : ℝ, m = 8 ∧ ∀ x y z : ℝ, x + 3*y + z = 6 → x*y + x*z + y*z ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l82_8221


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l82_8291

/-- A line passing through (1, 2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1, 2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x + y - 3 = 0 or 2x - y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y - 2 = l.k * (x - 1)) ∨
  (∀ x y, 2 * x - y = 0 ↔ y - 2 = l.k * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l82_8291


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l82_8277

/-- The coefficient of x^3 in the expansion of (1+ax)^5 -/
def coefficient_x3 (a : ℝ) : ℝ := 10 * a^3

theorem binomial_expansion_coefficient (a : ℝ) :
  coefficient_x3 a = -80 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l82_8277


namespace NUMINAMATH_CALUDE_gerald_initial_farthings_l82_8298

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- The cost of a meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def pfennigs_left : ℕ := 7

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

theorem gerald_initial_farthings :
  initial_farthings = 
    pie_cost * farthings_per_pfennig + pfennigs_left * farthings_per_pfennig :=
by sorry

end NUMINAMATH_CALUDE_gerald_initial_farthings_l82_8298


namespace NUMINAMATH_CALUDE_ivan_piggy_bank_l82_8225

/-- Represents the contents of Ivan's piggy bank -/
structure PiggyBank where
  dimes : Nat
  pennies : Nat

/-- The value of the piggy bank in cents -/
def PiggyBank.value (pb : PiggyBank) : Nat :=
  pb.dimes * 10 + pb.pennies

theorem ivan_piggy_bank :
  ∀ (pb : PiggyBank),
    pb.dimes = 50 →
    pb.value = 1200 →
    pb.pennies = 700 := by
  sorry

end NUMINAMATH_CALUDE_ivan_piggy_bank_l82_8225


namespace NUMINAMATH_CALUDE_puppies_left_l82_8219

theorem puppies_left (initial : ℕ) (given_away : ℕ) (h1 : initial = 7) (h2 : given_away = 5) :
  initial - given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_left_l82_8219


namespace NUMINAMATH_CALUDE_smallest_surface_areas_100_cubes_l82_8295

/-- Represents a polyhedron formed by unit cubes -/
structure Polyhedron :=
  (length width height : ℕ)
  (total_cubes : ℕ)
  (surface_area : ℕ)

/-- Calculates the surface area of a rectangular prism -/
def calculate_surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + l * h + w * h)

/-- Generates all possible polyhedra from 100 unit cubes -/
def generate_polyhedra (n : ℕ) : List Polyhedron :=
  sorry

/-- Finds the first 6 smallest surface areas -/
def first_6_surface_areas (polyhedra : List Polyhedron) : List ℕ :=
  sorry

theorem smallest_surface_areas_100_cubes :
  let polyhedra := generate_polyhedra 100
  let areas := first_6_surface_areas polyhedra
  areas = [130, 134, 136, 138, 140, 142] :=
sorry

end NUMINAMATH_CALUDE_smallest_surface_areas_100_cubes_l82_8295


namespace NUMINAMATH_CALUDE_scientific_notation_of_35_billion_l82_8211

-- Define 35 billion
def thirty_five_billion : ℝ := 35000000000

-- Theorem statement
theorem scientific_notation_of_35_billion :
  thirty_five_billion = 3.5 * (10 : ℝ) ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35_billion_l82_8211


namespace NUMINAMATH_CALUDE_new_student_weights_l82_8231

/-- Proves that given the class size changes and average weights, the weights of the four new students are as calculated. -/
theorem new_student_weights
  (original_size : ℕ)
  (original_avg : ℝ)
  (avg_after_first : ℝ)
  (avg_after_second : ℝ)
  (avg_after_third : ℝ)
  (final_avg : ℝ)
  (h_original_size : original_size = 29)
  (h_original_avg : original_avg = 28)
  (h_avg_after_first : avg_after_first = 27.2)
  (h_avg_after_second : avg_after_second = 27.8)
  (h_avg_after_third : avg_after_third = 27.6)
  (h_final_avg : final_avg = 28) :
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = 4 ∧
    w2 = 45.8 ∧
    w3 = 21.4 ∧
    w4 = 40.8 ∧
    (original_size : ℝ) * original_avg + w1 = (original_size + 1 : ℝ) * avg_after_first ∧
    (original_size + 1 : ℝ) * avg_after_first + w2 = (original_size + 2 : ℝ) * avg_after_second ∧
    (original_size + 2 : ℝ) * avg_after_second + w3 = (original_size + 3 : ℝ) * avg_after_third ∧
    (original_size + 3 : ℝ) * avg_after_third + w4 = (original_size + 4 : ℝ) * final_avg :=
by
  sorry

end NUMINAMATH_CALUDE_new_student_weights_l82_8231


namespace NUMINAMATH_CALUDE_special_integers_count_l82_8223

/-- Sum of all positive divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers j such that 1 ≤ j ≤ 5041 and g(j) = 1 + √j + j -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 20 := by sorry

end NUMINAMATH_CALUDE_special_integers_count_l82_8223


namespace NUMINAMATH_CALUDE_product_digit_sum_l82_8212

def number1 : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def number2 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

theorem product_digit_sum :
  let product := number1 * number2
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 13 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l82_8212


namespace NUMINAMATH_CALUDE_line_point_sum_l82_8209

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- The main theorem -/
theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 8.75 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l82_8209


namespace NUMINAMATH_CALUDE_new_lamp_height_is_correct_l82_8236

/-- The height of the old lamp in feet -/
def old_lamp_height : ℝ := 1

/-- The difference in height between the new and old lamp in feet -/
def height_difference : ℝ := 1.3333333333333333

/-- The height of the new lamp in feet -/
def new_lamp_height : ℝ := old_lamp_height + height_difference

theorem new_lamp_height_is_correct : new_lamp_height = 2.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_new_lamp_height_is_correct_l82_8236


namespace NUMINAMATH_CALUDE_victors_earnings_l82_8242

/-- Victor's earnings for two days of work --/
theorem victors_earnings (hourly_wage : ℕ) (hours_monday : ℕ) (hours_tuesday : ℕ) :
  hourly_wage = 6 →
  hours_monday = 5 →
  hours_tuesday = 5 →
  hourly_wage * (hours_monday + hours_tuesday) = 60 := by
  sorry

end NUMINAMATH_CALUDE_victors_earnings_l82_8242


namespace NUMINAMATH_CALUDE_triangle_formation_l82_8282

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 2 3 6) ∧
  ¬(can_form_triangle 3 4 8) ∧
  can_form_triangle 5 6 10 ∧
  ¬(can_form_triangle 5 6 11) := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l82_8282


namespace NUMINAMATH_CALUDE_sqrt_m_minus_n_l82_8202

theorem sqrt_m_minus_n (m n : ℝ) 
  (h1 : Real.sqrt (m - 3) = 3) 
  (h2 : Real.sqrt (n + 1) = 2) : 
  Real.sqrt (m - n) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_m_minus_n_l82_8202


namespace NUMINAMATH_CALUDE_mario_age_l82_8278

theorem mario_age : 
  ∀ (mario_age maria_age : ℕ), 
  mario_age + maria_age = 7 →
  mario_age = maria_age + 1 →
  mario_age = 4 := by
sorry

end NUMINAMATH_CALUDE_mario_age_l82_8278


namespace NUMINAMATH_CALUDE_monthly_interest_payment_l82_8224

/-- Calculate the monthly interest payment given the annual interest rate and investment amount -/
theorem monthly_interest_payment 
  (annual_rate : ℝ) 
  (investment : ℝ) 
  (h1 : annual_rate = 0.09) 
  (h2 : investment = 28800) : 
  (investment * annual_rate) / 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_monthly_interest_payment_l82_8224


namespace NUMINAMATH_CALUDE_mutually_exclusive_non_opposite_l82_8281

structure Ball where
  color : Bool  -- True for black, False for red

def Pocket : Finset Ball := sorry

-- Define the event of selecting exactly one black ball
def ExactlyOneBlack (selection : Finset Ball) : Prop :=
  selection.card = 2 ∧ (selection.filter (λ b => b.color)).card = 1

-- Define the event of selecting exactly two black balls
def ExactlyTwoBlack (selection : Finset Ball) : Prop :=
  selection.card = 2 ∧ (selection.filter (λ b => b.color)).card = 2

-- The main theorem
theorem mutually_exclusive_non_opposite :
  (∀ selection : Finset Ball, selection ⊆ Pocket → selection.card = 2 →
    ¬(ExactlyOneBlack selection ∧ ExactlyTwoBlack selection)) ∧
  (∃ selection : Finset Ball, selection ⊆ Pocket ∧ selection.card = 2 ∧
    ¬ExactlyOneBlack selection ∧ ¬ExactlyTwoBlack selection) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_non_opposite_l82_8281


namespace NUMINAMATH_CALUDE_value_range_sqrt_sum_bounds_are_tight_l82_8247

theorem value_range_sqrt_sum (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) ∧ 
  Real.sqrt 2 ≤ y ∧ y ≤ 2 :=
sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = Real.sqrt 2) ∧
  (∃ (x : ℝ), Real.sqrt (1 + 2*x) + Real.sqrt (1 - 2*x) = 2) :=
sorry

end NUMINAMATH_CALUDE_value_range_sqrt_sum_bounds_are_tight_l82_8247


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l82_8285

theorem geometric_sequence_second_term (a : ℝ) : 
  a > 0 ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ 25 * r = a ∧ a * r = 8/5) → 
  a = 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l82_8285


namespace NUMINAMATH_CALUDE_executive_board_selection_l82_8205

theorem executive_board_selection (n : ℕ) (r : ℕ) : n = 12 ∧ r = 5 → Nat.choose n r = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l82_8205


namespace NUMINAMATH_CALUDE_base10_to_base13_172_l82_8210

/-- Converts a number from base 10 to base 13 --/
def toBase13 (n : ℕ) : List ℕ := sorry

theorem base10_to_base13_172 :
  toBase13 172 = [1, 0, 3] := by sorry

end NUMINAMATH_CALUDE_base10_to_base13_172_l82_8210


namespace NUMINAMATH_CALUDE_first_fun_friday_l82_8261

/-- Represents a day of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The company's year starts on Thursday, March 1st -/
def yearStart : MarchDate :=
  { day := 1, dayOfWeek := DayOfWeek.thursday }

/-- March has 31 days -/
def marchDays : Nat := 31

/-- Determines if a given date is a Friday -/
def isFriday (date : MarchDate) : Prop :=
  date.dayOfWeek = DayOfWeek.friday

/-- Counts the number of Fridays up to and including a given date in March -/
def fridayCount (date : MarchDate) : Nat :=
  sorry

/-- Determines if a given date is a Fun Friday -/
def isFunFriday (date : MarchDate) : Prop :=
  isFriday date ∧ fridayCount date = 5

/-- The theorem to be proved -/
theorem first_fun_friday : 
  ∃ (date : MarchDate), date.day = 30 ∧ isFunFriday date :=
sorry

end NUMINAMATH_CALUDE_first_fun_friday_l82_8261


namespace NUMINAMATH_CALUDE_stating_special_numeral_satisfies_condition_l82_8294

/-- 
A numeral with two 1's where the difference between their place values is 99.99.
-/
def special_numeral : ℝ := 1.11

/-- 
The difference between the place values of the two 1's in the special numeral.
-/
def place_value_difference : ℝ := 99.99

/-- 
Theorem stating that the special_numeral satisfies the required condition.
-/
theorem special_numeral_satisfies_condition : 
  (100 : ℝ) - (1 / 100 : ℝ) = place_value_difference :=
by sorry

end NUMINAMATH_CALUDE_stating_special_numeral_satisfies_condition_l82_8294


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_l82_8238

theorem fixed_point_quadratic (k : ℝ) : 
  200 = 8 * (5 : ℝ)^2 + 3 * k * 5 - 5 * k := by sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_l82_8238


namespace NUMINAMATH_CALUDE_product_of_digits_for_non_divisible_by_five_l82_8203

def numbers : List Nat := [4750, 4760, 4775, 4785, 4790]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_for_non_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧ 
    units_digit n * tens_digit n = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_of_digits_for_non_divisible_by_five_l82_8203


namespace NUMINAMATH_CALUDE_fifteenth_number_with_digit_sum_14_l82_8297

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nth_number_with_digit_sum_14 (n : ℕ+) : ℕ+ := sorry

/-- The main theorem -/
theorem fifteenth_number_with_digit_sum_14 :
  nth_number_with_digit_sum_14 15 = 266 := by sorry

end NUMINAMATH_CALUDE_fifteenth_number_with_digit_sum_14_l82_8297


namespace NUMINAMATH_CALUDE_emma_reaches_jack_emma_reaches_jack_proof_l82_8250

/-- The time it takes for Emma to reach Jack given their initial conditions -/
theorem emma_reaches_jack : ℝ :=
  let initial_distance : ℝ := 30
  let combined_speed : ℝ := 2
  let jack_emma_speed_ratio : ℝ := 2
  let jack_stop_time : ℝ := 6
  
  33

theorem emma_reaches_jack_proof (initial_distance : ℝ) (combined_speed : ℝ) 
  (jack_emma_speed_ratio : ℝ) (jack_stop_time : ℝ) 
  (h1 : initial_distance = 30)
  (h2 : combined_speed = 2)
  (h3 : jack_emma_speed_ratio = 2)
  (h4 : jack_stop_time = 6) :
  emma_reaches_jack = 33 := by
  sorry

#check emma_reaches_jack_proof

end NUMINAMATH_CALUDE_emma_reaches_jack_emma_reaches_jack_proof_l82_8250


namespace NUMINAMATH_CALUDE_committees_with_restriction_l82_8256

def total_students : ℕ := 9
def committee_size : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem committees_with_restriction (total : ℕ) (size : ℕ) : 
  total = total_students → size = committee_size → 
  (choose total size) - (choose (total - 2) (size - 2)) = 91 := by
  sorry

end NUMINAMATH_CALUDE_committees_with_restriction_l82_8256


namespace NUMINAMATH_CALUDE_area_calculation_l82_8274

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 - 16*x + y^2 = 60

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 4 - x

/-- The region of interest -/
def region_of_interest (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y > 4 - x

/-- The area of the region of interest -/
noncomputable def area_of_region : ℝ := sorry

theorem area_calculation : area_of_region = 77.5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_calculation_l82_8274


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_nonnegative_with_zero_l82_8214

theorem sin_cos_fourth_power_nonnegative_with_zero (x : ℝ) :
  (∀ x, (Real.sin x + Real.cos x)^4 ≥ 0) ∧
  (∃ x, (Real.sin x + Real.cos x)^4 = 0) := by
sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_nonnegative_with_zero_l82_8214


namespace NUMINAMATH_CALUDE_triangle_value_l82_8271

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 85)
  (h2 : (triangle + p) + 3 * p = 154) : 
  triangle = 62 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l82_8271


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l82_8279

theorem quadratic_roots_sum_minus_product (a b : ℝ) : 
  a^2 - 3*a + 1 = 0 → b^2 - 3*b + 1 = 0 → a + b - a*b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_minus_product_l82_8279


namespace NUMINAMATH_CALUDE_running_to_basketball_ratio_l82_8258

def trumpet_time : ℕ := 40

theorem running_to_basketball_ratio :
  let running_time := trumpet_time / 2
  let basketball_time := running_time + trumpet_time
  (running_time : ℚ) / basketball_time = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_running_to_basketball_ratio_l82_8258


namespace NUMINAMATH_CALUDE_unique_function_solution_l82_8227

/-- Given a positive real number c, prove that the only function f: ℝ₊ → ℝ₊ 
    satisfying f((c+1)x + f(y)) = f(x + 2y) + 2cx for all x, y ∈ ℝ₊ is f(x) = 2x. -/
theorem unique_function_solution (c : ℝ) (hc : c > 0) :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  (∀ x y, x > 0 → y > 0 → f ((c + 1) * x + f y) = f (x + 2 * y) + 2 * c * x) →
  ∀ x, x > 0 → f x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l82_8227


namespace NUMINAMATH_CALUDE_proportional_scaling_l82_8276

/-- Proportional scaling of a rectangle -/
theorem proportional_scaling (w h new_w : ℝ) (hw : w > 0) (hh : h > 0) (hnew_w : new_w > 0) :
  let scale_factor := new_w / w
  let new_h := h * scale_factor
  w = 3 ∧ h = 2 ∧ new_w = 12 → new_h = 8 := by sorry

end NUMINAMATH_CALUDE_proportional_scaling_l82_8276


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l82_8263

/-- Sum of an arithmetic sequence with given parameters -/
def arithmeticSequenceSum (a₁ aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem stating the sum of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_sum :
  arithmeticSequenceSum 1000 5000 4 = 3003000 := by
  sorry

#eval arithmeticSequenceSum 1000 5000 4

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_sum_l82_8263


namespace NUMINAMATH_CALUDE_rahuls_share_l82_8259

/-- Calculates the share of payment for a worker in a joint work scenario -/
def calculate_share (days_worker1 days_worker2 total_payment : ℚ) : ℚ :=
  let worker1_rate := 1 / days_worker1
  let worker2_rate := 1 / days_worker2
  let combined_rate := worker1_rate + worker2_rate
  let share_ratio := worker1_rate / combined_rate
  share_ratio * total_payment

/-- Theorem stating that Rahul's share of the payment is $68 -/
theorem rahuls_share :
  calculate_share 3 2 170 = 68 := by
  sorry

#eval calculate_share 3 2 170

end NUMINAMATH_CALUDE_rahuls_share_l82_8259


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l82_8200

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) :
  r = 6 →
  circle_area = π * r^2 →
  rectangle_area = 3 * circle_area →
  ∃ (shorter_side longer_side : ℝ),
    shorter_side = 2 * r ∧
    rectangle_area = shorter_side * longer_side ∧
    longer_side = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l82_8200


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l82_8201

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ l}

-- Define the property that for all x in S, x² is also in S
def square_closed (m l : ℝ) : Prop :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1: If m = 1, then S = {1}
theorem proposition_1 (l : ℝ) (h : square_closed 1 l) :
  S 1 l = {1} :=
sorry

-- Theorem 2: If m = -1/3, then l ∈ [1/9, 1]
theorem proposition_3 (l : ℝ) (h : square_closed (-1/3) l) :
  1/9 ≤ l ∧ l ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l82_8201


namespace NUMINAMATH_CALUDE_hours_difference_l82_8272

/-- Represents the project and candidates' information -/
structure Project where
  total_pay : ℕ
  p_wage : ℕ
  q_wage : ℕ
  p_hours : ℕ
  q_hours : ℕ

/-- Conditions of the project -/
def project_conditions (proj : Project) : Prop :=
  proj.total_pay = 360 ∧
  proj.p_wage = proj.q_wage + proj.q_wage / 2 ∧
  proj.p_wage = proj.q_wage + 6 ∧
  proj.total_pay = proj.p_wage * proj.p_hours ∧
  proj.total_pay = proj.q_wage * proj.q_hours

/-- Theorem stating the difference in hours between candidates q and p -/
theorem hours_difference (proj : Project) 
  (h : project_conditions proj) : proj.q_hours - proj.p_hours = 10 := by
  sorry


end NUMINAMATH_CALUDE_hours_difference_l82_8272


namespace NUMINAMATH_CALUDE_ellipse_foci_l82_8251

/-- An ellipse defined by parametric equations -/
structure ParametricEllipse where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The foci of an ellipse -/
structure EllipseFoci where
  x : ℝ
  y : ℝ

/-- Theorem: The foci of the ellipse defined by x = 3cos(θ) and y = 5sin(θ) are (0, ±4) -/
theorem ellipse_foci (e : ParametricEllipse) 
    (hx : e.x = fun θ => 3 * Real.cos θ)
    (hy : e.y = fun θ => 5 * Real.sin θ) :
  ∃ (f₁ f₂ : EllipseFoci), f₁.x = 0 ∧ f₁.y = 4 ∧ f₂.x = 0 ∧ f₂.y = -4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l82_8251


namespace NUMINAMATH_CALUDE_alligators_count_l82_8287

/-- Given the number of alligators seen by Samara and her friends, prove the total number of alligators seen. -/
theorem alligators_count (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : 
  samara_count = 20 → friend_count = 3 → friend_average = 10 →
  samara_count + friend_count * friend_average = 50 := by
  sorry


end NUMINAMATH_CALUDE_alligators_count_l82_8287


namespace NUMINAMATH_CALUDE_squared_roots_equation_l82_8240

/-- Given a quadratic equation x^2 + px + q = 0, this theorem proves that
    the equation x^2 - (p^2 - 2q)x + q^2 = 0 has roots that are the squares
    of the roots of the original equation. -/
theorem squared_roots_equation (p q : ℝ) :
  let original_eq (x : ℝ) := x^2 + p*x + q
  let new_eq (x : ℝ) := x^2 - (p^2 - 2*q)*x + q^2
  ∀ (r : ℝ), original_eq r = 0 → new_eq (r^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_squared_roots_equation_l82_8240


namespace NUMINAMATH_CALUDE_reggie_loses_by_21_points_l82_8254

/-- Represents the types of basketball shots -/
inductive ShotType
  | Layup
  | FreeThrow
  | ThreePointer
  | HalfCourt

/-- Returns the point value for a given shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.Layup => 1
  | ShotType.FreeThrow => 2
  | ShotType.ThreePointer => 3
  | ShotType.HalfCourt => 5

/-- Calculates the total points for a set of shots -/
def totalPoints (layups freeThrows threePointers halfCourt : ℕ) : ℕ :=
  layups * pointValue ShotType.Layup +
  freeThrows * pointValue ShotType.FreeThrow +
  threePointers * pointValue ShotType.ThreePointer +
  halfCourt * pointValue ShotType.HalfCourt

/-- Theorem stating the difference in points between Reggie's brother and Reggie -/
theorem reggie_loses_by_21_points :
  totalPoints 3 2 5 4 - totalPoints 4 3 2 1 = 21 := by
  sorry

#eval totalPoints 3 2 5 4 - totalPoints 4 3 2 1

end NUMINAMATH_CALUDE_reggie_loses_by_21_points_l82_8254


namespace NUMINAMATH_CALUDE_unique_solution_equation_l82_8232

theorem unique_solution_equation (x : ℝ) (h1 : x ≠ 0) :
  (9 * x) ^ 18 = (27 * x) ^ 9 ↔ x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l82_8232


namespace NUMINAMATH_CALUDE_angle_conversion_l82_8268

theorem angle_conversion (π : ℝ) :
  (12 : ℝ) * (π / 180) = π / 15 :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l82_8268


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l82_8280

/-- The system of linear equations -/
def system (x₁ x₂ x₃ : ℝ) : Prop :=
  x₁ + 2*x₂ + 4*x₃ = 5 ∧
  2*x₁ + x₂ + 5*x₃ = 7 ∧
  3*x₁ + 2*x₂ + 6*x₃ = 9

/-- The solution satisfies the system of equations -/
theorem solution_satisfies_system :
  system 1 0 1 := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l82_8280


namespace NUMINAMATH_CALUDE_prob_b_not_lose_l82_8284

/-- The probability that Player A wins a chess match. -/
def prob_a_wins : ℝ := 0.5

/-- The probability of a draw in a chess match. -/
def prob_draw : ℝ := 0.1

/-- The probability that Player B does not lose is equal to 0.5. -/
theorem prob_b_not_lose : 1 - prob_a_wins = 0.5 := by sorry

end NUMINAMATH_CALUDE_prob_b_not_lose_l82_8284
