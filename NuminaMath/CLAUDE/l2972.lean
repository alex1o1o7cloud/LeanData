import Mathlib

namespace NUMINAMATH_CALUDE_jovanas_shells_l2972_297252

theorem jovanas_shells (x : ℝ) : x + 15 + 17 = 37 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_jovanas_shells_l2972_297252


namespace NUMINAMATH_CALUDE_magical_stack_size_l2972_297217

/-- A magical stack is a stack of cards with the following properties:
  1. There are 3n cards numbered consecutively from 1 to 3n.
  2. Cards are divided into three piles A, B, and C, each with n cards.
  3. Cards are restacked alternately from C, B, and A.
  4. At least one card from each pile occupies its original position after restacking.
  5. Card number 101 retains its original position. -/
structure MagicalStack (n : ℕ) :=
  (total_cards : ℕ := 3 * n)
  (pile_size : ℕ := n)
  (card_101_position : ℕ)
  (is_magical : Bool)
  (card_101_retained : Bool)

/-- The theorem states that for a magical stack where card 101 retains its position,
    the total number of cards is 303. -/
theorem magical_stack_size (stack : MagicalStack n) 
  (h1 : stack.is_magical = true) 
  (h2 : stack.card_101_retained = true) 
  (h3 : stack.card_101_position = 101) :
  stack.total_cards = 303 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_size_l2972_297217


namespace NUMINAMATH_CALUDE_largest_interesting_number_l2972_297225

def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length ≥ 3 ∧
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_interesting_number :
  (∀ m : ℕ, is_interesting m → m ≤ 96433469) ∧ is_interesting 96433469 := by
  sorry

end NUMINAMATH_CALUDE_largest_interesting_number_l2972_297225


namespace NUMINAMATH_CALUDE_equation_is_linear_l2972_297243

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 1 = 20 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem: The equation 2x - 1 = 20 is a linear equation -/
theorem equation_is_linear : is_linear_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l2972_297243


namespace NUMINAMATH_CALUDE_simplified_ratio_of_boys_to_girls_l2972_297207

def number_of_boys : ℕ := 12
def number_of_girls : ℕ := 18

theorem simplified_ratio_of_boys_to_girls :
  (number_of_boys : ℚ) / (number_of_girls : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_ratio_of_boys_to_girls_l2972_297207


namespace NUMINAMATH_CALUDE_x_divisibility_l2972_297235

def x : ℕ := 36^2 + 48^2 + 64^3 + 81^2

theorem x_divisibility :
  (∃ k : ℕ, x = 3 * k) ∧
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 9 * k) ∧
  ¬(∃ k : ℕ, x = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_x_divisibility_l2972_297235


namespace NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l2972_297216

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem derivative_of_periodic_is_periodic
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (T : ℝ) (hT : T ≠ 0)
  (hf : Differentiable ℝ f)
  (hf' : ∀ x, HasDerivAt f (f' x) x)
  (hperiodic : IsPeriodic f T) :
  IsPeriodic f' T :=
sorry

end NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l2972_297216


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l2972_297202

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 2 = Real.sqrt (4^a * 2^b)) :
  (2/a + 1/b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    2 = Real.sqrt (4^a₀ * 2^b₀) ∧ (2/a₀ + 1/b₀) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l2972_297202


namespace NUMINAMATH_CALUDE_athletes_simultaneous_return_l2972_297226

/-- The time in minutes for Athlete A to complete one lap -/
def timeA : ℕ := 4

/-- The time in minutes for Athlete B to complete one lap -/
def timeB : ℕ := 5

/-- The time in minutes for Athlete C to complete one lap -/
def timeC : ℕ := 6

/-- The length of the circular track in meters -/
def trackLength : ℕ := 1000

/-- The time when all athletes simultaneously return to the starting point -/
def simultaneousReturnTime : ℕ := 60

theorem athletes_simultaneous_return :
  Nat.lcm (Nat.lcm timeA timeB) timeC = simultaneousReturnTime :=
sorry

end NUMINAMATH_CALUDE_athletes_simultaneous_return_l2972_297226


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2972_297256

theorem two_digit_number_property : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2972_297256


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2972_297281

theorem complex_fraction_equality : ∃ (i : ℂ), i * i = -1 ∧ (7 + i) / (3 + 4 * i) = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2972_297281


namespace NUMINAMATH_CALUDE_playground_dimensions_l2972_297258

theorem playground_dimensions :
  ∃! n : ℕ, n = (Finset.filter (fun pair : ℕ × ℕ =>
    pair.2 > pair.1 ∧
    (pair.1 - 4) * (pair.2 - 4) = 2 * pair.1 * pair.2 / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_playground_dimensions_l2972_297258


namespace NUMINAMATH_CALUDE_total_students_in_halls_l2972_297213

theorem total_students_in_halls (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 ∧ 
  biology = 2 * general ∧ 
  math = (3 * (general + biology)) / 5 → 
  general + biology + math = 144 :=
by sorry

end NUMINAMATH_CALUDE_total_students_in_halls_l2972_297213


namespace NUMINAMATH_CALUDE_value_of_a_l2972_297299

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| = 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Theorem statement
theorem value_of_a (a : ℝ) : A ⊇ B a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2972_297299


namespace NUMINAMATH_CALUDE_sum_of_products_l2972_297201

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 139) :
  x*y + y*z + x*z = 80 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2972_297201


namespace NUMINAMATH_CALUDE_nancy_albums_l2972_297275

theorem nancy_albums (total_pictures : ℕ) (first_album : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pictures = 51)
  (h2 : first_album = 11)
  (h3 : pics_per_album = 5) :
  (total_pictures - first_album) / pics_per_album = 8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_albums_l2972_297275


namespace NUMINAMATH_CALUDE_three_digit_ends_in_five_divisible_by_five_l2972_297271

/-- A three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem three_digit_ends_in_five_divisible_by_five :
  ∀ N : ℕ, ThreeDigitInt N → onesDigit N = 5 → 
    (∃ k : ℕ, N = 5 * k) := by sorry

end NUMINAMATH_CALUDE_three_digit_ends_in_five_divisible_by_five_l2972_297271


namespace NUMINAMATH_CALUDE_wrong_observation_value_l2972_297208

theorem wrong_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 40)
  (h3 : correct_value = 45)
  (h4 : new_mean = 40.66)
  : ∃ (wrong_value : ℝ),
    n * new_mean - n * initial_mean = correct_value - wrong_value ∧
    wrong_value = 12 :=
by sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l2972_297208


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2972_297260

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 15) ↔ 
  (x < -3/2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2972_297260


namespace NUMINAMATH_CALUDE_median_in_70_74_interval_l2972_297214

/-- Represents the frequency of scores in each interval -/
structure ScoreFrequency where
  interval : ℕ × ℕ
  count : ℕ

/-- The list of score frequencies for the test -/
def scoreDistribution : List ScoreFrequency := [
  ⟨(80, 84), 16⟩,
  ⟨(75, 79), 12⟩,
  ⟨(70, 74), 6⟩,
  ⟨(65, 69), 3⟩,
  ⟨(60, 64), 2⟩,
  ⟨(55, 59), 20⟩,
  ⟨(50, 54), 22⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 81

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median score -/
def findMedianInterval (distribution : List ScoreFrequency) (medianPos : ℕ) : ℕ × ℕ :=
  sorry

/-- Theorem stating that the median score is in the interval 70-74 -/
theorem median_in_70_74_interval :
  findMedianInterval scoreDistribution medianPosition = (70, 74) := by
  sorry

end NUMINAMATH_CALUDE_median_in_70_74_interval_l2972_297214


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2972_297221

/-- Calculates the total cost of tickets for a high school musical performance. -/
def calculate_total_cost (adult_price : ℚ) (child_price : ℚ) (senior_price : ℚ) (student_price : ℚ)
  (num_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) (num_students : ℕ) : ℚ :=
  let adult_cost := num_adults * adult_price
  let child_cost := (num_children - 1) * child_price  -- Family package applied
  let senior_cost := num_seniors * senior_price * (1 - 1/10)  -- 10% senior discount
  let student_cost := 2 * student_price + (student_price / 2)  -- Student promotion
  adult_cost + child_cost + senior_cost + student_cost

/-- Theorem stating that the total cost for the given scenario is $103.30. -/
theorem total_cost_is_correct :
  calculate_total_cost 12 10 8 9 4 3 2 3 = 1033/10 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2972_297221


namespace NUMINAMATH_CALUDE_age_difference_l2972_297230

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 7 →
  albert_age - mary_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2972_297230


namespace NUMINAMATH_CALUDE_sam_above_average_l2972_297231

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := bridget_count - 2

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by the three observers -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

/-- Theorem stating that Sam counted 2 more shooting stars than the average -/
theorem sam_above_average : sam_count - average_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_sam_above_average_l2972_297231


namespace NUMINAMATH_CALUDE_smallest_number_with_three_prime_factors_ge_10_l2972_297293

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_three_prime_factors (n : ℕ) : Prop := sorry

def all_prime_factors_ge_10 (n : ℕ) : Prop := sorry

theorem smallest_number_with_three_prime_factors_ge_10 :
  ∀ n : ℕ, (has_exactly_three_prime_factors n ∧ all_prime_factors_ge_10 n) → n ≥ 2431 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_three_prime_factors_ge_10_l2972_297293


namespace NUMINAMATH_CALUDE_max_value_of_z_l2972_297250

theorem max_value_of_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = x + y / 2 ∧ z ≤ 7 ∧ ∀ (w : ℝ), w = x + y / 2 → w ≤ z :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2972_297250


namespace NUMINAMATH_CALUDE_remaining_squares_after_removal_l2972_297205

/-- Represents the initial arrangement of matchsticks -/
structure Arrangement where
  matchsticks : ℕ
  squares : ℕ

/-- Represents a claim about the arrangement after removing matchsticks -/
inductive Claim
  | A : Claim  -- 5 squares of size 1x1 remain
  | B : Claim  -- 3 squares of size 2x2 remain
  | C : Claim  -- All 3x3 squares remain
  | D : Claim  -- Removed matchsticks are all on different lines
  | E : Claim  -- Four of the removed matchsticks are on the same line

/-- The main theorem to be proved -/
theorem remaining_squares_after_removal 
  (initial : Arrangement)
  (removed : ℕ)
  (incorrect_claims : Finset Claim)
  (h1 : initial.matchsticks = 40)
  (h2 : initial.squares = 30)
  (h3 : removed = 5)
  (h4 : incorrect_claims.card = 2)
  (h5 : Claim.A ∈ incorrect_claims)
  (h6 : Claim.D ∈ incorrect_claims)
  (h7 : Claim.E ∉ incorrect_claims)
  (h8 : Claim.B ∉ incorrect_claims)
  (h9 : Claim.C ∉ incorrect_claims) :
  ∃ (final : Arrangement), final.squares = 28 :=
sorry

end NUMINAMATH_CALUDE_remaining_squares_after_removal_l2972_297205


namespace NUMINAMATH_CALUDE_fraction_ordering_l2972_297280

theorem fraction_ordering : (25 : ℚ) / 19 < 21 / 16 ∧ 21 / 16 < 23 / 17 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2972_297280


namespace NUMINAMATH_CALUDE_abs_sum_min_value_abs_sum_min_value_achieved_l2972_297241

theorem abs_sum_min_value (x : ℝ) : 
  |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| ≥ 6 :=
sorry

theorem abs_sum_min_value_achieved : 
  ∃ x : ℝ, |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| = 6 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_abs_sum_min_value_achieved_l2972_297241


namespace NUMINAMATH_CALUDE_inscribed_circle_areas_l2972_297249

-- Define the square with its diagonal
def square_diagonal : ℝ := 40

-- Define the theorem
theorem inscribed_circle_areas :
  let square_side := square_diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_radius := square_side / 2
  let circle_area := π * circle_radius ^ 2
  square_area = 800 ∧ circle_area = 200 * π := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_areas_l2972_297249


namespace NUMINAMATH_CALUDE_aladdin_gold_bars_l2972_297239

theorem aladdin_gold_bars (x : ℕ) : 
  (x + 1023000) / 1024 ≤ x := by sorry

end NUMINAMATH_CALUDE_aladdin_gold_bars_l2972_297239


namespace NUMINAMATH_CALUDE_power_of_three_division_l2972_297273

theorem power_of_three_division : (3 : ℕ) ^ 2023 / 9 = (3 : ℕ) ^ 2021 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_division_l2972_297273


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2972_297242

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2972_297242


namespace NUMINAMATH_CALUDE_max_right_angles_in_triangle_l2972_297203

theorem max_right_angles_in_triangle : ℕ :=
  -- Define the sum of angles in a triangle
  let sum_of_angles : ℝ := 180

  -- Define a right angle in degrees
  let right_angle : ℝ := 90

  -- Define the maximum number of right angles
  let max_right_angles : ℕ := 1

  -- Theorem statement
  max_right_angles

end NUMINAMATH_CALUDE_max_right_angles_in_triangle_l2972_297203


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2972_297262

theorem fraction_to_decimal : (51 : ℚ) / 160 = 0.31875 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2972_297262


namespace NUMINAMATH_CALUDE_container_capacity_l2972_297209

/-- Given a container where 8 liters represents 20% of its capacity,
    this theorem proves that the total capacity of 40 such containers is 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  (num_containers : ℝ) * container_capacity = 1600 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2972_297209


namespace NUMINAMATH_CALUDE_max_removed_squares_elegantly_destroyed_l2972_297220

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a domino --/
inductive Domino
  | Horizontal : Nat → Nat → Domino
  | Vertical : Nat → Nat → Domino

/-- Checks if a domino can be placed on the board --/
def canPlaceDomino (board : Chessboard) (d : Domino) : Prop :=
  match d with
  | Domino.Horizontal x y => 
      x < board.size ∧ y < board.size - 1 ∧ 
      (x, y) ∉ board.removed ∧ (x, y + 1) ∉ board.removed
  | Domino.Vertical x y => 
      x < board.size - 1 ∧ y < board.size ∧ 
      (x, y) ∉ board.removed ∧ (x + 1, y) ∉ board.removed

/-- Defines an "elegantly destroyed" board --/
def isElegantlyDestroyed (board : Chessboard) : Prop :=
  (∀ d : Domino, ¬canPlaceDomino board d) ∧
  (∀ s : Nat × Nat, s ∈ board.removed →
    ∃ d : Domino, canPlaceDomino { size := board.size, removed := board.removed.erase s } d)

/-- The main theorem --/
theorem max_removed_squares_elegantly_destroyed :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    isElegantlyDestroyed board ∧
    board.removed.card = 48 ∧
    (∀ (board' : Chessboard), board'.size = 8 →
      isElegantlyDestroyed board' →
      board'.removed.card ≤ 48) :=
  sorry

end NUMINAMATH_CALUDE_max_removed_squares_elegantly_destroyed_l2972_297220


namespace NUMINAMATH_CALUDE_passengers_in_nine_buses_l2972_297211

/-- Given that 110 passengers fit in 5 buses, prove that 198 passengers fit in 9 buses. -/
theorem passengers_in_nine_buses :
  ∀ (passengers_per_bus : ℕ),
    110 = 5 * passengers_per_bus →
    9 * passengers_per_bus = 198 := by
  sorry

end NUMINAMATH_CALUDE_passengers_in_nine_buses_l2972_297211


namespace NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l2972_297224

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def dodecahedral_die := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the dodecahedral die -/
def expected_value : ℚ := (dodecahedral_die.sum fun i => (i + 1 : ℚ) * prob i) / 1

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_dodecahedral_die_expected_value_l2972_297224


namespace NUMINAMATH_CALUDE_root_product_value_l2972_297265

theorem root_product_value : 
  ∀ (a b c d : ℝ), 
  (a^2 + 2000*a + 1 = 0) → 
  (b^2 + 2000*b + 1 = 0) → 
  (c^2 - 2008*c + 1 = 0) → 
  (d^2 - 2008*d + 1 = 0) → 
  (a+c)*(b+c)*(a-d)*(b-d) = 32064 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l2972_297265


namespace NUMINAMATH_CALUDE_ratio_proof_l2972_297295

def problem (A B : ℕ) : Prop :=
  A = 45 ∧ Nat.lcm A B = 180

theorem ratio_proof (A B : ℕ) (h : problem A B) : 
  A / B = 45 / 4 := by sorry

end NUMINAMATH_CALUDE_ratio_proof_l2972_297295


namespace NUMINAMATH_CALUDE_log_equation_solution_l2972_297272

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2972_297272


namespace NUMINAMATH_CALUDE_jake_snakes_l2972_297229

/-- The number of eggs each snake lays -/
def eggs_per_snake : ℕ := 2

/-- The price of a regular baby snake in dollars -/
def regular_price : ℕ := 250

/-- The price of the rare baby snake in dollars -/
def rare_price : ℕ := 4 * regular_price

/-- The total amount Jake received from selling the snakes in dollars -/
def total_revenue : ℕ := 2250

/-- The number of snakes Jake has -/
def num_snakes : ℕ := 3

theorem jake_snakes :
  num_snakes * eggs_per_snake * regular_price + (rare_price - regular_price) = total_revenue :=
sorry

end NUMINAMATH_CALUDE_jake_snakes_l2972_297229


namespace NUMINAMATH_CALUDE_eight_people_line_up_with_pair_l2972_297232

/-- The number of ways to arrange n people in a line. -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line, 
    with 2 specific people always standing together. -/
def arrangementsWithPair (n : ℕ) : ℕ :=
  2 * linearArrangements (n - 1)

/-- Theorem: There are 10080 ways for 8 people to line up
    with 2 specific people always standing together. -/
theorem eight_people_line_up_with_pair : 
  arrangementsWithPair 8 = 10080 := by
  sorry


end NUMINAMATH_CALUDE_eight_people_line_up_with_pair_l2972_297232


namespace NUMINAMATH_CALUDE_max_cylinder_surface_area_in_sphere_l2972_297204

/-- The maximum surface area of a cylinder inscribed in a sphere -/
theorem max_cylinder_surface_area_in_sphere (R : ℝ) (h_pos : R > 0) :
  ∃ (r h : ℝ),
    r > 0 ∧ h > 0 ∧
    R^2 = r^2 + (h/2)^2 ∧
    ∀ (r' h' : ℝ),
      r' > 0 → h' > 0 → R^2 = r'^2 + (h'/2)^2 →
      2 * π * r * (h + r) ≤ 2 * π * r' * (h' + r') →
      2 * π * r * (h + r) = R^2 * π * (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_max_cylinder_surface_area_in_sphere_l2972_297204


namespace NUMINAMATH_CALUDE_units_digit_of_p_squared_plus_3_to_p_l2972_297222

theorem units_digit_of_p_squared_plus_3_to_p (p : ℕ) : 
  p = 2017^3 + 3^2017 → (p^2 + 3^p) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_p_squared_plus_3_to_p_l2972_297222


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2972_297228

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2972_297228


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2972_297212

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 2 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 2000 →
  c = 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2972_297212


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l2972_297296

theorem quadratic_equation_value (x : ℝ) : 2*x^2 + 3*x + 7 = 8 → 9 - 4*x^2 - 6*x = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l2972_297296


namespace NUMINAMATH_CALUDE_cost_price_is_100_l2972_297210

/-- Given a toy's cost price, calculate the final selling price after markup and discount --/
def final_price (cost : ℝ) : ℝ := cost * 1.5 * 0.8

/-- The profit made on the toy --/
def profit (cost : ℝ) : ℝ := final_price cost - cost

/-- Theorem stating that if the profit is 20 yuan, the cost price must be 100 yuan --/
theorem cost_price_is_100 : 
  ∀ x : ℝ, profit x = 20 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_100_l2972_297210


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2972_297264

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 - 3 = p.1 - 2 ∧ p ≠ (2, 3)}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (M ∪ N)ᶜ = {(2, 3)} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2972_297264


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l2972_297215

theorem square_sum_equals_one (a b : ℝ) (h : a = 1 - b) : a^2 + 2*a*b + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l2972_297215


namespace NUMINAMATH_CALUDE_third_butcher_packages_l2972_297246

/-- Represents the number of packages delivered by each butcher and their delivery times -/
structure Delivery :=
  (x y z : ℕ)
  (t1 t2 t3 : ℕ)

/-- Defines the conditions of the delivery problem -/
def DeliveryProblem (d : Delivery) : Prop :=
  d.x = 10 ∧
  d.y = 7 ∧
  d.t1 = 8 ∧
  d.t2 = 10 ∧
  d.t3 = 18 ∧
  4 * d.x + 4 * d.y + 4 * d.z = 100

/-- Theorem stating that under the given conditions, the third butcher delivered 8 packages -/
theorem third_butcher_packages (d : Delivery) (h : DeliveryProblem d) : d.z = 8 := by
  sorry

end NUMINAMATH_CALUDE_third_butcher_packages_l2972_297246


namespace NUMINAMATH_CALUDE_distribute_four_among_five_l2972_297283

/-- The number of ways to distribute n identical objects among k people,
    where each person receives at most one object and all objects must be distributed. -/
def distribute (n k : ℕ) : ℕ :=
  if n = k - 1 then k else 0

theorem distribute_four_among_five :
  distribute 4 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_among_five_l2972_297283


namespace NUMINAMATH_CALUDE_inequality_proof_l2972_297236

theorem inequality_proof (x y z : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x < 2) (hy_bound : y < 2) (hz_bound : z < 2)
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  (3/2 : ℝ) < (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) ∧ 
  (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2972_297236


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2972_297263

/-- Given an arithmetic sequence {aₙ}, prove that a₁₈ = 8 when a₄ + a₈ = 10 and a₁₀ = 6 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 4 + a 8 = 10 →
  a 10 = 6 →
  a 18 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2972_297263


namespace NUMINAMATH_CALUDE_apple_to_mango_ratio_l2972_297257

/-- Represents the total produce of fruits in kilograms -/
structure FruitProduce where
  apples : ℝ
  mangoes : ℝ
  oranges : ℝ

/-- Represents the fruit selling details -/
structure FruitSale where
  price_per_kg : ℝ
  total_amount : ℝ

/-- Theorem stating the ratio of apple to mango production -/
theorem apple_to_mango_ratio (fp : FruitProduce) (fs : FruitSale) :
  fp.mangoes = 400 ∧
  fp.oranges = fp.mangoes + 200 ∧
  fs.price_per_kg = 50 ∧
  fs.total_amount = 90000 ∧
  fs.total_amount = fs.price_per_kg * (fp.apples + fp.mangoes + fp.oranges) →
  fp.apples / fp.mangoes = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_apple_to_mango_ratio_l2972_297257


namespace NUMINAMATH_CALUDE_average_study_time_difference_l2972_297200

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The differences in study time on weekdays -/
def weekday_differences : List ℤ := [5, -5, 15, 25, -15]

/-- The additional time Sasha studied on weekends compared to usual -/
def weekend_additional_time : ℤ := 15

/-- The average difference in study time per day -/
def average_difference : ℚ := 12

theorem average_study_time_difference :
  (weekday_differences.sum + 2 * (weekend_additional_time + 15)) / days_in_week = average_difference := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l2972_297200


namespace NUMINAMATH_CALUDE_rectangle_tiling_exists_l2972_297292

/-- A tiling of a rectangle using two layers of 1 × 2 bricks -/
structure Tiling (n m : ℕ) :=
  (layer1 : Fin n → Fin (2*m) → Bool)
  (layer2 : Fin n → Fin (2*m) → Bool)

/-- Predicate to check if a tiling is valid -/
def is_valid_tiling (n m : ℕ) (t : Tiling n m) : Prop :=
  (∀ i j, t.layer1 i j ∨ t.layer2 i j) ∧ 
  (∀ i j, ¬(t.layer1 i j ∧ t.layer2 i j))

/-- Main theorem: A valid tiling exists for any rectangle n × 2m where n > 1 -/
theorem rectangle_tiling_exists (n m : ℕ) (h : n > 1) : 
  ∃ t : Tiling n m, is_valid_tiling n m t :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_exists_l2972_297292


namespace NUMINAMATH_CALUDE_grid_solution_l2972_297227

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two cells are adjacent in the grid -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val))

/-- The given grid with known values -/
def given_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 1 then 1
    else if i = 0 ∧ j = 2 then 9
    else if i = 1 ∧ j = 0 then 3
    else if i = 1 ∧ j = 1 then 5
    else if i = 2 ∧ j = 2 then 7
    else 0  -- placeholder for unknown values

theorem grid_solution :
  ∀ g : Grid,
  (∀ i j, g i j ∈ Finset.range 10) →  -- all numbers are from 1 to 9
  (∀ a b, adjacent a b → g a.1 a.2 + g b.1 b.2 < 12) →  -- sum of adjacent cells < 12
  (∀ i j, given_grid i j ≠ 0 → g i j = given_grid i j) →  -- matches given values
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_solution_l2972_297227


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l2972_297282

theorem sqrt_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l2972_297282


namespace NUMINAMATH_CALUDE_amusement_park_visits_l2972_297234

theorem amusement_park_visits 
  (season_pass_cost : ℕ) 
  (cost_per_trip : ℕ) 
  (youngest_son_visits : ℕ) 
  (oldest_son_visits : ℕ) : 
  season_pass_cost = 100 → 
  cost_per_trip = 4 → 
  youngest_son_visits = 15 → 
  oldest_son_visits * cost_per_trip = season_pass_cost - (youngest_son_visits * cost_per_trip) → 
  oldest_son_visits = 10 := by
sorry

end NUMINAMATH_CALUDE_amusement_park_visits_l2972_297234


namespace NUMINAMATH_CALUDE_ant_path_theorem_l2972_297255

/-- Represents the three concentric square paths -/
structure SquarePaths where
  a : ℝ  -- Side length of the smallest square
  b : ℝ  -- Side length of the middle square
  c : ℝ  -- Side length of the largest square
  h1 : 0 < a
  h2 : a < b
  h3 : b < c

/-- Represents the positions of the three ants -/
structure AntPositions (p : SquarePaths) where
  mu : ℝ  -- Distance traveled by Mu
  ra : ℝ  -- Distance traveled by Ra
  vey : ℝ  -- Distance traveled by Vey
  h1 : mu = p.c  -- Mu reaches the lower-right corner of the largest square
  h2 : ra = p.c - 1  -- Ra's position on the right side of the middle square
  h3 : vey = 2 * (p.c - p.b + 1)  -- Vey's position on the right side of the smallest square

/-- The main theorem stating the conditions and the result -/
theorem ant_path_theorem (p : SquarePaths) (pos : AntPositions p) :
  (p.c - p.b = p.b - p.a) ∧ (p.b - p.a = 2) →
  p.a = 4 ∧ p.b = 6 ∧ p.c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_theorem_l2972_297255


namespace NUMINAMATH_CALUDE_roots_sum_powers_l2972_297286

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → 
  b^2 - 6*b + 8 = 0 → 
  a^5 + a^3*b^3 + b^5 = -568 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l2972_297286


namespace NUMINAMATH_CALUDE_square_room_tiles_and_triangles_l2972_297297

theorem square_room_tiles_and_triangles (n : ℕ) : 
  n > 0 →  -- Ensure the room has a positive side length
  (2 * n - 1 = 57) →  -- Total tiles on diagonals
  (n^2 = 841 ∧ 4 = 4) :=  -- Total tiles and number of triangles
by sorry

end NUMINAMATH_CALUDE_square_room_tiles_and_triangles_l2972_297297


namespace NUMINAMATH_CALUDE_factor_tree_value_l2972_297277

/-- Given a factor tree with the following relationships:
  X = Y * Z
  Y = 7 * F
  Z = 11 * G
  F = 7 * 2
  G = 3 * 2
  Prove that X = 12936 -/
theorem factor_tree_value (X Y Z F G : ℕ) 
  (h1 : X = Y * Z)
  (h2 : Y = 7 * F)
  (h3 : Z = 11 * G)
  (h4 : F = 7 * 2)
  (h5 : G = 3 * 2) : 
  X = 12936 := by
  sorry

#check factor_tree_value

end NUMINAMATH_CALUDE_factor_tree_value_l2972_297277


namespace NUMINAMATH_CALUDE_solution_set_and_roots_negative_at_two_implies_bound_l2972_297253

def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

theorem solution_set_and_roots (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := by sorry

theorem negative_at_two_implies_bound (b : ℝ) :
  (∀ a, f a b 2 < 0) →
  b < -1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_and_roots_negative_at_two_implies_bound_l2972_297253


namespace NUMINAMATH_CALUDE_venus_hall_rental_cost_l2972_297240

theorem venus_hall_rental_cost (caesars_rental : ℕ) (caesars_meal : ℕ) (venus_meal : ℕ) (guests : ℕ) :
  caesars_rental = 800 →
  caesars_meal = 30 →
  venus_meal = 35 →
  guests = 60 →
  ∃ venus_rental : ℕ, venus_rental = 500 ∧ 
    caesars_rental + guests * caesars_meal = venus_rental + guests * venus_meal :=
by sorry

end NUMINAMATH_CALUDE_venus_hall_rental_cost_l2972_297240


namespace NUMINAMATH_CALUDE_f_simplification_f_value_in_third_quadrant_l2972_297268

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by
  sorry

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : α > Real.pi ∧ α < 3 * Real.pi / 2) 
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_in_third_quadrant_l2972_297268


namespace NUMINAMATH_CALUDE_egg_grouping_l2972_297269

theorem egg_grouping (total_eggs : ℕ) (group_size : ℕ) (h1 : total_eggs = 9) (h2 : group_size = 3) :
  total_eggs / group_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_grouping_l2972_297269


namespace NUMINAMATH_CALUDE_allison_total_items_l2972_297288

/-- Represents the number of craft items bought by a person -/
structure CraftItems where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The problem setup -/
def craftProblem (marie allison : CraftItems) : Prop :=
  allison.glueSticks = marie.glueSticks + 8 ∧
  marie.constructionPaper = 6 * allison.constructionPaper ∧
  marie.glueSticks = 15 ∧
  marie.constructionPaper = 30

/-- The theorem to prove -/
theorem allison_total_items (marie allison : CraftItems) 
  (h : craftProblem marie allison) : 
  allison.glueSticks + allison.constructionPaper = 28 := by
  sorry


end NUMINAMATH_CALUDE_allison_total_items_l2972_297288


namespace NUMINAMATH_CALUDE_festival_guests_selection_l2972_297287

theorem festival_guests_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end NUMINAMATH_CALUDE_festival_guests_selection_l2972_297287


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2972_297298

/-- The complex number z = 2i / (1-i) corresponds to a point in the second quadrant of the complex plane. -/
theorem z_in_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (2 * Complex.I) / (1 - Complex.I) = Complex.mk x y := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2972_297298


namespace NUMINAMATH_CALUDE_valid_pairs_l2972_297284

def is_valid_pair (m n : ℕ+) : Prop :=
  let d := Nat.gcd m n
  m + n^2 + d^3 = m * n * d

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 6) ∨ (m = 5 ∧ n = 2) ∨ (m = 5 ∧ n = 3) :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_l2972_297284


namespace NUMINAMATH_CALUDE_operation_equations_l2972_297238

theorem operation_equations :
  (37.3 / (1/2) = 74 + 3/5) ∧
  (33/40 * 10/11 = 0.75) ∧
  (0.45 - 1/20 = 2/5) ∧
  (0.375 + 1/40 = 0.4) := by
sorry

end NUMINAMATH_CALUDE_operation_equations_l2972_297238


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2972_297233

theorem cylinder_surface_area (V : Real) (d : Real) (h : Real) : 
  V = 500 * Real.pi / 3 →  -- Volume of the sphere
  d = 8 →                  -- Diameter of the cylinder base
  h = 6 →                  -- Height of the cylinder (derived from the problem)
  2 * Real.pi * (d/2) * h + 2 * Real.pi * (d/2)^2 = 80 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2972_297233


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2972_297254

-- Define the first equation
def equation1 (x : ℝ) : Prop := 3 * x - 5 = 6 * x - 8

-- Define the second equation
def equation2 (x : ℝ) : Prop := (x + 1) / 2 - (2 * x - 1) / 3 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2972_297254


namespace NUMINAMATH_CALUDE_problem_statement_l2972_297223

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 3) 
  (h3 : a * x + b * y = 4) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2972_297223


namespace NUMINAMATH_CALUDE_chocolate_division_l2972_297247

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_shaina : ℕ) :
  total_chocolate = 35 / 4 ∧
  num_piles = 5 ∧
  piles_to_shaina = 2 →
  piles_to_shaina * (total_chocolate / num_piles) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l2972_297247


namespace NUMINAMATH_CALUDE_rubble_initial_money_l2972_297270

/-- The amount of money Rubble had initially -/
def initial_money : ℝ := 15

/-- The cost of a notebook -/
def notebook_cost : ℝ := 4

/-- The cost of a pen -/
def pen_cost : ℝ := 1.5

/-- The number of notebooks Rubble bought -/
def num_notebooks : ℕ := 2

/-- The number of pens Rubble bought -/
def num_pens : ℕ := 2

/-- The amount of money Rubble had left after the purchase -/
def money_left : ℝ := 4

theorem rubble_initial_money :
  initial_money = 
    (num_notebooks : ℝ) * notebook_cost + 
    (num_pens : ℝ) * pen_cost + 
    money_left :=
by
  sorry

end NUMINAMATH_CALUDE_rubble_initial_money_l2972_297270


namespace NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l2972_297237

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primesBetween1And15 : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15 ∧ isPrime n}

theorem short_bingo_first_column_possibilities :
  Nat.factorial 6 / Nat.factorial 1 = 720 :=
sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l2972_297237


namespace NUMINAMATH_CALUDE_stratified_sample_factory_a_l2972_297218

theorem stratified_sample_factory_a (total : ℕ) (factory_a : ℕ) (sample_size : ℕ)
  (h_total : total = 98)
  (h_factory_a : factory_a = 56)
  (h_sample_size : sample_size = 14) :
  (factory_a : ℚ) / total * sample_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_factory_a_l2972_297218


namespace NUMINAMATH_CALUDE_kristin_reads_half_l2972_297251

/-- The number of books Peter and Kristin need to read -/
def total_books : ℕ := 20

/-- Peter's reading speed in hours per book -/
def peter_speed : ℚ := 18

/-- The ratio of Kristin's reading speed to Peter's -/
def speed_ratio : ℚ := 3

/-- The time Kristin has to read in hours -/
def kristin_time : ℚ := 540

/-- The portion of books Kristin reads in the given time -/
def kristin_portion : ℚ := kristin_time / (peter_speed * speed_ratio * total_books)

theorem kristin_reads_half :
  kristin_portion = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_kristin_reads_half_l2972_297251


namespace NUMINAMATH_CALUDE_subway_speed_increase_l2972_297290

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- State the theorem
theorem subway_speed_increase (s : ℝ) : 
  0 ≤ s ∧ s ≤ 7 → 
  speed s = speed 5 + 28 → 
  s = 7 := by
  sorry

end NUMINAMATH_CALUDE_subway_speed_increase_l2972_297290


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2972_297289

theorem greatest_integer_fraction_inequality : 
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l2972_297289


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2972_297276

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2972_297276


namespace NUMINAMATH_CALUDE_parade_runner_time_l2972_297285

/-- The time taken for a runner to travel from the front to the end of a moving parade -/
theorem parade_runner_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_speed = 6 →
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_parade_runner_time_l2972_297285


namespace NUMINAMATH_CALUDE_dave_cleaning_time_l2972_297267

/-- Proves that Dave's cleaning time is 15 minutes, given Carla's cleaning time and the ratio of Dave's to Carla's time -/
theorem dave_cleaning_time (carla_time : ℕ) (dave_ratio : ℚ) : 
  carla_time = 40 → dave_ratio = 3/8 → dave_ratio * carla_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_cleaning_time_l2972_297267


namespace NUMINAMATH_CALUDE_book_page_words_l2972_297274

theorem book_page_words (total_pages : ℕ) (words_per_page : ℕ) : 
  total_pages = 150 →
  50 ≤ words_per_page →
  words_per_page ≤ 150 →
  (total_pages * words_per_page) % 221 = 217 →
  words_per_page = 135 := by
sorry

end NUMINAMATH_CALUDE_book_page_words_l2972_297274


namespace NUMINAMATH_CALUDE_ball_problem_l2972_297261

theorem ball_problem (x : ℕ) : 
  (x > 0) →                                      -- Ensure x is positive
  ((x + 1) / (2 * x + 1) - x / (2 * x) = 1 / 22) →  -- Probability condition
  (2 * x = 10) :=                                -- Conclusion
by sorry

end NUMINAMATH_CALUDE_ball_problem_l2972_297261


namespace NUMINAMATH_CALUDE_roden_gold_fish_l2972_297219

/-- The number of fish Roden bought in total -/
def total_fish : ℕ := 22

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := total_fish - blue_fish

theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_roden_gold_fish_l2972_297219


namespace NUMINAMATH_CALUDE_proposition_correctness_l2972_297279

theorem proposition_correctness : 
  (∃ (S : Finset (Prop)), 
    S.card = 4 ∧ 
    (∃ (incorrect : Finset (Prop)), 
      incorrect ⊆ S ∧ 
      incorrect.card = 2 ∧
      (∀ p ∈ S, p ∈ incorrect ↔ ¬p) ∧
      (∃ p ∈ S, p = (∀ (p q : Prop), p ∨ q → p ∧ q)) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
                   (∃ y : ℝ, y^2 - 4*y - 5 > 0 ∧ y ≤ 5)) ∧
      (∃ p ∈ S, p = ((¬∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0))) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → x^2 - 3*x + 2 ≠ 0)))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_correctness_l2972_297279


namespace NUMINAMATH_CALUDE_vegetables_in_box_l2972_297291

/-- Given a box with cabbages and radishes, we define the total number of vegetables -/
def total_vegetables (num_cabbages num_radishes : ℕ) : ℕ :=
  num_cabbages + num_radishes

/-- Theorem: In a box with 3 cabbages and 2 radishes, there are 5 vegetables in total -/
theorem vegetables_in_box : total_vegetables 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_in_box_l2972_297291


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_two_l2972_297244

theorem sum_of_coefficients_equals_negative_two :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_two_l2972_297244


namespace NUMINAMATH_CALUDE_cement_calculation_l2972_297266

theorem cement_calculation (initial bought total : ℕ) 
  (h1 : initial = 98)
  (h2 : bought = 215)
  (h3 : total = 450) :
  total - (initial + bought) = 137 := by
  sorry

end NUMINAMATH_CALUDE_cement_calculation_l2972_297266


namespace NUMINAMATH_CALUDE_bridgette_dog_baths_l2972_297259

/-- The number of times Bridgette bathes her dogs each month -/
def dog_baths_per_month : ℕ := sorry

/-- The number of dogs Bridgette has -/
def num_dogs : ℕ := 2

/-- The number of cats Bridgette has -/
def num_cats : ℕ := 3

/-- The number of birds Bridgette has -/
def num_birds : ℕ := 4

/-- The number of times Bridgette bathes her cats each month -/
def cat_baths_per_month : ℕ := 1

/-- The number of times Bridgette bathes her birds each month -/
def bird_baths_per_month : ℚ := 1/4

/-- The total number of baths Bridgette gives in a year -/
def total_baths_per_year : ℕ := 96

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem bridgette_dog_baths : 
  dog_baths_per_month = 2 :=
by sorry

end NUMINAMATH_CALUDE_bridgette_dog_baths_l2972_297259


namespace NUMINAMATH_CALUDE_inscribed_square_area_is_2210_l2972_297278

/-- Represents a triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  is_inscribed : square_side > 0

/-- The area of the inscribed square in the given triangle -/
def inscribed_square_area (t : TriangleWithInscribedSquare) : ℝ :=
  t.square_side^2

/-- Theorem: The area of the inscribed square is 2210 when PQ = 34 and PR = 65 -/
theorem inscribed_square_area_is_2210
    (t : TriangleWithInscribedSquare)
    (h_pq : t.pq = 34)
    (h_pr : t.pr = 65) :
    inscribed_square_area t = 2210 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_is_2210_l2972_297278


namespace NUMINAMATH_CALUDE_parabola_focus_l2972_297245

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/4) * p.1^2}

/-- The focus of a parabola is a point from which the distance to any point on the parabola
    is equal to the distance from that point to a fixed line called the directrix -/
def is_focus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (d : ℝ), ∀ x y : ℝ, (x, y) ∈ p → 
    (x - f.1)^2 + (y - f.2)^2 = (y + d)^2

/-- The theorem stating that the focus of the parabola y = (1/4)x^2 is at (0, 1) -/
theorem parabola_focus :
  is_focus (0, 1) parabola := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2972_297245


namespace NUMINAMATH_CALUDE_five_greater_than_two_sqrt_five_l2972_297294

theorem five_greater_than_two_sqrt_five : 5 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_five_greater_than_two_sqrt_five_l2972_297294


namespace NUMINAMATH_CALUDE_max_sum_same_color_as_center_l2972_297206

/-- Represents a 5x5 checkerboard grid with alternating colors -/
def Grid := Fin 5 → Fin 5 → Bool

/-- A valid numbering of the grid satisfies the adjacent consecutive property -/
def ValidNumbering (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : Prop := sorry

/-- The sum of numbers in squares of the same color as the center square -/
def SumSameColorAsCenter (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : ℕ := sorry

/-- The maximum sum of numbers in squares of the same color as the center square -/
def MaxSumSameColorAsCenter (g : Grid) : ℕ := sorry

theorem max_sum_same_color_as_center (g : Grid) :
  MaxSumSameColorAsCenter g = 169 := by sorry

end NUMINAMATH_CALUDE_max_sum_same_color_as_center_l2972_297206


namespace NUMINAMATH_CALUDE_faye_pencils_count_l2972_297248

theorem faye_pencils_count (rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : rows = 14) (h2 : pencils_per_row = 11) : 
  rows * pencils_per_row = 154 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_count_l2972_297248
