import Mathlib

namespace twenty_apples_fourteen_cucumbers_l107_10762

/-- Represents the cost of a single apple -/
def apple_cost : ℝ := sorry

/-- Represents the cost of a single banana -/
def banana_cost : ℝ := sorry

/-- Represents the cost of a single cucumber -/
def cucumber_cost : ℝ := sorry

/-- The cost of 10 apples equals the cost of 5 bananas -/
axiom ten_apples_five_bananas : 10 * apple_cost = 5 * banana_cost

/-- The cost of 5 bananas equals the cost of 7 cucumbers -/
axiom five_bananas_seven_cucumbers : 5 * banana_cost = 7 * cucumber_cost

/-- Theorem: The cost of 20 apples equals the cost of 14 cucumbers -/
theorem twenty_apples_fourteen_cucumbers : 20 * apple_cost = 14 * cucumber_cost := by
  sorry

end twenty_apples_fourteen_cucumbers_l107_10762


namespace peanuts_remaining_l107_10796

theorem peanuts_remaining (initial : ℕ) (eaten_by_bonita : ℕ) : 
  initial = 148 → 
  eaten_by_bonita = 29 → 
  82 = initial - (initial / 4) - eaten_by_bonita := by
  sorry

end peanuts_remaining_l107_10796


namespace planes_intersect_l107_10709

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (are_skew : Line → Line → Prop)
variable (is_perpendicular_to_plane : Line → Plane → Prop)
variable (are_intersecting : Plane → Plane → Prop)

-- State the theorem
theorem planes_intersect (a b : Line) (α β : Plane) 
  (h1 : are_skew a b)
  (h2 : is_perpendicular_to_plane a α)
  (h3 : is_perpendicular_to_plane b β) :
  are_intersecting α β :=
sorry

end planes_intersect_l107_10709


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l107_10730

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt 1) / 2
  let r₂ := (7 - Real.sqrt 1) / 2
  r₁ + r₂ = 7 := by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l107_10730


namespace shortened_card_area_l107_10765

/-- Represents a rectangular card with given dimensions -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Represents the amount by which each side is shortened -/
structure Shortening where
  length_reduction : ℝ
  width_reduction : ℝ

/-- Applies a shortening to a card -/
def apply_shortening (c : Card) (s : Shortening) : Card :=
  { length := c.length - s.length_reduction,
    width := c.width - s.width_reduction }

theorem shortened_card_area 
  (original : Card)
  (shortening : Shortening)
  (h1 : original.length = 5)
  (h2 : original.width = 7)
  (h3 : shortening.length_reduction = 2)
  (h4 : shortening.width_reduction = 1) :
  area (apply_shortening original shortening) = 18 := by
  sorry

end shortened_card_area_l107_10765


namespace weight_of_A_l107_10790

def avg_weight_ABC : ℝ := 60
def avg_weight_ABCD : ℝ := 65
def avg_weight_BCDE : ℝ := 64
def weight_difference_E_D : ℝ := 3

theorem weight_of_A (weight_A weight_B weight_C weight_D weight_E : ℝ) : 
  (weight_A + weight_B + weight_C) / 3 = avg_weight_ABC ∧
  (weight_A + weight_B + weight_C + weight_D) / 4 = avg_weight_ABCD ∧
  weight_E = weight_D + weight_difference_E_D ∧
  (weight_B + weight_C + weight_D + weight_E) / 4 = avg_weight_BCDE →
  weight_A = 87 := by
sorry

end weight_of_A_l107_10790


namespace unique_teammate_d_score_l107_10750

-- Define the scoring system
def single_points : ℕ := 1
def double_points : ℕ := 2
def triple_points : ℕ := 3
def home_run_points : ℕ := 4

-- Define the total team score
def total_team_score : ℕ := 68

-- Define Faye's score
def faye_score : ℕ := 28

-- Define Teammate A's score components
def teammate_a_singles : ℕ := 1
def teammate_a_doubles : ℕ := 3
def teammate_a_home_runs : ℕ := 1

-- Define Teammate B's score components
def teammate_b_singles : ℕ := 4
def teammate_b_doubles : ℕ := 2
def teammate_b_triples : ℕ := 1

-- Define Teammate C's score components
def teammate_c_singles : ℕ := 2
def teammate_c_doubles : ℕ := 1
def teammate_c_triples : ℕ := 2
def teammate_c_home_runs : ℕ := 1

-- Theorem: There must be exactly one more player (Teammate D) who scored 4 points
theorem unique_teammate_d_score : 
  ∃! teammate_d_score : ℕ, 
    faye_score + 
    (teammate_a_singles * single_points + teammate_a_doubles * double_points + teammate_a_home_runs * home_run_points) +
    (teammate_b_singles * single_points + teammate_b_doubles * double_points + teammate_b_triples * triple_points) +
    (teammate_c_singles * single_points + teammate_c_doubles * double_points + teammate_c_triples * triple_points + teammate_c_home_runs * home_run_points) +
    teammate_d_score = total_team_score ∧ 
    teammate_d_score = 4 := by sorry

end unique_teammate_d_score_l107_10750


namespace abc_sqrt_problem_l107_10776

theorem abc_sqrt_problem (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 72 := by
  sorry

end abc_sqrt_problem_l107_10776


namespace distance_philadelphia_los_angeles_l107_10794

/-- The distance between two points on a complex plane, where one point is at (1950, 1950) and the other is at (0, 0), is equal to 1950√2. -/
theorem distance_philadelphia_los_angeles : 
  let philadelphia : ℂ := 1950 + 1950 * Complex.I
  let los_angeles : ℂ := 0
  Complex.abs (philadelphia - los_angeles) = 1950 * Real.sqrt 2 := by
  sorry

end distance_philadelphia_los_angeles_l107_10794


namespace moon_permutations_eq_12_l107_10760

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_12 : moon_permutations = 12 := by
  sorry

end moon_permutations_eq_12_l107_10760


namespace octal_54321_to_decimal_l107_10782

/-- Converts a base-8 digit to its base-10 equivalent -/
def octalToDecimal (digit : ℕ) : ℕ := digit

/-- Computes the value of a digit in a specific position in base 8 -/
def octalDigitValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (8 ^ position)

/-- Theorem: The base-10 equivalent of 54321 in base-8 is 22737 -/
theorem octal_54321_to_decimal : 
  octalToDecimal 1 + 
  octalDigitValue 2 1 + 
  octalDigitValue 3 2 + 
  octalDigitValue 4 3 + 
  octalDigitValue 5 4 = 22737 :=
by sorry

end octal_54321_to_decimal_l107_10782


namespace two_aces_probability_l107_10743

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of Aces in a standard deck
def numAces : ℕ := 4

-- Define the probability of drawing two Aces
def probTwoAces : ℚ := 1 / 221

-- Theorem statement
theorem two_aces_probability :
  (numAces / totalCards) * ((numAces - 1) / (totalCards - 1)) = probTwoAces := by
  sorry

end two_aces_probability_l107_10743


namespace exactly_one_zero_iff_m_eq_zero_or_nine_l107_10739

/-- A quadratic function of the form y = mx² - 6x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-6)^2 - 4 * m * 1

/-- The function f has exactly one zero -/
def has_exactly_one_zero (m : ℝ) : Prop :=
  (m = 0 ∧ ∃! x, f m x = 0) ∨
  (m ≠ 0 ∧ discriminant m = 0)

theorem exactly_one_zero_iff_m_eq_zero_or_nine (m : ℝ) :
  has_exactly_one_zero m ↔ m = 0 ∨ m = 9 := by
  sorry

end exactly_one_zero_iff_m_eq_zero_or_nine_l107_10739


namespace no_solution_iff_k_eq_ten_l107_10744

theorem no_solution_iff_k_eq_ten (k : ℝ) : 
  (∀ x : ℝ, (3*x + 1 ≠ 0 ∧ 5*x + 4 ≠ 0) → ((2*x - 4)/(3*x + 1) ≠ (2*x - k)/(5*x + 4))) ↔ 
  k = 10 :=
by sorry

end no_solution_iff_k_eq_ten_l107_10744


namespace min_tiles_for_floor_l107_10749

-- Define the length and breadth of the floor in centimeters
def floor_length : ℚ := 1625 / 100
def floor_width : ℚ := 1275 / 100

-- Define the function to calculate the number of tiles
def num_tiles (length width : ℚ) : ℕ :=
  let gcd := (Nat.gcd (Nat.floor (length * 100)) (Nat.floor (width * 100))) / 100
  let tile_area := gcd * gcd
  let floor_area := length * width
  Nat.ceil (floor_area / tile_area)

-- Theorem statement
theorem min_tiles_for_floor : num_tiles floor_length floor_width = 3315 := by
  sorry

end min_tiles_for_floor_l107_10749


namespace cross_number_puzzle_digit_l107_10717

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def power_of_2 (m : ℕ) : ℕ := 2^m
def power_of_3 (n : ℕ) : ℕ := 3^n

def same_digit_position (a b : ℕ) (pos : ℕ) : Prop :=
  (a / 10^pos) % 10 = (b / 10^pos) % 10

theorem cross_number_puzzle_digit :
  ∃! d : ℕ, d < 10 ∧
    ∃ (m n pos : ℕ),
      is_three_digit (power_of_2 m) ∧
      is_three_digit (power_of_3 n) ∧
      same_digit_position (power_of_2 m) (power_of_3 n) pos ∧
      (power_of_2 m / 10^pos) % 10 = d :=
by
  sorry

end cross_number_puzzle_digit_l107_10717


namespace shoe_matching_probability_l107_10756

/-- Represents the number of pairs of shoes for each color -/
structure ShoeInventory :=
  (black : ℕ)
  (brown : ℕ)
  (gray : ℕ)
  (red : ℕ)

/-- Calculates the probability of picking a matching pair of different feet -/
def matchingProbability (inventory : ShoeInventory) : ℚ :=
  let totalShoes := 2 * (inventory.black + inventory.brown + inventory.gray + inventory.red)
  let matchingPairs := 
    inventory.black * (inventory.black - 1) +
    inventory.brown * (inventory.brown - 1) +
    inventory.gray * (inventory.gray - 1) +
    inventory.red * (inventory.red - 1)
  ↑matchingPairs / (totalShoes * (totalShoes - 1))

theorem shoe_matching_probability (inventory : ShoeInventory) :
  inventory.black = 8 ∧ 
  inventory.brown = 4 ∧ 
  inventory.gray = 3 ∧ 
  inventory.red = 2 →
  matchingProbability inventory = 93 / 551 :=
by sorry

end shoe_matching_probability_l107_10756


namespace square_minus_product_plus_square_l107_10706

theorem square_minus_product_plus_square : 7^2 - 4*5 + 2^2 = 33 := by
  sorry

end square_minus_product_plus_square_l107_10706


namespace largest_fraction_add_to_one_seventh_l107_10712

theorem largest_fraction_add_to_one_seventh :
  ∀ (a b : ℕ) (hb : 0 < b) (hb_lt_5 : b < 5),
    (1 : ℚ) / 7 + (a : ℚ) / b < 1 →
    (a : ℚ) / b ≤ 3 / 4 :=
by sorry

end largest_fraction_add_to_one_seventh_l107_10712


namespace four_digit_sum_l107_10774

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end four_digit_sum_l107_10774


namespace value_of_a_l107_10700

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

theorem value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) → a = 4 := by
  sorry

end value_of_a_l107_10700


namespace ellipse_intersection_theorem_l107_10763

/-- Given a line y = x - 1 intersecting an ellipse (x^2 / a^2) + (y^2 / (a^2 - 1)) = 1 
    where a > 1, if the circle with diameter AB (where A and B are intersection points) 
    passes through the left focus of the ellipse, then a = (√6 + √2) / 2 -/
theorem ellipse_intersection_theorem (a : ℝ) (h_a : a > 1) :
  let line := fun x : ℝ => x - 1
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / (a^2 - 1) = 1
  let intersection_points := {p : ℝ × ℝ | ellipse p.1 p.2 ∧ p.2 = line p.1}
  let circle := fun (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) => 
    (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2
  ∃ (A B : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ), 
    A ∈ intersection_points ∧ 
    B ∈ intersection_points ∧
    A ≠ B ∧
    circle c r A ∧
    circle c r B ∧
    circle c r (-1, 0) →
  a = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end ellipse_intersection_theorem_l107_10763


namespace ten_children_same_cards_l107_10724

/-- Represents the number of children who can form a specific word -/
structure WordCount where
  mama : ℕ
  nyanya : ℕ
  manya : ℕ

/-- Calculates the number of children with all three cards the same -/
def childrenWithSameCards (wc : WordCount) : ℕ :=
  wc.mama + wc.nyanya - wc.manya

/-- Theorem stating that 10 children have all three cards the same -/
theorem ten_children_same_cards (wc : WordCount) 
  (h_mama : wc.mama = 20)
  (h_nyanya : wc.nyanya = 30)
  (h_manya : wc.manya = 40) : 
  childrenWithSameCards wc = 10 := by
sorry

#eval childrenWithSameCards ⟨20, 30, 40⟩

end ten_children_same_cards_l107_10724


namespace area_two_sectors_l107_10793

/-- The area of a figure composed of two 45° sectors of a circle with radius 10 -/
theorem area_two_sectors (r : ℝ) (h : r = 10) : 
  2 * (π * r^2 * (45 / 360)) = 25 * π := by
  sorry

end area_two_sectors_l107_10793


namespace sum_of_z_values_l107_10785

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 2*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/2) := by sorry

end sum_of_z_values_l107_10785


namespace triangle_angle_B_l107_10704

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 2 → A = 30 * π / 180 → 
  (B = 45 * π / 180 ∨ B = 135 * π / 180) ↔ 
  (Real.sin B = b * Real.sin A / a) := by sorry

end triangle_angle_B_l107_10704


namespace cubic_function_extrema_l107_10778

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f' a x = 0 ∧ f' a y = 0

theorem cubic_function_extrema (a : ℝ) :
  has_max_and_min a → a < -3 ∨ a > 6 :=
sorry

end cubic_function_extrema_l107_10778


namespace calculation_proof_l107_10705

theorem calculation_proof :
  (2 * Real.sqrt 18 - 3 * Real.sqrt 2 - Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - 1)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2) = 3 - 2 * Real.sqrt 3) :=
by sorry

end calculation_proof_l107_10705


namespace expression_evaluation_l107_10786

theorem expression_evaluation : 
  let mixed_number : ℚ := 20 + 94 / 95
  let expression := (mixed_number * 1.65 - mixed_number + 7 / 20 * mixed_number) * 47.5 * 0.8 * 2.5
  expression = 1994 := by sorry

end expression_evaluation_l107_10786


namespace gate_code_combinations_l107_10755

theorem gate_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end gate_code_combinations_l107_10755


namespace max_distance_difference_l107_10758

def circle_C1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

def circle_C2 (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 9

def on_x_axis (x y : ℝ) : Prop := y = 0

theorem max_distance_difference :
  ∃ (max : ℝ),
    (∀ (Mx My Nx Ny Px Py : ℝ),
      circle_C1 Mx My →
      circle_C2 Nx Ny →
      on_x_axis Px Py →
      Real.sqrt ((Nx - Px)^2 + (Ny - Py)^2) -
      Real.sqrt ((Mx - Px)^2 + (My - Py)^2) ≤ max) ∧
    max = 4 + Real.sqrt 26 :=
by sorry

end max_distance_difference_l107_10758


namespace job_completion_proof_l107_10754

/-- The number of days initially planned for 6 workers to complete a job -/
def initial_days : ℕ := sorry

/-- The number of workers who started the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers joined -/
def days_before_joining : ℕ := 3

/-- The number of additional workers who joined -/
def additional_workers : ℕ := 4

/-- The number of days worked after additional workers joined -/
def days_after_joining : ℕ := 3

/-- The total number of worker-days required to complete the job -/
def total_worker_days : ℕ := initial_workers * initial_days

theorem job_completion_proof :
  total_worker_days = 
    initial_workers * days_before_joining + 
    (initial_workers + additional_workers) * days_after_joining ∧
  initial_days = 8 := by sorry

end job_completion_proof_l107_10754


namespace melanie_coin_count_l107_10732

/-- Represents the number of coins Melanie has or receives -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in dollars -/
def coinValue (coins : CoinCount) : ℚ :=
  (coins.dimes * 10 + coins.nickels * 5 + coins.quarters * 25) / 100

/-- Adds two CoinCount structures -/
def addCoins (a b : CoinCount) : CoinCount :=
  { dimes := a.dimes + b.dimes,
    nickels := a.nickels + b.nickels,
    quarters := a.quarters + b.quarters }

def initial : CoinCount := { dimes := 19, nickels := 12, quarters := 8 }
def fromDad : CoinCount := { dimes := 39, nickels := 22, quarters := 15 }
def fromSister : CoinCount := { dimes := 15, nickels := 7, quarters := 12 }
def fromMother : CoinCount := { dimes := 25, nickels := 10, quarters := 0 }
def fromGrandmother : CoinCount := { dimes := 0, nickels := 30, quarters := 3 }

theorem melanie_coin_count :
  let final := addCoins initial (addCoins fromDad (addCoins fromSister (addCoins fromMother fromGrandmother)))
  final.dimes = 98 ∧
  final.nickels = 81 ∧
  final.quarters = 38 ∧
  coinValue final = 2335 / 100 := by
  sorry

end melanie_coin_count_l107_10732


namespace exists_number_not_divisible_by_both_l107_10738

def numbers : List Nat := [3654, 3664, 3674, 3684, 3694]

def divisible_by_4 (n : Nat) : Prop := n % 4 = 0

def divisible_by_3 (n : Nat) : Prop := n % 3 = 0

def units_digit (n : Nat) : Nat := n % 10

def tens_digit (n : Nat) : Nat := (n / 10) % 10

theorem exists_number_not_divisible_by_both :
  ∃ n ∈ numbers, ¬(divisible_by_4 n ∧ divisible_by_3 n) ∧
  (units_digit n * tens_digit n = 28 ∨ units_digit n * tens_digit n = 36) :=
by sorry

end exists_number_not_divisible_by_both_l107_10738


namespace imaginary_part_of_z_l107_10707

theorem imaginary_part_of_z (z : ℂ) : z = (1 - Complex.I) / Complex.I → z.im = -1 := by
  sorry

end imaginary_part_of_z_l107_10707


namespace faster_speed_calculation_l107_10723

/-- Prove that given a person walks 50 km at 10 km/hr, if they walked at a faster speed,
    they would cover 20 km more in the same time, then the faster speed is 14 km/hr -/
theorem faster_speed_calculation (actual_distance : ℝ) (original_speed : ℝ) (additional_distance : ℝ)
    (h1 : actual_distance = 50)
    (h2 : original_speed = 10)
    (h3 : additional_distance = 20) :
  let total_distance := actual_distance + additional_distance
  let time := actual_distance / original_speed
  let faster_speed := total_distance / time
  faster_speed = 14 := by
sorry

end faster_speed_calculation_l107_10723


namespace expand_difference_of_squares_simplify_fraction_l107_10716

-- Define a as a real number
variable (a : ℝ)

-- Theorem 1: (a+2)(a-2) = a^2 - 4
theorem expand_difference_of_squares : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

-- Theorem 2: (a^2-4)/(a+2) + 2 = a
theorem simplify_fraction : (a^2 - 4) / (a + 2) + 2 = a := by
  sorry

end expand_difference_of_squares_simplify_fraction_l107_10716


namespace divisibility_of_expression_l107_10719

theorem divisibility_of_expression (x : ℤ) (h : Odd x) :
  ∃ k : ℤ, (8 * x + 6) * (8 * x + 10) * (4 * x + 4) = 384 * k := by
  sorry

end divisibility_of_expression_l107_10719


namespace final_cost_calculation_l107_10779

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def discount_rate : ℝ := 0.1

theorem final_cost_calculation :
  let total_cost : ℝ := washing_machine_cost + dryer_cost
  let discount_amount : ℝ := total_cost * discount_rate
  let final_cost : ℝ := total_cost - discount_amount
  final_cost = 153 := by sorry

end final_cost_calculation_l107_10779


namespace quadratic_radicals_same_type_l107_10729

theorem quadratic_radicals_same_type (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ a - 3 = k * (12 - 2*a)) → a = 5 := by
  sorry

end quadratic_radicals_same_type_l107_10729


namespace gcd_lcm_product_90_135_l107_10746

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end gcd_lcm_product_90_135_l107_10746


namespace remainder_two_power_33_mod_9_l107_10736

theorem remainder_two_power_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_two_power_33_mod_9_l107_10736


namespace number_of_possible_lists_l107_10710

def num_balls : ℕ := 15
def list_length : ℕ := 4

theorem number_of_possible_lists :
  (num_balls ^ list_length : ℕ) = 50625 := by
  sorry

end number_of_possible_lists_l107_10710


namespace prism_tetrahedron_surface_area_ratio_l107_10775

/-- The ratio of surface areas of a rectangular prism to a tetrahedron --/
theorem prism_tetrahedron_surface_area_ratio :
  let prism_dimensions : Fin 3 → ℝ := ![2, 3, 4]
  let prism_surface_area := 2 * (prism_dimensions 0 * prism_dimensions 1 + 
                                 prism_dimensions 1 * prism_dimensions 2 + 
                                 prism_dimensions 0 * prism_dimensions 2)
  let tetrahedron_edge_length := Real.sqrt 13
  let tetrahedron_surface_area := Real.sqrt 3 * tetrahedron_edge_length ^ 2
  prism_surface_area / tetrahedron_surface_area = 4 * Real.sqrt 3 / 3 := by
  sorry

end prism_tetrahedron_surface_area_ratio_l107_10775


namespace holiday_savings_l107_10718

theorem holiday_savings (victory_savings sam_savings : ℕ) : 
  victory_savings = sam_savings - 100 →
  victory_savings + sam_savings = 1900 →
  sam_savings = 1000 := by
sorry

end holiday_savings_l107_10718


namespace total_notes_count_l107_10713

/-- Proves that given a total amount of Rs. 10350 in Rs. 50 and Rs. 500 notes,
    with 77 notes of Rs. 50 denomination, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (notes_50_count : ℕ) (notes_50_value : ℕ) (notes_500_value : ℕ) :
  total_amount = 10350 →
  notes_50_count = 77 →
  notes_50_value = 50 →
  notes_500_value = 500 →
  ∃ (notes_500_count : ℕ),
    total_amount = notes_50_count * notes_50_value + notes_500_count * notes_500_value ∧
    notes_50_count + notes_500_count = 90 :=
by sorry

end total_notes_count_l107_10713


namespace triangle_side_length_l107_10777

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  -- Given conditions
  (Real.cos A = Real.sqrt 5 / 5) →
  (Real.cos B = Real.sqrt 10 / 10) →
  (c = Real.sqrt 2) →
  -- Sine rule
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c / Real.sin C = a / Real.sin A) →
  -- Prove
  a = 4 * Real.sqrt 5 / 5 := by
sorry

end triangle_side_length_l107_10777


namespace min_minutes_for_cheaper_plan_b_l107_10787

/-- Represents the cost of a phone plan in cents -/
def PlanCost := ℕ → ℕ

/-- Cost function for Plan A: 10 cents per minute -/
def planA : PlanCost := λ minutes => 10 * minutes

/-- Cost function for Plan B: $20 flat fee (2000 cents) plus 5 cents per minute -/
def planB : PlanCost := λ minutes => 2000 + 5 * minutes

/-- Theorem stating that 401 is the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_cheaper_plan_b : 
  (∀ m : ℕ, m < 401 → planA m ≤ planB m) ∧ 
  (∀ m : ℕ, m ≥ 401 → planB m < planA m) := by
  sorry

end min_minutes_for_cheaper_plan_b_l107_10787


namespace subset_pair_existence_l107_10767

theorem subset_pair_existence (n : ℕ) (A : Fin n → Set ℕ) :
  ∃ (X Y : ℕ), ∀ i : Fin n, (X ∈ A i ∧ Y ∈ A i) ∨ (X ∉ A i ∧ Y ∉ A i) := by
  sorry

end subset_pair_existence_l107_10767


namespace even_difference_of_coefficients_l107_10745

theorem even_difference_of_coefficients (a₁ a₂ b₁ b₂ m n : ℤ) : 
  a₁ ≠ a₂ →
  m ≠ n →
  (m^2 + a₁*m + b₁ = n^2 + a₂*n + b₂) →
  (m^2 + a₂*m + b₂ = n^2 + a₁*n + b₁) →
  ∃ k : ℤ, a₁ - a₂ = 2 * k :=
by sorry

end even_difference_of_coefficients_l107_10745


namespace zero_descriptions_l107_10789

theorem zero_descriptions (x : ℝ) :
  (x = 0) ↔ 
  (∀ (y : ℝ), x ≤ y ∧ x ≥ y → y = x) ∧ 
  (∀ (y : ℝ), x + y = y) ∧
  (∀ (y : ℝ), x * y = x) :=
sorry

end zero_descriptions_l107_10789


namespace stratified_sample_bulbs_l107_10715

/-- Represents the types of bulbs -/
inductive BulbType
  | W20
  | W40
  | W60

/-- Calculates the number of bulbs of a given type in a sample -/
def sampleSize (totalBulbs : ℕ) (sampleBulbs : ℕ) (ratio : ℕ) (totalRatio : ℕ) : ℕ :=
  (ratio * totalBulbs * sampleBulbs) / (totalRatio * totalBulbs)

theorem stratified_sample_bulbs :
  let totalBulbs : ℕ := 400
  let sampleBulbs : ℕ := 40
  let ratio20W : ℕ := 4
  let ratio40W : ℕ := 3
  let ratio60W : ℕ := 1
  let totalRatio : ℕ := ratio20W + ratio40W + ratio60W
  (sampleSize totalBulbs sampleBulbs ratio20W totalRatio = 20) ∧
  (sampleSize totalBulbs sampleBulbs ratio40W totalRatio = 15) ∧
  (sampleSize totalBulbs sampleBulbs ratio60W totalRatio = 5) :=
by sorry

end stratified_sample_bulbs_l107_10715


namespace percentage_difference_l107_10799

theorem percentage_difference (third : ℝ) (first second : ℝ) 
  (h1 : first = 0.75 * third) 
  (h2 : second = first - 0.06 * first) : 
  (third - second) / third = 0.295 := by
  sorry

end percentage_difference_l107_10799


namespace centers_connection_line_l107_10727

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem centers_connection_line :
  ∃ (x1 y1 x2 y2 : ℝ),
    (∀ x y, circle1 x y ↔ (x - x1)^2 + (y - y1)^2 = (x1^2 + y1^2)) ∧
    (∀ x y, circle2 x y ↔ (x - x2)^2 + (y - y2)^2 = x2^2) ∧
    line_equation x1 y1 ∧
    line_equation x2 y2 :=
sorry

end centers_connection_line_l107_10727


namespace prob_three_tails_in_eight_flips_l107_10722

/-- The probability of flipping a tail -/
def p_tail : ℚ := 3/4

/-- The probability of flipping a head -/
def p_head : ℚ := 1/4

/-- The number of coin flips -/
def n_flips : ℕ := 8

/-- The number of tails we want to get -/
def n_tails : ℕ := 3

/-- The probability of getting exactly n_tails in n_flips of an unfair coin -/
def prob_exact_tails (n_flips n_tails : ℕ) (p_tail : ℚ) : ℚ :=
  (n_flips.choose n_tails) * (p_tail ^ n_tails) * ((1 - p_tail) ^ (n_flips - n_tails))

theorem prob_three_tails_in_eight_flips : 
  prob_exact_tails n_flips n_tails p_tail = 189/512 := by
  sorry

end prob_three_tails_in_eight_flips_l107_10722


namespace special_triangle_properties_l107_10768

/-- Triangle ABC with specific conditions -/
structure SpecialTriangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  angle_sum : A + B + C = π
  side_condition : a + c = 3 * Real.sqrt 3 / 2
  side_b : b = Real.sqrt 3
  angle_condition : 2 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  t.B = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.B = 5 * Real.sqrt 3 / 16) :=
by sorry

end special_triangle_properties_l107_10768


namespace bus_stops_count_l107_10769

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : ℕ
  stops_ge_three : stops ≥ 3

/-- Represents the city's bus system -/
structure BusSystem where
  routes : Finset BusRoute
  route_count : routes.card = 57
  all_connected : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → ∃! (s : ℕ), s ≤ r₁.stops ∧ s ≤ r₂.stops
  stops_equal : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → r₁.stops = r₂.stops

theorem bus_stops_count (bs : BusSystem) : ∀ (r : BusRoute), r ∈ bs.routes → r.stops = 8 := by
  sorry

end bus_stops_count_l107_10769


namespace triangle_area_l107_10735

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 is 64 -/
theorem triangle_area : Real := by
  -- Define the lines
  let line1 : Real → Real := λ x ↦ x
  let line2 : Real → Real := λ x ↦ -x
  let line3 : Real → Real := λ _ ↦ 8

  -- Define the intersection points
  let A : (Real × Real) := (8, 8)
  let B : (Real × Real) := (-8, 8)
  let O : (Real × Real) := (0, 0)

  -- Calculate the base and height of the triangle
  let base : Real := A.1 - B.1
  let height : Real := line3 0 - O.2

  -- Calculate the area
  let area : Real := (1 / 2) * base * height

  -- Prove that the area is 64
  sorry

end triangle_area_l107_10735


namespace cube_root_and_square_roots_l107_10798

theorem cube_root_and_square_roots (a b m : ℝ) : 
  (3 * a - 5)^(1/3) = -2 ∧ 
  m^2 = b ∧ 
  (1 - 5*m)^2 = b →
  a = -1 ∧ b = 1/16 := by
  sorry

end cube_root_and_square_roots_l107_10798


namespace cos_inv_third_over_pi_irrational_l107_10780

theorem cos_inv_third_over_pi_irrational : Irrational ((Real.arccos (1/3)) / Real.pi) := by
  sorry

end cos_inv_third_over_pi_irrational_l107_10780


namespace task_completion_time_l107_10784

/-- Ram's efficiency is half of Krish's, and Ram takes 27 days to complete a task alone.
    This theorem proves that Ram and Krish working together will complete the task in 9 days. -/
theorem task_completion_time (ram_efficiency krish_efficiency : ℝ) 
  (h1 : ram_efficiency = (1 / 2) * krish_efficiency) 
  (h2 : ram_efficiency * 27 = 1) : 
  (ram_efficiency + krish_efficiency) * 9 = 1 := by
  sorry

end task_completion_time_l107_10784


namespace sum_and_count_equals_851_l107_10701

/-- Sum of integers from a to b, inclusive -/
def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- Count of even integers from a to b, inclusive -/
def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

/-- The sum of integers from 30 to 50 (inclusive) plus the count of even integers
    in the same range equals 851 -/
theorem sum_and_count_equals_851 : sumIntegers 30 50 + countEvenIntegers 30 50 = 851 := by
  sorry

end sum_and_count_equals_851_l107_10701


namespace ab_geq_one_implies_conditions_l107_10708

theorem ab_geq_one_implies_conditions (a b : ℝ) (h : a * b ≥ 1) :
  a^2 ≥ 1 / b^2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end ab_geq_one_implies_conditions_l107_10708


namespace symmetric_point_coordinates_l107_10703

/-- Given a point P with coordinates (-2, 3), prove that the coordinates of the point symmetric to the origin with respect to P are (2, -3). -/
theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point := (-P.1, -P.2)
  symmetric_point = (2, -3) := by
  sorry

end symmetric_point_coordinates_l107_10703


namespace cube_root_three_identity_l107_10748

theorem cube_root_three_identity (t : ℝ) : 
  t = 1 / (1 - Real.rpow 3 (1/3)) → 
  t = -(1 + Real.rpow 3 (1/3) + Real.rpow 3 (2/3)) / 2 := by
  sorry

end cube_root_three_identity_l107_10748


namespace staircase_perimeter_l107_10731

/-- A staircase-shaped region with specific properties -/
structure StaircaseRegion where
  num_sides : ℕ
  side_length : ℝ
  total_area : ℝ

/-- The perimeter of a StaircaseRegion -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating the perimeter of a specific StaircaseRegion -/
theorem staircase_perimeter :
  ∀ (r : StaircaseRegion),
    r.num_sides = 12 ∧
    r.side_length = 1 ∧
    r.total_area = 120 →
    perimeter r = 36 := by sorry

end staircase_perimeter_l107_10731


namespace imaginary_part_of_z_l107_10752

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end imaginary_part_of_z_l107_10752


namespace shopping_cart_fruit_ratio_l107_10766

theorem shopping_cart_fruit_ratio (apples oranges pears : ℕ) : 
  oranges = 3 * apples →
  pears = 4 * oranges →
  (apples : ℚ) / pears = 1 / 12 := by
  sorry

end shopping_cart_fruit_ratio_l107_10766


namespace inequality_proof_l107_10771

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end inequality_proof_l107_10771


namespace symmetric_points_sum_l107_10753

/-- Two points are symmetric about the x-axis if their x-coordinates are the same
    and their y-coordinates are opposite numbers -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (b a : ℝ) :
  symmetric_about_x_axis (-2, b) (a, -3) → a + b = 1 := by
  sorry

end symmetric_points_sum_l107_10753


namespace xyz_sum_equals_zero_l107_10720

theorem xyz_sum_equals_zero 
  (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 48)
  (h_eq2 : y^2 + y*z + z^2 = 25)
  (h_eq3 : z^2 + x*z + x^2 = 73) :
  x*y + y*z + x*z = 0 :=
sorry

end xyz_sum_equals_zero_l107_10720


namespace sum_of_2001_numbers_positive_l107_10772

theorem sum_of_2001_numbers_positive 
  (a : Fin 2001 → ℝ) 
  (h : ∀ (i j k l : Fin 2001), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    a i + a j + a k + a l > 0) : 
  Finset.sum Finset.univ a > 0 := by
sorry

end sum_of_2001_numbers_positive_l107_10772


namespace table_tennis_arrangements_l107_10725

def total_players : ℕ := 10
def main_players : ℕ := 3
def match_players : ℕ := 5
def remaining_players : ℕ := total_players - main_players

theorem table_tennis_arrangements :
  (main_players.factorial) * (remaining_players.choose 2) = 252 :=
by sorry

end table_tennis_arrangements_l107_10725


namespace pipe_B_fill_time_l107_10734

/-- Time for pipe A to fill the tank -/
def time_A : ℝ := 5

/-- Time for the tank to drain -/
def time_drain : ℝ := 20

/-- Time to fill the tank with both pipes on and drainage open -/
def time_combined : ℝ := 3.6363636363636362

/-- Time for pipe B to fill the tank -/
def time_B : ℝ := 1.0526315789473684

/-- Theorem stating the relationship between the given times -/
theorem pipe_B_fill_time :
  time_B = (time_A * time_drain * time_combined) / 
           (time_A * time_drain - time_A * time_combined - time_drain * time_combined) :=
by sorry

end pipe_B_fill_time_l107_10734


namespace dan_total_limes_l107_10726

/-- The number of limes Dan picked -/
def limes_picked : ℕ := 9

/-- The number of limes Sara gave to Dan -/
def limes_given : ℕ := 4

/-- The total number of limes Dan has now -/
def total_limes : ℕ := limes_picked + limes_given

theorem dan_total_limes : total_limes = 13 := by
  sorry

end dan_total_limes_l107_10726


namespace triangle_inequality_specific_l107_10702

/-- Triangle inequality theorem for a specific triangle --/
theorem triangle_inequality_specific (a b c : ℝ) (ha : a = 5) (hb : b = 8) (hc : c = 6) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by sorry

end triangle_inequality_specific_l107_10702


namespace next_235_time_91_minutes_l107_10757

def is_valid_time (h m : ℕ) : Prop :=
  h < 24 ∧ m < 60

def uses_digits_235_once (h m : ℕ) : Prop :=
  let digits := h.digits 10 ++ m.digits 10
  digits.count 2 = 1 ∧ digits.count 3 = 1 ∧ digits.count 5 = 1

def minutes_from_352_to (h m : ℕ) : ℕ :=
  if h < 3 ∨ (h = 3 ∧ m ≤ 52) then
    (h + 24 - 3) * 60 + (m - 52)
  else
    (h - 3) * 60 + (m - 52)

theorem next_235_time_91_minutes :
  ∃ (h m : ℕ), 
    is_valid_time h m ∧
    uses_digits_235_once h m ∧
    minutes_from_352_to h m = 91 ∧
    (∀ (h' m' : ℕ), 
      is_valid_time h' m' →
      uses_digits_235_once h' m' →
      minutes_from_352_to h' m' ≥ 91) :=
sorry

end next_235_time_91_minutes_l107_10757


namespace average_marks_chemistry_mathematics_l107_10733

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: Physics marks, C: Chemistry marks, M: Mathematics marks
  (h : P + C + M = P + 130) -- Total marks condition
  : (C + M) / 2 = 65 := by
  sorry

end average_marks_chemistry_mathematics_l107_10733


namespace gcd_1237_1957_l107_10741

theorem gcd_1237_1957 : Nat.gcd 1237 1957 = 1 := by
  sorry

end gcd_1237_1957_l107_10741


namespace circle_symmetry_l107_10740

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the property of symmetry
def is_symmetric (circle1 circle2 : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ 
    circle2 x2 y2 ∧ 
    line ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Define our target circle
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- The main theorem
theorem circle_symmetry :
  is_symmetric given_circle target_circle symmetry_line :=
sorry

end circle_symmetry_l107_10740


namespace center_octahedron_volume_ratio_l107_10791

/-- A regular octahedron -/
structure RegularOctahedron where
  -- We don't need to define the structure fully, just declare it exists
  mk :: (dummy : Unit)

/-- The octahedron formed by the centers of faces of a regular octahedron -/
def center_octahedron (o : RegularOctahedron) : RegularOctahedron :=
  RegularOctahedron.mk ()

/-- The volume of an octahedron -/
def volume (o : RegularOctahedron) : ℝ :=
  sorry

/-- The theorem stating the volume ratio of the center octahedron to the original octahedron -/
theorem center_octahedron_volume_ratio (o : RegularOctahedron) :
  volume (center_octahedron o) / volume o = 1 / 9 := by
  sorry

end center_octahedron_volume_ratio_l107_10791


namespace w_sequence_properties_l107_10711

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequence w_n
def w : ℕ → ℂ
  | 0 => 1
  | 1 => i
  | (n + 2) => 2 * w (n + 1) + 3 * w n

-- State the theorem
theorem w_sequence_properties :
  (∀ n : ℕ, w n = (1 + i) / 4 * 3^n + (3 - i) / 4 * (-1)^n) ∧
  (∀ n : ℕ, n ≥ 1 → |Complex.re (w n) - Complex.im (w n)| = 1) := by
  sorry


end w_sequence_properties_l107_10711


namespace quadratic_roots_range_l107_10761

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (7 * x₁^2 - (a + 13) * x₁ + a^2 - a - 2 = 0) ∧ 
    (7 * x₂^2 - (a + 13) * x₂ + a^2 - a - 2 = 0) ∧ 
    (0 < x₁) ∧ (x₁ < 1) ∧ (1 < x₂) ∧ (x₂ < 2)) →
  ((-2 < a ∧ a < -1) ∨ (3 < a ∧ a < 4)) :=
by sorry

end quadratic_roots_range_l107_10761


namespace no_general_rational_solution_l107_10764

theorem no_general_rational_solution (k : ℚ) : 
  ¬ ∃ (S : Set ℝ), ∀ (x : ℝ), x ∈ S → 
    ∃ (q : ℚ), x + k * Real.sqrt (x^2 + 1) - 1 / (x + k * Real.sqrt (x^2 + 1)) = q :=
by sorry

end no_general_rational_solution_l107_10764


namespace floor_ceiling_sum_l107_10751

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.7 : ℝ)⌉ = 31 := by sorry

end floor_ceiling_sum_l107_10751


namespace work_completion_time_l107_10795

theorem work_completion_time (a b : ℕ) (h1 : a = 20) (h2 : (4 : ℝ) * ((1 : ℝ) / a + (1 : ℝ) / b) = (1 : ℝ) / 3) : b = 30 := by
  sorry

end work_completion_time_l107_10795


namespace squares_characterization_l107_10737

class MyGroup (G : Type) extends Group G where
  g : G
  h : G
  g_four : g ^ 4 = 1
  g_two_ne_one : g ^ 2 ≠ 1
  h_seven : h ^ 7 = 1
  h_ne_one : h ≠ 1
  gh_relation : g * h * g⁻¹ * h = 1
  subgroup_condition : ∀ (H : Subgroup G), g ∈ H → h ∈ H → H = ⊤

variable {G : Type} [MyGroup G]

def squares (G : Type) [MyGroup G] : Set G :=
  {x : G | ∃ y : G, y ^ 2 = x}

theorem squares_characterization :
  squares G = {1, (MyGroup.g : G) ^ 2, MyGroup.h, MyGroup.h ^ 2, MyGroup.h ^ 3, MyGroup.h ^ 4, MyGroup.h ^ 5, MyGroup.h ^ 6} := by
  sorry

end squares_characterization_l107_10737


namespace sequence_sum_l107_10747

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ + 13*x₇ = 3)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ + 15*x₇ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ + 17*x₇ = 85) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ + 19*x₇ = 213 := by
  sorry

end sequence_sum_l107_10747


namespace intersection_of_P_and_complement_of_M_l107_10714

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x ≥ 3}
def M : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_of_P_and_complement_of_M :
  P ∩ (Set.univ \ M) = {x : ℝ | x ≥ 4} := by sorry

end intersection_of_P_and_complement_of_M_l107_10714


namespace solution_to_system_l107_10788

theorem solution_to_system (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x + y + z)^3) → 
  ((x = 0 ∧ y = 0 ∧ z = 1/Real.sqrt 3) ∨ 
   (x = 0 ∧ y = 0 ∧ z = -1/Real.sqrt 3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = 1/3 ∧ y = -1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = 1/3 ∧ z = -1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = 1/3) ∨ 
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) :=
by sorry

end solution_to_system_l107_10788


namespace guessing_game_difference_l107_10781

theorem guessing_game_difference : (2 * 51) - (3 * 33) = 3 := by
  sorry

end guessing_game_difference_l107_10781


namespace power_of_81_three_fourths_l107_10728

theorem power_of_81_three_fourths : (81 : ℝ) ^ (3/4 : ℝ) = 27 := by sorry

end power_of_81_three_fourths_l107_10728


namespace probability_odd_divisor_15_factorial_l107_10770

theorem probability_odd_divisor_15_factorial (n : ℕ) (h : n = 15) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (λ x => x > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ x => x > 0 ∧ x % 2 ≠ 0)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by
  sorry

end probability_odd_divisor_15_factorial_l107_10770


namespace total_pushups_is_53_l107_10759

/-- The number of push-ups David did -/
def david_pushups : ℕ := 51

/-- The difference between David's and Zachary's push-ups -/
def pushup_difference : ℕ := 49

/-- Calculates the total number of push-ups done by David and Zachary -/
def total_pushups : ℕ := david_pushups + (david_pushups - pushup_difference)

theorem total_pushups_is_53 : total_pushups = 53 := by
  sorry

end total_pushups_is_53_l107_10759


namespace horner_method_f_2_l107_10773

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 - 5x^4 + 3x^3 - 2x^2 + x -/
def f (x : ℚ) : ℚ :=
  horner [1, 0, -2, 3, -5, 3] x

theorem horner_method_f_2 :
  f 2 = 34 := by sorry

end horner_method_f_2_l107_10773


namespace pure_imaginary_fraction_l107_10783

theorem pure_imaginary_fraction (a : ℝ) : 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).re = 0 ∧ 
  (((1 : ℂ) + 2 * Complex.I) / (a + Complex.I)).im ≠ 0 → 
  a = -2 := by
  sorry

end pure_imaginary_fraction_l107_10783


namespace appliance_price_difference_l107_10792

theorem appliance_price_difference : 
  let in_store_price : ℚ := 109.99
  let tv_payment : ℚ := 24.99
  let tv_shipping : ℚ := 14.98
  let tv_price : ℚ := 4 * tv_payment + tv_shipping
  (tv_price - in_store_price) * 100 = 495 := by sorry

end appliance_price_difference_l107_10792


namespace machine_selling_price_l107_10721

/-- Calculates the selling price of a machine given its costs and desired profit percentage -/
def selling_price (purchase_price repair_cost transport_cost profit_percent : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percent / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 30000 Rs -/
theorem machine_selling_price :
  selling_price 14000 5000 1000 50 = 30000 := by
  sorry

end machine_selling_price_l107_10721


namespace gills_arrival_time_l107_10742

/-- Represents the travel details of Gill's train journey --/
structure TravelDetails where
  departure_time : Nat  -- in minutes past midnight
  first_segment_distance : Nat  -- in km
  second_segment_distance : Nat  -- in km
  speed : Nat  -- in km/h
  stop_duration : Nat  -- in minutes

/-- Calculates the arrival time given the travel details --/
def calculate_arrival_time (details : TravelDetails) : Nat :=
  let first_segment_time := details.first_segment_distance * 60 / details.speed
  let second_segment_time := details.second_segment_distance * 60 / details.speed
  let total_travel_time := first_segment_time + details.stop_duration + second_segment_time
  details.departure_time + total_travel_time

/-- Gill's travel details --/
def gills_travel : TravelDetails :=
  { departure_time := 9 * 60  -- 09:00 in minutes
    first_segment_distance := 27
    second_segment_distance := 29
    speed := 96
    stop_duration := 3 }

theorem gills_arrival_time :
  calculate_arrival_time gills_travel = 9 * 60 + 38 := by
  sorry

end gills_arrival_time_l107_10742


namespace class_reading_total_l107_10797

/-- Calculates the total number of books read by a class per week given the following conditions:
  * There are 12 girls and 10 boys in the class.
  * 5/6 of the girls and 4/5 of the boys are reading.
  * Girls read at an average rate of 3 books per week.
  * Boys read at an average rate of 2 books per week.
  * 20% of reading girls read at a faster rate of 5 books per week.
  * 10% of reading boys read at a slower rate of 1 book per week.
-/
theorem class_reading_total (girls : ℕ) (boys : ℕ) 
  (girls_reading_ratio : ℚ) (boys_reading_ratio : ℚ)
  (girls_avg_rate : ℕ) (boys_avg_rate : ℕ)
  (girls_faster_ratio : ℚ) (boys_slower_ratio : ℚ)
  (girls_faster_rate : ℕ) (boys_slower_rate : ℕ) :
  girls = 12 →
  boys = 10 →
  girls_reading_ratio = 5/6 →
  boys_reading_ratio = 4/5 →
  girls_avg_rate = 3 →
  boys_avg_rate = 2 →
  girls_faster_ratio = 1/5 →
  boys_slower_ratio = 1/10 →
  girls_faster_rate = 5 →
  boys_slower_rate = 1 →
  (girls_reading_ratio * girls * girls_avg_rate +
   boys_reading_ratio * boys * boys_avg_rate +
   girls_reading_ratio * girls * girls_faster_ratio * (girls_faster_rate - girls_avg_rate) +
   boys_reading_ratio * boys * boys_slower_ratio * (boys_slower_rate - boys_avg_rate)) = 49 := by
  sorry


end class_reading_total_l107_10797
