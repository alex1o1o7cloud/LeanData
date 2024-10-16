import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l713_71330

-- Define the set A
def A : Set ℝ := {x | x > 1}

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) (h_sum : m + n = 4) :
  n^2 / (m - 1) + m^2 / (n - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l713_71330


namespace NUMINAMATH_CALUDE_emily_fishing_total_weight_l713_71300

theorem emily_fishing_total_weight :
  let trout_count : ℕ := 4
  let catfish_count : ℕ := 3
  let bluegill_count : ℕ := 5
  let trout_weight : ℚ := 2
  let catfish_weight : ℚ := 1.5
  let bluegill_weight : ℚ := 2.5
  let total_weight : ℚ := trout_count * trout_weight + catfish_count * catfish_weight + bluegill_count * bluegill_weight
  total_weight = 25 := by sorry

end NUMINAMATH_CALUDE_emily_fishing_total_weight_l713_71300


namespace NUMINAMATH_CALUDE_age_difference_l713_71315

theorem age_difference (A B : ℕ) : B = 38 → A + 10 = 2 * (B - 10) → A - B = 8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l713_71315


namespace NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l713_71344

/-- The points scored by the winning team in the fourth quarter of a basketball game. -/
def fourth_quarter_points (first_quarter_losing : ℕ) 
                          (second_quarter_increase : ℕ) 
                          (third_quarter_increase : ℕ) 
                          (total_points : ℕ) : ℕ :=
  let first_quarter_winning := 2 * first_quarter_losing
  let second_quarter_winning := first_quarter_winning + second_quarter_increase
  let third_quarter_winning := second_quarter_winning + third_quarter_increase
  total_points - third_quarter_winning

/-- Theorem stating that the winning team scored 30 points in the fourth quarter. -/
theorem winning_team_fourth_quarter_points : 
  fourth_quarter_points 10 10 20 80 = 30 := by
  sorry

end NUMINAMATH_CALUDE_winning_team_fourth_quarter_points_l713_71344


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_relation_l713_71370

theorem smallest_integer_gcd_lcm_relation (m : ℕ) (h : m > 0) :
  (Nat.gcd 60 m * 20 = Nat.lcm 60 m) →
  (∀ k : ℕ, k > 0 ∧ k < m → Nat.gcd 60 k * 20 ≠ Nat.lcm 60 k) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_relation_l713_71370


namespace NUMINAMATH_CALUDE_bear_color_theorem_l713_71372

/-- Represents the Earth's surface --/
structure EarthSurface where
  latitude : ℝ
  longitude : ℝ

/-- Represents a bear --/
inductive Bear
| Polar
| Other

/-- Represents the hunter's position and orientation --/
structure HunterState where
  position : EarthSurface
  facing : EarthSurface

/-- Function to determine if a point is at the North Pole --/
def isNorthPole (p : EarthSurface) : Prop :=
  p.latitude = 90 -- Assuming 90 degrees latitude is the North Pole

/-- Function to move a point on the Earth's surface --/
def move (start : EarthSurface) (direction : String) (distance : ℝ) : EarthSurface :=
  sorry -- Implementation details omitted

/-- Function to determine the type of bear based on location --/
def bearType (location : EarthSurface) : Bear :=
  sorry -- Implementation details omitted

/-- The main theorem --/
theorem bear_color_theorem 
  (bear_position : EarthSurface)
  (initial_hunter_position : EarthSurface)
  (h1 : initial_hunter_position = move bear_position "south" 100)
  (h2 : let east_position := move initial_hunter_position "east" 100
        east_position.latitude = initial_hunter_position.latitude)
  (h3 : let final_hunter_state := HunterState.mk (move initial_hunter_position "east" 100) bear_position
        final_hunter_state.facing = bear_position)
  : bearType bear_position = Bear.Polar :=
sorry


end NUMINAMATH_CALUDE_bear_color_theorem_l713_71372


namespace NUMINAMATH_CALUDE_function_composition_equality_l713_71367

/-- Given real numbers a, b, c, d, k where k ≠ 0, and functions f and g defined as
    f(x) = ax + b and g(x) = k(cx + d), this theorem states that f(g(x)) = g(f(x))
    if and only if b(1 - kc) = k(d(1 - a)). -/
theorem function_composition_equality
  (a b c d k : ℝ)
  (hk : k ≠ 0)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = k * (c * x + d)) :
  (∀ x, f (g x) = g (f x)) ↔ b * (1 - k * c) = k * (d * (1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l713_71367


namespace NUMINAMATH_CALUDE_staircase_problem_l713_71319

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def jumps (step_size : ℕ) (total_steps : ℕ) : ℕ := 
  (total_steps + step_size - 1) / step_size

theorem staircase_problem (n : ℕ) : 
  is_prime n → 
  jumps 3 n - jumps 6 n = 25 → 
  ∃ m : ℕ, is_prime m ∧ 
           jumps 3 m - jumps 6 m = 25 ∧ 
           n + m = 300 :=
sorry

end NUMINAMATH_CALUDE_staircase_problem_l713_71319


namespace NUMINAMATH_CALUDE_initial_apples_count_l713_71325

theorem initial_apples_count (initial_apples : ℕ) : 
  initial_apples - 2 + (8 - 2 * 2) + (15 - (2 / 3 * 15)) = 14 → 
  initial_apples = 7 := by
sorry

end NUMINAMATH_CALUDE_initial_apples_count_l713_71325


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_divisibility_l713_71360

/-- Given three numbers in an arithmetic sequence with common difference d,
    where one of the numbers is divisible by d, their product is divisible by 6d³ -/
theorem arithmetic_sequence_product_divisibility
  (a b c d : ℤ) -- a, b, c are the three numbers, d is the common difference
  (h_arithmetic : b - a = d ∧ c - b = d) -- arithmetic sequence condition
  (h_divisible : a % d = 0 ∨ b % d = 0 ∨ c % d = 0) -- one number divisible by d
  : (6 * d^3) ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_divisibility_l713_71360


namespace NUMINAMATH_CALUDE_possible_two_black_one_white_l713_71379

/-- Represents the possible marble replacement operations -/
inductive Operation
  | replaceThreeBlackWithTwoBlack
  | replaceTwoBlackOneWhiteWithTwoWhite
  | replaceOneBlackTwoWhiteWithOneBlackOneWhite
  | replaceThreeWhiteWithTwoBlack

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replaceThreeBlackWithTwoBlack =>
      UrnState.mk state.white (state.black - 1)
  | Operation.replaceTwoBlackOneWhiteWithTwoWhite =>
      UrnState.mk (state.white + 1) (state.black - 2)
  | Operation.replaceOneBlackTwoWhiteWithOneBlackOneWhite =>
      UrnState.mk (state.white - 1) state.black
  | Operation.replaceThreeWhiteWithTwoBlack =>
      UrnState.mk (state.white - 3) (state.black + 2)

/-- Theorem: It is possible to reach a state of 2 black marbles and 1 white marble -/
theorem possible_two_black_one_white :
  ∃ (operations : List Operation),
    let initial_state := UrnState.mk 150 200
    let final_state := operations.foldl applyOperation initial_state
    final_state.white = 1 ∧ final_state.black = 2 :=
  sorry


end NUMINAMATH_CALUDE_possible_two_black_one_white_l713_71379


namespace NUMINAMATH_CALUDE_min_sum_with_conditions_l713_71356

theorem min_sum_with_conditions (a b : ℕ+) 
  (h1 : ¬ 5 ∣ a.val)
  (h2 : ¬ 5 ∣ b.val)
  (h3 : (5 : ℕ)^5 ∣ a.val^5 + b.val^5) :
  ∀ (x y : ℕ+), 
    (¬ 5 ∣ x.val) → 
    (¬ 5 ∣ y.val) → 
    ((5 : ℕ)^5 ∣ x.val^5 + y.val^5) → 
    (a.val + b.val ≤ x.val + y.val) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_conditions_l713_71356


namespace NUMINAMATH_CALUDE_max_value_on_interval_a_range_for_increasing_l713_71396

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

-- Theorem for part 1
theorem max_value_on_interval (a : ℝ) (h : f_prime a 1 = 3) :
  ∃ x ∈ Set.Icc 0 2, ∀ y ∈ Set.Icc 0 2, f a x ≥ f a y ∧ f a x = 8 :=
sorry

-- Theorem for part 2
theorem a_range_for_increasing (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x ≤ y → f a x ≤ f a y) ↔ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_a_range_for_increasing_l713_71396


namespace NUMINAMATH_CALUDE_range_of_a_l713_71390

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | |x - 1| > 2}

-- Theorem statement
theorem range_of_a (a : ℝ) : (A a ∩ B = A a) ↔ (a ≤ -1 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l713_71390


namespace NUMINAMATH_CALUDE_two_children_gender_combinations_l713_71364

/-- Represents the gender of a child -/
inductive Gender
  | Male
  | Female

/-- Represents a pair of children's genders -/
def ChildPair := Gender × Gender

/-- The set of all possible gender combinations for two children -/
def allGenderCombinations : Set ChildPair :=
  {(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)}

/-- Theorem stating that the set of all possible gender combinations
    for two children is equal to the expected set -/
theorem two_children_gender_combinations :
  {pair : ChildPair | True} = allGenderCombinations := by
  sorry

end NUMINAMATH_CALUDE_two_children_gender_combinations_l713_71364


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l713_71395

theorem triangle_cut_theorem (x : ℝ) : x ≥ 6 ↔ 
  (12 - x) + (18 - x) ≤ 24 - x ∧ 
  x ≤ 12 ∧ x ≤ 18 ∧ x ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l713_71395


namespace NUMINAMATH_CALUDE_product_simplification_l713_71377

theorem product_simplification (x : ℝ) (hx : x ≠ 0) :
  (10 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/4) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l713_71377


namespace NUMINAMATH_CALUDE_n_value_equality_l713_71322

theorem n_value_equality (n : ℕ) : 3 * (Nat.choose (n - 3) (n - 7)) = 5 * (Nat.factorial (n - 4) / Nat.factorial (n - 6)) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_n_value_equality_l713_71322


namespace NUMINAMATH_CALUDE_fraction_evaluation_l713_71333

theorem fraction_evaluation (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l713_71333


namespace NUMINAMATH_CALUDE_sum_abs_coeff_2x_minus_1_pow_5_l713_71363

/-- The sum of absolute values of coefficients (excluding constant term) 
    in the expansion of (2x-1)^5 is 242 -/
theorem sum_abs_coeff_2x_minus_1_pow_5 :
  let f : ℝ → ℝ := fun x ↦ (2*x - 1)^5
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (∀ x, f x = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) ∧
    |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 242 :=
by sorry

end NUMINAMATH_CALUDE_sum_abs_coeff_2x_minus_1_pow_5_l713_71363


namespace NUMINAMATH_CALUDE_rhombus_area_l713_71380

/-- The area of a rhombus with side length 13 and one diagonal 24 is 120 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (area : ℝ) : 
  side = 13 → diagonal1 = 24 → area = (diagonal1 * (2 * Real.sqrt (side^2 - (diagonal1/2)^2))) / 2 → area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l713_71380


namespace NUMINAMATH_CALUDE_stephanies_speed_l713_71328

/-- Given a distance of 15 miles and a time of 3 hours, prove that the speed is 5 miles per hour. -/
theorem stephanies_speed (distance : ℝ) (time : ℝ) (h1 : distance = 15) (h2 : time = 3) :
  distance / time = 5 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_speed_l713_71328


namespace NUMINAMATH_CALUDE_second_arrangement_column_size_l713_71376

/-- Represents a group of people that can be arranged in columns. -/
structure PeopleGroup where
  /-- The total number of people in the group -/
  total : ℕ
  /-- The number of columns formed when 30 people stand in each column -/
  columns_with_30 : ℕ
  /-- The number of columns formed in the second arrangement -/
  columns_in_second : ℕ
  /-- Ensures that 30 people per column forms the specified number of columns -/
  h_first_arrangement : total = 30 * columns_with_30

/-- 
Given a group of people where 30 people per column forms 16 columns,
if the same group is rearranged into 12 columns,
then there will be 40 people in each column of the second arrangement.
-/
theorem second_arrangement_column_size (g : PeopleGroup) 
    (h_16_columns : g.columns_with_30 = 16)
    (h_12_columns : g.columns_in_second = 12) :
    g.total / g.columns_in_second = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_arrangement_column_size_l713_71376


namespace NUMINAMATH_CALUDE_least_clock_equivalent_after_nine_l713_71303

/-- Definition of clock equivalence on a 12-hour clock -/
def clockEquivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 12 = 0

/-- Theorem stating that 13 is the least whole number greater than 9 
    that is clock equivalent to its square on a 12-hour clock -/
theorem least_clock_equivalent_after_nine :
  ∀ n : ℕ, n > 9 ∧ n < 13 → ¬ clockEquivalent n ∧ clockEquivalent 13 := by
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_after_nine_l713_71303


namespace NUMINAMATH_CALUDE_computer_preference_ratio_l713_71309

theorem computer_preference_ratio (total : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  (total - (mac_preference + no_preference)) = mac_preference :=
by sorry

end NUMINAMATH_CALUDE_computer_preference_ratio_l713_71309


namespace NUMINAMATH_CALUDE_school_distribution_l713_71308

theorem school_distribution (a b : ℝ) : 
  a + b = 100 →
  0.3 * a + 0.4 * b = 34 →
  a = 60 :=
by sorry

end NUMINAMATH_CALUDE_school_distribution_l713_71308


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l713_71347

/-- Proves that given the conditions of the cyclist's trip, the average speed for the return trip is 9 miles per hour. -/
theorem cyclist_return_speed (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : 
  total_distance = 36 →
  speed1 = 12 →
  speed2 = 10 →
  total_time = 7.3 →
  (total_distance / speed1 + total_distance / speed2 + total_distance / 9 = total_time) := by
sorry

end NUMINAMATH_CALUDE_cyclist_return_speed_l713_71347


namespace NUMINAMATH_CALUDE_pizza_distribution_l713_71335

theorem pizza_distribution (num_students : ℕ) (pieces_per_pizza : ℕ) (total_pieces : ℕ) :
  num_students = 10 →
  pieces_per_pizza = 6 →
  total_pieces = 1200 →
  (total_pieces / pieces_per_pizza) / num_students = 20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_distribution_l713_71335


namespace NUMINAMATH_CALUDE_inequality_proof_l713_71391

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  (a - b) * c^2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l713_71391


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l713_71366

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 ∧ y = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l713_71366


namespace NUMINAMATH_CALUDE_pants_cut_amount_l713_71340

def skirt_cut : ℝ := 0.75
def difference : ℝ := 0.25

theorem pants_cut_amount : ∃ (x : ℝ), x = skirt_cut - difference := by sorry

end NUMINAMATH_CALUDE_pants_cut_amount_l713_71340


namespace NUMINAMATH_CALUDE_multiply_fractions_l713_71393

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l713_71393


namespace NUMINAMATH_CALUDE_turquoise_more_green_count_l713_71337

/-- Represents the survey results about the perception of turquoise color --/
structure TurquoiseSurvey where
  total : Nat
  more_blue : Nat
  both : Nat
  neither : Nat

/-- Calculates the number of people who believe turquoise is "more green" --/
def more_green (survey : TurquoiseSurvey) : Nat :=
  survey.total - (survey.more_blue - survey.both) - survey.neither

/-- Theorem stating that given the survey conditions, 80 people believe turquoise is "more green" --/
theorem turquoise_more_green_count :
  ∀ (survey : TurquoiseSurvey),
  survey.total = 150 →
  survey.more_blue = 90 →
  survey.both = 40 →
  survey.neither = 20 →
  more_green survey = 80 := by
  sorry


end NUMINAMATH_CALUDE_turquoise_more_green_count_l713_71337


namespace NUMINAMATH_CALUDE_brothers_age_sum_l713_71386

/-- Two brothers with an age difference of 4 years -/
structure Brothers where
  older_age : ℕ
  younger_age : ℕ
  age_difference : older_age = younger_age + 4

/-- The sum of the brothers' ages -/
def age_sum (b : Brothers) : ℕ := b.older_age + b.younger_age

/-- Theorem: The sum of the ages of two brothers who are 4 years apart,
    where the older one is 16 years old, is 28 years. -/
theorem brothers_age_sum :
  ∀ (b : Brothers), b.older_age = 16 → age_sum b = 28 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_sum_l713_71386


namespace NUMINAMATH_CALUDE_complement_of_A_in_I_l713_71378

def I : Set ℕ := {x | 0 < x ∧ x < 6}
def A : Set ℕ := {1, 2, 3}

theorem complement_of_A_in_I :
  (I \ A) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_I_l713_71378


namespace NUMINAMATH_CALUDE_comic_book_pages_l713_71338

/-- Given that Trevor drew 220 pages in total over three months,
    and the third month's issue was four pages longer than the others,
    prove that the first issue had 72 pages. -/
theorem comic_book_pages :
  ∀ (x : ℕ),
  (x + x + (x + 4) = 220) →
  (x = 72) :=
by sorry

end NUMINAMATH_CALUDE_comic_book_pages_l713_71338


namespace NUMINAMATH_CALUDE_base_85_modulo_17_l713_71310

theorem base_85_modulo_17 (b : ℕ) : 
  0 ≤ b ∧ b ≤ 16 → (352936524 : ℕ) ≡ b [MOD 17] ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_85_modulo_17_l713_71310


namespace NUMINAMATH_CALUDE_set_intersection_equals_greater_equal_one_l713_71373

-- Define the sets S and T
def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | ∃ y, y = Real.log x}

-- State the theorem
theorem set_intersection_equals_greater_equal_one :
  S ∩ T = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equals_greater_equal_one_l713_71373


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l713_71329

/-- The average speed of a car given its distances traveled in two consecutive hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : 
  d1 = 90 → d2 = 40 → (d1 + d2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l713_71329


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l713_71381

theorem power_seven_mod_nine : 7^138 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l713_71381


namespace NUMINAMATH_CALUDE_base12_remainder_theorem_l713_71327

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (a b c d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

/-- The base-12 number 2563₁₂ --/
def base12Number : ℕ := base12ToDecimal 2 5 6 3

/-- The theorem stating that the remainder of 2563₁₂ divided by 17 is 1 --/
theorem base12_remainder_theorem : base12Number % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_base12_remainder_theorem_l713_71327


namespace NUMINAMATH_CALUDE_ball_count_and_probability_l713_71343

/-- Represents the colors of the balls -/
inductive Color
  | Red
  | White
  | Blue

/-- Represents the bag of balls -/
structure Bag where
  total : Nat
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the second bag with specific balls -/
structure SpecificBag where
  red1 : Bool
  white1 : Bool
  blue2 : Bool
  blue3 : Bool

def Bag.probability (b : Bag) (c : Color) : Rat :=
  match c with
  | Color.Red => b.red / b.total
  | Color.White => b.white / b.total
  | Color.Blue => b.blue / b.total

theorem ball_count_and_probability (b : Bag) :
  b.total = 24 ∧ b.blue = 3 ∧ b.probability Color.Red = 1/6 →
  b.red = 4 ∧
  (let sb : SpecificBag := ⟨true, true, true, true⟩
   (5 : Rat) / 12 = (Nat.choose 3 1 * Nat.choose 1 1) / (Nat.choose 4 2)) := by
  sorry


end NUMINAMATH_CALUDE_ball_count_and_probability_l713_71343


namespace NUMINAMATH_CALUDE_investment_of_c_is_120000_l713_71399

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Calculates the investment of partner C given the investments and profit shares of A and B -/
def calculateInvestmentC (a : Partner) (b : Partner) (profitShareDiffAC : ℕ) : ℕ :=
  let profitShareA := a.investment * b.profitShare / b.investment
  let profitShareC := profitShareA + profitShareDiffAC
  profitShareC * b.investment / b.profitShare

/-- Theorem stating that given the problem conditions, C's investment is 120000 -/
theorem investment_of_c_is_120000 : 
  let a : Partner := ⟨8000, 0⟩
  let b : Partner := ⟨10000, 1700⟩
  let profitShareDiffAC := 680
  calculateInvestmentC a b profitShareDiffAC = 120000 := by
  sorry

#eval calculateInvestmentC ⟨8000, 0⟩ ⟨10000, 1700⟩ 680

end NUMINAMATH_CALUDE_investment_of_c_is_120000_l713_71399


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l713_71359

theorem triangle_area_ratio (K J : ℝ) (x : ℝ) (h_positive : 0 < x) (h_less_than_one : x < 1)
  (h_ratio : J / K = x) : x = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l713_71359


namespace NUMINAMATH_CALUDE_test_scores_theorem_l713_71371

def is_valid_sequence (s : List Nat) : Prop :=
  s.length > 0 ∧ 
  s.Nodup ∧ 
  s.sum = 119 ∧ 
  (s.take 3).sum = 23 ∧ 
  (s.reverse.take 3).sum = 49

theorem test_scores_theorem (s : List Nat) (h : is_valid_sequence s) : 
  s.length = 10 ∧ s.maximum? = some 18 := by
  sorry

end NUMINAMATH_CALUDE_test_scores_theorem_l713_71371


namespace NUMINAMATH_CALUDE_largest_n_inequality_l713_71394

theorem largest_n_inequality : ∃ (n : ℕ), n = 14 ∧ 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) ≥ n * (a + b + c))) ∧
  (∀ (m : ℕ), m > 14 → 
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      (a^2 / (b/29 + c/31) + b^2 / (c/29 + a/31) + c^2 / (a/29 + b/31) < m * (a + b + c))) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_inequality_l713_71394


namespace NUMINAMATH_CALUDE_austin_surfboard_length_l713_71301

/-- Austin's surfing problem -/
theorem austin_surfboard_length 
  (H : ℝ) -- Austin's height
  (S : ℝ) -- Austin's surfboard length
  (highest_wave : 4 * H + 2 = 26) -- Highest wave is 2 feet higher than 4 times Austin's height
  (shortest_wave_height : H + 4 = S + 3) -- Shortest wave is 4 feet higher than Austin's height and 3 feet higher than surfboard length
  : S = 7 := by
  sorry


end NUMINAMATH_CALUDE_austin_surfboard_length_l713_71301


namespace NUMINAMATH_CALUDE_toms_calculation_l713_71354

theorem toms_calculation (y : ℝ) (h : 4 * y + 7 = 39) : (y + 7) * 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_toms_calculation_l713_71354


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l713_71307

/-- Given a parabola y = x^2 - 3mx + m + n, prove that for the parabola to intersect
    the x-axis for all real numbers m, n must satisfy n ≤ -1/9 -/
theorem parabola_intersects_x_axis (n : ℝ) :
  (∀ m : ℝ, ∃ x : ℝ, x^2 - 3*m*x + m + n = 0) ↔ n ≤ -1/9 := by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l713_71307


namespace NUMINAMATH_CALUDE_inequality_equivalence_l713_71384

theorem inequality_equivalence (x : ℝ) : 
  (1/3 : ℝ)^(x^2 - 8) > 3^(-2*x) ↔ -2 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l713_71384


namespace NUMINAMATH_CALUDE_eel_fat_l713_71352

/-- The amount of fat in ounces for each type of fish --/
structure FishFat where
  herring : ℝ
  eel : ℝ
  pike : ℝ

/-- The number of each type of fish cooked --/
def fish_count : ℝ := 40

/-- The total amount of fat served in ounces --/
def total_fat : ℝ := 3600

/-- Theorem stating the amount of fat in an eel --/
theorem eel_fat (f : FishFat) 
  (herring_fat : f.herring = 40)
  (pike_fat : f.pike = f.eel + 10)
  (total_fat_eq : fish_count * (f.herring + f.eel + f.pike) = total_fat) :
  f.eel = 20 := by
  sorry

end NUMINAMATH_CALUDE_eel_fat_l713_71352


namespace NUMINAMATH_CALUDE_graph_translation_up_one_unit_l713_71314

/-- Represents a vertical translation of a function --/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- The original quadratic function --/
def originalFunction : ℝ → ℝ := fun x ↦ x^2

theorem graph_translation_up_one_unit :
  verticalTranslation originalFunction 1 = fun x ↦ x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_graph_translation_up_one_unit_l713_71314


namespace NUMINAMATH_CALUDE_correct_graph_representation_l713_71355

/-- Represents a segment of Mike's trip -/
inductive TripSegment
  | CityDriving
  | HighwayDriving
  | Shopping
  | Refueling

/-- Represents the slope of a graph segment -/
inductive Slope
  | Flat
  | Gradual
  | Steep

/-- Represents Mike's trip -/
structure MikeTrip where
  segments : List TripSegment
  shoppingDuration : ℝ
  refuelingDuration : ℝ

/-- Represents a graph of Mike's trip -/
structure TripGraph where
  flatSections : Nat
  slopes : List Slope

/-- The correct graph representation of Mike's trip -/
def correctGraph : TripGraph :=
  { flatSections := 2
  , slopes := [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] }

theorem correct_graph_representation (trip : MikeTrip)
  (h1 : trip.segments = [TripSegment.CityDriving, TripSegment.HighwayDriving, TripSegment.Shopping, TripSegment.Refueling, TripSegment.HighwayDriving, TripSegment.CityDriving])
  (h2 : trip.shoppingDuration = 2)
  (h3 : trip.refuelingDuration = 0.5)
  : TripGraph.flatSections correctGraph = 2 ∧ 
    TripGraph.slopes correctGraph = [Slope.Gradual, Slope.Steep, Slope.Flat, Slope.Flat, Slope.Steep, Slope.Gradual] := by
  sorry

end NUMINAMATH_CALUDE_correct_graph_representation_l713_71355


namespace NUMINAMATH_CALUDE_percentage_of_fraction_equals_value_l713_71332

theorem percentage_of_fraction_equals_value : 
  let number : ℝ := 70.58823529411765
  let fraction : ℝ := 3 / 5
  let percentage : ℝ := 85 / 100
  percentage * (fraction * number) = 36 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_fraction_equals_value_l713_71332


namespace NUMINAMATH_CALUDE_odd_function_properties_l713_71316

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then -2^(-x)
  else if x = 0 then 0
  else if 0 < x ∧ x < 1 then 2^x
  else 0

theorem odd_function_properties (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x) →
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x ≤ 2*a) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x = -2^(-x)) ∧
  (f 0 = 0) ∧
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) ∧
  (a ≥ 1) := by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l713_71316


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_solution_set_eq_interval_l713_71358

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 1|

-- Theorem for part (I)
theorem solution_set_f (x : ℝ) : 
  f x ≥ 4*x + 3 ↔ x ∈ Set.Iic (-3/7) := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ 3*a^2 - a - 1) → 
  a ∈ Set.Icc (-1) (4/3) := by sorry

-- Define the set of solutions for part (I)
def solution_set : Set ℝ := {x : ℝ | f x ≥ 4*x + 3}

-- Theorem stating that the solution set is equal to (-∞, -3/7]
theorem solution_set_eq_interval : 
  solution_set = Set.Iic (-3/7) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_solution_set_eq_interval_l713_71358


namespace NUMINAMATH_CALUDE_length_AC_l713_71383

-- Define the right triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- Condition 1 and 2: ABC is a right triangle with angle C = 90°
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0 ∧
  -- Condition 3: AB = 9
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9 ∧
  -- Condition 4: cos B = 2/3
  (C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 2/3

-- Theorem statement
theorem length_AC (A B C : ℝ × ℝ) (h : triangle_ABC A B C) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_length_AC_l713_71383


namespace NUMINAMATH_CALUDE_perfect_square_property_l713_71306

theorem perfect_square_property : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 8 * n + 1 = k * k) ∧ 
  (∃ (m : ℕ), n = 2 * m) ∧
  (∃ (p : ℕ), n = p * p) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l713_71306


namespace NUMINAMATH_CALUDE_complex_modulus_example_l713_71368

theorem complex_modulus_example : Complex.abs (2 - (5/6) * Complex.I) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l713_71368


namespace NUMINAMATH_CALUDE_recycling_program_earnings_l713_71311

/-- Calculates the total money earned by Katrina and her friends in the recycling program -/
def total_money_earned (initial_signup : ℕ) (referral_bonus : ℕ) (friends_day1 : ℕ) (friends_week : ℕ) : ℕ :=
  let katrina_earnings := initial_signup + referral_bonus * (friends_day1 + friends_week)
  let friends_earnings := referral_bonus * (friends_day1 + friends_week)
  katrina_earnings + friends_earnings

/-- Theorem stating that the total money earned by Katrina and her friends is $125.00 -/
theorem recycling_program_earnings : 
  total_money_earned 5 5 5 7 = 125 := by
  sorry

#eval total_money_earned 5 5 5 7

end NUMINAMATH_CALUDE_recycling_program_earnings_l713_71311


namespace NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l713_71313

theorem prime_square_remainders_mod_180 :
  ∃! (s : Finset Nat), 
    (∀ r ∈ s, r < 180) ∧ 
    (∀ p : Nat, Prime p → p > 5 → (p^2 % 180) ∈ s) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_remainders_mod_180_l713_71313


namespace NUMINAMATH_CALUDE_intersection_range_l713_71362

/-- The curve y = 1 + √(4 - x²) intersects with the line y = k(x + 2) + 5 at two points
    if and only if k is in the range [-1, -3/4) --/
theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (1 + Real.sqrt (4 - x₁^2) = k * (x₁ + 2) + 5) ∧
    (1 + Real.sqrt (4 - x₂^2) = k * (x₂ + 2) + 5)) ↔ 
  (k ≥ -1 ∧ k < -3/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l713_71362


namespace NUMINAMATH_CALUDE_math_olympiad_properties_l713_71326

/-- Represents the math Olympiad team composition and assessment rules -/
structure MathOlympiadTeam where
  total_students : Nat
  grade_11_students : Nat
  grade_12_students : Nat
  grade_13_students : Nat
  selected_students : Nat
  prob_correct_easy : Rat
  prob_correct_hard : Rat
  points_easy : Nat
  points_hard : Nat
  excellent_threshold : Nat

/-- Calculates the probability of selecting exactly 2 students from Grade 11 -/
def prob_two_from_grade_11 (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the mathematical expectation of Zhang's score -/
def expected_score (team : MathOlympiadTeam) : Rat :=
  sorry

/-- Calculates the probability of Zhang being an excellent student -/
def prob_excellent_student (team : MathOlympiadTeam) : Rat :=
  sorry

/-- The main theorem proving the three required properties -/
theorem math_olympiad_properties (team : MathOlympiadTeam)
  (h1 : team.total_students = 20)
  (h2 : team.grade_11_students = 8)
  (h3 : team.grade_12_students = 7)
  (h4 : team.grade_13_students = 5)
  (h5 : team.selected_students = 3)
  (h6 : team.prob_correct_easy = 2/3)
  (h7 : team.prob_correct_hard = 1/2)
  (h8 : team.points_easy = 1)
  (h9 : team.points_hard = 2)
  (h10 : team.excellent_threshold = 5) :
  prob_two_from_grade_11 team = 28/95 ∧
  expected_score team = 10/3 ∧
  prob_excellent_student team = 2/9 :=
by
  sorry

end NUMINAMATH_CALUDE_math_olympiad_properties_l713_71326


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l713_71305

/-- An arithmetic sequence {a_n} where a_1 = 1/3, a_2 + a_5 = 4, and a_n = 33 has n = 50 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (n : ℕ) :
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 1 / 3 →
  a 2 + a 5 = 4 →
  a n = 33 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l713_71305


namespace NUMINAMATH_CALUDE_amy_chocolate_bars_l713_71361

/-- The number of chocolate bars Amy has -/
def chocolate_bars : ℕ := sorry

/-- The number of M&Ms Amy has -/
def m_and_ms : ℕ := 7 * chocolate_bars

/-- The number of marshmallows Amy has -/
def marshmallows : ℕ := 6 * m_and_ms

/-- The total number of candies Amy has -/
def total_candies : ℕ := chocolate_bars + m_and_ms + marshmallows

/-- The number of baskets Amy fills -/
def num_baskets : ℕ := 25

/-- The number of candies in each basket -/
def candies_per_basket : ℕ := 10

theorem amy_chocolate_bars : 
  chocolate_bars = 5 ∧ 
  total_candies = num_baskets * candies_per_basket := by
  sorry

end NUMINAMATH_CALUDE_amy_chocolate_bars_l713_71361


namespace NUMINAMATH_CALUDE_farm_cows_l713_71351

/-- Represents the number of bags of husk eaten by some cows in 45 days -/
def total_bags : ℕ := 45

/-- Represents the number of bags of husk eaten by one cow in 45 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 45 -/
theorem farm_cows : num_cows = 45 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_l713_71351


namespace NUMINAMATH_CALUDE_eight_power_32_sum_equals_2_power_99_l713_71341

theorem eight_power_32_sum_equals_2_power_99 :
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + 
  (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 + (8:ℕ)^32 = (2:ℕ)^99 :=
by sorry

end NUMINAMATH_CALUDE_eight_power_32_sum_equals_2_power_99_l713_71341


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l713_71342

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l713_71342


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_l713_71397

theorem sqrt_sum_equation (a b : ℚ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_l713_71397


namespace NUMINAMATH_CALUDE_multiply_by_48_equals_173_times_240_l713_71385

theorem multiply_by_48_equals_173_times_240 : 48 * 865 = 173 * 240 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_48_equals_173_times_240_l713_71385


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l713_71302

-- Define a random variable following normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def probability (ξ : normal_distribution 1 σ) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_distribution_symmetry 
  (σ : ℝ) 
  (ξ : normal_distribution 1 σ) 
  (h : probability ξ 0 1 = 0.4) : 
  probability ξ 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l713_71302


namespace NUMINAMATH_CALUDE_second_round_score_l713_71339

/-- Represents the number of darts thrown in each round -/
def darts_per_round : ℕ := 8

/-- Represents the minimum points per dart -/
def min_points_per_dart : ℕ := 3

/-- Represents the maximum points per dart -/
def max_points_per_dart : ℕ := 9

/-- Represents the points scored in the first round -/
def first_round_points : ℕ := 24

/-- Represents the ratio of points scored in the second round compared to the first round -/
def second_round_ratio : ℚ := 2

/-- Represents the ratio of points scored in the third round compared to the second round -/
def third_round_ratio : ℚ := (3/2 : ℚ)

/-- Theorem stating that Misha scored 48 points in the second round -/
theorem second_round_score : 
  first_round_points * second_round_ratio = 48 := by sorry

end NUMINAMATH_CALUDE_second_round_score_l713_71339


namespace NUMINAMATH_CALUDE_existence_of_four_numbers_l713_71375

theorem existence_of_four_numbers : ∃ (a b c d : ℕ+), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (3 ∣ a) ∧ (3 ∣ b) ∧ (3 ∣ c) ∧ (3 ∣ d) ∧
  (d ∣ (a + b + c)) ∧ (c ∣ (a + b + d)) ∧ 
  (b ∣ (a + c + d)) ∧ (a ∣ (b + c + d)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_four_numbers_l713_71375


namespace NUMINAMATH_CALUDE_inequality_solution_set_l713_71388

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 + k / (x - 1) ≤ 0 ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1)) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l713_71388


namespace NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l713_71321

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 6) :
  Real.cos (π / 6 + α) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_plus_alpha_l713_71321


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l713_71323

/-- The equation of a circle passing through (0, 0), (-2, 3), and (-4, 1) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + (19/5)*x - (9/5)*y = 0

/-- Theorem stating that the circle passes through the required points -/
theorem circle_passes_through_points :
  circle_equation 0 0 ∧ circle_equation (-2) 3 ∧ circle_equation (-4) 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l713_71323


namespace NUMINAMATH_CALUDE_polynomial_identity_l713_71324

/-- 
Given a, b, and c, prove that 
a(b - c)³ + b(c - a)³ + c(a - b)³ + (a - b)²(b - c)²(c - a)² = (a - b)(b - c)(c - a)(a + b + c + abc)
-/
theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 + (a - b)^2 * (b - c)^2 * (c - a)^2 = 
  (a - b) * (b - c) * (c - a) * (a + b + c + a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l713_71324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l713_71382

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a1 : a 1 = 5)
  (h_a5 : a 5 = 1) :
  a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sixth_term_l713_71382


namespace NUMINAMATH_CALUDE_total_cost_is_51_l713_71317

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℕ := 5

/-- The cost of a single hat in dollars -/
def hat_cost : ℕ := 4

/-- The cost of a single pair of jeans in dollars -/
def jeans_cost : ℕ := 10

/-- The number of shirts to be purchased -/
def num_shirts : ℕ := 3

/-- The number of hats to be purchased -/
def num_hats : ℕ := 4

/-- The number of pairs of jeans to be purchased -/
def num_jeans : ℕ := 2

/-- Theorem stating that the total cost of the purchase is $51 -/
theorem total_cost_is_51 : 
  num_shirts * shirt_cost + num_hats * hat_cost + num_jeans * jeans_cost = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_51_l713_71317


namespace NUMINAMATH_CALUDE_max_n_for_factorable_quadratic_l713_71357

/-- 
Given a quadratic expression 6x^2 + nx + 108 that can be factored as the product 
of two linear factors with integer coefficients, the maximum possible value of n is 649.
-/
theorem max_n_for_factorable_quadratic : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, 6 * A * B = 108 ∧ 6 * B + A = n) → 
  n ≤ 649 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorable_quadratic_l713_71357


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l713_71348

theorem cubic_root_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_inequality_l713_71348


namespace NUMINAMATH_CALUDE_corridor_length_is_95_meters_l713_71350

/-- Represents the scale of a blueprint in meters per centimeter. -/
def blueprint_scale : ℝ := 10

/-- Represents the length of the corridor in the blueprint in centimeters. -/
def blueprint_corridor_length : ℝ := 9.5

/-- Calculates the real-life length of the corridor in meters. -/
def real_life_corridor_length : ℝ := blueprint_scale * blueprint_corridor_length

/-- Theorem stating that the real-life length of the corridor is 95 meters. -/
theorem corridor_length_is_95_meters : real_life_corridor_length = 95 := by
  sorry

end NUMINAMATH_CALUDE_corridor_length_is_95_meters_l713_71350


namespace NUMINAMATH_CALUDE_marks_tomatoes_l713_71365

/-- Given that Mark bought tomatoes at $5 per pound and 5 pounds of apples at $6 per pound,
    spending a total of $40, prove that he bought 2 pounds of tomatoes. -/
theorem marks_tomatoes :
  ∀ (tomato_price apple_price : ℝ) (apple_pounds : ℝ) (total_spent : ℝ),
    tomato_price = 5 →
    apple_price = 6 →
    apple_pounds = 5 →
    total_spent = 40 →
    ∃ (tomato_pounds : ℝ),
      tomato_pounds * tomato_price + apple_pounds * apple_price = total_spent ∧
      tomato_pounds = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_tomatoes_l713_71365


namespace NUMINAMATH_CALUDE_computer_price_l713_71369

theorem computer_price (P : ℝ) 
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 := by
sorry

end NUMINAMATH_CALUDE_computer_price_l713_71369


namespace NUMINAMATH_CALUDE_arrangement_count_is_7200_l713_71398

/-- The number of consonants in the word "ИНТЕГРАЛ" -/
def num_consonants : ℕ := 5

/-- The number of vowels in the word "ИНТЕГРАЛ" -/
def num_vowels : ℕ := 3

/-- The total number of letters in the word "ИНТЕГРАЛ" -/
def total_letters : ℕ := num_consonants + num_vowels

/-- The number of positions that must be occupied by consonants -/
def required_consonant_positions : ℕ := 3

/-- The number of remaining positions after placing consonants in required positions -/
def remaining_positions : ℕ := total_letters - required_consonant_positions

/-- The number of ways to arrange the letters in "ИНТЕГРАЛ" with consonants in specific positions -/
def arrangement_count : ℕ := 
  (num_consonants.factorial / (num_consonants - required_consonant_positions).factorial) * 
  remaining_positions.factorial

theorem arrangement_count_is_7200 : arrangement_count = 7200 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_7200_l713_71398


namespace NUMINAMATH_CALUDE_minimize_m_l713_71349

theorem minimize_m (x y : ℝ) :
  let m := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9
  ∀ a b : ℝ, m ≤ (4 * a^2 - 12 * a * b + 10 * b^2 + 4 * b + 9) ∧
  m = 5 ∧ x = -3 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_m_l713_71349


namespace NUMINAMATH_CALUDE_negation_of_existence_l713_71345

theorem negation_of_existence (l : ℝ) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_existence_l713_71345


namespace NUMINAMATH_CALUDE_indeterminate_divisor_l713_71312

theorem indeterminate_divisor (x y : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + 7 = y * m + 12) →
  ¬ (∃! y : ℤ, ∃ m : ℤ, x + 7 = y * m + 12) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_divisor_l713_71312


namespace NUMINAMATH_CALUDE_probability_at_least_five_consecutive_heads_l713_71346

def num_flips : ℕ := 8
def min_consecutive_heads : ℕ := 5

def favorable_outcomes : ℕ := 10
def total_outcomes : ℕ := 2^num_flips

theorem probability_at_least_five_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 128 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_five_consecutive_heads_l713_71346


namespace NUMINAMATH_CALUDE_fire_water_requirement_l713_71334

theorem fire_water_requirement 
  (flow_rate : ℝ) 
  (num_firefighters : ℕ) 
  (time_taken : ℝ) 
  (h1 : flow_rate = 20) 
  (h2 : num_firefighters = 5) 
  (h3 : time_taken = 40) : 
  flow_rate * num_firefighters * time_taken = 4000 :=
by
  sorry

end NUMINAMATH_CALUDE_fire_water_requirement_l713_71334


namespace NUMINAMATH_CALUDE_inequality_system_solution_l713_71331

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 > 3 * (x - 1)) ∧ ((1 / 2) * x < 3) → -2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l713_71331


namespace NUMINAMATH_CALUDE_barbie_gave_four_pairs_l713_71318

/-- The number of pairs of earrings Barbie bought -/
def total_earrings : ℕ := 12

/-- The number of pairs of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := 4

/-- Alissa's total collection after receiving earrings from Barbie -/
def alissa_total (x : ℕ) : ℕ := 3 * x

theorem barbie_gave_four_pairs :
  earrings_given = 4 ∧
  alissa_total earrings_given + earrings_given = total_earrings :=
by sorry

end NUMINAMATH_CALUDE_barbie_gave_four_pairs_l713_71318


namespace NUMINAMATH_CALUDE_g_derivative_at_midpoint_sign_l713_71387

/-- The function g(x) defined as x + a * ln(x) - k * x^2 --/
noncomputable def g (a k x : ℝ) : ℝ := x + a * Real.log x - k * x^2

/-- The derivative of g(x) --/
noncomputable def g' (a k x : ℝ) : ℝ := 1 + a / x - 2 * k * x

theorem g_derivative_at_midpoint_sign (a k x₁ x₂ : ℝ) 
  (hk : k > 0) 
  (hx : x₁ ≠ x₂) 
  (hz₁ : g a k x₁ = 0) 
  (hz₂ : g a k x₂ = 0) :
  (a > 0 → g' a k ((x₁ + x₂) / 2) < 0) ∧
  (a < 0 → g' a k ((x₁ + x₂) / 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_g_derivative_at_midpoint_sign_l713_71387


namespace NUMINAMATH_CALUDE_ivan_remaining_money_l713_71389

def initial_amount : ℚ := 10
def cupcake_fraction : ℚ := 1/5
def milkshake_cost : ℚ := 5

theorem ivan_remaining_money :
  let cupcake_cost : ℚ := initial_amount * cupcake_fraction
  let remaining_after_cupcakes : ℚ := initial_amount - cupcake_cost
  let final_remaining : ℚ := remaining_after_cupcakes - milkshake_cost
  final_remaining = 3 := by sorry

end NUMINAMATH_CALUDE_ivan_remaining_money_l713_71389


namespace NUMINAMATH_CALUDE_marble_selection_problem_l713_71374

theorem marble_selection_problem (n : ℕ) (k : ℕ) (s : ℕ) (t : ℕ) :
  n = 15 ∧ k = 5 ∧ s = 4 ∧ t = 2 →
  (Nat.choose s t) * (Nat.choose (n - s) (k - t)) = 990 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l713_71374


namespace NUMINAMATH_CALUDE_two_small_triangles_exist_l713_71320

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- The unit triangle -/
def unitTriangle : Triangle :=
  sorry

/-- Theorem: Given 5 points in a unit triangle, there exist at least two distinct
    triangles formed by these points, each with an area not exceeding 1/4 -/
theorem two_small_triangles_exist (points : Finset Point)
    (h1 : points.card = 5)
    (h2 : ∀ p ∈ points, isInside p unitTriangle) :
    ∃ t1 t2 : Triangle,
      t1.a ∈ points ∧ t1.b ∈ points ∧ t1.c ∈ points ∧
      t2.a ∈ points ∧ t2.b ∈ points ∧ t2.c ∈ points ∧
      t1 ≠ t2 ∧
      area t1 ≤ 1/4 ∧ area t2 ≤ 1/4 :=
  sorry

end NUMINAMATH_CALUDE_two_small_triangles_exist_l713_71320


namespace NUMINAMATH_CALUDE_sand_bucket_calculation_l713_71304

/-- Given a total amount of sand and the amount per bucket, calculates the number of buckets. -/
def buckets_of_sand (total_sand : ℕ) (sand_per_bucket : ℕ) : ℕ :=
  total_sand / sand_per_bucket

/-- Theorem stating that given 34 pounds of sand total and 2 pounds per bucket, the number of buckets is 17. -/
theorem sand_bucket_calculation :
  buckets_of_sand 34 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sand_bucket_calculation_l713_71304


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l713_71336

/-- Represents a square tile pattern -/
structure TilePattern :=
  (side : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Extends a tile pattern by adding a border of white tiles -/
def extend_pattern (p : TilePattern) : TilePattern :=
  { side := p.side + 2,
    black_tiles := p.black_tiles,
    white_tiles := p.white_tiles + (p.side + 2)^2 - p.side^2 }

/-- The ratio of black tiles to white tiles in a pattern -/
def tile_ratio (p : TilePattern) : ℚ :=
  p.black_tiles / p.white_tiles

theorem extended_pattern_ratio :
  let original := TilePattern.mk 5 13 12
  let extended := extend_pattern original
  tile_ratio extended = 13 / 36 := by sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l713_71336


namespace NUMINAMATH_CALUDE_medical_team_formation_plans_l713_71392

theorem medical_team_formation_plans (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4) :
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2) +
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_formation_plans_l713_71392


namespace NUMINAMATH_CALUDE_jennifer_blue_sweets_l713_71353

theorem jennifer_blue_sweets (green : ℕ) (yellow : ℕ) (people : ℕ) (sweets_per_person : ℕ) :
  green = 212 →
  yellow = 502 →
  people = 4 →
  sweets_per_person = 256 →
  people * sweets_per_person - (green + yellow) = 310 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_blue_sweets_l713_71353
