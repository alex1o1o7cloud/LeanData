import Mathlib

namespace NUMINAMATH_CALUDE_projectile_max_height_l4106_410626

/-- The height of the projectile as a function of time -/
def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

/-- The time at which the projectile reaches its maximum height -/
def t_max : ℝ := 1

theorem projectile_max_height :
  ∃ (max_height : ℝ), max_height = h t_max ∧ 
  ∀ (t : ℝ), h t ≤ max_height ∧
  max_height = 45 := by
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l4106_410626


namespace NUMINAMATH_CALUDE_tops_count_l4106_410695

-- Define the number of marbles for each person
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Define the number of tops for each person
def laurie_tops : ℕ := laurie_marbles * 2
def kurt_tops : ℕ := kurt_marbles - 3
def dennis_tops : ℕ := dennis_marbles + 8
def jessica_tops : ℕ := jessica_marbles - 10

theorem tops_count :
  laurie_tops = 74 ∧
  kurt_tops = 22 ∧
  dennis_tops = 78 ∧
  jessica_tops = 52 := by sorry

end NUMINAMATH_CALUDE_tops_count_l4106_410695


namespace NUMINAMATH_CALUDE_xy_value_l4106_410685

theorem xy_value (x y : ℝ) : 
  |x - y + 1| + (y + 5)^2010 = 0 → x * y = 30 := by sorry

end NUMINAMATH_CALUDE_xy_value_l4106_410685


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l4106_410664

theorem simultaneous_equations_solution :
  ∀ a b : ℚ,
  (a + b) * (a^2 - b^2) = 4 ∧
  (a - b) * (a^2 + b^2) = 5/2 →
  ((a = 3/2 ∧ b = 1/2) ∨ (a = -1/2 ∧ b = -3/2)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l4106_410664


namespace NUMINAMATH_CALUDE_car_distance_theorem_l4106_410665

/-- Given a car traveling at a specific speed for a certain time, 
    calculate the distance covered. -/
theorem car_distance_theorem (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 160 → time = 5 → distance = speed * time → distance = 800 :=
by sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l4106_410665


namespace NUMINAMATH_CALUDE_quadrilateral_property_l4106_410661

-- Define a quadrilateral as a tuple of four natural numbers
def Quadrilateral := (ℕ × ℕ × ℕ × ℕ)

-- Define a property that each side divides the sum of the other three
def DivisibilityProperty (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  (a ∣ b + c + d) ∧ (b ∣ a + c + d) ∧ (c ∣ a + b + d) ∧ (d ∣ a + b + c)

-- Define a property that at least two sides are equal
def TwoSidesEqual (q : Quadrilateral) : Prop :=
  let (a, b, c, d) := q
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d

-- The main theorem
theorem quadrilateral_property (q : Quadrilateral) :
  DivisibilityProperty q → TwoSidesEqual q :=
by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_property_l4106_410661


namespace NUMINAMATH_CALUDE_train_stop_time_l4106_410677

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 45)
  (h2 : speed_with_stops = 30) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 20 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l4106_410677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4106_410625

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  positive : ∀ n, a n > 0
  initial : a 1 = 1
  geometric : (a 3) * (a 11) = (a 4 + 5/2)^2
  arithmetic : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) (m n : ℕ) 
  (h : m - n = 8) : seq.a m - seq.a n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4106_410625


namespace NUMINAMATH_CALUDE_sum_of_B_elements_l4106_410605

/-- A finite set with two elements -/
inductive TwoElementSet
  | e1
  | e2

/-- The mapping f from A to B -/
def f (x : TwoElementSet) : ℝ :=
  match x with
  | TwoElementSet.e1 => 1^2
  | TwoElementSet.e2 => 3^2

/-- The set B as a function from TwoElementSet to ℝ -/
def B : TwoElementSet → ℝ := f

theorem sum_of_B_elements : (B TwoElementSet.e1) + (B TwoElementSet.e2) = 10 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_B_elements_l4106_410605


namespace NUMINAMATH_CALUDE_pencil_price_l4106_410609

theorem pencil_price (x y : ℚ) 
  (eq1 : 3 * x + 5 * y = 345)
  (eq2 : 4 * x + 2 * y = 280) :
  y = 540 / 14 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l4106_410609


namespace NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_events_A_C_not_independent_l4106_410628

/-- Represents the possible outcomes when drawing a ball from Box A -/
inductive BoxA
| one
| two
| three
| four

/-- Represents the possible outcomes when drawing a ball from Box B -/
inductive BoxB
| five
| six
| seven
| eight

/-- The type of all possible outcomes when drawing one ball from each box -/
def Outcome := BoxA × BoxB

/-- The sum of the numbers on the balls drawn -/
def sum (o : Outcome) : ℕ :=
  match o with
  | (BoxA.one, b) => 1 + boxBToNat b
  | (BoxA.two, b) => 2 + boxBToNat b
  | (BoxA.three, b) => 3 + boxBToNat b
  | (BoxA.four, b) => 4 + boxBToNat b
where
  boxBToNat : BoxB → ℕ
  | BoxB.five => 5
  | BoxB.six => 6
  | BoxB.seven => 7
  | BoxB.eight => 8

/-- Event A: the sum of the numbers drawn is even -/
def eventA (o : Outcome) : Prop := Even (sum o)

/-- Event B: the sum of the numbers drawn is 9 -/
def eventB (o : Outcome) : Prop := sum o = 9

/-- Event C: the sum of the numbers drawn is greater than 9 -/
def eventC (o : Outcome) : Prop := sum o > 9

/-- The probability measure on the sample space -/
def P : Set Outcome → ℝ := sorry

theorem events_A_B_mutually_exclusive :
  ∀ o : Outcome, ¬(eventA o ∧ eventB o) := by sorry

theorem events_A_C_not_independent :
  P {o | eventA o ∧ eventC o} ≠ P {o | eventA o} * P {o | eventC o} := by sorry

end NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_events_A_C_not_independent_l4106_410628


namespace NUMINAMATH_CALUDE_juan_running_distance_l4106_410632

/-- Given that Juan ran for 80.0 hours at a speed of 10.0 miles per hour, 
    prove that the distance he ran is 800.0 miles. -/
theorem juan_running_distance (time : ℝ) (speed : ℝ) (distance : ℝ) : 
  time = 80.0 → speed = 10.0 → distance = time * speed → distance = 800.0 := by
  sorry

end NUMINAMATH_CALUDE_juan_running_distance_l4106_410632


namespace NUMINAMATH_CALUDE_print_325_pages_time_l4106_410683

/-- Calculates the time required to print a given number of pages with a printer that has a specific print rate and delay after every 100 pages. -/
def print_time (total_pages : ℕ) (pages_per_minute : ℕ) (delay_minutes : ℕ) : ℕ :=
  let print_time := total_pages / pages_per_minute
  let num_delays := total_pages / 100
  print_time + num_delays * delay_minutes

/-- Theorem stating that printing 325 pages takes 16 minutes with the given conditions. -/
theorem print_325_pages_time :
  print_time 325 25 1 = 16 :=
by sorry

end NUMINAMATH_CALUDE_print_325_pages_time_l4106_410683


namespace NUMINAMATH_CALUDE_multiples_equality_l4106_410634

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ+) : ℚ := 2 * n

/-- Theorem stating that if a^2 - b^2 = 0, then n = 10 -/
theorem multiples_equality (n : ℕ+) : a^2 - (b n)^2 = 0 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiples_equality_l4106_410634


namespace NUMINAMATH_CALUDE_repeating_decimal_6_is_two_thirds_l4106_410615

def repeating_decimal_6 : ℚ := 0.6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666

theorem repeating_decimal_6_is_two_thirds : repeating_decimal_6 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_6_is_two_thirds_l4106_410615


namespace NUMINAMATH_CALUDE_triangle_property_l4106_410687

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (a - b) = (Real.sin C - Real.sin B) * c →
  a = 4 →
  A = π / 3 ∧ (∀ b' c' : ℝ, b' > 0 → c' > 0 → 
    (Real.sin A + Real.sin B) * (a - b') = (Real.sin C - Real.sin B) * c' →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l4106_410687


namespace NUMINAMATH_CALUDE_sector_central_angle_l4106_410636

theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 10) (h2 : area = 100) :
  (2 * area) / (r^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4106_410636


namespace NUMINAMATH_CALUDE_six_lines_regions_l4106_410622

/-- The number of regions created by n lines in a plane where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := sorry

theorem six_lines_regions :
  general_position 6 → num_regions 6 = 22 := by sorry

end NUMINAMATH_CALUDE_six_lines_regions_l4106_410622


namespace NUMINAMATH_CALUDE_system_solution_l4106_410624

theorem system_solution :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4106_410624


namespace NUMINAMATH_CALUDE_not_always_true_from_false_l4106_410648

-- Define a proposition
variable (P Q R : Prop)

-- Define a logical argument
def logical_argument (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define soundness of logical derivation
def sound_derivation (arg : Prop → Prop) : Prop :=
  ∀ (X Y : Prop), (X → Y) → (arg X → arg Y)

-- Theorem statement
theorem not_always_true_from_false :
  ∃ (premises conclusion : Prop) (arg : Prop → Prop),
    (¬premises) ∧ 
    (sound_derivation arg) ∧
    (logical_argument premises conclusion) ∧
    (¬conclusion) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_from_false_l4106_410648


namespace NUMINAMATH_CALUDE_solution_is_negative_eight_l4106_410614

/-- An arithmetic sequence is defined by its first three terms -/
structure ArithmeticSequence :=
  (a₁ : ℚ)
  (a₂ : ℚ)
  (a₃ : ℚ)

/-- The common difference of an arithmetic sequence -/
def ArithmeticSequence.commonDifference (seq : ArithmeticSequence) : ℚ :=
  seq.a₂ - seq.a₁

/-- A sequence is arithmetic if the difference between the second and third terms
    is equal to the difference between the first and second terms -/
def ArithmeticSequence.isArithmetic (seq : ArithmeticSequence) : Prop :=
  seq.a₃ - seq.a₂ = seq.a₂ - seq.a₁

/-- The given sequence -/
def givenSequence (x : ℚ) : ArithmeticSequence :=
  { a₁ := 2
    a₂ := (2*x + 1) / 3
    a₃ := 2*x + 4 }

theorem solution_is_negative_eight :
  ∃ x : ℚ, (givenSequence x).isArithmetic ∧ x = -8 := by sorry

end NUMINAMATH_CALUDE_solution_is_negative_eight_l4106_410614


namespace NUMINAMATH_CALUDE_lcm_of_16_24_45_l4106_410620

theorem lcm_of_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_24_45_l4106_410620


namespace NUMINAMATH_CALUDE_amy_candy_distribution_l4106_410638

/-- Proves that Amy puts 10 candies in each basket given the conditions of the problem -/
theorem amy_candy_distribution (chocolate_bars : ℕ) (num_baskets : ℕ) : 
  chocolate_bars = 5 →
  num_baskets = 25 →
  (chocolate_bars + 7 * chocolate_bars + 6 * (7 * chocolate_bars)) / num_baskets = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_candy_distribution_l4106_410638


namespace NUMINAMATH_CALUDE_order_of_magnitude_l4106_410629

noncomputable def a : ℝ := Real.exp (Real.exp 1)
noncomputable def b : ℝ := Real.pi ^ Real.pi
noncomputable def c : ℝ := Real.exp Real.pi
noncomputable def d : ℝ := Real.pi ^ (Real.exp 1)

theorem order_of_magnitude : a < d ∧ d < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l4106_410629


namespace NUMINAMATH_CALUDE_die_roll_prime_probability_l4106_410671

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (x y : ℕ) : Prop := is_prime (x + y)

def count_prime_sums : ℕ := 22

def total_outcomes : ℕ := 48

theorem die_roll_prime_probability :
  (count_prime_sums : ℚ) / total_outcomes = 11 / 24 := by sorry

end NUMINAMATH_CALUDE_die_roll_prime_probability_l4106_410671


namespace NUMINAMATH_CALUDE_slices_left_over_l4106_410621

/-- Represents the number of pizzas shared --/
def total_pizzas : ℕ := 2

/-- Represents the number of slices in each pizza --/
def slices_per_pizza : ℕ := 12

/-- Represents the fraction of a pizza Bob ate --/
def bob_fraction : ℚ := 1/2

/-- Represents the fraction of a pizza Tom ate --/
def tom_fraction : ℚ := 1/3

/-- Represents the fraction of a pizza Sally ate --/
def sally_fraction : ℚ := 1/6

/-- Represents the fraction of a pizza Jerry ate --/
def jerry_fraction : ℚ := 1/4

/-- Theorem stating the number of slices left over --/
theorem slices_left_over :
  (total_pizzas * slices_per_pizza : ℚ) -
  (bob_fraction + tom_fraction + sally_fraction + jerry_fraction) * slices_per_pizza = 9 := by
  sorry

end NUMINAMATH_CALUDE_slices_left_over_l4106_410621


namespace NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l4106_410643

theorem exterior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_exterior_angles : ℝ),
  n = 8 → 
  sum_exterior_angles = 360 →
  (sum_exterior_angles / n : ℝ) = 45 := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_octagon_l4106_410643


namespace NUMINAMATH_CALUDE_unique_line_through_point_l4106_410644

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_line_through_point :
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    is_prime b ∧ 
    (6 : ℚ) / a + (5 : ℚ) / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_through_point_l4106_410644


namespace NUMINAMATH_CALUDE_line_through_points_l4106_410652

/-- Given three points on a line, find the y-coordinate of a fourth point on the same line -/
theorem line_through_points (x1 y1 x2 y2 x3 y3 x4 : ℝ) (h1 : y2 - y1 = (x2 - x1) * ((y3 - y1) / (x3 - x1))) 
  (h2 : y3 - y2 = (x3 - x2) * ((y3 - y1) / (x3 - x1))) : 
  let t := y1 + (x4 - x1) * ((y3 - y1) / (x3 - x1))
  (x1 = 2 ∧ y1 = 6 ∧ x2 = 5 ∧ y2 = 12 ∧ x3 = 8 ∧ y3 = 18 ∧ x4 = 20) → t = 42 := by
  sorry


end NUMINAMATH_CALUDE_line_through_points_l4106_410652


namespace NUMINAMATH_CALUDE_butter_amount_is_480_l4106_410682

/-- Represents the ingredients in a recipe --/
structure Ingredients where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Represents the ratios of ingredients in a recipe --/
structure Ratio where
  flour : ℝ
  butter : ℝ
  sugar : ℝ

/-- Calculates the total ingredients after mixing two recipes and adding extra flour --/
def mixRecipes (cake : Ingredients) (cream : Ingredients) (extraFlour : ℝ) : Ingredients :=
  { flour := cake.flour + extraFlour
  , butter := cake.butter + cream.butter
  , sugar := cake.sugar + cream.sugar }

/-- Checks if the given ingredients satisfy the required ratio --/
def satisfiesRatio (ingredients : Ingredients) (ratio : Ratio) : Prop :=
  ingredients.flour / ratio.flour = ingredients.butter / ratio.butter ∧
  ingredients.flour / ratio.flour = ingredients.sugar / ratio.sugar

/-- Main theorem: The amount of butter used is 480 grams --/
theorem butter_amount_is_480 
  (cake_ratio : Ratio)
  (cream_ratio : Ratio)
  (cookie_ratio : Ratio)
  (cake : Ingredients)
  (cream : Ingredients)
  (h1 : satisfiesRatio cake cake_ratio)
  (h2 : satisfiesRatio cream cream_ratio)
  (h3 : cake_ratio = { flour := 3, butter := 2, sugar := 1 })
  (h4 : cream_ratio = { flour := 0, butter := 2, sugar := 3 })
  (h5 : cookie_ratio = { flour := 5, butter := 3, sugar := 2 })
  (h6 : satisfiesRatio (mixRecipes cake cream 200) cookie_ratio) :
  cake.butter + cream.butter = 480 := by
  sorry


end NUMINAMATH_CALUDE_butter_amount_is_480_l4106_410682


namespace NUMINAMATH_CALUDE_game_result_l4106_410691

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls : List ℕ := [5, 6, 1, 2, 3]
def betty_rolls : List ℕ := [6, 1, 1, 2, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  (total_points allie_rolls) * (total_points betty_rolls) = 169 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l4106_410691


namespace NUMINAMATH_CALUDE_coin_pile_theorem_l4106_410699

/-- Represents the state of the three piles of coins -/
structure CoinState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Defines the allowed operations on the coin piles -/
inductive Operation
  | split_even : ℕ → Operation
  | remove_odd : ℕ → Operation

/-- Applies an operation to a CoinState -/
def apply_operation (state : CoinState) (op : Operation) : CoinState :=
  sorry

/-- Checks if a state has a pile with at least 2017^2017 coins -/
def has_large_pile (state : CoinState) : Prop :=
  sorry

/-- Theorem stating that any initial state except (2015, 2015, 2015) can reach a large pile -/
theorem coin_pile_theorem (initial : CoinState)
    (h1 : initial.a ≥ 2015)
    (h2 : initial.b ≥ 2015)
    (h3 : initial.c ≥ 2015)
    (h4 : ¬(initial.a = 2015 ∧ initial.b = 2015 ∧ initial.c = 2015)) :
    ∃ (ops : List Operation), has_large_pile (ops.foldl apply_operation initial) :=
  sorry

end NUMINAMATH_CALUDE_coin_pile_theorem_l4106_410699


namespace NUMINAMATH_CALUDE_f_at_five_halves_l4106_410631

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_symmetry (x : ℝ) : f ((-2) - x) = f ((-2) + x)
axiom f_period (x : ℝ) : f (x + 2) = f x
axiom f_definition (x : ℝ) (h : x ∈ Set.Icc (-3) (-2)) : f x = (x + 2)^2

-- State the theorem to be proved
theorem f_at_five_halves : f (5/2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_at_five_halves_l4106_410631


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l4106_410681

theorem consecutive_integers_square_difference (n : ℕ) : 
  (n > 0) → (n + (n + 1) = 105) → ((n + 1)^2 - n^2 = 105) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l4106_410681


namespace NUMINAMATH_CALUDE_percentage_increase_l4106_410650

theorem percentage_increase (original : ℝ) (difference : ℝ) (increase : ℝ) : 
  original = 80 →
  original + (increase / 100) * original - (original - 25 / 100 * original) = difference →
  difference = 30 →
  increase = 12.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l4106_410650


namespace NUMINAMATH_CALUDE_a_less_than_one_l4106_410630

-- Define the function f
def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

-- State the theorem
theorem a_less_than_one (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_one_l4106_410630


namespace NUMINAMATH_CALUDE_sanitizer_sprays_common_kill_percentage_l4106_410679

theorem sanitizer_sprays_common_kill_percentage 
  (spray1_kill : Real) 
  (spray2_kill : Real) 
  (combined_survival : Real) 
  (h1 : spray1_kill = 0.5) 
  (h2 : spray2_kill = 0.25) 
  (h3 : combined_survival = 0.3) : 
  spray1_kill + spray2_kill - (1 - combined_survival) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_sprays_common_kill_percentage_l4106_410679


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l4106_410675

theorem same_terminal_side_angle : ∃ θ : ℝ, 
  0 ≤ θ ∧ θ < 2*π ∧ 
  ∃ k : ℤ, θ = 2*k*π + (-4*π/3) ∧
  θ = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l4106_410675


namespace NUMINAMATH_CALUDE_max_value_of_expression_l4106_410612

theorem max_value_of_expression (a b c : ℝ) (h : a + 3 * b + c = 6) :
  (∀ x y z : ℝ, x + 3 * y + z = 6 → a * b + a * c + b * c ≥ x * y + x * z + y * z) →
  a * b + a * c + b * c = 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l4106_410612


namespace NUMINAMATH_CALUDE_seafood_price_seafood_price_proof_l4106_410617

/-- The regular price for two pounds of seafood given a 75% discount and $4 discounted price for one pound -/
theorem seafood_price : ℝ → ℝ → ℝ → Prop :=
  fun discount_percent discounted_price_per_pound regular_price_two_pounds =>
    discount_percent = 75 ∧
    discounted_price_per_pound = 4 →
    regular_price_two_pounds = 32

/-- Proof of the seafood price theorem -/
theorem seafood_price_proof :
  seafood_price 75 4 32 := by
  sorry

end NUMINAMATH_CALUDE_seafood_price_seafood_price_proof_l4106_410617


namespace NUMINAMATH_CALUDE_extremum_at_negative_one_l4106_410613

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extremum_at_negative_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a (-1) ≤ f a x ∨ f a (-1) ≥ f a x) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_negative_one_l4106_410613


namespace NUMINAMATH_CALUDE_complement_of_A_l4106_410600

def U : Set Nat := {2, 4, 6, 8, 10}
def A : Set Nat := {2, 6, 8}

theorem complement_of_A : (Aᶜ : Set Nat) = {4, 10} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l4106_410600


namespace NUMINAMATH_CALUDE_marble_distribution_l4106_410647

theorem marble_distribution (n : ℕ) : n = 720 → 
  (Finset.filter (fun x => x > 1 ∧ x < n) (Finset.range (n + 1))).card = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l4106_410647


namespace NUMINAMATH_CALUDE_paint_bottle_cost_l4106_410637

theorem paint_bottle_cost (num_cars num_paintbrushes num_paint_bottles : ℕ)
                          (car_cost paintbrush_cost total_spent : ℚ)
                          (h1 : num_cars = 5)
                          (h2 : num_paintbrushes = 5)
                          (h3 : num_paint_bottles = 5)
                          (h4 : car_cost = 20)
                          (h5 : paintbrush_cost = 2)
                          (h6 : total_spent = 160)
                          : (total_spent - (num_cars * car_cost + num_paintbrushes * paintbrush_cost)) / num_paint_bottles = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_bottle_cost_l4106_410637


namespace NUMINAMATH_CALUDE_davids_physics_marks_l4106_410616

def english_marks : ℕ := 36
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 57
def biology_marks : ℕ := 55
def average_marks : ℕ := 45
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks := total_marks - known_marks
  physics_marks = 42 := by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l4106_410616


namespace NUMINAMATH_CALUDE_cubic_inequality_l4106_410619

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l4106_410619


namespace NUMINAMATH_CALUDE_student_arrangement_count_l4106_410697

def num_male_students : ℕ := 3
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def adjacent_female_students : ℕ := 2

def num_arrangements : ℕ := 432

theorem student_arrangement_count :
  (num_male_students = 3) →
  (num_female_students = 3) →
  (total_students = num_male_students + num_female_students) →
  (adjacent_female_students = 2) →
  (num_arrangements = 432) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l4106_410697


namespace NUMINAMATH_CALUDE_cos_symmetry_center_l4106_410674

theorem cos_symmetry_center (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x + π / 3)
  let center : ℝ × ℝ := (π / 12, 0)
  ∀ t : ℝ, f (center.1 + t) = f (center.1 - t) :=
by sorry

end NUMINAMATH_CALUDE_cos_symmetry_center_l4106_410674


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l4106_410646

-- Define a line as a type
def Line := ℝ → ℝ → Prop

-- Define an angle as a pair of lines
def Angle := Line × Line

-- Define vertical angles
def VerticalAngles (a b : Angle) : Prop :=
  ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
    ((a.1 = l1 ∧ a.2 = l2) ∨ (a.1 = l2 ∧ a.2 = l1)) ∧
    ((b.1 = l1 ∧ b.2 = l2) ∨ (b.1 = l2 ∧ b.2 = l1))

-- Define angle measure
def AngleMeasure (a : Angle) : ℝ := sorry

-- Theorem: Vertical angles are always equal
theorem vertical_angles_equal (a b : Angle) :
  VerticalAngles a b → AngleMeasure a = AngleMeasure b := by
  sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l4106_410646


namespace NUMINAMATH_CALUDE_shaded_percentage_is_59_l4106_410672

def large_square_side_length : ℕ := 5
def small_square_side_length : ℕ := 1
def border_squares_count : ℕ := 16
def shaded_border_squares_count : ℕ := 8
def central_region_shaded_fraction : ℚ := 3 / 4

theorem shaded_percentage_is_59 :
  let total_area : ℚ := (large_square_side_length ^ 2 : ℚ)
  let border_area : ℚ := (border_squares_count * small_square_side_length ^ 2 : ℚ)
  let central_area : ℚ := total_area - border_area
  let shaded_border_area : ℚ := (shaded_border_squares_count * small_square_side_length ^ 2 : ℚ)
  let shaded_central_area : ℚ := central_region_shaded_fraction * central_area
  let total_shaded_area : ℚ := shaded_border_area + shaded_central_area
  (total_shaded_area / total_area) * 100 = 59 :=
by sorry

end NUMINAMATH_CALUDE_shaded_percentage_is_59_l4106_410672


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l4106_410611

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h1 : selling_price = 290)
  (h2 : cost_price = 241.67) : 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l4106_410611


namespace NUMINAMATH_CALUDE_half_sum_negative_l4106_410601

theorem half_sum_negative (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / 2 ∧ y < 0) ↔ (x + 3) / 2 < 0 := by
sorry

end NUMINAMATH_CALUDE_half_sum_negative_l4106_410601


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l4106_410618

theorem quadratic_coefficient (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 20 = (x + m)^2 + 8) → 
  b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l4106_410618


namespace NUMINAMATH_CALUDE_steve_calculation_l4106_410662

theorem steve_calculation (x : ℝ) : (x / 8) - 20 = 12 → (x * 8) + 20 = 2068 := by
  sorry

end NUMINAMATH_CALUDE_steve_calculation_l4106_410662


namespace NUMINAMATH_CALUDE_shaded_area_squares_l4106_410688

theorem shaded_area_squares (large_side small_side : ℝ) 
  (h1 : large_side = 14) 
  (h2 : small_side = 10) : 
  (large_side^2 - small_side^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_squares_l4106_410688


namespace NUMINAMATH_CALUDE_basketball_probability_l4106_410668

/-- A sequence of basketball shots where the probability of hitting each shot
    after the first two is equal to the proportion of shots hit so far. -/
def BasketballSequence (n : ℕ) : Type :=
  Fin n → Bool

/-- The probability of hitting exactly k shots out of n in a BasketballSequence. -/
def hitProbability (n k : ℕ) : ℚ :=
  if k = 0 ∨ k = n then 0
  else if k = 1 ∧ n = 2 then 1
  else 1 / (n - 1)

/-- The theorem stating the probability of hitting exactly k shots out of n. -/
theorem basketball_probability (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
    hitProbability n k = 1 / (n - 1) := by
  sorry

#eval hitProbability 100 50

end NUMINAMATH_CALUDE_basketball_probability_l4106_410668


namespace NUMINAMATH_CALUDE_camp_kids_count_l4106_410633

theorem camp_kids_count (total : ℕ) 
  (h1 : total / 2 = total / 2) -- Half of the kids are going to soccer camp
  (h2 : (total / 2) / 4 = (total / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  (h3 : ((total / 2) * 3) / 4 = 750) -- 750 kids go to soccer camp in the afternoon
  : total = 2000 := by
  sorry

end NUMINAMATH_CALUDE_camp_kids_count_l4106_410633


namespace NUMINAMATH_CALUDE_roots_sum_zero_l4106_410608

theorem roots_sum_zero (a b c : ℂ) : 
  a^3 - 2*a^2 + 3*a - 4 = 0 →
  b^3 - 2*b^2 + 3*b - 4 = 0 →
  c^3 - 2*c^2 + 3*c - 4 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1 / (a * (b^2 + c^2 - a^2)) + 1 / (b * (c^2 + a^2 - b^2)) + 1 / (c * (a^2 + b^2 - c^2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_zero_l4106_410608


namespace NUMINAMATH_CALUDE_cubic_equation_solution_product_l4106_410659

theorem cubic_equation_solution_product (d e f : ℝ) : 
  d^3 + 2*d^2 + 3*d - 5 = 0 ∧ 
  e^3 + 2*e^2 + 3*e - 5 = 0 ∧ 
  f^3 + 2*f^2 + 3*f - 5 = 0 → 
  (d - 1) * (e - 1) * (f - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_product_l4106_410659


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l4106_410645

theorem polynomial_coefficient_B (A : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ : ℕ+),
    (r₁ : ℤ) + r₂ + r₃ + r₄ = 7 ∧
    ∀ (z : ℂ), z^4 - 7*z^3 + A*z^2 + (-12)*z + 24 = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l4106_410645


namespace NUMINAMATH_CALUDE_sin_210_degrees_l4106_410604

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l4106_410604


namespace NUMINAMATH_CALUDE_expression_value_l4106_410603

theorem expression_value : 3 * (15 + 7)^2 - (15^2 + 7^2) = 1178 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4106_410603


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_other_root_l4106_410651

/-- The quadratic equation x^2 - 2x + m - 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m - 1 = 0

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, quadratic_equation y m ↔ y = x) →
  m = 2 :=
sorry

theorem quadratic_equation_other_root (m : ℝ) :
  (quadratic_equation 5 m) →
  (∃ x : ℝ, x ≠ 5 ∧ quadratic_equation x m ∧ x = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_other_root_l4106_410651


namespace NUMINAMATH_CALUDE_distribute_six_books_three_people_l4106_410623

/-- The number of ways to distribute n different books among k people, 
    with each person getting at least 1 book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 6 different books among 3 people, 
    with each person getting at least 1 book, can be done in 540 ways -/
theorem distribute_six_books_three_people : 
  distribute_books 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_distribute_six_books_three_people_l4106_410623


namespace NUMINAMATH_CALUDE_divisibility_property_l4106_410602

theorem divisibility_property (n : ℕ) : 
  n > 0 ∧ n^2 ∣ 2^n + 1 ↔ n = 1 ∨ n = 3 := by sorry

end NUMINAMATH_CALUDE_divisibility_property_l4106_410602


namespace NUMINAMATH_CALUDE_equality_check_l4106_410676

theorem equality_check : 
  (-3^2 ≠ -2^3) ∧ 
  (-6^3 = (-6)^3) ∧ 
  (-6^2 ≠ (-6)^2) ∧ 
  ((-3 * 2)^2 ≠ (-3) * 2^2) :=
by
  sorry

end NUMINAMATH_CALUDE_equality_check_l4106_410676


namespace NUMINAMATH_CALUDE_cakes_served_today_l4106_410670

theorem cakes_served_today (lunch_cakes dinner_cakes : ℕ) 
  (h1 : lunch_cakes = 6) 
  (h2 : dinner_cakes = 9) : 
  lunch_cakes + dinner_cakes = 15 := by
sorry

end NUMINAMATH_CALUDE_cakes_served_today_l4106_410670


namespace NUMINAMATH_CALUDE_distance_to_leg_intersection_l4106_410690

/-- An isosceles trapezoid with specific diagonal properties -/
structure IsoscelesTrapezoid where
  /-- The length of the longer segment of each diagonal -/
  long_segment : ℝ
  /-- The length of the shorter segment of each diagonal -/
  short_segment : ℝ
  /-- The angle between the diagonals formed by the legs -/
  diagonal_angle : ℝ
  /-- Condition: The longer segment is 7 -/
  long_is_7 : long_segment = 7
  /-- Condition: The shorter segment is 3 -/
  short_is_3 : short_segment = 3
  /-- Condition: The angle between diagonals is 60° -/
  angle_is_60 : diagonal_angle = 60

/-- The theorem stating the distance from diagonal intersection to leg intersection -/
theorem distance_to_leg_intersection (t : IsoscelesTrapezoid) :
  (t.long_segment / t.short_segment) * t.short_segment = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_leg_intersection_l4106_410690


namespace NUMINAMATH_CALUDE_function_inequality_l4106_410663

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x > 1 → y > x → f x < f y)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- State the theorem
theorem function_inequality : f (-1) < f 0 ∧ f 0 < f 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4106_410663


namespace NUMINAMATH_CALUDE_sequence_length_divisible_by_three_l4106_410698

theorem sequence_length_divisible_by_three (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 3)
  (h2 : ∀ i, a (i + n) = a i)
  (h3 : ∀ i, a i * a (i + 1) + 1 = a (i + 2)) :
  ∃ k : ℕ, n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_sequence_length_divisible_by_three_l4106_410698


namespace NUMINAMATH_CALUDE_green_blue_difference_l4106_410654

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 72) : 
  bag.green - bag.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l4106_410654


namespace NUMINAMATH_CALUDE_log_equality_l4106_410660

theorem log_equality : Real.log 81 / Real.log 4 = Real.log 9 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l4106_410660


namespace NUMINAMATH_CALUDE_rational_numbers_definition_l4106_410639

-- Define the set of rational numbers
def RationalNumbers : Set ℚ := {q : ℚ | true}

-- Define the set of integers as a subset of rational numbers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions as a subset of rational numbers
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem stating that rational numbers are the union of integers and fractions
theorem rational_numbers_definition : 
  RationalNumbers = Integers ∪ Fractions := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_definition_l4106_410639


namespace NUMINAMATH_CALUDE_hypotenuse_plus_diameter_eq_sum_of_legs_l4106_410635

/-- Represents a right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  a : ℝ     -- Length of one leg
  b : ℝ     -- Length of the other leg
  c : ℝ     -- Length of the hypotenuse
  ρ : ℝ     -- Radius of the inscribed circle
  h_right : a^2 + b^2 = c^2  -- Pythagorean theorem
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ρ > 0  -- Positive lengths

/-- 
The sum of the hypotenuse and the diameter of the inscribed circle 
is equal to the sum of the two legs in a right-angled triangle
-/
theorem hypotenuse_plus_diameter_eq_sum_of_legs 
  (t : RightTriangleWithInscribedCircle) : t.c + 2 * t.ρ = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_plus_diameter_eq_sum_of_legs_l4106_410635


namespace NUMINAMATH_CALUDE_complex_simplification_l4106_410680

theorem complex_simplification :
  ((2 + Complex.I) ^ 200) / ((2 - Complex.I) ^ 200) = Complex.exp (200 * Complex.I * Complex.arctan (4 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l4106_410680


namespace NUMINAMATH_CALUDE_f_of_3_eq_3_l4106_410640

/-- The exponent in the function definition -/
def n : ℕ := 2008

/-- The function f(x) is defined implicitly by this equation -/
def f_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (x^(3^n - 1) - 1) * f x = 
    (List.range n).foldl (λ acc i => acc * (x^(3^i) + 1)) (x + 1) + (x^2 - 1) - 1

/-- The theorem stating that f(3) = 3 -/
theorem f_of_3_eq_3 (f : ℝ → ℝ) (hf : ∀ x, f_equation f x) : f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_3_l4106_410640


namespace NUMINAMATH_CALUDE_base_b_perfect_square_l4106_410673

-- Define the representation of a number in base b
def base_representation (b : ℕ) : ℕ := b^2 + 4*b + 1

-- Theorem statement
theorem base_b_perfect_square (b : ℕ) (h : b > 4) :
  ∃ n : ℕ, base_representation b = n^2 :=
sorry

end NUMINAMATH_CALUDE_base_b_perfect_square_l4106_410673


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l4106_410658

/-- The area of a rhombus inscribed in a circle, which is in turn inscribed in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 16) : 
  let r := s / 2
  let d := s
  let rhombus_area := d * d / 2
  rhombus_area = 128 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l4106_410658


namespace NUMINAMATH_CALUDE_number_ratio_l4106_410606

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 5) = 117) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l4106_410606


namespace NUMINAMATH_CALUDE_linear_equation_implies_specific_value_l4106_410692

/-- 
If $2x^{2a-b}-y^{a+b-1}=3$ is a linear equation in $x$ and $y$, 
then $(a-2b)^{2023} = -1$.
-/
theorem linear_equation_implies_specific_value (a b : ℝ) : 
  (∀ x y, ∃ k₁ k₂ c : ℝ, 2 * x^(2*a-b) - y^(a+b-1) = k₁ * x + k₂ * y + c) → 
  (a - 2*b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_implies_specific_value_l4106_410692


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l4106_410696

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_two_digit_prime : Nat := 97

theorem product_of_specific_primes :
  (smallest_one_digit_primes.prod * largest_two_digit_prime) = 582 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l4106_410696


namespace NUMINAMATH_CALUDE_mrs_petersons_change_l4106_410693

def change_calculation (num_tumblers : ℕ) (cost_per_tumbler : ℚ) (discount_rate : ℚ) (num_bills : ℕ) (bill_value : ℚ) : ℚ :=
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * bill_value
  total_amount_paid - total_cost_after_discount

theorem mrs_petersons_change :
  change_calculation 10 45 (1/10) 5 100 = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_petersons_change_l4106_410693


namespace NUMINAMATH_CALUDE_intersection_slope_l4106_410694

/-- Given two lines m and n that intersect at (-4, 0), prove that the slope of line n is -9/4 -/
theorem intersection_slope (k : ℚ) : 
  (∀ x y, y = 2 * x + 8 → y = k * x - 9 → x = -4 ∧ y = 0) → 
  k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l4106_410694


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4106_410627

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4106_410627


namespace NUMINAMATH_CALUDE_week_net_change_l4106_410649

/-- The net change in stock exchange points for a week -/
def net_change (monday tuesday wednesday thursday friday : Int) : Int :=
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the net change for the given week is -119 -/
theorem week_net_change :
  net_change (-150) 106 (-47) 182 (-210) = -119 := by
  sorry

end NUMINAMATH_CALUDE_week_net_change_l4106_410649


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l4106_410656

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def applySimilarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let A : Point3D := ⟨1, 2, 2⟩
  let a : Plane := ⟨3, 0, -1, 5⟩
  let k : ℝ := -1/5
  let a' : Plane := applySimilarity a k
  pointOnPlane A a' := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l4106_410656


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l4106_410678

theorem quadratic_equation_roots_ratio (c : ℚ) : 
  (∃ x y : ℚ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + c = 0 ∧ y^2 + 10*y + c = 0) → 
  c = 75/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l4106_410678


namespace NUMINAMATH_CALUDE_division_problem_l4106_410689

theorem division_problem : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4106_410689


namespace NUMINAMATH_CALUDE_roots_of_equation_l4106_410686

theorem roots_of_equation : ∀ x : ℝ, 
  (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l4106_410686


namespace NUMINAMATH_CALUDE_sequence_range_l4106_410669

/-- Given a sequence {a_n} with the following properties:
  1) a_1 = a > 0
  2) a_(n+1) = -a_n^2 + t*a_n for n ∈ ℕ*
  3) There exists a real number t that makes {a_n} monotonically increasing
  Then the range of a is (0,1) -/
theorem sequence_range (a : ℝ) (t : ℝ) (a_n : ℕ → ℝ) :
  a > 0 →
  (∀ n : ℕ, n > 0 → a_n (n + 1) = -a_n n ^ 2 + t * a_n n) →
  (∃ t : ℝ, ∀ n : ℕ, n > 0 → a_n (n + 1) > a_n n) →
  a_n 1 = a →
  0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_range_l4106_410669


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4106_410642

theorem rectangle_dimensions (x y : ℝ) : 
  (2*x + y) * (2*y) = 90 ∧ x*y = 10 → x = 2 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4106_410642


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l4106_410684

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_iff_same_slope {a b c d : ℝ} (l1 l2 : ℝ → ℝ → Prop) :
  (∀ x y, l1 x y ↔ a * x + b * y = 0) →
  (∀ x y, l2 x y ↔ c * x + d * y = 1) →
  (∀ x y, l1 x y → l2 x y) ↔ a / b = c / d

/-- The line ax + 2y = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y = 0

/-- The line x + y = 1 -/
def line2 (x y : ℝ) : Prop := x + y = 1

/-- Theorem: a = 2 is both sufficient and necessary for line1 to be parallel to line2 -/
theorem parallel_iff_a_eq_two (a : ℝ) :
  (∀ x y, line1 a x y → line2 x y) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l4106_410684


namespace NUMINAMATH_CALUDE_water_displaced_squared_volume_l4106_410607

/-- The volume of water displaced by a cube in a cylindrical tank -/
def water_displaced (cube_side : ℝ) (tank_radius : ℝ) (tank_height : ℝ) : ℝ :=
  -- Definition left abstract
  sorry

/-- The main theorem stating the squared volume of water displaced -/
theorem water_displaced_squared_volume :
  let cube_side : ℝ := 10
  let tank_radius : ℝ := 5
  let tank_height : ℝ := 12
  (water_displaced cube_side tank_radius tank_height) ^ 2 = 79156.25 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_volume_l4106_410607


namespace NUMINAMATH_CALUDE_logical_equivalence_l4106_410657

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ R) → ¬Q) ↔ (Q → (¬P ∨ ¬R)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l4106_410657


namespace NUMINAMATH_CALUDE_inequality_and_not_all_greater_l4106_410667

theorem inequality_and_not_all_greater (m a b x y z : ℝ) : 
  m > 0 → 
  0 < x → x < 2 → 
  0 < y → y < 2 → 
  0 < z → z < 2 → 
  ((a + m * b) / (1 + m))^2 ≤ (a^2 + m * b^2) / (1 + m) ∧ 
  ¬(x * (2 - y) > 1 ∧ y * (2 - z) > 1 ∧ z * (2 - x) > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_not_all_greater_l4106_410667


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l4106_410666

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (0.3 * 0.6 * y) / y * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l4106_410666


namespace NUMINAMATH_CALUDE_swimming_pool_kids_jose_swimming_pool_l4106_410610

theorem swimming_pool_kids (kids_charge : ℕ) (adults_charge : ℕ) 
  (adults_per_day : ℕ) (weekly_earnings : ℕ) : ℕ :=
  let kids_per_day := 
    (weekly_earnings / 7 - adults_per_day * adults_charge) / kids_charge
  kids_per_day

theorem jose_swimming_pool : swimming_pool_kids 3 6 10 588 = 8 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_kids_jose_swimming_pool_l4106_410610


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4106_410655

theorem quadratic_roots_condition (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 2 ∧ 
   x₁^2 + 2*p*x₁ + q = 0 ∧ x₂^2 + 2*p*x₂ + q = 0) ↔ 
  (q > 0 ∧ p < -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4106_410655


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4106_410653

theorem exponent_multiplication (x : ℝ) (m n : ℕ) :
  x^m * x^n = x^(m + n) := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4106_410653


namespace NUMINAMATH_CALUDE_j_walking_speed_l4106_410641

/-- Represents the walking speed of J in kmph -/
def j_speed : ℝ := 5.945

/-- Represents the cycling speed of P in kmph -/
def p_speed : ℝ := 8

/-- Represents the time (in hours) between J's start and P's start -/
def time_difference : ℝ := 1.5

/-- Represents the total time (in hours) from J's start to when P catches up -/
def total_time : ℝ := 7.3

/-- Represents the time (in hours) P cycles before catching up to J -/
def p_cycle_time : ℝ := 5.8

/-- Represents the distance (in km) J is behind P when P catches up -/
def distance_behind : ℝ := 3

theorem j_walking_speed :
  p_speed * p_cycle_time = j_speed * total_time + distance_behind :=
sorry

end NUMINAMATH_CALUDE_j_walking_speed_l4106_410641
