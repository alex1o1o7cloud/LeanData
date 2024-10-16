import Mathlib

namespace NUMINAMATH_CALUDE_solution_values_l684_68423

-- Define the equations
def equation_1 (a x : ℝ) : Prop := a * x + 3 = 2 * (x - a)
def equation_2 (x : ℝ) : Prop := |x - 2| - 3 = 0

-- Theorem statement
theorem solution_values (a x : ℝ) :
  equation_1 a x ∧ equation_2 x → a = -5 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l684_68423


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l684_68406

/-- Given a cone with base radius 4 cm and unfolded lateral surface radius 5 cm,
    prove that its lateral surface area is 20π cm². -/
theorem cone_lateral_surface_area :
  ∀ (base_radius unfolded_radius : ℝ),
    base_radius = 4 →
    unfolded_radius = 5 →
    let lateral_area := (1/2) * unfolded_radius^2 * (2 * Real.pi * base_radius / unfolded_radius)
    lateral_area = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l684_68406


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l684_68493

/-- Represents the seven points on the circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven

/-- Determines if a CirclePoint is prime -/
def isPrime : CirclePoint → Bool
  | CirclePoint.two => true
  | CirclePoint.three => true
  | CirclePoint.five => true
  | CirclePoint.seven => true
  | _ => false

/-- Determines if a CirclePoint is composite -/
def isComposite : CirclePoint → Bool
  | CirclePoint.four => true
  | CirclePoint.six => true
  | _ => false

/-- Moves the bug according to the jumping rule -/
def move (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.one => CirclePoint.two
  | CirclePoint.two => CirclePoint.three
  | CirclePoint.three => CirclePoint.four
  | CirclePoint.four => CirclePoint.seven
  | CirclePoint.five => CirclePoint.six
  | CirclePoint.six => CirclePoint.two
  | CirclePoint.seven => CirclePoint.one

/-- Performs n jumps starting from a given point -/
def jumpN (start : CirclePoint) (n : Nat) : CirclePoint :=
  match n with
  | 0 => start
  | n + 1 => move (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN CirclePoint.seven 2023 = CirclePoint.two := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l684_68493


namespace NUMINAMATH_CALUDE_valentines_day_problem_l684_68458

theorem valentines_day_problem (boys girls : ℕ) : 
  boys * girls = boys + girls + 52 → boys * girls = 108 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_problem_l684_68458


namespace NUMINAMATH_CALUDE_decryption_theorem_l684_68498

-- Define the encryption functions
def encrypt_a (a : ℤ) : ℤ := a + 1
def encrypt_b (a b : ℤ) : ℤ := 2 * b + a
def encrypt_c (c : ℤ) : ℤ := 3 * c - 4

-- Define the theorem
theorem decryption_theorem (a b c : ℤ) :
  encrypt_a a = 21 ∧ encrypt_b a b = 22 ∧ encrypt_c c = 23 →
  a = 20 ∧ b = 1 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_decryption_theorem_l684_68498


namespace NUMINAMATH_CALUDE_greatest_integer_a_l684_68484

theorem greatest_integer_a : ∃ (a : ℤ), 
  (∀ (x : ℤ), (x - a) * (x - 7) + 3 ≠ 0) ∧ 
  (∃ (x : ℤ), (x - 11) * (x - 7) + 3 = 0) ∧
  (∀ (b : ℤ), b > 11 → ∀ (x : ℤ), (x - b) * (x - 7) + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_a_l684_68484


namespace NUMINAMATH_CALUDE_price_change_calculation_l684_68425

theorem price_change_calculation (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_decrease := 0.8 * initial_price
  let final_price := 1.04 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 30 :=
by sorry

end NUMINAMATH_CALUDE_price_change_calculation_l684_68425


namespace NUMINAMATH_CALUDE_jeff_peanut_butter_amount_l684_68441

/-- The amount of peanut butter in ounces for each jar size -/
def jar_sizes : List Nat := [16, 28, 40]

/-- The total number of jars Jeff has -/
def total_jars : Nat := 9

/-- Theorem stating that Jeff has 252 ounces of peanut butter -/
theorem jeff_peanut_butter_amount :
  (total_jars / jar_sizes.length) * (jar_sizes.sum) = 252 := by
  sorry

#check jeff_peanut_butter_amount

end NUMINAMATH_CALUDE_jeff_peanut_butter_amount_l684_68441


namespace NUMINAMATH_CALUDE_largest_house_number_l684_68421

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The largest 3-digit number with distinct digits whose sum equals the sum of digits in 5039821 -/
theorem largest_house_number : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    has_distinct_digits n ∧
    digit_sum n = digit_sum 5039821 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < 1000 ∧ 
      has_distinct_digits m ∧ 
      digit_sum m = digit_sum 5039821 → 
      m ≤ n ∧
    n = 981 := by sorry

end NUMINAMATH_CALUDE_largest_house_number_l684_68421


namespace NUMINAMATH_CALUDE_first_group_machines_correct_l684_68478

/-- The number of machines in the first group -/
def first_group_machines : ℕ := 5

/-- The production rate of the first group (units per machine-hour) -/
def first_group_rate : ℚ := 20 / (first_group_machines * 10)

/-- The production rate of the second group (units per machine-hour) -/
def second_group_rate : ℚ := 180 / (20 * 22.5)

/-- Theorem stating that the number of machines in the first group is correct -/
theorem first_group_machines_correct :
  first_group_rate = second_group_rate ∧
  first_group_machines * first_group_rate * 10 = 20 := by
  sorry

#check first_group_machines_correct

end NUMINAMATH_CALUDE_first_group_machines_correct_l684_68478


namespace NUMINAMATH_CALUDE_green_percentage_is_25_l684_68449

def amber_pieces : ℕ := 20
def green_pieces : ℕ := 35
def clear_pieces : ℕ := 85

def total_pieces : ℕ := amber_pieces + green_pieces + clear_pieces

def percentage_green : ℚ := (green_pieces : ℚ) / (total_pieces : ℚ) * 100

theorem green_percentage_is_25 : percentage_green = 25 := by
  sorry

end NUMINAMATH_CALUDE_green_percentage_is_25_l684_68449


namespace NUMINAMATH_CALUDE_a_142_equals_1995_and_unique_l684_68400

def p (n : ℕ) : ℕ := sorry

def q (n : ℕ) : ℕ := sorry

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n * p (a n) / q (a n)

theorem a_142_equals_1995_and_unique :
  a 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → a n ≠ 1995 := by sorry

end NUMINAMATH_CALUDE_a_142_equals_1995_and_unique_l684_68400


namespace NUMINAMATH_CALUDE_fraction_zero_implies_negative_one_l684_68402

theorem fraction_zero_implies_negative_one (x : ℝ) :
  (x^2 - 1) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_negative_one_l684_68402


namespace NUMINAMATH_CALUDE_mean_temperature_l684_68447

def temperatures : List ℝ := [-8, -5, -3, -5, 2, 4, 3, -1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℝ) = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l684_68447


namespace NUMINAMATH_CALUDE_island_area_l684_68451

/-- The area of a rectangular island with width 5 miles and length 10 miles is 50 square miles. -/
theorem island_area : 
  let width : ℝ := 5
  let length : ℝ := 10
  width * length = 50 := by sorry

end NUMINAMATH_CALUDE_island_area_l684_68451


namespace NUMINAMATH_CALUDE_term_position_98_l684_68487

/-- The sequence defined by a_n = n^2 / (n^2 + 1) -/
def a (n : ℕ) : ℚ := n^2 / (n^2 + 1)

/-- The theorem stating that 0.98 occurs at position 7 in the sequence -/
theorem term_position_98 : a 7 = 98/100 := by
  sorry

end NUMINAMATH_CALUDE_term_position_98_l684_68487


namespace NUMINAMATH_CALUDE_power_seven_mod_nine_l684_68424

theorem power_seven_mod_nine : 7^123 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nine_l684_68424


namespace NUMINAMATH_CALUDE_largest_root_of_equation_l684_68467

theorem largest_root_of_equation (x : ℝ) :
  (x - 37)^2 - 169 = 0 → x ≤ 50 ∧ ∃ y, (y - 37)^2 - 169 = 0 ∧ y = 50 := by
  sorry

end NUMINAMATH_CALUDE_largest_root_of_equation_l684_68467


namespace NUMINAMATH_CALUDE_stationery_store_problem_l684_68461

/-- Represents the weekly sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 80

/-- Represents the weekly profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

theorem stationery_store_problem 
  (h_price_range : ∀ x, 20 ≤ x ∧ x ≤ 28 → x ∈ Set.Icc 20 28)
  (h_sales_22 : sales_volume 22 = 36)
  (h_sales_24 : sales_volume 24 = 32) :
  (∀ x, sales_volume x = -2 * x + 80) ∧
  (∃ x ∈ Set.Icc 20 28, profit x = 150 ∧ x = 25) ∧
  (∀ x ∈ Set.Icc 20 28, profit x ≤ profit 28 ∧ profit 28 = 192) := by
  sorry

#check stationery_store_problem

end NUMINAMATH_CALUDE_stationery_store_problem_l684_68461


namespace NUMINAMATH_CALUDE_purchase_cost_l684_68452

/-- The cost of purchasing bananas and oranges -/
theorem purchase_cost (banana_quantity : ℕ) (orange_quantity : ℕ) 
  (banana_price : ℚ) (orange_price : ℚ) : 
  banana_quantity = 5 → 
  orange_quantity = 10 → 
  banana_price = 2 → 
  orange_price = (3/2) → 
  banana_quantity * banana_price + orange_quantity * orange_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l684_68452


namespace NUMINAMATH_CALUDE_function_continuity_l684_68445

-- Define a function f on the real line
variable (f : ℝ → ℝ)

-- Define the condition that f(x) + f(ax) is continuous for any a > 1
def condition (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, a > 1 → Continuous (fun x ↦ f x + f (a * x))

-- Theorem statement
theorem function_continuity (h : condition f) : Continuous f := by
  sorry

end NUMINAMATH_CALUDE_function_continuity_l684_68445


namespace NUMINAMATH_CALUDE_midpoint_chain_l684_68411

theorem midpoint_chain (A B C D E F G : ℝ) : 
  C = (A + B) / 2 →  -- C is midpoint of AB
  D = (A + C) / 2 →  -- D is midpoint of AC
  E = (A + D) / 2 →  -- E is midpoint of AD
  F = (A + E) / 2 →  -- F is midpoint of AE
  G = (A + F) / 2 →  -- G is midpoint of AF
  G - A = 2 →        -- AG = 2
  B - A = 64 :=      -- AB = 64
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l684_68411


namespace NUMINAMATH_CALUDE_power_calculation_l684_68479

theorem power_calculation : (10 ^ 6 : ℕ) * (10 ^ 2 : ℕ) ^ 3 / (10 ^ 4 : ℕ) = 10 ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l684_68479


namespace NUMINAMATH_CALUDE_simplified_ratio_l684_68489

def initial_money : ℕ := 91
def spent_money : ℕ := 21

def money_left : ℕ := initial_money - spent_money

def ratio_numerator : ℕ := money_left
def ratio_denominator : ℕ := spent_money

theorem simplified_ratio :
  (ratio_numerator / (Nat.gcd ratio_numerator ratio_denominator)) = 10 ∧
  (ratio_denominator / (Nat.gcd ratio_numerator ratio_denominator)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_ratio_l684_68489


namespace NUMINAMATH_CALUDE_sum_of_digits_l684_68464

/-- The decimal representation of 1/142857 -/
def decimal_rep : ℚ := 1 / 142857

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 7

/-- The sum of digits in one repeating sequence -/
def cycle_sum : ℕ := 7

/-- The number of digits we're considering after the decimal point -/
def digit_count : ℕ := 35

/-- Theorem: The sum of the first 35 digits after the decimal point
    in the decimal representation of 1/142857 is equal to 35 -/
theorem sum_of_digits :
  (digit_count / repeat_length) * cycle_sum = 35 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_l684_68464


namespace NUMINAMATH_CALUDE_bracelets_made_l684_68407

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The theorem stating the number of bracelets Nancy and Rose can make -/
theorem bracelets_made : total_beads / beads_per_bracelet = 20 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_made_l684_68407


namespace NUMINAMATH_CALUDE_land_area_scientific_notation_l684_68434

theorem land_area_scientific_notation :
  let land_area : ℝ := 9600000
  9.6 * (10 ^ 6) = land_area := by
  sorry

end NUMINAMATH_CALUDE_land_area_scientific_notation_l684_68434


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_divisible_by_61_l684_68475

theorem sum_of_odd_powers_divisible_by_61 
  (a₁ a₂ a₃ a₄ : ℤ) 
  (h : a₁^3 + a₂^3 + a₃^3 + a₄^3 = 0) :
  ∀ k : ℕ, k % 2 = 1 → k > 0 → 
  (61 : ℤ) ∣ (a₁^k + a₂^k + a₃^k + a₄^k) := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_divisible_by_61_l684_68475


namespace NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l684_68496

/-- Represents the state of the game at any point --/
structure GameState where
  numPiles : ℕ
  totalTokens : ℕ

/-- The invariant of the game --/
def invariant (state : GameState) : ℕ :=
  state.numPiles + state.totalTokens

/-- The initial state of the game --/
def initialState : GameState :=
  { numPiles := 1, totalTokens := 1001 }

/-- Theorem stating the impossibility of reaching a state with only piles of 3 tokens --/
theorem impossible_to_reach_all_threes :
  ¬∃ (k : ℕ), invariant initialState = 4 * k :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l684_68496


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l684_68470

theorem smallest_n_for_inequality : ∃ (n : ℕ+),
  (∀ (m : ℕ), 0 < m → m < 2001 → 
    ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n : ℚ) ∧ (k : ℚ) / (n : ℚ) < ((m + 1) : ℚ) / 2002) ∧
  (∀ (n' : ℕ+), 
    (∀ (m : ℕ), 0 < m → m < 2001 → 
      ∃ (k : ℤ), (m : ℚ) / 2001 < (k : ℚ) / (n' : ℚ) ∧ (k : ℚ) / (n' : ℚ) < ((m + 1) : ℚ) / 2002) →
    n ≤ n') ∧
  n = 4003 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l684_68470


namespace NUMINAMATH_CALUDE_largest_value_when_x_is_9_l684_68431

theorem largest_value_when_x_is_9 :
  let x : ℝ := 9
  (x / 2 > Real.sqrt x) ∧
  (x / 2 > x - 5) ∧
  (x / 2 > 40 / x) ∧
  (x / 2 > x^2 / 20) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_when_x_is_9_l684_68431


namespace NUMINAMATH_CALUDE_two_white_marbles_probability_l684_68480

/-- The probability of drawing two white marbles consecutively without replacement from a bag containing 5 red marbles and 7 white marbles is 7/22. -/
theorem two_white_marbles_probability :
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + white_marbles
  let prob_first_white : ℚ := white_marbles / total_marbles
  let prob_second_white : ℚ := (white_marbles - 1) / (total_marbles - 1)
  prob_first_white * prob_second_white = 7 / 22 :=
by sorry

end NUMINAMATH_CALUDE_two_white_marbles_probability_l684_68480


namespace NUMINAMATH_CALUDE_total_fabric_needed_l684_68437

/-- The number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- The number of pants Kingsley makes per day -/
def kingsley_pants_per_day : ℕ := 5

/-- The number of yards of fabric used for one shirt -/
def fabric_per_shirt : ℕ := 2

/-- The number of yards of fabric used for one pair of pants -/
def fabric_per_pants : ℕ := 5

/-- The number of days to calculate fabric for -/
def days : ℕ := 3

/-- Theorem stating the total yards of fabric needed every 3 days -/
theorem total_fabric_needed : 
  jenson_shirts_per_day * fabric_per_shirt * days + 
  kingsley_pants_per_day * fabric_per_pants * days = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_fabric_needed_l684_68437


namespace NUMINAMATH_CALUDE_sum_of_hundred_consecutive_integers_l684_68419

theorem sum_of_hundred_consecutive_integers : ∃ k : ℕ, 
  50 * (2 * k + 99) = 1627384950 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_hundred_consecutive_integers_l684_68419


namespace NUMINAMATH_CALUDE_oxen_grazing_problem_l684_68460

theorem oxen_grazing_problem (total_rent : ℕ) (a_months b_oxen b_months c_oxen c_months : ℕ) (c_share : ℕ) :
  total_rent = 175 →
  a_months = 7 →
  b_oxen = 12 →
  b_months = 5 →
  c_oxen = 15 →
  c_months = 3 →
  c_share = 45 →
  ∃ a_oxen : ℕ, a_oxen * a_months + b_oxen * b_months + c_oxen * c_months = total_rent ∧ a_oxen = 10 := by
  sorry


end NUMINAMATH_CALUDE_oxen_grazing_problem_l684_68460


namespace NUMINAMATH_CALUDE_number_relationship_l684_68409

/-- Proves that given a = 0.5^6, b = log_5(0.6), and c = 6^0.5, the relationship b < a < c holds. -/
theorem number_relationship :
  let a : ℝ := (1/2)^6
  let b : ℝ := Real.log 0.6 / Real.log 5
  let c : ℝ := 6^(1/2)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_number_relationship_l684_68409


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l684_68430

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ y : ℝ, x = y^2

-- Define what it means for a quadratic radical to be simplest
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ 
  ∀ y : ℝ, (∃ z : ℝ, y = z^2 ∧ x = y * z) → y = 1

-- State the theorem
theorem simplest_quadratic_radical :
  SimplestQuadraticRadical 6 ∧
  ¬SimplestQuadraticRadical 12 ∧
  ¬SimplestQuadraticRadical 0.3 ∧
  ¬SimplestQuadraticRadical (1/2) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l684_68430


namespace NUMINAMATH_CALUDE_minimum_candies_to_remove_l684_68477

/-- Represents the number of candies of each flavor in the bag -/
structure CandyBag where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial state of the candy bag -/
def initialBag : CandyBag := { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies (bag : CandyBag) : Nat :=
  bag.chocolate + bag.mint + bag.butterscotch

/-- Predicate to check if at least two candies of each flavor have been eaten -/
def atLeastTwoEachFlavor (removed : Nat) (bag : CandyBag) : Prop :=
  removed ≥ bag.chocolate - 1 ∧ removed ≥ bag.mint - 1 ∧ removed ≥ bag.butterscotch - 1

theorem minimum_candies_to_remove (bag : CandyBag) :
  totalCandies bag = 20 →
  bag = initialBag →
  ∃ (n : Nat), n = 18 ∧ 
    (∀ (m : Nat), m < n → ¬(atLeastTwoEachFlavor m bag)) ∧
    (atLeastTwoEachFlavor n bag) := by
  sorry

end NUMINAMATH_CALUDE_minimum_candies_to_remove_l684_68477


namespace NUMINAMATH_CALUDE_natasha_dimes_l684_68476

theorem natasha_dimes (n : ℕ) 
  (h1 : 100 < n ∧ n < 200)
  (h2 : n % 6 = 2)
  (h3 : n % 7 = 2)
  (h4 : n % 8 = 2) : 
  n = 170 := by sorry

end NUMINAMATH_CALUDE_natasha_dimes_l684_68476


namespace NUMINAMATH_CALUDE_min_length_shared_side_l684_68446

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the length of BC must be greater than 14. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 7 → AC = 15 → DC = 9 → BD = 23 → BC > 14 := by
  sorry

end NUMINAMATH_CALUDE_min_length_shared_side_l684_68446


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l684_68408

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l684_68408


namespace NUMINAMATH_CALUDE_exponent_value_l684_68485

theorem exponent_value : ∃ exponent : ℝ,
  (1/5 : ℝ)^35 * (1/4 : ℝ)^exponent = 1 / (2 * (10 : ℝ)^35) ∧ exponent = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_value_l684_68485


namespace NUMINAMATH_CALUDE_unique_triangle_l684_68440

/-- 
A triple of positive integers (a, a, b) represents an acute-angled isosceles triangle 
with perimeter 31 if and only if it satisfies the following conditions:
1. 2a + b = 31 (perimeter condition)
2. a < b < 2a (acute-angled isosceles condition)
-/
def is_valid_triangle (a b : ℕ) : Prop :=
  2 * a + b = 31 ∧ a < b ∧ b < 2 * a

/-- There exists exactly one triple of positive integers (a, a, b) that represents 
an acute-angled isosceles triangle with perimeter 31. -/
theorem unique_triangle : ∃! p : ℕ × ℕ, is_valid_triangle p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_l684_68440


namespace NUMINAMATH_CALUDE_solution_and_uniqueness_l684_68420

def equation (x : ℤ) : Prop :=
  (x + 1)^3 + (x + 2)^3 + (x + 3)^3 = (x + 4)^3

theorem solution_and_uniqueness :
  equation 2 ∧ ∀ x : ℤ, x ≠ 2 → ¬(equation x) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_uniqueness_l684_68420


namespace NUMINAMATH_CALUDE_range_of_a_l684_68482

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 ≤ 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, ¬(p x a) → ¬(q x))
  (h3 : ∃ x, ¬(p x a) ∧ q x) :
  0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l684_68482


namespace NUMINAMATH_CALUDE_paint_usage_fraction_l684_68483

theorem paint_usage_fraction (total_paint : ℚ) (first_week_fraction : ℚ) (total_used : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1 / 9 →
  total_used = 104 →
  let remaining_paint := total_paint - first_week_fraction * total_paint
  let second_week_usage := total_used - first_week_fraction * total_paint
  second_week_usage / remaining_paint = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_fraction_l684_68483


namespace NUMINAMATH_CALUDE_largest_x_value_l684_68472

theorem largest_x_value (x : ℝ) : 
  (3 * x / 7 + 2 / (9 * x) = 1) → 
  x ≤ (63 + Real.sqrt 2457) / 54 ∧ 
  ∃ y : ℝ, (3 * y / 7 + 2 / (9 * y) = 1) ∧ y = (63 + Real.sqrt 2457) / 54 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l684_68472


namespace NUMINAMATH_CALUDE_constant_sum_inverse_lengths_l684_68432

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a perpendicular line passing through F
def perp_line_through_F (k : ℝ) (x y : ℝ) : Prop := y = -(1/k) * (x - 1)

-- Define the theorem
theorem constant_sum_inverse_lengths 
  (k : ℝ) 
  (A B C D : ℝ × ℝ) 
  (hA : curve A.1 A.2) (hB : curve B.1 B.2) (hC : curve C.1 C.2) (hD : curve D.1 D.2)
  (hAB : line_through_F k A.1 A.2 ∧ line_through_F k B.1 B.2)
  (hCD : perp_line_through_F k C.1 C.2 ∧ perp_line_through_F k D.1 D.2)
  (hk : k ≠ 0) :
  1 / Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + 
  1 / Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 17/48 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_inverse_lengths_l684_68432


namespace NUMINAMATH_CALUDE_book_selection_problem_l684_68494

theorem book_selection_problem (total_books : ℕ) (novels : ℕ) (to_choose : ℕ) :
  total_books = 15 →
  novels = 5 →
  to_choose = 3 →
  (Nat.choose total_books to_choose) - (Nat.choose (total_books - novels) to_choose) = 335 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_problem_l684_68494


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l684_68413

theorem certain_number_divisibility : ∃ (k : ℕ), k = 65 ∧ 
  (∀ (n : ℕ), n < 6 → ¬(k ∣ 11 * n - 1)) ∧ 
  (k ∣ 11 * 6 - 1) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l684_68413


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l684_68438

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : 
  let A : ℝ × ℝ := (x, y)
  A = A :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l684_68438


namespace NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l684_68466

/-- Given the conversion rates between knicks, knacks, and knocks, 
    prove that 36 knocks are equal to 40 knicks. -/
theorem knicks_knacks_knocks_conversion :
  (∀ (knicks knacks knocks : ℚ),
    5 * knicks = 3 * knacks →
    4 * knacks = 6 * knocks →
    36 * knocks = 40 * knicks) :=
by sorry

end NUMINAMATH_CALUDE_knicks_knacks_knocks_conversion_l684_68466


namespace NUMINAMATH_CALUDE_rectangle_folding_cutting_perimeter_ratio_l684_68450

theorem rectangle_folding_cutting_perimeter_ratio :
  let initial_length : ℚ := 6
  let initial_width : ℚ := 4
  let folded_length : ℚ := initial_length / 2
  let folded_width : ℚ := initial_width
  let cut_length : ℚ := folded_length
  let cut_width : ℚ := folded_width / 2
  let small_perimeter : ℚ := 2 * (cut_length + cut_width)
  let large_perimeter : ℚ := 2 * (folded_length + folded_width)
  small_perimeter / large_perimeter = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_folding_cutting_perimeter_ratio_l684_68450


namespace NUMINAMATH_CALUDE_apples_in_basket_A_l684_68433

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of mangoes in basket B -/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C -/
def peaches_in_C : ℕ := 20

/-- The number of pears in basket D -/
def pears_in_D : ℕ := 25

/-- The number of bananas in basket E -/
def bananas_in_E : ℕ := 35

/-- The number of apples in basket A -/
def apples_in_A : ℕ := num_baskets * avg_fruits_per_basket - (mangoes_in_B + peaches_in_C + pears_in_D + bananas_in_E)

theorem apples_in_basket_A : apples_in_A = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_A_l684_68433


namespace NUMINAMATH_CALUDE_water_remaining_l684_68468

theorem water_remaining (total : ℚ) (used : ℚ) (h1 : total = 3) (h2 : used = 4/3) :
  total - used = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l684_68468


namespace NUMINAMATH_CALUDE_piano_practice_minutes_l684_68448

theorem piano_practice_minutes (practice_time_6days : ℕ) (practice_time_2days : ℕ) 
  (total_days : ℕ) (average_minutes : ℕ) :
  practice_time_6days = 100 →
  practice_time_2days = 80 →
  total_days = 9 →
  average_minutes = 100 →
  (6 * practice_time_6days + 2 * practice_time_2days + 
   (average_minutes * total_days - (6 * practice_time_6days + 2 * practice_time_2days))) / total_days = average_minutes :=
by
  sorry

end NUMINAMATH_CALUDE_piano_practice_minutes_l684_68448


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l684_68426

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b)) ≤ (a + b + c) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l684_68426


namespace NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l684_68456

def prob_boy_or_girl : ℚ := 1 / 2

def family_size : ℕ := 4

theorem prob_at_least_one_boy_one_girl :
  1 - (prob_boy_or_girl ^ family_size + prob_boy_or_girl ^ family_size) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_boy_one_girl_l684_68456


namespace NUMINAMATH_CALUDE_mikes_hourly_wage_l684_68443

/-- Given Mike's total earnings, earnings from his first job, and hours worked at his second job,
    calculate his hourly wage at the second job. -/
theorem mikes_hourly_wage (total_earnings : ℚ) (first_job_earnings : ℚ) (second_job_hours : ℚ) 
    (h1 : total_earnings = 160)
    (h2 : first_job_earnings = 52)
    (h3 : second_job_hours = 12) :
    (total_earnings - first_job_earnings) / second_job_hours = 9 := by
  sorry

end NUMINAMATH_CALUDE_mikes_hourly_wage_l684_68443


namespace NUMINAMATH_CALUDE_solve_x_l684_68442

theorem solve_x : ∃ x : ℝ, (0.4 * x = (1 / 3) * x + 110) ∧ (x = 1650) := by sorry

end NUMINAMATH_CALUDE_solve_x_l684_68442


namespace NUMINAMATH_CALUDE_tensor_A_equals_result_l684_68495

def A : Set ℕ := {0, 2, 3}

def tensor_operation (S : Set ℕ) : Set ℕ :=
  {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensor_operation A = {0, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_tensor_A_equals_result_l684_68495


namespace NUMINAMATH_CALUDE_cloth_gain_theorem_l684_68404

/-- Represents the gain percentage as a rational number -/
def gainPercentage : ℚ := 200 / 3

/-- Represents the number of meters of cloth sold -/
def metersSold : ℕ := 25

/-- Calculates the number of meters of cloth's selling price gained -/
def metersGained (gainPercentage : ℚ) (metersSold : ℕ) : ℚ :=
  (gainPercentage / 100) * metersSold / (1 + gainPercentage / 100)

/-- Theorem stating that the number of meters of cloth's selling price gained is 10 -/
theorem cloth_gain_theorem :
  metersGained gainPercentage metersSold = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_gain_theorem_l684_68404


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l684_68490

/-- Given a quadratic function f(x) = -2x^2 + cx - 8, 
    where f(x) < 0 only when x ∈ (-∞, 2) ∪ (6, ∞),
    prove that c = 16 -/
theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x < 2 ∨ x > 6) → 
  c = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l684_68490


namespace NUMINAMATH_CALUDE_hockey_team_boys_percentage_l684_68403

theorem hockey_team_boys_percentage
  (total_players : ℕ)
  (junior_girls : ℕ)
  (h1 : total_players = 50)
  (h2 : junior_girls = 10)
  (h3 : junior_girls = total_players - junior_girls - (total_players - 2 * junior_girls)) :
  (total_players - 2 * junior_girls : ℚ) / total_players = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_boys_percentage_l684_68403


namespace NUMINAMATH_CALUDE_equal_numbers_sum_l684_68457

theorem equal_numbers_sum (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d + e = 45 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_sum_l684_68457


namespace NUMINAMATH_CALUDE_objects_meeting_probability_l684_68428

/-- The probability of two objects meeting in a coordinate plane --/
theorem objects_meeting_probability :
  let start_A : ℕ × ℕ := (0, 0)
  let start_B : ℕ × ℕ := (3, 5)
  let steps : ℕ := 5
  let prob_A_right : ℚ := 1/2
  let prob_A_up : ℚ := 1/2
  let prob_B_left : ℚ := 1/2
  let prob_B_down : ℚ := 1/2
  ∃ (meeting_prob : ℚ), meeting_prob = 31/128 := by
  sorry

end NUMINAMATH_CALUDE_objects_meeting_probability_l684_68428


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l684_68436

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l684_68436


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l684_68455

theorem no_infinite_sequence_exists : ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, 
  (a (n + 2) : ℝ) = (a (n + 1) : ℝ) + Real.sqrt ((a (n + 1) : ℝ) + (a n : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l684_68455


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_a_range_l684_68491

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part 2
theorem part2_a_range :
  ∀ a : ℝ, (∀ y : ℝ, -1 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, g a x = y) →
  (a ≤ 1 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_a_range_l684_68491


namespace NUMINAMATH_CALUDE_parabola_focus_l684_68401

/-- A parabola is defined by the equation x^2 = 4y -/
def is_parabola (f : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, f (x, y) ↔ x^2 = 4*y

/-- The focus of a parabola is a point (h, k) -/
def is_focus (f : ℝ × ℝ → Prop) (h k : ℝ) : Prop :=
  is_parabola f ∧ (h, k) = (0, 1)

/-- Theorem: The focus of the parabola x^2 = 4y is (0, 1) -/
theorem parabola_focus :
  ∀ f : ℝ × ℝ → Prop, is_parabola f → is_focus f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l684_68401


namespace NUMINAMATH_CALUDE_projectile_speed_proof_l684_68486

/-- Proves that the speed of the first projectile is 445 km/h given the problem conditions -/
theorem projectile_speed_proof (v : ℝ) : 
  (v + 545) * (84 / 60) = 1386 → v = 445 := by
  sorry

end NUMINAMATH_CALUDE_projectile_speed_proof_l684_68486


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l684_68488

theorem rectangular_solid_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 20)
  (h_front : front_area = 15)
  (h_bottom : bottom_area = 12) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l684_68488


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l684_68471

/-- The normal vector of a plane -/
structure NormalVector where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Dot product of two normal vectors -/
def dot_product (v1 v2 : NormalVector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

/-- Two planes are perpendicular if their normal vectors are orthogonal -/
def perpendicular (v1 v2 : NormalVector) : Prop :=
  dot_product v1 v2 = 0

theorem plane_perpendicular_condition (k : ℝ) :
  let α : NormalVector := ⟨3, 1, -2⟩
  let β : NormalVector := ⟨-1, 1, k⟩
  perpendicular α β → k = -1 :=
by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l684_68471


namespace NUMINAMATH_CALUDE_total_weight_chromic_acid_sodium_hydroxide_l684_68459

/-- The total weight of Chromic acid and Sodium hydroxide in a neutralization reaction -/
theorem total_weight_chromic_acid_sodium_hydroxide 
  (moles_chromic_acid : ℝ) 
  (moles_sodium_hydroxide : ℝ) 
  (molar_mass_chromic_acid : ℝ) 
  (molar_mass_sodium_hydroxide : ℝ) : 
  moles_chromic_acid = 17.3 →
  moles_sodium_hydroxide = 8.5 →
  molar_mass_chromic_acid = 118.02 →
  molar_mass_sodium_hydroxide = 40.00 →
  moles_chromic_acid * molar_mass_chromic_acid + 
  moles_sodium_hydroxide * molar_mass_sodium_hydroxide = 2381.746 := by
  sorry

#check total_weight_chromic_acid_sodium_hydroxide

end NUMINAMATH_CALUDE_total_weight_chromic_acid_sodium_hydroxide_l684_68459


namespace NUMINAMATH_CALUDE_inequality_proof_l684_68465

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log x < x ∧ x < Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l684_68465


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l684_68435

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 20*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 24*t^2 - 16*t = -96 - 8*t := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l684_68435


namespace NUMINAMATH_CALUDE_factory_assignment_l684_68469

-- Define the workers and machines
inductive Worker : Type
  | Dan : Worker
  | Emma : Worker
  | Fiona : Worker

inductive Machine : Type
  | A : Machine
  | B : Machine
  | C : Machine

-- Define the assignment of workers to machines
def Assignment := Worker → Machine

-- Define the conditions
def condition1 (a : Assignment) : Prop := a Worker.Emma ≠ Machine.A
def condition2 (a : Assignment) : Prop := a Worker.Dan = Machine.C
def condition3 (a : Assignment) : Prop := a Worker.Fiona = Machine.B

-- Define the correct assignment
def correct_assignment : Assignment :=
  fun w => match w with
    | Worker.Dan => Machine.C
    | Worker.Emma => Machine.A
    | Worker.Fiona => Machine.B

-- Theorem statement
theorem factory_assignment :
  ∀ (a : Assignment),
    (a Worker.Dan ≠ a Worker.Emma ∧ a Worker.Dan ≠ a Worker.Fiona ∧ a Worker.Emma ≠ a Worker.Fiona) →
    ((condition1 a ∧ ¬condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ ¬condition2 a ∧ condition3 a)) →
    a = correct_assignment :=
  sorry

end NUMINAMATH_CALUDE_factory_assignment_l684_68469


namespace NUMINAMATH_CALUDE_gcd_upper_bound_from_lcm_lower_bound_l684_68429

theorem gcd_upper_bound_from_lcm_lower_bound 
  (a b : ℕ) 
  (ha : a < 10^7) 
  (hb : b < 10^7) 
  (hlcm : 10^11 ≤ Nat.lcm a b) : 
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_from_lcm_lower_bound_l684_68429


namespace NUMINAMATH_CALUDE_puppy_weight_l684_68414

/-- Given the weights of animals satisfying certain conditions, prove the puppy's weight is √2 -/
theorem puppy_weight (p s l r : ℝ) 
  (h1 : p + s + l + r = 40)
  (h2 : p^2 + l^2 = 4*s)
  (h3 : p^2 + s^2 = l^2) :
  p = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l684_68414


namespace NUMINAMATH_CALUDE_sum_greater_than_6_is_random_event_l684_68497

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_sum_greater_than_6 (selection : List ℕ) : Bool :=
  selection.sum > 6

theorem sum_greater_than_6_is_random_event :
  ∃ (selection₁ selection₂ : List ℕ),
    selection₁.length = 3 ∧
    selection₂.length = 3 ∧
    (∀ n ∈ selection₁, n ∈ numbers) ∧
    (∀ n ∈ selection₂, n ∈ numbers) ∧
    is_sum_greater_than_6 selection₁ ∧
    ¬is_sum_greater_than_6 selection₂ :=
by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_6_is_random_event_l684_68497


namespace NUMINAMATH_CALUDE_special_number_theorem_l684_68444

/-- The type of positive integers with at least seven divisors -/
def HasAtLeastSevenDivisors (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ d₄ < d₅ ∧ d₅ < d₆ ∧ d₆ < d₇ ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧ d₇ ∣ n

/-- The property that n + 1 is equal to the sum of squares of its 6th and 7th divisors -/
def SumOfSquaresProperty (n : ℕ) : Prop :=
  ∃ (d₆ d₇ : ℕ), d₆ < d₇ ∧ d₆ ∣ n ∧ d₇ ∣ n ∧
    (∀ d : ℕ, d ∣ n → d < d₆ ∨ d = d₆ ∨ d = d₇ ∨ d₇ < d) ∧
    n + 1 = d₆^2 + d₇^2

theorem special_number_theorem (n : ℕ) 
  (h1 : HasAtLeastSevenDivisors n)
  (h2 : SumOfSquaresProperty n) :
  n = 144 ∨ n = 1984 :=
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l684_68444


namespace NUMINAMATH_CALUDE_common_divisors_90_105_l684_68416

theorem common_divisors_90_105 : Finset.card (Finset.filter (· ∣ 105) (Nat.divisors 90)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_90_105_l684_68416


namespace NUMINAMATH_CALUDE_reptile_insect_consumption_l684_68453

theorem reptile_insect_consumption :
  let num_geckos : ℕ := 5
  let num_lizards : ℕ := 3
  let num_chameleons : ℕ := 4
  let num_iguanas : ℕ := 2
  let gecko_consumption : ℕ := 6
  let lizard_consumption : ℝ := 2 * gecko_consumption
  let chameleon_consumption : ℝ := 3.5 * gecko_consumption
  let iguana_consumption : ℝ := 0.75 * gecko_consumption
  
  (num_geckos * gecko_consumption : ℝ) +
  (num_lizards : ℝ) * lizard_consumption +
  (num_chameleons : ℝ) * chameleon_consumption +
  (num_iguanas : ℝ) * iguana_consumption = 159
  := by sorry

end NUMINAMATH_CALUDE_reptile_insect_consumption_l684_68453


namespace NUMINAMATH_CALUDE_positive_integer_division_problem_l684_68412

theorem positive_integer_division_problem (a b : ℕ) : 
  a > 1 → b > 1 → (∃k : ℕ, b + 1 = k * a) → (∃l : ℕ, a^3 - 1 = l * b) →
  ((b = a - 1) ∨ (∃p : ℕ, p = 1 ∨ p = 2 ∧ a = a^p ∧ b = a^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_division_problem_l684_68412


namespace NUMINAMATH_CALUDE_grid_arrangement_theorem_l684_68492

/-- A type representing the grid arrangement of digits -/
def GridArrangement := Fin 8 → Fin 9

/-- Function to check if a three-digit number is a multiple of k -/
def isMultipleOfK (n : ℕ) (k : ℕ) : Prop :=
  n % k = 0

/-- Function to extract a three-digit number from the grid -/
def extractNumber (g : GridArrangement) (start : Fin 8) : ℕ :=
  100 * (g start).val + 10 * (g ((start + 2) % 8)).val + (g ((start + 4) % 8)).val

/-- Predicate to check if all four numbers in the grid are multiples of k -/
def allMultiplesOfK (g : GridArrangement) (k : ℕ) : Prop :=
  ∀ i : Fin 4, isMultipleOfK (extractNumber g (2 * i)) k

/-- Predicate to check if a grid arrangement is valid (uses all digits 1 to 8 once) -/
def isValidArrangement (g : GridArrangement) : Prop :=
  ∀ i j : Fin 8, i ≠ j → g i ≠ g j

/-- The main theorem stating for which values of k a valid arrangement exists -/
theorem grid_arrangement_theorem :
  ∀ k : ℕ, 2 ≤ k → k ≤ 6 →
    (∃ g : GridArrangement, isValidArrangement g ∧ allMultiplesOfK g k) ↔ (k = 2 ∨ k = 3) :=
sorry

end NUMINAMATH_CALUDE_grid_arrangement_theorem_l684_68492


namespace NUMINAMATH_CALUDE_diet_soda_ratio_l684_68454

theorem diet_soda_ratio (total bottles : ℕ) (regular_soda diet_soda fruit_juice sparkling_water : ℕ) :
  total = 60 →
  regular_soda = 18 →
  diet_soda = 14 →
  fruit_juice = 8 →
  sparkling_water = 10 →
  total = regular_soda + diet_soda + fruit_juice + sparkling_water + (total - regular_soda - diet_soda - fruit_juice - sparkling_water) →
  (diet_soda : ℚ) / total = 7 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_diet_soda_ratio_l684_68454


namespace NUMINAMATH_CALUDE_product_35_42_base7_l684_68418

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Main theorem --/
theorem product_35_42_base7 :
  let a := toDecimal 35
  let b := toDecimal 42
  let product := a * b
  let base7Product := toBase7 product
  let digitSum := sumOfDigitsBase7 base7Product
  digitSum = 5 ∧ digitSum % 5 = 5 := by sorry

end NUMINAMATH_CALUDE_product_35_42_base7_l684_68418


namespace NUMINAMATH_CALUDE_initial_ball_count_l684_68439

theorem initial_ball_count (initial_blue : ℕ) (removed_blue : ℕ) (final_probability : ℚ) : 
  initial_blue = 7 → 
  removed_blue = 3 → 
  final_probability = 1/3 → 
  ∃ (total : ℕ), total = 15 ∧ 
    (initial_blue - removed_blue : ℚ) / (total - removed_blue : ℚ) = final_probability :=
by sorry

end NUMINAMATH_CALUDE_initial_ball_count_l684_68439


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l684_68405

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^5 + b^5 + c^5 = 98/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l684_68405


namespace NUMINAMATH_CALUDE_possible_values_of_a_l684_68463

def P : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x + 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, Q a ⊆ P → a = 0 ∨ a = -1/2 ∨ a = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l684_68463


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l684_68417

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 64 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l684_68417


namespace NUMINAMATH_CALUDE_dodecahedron_triangle_count_l684_68474

/-- The number of vertices in a regular dodecahedron -/
def dodecahedron_vertices : ℕ := 12

/-- The number of distinct triangles that can be formed by connecting three
    different vertices of a regular dodecahedron -/
def dodecahedron_triangles : ℕ := Nat.choose dodecahedron_vertices 3

theorem dodecahedron_triangle_count :
  dodecahedron_triangles = 220 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangle_count_l684_68474


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l684_68499

/-- Represents the number of ways to make 50 cents using 5 cent and 7 cent stamps -/
def stamp_combinations : Set (ℕ × ℕ) :=
  {(s, t) | 5 * s + 7 * t = 50 ∧ s ≥ 0 ∧ t ≥ 0}

/-- The total number of stamps used in a combination -/
def total_stamps (combination : ℕ × ℕ) : ℕ :=
  combination.1 + combination.2

theorem min_stamps_for_50_cents :
  ∃ (combination : ℕ × ℕ),
    combination ∈ stamp_combinations ∧
    (∀ other ∈ stamp_combinations, total_stamps combination ≤ total_stamps other) ∧
    total_stamps combination = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l684_68499


namespace NUMINAMATH_CALUDE_max_pairs_sum_bound_l684_68422

theorem max_pairs_sum_bound (k : ℕ) 
  (pairs : Fin k → (ℕ × ℕ))
  (h_range : ∀ i, (pairs i).1 ∈ Finset.range 3000 ∧ (pairs i).2 ∈ Finset.range 3000)
  (h_order : ∀ i, (pairs i).1 < (pairs i).2)
  (h_distinct : ∀ i j, i ≠ j → (pairs i).1 ≠ (pairs j).1 ∧ (pairs i).1 ≠ (pairs j).2 ∧
                                (pairs i).2 ≠ (pairs j).1 ∧ (pairs i).2 ≠ (pairs j).2)
  (h_sum_distinct : ∀ i j, i ≠ j → (pairs i).1 + (pairs i).2 ≠ (pairs j).1 + (pairs j).2)
  (h_sum_bound : ∀ i, (pairs i).1 + (pairs i).2 ≤ 4000) :
  k ≤ 1599 :=
sorry

end NUMINAMATH_CALUDE_max_pairs_sum_bound_l684_68422


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l684_68415

theorem simplify_fraction_multiplication :
  (175 : ℚ) / 1225 * 25 = 25 / 7 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l684_68415


namespace NUMINAMATH_CALUDE_total_weight_CaI2_is_1469_4_l684_68462

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of moles of calcium iodide -/
def moles_CaI2 : ℝ := 5

/-- The molecular weight of calcium iodide (CaI2) in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The total weight of calcium iodide in grams -/
def total_weight_CaI2 : ℝ := moles_CaI2 * molecular_weight_CaI2

theorem total_weight_CaI2_is_1469_4 :
  total_weight_CaI2 = 1469.4 := by sorry

end NUMINAMATH_CALUDE_total_weight_CaI2_is_1469_4_l684_68462


namespace NUMINAMATH_CALUDE_students_not_enrolled_l684_68473

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l684_68473


namespace NUMINAMATH_CALUDE_teacher_arrangements_eq_144_l684_68410

/-- The number of ways to arrange 6 teachers (3 math, 2 English, 1 Chinese) such that the 3 math teachers are not adjacent -/
def teacher_arrangements : ℕ :=
  Nat.factorial 3 * (Nat.factorial 3 * Nat.choose 4 3)

theorem teacher_arrangements_eq_144 : teacher_arrangements = 144 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangements_eq_144_l684_68410


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l684_68481

/-- Given three non-overlapping circles with radii r₁, r₂, r₃ where r₁ > r₂ and r₁ > r₃,
    the quadrilateral formed by their external common tangents has an inscribed circle
    with radius r = (r₁r₂r₃) / (r₁r₂ - r₁r₃ - r₂r₃) -/
theorem inscribed_circle_radius
  (r₁ r₂ r₃ : ℝ)
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h₄ : r₁ > r₂) (h₅ : r₁ > r₃)
  (h_non_overlap : r₁ < r₂ + r₃) :
  ∃ r : ℝ, r > 0 ∧ r = (r₁ * r₂ * r₃) / (r₁ * r₂ - r₁ * r₃ - r₂ * r₃) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l684_68481


namespace NUMINAMATH_CALUDE_total_travel_time_l684_68427

/-- Prove that the total time traveled is 4 hours -/
theorem total_travel_time (speed : ℝ) (distance_AB : ℝ) (h1 : speed = 60) (h2 : distance_AB = 120) :
  2 * distance_AB / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_l684_68427
