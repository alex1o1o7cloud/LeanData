import Mathlib

namespace NUMINAMATH_CALUDE_f_is_even_l1683_168391

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l1683_168391


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l1683_168380

theorem pokemon_card_ratio (mark_cards lloyd_cards michael_cards : ℕ) : 
  mark_cards = lloyd_cards →
  mark_cards = michael_cards - 10 →
  michael_cards = 100 →
  mark_cards + lloyd_cards + michael_cards + 80 = 300 →
  mark_cards = lloyd_cards :=
by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l1683_168380


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1683_168365

/-- The standard equation of a parabola with focus (2,0) is y^2 = 8x -/
theorem parabola_standard_equation (F : ℝ × ℝ) (h : F = (2, 0)) :
  ∃ (f : ℝ → ℝ), (∀ x y : ℝ, f y = 8*x ↔ y^2 = 8*x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1683_168365


namespace NUMINAMATH_CALUDE_tims_garden_carrots_l1683_168396

/-- Represents the number of carrots in Tim's garden -/
def carrots : ℕ := sorry

/-- Represents the number of potatoes in Tim's garden -/
def potatoes : ℕ := sorry

/-- The ratio of carrots to potatoes -/
def ratio : Rat := 3 / 4

/-- The initial number of potatoes -/
def initial_potatoes : ℕ := 32

/-- The number of potatoes added -/
def added_potatoes : ℕ := 28

theorem tims_garden_carrots : 
  (ratio = carrots / potatoes) → 
  (potatoes = initial_potatoes + added_potatoes) →
  carrots = 45 := by sorry

end NUMINAMATH_CALUDE_tims_garden_carrots_l1683_168396


namespace NUMINAMATH_CALUDE_all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l1683_168326

/-- Represents the numbers written by three schoolchildren on their notebooks. -/
structure SchoolchildrenNumbers where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Performs one iteration of the number writing process. -/
def iterate (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  { a := nums.c - nums.b
  , b := nums.a - nums.c
  , c := nums.b - nums.a }

/-- Performs n iterations of the number writing process. -/
def iterateN (n : ℕ) (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  match n with
  | 0 => nums
  | n + 1 => iterate (iterateN n nums)

/-- Theorem stating that after 4 iterations, all numbers are multiples of 3. -/
theorem all_multiples_of_three_after_four_iterations (initial : SchoolchildrenNumbers) :
  ∃ k l m : ℤ, 
    let result := iterateN 4 initial
    result.a = 3 * k ∧ result.b = 3 * l ∧ result.c = 3 * m :=
  sorry

/-- Theorem stating that 2020 cannot appear on the 10th page. -/
theorem no_2020_on_tenth_page (initial : SchoolchildrenNumbers) :
  let result := iterateN 9 initial
  result.a ≠ 2020 ∧ result.b ≠ 2020 ∧ result.c ≠ 2020 :=
  sorry

end NUMINAMATH_CALUDE_all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l1683_168326


namespace NUMINAMATH_CALUDE_equation_solutions_l1683_168378

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 1 = 8 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (x + 4)^3 = -64 ↔ x = -8) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1683_168378


namespace NUMINAMATH_CALUDE_unique_k_is_zero_l1683_168329

/-- A function f: ℕ → ℕ satisfying f^n(n) = n + k for all n ∈ ℕ, where k is a non-negative integer -/
def SatisfiesCondition (f : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, (f^[n] n) = n + k

/-- Theorem stating that if a function satisfies the condition, then k must be 0 -/
theorem unique_k_is_zero (f : ℕ → ℕ) (k : ℕ) (h : SatisfiesCondition f k) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_is_zero_l1683_168329


namespace NUMINAMATH_CALUDE_yogurt_shop_combinations_l1683_168364

/-- The number of combinations of one flavor from n flavors and two different toppings from m toppings -/
def yogurt_combinations (n m : ℕ) : ℕ :=
  n * (m.choose 2)

/-- Theorem: There are 105 combinations of one flavor from 5 flavors and two different toppings from 7 toppings -/
theorem yogurt_shop_combinations :
  yogurt_combinations 5 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_shop_combinations_l1683_168364


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1683_168373

/-- Represents the configuration of rectangles around a quadrilateral -/
structure RectangleConfiguration where
  s : ℝ  -- side length of the inner quadrilateral
  x : ℝ  -- shorter side of each rectangle
  y : ℝ  -- longer side of each rectangle
  h1 : s > 0  -- side length is positive
  h2 : x > 0  -- rectangle sides are positive
  h3 : y > 0
  h4 : (s + 2*x)^2 = 4*s^2  -- area relation
  h5 : s + 2*y = 2*s  -- relation for y sides

/-- The ratio of y to x is 1 in the given configuration -/
theorem rectangle_ratio (config : RectangleConfiguration) : config.y / config.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1683_168373


namespace NUMINAMATH_CALUDE_resultant_profit_is_four_percent_l1683_168345

/-- Calculates the resultant profit percentage when an item is sold twice -/
def resultantProfitPercentage (firstProfit : Real) (secondLoss : Real) : Real :=
  let firstSalePrice := 1 + firstProfit
  let secondSalePrice := firstSalePrice * (1 - secondLoss)
  (secondSalePrice - 1) * 100

/-- Theorem: The resultant profit percentage when an item is sold with 30% profit
    and then resold with 20% loss is 4% -/
theorem resultant_profit_is_four_percent :
  resultantProfitPercentage 0.3 0.2 = 4 := by sorry

end NUMINAMATH_CALUDE_resultant_profit_is_four_percent_l1683_168345


namespace NUMINAMATH_CALUDE_solution_set_a_3_range_of_a_non_negative_l1683_168334

-- Define the function f
def f (a x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

-- Theorem 1: Solution set when a = 3
theorem solution_set_a_3 :
  {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Theorem 2: Range of a for f(x) ≥ 0 for all x
theorem range_of_a_non_negative :
  {a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_3_range_of_a_non_negative_l1683_168334


namespace NUMINAMATH_CALUDE_equation_describes_ellipse_l1683_168370

def is_ellipse (f₁ f₂ : ℝ × ℝ) (c : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
               Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = c

theorem equation_describes_ellipse :
  is_ellipse (0, 2) (6, -4) 12 := by sorry

end NUMINAMATH_CALUDE_equation_describes_ellipse_l1683_168370


namespace NUMINAMATH_CALUDE_notebook_final_price_l1683_168354

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

def price_after_first_discount : ℝ := initial_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

theorem notebook_final_price : final_price = 9 := by
  sorry

end NUMINAMATH_CALUDE_notebook_final_price_l1683_168354


namespace NUMINAMATH_CALUDE_juggling_contest_winner_l1683_168375

/-- Represents the number of rotations for an object over 4 minutes -/
structure Rotations :=
  (minute1 : ℕ) (minute2 : ℕ) (minute3 : ℕ) (minute4 : ℕ)

/-- Calculates the total rotations for a contestant -/
def totalRotations (obj1Count : ℕ) (obj1Rotations : Rotations) 
                   (obj2Count : ℕ) (obj2Rotations : Rotations) : ℕ :=
  obj1Count * (obj1Rotations.minute1 + obj1Rotations.minute2 + obj1Rotations.minute3 + obj1Rotations.minute4) +
  obj2Count * (obj2Rotations.minute1 + obj2Rotations.minute2 + obj2Rotations.minute3 + obj2Rotations.minute4)

theorem juggling_contest_winner (tobyBaseballs : Rotations) (tobyFrisbees : Rotations)
                                (annaApples : Rotations) (annaOranges : Rotations)
                                (jackTennisBalls : Rotations) (jackWaterBalloons : Rotations) :
  tobyBaseballs = ⟨80, 85, 75, 90⟩ →
  tobyFrisbees = ⟨60, 70, 65, 80⟩ →
  annaApples = ⟨101, 99, 98, 102⟩ →
  annaOranges = ⟨95, 90, 92, 93⟩ →
  jackTennisBalls = ⟨82, 81, 85, 87⟩ →
  jackWaterBalloons = ⟨100, 96, 101, 97⟩ →
  (max (totalRotations 5 tobyBaseballs 3 tobyFrisbees)
       (max (totalRotations 4 annaApples 5 annaOranges)
            (totalRotations 6 jackTennisBalls 4 jackWaterBalloons))) = 3586 := by
  sorry

end NUMINAMATH_CALUDE_juggling_contest_winner_l1683_168375


namespace NUMINAMATH_CALUDE_increasing_function_property_l1683_168341

-- Define a function f on positive real numbers
variable (f : ℝ → ℝ)

-- Define the property of being increasing for positive real numbers
def IncreasingOnPositive (g : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → g x < g y

-- State the theorem
theorem increasing_function_property
  (h1 : IncreasingOnPositive (fun x => f x - x))
  (h2 : IncreasingOnPositive (fun x => f (x^2) - x^6)) :
  IncreasingOnPositive (fun x => f (x^3) - (Real.sqrt 3 / 2) * x^6) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_property_l1683_168341


namespace NUMINAMATH_CALUDE_hyperbola_foci_l1683_168312

/-- The hyperbola equation --/
def hyperbola_eq (x y : ℝ) : Prop := y^2 - x^2/3 = 1

/-- The focus coordinates --/
def focus_coords : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

/-- Theorem: The given coordinates are the foci of the hyperbola --/
theorem hyperbola_foci : 
  ∀ (x y : ℝ), hyperbola_eq x y ↔ ∃ (f : ℝ × ℝ), f ∈ focus_coords ∧ 
    (x - f.1)^2 + (y - f.2)^2 = ((x + f.1)^2 + (y + f.2)^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l1683_168312


namespace NUMINAMATH_CALUDE_woodworker_tables_l1683_168332

/-- Proves the number of tables made by a woodworker given the total number of furniture legs and chairs made -/
theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) : 
  total_legs = 40 → 
  chairs = 6 → 
  ∃ (tables : ℕ), 
    tables * 4 + chairs * 4 = total_legs ∧ 
    tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l1683_168332


namespace NUMINAMATH_CALUDE_jill_jane_doll_difference_l1683_168301

theorem jill_jane_doll_difference (total : ℕ) (jane : ℕ) (jill : ℕ) 
  (h1 : total = 32)
  (h2 : jane = 13)
  (h3 : total = jane + jill)
  (h4 : jill > jane) :
  jill - jane = 6 := by
  sorry

end NUMINAMATH_CALUDE_jill_jane_doll_difference_l1683_168301


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocal_l1683_168319

theorem arithmetic_mean_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  b = (a + c) / 2 →
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) →
  (∃ x : ℝ, x ≠ 0 ∧ (a = x ∧ b = x ∧ c = x ∨ a = -4*x ∧ b = -x ∧ c = 2*x)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocal_l1683_168319


namespace NUMINAMATH_CALUDE_square_root_of_36_l1683_168347

theorem square_root_of_36 : ∃ (x : ℝ), x^2 = 36 ↔ x = 6 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_36_l1683_168347


namespace NUMINAMATH_CALUDE_exists_fib_divisible_by_2007_l1683_168362

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem exists_fib_divisible_by_2007 : ∃ n : ℕ, n > 0 ∧ 2007 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_divisible_by_2007_l1683_168362


namespace NUMINAMATH_CALUDE_total_cows_is_570_l1683_168383

/-- The number of cows owned by Matthews -/
def matthews_cows : ℕ := 60

/-- The number of cows owned by Aaron -/
def aaron_cows : ℕ := 4 * matthews_cows

/-- The number of cows owned by Marovich -/
def marovich_cows : ℕ := aaron_cows + matthews_cows - 30

/-- The total number of cows owned by all three -/
def total_cows : ℕ := aaron_cows + matthews_cows + marovich_cows

theorem total_cows_is_570 : total_cows = 570 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_is_570_l1683_168383


namespace NUMINAMATH_CALUDE_max_books_purchasable_l1683_168379

theorem max_books_purchasable (book_price : ℚ) (budget : ℚ) : 
  book_price = 15 → budget = 200 → 
    ↑(⌊budget / book_price⌋) = (13 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_max_books_purchasable_l1683_168379


namespace NUMINAMATH_CALUDE_perpendicular_sum_limit_l1683_168320

/-- Given two distinct straight lines and alternating perpendiculars between them,
    prove that the sum of perpendicular lengths converges to a²/(a - b) -/
theorem perpendicular_sum_limit (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let r := b / a
  ∃ (S : ℝ), (∑' n, a * r^n) = S ∧ S = a^2 / (a - b) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_limit_l1683_168320


namespace NUMINAMATH_CALUDE_hot_dog_bun_distribution_l1683_168339

/-- Hot dog bun distribution problem -/
theorem hot_dog_bun_distribution
  (buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30) :
  (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 :=
sorry

end NUMINAMATH_CALUDE_hot_dog_bun_distribution_l1683_168339


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1683_168307

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1683_168307


namespace NUMINAMATH_CALUDE_probability_product_eight_l1683_168331

/-- A standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sample space of rolling a standard 6-sided die twice -/
def TwoRollsSampleSpace : Finset (ℕ × ℕ) :=
  (StandardDie.product StandardDie)

/-- The favorable outcomes where the product of two rolls is 8 -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  {(2, 4), (4, 2)}

/-- Probability of the product of two rolls being 8 -/
theorem probability_product_eight :
  (FavorableOutcomes.card : ℚ) / TwoRollsSampleSpace.card = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_probability_product_eight_l1683_168331


namespace NUMINAMATH_CALUDE_coach_rental_equation_l1683_168387

/-- Represents the equation for renting coaches to transport a group of people -/
theorem coach_rental_equation (total_people : ℕ) (school_bus_capacity : ℕ) (coach_capacity : ℕ) (x : ℕ) :
  total_people = 328 →
  school_bus_capacity = 64 →
  coach_capacity = 44 →
  44 * x + 64 = 328 :=
by sorry

end NUMINAMATH_CALUDE_coach_rental_equation_l1683_168387


namespace NUMINAMATH_CALUDE_probability_of_two_as_median_l1683_168393

def S : Finset ℕ := {2, 0, 1, 5}

def is_median (a b c : ℕ) : Prop :=
  (a ≤ b ∧ b ≤ c) ∨ (c ≤ b ∧ b ≤ a)

def favorable_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 2, 5), (1, 2, 5)}

def total_outcomes : Finset (ℕ × ℕ × ℕ) :=
  {(0, 1, 2), (0, 1, 5), (0, 2, 5), (1, 2, 5)}

theorem probability_of_two_as_median :
  (Finset.card favorable_outcomes : ℚ) / (Finset.card total_outcomes : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_as_median_l1683_168393


namespace NUMINAMATH_CALUDE_last_digit_theorem_l1683_168397

-- Define the property for the last digit of powers
def last_digit_property (a n k : ℕ) : Prop :=
  a^(4*n + k) % 10 = a^k % 10

-- Define the sum of specific powers
def sum_of_powers : ℕ :=
  (2^1997 + 3^1997 + 7^1997 + 9^1997) % 10

-- Theorem statement
theorem last_digit_theorem :
  (∀ (a n k : ℕ), last_digit_property a n k) ∧
  sum_of_powers = 1 := by
sorry

end NUMINAMATH_CALUDE_last_digit_theorem_l1683_168397


namespace NUMINAMATH_CALUDE_exercise_book_problem_l1683_168309

theorem exercise_book_problem :
  ∀ (x y : ℕ),
    x + y = 100 →
    2 * x + 4 * y = 250 →
    x = 75 ∧ y = 25 :=
by sorry

end NUMINAMATH_CALUDE_exercise_book_problem_l1683_168309


namespace NUMINAMATH_CALUDE_min_sum_and_inequality_range_l1683_168300

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 4 * a + b = a * b

-- Define the minimum value of a + b
def min_sum (a b : ℝ) : ℝ := a + b

-- Define the inequality condition
def inequality_condition (a b t : ℝ) : Prop :=
  ∀ x : ℝ, |x - a| + |x - b| ≥ t^2 - 2*t

-- Theorem statement
theorem min_sum_and_inequality_range :
  ∃ a b : ℝ, conditions a b ∧
    (∀ a' b' : ℝ, conditions a' b' → min_sum a b ≤ min_sum a' b') ∧
    min_sum a b = 9 ∧
    (∀ t : ℝ, inequality_condition a b t ↔ -1 ≤ t ∧ t ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_and_inequality_range_l1683_168300


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1683_168335

theorem inverse_proportion_ratio (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ ≠ 0) (h2 : a₂ ≠ 0) (h3 : b₁ ≠ 0) (h4 : b₂ ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a₁ * b₁ = k ∧ a₂ * b₂ = k) →
  a₁ / a₂ = 3 / 4 →
  b₁ / b₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1683_168335


namespace NUMINAMATH_CALUDE_unique_division_solution_l1683_168360

theorem unique_division_solution :
  ∀ (dividend divisor quotient : ℕ),
    divisor ≥ 100 ∧ divisor < 1000 →
    quotient ≥ 10000 ∧ quotient < 100000 →
    (quotient / 1000) % 10 = 7 →
    dividend = divisor * quotient →
    (dividend, divisor, quotient) = (12128316, 124, 97809) := by
  sorry

end NUMINAMATH_CALUDE_unique_division_solution_l1683_168360


namespace NUMINAMATH_CALUDE_subtraction_result_l1683_168357

theorem subtraction_result : 34.256 - 12.932 - 1.324 = 20 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l1683_168357


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_one_l1683_168315

theorem no_solution_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) + (2 + m*x)/(3 - x) ≠ -1) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_neg_one_l1683_168315


namespace NUMINAMATH_CALUDE_coach_spending_difference_l1683_168392

-- Define the purchases and discounts for each coach
def coach_A_basketballs : Nat := 10
def coach_A_basketball_price : ℝ := 29
def coach_A_soccer_balls : Nat := 5
def coach_A_soccer_ball_price : ℝ := 15
def coach_A_discount : ℝ := 0.05

def coach_B_baseballs : Nat := 14
def coach_B_baseball_price : ℝ := 2.50
def coach_B_baseball_bats : Nat := 1
def coach_B_baseball_bat_price : ℝ := 18
def coach_B_hockey_sticks : Nat := 4
def coach_B_hockey_stick_price : ℝ := 25
def coach_B_hockey_masks : Nat := 1
def coach_B_hockey_mask_price : ℝ := 72
def coach_B_discount : ℝ := 0.10

def coach_C_volleyball_nets : Nat := 8
def coach_C_volleyball_net_price : ℝ := 32
def coach_C_volleyballs : Nat := 12
def coach_C_volleyball_price : ℝ := 12
def coach_C_discount : ℝ := 0.07

-- Define the theorem
theorem coach_spending_difference :
  let coach_A_total := (1 - coach_A_discount) * (coach_A_basketballs * coach_A_basketball_price + coach_A_soccer_balls * coach_A_soccer_ball_price)
  let coach_B_total := (1 - coach_B_discount) * (coach_B_baseballs * coach_B_baseball_price + coach_B_baseball_bats * coach_B_baseball_bat_price + coach_B_hockey_sticks * coach_B_hockey_stick_price + coach_B_hockey_masks * coach_B_hockey_mask_price)
  let coach_C_total := (1 - coach_C_discount) * (coach_C_volleyball_nets * coach_C_volleyball_net_price + coach_C_volleyballs * coach_C_volleyball_price)
  coach_A_total - (coach_B_total + coach_C_total) = -227.75 := by
  sorry

end NUMINAMATH_CALUDE_coach_spending_difference_l1683_168392


namespace NUMINAMATH_CALUDE_congruence_conditions_and_smallest_n_l1683_168361

theorem congruence_conditions_and_smallest_n :
  ∀ (r s : ℕ+),
  (2^(r : ℕ) - 16^(s : ℕ)) % 7 = 5 →
  (r : ℕ) % 3 = 1 ∧ (s : ℕ) % 3 = 2 ∧
  (∀ (r' s' : ℕ+),
    (2^(r' : ℕ) - 16^(s' : ℕ)) % 7 = 5 →
    2^(r : ℕ) - 16^(s : ℕ) ≤ 2^(r' : ℕ) - 16^(s' : ℕ)) ∧
  2^(r : ℕ) - 16^(s : ℕ) = 768 :=
by sorry

end NUMINAMATH_CALUDE_congruence_conditions_and_smallest_n_l1683_168361


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1683_168352

/-- Given a dairy farm scenario where multiple cows eat multiple bags of husk over multiple days,
    this theorem proves that the number of days for one cow to eat one bag is the same as the
    total number of days for all cows to eat all bags. -/
theorem dairy_farm_husk_consumption
  (num_cows : ℕ)
  (num_bags : ℕ)
  (num_days : ℕ)
  (h_cows : num_cows = 46)
  (h_bags : num_bags = 46)
  (h_days : num_days = 46)
  : num_days = (num_days * num_cows) / num_cows :=
by
  sorry

#check dairy_farm_husk_consumption

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1683_168352


namespace NUMINAMATH_CALUDE_negation_of_forall_is_exists_not_l1683_168346

variable (S : Set ℝ)

-- Define the original property
def P (x : ℝ) : Prop := x^2 = x

-- State the theorem
theorem negation_of_forall_is_exists_not (h : ∀ x ∈ S, P x) : 
  ¬(∀ x ∈ S, P x) ↔ ∃ x ∈ S, ¬(P x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_is_exists_not_l1683_168346


namespace NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1683_168388

theorem quadratic_trinomial_decomposition (a b c : ℝ) :
  ∃ (p q r s t u : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = (p * x^2 + q * x + r) + (s * x^2 + t * x + u)) ∧
    (q^2 - 4*p*r = 0) ∧
    (t^2 - 4*s*u = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_decomposition_l1683_168388


namespace NUMINAMATH_CALUDE_product_sum_6545_l1683_168351

theorem product_sum_6545 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6545 ∧ 
  a + b = 162 := by
sorry

end NUMINAMATH_CALUDE_product_sum_6545_l1683_168351


namespace NUMINAMATH_CALUDE_vincent_outer_space_books_l1683_168348

/-- The number of books about outer space Vincent bought -/
def outer_space_books : ℕ := 1

/-- The number of books about animals Vincent bought -/
def animal_books : ℕ := 10

/-- The number of books about trains Vincent bought -/
def train_books : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 16

/-- The total amount spent on books in dollars -/
def total_spent : ℕ := 224

theorem vincent_outer_space_books :
  outer_space_books = 1 ∧
  animal_books * book_cost + outer_space_books * book_cost + train_books * book_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_vincent_outer_space_books_l1683_168348


namespace NUMINAMATH_CALUDE_playerB_is_best_choice_l1683_168302

-- Define a structure for a player
structure Player where
  name : String
  average : Float
  variance : Float

-- Define the players
def playerA : Player := { name := "A", average := 9.2, variance := 3.6 }
def playerB : Player := { name := "B", average := 9.5, variance := 3.6 }
def playerC : Player := { name := "C", average := 9.5, variance := 7.4 }
def playerD : Player := { name := "D", average := 9.2, variance := 8.1 }

def players : List Player := [playerA, playerB, playerC, playerD]

-- Function to determine if a player is the best choice
def isBestChoice (p : Player) (players : List Player) : Prop :=
  (∀ q ∈ players, p.average ≥ q.average) ∧
  (∀ q ∈ players, p.variance ≤ q.variance) ∧
  (∃ q ∈ players, p.average > q.average ∨ p.variance < q.variance)

-- Theorem stating that playerB is the best choice
theorem playerB_is_best_choice : isBestChoice playerB players := by
  sorry

end NUMINAMATH_CALUDE_playerB_is_best_choice_l1683_168302


namespace NUMINAMATH_CALUDE_candy_difference_l1683_168325

/- Define the number of candies each person can eat -/
def nellie_candies : ℕ := 12
def jacob_candies : ℕ := nellie_candies / 2
def lana_candies : ℕ := jacob_candies - 3

/- Define the total number of candies in the bucket -/
def total_candies : ℕ := 30

/- Define the number of remaining candies after they ate -/
def remaining_candies : ℕ := 9

/- Theorem statement -/
theorem candy_difference :
  jacob_candies - lana_candies = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_difference_l1683_168325


namespace NUMINAMATH_CALUDE_remainder_theorem_l1683_168343

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 25 * k - 1) :
  (n^2 + 3*n + 5) % 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1683_168343


namespace NUMINAMATH_CALUDE_andrews_cookie_expenditure_l1683_168358

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew buys each day -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 15

/-- The total amount Andrew spent on cookies in May -/
def total_spent : ℕ := days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem stating that Andrew spent 1395 dollars on cookies in May -/
theorem andrews_cookie_expenditure : total_spent = 1395 := by
  sorry

end NUMINAMATH_CALUDE_andrews_cookie_expenditure_l1683_168358


namespace NUMINAMATH_CALUDE_simple_interest_rate_interest_rate_problem_l1683_168350

/-- Simple interest calculation --/
theorem simple_interest_rate (principal amount : ℚ) (time : ℕ) (rate : ℚ) : 
  principal * (1 + rate * time) = amount →
  rate = (amount - principal) / (principal * time) :=
by
  sorry

/-- Prove that the interest rate is 5% given the problem conditions --/
theorem interest_rate_problem :
  let principal : ℚ := 600
  let amount : ℚ := 720
  let time : ℕ := 4
  let rate : ℚ := (amount - principal) / (principal * time)
  rate = 5 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_interest_rate_problem_l1683_168350


namespace NUMINAMATH_CALUDE_probability_two_white_balls_l1683_168327

def total_balls : ℕ := 7 + 8
def white_balls : ℕ := 7
def black_balls : ℕ := 8

theorem probability_two_white_balls :
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_l1683_168327


namespace NUMINAMATH_CALUDE_badge_exchange_l1683_168305

theorem badge_exchange (vasya_initial : ℕ) (tolya_initial : ℕ) : 
  (vasya_initial = tolya_initial + 5) →
  (vasya_initial - (24 * vasya_initial) / 100 + (20 * tolya_initial) / 100 = 
   tolya_initial - (20 * tolya_initial) / 100 + (24 * vasya_initial) / 100 - 1) →
  (vasya_initial = 50 ∧ tolya_initial = 45) :=
by sorry

#check badge_exchange

end NUMINAMATH_CALUDE_badge_exchange_l1683_168305


namespace NUMINAMATH_CALUDE_line_slope_and_inclination_l1683_168385

/-- Given a line l passing through points A(1,2) and B(4, 2+√3), 
    prove its slope and angle of inclination. -/
theorem line_slope_and_inclination :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (4, 2 + Real.sqrt 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  let angle := Real.arctan slope
  slope = Real.sqrt 3 / 3 ∧ angle = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_line_slope_and_inclination_l1683_168385


namespace NUMINAMATH_CALUDE_rectangular_field_dimension_l1683_168321

theorem rectangular_field_dimension (m : ℝ) : ∃! m : ℝ, (3*m + 5)*(m - 1) = 104 ∧ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimension_l1683_168321


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l1683_168371

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time (lawn_length lawn_width swath_width overlap mowing_rate : ℝ) :
  lawn_length = 120 →
  lawn_width = 180 →
  swath_width = 30 / 12 →
  overlap = 6 / 12 →
  mowing_rate = 4000 →
  (lawn_width / (swath_width - overlap) * lawn_length) / mowing_rate = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l1683_168371


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1683_168314

def a (n : ℕ) : ℚ := (1 - 2 * n^2) / (2 + 4 * n^2)

theorem limit_of_sequence_a : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1683_168314


namespace NUMINAMATH_CALUDE_circle_radius_is_6_sqrt_2_l1683_168363

/-- A right triangle with squares constructed on two sides --/
structure RightTriangleWithSquares where
  -- The lengths of the two sides of the right triangle
  PQ : ℝ
  QR : ℝ
  -- Assertion that the squares are constructed on these sides
  square_PQ_constructed : Bool
  square_QR_constructed : Bool
  -- Assertion that the corners of the squares lie on a circle
  corners_on_circle : Bool

/-- The radius of the circle passing through the corners of the squares --/
def circle_radius (t : RightTriangleWithSquares) : ℝ :=
  sorry

/-- Theorem stating that for a right triangle with PQ = 9 and QR = 12,
    and squares constructed on these sides, if the corners of the squares
    lie on a circle, then the radius of this circle is 6√2 --/
theorem circle_radius_is_6_sqrt_2 (t : RightTriangleWithSquares)
    (h1 : t.PQ = 9)
    (h2 : t.QR = 12)
    (h3 : t.square_PQ_constructed = true)
    (h4 : t.square_QR_constructed = true)
    (h5 : t.corners_on_circle = true) :
  circle_radius t = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_6_sqrt_2_l1683_168363


namespace NUMINAMATH_CALUDE_complex_square_pure_imaginary_l1683_168333

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_square_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_pure_imaginary_l1683_168333


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1683_168349

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 14*x^2 - 20*x + 15 = (x - 3)*(x^4 + 3*x^3 - 16*x^2 - 34*x - 122) + (-291) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1683_168349


namespace NUMINAMATH_CALUDE_drug_price_reduction_equation_l1683_168303

/-- Represents the price reduction scenario for a drug -/
def price_reduction (initial_price final_price : ℝ) (num_reductions : ℕ) (reduction_percentage : ℝ) : Prop :=
  initial_price * (1 - reduction_percentage) ^ num_reductions = final_price

/-- Theorem stating the equation for the drug price reduction scenario -/
theorem drug_price_reduction_equation :
  ∃ (x : ℝ), price_reduction 144 81 2 x :=
sorry

end NUMINAMATH_CALUDE_drug_price_reduction_equation_l1683_168303


namespace NUMINAMATH_CALUDE_mrs_hilt_daily_reading_l1683_168318

/-- The number of books Mrs. Hilt read in one week -/
def books_per_week : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of books Mrs. Hilt read per day -/
def books_per_day : ℚ := books_per_week / days_in_week

theorem mrs_hilt_daily_reading : books_per_day = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_daily_reading_l1683_168318


namespace NUMINAMATH_CALUDE_sum_of_series_equals_two_l1683_168322

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-2)/(3^n) is equal to 2. -/
theorem sum_of_series_equals_two :
  ∑' n : ℕ, (4 * n - 2 : ℝ) / (3 ^ n) = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_two_l1683_168322


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1683_168330

/-- Represents a triangle divided into three triangles and one quadrilateral -/
structure DividedTriangle where
  -- Areas of the three triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  quad_area : ℝ

/-- Theorem stating that if the areas of the three triangles are 6, 9, and 15,
    then the area of the quadrilateral is 65 -/
theorem quadrilateral_area (t : DividedTriangle) :
  t.area1 = 6 ∧ t.area2 = 9 ∧ t.area3 = 15 → t.quad_area = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1683_168330


namespace NUMINAMATH_CALUDE_negative_four_squared_l1683_168372

theorem negative_four_squared : -4^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_l1683_168372


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1683_168381

/-- The complex number z = (3+i)/(1-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 + Complex.I) / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1683_168381


namespace NUMINAMATH_CALUDE_correct_election_result_l1683_168390

/-- Election results with three candidates -/
structure ElectionResult where
  total_votes : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Conditions of the election -/
def election_conditions (result : ElectionResult) : Prop :=
  (result.votes_a : ℚ) / result.total_votes = 45 / 100 ∧
  (result.votes_b : ℚ) / result.total_votes = 35 / 100 ∧
  (result.votes_c : ℚ) / result.total_votes = 20 / 100 ∧
  result.votes_a - result.votes_b = 2500 ∧
  result.total_votes = result.votes_a + result.votes_b + result.votes_c

/-- Theorem stating the correct election results -/
theorem correct_election_result :
  ∃ (result : ElectionResult),
    election_conditions result ∧
    result.total_votes = 25000 ∧
    result.votes_a = 11250 ∧
    result.votes_b = 8750 ∧
    result.votes_c = 5000 := by
  sorry

end NUMINAMATH_CALUDE_correct_election_result_l1683_168390


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1683_168355

theorem quadratic_minimum (k : ℝ) : 
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → (1/2) * (x - 1)^2 + k ≥ 3) ∧
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 5 ∧ (1/2) * (x - 1)^2 + k = 3) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1683_168355


namespace NUMINAMATH_CALUDE_correct_sunset_time_l1683_168317

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry, by sorry⟩

theorem correct_sunset_time :
  let sunrise : Time := ⟨7, 12, by sorry, by sorry⟩
  let incorrectDaylight : Nat := 11 * 60 + 15 -- 11 hours and 15 minutes in minutes
  let calculatedSunset := addMinutes sunrise incorrectDaylight
  calculatedSunset.hours = 18 ∧ calculatedSunset.minutes = 27 :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l1683_168317


namespace NUMINAMATH_CALUDE_soccer_team_games_l1683_168382

/-- Calculates the total number of games played by a soccer team given their win:loss:tie ratio and the number of games lost. -/
def total_games (win_ratio : ℕ) (loss_ratio : ℕ) (tie_ratio : ℕ) (games_lost : ℕ) : ℕ :=
  let games_per_part := games_lost / loss_ratio
  let total_parts := win_ratio + loss_ratio + tie_ratio
  total_parts * games_per_part

/-- Theorem stating that for a soccer team with a win:loss:tie ratio of 4:3:1 and 9 losses, the total number of games played is 24. -/
theorem soccer_team_games : 
  total_games 4 3 1 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l1683_168382


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l1683_168374

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2 - x^3

def point_P : ℝ × ℝ := (0, -4)

def is_tangent_point (a : ℝ) : Prop :=
  ∃ (m : ℝ), curve a = 2 * a^2 - a^3 ∧
             m * a + (2 * a^2 - a^3) = -4 ∧
             m = 4 * a - 3 * a^2

theorem tangent_line_theorem :
  ∃ (a : ℝ), is_tangent_point a ∧ a = -1 ∧
  ∃ (m : ℝ), m = -7 ∧ 
  (∀ (x y : ℝ), y = m * x - 4 ↔ 7 * x + y + 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l1683_168374


namespace NUMINAMATH_CALUDE_square_sum_seventeen_l1683_168369

theorem square_sum_seventeen (x y : ℝ) 
  (h1 : y + 7 = (x - 3)^2) 
  (h2 : x + 7 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 17 := by
sorry

end NUMINAMATH_CALUDE_square_sum_seventeen_l1683_168369


namespace NUMINAMATH_CALUDE_singles_on_itunes_l1683_168395

def total_songs : ℕ := 55
def albums_15_songs : ℕ := 2
def songs_per_album_15 : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_album_20 : ℕ := 20

theorem singles_on_itunes : 
  total_songs - (albums_15_songs * songs_per_album_15 + albums_20_songs * songs_per_album_20) = 5 := by
  sorry

end NUMINAMATH_CALUDE_singles_on_itunes_l1683_168395


namespace NUMINAMATH_CALUDE_roofing_cost_calculation_l1683_168344

theorem roofing_cost_calculation (total_needed : ℕ) (cost_per_foot : ℕ) (free_roofing : ℕ) : 
  total_needed = 300 → 
  cost_per_foot = 8 → 
  free_roofing = 250 → 
  (total_needed - free_roofing) * cost_per_foot = 400 := by
  sorry

end NUMINAMATH_CALUDE_roofing_cost_calculation_l1683_168344


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1683_168316

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2*x*y = 8) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b + 2*a*b = 8 → x + 2*y ≤ a + 2*b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1683_168316


namespace NUMINAMATH_CALUDE_translated_line_y_axis_intersection_l1683_168366

/-- The intersection point of a line translated upward with the y-axis -/
theorem translated_line_y_axis_intersection
  (original_line : ℝ → ℝ)
  (h_original : ∀ x, original_line x = x - 3)
  (translation : ℝ)
  (h_translation : translation = 2)
  (translated_line : ℝ → ℝ)
  (h_translated : ∀ x, translated_line x = original_line x + translation)
  : translated_line 0 = -1 :=
by sorry

end NUMINAMATH_CALUDE_translated_line_y_axis_intersection_l1683_168366


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l1683_168359

-- Define the population size
def population_size : ℕ := 100

-- Define the sample size
def sample_size : ℕ := 5

-- Define the probability of an individual being drawn
def prob_individual_drawn (n : ℕ) (k : ℕ) : ℚ := k / n

-- Theorem statement
theorem simple_random_sampling_probability :
  prob_individual_drawn population_size sample_size = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l1683_168359


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1683_168311

theorem triangle_side_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1683_168311


namespace NUMINAMATH_CALUDE_stereographic_projection_is_inversion_l1683_168310

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a sphere (Earth) -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Stereographic projection from a pole onto a plane -/
def stereographicProjection (sphere : Sphere) (pole : Point3D) (plane : Plane) (point : Point3D) : Point3D :=
  sorry

/-- The mapping between corresponding points on two planes -/
def planeMapping (sphere : Sphere) (plane1 : Plane) (plane2 : Plane) (point : Point3D) : Point3D :=
  sorry

/-- Definition of inversion -/
def isInversion (f : Point3D → Point3D) (center : Point3D) (radius : ℝ) : Prop :=
  sorry

theorem stereographic_projection_is_inversion
  (sphere : Sphere)
  (northPole : Point3D)
  (southPole : Point3D)
  (plane1 : Plane)
  (plane2 : Plane)
  (h1 : plane1.point = northPole)
  (h2 : plane2.point = southPole)
  (h3 : northPole.z = sphere.radius)
  (h4 : southPole.z = -sphere.radius) :
  ∃ (center : Point3D) (radius : ℝ),
    isInversion (planeMapping sphere plane1 plane2) center radius :=
  sorry

end NUMINAMATH_CALUDE_stereographic_projection_is_inversion_l1683_168310


namespace NUMINAMATH_CALUDE_integer_x_is_seven_l1683_168389

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 8)
  (h3 : -2 < x ∧ x < 9)
  (h4 : 0 < x ∧ x < 8)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_x_is_seven_l1683_168389


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l1683_168367

/-- Calculates the average speed of a car trip given specific conditions -/
theorem car_trip_average_speed :
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let first_part_speed : ℝ := 35
  let second_part_speed : ℝ := 44
  let total_distance : ℝ := first_part_speed * first_part_time + 
                             second_part_speed * (total_time - first_part_time)
  total_distance / total_time = 38 := by
sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l1683_168367


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1683_168384

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 160 → 
  train_speed_kmh = 45 → 
  bridge_length = 215 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1683_168384


namespace NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l1683_168336

/-- The sum of an arithmetic sequence of 20 terms -/
def sum_20_consecutive (first : ℕ) : ℕ :=
  20 * (2 * first + 19) / 2

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem smallest_square_sum_20_consecutive :
  (∀ n : ℕ, n < 490 → ¬(is_perfect_square n ∧ ∃ k : ℕ, sum_20_consecutive k = n)) ∧
  (is_perfect_square 490 ∧ ∃ k : ℕ, sum_20_consecutive k = 490) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l1683_168336


namespace NUMINAMATH_CALUDE_natalia_novels_l1683_168399

/-- The number of novels Natalia has in her library -/
def number_of_novels : ℕ := sorry

/-- The number of comics in Natalia's library -/
def comics : ℕ := 271

/-- The number of documentaries in Natalia's library -/
def documentaries : ℕ := 419

/-- The number of albums in Natalia's library -/
def albums : ℕ := 209

/-- The capacity of each crate -/
def crate_capacity : ℕ := 9

/-- The total number of crates used -/
def total_crates : ℕ := 116

theorem natalia_novels :
  number_of_novels = 145 ∧
  comics + documentaries + albums + number_of_novels = crate_capacity * total_crates :=
by sorry

end NUMINAMATH_CALUDE_natalia_novels_l1683_168399


namespace NUMINAMATH_CALUDE_computer_price_decrease_l1683_168342

/-- The price of a computer after a certain number of years, given an initial price and a rate of decrease every 3 years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 3)

/-- Theorem stating that the price of a computer initially priced at 8100 yuan,
    decreasing by 1/3 every 3 years, will be 2400 yuan after 9 years. -/
theorem computer_price_decrease :
  price_after_years 8100 (1/3) 9 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_decrease_l1683_168342


namespace NUMINAMATH_CALUDE_prism_lateral_edges_parallel_equal_l1683_168386

structure Prism where
  -- A prism is a polyhedron
  is_polyhedron : Bool
  -- A prism has two congruent and parallel bases
  has_congruent_parallel_bases : Bool
  -- The lateral faces of a prism are parallelograms
  lateral_faces_are_parallelograms : Bool

/-- The lateral edges of a prism are parallel and equal in length -/
theorem prism_lateral_edges_parallel_equal (p : Prism) :
  p.is_polyhedron ∧ p.has_congruent_parallel_bases ∧ p.lateral_faces_are_parallelograms →
  (lateral_edges_parallel : Bool) ∧ (lateral_edges_equal_length : Bool) :=
by sorry

end NUMINAMATH_CALUDE_prism_lateral_edges_parallel_equal_l1683_168386


namespace NUMINAMATH_CALUDE_expression_factorization_l1683_168376

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 - 36 * x^4) - (4 * x^6 - 9 * x^4 + 12) = 3 * x^4 * (2 * x + 3) * (2 * x - 3) - 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1683_168376


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l1683_168377

/-- 
Given real numbers a and b, prove that the expression 
x^2 - 4bx + 4ab + p^2 - 2px is a perfect square when p = a - b
-/
theorem expression_is_perfect_square (a b x : ℝ) : 
  ∃ k : ℝ, x^2 - 4*b*x + 4*a*b + (a - b)^2 - 2*(a - b)*x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l1683_168377


namespace NUMINAMATH_CALUDE_statement_IV_must_be_false_l1683_168338

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Represents the four statements about the number -/
structure Statements (n : TwoDigitNumber) where
  I : Bool
  II : Bool
  III : Bool
  IV : Bool
  I_def : I ↔ n.value = 12
  II_def : II ↔ n.value % 10 ≠ 2
  III_def : III ↔ n.value = 35
  IV_def : IV ↔ n.value % 10 ≠ 5
  three_true : I + II + III + IV = 3

theorem statement_IV_must_be_false (n : TwoDigitNumber) (s : Statements n) :
  s.IV = false :=
sorry

end NUMINAMATH_CALUDE_statement_IV_must_be_false_l1683_168338


namespace NUMINAMATH_CALUDE_apple_pie_count_l1683_168394

/-- Given a box of apples weighing 120 pounds, using half for applesauce and the rest for pies,
    with each pie requiring 4 pounds of apples, prove that 15 pies can be made. -/
theorem apple_pie_count (total_weight : ℕ) (pie_weight : ℕ) : 
  total_weight = 120 →
  pie_weight = 4 →
  (total_weight / 2) / pie_weight = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l1683_168394


namespace NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l1683_168328

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ
  orange : ℕ

/-- Calculates the total number of gumdrops in the jar -/
def GumdropJar.total (jar : GumdropJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green + jar.orange

/-- Represents the initial distribution of gumdrops -/
def initial_jar : GumdropJar :=
  { blue := 40
    brown := 15
    red := 10
    yellow := 5
    green := 20
    orange := 10 }

/-- Theorem stating that after replacing a third of blue gumdrops with orange,
    the number of orange gumdrops will be 23 -/
theorem orange_gumdrops_after_replacement (jar : GumdropJar)
    (h1 : jar = initial_jar)
    (h2 : jar.total = 100)
    (h3 : jar.blue / 3 = 13) :
  (⟨jar.blue - 13, jar.brown, jar.red, jar.yellow, jar.green, jar.orange + 13⟩ : GumdropJar).orange = 23 := by
  sorry

end NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l1683_168328


namespace NUMINAMATH_CALUDE_gold_beads_undetermined_l1683_168353

/-- Represents the types of beads used in the corset --/
inductive BeadType
  | Purple
  | Blue
  | Gold

/-- Represents a row of beads --/
structure BeadRow where
  beadType : BeadType
  beadsPerRow : ℕ
  rowCount : ℕ

/-- Represents the corset design --/
structure CorsetDesign where
  purpleRows : BeadRow
  blueRows : BeadRow
  goldBeads : ℕ
  totalCost : ℚ

def carlyDesign : CorsetDesign :=
  { purpleRows := { beadType := BeadType.Purple, beadsPerRow := 20, rowCount := 50 }
  , blueRows := { beadType := BeadType.Blue, beadsPerRow := 18, rowCount := 40 }
  , goldBeads := 0  -- This is what we're trying to determine
  , totalCost := 180 }

/-- The theorem stating that the number of gold beads cannot be determined --/
theorem gold_beads_undetermined (design : CorsetDesign) : 
  design.purpleRows.beadsPerRow = carlyDesign.purpleRows.beadsPerRow ∧ 
  design.purpleRows.rowCount = carlyDesign.purpleRows.rowCount ∧
  design.blueRows.beadsPerRow = carlyDesign.blueRows.beadsPerRow ∧
  design.blueRows.rowCount = carlyDesign.blueRows.rowCount ∧
  design.totalCost = carlyDesign.totalCost →
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (design1 design2 : CorsetDesign), 
      design1.goldBeads = x ∧ 
      design2.goldBeads = y ∧
      design1.purpleRows = design.purpleRows ∧
      design1.blueRows = design.blueRows ∧
      design1.totalCost = design.totalCost ∧
      design2.purpleRows = design.purpleRows ∧
      design2.blueRows = design.blueRows ∧
      design2.totalCost = design.totalCost) :=
by
  sorry

end NUMINAMATH_CALUDE_gold_beads_undetermined_l1683_168353


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l1683_168368

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l1683_168368


namespace NUMINAMATH_CALUDE_company_n_profit_change_l1683_168306

def CompanyN (R : ℝ) : Prop :=
  let profit1998 := 0.10 * R
  let revenue1999 := 0.70 * R
  let profit1999 := 0.15 * revenue1999
  let revenue2000 := 1.20 * revenue1999
  let profit2000 := 0.18 * revenue2000
  let percentageChange := (profit2000 - profit1998) / profit1998 * 100
  percentageChange = 51.2

theorem company_n_profit_change (R : ℝ) (h : R > 0) : CompanyN R := by
  sorry

end NUMINAMATH_CALUDE_company_n_profit_change_l1683_168306


namespace NUMINAMATH_CALUDE_difference_of_squares_601_597_l1683_168337

theorem difference_of_squares_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_597_l1683_168337


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l1683_168323

theorem solution_to_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l1683_168323


namespace NUMINAMATH_CALUDE_percentage_of_50_to_125_l1683_168356

theorem percentage_of_50_to_125 : (50 / 125) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_50_to_125_l1683_168356


namespace NUMINAMATH_CALUDE_set_equivalence_l1683_168313

theorem set_equivalence : 
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
sorry

end NUMINAMATH_CALUDE_set_equivalence_l1683_168313


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1683_168304

/-- Given a quadratic inequality mx^2 + 8mx + 28 < 0 with solution set {x | -7 < x < -1},
    prove that m = 4 -/
theorem quadratic_inequality_solution (m : ℝ) 
  (h : ∀ x, mx^2 + 8*m*x + 28 < 0 ↔ -7 < x ∧ x < -1) : 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1683_168304


namespace NUMINAMATH_CALUDE_cookies_per_tray_l1683_168308

/-- Given that Marian baked 276 oatmeal cookies and used 23 trays,
    prove that she can place 12 cookies on a tray at a time. -/
theorem cookies_per_tray (total_cookies : ℕ) (num_trays : ℕ) 
  (h1 : total_cookies = 276) (h2 : num_trays = 23) :
  total_cookies / num_trays = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_tray_l1683_168308


namespace NUMINAMATH_CALUDE_denominator_numerator_difference_l1683_168340

/-- Represents a base-12 number as a pair of integers (numerator, denominator) -/
def Base12Fraction := ℤ × ℤ

/-- Converts a repeating decimal in base 12 to a fraction -/
def repeating_decimal_to_fraction (digits : List ℕ) : Base12Fraction := sorry

/-- Simplifies a fraction to its lowest terms -/
def simplify_fraction (f : Base12Fraction) : Base12Fraction := sorry

/-- The infinite repeating decimal 0.127127127... in base 12 -/
def G : Base12Fraction := repeating_decimal_to_fraction [1, 2, 7]

theorem denominator_numerator_difference :
  let simplified_G := simplify_fraction G
  (simplified_G.2 - simplified_G.1) = 342 := by sorry

end NUMINAMATH_CALUDE_denominator_numerator_difference_l1683_168340


namespace NUMINAMATH_CALUDE_symmetry_and_inverse_l1683_168398

/-- A function that is symmetric about the line y = x + 1 -/
def SymmetricAboutXPlus1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (y - 1) = x + 1

/-- Definition of g in terms of f and b -/
def g (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f (x + b)

/-- A function that is identical to its inverse -/
def IdenticalToInverse (h : ℝ → ℝ) : Prop :=
  ∀ x, h (h x) = x

theorem symmetry_and_inverse (f : ℝ → ℝ) (b : ℝ) 
  (h_sym : SymmetricAboutXPlus1 f) :
  IdenticalToInverse (g f b) ↔ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_and_inverse_l1683_168398


namespace NUMINAMATH_CALUDE_tan_double_angle_l1683_168324

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1683_168324
