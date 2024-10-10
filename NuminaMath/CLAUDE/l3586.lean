import Mathlib

namespace persistent_is_two_l3586_358661

/-- A number T is persistent if for any a, b, c, d ≠ 0, 1:
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T implies 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def IsPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

theorem persistent_is_two (T : ℝ) : IsPersistent T → T = 2 := by
  sorry

end persistent_is_two_l3586_358661


namespace repeating_decimal_sum_l3586_358648

/-- Represents a repeating decimal with a two-digit repeating sequence -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.474747... -/
def x : ℚ := RepeatingDecimal 4 7

/-- The sum of the numerator and denominator of a fraction -/
def sumNumeratorDenominator (q : ℚ) : ℕ :=
  q.num.natAbs + q.den

theorem repeating_decimal_sum : sumNumeratorDenominator x = 146 := by
  sorry

end repeating_decimal_sum_l3586_358648


namespace class_size_difference_l3586_358677

theorem class_size_difference (total_students : ℕ) (total_professors : ℕ) (class_sizes : List ℕ) :
  total_students = 200 →
  total_professors = 4 →
  class_sizes = [100, 50, 30, 20] →
  (class_sizes.sum = total_students) →
  let t := (class_sizes.sum : ℚ) / total_professors
  let s := (class_sizes.map (λ size => size * size)).sum / total_students
  t - s = -19 := by
  sorry

end class_size_difference_l3586_358677


namespace divisibility_by_360_l3586_358644

theorem divisibility_by_360 (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  360 ∣ (p^4 - 5*p^2 + 4) := by
sorry

end divisibility_by_360_l3586_358644


namespace divisibility_condition_l3586_358623

-- Define the predicate for divisibility
def divides (m n : ℤ) : Prop := ∃ k : ℤ, n = m * k

theorem divisibility_condition (p a : ℤ) : 
  (p ≥ 2) → 
  (a ≥ 1) → 
  Prime p → 
  p ≠ a → 
  (divides (a + p) (a^2 + p^2) ↔ 
    ((a = p) ∨ (a = p^2 - p) ∨ (a = 2*p^2 - p))) :=
by sorry

end divisibility_condition_l3586_358623


namespace tangent_circle_center_l3586_358622

/-- A circle tangent to two parallel lines with its center on a third line -/
structure TangentCircle where
  -- First tangent line: 3x - 4y = 20
  tangent_line1 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = 20
  -- Second tangent line: 3x - 4y = -40
  tangent_line2 : (ℝ × ℝ) → Prop := fun (x, y) ↦ 3 * x - 4 * y = -40
  -- Line containing the center: x - 3y = 0
  center_line : (ℝ × ℝ) → Prop := fun (x, y) ↦ x - 3 * y = 0

/-- The center of the tangent circle is at (-6, -2) -/
theorem tangent_circle_center (c : TangentCircle) : 
  ∃ (x y : ℝ), c.center_line (x, y) ∧ x = -6 ∧ y = -2 :=
sorry

end tangent_circle_center_l3586_358622


namespace same_color_probability_eight_nine_l3586_358606

/-- The probability of drawing two balls of the same color from a box containing 
    8 white balls and 9 black balls. -/
def same_color_probability (white : ℕ) (black : ℕ) : ℚ :=
  let total := white + black
  let same_color_ways := (white.choose 2) + (black.choose 2)
  let total_ways := total.choose 2
  same_color_ways / total_ways

/-- Theorem stating that the probability of drawing two balls of the same color 
    from a box with 8 white balls and 9 black balls is 8/17. -/
theorem same_color_probability_eight_nine : 
  same_color_probability 8 9 = 8 / 17 := by
  sorry

end same_color_probability_eight_nine_l3586_358606


namespace ticket_price_possibilities_l3586_358610

theorem ticket_price_possibilities : ∃ (S : Finset ℕ), 
  (∀ y ∈ S, y > 0 ∧ 42 % y = 0 ∧ 70 % y = 0) ∧ 
  (∀ y : ℕ, y > 0 → 42 % y = 0 → 70 % y = 0 → y ∈ S) ∧
  Finset.card S = 4 := by
  sorry

end ticket_price_possibilities_l3586_358610


namespace four_circles_common_tangent_l3586_358617

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the length of a common tangent between two circles -/
def tangentLength (c1 c2 : Circle) : ℝ := sorry

/-- 
Given four circles α, β, γ, and δ satisfying the tangent length equation,
there exists a circle tangent to all four circles.
-/
theorem four_circles_common_tangent 
  (α β γ δ : Circle)
  (h : tangentLength α β * tangentLength γ δ + 
       tangentLength β γ * tangentLength δ α = 
       tangentLength α γ * tangentLength β δ) :
  ∃ (σ : Circle), 
    (tangentLength σ α = 0) ∧ 
    (tangentLength σ β = 0) ∧ 
    (tangentLength σ γ = 0) ∧ 
    (tangentLength σ δ = 0) :=
sorry

end four_circles_common_tangent_l3586_358617


namespace victoria_money_l3586_358600

/-- The amount of money Victoria was given by her mother -/
def total_money : ℕ := sorry

/-- The cost of one box of pizza -/
def pizza_cost : ℕ := 12

/-- The number of pizza boxes bought -/
def pizza_boxes : ℕ := 2

/-- The cost of one pack of juice drinks -/
def juice_cost : ℕ := 2

/-- The number of juice drink packs bought -/
def juice_packs : ℕ := 2

/-- The amount Victoria should return to her mother -/
def return_amount : ℕ := 22

/-- Theorem stating that the total money Victoria was given equals $50 -/
theorem victoria_money : 
  total_money = pizza_cost * pizza_boxes + juice_cost * juice_packs + return_amount :=
by sorry

end victoria_money_l3586_358600


namespace sharon_wants_254_supplies_l3586_358694

/-- Calculates the total number of kitchen supplies Sharon wants to buy -/
def sharons_kitchen_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 6 + 3 * angela_pots
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_kitchen_supplies 20 = 254 := by
  sorry

end sharon_wants_254_supplies_l3586_358694


namespace sufficient_not_necessary_condition_range_l3586_358660

/-- Given two predicates p and q on real numbers, where p is a sufficient but not necessary condition for q, 
    this theorem states that the lower bound of k in p is 2, and k can be any real number greater than 2. -/
theorem sufficient_not_necessary_condition_range (k : ℝ) : 
  (∀ x, x ≥ k → (2 - x) / (x + 1) < 0) ∧ 
  (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → 
  k > 2 :=
sorry

end sufficient_not_necessary_condition_range_l3586_358660


namespace pascal_triangle_row_15_fifth_number_l3586_358612

theorem pascal_triangle_row_15_fifth_number : 
  let row := List.range 16
  let pascal_row := row.map (fun k => Nat.choose 15 k)
  pascal_row[4] = 1365 := by
  sorry

end pascal_triangle_row_15_fifth_number_l3586_358612


namespace triangle_properties_l3586_358621

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 * b ∧
  Real.sin A + Real.sin B = 2 * Real.sin C ∧
  (1 / 2) * b * c * Real.sin A = (8 * Real.sqrt 15) / 3 →
  Real.cos A = -1 / 4 ∧ c = 4 * Real.sqrt 2 := by
  sorry

end triangle_properties_l3586_358621


namespace distribute_five_books_four_students_l3586_358658

/-- The number of ways to distribute n different books to k students,
    with each student getting at least one book -/
def distribute (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2) * (k-2)^n

/-- Theorem: There are 240 ways to distribute 5 different books to 4 students,
    with each student getting at least one book -/
theorem distribute_five_books_four_students :
  distribute 5 4 = 240 := by
  sorry

end distribute_five_books_four_students_l3586_358658


namespace super_extra_yield_interest_l3586_358680

/-- Calculates the interest earned on a compound interest savings account -/
theorem super_extra_yield_interest
  (principal : ℝ)
  (rate : ℝ)
  (years : ℕ)
  (h_principal : principal = 1500)
  (h_rate : rate = 0.02)
  (h_years : years = 5) :
  ⌊(principal * (1 + rate) ^ years - principal)⌋ = 156 := by
  sorry

end super_extra_yield_interest_l3586_358680


namespace other_man_age_is_36_l3586_358646

/-- The age of the other man in the group problem -/
def other_man_age : ℕ := 36

/-- The number of men in the initial group -/
def num_men : ℕ := 9

/-- The increase in average age when two women replace two men -/
def avg_age_increase : ℕ := 4

/-- The age of one of the men in the group -/
def known_man_age : ℕ := 32

/-- The average age of the two women -/
def women_avg_age : ℕ := 52

/-- The theorem stating that given the conditions, the age of the other man is 36 -/
theorem other_man_age_is_36 :
  (num_men * avg_age_increase = 2 * women_avg_age - (other_man_age + known_man_age)) →
  other_man_age = 36 := by
  sorry

#check other_man_age_is_36

end other_man_age_is_36_l3586_358646


namespace simplify_expression_l3586_358640

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end simplify_expression_l3586_358640


namespace sum_and_count_result_l3586_358631

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def x : ℕ := sum_of_integers 10 20

def y : ℕ := count_even_integers 10 20

theorem sum_and_count_result : x + y = 171 := by
  sorry

end sum_and_count_result_l3586_358631


namespace sum_of_specific_S_values_l3586_358696

-- Define the sequence Sn
def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

-- State the theorem
theorem sum_of_specific_S_values : S 19 + S 37 + S 52 = 3 := by
  sorry

end sum_of_specific_S_values_l3586_358696


namespace tree_height_after_three_years_l3586_358685

/-- The height of a tree that doubles every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (2 ^ years)

theorem tree_height_after_three_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 6 = 32 ∧
    tree_height initial_height 3 = 4 := by
  sorry

#check tree_height_after_three_years

end tree_height_after_three_years_l3586_358685


namespace win_sector_area_l3586_358604

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/3) :
  p * π * r^2 = 48 * π := by
  sorry

end win_sector_area_l3586_358604


namespace intersection_product_l3586_358645

-- Define the sets T and S
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

-- State the theorem
theorem intersection_product (a b : ℝ) : 
  S b ∩ T a = {(2, 1)} → a * b = 1 := by
  sorry

end intersection_product_l3586_358645


namespace max_value_theorem_l3586_358616

theorem max_value_theorem (x y : ℝ) : 
  (2*x + 3*y + 5) / Real.sqrt (x^2 + 2*y^2 + 2) ≤ Real.sqrt 38 ∧ 
  ∃ (x₀ y₀ : ℝ), (2*x₀ + 3*y₀ + 5) / Real.sqrt (x₀^2 + 2*y₀^2 + 2) = Real.sqrt 38 := by
  sorry

end max_value_theorem_l3586_358616


namespace part_one_part_two_l3586_358625

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 5|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2 * |x + 5|} = {x : ℝ | x ≤ -2} := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 8) → (a ≥ 3 ∨ a ≤ -13) := by sorry

end part_one_part_two_l3586_358625


namespace card_game_profit_general_card_game_profit_l3586_358641

/-- Expected profit function for the card guessing game -/
def expected_profit (r b g : ℕ) : ℚ :=
  (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g

/-- Theorem stating the expected profit for the specific game instance -/
theorem card_game_profit :
  expected_profit 2011 2012 2011 = 1 / 4023 := by
  sorry

/-- Theorem for the general case of the card guessing game -/
theorem general_card_game_profit (r b g : ℕ) (h : r + b > 0) :
  expected_profit r b g =
    (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g := by
  sorry

end card_game_profit_general_card_game_profit_l3586_358641


namespace box_volume_l3586_358605

/-- Given a box with specified dimensions, prove its volume is 3888 cubic inches. -/
theorem box_volume : 
  ∀ (height length width : ℝ),
    height = 12 →
    length = 3 * height →
    length = 4 * width →
    height * length * width = 3888 := by
  sorry

end box_volume_l3586_358605


namespace x_range_l3586_358608

theorem x_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, x^2 + 2*x < a/b + 16*b/a → -4 < x ∧ x < 2 :=
sorry

end x_range_l3586_358608


namespace ryegrass_percentage_in_x_l3586_358635

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combined_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * x_proportion + y.ryegrass * (1 - x_proportion)
  , bluegrass := x.bluegrass * x_proportion + y.bluegrass * (1 - x_proportion)
  , fescue := x.fescue * x_proportion + y.fescue * (1 - x_proportion) }

theorem ryegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture) 
  (h1 : x.bluegrass = 60)
  (h2 : x.ryegrass + x.bluegrass + x.fescue = 100)
  (h3 : y.ryegrass = 25)
  (h4 : y.fescue = 75)
  (h5 : y.ryegrass + y.bluegrass + y.fescue = 100)
  (h6 : (combined_mixture x y (2/3)).ryegrass = 35) :
  x.ryegrass = 40 := by
    sorry

#check ryegrass_percentage_in_x

end ryegrass_percentage_in_x_l3586_358635


namespace complex_modulus_l3586_358683

theorem complex_modulus (z : ℂ) : z = (1 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_l3586_358683


namespace cubic_equation_roots_l3586_358682

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x^2 - a = 0 ∧ 
    y^3 - 3*y^2 - a = 0 ∧ 
    z^3 - 3*z^2 - a = 0) → 
  -4 < a ∧ a < 0 :=
by sorry

end cubic_equation_roots_l3586_358682


namespace max_oranges_for_teacher_l3586_358678

theorem max_oranges_for_teacher (n : ℕ) : 
  let k := 8
  let remainder := n % k
  remainder ≤ 7 ∧ ∃ m : ℕ, n = m * k + 7 :=
by sorry

end max_oranges_for_teacher_l3586_358678


namespace female_democrats_count_l3586_358668

-- Define the total number of participants
def total_participants : ℕ := 840

-- Define the ratio of female Democrats to total females
def female_democrat_ratio : ℚ := 1/2

-- Define the ratio of male Democrats to total males
def male_democrat_ratio : ℚ := 1/4

-- Define the ratio of all Democrats to total participants
def total_democrat_ratio : ℚ := 1/3

-- Theorem statement
theorem female_democrats_count :
  ∃ (female_participants male_participants : ℕ),
    female_participants + male_participants = total_participants ∧
    (female_democrat_ratio * female_participants + male_democrat_ratio * male_participants : ℚ) = 
      total_democrat_ratio * total_participants ∧
    female_democrat_ratio * female_participants = 140 :=
sorry

end female_democrats_count_l3586_358668


namespace intersection_of_M_and_N_l3586_358663

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (3 - x^2)}

-- State the theorem
theorem intersection_of_M_and_N :
  (M ∩ N : Set ℝ) = {x | -1 ≤ x ∧ x ≤ Real.sqrt 3} := by sorry

end intersection_of_M_and_N_l3586_358663


namespace m_range_theorem_l3586_358672

/-- The function f(x) = x^2 + mx + 1 has two distinct roots -/
def has_two_distinct_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- There exists an x such that 4x^2 + 4(m-2)x + 1 ≤ 0 -/
def exists_nonpositive (m : ℝ) : Prop :=
  ∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≤ 0

/-- The range of m is (-∞, -2) ∪ [3, +∞) -/
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m ≥ 3

theorem m_range_theorem (m : ℝ) :
  has_two_distinct_roots m ∧ exists_nonpositive m → m_range m := by
  sorry

end m_range_theorem_l3586_358672


namespace smallest_sum_of_reciprocals_l3586_358603

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12 → (x : ℕ) + y ≥ 49 := by
  sorry

end smallest_sum_of_reciprocals_l3586_358603


namespace function_parity_l3586_358611

-- Define the property of the function
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

-- Define even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem function_parity (f : ℝ → ℝ) (h : satisfies_property f) :
  (is_even f ∨ is_odd f) :=
sorry

end function_parity_l3586_358611


namespace total_birds_is_168_l3586_358632

/-- Represents the number of birds of each species -/
structure BirdCounts where
  bluebirds : ℕ
  cardinals : ℕ
  goldfinches : ℕ
  sparrows : ℕ
  swallows : ℕ

/-- Conditions for the bird counts -/
def validBirdCounts (b : BirdCounts) : Prop :=
  b.cardinals = 2 * b.bluebirds ∧
  b.goldfinches = 4 * b.bluebirds ∧
  b.sparrows = (b.cardinals + b.goldfinches) / 2 ∧
  b.swallows = 8 ∧
  b.bluebirds = 2 * b.swallows

/-- The total number of birds -/
def totalBirds (b : BirdCounts) : ℕ :=
  b.bluebirds + b.cardinals + b.goldfinches + b.sparrows + b.swallows

/-- Theorem: The total number of birds is 168 -/
theorem total_birds_is_168 :
  ∀ b : BirdCounts, validBirdCounts b → totalBirds b = 168 := by
  sorry

end total_birds_is_168_l3586_358632


namespace quadratic_roots_range_l3586_358684

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) → 
  m ∈ Set.Ioo (-5) (-4) ∪ {-4} :=
sorry

end quadratic_roots_range_l3586_358684


namespace charity_boxes_theorem_l3586_358670

/-- Calculates the total number of boxes a charity can pack given initial conditions --/
theorem charity_boxes_theorem (initial_boxes : ℕ) (food_cost : ℕ) (supplies_cost : ℕ) (donation_multiplier : ℕ) : 
  initial_boxes = 400 → 
  food_cost = 80 → 
  supplies_cost = 165 → 
  donation_multiplier = 4 → 
  (initial_boxes + (donation_multiplier * initial_boxes * (food_cost + supplies_cost)) / (food_cost + supplies_cost) : ℕ) = 2000 := by
  sorry

#check charity_boxes_theorem

end charity_boxes_theorem_l3586_358670


namespace time_after_1567_minutes_l3586_358667

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a day and time -/
structure DayTime where
  days : Nat
  time : Time
  deriving Repr

def addMinutes (start : Time) (minutes : Nat) : DayTime :=
  let totalMinutes := start.minutes + minutes
  let totalHours := start.hours + totalMinutes / 60
  let finalMinutes := totalMinutes % 60
  let finalHours := totalHours % 24
  let days := totalHours / 24
  { days := days
  , time := { hours := finalHours, minutes := finalMinutes } }

theorem time_after_1567_minutes :
  let start := Time.mk 17 0  -- 5:00 p.m.
  let result := addMinutes start 1567
  result = DayTime.mk 1 (Time.mk 19 7)  -- 7:07 p.m. next day
  := by sorry

end time_after_1567_minutes_l3586_358667


namespace eight_star_three_equals_fiftythree_l3586_358618

-- Define the operation ⋆
def star (a b : ℤ) : ℤ := 4*a + 6*b + 3

-- Theorem statement
theorem eight_star_three_equals_fiftythree : star 8 3 = 53 := by sorry

end eight_star_three_equals_fiftythree_l3586_358618


namespace polynomial_factor_problem_l3586_358675

theorem polynomial_factor_problem (b c : ℤ) :
  let p : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  (∃ q : ℝ → ℝ, (fun x ↦ x^4 + 8*x^2 + 49) = fun x ↦ p x * q x) ∧
  (∃ r : ℝ → ℝ, (fun x ↦ 2*x^4 + 5*x^2 + 32*x + 8) = fun x ↦ p x * r x) →
  p 1 = 24 := by
sorry

end polynomial_factor_problem_l3586_358675


namespace no_solution_for_four_divides_sum_of_squares_plus_one_l3586_358601

theorem no_solution_for_four_divides_sum_of_squares_plus_one :
  ∀ (a b : ℤ), ¬(4 ∣ a^2 + b^2 + 1) :=
by sorry

end no_solution_for_four_divides_sum_of_squares_plus_one_l3586_358601


namespace car_profit_percentage_l3586_358639

theorem car_profit_percentage (original_price : ℝ) (h : original_price > 0) :
  let discount_rate := 0.20
  let purchase_price := original_price * (1 - discount_rate)
  let sale_price := purchase_price * 2
  let profit := sale_price - original_price
  let profit_percentage := (profit / original_price) * 100
  profit_percentage = 60 := by sorry

end car_profit_percentage_l3586_358639


namespace four_digit_perfect_square_prefix_l3586_358626

theorem four_digit_perfect_square_prefix : ∃ (N : ℕ), 
  (1000 ≤ N ∧ N < 10000) ∧ 
  (∃ (k : ℕ), 4000000 + N = k^2) ∧
  (N = 4001 ∨ N = 8004) := by
  sorry

end four_digit_perfect_square_prefix_l3586_358626


namespace equilateral_triangle_area_l3586_358665

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := side * h / 2
  area = 4 * Real.sqrt 3 := by sorry

end equilateral_triangle_area_l3586_358665


namespace johns_daily_wage_without_bonus_l3586_358629

/-- John's work scenario -/
structure WorkScenario where
  regular_hours : ℕ
  bonus_hours : ℕ
  bonus_amount : ℕ
  hourly_rate_with_bonus : ℕ

/-- Calculates John's daily wage without bonus -/
def daily_wage_without_bonus (w : WorkScenario) : ℕ :=
  w.hourly_rate_with_bonus * w.bonus_hours - w.bonus_amount

/-- Theorem: John's daily wage without bonus is $80 -/
theorem johns_daily_wage_without_bonus :
  let w : WorkScenario := {
    regular_hours := 8,
    bonus_hours := 10,
    bonus_amount := 20,
    hourly_rate_with_bonus := 10
  }
  daily_wage_without_bonus w = 80 := by
  sorry


end johns_daily_wage_without_bonus_l3586_358629


namespace consecutive_even_integers_sum_l3586_358642

theorem consecutive_even_integers_sum (n : ℤ) :
  (n + (n + 6) = 160) →
  ((n + 2) + (n + 4) = 160) :=
by sorry

end consecutive_even_integers_sum_l3586_358642


namespace we_the_people_cows_l3586_358652

theorem we_the_people_cows (W : ℕ) : 
  W + (3 * W + 2) = 70 → W = 17 := by
  sorry

end we_the_people_cows_l3586_358652


namespace harkamal_fruit_purchase_cost_l3586_358697

/-- The total cost of fruits purchased by Harkamal -/
def total_cost (grapes_kg : ℕ) (grapes_price : ℕ) 
               (mangoes_kg : ℕ) (mangoes_price : ℕ)
               (apples_kg : ℕ) (apples_price : ℕ)
               (strawberries_kg : ℕ) (strawberries_price : ℕ) : ℕ :=
  grapes_kg * grapes_price + 
  mangoes_kg * mangoes_price + 
  apples_kg * apples_price + 
  strawberries_kg * strawberries_price

/-- Theorem stating the total cost of fruits purchased by Harkamal -/
theorem harkamal_fruit_purchase_cost : 
  total_cost 8 70 9 45 5 30 3 100 = 1415 := by
  sorry

end harkamal_fruit_purchase_cost_l3586_358697


namespace valid_selections_eq_sixteen_l3586_358687

/-- The number of people to choose from -/
def n : ℕ := 5

/-- The number of positions to fill (team leader and deputy team leader) -/
def k : ℕ := 2

/-- The number of ways to select k people from n people, where order matters -/
def permutations (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

/-- The number of ways to select a team leader and deputy team leader
    when one specific person cannot be the deputy team leader -/
def valid_selections (n k : ℕ) : ℕ :=
  permutations n k - permutations (n - 1) (k - 1)

/-- The main theorem: prove that the number of valid selections is 16 -/
theorem valid_selections_eq_sixteen : valid_selections n k = 16 := by
  sorry

end valid_selections_eq_sixteen_l3586_358687


namespace gcf_of_40_and_14_l3586_358649

theorem gcf_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 10 := by
sorry

end gcf_of_40_and_14_l3586_358649


namespace sticker_distribution_l3586_358636

/-- The number of ways to distribute n identical objects into k identical containers -/
def distribute_objects (n k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 25 ways to distribute 10 identical stickers onto 5 identical sheets of paper -/
theorem sticker_distribution : distribute_objects 10 5 = 25 := by sorry

end sticker_distribution_l3586_358636


namespace sufficient_material_for_box_l3586_358698

/-- A rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculate the volume of a box -/
def volume (b : Box) : ℕ :=
  b.length * b.width * b.height

/-- Calculate the surface area of a box -/
def surface_area (b : Box) : ℕ :=
  2 * (b.length * b.width + b.length * b.height + b.width * b.height)

/-- Theorem: There exists a box with volume at least 1995 and surface area exactly 958 -/
theorem sufficient_material_for_box : 
  ∃ (b : Box), volume b ≥ 1995 ∧ surface_area b = 958 :=
by
  sorry

end sufficient_material_for_box_l3586_358698


namespace range_of_m_l3586_358695

def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬¬(q m)) : m ≤ 1 :=
sorry

end range_of_m_l3586_358695


namespace cubs_cardinals_home_run_difference_l3586_358620

/-- The number of home runs scored by the Chicago Cubs in the game -/
def cubs_home_runs : ℕ := 2 + 1 + 2

/-- The number of home runs scored by the Cardinals in the game -/
def cardinals_home_runs : ℕ := 1 + 1

/-- The difference in home runs between the Cubs and the Cardinals -/
def home_run_difference : ℕ := cubs_home_runs - cardinals_home_runs

theorem cubs_cardinals_home_run_difference :
  home_run_difference = 3 :=
by sorry

end cubs_cardinals_home_run_difference_l3586_358620


namespace hannah_strawberry_harvest_l3586_358693

/-- Hannah's strawberry harvest problem -/
theorem hannah_strawberry_harvest :
  let daily_harvest : ℕ := 5
  let days_in_april : ℕ := 30
  let given_away : ℕ := 20
  let stolen : ℕ := 30
  let total_harvested : ℕ := daily_harvest * days_in_april
  let remaining_after_giving : ℕ := total_harvested - given_away
  let final_count : ℕ := remaining_after_giving - stolen
  final_count = 100 := by sorry

end hannah_strawberry_harvest_l3586_358693


namespace imaginary_part_of_complex_fraction_l3586_358634

/-- The imaginary part of (3-2i)/(1-i) is 1/2 -/
theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 - 2*I) / (1 - I)
  Complex.im z = 1/2 := by sorry

end imaginary_part_of_complex_fraction_l3586_358634


namespace customers_without_tip_greasy_spoon_tip_problem_l3586_358653

/-- The number of customers who didn't leave a tip at 'The Greasy Spoon' restaurant --/
theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  initial_customers + additional_customers - customers_with_tip

/-- Proof that 34 customers didn't leave a tip --/
theorem greasy_spoon_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end customers_without_tip_greasy_spoon_tip_problem_l3586_358653


namespace parallelogram_vector_l3586_358628

/-- A parallelogram on the complex plane -/
structure Parallelogram :=
  (A B C D : ℂ)
  (parallelogram_condition : (C - A) = (D - B))

/-- The theorem statement -/
theorem parallelogram_vector (ABCD : Parallelogram) 
  (hAC : ABCD.C - ABCD.A = 6 + 8*I) 
  (hBD : ABCD.D - ABCD.B = -4 + 6*I) : 
  ABCD.A - ABCD.D = -1 - 7*I :=
sorry

end parallelogram_vector_l3586_358628


namespace expression_evaluation_l3586_358666

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℤ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end expression_evaluation_l3586_358666


namespace equation_solution_l3586_358627

theorem equation_solution (y : ℝ) : (24 / 36 : ℝ) = Real.sqrt (y / 36) → y = 16 := by
  sorry

end equation_solution_l3586_358627


namespace power_of_point_outside_circle_l3586_358650

/-- Given a circle with radius R and a point M outside the circle at distance d from the center,
    prove that for any line through M intersecting the circle at A and B, MA * MB = d² - R² -/
theorem power_of_point_outside_circle (R d : ℝ) (h : 0 < R) (h' : R < d) :
  ∀ (M A B : ℝ × ℝ),
    ‖M - (0, 0)‖ = d →
    ‖A - (0, 0)‖ = R →
    ‖B - (0, 0)‖ = R →
    (∃ t : ℝ, A = M + t • (B - M)) →
    ‖M - A‖ * ‖M - B‖ = d^2 - R^2 :=
by sorry

end power_of_point_outside_circle_l3586_358650


namespace three_numbers_problem_l3586_358688

theorem three_numbers_problem (a b c : ℝ) :
  (a + 1) * (b + 1) * (c + 1) = a * b * c + 1 ∧
  (a + 2) * (b + 2) * (c + 2) = a * b * c + 2 →
  a = -1 ∧ b = -1 ∧ c = -1 := by
sorry

end three_numbers_problem_l3586_358688


namespace lemonade_mixture_problem_l3586_358674

theorem lemonade_mixture_problem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 →  -- Percentage of lemonade in first solution
  (0.6799999999999997 * (100 - x) + 0.32 * 55 = 72) →  -- Mixture equation
  x = 20 := by sorry

end lemonade_mixture_problem_l3586_358674


namespace perfect_square_trinomial_l3586_358656

/-- A quadratic trinomial of the form x^2 + kx + 9 is a perfect square if and only if k = ±6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := by
  sorry

end perfect_square_trinomial_l3586_358656


namespace insurance_cost_decade_l3586_358699

/-- Benjamin's yearly car insurance cost in dollars -/
def yearly_cost : ℕ := 3000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Benjamin's car insurance cost over a decade -/
theorem insurance_cost_decade : yearly_cost * decade = 30000 := by
  sorry

end insurance_cost_decade_l3586_358699


namespace negation_of_universal_proposition_l3586_358679

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_universal_proposition_l3586_358679


namespace multiple_of_six_between_14_and_30_l3586_358655

theorem multiple_of_six_between_14_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 196)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end multiple_of_six_between_14_and_30_l3586_358655


namespace mutual_correlation_sign_change_l3586_358602

/-- A stationary stochastic process -/
class StationaryStochasticProcess (X : ℝ → ℝ) : Prop where
  -- Add any necessary properties for a stationary stochastic process

/-- The derivative of a function -/
def derivative (f : ℝ → ℝ) : ℝ → ℝ :=
  fun t => sorry -- Definition of derivative

/-- Mutual correlation function of a process and its derivative -/
def mutualCorrelationFunction (X : ℝ → ℝ) (t₁ t₂ : ℝ) : ℝ :=
  sorry -- Definition of mutual correlation function

/-- Theorem: The mutual correlation function changes sign when arguments are swapped -/
theorem mutual_correlation_sign_change
  (X : ℝ → ℝ) [StationaryStochasticProcess X] (t₁ t₂ : ℝ) :
  mutualCorrelationFunction X t₁ t₂ = -mutualCorrelationFunction X t₂ t₁ :=
by sorry

end mutual_correlation_sign_change_l3586_358602


namespace larger_number_proof_l3586_358673

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 7 * S + 15) : 
  L = 1590 := by
  sorry

end larger_number_proof_l3586_358673


namespace luke_score_l3586_358689

/-- A trivia game where a player gains points each round -/
structure TriviaGame where
  points_per_round : ℕ
  num_rounds : ℕ

/-- Calculate the total points scored in a trivia game -/
def total_points (game : TriviaGame) : ℕ :=
  game.points_per_round * game.num_rounds

/-- Luke's trivia game -/
def luke_game : TriviaGame :=
  { points_per_round := 3
    num_rounds := 26 }

/-- Theorem: Luke scored 78 points in the trivia game -/
theorem luke_score : total_points luke_game = 78 := by
  sorry

end luke_score_l3586_358689


namespace area_ratio_of_specific_trapezoid_l3586_358643

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Length of the shorter base
  pq : ℝ
  -- Length of the longer base
  rs : ℝ
  -- Point where extended legs meet
  t : Point

/-- Calculates the ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def areaRatio (trap : ExtendedTrapezoid) : ℚ :=
  100 / 429

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_of_specific_trapezoid :
  ∃ (trap : ExtendedTrapezoid),
    trap.pq = 10 ∧ trap.rs = 23 ∧ areaRatio trap = 100 / 429 := by
  sorry

end area_ratio_of_specific_trapezoid_l3586_358643


namespace arthurs_walk_distance_l3586_358637

/-- Represents the distance walked in each direction --/
structure WalkDistance where
  east : ℕ
  north : ℕ
  west : ℕ

/-- Calculates the total distance walked in miles --/
def total_distance (walk : WalkDistance) (block_length : ℚ) : ℚ :=
  ((walk.east + walk.north + walk.west) : ℚ) * block_length

/-- Theorem: Arthur's walk totals 6.5 miles --/
theorem arthurs_walk_distance :
  let walk := WalkDistance.mk 8 15 3
  let block_length : ℚ := 1/4
  total_distance walk block_length = 13/2 := by
  sorry

end arthurs_walk_distance_l3586_358637


namespace smaller_of_two_numbers_l3586_358607

theorem smaller_of_two_numbers (x y a b c : ℝ) : 
  x > 0 → y > 0 → x * y = c → x^2 - b*x + a*y = 0 → 0 < a → a < b → 
  min x y = c / a :=
by sorry

end smaller_of_two_numbers_l3586_358607


namespace cone_lateral_surface_area_l3586_358669

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- base radius
  let h : ℝ := 4  -- height
  let l : ℝ := (r^2 + h^2).sqrt  -- slant height
  π * r * l = 15 * π :=
by sorry

end cone_lateral_surface_area_l3586_358669


namespace sets_and_range_l3586_358676

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 2 < x ∧ x < a}
def B : Set ℝ := {x | 3 / (x - 1) ≥ 1}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ≤ 1 ∨ x > 4}

-- State the theorem
theorem sets_and_range (a : ℝ) : 
  A a ⊆ complement_B → (a ≤ 1 ∨ a ≥ 2) := by sorry

end sets_and_range_l3586_358676


namespace common_point_intersection_l3586_358662

/-- The common point of intersection for a family of lines -/
def common_point : ℝ × ℝ := (-1, 1)

/-- The equation of lines in the family -/
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y = c

/-- The arithmetic progression condition -/
def arithmetic_progression (a b c d : ℝ) : Prop := b = a - d ∧ c = a - 2 * d

theorem common_point_intersection :
  ∀ (a b c d x y : ℝ),
    arithmetic_progression a b c d →
    (x, y) = common_point ↔ line_equation a b c x y :=
by sorry

end common_point_intersection_l3586_358662


namespace geometric_sequence_terms_l3586_358619

theorem geometric_sequence_terms (n : ℕ) (a₁ q : ℝ) 
  (h1 : a₁^3 * q^3 = 3)
  (h2 : a₁^3 * q^(3*n - 6) = 9)
  (h3 : a₁^n * q^(n*(n-1)/2) = 729) :
  n = 12 := by sorry

end geometric_sequence_terms_l3586_358619


namespace lexiCement_is_10_l3586_358654

/-- The amount of cement used for Lexi's street -/
def lexiCement : ℝ := sorry

/-- The amount of cement used for Tess's street -/
def tessCement : ℝ := 5.1

/-- The total amount of cement used -/
def totalCement : ℝ := 15.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexiCement_is_10 : lexiCement = 10 :=
by
  have h1 : lexiCement = totalCement - tessCement := sorry
  sorry


end lexiCement_is_10_l3586_358654


namespace min_value_f_l3586_358638

/-- Given that 2x^2 + 3xy + 2y^2 = 1, the minimum value of f(x, y) = x + y + xy is -9/8 -/
theorem min_value_f (x y : ℝ) (h : 2*x^2 + 3*x*y + 2*y^2 = 1) :
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (a b : ℝ), 2*a^2 + 3*a*b + 2*b^2 = 1 → a + b + a*b ≥ m :=
by sorry

end min_value_f_l3586_358638


namespace N_inverse_proof_l3586_358615

def N : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, -1; 4, -3, 2; -3, 5, 0]

theorem N_inverse_proof :
  let N_inv : Matrix (Fin 3) (Fin 3) ℝ := !![5/21, 5/14, -1/21; 3/14, 1/14, 5/42; -1/21, -19/42, 11/42]
  N * N_inv = 1 ∧ N_inv * N = 1 := by sorry

end N_inverse_proof_l3586_358615


namespace sum_of_x_coordinates_is_zero_l3586_358651

/-- A piecewise linear function composed of six line segments -/
def PiecewiseLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    (∀ x, (x ≤ x₁ ∨ x₁ < x ∧ x ≤ x₂ ∨ x₂ < x ∧ x ≤ x₃ ∨
           x₃ < x ∧ x ≤ x₄ ∨ x₄ < x ∧ x ≤ x₅ ∨ x₅ < x) →
      ∃ (a b : ℝ), ∀ y ∈ Set.Icc x₁ x, f y = a * y + b)

/-- The graph of g intersects with y = x - 1 at exactly three points -/
def ThreeIntersections (g : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem sum_of_x_coordinates_is_zero
  (g : ℝ → ℝ)
  (h₁ : PiecewiseLinearFunction g)
  (h₂ : ThreeIntersections g) :
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x, g x = x - 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
                    x₁ + x₂ + x₃ = 0 :=
sorry

end sum_of_x_coordinates_is_zero_l3586_358651


namespace only_one_divides_power_minus_one_l3586_358630

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ∣ (2^n - 1) ↔ n = 1 := by sorry

end only_one_divides_power_minus_one_l3586_358630


namespace plum_cost_l3586_358614

theorem plum_cost (total_fruits : ℕ) (total_cost : ℕ) (peach_cost : ℕ) (plum_count : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plum_count = 20 →
  ∃ (plum_cost : ℕ), plum_cost = 2 ∧ 
    plum_cost * plum_count + peach_cost * (total_fruits - plum_count) = total_cost :=
by sorry

end plum_cost_l3586_358614


namespace cylindrical_tank_capacity_l3586_358659

theorem cylindrical_tank_capacity (x : ℝ) 
  (h1 : 0.24 * x = 72) : x = 300 := by
  sorry

end cylindrical_tank_capacity_l3586_358659


namespace fourth_number_in_sequence_l3586_358613

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_fib : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 :=
sorry

end fourth_number_in_sequence_l3586_358613


namespace brick_width_l3586_358633

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: For a rectangular prism with length 10, height 3, and surface area 164, the width is 4 -/
theorem brick_width (l h : ℝ) (w : ℝ) (h₁ : l = 10) (h₂ : h = 3) (h₃ : surface_area l w h = 164) : w = 4 := by
  sorry

end brick_width_l3586_358633


namespace polynomial_factorization_l3586_358609

/-- For any a, b, and c, the expression a^4(b^3 - c^3) + b^4(c^3 - a^3) + c^4(a^3 - b^3)
    can be factored as (a - b)(b - c)(c - a) multiplied by a specific polynomial in a, b, and c. -/
theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^3*b + a^3*c + a^2*b^2 + a^2*b*c + a^2*c^2 + a*b^3 + a*b*c^2 + a*c^3 + b^3*c + b^2*c^2 + b*c^3) :=
by sorry

end polynomial_factorization_l3586_358609


namespace triangle_side_length_l3586_358686

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 →
  Real.sin A = Real.sqrt 3 / 2 →
  B = π / 6 →
  b = a * Real.sin B / Real.sin A →
  b = 1 := by
  sorry

end triangle_side_length_l3586_358686


namespace eleven_overtake_points_l3586_358624

/-- Represents a point on a circular track -/
structure TrackPoint where
  position : ℝ
  mk_mod : position ≥ 0 ∧ position < 1

/-- Represents the movement of a person on the track -/
structure Movement where
  speed : ℝ
  startPoint : TrackPoint

/-- Calculates the number of distinct overtake points -/
def countOvertakePoints (pedestrian : Movement) (cyclist : Movement) : ℕ :=
  sorry

/-- Main theorem: There are exactly 11 distinct overtake points -/
theorem eleven_overtake_points :
  ∀ (start : TrackPoint) (pedSpeed : ℝ),
    pedSpeed > 0 →
    let cycSpeed := pedSpeed * 1.55
    let pedestrian := Movement.mk pedSpeed start
    let cyclist := Movement.mk cycSpeed start
    countOvertakePoints pedestrian cyclist = 11 :=
  sorry

end eleven_overtake_points_l3586_358624


namespace folded_paper_length_l3586_358690

def paper_length : ℝ := 12

theorem folded_paper_length : 
  paper_length / 2 = 6 := by sorry

end folded_paper_length_l3586_358690


namespace q_factor_change_l3586_358664

/-- Given a function q defined in terms of w, h, and z, prove that when w is quadrupled,
    h is doubled, and z is tripled, q is multiplied by 5/18. -/
theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
    (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (5/18) * q w h z := by
  sorry

end q_factor_change_l3586_358664


namespace total_stickers_l3586_358671

theorem total_stickers (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 := by
sorry

end total_stickers_l3586_358671


namespace two_parts_problem_l3586_358657

theorem two_parts_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : x = 13 := by
  sorry

end two_parts_problem_l3586_358657


namespace farmer_apples_l3586_358692

theorem farmer_apples (initial_apples given_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : given_apples = 88) :
  initial_apples - given_apples = 39 := by
  sorry

end farmer_apples_l3586_358692


namespace probability_of_drawing_ball_two_l3586_358681

/-- A box containing labeled balls. -/
structure Box where
  balls : Finset ℕ
  labels_distinct : balls.card = balls.toList.length

/-- The probability of drawing a specific ball from a box. -/
def probability_of_drawing (box : Box) (ball : ℕ) : ℚ :=
  if ball ∈ box.balls then 1 / box.balls.card else 0

/-- Theorem stating the probability of drawing ball 2 from a box with 3 balls labeled 1, 2, and 3. -/
theorem probability_of_drawing_ball_two :
  ∃ (box : Box), box.balls = {1, 2, 3} ∧ probability_of_drawing box 2 = 1/3 := by
  sorry

end probability_of_drawing_ball_two_l3586_358681


namespace expand_product_l3586_358691

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l3586_358691


namespace cyclic_ratio_inequality_l3586_358647

theorem cyclic_ratio_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end cyclic_ratio_inequality_l3586_358647
