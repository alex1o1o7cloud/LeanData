import Mathlib

namespace NUMINAMATH_CALUDE_bouncy_balls_cost_l3420_342030

def red_packs : ℕ := 5
def yellow_packs : ℕ := 4
def blue_packs : ℕ := 3

def red_balls_per_pack : ℕ := 18
def yellow_balls_per_pack : ℕ := 15
def blue_balls_per_pack : ℕ := 12

def red_price : ℚ := 3/2
def yellow_price : ℚ := 5/4
def blue_price : ℚ := 1

def red_discount : ℚ := 1/10
def blue_discount : ℚ := 1/20

def total_cost (packs : ℕ) (balls_per_pack : ℕ) (price : ℚ) : ℚ :=
  (packs * balls_per_pack : ℚ) * price

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost * (1 - discount)

theorem bouncy_balls_cost :
  discounted_cost (total_cost red_packs red_balls_per_pack red_price) red_discount = 243/2 ∧
  total_cost yellow_packs yellow_balls_per_pack yellow_price = 75 ∧
  discounted_cost (total_cost blue_packs blue_balls_per_pack blue_price) blue_discount = 342/10 :=
by sorry

end NUMINAMATH_CALUDE_bouncy_balls_cost_l3420_342030


namespace NUMINAMATH_CALUDE_cassini_oval_properties_l3420_342097

-- Define the curve Γ
def Γ (m : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) * Real.sqrt ((x - 1)^2 + y^2) = m ∧ m > 0

-- Define a single-track curve
def SingleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, C x (f x)

-- Define a double-track curve
def DoubleTrackCurve (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (f g : ℝ → ℝ), (∀ x, C x (f x) ∨ C x (g x)) ∧ 
  (∃ x, f x ≠ g x)

-- The main theorem
theorem cassini_oval_properties :
  (∃ m : ℝ, m > 1 ∧ SingleTrackCurve (Γ m)) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ DoubleTrackCurve (Γ m)) := by
  sorry

end NUMINAMATH_CALUDE_cassini_oval_properties_l3420_342097


namespace NUMINAMATH_CALUDE_square_sum_halving_l3420_342053

theorem square_sum_halving (a b : ℕ) (h : a^2 + b^2 = 18728) :
  ∃ (n m : ℕ), n^2 + m^2 = 9364 ∧ ((n = 30 ∧ m = 92) ∨ (n = 92 ∧ m = 30)) :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_halving_l3420_342053


namespace NUMINAMATH_CALUDE_max_black_pieces_l3420_342085

/-- Represents a piece color -/
inductive Color
| Black
| White

/-- Represents the state of the circle -/
def CircleState := List Color

/-- Applies the rule to place new pieces between existing ones -/
def applyRule (state : CircleState) : CircleState :=
  sorry

/-- Removes the original pieces from the circle -/
def removeOriginal (state : CircleState) : CircleState :=
  sorry

/-- Counts the number of black pieces in the circle -/
def countBlack (state : CircleState) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of black pieces is 4 -/
theorem max_black_pieces (initial : CircleState) : 
  initial.length = 5 → 
  ∀ (n : Nat), countBlack (removeOriginal (applyRule initial)) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_black_pieces_l3420_342085


namespace NUMINAMATH_CALUDE_max_distinct_sums_diffs_is_64_l3420_342077

/-- Given a set of five natural numbers including 100, 200, and 400,
    this function returns the maximum number of distinct non-zero natural numbers
    that can be obtained by performing addition and subtraction operations,
    where each number is used at most once in each expression
    and at least two numbers are used. -/
def max_distinct_sums_diffs (a b : ℕ) : ℕ :=
  64

/-- Theorem stating that the maximum number of distinct non-zero natural numbers
    obtainable from the given set of numbers under the specified conditions is 64. -/
theorem max_distinct_sums_diffs_is_64 (a b : ℕ) :
  max_distinct_sums_diffs a b = 64 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_sums_diffs_is_64_l3420_342077


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3420_342042

-- Define the inequality function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

-- Define the solution set based on the value of a
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | 1 < x}
  else if 0 < a ∧ a < 1 then {x | 1 < x ∧ x < 1/a}
  else if a = 1 then ∅
  else if a > 1 then {x | 1/a < x ∧ x < 1}
  else {x | x < 1/a ∨ 1 < x}

-- Theorem statement
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | f a x < 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3420_342042


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3420_342027

/-- The number of ways to place n different balls into m different boxes --/
def placeWays (n m : ℕ) : ℕ := sorry

/-- The number of ways to place n different balls into m different boxes, leaving k boxes empty --/
def placeWaysWithEmpty (n m k : ℕ) : ℕ := sorry

theorem ball_placement_theorem :
  (placeWaysWithEmpty 4 4 1 = 144) ∧ (placeWaysWithEmpty 4 4 2 = 84) := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3420_342027


namespace NUMINAMATH_CALUDE_average_age_decrease_l3420_342059

theorem average_age_decrease (initial_average : ℝ) : 
  let original_total := 10 * initial_average
  let new_total := original_total - 44 + 14
  let new_average := new_total / 10
  initial_average - new_average = 3 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l3420_342059


namespace NUMINAMATH_CALUDE_previous_day_visitors_l3420_342008

def total_visitors : ℕ := 406
def current_day_visitors : ℕ := 132

theorem previous_day_visitors : 
  total_visitors - current_day_visitors = 274 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitors_l3420_342008


namespace NUMINAMATH_CALUDE_greatest_b_value_l3420_342024

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l3420_342024


namespace NUMINAMATH_CALUDE_apple_distribution_theorem_l3420_342067

/-- Represents the distribution of apples in bags -/
structure AppleDistribution where
  totalApples : Nat
  totalBags : Nat
  xApples : Nat
  threeAppleBags : Nat
  xAppleBags : Nat

/-- Checks if the apple distribution is valid -/
def isValidDistribution (d : AppleDistribution) : Prop :=
  d.totalApples = 109 ∧
  d.totalBags = 20 ∧
  d.threeAppleBags + d.xAppleBags = d.totalBags ∧
  d.xApples * d.xAppleBags + 3 * d.threeAppleBags = d.totalApples

/-- Theorem stating the possible values of x -/
theorem apple_distribution_theorem :
  ∀ d : AppleDistribution,
    isValidDistribution d →
    d.xApples = 10 ∨ d.xApples = 52 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_theorem_l3420_342067


namespace NUMINAMATH_CALUDE_local_max_range_l3420_342002

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem local_max_range (a : ℝ) :
  (∀ x, HasDerivAt f (f_deriv a x) x) →  -- f' is the derivative of f
  (∃ δ > 0, ∀ x, x ≠ a → |x - a| < δ → f x ≤ f a) →  -- local maximum at x = a
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_local_max_range_l3420_342002


namespace NUMINAMATH_CALUDE_trig_identity_l3420_342028

theorem trig_identity (x z : ℝ) : 
  (Real.sin x)^2 + (Real.sin (x + z))^2 - 2 * (Real.sin x) * (Real.sin z) * (Real.sin (x + z)) = (Real.sin z)^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3420_342028


namespace NUMINAMATH_CALUDE_min_value_of_even_function_l3420_342065

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a^2 - 1) * x - 3 * a

-- State the theorem
theorem min_value_of_even_function (a : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ∈ Set.range (f a)) →  -- domain of f is [4a+2, a^2+1]
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1) →  -- -1 is in the range of f
  (∀ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) → f a x ≥ -1) →  -- -1 is the minimum value
  (∃ x, x ∈ Set.Icc (4 * a + 2) (a^2 + 1) ∧ f a x = -1)  -- the minimum value of f(x) is -1
  := by sorry

end NUMINAMATH_CALUDE_min_value_of_even_function_l3420_342065


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3420_342006

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1/a > 1/b) ∧
  ¬(∀ a b : ℝ, 1/a > 1/b → b > a ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3420_342006


namespace NUMINAMATH_CALUDE_range_of_a_l3420_342020

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := a * x^3 - x^2 + 4*x + 3

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-2 : ℝ) 1 → f x a ≥ 0) → 
  a ∈ Set.Icc (-6 : ℝ) (-2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3420_342020


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3420_342081

theorem cubic_equation_roots (p q : ℝ) : 
  (3 * p^2 + 4 * p - 7 = 0) → 
  (3 * q^2 + 4 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3420_342081


namespace NUMINAMATH_CALUDE_alla_boris_meeting_point_l3420_342071

/-- The number of lanterns along the alley -/
def total_lanterns : ℕ := 400

/-- Alla's position when the first observation is made -/
def alla_position : ℕ := 55

/-- Boris's position when the first observation is made -/
def boris_position : ℕ := 321

/-- The meeting point of Alla and Boris -/
def meeting_point : ℕ := 163

/-- Theorem stating that Alla and Boris will meet at the calculated meeting point -/
theorem alla_boris_meeting_point :
  ∀ (alla_start boris_start : ℕ),
  alla_start = 1 ∧ boris_start = total_lanterns ∧
  alla_position > alla_start ∧ boris_position < boris_start ∧
  (alla_position - alla_start) / (total_lanterns - alla_position - (boris_start - boris_position)) =
  (meeting_point - alla_start) / (boris_start - meeting_point) :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_point_l3420_342071


namespace NUMINAMATH_CALUDE_find_set_C_l3420_342046

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 1, 2}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ A ∪ B a = A) :=
by sorry

end NUMINAMATH_CALUDE_find_set_C_l3420_342046


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l3420_342084

theorem pentagon_angle_sum (A B C D E : ℝ) (x y : ℝ) : 
  A = 34 → 
  B = 70 → 
  C = 30 → 
  D = 90 → 
  A + B + C + D + E = 540 → 
  E = 360 - x → 
  180 - y = 120 →
  x + y = 134 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l3420_342084


namespace NUMINAMATH_CALUDE_simplify_fraction_l3420_342068

theorem simplify_fraction : 20 * (9 / 14) * (1 / 18) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3420_342068


namespace NUMINAMATH_CALUDE_min_value_of_t_l3420_342058

theorem min_value_of_t (x y t : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 3 * x + y + x * y - 13 = 0) 
  (h2 : ∃ (t : ℝ), t ≥ 2 * y + x) : 
  ∀ t, t ≥ 2 * y + x → t ≥ 8 * Real.sqrt 2 - 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_t_l3420_342058


namespace NUMINAMATH_CALUDE_polynomial_value_l3420_342089

theorem polynomial_value : 
  let x : ℚ := 1/2
  2*x^2 - 5*x + x^2 + 4*x - 3*x^2 - 2 = -5/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l3420_342089


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3420_342090

theorem min_value_quadratic_form :
  (∀ x y z : ℝ, x^2 + x*y + y^2 + z^2 ≥ 0) ∧
  (∃ x y z : ℝ, x^2 + x*y + y^2 + z^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3420_342090


namespace NUMINAMATH_CALUDE_floor_length_calculation_l3420_342045

theorem floor_length_calculation (floor_width : ℝ) (strip_width : ℝ) (rug_area : ℝ) :
  floor_width = 20 →
  strip_width = 4 →
  rug_area = 204 →
  (floor_width - 2 * strip_width) * (floor_length - 2 * strip_width) = rug_area →
  floor_length = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_floor_length_calculation_l3420_342045


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3420_342040

theorem solution_implies_a_value (a : ℝ) : 
  (2 * 1 + 3 * a = -1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3420_342040


namespace NUMINAMATH_CALUDE_perimeter_of_modified_square_l3420_342096

/-- The perimeter of a figure ABFCDE formed by cutting a right triangle from a square and translating it -/
theorem perimeter_of_modified_square (side_length : ℝ) (triangle_leg : ℝ) 
  (h1 : side_length = 20)
  (h2 : triangle_leg = 12) : 
  let hypotenuse := Real.sqrt (2 * triangle_leg ^ 2)
  let perimeter := 2 * side_length + (side_length - triangle_leg) + hypotenuse + 2 * triangle_leg
  perimeter = 72 + 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_modified_square_l3420_342096


namespace NUMINAMATH_CALUDE_sector_central_angle_l3420_342032

/-- Given a sector with radius R and a perimeter equal to half the circumference of its circle,
    the central angle of the sector is (π - 2) radians. -/
theorem sector_central_angle (R : ℝ) (h : R > 0) : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 2 * π ∧ 
  (2 * R + R * θ = π * R) → θ = π - 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3420_342032


namespace NUMINAMATH_CALUDE_movie_tickets_difference_l3420_342019

theorem movie_tickets_difference (x y : ℕ) : 
  x + y = 30 →
  10 * x + 20 * y = 500 →
  y > x →
  y - x = 10 :=
by sorry

end NUMINAMATH_CALUDE_movie_tickets_difference_l3420_342019


namespace NUMINAMATH_CALUDE_kenneth_remaining_money_l3420_342056

-- Define the initial amount Kenneth has
def initial_amount : ℕ := 50

-- Define the number of baguettes and bottles of water
def num_baguettes : ℕ := 2
def num_water_bottles : ℕ := 2

-- Define the cost of each baguette and bottle of water
def cost_baguette : ℕ := 2
def cost_water : ℕ := 1

-- Define the total cost of purchases
def total_cost : ℕ := num_baguettes * cost_baguette + num_water_bottles * cost_water

-- Define the remaining money after purchases
def remaining_money : ℕ := initial_amount - total_cost

-- Theorem statement
theorem kenneth_remaining_money :
  remaining_money = 44 :=
by sorry

end NUMINAMATH_CALUDE_kenneth_remaining_money_l3420_342056


namespace NUMINAMATH_CALUDE_black_ball_probability_l3420_342001

/-- Given a bag of 100 balls with 45 red balls and a probability of 0.23 for drawing a white ball,
    the probability of drawing a black ball is 0.32. -/
theorem black_ball_probability
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_prob : ℝ)
  (h_total : total_balls = 100)
  (h_red : red_balls = 45)
  (h_white_prob : white_prob = 0.23)
  : (total_balls - red_balls - (white_prob * total_balls)) / total_balls = 0.32 := by
  sorry


end NUMINAMATH_CALUDE_black_ball_probability_l3420_342001


namespace NUMINAMATH_CALUDE_prob_one_white_correct_prob_red_given_red_correct_l3420_342098

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing exactly one white ball
def prob_one_white : ℚ := 3/5

-- Define the probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
def prob_red_given_red : ℚ := 3/5

-- Theorem 1: Probability of drawing exactly one white ball
theorem prob_one_white_correct :
  (Nat.choose white_balls 1 * Nat.choose red_balls (balls_drawn - 1)) / 
  Nat.choose total_balls balls_drawn = prob_one_white := by sorry

-- Theorem 2: Probability of drawing a red ball on the second draw given a red ball was drawn on the first draw
theorem prob_red_given_red_correct :
  (red_balls - 1) / (total_balls - 1) = prob_red_given_red := by sorry

end NUMINAMATH_CALUDE_prob_one_white_correct_prob_red_given_red_correct_l3420_342098


namespace NUMINAMATH_CALUDE_natural_subset_rational_l3420_342011

theorem natural_subset_rational :
  (∀ x : ℕ, ∃ y : ℚ, (x : ℚ) = y) ∧
  (∃ z : ℚ, ∀ w : ℕ, (w : ℚ) ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_natural_subset_rational_l3420_342011


namespace NUMINAMATH_CALUDE_power_function_m_value_l3420_342052

/-- A function y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant -/
def is_power_and_increasing (m : ℝ) : Prop :=
  (m^2 + 2*m - 2 = 1) ∧ (m > 0)

/-- If y = (m^2 + 2m - 2)x^m is a power function and increasing in the first quadrant, then m = 1 -/
theorem power_function_m_value :
  ∀ m : ℝ, is_power_and_increasing m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l3420_342052


namespace NUMINAMATH_CALUDE_complement_of_union_l3420_342043

def U : Set ℕ := {0, 1, 2, 3, 4, 5}

def A : Set ℕ := {1, 2}

def B : Set ℕ := {x ∈ U | x^2 - 5*x + 4 < 0}

theorem complement_of_union (U A B : Set ℕ) :
  (A ∪ B)ᶜ = {0, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3420_342043


namespace NUMINAMATH_CALUDE_unique_b_value_l3420_342025

theorem unique_b_value : ∃! b : ℝ, ∃ x : ℝ, x^2 + b*x + 1 = 0 ∧ x^2 + x + b = 0 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l3420_342025


namespace NUMINAMATH_CALUDE_susie_earnings_l3420_342034

/-- Calculates the total earnings from selling pizza slices and whole pizzas --/
def calculate_earnings (price_per_slice : ℚ) (price_per_whole : ℚ) (slices_sold : ℕ) (whole_sold : ℕ) : ℚ :=
  price_per_slice * slices_sold + price_per_whole * whole_sold

/-- Proves that Susie's earnings are $117 given the specified prices and sales --/
theorem susie_earnings : 
  let price_per_slice : ℚ := 3
  let price_per_whole : ℚ := 15
  let slices_sold : ℕ := 24
  let whole_sold : ℕ := 3
  calculate_earnings price_per_slice price_per_whole slices_sold whole_sold = 117 := by
  sorry

end NUMINAMATH_CALUDE_susie_earnings_l3420_342034


namespace NUMINAMATH_CALUDE_unique_number_exists_l3420_342094

theorem unique_number_exists : ∃! N : ℕ, 
  (∃ Q : ℕ, N = 11 * Q) ∧ 
  (N / 11 + N + 11 = 71) := by
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l3420_342094


namespace NUMINAMATH_CALUDE_weight_of_a_l3420_342095

theorem weight_of_a (a b c d : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  ∃ e : ℝ, e = d + 5 ∧ (b + c + d + e) / 4 = 79 →
  a = 77 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l3420_342095


namespace NUMINAMATH_CALUDE_max_quotient_four_digit_number_l3420_342031

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem max_quotient_four_digit_number (a b c d : ℕ) 
  (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (hd : is_digit d)
  (hdiff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤ 329.2 := by
  sorry

end NUMINAMATH_CALUDE_max_quotient_four_digit_number_l3420_342031


namespace NUMINAMATH_CALUDE_project_completion_days_l3420_342021

/-- Represents the time in days for a worker to complete the project alone -/
structure WorkerRate where
  days : ℕ
  days_pos : days > 0

/-- Represents the project completion scenario -/
structure ProjectCompletion where
  worker_a : WorkerRate
  worker_b : WorkerRate
  worker_c : WorkerRate
  a_quit_before_end : ℕ

/-- Calculates the total days to complete the project -/
def total_days (p : ProjectCompletion) : ℕ := 
  sorry

/-- Theorem stating that the project will be completed in 18 days -/
theorem project_completion_days (p : ProjectCompletion) 
  (h1 : p.worker_a.days = 20)
  (h2 : p.worker_b.days = 30)
  (h3 : p.worker_c.days = 40)
  (h4 : p.a_quit_before_end = 18) :
  total_days p = 18 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_days_l3420_342021


namespace NUMINAMATH_CALUDE_first_digit_1025_base12_l3420_342004

/-- The first digit of a number in a given base -/
def firstDigitInBase (n : ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Theorem: The first digit of 1025 (base 10) in base 12 is 7 -/
theorem first_digit_1025_base12 : firstDigitInBase 1025 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_1025_base12_l3420_342004


namespace NUMINAMATH_CALUDE_evaluate_expression_l3420_342037

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 5) : y * (y - 3 * x) = -5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3420_342037


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_eight_l3420_342088

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  outerWidth : ℝ
  outerHeight : ℝ
  borderWidth : ℝ

/-- Calculate the area of the frame -/
def frameArea (frame : PictureFrame) : ℝ :=
  frame.outerWidth * frame.outerHeight - (frame.outerWidth - 2 * frame.borderWidth) * (frame.outerHeight - 2 * frame.borderWidth)

/-- Calculate the sum of the interior edge lengths -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * ((frame.outerWidth - 2 * frame.borderWidth) + (frame.outerHeight - 2 * frame.borderWidth))

/-- Theorem: The sum of interior edges is 8 inches for a frame with given properties -/
theorem interior_edge_sum_is_eight (frame : PictureFrame) 
  (h1 : frame.borderWidth = 2)
  (h2 : frameArea frame = 32)
  (h3 : frame.outerWidth = 7) : 
  interiorEdgeSum frame = 8 := by
  sorry


end NUMINAMATH_CALUDE_interior_edge_sum_is_eight_l3420_342088


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3420_342054

theorem algebraic_expression_value (x y : ℝ) : 
  5 * x^2 - 4 * x * y - 1 = -11 → -10 * x^2 + 8 * x * y + 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3420_342054


namespace NUMINAMATH_CALUDE_number_equation_solution_l3420_342038

theorem number_equation_solution : ∃ n : ℝ, 7 * n = 3 * n + 12 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3420_342038


namespace NUMINAMATH_CALUDE_closest_point_l3420_342079

/-- The vector v as a function of t -/
def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 1 + 5*t
  | 1 => -2 + 4*t
  | 2 => -4 - 2*t

/-- The vector a -/
def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 2
  | 2 => 6

/-- The direction vector of v -/
def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 4
  | 2 => -2

/-- Theorem: The value of t that minimizes the distance between v and a is 2/15 -/
theorem closest_point : 
  (∀ t : ℝ, (v t - a) • direction = 0 → t = 2/15) ∧ 
  (v (2/15) - a) • direction = 0 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_l3420_342079


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3420_342060

def polynomial (y : ℝ) : ℝ := y^5 - 8*y^4 + 12*y^3 + 25*y^2 - 40*y + 24

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, polynomial = (λ y => (y - 4) * q y + 8) := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3420_342060


namespace NUMINAMATH_CALUDE_arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l3420_342015

/-- The number of people standing in a row -/
def n : ℕ := 7

/-- The number of arrangements with A and B at the ends -/
def arrangements_ends : ℕ := 240

/-- The number of arrangements with A, B, and C together -/
def arrangements_together : ℕ := 720

/-- The number of arrangements with A not at beginning and B not at end -/
def arrangements_not_ends : ℕ := 3720

/-- Theorem for the number of arrangements with A and B at the ends -/
theorem arrangements_ends_correct : 
  arrangements_ends = 2 * Nat.factorial (n - 2) := by sorry

/-- Theorem for the number of arrangements with A, B, and C together -/
theorem arrangements_together_correct : 
  arrangements_together = 6 * Nat.factorial (n - 3) := by sorry

/-- Theorem for the number of arrangements with A not at beginning and B not at end -/
theorem arrangements_not_ends_correct : 
  arrangements_not_ends = Nat.factorial n - 2 * Nat.factorial (n - 1) + Nat.factorial (n - 2) := by sorry

end NUMINAMATH_CALUDE_arrangements_ends_correct_arrangements_together_correct_arrangements_not_ends_correct_l3420_342015


namespace NUMINAMATH_CALUDE_division_of_decimals_l3420_342083

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3420_342083


namespace NUMINAMATH_CALUDE_expression_evaluation_l3420_342035

theorem expression_evaluation :
  (2^2003 * 3^2002 * 5) / 6^2003 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3420_342035


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3420_342074

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x

-- Theorem statement
theorem tangent_line_at_negative_one :
  ∃ (m b : ℝ), 
    (f' (-1) = m) ∧ 
    (f (-1) = -2) ∧ 
    (∀ x y : ℝ, y = m * (x + 1) - 2 ↔ m * x - y + b = 0) ∧
    (2 * x - y = 0 ↔ m * x - y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3420_342074


namespace NUMINAMATH_CALUDE_prize_orders_count_l3420_342041

/-- Represents a bowling tournament with 6 players. -/
structure BowlingTournament :=
  (players : Fin 6)
  (match_structure : List (Fin 6 × Fin 6))
  (special_rule : Bool)

/-- Calculates the number of possible prize orders in the tournament. -/
def count_prize_orders (tournament : BowlingTournament) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 32 possible prize orders. -/
theorem prize_orders_count :
  ∀ (tournament : BowlingTournament),
  count_prize_orders tournament = 32 :=
sorry

end NUMINAMATH_CALUDE_prize_orders_count_l3420_342041


namespace NUMINAMATH_CALUDE_scorpion_millipede_calculation_l3420_342075

/-- Calculates the number of additional millipedes needed to reach a daily segment requirement -/
theorem scorpion_millipede_calculation 
  (daily_requirement : ℕ) 
  (eaten_millipede_segments : ℕ) 
  (eaten_long_millipedes : ℕ) 
  (additional_millipede_segments : ℕ) 
  (h1 : daily_requirement = 800)
  (h2 : eaten_millipede_segments = 60)
  (h3 : eaten_long_millipedes = 2)
  (h4 : additional_millipede_segments = 50) :
  (daily_requirement - (eaten_millipede_segments + eaten_long_millipedes * eaten_millipede_segments * 2)) / additional_millipede_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_millipede_calculation_l3420_342075


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3420_342078

/-- Given an ellipse with the following properties:
    1. The chord passing through the focus and perpendicular to the major axis has a length of √2
    2. The distance from the focus to the corresponding directrix is 1
    This theorem states that the eccentricity of the ellipse is √2/2 -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 2 * b^2 / a = Real.sqrt 2) (h4 : a^2 / c - c = 1) : 
  c / a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3420_342078


namespace NUMINAMATH_CALUDE_packaging_cost_per_cake_l3420_342029

/-- Proves that the cost of packaging per cake is $1 -/
theorem packaging_cost_per_cake
  (ingredient_cost_two_cakes : ℝ)
  (selling_price_per_cake : ℝ)
  (profit_per_cake : ℝ)
  (h1 : ingredient_cost_two_cakes = 12)
  (h2 : selling_price_per_cake = 15)
  (h3 : profit_per_cake = 8) :
  selling_price_per_cake - profit_per_cake - (ingredient_cost_two_cakes / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_packaging_cost_per_cake_l3420_342029


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_three_l3420_342086

theorem trigonometric_sum_equals_sqrt_three (x : ℝ) 
  (h : Real.tan (4 * x) = Real.sqrt 3 / 3) : 
  (Real.sin (4 * x)) / (Real.cos (8 * x) * Real.cos (4 * x)) + 
  (Real.sin (2 * x)) / (Real.cos (4 * x) * Real.cos (2 * x)) + 
  (Real.sin x) / (Real.cos (2 * x) * Real.cos x) + 
  (Real.sin x) / (Real.cos x) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_sqrt_three_l3420_342086


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3420_342017

/-- Given a triangle ABC and a point M, prove that a certain line always passes through a fixed point -/
theorem fixed_point_theorem (a b c t m : ℝ) : 
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (t, m)
  let D : ℝ × ℝ := ((b + c) / 3, a / 3)  -- Centroid
  let E : ℝ × ℝ := ((t + b) / 2, m / 2)  -- Midpoint of MB
  let F : ℝ × ℝ := ((t + c) / 2, m / 2)  -- Midpoint of MC
  let P : ℝ × ℝ := ((t + b) / 2, a * (1 - (t + b) / (2 * b)))  -- Intersection of AB and perpendicular through E
  let Q : ℝ × ℝ := ((t + c) / 2, a * (1 - (t + c) / (2 * c)))  -- Intersection of AC and perpendicular through F
  let slope_PQ : ℝ := (a * t) / (b * c)
  let perpendicular_slope : ℝ := -b * c / (a * t)
  True → ∃ k : ℝ, (0, m + b * c / a) = (t + k, m + k * perpendicular_slope) :=
by
  sorry


end NUMINAMATH_CALUDE_fixed_point_theorem_l3420_342017


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3420_342069

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 8 / 1 → a / b = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3420_342069


namespace NUMINAMATH_CALUDE_second_day_distance_l3420_342080

def distance_day1 : ℝ := 240
def speed : ℝ := 60
def time_difference : ℝ := 3

theorem second_day_distance :
  ∃ (distance_day2 : ℝ),
    distance_day2 / speed = distance_day1 / speed + time_difference ∧
    distance_day2 = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_second_day_distance_l3420_342080


namespace NUMINAMATH_CALUDE_problem_solution_l3420_342070

theorem problem_solution (x : ℝ) : (1 / (2 + 3)) * (1 / (3 + 4)) = 1 / (x + 5) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3420_342070


namespace NUMINAMATH_CALUDE_factorization_equality_l3420_342063

theorem factorization_equality (a : ℝ) : 2*a^2 + 4*a + 2 = 2*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3420_342063


namespace NUMINAMATH_CALUDE_same_solution_equations_l3420_342039

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 11 = 5) ∧ (c * x - 14 = -4) → c = -5 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l3420_342039


namespace NUMINAMATH_CALUDE_M_geq_N_l3420_342003

theorem M_geq_N (a : ℝ) : 2 * a * (a - 2) + 3 ≥ (a - 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_geq_N_l3420_342003


namespace NUMINAMATH_CALUDE_min_orange_weight_l3420_342007

theorem min_orange_weight (a o : ℝ) 
  (h1 : a ≥ 8 + 3 * o) 
  (h2 : a ≤ 4 * o) : 
  o ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_orange_weight_l3420_342007


namespace NUMINAMATH_CALUDE_population_growth_s_curve_l3420_342036

/-- Represents the population size at a given time -/
def PopulationSize := ℝ

/-- Represents time -/
def Time := ℝ

/-- Represents the carrying capacity of the environment -/
def CarryingCapacity := ℝ

/-- Represents the growth rate of the population -/
def GrowthRate := ℝ

/-- A function that models population growth over time -/
def populationGrowthModel (t : Time) (K : CarryingCapacity) (r : GrowthRate) : PopulationSize :=
  sorry

/-- Predicate that checks if a function exhibits an S-curve pattern -/
def isSCurve (f : Time → PopulationSize) : Prop :=
  sorry

/-- Theorem stating that population growth often exhibits an S-curve in nature -/
theorem population_growth_s_curve 
  (limitedEnvironment : CarryingCapacity → Prop)
  (environmentalFactors : (Time → PopulationSize) → Prop) :
  ∃ (K : CarryingCapacity) (r : GrowthRate),
    limitedEnvironment K ∧ 
    environmentalFactors (populationGrowthModel · K r) ∧
    isSCurve (populationGrowthModel · K r) :=
  sorry

end NUMINAMATH_CALUDE_population_growth_s_curve_l3420_342036


namespace NUMINAMATH_CALUDE_train_passengers_l3420_342012

theorem train_passengers (adults_first : ℕ) (children_first : ℕ) 
  (adults_second : ℕ) (children_second : ℕ) (got_off : ℕ) (total : ℕ) : 
  children_first = adults_first - 17 →
  adults_second = 57 →
  children_second = 18 →
  got_off = 44 →
  total = 502 →
  adults_first + children_first + adults_second + children_second - got_off = total →
  adults_first = 244 := by
sorry

end NUMINAMATH_CALUDE_train_passengers_l3420_342012


namespace NUMINAMATH_CALUDE_triangle_inequality_l3420_342033

/-- Given two triangles ABC and A₁B₁C₁, where b₁ and c₁ have areas S and S₁ respectively,
    prove the inequality and its equality condition. -/
theorem triangle_inequality (a b c a₁ b₁ c₁ S S₁ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a₁ > 0 ∧ b₁ > 0 ∧ c₁ > 0 ∧ S > 0 ∧ S₁ > 0 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ →
  a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) ≥ 16 * S * S₁ ∧
  (a₁^2 * (-a^2 + b^2 + c^2) + b₁^2 * (a^2 - b^2 + c^2) + c₁^2 * (a^2 + b^2 - c^2) = 16 * S * S₁ ↔
   ∃ k : ℝ, k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3420_342033


namespace NUMINAMATH_CALUDE_symmetric_points_on_ellipse_l3420_342066

/-- Given an ellipse C and a line l, prove the range of m for which there are always two points on C symmetric with respect to l -/
theorem symmetric_points_on_ellipse (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧ 
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧
    (y₁ = 4*x₁ + m) ∧ 
    (y₂ = 4*x₂ + m) ∧ 
    (x₁ ≠ x₂) ∧
    (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2 ∧ y₀ = 4*x₀ + m)) ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_ellipse_l3420_342066


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3420_342022

/-- Given an inverse proportion function y = (k+1)/x passing through the point (1, -2),
    prove that the value of k is -3. -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = (k + 1) / x) ∧ f 1 = -2) → k = -3 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3420_342022


namespace NUMINAMATH_CALUDE_sports_club_members_l3420_342048

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 17)
  (h2 : tennis = 19)
  (h3 : both = 8)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l3420_342048


namespace NUMINAMATH_CALUDE_equal_selection_probability_l3420_342093

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents the probability of an individual being selected in a sampling method -/
def selectionProbability (method : SamplingMethod) (individual : ℕ) : ℝ := sorry

/-- The theorem stating that all three sampling methods have equal selection probability for all individuals -/
theorem equal_selection_probability (population : Finset ℕ) :
  ∀ (method : SamplingMethod) (i j : ℕ), i ∈ population → j ∈ population →
    selectionProbability method i = selectionProbability method j :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l3420_342093


namespace NUMINAMATH_CALUDE_third_player_win_probability_l3420_342061

/-- Represents a fair six-sided die --/
def FairDie : Finset ℕ := Finset.range 6

/-- The probability of rolling a 6 on a fair die --/
def probWin : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a fair die --/
def probLose : ℚ := 1 - probWin

/-- The number of players --/
def numPlayers : ℕ := 3

/-- The probability that the third player wins the game --/
def probThirdPlayerWins : ℚ := 1 / 91

theorem third_player_win_probability :
  probThirdPlayerWins = (probWin^numPlayers) / (1 - probLose^numPlayers) :=
by sorry

end NUMINAMATH_CALUDE_third_player_win_probability_l3420_342061


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l3420_342055

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l3420_342055


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3420_342064

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry about the origin -/
def symmetricAboutOrigin (c1 c2 : Circle) : Prop :=
  c2.center = (-c1.center.1, -c1.center.2) ∧ c2.radius = c1.radius

/-- The main theorem -/
theorem symmetric_circle_equation (c1 c2 : Circle) :
  c1.equation = λ x y => (x + 2)^2 + y^2 = 5 →
  symmetricAboutOrigin c1 c2 →
  c2.equation = λ x y => (x - 2)^2 + y^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3420_342064


namespace NUMINAMATH_CALUDE_alexanders_apples_l3420_342026

/-- Prove that Alexander bought 5 apples given the conditions of his shopping trip -/
theorem alexanders_apples : 
  ∀ (apple_price orange_price total_spent num_oranges : ℕ),
    apple_price = 1 →
    orange_price = 2 →
    num_oranges = 2 →
    total_spent = 9 →
    ∃ (num_apples : ℕ), 
      num_apples * apple_price + num_oranges * orange_price = total_spent ∧
      num_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_alexanders_apples_l3420_342026


namespace NUMINAMATH_CALUDE_gunther_free_time_l3420_342014

/-- Represents the time in minutes for each cleaning task and the total free time --/
structure CleaningTime where
  vacuuming : ℕ
  dusting : ℕ
  mopping : ℕ
  brushing_per_cat : ℕ
  num_cats : ℕ
  free_time : ℕ

/-- Calculates the remaining free time after cleaning --/
def remaining_free_time (ct : CleaningTime) : ℕ :=
  ct.free_time - (ct.vacuuming + ct.dusting + ct.mopping + ct.brushing_per_cat * ct.num_cats)

/-- Theorem stating that Gunther will have 30 minutes of free time left --/
theorem gunther_free_time :
  let ct : CleaningTime := {
    vacuuming := 45,
    dusting := 60,
    mopping := 30,
    brushing_per_cat := 5,
    num_cats := 3,
    free_time := 180
  }
  remaining_free_time ct = 30 := by
  sorry


end NUMINAMATH_CALUDE_gunther_free_time_l3420_342014


namespace NUMINAMATH_CALUDE_proposition_logic_proof_l3420_342082

theorem proposition_logic_proof (p q : Prop) 
  (hp : p ↔ (3 ≥ 3)) 
  (hq : q ↔ (3 > 4)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_proof_l3420_342082


namespace NUMINAMATH_CALUDE_nail_trimming_sounds_l3420_342005

/-- Represents the number of customers --/
def num_customers : Nat := 3

/-- Represents the number of appendages per customer --/
def appendages_per_customer : Nat := 4

/-- Represents the number of nails per appendage --/
def nails_per_appendage : Nat := 4

/-- Calculates the total number of nail trimming sounds --/
def total_nail_sounds : Nat :=
  num_customers * appendages_per_customer * nails_per_appendage

/-- Theorem stating that the total number of nail trimming sounds is 48 --/
theorem nail_trimming_sounds :
  total_nail_sounds = 48 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_sounds_l3420_342005


namespace NUMINAMATH_CALUDE_cost_45_roses_l3420_342099

/-- The cost of a bouquet is directly proportional to the number of roses it contains -/
axiom price_proportional_to_roses (n : ℕ) (price : ℚ) : n > 0 → price > 0 → ∃ k : ℚ, k > 0 ∧ price = k * n

/-- The cost of a bouquet with 15 roses -/
def cost_15 : ℚ := 25

/-- The number of roses in the first bouquet -/
def roses_15 : ℕ := 15

/-- The number of roses in the second bouquet -/
def roses_45 : ℕ := 45

/-- The theorem to prove -/
theorem cost_45_roses : 
  ∃ (k : ℚ), k > 0 ∧ cost_15 = k * roses_15 → k * roses_45 = 75 :=
sorry

end NUMINAMATH_CALUDE_cost_45_roses_l3420_342099


namespace NUMINAMATH_CALUDE_min_sum_of_product_1806_l3420_342010

theorem min_sum_of_product_1806 (a b c : ℕ+) : 
  a * b * c = 1806 → 
  (Even a ∨ Even b ∨ Even c) → 
  (∀ x y z : ℕ+, x * y * z = 1806 → (Even x ∨ Even y ∨ Even z) → a + b + c ≤ x + y + z) →
  a + b + c = 112 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1806_l3420_342010


namespace NUMINAMATH_CALUDE_min_value_theorem_l3420_342044

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3420_342044


namespace NUMINAMATH_CALUDE_complex_number_problems_l3420_342018

open Complex

theorem complex_number_problems (z₁ z₂ z : ℂ) (b : ℝ) :
  z₁ = 1 - I ∧ z₂ = 4 + 6 * I ∧ z = 1 + b * I ∧ (z + z₁).im = 0 →
  z₂ / z₁ = -1 + 5 * I ∧ abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problems_l3420_342018


namespace NUMINAMATH_CALUDE_right_triangle_area_l3420_342013

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 225) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3420_342013


namespace NUMINAMATH_CALUDE_prime_differences_l3420_342076

theorem prime_differences (x y : ℝ) 
  (h1 : Prime (x - y))
  (h2 : Prime (x^2 - y^2))
  (h3 : Prime (x^3 - y^3)) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_differences_l3420_342076


namespace NUMINAMATH_CALUDE_largest_n_for_integer_differences_l3420_342091

theorem largest_n_for_integer_differences : ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ),
  (∀ k : ℕ, k ≤ 9 → 
    (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |x₁ - x₂| ∨ k = |x₁ - x₃| ∨ k = |x₁ - x₄| ∨ 
                               k = |x₂ - x₃| ∨ k = |x₂ - x₄| ∨ k = |x₃ - x₄| ∨
                               k = |y₁ - y₂| ∨ k = |y₁ - y₃| ∨ k = |y₁ - y₄| ∨ 
                               k = |y₂ - y₃| ∨ k = |y₂ - y₄| ∨ k = |y₃ - y₄|)) ∧
  (∀ n : ℕ, n > 9 → 
    ¬∃ (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℤ),
      ∀ k : ℕ, k ≤ n → 
        (∃ (i j : Fin 4), i < j ∧ (k : ℤ) = |a₁ - a₂| ∨ k = |a₁ - a₃| ∨ k = |a₁ - a₄| ∨ 
                                   k = |a₂ - a₃| ∨ k = |a₂ - a₄| ∨ k = |a₃ - a₄| ∨
                                   k = |b₁ - b₂| ∨ k = |b₁ - b₃| ∨ k = |b₁ - b₄| ∨ 
                                   k = |b₂ - b₃| ∨ k = |b₂ - b₄| ∨ k = |b₃ - b₄|)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_integer_differences_l3420_342091


namespace NUMINAMATH_CALUDE_phil_bought_cards_for_52_weeks_l3420_342087

/-- Represents the number of weeks Phil bought baseball card packs --/
def weeks_buying_cards (cards_per_pack : ℕ) (cards_after_fire : ℕ) : ℕ :=
  (2 * cards_after_fire) / cards_per_pack

/-- Theorem stating that Phil bought cards for 52 weeks --/
theorem phil_bought_cards_for_52_weeks :
  weeks_buying_cards 20 520 = 52 := by
  sorry

end NUMINAMATH_CALUDE_phil_bought_cards_for_52_weeks_l3420_342087


namespace NUMINAMATH_CALUDE_cube_root_of_negative_64_l3420_342057

theorem cube_root_of_negative_64 : ∃ x : ℝ, x^3 = -64 ∧ x = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_64_l3420_342057


namespace NUMINAMATH_CALUDE_original_savings_proof_l3420_342092

def lindas_savings : ℝ := 880
def tv_cost : ℝ := 220

theorem original_savings_proof :
  (1 / 4 : ℝ) * lindas_savings = tv_cost →
  lindas_savings = 880 := by
sorry

end NUMINAMATH_CALUDE_original_savings_proof_l3420_342092


namespace NUMINAMATH_CALUDE_oil_in_barrels_l3420_342023

theorem oil_in_barrels (barrel_a barrel_b : ℚ) : 
  barrel_a = 3/4 → 
  barrel_b = barrel_a + 1/10 → 
  barrel_a + barrel_b = 8/5 := by
sorry

end NUMINAMATH_CALUDE_oil_in_barrels_l3420_342023


namespace NUMINAMATH_CALUDE_abs_sine_period_l3420_342051

-- Define the sine function and its period
noncomputable def sine_period : ℝ := 2 * Real.pi

-- Define the property that sine has this period
axiom sine_periodic (x : ℝ) : Real.sin (x + sine_period) = Real.sin x

-- Define the property that taking absolute value halves the period
axiom abs_halves_period {f : ℝ → ℝ} {p : ℝ} (h : ∀ x, f (x + p) = f x) :
  ∀ x, |f (x + p/2)| = |f x|

-- State the theorem
theorem abs_sine_period : 
  ∃ p : ℝ, p > 0 ∧ p = Real.pi ∧ ∀ x, |Real.sin (x + p)| = |Real.sin x| ∧
  ∀ q, q > 0 → (∀ x, |Real.sin (x + q)| = |Real.sin x|) → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_abs_sine_period_l3420_342051


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3420_342049

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 34 → 
  x = Real.sqrt a - b → 
  a + b = 64 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3420_342049


namespace NUMINAMATH_CALUDE_characterize_S_l3420_342000

/-- The function f(A, B, C) = A^3 + B^3 + C^3 - 3ABC -/
def f (A B C : ℕ) : ℤ := A^3 + B^3 + C^3 - 3 * A * B * C

/-- The set of all possible values of f(A, B, C) -/
def S : Set ℤ := {n | ∃ (A B C : ℕ), f A B C = n}

/-- The theorem stating the characterization of S -/
theorem characterize_S : S = {n : ℤ | n ≥ 0 ∧ n % 9 ≠ 3 ∧ n % 9 ≠ 6} := by sorry

end NUMINAMATH_CALUDE_characterize_S_l3420_342000


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l3420_342073

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : Nat) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_divisible_by_smallest_primes :
  ∃ (n : Nat), is_five_digit n ∧ 
    (∀ p ∈ smallest_primes, n % p = 0) ∧
    (∀ m : Nat, is_five_digit m ∧ (∀ p ∈ smallest_primes, m % p = 0) → n ≤ m) ∧
    n = 11550 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l3420_342073


namespace NUMINAMATH_CALUDE_train_length_l3420_342072

/-- Calculates the length of a train given its speed, the speed of a vehicle it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 →
  motorbike_speed = 64 →
  overtake_time = 40 →
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 400 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3420_342072


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l3420_342050

theorem intersection_point_coordinates
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let line1 := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let line2 := {(x, y) : ℝ × ℝ | b * x + c * y = a}
  let line3 := {(x, y) : ℝ × ℝ | y = 2 * x}
  (∀ (p q : ℝ × ℝ), p ∈ line1 ∧ q ∈ line2 → (p.1 - q.1) * (p.2 - q.2) = -1) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3 ∧ P = (-3/5, -6/5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l3420_342050


namespace NUMINAMATH_CALUDE_diagonal_cuboids_count_l3420_342047

def cuboid_count (a b c L : ℕ) : ℕ :=
  L / a + L / b + L / c - L / (a * b) - L / (a * c) - L / (b * c) + L / (a * b * c)

theorem diagonal_cuboids_count : 
  let a : ℕ := 2
  let b : ℕ := 7
  let c : ℕ := 13
  let L : ℕ := 2002
  let lcm : ℕ := a * b * c
  (L / lcm) * cuboid_count a b c lcm = 1210 := by sorry

end NUMINAMATH_CALUDE_diagonal_cuboids_count_l3420_342047


namespace NUMINAMATH_CALUDE_least_common_multiple_9_6_l3420_342009

theorem least_common_multiple_9_6 : Nat.lcm 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_9_6_l3420_342009


namespace NUMINAMATH_CALUDE_log_problem_l3420_342016

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the conditions and the theorem
theorem log_problem (x y : ℝ) (h1 : log (x * y^5) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l3420_342016


namespace NUMINAMATH_CALUDE_lidia_money_is_66_l3420_342062

/-- The amount of money Lidia has for buying apps -/
def lidia_money (app_cost : ℝ) (num_apps : ℕ) (remaining : ℝ) : ℝ :=
  app_cost * (num_apps : ℝ) + remaining

/-- Theorem stating that Lidia has $66 for buying apps -/
theorem lidia_money_is_66 :
  lidia_money 4 15 6 = 66 := by
  sorry

end NUMINAMATH_CALUDE_lidia_money_is_66_l3420_342062
