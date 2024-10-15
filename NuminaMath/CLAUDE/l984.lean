import Mathlib

namespace NUMINAMATH_CALUDE_remainder_problem_l984_98456

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l984_98456


namespace NUMINAMATH_CALUDE_sale_price_percentage_l984_98471

theorem sale_price_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := original_price * 0.5
  let final_price := first_sale_price * 0.9
  (final_price / original_price) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_percentage_l984_98471


namespace NUMINAMATH_CALUDE_sequence_max_value_l984_98447

def a (n : ℕ+) : ℚ := n / (n^2 + 156)

theorem sequence_max_value :
  (∃ (k : ℕ+), a k = 1/25 ∧ 
   ∀ (n : ℕ+), a n ≤ 1/25) ∧
  (∀ (n : ℕ+), a n = 1/25 → (n = 12 ∨ n = 13)) :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l984_98447


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_59_million_l984_98413

/-- Expresses 1.59 million in scientific notation -/
theorem scientific_notation_of_1_59_million :
  (1.59 : ℝ) * 1000000 = 1.59 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_59_million_l984_98413


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l984_98424

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a + b + c = a * b + b * c + c * a) 
  (h5 : a + b + c > 0) : 
  (Real.sqrt (b * c) * (a + 1) ≥ 2) ∧ 
  (Real.sqrt (b * c) * (a + 1) = 2 ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 0 ∧ b = 2 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l984_98424


namespace NUMINAMATH_CALUDE_unique_4digit_number_l984_98465

def is_3digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_4digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem unique_4digit_number :
  ∃! n : ℕ, 
    is_4digit n ∧ 
    (∃ a : ℕ, is_3digit (400 + 10*a + 3) ∧ n = (400 + 10*a + 3) + 984) ∧
    n % 11 = 0 ∧
    (∃ h : ℕ, 10 ≤ h ∧ h ≤ 19 ∧ a + (h - 10) = 10 ∧ n = 1000*h + (n % 1000)) ∧
    n = 1397 :=
sorry

end NUMINAMATH_CALUDE_unique_4digit_number_l984_98465


namespace NUMINAMATH_CALUDE_dispersion_measures_l984_98485

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define statistics
def standardDeviation (s : Sample) : Real :=
  sorry

def median (s : Sample) : Real :=
  sorry

def range (s : Sample) : Real :=
  sorry

def mean (s : Sample) : Real :=
  sorry

-- Define a predicate for measures of dispersion
def measuresDispersion (f : Sample → Real) : Prop :=
  sorry

-- Theorem statement
theorem dispersion_measures (s : Sample) :
  measuresDispersion (standardDeviation) ∧
  measuresDispersion (range) ∧
  ¬measuresDispersion (median) ∧
  ¬measuresDispersion (mean) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l984_98485


namespace NUMINAMATH_CALUDE_perfect_square_conditions_l984_98430

theorem perfect_square_conditions (a b c d e f : ℝ) :
  (∃ (p q r : ℝ), ∀ (x y z : ℝ),
    a * x^2 + b * y^2 + c * z^2 + 2 * d * x * y + 2 * e * y * z + 2 * f * z * x = (p * x + q * y + r * z)^2)
  ↔
  (a * b = d^2 ∧ b * c = e^2 ∧ c * a = f^2 ∧ a * e = d * f ∧ b * f = d * e ∧ c * d = e * f) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_conditions_l984_98430


namespace NUMINAMATH_CALUDE_carol_cupcakes_theorem_l984_98448

/-- Calculates the number of cupcakes made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial - sold)

/-- Proves that Carol made 28 cupcakes after selling the first batch -/
theorem carol_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final_total : ℕ)
    (h1 : initial = 30)
    (h2 : sold = 9)
    (h3 : final_total = 49) :
    cupcakes_made_after initial sold final_total = 28 := by
  sorry

end NUMINAMATH_CALUDE_carol_cupcakes_theorem_l984_98448


namespace NUMINAMATH_CALUDE_equation_solutions_l984_98497

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (Real.sqrt x) = 15 / (8 - Real.sqrt (Real.sqrt x))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 625 ∨ x = 81) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l984_98497


namespace NUMINAMATH_CALUDE_total_cost_of_cows_l984_98409

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of hearts per card
def hearts_per_card : ℕ := 4

-- Define the cost per cow
def cost_per_cow : ℕ := 200

-- Define the number of cows in Devonshire
def cows_in_devonshire : ℕ := 2 * (standard_deck_size * hearts_per_card)

-- State the theorem
theorem total_cost_of_cows : cows_in_devonshire * cost_per_cow = 83200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_cows_l984_98409


namespace NUMINAMATH_CALUDE_randy_blocks_left_l984_98451

/-- The number of blocks Randy has left after constructions -/
def blocks_left (initial : ℕ) (tower : ℕ) (house : ℕ) : ℕ :=
  let remaining_after_tower := initial - tower
  let bridge := remaining_after_tower / 2
  let remaining_after_bridge := remaining_after_tower - bridge
  remaining_after_bridge - house

/-- Theorem stating that Randy has 19 blocks left after constructions -/
theorem randy_blocks_left :
  blocks_left 78 19 11 = 19 := by sorry

end NUMINAMATH_CALUDE_randy_blocks_left_l984_98451


namespace NUMINAMATH_CALUDE_sue_grocery_spending_l984_98469

/-- Calculates Sue's spending on a grocery shopping trip with specific conditions --/
theorem sue_grocery_spending : 
  let apple_price : ℚ := 2
  let apple_quantity : ℕ := 4
  let juice_price : ℚ := 6
  let juice_quantity : ℕ := 2
  let bread_price : ℚ := 3
  let bread_quantity : ℕ := 3
  let cheese_price : ℚ := 4
  let cheese_quantity : ℕ := 2
  let cereal_price : ℚ := 8
  let cereal_quantity : ℕ := 1
  let cheese_discount : ℚ := 0.25
  let order_discount_threshold : ℚ := 40
  let order_discount_rate : ℚ := 0.1

  let discounted_cheese_price : ℚ := cheese_price * (1 - cheese_discount)
  let subtotal : ℚ := 
    apple_price * apple_quantity +
    juice_price * juice_quantity +
    bread_price * bread_quantity +
    discounted_cheese_price * cheese_quantity +
    cereal_price * cereal_quantity

  let final_total : ℚ := 
    if subtotal ≥ order_discount_threshold
    then subtotal * (1 - order_discount_rate)
    else subtotal

  final_total = 387/10 := by sorry

end NUMINAMATH_CALUDE_sue_grocery_spending_l984_98469


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_504_l984_98419

theorem factorial_ratio_equals_504 : ∃! n : ℕ, n > 0 ∧ n.factorial / (n - 3).factorial = 504 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_504_l984_98419


namespace NUMINAMATH_CALUDE_even_sum_digits_all_residues_l984_98476

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Main theorem -/
theorem even_sum_digits_all_residues (k : ℕ) (h : k ≥ 2) :
  ∀ r, r < k → ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r :=
by sorry

end NUMINAMATH_CALUDE_even_sum_digits_all_residues_l984_98476


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l984_98492

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^3 - 3ax + a^2 is exactly one. -/
theorem line_intersects_parabola_vertex_once :
  ∃! a : ℝ, ∃ x y : ℝ,
    (y = x + a) ∧                   -- Line equation
    (y = x^3 - 3*a*x + a^2) ∧       -- Parabola equation
    (∀ x' : ℝ, x'^3 - 3*a*x' + a^2 ≤ x^3 - 3*a*x + a^2) -- Vertex condition
    := by sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_once_l984_98492


namespace NUMINAMATH_CALUDE_iphone_case_cost_percentage_l984_98441

/-- Proves that the percentage of the case cost relative to the phone cost is 20% --/
theorem iphone_case_cost_percentage :
  let phone_cost : ℝ := 1000
  let monthly_contract_cost : ℝ := 200
  let case_cost_percentage : ℝ → ℝ := λ x => x / 100 * phone_cost
  let headphones_cost : ℝ → ℝ := λ x => (1 / 2) * case_cost_percentage x
  let total_yearly_cost : ℝ → ℝ := λ x => 
    phone_cost + 12 * monthly_contract_cost + case_cost_percentage x + headphones_cost x
  ∃ x : ℝ, total_yearly_cost x = 3700 ∧ x = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_iphone_case_cost_percentage_l984_98441


namespace NUMINAMATH_CALUDE_arrangements_count_l984_98464

/-- The number of arrangements for 7 students with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let ways_to_place_together : ℕ := 2  -- left or right of middle
  let arrangements_within_together : ℕ := 2  -- B-C or C-B
  let permutations_of_remaining : ℕ := Nat.factorial remaining_students
  ways_to_place_together * arrangements_within_together * permutations_of_remaining

theorem arrangements_count : num_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l984_98464


namespace NUMINAMATH_CALUDE_probability_for_given_dice_l984_98421

/-- Represents a 20-sided die with color distributions -/
structure TwentySidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total : Nat)
  (sum_eq_total : maroon + teal + cyan + sparkly = total)

/-- Calculate the probability of two 20-sided dice showing the same color
    and a 6-sided die showing a number greater than 4 -/
def probability_same_color_and_high_roll 
  (die1 : TwentySidedDie) 
  (die2 : TwentySidedDie) : ℚ :=
  let same_color_prob := 
    (die1.maroon * die2.maroon + 
     die1.teal * die2.teal + 
     die1.cyan * die2.cyan + 
     die1.sparkly * die2.sparkly : ℚ) / 
    (die1.total * die2.total : ℚ)
  let high_roll_prob : ℚ := 1 / 3
  same_color_prob * high_roll_prob

/-- The main theorem stating the probability for the given dice configuration -/
theorem probability_for_given_dice : 
  let die1 : TwentySidedDie := ⟨3, 9, 7, 1, 20, by norm_num⟩
  let die2 : TwentySidedDie := ⟨5, 6, 8, 1, 20, by norm_num⟩
  probability_same_color_and_high_roll die1 die2 = 21 / 200 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_given_dice_l984_98421


namespace NUMINAMATH_CALUDE_find_tap_a_turnoff_time_l984_98461

/-- Represents the time it takes for a tap to fill the cistern -/
structure TapFillTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents the state of the cistern filling process -/
structure CisternFilling where
  tapA : TapFillTime
  tapB : TapFillTime
  remainingTime : ℝ
  positive : remainingTime > 0

/-- The main theorem statement -/
theorem find_tap_a_turnoff_time (c : CisternFilling) 
    (h1 : c.tapA.minutes = 12)
    (h2 : c.tapB.minutes = 18)
    (h3 : c.remainingTime = 8) : 
  ∃ t : ℝ, t > 0 ∧ t = 4 ∧
    (t * (1 / c.tapA.minutes + 1 / c.tapB.minutes) + 
     c.remainingTime * (1 / c.tapB.minutes) = 1) := by
  sorry

#check find_tap_a_turnoff_time

end NUMINAMATH_CALUDE_find_tap_a_turnoff_time_l984_98461


namespace NUMINAMATH_CALUDE_integer_solution_existence_l984_98470

theorem integer_solution_existence (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_existence_l984_98470


namespace NUMINAMATH_CALUDE_smaller_field_area_l984_98499

theorem smaller_field_area (total_area : ℝ) (smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area + larger_area = total_area →
  larger_area - smaller_area = (smaller_area + larger_area) / 10 →
  smaller_area = 225 := by
sorry

end NUMINAMATH_CALUDE_smaller_field_area_l984_98499


namespace NUMINAMATH_CALUDE_max_value_theorem_l984_98484

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  2*a*b + 2*b*c*Real.sqrt 2 + 2*a*c ≤ 2*(1 + Real.sqrt 2)/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l984_98484


namespace NUMINAMATH_CALUDE_fountain_area_l984_98429

theorem fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : ∃ (area : ℝ), area = 244 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_fountain_area_l984_98429


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l984_98487

theorem min_value_sum_of_reciprocals (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l984_98487


namespace NUMINAMATH_CALUDE_carmen_cookie_sales_l984_98436

/-- Represents the number of boxes of each type of cookie sold --/
structure CookieSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ

/-- Represents the price of each type of cookie --/
structure CookiePrices where
  samoas : ℚ
  thinMints : ℚ
  fudgeDelights : ℚ
  sugarCookies : ℚ

/-- Calculates the total revenue from cookie sales --/
def totalRevenue (sales : CookieSales) (prices : CookiePrices) : ℚ :=
  sales.samoas * prices.samoas +
  sales.thinMints * prices.thinMints +
  sales.fudgeDelights * prices.fudgeDelights +
  sales.sugarCookies * prices.sugarCookies

/-- The main theorem representing Carmen's cookie sales --/
theorem carmen_cookie_sales 
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thinMints = 2)
  (h3 : sales.fudgeDelights = 1)
  (h4 : prices.samoas = 4)
  (h5 : prices.thinMints = 7/2)
  (h6 : prices.fudgeDelights = 5)
  (h7 : prices.sugarCookies = 2)
  (h8 : totalRevenue sales prices = 42) :
  sales.sugarCookies = 9 := by
  sorry

end NUMINAMATH_CALUDE_carmen_cookie_sales_l984_98436


namespace NUMINAMATH_CALUDE_nowhere_negative_polynomial_is_sum_of_squares_l984_98460

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- A polynomial is nowhere negative if it's non-negative for all real inputs -/
def NowhereNegative (p : RealPolynomial) : Prop :=
  ∀ x : ℝ, p x ≥ 0

/-- Theorem: Any nowhere negative real polynomial can be expressed as a sum of squares -/
theorem nowhere_negative_polynomial_is_sum_of_squares :
  ∀ p : RealPolynomial, NowhereNegative p →
  ∃ q r s : RealPolynomial, ∀ x : ℝ, p x = (q x)^2 * ((r x)^2 + (s x)^2) :=
sorry

end NUMINAMATH_CALUDE_nowhere_negative_polynomial_is_sum_of_squares_l984_98460


namespace NUMINAMATH_CALUDE_trig_simplification_l984_98489

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) =
  1/2 * (1 - Real.cos x ^ 2 - Real.cos y ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l984_98489


namespace NUMINAMATH_CALUDE_d_share_is_300_l984_98422

/-- Calculates the share of profit for an investor given the investments and total profit -/
def calculate_share (investment_c : ℚ) (investment_d : ℚ) (total_profit : ℚ) : ℚ :=
  (investment_d / (investment_c + investment_d)) * total_profit

/-- Theorem stating that D's share of the profit is 300 given the specified investments and total profit -/
theorem d_share_is_300 
  (investment_c : ℚ) 
  (investment_d : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_c = 1000)
  (h2 : investment_d = 1500)
  (h3 : total_profit = 500) :
  calculate_share investment_c investment_d total_profit = 300 := by
  sorry

#eval calculate_share 1000 1500 500

end NUMINAMATH_CALUDE_d_share_is_300_l984_98422


namespace NUMINAMATH_CALUDE_fraction_equality_l984_98418

theorem fraction_equality (m n : ℚ) (h : 2/3 * m = 5/6 * n) : (m - n) / n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l984_98418


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l984_98402

/-- Theorem: Given two cyclists on a 45-mile course, starting from opposite ends at the same time,
    where one cyclist rides at 14 mph and they meet after 1.5 hours, the speed of the second cyclist is 16 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (first_speed : ℝ) (meeting_time : ℝ) :
  course_length = 45 ∧ first_speed = 14 ∧ meeting_time = 1.5 →
  ∃ second_speed : ℝ, second_speed = 16 ∧ course_length = (first_speed + second_speed) * meeting_time :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_speed_problem_l984_98402


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l984_98453

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 600 →
  fence_length = 130 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l984_98453


namespace NUMINAMATH_CALUDE_ducks_in_lake_l984_98404

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l984_98404


namespace NUMINAMATH_CALUDE_triangle_inequality_l984_98431

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l984_98431


namespace NUMINAMATH_CALUDE_family_reunion_attendance_l984_98435

/-- The number of male adults at the family reunion -/
def male_adults : ℕ := 100

/-- The number of female adults at the family reunion -/
def female_adults : ℕ := male_adults + 50

/-- The total number of adults at the family reunion -/
def total_adults : ℕ := male_adults + female_adults

/-- The number of children at the family reunion -/
def children : ℕ := 2 * total_adults

/-- The total number of attendees at the family reunion -/
def total_attendees : ℕ := total_adults + children

theorem family_reunion_attendance : 
  female_adults = male_adults + 50 ∧ 
  children = 2 * total_adults ∧ 
  total_attendees = 750 → 
  male_adults = 100 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_attendance_l984_98435


namespace NUMINAMATH_CALUDE_hyperbola_equation_l984_98446

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ c : ℝ, c = Real.sqrt 5 ∧ c^2 = a^2 + b^2) → 
  (b / a = 1 / 2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l984_98446


namespace NUMINAMATH_CALUDE_benson_ticket_cost_l984_98403

/-- Calculates the total cost of concert tickets for Mr. Benson -/
def concert_ticket_cost (base_price : ℝ) (general_count : ℕ) (vip_count : ℕ) (premium_count : ℕ) 
  (vip_markup : ℝ) (premium_markup : ℝ) (discount_threshold : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_count := general_count + vip_count + premium_count
  let vip_price := base_price * (1 + vip_markup)
  let premium_price := base_price * (1 + premium_markup)
  let discounted_count := max (total_count - discount_threshold) 0
  let general_cost := base_price * general_count
  let vip_cost := if vip_count ≤ discounted_count
                  then vip_price * vip_count * (1 - discount_rate)
                  else vip_price * (vip_count - discounted_count) + 
                       vip_price * discounted_count * (1 - discount_rate)
  let premium_cost := if premium_count ≤ (discounted_count - vip_count)
                      then premium_price * premium_count * (1 - discount_rate)
                      else premium_price * (premium_count - (discounted_count - vip_count)) +
                           premium_price * (discounted_count - vip_count) * (1 - discount_rate)
  general_cost + vip_cost + premium_cost

/-- Theorem stating that the total cost for Mr. Benson's tickets is $650.80 -/
theorem benson_ticket_cost : 
  concert_ticket_cost 40 10 3 2 0.2 0.5 10 0.05 = 650.80 := by
  sorry


end NUMINAMATH_CALUDE_benson_ticket_cost_l984_98403


namespace NUMINAMATH_CALUDE_lionel_graham_crackers_left_l984_98433

/-- Represents the ingredients for making Oreo cheesecakes -/
structure Ingredients where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Represents the recipe requirements for one Oreo cheesecake -/
structure Recipe where
  graham_crackers : ℕ
  oreos : ℕ
  cream_cheese : ℕ

/-- Calculates the maximum number of cheesecakes that can be made given the ingredients and recipe -/
def max_cheesecakes (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  min (ingredients.graham_crackers / recipe.graham_crackers)
      (min (ingredients.oreos / recipe.oreos)
           (ingredients.cream_cheese / recipe.cream_cheese))

/-- Calculates the number of Graham cracker boxes left over after making the maximum number of cheesecakes -/
def graham_crackers_left (ingredients : Ingredients) (recipe : Recipe) : ℕ :=
  ingredients.graham_crackers - (max_cheesecakes ingredients recipe * recipe.graham_crackers)

/-- Theorem stating that Lionel will have 4 boxes of Graham crackers left over -/
theorem lionel_graham_crackers_left :
  let ingredients := Ingredients.mk 14 15 36
  let recipe := Recipe.mk 2 3 4
  graham_crackers_left ingredients recipe = 4 := by
  sorry

end NUMINAMATH_CALUDE_lionel_graham_crackers_left_l984_98433


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l984_98458

/-- The ratio of the volume of a cone to the volume of a cylinder with the same base radius,
    where the cone's height is one-third of the cylinder's height, is 1/9. -/
theorem cone_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l984_98458


namespace NUMINAMATH_CALUDE_triangle_property_l984_98400

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  -- Given condition
  (2 * Real.cos B * Real.cos C + 1 = 2 * Real.sin B * Real.sin C) →
  (b + c = 4) →
  -- Conclusions
  (A = π / 3) ∧
  (∀ (area : Real), area = 1/2 * b * c * Real.sin A → area ≤ Real.sqrt 3) ∧
  (∃ (area : Real), area = 1/2 * b * c * Real.sin A ∧ area = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l984_98400


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l984_98462

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 8

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen spent -/
def money_spent : ℕ := 4

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem gwen_birthday_money : difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l984_98462


namespace NUMINAMATH_CALUDE_expression_simplification_l984_98475

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : 3 * x + y / 3 + 2 * z ≠ 0) :
  (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹) = 
  (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l984_98475


namespace NUMINAMATH_CALUDE_sequence_general_term_l984_98466

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l984_98466


namespace NUMINAMATH_CALUDE_journey_speed_theorem_l984_98408

/-- Proves that given a round trip journey where the time taken to go up is twice the time taken to come down,
    the total journey time is 6 hours, and the average speed for the whole journey is 4 km/h,
    then the average speed while going up is 3 km/h. -/
theorem journey_speed_theorem (time_up : ℝ) (time_down : ℝ) (total_distance : ℝ) :
  time_up = 2 * time_down →
  time_up + time_down = 6 →
  total_distance / (time_up + time_down) = 4 →
  (total_distance / 2) / time_up = 3 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_theorem_l984_98408


namespace NUMINAMATH_CALUDE_school_students_count_l984_98425

theorem school_students_count (total : ℕ) 
  (chess_ratio : Real) 
  (swimming_ratio : Real) 
  (swimming_count : ℕ) 
  (h1 : chess_ratio = 0.25)
  (h2 : swimming_ratio = 0.50)
  (h3 : swimming_count = 125)
  (h4 : ↑swimming_count = swimming_ratio * (chess_ratio * total)) :
  total = 1000 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l984_98425


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l984_98468

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / a - y / b = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l984_98468


namespace NUMINAMATH_CALUDE_three_stamps_cost_l984_98417

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34 / 100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68 / 100

/-- Theorem: The cost of three stamps is $1.02 -/
theorem three_stamps_cost : stamp_cost * 3 = 102 / 100 := by
  sorry

end NUMINAMATH_CALUDE_three_stamps_cost_l984_98417


namespace NUMINAMATH_CALUDE_equation_equivalence_l984_98459

theorem equation_equivalence (x y : ℝ) :
  (2*x - 3*y)^2 = 4*x^2 + 9*y^2 ↔ x*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l984_98459


namespace NUMINAMATH_CALUDE_special_line_equation_l984_98440

/-- A line passing through (9, 4) with x-intercept 5 units greater than y-intercept -/
def SpecialLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b ∧ (9, 4) ∈ {q : ℝ × ℝ | q.2 = m * q.1 + b} ∧
    ∃ (x y : ℝ), x = y + 5 ∧ 0 = m * x + b ∧ y = b}

/-- The three possible equations of the special line -/
def PossibleEquations : Set (ℝ × ℝ → Prop) :=
  {(λ p : ℝ × ℝ ↦ 2 * p.1 + 3 * p.2 - 30 = 0),
   (λ p : ℝ × ℝ ↦ 2 * p.1 - 3 * p.2 - 6 = 0),
   (λ p : ℝ × ℝ ↦ p.1 - p.2 - 5 = 0)}

theorem special_line_equation :
  ∃ (eq : ℝ × ℝ → Prop), eq ∈ PossibleEquations ∧ ∀ p : ℝ × ℝ, p ∈ SpecialLine ↔ eq p :=
by sorry

end NUMINAMATH_CALUDE_special_line_equation_l984_98440


namespace NUMINAMATH_CALUDE_combined_average_score_l984_98412

/-- Given three classes with average scores and student ratios, prove the combined average score -/
theorem combined_average_score 
  (score_U score_B score_C : ℝ)
  (ratio_U ratio_B ratio_C : ℕ)
  (h1 : score_U = 65)
  (h2 : score_B = 80)
  (h3 : score_C = 77)
  (h4 : ratio_U = 4)
  (h5 : ratio_B = 6)
  (h6 : ratio_C = 5) :
  (score_U * ratio_U + score_B * ratio_B + score_C * ratio_C) / (ratio_U + ratio_B + ratio_C) = 75 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l984_98412


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l984_98455

theorem concentric_circles_radii_difference
  (r R : ℝ) -- radii of the smaller and larger circles
  (h : r > 0) -- radius is positive
  (area_ratio : π * R^2 = 4 * (π * r^2)) -- area ratio is 1:4
  : R - r = r := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l984_98455


namespace NUMINAMATH_CALUDE_yellow_cards_per_player_l984_98407

theorem yellow_cards_per_player (total_players : ℕ) (uncautioned_players : ℕ) (red_cards : ℕ) 
  (h1 : total_players = 11)
  (h2 : uncautioned_players = 5)
  (h3 : red_cards = 3) :
  (total_players - uncautioned_players) * ((red_cards * 2) / (total_players - uncautioned_players)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_yellow_cards_per_player_l984_98407


namespace NUMINAMATH_CALUDE_star_op_specific_value_l984_98415

-- Define the * operation for non-zero integers
def star_op (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_op_specific_value :
  ∀ a b : ℕ+, 
  (a : ℤ) + (b : ℤ) = 15 → 
  (a : ℤ) * (b : ℤ) = 36 → 
  star_op a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_op_specific_value_l984_98415


namespace NUMINAMATH_CALUDE_innocent_knight_convincing_l984_98414

-- Define the types of people
inductive PersonType
| Normal
| Knight
| Liar

-- Define the properties of a person
structure Person where
  type : PersonType
  guilty : Bool

-- Define the criminal
def criminal : Person := { type := PersonType.Liar, guilty := true }

-- Define the statement made by the person
def statement (p : Person) : Prop := p.type = PersonType.Knight ∧ ¬p.guilty

-- Theorem to prove
theorem innocent_knight_convincing (p : Person) 
  (h1 : p.type ≠ PersonType.Normal) 
  (h2 : ¬p.guilty) 
  (h3 : p.type ≠ PersonType.Liar) :
  statement p → (¬p.guilty ∧ p.type ≠ PersonType.Liar) :=
by sorry

end NUMINAMATH_CALUDE_innocent_knight_convincing_l984_98414


namespace NUMINAMATH_CALUDE_rectangle_sides_from_ratio_and_area_l984_98483

theorem rectangle_sides_from_ratio_and_area 
  (m n S : ℝ) (hm : m > 0) (hn : n > 0) (hS : S > 0) :
  ∃ (x y : ℝ), 
    x / y = m / n ∧ 
    x * y = S ∧ 
    x = Real.sqrt ((m * S) / n) ∧ 
    y = Real.sqrt ((n * S) / m) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_from_ratio_and_area_l984_98483


namespace NUMINAMATH_CALUDE_work_completed_by_two_workers_l984_98450

/-- The fraction of work completed by two workers in one day -/
def work_completed_together (days_a : ℕ) (days_b : ℕ) : ℚ :=
  1 / days_a + 1 / days_b

/-- Theorem: Two workers A and B, where A takes 12 days and B takes half the time of A,
    can complete 1/4 of the work in one day when working together -/
theorem work_completed_by_two_workers :
  let days_a : ℕ := 12
  let days_b : ℕ := days_a / 2
  work_completed_together days_a days_b = 1 / 4 := by
sorry


end NUMINAMATH_CALUDE_work_completed_by_two_workers_l984_98450


namespace NUMINAMATH_CALUDE_square_expansion_l984_98490

theorem square_expansion (n : ℕ) (h : ∃ k : ℕ, k > 0 ∧ (n + k)^2 - n^2 = 47) : n = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_expansion_l984_98490


namespace NUMINAMATH_CALUDE_completing_square_l984_98481

theorem completing_square (x : ℝ) : x^2 - 4*x = 6 ↔ (x - 2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l984_98481


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l984_98463

/-- The function f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 + a*x^2 + 3*x - 9
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-3-ε) (-3+ε), f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x^3 + ax^2 + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ a : ℝ, has_extreme_value_at_neg_three a → a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l984_98463


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l984_98427

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 72) 
  (h2 : height = 6) 
  (h3 : area = base * height) : 
  base = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l984_98427


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_is_real_line_l984_98426

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_is_real_line 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_is_real_line_l984_98426


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l984_98478

/-- Given two points (1,6) and (5,2) as adjacent vertices of a square, prove that the area of the square is 32. -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, 6)
  let p2 : ℝ × ℝ := (5, 2)
  32 = (((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) : ℝ) := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l984_98478


namespace NUMINAMATH_CALUDE_lassis_from_nine_mangoes_l984_98449

/-- The number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  5 * mangoes

/-- The cost of a given number of mangoes -/
def mango_cost (mangoes : ℕ) : ℕ :=
  2 * mangoes

theorem lassis_from_nine_mangoes :
  lassis_from_mangoes 9 = 45 :=
by sorry

end NUMINAMATH_CALUDE_lassis_from_nine_mangoes_l984_98449


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l984_98480

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : ∃ k : ℕ, A = 2 * k ∧ B = 3 * k) : 
  gcd A B = 30 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l984_98480


namespace NUMINAMATH_CALUDE_francine_work_weeks_francine_work_weeks_solution_l984_98438

theorem francine_work_weeks 
  (daily_distance : ℕ) 
  (workdays_per_week : ℕ) 
  (total_distance : ℕ) : ℕ :=
  let weekly_distance := daily_distance * workdays_per_week
  total_distance / weekly_distance

#check francine_work_weeks 140 4 2240

theorem francine_work_weeks_solution :
  francine_work_weeks 140 4 2240 = 4 := by
  sorry

end NUMINAMATH_CALUDE_francine_work_weeks_francine_work_weeks_solution_l984_98438


namespace NUMINAMATH_CALUDE_min_distinct_prime_factors_l984_98452

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  p ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) ∧
  q ∣ (m * (n + 9) * (m + 2 * n^2 + 3)) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_prime_factors_l984_98452


namespace NUMINAMATH_CALUDE_money_ratio_l984_98488

/-- Prove that the ratio of Alison's money to Brittany's money is 1:2 -/
theorem money_ratio (kent_money : ℝ) (brooke_money : ℝ) (brittany_money : ℝ) (alison_money : ℝ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : brittany_money = 4 * brooke_money)
  (h4 : alison_money = 4000) :
  alison_money / brittany_money = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_l984_98488


namespace NUMINAMATH_CALUDE_fraction_of_25_l984_98473

theorem fraction_of_25 : ∃ x : ℚ, x * 25 = 0.9 * 40 - 16 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l984_98473


namespace NUMINAMATH_CALUDE_indeterminate_magnitude_l984_98467

/-- Given two approximate numbers A and B, prove that their relative magnitude cannot be determined. -/
theorem indeterminate_magnitude (A B : ℝ) (hA : 3.55 ≤ A ∧ A < 3.65) (hB : 3.595 ≤ B ∧ B < 3.605) :
  ¬(A > B ∨ A = B ∨ A < B) := by
  sorry

#check indeterminate_magnitude

end NUMINAMATH_CALUDE_indeterminate_magnitude_l984_98467


namespace NUMINAMATH_CALUDE_sin_squared_value_l984_98482

theorem sin_squared_value (α : Real) (h : Real.tan (α + π/4) = 3/4) :
  Real.sin (π/4 - α) ^ 2 = 16/25 := by sorry

end NUMINAMATH_CALUDE_sin_squared_value_l984_98482


namespace NUMINAMATH_CALUDE_quadratic_roots_value_l984_98496

theorem quadratic_roots_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 49/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_value_l984_98496


namespace NUMINAMATH_CALUDE_pi_digits_difference_l984_98432

theorem pi_digits_difference (mina_digits : ℕ) (mina_carlos_ratio : ℕ) (sam_digits : ℕ)
  (h1 : mina_digits = 24)
  (h2 : mina_digits = mina_carlos_ratio * (sam_digits - 6))
  (h3 : sam_digits = 10) :
  sam_digits - (mina_digits / mina_carlos_ratio) = 6 := by
sorry

end NUMINAMATH_CALUDE_pi_digits_difference_l984_98432


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_l984_98474

/-- Represents the eating habits in a family -/
structure FamilyDiet where
  onlyVegetarian : ℕ
  onlyNonVegetarian : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def vegetarianEaters (f : FamilyDiet) : ℕ :=
  f.onlyVegetarian + f.both

/-- Theorem: Given the family diet information, prove that the number of vegetarian eaters
    is the sum of those who eat only vegetarian and those who eat both -/
theorem vegetarian_eaters_count (f : FamilyDiet) 
    (h1 : f.onlyVegetarian = 13)
    (h2 : f.onlyNonVegetarian = 7)
    (h3 : f.both = 8) :
    vegetarianEaters f = 21 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_l984_98474


namespace NUMINAMATH_CALUDE_number_of_students_l984_98472

theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_average : ℝ) :
  initial_average = 100 →
  wrong_mark = 90 →
  correct_mark = 10 →
  correct_average = 92 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * initial_average - (wrong_mark - correct_mark) = (n : ℝ) * correct_average ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l984_98472


namespace NUMINAMATH_CALUDE_greatest_c_value_l984_98442

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_c_value_l984_98442


namespace NUMINAMATH_CALUDE_existence_of_square_between_l984_98454

theorem existence_of_square_between (a b c d : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, (↑a : ℝ) < m^2 ∧ (m^2 : ℝ) < ↑d :=
by sorry

end NUMINAMATH_CALUDE_existence_of_square_between_l984_98454


namespace NUMINAMATH_CALUDE_min_value_of_linear_function_l984_98439

/-- Given a system of linear inequalities, prove that the minimum value of a linear function is -6. -/
theorem min_value_of_linear_function (x y : ℝ) :
  x - 2*y + 2 ≥ 0 →
  2*x - y - 2 ≤ 0 →
  y ≥ 0 →
  ∀ z : ℝ, z = 3*x + y → z ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_linear_function_l984_98439


namespace NUMINAMATH_CALUDE_art_collection_cost_l984_98477

/-- The total cost of John's art collection --/
def total_cost (first_3_price : ℚ) : ℚ :=
  -- Cost of first 3 pieces
  3 * first_3_price +
  -- Cost of next 2 pieces (25% more expensive)
  2 * (first_3_price * (1 + 1/4)) +
  -- Cost of last 3 pieces (50% more expensive)
  3 * (first_3_price * (1 + 1/2))

/-- Theorem stating the total cost of John's art collection --/
theorem art_collection_cost :
  ∃ (first_3_price : ℚ),
    first_3_price > 0 ∧
    3 * first_3_price = 45000 ∧
    total_cost first_3_price = 150000 := by
  sorry


end NUMINAMATH_CALUDE_art_collection_cost_l984_98477


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l984_98434

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 + x*y - y = 2 ↔ (x = 2 ∧ y = -2) ∨ (x = 0 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l984_98434


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l984_98410

theorem max_distance_between_functions (a : ℝ) : 
  let f (x : ℝ) := 2 * (Real.cos (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x)
  let distance := |f a - g a|
  ∃ (max_distance : ℝ), max_distance = 3 ∧ distance ≤ max_distance ∧
    ∀ (b : ℝ), |f b - g b| ≤ max_distance :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l984_98410


namespace NUMINAMATH_CALUDE_percentage_of_girls_l984_98416

/-- The percentage of girls in a school, given the total number of students and the number of boys. -/
theorem percentage_of_girls (total : ℕ) (boys : ℕ) (h1 : total = 100) (h2 : boys = 50) :
  (total - boys : ℚ) / total * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_l984_98416


namespace NUMINAMATH_CALUDE_first_boy_speed_l984_98401

/-- The speed of the second boy in km/h -/
def second_boy_speed : ℝ := 7.5

/-- The time the boys walk in hours -/
def walking_time : ℝ := 16

/-- The distance between the boys after walking in km -/
def final_distance : ℝ := 32

/-- Theorem stating the speed of the first boy -/
theorem first_boy_speed (x : ℝ) : 
  (x - second_boy_speed) * walking_time = final_distance → x = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_first_boy_speed_l984_98401


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l984_98495

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sample : Nat

/-- Calculates the interval between samples -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the first sampled number -/
def first_sample (s : SystematicSampling) : Nat :=
  s.last_sample % (sample_interval s)

/-- Calculates the second sampled number -/
def second_sample (s : SystematicSampling) : Nat :=
  first_sample s + sample_interval s

/-- Theorem stating the first two sampled numbers for the given scenario -/
theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sample = 7900) :
  first_sample s = 60 ∧ second_sample s = 220 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l984_98495


namespace NUMINAMATH_CALUDE_cube_edge_length_l984_98420

/-- Represents a cube with a given total edge length. -/
structure Cube where
  total_edge_length : ℝ
  total_edge_length_positive : 0 < total_edge_length

/-- The number of edges in a cube. -/
def num_edges : ℕ := 12

/-- Theorem: In a cube where the sum of all edge lengths is 108 cm, each edge is 9 cm long. -/
theorem cube_edge_length (c : Cube) (h : c.total_edge_length = 108) :
  c.total_edge_length / num_edges = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l984_98420


namespace NUMINAMATH_CALUDE_candy_bar_multiple_l984_98444

theorem candy_bar_multiple (max_sales seth_sales : ℕ) (m : ℚ) 
  (h1 : max_sales = 24)
  (h2 : seth_sales = 78)
  (h3 : seth_sales = m * max_sales + 6) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_multiple_l984_98444


namespace NUMINAMATH_CALUDE_octal_subtraction_example_l984_98494

/-- Subtraction in octal (base 8) number system --/
def octal_subtraction (a b : Nat) : Nat :=
  -- Define octal subtraction here
  sorry

/-- Conversion from decimal to octal --/
def decimal_to_octal (n : Nat) : Nat :=
  -- Define decimal to octal conversion here
  sorry

/-- Conversion from octal to decimal --/
def octal_to_decimal (n : Nat) : Nat :=
  -- Define octal to decimal conversion here
  sorry

theorem octal_subtraction_example : octal_subtraction 325 237 = 66 := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_example_l984_98494


namespace NUMINAMATH_CALUDE_total_tickets_sold_l984_98445

/-- Represents the price of an adult ticket in dollars -/
def adult_price : ℕ := 15

/-- Represents the price of a child ticket in dollars -/
def child_price : ℕ := 8

/-- Represents the total receipts for the day in dollars -/
def total_receipts : ℕ := 5086

/-- Represents the number of adult tickets sold -/
def adult_tickets : ℕ := 130

/-- Theorem stating that the total number of tickets sold is 522 -/
theorem total_tickets_sold : 
  ∃ (child_tickets : ℕ), 
    adult_tickets * adult_price + child_tickets * child_price = total_receipts ∧
    adult_tickets + child_tickets = 522 :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l984_98445


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l984_98493

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l984_98493


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l984_98405

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

-- Define the condition |PA| = |PB|
def equal_chords (x y : ℝ) : Prop := 
  ∃ (xa ya xb yb : ℝ), C₁ xa ya ∧ C₂ xb yb ∧ 
  (x - xa)^2 + (y - ya)^2 = (x - xb)^2 + (y - yb)^2

-- Theorem statement
theorem min_distance_to_origin : 
  ∀ (x y : ℝ), equal_chords x y → 
  ∃ (x' y' : ℝ), equal_chords x' y' ∧ 
  ∀ (x'' y'' : ℝ), equal_chords x'' y'' → 
  (x'^2 + y'^2 : ℝ) ≤ x''^2 + y''^2 ∧
  (x'^2 + y'^2 : ℝ) = (4/5)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l984_98405


namespace NUMINAMATH_CALUDE_sum_of_positive_numbers_l984_98491

theorem sum_of_positive_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y = 30 → x * z = 60 → y * z = 90 → 
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_positive_numbers_l984_98491


namespace NUMINAMATH_CALUDE_martin_goldfish_purchase_l984_98486

/-- The number of new goldfish Martin purchases every week -/
def new_goldfish_per_week : ℕ := sorry

theorem martin_goldfish_purchase :
  let initial_goldfish : ℕ := 18
  let dying_goldfish_per_week : ℕ := 5
  let weeks : ℕ := 7
  let final_goldfish : ℕ := 4
  final_goldfish = initial_goldfish + (new_goldfish_per_week - dying_goldfish_per_week) * weeks →
  new_goldfish_per_week = 3 := by
  sorry

end NUMINAMATH_CALUDE_martin_goldfish_purchase_l984_98486


namespace NUMINAMATH_CALUDE_rectangle_count_num_rectangles_nat_l984_98498

/-- The number of rectangles formed in a rectangle ABCD with additional points and lines -/
theorem rectangle_count (m n : ℕ) : 
  (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4 = 
  (Nat.choose (m + 2) 2) * (Nat.choose (n + 2) 2) :=
by sorry

/-- The formula for the number of rectangles formed -/
def num_rectangles (m n : ℕ) : ℕ := (m + 2) * (m + 1) * (n + 2) * (n + 1) / 4

/-- The number of rectangles is always a natural number -/
theorem num_rectangles_nat (m n : ℕ) : 
  ∃ k : ℕ, num_rectangles m n = k :=
by sorry

end NUMINAMATH_CALUDE_rectangle_count_num_rectangles_nat_l984_98498


namespace NUMINAMATH_CALUDE_amit_work_days_l984_98479

theorem amit_work_days (ananthu_days : ℕ) (amit_worked : ℕ) (total_days : ℕ) :
  ananthu_days = 30 →
  amit_worked = 3 →
  total_days = 27 →
  ∃ (amit_days : ℕ),
    amit_days = 15 ∧
    (3 : ℝ) / amit_days + (total_days - amit_worked : ℝ) / ananthu_days = 1 :=
by sorry

end NUMINAMATH_CALUDE_amit_work_days_l984_98479


namespace NUMINAMATH_CALUDE_parallel_transitivity_l984_98437

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "in plane" relation for a line
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (a b : Line) (α : Plane) :
  parallel_line a b →
  ¬ in_plane a α →
  ¬ in_plane b α →
  parallel_line_plane a α →
  parallel_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l984_98437


namespace NUMINAMATH_CALUDE_all_a_equal_one_l984_98406

def cyclic_index (i : ℕ) : ℕ :=
  match i % 100 with
  | 0 => 100
  | n => n

theorem all_a_equal_one (a : ℕ → ℝ) 
  (h_ineq : ∀ i, a (cyclic_index i) - 4 * a (cyclic_index (i + 1)) + 3 * a (cyclic_index (i + 2)) ≥ 0)
  (h_a1 : a 1 = 1) :
  ∀ i, a i = 1 := by
sorry

end NUMINAMATH_CALUDE_all_a_equal_one_l984_98406


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l984_98443

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l984_98443


namespace NUMINAMATH_CALUDE_population_after_20_years_l984_98423

/-- The population growth over time with a constant growth rate -/
def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- The theorem stating the population after 20 years with 1% growth rate -/
theorem population_after_20_years :
  population_growth 13 0.01 20 = 13 * (1 + 0.01)^20 := by
  sorry

#eval population_growth 13 0.01 20

end NUMINAMATH_CALUDE_population_after_20_years_l984_98423


namespace NUMINAMATH_CALUDE_volume_third_number_l984_98457

/-- Given a volume that is the product of three numbers, where two numbers are 12 and 18,
    and 48 cubes of edge 3 can be inserted into it, the third number in the product is 6. -/
theorem volume_third_number (volume : ℕ) (x : ℕ) : 
  volume = 12 * 18 * x →
  volume = 48 * 3^3 →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_volume_third_number_l984_98457


namespace NUMINAMATH_CALUDE_smallest_product_l984_98428

def digits : List Nat := [3, 4, 5, 6]

def valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    valid_arrangement a b c d →
    product a b c d ≥ 1610 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l984_98428


namespace NUMINAMATH_CALUDE_hard_hats_remaining_l984_98411

/-- The number of hard hats remaining in a truck after some are removed --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ)
  (pink_carl pink_john green_john : ℕ) : ℕ :=
  (pink_initial - pink_carl - pink_john) +
  (green_initial - green_john) +
  yellow_initial

theorem hard_hats_remaining :
  remaining_hard_hats 26 15 24 4 6 12 = 43 := by
  sorry

end NUMINAMATH_CALUDE_hard_hats_remaining_l984_98411
