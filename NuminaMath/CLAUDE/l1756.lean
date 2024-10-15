import Mathlib

namespace NUMINAMATH_CALUDE_balloon_count_l1756_175669

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := 41

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := joan_balloons + melanie_balloons

theorem balloon_count : total_balloons = 81 := by sorry

end NUMINAMATH_CALUDE_balloon_count_l1756_175669


namespace NUMINAMATH_CALUDE_cubic_root_power_sum_l1756_175644

theorem cubic_root_power_sum (p q n : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁^2 + q*x₁ + n = 0 →
  x₂^3 + p*x₂^2 + q*x₂ + n = 0 →
  x₃^3 + p*x₃^2 + q*x₃ + n = 0 →
  q^2 = 2*n*p →
  x₁^4 + x₂^4 + x₃^4 = (x₁^2 + x₂^2 + x₃^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_power_sum_l1756_175644


namespace NUMINAMATH_CALUDE_sharp_composition_72_l1756_175606

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem sharp_composition_72 : sharp (sharp (sharp 72)) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_72_l1756_175606


namespace NUMINAMATH_CALUDE_tangerines_left_l1756_175620

theorem tangerines_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 7) :
  initial - eaten = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l1756_175620


namespace NUMINAMATH_CALUDE_tan_x_plus_pi_third_l1756_175636

theorem tan_x_plus_pi_third (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + π / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_plus_pi_third_l1756_175636


namespace NUMINAMATH_CALUDE_confetti_area_difference_l1756_175603

/-- The difference between the area of a square with side length 8 cm and 
    the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem confetti_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by sorry

end NUMINAMATH_CALUDE_confetti_area_difference_l1756_175603


namespace NUMINAMATH_CALUDE_probability_multiple_4_5_7_l1756_175672

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

theorem probability_multiple_4_5_7 (max : ℕ) (h : max = 150) :
  (count_multiples max 4 + count_multiples max 5 + count_multiples max 7
   - count_multiples max 20 - count_multiples max 28 - count_multiples max 35
   + count_multiples max 140) / max = 73 / 150 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_4_5_7_l1756_175672


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l1756_175626

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_roots_bound (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a b (f a b x₁) = 0 ∧ f a b (f a b x₂) = 0 ∧ f a b (f a b x₃) = 0 ∧ f a b (f a b x₄) = 0) →
  (∃ y₁ y₂ : ℝ, f a b (f a b y₁) = 0 ∧ f a b (f a b y₂) = 0 ∧ y₁ + y₂ = -1) →
  b ≤ -1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l1756_175626


namespace NUMINAMATH_CALUDE_no_90_cents_possible_l1756_175657

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a selection of coins --/
structure CoinSelection :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

/-- Checks if a coin selection is valid according to the problem constraints --/
def isValidSelection (s : CoinSelection) : Prop :=
  s.pennies + s.nickels + s.dimes + s.quarters = 6 ∧
  s.pennies ≤ 4 ∧ s.nickels ≤ 4 ∧ s.dimes ≤ 4 ∧ s.quarters ≤ 4

/-- Calculates the total value of a coin selection in cents --/
def totalValue (s : CoinSelection) : Nat :=
  s.pennies * coinValue Coin.Penny +
  s.nickels * coinValue Coin.Nickel +
  s.dimes * coinValue Coin.Dime +
  s.quarters * coinValue Coin.Quarter

/-- Theorem stating that it's impossible to make 90 cents with a valid coin selection --/
theorem no_90_cents_possible :
  ¬∃ (s : CoinSelection), isValidSelection s ∧ totalValue s = 90 := by
  sorry


end NUMINAMATH_CALUDE_no_90_cents_possible_l1756_175657


namespace NUMINAMATH_CALUDE_bakers_sales_l1756_175662

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  cakes_sold - pastries_sold = 11 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l1756_175662


namespace NUMINAMATH_CALUDE_sin_cos_sum_negative_sqrt_two_l1756_175683

theorem sin_cos_sum_negative_sqrt_two (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → Real.sin x + Real.cos x = -Real.sqrt 2 → x = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_negative_sqrt_two_l1756_175683


namespace NUMINAMATH_CALUDE_third_number_in_sequence_l1756_175637

theorem third_number_in_sequence (x : ℕ) 
  (h : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 60) : 
  x + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_sequence_l1756_175637


namespace NUMINAMATH_CALUDE_log_equality_implies_equal_bases_l1756_175691

/-- Proves that for x, y ∈ (0,1) and a > 0, a ≠ 1, if log_x(a) + log_y(a) = 4 log_xy(a), then x = y -/
theorem log_equality_implies_equal_bases
  (x y a : ℝ)
  (h_x : 0 < x ∧ x < 1)
  (h_y : 0 < y ∧ y < 1)
  (h_a : a > 0 ∧ a ≠ 1)
  (h_log : Real.log a / Real.log x + Real.log a / Real.log y = 4 * Real.log a / Real.log (x * y)) :
  x = y :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_equal_bases_l1756_175691


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_circle_center_l1756_175608

/-- A circle with center (1, 3) tangent to the line 3x - 4y - 6 = 0 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 3)^2 = 9}

/-- The line 3x - 4y - 6 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 6 = 0}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p) :=
by sorry

theorem circle_center :
  ∀ (p : ℝ × ℝ), p ∈ TangentCircle → (p.1 - 1)^2 + (p.2 - 3)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_circle_center_l1756_175608


namespace NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l1756_175692

theorem tens_digit_of_36_pow_12 : ∃ n : ℕ, 36^12 ≡ 10*n + 1 [MOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l1756_175692


namespace NUMINAMATH_CALUDE_license_plate_count_l1756_175668

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible characters for the second position (letters + digits) -/
def num_second_choices : ℕ := num_letters + num_digits

/-- The length of the license plate -/
def plate_length : ℕ := 4

/-- Calculates the number of possible license plates given the constraints -/
def num_license_plates : ℕ :=
  num_letters * num_second_choices * 1 * num_digits

/-- Theorem stating that the number of possible license plates is 9360 -/
theorem license_plate_count :
  num_license_plates = 9360 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1756_175668


namespace NUMINAMATH_CALUDE_arrangement_remainder_l1756_175625

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that can be arranged with the blue marbles
    such that the number of marbles with same-color neighbors equals the number of
    marbles with different-color neighbors --/
def max_yellow_marbles : ℕ := 17

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + max_yellow_marbles

/-- The number of possible arrangements of the marbles --/
def num_arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem arrangement_remainder :
  num_arrangements % 1000 = 376 := by sorry

end NUMINAMATH_CALUDE_arrangement_remainder_l1756_175625


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l1756_175682

theorem sufficient_not_necessary_negation 
  (p q : Prop) 
  (h_suff : p → q) 
  (h_not_nec : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_negation_l1756_175682


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1756_175634

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - x + 3

-- Theorem statement
theorem quadratic_minimum :
  ∀ x : ℝ, f x ≥ 11/4 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1756_175634


namespace NUMINAMATH_CALUDE_system_solution_l1756_175605

theorem system_solution :
  ∃ (x y : ℚ), 
    (3 * (x + y) - 4 * (x - y) = 5) ∧
    ((x + y) / 2 + (x - y) / 6 = 0) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1756_175605


namespace NUMINAMATH_CALUDE_eight_times_seven_divided_by_three_l1756_175621

theorem eight_times_seven_divided_by_three :
  (∃ (a b c : ℕ), a = 5 ∧ b = 6 ∧ c = 7 ∧ a * b = 30 ∧ b * c = 42 ∧ c * 8 = 56) →
  (8 * 7) / 3 = 18 ∧ (8 * 7) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_eight_times_seven_divided_by_three_l1756_175621


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l1756_175684

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem folded_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
    perimeter r1 = perimeter r2 + 20 →
    perimeter r2 = perimeter r3 + 16 →
    r1.length = r2.length →
    r2.length = r3.length →
    r1.width = r2.width + 10 →
    r2.width = r3.width + 8 →
    area r1 = 504 :=
by sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l1756_175684


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1756_175663

theorem arithmetic_calculations :
  (156 - 135 / 9 = 141) ∧
  ((124 - 56) / 4 = 17) ∧
  (55 * 6 + 45 * 6 = 600) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1756_175663


namespace NUMINAMATH_CALUDE_derivative_x_sin_x_at_pi_l1756_175614

/-- The derivative of f(x) = x * sin(x) evaluated at π is equal to -π. -/
theorem derivative_x_sin_x_at_pi :
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x
  (deriv f) π = -π :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_sin_x_at_pi_l1756_175614


namespace NUMINAMATH_CALUDE_female_officers_count_l1756_175658

/-- Proves the total number of female officers on a police force given certain conditions -/
theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 152 →
  female_on_duty_percent = 19 / 100 →
  ∃ (total_female : ℕ),
    total_female = 400 ∧
    (total_female : ℚ) * female_on_duty_percent = total_on_duty / 2 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l1756_175658


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_neg_80_l1756_175674

theorem largest_multiple_of_8_less_than_neg_80 :
  ∀ n : ℤ, n % 8 = 0 ∧ n < -80 → n ≤ -88 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_neg_80_l1756_175674


namespace NUMINAMATH_CALUDE_annes_journey_l1756_175693

/-- Calculates the distance traveled given time and speed -/
def distance_traveled (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 3 hours at 2 miles per hour results in a 6-mile journey -/
theorem annes_journey : distance_traveled 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_journey_l1756_175693


namespace NUMINAMATH_CALUDE_zoo_bus_children_l1756_175632

/-- The number of children taking the bus to the zoo -/
def children_count : ℕ := 58

/-- The number of seats needed -/
def seats_needed : ℕ := 29

/-- The number of children per seat -/
def children_per_seat : ℕ := 2

/-- Theorem: The number of children taking the bus to the zoo is 58,
    given that they sit 2 children in every seat and need 29 seats in total. -/
theorem zoo_bus_children :
  children_count = seats_needed * children_per_seat :=
by sorry

end NUMINAMATH_CALUDE_zoo_bus_children_l1756_175632


namespace NUMINAMATH_CALUDE_fruit_picking_orders_l1756_175666

/-- The number of fruits in the basket -/
def n : ℕ := 5

/-- The number of fruits to be picked -/
def k : ℕ := 2

/-- Calculates the number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that picking 2 fruits out of 5 distinct fruits, where order matters, results in 20 different orders -/
theorem fruit_picking_orders : permutations n k = 20 := by sorry

end NUMINAMATH_CALUDE_fruit_picking_orders_l1756_175666


namespace NUMINAMATH_CALUDE_inequality_proof_l1756_175624

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1756_175624


namespace NUMINAMATH_CALUDE_f_monotonicity_and_roots_l1756_175680

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x + 1 - Real.exp (a * x)

theorem f_monotonicity_and_roots :
  (∀ x y : ℝ, x < y → a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y : ℝ, x < y → x < (1/a) * Real.log (2/a) → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > (1/a) * Real.log (2/a) → f a x > f a y)) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ = 1 → f a x₂ = 1 → x₁ + x₂ > 2/a) :=
by sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_roots_l1756_175680


namespace NUMINAMATH_CALUDE_rational_nonzero_l1756_175611

theorem rational_nonzero (a b : ℚ) (h1 : a * b > a) (h2 : a - b > b) : a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_nonzero_l1756_175611


namespace NUMINAMATH_CALUDE_unique_permutations_3_3_3_6_eq_4_l1756_175671

/-- The number of unique permutations of a multiset with 4 elements, where 3 elements are identical --/
def unique_permutations_3_3_3_6 : ℕ :=
  Nat.factorial 4 / Nat.factorial 3

theorem unique_permutations_3_3_3_6_eq_4 : 
  unique_permutations_3_3_3_6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_3_3_3_6_eq_4_l1756_175671


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1756_175677

theorem arithmetic_equality : 245 - 57 + 136 + 14 - 38 = 300 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1756_175677


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l1756_175615

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2019 + b^2018 = -1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l1756_175615


namespace NUMINAMATH_CALUDE_interest_calculation_l1756_175613

/-- Given a principal amount and number of years, if the simple interest
    at 5% per annum is Rs. 56 and the compound interest at the same rate
    is Rs. 57.40, then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 56 →
  P * ((1 + 5/100)^n - 1) = 57.40 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l1756_175613


namespace NUMINAMATH_CALUDE_rope_length_for_second_post_l1756_175633

theorem rope_length_for_second_post
  (total_rope : ℕ)
  (first_post : ℕ)
  (third_post : ℕ)
  (fourth_post : ℕ)
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : third_post = 14)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + third_post + fourth_post) = 20 := by
  sorry

#check rope_length_for_second_post

end NUMINAMATH_CALUDE_rope_length_for_second_post_l1756_175633


namespace NUMINAMATH_CALUDE_jelly_bean_color_match_probability_l1756_175622

def claire_green : ℕ := 2
def claire_red : ℕ := 2
def daniel_green : ℕ := 2
def daniel_yellow : ℕ := 3
def daniel_red : ℕ := 4

def claire_total : ℕ := claire_green + claire_red
def daniel_total : ℕ := daniel_green + daniel_yellow + daniel_red

theorem jelly_bean_color_match_probability :
  (claire_green / claire_total : ℚ) * (daniel_green / daniel_total : ℚ) +
  (claire_red / claire_total : ℚ) * (daniel_red / daniel_total : ℚ) =
  1 / 3 :=
sorry

end NUMINAMATH_CALUDE_jelly_bean_color_match_probability_l1756_175622


namespace NUMINAMATH_CALUDE_quadratic_coincidence_l1756_175610

-- Define the type for 2D points
def Point := ℝ × ℝ

-- Define a line in 2D
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadratic function
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a segment cut by a quadratic function on a line
def SegmentCut (f : QuadraticFunction) (l : Line) : ℝ :=
  sorry

-- Non-parallel lines
def NonParallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b ≠ l₁.b * l₂.a

-- Theorem statement
theorem quadratic_coincidence (f₁ f₂ : QuadraticFunction) (l₁ l₂ : Line) :
  NonParallel l₁ l₂ →
  SegmentCut f₁ l₁ = SegmentCut f₂ l₁ →
  SegmentCut f₁ l₂ = SegmentCut f₂ l₂ →
  f₁ = f₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_coincidence_l1756_175610


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l1756_175695

/-- Proves that a regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 :=
by
  sorry

#check regular_polygon_150_degrees_has_12_sides

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l1756_175695


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_l1756_175690

/-- The cost of paint per quart for a cube with given dimensions and total cost -/
theorem paint_cost_per_quart (edge_length : ℝ) (total_cost : ℝ) (coverage_per_quart : ℝ) : 
  edge_length = 10 →
  total_cost = 192 →
  coverage_per_quart = 10 →
  (total_cost / (6 * edge_length^2 / coverage_per_quart)) = 3.2 := by
sorry

end NUMINAMATH_CALUDE_paint_cost_per_quart_l1756_175690


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1756_175678

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 7)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 6)) ∧ (4 ∣ (m - 7))) → false) ∧
  n = 111 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1756_175678


namespace NUMINAMATH_CALUDE_some_club_members_not_debate_team_l1756_175665

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Club : U → Prop)  -- x is a club member
variable (Punctual : U → Prop)  -- x is punctual
variable (DebateTeam : U → Prop)  -- x is a debate team member

-- Define the premises
variable (h1 : ∃ x, Club x ∧ ¬Punctual x)  -- Some club members are not punctual
variable (h2 : ∀ x, DebateTeam x → Punctual x)  -- All members of the debate team are punctual

-- State the theorem
theorem some_club_members_not_debate_team :
  ∃ x, Club x ∧ ¬DebateTeam x :=
by sorry

end NUMINAMATH_CALUDE_some_club_members_not_debate_team_l1756_175665


namespace NUMINAMATH_CALUDE_complex_multiplication_l1756_175628

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1756_175628


namespace NUMINAMATH_CALUDE_pages_per_donut_l1756_175629

/-- Given Jean's writing and eating habits, calculate the number of pages she writes per donut. -/
theorem pages_per_donut (pages_written : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ)
  (h1 : pages_written = 12)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories = 900) :
  pages_written / (total_calories / calories_per_donut) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pages_per_donut_l1756_175629


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l1756_175699

/-- Represents an isosceles triangle with one angle of 50 degrees -/
structure IsoscelesTriangle where
  /-- The measure of the first angle in degrees -/
  angle1 : ℝ
  /-- The measure of the second angle in degrees -/
  angle2 : ℝ
  /-- The measure of the third angle in degrees -/
  angle3 : ℝ
  /-- The sum of all angles is 180 degrees -/
  sum_of_angles : angle1 + angle2 + angle3 = 180
  /-- One angle is 50 degrees -/
  has_50_degree_angle : angle1 = 50 ∨ angle2 = 50 ∨ angle3 = 50
  /-- The triangle is isosceles (two angles are equal) -/
  is_isosceles : (angle1 = angle2) ∨ (angle2 = angle3) ∨ (angle1 = angle3)

/-- Theorem: In an isosceles triangle with one angle of 50°, the other two angles are 50° and 80° -/
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 80 ∧ t.angle3 = 50) ∨
  (t.angle1 = 80 ∧ t.angle2 = 50 ∧ t.angle3 = 50) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angles_l1756_175699


namespace NUMINAMATH_CALUDE_power_equation_solver_l1756_175630

theorem power_equation_solver (m : ℕ) : 5^m = 5 * 25^3 * 125^2 → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solver_l1756_175630


namespace NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l1756_175604

/-- Calculates the number of permits Andrew stamps in a day -/
def permits_stamped (num_appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamps_per_hour : ℕ) : ℕ :=
  let appointment_time := num_appointments * appointment_duration
  let stamping_time := workday_hours - appointment_time
  stamping_time * stamps_per_hour

/-- Proves that Andrew stamps 100 permits given the specified conditions -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l1756_175604


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1756_175670

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {2, 4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1756_175670


namespace NUMINAMATH_CALUDE_fraction_simplification_l1756_175641

theorem fraction_simplification (d : ℝ) : (5 + 4 * d) / 9 + 3 = (32 + 4 * d) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1756_175641


namespace NUMINAMATH_CALUDE_anne_solo_time_l1756_175654

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom clean_together : bruce_rate + anne_rate = 1 / 4
axiom clean_anne_double : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_solo_time : 1 / anne_rate = 12 := by sorry

end NUMINAMATH_CALUDE_anne_solo_time_l1756_175654


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l1756_175656

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + x - k ≠ 0) ↔ k < -1/8 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l1756_175656


namespace NUMINAMATH_CALUDE_circle_equation_with_radius_5_l1756_175698

/-- Given a circle with equation x^2 - 2x + y^2 + 6y + c = 0 and radius 5, prove c = -15 -/
theorem circle_equation_with_radius_5 (c : ℝ) :
  (∀ x y : ℝ, x^2 - 2*x + y^2 + 6*y + c = 0 ↔ (x - 1)^2 + (y + 3)^2 = 5^2) →
  c = -15 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_radius_5_l1756_175698


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1756_175655

theorem slope_angle_of_line (x y : ℝ) (α : ℝ) :
  x * Real.sin (2 * π / 5) + y * Real.cos (2 * π / 5) = 0 →
  α = 3 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1756_175655


namespace NUMINAMATH_CALUDE_parabola_properties_l1756_175638

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus
def Focus : ℝ × ℝ := (-1, 0)

-- Define the line passing through the focus with slope 45°
def Line (x y : ℝ) : Prop := y = x + 1

-- Define the chord length
def ChordLength : ℝ := 8

theorem parabola_properties :
  -- The parabola passes through (-2, 2√2)
  Parabola (-2) (2 * Real.sqrt 2) ∧
  -- The focus is at (-1, 0)
  Focus = (-1, 0) ∧
  -- The chord formed by the intersection of the parabola and the line has length 8
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = ChordLength :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1756_175638


namespace NUMINAMATH_CALUDE_ellipse_inequality_l1756_175667

noncomputable section

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define the right vertex C
def C : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define a point A on the ellipse in the first quadrant
def A (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Define point B symmetric to A with respect to the origin
def B (α : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 * Real.cos α, -Real.sqrt 2 * Real.sin α)

-- Define point D
def D (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, 
  (Real.sqrt 2 * Real.sin α * (1 - Real.cos α)) / (1 + Real.cos α))

-- State the theorem
theorem ellipse_inequality (α : ℝ) 
  (h1 : 0 < α ∧ α < π/2)  -- Ensure A is in the first quadrant
  (h2 : Ellipse (A α).1 (A α).2)  -- Ensure A is on the ellipse
  : ‖A α - C‖^2 < ‖C - D α‖ * ‖D α - B α‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ellipse_inequality_l1756_175667


namespace NUMINAMATH_CALUDE_prob_red_fifth_black_tenth_correct_l1756_175688

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card based on its number -/
def card_color (n : Fin 52) : Color :=
  if n.val < 26 then Color.Red else Color.Black

/-- Probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
def prob_red_fifth_black_tenth (d : Deck) : ℚ :=
  13 / 51

/-- Theorem stating the probability of drawing a red card as the fifth and a black card as the tenth from a shuffled deck -/
theorem prob_red_fifth_black_tenth_correct (d : Deck) :
  prob_red_fifth_black_tenth d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_fifth_black_tenth_correct_l1756_175688


namespace NUMINAMATH_CALUDE_river_depth_l1756_175617

/-- The depth of a river given its width, flow rate, and volume of water per minute -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) : 
  width = 65 →
  flow_rate = 6 →
  volume_per_minute = 26000 →
  (width * (flow_rate * 1000 / 60) * 4 = volume_per_minute) := by sorry

end NUMINAMATH_CALUDE_river_depth_l1756_175617


namespace NUMINAMATH_CALUDE_min_p_plus_q_l1756_175645

theorem min_p_plus_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 15 * (p + 1) = 29 * (q + 1)) : 
  ∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ 15 * (p' + 1) = 29 * (q' + 1) ∧ 
    p' + q' = 45 ∧ ∀ (p'' q'' : ℕ), p'' > 1 → q'' > 1 → 
      15 * (p'' + 1) = 29 * (q'' + 1) → p'' + q'' ≥ 45 :=
by sorry

end NUMINAMATH_CALUDE_min_p_plus_q_l1756_175645


namespace NUMINAMATH_CALUDE_expo_park_arrangements_l1756_175647

/-- The number of ways to arrange school visits to an Expo Park -/
def schoolVisitArrangements (totalDays : ℕ) (totalSchools : ℕ) (largeSchoolDays : ℕ) : ℕ :=
  Nat.choose (totalDays - 1) 1 * (Nat.factorial (totalDays - largeSchoolDays) / Nat.factorial (totalDays - largeSchoolDays - (totalSchools - 1)))

/-- Theorem stating the number of arrangements for the given scenario -/
theorem expo_park_arrangements :
  schoolVisitArrangements 30 10 2 = Nat.choose 29 1 * (Nat.factorial 28 / Nat.factorial 19) :=
by
  sorry

#eval schoolVisitArrangements 30 10 2

end NUMINAMATH_CALUDE_expo_park_arrangements_l1756_175647


namespace NUMINAMATH_CALUDE_soccer_league_games_l1756_175646

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league of 15 teams where each team plays every other team once, 
    the total number of games played is 105 -/
theorem soccer_league_games : games_played 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1756_175646


namespace NUMINAMATH_CALUDE_exists_fixed_point_with_iteration_l1756_175681

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ x, 1 ≤ f x - x ∧ f x - x ≤ 2019) ∧
  (∀ x, f (f x) % 2019 = x % 2019)

/-- The main theorem -/
theorem exists_fixed_point_with_iteration (f : ℕ → ℕ) (h : SatisfyingFunction f) :
  ∃ x, ∀ k, f^[k] x = x + 2019 * k :=
sorry

end NUMINAMATH_CALUDE_exists_fixed_point_with_iteration_l1756_175681


namespace NUMINAMATH_CALUDE_fir_trees_count_l1756_175659

/-- Represents the statements made by each child -/
inductive Statement
| anya : Statement
| borya : Statement
| vera : Statement
| gena : Statement

/-- Represents the gender of each child -/
inductive Gender
| boy : Gender
| girl : Gender

/-- Checks if a statement is true given the number of trees -/
def isTrue (s : Statement) (n : Nat) : Prop :=
  match s with
  | .anya => n = 15
  | .borya => n % 11 = 0
  | .vera => n < 25
  | .gena => n % 22 = 0

/-- Assigns a gender to each child -/
def gender (s : Statement) : Gender :=
  match s with
  | .anya => .girl
  | .borya => .boy
  | .vera => .girl
  | .gena => .boy

/-- The main theorem to prove -/
theorem fir_trees_count : 
  ∃ (n : Nat), n = 11 ∧ 
  ∃ (s1 s2 : Statement), s1 ≠ s2 ∧ 
  gender s1 ≠ gender s2 ∧
  isTrue s1 n ∧ isTrue s2 n ∧
  ∀ (s : Statement), s ≠ s1 ∧ s ≠ s2 → ¬(isTrue s n) :=
by sorry

end NUMINAMATH_CALUDE_fir_trees_count_l1756_175659


namespace NUMINAMATH_CALUDE_min_value_fraction_l1756_175643

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c)) →
  (x + y) / (x * y * z) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1756_175643


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l1756_175627

/-- Given a mixture of two types of candy, prove the cost of the first type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (selling_price : ℝ)
  (expensive_candy_amount : ℝ)
  (expensive_candy_price : ℝ)
  (h1 : total_mixture = 80)
  (h2 : selling_price = 2.20)
  (h3 : expensive_candy_amount = 16)
  (h4 : expensive_candy_price = 3) :
  ∃ (cheap_candy_price : ℝ),
    cheap_candy_price * (total_mixture - expensive_candy_amount) +
    expensive_candy_price * expensive_candy_amount =
    selling_price * total_mixture ∧
    cheap_candy_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l1756_175627


namespace NUMINAMATH_CALUDE_ben_car_payment_l1756_175689

/-- Ben's monthly finances -/
structure BenFinances where
  gross_income : ℝ
  tax_rate : ℝ
  car_expense_rate : ℝ

/-- Calculate Ben's car payment given his financial structure -/
def car_payment (bf : BenFinances) : ℝ :=
  bf.gross_income * (1 - bf.tax_rate) * bf.car_expense_rate

/-- Theorem: Ben's car payment is $400 given the specified conditions -/
theorem ben_car_payment :
  let bf : BenFinances := {
    gross_income := 3000,
    tax_rate := 1/3,
    car_expense_rate := 0.20
  }
  car_payment bf = 400 := by
  sorry


end NUMINAMATH_CALUDE_ben_car_payment_l1756_175689


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l1756_175602

theorem consecutive_pages_sum (n : ℕ) : n > 0 ∧ n + (n + 1) = 185 → n = 92 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l1756_175602


namespace NUMINAMATH_CALUDE_fraction_equality_l1756_175651

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : (x - y) / (x^2 - y^2) = 1 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1756_175651


namespace NUMINAMATH_CALUDE_triangle_area_equivalence_l1756_175607

/-- Given a triangle with sides a, b, c, semi-perimeter s, and opposite angles α, β, γ,
    prove that the area formula using sines of half-angles is equivalent to Heron's formula. -/
theorem triangle_area_equivalence (a b c s : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_semi_perimeter : s = (a + b + c) / 2)
  (h_angles : α + β + γ = Real.pi) :
  Real.sqrt (a * b * c * s * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2)) = 
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equivalence_l1756_175607


namespace NUMINAMATH_CALUDE_no_outliers_l1756_175640

def data_set : List ℝ := [2, 11, 23, 23, 25, 35, 41, 41, 55, 67, 85]
def Q1 : ℝ := 23
def Q2 : ℝ := 35
def Q3 : ℝ := 55

def is_outlier (x : ℝ) : Prop :=
  let IQR := Q3 - Q1
  x < Q1 - 2 * IQR ∨ x > Q3 + 2 * IQR

theorem no_outliers : ∀ x ∈ data_set, ¬(is_outlier x) := by sorry

end NUMINAMATH_CALUDE_no_outliers_l1756_175640


namespace NUMINAMATH_CALUDE_two_digit_number_twice_product_of_digits_l1756_175696

theorem two_digit_number_twice_product_of_digits : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ n = 2 * (n / 10) * (n % 10) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_two_digit_number_twice_product_of_digits_l1756_175696


namespace NUMINAMATH_CALUDE_scientific_notation_of_2310000_l1756_175661

/-- Proves that 2,310,000 is equal to 2.31 × 10^6 in scientific notation -/
theorem scientific_notation_of_2310000 : 
  2310000 = 2.31 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2310000_l1756_175661


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1756_175675

theorem geometric_progression_fourth_term 
  (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 2^(1/2 : ℝ)) 
  (h₂ : a₂ = 2^(1/4 : ℝ)) 
  (h₃ : a₃ = 2^(1/8 : ℝ)) 
  (h_geo : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) :
  ∃ a₄ : ℝ, a₄ = 2^(1/16 : ℝ) ∧ a₄ = a₃ * (a₃ / a₂) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1756_175675


namespace NUMINAMATH_CALUDE_exists_cube_root_of_3_15_l1756_175616

theorem exists_cube_root_of_3_15 : ∃ n : ℕ, 3^12 * 3^3 = n^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_cube_root_of_3_15_l1756_175616


namespace NUMINAMATH_CALUDE_graph_is_pair_of_lines_l1756_175687

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_lines_l1756_175687


namespace NUMINAMATH_CALUDE_major_premise_is_false_l1756_175635

/-- A plane in 3D space -/
structure Plane3D where
  -- Define plane properties here
  
/-- A line in 3D space -/
structure Line3D where
  -- Define line properties here

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines when two lines are perpendicular -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Theorem stating that the major premise is false -/
theorem major_premise_is_false :
  ¬ ∀ (l : Line3D) (p : Plane3D) (l_in_p : Line3D),
    parallel_line_plane l p →
    line_in_plane l_in_p p →
    parallel_lines l l_in_p :=
  sorry

end NUMINAMATH_CALUDE_major_premise_is_false_l1756_175635


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l1756_175652

-- Define the slopes of the two lines
def slope1 (b : ℝ) : ℝ := 4
def slope2 (b : ℝ) : ℝ := b - 3

-- Define the condition for parallel lines
def are_parallel (b : ℝ) : Prop := slope1 b = slope2 b

-- Theorem statement
theorem parallel_lines_b_value :
  ∃ b : ℝ, are_parallel b ∧ b = 7 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l1756_175652


namespace NUMINAMATH_CALUDE_expression_equals_percentage_of_y_l1756_175648

theorem expression_equals_percentage_of_y (y d : ℝ) (h1 : y > 0) :
  (7 * y / 20 + 3 * y / d) = 0.6499999999999999 * y → d = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_percentage_of_y_l1756_175648


namespace NUMINAMATH_CALUDE_sin_two_theta_value_l1756_175686

theorem sin_two_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) : 
  Real.sin (2*θ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_value_l1756_175686


namespace NUMINAMATH_CALUDE_fraction_division_equality_l1756_175697

theorem fraction_division_equality : 
  (-1/42) / (1/6 - 3/14 + 2/3 - 2/7) = -1/14 := by sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l1756_175697


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1756_175679

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a + b > 0) ∧
  (∃ a b : ℝ, a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1756_175679


namespace NUMINAMATH_CALUDE_even_swaps_not_restore_order_l1756_175660

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The identity permutation (ascending order) -/
def id_perm (n : ℕ) : Permutation n := fun i => i

/-- Swap two elements in a permutation -/
def swap (p : Permutation n) (i j : Fin n) : Permutation n :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- Apply a sequence of swaps to a permutation -/
def apply_swaps (p : Permutation n) (swaps : List (Fin n × Fin n)) : Permutation n :=
  swaps.foldl (fun p' (i, j) => swap p' i j) p

/-- The main theorem -/
theorem even_swaps_not_restore_order (n : ℕ) (swaps : List (Fin n × Fin n)) :
  swaps.length % 2 = 0 → apply_swaps (id_perm n) swaps ≠ id_perm n :=
sorry

end NUMINAMATH_CALUDE_even_swaps_not_restore_order_l1756_175660


namespace NUMINAMATH_CALUDE_smallest_n_with_divisible_sum_or_diff_l1756_175631

theorem smallest_n_with_divisible_sum_or_diff (n : ℕ) : n = 1006 ↔ 
  (∀ (S : Finset ℤ), S.card = n → 
    ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b : ℤ), a ∈ T → b ∈ T → a ≠ b → ¬(2009 ∣ (a + b) ∨ 2009 ∣ (a - b))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_divisible_sum_or_diff_l1756_175631


namespace NUMINAMATH_CALUDE_zoom_setup_ratio_l1756_175609

/-- Represents the time spent on various activities during Mary's Zoom setup and call -/
structure ZoomSetup where
  mac_download : ℕ
  windows_download : ℕ
  audio_glitch_duration : ℕ
  audio_glitch_count : ℕ
  video_glitch_duration : ℕ
  total_time : ℕ

/-- Calculates the ratio of time spent talking without glitches to time spent with glitches -/
def talkTimeRatio (setup : ZoomSetup) : Rat :=
  let total_download_time := setup.mac_download + setup.windows_download
  let total_glitch_time := setup.audio_glitch_duration * setup.audio_glitch_count + setup.video_glitch_duration
  let total_talk_time := setup.total_time - total_download_time
  let talk_time_without_glitches := total_talk_time - total_glitch_time
  talk_time_without_glitches / total_glitch_time

/-- Theorem stating that given the specific conditions, the talk time ratio is 2:1 -/
theorem zoom_setup_ratio : 
  ∀ (setup : ZoomSetup), 
    setup.mac_download = 10 ∧ 
    setup.windows_download = 3 * setup.mac_download ∧
    setup.audio_glitch_duration = 4 ∧
    setup.audio_glitch_count = 2 ∧
    setup.video_glitch_duration = 6 ∧
    setup.total_time = 82 →
    talkTimeRatio setup = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoom_setup_ratio_l1756_175609


namespace NUMINAMATH_CALUDE_solve_equation_l1756_175673

theorem solve_equation (x : ℝ) (h : (8 / x) + 6 = 8) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1756_175673


namespace NUMINAMATH_CALUDE_bad_games_count_l1756_175676

theorem bad_games_count (total_games working_games : ℕ) 
  (h1 : total_games = 11)
  (h2 : working_games = 6) :
  total_games - working_games = 5 := by
sorry

end NUMINAMATH_CALUDE_bad_games_count_l1756_175676


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l1756_175685

/-- Given that z varies inversely as √w, prove that w = 16 when z = 2, 
    given that z = 4 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (4 * Real.sqrt 4 = 4 * Real.sqrt w) → (2 * Real.sqrt w = 4 * Real.sqrt 4) → w = 16 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l1756_175685


namespace NUMINAMATH_CALUDE_team_win_percentage_l1756_175653

theorem team_win_percentage (total_games : ℕ) (win_rate : ℚ) : 
  total_games = 75 → win_rate = 65/100 → 
  (win_rate * total_games) / total_games = 65/100 := by
  sorry

end NUMINAMATH_CALUDE_team_win_percentage_l1756_175653


namespace NUMINAMATH_CALUDE_smallest_nth_root_of_unity_l1756_175639

theorem smallest_nth_root_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_nth_root_of_unity_l1756_175639


namespace NUMINAMATH_CALUDE_train_platform_time_l1756_175618

/-- The time taken for a train to pass a platform -/
def time_to_pass_platform (l : ℝ) (t : ℝ) : ℝ :=
  5 * t

/-- Theorem: The time taken for a train of length l, traveling at a constant velocity, 
    to pass a platform of length 4l is 5 times the time it takes to pass a pole, 
    given that it takes t seconds to pass the pole. -/
theorem train_platform_time (l : ℝ) (t : ℝ) (v : ℝ) :
  l > 0 → t > 0 → v > 0 →
  (l / v = t) →  -- Time to pass pole
  ((l + 4 * l) / v = time_to_pass_platform l t) :=
by sorry

end NUMINAMATH_CALUDE_train_platform_time_l1756_175618


namespace NUMINAMATH_CALUDE_eight_chickens_ten_eggs_l1756_175612

/-- Given that 5 chickens lay 7 eggs in 4 days, this function calculates
    the number of days it takes for 8 chickens to lay 10 eggs. -/
def days_to_lay_eggs (initial_chickens : ℕ) (initial_eggs : ℕ) (initial_days : ℕ)
                     (target_chickens : ℕ) (target_eggs : ℕ) : ℚ :=
  (initial_chickens * initial_days * target_eggs : ℚ) /
  (initial_eggs * target_chickens : ℚ)

/-- Theorem stating that 8 chickens will take 50/7 days to lay 10 eggs,
    given that 5 chickens lay 7 eggs in 4 days. -/
theorem eight_chickens_ten_eggs :
  days_to_lay_eggs 5 7 4 8 10 = 50 / 7 := by
  sorry

#eval days_to_lay_eggs 5 7 4 8 10

end NUMINAMATH_CALUDE_eight_chickens_ten_eggs_l1756_175612


namespace NUMINAMATH_CALUDE_red_chips_count_l1756_175601

def total_chips : ℕ := 60
def green_chips : ℕ := 16

def blue_chips : ℕ := total_chips / 6

def red_chips : ℕ := total_chips - blue_chips - green_chips

theorem red_chips_count : red_chips = 34 := by
  sorry

end NUMINAMATH_CALUDE_red_chips_count_l1756_175601


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l1756_175650

theorem arcsin_equation_solution : 
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ Real.arcsin x + Real.arcsin (3 * x) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l1756_175650


namespace NUMINAMATH_CALUDE_isabel_candy_count_l1756_175694

/-- Calculates the total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given Isabel's initial candy count and the additional pieces she received,
    prove that her total candy count is 93 -/
theorem isabel_candy_count :
  let initial := 68
  let additional := 25
  total_candy initial additional = 93 := by
  sorry

end NUMINAMATH_CALUDE_isabel_candy_count_l1756_175694


namespace NUMINAMATH_CALUDE_converse_proposition_false_l1756_175600

/-- Vectors a and b are collinear -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- The converse of the proposition "If x = 1, then the vectors (-2x, 1) and (-2, x) are collinear" is false -/
theorem converse_proposition_false : ¬ ∀ x : ℝ, 
  are_collinear (-2*x, 1) (-2, x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_converse_proposition_false_l1756_175600


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1756_175642

/-- Given a square ABCD with side length 1, E is the midpoint of AB, 
    F is the intersection of ED and AC, and G is the intersection of EC and BD. 
    The radius r of the circle inscribed in quadrilateral EFPG is equal to |EF| - |FP|. -/
theorem inscribed_circle_radius (A B C D E F G P : ℝ × ℝ) (r : ℝ) : 
  A = (0, 1) →
  B = (1, 1) →
  C = (1, 0) →
  D = (0, 0) →
  E = (1/2, 1) →
  F = (0, 1) →
  G = (2/3, 2/3) →
  P = (1/2, 1/2) →
  r = |EF| - |FP| :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1756_175642


namespace NUMINAMATH_CALUDE_product_change_l1756_175619

theorem product_change (a b : ℝ) (h : a * b = 1620) :
  (4 * a) * (b / 2) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_product_change_l1756_175619


namespace NUMINAMATH_CALUDE_base5_division_theorem_l1756_175623

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base5_division_theorem :
  let dividend := [1, 3, 4, 2]  -- 2431₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [3, 0, 1]     -- 103₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l1756_175623


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1756_175649

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l1756_175649


namespace NUMINAMATH_CALUDE_solutions_correct_l1756_175664

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + y = 4 ∧ 3 * x + 2 * y = 6

def system2 (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ 3 * x - 5 * y = 11

-- State the theorem
theorem solutions_correct :
  (∃ x y : ℝ, system1 x y ∧ x = 2/3 ∧ y = 2) ∧
  (∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_solutions_correct_l1756_175664
