import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1631_163171

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, 
    prove that the 7th term a₇ = 64. -/
theorem geometric_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 2 + a 3 = 6) 
  (h_geom : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = a 2 / a 1) : 
  a 7 = 64 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1631_163171


namespace NUMINAMATH_CALUDE_gcd_of_256_180_720_l1631_163111

theorem gcd_of_256_180_720 : Nat.gcd 256 (Nat.gcd 180 720) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_720_l1631_163111


namespace NUMINAMATH_CALUDE_popsicles_left_l1631_163113

def initial_grape : ℕ := 2
def initial_cherry : ℕ := 13
def initial_banana : ℕ := 2
def initial_mango : ℕ := 8
def initial_strawberry : ℕ := 4
def initial_orange : ℕ := 6

def cherry_eaten : ℕ := 3
def grape_eaten : ℕ := 1

def total_initial : ℕ := initial_grape + initial_cherry + initial_banana + initial_mango + initial_strawberry + initial_orange

def total_eaten : ℕ := cherry_eaten + grape_eaten

theorem popsicles_left : total_initial - total_eaten = 31 := by
  sorry

end NUMINAMATH_CALUDE_popsicles_left_l1631_163113


namespace NUMINAMATH_CALUDE_parking_lot_spaces_l1631_163163

/-- The number of spaces a single caravan occupies -/
def spaces_per_caravan : ℕ := 2

/-- The number of caravans currently parked -/
def number_of_caravans : ℕ := 3

/-- The number of spaces left for other vehicles -/
def spaces_left : ℕ := 24

/-- The total number of spaces in the parking lot -/
def total_spaces : ℕ := spaces_per_caravan * number_of_caravans + spaces_left

theorem parking_lot_spaces : total_spaces = 30 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_spaces_l1631_163163


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l1631_163120

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_mean (x y z : ℕ) : Prop :=
  x * z = y * y

theorem four_digit_number_problem (a b c d : ℕ) :
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 0 ∧ b ≤ 9 ∧
  c ≥ 0 ∧ c ≤ 9 ∧
  d ≥ 0 ∧ d ≤ 9 ∧
  is_arithmetic_sequence a b c d ∧
  is_geometric_mean a b d ∧
  1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 11110 →
  (a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5) ∨ (a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l1631_163120


namespace NUMINAMATH_CALUDE_intersection_and_complement_eq_union_l1631_163195

/-- Given the universal set ℝ, prove that the intersection of M and the complement of N in ℝ
    is the union of {x | x < -2} and {x | x ≥ 3} -/
theorem intersection_and_complement_eq_union (M N : Set ℝ) : 
  M = {x : ℝ | x^2 > 4} →
  N = {x : ℝ | (x - 3) / (x + 1) < 0} →
  M ∩ (Set.univ \ N) = {x : ℝ | x < -2} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_eq_union_l1631_163195


namespace NUMINAMATH_CALUDE_system_solution_l1631_163139

theorem system_solution (x y z : ℝ) : 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^3 = 2*y^2 - z ∧
  y^3 = 2*z^2 - x ∧
  z^3 = 2*x^2 - y →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1631_163139


namespace NUMINAMATH_CALUDE_coral_reef_decrease_l1631_163145

def coral_decrease_rate : ℝ := 0.3
def target_percentage : ℝ := 0.05
def years_since_2010 : ℕ := 10

theorem coral_reef_decrease :
  (1 - coral_decrease_rate) ^ years_since_2010 < target_percentage := by
  sorry

end NUMINAMATH_CALUDE_coral_reef_decrease_l1631_163145


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_2718_and_gcd_l1631_163199

theorem largest_four_digit_divisible_by_2718_and_gcd : ∃ (n : ℕ), n ≤ 9999 ∧ n % 2718 = 0 ∧ 
  (∀ m : ℕ, m ≤ 9999 ∧ m % 2718 = 0 → m ≤ n) ∧
  n = 8154 ∧
  Nat.gcd n 8640 = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_2718_and_gcd_l1631_163199


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l1631_163123

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 flat discount) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (20% off the amount over $150) -/
def savingsC (price : ℝ) : ℝ := 0.20 * (price - 150)

/-- The theorem to be proved -/
theorem coupon_savings_difference : 
  ∃ (min_price max_price : ℝ),
    (∀ price, price > 150 → 
      (savingsA price > savingsB ∧ savingsA price > savingsC price) ↔ 
      (min_price ≤ price ∧ price ≤ max_price)) ∧
    max_price - min_price = 400 :=
sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l1631_163123


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1631_163156

/-- The sum of the infinite series ∑(n=0 to ∞) (2^n / 5^n) is equal to 5/3 -/
theorem geometric_series_sum : 
  let a : ℕ → ℝ := λ n => (2 : ℝ)^n
  (∑' n, a n / 5^n) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1631_163156


namespace NUMINAMATH_CALUDE_fraction_simplification_l1631_163140

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1631_163140


namespace NUMINAMATH_CALUDE_purchases_per_customer_l1631_163108

/-- Given a parking lot scenario, prove that each customer makes exactly one purchase. -/
theorem purchases_per_customer (num_cars : ℕ) (customers_per_car : ℕ) 
  (sports_sales : ℕ) (music_sales : ℕ) 
  (h1 : num_cars = 10) 
  (h2 : customers_per_car = 5) 
  (h3 : sports_sales = 20) 
  (h4 : music_sales = 30) : 
  (sports_sales + music_sales) / (num_cars * customers_per_car) = 1 :=
by sorry

end NUMINAMATH_CALUDE_purchases_per_customer_l1631_163108


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1631_163168

theorem complex_fraction_simplification :
  (7 : ℂ) + 9*I / (3 : ℂ) - 4*I = 57/25 + 55/25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1631_163168


namespace NUMINAMATH_CALUDE_triangle_side_length_l1631_163112

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.α > 0 ∧ t.β > 0 ∧ t.γ > 0 ∧
  t.α + t.β + t.γ = Real.pi ∧
  3 * t.α + 2 * t.β = Real.pi

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h : TriangleProperties t) 
  (ha : t.a = 2) 
  (hb : t.b = 3) : 
  t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1631_163112


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1631_163146

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.height

/-- The dimensions of the larger box -/
def largeBox : BoxDimensions :=
  { length := 4, width := 3, height := 3 }

/-- The dimensions of the smaller block -/
def smallBlock : BoxDimensions :=
  { length := 3, width := 2, height := 1 }

/-- Theorem: The maximum number of small blocks that can fit in the large box is 6 -/
theorem max_blocks_fit : 
  (volume largeBox) / (volume smallBlock) = 6 ∧ 
  (2 * smallBlock.length ≤ largeBox.length) ∧
  (smallBlock.width ≤ largeBox.width) ∧
  (2 * smallBlock.height ≤ largeBox.height) := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1631_163146


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1631_163110

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 37 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1631_163110


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1631_163131

open Real

theorem sufficient_but_not_necessary (θ : ℝ) : 
  (∀ θ, |θ - π/12| < π/12 → sin θ < 1/2) ∧ 
  (∃ θ, sin θ < 1/2 ∧ |θ - π/12| ≥ π/12) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1631_163131


namespace NUMINAMATH_CALUDE_train_length_l1631_163106

/-- Given a train that crosses a platform in 39 seconds, crosses a signal pole in 20 seconds,
    and the platform length is 285 meters, the length of the train is 300 meters. -/
theorem train_length (crossing_time_platform : ℝ) (crossing_time_pole : ℝ) (platform_length : ℝ)
    (h1 : crossing_time_platform = 39)
    (h2 : crossing_time_pole = 20)
    (h3 : platform_length = 285) :
    ∃ train_length : ℝ, train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1631_163106


namespace NUMINAMATH_CALUDE_expression_decrease_l1631_163192

theorem expression_decrease (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  (x' * y' ^ 2) / (x * y ^ 2) = 0.216 := by sorry

end NUMINAMATH_CALUDE_expression_decrease_l1631_163192


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l1631_163104

theorem hexagon_angle_measure :
  ∀ (a b c d e : ℝ),
    a = 135 ∧ b = 150 ∧ c = 120 ∧ d = 130 ∧ e = 100 →
    ∃ (q : ℝ),
      q = 85 ∧
      a + b + c + d + e + q = 720 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l1631_163104


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1631_163196

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1631_163196


namespace NUMINAMATH_CALUDE_students_in_sunghoons_class_l1631_163116

theorem students_in_sunghoons_class 
  (jisoo_students : ℕ) 
  (product : ℕ) 
  (h1 : jisoo_students = 36)
  (h2 : jisoo_students * sunghoon_students = product)
  (h3 : product = 1008) : 
  sunghoon_students = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_sunghoons_class_l1631_163116


namespace NUMINAMATH_CALUDE_circle_division_l1631_163138

theorem circle_division (OA : ℝ) (OA_pos : OA > 0) :
  ∃ (OC OB : ℝ),
    OC = (OA * Real.sqrt 3) / 3 ∧
    OB = (OA * Real.sqrt 6) / 3 ∧
    π * OC^2 = π * (OB^2 - OC^2) ∧
    π * (OB^2 - OC^2) = π * (OA^2 - OB^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_division_l1631_163138


namespace NUMINAMATH_CALUDE_trigonometric_product_sqrt_l1631_163100

theorem trigonometric_product_sqrt (h1 : Real.sin (π / 6) = 1 / 2)
                                   (h2 : Real.sin (π / 4) = Real.sqrt 2 / 2)
                                   (h3 : Real.sin (π / 3) = Real.sqrt 3 / 2) :
  Real.sqrt ((2 - (Real.sin (π / 6))^2) * (2 - (Real.sin (π / 4))^2) * (2 - (Real.sin (π / 3))^2)) = Real.sqrt 210 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_sqrt_l1631_163100


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l1631_163127

/-- The minimum distance between a point on the line y = (15/8)x - 4 and a point on the parabola y = x^2 is 47/32 -/
theorem min_distance_line_parabola :
  let line := fun (x : ℝ) => (15/8) * x - 4
  let parabola := fun (x : ℝ) => x^2
  ∃ (x₁ x₂ : ℝ),
    (∀ (y₁ y₂ : ℝ),
      (line y₁ = (15/8) * y₁ - 4) →
      (parabola y₂ = y₂^2) →
      ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) ≤ ((y₂ - y₁)^2 + (parabola y₂ - line y₁)^2)^(1/2)) ∧
    ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)^(1/2) = 47/32 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l1631_163127


namespace NUMINAMATH_CALUDE_min_students_in_class_l1631_163174

theorem min_students_in_class (b g : ℕ) : 
  b = 2 * g →  -- ratio of boys to girls is 2:1
  (3 * b) / 5 = (5 * g) / 8 →  -- number of boys who passed equals number of girls who passed
  b + g ≥ 120 ∧ ∀ n < 120, ¬(∃ b' g', b' = 2 * g' ∧ (3 * b') / 5 = (5 * g') / 8 ∧ b' + g' = n) :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_class_l1631_163174


namespace NUMINAMATH_CALUDE_function_properties_l1631_163126

-- Define the function f
def f (a c : ℕ) (x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- State the theorem
theorem function_properties (a c : ℕ) (h1 : a ≠ 0) (h2 : c ≠ 0) :
  (f a c 1 = 5) →
  (6 < f a c 2 ∧ f a c 2 < 11) →
  (a = 1 ∧ c = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, f a c x - 2 * m * x ≤ 1) → m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1631_163126


namespace NUMINAMATH_CALUDE_power_product_equals_1938400_l1631_163125

theorem power_product_equals_1938400 : 2^4 * 3^2 * 5^2 * 7^2 * 11 = 1938400 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_1938400_l1631_163125


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1631_163160

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), (x = 6 + (182 : ℚ) / 999) ∧ (1000 * x - x = 6182 - 6) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1631_163160


namespace NUMINAMATH_CALUDE_same_parity_of_extrema_l1631_163169

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest_element (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_of_extrema :
  is_even (smallest_element A_P) ↔ is_even (largest_element A_P) := by
  sorry

end NUMINAMATH_CALUDE_same_parity_of_extrema_l1631_163169


namespace NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l1631_163128

/-- Calculates the number of potatoes Christen peeled --/
def christenPotatoesCount (totalPotatoes : ℕ) (homerRate : ℕ) (christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerInitialPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := totalPotatoes - homerInitialPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

theorem christen_peeled_21_potatoes :
  christenPotatoesCount 60 4 6 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l1631_163128


namespace NUMINAMATH_CALUDE_estimate_event_knowledge_chengdu_games_knowledge_estimate_l1631_163155

/-- Estimates the number of people in a population who know about an event,
    given a sample survey result. -/
theorem estimate_event_knowledge (total_population : ℕ) 
                                  (sample_size : ℕ) 
                                  (sample_positive : ℕ) : ℕ :=
  let estimate := (sample_positive * total_population) / sample_size
  estimate

/-- Proves that the estimated number of people who know about the event
    in a population of 10,000, given 125 out of 200 know in a sample, is 6250. -/
theorem chengdu_games_knowledge_estimate :
  estimate_event_knowledge 10000 200 125 = 6250 := by
  sorry

end NUMINAMATH_CALUDE_estimate_event_knowledge_chengdu_games_knowledge_estimate_l1631_163155


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1631_163150

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 18 = -15) →
  (a 2 * a 18 = 16) →
  a 3 * a 10 * a 17 = -64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1631_163150


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_l1631_163162

theorem tic_tac_toe_tie (amy_win : ℚ) (lily_win : ℚ) (tie : ℚ) : 
  amy_win = 5/12 → lily_win = 1/4 → tie = 1 - (amy_win + lily_win) → tie = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_l1631_163162


namespace NUMINAMATH_CALUDE_parallel_vectors_l1631_163180

/-- Given vectors a and b in ℝ², prove that if a is parallel to b, then λ = 8/5 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.1 / b.1 = a.2 / b.2) :
  a = (2, 5) → b.2 = 4 → b.1 = 8/5 := by
  sorry

#check parallel_vectors

end NUMINAMATH_CALUDE_parallel_vectors_l1631_163180


namespace NUMINAMATH_CALUDE_extra_sweets_per_child_l1631_163172

theorem extra_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (sweets_per_present_child : ℕ) :
  total_children = 190 →
  absent_children = 70 →
  sweets_per_present_child = 38 →
  (total_children - absent_children) * sweets_per_present_child / total_children - 
    ((total_children - absent_children) * sweets_per_present_child / total_children) = 14 := by
  sorry

end NUMINAMATH_CALUDE_extra_sweets_per_child_l1631_163172


namespace NUMINAMATH_CALUDE_entree_cost_l1631_163151

theorem entree_cost (total : ℝ) (difference : ℝ) (entree : ℝ) (dessert : ℝ)
  (h1 : total = 23)
  (h2 : difference = 5)
  (h3 : entree = dessert + difference)
  (h4 : total = entree + dessert) :
  entree = 14 := by
sorry

end NUMINAMATH_CALUDE_entree_cost_l1631_163151


namespace NUMINAMATH_CALUDE_positive_difference_l1631_163191

theorem positive_difference (x y w : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hw : 0.5 < w ∧ w < 1) : 
  w - y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_l1631_163191


namespace NUMINAMATH_CALUDE_magical_coin_expected_winnings_l1631_163157

/-- Represents the outcomes of the magical coin flip -/
inductive Outcome
  | Heads
  | Tails
  | Edge
  | Disappear

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 3/8
  | Outcome.Tails => 1/4
  | Outcome.Edge => 1/8
  | Outcome.Disappear => 1/4

/-- The winnings (or losses) for each outcome -/
def winnings (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 2
  | Outcome.Tails => 5
  | Outcome.Edge => -2
  | Outcome.Disappear => -6

/-- The expected winnings of flipping the magical coin -/
def expected_winnings : ℚ :=
  (probability Outcome.Heads * winnings Outcome.Heads) +
  (probability Outcome.Tails * winnings Outcome.Tails) +
  (probability Outcome.Edge * winnings Outcome.Edge) +
  (probability Outcome.Disappear * winnings Outcome.Disappear)

theorem magical_coin_expected_winnings :
  expected_winnings = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_magical_coin_expected_winnings_l1631_163157


namespace NUMINAMATH_CALUDE_quotient_negative_one_sum_zero_l1631_163133

theorem quotient_negative_one_sum_zero (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = -1 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quotient_negative_one_sum_zero_l1631_163133


namespace NUMINAMATH_CALUDE_complex_equality_squared_l1631_163122

theorem complex_equality_squared (m n : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : m * (1 + i) = 1 + n * i) : 
  ((m + n * i) / (m - n * i))^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_squared_l1631_163122


namespace NUMINAMATH_CALUDE_pie_eating_contest_l1631_163183

theorem pie_eating_contest (first_participant second_participant : ℚ) : 
  first_participant = 5/6 → second_participant = 2/3 → 
  first_participant - second_participant = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l1631_163183


namespace NUMINAMATH_CALUDE_division_problem_l1631_163132

theorem division_problem (n : ℕ) : n / 20 = 10 ∧ n % 20 = 10 ↔ n = 210 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1631_163132


namespace NUMINAMATH_CALUDE_line_properties_l1631_163197

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 2

theorem line_properties :
  -- Part 1: The line passes through (1, 1) for all real m
  (∀ m : ℝ, line_equation m 1 1) ∧
  -- Part 2: When the line is tangent to the circle, m = -1
  (∃ m : ℝ, (∀ x y : ℝ, line_equation m x y → circle_equation x y → 
    (x - 0)^2 + (y - 0)^2 = (1 - m)^2 / (m^2 + 1)) ∧ m = -1) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1631_163197


namespace NUMINAMATH_CALUDE_solve_for_m_l1631_163152

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (m : ℝ) : Prop :=
  (1 - m * i) / (i^3) = 1 + i

-- Theorem statement
theorem solve_for_m :
  ∃ (m : ℝ), equation m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_solve_for_m_l1631_163152


namespace NUMINAMATH_CALUDE_remainder_theorem_l1631_163177

theorem remainder_theorem : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1631_163177


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1631_163176

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ
  /-- The center of the circle lies on the longer base of the trapezoid -/
  center_on_longer_base : Bool

/-- The area of an inscribed isosceles trapezoid -/
def area (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific inscribed trapezoid is 32 -/
theorem area_of_specific_trapezoid :
  let t : InscribedTrapezoid := ⟨5, 6, true⟩
  area t = 32 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1631_163176


namespace NUMINAMATH_CALUDE_sector_angle_l1631_163193

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) : 
  (1/2 * θ * r^2 = 1) → (2*r + θ*r = 4) → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1631_163193


namespace NUMINAMATH_CALUDE_positive_real_solution_l1631_163149

theorem positive_real_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : 3 * Real.sqrt (x^2 + x) + 3 * Real.sqrt (x^2 - x) = 6 * Real.sqrt 2) : 
  x = 4 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l1631_163149


namespace NUMINAMATH_CALUDE_intersection_equals_B_B_proper_superset_A_l1631_163129

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

-- Theorem for part 1
theorem intersection_equals_B (m : ℝ) : (A ∩ B m) = B m ↔ m ≤ 1 := by sorry

-- Theorem for part 2
theorem B_proper_superset_A (m : ℝ) : A ⊂ B m ↔ m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_B_proper_superset_A_l1631_163129


namespace NUMINAMATH_CALUDE_divisibility_proof_l1631_163173

theorem divisibility_proof (n : ℕ) (x : ℝ) :
  ∃ P : ℝ → ℝ, (x + 1)^(2*n) - x^(2*n) - 2*x - 1 = x * (x + 1) * (2*x + 1) * P x :=
by sorry

end NUMINAMATH_CALUDE_divisibility_proof_l1631_163173


namespace NUMINAMATH_CALUDE_brandy_caffeine_excess_l1631_163186

/-- Represents the caffeine consumption and limits for an individual -/
structure CaffeineProfile where
  weight : ℝ
  additionalTolerance : ℝ
  coffeeConsumption : ℕ
  coffeinePer : ℝ
  energyDrinkConsumption : ℕ
  energyDrinkCaffeine : ℝ
  standardLimit : ℝ

/-- Calculates the total safe caffeine amount for an individual -/
def totalSafeAmount (profile : CaffeineProfile) : ℝ :=
  profile.weight * profile.standardLimit + profile.additionalTolerance

/-- Calculates the total caffeine consumed -/
def totalConsumed (profile : CaffeineProfile) : ℝ :=
  (profile.coffeeConsumption : ℝ) * profile.coffeinePer +
  (profile.energyDrinkConsumption : ℝ) * profile.energyDrinkCaffeine

/-- Theorem stating that Brandy has exceeded her safe caffeine limit by 470 mg -/
theorem brandy_caffeine_excess (brandy : CaffeineProfile)
  (h1 : brandy.weight = 60)
  (h2 : brandy.additionalTolerance = 50)
  (h3 : brandy.coffeeConsumption = 2)
  (h4 : brandy.coffeinePer = 95)
  (h5 : brandy.energyDrinkConsumption = 4)
  (h6 : brandy.energyDrinkCaffeine = 120)
  (h7 : brandy.standardLimit = 2.5) :
  totalConsumed brandy - totalSafeAmount brandy = 470 := by
  sorry

end NUMINAMATH_CALUDE_brandy_caffeine_excess_l1631_163186


namespace NUMINAMATH_CALUDE_memory_card_cost_memory_card_cost_is_60_l1631_163178

/-- The cost of a single memory card given the following conditions:
  * John takes 10 pictures daily for 3 years
  * Each memory card stores 50 images
  * The total spent on memory cards is $13,140 -/
theorem memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (total_spent : ℕ) : ℕ :=
  let days_per_year : ℕ := 365
  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := total_pictures / images_per_card
  total_spent / cards_needed

/-- Proof that the cost of each memory card is $60 -/
theorem memory_card_cost_is_60 : memory_card_cost 10 3 50 13140 = 60 := by
  sorry

end NUMINAMATH_CALUDE_memory_card_cost_memory_card_cost_is_60_l1631_163178


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_segment_cut_on_x_axis_l1631_163161

-- Define the quadratic function
def f (k x : ℝ) : ℝ := (k + 2) * x^2 - 2 * k * x + 3 * k

-- Theorem for the vertex on x-axis
theorem vertex_on_x_axis (k : ℝ) :
  (∃ x, f k x = 0 ∧ ∀ y, f k y ≥ f k x) ↔ k = 0 ∨ k = -3 := by sorry

-- Theorem for the segment cut on x-axis
theorem segment_cut_on_x_axis (k : ℝ) :
  (∃ a b, a > b ∧ f k a = 0 ∧ f k b = 0 ∧ a - b = 4) ↔ k = -8/3 ∨ k = -1 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_segment_cut_on_x_axis_l1631_163161


namespace NUMINAMATH_CALUDE_cos_72_degrees_l1631_163175

theorem cos_72_degrees : Real.cos (72 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_degrees_l1631_163175


namespace NUMINAMATH_CALUDE_composition_f_one_ninth_l1631_163118

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem composition_f_one_ninth :
  f (f (1/9)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_composition_f_one_ninth_l1631_163118


namespace NUMINAMATH_CALUDE_picnic_problem_l1631_163154

theorem picnic_problem (total : ℕ) (men_excess : ℕ) (adult_excess : ℕ) :
  total = 240 →
  men_excess = 80 →
  adult_excess = 80 →
  ∃ (men women adults children : ℕ),
    men = women + men_excess ∧
    adults = children + adult_excess ∧
    men + women = adults ∧
    adults + children = total ∧
    men = 120 := by
  sorry

end NUMINAMATH_CALUDE_picnic_problem_l1631_163154


namespace NUMINAMATH_CALUDE_car_dealership_monthly_payment_l1631_163143

/-- Calculates the total monthly payment for employees in a car dealership --/
theorem car_dealership_monthly_payment 
  (fiona_hours : ℕ) 
  (john_hours : ℕ) 
  (jeremy_hours : ℕ) 
  (hourly_rate : ℕ) 
  (h1 : fiona_hours = 40)
  (h2 : john_hours = 30)
  (h3 : jeremy_hours = 25)
  (h4 : hourly_rate = 20)
  : (fiona_hours + john_hours + jeremy_hours) * hourly_rate * 4 = 7600 := by
  sorry


end NUMINAMATH_CALUDE_car_dealership_monthly_payment_l1631_163143


namespace NUMINAMATH_CALUDE_square_side_length_l1631_163119

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 8 →
  rectangle_width = 10 →
  4 * square_side = 2 * (rectangle_length + rectangle_width) →
  square_side = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1631_163119


namespace NUMINAMATH_CALUDE_constant_calculation_l1631_163115

theorem constant_calculation (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 8100) → (∀ m : ℤ, m ≤ 8) → c = 126.5625 := by
sorry

end NUMINAMATH_CALUDE_constant_calculation_l1631_163115


namespace NUMINAMATH_CALUDE_man_walked_five_minutes_l1631_163147

/-- Represents the scenario of a man walking home and being picked up by his wife --/
structure WalkingScenario where
  usual_travel_time : ℕ  -- Time it usually takes to drive from station to home
  early_arrival : ℕ      -- How early the man arrived at the station (in minutes)
  time_saved : ℕ         -- How much earlier they arrived home than usual

/-- Calculates the time the man spent walking given a WalkingScenario --/
def time_spent_walking (scenario : WalkingScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific scenario, the man spent 5 minutes walking --/
theorem man_walked_five_minutes :
  let scenario : WalkingScenario := {
    usual_travel_time := 10,
    early_arrival := 60,
    time_saved := 10
  }
  time_spent_walking scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_walked_five_minutes_l1631_163147


namespace NUMINAMATH_CALUDE_sqrt_product_plus_factorial_equals_1114_l1631_163148

theorem sqrt_product_plus_factorial_equals_1114 : 
  Real.sqrt ((35 * 34 * 33 * 32) + 24) = 1114 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_factorial_equals_1114_l1631_163148


namespace NUMINAMATH_CALUDE_problem_solution_l1631_163101

def M : Set ℝ := {x | x^2 - 3*x ≤ 10}
def N (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

theorem problem_solution :
  (∀ x, x ∈ (Set.univ \ M) ∪ (N 2) ↔ x > 5 ∨ x < -2) ∧
  (∀ a, M ∪ N a = M ↔ a < -2 ∨ (-1 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1631_163101


namespace NUMINAMATH_CALUDE_N2O_molecular_weight_l1631_163141

/-- The atomic weight of nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of N2O in g/mol -/
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

/-- Theorem stating that the molecular weight of N2O is 44.02 g/mol -/
theorem N2O_molecular_weight : molecular_weight_N2O = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O_molecular_weight_l1631_163141


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1631_163164

theorem triangle_area_inequality (a b c T : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hT : T > 0) (triangle_area : T^2 = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c) / 16) :
  T^2 ≤ a * b * c * (a + b + c) / 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1631_163164


namespace NUMINAMATH_CALUDE_x_squared_plus_k_factorization_l1631_163170

theorem x_squared_plus_k_factorization (k : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k = (x - a) * (x - b)) ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_k_factorization_l1631_163170


namespace NUMINAMATH_CALUDE_square_root_range_l1631_163105

theorem square_root_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_square_root_range_l1631_163105


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1631_163117

theorem floor_equation_solution :
  ∀ m n : ℕ+,
  (⌊(m^2 : ℚ) / n⌋ + ⌊(n^2 : ℚ) / m⌋ = ⌊(m : ℚ) / n + (n : ℚ) / m⌋ + m * n) ↔ (m = 2 ∧ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1631_163117


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l1631_163181

theorem volunteer_arrangement_count : 
  (n : ℕ) → 
  (total : ℕ) → 
  (day1 : ℕ) → 
  (day2 : ℕ) → 
  (day3 : ℕ) → 
  n = 4 → 
  total = 5 → 
  day1 = 1 → 
  day2 = 2 → 
  day3 = 1 → 
  day1 + day2 + day3 = n →
  (total.choose day1) * ((total - day1).choose day2) * ((total - day1 - day2).choose day3) = 60 := by
sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l1631_163181


namespace NUMINAMATH_CALUDE_max_people_served_l1631_163142

theorem max_people_served (total_budget : ℚ) (min_food_spend : ℚ) (cheapest_food_cost : ℚ) (cheapest_drink_cost : ℚ) 
  (h1 : total_budget = 12.5)
  (h2 : min_food_spend = 10)
  (h3 : cheapest_food_cost = 0.6)
  (h4 : cheapest_drink_cost = 0.5) :
  ∃ (n : ℕ), n = 10 ∧ 
    n * (cheapest_food_cost + cheapest_drink_cost) ≤ total_budget ∧
    n * cheapest_food_cost ≥ min_food_spend ∧
    ∀ (m : ℕ), m > n → 
      m * (cheapest_food_cost + cheapest_drink_cost) > total_budget ∨
      m * cheapest_food_cost < min_food_spend :=
by
  sorry

#check max_people_served

end NUMINAMATH_CALUDE_max_people_served_l1631_163142


namespace NUMINAMATH_CALUDE_sapling_growth_relation_l1631_163136

/-- Represents the height of a sapling over time -/
def sapling_height (x : ℝ) : ℝ :=
  50 * x + 100

theorem sapling_growth_relation (x : ℝ) (y : ℝ) 
  (h1 : sapling_height 0 = 100) 
  (h2 : ∀ x1 x2, sapling_height x2 - sapling_height x1 = 50 * (x2 - x1)) :
  y = sapling_height x :=
by sorry

end NUMINAMATH_CALUDE_sapling_growth_relation_l1631_163136


namespace NUMINAMATH_CALUDE_cos_fourteen_pi_thirds_l1631_163194

theorem cos_fourteen_pi_thirds : Real.cos (14 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_fourteen_pi_thirds_l1631_163194


namespace NUMINAMATH_CALUDE_combine_terms_implies_zero_sum_l1631_163159

theorem combine_terms_implies_zero_sum (a b x y : ℝ) : 
  (∃ k : ℝ, -3 * a^(2*x-1) * b = k * 5 * a * b^(y+4)) → 
  (x - 2)^2016 + (y + 2)^2017 = 0 := by
sorry

end NUMINAMATH_CALUDE_combine_terms_implies_zero_sum_l1631_163159


namespace NUMINAMATH_CALUDE_equation_solution_l1631_163124

theorem equation_solution :
  ∃ (a b c d : ℚ),
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1631_163124


namespace NUMINAMATH_CALUDE_divisible_by_21_with_sqrt_between_30_and_30_5_l1631_163153

theorem divisible_by_21_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (21 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_21_with_sqrt_between_30_and_30_5_l1631_163153


namespace NUMINAMATH_CALUDE_innings_count_l1631_163158

/-- Represents the batting statistics of a batsman -/
structure BattingStats where
  n : ℕ                -- Total number of innings
  highest : ℕ          -- Highest score
  lowest : ℕ           -- Lowest score
  average : ℚ          -- Average score
  newAverage : ℚ       -- Average after excluding highest and lowest scores

/-- Theorem stating the conditions and the result to be proved -/
theorem innings_count (stats : BattingStats) : 
  stats.average = 50 ∧ 
  stats.highest - stats.lowest = 172 ∧
  stats.newAverage = stats.average - 2 ∧
  stats.highest = 174 →
  stats.n = 40 := by
  sorry


end NUMINAMATH_CALUDE_innings_count_l1631_163158


namespace NUMINAMATH_CALUDE_budget_calculation_l1631_163165

/-- The total budget for purchasing a TV, computer, and fridge -/
def total_budget (tv_cost computer_cost fridge_extra_cost : ℕ) : ℕ :=
  tv_cost + computer_cost + (computer_cost + fridge_extra_cost)

/-- Theorem stating that the total budget for the given costs is 1600 -/
theorem budget_calculation :
  total_budget 600 250 500 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_budget_calculation_l1631_163165


namespace NUMINAMATH_CALUDE_max_range_of_five_numbers_l1631_163185

theorem max_range_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- Distinct and ordered
  (a + b + c + d + e) / 5 = 13 →   -- Average is 13
  c = 15 →                         -- Median is 15
  e - a ≤ 33 :=                    -- Maximum range is at most 33
by sorry

end NUMINAMATH_CALUDE_max_range_of_five_numbers_l1631_163185


namespace NUMINAMATH_CALUDE_field_trip_cost_theorem_l1631_163198

/-- Calculates the total cost of a field trip with a group discount --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ)
  (student_fee : ℚ) (adult_fee : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  let student_cost := total_students * student_fee
  let adult_cost := total_adults * adult_fee
  let total_cost := student_cost + adult_cost
  let discount := if total_students > discount_threshold then discount_rate * student_cost else 0
  total_cost - discount

/-- The total cost of the field trip is $987.60 --/
theorem field_trip_cost_theorem :
  field_trip_cost 4 42 6 (11/2) (13/2) (1/10) 40 = 9876/10 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_cost_theorem_l1631_163198


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1631_163190

theorem quadratic_roots_property (f g : ℝ) : 
  (3 * f^2 + 5 * f - 8 = 0) → 
  (3 * g^2 + 5 * g - 8 = 0) → 
  (f - 2) * (g - 2) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1631_163190


namespace NUMINAMATH_CALUDE_polynomial_division_degree_l1631_163102

open Polynomial

theorem polynomial_division_degree (f d q r : ℝ[X]) : 
  degree f = 15 →
  f = d * q + r →
  degree q = 9 →
  degree r = 4 →
  degree r < degree d →
  degree d = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_degree_l1631_163102


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_integers_l1631_163144

/-- A function that returns the nth prime number -/
noncomputable def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns the product of the first n primes -/
noncomputable def productOfFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Definition of an interesting number -/
def isInteresting (k : ℕ) : Prop :=
  k > 0 ∧ (productOfFirstNPrimes k) % k = 0

/-- Theorem stating that the maximal number of consecutive interesting integers is 7 -/
theorem max_consecutive_interesting_integers :
  ∃ n : ℕ, n > 0 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ n → isInteresting k) ∧
  (∀ m : ℕ, m > n → ¬(∀ k : ℕ, k > 0 ∧ k ≤ m → isInteresting k)) ∧
  n = 7 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_interesting_integers_l1631_163144


namespace NUMINAMATH_CALUDE_max_value_of_f_l1631_163184

/-- The function f(x) defined as sin(2x) - 2√3 * sin²(x) has a maximum value of 2 - √3 -/
theorem max_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x) - 2 * Real.sqrt 3 * (Real.sin x) ^ 2
  ∃ (M : ℝ), M = 2 - Real.sqrt 3 ∧ ∀ x, f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1631_163184


namespace NUMINAMATH_CALUDE_value_of_A_minus_2B_A_minus_2B_independent_of_y_l1631_163188

/-- Definition of A in terms of x and y -/
def A (x y : ℝ) : ℝ := 2 * x^2 + x * y + 3 * y

/-- Definition of B in terms of x and y -/
def B (x y : ℝ) : ℝ := x^2 - x * y

/-- Theorem stating the value of A - 2B under the given condition -/
theorem value_of_A_minus_2B (x y : ℝ) :
  (x + 2)^2 + |y - 3| = 0 → A x y - 2 * B x y = -9 := by sorry

/-- Theorem stating the condition for A - 2B to be independent of y -/
theorem A_minus_2B_independent_of_y (x : ℝ) :
  (∀ y : ℝ, ∃ k : ℝ, A x y - 2 * B x y = k) ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_value_of_A_minus_2B_A_minus_2B_independent_of_y_l1631_163188


namespace NUMINAMATH_CALUDE_remaining_money_l1631_163182

def initial_amount : ℚ := 100
def apple_price : ℚ := 1.5
def orange_price : ℚ := 2
def pear_price : ℚ := 2.25
def apple_quantity : ℕ := 5
def orange_quantity : ℕ := 10
def pear_quantity : ℕ := 4

theorem remaining_money :
  initial_amount - 
  (apple_price * apple_quantity + 
   orange_price * orange_quantity + 
   pear_price * pear_quantity) = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1631_163182


namespace NUMINAMATH_CALUDE_integer_equation_solution_l1631_163167

theorem integer_equation_solution (x y : ℤ) : x^4 - 2*y^2 = 1 → (x = 1 ∨ x = -1) ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_equation_solution_l1631_163167


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l1631_163137

theorem greatest_power_under_500 (a b : ℕ) :
  a > 0 → b > 1 → a^b < 500 →
  (∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → x^y ≤ a^b) →
  a + b = 24 := by
sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l1631_163137


namespace NUMINAMATH_CALUDE_house_prices_and_yields_l1631_163187

theorem house_prices_and_yields :
  ∀ (price1 price2 yield1 yield2 : ℝ),
  price1 > 0 ∧ price2 > 0 ∧ yield1 > 0 ∧ yield2 > 0 →
  425 = (yield1 / 100) * price1 →
  459 = (yield2 / 100) * price2 →
  price2 = (6 / 5) * price1 →
  yield2 = yield1 - (1 / 2) →
  price1 = 8500 ∧ price2 = 10200 ∧ yield1 = 5 ∧ yield2 = (9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_house_prices_and_yields_l1631_163187


namespace NUMINAMATH_CALUDE_total_books_count_l1631_163130

/-- The number of bookshelves -/
def num_bookshelves : ℕ := 1250

/-- The number of books on each bookshelf -/
def books_per_shelf : ℕ := 45

/-- The total number of books on all shelves -/
def total_books : ℕ := num_bookshelves * books_per_shelf

theorem total_books_count : total_books = 56250 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1631_163130


namespace NUMINAMATH_CALUDE_moneybox_fills_in_60_weeks_l1631_163107

/-- The number of weeks it takes for Monica's moneybox to get full -/
def weeks_to_fill : ℕ := sorry

/-- The amount Monica puts into her moneybox each week -/
def weekly_savings : ℕ := 15

/-- The number of times Monica repeats the saving process -/
def repetitions : ℕ := 5

/-- The total amount Monica takes to the bank -/
def total_savings : ℕ := 4500

/-- Theorem stating that the moneybox gets full in 60 weeks -/
theorem moneybox_fills_in_60_weeks : 
  weeks_to_fill = 60 ∧ 
  weekly_savings * weeks_to_fill * repetitions = total_savings :=
sorry

end NUMINAMATH_CALUDE_moneybox_fills_in_60_weeks_l1631_163107


namespace NUMINAMATH_CALUDE_parallelogram_xy_sum_l1631_163121

/-- A parallelogram with sides a, b, c, d where opposite sides are equal -/
structure Parallelogram where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  opposite_sides_equal : a = c ∧ b = d

/-- The specific parallelogram from the problem -/
def problem_parallelogram (x y : ℝ) : Parallelogram where
  a := 6 * y - 2
  b := 12
  c := 3 * x + 4
  d := 9
  opposite_sides_equal := by sorry

theorem parallelogram_xy_sum (x y : ℝ) :
  (problem_parallelogram x y).a = (problem_parallelogram x y).c ∧
  (problem_parallelogram x y).b = (problem_parallelogram x y).d →
  x + y = 4 := by sorry

end NUMINAMATH_CALUDE_parallelogram_xy_sum_l1631_163121


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1631_163179

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l1631_163179


namespace NUMINAMATH_CALUDE_gus_ate_fourteen_eggs_l1631_163109

/-- Represents the number of eggs in each dish Gus ate throughout the day -/
def eggs_per_dish : List Nat := [2, 1, 3, 2, 1, 2, 3]

/-- The total number of eggs Gus ate -/
def total_eggs : Nat := eggs_per_dish.sum

/-- Theorem stating that the total number of eggs Gus ate is 14 -/
theorem gus_ate_fourteen_eggs : total_eggs = 14 := by sorry

end NUMINAMATH_CALUDE_gus_ate_fourteen_eggs_l1631_163109


namespace NUMINAMATH_CALUDE_effective_annual_rate_l1631_163135

/-- The effective annual compound interest rate for a 4-year investment -/
theorem effective_annual_rate (initial_investment final_amount : ℝ)
  (rate1 rate2 rate3 rate4 : ℝ) (h_initial : initial_investment = 810)
  (h_final : final_amount = 1550) (h_rate1 : rate1 = 0.05)
  (h_rate2 : rate2 = 0.07) (h_rate3 : rate3 = 0.06) (h_rate4 : rate4 = 0.04) :
  ∃ (r : ℝ), (abs (r - 0.1755) < 0.0001 ∧
  final_amount = initial_investment * ((1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)) ∧
  final_amount = initial_investment * (1 + r)^4) :=
sorry

end NUMINAMATH_CALUDE_effective_annual_rate_l1631_163135


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l1631_163166

/-- The area of a square with perimeter 32 cm is 64 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (side : ℝ) (area : ℝ) :
  perimeter = 32 →
  side = perimeter / 4 →
  area = side * side →
  area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l1631_163166


namespace NUMINAMATH_CALUDE_max_xy_value_l1631_163189

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 18) :
  x * y ≤ 81 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 18 ∧ x * y = 81 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1631_163189


namespace NUMINAMATH_CALUDE_duck_purchase_difference_l1631_163103

/-- Represents the number of ducks bought by each person -/
structure DuckPurchase where
  adelaide : ℕ
  ephraim : ℕ
  kolton : ℕ

/-- The conditions of the duck purchase problem -/
def DuckProblemConditions (d : DuckPurchase) : Prop :=
  d.adelaide = 2 * d.ephraim ∧
  d.adelaide = 30 ∧
  (d.adelaide + d.ephraim + d.kolton) / 3 = 35

/-- The theorem stating the difference between Kolton's and Ephraim's duck purchases -/
theorem duck_purchase_difference (d : DuckPurchase) :
  DuckProblemConditions d → d.kolton - d.ephraim = 45 := by
  sorry


end NUMINAMATH_CALUDE_duck_purchase_difference_l1631_163103


namespace NUMINAMATH_CALUDE_parallelogram_area_18_16_l1631_163134

/-- The area of a parallelogram given its base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 16 cm is 288 square centimeters -/
theorem parallelogram_area_18_16 : 
  parallelogram_area 18 16 = 288 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_16_l1631_163134


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1631_163114

theorem right_triangle_acute_angles (α β : Real) : 
  α = 30 → β = 90 → ∃ γ : Real, γ = 60 ∧ α + β + γ = 180 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1631_163114
