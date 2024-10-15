import Mathlib

namespace NUMINAMATH_CALUDE_zeros_properties_l3005_300554

noncomputable def f (x : ℝ) : ℝ := (2 * x / (x - 2))^2 - 3^x

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 3) - 4 / (x - 2) - 2

theorem zeros_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 2) (h₂ : x₂ > 2) 
  (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) 
  (hg₁ : g x₁ = 0) (hg₂ : g x₂ = 0) :
  x₂ > 3 ∧ 2*x₁ + 2*x₂ = x₁*x₂ ∧ x₁*x₂ > 16 :=
by sorry

end NUMINAMATH_CALUDE_zeros_properties_l3005_300554


namespace NUMINAMATH_CALUDE_common_chord_length_l3005_300542

/-- The length of the common chord of two circles -/
theorem common_chord_length (c1 c2 : ℝ × ℝ → Prop) : 
  (∀ x y, c1 (x, y) ↔ x^2 + y^2 = 4) →
  (∀ x y, c2 (x, y) ↔ x^2 + y^2 - 2*y - 6 = 0) →
  ∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
    ∀ x y, (c1 (x, y) ∧ c2 (x, y)) → 
      (x^2 + y^2 = 4 ∧ y = -1) ∨ 
      (x^2 + y^2 - 2*y - 6 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3005_300542


namespace NUMINAMATH_CALUDE_unique_solution_l3005_300540

theorem unique_solution : ∃! (x : ℕ), 
  x > 0 ∧ 
  let n := x^2 + 4*x + 23
  let d := 3*x + 7
  n = d*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3005_300540


namespace NUMINAMATH_CALUDE_computer_pricing_l3005_300528

/-- Represents the prices of computer components -/
structure Prices where
  basic_computer : ℝ
  printer : ℝ
  regular_monitor : ℝ

/-- Proves the correct prices given the problem conditions -/
theorem computer_pricing (prices : Prices) 
  (total_basic : prices.basic_computer + prices.printer + prices.regular_monitor = 3000)
  (enhanced_printer_ratio : prices.printer = (1/4) * (prices.basic_computer + 500 + prices.printer + prices.regular_monitor + 300)) :
  prices.printer = 950 ∧ prices.basic_computer + prices.regular_monitor = 2050 := by
  sorry


end NUMINAMATH_CALUDE_computer_pricing_l3005_300528


namespace NUMINAMATH_CALUDE_jessica_withdrawal_l3005_300533

/-- Proves that given the conditions of Jessica's bank transactions, 
    the amount she initially withdrew was $200. -/
theorem jessica_withdrawal (B : ℚ) : 
  (3/5 * B + 1/5 * (3/5 * B) = 360) → 
  (2/5 * B = 200) := by
  sorry

#eval (2/5 : ℚ) * 500 -- Optional: to verify the result

end NUMINAMATH_CALUDE_jessica_withdrawal_l3005_300533


namespace NUMINAMATH_CALUDE_transaction_gain_per_year_l3005_300520

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem transaction_gain_per_year
  (principal : ℚ)
  (borrowRate lendRate : ℚ)
  (time : ℚ)
  (h1 : principal = 8000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 160 := by
  sorry

end NUMINAMATH_CALUDE_transaction_gain_per_year_l3005_300520


namespace NUMINAMATH_CALUDE_amount_lent_to_C_l3005_300549

/-- Amount lent to B in rupees -/
def amount_B : ℝ := 5000

/-- Duration of loan to B in years -/
def duration_B : ℝ := 2

/-- Duration of loan to C in years -/
def duration_C : ℝ := 4

/-- Annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- Total interest received from both B and C in rupees -/
def total_interest : ℝ := 1760

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem amount_lent_to_C : ∃ (amount_C : ℝ),
  amount_C = 3000 ∧
  simple_interest amount_B interest_rate duration_B +
  simple_interest amount_C interest_rate duration_C = total_interest :=
sorry

end NUMINAMATH_CALUDE_amount_lent_to_C_l3005_300549


namespace NUMINAMATH_CALUDE_x_over_y_equals_four_l3005_300519

theorem x_over_y_equals_four (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_equals_four_l3005_300519


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3005_300571

def alice_number : ℕ := 30

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3005_300571


namespace NUMINAMATH_CALUDE_min_buses_required_l3005_300570

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 625) (h2 : bus_capacity = 47) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_buses_required_l3005_300570


namespace NUMINAMATH_CALUDE_parallelogram_count_l3005_300534

/-- The number of parallelograms formed by lines passing through each grid point in a triangle -/
def f (n : ℕ) : ℕ := 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24

/-- Theorem stating that f(n) correctly calculates the number of parallelograms -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l3005_300534


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l3005_300557

/-- Calculates the downstream speed of a rower given their upstream and still water speeds -/
def downstreamSpeed (upstreamSpeed stillWaterSpeed : ℝ) : ℝ :=
  2 * stillWaterSpeed - upstreamSpeed

theorem rower_downstream_speed
  (upstreamSpeed : ℝ)
  (stillWaterSpeed : ℝ)
  (h1 : upstreamSpeed = 25)
  (h2 : stillWaterSpeed = 33) :
  downstreamSpeed upstreamSpeed stillWaterSpeed = 41 := by
  sorry

#eval downstreamSpeed 25 33

end NUMINAMATH_CALUDE_rower_downstream_speed_l3005_300557


namespace NUMINAMATH_CALUDE_tangent_circles_count_l3005_300573

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius + c2.radius)^2 ∨
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius - c2.radius)^2

/-- A circle is tangent to two other circles if it's tangent to both of them -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

/-- The main theorem: there are exactly 6 circles of radius 5 tangent to two tangent circles of radius 2 -/
theorem tangent_circles_count (c1 c2 : Circle) 
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), (∀ c ∈ s, c.radius = 5 ∧ is_tangent_to_both c c1 c2) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l3005_300573


namespace NUMINAMATH_CALUDE_bruce_egg_count_after_loss_l3005_300595

/-- Given Bruce's initial egg count and the number of eggs he loses,
    calculate Bruce's final egg count. -/
def bruces_final_egg_count (initial_count : ℕ) (eggs_lost : ℕ) : ℕ :=
  initial_count - eggs_lost

/-- Theorem stating that given Bruce's initial egg count of 215 and a loss of 137 eggs,
    Bruce's final egg count is 78. -/
theorem bruce_egg_count_after_loss :
  bruces_final_egg_count 215 137 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bruce_egg_count_after_loss_l3005_300595


namespace NUMINAMATH_CALUDE_cat_count_l3005_300514

/-- The number of cats that can jump -/
def jump : ℕ := 45

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can meow -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 20

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 23

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the training center -/
def total_cats : ℕ := 67

theorem cat_count : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = total_cats :=
by sorry

end NUMINAMATH_CALUDE_cat_count_l3005_300514


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3005_300588

/-- Represents the number of pots for each color of chrysanthemums -/
structure ChrysanthemumPots where
  yellow : Nat
  white : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemum pots -/
def countArrangements (pots : ChrysanthemumPots) : Nat :=
  sorry

/-- Theorem stating the number of arrangements for the given conditions -/
theorem chrysanthemum_arrangement_count :
  let pots : ChrysanthemumPots := { yellow := 2, white := 2, red := 1 }
  countArrangements pots = 24 := by
  sorry

end NUMINAMATH_CALUDE_chrysanthemum_arrangement_count_l3005_300588


namespace NUMINAMATH_CALUDE_correct_amount_paid_l3005_300579

/-- The amount paid by Mr. Doré given the costs of items and change received -/
def amount_paid (pants_cost shirt_cost tie_cost change : ℕ) : ℕ :=
  pants_cost + shirt_cost + tie_cost + change

/-- Theorem stating that the amount paid is correct given the problem conditions -/
theorem correct_amount_paid :
  amount_paid 140 43 15 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_paid_l3005_300579


namespace NUMINAMATH_CALUDE_parabola_directrix_l3005_300569

/-- The equation of the directrix of the parabola y = x^2 is 4y + 1 = 0 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), k * y + 1 = 0 ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3005_300569


namespace NUMINAMATH_CALUDE_line_mb_equals_two_l3005_300572

/-- Given a line passing through points (0, -1) and (-1, 1) with equation y = mx + b, prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  ((-1 : ℝ) = m * 0 + b) →      -- Point (0, -1)
  (1 : ℝ) = m * (-1) + b →      -- Point (-1, 1)
  m * b = 2 := by
sorry

end NUMINAMATH_CALUDE_line_mb_equals_two_l3005_300572


namespace NUMINAMATH_CALUDE_exercise_book_count_l3005_300560

/-- Given a ratio of pencils to exercise books and the number of pencils,
    calculate the number of exercise books -/
def calculate_exercise_books (pencil_ratio : ℕ) (book_ratio : ℕ) (num_pencils : ℕ) : ℕ :=
  (num_pencils / pencil_ratio) * book_ratio

/-- Theorem: In a shop with 140 pencils and a pencil to exercise book ratio of 14:3,
    there are 30 exercise books -/
theorem exercise_book_count :
  calculate_exercise_books 14 3 140 = 30 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l3005_300560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3005_300564

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3005_300564


namespace NUMINAMATH_CALUDE_sin_cos_equality_l3005_300561

theorem sin_cos_equality (θ : Real) (h : Real.sin θ * Real.cos θ = 1/2) :
  Real.sin θ - Real.cos θ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_l3005_300561


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3005_300547

theorem trig_expression_equals_one : 
  (Real.cos (36 * π / 180) * Real.sin (24 * π / 180) + 
   Real.sin (144 * π / 180) * Real.sin (84 * π / 180)) / 
  (Real.cos (44 * π / 180) * Real.sin (16 * π / 180) + 
   Real.sin (136 * π / 180) * Real.sin (76 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3005_300547


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3005_300567

theorem asterisk_replacement : (42 / 21) * (42 / 84) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3005_300567


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l3005_300503

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l3005_300503


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l3005_300513

theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2 * x - a ≤ -1 ↔ x ≤ 1) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l3005_300513


namespace NUMINAMATH_CALUDE_power_addition_l3005_300591

theorem power_addition (x y : ℝ) (a b : ℝ) 
  (h1 : (4 : ℝ) ^ x = a) 
  (h2 : (4 : ℝ) ^ y = b) : 
  (4 : ℝ) ^ (x + y) = a * b := by
  sorry

end NUMINAMATH_CALUDE_power_addition_l3005_300591


namespace NUMINAMATH_CALUDE_expected_sides_is_four_l3005_300559

/-- The number of cuts made in one hour -/
def k : ℕ := 3600

/-- The initial number of sides of the rectangular sheet -/
def initial_sides : ℕ := 4

/-- The total number of sides after k cuts -/
def total_sides (k : ℕ) : ℕ := initial_sides + 4 * k

/-- The total number of polygons after k cuts -/
def total_polygons (k : ℕ) : ℕ := k + 1

/-- The expected number of sides of a randomly picked polygon after k cuts -/
def expected_sides (k : ℕ) : ℚ :=
  (total_sides k : ℚ) / (total_polygons k : ℚ)

theorem expected_sides_is_four :
  expected_sides k = 4 := by sorry

end NUMINAMATH_CALUDE_expected_sides_is_four_l3005_300559


namespace NUMINAMATH_CALUDE_find_number_l3005_300574

theorem find_number : ∃! x : ℝ, 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3005_300574


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_side_condition_l3005_300530

theorem triangle_isosceles_from_side_condition (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_condition : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) :
  a = b ∨ b = c ∨ c = a := by
sorry


end NUMINAMATH_CALUDE_triangle_isosceles_from_side_condition_l3005_300530


namespace NUMINAMATH_CALUDE_product_of_squares_l3005_300507

theorem product_of_squares (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l3005_300507


namespace NUMINAMATH_CALUDE_range_of_a_l3005_300586

theorem range_of_a (a : ℝ) : 
  (a - 3*3 < 4*a*3 + 2) → 
  (a - 3*0 < 4*a*0 + 2) → 
  (-1 < a ∧ a < 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3005_300586


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3005_300510

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -3]

theorem vector_addition_and_scalar_multiplication :
  (a + 2 • b) = ![4, -5] := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3005_300510


namespace NUMINAMATH_CALUDE_brother_payment_l3005_300529

/-- Margaux's daily earnings from her money lending company -/
structure DailyEarnings where
  friend : ℝ
  brother : ℝ
  cousin : ℝ

/-- The total earnings after a given number of days -/
def total_earnings (e : DailyEarnings) (days : ℝ) : ℝ :=
  (e.friend + e.brother + e.cousin) * days

/-- Theorem stating that Margaux's brother pays $8 per day -/
theorem brother_payment (e : DailyEarnings) :
  e.friend = 5 ∧ e.cousin = 4 ∧ total_earnings e 7 = 119 → e.brother = 8 := by
  sorry

end NUMINAMATH_CALUDE_brother_payment_l3005_300529


namespace NUMINAMATH_CALUDE_pentagonal_pillar_faces_l3005_300518

/-- Represents a pentagonal pillar -/
structure PentagonalPillar :=
  (rectangular_faces : Nat)
  (pentagonal_faces : Nat)

/-- The total number of faces of a pentagonal pillar -/
def total_faces (p : PentagonalPillar) : Nat :=
  p.rectangular_faces + p.pentagonal_faces

/-- Theorem stating that a pentagonal pillar has 7 faces -/
theorem pentagonal_pillar_faces :
  ∀ (p : PentagonalPillar),
  p.rectangular_faces = 5 ∧ p.pentagonal_faces = 2 →
  total_faces p = 7 := by
  sorry

#check pentagonal_pillar_faces

end NUMINAMATH_CALUDE_pentagonal_pillar_faces_l3005_300518


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l3005_300563

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, prove that the equation of the ellipse
    whose foci are the vertices of the hyperbola and whose vertices are the foci of the hyperbola
    is x²/4 + y²/16 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
  ((x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l3005_300563


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l3005_300558

theorem inverse_proportion_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -4 / 1 → y₂ = -4 / 2 → y₃ = -4 / (-3) → y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l3005_300558


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l3005_300526

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l3005_300526


namespace NUMINAMATH_CALUDE_expression_simplification_l3005_300500

theorem expression_simplification (x y : ℝ) :
  (-2 * x^2 * y) * (-3 * x * y)^2 / (3 * x * y^2) = -6 * x^3 * y := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3005_300500


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3005_300590

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 8 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3005_300590


namespace NUMINAMATH_CALUDE_last_two_digits_of_2007_power_20077_l3005_300501

theorem last_two_digits_of_2007_power_20077 : 2007^20077 % 100 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2007_power_20077_l3005_300501


namespace NUMINAMATH_CALUDE_pie_eating_difference_l3005_300593

theorem pie_eating_difference :
  let first_participant : ℚ := 5/6
  let second_participant : ℚ := 2/3
  first_participant - second_participant = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_difference_l3005_300593


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l3005_300580

/-- The total number of technical personnel --/
def total_personnel : ℕ := 37

/-- The number of attendees --/
def n : ℕ := 18

/-- Proves that n satisfies the conditions of the systematic sampling problem --/
theorem systematic_sampling_proof :
  (total_personnel - 1) % n = 0 ∧ 
  (total_personnel - 3) % (n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l3005_300580


namespace NUMINAMATH_CALUDE_rationalize_sqrt3_minus1_l3005_300545

theorem rationalize_sqrt3_minus1 : 1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt3_minus1_l3005_300545


namespace NUMINAMATH_CALUDE_factorization_equality_l3005_300598

theorem factorization_equality (a : ℝ) : 2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3005_300598


namespace NUMINAMATH_CALUDE_sphere_radius_is_zero_l3005_300582

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The configuration of points and lines in the problem -/
structure Configuration where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  m : Line3D
  n : Line3D
  a : ℝ
  b : ℝ
  θ : ℝ

/-- Checks if two points are distinct -/
def are_distinct (p q : Point3D) : Prop :=
  p ≠ q

/-- Checks if a line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a point is on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Calculates the radius of a sphere passing through four points -/
def sphere_radius (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the radius of the sphere is zero -/
theorem sphere_radius_is_zero (config : Configuration) :
  are_distinct config.A config.B ∧
  is_perpendicular config.m (Line3D.mk config.A config.B) ∧
  is_perpendicular config.n (Line3D.mk config.A config.B) ∧
  point_on_line config.C config.m ∧
  are_distinct config.A config.C ∧
  point_on_line config.D config.n ∧
  are_distinct config.B config.D ∧
  distance config.A config.B = config.a ∧
  distance config.C config.D = config.b ∧
  angle_between_lines config.m config.n = config.θ
  →
  sphere_radius config.A config.B config.C config.D = 0 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_is_zero_l3005_300582


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3005_300517

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 2 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = (x - 1) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3005_300517


namespace NUMINAMATH_CALUDE_problem_statement_l3005_300522

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a^2 + b^2 ≥ x^2 + y^2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (a^2 + b^2 = 1/5) ∧
  (a*b ≤ 1/8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x*y ≤ 1/8) ∧
  (1/a + 1/b ≥ 3 + 2*Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3005_300522


namespace NUMINAMATH_CALUDE_diamond_value_l3005_300594

theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) 
  (h2 : diamond * 9 + 5 = diamond * 10 + 2) : diamond = 3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l3005_300594


namespace NUMINAMATH_CALUDE_combined_probability_l3005_300575

/-- The probability that Xavier solves Problem A -/
def p_xa : ℚ := 1/5

/-- The probability that Yvonne solves Problem A -/
def p_ya : ℚ := 1/2

/-- The probability that Zelda solves Problem A -/
def p_za : ℚ := 5/8

/-- The probability that Xavier solves Problem B -/
def p_xb : ℚ := 2/9

/-- The probability that Yvonne solves Problem B -/
def p_yb : ℚ := 3/5

/-- The probability that Zelda solves Problem B -/
def p_zb : ℚ := 1/4

/-- The probability that Xavier solves Problem C -/
def p_xc : ℚ := 1/4

/-- The probability that Yvonne solves Problem C -/
def p_yc : ℚ := 3/8

/-- The probability that Zelda solves Problem C -/
def p_zc : ℚ := 9/16

/-- The theorem stating the probability of the combined event -/
theorem combined_probability : 
  p_xa * p_ya * p_yb * (1 - p_yc) * (1 - p_xc) * (1 - p_zc) = 63/2048 := by
  sorry

end NUMINAMATH_CALUDE_combined_probability_l3005_300575


namespace NUMINAMATH_CALUDE_min_value_problem_l3005_300546

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3005_300546


namespace NUMINAMATH_CALUDE_female_officers_count_l3005_300508

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_on_duty_percent = 17 / 100 →
  (total_on_duty / 2 : ℚ) = female_on_duty_percent * (600 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l3005_300508


namespace NUMINAMATH_CALUDE_system_solution_l3005_300504

theorem system_solution : ∃ (x y z : ℝ), 
  (x + 2*y + 3*z = 3) ∧ 
  (3*x + y + 2*z = 7) ∧ 
  (2*x + 3*y + z = 2) ∧
  (x = 2) ∧ (y = -1) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3005_300504


namespace NUMINAMATH_CALUDE_jay_scored_six_more_l3005_300543

/-- Represents the scores of players in a basketball game. -/
structure BasketballScores where
  tobee : ℕ
  jay : ℕ
  sean : ℕ

/-- Conditions of the basketball game scores. -/
def validScores (scores : BasketballScores) : Prop :=
  scores.tobee = 4 ∧
  scores.jay > scores.tobee ∧
  scores.sean = scores.tobee + scores.jay - 2 ∧
  scores.tobee + scores.jay + scores.sean = 26

/-- Theorem stating that Jay scored 6 more points than Tobee. -/
theorem jay_scored_six_more (scores : BasketballScores) 
  (h : validScores scores) : scores.jay = scores.tobee + 6 := by
  sorry

#check jay_scored_six_more

end NUMINAMATH_CALUDE_jay_scored_six_more_l3005_300543


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l3005_300578

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  /-- The number of small cubes along each edge of the large cube -/
  edge_count : ℕ
  /-- All faces of the large cube are painted -/
  all_faces_painted : Bool
  /-- The number of small cubes with three faces colored -/
  three_face_colored_count : ℕ

/-- Theorem: If a cube is cut so that 8 small cubes have three faces colored, 
    then the total number of small cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) 
  (h1 : c.all_faces_painted = true) 
  (h2 : c.three_face_colored_count = 8) : 
  c.edge_count ^ 3 = 8 := by
  sorry

#check cut_cube_theorem

end NUMINAMATH_CALUDE_cut_cube_theorem_l3005_300578


namespace NUMINAMATH_CALUDE_book_sale_revenue_book_sale_revenue_proof_l3005_300541

theorem book_sale_revenue : ℕ → ℕ → ℕ → Prop :=
  fun total_books sold_price remaining_books =>
    (2 * total_books = 3 * remaining_books) →
    (total_books - remaining_books) * sold_price = 288

-- Proof
theorem book_sale_revenue_proof :
  book_sale_revenue 108 4 36 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_book_sale_revenue_proof_l3005_300541


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3005_300524

theorem smallest_number_with_remainders : ∃ N : ℕ, 
  N > 0 ∧ 
  N % 13 = 2 ∧ 
  N % 15 = 4 ∧ 
  (∀ M : ℕ, M > 0 → M % 13 = 2 → M % 15 = 4 → N ≤ M) ∧
  N = 184 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3005_300524


namespace NUMINAMATH_CALUDE_sallys_class_size_l3005_300539

theorem sallys_class_size (school_money : ℕ) (book_cost : ℕ) (out_of_pocket : ℕ) :
  school_money = 320 →
  book_cost = 12 →
  out_of_pocket = 40 →
  (school_money + out_of_pocket) / book_cost = 30 := by
sorry

end NUMINAMATH_CALUDE_sallys_class_size_l3005_300539


namespace NUMINAMATH_CALUDE_video_game_points_l3005_300576

/-- 
Given a video game level with the following conditions:
- There are 6 enemies in total
- Each defeated enemy gives 3 points
- 2 enemies are not defeated

Prove that the total points earned is 12.
-/
theorem video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (undefeated_enemies : ℕ) :
  total_enemies = 6 →
  points_per_enemy = 3 →
  undefeated_enemies = 2 →
  (total_enemies - undefeated_enemies) * points_per_enemy = 12 :=
by sorry

end NUMINAMATH_CALUDE_video_game_points_l3005_300576


namespace NUMINAMATH_CALUDE_math_book_cost_l3005_300538

theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_price = 373 →
  math_books = 27 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + history_book_cost * (total_books - math_books) = total_price ∧
    math_book_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_book_cost_l3005_300538


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l3005_300553

theorem derivative_at_pi_third (f : ℝ → ℝ) (h : f = λ x => Real.cos x + Real.sqrt 3 * Real.sin x) :
  deriv f (π / 3) = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l3005_300553


namespace NUMINAMATH_CALUDE_triangle_with_prime_angles_l3005_300562

theorem triangle_with_prime_angles (a b c : ℕ) : 
  a + b + c = 180 →
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) →
  a = 2 ∨ b = 2 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_prime_angles_l3005_300562


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3005_300577

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (3 - 1 - a - b - c)) →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 51 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l3005_300577


namespace NUMINAMATH_CALUDE_apple_sale_profit_percentage_l3005_300550

/-- Represents the shopkeeper's apple selling scenario -/
structure AppleSale where
  total_apples : ℝ
  sell_percent_1 : ℝ
  profit_percent_1 : ℝ
  sell_percent_2 : ℝ
  profit_percent_2 : ℝ
  sell_percent_3 : ℝ
  profit_percent_3 : ℝ
  unsold_percent : ℝ
  additional_expenses : ℝ

/-- Calculates the effective profit percentage for the given apple sale scenario -/
def effectiveProfitPercentage (sale : AppleSale) : ℝ :=
  sorry

/-- Theorem stating the effective profit percentage for the given scenario -/
theorem apple_sale_profit_percentage :
  let sale : AppleSale := {
    total_apples := 120,
    sell_percent_1 := 0.4,
    profit_percent_1 := 0.25,
    sell_percent_2 := 0.3,
    profit_percent_2 := 0.35,
    sell_percent_3 := 0.2,
    profit_percent_3 := 0.2,
    unsold_percent := 0.1,
    additional_expenses := 20
  }
  ∃ (ε : ℝ), ε > 0 ∧ abs (effectiveProfitPercentage sale + 2.407) < ε :=
sorry

end NUMINAMATH_CALUDE_apple_sale_profit_percentage_l3005_300550


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l3005_300597

-- Define the sum function
def sum (A B : Nat) : Nat :=
  (100 * A + 10 * A + B) + (10 * B + A) + B

-- Theorem statement
theorem largest_three_digit_sum :
  ∃ (A B : Nat),
    A ≠ B ∧
    A < 10 ∧
    B < 10 ∧
    sum A B ≤ 999 ∧
    ∀ (X Y : Nat),
      X ≠ Y →
      X < 10 →
      Y < 10 →
      sum X Y ≤ 999 →
      sum X Y ≤ sum A B :=
by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l3005_300597


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3005_300587

theorem contrapositive_equivalence (a b : ℤ) :
  ((Odd a ∧ Odd b) → Even (a + b)) ↔
  (¬Even (a + b) → ¬(Odd a ∧ Odd b)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3005_300587


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3005_300589

theorem cubic_roots_sum_of_squares_reciprocal :
  ∀ a b c : ℝ,
  (a^3 - 6*a^2 + 11*a - 6 = 0) →
  (b^3 - 6*b^2 + 11*b - 6 = 0) →
  (c^3 - 6*c^2 + 11*c - 6 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3005_300589


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l3005_300585

def initial_volume : ℚ := 45
def initial_milk_ratio : ℚ := 4
def initial_water_ratio : ℚ := 1
def added_water : ℚ := 23

theorem milk_water_ratio_after_addition :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  (initial_milk : ℚ) / final_water = 9 / 8 := by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l3005_300585


namespace NUMINAMATH_CALUDE_equation_solution_l3005_300565

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 9 ∧ x = 72 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3005_300565


namespace NUMINAMATH_CALUDE_dropped_student_score_l3005_300556

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : remaining_average = 62)
  (h5 : initial_students = remaining_students + 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * remaining_average = 70 := by
  sorry

#check dropped_student_score

end NUMINAMATH_CALUDE_dropped_student_score_l3005_300556


namespace NUMINAMATH_CALUDE_solution_implies_expression_value_l3005_300512

theorem solution_implies_expression_value
  (a b : ℝ)
  (h : a * (-2) - b = 1) :
  4 * a + 2 * b + 7 = 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_implies_expression_value_l3005_300512


namespace NUMINAMATH_CALUDE_height_difference_after_growth_spurt_l3005_300551

theorem height_difference_after_growth_spurt 
  (uncle_height : ℝ) 
  (james_initial_ratio : ℝ) 
  (sarah_initial_ratio : ℝ) 
  (james_growth : ℝ) 
  (sarah_growth : ℝ) 
  (h1 : uncle_height = 72) 
  (h2 : james_initial_ratio = 2/3) 
  (h3 : sarah_initial_ratio = 3/4) 
  (h4 : james_growth = 10) 
  (h5 : sarah_growth = 12) : 
  (james_initial_ratio * uncle_height + james_growth + 
   sarah_initial_ratio * james_initial_ratio * uncle_height + sarah_growth) - uncle_height = 34 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_after_growth_spurt_l3005_300551


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3005_300523

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 8 ∧ c = 4 → -- Two sides are 8, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  a + b + c = 20 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3005_300523


namespace NUMINAMATH_CALUDE_probability_collinear_dots_l3005_300596

/-- Represents a rectangular array of dots -/
structure DotArray where
  rows : ℕ
  cols : ℕ
  total_dots : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of collinear sets of 4 dots in a vertical line -/
def vertical_collinear_sets (arr : DotArray) : ℕ := 
  arr.cols * choose arr.rows 4

/-- Main theorem: Probability of choosing 4 collinear dots -/
theorem probability_collinear_dots (arr : DotArray) 
  (h1 : arr.rows = 5) 
  (h2 : arr.cols = 4) 
  (h3 : arr.total_dots = 20) : 
  (vertical_collinear_sets arr : ℚ) / (choose arr.total_dots 4) = 4 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_collinear_dots_l3005_300596


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l3005_300581

/-- The line equation: x - y + 1 = 0 -/
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- The circle equation: (x + 1)^2 + y^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l3005_300581


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3005_300555

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3005_300555


namespace NUMINAMATH_CALUDE_johns_age_l3005_300531

theorem johns_age (j d : ℕ) 
  (h1 : j + 28 = d)
  (h2 : j + d = 76)
  (h3 : d = 2 * (j - 4)) :
  j = 24 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l3005_300531


namespace NUMINAMATH_CALUDE_share_difference_l3005_300535

/-- Represents the distribution of money among five people -/
structure MoneyDistribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ
  priya : ℕ
  elina : ℕ

/-- Theorem stating the difference in shares based on the given conditions -/
theorem share_difference (d : MoneyDistribution) :
  d.faruk = 3 * 600 ∧
  d.vasim = 5 * 600 ∧
  d.ranjith = 9 * 600 ∧
  d.priya = 7 * 600 ∧
  d.elina = 11 * 600 ∧
  d.vasim = 3000 →
  (d.faruk + d.ranjith + d.elina) - (d.vasim + d.priya) = 6600 := by
  sorry


end NUMINAMATH_CALUDE_share_difference_l3005_300535


namespace NUMINAMATH_CALUDE_point_outside_region_l3005_300584

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_outside_region :
  ¬(planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
sorry

end NUMINAMATH_CALUDE_point_outside_region_l3005_300584


namespace NUMINAMATH_CALUDE_rmb_notes_problem_l3005_300548

theorem rmb_notes_problem (x y z : ℕ) : 
  x + y + z = 33 →
  x + 5 * y + 10 * z = 187 →
  y = x - 5 →
  (x = 12 ∧ y = 7 ∧ z = 14) :=
by sorry

end NUMINAMATH_CALUDE_rmb_notes_problem_l3005_300548


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l3005_300505

theorem sqrt_fraction_equality : Real.sqrt (9/4) - Real.sqrt (4/9) + 1/6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l3005_300505


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3005_300527

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 3^n + n^2
  let r : ℕ := 4^t - t^2
  r = 2^72 - 1296 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3005_300527


namespace NUMINAMATH_CALUDE_one_third_of_twelve_x_plus_five_l3005_300525

theorem one_third_of_twelve_x_plus_five (x : ℚ) : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_twelve_x_plus_five_l3005_300525


namespace NUMINAMATH_CALUDE_equation_solution_l3005_300536

theorem equation_solution : 
  ∀ t : ℝ, t ≠ 6 ∧ t ≠ -4 →
  ((t^2 - 3*t - 18) / (t - 6) = 2 / (t + 4)) ↔ (t = -2 ∨ t = -5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3005_300536


namespace NUMINAMATH_CALUDE_erased_number_theorem_l3005_300502

theorem erased_number_theorem :
  ∀ x : ℕ, x ∈ Finset.range 21 \ {0} →
  (∃ y ∈ Finset.range 21 \ {0, x},
    19 * y = (Finset.sum (Finset.range 21 \ {0, x}) id)) ↔
  x = 1 ∨ x = 20 :=
by sorry

end NUMINAMATH_CALUDE_erased_number_theorem_l3005_300502


namespace NUMINAMATH_CALUDE_fibonacci_150_mod_9_l3005_300537

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_150_mod_9_l3005_300537


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3005_300511

/-- Given a hyperbola C and an ellipse with the following properties:
    - The general equation of C is (x²/a²) - (y²/b²) = 1 where a > 0 and b > 0
    - C has an asymptote equation y = (√5/2)x
    - C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then, the specific equation of hyperbola C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2) ∧ 
  (b / a = Real.sqrt 5 / 2) ∧
  (c^2 = 3^2) →
  a^2 = 4 ∧ b^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3005_300511


namespace NUMINAMATH_CALUDE_max_expectation_exp_l3005_300509

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define probability measure P
variable (P : Set ℝ → ℝ)

-- Define expectation E
variable (E : (ℝ → ℝ) → ℝ)

-- Define variance D
variable (D : (ℝ → ℝ) → ℝ)

-- Constants σ and b
variable (σ b : ℝ)

-- Conditions
variable (h1 : P {x | |X x| ≤ b} = 1)
variable (h2 : E X = 0)
variable (h3 : D X = σ^2)
variable (h4 : σ > 0)
variable (h5 : b > 0)

-- Theorem statement
theorem max_expectation_exp :
  (∀ Y : ℝ → ℝ, P {x | |Y x| ≤ b} = 1 → E Y = 0 → D Y = σ^2 →
    E (fun x => Real.exp (Y x)) ≤ (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) ∧
  (E (fun x => Real.exp (X x)) = (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_max_expectation_exp_l3005_300509


namespace NUMINAMATH_CALUDE_specific_triangle_angle_l3005_300515

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem specific_triangle_angle (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 2)
  (h3 : t.A = 45) :
  t.B = 67.5 := by
  sorry


end NUMINAMATH_CALUDE_specific_triangle_angle_l3005_300515


namespace NUMINAMATH_CALUDE_evaluate_expression_l3005_300516

theorem evaluate_expression (y : ℚ) (h : y = -3) :
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3005_300516


namespace NUMINAMATH_CALUDE_branch_fraction_l3005_300566

theorem branch_fraction (L : ℝ) (F : ℝ) : 
  L = 3 →  -- The branch length is 3 meters
  0 < F → F < 1 →  -- F is a proper fraction
  L - (L / 3 + F * L) = 0.6 * L →  -- Remaining length after removal
  F = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_branch_fraction_l3005_300566


namespace NUMINAMATH_CALUDE_price_difference_is_80_cents_l3005_300599

/-- Represents the price calculation methods in Lintonville Fashion Store --/
def price_calculation (original_price discount_rate tax_rate coupon : ℝ) : ℝ × ℝ := 
  let bob_total := (original_price * (1 + tax_rate) * (1 - discount_rate)) - coupon
  let alice_total := (original_price * (1 - discount_rate) - coupon) * (1 + tax_rate)
  (bob_total, alice_total)

/-- The difference between Bob's and Alice's calculations is $0.80 --/
theorem price_difference_is_80_cents 
  (h_original_price : ℝ) 
  (h_discount_rate : ℝ) 
  (h_tax_rate : ℝ) 
  (h_coupon : ℝ) 
  (h_op : h_original_price = 120)
  (h_dr : h_discount_rate = 0.15)
  (h_tr : h_tax_rate = 0.08)
  (h_c : h_coupon = 10) : 
  let (bob_total, alice_total) := price_calculation h_original_price h_discount_rate h_tax_rate h_coupon
  bob_total - alice_total = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_is_80_cents_l3005_300599


namespace NUMINAMATH_CALUDE_simplify_expression_l3005_300552

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (5 + 7 * z^2) = -2 - 12 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3005_300552


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3005_300506

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3005_300506


namespace NUMINAMATH_CALUDE_sine_inequality_holds_only_at_zero_l3005_300583

theorem sine_inequality_holds_only_at_zero (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_sine_inequality_holds_only_at_zero_l3005_300583


namespace NUMINAMATH_CALUDE_area_of_triangle_MOI_l3005_300568

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The given triangle -/
def givenTriangle : Triangle := { AB := 15, AC := 8, BC := 17 }

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circumcenter of the triangle -/
noncomputable def O : Point := sorry

/-- Incenter of the triangle -/
noncomputable def I : Point := sorry

/-- Center of the circle tangent to AC, BC, and the circumcircle -/
noncomputable def M : Point := sorry

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The main theorem -/
theorem area_of_triangle_MOI (t : Triangle) (h : t = givenTriangle) : 
  triangleArea O I M = 3.4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MOI_l3005_300568


namespace NUMINAMATH_CALUDE_water_bottles_count_l3005_300544

theorem water_bottles_count (water_bottles : ℕ) (apple_bottles : ℕ) : 
  apple_bottles = water_bottles + 6 →
  water_bottles + apple_bottles = 54 →
  water_bottles = 24 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_count_l3005_300544


namespace NUMINAMATH_CALUDE_sqrt_abs_equation_l3005_300521

theorem sqrt_abs_equation (a b : ℤ) :
  (Real.sqrt (a - 2023 : ℝ) + |b + 2023| - 1 = 0) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_equation_l3005_300521


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3005_300592

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3005_300592


namespace NUMINAMATH_CALUDE_proposition_values_l3005_300532

theorem proposition_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  ¬p ∧ (q ∨ ¬q) :=
sorry

end NUMINAMATH_CALUDE_proposition_values_l3005_300532
