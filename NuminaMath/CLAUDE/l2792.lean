import Mathlib

namespace NUMINAMATH_CALUDE_bakery_sales_projection_l2792_279255

theorem bakery_sales_projection (white_bread_ratio : ℕ) (wheat_bread_ratio : ℕ) 
  (projected_white_bread : ℕ) (expected_wheat_bread : ℕ) : 
  white_bread_ratio = 5 → 
  wheat_bread_ratio = 8 → 
  projected_white_bread = 45 →
  expected_wheat_bread = wheat_bread_ratio * projected_white_bread / white_bread_ratio →
  expected_wheat_bread = 72 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sales_projection_l2792_279255


namespace NUMINAMATH_CALUDE_apple_count_theorem_l2792_279284

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l2792_279284


namespace NUMINAMATH_CALUDE_min_value_f_l2792_279243

/-- Given positive real numbers x₁ and x₂, and a function f satisfying certain conditions,
    the value of f(x₁ + x₂) has a lower bound of 4/5. -/
theorem min_value_f (x₁ x₂ : ℝ) (f : ℝ → ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hf : ∀ x, 4^x = (1 + f x) / (1 - f x))
  (hsum : f x₁ + f x₂ = 1) :
  f (x₁ + x₂) ≥ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l2792_279243


namespace NUMINAMATH_CALUDE_line_equations_l2792_279254

-- Define a line passing through (-1, 3) with equal absolute intercepts
def line_through_point_with_equal_intercepts (a b c : ℝ) : Prop :=
  -- The line passes through (-1, 3)
  a * (-1) + b * 3 + c = 0 ∧
  -- The line has intercepts of equal absolute values on x and y axes
  ∃ k : ℝ, k ≠ 0 ∧ (a * k + c = 0 ∨ b * k + c = 0) ∧ (a * (-k) + c = 0 ∨ b * (-k) + c = 0)

-- Theorem stating the possible equations of the line
theorem line_equations :
  ∃ (a b c : ℝ),
    line_through_point_with_equal_intercepts a b c ∧
    ((a = 3 ∧ b = 1 ∧ c = 0) ∨
     (a = 1 ∧ b = -1 ∧ c = -4) ∨
     (a = 1 ∧ b = 1 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l2792_279254


namespace NUMINAMATH_CALUDE_average_height_calculation_l2792_279285

theorem average_height_calculation (north_count : ℕ) (south_count : ℕ) 
  (north_avg : ℝ) (south_avg : ℝ) :
  north_count = 300 →
  south_count = 200 →
  north_avg = 1.60 →
  south_avg = 1.50 →
  (north_count * north_avg + south_count * south_avg) / (north_count + south_count) = 1.56 := by
  sorry

end NUMINAMATH_CALUDE_average_height_calculation_l2792_279285


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2792_279205

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧ 
  given_point.y = 3 ∧
  result_line.a = 1 ∧ 
  result_line.b = -2 ∧ 
  result_line.c = 7 →
  point_on_line given_point result_line ∧ 
  parallel_lines given_line result_line

-- The proof goes here
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 1 (-2) 3) 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 7) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2792_279205


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l2792_279217

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_values_of_f :
  ∃ (min_val max_val : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min_val) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max_val) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max_val) ∧
    min_val = -3 * Real.pi / 2 ∧
    max_val = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l2792_279217


namespace NUMINAMATH_CALUDE_cheerleaders_who_quit_l2792_279221

theorem cheerleaders_who_quit 
  (initial_football_players : Nat) 
  (initial_cheerleaders : Nat)
  (football_players_quit : Nat)
  (total_left : Nat)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : football_players_quit = 10)
  (h4 : total_left = 15)
  (h5 : initial_football_players - football_players_quit + initial_cheerleaders - cheerleaders_quit = total_left)
  : cheerleaders_quit = 4 :=
by
  sorry

#check cheerleaders_who_quit

end NUMINAMATH_CALUDE_cheerleaders_who_quit_l2792_279221


namespace NUMINAMATH_CALUDE_sin_cos_graph_shift_l2792_279227

theorem sin_cos_graph_shift (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (2 * x - Real.pi / 4)
  let g : ℝ → ℝ := λ x => Real.cos (2 * x)
  ∃ (shift : ℝ), shift = 3 * Real.pi / 8 ∧
    f x = g (x - shift) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_graph_shift_l2792_279227


namespace NUMINAMATH_CALUDE_characterization_of_good_numbers_l2792_279241

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_good_numbers_l2792_279241


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2792_279245

theorem complex_fraction_equality : (2 - I) / (1 + 2*I) = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2792_279245


namespace NUMINAMATH_CALUDE_equation_solutions_l2792_279253

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1/((x - 2)*(x - 3)) + 1/((x - 3)*(x - 4)) + 1/((x - 4)*(x - 5))
  ∀ x : ℝ, f x = 1/8 ↔ x = 13 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2792_279253


namespace NUMINAMATH_CALUDE_expression_equivalence_l2792_279273

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2792_279273


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l2792_279274

theorem quadratic_roots_and_triangle (α β : ℝ) (p k : ℝ) : 
  α^2 - 10*α + 20 = 0 →
  β^2 - 10*β + 20 = 0 →
  p = α^2 + β^2 →
  k * Real.sqrt 3 = (p^2 / 36) * Real.sqrt 3 →
  p = 60 ∧ k = p^2 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_l2792_279274


namespace NUMINAMATH_CALUDE_unique_prime_in_sequence_l2792_279201

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 9031511 * 10 + B

theorem unique_prime_in_sequence :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 9031517 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_sequence_l2792_279201


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l2792_279239

theorem max_value_x_sqrt_3_minus_x_squared (x : ℝ) (h1 : 0 < x) (h2 : x < Real.sqrt 3) :
  (∀ y, 0 < y ∧ y < Real.sqrt 3 → x * Real.sqrt (3 - x^2) ≥ y * Real.sqrt (3 - y^2)) →
  x * Real.sqrt (3 - x^2) = 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_3_minus_x_squared_l2792_279239


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2792_279264

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → is_nonprime (start + i)

theorem smallest_prime_after_seven_nonprimes :
  ∃ start : ℕ, consecutive_nonprimes start ∧ 
    is_prime 97 ∧
    (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ p > start + 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2792_279264


namespace NUMINAMATH_CALUDE_michaels_pets_percentage_l2792_279228

theorem michaels_pets_percentage (total_pets : ℕ) (cat_percentage : ℚ) (num_bunnies : ℕ) :
  total_pets = 36 →
  cat_percentage = 1/2 →
  num_bunnies = 9 →
  (total_pets : ℚ) * (1 - cat_percentage) - num_bunnies = (total_pets : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_michaels_pets_percentage_l2792_279228


namespace NUMINAMATH_CALUDE_base7_product_digit_sum_l2792_279233

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list --/
def sumDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem statement --/
theorem base7_product_digit_sum :
  let a := base7ToDecimal 34
  let b := base7ToDecimal 52
  let product := a * b
  let base9Product := decimalToBase9 product
  sumDigits base9Product = 10 := by sorry

end NUMINAMATH_CALUDE_base7_product_digit_sum_l2792_279233


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2792_279236

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2) = x^2 / 2 + 1 / (2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2792_279236


namespace NUMINAMATH_CALUDE_average_cookies_l2792_279259

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies : 
  (List.sum cookie_counts) / (List.length cookie_counts) = 125 / 9 := by
  sorry

end NUMINAMATH_CALUDE_average_cookies_l2792_279259


namespace NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2792_279281

def base_seven_to_ten (n : List Nat) : Nat :=
  List.foldr (λ (digit : Nat) (acc : Nat) => 7 * acc + digit) 0 n

theorem base_seven_23456_equals_6068 :
  base_seven_to_ten [2, 3, 4, 5, 6] = 6068 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_23456_equals_6068_l2792_279281


namespace NUMINAMATH_CALUDE_min_value_theorem_l2792_279216

theorem min_value_theorem (a : ℝ) (h : a > 2) :
  a + 1 / (a - 2) ≥ 4 ∧ (a + 1 / (a - 2) = 4 ↔ a = 3) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2792_279216


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l2792_279203

theorem simplified_fraction_sum (a b : ℕ) (h : a = 49 ∧ b = 84) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l2792_279203


namespace NUMINAMATH_CALUDE_bug_meeting_point_l2792_279250

/-- Triangle PQR with side lengths PQ=6, QR=8, and PR=9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)

/-- Two bugs crawling along the perimeter of the triangle -/
structure BugMeeting (t : Triangle) :=
  (S : ℝ)  -- Position of meeting point S on side QR

/-- Main theorem: QS = 5.5 when bugs meet -/
theorem bug_meeting_point (t : Triangle) (b : BugMeeting t) :
  t.PQ = 6 → t.QR = 8 → t.PR = 9 → b.S = 5.5 := by
  sorry

#check bug_meeting_point

end NUMINAMATH_CALUDE_bug_meeting_point_l2792_279250


namespace NUMINAMATH_CALUDE_sector_arc_length_l2792_279260

theorem sector_arc_length (central_angle : Real) (radius : Real) 
  (h1 : central_angle = 1/5)
  (h2 : radius = 5) : 
  central_angle * radius = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2792_279260


namespace NUMINAMATH_CALUDE_trig_identity_simplification_l2792_279226

theorem trig_identity_simplification (θ : Real) :
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) =
  4 * (Real.sin (2 * θ)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_simplification_l2792_279226


namespace NUMINAMATH_CALUDE_complex_sum_equals_polar_form_l2792_279278

theorem complex_sum_equals_polar_form : 
  5 * Complex.exp (Complex.I * (3 * Real.pi / 7)) + 
  15 * Complex.exp (Complex.I * (23 * Real.pi / 14)) = 
  20 * Real.sqrt ((3 + Real.cos (13 * Real.pi / 14)) / 4) * 
  Complex.exp (Complex.I * (29 * Real.pi / 28)) := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_polar_form_l2792_279278


namespace NUMINAMATH_CALUDE_impossibility_of_crossing_plan_l2792_279272

/-- Represents a group of friends -/
def FriendGroup := Finset (Fin 5)

/-- The set of all possible non-empty groups of friends -/
def AllGroups : Set FriendGroup :=
  {g : FriendGroup | g.Nonempty}

/-- A crossing plan is a function that assigns each group to a number of crossings -/
def CrossingPlan := FriendGroup → ℕ

/-- A valid crossing plan assigns exactly one crossing to each non-empty group -/
def IsValidPlan (plan : CrossingPlan) : Prop :=
  ∀ g : FriendGroup, g ∈ AllGroups → plan g = 1

theorem impossibility_of_crossing_plan :
  ¬∃ (plan : CrossingPlan), IsValidPlan plan :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_crossing_plan_l2792_279272


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2792_279219

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2792_279219


namespace NUMINAMATH_CALUDE_probability_diamond_is_one_fourth_l2792_279244

/-- A special deck of cards -/
structure SpecialDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h1 : total_cards = num_ranks * num_suits)
  (h2 : cards_per_suit = num_ranks)

/-- The probability of drawing a diamond from the special deck -/
def probability_diamond (deck : SpecialDeck) : ℚ :=
  deck.cards_per_suit / deck.total_cards

/-- Theorem stating that the probability of drawing a diamond is 1/4 -/
theorem probability_diamond_is_one_fourth (deck : SpecialDeck) 
  (h3 : deck.num_suits = 4) : 
  probability_diamond deck = 1/4 := by
  sorry

#check probability_diamond_is_one_fourth

end NUMINAMATH_CALUDE_probability_diamond_is_one_fourth_l2792_279244


namespace NUMINAMATH_CALUDE_speed_ratio_in_race_l2792_279283

/-- In a race, contestant A has a head start and wins. This theorem proves the ratio of their speeds. -/
theorem speed_ratio_in_race (total_distance : ℝ) (head_start : ℝ) (win_margin : ℝ)
  (h1 : total_distance = 500)
  (h2 : head_start = 300)
  (h3 : win_margin = 100)
  : (total_distance - head_start) / (total_distance - win_margin) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_in_race_l2792_279283


namespace NUMINAMATH_CALUDE_ice_cream_cup_cost_l2792_279235

/-- Given Alok's order and payment, prove the cost of each ice-cream cup --/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 1021) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cup_cost_l2792_279235


namespace NUMINAMATH_CALUDE_count_three_painted_faces_4x4x4_l2792_279266

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (painted_faces : Fin 6 → Bool)

/-- Counts the number of subcubes with at least three painted faces -/
def count_subcubes_with_three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Theorem: In a 4x4x4 cube with all outer faces painted, 
    the number of 1x1x1 subcubes with at least three painted faces is 8 -/
theorem count_three_painted_faces_4x4x4 : 
  ∀ (cube : PaintedCube), 
  cube.size = 4 → 
  (∀ (f : Fin 6), cube.painted_faces f = true) →
  count_subcubes_with_three_painted_faces cube = 8 := by sorry

end NUMINAMATH_CALUDE_count_three_painted_faces_4x4x4_l2792_279266


namespace NUMINAMATH_CALUDE_coefficient_a3_equals_84_l2792_279286

theorem coefficient_a3_equals_84 (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (a * x - 1)^9 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 
                        a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 0) →
  a₃ = 84 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_equals_84_l2792_279286


namespace NUMINAMATH_CALUDE_right_triangle_sum_of_legs_l2792_279204

theorem right_triangle_sum_of_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 50 →           -- Length of hypotenuse
  (a * b) / 2 = 600 →  -- Area of the triangle
  a + b = 70 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_of_legs_l2792_279204


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2792_279234

theorem scientific_notation_equality : 2912000 = 2.912 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2792_279234


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2792_279231

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : ℕ), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l2792_279231


namespace NUMINAMATH_CALUDE_triangle_problem_l2792_279263

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (Real.sin t.B) / (2 * Real.sin t.A - Real.sin t.C) = 1 / (2 * Real.cos t.C))
  (h2 : t.a = 1)
  (h3 : t.b = Real.sqrt 7) :
  t.B = π / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2792_279263


namespace NUMINAMATH_CALUDE_x_between_one_third_and_two_thirds_l2792_279251

theorem x_between_one_third_and_two_thirds (x : ℝ) :
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) → (1/3 < x ∧ x < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_x_between_one_third_and_two_thirds_l2792_279251


namespace NUMINAMATH_CALUDE_smallest_integer_l2792_279288

theorem smallest_integer (x : ℕ) (m n : ℕ) (h1 : n = 36) (h2 : 0 < x) 
  (h3 : Nat.gcd m n = x + 3) (h4 : Nat.lcm m n = x * (x + 3)) :
  m ≥ 3 ∧ (∃ (x : ℕ), m = 3 ∧ 0 < x ∧ 
    Nat.gcd 3 36 = x + 3 ∧ Nat.lcm 3 36 = x * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l2792_279288


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l2792_279257

/-- Converts a binary (base 2) number to its base 4 representation -/
def binary_to_base4 (b : ℕ) : ℕ := sorry

/-- The binary representation of the number -/
def binary_num : ℕ := 11011001

/-- The base 4 representation of the number -/
def base4_num : ℕ := 3121

theorem binary_to_base4_conversion :
  binary_to_base4 binary_num = base4_num := by sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l2792_279257


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2792_279206

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k*y + 5*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 2/15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2792_279206


namespace NUMINAMATH_CALUDE_B_current_age_l2792_279211

-- Define variables for A's and B's current ages
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 6

-- Theorem statement
theorem B_current_age (h1 : condition1 A B) (h2 : condition2 A B) : B = 36 := by
  sorry

end NUMINAMATH_CALUDE_B_current_age_l2792_279211


namespace NUMINAMATH_CALUDE_budget_supplies_percentage_l2792_279246

theorem budget_supplies_percentage (transportation research_development utilities equipment salaries supplies : ℝ)
  (h1 : transportation = 15)
  (h2 : research_development = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : salaries = 234 / 360 * 100)
  (h6 : transportation + research_development + utilities + equipment + salaries + supplies = 100) :
  supplies = 2 := by
  sorry

end NUMINAMATH_CALUDE_budget_supplies_percentage_l2792_279246


namespace NUMINAMATH_CALUDE_first_number_in_ratio_l2792_279279

/-- Given two positive integers with a ratio of 3:4 and an LCM of 84, prove that the first number is 21 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 4 → 
  Nat.lcm a b = 84 → 
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_ratio_l2792_279279


namespace NUMINAMATH_CALUDE_geometric_sequence_nth_term_l2792_279268

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_nth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 2 + a 4 = 5) :
  ∃ q : ℝ, ∀ n : ℕ, a n = 2^(4 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_nth_term_l2792_279268


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2792_279292

/-- A trinomial x^2 - kx + 9 is a perfect square if and only if k = 6 or k = -6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - k*x + 9 = (a*x + b)^2) ↔ (k = 6 ∨ k = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2792_279292


namespace NUMINAMATH_CALUDE_strange_clock_time_l2792_279224

/-- Represents a hand on the strange clock -/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand -/
structure HandPosition where
  exactHourMark : Bool
  slightlyBeforeHourMark : Bool

/-- Represents the strange clock -/
structure StrangeClock where
  hands : ClockHand → HandPosition
  sameLength : Bool
  noNumbers : Bool
  unclearTop : Bool

/-- Determines if a given time matches the strange clock configuration -/
def matchesClockConfiguration (clock : StrangeClock) (hours : Nat) (minutes : Nat) : Prop :=
  hours = 16 ∧ minutes = 50 ∧
  clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false } ∧
  clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true } ∧
  clock.sameLength ∧ clock.noNumbers ∧ clock.unclearTop

theorem strange_clock_time (clock : StrangeClock) 
  (h1 : clock.hands ClockHand.A = { exactHourMark := true, slightlyBeforeHourMark := false })
  (h2 : clock.hands ClockHand.B = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h3 : clock.hands ClockHand.C = { exactHourMark := false, slightlyBeforeHourMark := true })
  (h4 : clock.sameLength)
  (h5 : clock.noNumbers)
  (h6 : clock.unclearTop) :
  ∃ (hours minutes : Nat), matchesClockConfiguration clock hours minutes :=
by
  sorry

end NUMINAMATH_CALUDE_strange_clock_time_l2792_279224


namespace NUMINAMATH_CALUDE_lyras_remaining_budget_l2792_279248

/-- Calculates the remaining budget after food purchases -/
def remaining_budget (weekly_budget : ℕ) (chicken_cost : ℕ) (beef_price_per_pound : ℕ) (beef_pounds : ℕ) : ℕ :=
  weekly_budget - (chicken_cost + beef_price_per_pound * beef_pounds)

/-- Proves that Lyra's remaining budget is $53 -/
theorem lyras_remaining_budget :
  remaining_budget 80 12 3 5 = 53 := by
  sorry

end NUMINAMATH_CALUDE_lyras_remaining_budget_l2792_279248


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l2792_279209

theorem cubic_root_equation_solution (y : ℝ) : 
  (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 2 → y = 12 ∨ y = -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l2792_279209


namespace NUMINAMATH_CALUDE_rearranged_balls_theorem_l2792_279276

/-- Represents a ball with its initial and final pile sizes -/
structure Ball where
  initialPileSize : ℕ+
  finalPileSize : ℕ+

/-- The problem statement -/
theorem rearranged_balls_theorem (n k : ℕ+) (balls : Finset Ball) 
    (h_initial_piles : (balls.sum fun b => (1 : ℚ) / b.initialPileSize) = n)
    (h_final_piles : (balls.sum fun b => (1 : ℚ) / b.finalPileSize) = n + k) :
    ∃ (subset : Finset Ball), subset.card = k + 1 ∧ 
    ∀ b ∈ subset, b.initialPileSize > b.finalPileSize :=
  sorry

end NUMINAMATH_CALUDE_rearranged_balls_theorem_l2792_279276


namespace NUMINAMATH_CALUDE_purple_balls_count_l2792_279200

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  prob_not_red_purple = 60 / 100 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + red) = 3 := by
  sorry

end NUMINAMATH_CALUDE_purple_balls_count_l2792_279200


namespace NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l2792_279225

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 84 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_6_balls_4_boxes :
  distribute_balls 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_distribute_6_balls_4_boxes_l2792_279225


namespace NUMINAMATH_CALUDE_child_share_proof_l2792_279238

theorem child_share_proof (total_amount : ℕ) (ratio_a ratio_b ratio_c : ℕ) 
  (h1 : total_amount = 2700)
  (h2 : ratio_a = 2)
  (h3 : ratio_b = 3)
  (h4 : ratio_c = 4) : 
  (total_amount * ratio_b) / (ratio_a + ratio_b + ratio_c) = 900 := by
  sorry

end NUMINAMATH_CALUDE_child_share_proof_l2792_279238


namespace NUMINAMATH_CALUDE_meat_for_35_tacos_l2792_279277

/-- The amount of meat (in pounds) needed to make a given number of tacos, 
    given that 4 pounds of meat make 10 tacos -/
def meat_needed (tacos : ℕ) : ℚ :=
  (4 : ℚ) * tacos / 10

theorem meat_for_35_tacos : meat_needed 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_35_tacos_l2792_279277


namespace NUMINAMATH_CALUDE_parabola_directrix_l2792_279215

/-- A parabola with equation y^2 = 2px and focus on the line 2x + 3y - 4 = 0 has directrix x = -2 -/
theorem parabola_directrix (p : ℝ) : 
  ∃ (f : ℝ × ℝ), 
    (∀ (x y : ℝ), y^2 = 2*p*x ↔ ((x - f.1)^2 + (y - f.2)^2 = (x + f.1)^2)) ∧ 
    (2*f.1 + 3*f.2 - 4 = 0) → 
    (f.1 = 2 ∧ f.2 = 0 ∧ ∀ (x : ℝ), x = -2 ↔ x = f.1 - p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2792_279215


namespace NUMINAMATH_CALUDE_difference_in_circumferences_l2792_279249

/-- The difference in circumferences of two concentric circular paths -/
theorem difference_in_circumferences 
  (inner_radius : ℝ) 
  (width_difference : ℝ) 
  (h1 : inner_radius = 25) 
  (h2 : width_difference = 15) : 
  2 * π * (inner_radius + width_difference) - 2 * π * inner_radius = 30 * π := by
sorry

end NUMINAMATH_CALUDE_difference_in_circumferences_l2792_279249


namespace NUMINAMATH_CALUDE_bennys_savings_l2792_279280

/-- Proves that Benny's savings in January (and February) must be $19 given the conditions -/
theorem bennys_savings (x : ℕ) : 2 * x + 8 = 46 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_bennys_savings_l2792_279280


namespace NUMINAMATH_CALUDE_system_solution_l2792_279252

theorem system_solution (x y z : ℚ) : 
  (x * y = 6 * (x + y) ∧ 
   x * z = 4 * (x + z) ∧ 
   y * z = 2 * (y + z)) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = -24 ∧ y = 24/5 ∧ z = 24/7)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2792_279252


namespace NUMINAMATH_CALUDE_equal_shaded_unshaded_probability_l2792_279214

/-- Represents a grid of squares -/
structure Grid :=
  (size : ℕ)
  (square_size : ℝ)

/-- Represents a circle -/
structure Circle :=
  (diameter : ℝ)

/-- Represents a position on the grid -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Counts the number of favorable positions -/
def count_favorable_positions (g : Grid) (c : Circle) : ℕ := sorry

/-- Counts the total number of possible positions -/
def count_total_positions (g : Grid) : ℕ := sorry

/-- Calculates the probability of placing the circle in a favorable position -/
def probability_favorable_position (g : Grid) (c : Circle) : ℚ :=
  (count_favorable_positions g c : ℚ) / (count_total_positions g : ℚ)

theorem equal_shaded_unshaded_probability 
  (g : Grid) 
  (c : Circle) 
  (h1 : g.square_size = 2)
  (h2 : c.diameter = 3)
  (h3 : g.size = 5) :
  probability_favorable_position g c = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_equal_shaded_unshaded_probability_l2792_279214


namespace NUMINAMATH_CALUDE_inequality_proof_l2792_279270

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧ 
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2792_279270


namespace NUMINAMATH_CALUDE_students_without_scholarships_l2792_279297

def total_students : ℕ := 300

def full_merit_percent : ℚ := 5 / 100
def half_merit_percent : ℚ := 10 / 100
def sports_percent : ℚ := 3 / 100
def need_based_percent : ℚ := 7 / 100

def full_merit_and_sports_percent : ℚ := 1 / 100
def half_merit_and_need_based_percent : ℚ := 2 / 100
def sports_and_need_based_percent : ℚ := 1 / 200

theorem students_without_scholarships :
  (total_students : ℚ) - 
  (((full_merit_percent + half_merit_percent + sports_percent + need_based_percent) * total_students) -
   ((full_merit_and_sports_percent + half_merit_and_need_based_percent + sports_and_need_based_percent) * total_students)) = 236 := by
  sorry

end NUMINAMATH_CALUDE_students_without_scholarships_l2792_279297


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2792_279282

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2792_279282


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2792_279265

theorem sufficient_not_necessary_condition :
  ∃ (q : ℝ → Prop), 
    (∀ x, q x → x^2 - x - 6 < 0) ∧ 
    (∃ x, x^2 - x - 6 < 0 ∧ ¬(q x)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2792_279265


namespace NUMINAMATH_CALUDE_floor_inequality_l2792_279237

theorem floor_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := by
  sorry

#check floor_inequality

end NUMINAMATH_CALUDE_floor_inequality_l2792_279237


namespace NUMINAMATH_CALUDE_race_finish_orders_l2792_279296

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def num_racers : ℕ := 3

/-- Theorem: The number of different possible orders for three distinct individuals 
    to finish a race without ties is equal to 6 -/
theorem race_finish_orders : permutations num_racers = 6 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l2792_279296


namespace NUMINAMATH_CALUDE_root_implies_a_value_l2792_279261

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 48

-- State the theorem
theorem root_implies_a_value (a b : ℚ) :
  f a b (-2 - 3 * Real.sqrt 3) = 0 → a = 44 / 23 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l2792_279261


namespace NUMINAMATH_CALUDE_simplify_expression_l2792_279213

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 + 24 = 45*w + 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2792_279213


namespace NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l2792_279218

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1/5) * total) 
  (h2 : vcr = (1/10) * total) 
  (h3 : both = (1/3) * cable) :
  (total - (cable + vcr - both)) / total = 7/10 := by
sorry

end NUMINAMATH_CALUDE_housing_units_without_cable_or_vcr_l2792_279218


namespace NUMINAMATH_CALUDE_craig_walk_distance_l2792_279262

/-- The distance Craig walked from school to David's house in miles -/
def school_to_david : ℝ := 0.2

/-- The total distance Craig walked in miles -/
def total_distance : ℝ := 0.9

/-- The distance Craig walked from David's house to his own house in miles -/
def david_to_craig : ℝ := total_distance - school_to_david

theorem craig_walk_distance : david_to_craig = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_craig_walk_distance_l2792_279262


namespace NUMINAMATH_CALUDE_area_of_rectangle_l2792_279202

-- Define the points
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (3, 0)

-- Define the length of PR
def PR_length : ℝ := 5

-- Define the property that PQR is a right triangle
def is_right_triangle (P Q R : ℝ × ℝ) (PR_length : ℝ) : Prop :=
  (Q.1 - P.1)^2 + (R.2 - Q.2)^2 = PR_length^2

-- Define the area of the rectangle
def rectangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (Q.1 - P.1) * (Q.2 - R.2)

-- The theorem to be proved
theorem area_of_rectangle : 
  is_right_triangle P Q R PR_length → rectangle_area P Q R = 12 :=
by sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l2792_279202


namespace NUMINAMATH_CALUDE_complex_subtraction_l2792_279291

theorem complex_subtraction (z₁ z₂ : ℂ) (h1 : z₁ = -2 - I) (h2 : z₂ = I) :
  z₁ - 2 * z₂ = -2 - 3 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2792_279291


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l2792_279271

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of extra topping options -/
def extra_toppings : ℕ := 3

/-- The number of topping combinations (including no extra toppings) -/
def topping_combinations : ℕ := 2^extra_toppings

/-- The total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count : total_varieties = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l2792_279271


namespace NUMINAMATH_CALUDE_equation_solution_l2792_279212

theorem equation_solution (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x ≠ 1/16) 
  (h3 : x ≠ 1/2) 
  (h4 : x ≠ 1) : 
  (Real.log 2 / Real.log (4 * Real.sqrt x)) / (Real.log 2 / Real.log (2 * x)) + 
  (Real.log 2 / Real.log (2 * x)) * (Real.log (2 * x) / Real.log (1/2)) = 0 ↔ 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2792_279212


namespace NUMINAMATH_CALUDE_duty_arrangements_eq_180_l2792_279229

/-- The number of different duty arrangements for 3 staff members over 5 days -/
def duty_arrangements (num_staff : ℕ) (num_days : ℕ) (max_days_per_staff : ℕ) : ℕ :=
  -- Number of ways to choose the person working only one day
  num_staff *
  -- Number of ways to permute the duties
  (Nat.factorial num_days / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)) *
  -- Number of ways to assign the two-day duties to the remaining two staff members
  Nat.factorial 2

/-- Theorem stating that the number of duty arrangements for the given conditions is 180 -/
theorem duty_arrangements_eq_180 :
  duty_arrangements 3 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_duty_arrangements_eq_180_l2792_279229


namespace NUMINAMATH_CALUDE_sin_inequality_l2792_279258

theorem sin_inequality (α : Real) (h : 0 < α ∧ α < π / 2) :
  Real.sin (2 * α) + 2 / Real.sin (2 * α) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l2792_279258


namespace NUMINAMATH_CALUDE_complex_plane_theorem_l2792_279290

def complex_plane_problem (z : ℂ) : Prop :=
  let x := z.re
  let y := z.im
  Complex.abs z = Real.sqrt 2 ∧
  (z^2).im = 2 ∧
  x > 0 ∧ y > 0 →
  z = Complex.mk 1 1 ∧
  let A := z
  let B := z^2
  let C := z - z^2
  let cos_ABC := ((B.re - A.re) * (C.re - B.re) + (B.im - A.im) * (C.im - B.im)) /
                 (Complex.abs (B - A) * Complex.abs (C - B))
  cos_ABC = 3 * Real.sqrt 10 / 10

theorem complex_plane_theorem :
  ∃ z : ℂ, complex_plane_problem z :=
sorry

end NUMINAMATH_CALUDE_complex_plane_theorem_l2792_279290


namespace NUMINAMATH_CALUDE_function_range_complement_l2792_279293

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 2

theorem function_range_complement :
  {k : ℝ | ∀ x, f x ≠ k} = Set.Iio (-3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_complement_l2792_279293


namespace NUMINAMATH_CALUDE_x_value_l2792_279242

theorem x_value : ∃ x : ℝ, 0.25 * x = 0.20 * 1000 - 30 ∧ x = 680 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2792_279242


namespace NUMINAMATH_CALUDE_smallest_intersection_percentage_l2792_279299

theorem smallest_intersection_percentage (S J : ℝ) : 
  S = 90 → J = 80 → 
  ∃ (I : ℝ), I ≥ 70 ∧ I ≤ S ∧ I ≤ J ∧ 
  ∀ (I' : ℝ), I' ≤ S ∧ I' ≤ J → I' ≤ I := by
  sorry

end NUMINAMATH_CALUDE_smallest_intersection_percentage_l2792_279299


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_in_open_unit_interval_l2792_279275

/-- The function f(x) = x³ - 3ax + 1 has a local minimum in the interval (0,1) -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 1 ∧ ∀ y ∈ Set.Ioo 0 1, f y ≥ f x

/-- The main theorem stating that if f(x) = x³ - 3ax + 1 has a local minimum 
    in the interval (0,1), then 0 < a < 1 -/
theorem local_minimum_implies_a_in_open_unit_interval (a : ℝ) :
  has_local_minimum (fun x => x^3 - 3*a*x + 1) a → 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_in_open_unit_interval_l2792_279275


namespace NUMINAMATH_CALUDE_complex_calculation_l2792_279267

theorem complex_calculation (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 2 + 4*I) :
  3*a - 4*b = 7 - 25*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2792_279267


namespace NUMINAMATH_CALUDE_largest_number_with_123_l2792_279294

theorem largest_number_with_123 :
  let a := 321
  let b := 21^3
  let c := 3^21
  let d := 2^31
  (c > a) ∧ (c > b) ∧ (c > d) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_123_l2792_279294


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2792_279287

theorem smallest_number_divisible (n : ℕ) : n = 92160 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 37 * 47 * 53 * k)) ∧ 
  (∃ k : ℕ, (n + 7) = 37 * 47 * 53 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2792_279287


namespace NUMINAMATH_CALUDE_complex_product_real_l2792_279220

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 2 + b * Complex.I
  (z₁ * z₂).im = 0 → b = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2792_279220


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l2792_279223

theorem units_digit_47_power_47 : (47^47) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l2792_279223


namespace NUMINAMATH_CALUDE_homework_policy_for_25_points_l2792_279208

def homework_assignments (n : ℕ) : ℕ :=
  if n ≤ 3 then 0
  else ((n - 3 - 1) / 5 + 1)

def total_assignments (total_points : ℕ) : ℕ :=
  (List.range total_points).map homework_assignments |>.sum

theorem homework_policy_for_25_points :
  total_assignments 25 = 60 := by
  sorry

end NUMINAMATH_CALUDE_homework_policy_for_25_points_l2792_279208


namespace NUMINAMATH_CALUDE_ducks_to_chickens_ratio_l2792_279269

/-- Represents the farm animals --/
structure Farm :=
  (chickens : ℕ)
  (ducks : ℕ)
  (turkeys : ℕ)

/-- The conditions of Mr. Valentino's farm --/
def valentino_farm (f : Farm) : Prop :=
  f.chickens = 200 ∧
  f.turkeys = 3 * f.ducks ∧
  f.chickens + f.ducks + f.turkeys = 1800

/-- The theorem stating the ratio of ducks to chickens --/
theorem ducks_to_chickens_ratio (f : Farm) :
  valentino_farm f → (f.ducks : ℚ) / f.chickens = 2 := by
  sorry

#check ducks_to_chickens_ratio

end NUMINAMATH_CALUDE_ducks_to_chickens_ratio_l2792_279269


namespace NUMINAMATH_CALUDE_fraction_equality_l2792_279289

theorem fraction_equality : (250 : ℚ) / ((20 + 15 * 3) - 10) = 250 / 55 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2792_279289


namespace NUMINAMATH_CALUDE_gcf_of_lcms_eq_210_l2792_279247

theorem gcf_of_lcms_eq_210 : Nat.gcd (Nat.lcm 10 21) (Nat.lcm 14 15) = 210 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_eq_210_l2792_279247


namespace NUMINAMATH_CALUDE_two_year_increase_l2792_279295

/-- 
Given an initial amount that increases by 1/8th of itself each year, 
this theorem proves that after two years, the amount will be as calculated.
-/
theorem two_year_increase (initial_amount : ℝ) : 
  initial_amount = 70400 → 
  (initial_amount * (9/8) * (9/8) : ℝ) = 89070 := by
  sorry

end NUMINAMATH_CALUDE_two_year_increase_l2792_279295


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2792_279230

/-- Given a rectangular prism with side face areas of √2, √3, and √6, its volume is √6 -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) : 
  a * b * c = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2792_279230


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2792_279232

def parabola1 (x y : ℝ) : Prop := y = 3 * x^2 - 4 * x + 2
def parabola2 (x y : ℝ) : Prop := y = x^3 - 2 * x^2 + x + 2

def intersection_points : Set (ℝ × ℝ) :=
  {(0, 2),
   ((5 + Real.sqrt 5) / 2, 3 * ((5 + Real.sqrt 5) / 2)^2 - 4 * ((5 + Real.sqrt 5) / 2) + 2),
   ((5 - Real.sqrt 5) / 2, 3 * ((5 - Real.sqrt 5) / 2)^2 - 4 * ((5 - Real.sqrt 5) / 2) + 2)}

theorem parabolas_intersection :
  ∀ x y : ℝ, (parabola1 x y ∧ parabola2 x y) ↔ (x, y) ∈ intersection_points := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2792_279232


namespace NUMINAMATH_CALUDE_contingency_fund_allocation_l2792_279222

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def local_crisis_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def local_crisis : ℚ := total_donation * local_crisis_ratio

def remaining_after_main : ℚ := total_donation - (community_pantry + local_crisis)
def livelihood : ℚ := remaining_after_main * livelihood_ratio

def contingency : ℚ := remaining_after_main - livelihood

theorem contingency_fund_allocation :
  contingency = 30 := by sorry

end NUMINAMATH_CALUDE_contingency_fund_allocation_l2792_279222


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l2792_279207

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l2792_279207


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2792_279256

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_2 * a_3 = 2 * a_1 and 5/4 is the arithmetic mean of a_4 and 2 * a_7,
    then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h3 : a 2 * a 3 = 2 * a 1)
  (h4 : (a 4 + 2 * a 7) / 2 = 5 / 4)
  : q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2792_279256


namespace NUMINAMATH_CALUDE_melted_ice_cream_depth_l2792_279240

/-- Given a sphere of radius 3 inches and a cylinder of radius 12 inches with the same volume,
    prove that the height of the cylinder is 1/4 inch. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  sphere_radius = 3 →
  cylinder_radius = 12 →
  (4 / 3) * Real.pi * sphere_radius ^ 3 = Real.pi * cylinder_radius ^ 2 * cylinder_height →
  cylinder_height = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_melted_ice_cream_depth_l2792_279240


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l2792_279298

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 157 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l2792_279298


namespace NUMINAMATH_CALUDE_units_digit_of_n_l2792_279210

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^8 → 
  m % 10 = 7 → 
  n % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l2792_279210
