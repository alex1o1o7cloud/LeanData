import Mathlib

namespace NUMINAMATH_CALUDE_original_list_size_l752_75240

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = (m + 3) * (n + 1) + 2 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_original_list_size_l752_75240


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l752_75296

theorem consecutive_odd_squares_difference (x : ℤ) : 
  Odd x → Odd (x + 2) → (x + 2)^2 - x^2 = 2000 → (x = 499 ∨ x = -501) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l752_75296


namespace NUMINAMATH_CALUDE_no_positive_integers_divisible_by_three_l752_75221

theorem no_positive_integers_divisible_by_three (n : ℕ) : 
  n > 0 ∧ 3 ∣ n → ¬(28 - 6 * n > 14) :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integers_divisible_by_three_l752_75221


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l752_75204

-- Define the two circles
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
def MovingCircle (x y r : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ x₀ y₀ ∧ C₂ x₀ y₀ ∧
  ((x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  ((x - 4)^2 + y^2 = (13 - r)^2) ∧
  ((x + 4)^2 + y^2 = (r + 3)^2)

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), (∃ (r : ℝ), MovingCircle x y r) →
  (x^2 / 64 + y^2 / 48 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l752_75204


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l752_75241

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  x^2 + y^2 = n^2 →  -- x^2 + y^2 = n^2
  x + y + n = 132 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l752_75241


namespace NUMINAMATH_CALUDE_sports_meet_participation_l752_75270

/-- The number of students participating in both track and field and ball games -/
def students_in_track_and_ball (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ)
  (swimming_and_track : ℕ) (swimming_and_ball : ℕ) : ℕ :=
  swimming + track + ball - swimming_and_track - swimming_and_ball - total

theorem sports_meet_participation :
  students_in_track_and_ball 28 15 8 14 3 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sports_meet_participation_l752_75270


namespace NUMINAMATH_CALUDE_mrs_hilt_remaining_money_l752_75260

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℚ) (pencil notebook pens : ℚ) : ℚ :=
  initial - (pencil + notebook + pens)

/-- Proves that Mrs. Hilt's remaining money is $3.00 -/
theorem mrs_hilt_remaining_money :
  remaining_money 12.5 1.25 3.45 4.8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_remaining_money_l752_75260


namespace NUMINAMATH_CALUDE_polygon_diagonals_theorem_l752_75252

theorem polygon_diagonals_theorem (n : ℕ) :
  n ≥ 3 →
  (n - 2 = 6) →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_theorem_l752_75252


namespace NUMINAMATH_CALUDE_cat_whiskers_problem_l752_75298

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  whisper : ℕ
  bella : ℕ
  max : ℕ
  felix : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem cat_whiskers_problem (c : CatWhiskers) : 
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 ∧
  c.whisper = 2 * c.puffy ∧
  c.whisper = c.scruffy / 3 ∧
  c.bella = c.juniper + c.puffy - 4 ∧
  c.max = c.scruffy + c.buffy ∧
  c.felix = min c.juniper (min c.puffy (min c.scruffy (min c.buffy (min c.whisper (min c.bella c.max)))))
  →
  c.max = 112 := by
sorry

end NUMINAMATH_CALUDE_cat_whiskers_problem_l752_75298


namespace NUMINAMATH_CALUDE_peters_birdseed_calculation_l752_75227

/-- The amount of birdseed needed for a week given the number of birds and their daily consumption --/
def birdseed_needed (parakeet_count : ℕ) (parrot_count : ℕ) (finch_count : ℕ) 
  (parakeet_consumption : ℕ) (parrot_consumption : ℕ) (days : ℕ) : ℕ :=
  let finch_consumption := parakeet_consumption / 2
  let daily_total := parakeet_count * parakeet_consumption + 
                     parrot_count * parrot_consumption + 
                     finch_count * finch_consumption
  daily_total * days

/-- Theorem stating that Peter needs to buy 266 grams of birdseed for a week --/
theorem peters_birdseed_calculation :
  birdseed_needed 3 2 4 2 14 7 = 266 := by
  sorry


end NUMINAMATH_CALUDE_peters_birdseed_calculation_l752_75227


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l752_75285

/-- Given a triangle ABC with sides a, b, c, inradius r, circumradius R, and excircle radii rA, rB, rC,
    prove the following equation. -/
theorem triangle_radii_relation (a b c r R rA rB rC : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ rA > 0 ∧ rB > 0 ∧ rC > 0) :
  a^2 * (2/rA - r/(rB*rC)) + b^2 * (2/rB - r/(rA*rC)) + c^2 * (2/rC - r/(rA*rB)) = 4*(R + 3*r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_relation_l752_75285


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l752_75220

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l752_75220


namespace NUMINAMATH_CALUDE_problem_statement_l752_75253

theorem problem_statement (x : ℝ) (h : x = -1) : 
  2 * (-x^2 + 3*x^3) - (2*x^3 - 2*x^2) + 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l752_75253


namespace NUMINAMATH_CALUDE_rational_identity_product_l752_75299

theorem rational_identity_product (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 55) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = 200 := by
sorry

end NUMINAMATH_CALUDE_rational_identity_product_l752_75299


namespace NUMINAMATH_CALUDE_percent_cat_owners_l752_75264

def total_students : ℕ := 500
def cat_owners : ℕ := 90

theorem percent_cat_owners : (cat_owners : ℚ) / total_students * 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_percent_cat_owners_l752_75264


namespace NUMINAMATH_CALUDE_probability_white_balls_l752_75228

def total_balls : ℕ := 16
def white_balls : ℕ := 8
def black_balls : ℕ := 8
def drawn_balls : ℕ := 2

theorem probability_white_balls (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : total_balls = 16)
  (h3 : white_balls = 8)
  (h4 : black_balls = 8)
  (h5 : drawn_balls = 2) :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 30 ∧
  1 - (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 23 / 30 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_balls_l752_75228


namespace NUMINAMATH_CALUDE_complex_modulus_one_l752_75258

theorem complex_modulus_one (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a ∧ a < (n + 1) / (n - 1 : ℝ)) 
  (h3 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l752_75258


namespace NUMINAMATH_CALUDE_angelina_speed_l752_75286

/-- Proves that Angelina's speed from the grocery to the gym is 3 meters per second --/
theorem angelina_speed (home_to_grocery : ℝ) (grocery_to_gym : ℝ) (v : ℝ) :
  home_to_grocery = 180 →
  grocery_to_gym = 240 →
  (home_to_grocery / v) - (grocery_to_gym / (2 * v)) = 40 →
  2 * v = 3 := by
  sorry

end NUMINAMATH_CALUDE_angelina_speed_l752_75286


namespace NUMINAMATH_CALUDE_yellow_then_not_yellow_probability_l752_75259

/-- A deck of cards with 5 suits and 13 ranks. -/
structure Deck :=
  (cards : Finset (Fin 5 × Fin 13))
  (card_count : cards.card = 65)
  (suit_rank_unique : ∀ (s : Fin 5) (r : Fin 13), (s, r) ∈ cards)

/-- The probability of drawing a yellow card followed by a non-yellow card from a shuffled deck. -/
def yellow_then_not_yellow_prob (d : Deck) : ℚ :=
  169 / 1040

/-- Theorem stating the probability of drawing a yellow card followed by a non-yellow card. -/
theorem yellow_then_not_yellow_probability (d : Deck) :
  yellow_then_not_yellow_prob d = 169 / 1040 := by
  sorry

end NUMINAMATH_CALUDE_yellow_then_not_yellow_probability_l752_75259


namespace NUMINAMATH_CALUDE_tangent_line_minimum_two_roots_inequality_l752_75278

noncomputable section

variables (m : ℝ) (x x₁ x₂ : ℝ) (a b n : ℝ)

def f (x : ℝ) : ℝ := Real.log x - m * x

theorem tangent_line_minimum (h : f e x = a * x + b) :
  ∃ (x₀ : ℝ), a + 2 * b = 1 / x₀ + 2 * Real.log x₀ - e - 2 ∧ 
  ∀ (x : ℝ), 1 / x + 2 * Real.log x - e - 2 ≥ 1 / x₀ + 2 * Real.log x₀ - e - 2 :=
sorry

theorem two_roots_inequality (h1 : f m x₁ = (2 - m) * x₁ + n) 
                             (h2 : f m x₂ = (2 - m) * x₂ + n) 
                             (h3 : x₁ < x₂) :
  2 * x₁ + x₂ > e / 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_two_roots_inequality_l752_75278


namespace NUMINAMATH_CALUDE_triangle_area_circumradius_angles_l752_75217

theorem triangle_area_circumradius_angles 
  (α β γ : Real) (R : Real) (S_Δ : Real) :
  (α + β + γ = π) →
  (R > 0) →
  (S_Δ > 0) →
  (S_Δ = 2 * R^2 * Real.sin α * Real.sin β * Real.sin γ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_circumradius_angles_l752_75217


namespace NUMINAMATH_CALUDE_equivalent_ratios_l752_75211

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = 3 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_ratios_l752_75211


namespace NUMINAMATH_CALUDE_fraction_sum_bounds_l752_75244

theorem fraction_sum_bounds (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_log_sum : Real.log a + Real.log b + Real.log c = 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 := by
  sorry


end NUMINAMATH_CALUDE_fraction_sum_bounds_l752_75244


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l752_75248

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon : 
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l752_75248


namespace NUMINAMATH_CALUDE_crayon_difference_l752_75271

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (hw : willy_crayons = 5092) 
  (hl : lucy_crayons = 3971) : 
  willy_crayons - lucy_crayons = 1121 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l752_75271


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l752_75273

theorem sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) 
  (h : ∀ x, 3 * x^2 - 2 * p * x + q = 0 ↔ x = a ∨ x = b) :
  a^2 + b^2 = 4 * p^2 - 6 * q := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l752_75273


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l752_75297

theorem arithmetic_progression_problem (a d : ℝ) : 
  2 * (a - d) * a * (a + d + 7) = 1000 ∧ 
  a^2 = 2 * (a - d) * (a + d + 7) →
  d = 8 ∨ d = -8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l752_75297


namespace NUMINAMATH_CALUDE_optimal_production_plan_l752_75215

/-- Represents a production plan with quantities of products A and B. -/
structure ProductionPlan where
  productA : ℕ
  productB : ℕ

/-- Represents the available raw materials and profit data. -/
structure FactoryData where
  rawMaterialA : ℝ
  rawMaterialB : ℝ
  totalProducts : ℕ
  materialAForProductA : ℝ
  materialBForProductA : ℝ
  materialAForProductB : ℝ
  materialBForProductB : ℝ
  profitProductA : ℕ
  profitProductB : ℕ

/-- Checks if a production plan is valid given the factory data. -/
def isValidPlan (plan : ProductionPlan) (data : FactoryData) : Prop :=
  plan.productA + plan.productB = data.totalProducts ∧
  plan.productA * data.materialAForProductA + plan.productB * data.materialAForProductB ≤ data.rawMaterialA ∧
  plan.productA * data.materialBForProductA + plan.productB * data.materialBForProductB ≤ data.rawMaterialB

/-- Calculates the profit for a given production plan. -/
def calculateProfit (plan : ProductionPlan) (data : FactoryData) : ℕ :=
  plan.productA * data.profitProductA + plan.productB * data.profitProductB

/-- The main theorem to prove. -/
theorem optimal_production_plan (data : FactoryData)
  (h_data : data.rawMaterialA = 66 ∧ data.rawMaterialB = 66.4 ∧ data.totalProducts = 90 ∧
            data.materialAForProductA = 0.5 ∧ data.materialBForProductA = 0.8 ∧
            data.materialAForProductB = 1.2 ∧ data.materialBForProductB = 0.6 ∧
            data.profitProductA = 30 ∧ data.profitProductB = 20) :
  ∃ (optimalPlan : ProductionPlan),
    isValidPlan optimalPlan data ∧
    calculateProfit optimalPlan data = 2420 ∧
    ∀ (plan : ProductionPlan), isValidPlan plan data → calculateProfit plan data ≤ 2420 :=
  sorry

end NUMINAMATH_CALUDE_optimal_production_plan_l752_75215


namespace NUMINAMATH_CALUDE_gcd_nine_digit_repeats_l752_75265

/-- The set of all nine-digit integers formed by repeating a three-digit integer three times -/
def NineDigitRepeats : Set ℕ :=
  {n | ∃ k : ℕ, 100 ≤ k ∧ k ≤ 999 ∧ n = 1001001 * k}

/-- The greatest common divisor of all numbers in NineDigitRepeats is 1001001 -/
theorem gcd_nine_digit_repeats :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ NineDigitRepeats, d ∣ n) ∧
  (∀ m : ℕ, m > 0 → (∀ n ∈ NineDigitRepeats, m ∣ n) → m ≤ d) ∧
  d = 1001001 := by
  sorry


end NUMINAMATH_CALUDE_gcd_nine_digit_repeats_l752_75265


namespace NUMINAMATH_CALUDE_parallelogram_area_l752_75229

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (5, 12), and (8, 12) is 36 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (5, 12)
  let v4 : ℝ × ℝ := (8, 12)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l752_75229


namespace NUMINAMATH_CALUDE_smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l752_75238

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_n_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(factorial n % 2016 = 0) ∧ (factorial 8 % 2016 = 0) := by sorry

theorem smallest_n_factorial_divisible_by_2016_pow_10 :
  ∀ n : ℕ, n < 63 → ¬(factorial n % (2016^10) = 0) ∧ (factorial 63 % (2016^10) = 0) := by sorry

end NUMINAMATH_CALUDE_smallest_n_factorial_divisible_by_2016_smallest_n_factorial_divisible_by_2016_pow_10_l752_75238


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l752_75232

def f (x : ℕ) : ℕ := 3 * x + 2

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l752_75232


namespace NUMINAMATH_CALUDE_problem_statement_l752_75233

theorem problem_statement (x y : ℝ) (h1 : x = 3) (h2 : y = 1) :
  let n := x - y^(2*(x+y))
  n = 2 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l752_75233


namespace NUMINAMATH_CALUDE_square_difference_emily_calculation_l752_75284

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem emily_calculation : 39^2 = 40^2 - 79 := by sorry

end NUMINAMATH_CALUDE_square_difference_emily_calculation_l752_75284


namespace NUMINAMATH_CALUDE_car_speed_problem_l752_75222

/-- A car travels uphill and downhill. This theorem proves the downhill speed given certain conditions. -/
theorem car_speed_problem (uphill_speed : ℝ) (total_distance : ℝ) (total_time uphill_time downhill_time : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : total_distance = 650)
  (h3 : total_time = 15)
  (h4 : uphill_time = 5)
  (h5 : downhill_time = 5) :
  ∃ downhill_speed : ℝ, 
    downhill_speed * downhill_time + uphill_speed * uphill_time = total_distance ∧ 
    downhill_speed = 100 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l752_75222


namespace NUMINAMATH_CALUDE_average_of_numbers_l752_75230

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l752_75230


namespace NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l752_75251

/-- The number of coins Lucy flips -/
def n : ℕ := 12

/-- The probability of getting fewer heads than tails when flipping n coins -/
def prob_fewer_heads (n : ℕ) : ℚ :=
  793 / 2048

theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n = 793 / 2048 := by sorry

end NUMINAMATH_CALUDE_prob_fewer_heads_12_coins_l752_75251


namespace NUMINAMATH_CALUDE_constant_sum_of_squares_l752_75214

/-- Defines an ellipse C with equation x²/4 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Defines a point P on the major axis of C -/
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Defines the direction vector of line l -/
def direction_vector : ℝ × ℝ := (2, 1)

/-- Defines the line l passing through P with the given direction vector -/
def line_l (m t : ℝ) : ℝ × ℝ := (m + 2*t, t)

/-- Defines the intersection points of line l and ellipse C -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l m t ∧ ellipse_C p.1 p.2}

/-- States that |PA|² + |PB|² is constant for all valid m -/
theorem constant_sum_of_squares (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) :
  ∃ A B, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
    (A.1 - m)^2 + A.2^2 + (B.1 - m)^2 + B.2^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_constant_sum_of_squares_l752_75214


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_and_integer_intersection_l752_75263

/-- Represents a quadratic function y = mx^2 - (m+2)x + 2 --/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

theorem parabola_intersects_x_axis_twice_and_integer_intersection (m : ℝ) 
  (hm_nonzero : m ≠ 0) (hm_not_two : m ≠ 2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_function m x1 = 0 ∧ quadratic_function m x2 = 0) ∧
  (∃! m : ℕ+, m ≠ 2 ∧ ∃ x1 x2 : ℤ, quadratic_function ↑m ↑x1 = 0 ∧ quadratic_function ↑m ↑x2 = 0 ∧ x1 ≠ x2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_and_integer_intersection_l752_75263


namespace NUMINAMATH_CALUDE_number_problem_l752_75295

theorem number_problem (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l752_75295


namespace NUMINAMATH_CALUDE_bianca_next_day_miles_l752_75242

/-- The number of miles Bianca ran on the first day -/
def first_day_miles : ℕ := 8

/-- The total number of miles Bianca ran over two days -/
def total_miles : ℕ := 12

/-- The number of miles Bianca ran on the next day -/
def next_day_miles : ℕ := total_miles - first_day_miles

theorem bianca_next_day_miles :
  next_day_miles = 4 :=
by sorry

end NUMINAMATH_CALUDE_bianca_next_day_miles_l752_75242


namespace NUMINAMATH_CALUDE_chord_intersection_tangent_circle_l752_75291

/-- Given a point A and a circle S with center O and radius R, 
    prove that the line through A intersecting S in a chord PQ of length d 
    is tangent to a circle with center O and radius sqrt(R^2 - d^2/4) -/
theorem chord_intersection_tangent_circle 
  (A O : ℝ × ℝ) (R d : ℝ) (S : Set (ℝ × ℝ)) :
  let circle_S := {p : ℝ × ℝ | dist p O = R}
  let chord_length (l : Set (ℝ × ℝ)) := ∃ P Q : ℝ × ℝ, P ∈ l ∩ S ∧ Q ∈ l ∩ S ∧ dist P Q = d
  let tangent_circle := {p : ℝ × ℝ | dist p O = Real.sqrt (R^2 - d^2/4)}
  ∀ l : Set (ℝ × ℝ), A ∈ l → S = circle_S → chord_length l → 
    ∃ p : ℝ × ℝ, p ∈ l ∩ tangent_circle :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_tangent_circle_l752_75291


namespace NUMINAMATH_CALUDE_function_inequality_l752_75203

-- Define the function f
variable {f : ℝ → ℝ}

-- State the theorem
theorem function_inequality
  (h : ∀ x y : ℝ, f y - f x ≤ (y - x)^2)
  (n : ℕ)
  (hn : n > 0)
  (a b : ℝ) :
  |f b - f a| ≤ (1 / n : ℝ) * (b - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l752_75203


namespace NUMINAMATH_CALUDE_house_transaction_loss_l752_75239

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

theorem house_transaction_loss :
  let first_sale := initial_value * (1 - loss_percentage)
  let second_sale := first_sale * (1 + gain_percentage)
  second_sale - initial_value = 240 := by sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l752_75239


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1010th_term_l752_75269

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (p r : ℚ) : ℕ → ℚ
| 0 => p / 2
| 1 => 18
| 2 => 2 * p - r
| 3 => 2 * p + r
| n + 4 => arithmetic_sequence p r 3 + (n + 1) * (arithmetic_sequence p r 3 - arithmetic_sequence p r 2)

/-- The 1010th term of the sequence is 72774/11 -/
theorem arithmetic_sequence_1010th_term (p r : ℚ) :
  arithmetic_sequence p r 1009 = 72774 / 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1010th_term_l752_75269


namespace NUMINAMATH_CALUDE_f_is_even_l752_75250

def f (x : ℝ) : ℝ := -3 * x^4

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l752_75250


namespace NUMINAMATH_CALUDE_complex_sum_argument_l752_75205

theorem complex_sum_argument : 
  let z : ℂ := Complex.exp (7 * π * I / 60) + Complex.exp (17 * π * I / 60) + 
                Complex.exp (27 * π * I / 60) + Complex.exp (37 * π * I / 60) + 
                Complex.exp (47 * π * I / 60)
  Complex.arg z = 9 * π / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_argument_l752_75205


namespace NUMINAMATH_CALUDE_expression_result_l752_75293

theorem expression_result : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l752_75293


namespace NUMINAMATH_CALUDE_identity_proof_l752_75201

theorem identity_proof (a b c d x y z u : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2
  = (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l752_75201


namespace NUMINAMATH_CALUDE_vector_dot_product_equation_l752_75213

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equation (x : ℝ) :
  dot_product a (b x) = 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equation_l752_75213


namespace NUMINAMATH_CALUDE_seven_layer_tower_lights_l752_75245

/-- Represents a tower with lights -/
structure LightTower where
  layers : ℕ
  top_lights : ℕ
  total_lights : ℕ

/-- The sum of a geometric sequence -/
def geometricSum (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * (r^n - 1) / (r - 1)

/-- The theorem statement -/
theorem seven_layer_tower_lights (tower : LightTower) :
  tower.layers = 7 ∧
  tower.total_lights = 381 ∧
  (∀ i : ℕ, i < 7 → geometricSum tower.top_lights 2 (i + 1) ≤ tower.total_lights) →
  tower.top_lights = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_layer_tower_lights_l752_75245


namespace NUMINAMATH_CALUDE_cubic_root_sum_cube_l752_75280

theorem cubic_root_sum_cube (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cube_l752_75280


namespace NUMINAMATH_CALUDE_april_coffee_expenditure_l752_75292

/-- Calculates the total expenditure on coffee for a month given the number of coffees per day, 
    cost per coffee, and number of days in the month. -/
def coffee_expenditure (coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_month : ℕ) : ℕ :=
  coffees_per_day * cost_per_coffee * days_in_month

theorem april_coffee_expenditure :
  coffee_expenditure 2 2 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_expenditure_l752_75292


namespace NUMINAMATH_CALUDE_magnitude_v_l752_75289

theorem magnitude_v (u v : ℂ) (h1 : u * v = 24 - 10 * I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 26 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_v_l752_75289


namespace NUMINAMATH_CALUDE_line_intersects_circle_l752_75257

/-- The line y = x + 1 intersects the circle x^2 + y^2 = 1 -/
theorem line_intersects_circle :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let circle : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = 1
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 1
  let distance_to_line : ℝ := |1| / Real.sqrt 2
  distance_to_line < radius →
  ∃ p : ℝ × ℝ, line p.1 = p.2 ∧ circle p :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l752_75257


namespace NUMINAMATH_CALUDE_acid_concentration_problem_l752_75209

theorem acid_concentration_problem (acid1 acid2 acid3 : ℝ) (water : ℝ) :
  acid1 = 10 →
  acid2 = 20 →
  acid3 = 30 →
  acid1 / (acid1 + (water * (1/20))) = 1/20 →
  acid2 / (acid2 + (water * (13/30))) = 7/30 →
  acid3 / (acid3 + water) = 21/200 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_problem_l752_75209


namespace NUMINAMATH_CALUDE_at_operation_four_three_l752_75243

def at_operation (a b : ℝ) : ℝ := 4 * a^2 - 2 * b

theorem at_operation_four_three : at_operation 4 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_four_three_l752_75243


namespace NUMINAMATH_CALUDE_strictly_increasing_function_l752_75261

theorem strictly_increasing_function
  (a b c d : ℝ)
  (h1 : a > c)
  (h2 : c > d)
  (h3 : d > b)
  (h4 : b > 1)
  (h5 : a * b > c * d) :
  let f : ℝ → ℝ := λ x ↦ a^x + b^x - c^x - d^x
  ∀ x ≥ 0, (deriv f) x > 0 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_l752_75261


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l752_75202

/-- The number of squares of a given size in a 6x6 grid -/
def squares_of_size (n : Nat) : Nat :=
  (7 - n) * (7 - n)

/-- The total number of squares in a 6x6 grid -/
def total_squares : Nat :=
  (squares_of_size 1) + (squares_of_size 2) + (squares_of_size 3) + 
  (squares_of_size 4) + (squares_of_size 5)

/-- Theorem: The total number of squares in a 6x6 grid is 55 -/
theorem six_by_six_grid_squares : total_squares = 55 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l752_75202


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l752_75236

theorem or_necessary_not_sufficient_for_and (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l752_75236


namespace NUMINAMATH_CALUDE_symmetry_lines_sum_l752_75275

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(9 - a)^2 + b = 0
  h4 : (9 - c)^2 + d = 0

/-- The sum of x-axis symmetry lines of two intersecting parabolas equals 12 -/
theorem symmetry_lines_sum (p : IntersectingParabolas) : p.a + p.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_lines_sum_l752_75275


namespace NUMINAMATH_CALUDE_three_number_sum_l752_75226

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 8)
  (h4 : (a + b + c) / 3 = c - 18)
  (h5 : b = 12) :
  a + b + c = 66 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l752_75226


namespace NUMINAMATH_CALUDE_c_minus_three_equals_negative_two_l752_75218

/-- An invertible function g : ℝ → ℝ -/
def g : ℝ → ℝ :=
  sorry

/-- c is a real number such that g(c) = 3 and g(3) = c -/
def c : ℝ :=
  sorry

theorem c_minus_three_equals_negative_two (h1 : Function.Injective g) (h2 : g c = 3) (h3 : g 3 = c) :
  c - 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_c_minus_three_equals_negative_two_l752_75218


namespace NUMINAMATH_CALUDE_seating_arrangements_3_8_l752_75294

/-- The number of distinct seating arrangements for 3 people in a row of 8 seats,
    with empty seats on both sides of each person. -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of seating arrangements
    for 3 people in 8 seats is 24. -/
theorem seating_arrangements_3_8 :
  seating_arrangements 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_3_8_l752_75294


namespace NUMINAMATH_CALUDE_f_minus_one_eq_neg_two_l752_75237

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_one_eq_neg_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = 2^x) :
  f (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_f_minus_one_eq_neg_two_l752_75237


namespace NUMINAMATH_CALUDE_line_erased_length_l752_75254

theorem line_erased_length (initial_length : ℝ) (final_length : ℝ) (erased_length : ℝ) : 
  initial_length = 1 →
  final_length = 0.67 →
  erased_length = initial_length * 100 - final_length * 100 →
  erased_length = 33 := by
sorry

end NUMINAMATH_CALUDE_line_erased_length_l752_75254


namespace NUMINAMATH_CALUDE_okinawa_sales_ratio_l752_75208

/-- Proves the ratio of Okinawa-flavored milk tea sales to total sales -/
theorem okinawa_sales_ratio (total_sales : ℕ) (winter_melon_sales : ℕ) (chocolate_sales : ℕ) 
  (h1 : total_sales = 50)
  (h2 : winter_melon_sales = 2 * total_sales / 5)
  (h3 : chocolate_sales = 15)
  (h4 : winter_melon_sales + chocolate_sales + (total_sales - winter_melon_sales - chocolate_sales) = total_sales) :
  (total_sales - winter_melon_sales - chocolate_sales) / total_sales = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_okinawa_sales_ratio_l752_75208


namespace NUMINAMATH_CALUDE_ellipse_foci_l752_75274

theorem ellipse_foci (x y : ℝ) :
  (x^2 / 25 + y^2 / 169 = 1) →
  (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁ = (0, 12) ∧ f₂ = (0, -12)) ∧
    (∀ p : ℝ × ℝ, p.1^2 / 25 + p.2^2 / 169 = 1 →
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
       2 * Real.sqrt 169))) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_l752_75274


namespace NUMINAMATH_CALUDE_mushroom_collection_l752_75262

theorem mushroom_collection : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    (n / 100 + (n / 10) % 10 + n % 10 = 14) ∧  -- sum of digits is 14
    n % 50 = 0 ∧  -- divisible by 50
    n = 950 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l752_75262


namespace NUMINAMATH_CALUDE_least_N_for_P_condition_l752_75277

/-- The probability that at least 3/5 of N green balls are on the same side of a red ball
    when arranged randomly in a line -/
def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + 1 + (N - ⌈(3 * N : ℚ) / 5⌉ + 1)) / (N + 1)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N % 5 = 0 → N > 0 →
    (∀ k : ℕ, k % 5 = 0 → k > 0 → k < N → P k ≥ 321/400) ∧
    P N < 321/400 →
    N = 480 :=
sorry

end NUMINAMATH_CALUDE_least_N_for_P_condition_l752_75277


namespace NUMINAMATH_CALUDE_max_donated_cookies_l752_75287

def distribute_cookies (total : Nat) (employees : Nat) : Nat :=
  total - (employees * (total / employees))

theorem max_donated_cookies :
  distribute_cookies 120 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_donated_cookies_l752_75287


namespace NUMINAMATH_CALUDE_kangaroo_population_change_l752_75283

theorem kangaroo_population_change 
  (G : ℝ) -- Initial number of grey kangaroos
  (R : ℝ) -- Initial number of red kangaroos
  (h1 : G > 0) -- Assumption: initial grey kangaroo population is positive
  (h2 : R > 0) -- Assumption: initial red kangaroo population is positive
  (h3 : 1.28 * G / (0.72 * R) = R / G) -- Ratio reversal condition
  : (2.24 * G) / ((7/3) * G) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_population_change_l752_75283


namespace NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l752_75235

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l752_75235


namespace NUMINAMATH_CALUDE_jackie_apple_count_l752_75231

/-- Given that Adam has 9 apples and 3 more apples than Jackie, prove that Jackie has 6 apples. -/
theorem jackie_apple_count (adam_apple_count : ℕ) (adam_extra_apples : ℕ) (jackie_apple_count : ℕ)
  (h1 : adam_apple_count = 9)
  (h2 : adam_apple_count = jackie_apple_count + adam_extra_apples)
  (h3 : adam_extra_apples = 3) :
  jackie_apple_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_jackie_apple_count_l752_75231


namespace NUMINAMATH_CALUDE_trigonometric_problem_l752_75267

theorem trigonometric_problem (x : ℝ) (h : 3 * Real.sin (x/2) - Real.cos (x/2) = 0) :
  Real.tan x = 3/4 ∧ (Real.cos (2*x)) / (Real.sqrt 2 * Real.cos (π/4 + x) * Real.sin x) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l752_75267


namespace NUMINAMATH_CALUDE_simplest_form_iff_coprime_l752_75223

/-- A fraction is a pair of integers where the denominator is non-zero -/
structure Fraction where
  numerator : Int
  denominator : Int
  denom_nonzero : denominator ≠ 0

/-- A fraction is in its simplest form if it cannot be reduced further -/
def is_simplest_form (f : Fraction) : Prop :=
  ∀ k : Int, k ≠ 0 → ¬(k ∣ f.numerator ∧ k ∣ f.denominator)

/-- Two integers are coprime if their greatest common divisor is 1 -/
def are_coprime (a b : Int) : Prop :=
  Int.gcd a b = 1

/-- Theorem: A fraction is in its simplest form if and only if its numerator and denominator are coprime -/
theorem simplest_form_iff_coprime (f : Fraction) :
  is_simplest_form f ↔ are_coprime f.numerator f.denominator := by
  sorry

end NUMINAMATH_CALUDE_simplest_form_iff_coprime_l752_75223


namespace NUMINAMATH_CALUDE_tony_squat_weight_l752_75282

/-- Represents Tony's weight lifting capabilities -/
structure WeightLifter where
  curl_weight : ℕ
  military_press_multiplier : ℕ
  squat_multiplier : ℕ

/-- Calculates the weight Tony can lift in the squat exercise -/
def squat_weight (lifter : WeightLifter) : ℕ :=
  lifter.curl_weight * lifter.military_press_multiplier * lifter.squat_multiplier

/-- Theorem: Tony can lift 900 pounds in the squat exercise -/
theorem tony_squat_weight :
  ∃ (tony : WeightLifter),
    tony.curl_weight = 90 ∧
    tony.military_press_multiplier = 2 ∧
    tony.squat_multiplier = 5 ∧
    squat_weight tony = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_squat_weight_l752_75282


namespace NUMINAMATH_CALUDE_log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l752_75207

-- Define the arithmetic mean of logarithms
def log_arithmetic_mean (x y z : ℝ) : Prop :=
  Real.log y = (Real.log x + Real.log z) / 2

-- Define the geometric mean
def geometric_mean (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem log_arithmetic_mean_implies_geometric_mean
  (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  log_arithmetic_mean x y z → geometric_mean x y z :=
sorry

theorem geometric_mean_not_implies_log_arithmetic_mean :
  ∃ x y z : ℝ, geometric_mean x y z ∧ ¬log_arithmetic_mean x y z :=
sorry

end NUMINAMATH_CALUDE_log_arithmetic_mean_implies_geometric_mean_geometric_mean_not_implies_log_arithmetic_mean_l752_75207


namespace NUMINAMATH_CALUDE_first_three_digits_of_expression_l752_75256

-- Define the expression
def expression : ℝ := (10^2003 + 1)^(11/9)

-- Define a function to get the first three digits after the decimal point
def firstThreeDecimalDigits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- Theorem statement
theorem first_three_digits_of_expression :
  firstThreeDecimalDigits expression = (2, 2, 2) := by sorry

end NUMINAMATH_CALUDE_first_three_digits_of_expression_l752_75256


namespace NUMINAMATH_CALUDE_small_cylinder_radius_l752_75290

/-- Proves that the radius of smaller cylinders is √(24/5) meters given the specified conditions -/
theorem small_cylinder_radius 
  (large_diameter : ℝ) 
  (large_height : ℝ) 
  (small_height : ℝ) 
  (num_small_cylinders : ℕ) 
  (h_large_diameter : large_diameter = 6)
  (h_large_height : large_height = 8)
  (h_small_height : small_height = 5)
  (h_num_small_cylinders : num_small_cylinders = 3)
  : ∃ (small_radius : ℝ), small_radius = Real.sqrt (24 / 5) := by
  sorry

#check small_cylinder_radius

end NUMINAMATH_CALUDE_small_cylinder_radius_l752_75290


namespace NUMINAMATH_CALUDE_triangular_pyramid_base_balls_l752_75249

/-- The number of balls in a triangular pyramid with n rows -/
def triangular_pyramid_balls (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of balls in the base of a triangular pyramid with n rows -/
def base_balls (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with 165 tightly packed identical balls,
    the number of balls in the base is 45 -/
theorem triangular_pyramid_base_balls :
  ∃ n : ℕ, triangular_pyramid_balls n = 165 ∧ base_balls n = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_base_balls_l752_75249


namespace NUMINAMATH_CALUDE_max_candy_leftover_l752_75276

theorem max_candy_leftover (y : ℕ) : ∃ (q r : ℕ), y = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l752_75276


namespace NUMINAMATH_CALUDE_committee_formation_count_l752_75200

theorem committee_formation_count (n : ℕ) (k : ℕ) : n = 30 → k = 5 →
  (n.choose 1) * ((n - 1).choose 1) * ((n - 2).choose (k - 2)) = 2850360 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l752_75200


namespace NUMINAMATH_CALUDE_height_weight_most_suitable_for_regression_l752_75255

-- Define the types of variables
inductive Variable
| CircleArea
| CircleRadius
| Height
| Weight
| ColorBlindness
| Gender
| AcademicPerformance

-- Define a pair of variables
structure VariablePair where
  var1 : Variable
  var2 : Variable

-- Define the types of relationships between variables
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Function to determine the relationship between a pair of variables
def relationshipBetween (pair : VariablePair) : Relationship :=
  match pair with
  | ⟨Variable.CircleArea, Variable.CircleRadius⟩ => Relationship.Functional
  | ⟨Variable.ColorBlindness, Variable.Gender⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.AcademicPerformance⟩ => Relationship.Unrelated
  | ⟨Variable.Height, Variable.Weight⟩ => Relationship.Correlated
  | _ => Relationship.Unrelated  -- Default case

-- Function to determine if a pair is suitable for regression analysis
def suitableForRegression (pair : VariablePair) : Prop :=
  relationshipBetween pair = Relationship.Correlated

-- Theorem stating that height and weight is the most suitable pair for regression
theorem height_weight_most_suitable_for_regression :
  suitableForRegression ⟨Variable.Height, Variable.Weight⟩ ∧
  ¬suitableForRegression ⟨Variable.CircleArea, Variable.CircleRadius⟩ ∧
  ¬suitableForRegression ⟨Variable.ColorBlindness, Variable.Gender⟩ ∧
  ¬suitableForRegression ⟨Variable.Height, Variable.AcademicPerformance⟩ :=
by sorry

end NUMINAMATH_CALUDE_height_weight_most_suitable_for_regression_l752_75255


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l752_75272

theorem tan_alpha_plus_pi_third (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - Real.pi/3) = 1/4) : 
  Real.tan (α + Real.pi/3) = 7/23 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l752_75272


namespace NUMINAMATH_CALUDE_ella_work_days_l752_75247

/-- Represents the number of days Ella worked at each age --/
structure WorkDays where
  age10 : ℕ
  age11 : ℕ
  age12 : ℕ

/-- Calculates the total pay for the given work days --/
def totalPay (w : WorkDays) : ℕ :=
  4 * (10 * w.age10 + 11 * w.age11 + 12 * w.age12)

theorem ella_work_days :
  ∃ (w : WorkDays),
    w.age10 + w.age11 + w.age12 = 180 ∧
    totalPay w = 7920 ∧
    w.age11 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ella_work_days_l752_75247


namespace NUMINAMATH_CALUDE_girls_tried_out_l752_75210

/-- The number of girls who tried out for the basketball team -/
def girls : ℕ := 39

/-- The number of boys who tried out for the basketball team -/
def boys : ℕ := 4

/-- The number of students who got called back -/
def called_back : ℕ := 26

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 17

/-- The total number of students who tried out -/
def total_students : ℕ := called_back + didnt_make_cut

theorem girls_tried_out : girls = total_students - boys := by
  sorry

end NUMINAMATH_CALUDE_girls_tried_out_l752_75210


namespace NUMINAMATH_CALUDE_olivias_chips_quarters_l752_75234

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total amount Olivia pays in dollars -/
def total_dollars : ℕ := 4

/-- The number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- The number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := total_dollars * quarters_per_dollar - quarters_for_soda

theorem olivias_chips_quarters : quarters_for_chips = 4 := by
  sorry

end NUMINAMATH_CALUDE_olivias_chips_quarters_l752_75234


namespace NUMINAMATH_CALUDE_equality_from_sum_squares_l752_75246

theorem equality_from_sum_squares (x y z : ℝ) :
  x^2 + y^2 + z^2 = x*y + y*z + z*x → x = y ∧ y = z := by
  sorry

end NUMINAMATH_CALUDE_equality_from_sum_squares_l752_75246


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l752_75216

/-- A geometric sequence with positive terms -/
structure PositiveGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- The property that 2a_1, (1/2)a_3, and a_2 form an arithmetic sequence -/
def ArithmeticProperty (s : PositiveGeometricSequence) : Prop :=
  2 * s.a 1 + s.a 2 = 2 * ((1/2) * s.a 3)

theorem geometric_sequence_property (s : PositiveGeometricSequence) 
  (h_arith : ArithmeticProperty s) : s.q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l752_75216


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l752_75224

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l752_75224


namespace NUMINAMATH_CALUDE_meet_at_starting_line_l752_75206

theorem meet_at_starting_line (henry_time margo_time cameron_time : ℕ) 
  (henry_eq : henry_time = 7)
  (margo_eq : margo_time = 12)
  (cameron_eq : cameron_time = 9) :
  Nat.lcm (Nat.lcm henry_time margo_time) cameron_time = 252 := by
  sorry

end NUMINAMATH_CALUDE_meet_at_starting_line_l752_75206


namespace NUMINAMATH_CALUDE_triangle_formation_l752_75288

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 3 6 3) ∧
  (can_form_triangle 3 4 5) ∧
  (can_form_triangle (Real.sqrt 3) 1 2) ∧
  (can_form_triangle 1.5 2.5 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l752_75288


namespace NUMINAMATH_CALUDE_existence_of_bounded_irreducible_factorization_l752_75225

def is_irreducible (S : Set ℕ) (x : ℕ) : Prop :=
  x ∈ S ∧ ∀ y z : ℕ, y ∈ S → z ∈ S → x = y * z → (y = 1 ∨ z = 1)

theorem existence_of_bounded_irreducible_factorization 
  (a b : ℕ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_gcd : ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.gcd a b ∧ q ∣ Nat.gcd a b) :
  ∃ t : ℕ, ∀ x ∈ {n : ℕ | n > 0 ∧ n % b = a % b}, 
    ∃ (factors : List ℕ), 
      (∀ f ∈ factors, is_irreducible {n : ℕ | n > 0 ∧ n % b = a % b} f) ∧
      (factors.prod = x) ∧
      (factors.length ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_bounded_irreducible_factorization_l752_75225


namespace NUMINAMATH_CALUDE_smallest_norm_w_l752_75219

variable (w : ℝ × ℝ)

def v : ℝ × ℝ := (4, 2)

theorem smallest_norm_w (h : ‖w + v‖ = 10) :
  ∃ (w_min : ℝ × ℝ), ‖w_min‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ w', ‖w' + v‖ = 10 → ‖w'‖ ≥ ‖w_min‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l752_75219


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l752_75268

theorem quadratic_equation_roots (a : ℝ) :
  let f : ℝ → ℝ := λ x => 4 * x^2 - 4 * (a + 2) * x + a^2 + 11
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ - x₂ = 3 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l752_75268


namespace NUMINAMATH_CALUDE_f_decreasing_l752_75212

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin x else -Real.sin x

theorem f_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(Real.pi / 2) + 2 * k * Real.pi) ((Real.pi / 2) + 2 * k * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_l752_75212


namespace NUMINAMATH_CALUDE_larger_box_capacity_l752_75279

/-- Represents the capacity of a box in terms of volume and paperclips -/
structure Box where
  volume : ℝ
  paperclipCapacity : ℕ

/-- The maximum number of paperclips a box can hold -/
def maxPaperclips (b : Box) : ℕ := b.paperclipCapacity

theorem larger_box_capacity (smallBox largeBox : Box)
  (h1 : smallBox.volume = 20)
  (h2 : smallBox.paperclipCapacity = 80)
  (h3 : largeBox.volume = 100)
  (h4 : largeBox.paperclipCapacity = 380) :
  maxPaperclips largeBox = 380 := by
  sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l752_75279


namespace NUMINAMATH_CALUDE_g_inv_composition_l752_75281

-- Define the function g
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

-- Define the inverse function g⁻¹
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 2
| 4 => 1
| 5 => 4

-- State the theorem
theorem g_inv_composition :
  g_inv (g_inv (g_inv 3)) = 4 := by sorry

end NUMINAMATH_CALUDE_g_inv_composition_l752_75281


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l752_75266

theorem least_possible_third_side_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 →
  (a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + b^2 = c^2) →
  c ≥ Real.sqrt 119 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l752_75266
