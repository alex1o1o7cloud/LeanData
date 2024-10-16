import Mathlib

namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3608_360877

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →
  a + c = 45 →
  a < c →
  (a = (45 - 15 * Real.sqrt 5) / 2 ∧ c = (45 + 15 * Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3608_360877


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_zero_l3608_360840

theorem sum_of_even_coefficients_zero (a : Fin 7 → ℝ) :
  (∀ x : ℝ, (x - 1)^6 = a 0 * x^6 + a 1 * x^5 + a 2 * x^4 + a 3 * x^3 + a 4 * x^2 + a 5 * x + a 6) →
  a 0 + a 2 + a 4 + a 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_zero_l3608_360840


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3608_360818

theorem cube_volume_surface_area (y : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*y ∧ 6*s^2 = 2*y) → y = 5832 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3608_360818


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3608_360814

theorem quadratic_equation_solution (x : ℝ) : -x^2 - (-16 + 10)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3608_360814


namespace NUMINAMATH_CALUDE_tangent_angle_range_l3608_360858

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 2

noncomputable def α (x : ℝ) : ℝ := Real.arctan (3 * x^2 - 1)

theorem tangent_angle_range :
  ∀ x : ℝ, α x ∈ Set.Icc 0 (Real.pi / 2) ∪ Set.Icc (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_range_l3608_360858


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_equals_one_l3608_360853

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Theorem statement
theorem tangent_perpendicular_implies_a_equals_one (a : ℝ) :
  (f_deriv a 1 * (-1/4) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_equals_one_l3608_360853


namespace NUMINAMATH_CALUDE_smallest_class_number_l3608_360807

/-- Systematic sampling function that returns the set of selected numbers -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (startNum : ℕ) : Finset ℕ :=
  sorry

/-- Theorem stating that in a systematic sampling of 5 from 30 starting with 26, the smallest number is 2 -/
theorem smallest_class_number
  (h1 : systematicSample 30 5 26 = {2, 8, 14, 20, 26}) :
  2 = Finset.min' (systematicSample 30 5 26) sorry :=
sorry

end NUMINAMATH_CALUDE_smallest_class_number_l3608_360807


namespace NUMINAMATH_CALUDE_expression_evaluation_l3608_360847

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 - 2*x*y = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3608_360847


namespace NUMINAMATH_CALUDE_most_cars_are_blue_l3608_360892

theorem most_cars_are_blue (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end NUMINAMATH_CALUDE_most_cars_are_blue_l3608_360892


namespace NUMINAMATH_CALUDE_power_relations_l3608_360860

/-- Given real numbers a, b, c, d satisfying certain conditions, 
    prove statements about their powers. -/
theorem power_relations (a b c d : ℝ) 
    (sum_eq : a + b = c + d) 
    (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
    (a^5 + b^5 = c^5 + d^5) ∧ 
    ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
  sorry


end NUMINAMATH_CALUDE_power_relations_l3608_360860


namespace NUMINAMATH_CALUDE_fixed_points_specific_case_range_of_a_for_two_fixed_points_l3608_360827

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Part 1: Fixed points when a = 1 and b = -2
theorem fixed_points_specific_case :
  is_fixed_point (f 1 (-2)) (-1) ∧ is_fixed_point (f 1 (-2)) 3 :=
sorry

-- Part 2: Range of a for two distinct fixed points
theorem range_of_a_for_two_fixed_points :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) →
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_specific_case_range_of_a_for_two_fixed_points_l3608_360827


namespace NUMINAMATH_CALUDE_sequence_term_16_l3608_360886

theorem sequence_term_16 (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = (Real.sqrt 2) ^ (n - 1)) →
  ∃ n : ℕ, n > 0 ∧ a n = 16 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_16_l3608_360886


namespace NUMINAMATH_CALUDE_normal_dist_probability_l3608_360855

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (hσ : σ > 0) : Type := Unit

-- Define the probability function
def P (X : normal_dist 1 σ hσ) (a b : ℝ) : ℝ := sorry

-- Define our theorem
theorem normal_dist_probability 
  (σ : ℝ) (hσ : σ > 0) (X : normal_dist 1 σ hσ) 
  (h : P X 0 1 = 0.4) : P X 0 2 = 0.8 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l3608_360855


namespace NUMINAMATH_CALUDE_correlation_relationships_l3608_360843

/-- Represents a relationship between two variables -/
structure Relationship where
  variable1 : String
  variable2 : String

/-- Determines if a relationship represents a correlation -/
def is_correlation (r : Relationship) : Prop :=
  match r with
  | ⟨"snowfall", "traffic accidents"⟩ => True
  | ⟨"brain capacity", "intelligence"⟩ => True
  | ⟨"age", "weight"⟩ => False
  | ⟨"rainfall", "crop yield"⟩ => True
  | _ => False

/-- The main theorem stating which relationships represent correlations -/
theorem correlation_relationships :
  let r1 : Relationship := ⟨"snowfall", "traffic accidents"⟩
  let r2 : Relationship := ⟨"brain capacity", "intelligence"⟩
  let r3 : Relationship := ⟨"age", "weight"⟩
  let r4 : Relationship := ⟨"rainfall", "crop yield"⟩
  is_correlation r1 ∧ is_correlation r2 ∧ ¬is_correlation r3 ∧ is_correlation r4 :=
by sorry


end NUMINAMATH_CALUDE_correlation_relationships_l3608_360843


namespace NUMINAMATH_CALUDE_sum_inequality_l3608_360849

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) +
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) +
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3608_360849


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3608_360873

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 3 - 6 * Complex.I) : 
  z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3608_360873


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3608_360815

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : (a - 2*i) * i = b - i) : 
  a + b*i = -1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3608_360815


namespace NUMINAMATH_CALUDE_expand_product_l3608_360817

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x - 5 * x^2 + 20 / x^3) = 3 / x - 15 * x^2 / 4 + 15 / x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3608_360817


namespace NUMINAMATH_CALUDE_disjunction_truth_l3608_360895

theorem disjunction_truth (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_truth_l3608_360895


namespace NUMINAMATH_CALUDE_rotation_transformation_l3608_360832

-- Define the triangles
def triangle_DEF : List (ℝ × ℝ) := [(0, 0), (0, 10), (14, 0)]
def triangle_DEF_prime : List (ℝ × ℝ) := [(28, 14), (40, 14), (28, 4)]

-- Define the rotation function
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

theorem rotation_transformation (n p q : ℝ) :
  0 < n → n < 180 →
  (∀ (point : ℝ × ℝ), point ∈ triangle_DEF →
    rotate (p, q) n point ∈ triangle_DEF_prime) →
  n + p + q = 104 := by sorry

end NUMINAMATH_CALUDE_rotation_transformation_l3608_360832


namespace NUMINAMATH_CALUDE_april_roses_unsold_l3608_360894

/-- Calculates the number of roses left unsold given the initial number of roses,
    the price per rose, and the total amount earned from sales. -/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Proves that the number of roses left unsold is 4 given the problem conditions. -/
theorem april_roses_unsold :
  roses_left_unsold 9 7 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_roses_unsold_l3608_360894


namespace NUMINAMATH_CALUDE_line_equation_l3608_360890

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line forms an isosceles right triangle with the coordinate axes -/
def formsIsoscelesRightTriangle (l : Line) : Prop :=
  ∃ a : ℝ, (l.a = 1 ∧ l.b = 1 ∧ l.c = -a) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = a)

/-- The main theorem -/
theorem line_equation (l : Line) 
  (h1 : pointOnLine l 2 3)
  (h2 : formsIsoscelesRightTriangle l) :
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -5) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3608_360890


namespace NUMINAMATH_CALUDE_king_of_red_suit_probability_l3608_360883

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)

/-- Represents the number of Kings of red suits in a standard deck -/
structure RedKings :=
  (count : Nat)

/-- The probability of selecting a specific card from a deck -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

theorem king_of_red_suit_probability (d : Deck) (rk : RedKings) :
  d.cards = 52 → rk.count = 2 → probability rk.count d.cards = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_king_of_red_suit_probability_l3608_360883


namespace NUMINAMATH_CALUDE_rhombus_shorter_diagonal_l3608_360805

/-- A rhombus with perimeter 9.6 and adjacent angles in ratio 1:2 has a shorter diagonal of length 2.4 -/
theorem rhombus_shorter_diagonal (p : ℝ) (r : ℚ) (d : ℝ) : 
  p = 9.6 → -- perimeter is 9.6
  r = 1/2 → -- ratio of adjacent angles is 1:2
  d = 2.4 -- shorter diagonal is 2.4
  := by sorry

end NUMINAMATH_CALUDE_rhombus_shorter_diagonal_l3608_360805


namespace NUMINAMATH_CALUDE_function_inequality_l3608_360887

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
    (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3608_360887


namespace NUMINAMATH_CALUDE_min_balls_for_target_color_l3608_360842

def orange_balls : ℕ := 26
def purple_balls : ℕ := 21
def brown_balls : ℕ := 20
def gray_balls : ℕ := 15
def silver_balls : ℕ := 12
def golden_balls : ℕ := 10

def target_count : ℕ := 17

theorem min_balls_for_target_color :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → 
      ∃ (o p b g s g' : ℕ), 
        o + p + b + g + s + g' = m ∧ 
        o ≤ orange_balls ∧ 
        p ≤ purple_balls ∧ 
        b ≤ brown_balls ∧ 
        g ≤ gray_balls ∧ 
        s ≤ silver_balls ∧ 
        g' ≤ golden_balls ∧
        o < target_count ∧ 
        p < target_count ∧ 
        b < target_count ∧ 
        g < target_count ∧ 
        s < target_count ∧ 
        g' < target_count) ∧
    (∀ (o p b g s g' : ℕ), 
      o + p + b + g + s + g' = n → 
      o ≤ orange_balls → 
      p ≤ purple_balls → 
      b ≤ brown_balls → 
      g ≤ gray_balls → 
      s ≤ silver_balls → 
      g' ≤ golden_balls →
      o ≥ target_count ∨ 
      p ≥ target_count ∨ 
      b ≥ target_count ∨ 
      g ≥ target_count ∨ 
      s ≥ target_count ∨ 
      g' ≥ target_count) ∧
    n = 86 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_for_target_color_l3608_360842


namespace NUMINAMATH_CALUDE_tan_2a_values_l3608_360879

theorem tan_2a_values (a : ℝ) (h : 2 * Real.sin (2 * a) = 1 + Real.cos (2 * a)) :
  Real.tan (2 * a) = 4 / 3 ∨ Real.tan (2 * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2a_values_l3608_360879


namespace NUMINAMATH_CALUDE_tray_height_is_five_l3608_360850

/-- The height of a tray formed by cutting and folding a square piece of paper -/
def trayHeight (sideLength : ℝ) (cutDistance : ℝ) (cutAngle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the height of the tray is 5 under given conditions -/
theorem tray_height_is_five :
  trayHeight 120 (Real.sqrt 25) (π / 4) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tray_height_is_five_l3608_360850


namespace NUMINAMATH_CALUDE_power_function_m_squared_minus_three_l3608_360888

/-- A function f(x) is a power function if it can be written as f(x) = ax^n, where a and n are constants and n ≠ 0. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), n ≠ 0 ∧ ∀ x, f x = a * x^n

/-- Given that y = (m^2 - 3)x^(2m) is a power function with respect to x, prove that m = ±2. -/
theorem power_function_m_squared_minus_three (m : ℝ) :
  is_power_function (λ x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_squared_minus_three_l3608_360888


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3608_360829

theorem product_of_sum_of_squares (p q r s : ℤ) :
  ∃ (x y : ℤ), (p^2 + q^2) * (r^2 + s^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3608_360829


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3608_360874

theorem proof_by_contradiction_assumption 
  (P : ℝ → ℝ → Prop) 
  (Q : ℝ → Prop) 
  (R : ℝ → Prop) 
  (h : ∀ x y, P x y → (Q x ∨ R y)) :
  (∀ x y, P x y → (Q x ∨ R y)) ↔ 
  (∀ x y, P x y ∧ ¬Q x ∧ ¬R y → False) := by
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3608_360874


namespace NUMINAMATH_CALUDE_chess_probabilities_l3608_360875

theorem chess_probabilities (p_draw p_b_win : ℝ) 
  (h_draw : p_draw = 1/2)
  (h_b_win : p_b_win = 1/3) :
  let p_a_win := 1 - p_draw - p_b_win
  let p_a_not_lose := p_draw + p_a_win
  (p_a_win = 1/6) ∧ (p_a_not_lose = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_chess_probabilities_l3608_360875


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3608_360830

theorem divisibility_by_five (a b c d e f g : ℕ) 
  (h1 : (a + b + c + d + e + f) % 5 = 0)
  (h2 : (a + b + c + d + e + g) % 5 = 0)
  (h3 : (a + b + c + d + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + d + e + f + g) % 5 = 0)
  (h6 : (a + c + d + e + f + g) % 5 = 0)
  (h7 : (b + c + d + e + f + g) % 5 = 0) :
  (a % 5 = 0) ∧ (b % 5 = 0) ∧ (c % 5 = 0) ∧ (d % 5 = 0) ∧ 
  (e % 5 = 0) ∧ (f % 5 = 0) ∧ (g % 5 = 0) := by
  sorry

#check divisibility_by_five

end NUMINAMATH_CALUDE_divisibility_by_five_l3608_360830


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_l3608_360806

/-- Given a family where some members eat vegetarian, some eat non-vegetarian, and some eat both,
    this theorem proves that the total number of people who eat vegetarian food is 28. -/
theorem vegetarian_eaters_count (only_veg : ℕ) (both : ℕ) 
    (h1 : only_veg = 16) (h2 : both = 12) : only_veg + both = 28 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_l3608_360806


namespace NUMINAMATH_CALUDE_line_direction_vector_l3608_360880

/-- Given a line passing through points (-5, 0) and (-2, 2), if its direction vector
    is of the form (2, b), then b = 4/3 -/
theorem line_direction_vector (b : ℚ) : 
  let p1 : ℚ × ℚ := (-5, 0)
  let p2 : ℚ × ℚ := (-2, 2)
  let dir : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = dir) → b = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3608_360880


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3608_360833

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3608_360833


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3608_360869

/-- Given a boat traveling downstream, prove its speed in still water. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 216)
  (h3 : downstream_time = 8) :
  downstream_distance / downstream_time - stream_speed = 22 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3608_360869


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3608_360834

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbonAtoms hydrogenAtoms oxygenAtoms : ℕ) 
  (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  (carbonAtoms : ℝ) * carbonWeight + 
  (hydrogenAtoms : ℝ) * hydrogenWeight + 
  (oxygenAtoms : ℝ) * oxygenWeight

/-- The molecular weight of the given compound is approximately 58.078 g/mol -/
theorem compound_molecular_weight :
  let carbonAtoms : ℕ := 3
  let hydrogenAtoms : ℕ := 6
  let oxygenAtoms : ℕ := 1
  let carbonWeight : ℝ := 12.01
  let hydrogenWeight : ℝ := 1.008
  let oxygenWeight : ℝ := 16.00
  abs (molecularWeight carbonAtoms hydrogenAtoms oxygenAtoms 
    carbonWeight hydrogenWeight oxygenWeight - 58.078) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3608_360834


namespace NUMINAMATH_CALUDE_mall_sale_plate_cost_l3608_360825

theorem mall_sale_plate_cost 
  (treadmill_price : ℝ)
  (discount_rate : ℝ)
  (num_plates : ℕ)
  (plate_price : ℝ)
  (total_paid : ℝ)
  (h1 : treadmill_price = 1350)
  (h2 : discount_rate = 0.3)
  (h3 : num_plates = 2)
  (h4 : plate_price = 50)
  (h5 : total_paid = 1045) :
  treadmill_price * (1 - discount_rate) + num_plates * plate_price = total_paid ∧
  num_plates * plate_price = 100 := by
sorry

end NUMINAMATH_CALUDE_mall_sale_plate_cost_l3608_360825


namespace NUMINAMATH_CALUDE_bacon_count_l3608_360859

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 228

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 23

/-- The difference between students suggesting bacon and tomatoes -/
def bacon_tomato_diff : ℕ := 314

/-- The number of students who suggested bacon -/
def bacon : ℕ := tomatoes + bacon_tomato_diff

theorem bacon_count : bacon = 337 := by
  sorry

end NUMINAMATH_CALUDE_bacon_count_l3608_360859


namespace NUMINAMATH_CALUDE_friend_rides_80_times_more_l3608_360816

/-- Tommy's effective riding area in square blocks -/
def tommy_area : ℚ := 1

/-- Tommy's friend's riding area in square blocks -/
def friend_area : ℚ := 80

/-- The ratio of Tommy's friend's riding area to Tommy's effective riding area -/
def area_ratio : ℚ := friend_area / tommy_area

theorem friend_rides_80_times_more : area_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_friend_rides_80_times_more_l3608_360816


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3608_360848

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (x^3 - 5*x^2 + 13*x - 7 = (x - a) * (x - b) * (x - c)) → 
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 490 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3608_360848


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l3608_360812

/-- The number of Popsicles Megan can finish in 5 hours -/
def popsicles_in_5_hours : ℕ := 15

/-- The time in minutes it takes Megan to eat one Popsicle -/
def minutes_per_popsicle : ℕ := 20

/-- The number of hours given in the problem -/
def hours : ℕ := 5

/-- Theorem stating that Megan can finish 15 Popsicles in 5 hours -/
theorem megan_popsicle_consumption :
  popsicles_in_5_hours = (hours * 60) / minutes_per_popsicle :=
by sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l3608_360812


namespace NUMINAMATH_CALUDE_coronavirus_cases_day3_l3608_360836

/-- Represents the number of Coronavirus cases over three days -/
structure CoronavirusCases where
  initial_cases : ℕ
  day2_increase : ℕ
  day2_recoveries : ℕ
  day3_recoveries : ℕ
  final_total : ℕ

/-- Calculates the number of new cases on day 3 -/
def new_cases_day3 (c : CoronavirusCases) : ℕ :=
  c.final_total - (c.initial_cases + c.day2_increase - c.day2_recoveries - c.day3_recoveries)

/-- Theorem stating that given the conditions, the number of new cases on day 3 is 1500 -/
theorem coronavirus_cases_day3 (c : CoronavirusCases) 
  (h1 : c.initial_cases = 2000)
  (h2 : c.day2_increase = 500)
  (h3 : c.day2_recoveries = 50)
  (h4 : c.day3_recoveries = 200)
  (h5 : c.final_total = 3750) :
  new_cases_day3 c = 1500 := by
  sorry

end NUMINAMATH_CALUDE_coronavirus_cases_day3_l3608_360836


namespace NUMINAMATH_CALUDE_roots_expression_l3608_360823

theorem roots_expression (p q α β γ δ : ℝ) : 
  (α^2 - p*α + 1 = 0) → 
  (β^2 - p*β + 1 = 0) → 
  (γ^2 - q*γ + 1 = 0) → 
  (δ^2 - q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l3608_360823


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l3608_360866

theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1200) 
  (h2 : breadth = 240) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l3608_360866


namespace NUMINAMATH_CALUDE_even_function_sum_of_angles_l3608_360854

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_sum_of_angles (θ φ : ℝ) :
  IsEven (fun x ↦ Real.cos (x + θ) + Real.sqrt 2 * Real.sin (x + φ)) →
  0 < θ ∧ θ < π / 2 →
  0 < φ ∧ φ < π / 2 →
  Real.cos θ = Real.sqrt 6 / 3 * Real.sin φ →
  θ + φ = 7 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_of_angles_l3608_360854


namespace NUMINAMATH_CALUDE_smallest_a_divisible_by_65_l3608_360841

theorem smallest_a_divisible_by_65 :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (n : ℤ), 65 ∣ (5 * n^13 + 13 * n^5 + 9 * a * n)) ∧
  (∀ (b : ℕ), b > 0 → b < a → 
    ∃ (m : ℤ), ¬(65 ∣ (5 * m^13 + 13 * m^5 + 9 * b * m))) ∧
  a = 63 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_divisible_by_65_l3608_360841


namespace NUMINAMATH_CALUDE_square_not_end_two_odd_digits_l3608_360885

theorem square_not_end_two_odd_digits (n : ℕ) : 
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ n^2 % 100 = 10 * d₁ + d₂ → (d₁ % 2 = 0 ∨ d₂ % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_not_end_two_odd_digits_l3608_360885


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_plus_one_ge_one_l3608_360857

theorem sum_reciprocal_squared_plus_one_ge_one (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_plus_one_ge_one_l3608_360857


namespace NUMINAMATH_CALUDE_train_length_l3608_360897

theorem train_length (t : ℝ) 
  (h1 : (t + 100) / 15 = (t + 250) / 20) : t = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3608_360897


namespace NUMINAMATH_CALUDE_largest_n_digit_divisible_by_61_l3608_360839

theorem largest_n_digit_divisible_by_61 (n : ℕ+) :
  ∃ (k : ℕ), k = (10^n.val - 1) - ((10^n.val - 1) % 61) ∧ 
  k % 61 = 0 ∧
  k ≤ 10^n.val - 1 ∧
  ∀ m : ℕ, m % 61 = 0 → m ≤ 10^n.val - 1 → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_n_digit_divisible_by_61_l3608_360839


namespace NUMINAMATH_CALUDE_bob_has_31_pennies_l3608_360868

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 1 = 4 * (alex_pennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 1 = 3 * (alex_pennies + 1)

/-- Bob has 31 pennies -/
theorem bob_has_31_pennies : bob_pennies = 31 := by sorry

end NUMINAMATH_CALUDE_bob_has_31_pennies_l3608_360868


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l3608_360863

/-- Proves that a rectangular garden with length three times its width and area 588 square meters has a width of 14 meters. -/
theorem rectangular_garden_width :
  ∀ (width length area : ℝ),
    length = 3 * width →
    area = length * width →
    area = 588 →
    width = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l3608_360863


namespace NUMINAMATH_CALUDE_principal_calculation_l3608_360835

theorem principal_calculation (P R : ℝ) 
  (h1 : P + (P * R * 2) / 100 = 660)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 516 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3608_360835


namespace NUMINAMATH_CALUDE_subtract_abs_from_local_value_l3608_360811

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  let digits := n.digits 10
  let index := digits.findIndex (· = d)
  10 ^ (digits.length - index - 1) * d

def absolute_value (n : ℤ) : ℕ := n.natAbs

theorem subtract_abs_from_local_value :
  local_value 564823 4 - absolute_value 4 = 39996 := by
  sorry

end NUMINAMATH_CALUDE_subtract_abs_from_local_value_l3608_360811


namespace NUMINAMATH_CALUDE_natural_number_pairs_satisfying_equation_l3608_360809

theorem natural_number_pairs_satisfying_equation :
  ∀ (x y : ℕ), 2^x - 3^y = 7 ↔ (x = 3 ∧ y = 0) ∨ (x = 4 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_pairs_satisfying_equation_l3608_360809


namespace NUMINAMATH_CALUDE_no_integer_solution_l3608_360819

theorem no_integer_solution :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x*y*z - 1 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3608_360819


namespace NUMINAMATH_CALUDE_bicycle_price_l3608_360822

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 120)
  (h2 : upfront_percentage = 0.2) :
  upfront_payment / upfront_percentage = 600 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_l3608_360822


namespace NUMINAMATH_CALUDE_union_M_N_equals_real_l3608_360804

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | x < 3}

-- Statement to prove
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_real_l3608_360804


namespace NUMINAMATH_CALUDE_school_total_is_125_l3608_360810

/-- Represents the number of students in a school with specific age distribution. -/
structure School where
  /-- The number of students who are 8 years old -/
  eight_years : ℕ
  /-- The proportion of students below 8 years old -/
  below_eight_percent : ℚ
  /-- The ratio of students above 8 years old to students who are 8 years old -/
  above_eight_ratio : ℚ

/-- Calculates the total number of students in the school -/
def total_students (s : School) : ℕ :=
  sorry

/-- Theorem stating that for a school with given age distribution, 
    the total number of students is 125 -/
theorem school_total_is_125 (s : School) 
  (h1 : s.eight_years = 60)
  (h2 : s.below_eight_percent = 1/5)
  (h3 : s.above_eight_ratio = 2/3) : 
  total_students s = 125 := by
  sorry

end NUMINAMATH_CALUDE_school_total_is_125_l3608_360810


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l3608_360828

/-- The function f(x) = ln(2x) - 1 has a root in the interval (1, 2) -/
theorem f_has_root_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ Real.log (2 * x) - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_root_in_interval_l3608_360828


namespace NUMINAMATH_CALUDE_probability_of_integer_occurrence_l3608_360803

theorem probability_of_integer_occurrence (a b : ℤ) (h : a ≤ b) :
  let range := b - a + 1
  (∀ k : ℤ, a ≤ k ∧ k ≤ b → (1 : ℚ) / range = (1 : ℚ) / range) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_integer_occurrence_l3608_360803


namespace NUMINAMATH_CALUDE_friends_score_l3608_360867

/-- Given that Edward and his friend scored a total of 13 points in basketball,
    and Edward scored 7 points, prove that Edward's friend scored 6 points. -/
theorem friends_score (total : ℕ) (edward : ℕ) (friend : ℕ)
    (h1 : total = 13)
    (h2 : edward = 7)
    (h3 : total = edward + friend) :
  friend = 6 := by
sorry

end NUMINAMATH_CALUDE_friends_score_l3608_360867


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l3608_360872

theorem system_inequalities_solution (x : ℝ) :
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l3608_360872


namespace NUMINAMATH_CALUDE_divisors_of_square_l3608_360837

theorem divisors_of_square (n : ℕ) : 
  (∃ p : ℕ, Prime p ∧ n = p^3) → 
  (Finset.card (Nat.divisors n) = 4) → 
  (Finset.card (Nat.divisors (n^2)) = 7) := by
sorry

end NUMINAMATH_CALUDE_divisors_of_square_l3608_360837


namespace NUMINAMATH_CALUDE_tomatoes_left_l3608_360800

theorem tomatoes_left (initial_tomatoes : ℕ) (eaten_fraction : ℚ) : initial_tomatoes = 21 ∧ eaten_fraction = 1/3 → initial_tomatoes - (initial_tomatoes * eaten_fraction).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l3608_360800


namespace NUMINAMATH_CALUDE_uniform_price_calculation_l3608_360844

/-- Calculates the price of a uniform given the conditions of a servant's employment --/
def uniform_price (full_year_salary : ℚ) (actual_salary : ℚ) (months_worked : ℕ) : ℚ :=
  full_year_salary * (months_worked / 12) - actual_salary

theorem uniform_price_calculation :
  uniform_price 900 650 9 = 25 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_calculation_l3608_360844


namespace NUMINAMATH_CALUDE_equipment_cost_theorem_l3608_360808

/-- The total cost of equipment requests for three departments -/
theorem equipment_cost_theorem 
  (cost1 cost2 cost3 : ℝ) 
  (h1 : cost1 = 0.45 * cost2)
  (h2 : cost2 = 0.8 * cost3)
  (h3 : cost3 = cost1 + 640) :
  cost1 + cost2 + cost3 = 2160 := by
  sorry

#check equipment_cost_theorem

end NUMINAMATH_CALUDE_equipment_cost_theorem_l3608_360808


namespace NUMINAMATH_CALUDE_gcd_32_24_l3608_360884

theorem gcd_32_24 : Nat.gcd 32 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_32_24_l3608_360884


namespace NUMINAMATH_CALUDE_red_balls_in_box_l3608_360845

theorem red_balls_in_box (total_balls : ℕ) (prob_red : ℚ) (num_red : ℕ) : 
  total_balls = 6 → 
  prob_red = 1/3 → 
  (num_red : ℚ) / total_balls = prob_red → 
  num_red = 2 := by
sorry

end NUMINAMATH_CALUDE_red_balls_in_box_l3608_360845


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l3608_360865

theorem average_marks_math_chem (M P C : ℕ) : 
  M + P = 20 →
  C = P + 20 →
  (M + C) / 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l3608_360865


namespace NUMINAMATH_CALUDE_direction_vector_y_component_l3608_360899

/-- Given a line determined by two points in 2D space, prove that if the direction vector
    has a specific x-component, then its y-component has a specific value. -/
theorem direction_vector_y_component 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-1, -1)) 
  (h2 : p2 = (3, 4)) 
  (direction : ℝ × ℝ) 
  (h_x_component : direction.1 = 3) : 
  direction.2 = 15/4 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_y_component_l3608_360899


namespace NUMINAMATH_CALUDE_smallest_square_covered_by_rectangles_l3608_360870

theorem smallest_square_covered_by_rectangles :
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (∃ (s : ℕ), 
      (s > 0) ∧ 
      (s * s = 12 * n) ∧ 
      (∀ (t : ℕ), (0 < t ∧ t < s) → ¬(∃ (m : ℕ), t * t = 12 * m)) ∧
      (n = 9)) := by
sorry

end NUMINAMATH_CALUDE_smallest_square_covered_by_rectangles_l3608_360870


namespace NUMINAMATH_CALUDE_park_journey_distance_sum_l3608_360871

/-- Represents the speed and start time of a traveler -/
structure Traveler where
  speed : ℚ
  startTime : ℚ

/-- The problem setup -/
def ParkJourney (d : ℚ) (patrick tanya jose : Traveler) : Prop :=
  patrick.speed > 0 ∧
  patrick.startTime = 0 ∧
  tanya.speed = patrick.speed + 2 ∧
  tanya.startTime = patrick.startTime + 1 ∧
  jose.speed = tanya.speed + 7 ∧
  jose.startTime = tanya.startTime + 1 ∧
  d / patrick.speed = (d / tanya.speed) + 1 ∧
  d / patrick.speed = (d / jose.speed) + 2

theorem park_journey_distance_sum :
  ∀ (d : ℚ) (patrick tanya jose : Traveler),
  ParkJourney d patrick tanya jose →
  ∃ (m n : ℕ), m.Coprime n ∧ d = m / n ∧ m + n = 277 := by
  sorry

end NUMINAMATH_CALUDE_park_journey_distance_sum_l3608_360871


namespace NUMINAMATH_CALUDE_police_catch_thief_time_l3608_360856

/-- Proves that the time taken by the police to catch the thief is 2 hours -/
theorem police_catch_thief_time
  (thief_speed : ℝ)
  (police_station_distance : ℝ)
  (police_delay : ℝ)
  (police_speed : ℝ)
  (h1 : thief_speed = 20)
  (h2 : police_station_distance = 60)
  (h3 : police_delay = 1)
  (h4 : police_speed = 40)
  : ℝ :=
by
  sorry

#check police_catch_thief_time

end NUMINAMATH_CALUDE_police_catch_thief_time_l3608_360856


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3608_360878

-- 1. 78 × 4 + 488
theorem problem_1 : 78 * 4 + 488 = 800 := by sorry

-- 2. 350 × (12 + 342 ÷ 9)
theorem problem_2 : 350 * (12 + 342 / 9) = 17500 := by sorry

-- 3. (3600 - 18 × 200) ÷ 253
theorem problem_3 : (3600 - 18 * 200) / 253 = 0 := by sorry

-- 4. 1903 - 475 × 4
theorem problem_4 : 1903 - 475 * 4 = 3 := by sorry

-- 5. 480 ÷ (125 - 117)
theorem problem_5 : 480 / (125 - 117) = 60 := by sorry

-- 6. (243 - 162) ÷ 27 × 380
theorem problem_6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l3608_360878


namespace NUMINAMATH_CALUDE_segments_form_triangle_l3608_360898

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (4, 5, 7) can form a triangle -/
theorem segments_form_triangle : can_form_triangle 4 5 7 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l3608_360898


namespace NUMINAMATH_CALUDE_primitive_poly_count_l3608_360838

/-- A polynomial with integer coefficients -/
structure IntPoly :=
  (a₂ a₁ a₀ : ℤ)

/-- The set of integers from 1 to 5 -/
def S : Set ℤ := {1, 2, 3, 4, 5}

/-- A polynomial is primitive if the gcd of its coefficients is 1 -/
def isPrimitive (p : IntPoly) : Prop :=
  Nat.gcd p.a₂.natAbs (Nat.gcd p.a₁.natAbs p.a₀.natAbs) = 1

/-- The product of two polynomials -/
def polyMul (p q : IntPoly) : IntPoly :=
  ⟨p.a₂ * q.a₂,
   p.a₂ * q.a₁ + p.a₁ * q.a₂,
   p.a₂ * q.a₀ + p.a₁ * q.a₁ + p.a₀ * q.a₂⟩

/-- The number of pairs of polynomials (f, g) such that f * g is primitive -/
def N : ℕ := sorry

theorem primitive_poly_count :
  N ≡ 689 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_primitive_poly_count_l3608_360838


namespace NUMINAMATH_CALUDE_average_age_combined_l3608_360882

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 40 →
  n_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents) = 28.8 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l3608_360882


namespace NUMINAMATH_CALUDE_clock_equivalent_hours_l3608_360889

theorem clock_equivalent_hours : ∃ (n : ℕ), n > 6 ∧ n ≡ n^2 [ZMOD 24] ∧
  ∀ (m : ℕ), m > 6 ∧ m < n → ¬(m ≡ m^2 [ZMOD 24]) :=
by sorry

end NUMINAMATH_CALUDE_clock_equivalent_hours_l3608_360889


namespace NUMINAMATH_CALUDE_odot_equation_solution_l3608_360891

/-- The operation ⊙ defined as a ⊙ b = a + √(b² + √(b² + √(b² + ...))) -/
noncomputable def odot (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2))))

/-- If 9 ⊙ h = 12, then h = √6 -/
theorem odot_equation_solution :
  ∃ h : ℝ, odot 9 h = 12 ∧ h = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_odot_equation_solution_l3608_360891


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l3608_360826

/-- A linear function y = (2m + 2)x + 5 is decreasing if and only if m < -1 -/
theorem linear_function_decreasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → ((2*m + 2)*x₁ + 5) > ((2*m + 2)*x₂ + 5)) ↔ m < -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l3608_360826


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3608_360813

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3608_360813


namespace NUMINAMATH_CALUDE_savings_account_decrease_l3608_360824

theorem savings_account_decrease (initial_balance : ℝ) (increase_percent : ℝ) (final_balance_percent : ℝ) :
  initial_balance = 125 →
  increase_percent = 25 →
  final_balance_percent = 100 →
  let increased_balance := initial_balance * (1 + increase_percent / 100)
  let final_balance := initial_balance * (final_balance_percent / 100)
  let decrease_amount := increased_balance - final_balance
  let decrease_percent := (decrease_amount / increased_balance) * 100
  decrease_percent = 20 := by
sorry

end NUMINAMATH_CALUDE_savings_account_decrease_l3608_360824


namespace NUMINAMATH_CALUDE_inscribed_iff_side_length_le_l3608_360876

/-- A regular polygon -/
structure RegularPolygon where
  n : ℕ
  sideLength : ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a regular polygon is inscribed in a circle -/
def isInscribed (p : RegularPolygon) (c : Circle) : Prop :=
  sorry

/-- The side length of an inscribed regular n-gon in a given circle -/
def inscribedSideLength (n : ℕ) (c : Circle) : ℝ :=
  sorry

theorem inscribed_iff_side_length_le
  (n : ℕ) (c : Circle) (p : RegularPolygon) 
  (h1 : p.n = n) :
  isInscribed p c ↔ p.sideLength ≤ inscribedSideLength n c :=
sorry

end NUMINAMATH_CALUDE_inscribed_iff_side_length_le_l3608_360876


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3608_360821

theorem trigonometric_equation_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  ∀ x : ℝ, 0 < x → x < 2 * Real.pi →
    (Real.sin (3 * x) + a * Real.sin (2 * x) + 2 * Real.sin x = 0) →
    (x = 0 ∨ x = Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3608_360821


namespace NUMINAMATH_CALUDE_soccer_players_count_l3608_360852

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l3608_360852


namespace NUMINAMATH_CALUDE_parking_lot_valid_tickets_percentage_l3608_360864

theorem parking_lot_valid_tickets_percentage 
  (total_cars : ℕ) 
  (unpaid_cars : ℕ) 
  (valid_ticket_percentage : ℝ) :
  total_cars = 300 →
  unpaid_cars = 30 →
  (valid_ticket_percentage / 5 + valid_ticket_percentage) * total_cars / 100 = total_cars - unpaid_cars →
  valid_ticket_percentage = 75 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_valid_tickets_percentage_l3608_360864


namespace NUMINAMATH_CALUDE_largest_sphere_in_folded_rectangle_l3608_360861

/-- Represents a rectangle ABCD folded into a tetrahedron D-ABC -/
structure FoldedRectangle where
  ab : ℝ
  bc : ℝ
  d_projects_on_ab : Bool

/-- The radius of the largest inscribed sphere in the tetrahedron formed by folding the rectangle -/
def largest_inscribed_sphere_radius (r : FoldedRectangle) : ℝ := 
  sorry

/-- Theorem stating that for a rectangle with AB = 4 and BC = 3, folded into a tetrahedron
    where D projects onto AB, the radius of the largest inscribed sphere is 3/2 -/
theorem largest_sphere_in_folded_rectangle :
  ∀ (r : FoldedRectangle), 
    r.ab = 4 ∧ r.bc = 3 ∧ r.d_projects_on_ab = true →
    largest_inscribed_sphere_radius r = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_sphere_in_folded_rectangle_l3608_360861


namespace NUMINAMATH_CALUDE_increasing_cubic_range_l3608_360862

/-- A cubic function f(x) = x³ + ax² + 7ax -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 7*a*x

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 7*a

theorem increasing_cubic_range (a : ℝ) :
  (∀ x : ℝ, (f_deriv a x) ≥ 0) → 0 ≤ a ∧ a ≤ 21 :=
sorry

end NUMINAMATH_CALUDE_increasing_cubic_range_l3608_360862


namespace NUMINAMATH_CALUDE_remainder_of_2543_base12_div_7_l3608_360846

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 2543₁₂ --/
def number : List Nat := [3, 4, 5, 2]

theorem remainder_of_2543_base12_div_7 :
  (base12ToDecimal number) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2543_base12_div_7_l3608_360846


namespace NUMINAMATH_CALUDE_opposite_expressions_l3608_360802

theorem opposite_expressions (x : ℚ) : x = -3/2 → (3 + x/3 = -(x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l3608_360802


namespace NUMINAMATH_CALUDE_square_land_side_length_l3608_360801

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l3608_360801


namespace NUMINAMATH_CALUDE_triangle_function_sign_l3608_360851

/-- Triangle with ordered sides -/
structure OrderedTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : OrderedTriangle) : ℝ := sorry

/-- The inradius of a triangle -/
noncomputable def inradius (t : OrderedTriangle) : ℝ := sorry

/-- The angle C of a triangle -/
noncomputable def angle_C (t : OrderedTriangle) : ℝ := sorry

theorem triangle_function_sign (t : OrderedTriangle) :
  let f := t.a + t.b - 2 * circumradius t - 2 * inradius t
  let C := angle_C t
  (π / 3 ≤ C ∧ C < π / 2 → f > 0) ∧
  (C = π / 2 → f = 0) ∧
  (π / 2 < C ∧ C < π → f < 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_function_sign_l3608_360851


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3608_360820

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 3).sqrt / (x - 1)) ↔ x ≥ -3/2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3608_360820


namespace NUMINAMATH_CALUDE_symmetric_point_origin_specific_symmetric_point_l3608_360831

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem specific_symmetric_point : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_origin_specific_symmetric_point_l3608_360831


namespace NUMINAMATH_CALUDE_strip_covers_cube_l3608_360893

/-- A rectangular strip can cover a cube in two layers -/
theorem strip_covers_cube (strip_length : ℝ) (strip_width : ℝ) (cube_edge : ℝ) :
  strip_length = 12 →
  strip_width = 1 →
  cube_edge = 1 →
  strip_length * strip_width = 2 * 6 * cube_edge ^ 2 := by
  sorry

#check strip_covers_cube

end NUMINAMATH_CALUDE_strip_covers_cube_l3608_360893


namespace NUMINAMATH_CALUDE_triangle_nested_calc_l3608_360881

-- Define the triangle operation
def triangle (a b : ℤ) : ℤ := a^2 - 2*b

-- State the theorem
theorem triangle_nested_calc : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nested_calc_l3608_360881


namespace NUMINAMATH_CALUDE_juans_speed_l3608_360896

/-- Given a distance of 800 miles traveled in 80.0 hours, prove that the speed is 10 miles per hour -/
theorem juans_speed (distance : ℝ) (time : ℝ) (h1 : distance = 800) (h2 : time = 80) :
  distance / time = 10 := by
  sorry

end NUMINAMATH_CALUDE_juans_speed_l3608_360896
