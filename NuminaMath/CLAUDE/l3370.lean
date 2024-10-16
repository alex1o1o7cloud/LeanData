import Mathlib

namespace NUMINAMATH_CALUDE_complex_division_result_l3370_337038

/-- Given that z = a^2 - 1 + (1 + a)i where a ∈ ℝ is a purely imaginary number,
    prove that z / (2 + i) = 2/5 + 4/5 * i -/
theorem complex_division_result (a : ℝ) (i : ℂ) (z : ℂ) :
  i^2 = -1 →
  z = a^2 - 1 + (1 + a) * i →
  z.re = 0 →
  z / (2 + i) = 2/5 + 4/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_result_l3370_337038


namespace NUMINAMATH_CALUDE_chantel_final_bracelets_l3370_337037

def bracelets_made_first_period : ℕ := 5 * 2
def bracelets_given_school : ℕ := 3
def bracelets_made_second_period : ℕ := 4 * 3
def bracelets_given_soccer : ℕ := 6

theorem chantel_final_bracelets :
  bracelets_made_first_period - bracelets_given_school + bracelets_made_second_period - bracelets_given_soccer = 13 := by
  sorry

end NUMINAMATH_CALUDE_chantel_final_bracelets_l3370_337037


namespace NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l3370_337059

theorem lcm_gcd_product_10_15 :
  Nat.lcm 10 15 * Nat.gcd 10 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_10_15_l3370_337059


namespace NUMINAMATH_CALUDE_parallelogram_altitude_l3370_337071

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Lengths of sides and segments
  DC : ℝ
  EB : ℝ
  DE : ℝ
  -- Condition that ABCD is a parallelogram
  is_parallelogram : True

/-- Theorem: In a parallelogram ABCD with given conditions, DF = 7 -/
theorem parallelogram_altitude (p : Parallelogram)
  (h1 : p.DC = 15)
  (h2 : p.EB = 5)
  (h3 : p.DE = 7) :
  ∃ DF : ℝ, DF = 7 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_l3370_337071


namespace NUMINAMATH_CALUDE_sue_votes_l3370_337047

/-- Given 1000 total votes and Sue receiving 35% of the votes, prove that Sue received 350 votes. -/
theorem sue_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  ↑total_votes * sue_percentage = 350 := by
  sorry

end NUMINAMATH_CALUDE_sue_votes_l3370_337047


namespace NUMINAMATH_CALUDE_road_width_calculation_l3370_337077

/-- Represents the width of the roads in meters -/
def road_width : ℝ := 10

/-- The length of the lawn in meters -/
def lawn_length : ℝ := 80

/-- The breadth of the lawn in meters -/
def lawn_breadth : ℝ := 60

/-- The cost per square meter in Rupees -/
def cost_per_sq_m : ℝ := 5

/-- The total cost of traveling the two roads in Rupees -/
def total_cost : ℝ := 6500

theorem road_width_calculation :
  (lawn_length * road_width + lawn_breadth * road_width - road_width^2) * cost_per_sq_m = total_cost :=
sorry

end NUMINAMATH_CALUDE_road_width_calculation_l3370_337077


namespace NUMINAMATH_CALUDE_simplify_expression_l3370_337049

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3370_337049


namespace NUMINAMATH_CALUDE_print_360_pages_in_15_minutes_l3370_337027

/-- Calculates the time needed to print a given number of pages at a specific rate. -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  pages / rate

/-- Theorem stating that printing 360 pages at a rate of 24 pages per minute takes 15 minutes. -/
theorem print_360_pages_in_15_minutes :
  print_time 360 24 = 15 := by
  sorry

end NUMINAMATH_CALUDE_print_360_pages_in_15_minutes_l3370_337027


namespace NUMINAMATH_CALUDE_set_equality_l3370_337073

def S : Set (ℕ × ℕ) := {p | p.1 + p.2 = 3}

theorem set_equality : S = {(1, 2), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3370_337073


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l3370_337078

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 108 → 
  5 * b = 3 * a → 
  4 * b = 3 * c → 
  b = 27 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l3370_337078


namespace NUMINAMATH_CALUDE_complex_number_equality_l3370_337036

theorem complex_number_equality (b : ℝ) : 
  (Complex.re ((1 + b * Complex.I) / (2 + Complex.I)) = 
   Complex.im ((1 + b * Complex.I) / (2 + Complex.I))) → 
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3370_337036


namespace NUMINAMATH_CALUDE_weight_difference_l3370_337057

/-- Proves that given Robbie's weight, Patty's initial weight relative to Robbie's, and Patty's weight loss, 
    the difference between Patty's current weight and Robbie's weight is 115 pounds. -/
theorem weight_difference (robbie_weight : ℝ) (patty_initial_factor : ℝ) (patty_weight_loss : ℝ) : 
  robbie_weight = 100 → 
  patty_initial_factor = 4.5 → 
  patty_weight_loss = 235 → 
  patty_initial_factor * robbie_weight - patty_weight_loss - robbie_weight = 115 := by
  sorry


end NUMINAMATH_CALUDE_weight_difference_l3370_337057


namespace NUMINAMATH_CALUDE_perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l3370_337043

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P_on_C (P : ℝ × ℝ) : Prop := hyperbola_C P.1 P.2

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem 1: Perimeter of triangle PF₁F₂
theorem perimeter_triangle_PF₁F₂ (P : ℝ × ℝ) (h₁ : P_on_C P) (h₂ : distance P F₁ = 2 * distance P F₂) :
  distance P F₁ + distance P F₂ + distance F₁ F₂ = 28 := sorry

-- Theorem 2: Shortest distance between opposite branches
theorem shortest_distance_opposite_branches :
  ∃ (P Q : ℝ × ℝ), P_on_C P ∧ P_on_C Q ∧ 
    (∀ (R S : ℝ × ℝ), P_on_C R → P_on_C S → R.1 * S.1 < 0 → distance P Q ≤ distance R S) ∧
    distance P Q = 6 := sorry

end NUMINAMATH_CALUDE_perimeter_triangle_PF₁F₂_shortest_distance_opposite_branches_l3370_337043


namespace NUMINAMATH_CALUDE_isabellas_house_paintable_area_l3370_337031

/-- Represents the dimensions of a bedroom -/
structure BedroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable wall area in all bedrooms -/
def totalPaintableArea (dimensions : BedroomDimensions) (numBedrooms : ℕ) (nonPaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableAreaPerRoom := wallArea - nonPaintableArea
  numBedrooms * paintableAreaPerRoom

/-- Theorem stating that the total paintable wall area in Isabella's house is 1194 square feet -/
theorem isabellas_house_paintable_area :
  let dimensions : BedroomDimensions := { length := 15, width := 11, height := 9 }
  let numBedrooms : ℕ := 3
  let nonPaintableArea : ℝ := 70
  totalPaintableArea dimensions numBedrooms nonPaintableArea = 1194 := by
  sorry


end NUMINAMATH_CALUDE_isabellas_house_paintable_area_l3370_337031


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3370_337061

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a - 3) * x₁^2 - 4 * x₁ - 1 = 0 ∧ (a - 3) * x₂^2 - 4 * x₂ - 1 = 0) ↔
  (a > -1 ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3370_337061


namespace NUMINAMATH_CALUDE_eight_digit_numbers_divisibility_l3370_337044

def first_number (A B C : ℕ) : ℕ := 84000000 + A * 100000 + 53000 + B * 100 + 10 + C
def second_number (A B C D : ℕ) : ℕ := 32700000 + A * 10000 + B * 1000 + 500 + C * 10 + D

theorem eight_digit_numbers_divisibility (A B C D : ℕ) 
  (h1 : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h2 : first_number A B C % 4 = 0) 
  (h3 : second_number A B C D % 3 = 0) : 
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_divisibility_l3370_337044


namespace NUMINAMATH_CALUDE_trig_calculation_l3370_337029

theorem trig_calculation :
  Real.sin (π / 3) + Real.tan (π / 4) - Real.cos (π / 6) * Real.tan (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_calculation_l3370_337029


namespace NUMINAMATH_CALUDE_function_properties_l3370_337091

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * |x - a|

theorem function_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (a = 1/2 → ∀ x, (x ≤ -1 ∨ (1/2 ≤ x ∧ x ≤ 1)) → 
    ∀ y, y < x → f (1/2) y < f (1/2) x) ∧
  (a > 0 → (∀ x : ℝ, x ≥ 0 → f a (x - 1) ≥ 2 * f a x) ↔ 
    (Real.sqrt 6 - 2 ≤ a ∧ a ≤ 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3370_337091


namespace NUMINAMATH_CALUDE_total_caffeine_consumption_l3370_337007

/-- Calculates the total caffeine consumption given the specifications of three drinks and a pill -/
theorem total_caffeine_consumption
  (drink1_oz : ℝ)
  (drink1_caffeine : ℝ)
  (drink2_oz : ℝ)
  (drink2_caffeine_multiplier : ℝ)
  (drink3_caffeine_per_ml : ℝ)
  (drink3_ml_consumed : ℝ) :
  drink1_oz = 12 →
  drink1_caffeine = 250 →
  drink2_oz = 8 →
  drink2_caffeine_multiplier = 3 →
  drink3_caffeine_per_ml = 18 →
  drink3_ml_consumed = 150 →
  let drink2_caffeine := (drink1_caffeine / drink1_oz) * drink2_caffeine_multiplier * drink2_oz
  let drink3_caffeine := drink3_caffeine_per_ml * drink3_ml_consumed
  let pill_caffeine := drink1_caffeine + drink2_caffeine + drink3_caffeine
  drink1_caffeine + drink2_caffeine + drink3_caffeine + pill_caffeine = 6900 := by
  sorry


end NUMINAMATH_CALUDE_total_caffeine_consumption_l3370_337007


namespace NUMINAMATH_CALUDE_john_final_height_l3370_337024

/-- Calculates the final height in feet given initial height, growth rate, and duration -/
def final_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (duration : ℕ) : ℚ :=
  (initial_height + growth_rate * duration) / 12

/-- Theorem stating that given the specific conditions, the final height is 6 feet -/
theorem john_final_height :
  final_height_in_feet 66 2 3 = 6 := by sorry

end NUMINAMATH_CALUDE_john_final_height_l3370_337024


namespace NUMINAMATH_CALUDE_sqrt_of_neg_seven_squared_l3370_337054

theorem sqrt_of_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_seven_squared_l3370_337054


namespace NUMINAMATH_CALUDE_binomial_congruence_l3370_337092

theorem binomial_congruence (p a b : ℕ) (hp : Nat.Prime p) (hab : a ≥ b) (hb : b ≥ 0) :
  (Nat.choose (p * (a - b)) p) ≡ (Nat.choose a b) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l3370_337092


namespace NUMINAMATH_CALUDE_quadratic_root_square_implies_s_l3370_337089

theorem quadratic_root_square_implies_s (r s : ℝ) :
  (∃ x : ℂ, 3 * x^2 + r * x + s = 0 ∧ x^2 = 4 - 3*I) →
  s = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_implies_s_l3370_337089


namespace NUMINAMATH_CALUDE_candy_mixture_l3370_337048

theorem candy_mixture :
  ∀ (x y : ℝ),
  x + y = 100 →
  18 * x + 10 * y = 15 * 100 →
  x = 62.5 ∧ y = 37.5 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_l3370_337048


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_half_right_angle_l3370_337019

/-- Given an angle that is half of 90 degrees, prove that the degree measure of
    the supplement of its complement is 135 degrees. -/
theorem supplement_of_complement_of_half_right_angle :
  let α : ℝ := 90 / 2
  let complement_α : ℝ := 90 - α
  let supplement_complement_α : ℝ := 180 - complement_α
  supplement_complement_α = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_half_right_angle_l3370_337019


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3370_337000

/-- Given a square floor with side length s, where tiles along the diagonals
    are marked blue, prove that if there are 225 blue tiles, then the total
    number of tiles on the floor is 12769. -/
theorem square_floor_tiles (s : ℕ) : 
  (2 * s - 1 = 225) → s^2 = 12769 := by sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3370_337000


namespace NUMINAMATH_CALUDE_sandy_paint_area_l3370_337004

/-- The area to be painted on Sandy's bedroom wall -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Theorem: The area Sandy needs to paint is 131 square feet -/
theorem sandy_paint_area :
  areaToPaint 10 15 3 5 2 2 = 131 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l3370_337004


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_equality_condition_l3370_337018

theorem min_value_exponential_sum (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  2^a + 4^b ≥ Real.sqrt 2 / 2 :=
by sorry

theorem equality_condition (a b : ℝ) (h : a + 2 * b + 3 = 0) :
  ∃ (a₀ b₀ : ℝ), a₀ + 2 * b₀ + 3 = 0 ∧ 2^a₀ + 4^b₀ = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_equality_condition_l3370_337018


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_pairs_l3370_337053

theorem arithmetic_geometric_sequence_pairs : ∃ (a₁ a₂ a₃ a₄ g₁ g₂ g₃ g₄ : ℕ) 
  (b₁ b₂ b₃ b₄ h₁ h₂ h₃ h₄ : ℕ) 
  (c₁ c₂ c₃ c₄ i₁ i₂ i₃ i₄ : ℕ) 
  (d₁ d₂ d₃ d₄ j₁ j₂ j₃ j₄ : ℕ),
  (a₁ = 1 ∧ g₁ = 1 ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ g₁ < g₂ ∧ g₂ < g₃ ∧ g₃ < g₄ ∧
   a₁ + a₂ + a₃ + a₄ = g₁ + g₂ + g₃ + g₄) ∧
  (b₁ = 1 ∧ h₁ = 1 ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
   b₁ + b₂ + b₃ + b₄ = h₁ + h₂ + h₃ + h₄) ∧
  (c₁ = 1 ∧ i₁ = 1 ∧ c₁ < c₂ ∧ c₂ < c₃ ∧ c₃ < c₄ ∧ i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧
   c₁ + c₂ + c₃ + c₄ = i₁ + i₂ + i₃ + i₄) ∧
  (d₁ = 1 ∧ j₁ = 1 ∧ d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ j₁ < j₂ ∧ j₂ < j₃ ∧ j₃ < j₄ ∧
   d₁ + d₂ + d₃ + d₄ = j₁ + j₂ + j₃ + j₄) ∧
  (a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃) ∧
  (b₂ - b₁ = b₃ - b₂ ∧ b₃ - b₂ = b₄ - b₃) ∧
  (c₂ - c₁ = c₃ - c₂ ∧ c₃ - c₂ = c₄ - c₃) ∧
  (d₂ - d₁ = d₃ - d₂ ∧ d₃ - d₂ = d₄ - d₃) ∧
  (g₂ / g₁ = g₃ / g₂ ∧ g₃ / g₂ = g₄ / g₃) ∧
  (h₂ / h₁ = h₃ / h₂ ∧ h₃ / h₂ = h₄ / h₃) ∧
  (i₂ / i₁ = i₃ / i₂ ∧ i₃ / i₂ = i₄ / i₃) ∧
  (j₂ / j₁ = j₃ / j₂ ∧ j₃ / j₂ = j₄ / j₃) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_pairs_l3370_337053


namespace NUMINAMATH_CALUDE_exist_same_color_perfect_square_diff_l3370_337010

/-- A coloring of integers using three colors. -/
def Coloring := ℤ → Fin 3

/-- Predicate to check if a number is a perfect square. -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- Main theorem: For any coloring of integers using three colors,
    there exist two different integers of the same color
    whose difference is a perfect square. -/
theorem exist_same_color_perfect_square_diff (c : Coloring) :
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ is_perfect_square (a - b) := by
  sorry


end NUMINAMATH_CALUDE_exist_same_color_perfect_square_diff_l3370_337010


namespace NUMINAMATH_CALUDE_inequality_proof_l3370_337042

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3370_337042


namespace NUMINAMATH_CALUDE_stone_68_is_10_l3370_337095

/-- The number of stones in the circle -/
def n : ℕ := 15

/-- The length of a full cycle (clockwise + counterclockwise) -/
def cycle_length : ℕ := n + (n - 1)

/-- The stone number corresponding to a given count -/
def stone_number (count : ℕ) : ℕ :=
  let effective_count := count % cycle_length
  if effective_count ≤ n then effective_count else n - (effective_count - n)

theorem stone_68_is_10 : stone_number 68 = 10 := by sorry

end NUMINAMATH_CALUDE_stone_68_is_10_l3370_337095


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3370_337062

theorem infinite_geometric_series_first_term
  (a r : ℝ)
  (h1 : 0 ≤ r ∧ r < 1)  -- Condition for convergence of infinite geometric series
  (h2 : a / (1 - r) = 15)  -- Sum of the series
  (h3 : a^2 / (1 - r^2) = 45)  -- Sum of the squares of the terms
  : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3370_337062


namespace NUMINAMATH_CALUDE_pizza_area_increase_l3370_337040

theorem pizza_area_increase : 
  let r1 : ℝ := 2
  let r2 : ℝ := 5
  let area1 := π * r1^2
  let area2 := π * r2^2
  (area2 - area1) / area1 * 100 = 525 :=
by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l3370_337040


namespace NUMINAMATH_CALUDE_bin_drawing_probability_l3370_337012

def bin_probability (black white : ℕ) : ℚ :=
  let total := black + white
  let favorable := (black.choose 2 * white) + (black * white.choose 2)
  favorable / total.choose 3

theorem bin_drawing_probability :
  bin_probability 10 4 = 60 / 91 := by
  sorry

end NUMINAMATH_CALUDE_bin_drawing_probability_l3370_337012


namespace NUMINAMATH_CALUDE_forty_men_handshakes_l3370_337064

/-- The maximum number of handshakes without cyclic handshakes for n people -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 40 men, the maximum number of handshakes without cyclic handshakes is 780 -/
theorem forty_men_handshakes : max_handshakes 40 = 780 := by
  sorry

end NUMINAMATH_CALUDE_forty_men_handshakes_l3370_337064


namespace NUMINAMATH_CALUDE_roses_per_friend_l3370_337088

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- Prove that each dancer friend gave Bella 2 roses -/
theorem roses_per_friend (
  parents_roses : ℕ) 
  (dancer_friends : ℕ) 
  (total_roses : ℕ) 
  (h1 : parents_roses = 2 * dozen)
  (h2 : dancer_friends = 10)
  (h3 : total_roses = 44) :
  (total_roses - parents_roses) / dancer_friends = 2 := by
  sorry

#check roses_per_friend

end NUMINAMATH_CALUDE_roses_per_friend_l3370_337088


namespace NUMINAMATH_CALUDE_total_charcoal_needed_l3370_337074

-- Define the ratios and water amounts for each batch
def batch1_ratio : ℚ := 2 / 30
def batch1_water : ℚ := 900

def batch2_ratio : ℚ := 3 / 50
def batch2_water : ℚ := 1150

def batch3_ratio : ℚ := 4 / 80
def batch3_water : ℚ := 1615

def batch4_ratio : ℚ := 2.3 / 25
def batch4_water : ℚ := 675

def batch5_ratio : ℚ := 5.5 / 115
def batch5_water : ℚ := 1930

-- Function to calculate charcoal needed for a batch
def charcoal_needed (ratio : ℚ) (water : ℚ) : ℚ :=
  ratio * water

-- Theorem stating the total charcoal needed is 363.28 grams
theorem total_charcoal_needed :
  (charcoal_needed batch1_ratio batch1_water +
   charcoal_needed batch2_ratio batch2_water +
   charcoal_needed batch3_ratio batch3_water +
   charcoal_needed batch4_ratio batch4_water +
   charcoal_needed batch5_ratio batch5_water) = 363.28 := by
  sorry

end NUMINAMATH_CALUDE_total_charcoal_needed_l3370_337074


namespace NUMINAMATH_CALUDE_inequality_proof_l3370_337028

theorem inequality_proof (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3370_337028


namespace NUMINAMATH_CALUDE_floor_sqrt_72_l3370_337020

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_72_l3370_337020


namespace NUMINAMATH_CALUDE_savings_is_six_dollars_l3370_337035

-- Define the number of notebooks
def num_notebooks : ℕ := 8

-- Define the original price per notebook
def original_price : ℚ := 3

-- Define the discount rate
def discount_rate : ℚ := 1/4

-- Define the function to calculate savings
def calculate_savings (n : ℕ) (p : ℚ) (d : ℚ) : ℚ :=
  n * p * d

-- Theorem stating that the savings is $6.00
theorem savings_is_six_dollars :
  calculate_savings num_notebooks original_price discount_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_savings_is_six_dollars_l3370_337035


namespace NUMINAMATH_CALUDE_floor_difference_equals_eight_l3370_337067

theorem floor_difference_equals_eight :
  ⌊(101^3 : ℝ) / (99 * 100) - (99^3 : ℝ) / (100 * 101)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_difference_equals_eight_l3370_337067


namespace NUMINAMATH_CALUDE_max_sum_of_digits_of_sum_l3370_337017

/-- Represents a three-digit positive integer with distinct digits from 1 to 9 -/
structure ThreeDigitNumber :=
  (value : ℕ)
  (is_three_digit : 100 ≤ value ∧ value ≤ 999)
  (distinct_digits : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃)
  (digits_range : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → 1 ≤ d₁ ∧ d₁ ≤ 9 ∧ 1 ≤ d₂ ∧ d₂ ≤ 9 ∧ 1 ≤ d₃ ∧ d₃ ≤ 9)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem max_sum_of_digits_of_sum (a b : ThreeDigitNumber) :
  let S := a.value + b.value
  100 ≤ S ∧ S ≤ 999 →
  sum_of_digits S ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_of_sum_l3370_337017


namespace NUMINAMATH_CALUDE_saree_price_l3370_337056

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 331.2)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.08) : 
  ∃ original_price : ℝ, 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
sorry

end NUMINAMATH_CALUDE_saree_price_l3370_337056


namespace NUMINAMATH_CALUDE_fifth_term_geometric_progression_l3370_337051

theorem fifth_term_geometric_progression :
  let x : ℝ := -1 + Real.sqrt 5
  let r : ℝ := (1 + Real.sqrt 5) / (-1 + Real.sqrt 5)
  let a₁ : ℝ := x
  let a₂ : ℝ := x + 2
  let a₃ : ℝ := 2 * x + 6
  let a₅ : ℝ := r^4 * a₁
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) →
  a₅ = ((1 + Real.sqrt 5) / (-1 + Real.sqrt 5)) * (4 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_progression_l3370_337051


namespace NUMINAMATH_CALUDE_repeated_root_fraction_equation_l3370_337096

theorem repeated_root_fraction_equation (x m : ℝ) : 
  (∃ x, (x / (x - 3) + 1 = m / (x - 3)) ∧ 
        (∀ y, y ≠ x → y / (y - 3) + 1 ≠ m / (y - 3))) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_repeated_root_fraction_equation_l3370_337096


namespace NUMINAMATH_CALUDE_function_inequality_implies_constant_l3370_337033

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x^2+y) for all x, y ∈ ℝ is constant. -/
theorem function_inequality_implies_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_constant_l3370_337033


namespace NUMINAMATH_CALUDE_prob_not_at_ends_eight_chairs_l3370_337032

/-- The number of chairs in the row -/
def n : ℕ := 8

/-- The probability of two people not sitting at either end when randomly choosing seats in a row of n chairs -/
def prob_not_at_ends (n : ℕ) : ℚ :=
  1 - (2 + 2 * (n - 2)) / (n.choose 2)

theorem prob_not_at_ends_eight_chairs :
  prob_not_at_ends n = 3/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_at_ends_eight_chairs_l3370_337032


namespace NUMINAMATH_CALUDE_max_value_theorem_l3370_337003

/-- The sum of the first m positive even numbers -/
def sumEvenNumbers (m : ℕ) : ℕ := m * (m + 1)

/-- The sum of the first n positive odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The constraint that the sum of m even numbers and n odd numbers is 1987 -/
def constraint (m n : ℕ) : Prop := sumEvenNumbers m + sumOddNumbers n = 1987

/-- The objective function to be maximized -/
def objective (m n : ℕ) : ℕ := 3 * m + 4 * n

theorem max_value_theorem :
  ∃ m n : ℕ, constraint m n ∧ 
    ∀ m' n' : ℕ, constraint m' n' → objective m' n' ≤ objective m n ∧
    objective m n = 219 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3370_337003


namespace NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l3370_337022

theorem equal_number_of_boys_and_girls 
  (m : ℕ) (d : ℕ) (M : ℝ) (D : ℝ) 
  (h1 : M / m ≠ D / d) 
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : 
  m = d := by sorry

end NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l3370_337022


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l3370_337066

theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (income : ℚ) : 
  tips = (3 : ℚ) / 4 * salary →
  income = salary + tips →
  tips / income = (3 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l3370_337066


namespace NUMINAMATH_CALUDE_ratio_problem_l3370_337046

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 5)
  (h3 : s / t = 1 / 4)
  (h4 : u / t = 3)
  : u / q = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3370_337046


namespace NUMINAMATH_CALUDE_hyperbola_condition_l3370_337052

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 5) - y^2 / (k + 2) = 1 ∧ 
  (k - 5 > 0 ∧ k + 2 > 0)

-- State the theorem
theorem hyperbola_condition (k : ℝ) :
  (is_hyperbola k → k > 5) ∧ 
  ¬(k > 5 → is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l3370_337052


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l3370_337011

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 101 in base 2. -/
def binary_101 : List Bool := [true, false, true]

theorem binary_101_to_decimal :
  binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l3370_337011


namespace NUMINAMATH_CALUDE_function_equality_condition_l3370_337021

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a x : ℝ) : ℝ := a*x - 1

-- Define the domain interval
def I : Set ℝ := Set.Icc (-1) 2

-- Define the theorem
theorem function_equality_condition (a : ℝ) : 
  (∀ x₁ ∈ I, ∃ x₂ ∈ I, f x₁ = g a x₂) ↔ 
  (a ≤ -4 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_function_equality_condition_l3370_337021


namespace NUMINAMATH_CALUDE_great_pyramid_height_l3370_337026

theorem great_pyramid_height (h w : ℝ) : 
  h > 500 → 
  w = h + 234 → 
  h + w = 1274 → 
  h - 500 = 20 := by
sorry

end NUMINAMATH_CALUDE_great_pyramid_height_l3370_337026


namespace NUMINAMATH_CALUDE_rectangle_area_l3370_337041

theorem rectangle_area (length width : ℝ) (h_diagonal : length^2 + width^2 = 41) 
  (h_perimeter : 2 * (length + width) = 18) : length * width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3370_337041


namespace NUMINAMATH_CALUDE_abs_T_equals_128_sqrt_2_l3370_337058

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^15 - (1 - i)^15

-- Theorem statement
theorem abs_T_equals_128_sqrt_2 : Complex.abs T = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_T_equals_128_sqrt_2_l3370_337058


namespace NUMINAMATH_CALUDE_ball_max_height_l3370_337079

/-- The height function of the ball's parabolic path -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

/-- Theorem stating that the maximum height of the ball is 116 feet -/
theorem ball_max_height : 
  ∃ (max : ℝ), max = 116 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3370_337079


namespace NUMINAMATH_CALUDE_locus_of_R_l3370_337070

-- Define the square ABCD
structure Square :=
  (A B C D : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the perimeter of a square
def perimeter (s : Square) : Set Point := sorry

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (P Q R : Point)

-- Define a rotation around a point
def rotate (center : Point) (angle : ℝ) (p : Point) : Point := sorry

-- Define the theorem
theorem locus_of_R (ABCD : Square) (Q : Point) :
  ∀ P ∈ perimeter ABCD,
  Q ∉ perimeter ABCD →
  ∃ (PQR : EquilateralTriangle),
  PQR.P = P ∧ PQR.Q = Q →
  ∃ (A₁B₁C₁D₁ A₂B₂C₂D₂ : Square),
  A₁B₁C₁D₁ = Square.mk (rotate Q (π/3) ABCD.A) (rotate Q (π/3) ABCD.B) (rotate Q (π/3) ABCD.C) (rotate Q (π/3) ABCD.D) ∧
  A₂B₂C₂D₂ = Square.mk (rotate Q (-π/3) ABCD.A) (rotate Q (-π/3) ABCD.B) (rotate Q (-π/3) ABCD.C) (rotate Q (-π/3) ABCD.D) ∧
  PQR.R ∈ perimeter A₁B₁C₁D₁ ∪ perimeter A₂B₂C₂D₂ :=
by sorry

end NUMINAMATH_CALUDE_locus_of_R_l3370_337070


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l3370_337016

theorem fraction_sum_product_equality (a b c : ℝ) 
  (h1 : 1 + b * c ≠ 0) (h2 : 1 + c * a ≠ 0) (h3 : 1 + a * b ≠ 0) : 
  (b - c) / (1 + b * c) + (c - a) / (1 + c * a) + (a - b) / (1 + a * b) = 
  (b - c) / (1 + b * c) * (c - a) / (1 + c * a) * (a - b) / (1 + a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l3370_337016


namespace NUMINAMATH_CALUDE_total_seashells_l3370_337013

-- Define the number of seashells found by Sam
def sam_shells : ℕ := 18

-- Define the number of seashells found by Mary
def mary_shells : ℕ := 47

-- Theorem stating the total number of seashells found
theorem total_seashells : sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l3370_337013


namespace NUMINAMATH_CALUDE_clock_hands_right_angle_count_l3370_337083

/-- The number of times clock hands form a right angle in a 12-hour period -/
def right_angles_12h : ℕ := 22

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days we're considering -/
def days : ℕ := 2

theorem clock_hands_right_angle_count :
  (right_angles_12h * hours_per_day * days) / 12 = 88 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_right_angle_count_l3370_337083


namespace NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l3370_337065

/-- The area of a right triangle formed by the diagonal of a rectangle. -/
theorem rectangle_diagonal_triangle_area
  (length width : ℝ)
  (h_length : length = 35)
  (h_width : width = 48) :
  (1 / 2) * length * width = 840 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_triangle_area_l3370_337065


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l3370_337075

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l3370_337075


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l3370_337068

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62) 
  (h2 : pies = 6) 
  (h3 : apples_per_pie = 9) :
  initial_apples - pies * apples_per_pie = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l3370_337068


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3370_337014

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  bc_length : BC = 32
  ap_length : AP = 24
  dq_length : DQ = 18
  ab_length : AB = 29
  cd_length : CD = 35

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem: The perimeter of the trapezoid is 170 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 170 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3370_337014


namespace NUMINAMATH_CALUDE_original_price_calculation_l3370_337080

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 220)
  (h2 : profit_percentage = 0.1) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage) ∧ 
    original_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3370_337080


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1337_l3370_337030

theorem largest_prime_factor_of_1337 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1337 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1337 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1337_l3370_337030


namespace NUMINAMATH_CALUDE_percentage_reduction_l3370_337055

theorem percentage_reduction (P : ℝ) : (85 * P / 100) - 11 = 23 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_reduction_l3370_337055


namespace NUMINAMATH_CALUDE_distance_to_work_is_18_l3370_337072

/-- The distance Esther drives to work -/
def distance_to_work : ℝ := 18

/-- The average speed to work in miles per hour -/
def speed_to_work : ℝ := 45

/-- The average speed from work in miles per hour -/
def speed_from_work : ℝ := 30

/-- The total commute time in hours -/
def total_commute_time : ℝ := 1

/-- Theorem stating that the distance to work is 18 miles given the conditions -/
theorem distance_to_work_is_18 :
  (distance_to_work / speed_to_work) + (distance_to_work / speed_from_work) = total_commute_time :=
by sorry

end NUMINAMATH_CALUDE_distance_to_work_is_18_l3370_337072


namespace NUMINAMATH_CALUDE_line_always_intersects_hyperbola_iff_k_in_range_l3370_337069

/-- A line intersects a hyperbola if their equations have a common solution -/
def intersects (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1

/-- The main theorem: if a line always intersects the hyperbola, then k is in the open interval (-√2/2, √2/2) -/
theorem line_always_intersects_hyperbola_iff_k_in_range (k : ℝ) :
  (∀ b : ℝ, intersects k b) ↔ -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_always_intersects_hyperbola_iff_k_in_range_l3370_337069


namespace NUMINAMATH_CALUDE_transform_to_zero_y_l3370_337086

-- Define the set M
def M : Set (ℤ × ℤ) := Set.univ

-- Define transformation S
def S (p : ℤ × ℤ) : ℤ × ℤ := (p.1 + p.2, p.2)

-- Define transformation T
def T (p : ℤ × ℤ) : ℤ × ℤ := (-p.2, p.1)

-- Define the type of transformations
inductive Transform
| S : Transform
| T : Transform

-- Define the application of a sequence of transformations
def applyTransforms : List Transform → ℤ × ℤ → ℤ × ℤ
| [], p => p
| (Transform.S :: ts), p => applyTransforms ts (S p)
| (Transform.T :: ts), p => applyTransforms ts (T p)

-- The main theorem
theorem transform_to_zero_y (p : ℤ × ℤ) : 
  ∃ (ts : List Transform) (g : ℤ), applyTransforms ts p = (g, 0) := by
  sorry


end NUMINAMATH_CALUDE_transform_to_zero_y_l3370_337086


namespace NUMINAMATH_CALUDE_egg_count_l3370_337063

theorem egg_count (initial_eggs : ℕ) (added_eggs : ℕ) : 
  initial_eggs = 7 → added_eggs = 4 → initial_eggs + added_eggs = 11 := by
  sorry

#check egg_count

end NUMINAMATH_CALUDE_egg_count_l3370_337063


namespace NUMINAMATH_CALUDE_vectors_same_direction_l3370_337087

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

-- Define points A, B, C
variable (A B C : V)

-- Define the vectors
def AB : V := B - A
def AC : V := C - A
def BC : V := C - B

-- Define the theorem
theorem vectors_same_direction (h : ‖AB A B‖ = ‖AC A C‖ + ‖BC B C‖) :
  ∃ (k : ℝ), k > 0 ∧ AC A C = k • (BC B C) := by
  sorry

end NUMINAMATH_CALUDE_vectors_same_direction_l3370_337087


namespace NUMINAMATH_CALUDE_red_light_time_proof_l3370_337025

/-- Represents the time added by each red light -/
def time_per_red_light : ℕ := sorry

/-- Time for the first route with all green lights -/
def green_route_time : ℕ := 10

/-- Time for the second route -/
def second_route_time : ℕ := 14

/-- Number of stoplights on the first route -/
def num_stoplights : ℕ := 3

theorem red_light_time_proof :
  (green_route_time + num_stoplights * time_per_red_light = second_route_time + 5) ∧
  (time_per_red_light = 3) := by sorry

end NUMINAMATH_CALUDE_red_light_time_proof_l3370_337025


namespace NUMINAMATH_CALUDE_one_meeting_l3370_337085

/-- Represents the movement and meeting of a jogger and an aid vehicle --/
structure JoggerVehicleSystem where
  jogger_speed : ℝ
  vehicle_speed : ℝ
  station_distance : ℝ
  vehicle_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between the jogger and the vehicle --/
def number_of_meetings (sys : JoggerVehicleSystem) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : JoggerVehicleSystem :=
  { jogger_speed := 6
  , vehicle_speed := 12
  , station_distance := 300
  , vehicle_stop_time := 20
  , initial_distance := 300 }

/-- Theorem stating that in the given scenario, there is exactly one meeting --/
theorem one_meeting :
  number_of_meetings problem_scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_one_meeting_l3370_337085


namespace NUMINAMATH_CALUDE_partition_product_property_l3370_337001

theorem partition_product_property (S : Finset ℕ) (h : S = Finset.range (3^5 - 2) ∪ {3^5}) :
  ∀ (A B : Finset ℕ), A ∪ B = S → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a * b = c) :=
by sorry

end NUMINAMATH_CALUDE_partition_product_property_l3370_337001


namespace NUMINAMATH_CALUDE_janets_height_l3370_337090

/-- Given the heights of various people, prove Janet's height --/
theorem janets_height :
  ∀ (ruby pablo charlene janet : ℝ),
  ruby = pablo - 2 →
  pablo = charlene + 70 →
  charlene = 2 * janet →
  ruby = 192 →
  janet = 62 := by
sorry

end NUMINAMATH_CALUDE_janets_height_l3370_337090


namespace NUMINAMATH_CALUDE_perimeter_of_externally_touching_circles_l3370_337093

/-- Given two externally touching circles with radii in the ratio 3:1 and a common external tangent
    of length 6√3, the perimeter of the figure formed by the external tangents and the external
    parts of the circles is 14π + 12√3. -/
theorem perimeter_of_externally_touching_circles (r R : ℝ) (h1 : R = 3 * r) 
    (h2 : r > 0) (h3 : 6 * Real.sqrt 3 = 2 * r * Real.sqrt 3) : 
    2 * (6 * Real.sqrt 3) + 2 * π * r * (1/3) + 2 * π * R * (2/3) = 14 * π + 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_externally_touching_circles_l3370_337093


namespace NUMINAMATH_CALUDE_floor_of_e_l3370_337094

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l3370_337094


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3370_337023

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x - 3 ≥ 0 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3370_337023


namespace NUMINAMATH_CALUDE_polynomial_value_l3370_337050

theorem polynomial_value (x : ℝ) (h : x^2 + 3*x = 1) : 3*x^2 + 9*x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3370_337050


namespace NUMINAMATH_CALUDE_construct_75_degree_angle_l3370_337082

/-- Given an angle of 19°, it is possible to construct an angle of 75°. -/
theorem construct_75_degree_angle (angle : ℝ) (h : angle = 19) : 
  ∃ (constructed_angle : ℝ), constructed_angle = 75 := by
sorry

end NUMINAMATH_CALUDE_construct_75_degree_angle_l3370_337082


namespace NUMINAMATH_CALUDE_inequality_theorem_l3370_337008

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, DifferentiableAt ℝ (deriv f) x) ∧
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  ∀ x ≥ 0, deriv (deriv f) x - 5 * deriv f x + 6 * f x ≥ 0

/-- The main theorem -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x ≥ 0, f x ≥ 3 * Real.exp (2 * x) - 2 * Real.exp (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3370_337008


namespace NUMINAMATH_CALUDE_childs_movie_ticket_cost_l3370_337002

/-- Proves that the cost of a child's movie ticket is $3 given the specified conditions. -/
theorem childs_movie_ticket_cost (total_money : ℚ) (adult_ticket_cost : ℚ) (num_children : ℕ) 
  (h1 : total_money = 35)
  (h2 : adult_ticket_cost = 8)
  (h3 : num_children = 9) :
  ∃ (child_ticket_cost : ℚ), 
    child_ticket_cost = 3 ∧ 
    adult_ticket_cost + num_children * child_ticket_cost ≤ total_money :=
by sorry

end NUMINAMATH_CALUDE_childs_movie_ticket_cost_l3370_337002


namespace NUMINAMATH_CALUDE_system_condition_l3370_337081

theorem system_condition : 
  (∀ x y : ℝ, x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
  (∃ x y : ℝ, x + y > 5 ∧ x * y > 6 ∧ ¬(x > 2 ∧ y > 3)) := by
sorry

end NUMINAMATH_CALUDE_system_condition_l3370_337081


namespace NUMINAMATH_CALUDE_car_expense_difference_l3370_337076

/-- The difference in car expenses between Alberto and Samara -/
def expense_difference (alberto_expense : ℕ) (samara_oil : ℕ) (samara_tires : ℕ) (samara_detailing : ℕ) : ℕ :=
  alberto_expense - (samara_oil + samara_tires + samara_detailing)

/-- Theorem stating the difference in car expenses between Alberto and Samara -/
theorem car_expense_difference :
  expense_difference 2457 25 467 79 = 1886 := by
  sorry

end NUMINAMATH_CALUDE_car_expense_difference_l3370_337076


namespace NUMINAMATH_CALUDE_complex_eighth_power_sum_l3370_337005

theorem complex_eighth_power_sum : (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 8 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_eighth_power_sum_l3370_337005


namespace NUMINAMATH_CALUDE_johns_investment_l3370_337098

theorem johns_investment (total_interest rate1 rate_difference investment1 : ℝ) 
  (h1 : total_interest = 1282)
  (h2 : rate1 = 0.11)
  (h3 : rate_difference = 0.015)
  (h4 : investment1 = 4000) : 
  ∃ investment2 : ℝ, 
    investment2 = 6736 ∧ 
    total_interest = investment1 * rate1 + investment2 * (rate1 + rate_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_johns_investment_l3370_337098


namespace NUMINAMATH_CALUDE_janeth_round_balloons_l3370_337084

/-- The number of bags of round balloons Janeth bought -/
def R : ℕ := sorry

/-- The number of balloons in each bag of round balloons -/
def round_per_bag : ℕ := 20

/-- The number of bags of long balloons Janeth bought -/
def long_bags : ℕ := 4

/-- The number of balloons in each bag of long balloons -/
def long_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst : ℕ := 5

/-- The total number of balloons left -/
def total_left : ℕ := 215

theorem janeth_round_balloons :
  R * round_per_bag + long_bags * long_per_bag - burst = total_left ∧ R = 5 := by
  sorry

end NUMINAMATH_CALUDE_janeth_round_balloons_l3370_337084


namespace NUMINAMATH_CALUDE_ninth_term_is_seven_l3370_337034

/-- A sequence where each term is 1/2 more than the previous term -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = a n + 1/2

/-- The 9th term of the arithmetic sequence is 7 -/
theorem ninth_term_is_seven (a : ℕ → ℚ) (h : arithmeticSequence a) : a 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_seven_l3370_337034


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3370_337015

-- Define the complex number z
def z (a : ℝ) : ℂ := (2 - Complex.I) * (2 + a * Complex.I)

-- Define the condition for z to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem complex_in_fourth_quadrant :
  ∃ a : ℝ, in_fourth_quadrant (z a) ∧ a = -2 :=
sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l3370_337015


namespace NUMINAMATH_CALUDE_wild_weatherman_proof_l3370_337045

structure TextContent where
  content : String

structure WritingStyle where
  style : String

structure CareerAspiration where
  aspiration : String

structure WeatherForecastingTechnology where
  accuracy : String
  perfection : Bool

structure WeatherScienceStudy where
  name : String

def text_content : TextContent := ⟨"[Full text content]"⟩

theorem wild_weatherman_proof 
  (text : TextContent) 
  (writing_style : WritingStyle) 
  (sam_aspiration : CareerAspiration) 
  (weather_tech : WeatherForecastingTechnology) 
  (weather_study : WeatherScienceStudy) : 
  writing_style.style = "interview" ∧ 
  sam_aspiration.aspiration = "news reporter" ∧ 
  weather_tech.accuracy = "more exact" ∧ 
  ¬weather_tech.perfection ∧
  weather_study.name = "meteorology" := by
  sorry

#check wild_weatherman_proof text_content

end NUMINAMATH_CALUDE_wild_weatherman_proof_l3370_337045


namespace NUMINAMATH_CALUDE_min_qr_length_l3370_337039

/-- Given two triangles PQR and SQR sharing side QR, with known side lengths,
    prove that the least possible integral length of QR is 15 cm. -/
theorem min_qr_length (pq pr sr sq : ℝ) (h_pq : pq = 7)
                      (h_pr : pr = 15) (h_sr : sr = 10) (h_sq : sq = 25) :
  ∀ qr : ℝ, qr > pr - pq ∧ qr > sq - sr → qr ≥ 15 := by sorry

end NUMINAMATH_CALUDE_min_qr_length_l3370_337039


namespace NUMINAMATH_CALUDE_possible_b_values_l3370_337099

/-- The cubic polynomial p(x) = x^3 + ax + b -/
def p (a b x : ℝ) : ℝ := x^3 + a*x + b

/-- The cubic polynomial q(x) = x^3 + ax + b + 150 -/
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 150

/-- Theorem stating the possible values of b given the conditions -/
theorem possible_b_values (a b r s : ℝ) : 
  (p a b r = 0 ∧ p a b s = 0) →  -- r and s are roots of p(x)
  (q a b (r+3) = 0 ∧ q a b (s-5) = 0) →  -- r+3 and s-5 are roots of q(x)
  b = 0 ∨ b = 12082 := by
sorry

end NUMINAMATH_CALUDE_possible_b_values_l3370_337099


namespace NUMINAMATH_CALUDE_k_squared_minus_3k_minus_4_l3370_337009

theorem k_squared_minus_3k_minus_4 (a b c d k : ℝ) :
  (2 * a / (b + c + d) = k) ∧
  (2 * b / (a + c + d) = k) ∧
  (2 * c / (a + b + d) = k) ∧
  (2 * d / (a + b + c) = k) →
  (k^2 - 3*k - 4 = -50/9) ∨ (k^2 - 3*k - 4 = 6) :=
by sorry

end NUMINAMATH_CALUDE_k_squared_minus_3k_minus_4_l3370_337009


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3370_337006

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The sum of the 6th, 7th, and 8th terms equals 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3370_337006


namespace NUMINAMATH_CALUDE_max_value_complex_fraction_l3370_337060

theorem max_value_complex_fraction (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (max_val : ℝ), max_val = (2 * Real.sqrt 5) / 3 ∧
  ∀ (w : ℂ), Complex.abs w = 1 →
    Complex.abs ((w + Complex.I) / (w + 2)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_fraction_l3370_337060


namespace NUMINAMATH_CALUDE_johns_presents_worth_l3370_337097

/-- The total worth of John's presents to his fiancee -/
def total_worth (ring_cost car_cost brace_cost : ℕ) : ℕ :=
  ring_cost + car_cost + brace_cost

/-- Theorem stating the total worth of John's presents -/
theorem johns_presents_worth :
  ∃ (ring_cost car_cost brace_cost : ℕ),
    ring_cost = 4000 ∧
    car_cost = 2000 ∧
    brace_cost = 2 * ring_cost ∧
    total_worth ring_cost car_cost brace_cost = 14000 := by
  sorry

end NUMINAMATH_CALUDE_johns_presents_worth_l3370_337097
