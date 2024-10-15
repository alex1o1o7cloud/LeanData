import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_solution_l3260_326027

theorem linear_equation_solution (x₁ y₁ x₂ y₂ : ℤ) :
  x₁ = 1 ∧ y₁ = -2 ∧ x₂ = -1 ∧ y₂ = -4 →
  x₁ - y₁ = 3 ∧ x₂ - y₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3260_326027


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_common_chord_l3260_326054

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the polar equation of the perpendicular bisector
def perpendicular_bisector (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - Real.pi/4) = Real.sqrt 2

-- Theorem statement
theorem perpendicular_bisector_of_common_chord :
  ∀ (x y ρ θ : ℝ), C1 x y → C2 x y →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  perpendicular_bisector ρ θ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_common_chord_l3260_326054


namespace NUMINAMATH_CALUDE_chocolate_bar_revenue_increase_l3260_326083

theorem chocolate_bar_revenue_increase 
  (original_weight : ℝ) (original_price : ℝ) 
  (new_weight : ℝ) (new_price : ℝ) :
  original_weight = 400 →
  original_price = 150 →
  new_weight = 300 →
  new_price = 180 →
  let original_revenue := original_price / original_weight
  let new_revenue := new_price / new_weight
  let revenue_increase := (new_revenue - original_revenue) / original_revenue
  revenue_increase = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_revenue_increase_l3260_326083


namespace NUMINAMATH_CALUDE_aubree_animal_count_l3260_326079

def total_animals (initial_beavers : ℕ) (initial_chipmunks : ℕ) : ℕ :=
  let final_beavers := 2 * initial_beavers
  let final_chipmunks := initial_chipmunks - 10
  initial_beavers + initial_chipmunks + final_beavers + final_chipmunks

theorem aubree_animal_count :
  total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aubree_animal_count_l3260_326079


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3260_326011

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3260_326011


namespace NUMINAMATH_CALUDE_five_heads_in_nine_flips_l3260_326096

/-- The probability of getting exactly k heads when flipping n fair coins -/
def coinFlipProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 5 heads when flipping 9 fair coins is 63/256 -/
theorem five_heads_in_nine_flips :
  coinFlipProbability 9 5 = 63 / 256 := by
  sorry

end NUMINAMATH_CALUDE_five_heads_in_nine_flips_l3260_326096


namespace NUMINAMATH_CALUDE_square_root_identity_specific_square_roots_l3260_326063

theorem square_root_identity (n : ℕ) :
  Real.sqrt (1 - (2 * n + 1) / ((n + 1) ^ 2)) = n / (n + 1) :=
sorry

theorem specific_square_roots :
  Real.sqrt (1 - 9 / 25) = 4 / 5 ∧ Real.sqrt (1 - 15 / 64) = 7 / 8 :=
sorry

end NUMINAMATH_CALUDE_square_root_identity_specific_square_roots_l3260_326063


namespace NUMINAMATH_CALUDE_division_problem_l3260_326012

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 127 → 
  divisor = 25 → 
  remainder = 2 → 
  dividend = divisor * quotient + remainder → 
  quotient = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3260_326012


namespace NUMINAMATH_CALUDE_even_function_sum_l3260_326021

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property of an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc b 3, f a x = f a (-x)) →
  is_even (f a) →
  a + b = -3 :=
sorry

end NUMINAMATH_CALUDE_even_function_sum_l3260_326021


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l3260_326050

/-- The probability of getting exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 3 heads in 8 flips of an unfair coin -/
theorem unfair_coin_probability : 
  binomial_probability 8 3 (1/3) = 1792/6561 := by
  sorry


end NUMINAMATH_CALUDE_unfair_coin_probability_l3260_326050


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3260_326092

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧ 
  (∃ y : ℝ, y < 1 ∧ |y + 1| + |y - 1| = 2 * |y|) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3260_326092


namespace NUMINAMATH_CALUDE_lcm_14_18_20_l3260_326019

theorem lcm_14_18_20 : Nat.lcm 14 (Nat.lcm 18 20) = 1260 := by sorry

end NUMINAMATH_CALUDE_lcm_14_18_20_l3260_326019


namespace NUMINAMATH_CALUDE_exact_arrival_speed_l3260_326040

theorem exact_arrival_speed 
  (d : ℝ) (t : ℝ) 
  (h1 : d = 30 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/12)) : 
  d / t = 42 := by
sorry

end NUMINAMATH_CALUDE_exact_arrival_speed_l3260_326040


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3260_326005

theorem unique_solution_quadratic_inequality (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3260_326005


namespace NUMINAMATH_CALUDE_plane_line_relationship_l3260_326091

-- Define the types for planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (a : Set (ℝ × ℝ × ℝ))

-- Define the perpendicular relation
def perpendicular (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the parallel relation
def parallel (S T : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the contained relation
def contained (L P : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_line_relationship 
  (h1 : perpendicular α β) 
  (h2 : perpendicular a β) : 
  contained a α ∨ parallel a α := by sorry

end NUMINAMATH_CALUDE_plane_line_relationship_l3260_326091


namespace NUMINAMATH_CALUDE_divisibility_of_prime_square_minus_one_l3260_326058

theorem divisibility_of_prime_square_minus_one (p : ℕ) (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) :
  24 ∣ (p^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_prime_square_minus_one_l3260_326058


namespace NUMINAMATH_CALUDE_employee_salary_proof_l3260_326073

-- Define the total weekly salary
def total_salary : ℝ := 594

-- Define the ratio of m's salary to n's salary
def salary_ratio : ℝ := 1.2

-- Define n's salary
def n_salary : ℝ := 270

-- Theorem statement
theorem employee_salary_proof :
  n_salary * (1 + salary_ratio) = total_salary :=
by sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l3260_326073


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l3260_326043

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l3260_326043


namespace NUMINAMATH_CALUDE_store_change_calculation_l3260_326061

def payment : ℕ := 20
def num_items : ℕ := 3
def item_cost : ℕ := 2

theorem store_change_calculation :
  payment - (num_items * item_cost) = 14 := by
  sorry

end NUMINAMATH_CALUDE_store_change_calculation_l3260_326061


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l3260_326008

theorem square_root_equation_solution (x : ℝ) :
  Real.sqrt (2 - 5 * x + x^2) = 9 ↔ x = (5 + Real.sqrt 341) / 2 ∨ x = (5 - Real.sqrt 341) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l3260_326008


namespace NUMINAMATH_CALUDE_function_property_l3260_326044

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3260_326044


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1997_l3260_326009

theorem rightmost_three_digits_of_7_to_1997 :
  7^1997 % 1000 = 207 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1997_l3260_326009


namespace NUMINAMATH_CALUDE_cubic_three_roots_m_range_l3260_326068

/-- The cubic polynomial f(x) = x³ - 6x² + 9x -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

/-- Theorem stating the range of m for which x³ - 6x² + 9x + m = 0 has exactly three distinct real roots -/
theorem cubic_three_roots_m_range :
  ∀ m : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ + m = 0 ∧ f r₂ + m = 0 ∧ f r₃ + m = 0) ↔ 
  -4 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_m_range_l3260_326068


namespace NUMINAMATH_CALUDE_money_sharing_l3260_326001

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda * 5 = ben * 3 →
  carlos * 5 = ben * 12 →
  ben = 25 →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3260_326001


namespace NUMINAMATH_CALUDE_square_pyramid_volume_l3260_326049

/-- The volume of a square pyramid inscribed in a cube -/
theorem square_pyramid_volume (cube_side_length : ℝ) (pyramid_volume : ℝ) :
  cube_side_length = 3 →
  pyramid_volume = (1 / 3) * (cube_side_length ^ 3) →
  pyramid_volume = 9 := by
sorry

end NUMINAMATH_CALUDE_square_pyramid_volume_l3260_326049


namespace NUMINAMATH_CALUDE_constant_segment_shadow_ratio_l3260_326029

-- Define a structure for a segment and its shadow
structure SegmentWithShadow where
  segment_length : ℝ
  shadow_length : ℝ

-- Define the fixed conditions (lines and projection direction)
axiom fixed_conditions : Prop

-- Define the theorem
theorem constant_segment_shadow_ratio 
  (s1 s2 : SegmentWithShadow) 
  (h : fixed_conditions) : 
  s1.segment_length / s1.shadow_length = s2.segment_length / s2.shadow_length :=
sorry

end NUMINAMATH_CALUDE_constant_segment_shadow_ratio_l3260_326029


namespace NUMINAMATH_CALUDE_player_B_more_consistent_l3260_326074

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (λ x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  variance player_B_scores < variance player_A_scores :=
by sorry

end NUMINAMATH_CALUDE_player_B_more_consistent_l3260_326074


namespace NUMINAMATH_CALUDE_duck_park_population_l3260_326025

theorem duck_park_population (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) : 
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  (initial_ducks * 2 - 10) - leaving_geese - (initial_ducks + arriving_ducks) = 1 :=
by sorry

end NUMINAMATH_CALUDE_duck_park_population_l3260_326025


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3260_326078

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest natural number with exactly k distinct divisors -/
def is_smallest_with_divisors (n k : ℕ) : Prop :=
  num_divisors n = k ∧ ∀ m < n, num_divisors m ≠ k

theorem smallest_number_with_2020_divisors :
  is_smallest_with_divisors (2^100 * 3^4 * 5 * 7) 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l3260_326078


namespace NUMINAMATH_CALUDE_f_at_4_l3260_326089

/-- The polynomial function f(x) = x^5 + 3x^4 - 5x^3 + 7x^2 - 9x + 11 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

/-- Theorem: The value of f(4) is 1559 -/
theorem f_at_4 : f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l3260_326089


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l3260_326057

/-- The number of sides in a convex polygon where the sum of n-1 internal angles is 2009 degrees -/
def polygon_sides : ℕ := 14

theorem convex_polygon_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 1) * 180 < 2009 →
  n * 180 > 2009 →
  n = polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l3260_326057


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l3260_326041

theorem rectangle_width_decrease (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.3 * L) (h2 : L * W = L' * W') : 
  (W - W') / W = 23.08 / 100 :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l3260_326041


namespace NUMINAMATH_CALUDE_product_of_fractions_l3260_326039

theorem product_of_fractions : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3260_326039


namespace NUMINAMATH_CALUDE_probability_123_in_10_rolls_l3260_326002

theorem probability_123_in_10_rolls (n : ℕ) (h : n = 10) :
  let total_outcomes := 6^n
  let favorable_outcomes := 8 * 6^7 - 15 * 6^4 + 4 * 6
  (favorable_outcomes : ℚ) / total_outcomes = 2220072 / 6^10 :=
by sorry

end NUMINAMATH_CALUDE_probability_123_in_10_rolls_l3260_326002


namespace NUMINAMATH_CALUDE_semicircles_area_ratio_l3260_326018

theorem semicircles_area_ratio (r : ℝ) (hr : r > 0) : 
  let circle_area := π * r^2
  let semicircle1_area := π * (r/2)^2 / 2
  let semicircle2_area := π * (r/3)^2 / 2
  (semicircle1_area + semicircle2_area) / circle_area = 13/72 := by
  sorry

end NUMINAMATH_CALUDE_semicircles_area_ratio_l3260_326018


namespace NUMINAMATH_CALUDE_ice_cream_permutations_l3260_326006

/-- The number of distinct permutations of n items, where some items may be identical -/
def distinctPermutations (n : ℕ) (itemCounts : List ℕ) : ℕ :=
  Nat.factorial n / (itemCounts.map Nat.factorial).prod

theorem ice_cream_permutations :
  distinctPermutations 4 [2, 1, 1] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_permutations_l3260_326006


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3260_326048

theorem quadratic_roots_property (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 4 = 0 ∧ 
   x₂^2 + a*x₂ + 4 = 0 ∧ 
   x₁^2 - 20/(3*x₂^3) = x₂^2 - 20/(3*x₁^3)) → 
  a = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3260_326048


namespace NUMINAMATH_CALUDE_cinema_chairs_l3260_326042

/-- The total number of chairs in a cinema with a given number of rows and chairs per row. -/
def total_chairs (rows : ℕ) (chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

/-- Theorem: The total number of chairs in a cinema with 4 rows and 8 chairs per row is 32. -/
theorem cinema_chairs : total_chairs 4 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_chairs_l3260_326042


namespace NUMINAMATH_CALUDE_symmetric_quadratic_comparison_l3260_326069

/-- A quadratic function that opens upward and is symmetric about x = 2013 -/
class SymmetricQuadratic (f : ℝ → ℝ) :=
  (opens_upward : ∃ (a b c : ℝ), a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (symmetric : ∀ x, f (2013 + x) = f (2013 - x))

/-- Theorem: For a symmetric quadratic function f that opens upward,
    f(2011) is greater than f(2014) -/
theorem symmetric_quadratic_comparison
  (f : ℝ → ℝ) [SymmetricQuadratic f] :
  f 2011 > f 2014 :=
sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_comparison_l3260_326069


namespace NUMINAMATH_CALUDE_circle_symmetry_orthogonality_l3260_326028

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the symmetry line
def symmetry_line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the orthogonality condition
def orthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1*x2 + y1*y2 = 0

theorem circle_symmetry_orthogonality :
  ∃ (m : ℝ) (x1 y1 x2 y2 : ℝ),
    curve x1 y1 ∧ curve x2 y2 ∧
    (∃ (x0 y0 : ℝ), symmetry_line m x0 y0 ∧ 
      (x1 - x0)^2 + (y1 - y0)^2 = (x2 - x0)^2 + (y2 - y0)^2) ∧
    orthogonal x1 y1 x2 y2 →
    m = -1 ∧ y2 - y1 = -(x2 - x1) := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_orthogonality_l3260_326028


namespace NUMINAMATH_CALUDE_total_investment_l3260_326085

/-- Given two investments with different interest rates, proves the total amount invested. -/
theorem total_investment (amount_at_8_percent : ℝ) (amount_at_9_percent : ℝ) 
  (h1 : amount_at_8_percent = 6000)
  (h2 : 0.08 * amount_at_8_percent + 0.09 * amount_at_9_percent = 840) :
  amount_at_8_percent + amount_at_9_percent = 10000 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_l3260_326085


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_l3260_326090

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define a point on the ellipse
def point_on_ellipse (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ellipse a b p.1 p.2

-- Define a diameter of the ellipse
def is_diameter (a b : ℝ) (c d : ℝ × ℝ) : Prop :=
  point_on_ellipse a b c ∧ point_on_ellipse a b d ∧ 
  c.1 = -d.1 ∧ c.2 = -d.2

-- Define a line parallel to CD passing through A
def parallel_line (a b : ℝ) (c d n m : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, n.1 = -a + t * (d.1 - c.1) ∧ 
            n.2 = t * (d.2 - c.2) ∧
            m.1 = -a + (a / (d.1 - c.1)) * (d.1 - c.1) ∧
            m.2 = (a / (d.1 - c.1)) * (d.2 - c.2)

-- Theorem statement
theorem ellipse_intersection_product (a b : ℝ) (c d n m : ℝ × ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hcd : is_diameter a b c d)
  (hnm : parallel_line a b c d n m)
  (hn : point_on_ellipse a b n) :
  let a := (-a, 0)
  let o := (0, 0)
  (dist a m) * (dist a n) = (dist c o) * (dist c d) := by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_product_l3260_326090


namespace NUMINAMATH_CALUDE_solution_sets_equivalent_l3260_326034

theorem solution_sets_equivalent : 
  {x : ℝ | |8*x + 9| < 7} = {x : ℝ | -4*x^2 - 9*x - 2 > 0} := by sorry

end NUMINAMATH_CALUDE_solution_sets_equivalent_l3260_326034


namespace NUMINAMATH_CALUDE_car_speed_l3260_326066

-- Define the problem parameters
def gallons_per_40_miles : ℝ := 1
def tank_capacity : ℝ := 12
def travel_time : ℝ := 5
def fuel_used_fraction : ℝ := 0.4166666666666667

-- Define the theorem
theorem car_speed (speed : ℝ) : 
  (gallons_per_40_miles * speed * travel_time / 40 = fuel_used_fraction * tank_capacity) →
  speed = 40 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_l3260_326066


namespace NUMINAMATH_CALUDE_group_size_is_16_l3260_326095

/-- The number of children whose height increases -/
def num_taller_children : ℕ := 12

/-- The height increase for each of the taller children in cm -/
def height_increase : ℕ := 8

/-- The total height increase in cm -/
def total_height_increase : ℕ := num_taller_children * height_increase

/-- The mean height increase in cm -/
def mean_height_increase : ℕ := 6

theorem group_size_is_16 :
  ∃ n : ℕ, n > 0 ∧ (total_height_increase : ℚ) / n = mean_height_increase ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_group_size_is_16_l3260_326095


namespace NUMINAMATH_CALUDE_gcd_126_105_l3260_326077

theorem gcd_126_105 : Nat.gcd 126 105 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_126_105_l3260_326077


namespace NUMINAMATH_CALUDE_xyz_equals_five_l3260_326081

-- Define the variables
variable (x y z : ℝ)

-- Define the theorem
theorem xyz_equals_five
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l3260_326081


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l3260_326080

theorem sandy_shopping_money (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 320 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l3260_326080


namespace NUMINAMATH_CALUDE_equation_solution_l3260_326084

theorem equation_solution : 
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3260_326084


namespace NUMINAMATH_CALUDE_factor_implies_s_value_l3260_326038

theorem factor_implies_s_value (m s : ℤ) : 
  (∃ k : ℤ, m^2 - s*m - 24 = (m - 8) * k) → s = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_s_value_l3260_326038


namespace NUMINAMATH_CALUDE_more_male_students_l3260_326097

theorem more_male_students (total : ℕ) (female : ℕ) (h1 : total = 280) (h2 : female = 127) :
  total - female - female = 26 := by
  sorry

end NUMINAMATH_CALUDE_more_male_students_l3260_326097


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l3260_326003

theorem quadratic_root_proof : let x : ℝ := (-15 - Real.sqrt 181) / 8
  ∀ u : ℝ, u = 2.75 → 4 * x^2 + 15 * x + u = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l3260_326003


namespace NUMINAMATH_CALUDE_four_divisions_for_400_to_25_l3260_326056

/-- The number of divisions needed to reduce a collection of books to a target group size -/
def divisions_needed (total_books : ℕ) (target_group_size : ℕ) : ℕ :=
  if total_books ≤ target_group_size then 0
  else 1 + divisions_needed (total_books / 2) target_group_size

/-- Theorem stating that 4 divisions are needed to reduce 400 books to groups of 25 -/
theorem four_divisions_for_400_to_25 :
  divisions_needed 400 25 = 4 := by
sorry

end NUMINAMATH_CALUDE_four_divisions_for_400_to_25_l3260_326056


namespace NUMINAMATH_CALUDE_problem_solution_l3260_326033

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := a > 0 ∧ ∃ c, c > 0 ∧ c < a ∧
  ∀ x y, x^2/2 + y^2/a = 1 → y^2 ≥ c*(1 - x^2/2)

-- Define the main theorem
theorem problem_solution (a : ℝ) 
  (h1 : ¬(q a))
  (h2 : p a ∨ q a) :
  (-1 < a ∧ a ≤ 0) ∧
  ((a = 1 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) → y = 0) ∧
   (-1 < a ∧ a < 0 → ∃ b c, b > c ∧ c > 0 ∧
     ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
       x^2/b^2 + y^2/c^2 = 1) ∧
   (a = 0 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
     x^2 + y^2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3260_326033


namespace NUMINAMATH_CALUDE_intersection_range_is_correct_l3260_326016

/-- Line l with parameter t -/
structure Line where
  a : ℝ
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ t, x t = a - 2 * t * (y t)
  h2 : ∀ t, y t = -4 * t

/-- Circle C with parameter θ -/
structure Circle where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h1 : ∀ θ, x θ = 4 * Real.cos θ
  h2 : ∀ θ, y θ = 4 * Real.sin θ

/-- The range of a for which line l intersects circle C -/
def intersectionRange (l : Line) (c : Circle) : Set ℝ :=
  { a | ∃ t θ, l.x t = c.x θ ∧ l.y t = c.y θ }

theorem intersection_range_is_correct (l : Line) (c : Circle) :
  intersectionRange l c = Set.Icc (-4 * Real.sqrt 5) (4 * Real.sqrt 5) := by
  sorry

#check intersection_range_is_correct

end NUMINAMATH_CALUDE_intersection_range_is_correct_l3260_326016


namespace NUMINAMATH_CALUDE_fraction_problem_l3260_326086

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (4 * n - 4) = 1 / 2 → n = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3260_326086


namespace NUMINAMATH_CALUDE_equation_solution_l3260_326000

theorem equation_solution :
  ∃ x : ℚ, (3 * x + 4 * x = 600 - (5 * x + 6 * x)) ∧ (x = 100 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3260_326000


namespace NUMINAMATH_CALUDE_place_face_difference_46_4_l3260_326022

/-- The place value of a digit in a two-digit number -/
def placeValue (n : ℕ) (d : ℕ) : ℕ :=
  if n ≥ 10 ∧ n < 100 ∧ d = n / 10 then d * 10 else 0

/-- The face value of a digit -/
def faceValue (d : ℕ) : ℕ := d

/-- The difference between place value and face value for a digit in a two-digit number -/
def placeFaceDifference (n : ℕ) (d : ℕ) : ℕ :=
  placeValue n d - faceValue d

theorem place_face_difference_46_4 : 
  placeFaceDifference 46 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_place_face_difference_46_4_l3260_326022


namespace NUMINAMATH_CALUDE_melanie_initial_dimes_l3260_326015

/-- The number of dimes Melanie initially had in her bank -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Melanie's dad gave her -/
def dimes_from_dad : ℕ := 8

/-- The number of dimes Melanie gave to her mother -/
def dimes_to_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 11

theorem melanie_initial_dimes : 
  initial_dimes + dimes_from_dad - dimes_to_mother = current_dimes := by sorry

end NUMINAMATH_CALUDE_melanie_initial_dimes_l3260_326015


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3260_326047

-- Define the cubic function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem stating the properties of f(x)
theorem cubic_function_properties :
  (∀ x, f' x = 0 ↔ x = 1 ∨ x = -1) ∧
  f (-2) = -4 ∧
  f (-1) = 0 ∧
  f 1 = -4 ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3260_326047


namespace NUMINAMATH_CALUDE_max_slices_formula_l3260_326093

/-- Represents a triangular cake with candles -/
structure TriangularCake where
  numCandles : ℕ
  candlesNotCollinear : True  -- Placeholder for the condition that no three candles are collinear

/-- The maximum number of triangular slices for a given cake -/
def maxSlices (cake : TriangularCake) : ℕ :=
  2 * cake.numCandles - 5

/-- Theorem stating the maximum number of slices for a cake with k candles -/
theorem max_slices_formula (k : ℕ) (h : k ≥ 3) :
  ∀ (cake : TriangularCake), cake.numCandles = k →
    maxSlices cake = 2 * k - 5 := by
  sorry

end NUMINAMATH_CALUDE_max_slices_formula_l3260_326093


namespace NUMINAMATH_CALUDE_perfect_square_minus_seven_l3260_326017

theorem perfect_square_minus_seven (k : ℕ+) : 
  ∃ (n m : ℕ+), n * 2^k.val - 7 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_minus_seven_l3260_326017


namespace NUMINAMATH_CALUDE_additional_money_needed_l3260_326031

def new_computer_cost : ℕ := 80
def initial_savings : ℕ := 50
def old_computer_sale : ℕ := 20

theorem additional_money_needed : 
  new_computer_cost - (initial_savings + old_computer_sale) = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l3260_326031


namespace NUMINAMATH_CALUDE_youngest_child_age_l3260_326046

def mother_charge : ℝ := 5.05
def child_charge_per_year : ℝ := 0.55
def total_bill : ℝ := 11.05

def is_valid_age_combination (twin_age : ℕ) (youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem youngest_child_age :
  ∀ youngest_age : ℕ,
    (∃ twin_age : ℕ, is_valid_age_combination twin_age youngest_age) ↔
    (youngest_age = 1 ∨ youngest_age = 3) :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3260_326046


namespace NUMINAMATH_CALUDE_kids_at_camp_difference_l3260_326082

theorem kids_at_camp_difference (camp_kids home_kids : ℕ) 
  (h1 : camp_kids = 819058) 
  (h2 : home_kids = 668278) : 
  camp_kids - home_kids = 150780 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_difference_l3260_326082


namespace NUMINAMATH_CALUDE_candy_chocolate_choices_l3260_326099

theorem candy_chocolate_choices (num_candies num_chocolates : ℕ) : 
  num_candies = 2 → num_chocolates = 3 → num_candies + num_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_choices_l3260_326099


namespace NUMINAMATH_CALUDE_cheddar_cheese_sticks_l3260_326060

theorem cheddar_cheese_sticks (mozzarella : ℕ) (pepperjack : ℕ) (p_pepperjack : ℚ) : ℕ :=
  let total := pepperjack * 2
  let cheddar := total - mozzarella - pepperjack
  by
    have h1 : mozzarella = 30 := by sorry
    have h2 : pepperjack = 45 := by sorry
    have h3 : p_pepperjack = 1/2 := by sorry
    exact 15

#check cheddar_cheese_sticks

end NUMINAMATH_CALUDE_cheddar_cheese_sticks_l3260_326060


namespace NUMINAMATH_CALUDE_largest_difference_l3260_326064

def A : ℕ := 3 * 1005^1006
def B : ℕ := 1005^1006
def C : ℕ := 1004 * 1005^1005
def D : ℕ := 3 * 1005^1005
def E : ℕ := 1005^1005
def F : ℕ := 1005^1004

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l3260_326064


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l3260_326098

/-- Calculates the required reduction in food consumption per student to maintain
    the same total cost given a decrease in student population and an increase in food price. -/
theorem food_consumption_reduction
  (initial_students : ℕ)
  (initial_price : ℝ)
  (student_decrease_rate : ℝ)
  (price_increase_rate : ℝ)
  (h1 : student_decrease_rate = 0.1)
  (h2 : price_increase_rate = 0.2)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_students : ℝ := initial_students * (1 - student_decrease_rate)
  let new_price : ℝ := initial_price * (1 + price_increase_rate)
  let consumption_ratio : ℝ := (initial_students * initial_price) / (new_students * new_price)
  abs (1 - consumption_ratio - 0.0741) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_food_consumption_reduction_l3260_326098


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_eight_l3260_326065

theorem sqrt_expression_equals_eight :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_eight_l3260_326065


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l3260_326037

theorem rectangular_field_dimensions : ∃ m : ℝ, m > 3 ∧ (3*m + 8)*(m - 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l3260_326037


namespace NUMINAMATH_CALUDE_expand_expression_l3260_326087

theorem expand_expression (x y : ℝ) : (-x + 2*y) * (-x - 2*y) = x^2 - 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3260_326087


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l3260_326075

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

theorem line_plane_parallelism 
  (m n : Line) (α : Plane)
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l3260_326075


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l3260_326052

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x - 2) * (6*x^8 + 3*x^7 - 2*x^3 + x) = 18*x^9 - 3*x^8 - 6*x^7 - 6*x^4 - 4*x^3 + x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l3260_326052


namespace NUMINAMATH_CALUDE_quadratic_sequence_inconsistency_l3260_326062

def isQuadraticSequence (seq : List ℤ) : Prop :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  secondDiffs.all (· = secondDiffs.head!)

def findInconsistentTerm (seq : List ℤ) : Option ℤ :=
  let firstDiffs := List.zipWith (·-·) seq.tail seq
  let secondDiffs := List.zipWith (·-·) firstDiffs.tail firstDiffs
  if h : secondDiffs.all (· = secondDiffs.head!) then
    none
  else
    some (seq[secondDiffs.findIndex (· ≠ secondDiffs.head!) + 1]!)

theorem quadratic_sequence_inconsistency 
  (seq : List ℤ) 
  (hseq : seq = [2107, 2250, 2402, 2574, 2738, 2920, 3094, 3286]) : 
  ¬isQuadraticSequence seq ∧ findInconsistentTerm seq = some 2574 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_inconsistency_l3260_326062


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3260_326094

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (5 : ℂ) / (2 - i) - i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3260_326094


namespace NUMINAMATH_CALUDE_tree_scenario_result_l3260_326014

/-- Represents the number of caterpillars and leaves eaten in a tree scenario -/
def tree_scenario (initial_caterpillars storm_fallen hatched_eggs 
                   baby_leaves_eaten cocoon_left moth_ratio
                   moth_daily_consumption days : ℕ) : ℕ × ℕ :=
  let remaining_after_storm := initial_caterpillars - storm_fallen
  let total_after_hatch := remaining_after_storm + hatched_eggs
  let remaining_after_cocoon := total_after_hatch - cocoon_left
  let moth_caterpillars := remaining_after_cocoon / 2
  let total_leaves_eaten := baby_leaves_eaten + 
    moth_caterpillars * moth_daily_consumption * days
  (remaining_after_cocoon, total_leaves_eaten)

/-- Theorem stating the result of the tree scenario -/
theorem tree_scenario_result : 
  tree_scenario 14 3 6 18 9 2 4 7 = (8, 130) :=
by sorry

end NUMINAMATH_CALUDE_tree_scenario_result_l3260_326014


namespace NUMINAMATH_CALUDE_largest_number_l3260_326051

/-- Represents a real number with a repeating decimal expansion -/
def RepeatingDecimal (whole : ℕ) (nonRepeating : List ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The number 8.12356 -/
def num1 : ℚ := 8.12356

/-- The number 8.123$\overline{5}$ -/
def num2 : ℚ := RepeatingDecimal 8 [1, 2, 3] [5]

/-- The number 8.12$\overline{356}$ -/
def num3 : ℚ := RepeatingDecimal 8 [1, 2] [3, 5, 6]

/-- The number 8.1$\overline{2356}$ -/
def num4 : ℚ := RepeatingDecimal 8 [1] [2, 3, 5, 6]

/-- The number 8.$\overline{12356}$ -/
def num5 : ℚ := RepeatingDecimal 8 [] [1, 2, 3, 5, 6]

theorem largest_number : 
  num2 > num1 ∧ num2 > num3 ∧ num2 > num4 ∧ num2 > num5 :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3260_326051


namespace NUMINAMATH_CALUDE_cos_minus_sin_for_point_l3260_326076

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (x y : Real), x = 3/5 ∧ y = -4/5 ∧ x = Real.cos α ∧ y = Real.sin α) →
  Real.cos α - Real.sin α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_for_point_l3260_326076


namespace NUMINAMATH_CALUDE_cone_volume_l3260_326013

theorem cone_volume (central_angle : Real) (sector_area : Real) :
  central_angle = 120 * Real.pi / 180 →
  sector_area = 3 * Real.pi →
  ∃ (volume : Real), volume = (2 * Real.sqrt 2 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3260_326013


namespace NUMINAMATH_CALUDE_library_book_distribution_l3260_326032

/-- The number of books bought for each grade -/
def BookDistribution := Fin 4 → ℕ

/-- The total number of books bought -/
def total_books (d : BookDistribution) : ℕ :=
  d 0 + d 1 + d 2 + d 3

theorem library_book_distribution :
  ∃ (d : BookDistribution),
    d 0 = 37 ∧
    d 1 = 39 ∧
    d 2 = 43 ∧
    d 3 = 28 ∧
    d 1 + d 2 + d 3 = 110 ∧
    d 0 + d 2 + d 3 = 108 ∧
    d 0 + d 1 + d 3 = 104 ∧
    d 0 + d 1 + d 2 = 119 ∧
    total_books d = 147 :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l3260_326032


namespace NUMINAMATH_CALUDE_confidence_level_interpretation_l3260_326035

theorem confidence_level_interpretation 
  (confidence_level : ℝ) 
  (hypothesis_test : Type) 
  (is_valid_test : hypothesis_test → Prop) 
  (test_result : hypothesis_test → Bool) 
  (h_confidence : confidence_level = 0.95) :
  ∃ (error_probability : ℝ), 
    error_probability = 1 - confidence_level ∧ 
    error_probability = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_confidence_level_interpretation_l3260_326035


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3260_326072

/-- Given a quadratic equation (a-1)x² + x + a² - 1 = 0 where one root is 0, prove that a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∀ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3260_326072


namespace NUMINAMATH_CALUDE_inequality_for_positive_reals_l3260_326030

theorem inequality_for_positive_reals : ∀ x : ℝ, x > 0 → x + 4/x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_reals_l3260_326030


namespace NUMINAMATH_CALUDE_stratified_sampling_athletes_l3260_326070

theorem stratified_sampling_athletes (total_male : ℕ) (total_female : ℕ) 
  (drawn_male : ℕ) (drawn_female : ℕ) : 
  total_male = 64 → total_female = 56 → drawn_male = 8 →
  (drawn_male : ℚ) / total_male = (drawn_female : ℚ) / total_female →
  drawn_female = 7 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_athletes_l3260_326070


namespace NUMINAMATH_CALUDE_circular_garden_area_increase_l3260_326071

theorem circular_garden_area_increase : 
  let original_diameter : ℝ := 20
  let new_diameter : ℝ := 30
  let original_area := π * (original_diameter / 2)^2
  let new_area := π * (new_diameter / 2)^2
  let area_increase := new_area - original_area
  let percent_increase := (area_increase / original_area) * 100
  percent_increase = 125 := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_increase_l3260_326071


namespace NUMINAMATH_CALUDE_min_value_theorem_l3260_326053

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3260_326053


namespace NUMINAMATH_CALUDE_least_number_of_marbles_l3260_326036

theorem least_number_of_marbles (x : ℕ) : x = 50 ↔ 
  x > 0 ∧ 
  x % 6 = 2 ∧ 
  x % 4 = 3 ∧ 
  ∀ y : ℕ, y > 0 ∧ y % 6 = 2 ∧ y % 4 = 3 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_marbles_l3260_326036


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l3260_326010

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem two_digit_reverse_sum_square :
  {n : ℕ | is_two_digit n ∧ is_perfect_square (n + reverse_digits n)} =
  {29, 38, 47, 56, 65, 74, 83, 92} := by sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l3260_326010


namespace NUMINAMATH_CALUDE_average_book_cost_l3260_326088

theorem average_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 236 → 
  books_bought = 6 → 
  amount_left = 14 → 
  (initial_amount - amount_left) / books_bought = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_book_cost_l3260_326088


namespace NUMINAMATH_CALUDE_chord_equation_l3260_326026

/-- Given an ellipse and a point M, prove the equation of the line containing the chord with midpoint M -/
theorem chord_equation (x y : ℝ) :
  (x^2 / 4 + y^2 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 4 + y₁^2 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 4 + y₂^2 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧    -- x-coordinate of midpoint M
    ((y₁ + y₂) / 2 = 1/2) ∧  -- y-coordinate of midpoint M
    (y - 1/2 = -(1/2) * (x - 1))) →  -- Equation of the line through M with slope -1/2
  x + 2*y - 2 = 0  -- Resulting equation of the line
:= by sorry

end NUMINAMATH_CALUDE_chord_equation_l3260_326026


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l3260_326007

theorem greatest_common_divisor_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 540 ∧ 
  n < 60 ∧ 
  n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 → m < 60 → m ∣ 180 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l3260_326007


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3260_326004

def total_toppings : ℕ := 9
def toppings_to_select : ℕ := 4
def required_toppings : ℕ := 2

theorem pizza_toppings_combinations :
  (Nat.choose total_toppings toppings_to_select) -
  (Nat.choose (total_toppings - required_toppings) toppings_to_select) = 91 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3260_326004


namespace NUMINAMATH_CALUDE_break_time_is_30_minutes_l3260_326023

/-- Represents the travel scenario with three train stations -/
structure TravelScenario where
  /-- Time between each station in hours -/
  station_distance : ℝ
  /-- Total travel time including break in minutes -/
  total_time : ℝ

/-- Calculates the break time at the second station -/
def break_time (scenario : TravelScenario) : ℝ :=
  scenario.total_time - 2 * (scenario.station_distance * 60)

/-- Theorem stating that the break time is 30 minutes -/
theorem break_time_is_30_minutes (scenario : TravelScenario) 
  (h1 : scenario.station_distance = 2)
  (h2 : scenario.total_time = 270) : 
  break_time scenario = 30 := by
  sorry

end NUMINAMATH_CALUDE_break_time_is_30_minutes_l3260_326023


namespace NUMINAMATH_CALUDE_remainder_problem_l3260_326024

theorem remainder_problem (m n : ℕ) (h1 : m > n) (h2 : n % 6 = 3) (h3 : (m - n) % 6 = 5) :
  m % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3260_326024


namespace NUMINAMATH_CALUDE_quadratic_max_l3260_326055

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 2)^2 - 3

-- State the theorem
theorem quadratic_max (x : ℝ) : 
  (∀ y : ℝ, f y ≤ f 2) ∧ f 2 = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_l3260_326055


namespace NUMINAMATH_CALUDE_log_sum_sqrt_equality_l3260_326059

theorem log_sum_sqrt_equality : Real.sqrt (Real.log 8 / Real.log 4 + Real.log 16 / Real.log 8) = Real.sqrt (17 / 6) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_sqrt_equality_l3260_326059


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3260_326020

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3260_326020


namespace NUMINAMATH_CALUDE_paper_parts_cannot_reach_2020_can_reach_2023_l3260_326045

def paper_sequence : Nat → Nat
  | 0 => 1
  | n + 1 => paper_sequence n + 2

theorem paper_parts (n : Nat) : 
  paper_sequence n = 2 * n + 1 := by sorry

theorem cannot_reach_2020 : 
  ∀ n, paper_sequence n ≠ 2020 := by sorry

theorem can_reach_2023 : 
  ∃ n, paper_sequence n = 2023 := by sorry

end NUMINAMATH_CALUDE_paper_parts_cannot_reach_2020_can_reach_2023_l3260_326045


namespace NUMINAMATH_CALUDE_number_count_l3260_326067

theorem number_count (avg_all : ℝ) (avg_pair1 : ℝ) (avg_pair2 : ℝ) (avg_pair3 : ℝ) 
  (h1 : avg_all = 4.60)
  (h2 : avg_pair1 = 3.4)
  (h3 : avg_pair2 = 3.8)
  (h4 : avg_pair3 = 6.6) :
  ∃ n : ℕ, n = 6 ∧ n * avg_all = 2 * (avg_pair1 + avg_pair2 + avg_pair3) := by
  sorry

end NUMINAMATH_CALUDE_number_count_l3260_326067
