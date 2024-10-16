import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_sum_condition_l2307_230729

/-- For distinct positive numbers a, b, c that are not perfect squares,
    √a + √b = √c holds if and only if 2√(ab) = c - (a + b) and ab is a perfect square -/
theorem sqrt_sum_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (hna : ¬ ∃ (n : ℕ), a = n^2)
  (hnb : ¬ ∃ (n : ℕ), b = n^2)
  (hnc : ¬ ∃ (n : ℕ), c = n^2) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt c) ↔ 
  (2 * Real.sqrt (a * b) = c - (a + b) ∧ ∃ (n : ℕ), a * b = n^2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_condition_l2307_230729


namespace NUMINAMATH_CALUDE_triangle_problem_l2307_230745

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 2 →
  Real.cos A = 1/2 →
  -- (I)
  Real.sin B = Real.sqrt 3 / 3 ∧
  -- (II)
  c = 1 + Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2307_230745


namespace NUMINAMATH_CALUDE_parabola_directrix_l2307_230796

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -49/16

/-- Theorem stating that the directrix of the given parabola is y = -49/16 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.2 - d) = (x^2 + (y + 3 - 1/(16:ℝ))^2) / (4 * 1/(16:ℝ))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2307_230796


namespace NUMINAMATH_CALUDE_monomial_sum_implies_expression_l2307_230774

/-- If the sum of two monomials is still a monomial, then a specific expression evaluates to -1 --/
theorem monomial_sum_implies_expression (m n : ℝ) : 
  (∃ (a : ℝ), ∃ (k : ℕ), ∃ (l : ℕ), 3 * (X : ℝ → ℝ → ℝ) k l + (-2) * (X : ℝ → ℝ → ℝ) (2*m+3) 3 = a * (X : ℝ → ℝ → ℝ) k l) →
  (4*m - n)^n = -1 := by
  sorry

/-- Helper function to represent monomials --/
def X (i j : ℕ) : ℝ → ℝ → ℝ := fun x y ↦ x^i * y^j

end NUMINAMATH_CALUDE_monomial_sum_implies_expression_l2307_230774


namespace NUMINAMATH_CALUDE_inequality_of_powers_l2307_230776

theorem inequality_of_powers (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l2307_230776


namespace NUMINAMATH_CALUDE_fraction_calculation_l2307_230705

theorem fraction_calculation (x y : ℚ) (hx : x = 4/7) (hy : y = 5/8) :
  (7*x + 5*y) / (70*x*y) = 57/400 := by
sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2307_230705


namespace NUMINAMATH_CALUDE_athletic_groups_l2307_230756

theorem athletic_groups (x y : ℕ) : 
  (7 * y + 3 = x) ∧ (8 * y - 5 = x) → x = 59 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_athletic_groups_l2307_230756


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_minus_one_l2307_230724

theorem binomial_coefficient_n_plus_one_choose_n_minus_one (n : ℕ+) :
  Nat.choose (n + 1) (n - 1) = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_plus_one_choose_n_minus_one_l2307_230724


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2307_230799

theorem gcd_digits_bound (a b : ℕ) : 
  100000 ≤ a ∧ a < 1000000 ∧ 
  100000 ≤ b ∧ b < 1000000 ∧ 
  1000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000 →
  Nat.gcd a b < 1000 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2307_230799


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_three_l2307_230751

theorem trigonometric_expression_equals_three (α : ℝ) 
  (h : Real.tan (3 * Real.pi + α) = 3) : 
  (Real.sin (α - 3 * Real.pi) + Real.cos (Real.pi - α) + 
   Real.sin (Real.pi / 2 - α) - 2 * Real.cos (Real.pi / 2 + α)) / 
  (-Real.sin (-α) + Real.cos (Real.pi + α)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_three_l2307_230751


namespace NUMINAMATH_CALUDE_line_l_properties_l2307_230738

-- Define the line l: ax + y + a = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x + y + a = 0

-- Theorem statement
theorem line_l_properties (a : ℝ) :
  -- 1. The line passes through the point (-1, 0)
  line_l a (-1) 0 ∧
  -- 2. When a = -1, the line is perpendicular to x + y - 2 = 0
  (a = -1 → ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    line_l (-1) x₁ y₁ ∧
    line_l (-1) x₂ y₂ ∧
    x₁ + y₁ - 2 = 0 ∧
    x₂ + y₂ - 2 = 0 ∧
    (y₂ - y₁) * (x₂ - x₁) = -1) ∧
  -- 3. When a > 0, the line passes through the second, third, and fourth quadrants
  (a > 0 → ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line_l a x₁ y₁ ∧ x₁ < 0 ∧ y₁ > 0 ∧  -- Second quadrant
    line_l a x₂ y₂ ∧ x₂ < 0 ∧ y₂ < 0 ∧  -- Third quadrant
    line_l a x₃ y₃ ∧ x₃ > 0 ∧ y₃ < 0)   -- Fourth quadrant
  := by sorry

end NUMINAMATH_CALUDE_line_l_properties_l2307_230738


namespace NUMINAMATH_CALUDE_set_operation_result_l2307_230703

def A : Set Int := {-1, 0}
def B : Set Int := {0, 1}
def C : Set Int := {1, 2}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l2307_230703


namespace NUMINAMATH_CALUDE_midpoint_product_zero_l2307_230753

/-- Given that C = (4, 3) is the midpoint of line segment AB where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product_zero (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_product_zero_l2307_230753


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l2307_230708

/-- Given vectors a and b in ℝ², prove that the value of t that makes (a - b) perpendicular to (a - t*b) is -11/30 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (h1 : a = (-3, 1)) (h2 : b = (2, 5)) :
  ∃ t : ℝ, t = -11/30 ∧ (a.1 - b.1, a.2 - b.2) • (a.1 - t * b.1, a.2 - t * b.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l2307_230708


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2307_230714

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The cumulative distribution function (CDF) for a normal distribution -/
noncomputable def normalCDF (nd : NormalDistribution) (x : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (nd : NormalDistribution) 
  (h_mean : nd.μ = 85) 
  (h_cdf : normalCDF nd 122 = 0.96) :
  normalCDF nd 48 = 0.04 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2307_230714


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2307_230720

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + a 9 = 10) :
  a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2307_230720


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l2307_230749

theorem smallest_integer_y (y : ℤ) : (7 - 5 * y < 22) ↔ (y > -3) :=
  sorry

theorem smallest_integer_solution : ∃ y : ℤ, (∀ z : ℤ, 7 - 5 * z < 22 → y ≤ z) ∧ (7 - 5 * y < 22) ∧ y = -2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l2307_230749


namespace NUMINAMATH_CALUDE_divisibility_condition_l2307_230707

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2307_230707


namespace NUMINAMATH_CALUDE_modulus_of_complex_fourth_power_l2307_230779

theorem modulus_of_complex_fourth_power : 
  Complex.abs ((2 : ℂ) + (3 * Real.sqrt 2) * Complex.I) ^ 4 = 484 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fourth_power_l2307_230779


namespace NUMINAMATH_CALUDE_combined_distance_of_trains_l2307_230715

-- Define the speeds of the trains in km/h
def train_A_speed : ℝ := 120
def train_B_speed : ℝ := 160

-- Define the time in hours (45 minutes = 0.75 hours)
def time : ℝ := 0.75

-- Theorem statement
theorem combined_distance_of_trains :
  train_A_speed * time + train_B_speed * time = 210 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_of_trains_l2307_230715


namespace NUMINAMATH_CALUDE_cakes_served_total_l2307_230775

/-- The number of cakes served in a restaurant over two days. -/
def total_cakes (lunch_today dinner_today yesterday : ℕ) : ℕ :=
  lunch_today + dinner_today + yesterday

/-- Theorem stating that the total number of cakes served is 14 -/
theorem cakes_served_total :
  total_cakes 5 6 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_total_l2307_230775


namespace NUMINAMATH_CALUDE_king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l2307_230730

/-- Represents a network of cities and roads -/
structure CityNetwork where
  n : ℕ                -- number of cities
  roads : ℕ             -- number of roads
  connected : Prop      -- any city can be reached from any other city
  distances : Finset ℕ  -- set of shortest distances between pairs of cities

/-- The condition for a valid city network -/
def validNetwork (net : CityNetwork) : Prop :=
  net.roads = net.n - 1 ∧
  net.connected ∧
  net.distances = Finset.range (net.n * (net.n - 1) / 2 + 1) \ {0}

/-- The condition for the network to meet the king's requirement -/
def meetsRequirement (n : ℕ) : Prop :=
  ∃ (net : CityNetwork), net.n = n ∧ validNetwork net

/-- The main theorem -/
theorem king_requirement (n : ℕ) :
  meetsRequirement n ↔ (∃ k : ℕ, n = k^2) ∨ (∃ k : ℕ, n = k^2 + 2) :=
sorry

/-- The requirement can be met for n = 6 -/
theorem requirement_met_for_6 : meetsRequirement 6 :=
sorry

/-- The requirement cannot be met for n = 1986 -/
theorem requirement_not_met_for_1986 : ¬meetsRequirement 1986 :=
sorry

end NUMINAMATH_CALUDE_king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l2307_230730


namespace NUMINAMATH_CALUDE_cathy_cookies_l2307_230734

theorem cathy_cookies (total : ℝ) (amy_fraction : ℝ) (bob_cookies : ℝ) : 
  total = 18 → 
  amy_fraction = 1/3 → 
  bob_cookies = 2.5 → 
  total - (amy_fraction * total + bob_cookies) = 9.5 := by
sorry

end NUMINAMATH_CALUDE_cathy_cookies_l2307_230734


namespace NUMINAMATH_CALUDE_cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l2307_230702

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the line l in parametric form
def line_l (t x y : ℝ) : Prop := x = 3 + (1/2) * t ∧ y = -3 + (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of circle C
theorem cartesian_equation_circle_C :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, circle_C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ 
  (x - 3)^2 + y^2 = 9 :=
sorry

-- Define a function to represent the ratio of arc lengths
def arc_length_ratio (r₁ r₂ : ℝ) : Prop := r₁ / r₂ = 1 / 2

-- Theorem for the ratio of arc lengths
theorem arc_length_ratio_circle_C_line_l :
  ∃ r₁ r₂ : ℝ, arc_length_ratio r₁ r₂ ∧ 
  (∀ x y : ℝ, (x - 3)^2 + y^2 = 9 → 
    (∃ t : ℝ, line_l t x y) → 
    (r₁ + r₂ = 2 * Real.pi * 3 ∧ r₁ ≤ r₂)) :=
sorry

end NUMINAMATH_CALUDE_cartesian_equation_circle_C_arc_length_ratio_circle_C_line_l_l2307_230702


namespace NUMINAMATH_CALUDE_min_value_inequality_l2307_230718

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2307_230718


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_three_halves_l2307_230750

/-- Given real numbers a, b, and c satisfying the conditions,
    prove that (a/(b+c)) + (b/(c+a)) + (c/(a+b)) = 3/2 -/
theorem sum_of_fractions_equals_three_halves
  (a b c : ℝ)
  (h1 : a^3 + b^3 + c^3 = 3*a*b*c)
  (h2 : Matrix.det !![a, b, c; c, a, b; b, c, a] = 0) :
  a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_three_halves_l2307_230750


namespace NUMINAMATH_CALUDE_floor_tiles_count_l2307_230771

/-- Represents a square floor tiled with square tiles -/
structure SquareFloor where
  side_length : ℕ
  is_square : side_length > 0

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

theorem floor_tiles_count (floor : SquareFloor) 
  (h : black_tiles floor = 101) : 
  total_tiles floor = 2601 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_count_l2307_230771


namespace NUMINAMATH_CALUDE_thomas_lost_pieces_l2307_230798

theorem thomas_lost_pieces (total_start : Nat) (player_start : Nat) (audrey_lost : Nat) (total_end : Nat) :
  total_start = 32 →
  player_start = 16 →
  audrey_lost = 6 →
  total_end = 21 →
  player_start - (total_end - (player_start - audrey_lost)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_thomas_lost_pieces_l2307_230798


namespace NUMINAMATH_CALUDE_corner_sum_possibilities_l2307_230737

/-- Represents the color of a cell on the board -/
inductive CellColor
| Gold
| Silver

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (colorAt : Nat → Nat → CellColor)
  (numberAt : Nat → Nat → Nat)

/-- Defines a valid board configuration -/
def validBoard (b : Board) : Prop :=
  b.rows = 2016 ∧ b.cols = 2017 ∧
  (∀ i j, b.numberAt i j = 0 ∨ b.numberAt i j = 1) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt i (j+1)) ∧
  (∀ i j, b.colorAt i j ≠ b.colorAt (i+1) j) ∧
  (∀ i j, b.colorAt i j = CellColor.Gold →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 0) ∧
  (∀ i j, b.colorAt i j = CellColor.Silver →
    (b.numberAt i j + b.numberAt (i+1) j + b.numberAt i (j+1) + b.numberAt (i+1) (j+1)) % 2 = 1)

/-- The theorem to be proved -/
theorem corner_sum_possibilities (b : Board) (h : validBoard b) :
  let cornerSum := b.numberAt 0 0 + b.numberAt 0 (b.cols-1) + b.numberAt (b.rows-1) 0 + b.numberAt (b.rows-1) (b.cols-1)
  cornerSum = 0 ∨ cornerSum = 2 ∨ cornerSum = 4 :=
sorry

end NUMINAMATH_CALUDE_corner_sum_possibilities_l2307_230737


namespace NUMINAMATH_CALUDE_parabola_vertex_relationship_l2307_230760

/-- Given a parabola y = x^2 - 2mx + 2m^2 - 3m + 1, prove that the functional relationship
    between the vertical coordinate y and the horizontal coordinate x of its vertex
    is y = x^2 - 3x + 1, regardless of the value of m. -/
theorem parabola_vertex_relationship (m x y : ℝ) :
  y = x^2 - 2*m*x + 2*m^2 - 3*m + 1 →
  (x = m ∧ y = m^2 - 3*m + 1) →
  y = x^2 - 3*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_relationship_l2307_230760


namespace NUMINAMATH_CALUDE_floor_area_closest_to_160000_l2307_230792

def hand_length : ℝ := 20

def floor_width (hl : ℝ) : ℝ := 18 * hl
def floor_length (hl : ℝ) : ℝ := 22 * hl

def floor_area (w l : ℝ) : ℝ := w * l

def closest_area : ℝ := 160000

theorem floor_area_closest_to_160000 :
  ∀ (ε : ℝ), ε > 0 →
  |floor_area (floor_width hand_length) (floor_length hand_length) - closest_area| < ε →
  ∀ (other_area : ℝ), other_area ≠ closest_area →
  |floor_area (floor_width hand_length) (floor_length hand_length) - other_area| ≥ ε :=
by sorry

end NUMINAMATH_CALUDE_floor_area_closest_to_160000_l2307_230792


namespace NUMINAMATH_CALUDE_brians_net_commission_l2307_230709

def house_price_1 : ℝ := 157000
def house_price_2 : ℝ := 499000
def house_price_3 : ℝ := 125000
def house_price_4 : ℝ := 275000
def house_price_5 : ℝ := 350000

def commission_rate_1 : ℝ := 0.025
def commission_rate_2 : ℝ := 0.018
def commission_rate_3 : ℝ := 0.02
def commission_rate_4 : ℝ := 0.022
def commission_rate_5 : ℝ := 0.023

def administrative_fee : ℝ := 500

def total_commission : ℝ := 
  house_price_1 * commission_rate_1 +
  house_price_2 * commission_rate_2 +
  house_price_3 * commission_rate_3 +
  house_price_4 * commission_rate_4 +
  house_price_5 * commission_rate_5

def net_commission : ℝ := total_commission - administrative_fee

theorem brians_net_commission : 
  net_commission = 29007 := by sorry

end NUMINAMATH_CALUDE_brians_net_commission_l2307_230709


namespace NUMINAMATH_CALUDE_best_of_three_prob_l2307_230757

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best of 3 sets match -/
def match_win_prob : ℝ := p^2 + 3 * p^2 * (1 - p)

theorem best_of_three_prob : match_win_prob = 0.648 := by sorry

end NUMINAMATH_CALUDE_best_of_three_prob_l2307_230757


namespace NUMINAMATH_CALUDE_product_45_360_trailing_zeros_l2307_230787

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 45 and 360 has 2 trailing zeros -/
theorem product_45_360_trailing_zeros : trailingZeros (45 * 360) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_45_360_trailing_zeros_l2307_230787


namespace NUMINAMATH_CALUDE_largest_divisor_power_l2307_230791

theorem largest_divisor_power (k : ℕ+) : 
  (∀ m : ℕ+, m ≤ k → (1991 : ℤ)^(m : ℕ) ∣ 1990^19911992 + 1992^19911990) ∧ 
  ¬((1991 : ℤ)^((k + 1) : ℕ) ∣ 1990^19911992 + 1992^19911990) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_largest_divisor_power_l2307_230791


namespace NUMINAMATH_CALUDE_round_trip_speed_l2307_230710

/-- Given a round trip between two points A and B, this theorem proves
    that if the distance is 120 miles, the speed from A to B is 60 mph,
    and the average speed for the entire trip is 45 mph, then the speed
    from B to A must be 36 mph. -/
theorem round_trip_speed (d : ℝ) (v_ab : ℝ) (v_avg : ℝ) (v_ba : ℝ) : 
  d = 120 → v_ab = 60 → v_avg = 45 → 
  (2 * d) / (d / v_ab + d / v_ba) = v_avg →
  v_ba = 36 := by
  sorry

#check round_trip_speed

end NUMINAMATH_CALUDE_round_trip_speed_l2307_230710


namespace NUMINAMATH_CALUDE_max_six_yuan_items_l2307_230728

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  twoYuan : ℕ
  fourYuan : ℕ
  sixYuan : ℕ

/-- The problem constraints -/
def isValidPurchase (items : ItemCounts) : Prop :=
  items.twoYuan + items.fourYuan + items.sixYuan = 16 ∧
  2 * items.twoYuan + 4 * items.fourYuan + 6 * items.sixYuan = 60

/-- The theorem stating the maximum number of 6-yuan items -/
theorem max_six_yuan_items :
  ∃ (max : ℕ), max = 7 ∧
  (∀ (items : ItemCounts), isValidPurchase items → items.sixYuan ≤ max) ∧
  (∃ (items : ItemCounts), isValidPurchase items ∧ items.sixYuan = max) := by
  sorry

end NUMINAMATH_CALUDE_max_six_yuan_items_l2307_230728


namespace NUMINAMATH_CALUDE_number_divisible_by_nine_missing_digit_correct_l2307_230758

/-- The missing digit in the five-digit number 385_7 that makes it divisible by 9 -/
def missing_digit : ℕ := 4

/-- The five-digit number with the missing digit filled in -/
def number : ℕ := 38547

theorem number_divisible_by_nine :
  number % 9 = 0 :=
sorry

theorem missing_digit_correct :
  ∃ (d : ℕ), d < 10 ∧ 38500 + d * 10 + 7 = number ∧ (38500 + d * 10 + 7) % 9 = 0 → d = missing_digit :=
sorry

end NUMINAMATH_CALUDE_number_divisible_by_nine_missing_digit_correct_l2307_230758


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2307_230725

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    ((x = -7 ∧ y = -99) ∨ 
     (x = -1 ∧ y = -9) ∨ 
     (x = 1 ∧ y = 5) ∨ 
     (x = 7 ∧ y = -97)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2307_230725


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2307_230783

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/4
  let S := ∑' n, a * r^n
  S = 4/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2307_230783


namespace NUMINAMATH_CALUDE_triangle_angle_B_l2307_230795

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  c = 2 →
  b = 2 * Real.sqrt 3 →
  C = 30 * π / 180 →
  (B = 60 * π / 180 ∨ B = 120 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l2307_230795


namespace NUMINAMATH_CALUDE_apollo_pays_168_apples_l2307_230736

/-- Represents the number of months in a year --/
def months_in_year : ℕ := 12

/-- Represents Hephaestus's charging rate for the first half of the year --/
def hephaestus_rate_first_half : ℕ := 3

/-- Represents Hephaestus's charging rate for the second half of the year --/
def hephaestus_rate_second_half : ℕ := 2 * hephaestus_rate_first_half

/-- Represents Athena's charging rate for the entire year --/
def athena_rate : ℕ := 5

/-- Represents Ares's charging rate for the first 9 months --/
def ares_rate_first_nine : ℕ := 4

/-- Represents Ares's charging rate for the last 3 months --/
def ares_rate_last_three : ℕ := 6

/-- Calculates the total number of golden apples Apollo pays for a year --/
def total_golden_apples : ℕ :=
  (hephaestus_rate_first_half * 6 + hephaestus_rate_second_half * 6) +
  (athena_rate * months_in_year) +
  (ares_rate_first_nine * 9 + ares_rate_last_three * 3)

/-- Theorem stating that the total number of golden apples Apollo pays is 168 --/
theorem apollo_pays_168_apples : total_golden_apples = 168 := by
  sorry

end NUMINAMATH_CALUDE_apollo_pays_168_apples_l2307_230736


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2307_230762

def integer_range : List Int := List.range 10 |>.map (λ x => x - 3)

theorem arithmetic_mean_of_range : 
  (integer_range.sum : ℚ) / integer_range.length = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_l2307_230762


namespace NUMINAMATH_CALUDE_counterexample_ten_l2307_230763

theorem counterexample_ten : 
  ¬(¬(Nat.Prime 10) → Nat.Prime (10 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_ten_l2307_230763


namespace NUMINAMATH_CALUDE_inequality_proof_l2307_230717

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2307_230717


namespace NUMINAMATH_CALUDE_vector_range_l2307_230769

/-- Given unit vectors i and j along x and y axes respectively, and a vector a satisfying 
    |a - i| + |a - 2j| = √5, prove that the range of |a + 2i| is [6√5/5, 3]. -/
theorem vector_range (i j a : ℝ × ℝ) : 
  i = (1, 0) → 
  j = (0, 1) → 
  ‖a - i‖ + ‖a - 2 • j‖ = Real.sqrt 5 → 
  6 * Real.sqrt 5 / 5 ≤ ‖a + 2 • i‖ ∧ ‖a + 2 • i‖ ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_range_l2307_230769


namespace NUMINAMATH_CALUDE_sum_reciprocals_l2307_230721

theorem sum_reciprocals (a b c d e : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 → e ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) + 1 / (e + ω) : ℂ) = 3 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) + 1 / (e + 1) : ℝ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l2307_230721


namespace NUMINAMATH_CALUDE_parabola_directrix_l2307_230743

/-- Given a parabola y = -4x^2 + 8x - 1, its directrix is y = 49/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = -4 * x^2 + 8 * x - 1 →
  ∃ (k : ℝ), k = 49/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = -4 * x₀^2 + 8 * x₀ - 1 →
    ∃ (x₁ : ℝ), (x₀ - x₁)^2 + (y₀ - k)^2 = (y₀ - k)^2 / 4) :=
by sorry


end NUMINAMATH_CALUDE_parabola_directrix_l2307_230743


namespace NUMINAMATH_CALUDE_rectangle_division_l2307_230748

theorem rectangle_division (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (area1 : a * b = 18) (area2 : a * c = 27) (area3 : b * d = 12) :
  c * d = 93 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l2307_230748


namespace NUMINAMATH_CALUDE_parabola_theorem_l2307_230755

/-- Parabola passing through two points with specific properties -/
def Parabola (a b : ℝ) : Prop :=
  (a * 1^2 + b * 1 + 1 = -2) ∧ 
  (a * (-2)^2 + b * (-2) + 1 = 13)

/-- Two points on the parabola with a specific relationship -/
def ParabolaPoints (a b m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ,
    (a * 5^2 + b * 5 + 1 = y₁) ∧
    (a * m^2 + b * m + 1 = y₂) ∧
    (y₂ = 12 - y₁)

/-- Main theorem -/
theorem parabola_theorem :
  ∀ a b m : ℝ, Parabola a b → ParabolaPoints a b m →
    (a = 1 ∧ b = -4 ∧ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_theorem_l2307_230755


namespace NUMINAMATH_CALUDE_admission_difference_theorem_l2307_230754

/-- Represents the admission plan for a university -/
structure AdmissionPlan where
  firstTier : ℕ
  secondTier : ℕ
  thirdTier : ℕ
  ratio : firstTier + secondTier + thirdTier > 0
  firstRatio : firstTier * 5 = secondTier * 2
  secondRatio : secondTier * 3 = thirdTier * 5

/-- The admission difference between second-tier and first-tier universities -/
def admissionDifference (plan : AdmissionPlan) : ℕ :=
  plan.secondTier - plan.firstTier

/-- Theorem stating the admission difference for the given conditions -/
theorem admission_difference_theorem (plan : AdmissionPlan) 
    (h : plan.thirdTier = 1500) : 
    admissionDifference plan = 1500 := by
  sorry

#check admission_difference_theorem

end NUMINAMATH_CALUDE_admission_difference_theorem_l2307_230754


namespace NUMINAMATH_CALUDE_unique_solution_l2307_230732

/-- The system of equations for a given n ≥ 4 -/
def system_of_equations (n : ℕ) (a : ℕ → ℝ) : Prop :=
  n ≥ 4 ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ 2*n → a i > 0) ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ n → 
    a (2*i - 1) = 1 / a (2*n) + 1 / a (2*i) ∧
    a (2*i) = a (2*i - 1) + a (2*i + 1)) ∧
  a (2*n) = a (2*n - 1) + a 1

/-- The solution sequence for a given n ≥ 4 -/
def solution_sequence (n : ℕ) (a : ℕ → ℝ) : Prop :=
  n ≥ 4 ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ 2*n → 
    (i % 2 = 1 → a i = 1) ∧
    (i % 2 = 0 → a i = 2))

/-- The theorem stating that the solution_sequence is the only solution to the system_of_equations -/
theorem unique_solution (n : ℕ) (a : ℕ → ℝ) :
  system_of_equations n a ↔ solution_sequence n a :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2307_230732


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2307_230747

/-- The cost relationships between fruits at Minnie's Orchard -/
structure FruitCosts where
  banana_pear : ℚ  -- ratio of bananas to pears
  pear_apple : ℚ   -- ratio of pears to apples
  apple_orange : ℚ -- ratio of apples to oranges

/-- The number of oranges equivalent in cost to a given number of bananas -/
def bananas_to_oranges (costs : FruitCosts) (num_bananas : ℚ) : ℚ :=
  num_bananas * costs.banana_pear * costs.pear_apple * costs.apple_orange

/-- Theorem stating that 80 bananas are equivalent in cost to 18 oranges -/
theorem banana_orange_equivalence (costs : FruitCosts) 
  (h1 : costs.banana_pear = 4/5)
  (h2 : costs.pear_apple = 3/8)
  (h3 : costs.apple_orange = 9/12) :
  bananas_to_oranges costs 80 = 18 := by
  sorry

#eval bananas_to_oranges ⟨4/5, 3/8, 9/12⟩ 80

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2307_230747


namespace NUMINAMATH_CALUDE_log_relation_l2307_230768

theorem log_relation (a b : ℝ) (h1 : a = Real.log 256 / Real.log 4) (h2 : b = Real.log 27 / Real.log 3) :
  a = (4/3) * b := by sorry

end NUMINAMATH_CALUDE_log_relation_l2307_230768


namespace NUMINAMATH_CALUDE_train_length_l2307_230784

/-- Given a train crossing a bridge, calculate its length. -/
theorem train_length (train_speed : Real) (crossing_time : Real) (bridge_length : Real) :
  train_speed = 72 → -- km/hr
  crossing_time = 12.299016078713702 → -- seconds
  bridge_length = 136 → -- meters
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 110.98032157427404 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2307_230784


namespace NUMINAMATH_CALUDE_expression_evaluation_l2307_230742

theorem expression_evaluation : 
  let c : ℝ := 2
  let d : ℝ := 1/4
  (Real.sqrt (c - d) / (c^2 * Real.sqrt (2*c))) * 
  (Real.sqrt ((c - d)/(c + d)) + Real.sqrt ((c^2 + c*d)/(c^2 - c*d))) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2307_230742


namespace NUMINAMATH_CALUDE_starters_with_twin_restriction_l2307_230767

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of players in the team -/
def total_players : ℕ := 16

/-- The number of starters to be chosen -/
def starters : ℕ := 5

/-- The number of players excluding both sets of twins -/
def players_excluding_twins : ℕ := total_players - 4

/-- The number of ways to choose starters with the twin restriction -/
def ways_to_choose_starters : ℕ :=
  binomial total_players starters -
  2 * binomial (total_players - 2) (starters - 2) +
  binomial (total_players - 4) (starters - 4)

theorem starters_with_twin_restriction :
  ways_to_choose_starters = 3652 :=
sorry

end NUMINAMATH_CALUDE_starters_with_twin_restriction_l2307_230767


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2307_230764

-- Problem 1
theorem problem_1 : (-6) + (-13) = -19 := by sorry

-- Problem 2
theorem problem_2 : (3 : ℚ) / 5 + (-3 / 4) = -3 / 20 := by sorry

-- Problem 3
theorem problem_3 : (4.7 : ℝ) + (-0.8) + 5.3 + (-8.2) = 1 := by sorry

-- Problem 4
theorem problem_4 : (-1 : ℚ) / 6 + 1 / 3 + (-1 / 12) = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2307_230764


namespace NUMINAMATH_CALUDE_max_b_value_l2307_230712

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) 
  (h_order : 1 < c ∧ c < b ∧ b < a) (h_prime : Nat.Prime c) :
  b ≤ 12 ∧ ∃ (a' b' c' : ℕ), a' * b' * c' = 360 ∧ 1 < c' ∧ c' < b' ∧ b' < a' ∧ Nat.Prime c' ∧ b' = 12 :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l2307_230712


namespace NUMINAMATH_CALUDE_timothys_journey_speed_l2307_230726

/-- Proves that given the conditions of Timothy's journey, his average speed during the first part was 10 mph. -/
theorem timothys_journey_speed (v : ℝ) (T : ℝ) (h1 : T > 0) :
  v * (0.25 * T) + 50 * (0.75 * T) = 40 * T →
  v = 10 := by
  sorry

end NUMINAMATH_CALUDE_timothys_journey_speed_l2307_230726


namespace NUMINAMATH_CALUDE_parabola_max_area_l2307_230701

/-- A parabola with y-axis symmetry -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The equation of a parabola -/
def Parabola.equation (p : Parabola) (x : ℝ) : ℝ := p.a * x^2 + p.c

/-- The condition that the parabola is concave up -/
def Parabola.concaveUp (p : Parabola) : Prop := p.a > 0

/-- The condition that the parabola touches the graph y = 1 - |x| -/
def Parabola.touchesGraph (p : Parabola) : Prop :=
  ∃ x₀ : ℝ, p.equation x₀ = 1 - |x₀| ∧ 
    (deriv p.equation) x₀ = if x₀ ≥ 0 then -1 else 1

/-- The area between the parabola and the x-axis -/
noncomputable def Parabola.area (p : Parabola) : ℝ :=
  ∫ x in (-Real.sqrt (1/p.a))..(Real.sqrt (1/p.a)), p.equation x

/-- The theorem statement -/
theorem parabola_max_area :
  ∀ p : Parabola, 
    p.concaveUp → 
    p.touchesGraph → 
    p.area ≤ Parabola.area ⟨1, 3/4⟩ :=
sorry

end NUMINAMATH_CALUDE_parabola_max_area_l2307_230701


namespace NUMINAMATH_CALUDE_inverse_between_zero_and_one_l2307_230713

theorem inverse_between_zero_and_one (x : ℝ) : 0 < (1 : ℝ) / x ∧ (1 : ℝ) / x < 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_between_zero_and_one_l2307_230713


namespace NUMINAMATH_CALUDE_total_spent_equals_79_09_l2307_230788

def shorts_price : Float := 15.00
def jacket_price : Float := 14.82
def shirt_price : Float := 12.51
def shoes_price : Float := 21.67
def hat_price : Float := 8.75
def belt_price : Float := 6.34

def total_spent : Float := shorts_price + jacket_price + shirt_price + shoes_price + hat_price + belt_price

theorem total_spent_equals_79_09 : total_spent = 79.09 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_79_09_l2307_230788


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2307_230716

theorem triangle_angle_measure (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h1 : Real.sin B ^ 2 - Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sqrt 3 * Real.sin A * Real.sin C) : 
  B = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2307_230716


namespace NUMINAMATH_CALUDE_vector_collinear_same_direction_l2307_230790

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have the same direction if their corresponding components have the same sign -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 ≥ 0) ∧ (a.2 * b.2 ≥ 0)

/-- The theorem statement -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b ∧ same_direction a b → x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinear_same_direction_l2307_230790


namespace NUMINAMATH_CALUDE_difference_d_minus_b_l2307_230761

theorem difference_d_minus_b (a b c d : ℕ+) 
  (h1 : a^5 = b^4) 
  (h2 : c^3 = d^2) 
  (h3 : c - a = 19) : 
  d - b = 757 := by sorry

end NUMINAMATH_CALUDE_difference_d_minus_b_l2307_230761


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_l2307_230793

/-- Represents different types of statistical graphs -/
inductive StatGraph
  | PieChart
  | LineChart
  | BarChart

/-- Represents a substance composed of various components -/
structure Substance where
  components : List String

/-- Determines if a statistical graph is suitable for representing the composition of a substance -/
def is_suitable (graph : StatGraph) (substance : Substance) : Prop :=
  match graph with
  | StatGraph.PieChart => substance.components.length > 1
  | _ => False

/-- Air is a substance composed of various gases -/
def air : Substance :=
  { components := ["nitrogen", "oxygen", "argon", "carbon dioxide", "other gases"] }

/-- Theorem stating that a pie chart is the most suitable graph for representing air composition -/
theorem pie_chart_most_suitable_for_air :
  is_suitable StatGraph.PieChart air ∧
  ∀ (graph : StatGraph), graph ≠ StatGraph.PieChart → ¬(is_suitable graph air) :=
sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_air_l2307_230793


namespace NUMINAMATH_CALUDE_max_product_of_roots_l2307_230766

/-- Given a quadratic equation 5x^2 - 10x + m = 0 with real roots,
    the maximum value of the product of its roots is 1. -/
theorem max_product_of_roots :
  ∀ m : ℝ,
  (∃ x : ℝ, 5 * x^2 - 10 * x + m = 0) →
  (∀ k : ℝ, (∃ x : ℝ, 5 * x^2 - 10 * x + k = 0) → m / 5 ≥ k / 5) →
  m / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_roots_l2307_230766


namespace NUMINAMATH_CALUDE_mitzi_remaining_money_l2307_230735

def amusement_park_spending (initial_amount ticket_cost food_cost tshirt_cost : ℕ) : ℕ :=
  initial_amount - (ticket_cost + food_cost + tshirt_cost)

theorem mitzi_remaining_money :
  amusement_park_spending 75 30 13 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mitzi_remaining_money_l2307_230735


namespace NUMINAMATH_CALUDE_expression_minimum_l2307_230780

theorem expression_minimum (x : ℝ) (h : 1 < x ∧ x < 5) : 
  ∃ (y : ℝ), y = (x^2 - 4*x + 5) / (2*x - 6) ∧ 
  (∀ (z : ℝ), 1 < z ∧ z < 5 → (z^2 - 4*z + 5) / (2*z - 6) ≥ y) ∧
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_minimum_l2307_230780


namespace NUMINAMATH_CALUDE_count_sets_satisfying_union_l2307_230706

theorem count_sets_satisfying_union (A B : Set ℕ) : 
  A = {1, 2} → 
  (A ∪ B = {1, 2, 3, 4, 5}) → 
  (∃! (count : ℕ), ∃ (S : Finset (Set ℕ)), 
    (Finset.card S = count) ∧ 
    (∀ C ∈ S, A ∪ C = {1, 2, 3, 4, 5}) ∧
    (∀ D, A ∪ D = {1, 2, 3, 4, 5} → D ∈ S) ∧
    count = 4) :=
by sorry

end NUMINAMATH_CALUDE_count_sets_satisfying_union_l2307_230706


namespace NUMINAMATH_CALUDE_rectangle_ratio_in_square_config_l2307_230778

-- Define the structure of our square-rectangle configuration
structure SquareRectConfig where
  inner_side : ℝ
  rect_short : ℝ
  rect_long : ℝ

-- State the theorem
theorem rectangle_ratio_in_square_config (config : SquareRectConfig) :
  -- The outer square's side is composed of the inner square's side and two short sides of rectangles
  config.inner_side + 2 * 2 * config.rect_short = 3 * config.inner_side →
  -- Two long sides and one short side of rectangles make up the outer square's side
  2 * config.rect_long + config.rect_short = 3 * config.inner_side →
  -- The ratio of long to short sides of the rectangle is 2.5
  config.rect_long / config.rect_short = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_in_square_config_l2307_230778


namespace NUMINAMATH_CALUDE_sum_of_occurrences_l2307_230765

theorem sum_of_occurrences (a₀ a₁ a₂ a₃ a₄ : ℕ) 
  (sum_constraint : a₀ + a₁ + a₂ + a₃ + a₄ = 5)
  (value_constraint : 0*a₀ + 1*a₁ + 2*a₂ + 3*a₃ + 4*a₄ = 5) :
  a₀ + a₁ + a₂ + a₃ = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_occurrences_l2307_230765


namespace NUMINAMATH_CALUDE_probability_same_color_l2307_230781

def num_green_balls : ℕ := 7
def num_white_balls : ℕ := 7

def total_balls : ℕ := num_green_balls + num_white_balls

def same_color_combinations : ℕ := (num_green_balls.choose 2) + (num_white_balls.choose 2)
def total_combinations : ℕ := total_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 42 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_l2307_230781


namespace NUMINAMATH_CALUDE_sin_symmetric_angles_l2307_230770

def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, α + β = Real.pi + 2 * k * Real.pi

theorem sin_symmetric_angles (α β : Real) 
  (h_symmetric : symmetric_angles α β) (h_sin_α : Real.sin α = 1/3) : 
  Real.sin β = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_symmetric_angles_l2307_230770


namespace NUMINAMATH_CALUDE_floor_x_floor_x_equals_20_l2307_230731

theorem floor_x_floor_x_equals_20 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by sorry

end NUMINAMATH_CALUDE_floor_x_floor_x_equals_20_l2307_230731


namespace NUMINAMATH_CALUDE_square_of_105_l2307_230752

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l2307_230752


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_36_l2307_230733

theorem consecutive_integers_sum_36 : 
  ∃! (a : ℕ), a > 0 ∧ a + (a + 1) + (a + 2) = 36 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_36_l2307_230733


namespace NUMINAMATH_CALUDE_tiles_per_row_l2307_230785

-- Define the area of the room in square feet
def room_area : ℝ := 324

-- Define the side length of a tile in inches
def tile_side : ℝ := 9

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem tiles_per_row : 
  ⌊(feet_to_inches * Real.sqrt room_area) / tile_side⌋ = 24 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l2307_230785


namespace NUMINAMATH_CALUDE_problem_statements_l2307_230782

theorem problem_statements :
  (∀ x : ℤ, x^2 + 1 > 0) ∧
  (∃ x y : ℝ, x + y > 5 ∧ ¬(x > 2 ∧ y > 3)) ∧
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ y : ℝ, y ≤ 3 → ∃ x : ℝ, y = -x^2 + 2*x + 2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 2 ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l2307_230782


namespace NUMINAMATH_CALUDE_blake_change_l2307_230704

/-- The amount Blake spends on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spends on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spends on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake has -/
def initial_amount : ℕ := 300

/-- The change given to Blake -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by
  sorry

end NUMINAMATH_CALUDE_blake_change_l2307_230704


namespace NUMINAMATH_CALUDE_triangle_properties_l2307_230759

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the main theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : Real.tan abc.B + Real.tan abc.C = (2 * Real.sin abc.A) / Real.cos abc.C)
  (h2 : abc.a = abc.c + 2)
  (h3 : ∃ θ : Real, θ > π / 2 ∧ (θ = abc.A ∨ θ = abc.B ∨ θ = abc.C)) :
  abc.B = π / 3 ∧ (0 < abc.c ∧ abc.c < 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2307_230759


namespace NUMINAMATH_CALUDE_profit_distribution_l2307_230711

theorem profit_distribution (share_a share_b share_c : ℕ) (total_profit : ℕ) : 
  share_a + share_b + share_c = total_profit →
  2 * share_a = 3 * share_b →
  3 * share_b = 5 * share_c →
  share_c - share_b = 4000 →
  total_profit = 20000 := by
sorry

end NUMINAMATH_CALUDE_profit_distribution_l2307_230711


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_6_l2307_230741

/-- A function f(x) = x^2 - 2(a-1)x + 2 is decreasing on the interval (-∞, 5] -/
def is_decreasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 5 → y ≤ 5 → f x ≥ f y

/-- The quadratic function f(x) = x^2 - 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a-1)*x + 2

theorem decreasing_quadratic_implies_a_geq_6 :
  ∀ a : ℝ, is_decreasing_on_interval (f a) a → a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_6_l2307_230741


namespace NUMINAMATH_CALUDE_candied_grape_price_l2307_230777

-- Define the number of candied apples
def num_apples : ℕ := 15

-- Define the price of each candied apple
def price_apple : ℚ := 2

-- Define the number of candied grapes
def num_grapes : ℕ := 12

-- Define the total revenue
def total_revenue : ℚ := 48

-- Define the price of each candied grape
def price_grape : ℚ := 1.5

theorem candied_grape_price :
  price_grape * num_grapes + price_apple * num_apples = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_candied_grape_price_l2307_230777


namespace NUMINAMATH_CALUDE_multiples_properties_l2307_230727

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ k : ℤ, b = 8 * k) : 
  (∃ k : ℤ, b = 4 * k) ∧ 
  (∃ k : ℤ, a - b = 4 * k) ∧ 
  (∃ k : ℤ, a + b = 2 * k) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l2307_230727


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2307_230786

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2307_230786


namespace NUMINAMATH_CALUDE_winston_gas_tank_capacity_l2307_230746

/-- Represents the gas tank of Winston's car -/
structure GasTank where
  initialGas : ℕ
  usedToStore : ℕ
  usedToDoctor : ℕ
  neededToRefill : ℕ

/-- Calculates the maximum capacity of the gas tank -/
def maxCapacity (tank : GasTank) : ℕ :=
  tank.initialGas - tank.usedToStore - tank.usedToDoctor + tank.neededToRefill

/-- Theorem stating that the maximum capacity of Winston's gas tank is 12 gallons -/
theorem winston_gas_tank_capacity :
  let tank : GasTank := {
    initialGas := 10,
    usedToStore := 6,
    usedToDoctor := 2,
    neededToRefill := 10
  }
  maxCapacity tank = 12 := by sorry

end NUMINAMATH_CALUDE_winston_gas_tank_capacity_l2307_230746


namespace NUMINAMATH_CALUDE_single_elimination_games_l2307_230789

/-- A single-elimination tournament with no ties. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games played in a single-elimination tournament. -/
def games_played (t : Tournament) : ℕ :=
  t.num_teams - 1

theorem single_elimination_games (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.no_ties = true) : 
  games_played t = 22 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_games_l2307_230789


namespace NUMINAMATH_CALUDE_no_triple_perfect_squares_l2307_230773

theorem no_triple_perfect_squares (n : ℕ+) : 
  ¬(∃ a b c : ℕ, (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_perfect_squares_l2307_230773


namespace NUMINAMATH_CALUDE_circle_area_circumference_difference_l2307_230722

theorem circle_area_circumference_difference (a b c : ℝ) (h1 : a = 24) (h2 : b = 70) (h3 : c = 74) 
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) : 
  let r := c / 2
  (π * r ^ 2) - (2 * π * r) = 1295 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_circumference_difference_l2307_230722


namespace NUMINAMATH_CALUDE_increasing_sequence_count_remainder_mod_1000_l2307_230739

def sequence_count (n : ℕ) (k : ℕ) (max : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

theorem increasing_sequence_count : 
  sequence_count 998 12 2008 = Nat.choose 1009 12 :=
sorry

theorem remainder_mod_1000 : 
  1009 % 1000 = 9 :=
sorry

end NUMINAMATH_CALUDE_increasing_sequence_count_remainder_mod_1000_l2307_230739


namespace NUMINAMATH_CALUDE_road_travel_cost_l2307_230719

/-- Calculate the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 40)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 3) :
  (((lawn_length * road_width + lawn_width * road_width) - road_width * road_width) : ℚ) * cost_per_sqm = 3300 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l2307_230719


namespace NUMINAMATH_CALUDE_infinitely_many_common_divisors_l2307_230744

theorem infinitely_many_common_divisors :
  Set.Infinite {n : ℕ | ∃ d : ℕ, d > 1 ∧ d ∣ (2*n - 3) ∧ d ∣ (3*n - 2)} :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_common_divisors_l2307_230744


namespace NUMINAMATH_CALUDE_area_ratio_is_one_seventh_l2307_230772

/-- Given a triangle XYZ with sides XY, YZ, XZ and points P on XY and Q on XZ,
    this function calculates the ratio of the area of triangle XPQ to the area of quadrilateral PQYZ -/
def areaRatio (XY YZ XZ XP XQ : ℝ) : ℝ :=
  -- Define the ratio calculation here
  sorry

/-- Theorem stating that for the given triangle and points, the area ratio is 1/7 -/
theorem area_ratio_is_one_seventh :
  areaRatio 24 52 60 12 20 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_seventh_l2307_230772


namespace NUMINAMATH_CALUDE_function_property_l2307_230740

theorem function_property (f : ℕ+ → ℝ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ (n1 n2 : ℕ+), f (n1 + n2) = f n1 * f n2) : 
  ∀ (n : ℕ+), f n = 2^(n:ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2307_230740


namespace NUMINAMATH_CALUDE_sin_2x_minus_y_equals_neg_one_l2307_230700

-- Define the equations as functions
def equation1 (x : ℝ) : Prop := x + Real.sin x * Real.cos x - 1 = 0
def equation2 (y : ℝ) : Prop := 2 * Real.cos y - 2 * y + Real.pi + 4 = 0

-- State the theorem
theorem sin_2x_minus_y_equals_neg_one (x y : ℝ) 
  (h1 : equation1 x) (h2 : equation2 y) : 
  Real.sin (2 * x - y) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_2x_minus_y_equals_neg_one_l2307_230700


namespace NUMINAMATH_CALUDE_prime_extension_l2307_230797

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l2307_230797


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l2307_230723

/-- The total path length of a vertex of an equilateral triangle rotating inside a square --/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotation_angle : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotation_angle = 60 * π / 180) : 
  (4 : ℝ) * 3 * triangle_side * rotation_angle = 12 * π := by
  sorry

#check triangle_rotation_path_length

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l2307_230723


namespace NUMINAMATH_CALUDE_isabellas_final_hair_length_l2307_230794

/-- The final hair length given an initial length and growth --/
def finalHairLength (initialLength growth : ℕ) : ℕ :=
  initialLength + growth

/-- Theorem: Isabella's final hair length is 24 inches --/
theorem isabellas_final_hair_length :
  finalHairLength 18 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_final_hair_length_l2307_230794
