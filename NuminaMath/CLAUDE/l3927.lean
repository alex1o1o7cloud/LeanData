import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_f_l3927_392727

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) 
  (pos₁ : 0 < a₁) (pos₂ : 0 < a₂) (pos₃ : 0 < a₃) (pos₄ : 0 < a₄)
  (sum_cond : a₁ + 2*a₂ + 3*a₃ + 4*a₄ ≤ 10)
  (lower_bound₁ : a₁ ≥ 1/8) (lower_bound₂ : a₂ ≥ 1/4)
  (lower_bound₃ : a₃ ≥ 1/2) (lower_bound₄ : a₄ ≥ 1) : 
  1/(1 + a₁) + 1/(1 + a₂^2) + 1/(1 + a₃^3) + 1/(1 + a₄^4) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3927_392727


namespace NUMINAMATH_CALUDE_alternate_seating_four_boys_three_girls_l3927_392733

/-- The number of ways to seat 4 boys and 3 girls in a row alternately -/
def alternate_seating (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys = 4 ∧ num_girls = 3 then
    2 * (Nat.factorial num_boys * Nat.factorial num_girls)
  else
    0

theorem alternate_seating_four_boys_three_girls :
  alternate_seating 4 3 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternate_seating_four_boys_three_girls_l3927_392733


namespace NUMINAMATH_CALUDE_expression_evaluation_l3927_392761

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3927_392761


namespace NUMINAMATH_CALUDE_intersection_ordinate_l3927_392772

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 - 3

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point of the parabola and the y-axis -/
def intersection_point (x y : ℝ) : Prop := parabola x y ∧ y_axis x

theorem intersection_ordinate :
  ∃ x y : ℝ, intersection_point x y ∧ y = -5 := by sorry

end NUMINAMATH_CALUDE_intersection_ordinate_l3927_392772


namespace NUMINAMATH_CALUDE_rectangle_area_l3927_392707

theorem rectangle_area (square_area : ℝ) (rectangle_length_multiplier : ℝ) : 
  square_area = 36 → 
  rectangle_length_multiplier = 3 → 
  let square_side := Real.sqrt square_area
  let rectangle_width := square_side
  let rectangle_length := rectangle_length_multiplier * rectangle_width
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3927_392707


namespace NUMINAMATH_CALUDE_indoor_tables_count_l3927_392788

/-- The number of indoor tables in a bakery. -/
def num_indoor_tables : ℕ := sorry

/-- The number of outdoor tables in a bakery. -/
def num_outdoor_tables : ℕ := 12

/-- The number of chairs per indoor table. -/
def chairs_per_indoor_table : ℕ := 3

/-- The number of chairs per outdoor table. -/
def chairs_per_outdoor_table : ℕ := 3

/-- The total number of chairs in the bakery. -/
def total_chairs : ℕ := 60

/-- Theorem stating that the number of indoor tables is 8. -/
theorem indoor_tables_count : num_indoor_tables = 8 := by
  sorry

end NUMINAMATH_CALUDE_indoor_tables_count_l3927_392788


namespace NUMINAMATH_CALUDE_scalar_cross_product_sum_l3927_392797

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem scalar_cross_product_sum (a b c d : V) (h : a + b + c + d = 0) :
  ∃! k : ℝ, ∀ (a b c d : V), a + b + c + d = 0 →
    k • (cross_product c b) + cross_product b c + cross_product c a + 
    cross_product a d + cross_product d d = 0 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_scalar_cross_product_sum_l3927_392797


namespace NUMINAMATH_CALUDE_modular_difference_in_range_l3927_392701

theorem modular_difference_in_range (a b : ℤ) : 
  a ≡ 25 [ZMOD 60] →
  b ≡ 84 [ZMOD 60] →
  ∃! n : ℤ, 150 ≤ n ∧ n ≤ 200 ∧ a - b ≡ n [ZMOD 60] ∧ n = 181 :=
by sorry

end NUMINAMATH_CALUDE_modular_difference_in_range_l3927_392701


namespace NUMINAMATH_CALUDE_instrument_purchase_plan_l3927_392755

-- Define the cost prices of instruments A and B
def cost_A : ℕ := 400
def cost_B : ℕ := 300

-- Define the selling prices of instruments A and B
def sell_A : ℕ := 760
def sell_B : ℕ := 540

-- Define the function for the number of B given A
def num_B (a : ℕ) : ℕ := 3 * a + 10

-- Define the total cost function
def total_cost (a : ℕ) : ℕ := cost_A * a + cost_B * (num_B a)

-- Define the profit function
def profit (a : ℕ) : ℕ := (sell_A - cost_A) * a + (sell_B - cost_B) * (num_B a)

-- Theorem statement
theorem instrument_purchase_plan :
  (2 * cost_A + 3 * cost_B = 1700) ∧
  (3 * cost_A + cost_B = 1500) ∧
  (∀ a : ℕ, total_cost a ≤ 30000 → profit a ≥ 21600 → 
    (a = 18 ∧ num_B a = 64) ∨ 
    (a = 19 ∧ num_B a = 67) ∨ 
    (a = 20 ∧ num_B a = 70)) :=
by sorry

end NUMINAMATH_CALUDE_instrument_purchase_plan_l3927_392755


namespace NUMINAMATH_CALUDE_jack_pounds_l3927_392740

/-- Proves that Jack has 42 pounds given the problem conditions -/
theorem jack_pounds : 
  ∀ (p : ℝ) (e : ℝ) (y : ℝ),
  e = 11 →
  y = 3000 →
  2 * e + p + y / 100 = 9400 / 100 →
  p = 42 := by
  sorry


end NUMINAMATH_CALUDE_jack_pounds_l3927_392740


namespace NUMINAMATH_CALUDE_walkway_area_is_296_l3927_392723

/-- Represents a garden with flower beds and walkways -/
structure Garden where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  walkway_width : Nat

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : Nat :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden is 296 square feet -/
theorem walkway_area_is_296 (g : Garden) 
  (h_rows : g.rows = 4)
  (h_columns : g.columns = 3)
  (h_bed_width : g.bed_width = 4)
  (h_bed_height : g.bed_height = 3)
  (h_walkway_width : g.walkway_width = 2) :
  walkway_area g = 296 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_296_l3927_392723


namespace NUMINAMATH_CALUDE_sugar_price_inconsistency_l3927_392745

/-- Represents the price and consumption change of sugar -/
structure SugarPriceChange where
  p₀ : ℝ  -- Initial price
  p₁ : ℝ  -- New price
  r : ℝ   -- Reduction in consumption (as a decimal)

/-- Checks if the given sugar price change is consistent -/
def is_consistent (s : SugarPriceChange) : Prop :=
  s.r = (s.p₁ - s.p₀) / s.p₁

/-- Theorem stating that the given conditions are inconsistent -/
theorem sugar_price_inconsistency :
  ¬ is_consistent ⟨3, 5, 0.4⟩ := by
  sorry

end NUMINAMATH_CALUDE_sugar_price_inconsistency_l3927_392745


namespace NUMINAMATH_CALUDE_football_team_analysis_l3927_392746

structure FootballTeam where
  total_matches : Nat
  played_matches : Nat
  lost_matches : Nat
  points : Nat

def win_points : Nat := 3
def draw_points : Nat := 1
def loss_points : Nat := 0

def team : FootballTeam := {
  total_matches := 14,
  played_matches := 8,
  lost_matches := 1,
  points := 17
}

def wins_in_first_8 (t : FootballTeam) : Nat :=
  (t.points - (t.played_matches - t.lost_matches - 1)) / 2

def max_possible_points (t : FootballTeam) : Nat :=
  t.points + (t.total_matches - t.played_matches) * win_points

def min_wins_needed (t : FootballTeam) (target : Nat) : Nat :=
  ((target - t.points + 2) / win_points).min (t.total_matches - t.played_matches)

theorem football_team_analysis (t : FootballTeam) :
  wins_in_first_8 t = 5 ∧
  max_possible_points t = 35 ∧
  min_wins_needed t 29 = 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_analysis_l3927_392746


namespace NUMINAMATH_CALUDE_matchsticks_100th_stage_l3927_392794

/-- Represents the number of matchsticks in a stage of the pattern -/
def matchsticks (n : ℕ) : ℕ := 4 + (n - 1) * 4

/-- Proves that the 100th stage of the pattern contains 400 matchsticks -/
theorem matchsticks_100th_stage : matchsticks 100 = 400 := by
  sorry

end NUMINAMATH_CALUDE_matchsticks_100th_stage_l3927_392794


namespace NUMINAMATH_CALUDE_trapezoid_area_l3927_392732

/-- A trapezoid with the given properties -/
structure Trapezoid where
  /-- Length of one diagonal -/
  diagonal1 : ℝ
  /-- Length of the other diagonal -/
  diagonal2 : ℝ
  /-- Length of the segment connecting the midpoints of the bases -/
  midpoint_segment : ℝ
  /-- The first diagonal is 3 -/
  h1 : diagonal1 = 3
  /-- The second diagonal is 5 -/
  h2 : diagonal2 = 5
  /-- The segment connecting the midpoints of the bases is 2 -/
  h3 : midpoint_segment = 2

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := 6

/-- Theorem stating that the area of the trapezoid with the given properties is 6 -/
theorem trapezoid_area (t : Trapezoid) : area t = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3927_392732


namespace NUMINAMATH_CALUDE_rect_to_spherical_l3927_392759

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical (x y z : ℝ) :
  x = 1 ∧ y = Real.sqrt 3 ∧ z = 2 →
  ∃ (ρ θ φ : ℝ),
    ρ > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    ρ = 3 ∧
    θ = Real.pi / 3 ∧
    φ = Real.arccos (2/3) ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    z = ρ * Real.cos φ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_spherical_l3927_392759


namespace NUMINAMATH_CALUDE_parabola_with_directrix_y_2_l3927_392753

/-- Represents a parabola in 2D space -/
structure Parabola where
  /-- The equation of the parabola in the form x² = ky, where k is a non-zero real number -/
  equation : ℝ → ℝ → Prop

/-- Represents the directrix of a parabola -/
structure Directrix where
  /-- The y-coordinate of the horizontal directrix -/
  y : ℝ

/-- 
Given a parabola with a horizontal directrix y = 2, 
prove that its standard equation is x² = -8y 
-/
theorem parabola_with_directrix_y_2 (p : Parabola) (d : Directrix) :
  d.y = 2 → p.equation = fun x y ↦ x^2 = -8*y := by
  sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_y_2_l3927_392753


namespace NUMINAMATH_CALUDE_max_value_theorem_l3927_392743

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*b*c*Real.sqrt 3 ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3927_392743


namespace NUMINAMATH_CALUDE_digit_2023_of_17_19_l3927_392741

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit (n d k : ℕ) : ℕ := sorry

theorem digit_2023_of_17_19 : nth_digit 17 19 2023 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_2023_of_17_19_l3927_392741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3927_392779

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is the arithmetic sequence
  (S : ℕ → ℝ)  -- S is the sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- Given condition
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum of arithmetic sequence
  : ∃ d : ℝ, (∀ n, a (n+1) - a n = d) ∧ d = -2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3927_392779


namespace NUMINAMATH_CALUDE_simplify_expression_l3927_392713

theorem simplify_expression (x y : ℝ) : 7*x + 9 - 2*x + 3*y = 5*x + 3*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3927_392713


namespace NUMINAMATH_CALUDE_right_triangle_geometric_sequence_ratio_l3927_392703

theorem right_triangle_geometric_sequence_ratio :
  ∀ (a b c : ℝ),
    a > 0 →
    b > 0 →
    c > 0 →
    a < b →
    b < c →
    a^2 + b^2 = c^2 →
    (∃ r : ℝ, r > 1 ∧ b = a * r ∧ c = a * r^2) →
    c / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_geometric_sequence_ratio_l3927_392703


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3927_392782

open Real

theorem solution_set_characterization 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)
  (h_initial : f 0 = 2)
  (h_bound : ∀ x, f x + f' x > 1) :
  ∀ x, (exp x * f x > exp x + 1) ↔ x > 0 := by
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3927_392782


namespace NUMINAMATH_CALUDE_bucket_full_weight_bucket_full_weight_proof_l3927_392780

/-- Given a bucket with known weights at different fill levels, 
    calculate the weight when completely full. -/
theorem bucket_full_weight (c d : ℝ) : ℝ :=
  let three_fourths_weight := c
  let one_third_weight := d
  let full_weight := (8/5 : ℝ) * c - (3/5 : ℝ) * d
  full_weight

/-- Prove that the calculated full weight is correct -/
theorem bucket_full_weight_proof (c d : ℝ) :
  bucket_full_weight c d = (8/5 : ℝ) * c - (3/5 : ℝ) * d := by
  sorry

end NUMINAMATH_CALUDE_bucket_full_weight_bucket_full_weight_proof_l3927_392780


namespace NUMINAMATH_CALUDE_max_square_plots_l3927_392767

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFencing : ℕ := 1994

/-- Represents the field dimensions -/
def field : FieldDimensions := { width := 24, length := 52 }

/-- Calculates the number of square plots given the number of plots in a column -/
def numPlots (n : ℕ) : ℕ :=
  (13 * n * n) / 6

/-- Calculates the length of internal fencing needed for n plots in a column -/
def fencingNeeded (n : ℕ) : ℕ :=
  104 * n - 76

/-- Theorem stating the maximum number of square test plots -/
theorem max_square_plots :
  ∃ (n : ℕ), n ≤ 18 ∧ 6 ∣ n ∧
  fencingNeeded n ≤ availableFencing ∧
  (∀ (m : ℕ), m > n → fencingNeeded m > availableFencing ∨ ¬(6 ∣ m)) ∧
  numPlots n = 702 :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3927_392767


namespace NUMINAMATH_CALUDE_invertible_elements_mod_8_l3927_392720

theorem invertible_elements_mod_8 :
  ∀ a : ℤ, a ∈ ({1, 3, 5, 7} : Set ℤ) ↔
    (∃ b : ℤ, (a * b) % 8 = 1 ∧ (a * a) % 8 = 1) :=
by sorry

end NUMINAMATH_CALUDE_invertible_elements_mod_8_l3927_392720


namespace NUMINAMATH_CALUDE_wine_bottle_cost_l3927_392784

/-- The cost of a bottle of wine with a cork, given the price of the cork and the price difference between a bottle without a cork and the cork itself. -/
theorem wine_bottle_cost (cork_price : ℝ) (price_difference : ℝ) : 
  cork_price = 0.05 →
  price_difference = 2.00 →
  cork_price + (cork_price + price_difference) = 2.10 :=
by sorry

end NUMINAMATH_CALUDE_wine_bottle_cost_l3927_392784


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3927_392793

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (57890 + N).mod 6 = 0 → N ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l3927_392793


namespace NUMINAMATH_CALUDE_max_factors_b_power_n_l3927_392742

def count_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_b_power_n (b n : ℕ+) (h1 : b ≤ 20) (h2 : n = 10) :
  (∃ (b' : ℕ+), b' ≤ 20 ∧ count_factors b' n = 231) ∧
  (∀ (b' : ℕ+), b' ≤ 20 → count_factors b' n ≤ 231) :=
sorry

end NUMINAMATH_CALUDE_max_factors_b_power_n_l3927_392742


namespace NUMINAMATH_CALUDE_sqrt_sum_expression_l3927_392708

theorem sqrt_sum_expression (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_minimal : ∀ (a' b' c' : ℕ), a' > 0 → b' > 0 → c' > 0 → 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c' = a' * Real.sqrt 3 + b' * Real.sqrt 7 
    → c ≤ c')
  (h_equality : (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c = a * Real.sqrt 3 + b * Real.sqrt 7) :
  a + b + c = 73 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_expression_l3927_392708


namespace NUMINAMATH_CALUDE_apple_count_l3927_392776

/-- Represents the total number of apples -/
def total_apples : ℕ := sorry

/-- Represents the price of a sweet apple in dollars -/
def sweet_price : ℚ := 1/2

/-- Represents the price of a sour apple in dollars -/
def sour_price : ℚ := 1/10

/-- Represents the proportion of sweet apples -/
def sweet_proportion : ℚ := 3/4

/-- Represents the proportion of sour apples -/
def sour_proportion : ℚ := 1/4

/-- Represents the total earnings in dollars -/
def total_earnings : ℚ := 40

theorem apple_count : 
  sweet_proportion * total_apples * sweet_price + 
  sour_proportion * total_apples * sour_price = total_earnings ∧
  total_apples = 100 := by sorry

end NUMINAMATH_CALUDE_apple_count_l3927_392776


namespace NUMINAMATH_CALUDE_num_perfect_square_factors_specific_l3927_392738

/-- The number of positive perfect square integers that are factors of (2^12)(3^10)(5^18)(7^8) -/
def num_perfect_square_factors (a b c d : ℕ) : ℕ :=
  (a / 2 + 1) * (b / 2 + 1) * (c / 2 + 1) * (d / 2 + 1)

/-- Theorem stating that the number of positive perfect square integers
    that are factors of (2^12)(3^10)(5^18)(7^8) is 2100 -/
theorem num_perfect_square_factors_specific :
  num_perfect_square_factors 12 10 18 8 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_num_perfect_square_factors_specific_l3927_392738


namespace NUMINAMATH_CALUDE_cubic_root_theorem_l3927_392775

theorem cubic_root_theorem :
  ∃ (a b c : ℕ+) (x : ℝ),
    a = 1 ∧ b = 9 ∧ c = 1 ∧
    x = (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 1) / c ∧
    27 * x^3 - 9 * x^2 - 9 * x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_theorem_l3927_392775


namespace NUMINAMATH_CALUDE_notebook_cost_l3927_392789

theorem notebook_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 380)
  (eq2 : 3 * x + 6 * y = 354) :
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l3927_392789


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3927_392787

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3927_392787


namespace NUMINAMATH_CALUDE_election_ratio_l3927_392739

theorem election_ratio (total_votes_x : ℝ) (total_votes_y : ℝ)
  (h1 : total_votes_x > 0)
  (h2 : total_votes_y > 0)
  (h3 : 0.62 * total_votes_x + 0.38 * total_votes_y = 0.54 * (total_votes_x + total_votes_y)) :
  total_votes_x / total_votes_y = 2 :=
by sorry

end NUMINAMATH_CALUDE_election_ratio_l3927_392739


namespace NUMINAMATH_CALUDE_difference_of_squares_l3927_392730

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3927_392730


namespace NUMINAMATH_CALUDE_hyperbola_right_angle_triangle_area_l3927_392771

/-- Hyperbola type representing the equation x²/9 - y²/16 = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop := h.equation p.x p.y

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_right_angle_triangle_area 
  (h : Hyperbola) 
  (p : Point) 
  (hP : isOnHyperbola h p) 
  (f1 f2 : Point) 
  (hFoci : foci h = (f1, f2)) 
  (hAngle : angle f1 p f2 = 90) : 
  triangleArea (Triangle.mk f1 p f2) = 16 := by sorry

end NUMINAMATH_CALUDE_hyperbola_right_angle_triangle_area_l3927_392771


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3927_392710

/-- Given a geometric sequence with first term a₁ and common ratio r,
    a_n represents the nth term of the sequence. -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem fifth_term_of_geometric_sequence :
  let a₁ : ℝ := 5
  let r : ℝ := -2
  geometric_sequence a₁ r 5 = 80 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3927_392710


namespace NUMINAMATH_CALUDE_gcd_of_1054_and_986_l3927_392736

theorem gcd_of_1054_and_986 : Nat.gcd 1054 986 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_1054_and_986_l3927_392736


namespace NUMINAMATH_CALUDE_prob_even_odd_is_one_fourth_l3927_392709

/-- Represents a six-sided die -/
def Die := Fin 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even (d : Die) : ℚ := 1/2

/-- The probability of rolling an odd number on a six-sided die -/
def prob_odd (d : Die) : ℚ := 1/2

/-- The probability of rolling an even number on the first die and an odd number on the second die -/
def prob_even_odd (d1 d2 : Die) : ℚ := prob_even d1 * prob_odd d2

theorem prob_even_odd_is_one_fourth (d1 d2 : Die) :
  prob_even_odd d1 d2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_even_odd_is_one_fourth_l3927_392709


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_l3927_392722

/-- A polynomial function of degree 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

/-- Theorem stating that f(-2) = -2 given the conditions -/
theorem f_neg_two_eq_neg_two (a b c : ℝ) :
  (f a b c 5 + f a b c (-5) = 6) →
  (f a b c 2 = 8) →
  f a b c (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_l3927_392722


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3927_392760

theorem polynomial_factorization (y : ℝ) :
  (16 * y^7 - 36 * y^5 + 8 * y) - (4 * y^7 - 12 * y^5 - 8 * y) = 8 * y * (3 * y^6 - 6 * y^4 + 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3927_392760


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3927_392744

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (29*x)/(53*y)) :
  Real.sqrt x / Real.sqrt y = 91/42 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l3927_392744


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l3927_392702

/-- The coordinates of a point on the unit circle after moving counterclockwise from (1,0) by an arc length of 4π/3 -/
theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (∃ θ : ℝ, θ = 4 * Real.pi / 3 ∧ 
   Q.1 = Real.cos θ ∧ 
   Q.2 = Real.sin θ) →
  Q = (-1/2, -Real.sqrt 3 / 2) := by
sorry


end NUMINAMATH_CALUDE_point_on_unit_circle_l3927_392702


namespace NUMINAMATH_CALUDE_two_car_problem_l3927_392749

/-- Proves that given the conditions of the two-car problem, the speeds of cars A and B are 30 km/h and 25 km/h respectively. -/
theorem two_car_problem (distance_A distance_B : ℝ) (speed_difference : ℝ) 
  (h1 : distance_A = 300)
  (h2 : distance_B = 250)
  (h3 : speed_difference = 5)
  (h4 : ∃ (t : ℝ), t > 0 ∧ distance_A / (speed_B + speed_difference) = t ∧ distance_B / speed_B = t) :
  ∃ (speed_A speed_B : ℝ), 
    speed_A = 30 ∧ 
    speed_B = 25 ∧ 
    speed_A = speed_B + speed_difference ∧
    distance_A / speed_A = distance_B / speed_B :=
by
  sorry


end NUMINAMATH_CALUDE_two_car_problem_l3927_392749


namespace NUMINAMATH_CALUDE_max_popsicles_lucy_can_buy_l3927_392786

theorem max_popsicles_lucy_can_buy (lucy_money : ℝ) (popsicle_price : ℝ) :
  lucy_money = 19.23 →
  popsicle_price = 1.60 →
  ∃ n : ℕ, n * popsicle_price ≤ lucy_money ∧
    ∀ m : ℕ, m * popsicle_price ≤ lucy_money → m ≤ n ∧
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_lucy_can_buy_l3927_392786


namespace NUMINAMATH_CALUDE_b_value_l3927_392719

def consecutive_odd_numbers (a b c d e : ℤ) : Prop :=
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 ∧ e = d + 2

theorem b_value (a b c d e : ℤ) 
  (h1 : consecutive_odd_numbers a b c d e)
  (h2 : a + c = 146)
  (h3 : e = 79) : 
  b = 73 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l3927_392719


namespace NUMINAMATH_CALUDE_boys_ages_l3927_392790

theorem boys_ages (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_ages_l3927_392790


namespace NUMINAMATH_CALUDE_bernardo_always_less_than_silvia_l3927_392711

def bernardo_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def silvia_set : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def bernardo_number (a b c : ℕ) : ℕ := 
  100 * min a (min b c) + 10 * (a + b + c - max a (max b c) - min a (min b c)) + max a (max b c)

def silvia_number (x y z : ℕ) : ℕ := 
  100 * max x (max y z) + 10 * (x + y + z - max x (max y z) - min x (min y z)) + min x (min y z)

theorem bernardo_always_less_than_silvia :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a ∈ bernardo_set → b ∈ bernardo_set → c ∈ bernardo_set →
    x ∈ silvia_set → y ∈ silvia_set → z ∈ silvia_set →
    a ≠ b → b ≠ c → a ≠ c →
    x ≠ y → y ≠ z → x ≠ z →
    bernardo_number a b c < silvia_number x y z := by
  sorry

end NUMINAMATH_CALUDE_bernardo_always_less_than_silvia_l3927_392711


namespace NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3927_392751

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_side_length : ℝ := 2 * Real.sqrt 2
  let cube_surface_area : ℝ := 6 * cube_side_length^2
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length^2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_tetrahedron_surface_area_ratio_l3927_392751


namespace NUMINAMATH_CALUDE_class_test_probability_l3927_392763

theorem class_test_probability (p_first p_second p_neither : ℝ) 
  (h1 : p_first = 0.63)
  (h2 : p_second = 0.49)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.32 := by
    sorry

end NUMINAMATH_CALUDE_class_test_probability_l3927_392763


namespace NUMINAMATH_CALUDE_monotone_increasing_range_l3927_392718

/-- The function f(x) = lg(x^2 + ax - a - 1) is monotonically increasing in [2, +∞) -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y, 2 ≤ x → 2 ≤ y → x ≤ y →
    Real.log (x^2 + a*x - a - 1) ≤ Real.log (y^2 + a*y - a - 1)

/-- The theorem stating the range of a for which f(x) is monotonically increasing -/
theorem monotone_increasing_range :
  {a : ℝ | is_monotone_increasing a} = Set.Ioi (-3) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_range_l3927_392718


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3927_392795

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (-1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3927_392795


namespace NUMINAMATH_CALUDE_teachers_in_middle_probability_l3927_392783

def num_students : ℕ := 3
def num_teachers : ℕ := 2
def num_parents : ℕ := 3
def total_people : ℕ := num_students + num_teachers + num_parents

def probability_teachers_in_middle : ℚ :=
  (Nat.factorial (total_people - num_teachers)) / (Nat.factorial total_people)

theorem teachers_in_middle_probability :
  probability_teachers_in_middle = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_teachers_in_middle_probability_l3927_392783


namespace NUMINAMATH_CALUDE_not_linear_in_M_exp_in_M_sin_in_M_iff_l3927_392750

/-- The set M of functions satisfying f(x+T) = T⋅f(x) for some non-zero T -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x}

theorem not_linear_in_M :
    ∀ T : ℝ, T ≠ 0 → ∃ x : ℝ, x + T ≠ T * x := by sorry

theorem exp_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
    (∃ T : ℝ, T > 0 ∧ a^T = T) → (fun x ↦ a^x) ∈ M := by sorry

theorem sin_in_M_iff (k : ℝ) :
    (fun x ↦ Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_not_linear_in_M_exp_in_M_sin_in_M_iff_l3927_392750


namespace NUMINAMATH_CALUDE_pig_farm_fence_length_l3927_392778

theorem pig_farm_fence_length 
  (area : ℝ) 
  (short_side : ℝ) 
  (long_side : ℝ) :
  area = 1250 ∧ 
  long_side = 2 * short_side ∧ 
  area = long_side * short_side →
  short_side + short_side + long_side = 100 := by
sorry

end NUMINAMATH_CALUDE_pig_farm_fence_length_l3927_392778


namespace NUMINAMATH_CALUDE_cylinder_height_problem_l3927_392766

/-- The height of cylinder B given the conditions of the problem -/
def height_cylinder_B : ℝ := 75

/-- The base radius of cylinder A in cm -/
def radius_A : ℝ := 10

/-- The height of cylinder A in cm -/
def height_A : ℝ := 8

/-- The base radius of cylinder B in cm -/
def radius_B : ℝ := 4

/-- The volume ratio of cylinder B to cylinder A -/
def volume_ratio : ℝ := 1.5

theorem cylinder_height_problem :
  volume_ratio * (Real.pi * radius_A^2 * height_A) = Real.pi * radius_B^2 * height_cylinder_B :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_problem_l3927_392766


namespace NUMINAMATH_CALUDE_division_problem_l3927_392774

theorem division_problem : (70 / 4 + 90 / 4) / 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3927_392774


namespace NUMINAMATH_CALUDE_dog_bones_problem_l3927_392756

theorem dog_bones_problem (initial_bones : ℕ) : 
  initial_bones + 8 = 23 → initial_bones = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_problem_l3927_392756


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3927_392796

/-- A function that generates all valid four-digit even numbers greater than 2000
    using digits 0, 1, 2, 3, 4, 5 without repetition -/
def validNumbers : Finset Nat := sorry

/-- The cardinality of the set of valid numbers -/
theorem count_valid_numbers : Finset.card validNumbers = 120 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3927_392796


namespace NUMINAMATH_CALUDE_candy_distribution_l3927_392758

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) : 
  total_candy = 30 → num_friends = 4 → 
  ∃ (removed : ℕ) (equal_share : ℕ), 
    removed ≤ 2 ∧ 
    (total_candy - removed) % num_friends = 0 ∧ 
    (total_candy - removed) / num_friends = equal_share ∧
    ∀ (r : ℕ), r < removed → (total_candy - r) % num_friends ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3927_392758


namespace NUMINAMATH_CALUDE_consecutive_integers_product_divisibility_l3927_392728

theorem consecutive_integers_product_divisibility
  (m n : ℕ) (h : m < n) :
  ∀ (a : ℕ), ∃ (i j : ℕ), i ≠ j ∧ i < n ∧ j < n ∧ (mn ∣ (a + i) * (a + j)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_divisibility_l3927_392728


namespace NUMINAMATH_CALUDE_equation_solution_l3927_392792

theorem equation_solution :
  let f (x : ℂ) := (x^2 + 4*x + 20) / (x^2 - 7*x + 12)
  let g (x : ℂ) := (x - 3) / (x - 1)
  ∀ x : ℂ, f x = g x ↔ x = (17 + Complex.I * Real.sqrt 543) / 26 ∨ x = (17 - Complex.I * Real.sqrt 543) / 26 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3927_392792


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3927_392717

theorem solution_set_of_inequality (x : ℝ) :
  (∃ (S : Set ℝ), S = {x | 4*x^2 - 4*x + 1 ≤ 0}) ↔ {1/2} = {x | 4*x^2 - 4*x + 1 ≤ 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3927_392717


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l3927_392791

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → 
  (40/100 : ℝ) * N = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l3927_392791


namespace NUMINAMATH_CALUDE_translation_proof_l3927_392768

def original_function (x : ℝ) : ℝ := 4 * x + 3

def translated_function (x : ℝ) : ℝ := 4 * x + 16

def translation_vector : ℝ × ℝ := (-3, 1)

theorem translation_proof :
  ∀ x y : ℝ, 
    y = original_function x → 
    y + (translation_vector.2) = translated_function (x + translation_vector.1) :=
by sorry

end NUMINAMATH_CALUDE_translation_proof_l3927_392768


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3927_392735

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ)^a * (3 : ℝ)^b = 3 → 
  ∀ x y : ℝ, x > 0 → y > 0 → (3 : ℝ)^x * (3 : ℝ)^y = 3 → 
  1/a + 1/b ≤ 1/x + 1/y := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3927_392735


namespace NUMINAMATH_CALUDE_units_digit_factorial_50_l3927_392773

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_50 :
  ∃ k : ℕ, factorial 50 = 10 * k :=
sorry

end NUMINAMATH_CALUDE_units_digit_factorial_50_l3927_392773


namespace NUMINAMATH_CALUDE_equation_solution_l3927_392700

theorem equation_solution : ∃ x : ℝ, (4 / 7) * (1 / 8) * x = 12 ∧ x = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3927_392700


namespace NUMINAMATH_CALUDE_recurrence_sequence_a9_l3927_392799

/-- An increasing sequence of positive integers satisfying the given recurrence relation. -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1 ≤ n → a (n + 2) = a (n + 1) + a n)

theorem recurrence_sequence_a9 (a : ℕ → ℕ) (h : RecurrenceSequence a) (h6 : a 6 = 56) :
  a 9 = 270 := by
  sorry

#check recurrence_sequence_a9

end NUMINAMATH_CALUDE_recurrence_sequence_a9_l3927_392799


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l3927_392764

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 5BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 11/20. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line
  (B - A) = 4 * (D - A) →  -- AB = 4AD
  (B - A) = 5 * (C - B) →  -- AB = 5BC
  (D - C) / (B - A) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l3927_392764


namespace NUMINAMATH_CALUDE_sqrt_product_equals_two_l3927_392765

theorem sqrt_product_equals_two : Real.sqrt 20 * Real.sqrt (1/5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_two_l3927_392765


namespace NUMINAMATH_CALUDE_prob_C_is_one_fourth_l3927_392729

/-- A game spinner with four regions A, B, C, and D -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The probability of all regions in a spinner sum to 1 -/
def valid_spinner (s : Spinner) : Prop :=
  s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given a valid spinner with probA = 1/4, probB = 1/3, and probD = 1/6, 
    the probability of region C is 1/4 -/
theorem prob_C_is_one_fourth (s : Spinner) 
  (h_valid : valid_spinner s)
  (h_probA : s.probA = 1/4)
  (h_probB : s.probB = 1/3)
  (h_probD : s.probD = 1/6) :
  s.probC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_C_is_one_fourth_l3927_392729


namespace NUMINAMATH_CALUDE_toothpicks_for_1002_base_l3927_392734

/-- Calculates the number of toothpicks required for a large equilateral triangle
    constructed with rows of small equilateral triangles. -/
def toothpicks_count (base_triangles : ℕ) : ℕ :=
  let total_triangles := base_triangles * (base_triangles + 1) / 2
  let total_sides := 3 * total_triangles
  let boundary_sides := 3 * base_triangles
  (total_sides - boundary_sides) / 2 + boundary_sides

/-- Theorem stating that for a large equilateral triangle with 1002 small triangles
    in its base, the total number of toothpicks required is 752253. -/
theorem toothpicks_for_1002_base : toothpicks_count 1002 = 752253 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_for_1002_base_l3927_392734


namespace NUMINAMATH_CALUDE_total_ballpoint_pens_l3927_392716

theorem total_ballpoint_pens (red_pens blue_pens : ℕ) 
  (h1 : red_pens = 37) 
  (h2 : blue_pens = 17) : 
  red_pens + blue_pens = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_ballpoint_pens_l3927_392716


namespace NUMINAMATH_CALUDE_T_values_l3927_392731

theorem T_values (θ : Real) :
  (∃ T : Real, T = Real.sqrt (1 + Real.sin (2 * θ))) →
  (((Real.sin (π - θ) = 3/5 ∧ π/2 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = 1/5) ∧
   ((Real.cos (π/2 - θ) = m ∧ π/2 < θ ∧ θ < 3*π/4) →
    Real.sqrt (1 + Real.sin (2 * θ)) = m - Real.sqrt (1 - m^2)) ∧
   ((Real.cos (π/2 - θ) = m ∧ 3*π/4 < θ ∧ θ < π) →
    Real.sqrt (1 + Real.sin (2 * θ)) = -m + Real.sqrt (1 - m^2))) :=
by sorry

end NUMINAMATH_CALUDE_T_values_l3927_392731


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3927_392715

def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

theorem projection_matrix_values :
  ∀ (a c : ℚ),
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![a, 3/7; c, 4/7]
  is_projection_matrix P ↔ a = 1 ∧ c = 3/7 := by
sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3927_392715


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3927_392798

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → n ≥ 143 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l3927_392798


namespace NUMINAMATH_CALUDE_import_tax_problem_l3927_392754

/-- The import tax problem -/
theorem import_tax_problem (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) 
  (h1 : tax_rate = 0.07)
  (h2 : tax_paid = 112.70)
  (h3 : total_value = 2610) :
  ∃ (excess : ℝ), excess = 1000 ∧ tax_rate * (total_value - excess) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_problem_l3927_392754


namespace NUMINAMATH_CALUDE_gcd_equality_inequality_l3927_392712

theorem gcd_equality_inequality (S : Set ℕ) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    Nat.gcd a b ≠ Nat.gcd c d) :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x y = Nat.gcd y z ∧ Nat.gcd y z ≠ Nat.gcd z x :=
by sorry

end NUMINAMATH_CALUDE_gcd_equality_inequality_l3927_392712


namespace NUMINAMATH_CALUDE_sequence_term_number_l3927_392726

theorem sequence_term_number : 
  let a : ℕ → ℝ := fun n => Real.sqrt (2 * n - 1)
  ∃ n : ℕ, n = 23 ∧ a n = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_number_l3927_392726


namespace NUMINAMATH_CALUDE_angle_measure_quadrilateral_l3927_392724

/-- The measure of angle P in a quadrilateral PQRS where ∠P = 3∠Q = 4∠R = 6∠S -/
theorem angle_measure_quadrilateral (P Q R S : ℝ) 
  (quad_sum : P + Q + R + S = 360)
  (PQ_rel : P = 3 * Q)
  (PR_rel : P = 4 * R)
  (PS_rel : P = 6 * S) :
  P = 1440 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_quadrilateral_l3927_392724


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3927_392770

theorem absolute_value_equality (b : ℝ) : 
  (|1 - b| = |3 - b|) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3927_392770


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l3927_392747

/-- Triangle side condition -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem -/
theorem triangle_inequality_squared (a b c : ℝ) 
  (h : is_triangle a b c) : 
  a^4 + b^4 + c^4 - 2*a^2*b^2 - 2*b^2*c^2 - 2*c^2*a^2 < 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l3927_392747


namespace NUMINAMATH_CALUDE_chimps_moved_correct_l3927_392769

/-- The number of chimpanzees being moved to a new cage -/
def chimps_moved (total : ℕ) (staying : ℕ) : ℕ := total - staying

/-- Theorem stating that the number of chimpanzees moved is correct -/
theorem chimps_moved_correct (total : ℕ) (staying : ℕ) 
  (h1 : total = 45) (h2 : staying = 27) : 
  chimps_moved total staying = 18 := by
  sorry

end NUMINAMATH_CALUDE_chimps_moved_correct_l3927_392769


namespace NUMINAMATH_CALUDE_gcd_14658_11241_l3927_392714

theorem gcd_14658_11241 : Nat.gcd 14658 11241 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_14658_11241_l3927_392714


namespace NUMINAMATH_CALUDE_sum_of_four_triangles_l3927_392721

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- All squares have the same value -/
axiom square_constant : ∀ s : ℝ, s = square_value

/-- All triangles have the same value -/
axiom triangle_constant : ∀ t : ℝ, t = triangle_value

/-- First equation: square + triangle + square + triangle + square = 27 -/
axiom equation_1 : 3 * square_value + 2 * triangle_value = 27

/-- Second equation: triangle + square + triangle + square + triangle = 23 -/
axiom equation_2 : 2 * square_value + 3 * triangle_value = 23

/-- The sum of four triangles equals 12 -/
theorem sum_of_four_triangles : 4 * triangle_value = 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_triangles_l3927_392721


namespace NUMINAMATH_CALUDE_dried_grapes_weight_l3927_392757

/-- Calculates the weight of dried grapes from fresh grapes -/
theorem dried_grapes_weight
  (fresh_weight : ℝ)
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (h1 : fresh_weight = 40)
  (h2 : fresh_water_content = 0.9)
  (h3 : dried_water_content = 0.2) :
  (fresh_weight * (1 - fresh_water_content)) / (1 - dried_water_content) = 5 := by
  sorry

end NUMINAMATH_CALUDE_dried_grapes_weight_l3927_392757


namespace NUMINAMATH_CALUDE_first_apartment_utility_cost_l3927_392748

/-- Represents the monthly cost structure for an apartment --/
structure ApartmentCost where
  rent : ℝ
  utilities : ℝ
  drivingDistance : ℝ

/-- Calculates the total monthly cost for an apartment --/
def totalMonthlyCost (apt : ApartmentCost) (drivingCostPerMile : ℝ) (workingDays : ℝ) : ℝ :=
  apt.rent + apt.utilities + (apt.drivingDistance * drivingCostPerMile * workingDays)

/-- Theorem stating the utility cost of the first apartment --/
theorem first_apartment_utility_cost :
  let firstApt : ApartmentCost := { rent := 800, utilities := U, drivingDistance := 31 }
  let secondApt : ApartmentCost := { rent := 900, utilities := 200, drivingDistance := 21 }
  let drivingCostPerMile : ℝ := 0.58
  let workingDays : ℝ := 20
  totalMonthlyCost firstApt drivingCostPerMile workingDays - 
    totalMonthlyCost secondApt drivingCostPerMile workingDays = 76 →
  U = 259.60 := by
  sorry


end NUMINAMATH_CALUDE_first_apartment_utility_cost_l3927_392748


namespace NUMINAMATH_CALUDE_shaded_quadrilateral_area_l3927_392704

theorem shaded_quadrilateral_area (s : ℝ) (a b : ℝ) : 
  s = 20 → a = 15 → b = 20 →
  s^2 - (1/2 * a * b) - (1/2 * (s * b / (a^2 + b^2).sqrt) * (s * a / (a^2 + b^2).sqrt)) = 154 :=
by sorry

end NUMINAMATH_CALUDE_shaded_quadrilateral_area_l3927_392704


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l3927_392777

/-- The number of small boxes in the large box -/
def small_boxes : ℕ := 15

/-- The number of chocolate bars in each small box -/
def bars_per_box : ℕ := 25

/-- The total number of chocolate bars in the large box -/
def total_bars : ℕ := small_boxes * bars_per_box

theorem chocolate_bar_count : total_bars = 375 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l3927_392777


namespace NUMINAMATH_CALUDE_grade_10_sample_size_l3927_392725

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
  | 0 => 2  -- Grade 10
  | 1 => 2  -- Grade 11
  | 2 => 1  -- Grade 12

/-- Total sample size -/
def sample_size : ℕ := 45

/-- Calculates the number of students sampled from a specific grade -/
def students_sampled (grade : Fin 3) : ℕ :=
  (sample_size * grade_ratio grade) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

/-- Theorem stating that the number of grade 10 students in the sample is 18 -/
theorem grade_10_sample_size :
  students_sampled 0 = 18 := by sorry

end NUMINAMATH_CALUDE_grade_10_sample_size_l3927_392725


namespace NUMINAMATH_CALUDE_parabola_through_points_point_not_on_parabola_l3927_392705

/-- A parabola of the form y = ax² + bx passing through (1, 3) and (-1, -1) -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x

theorem parabola_through_points (a b : ℝ) :
  Parabola a b 1 = 3 ∧ Parabola a b (-1) = -1 → Parabola a b = λ x => x^2 + 2*x :=
sorry

theorem point_not_on_parabola :
  Parabola 1 2 2 ≠ 6 :=
sorry

end NUMINAMATH_CALUDE_parabola_through_points_point_not_on_parabola_l3927_392705


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3927_392737

/-- The area of a regular square inscribed in a circle with area 324π is 648 square units. -/
theorem inscribed_square_area (circle_area : ℝ) (h : circle_area = 324 * Real.pi) :
  let r : ℝ := Real.sqrt (circle_area / Real.pi)
  let square_side : ℝ := Real.sqrt 2 * r
  square_side ^ 2 = 648 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3927_392737


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3927_392706

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos α = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3927_392706


namespace NUMINAMATH_CALUDE_overall_gain_loss_percent_zero_l3927_392785

def article_A_cost : ℝ := 600
def article_B_cost : ℝ := 700
def article_C_cost : ℝ := 800
def article_A_sell : ℝ := 450
def article_B_sell : ℝ := 750
def article_C_sell : ℝ := 900

def total_cost : ℝ := article_A_cost + article_B_cost + article_C_cost
def total_sell : ℝ := article_A_sell + article_B_sell + article_C_sell

theorem overall_gain_loss_percent_zero :
  (total_sell - total_cost) / total_cost * 100 = 0 := by sorry

end NUMINAMATH_CALUDE_overall_gain_loss_percent_zero_l3927_392785


namespace NUMINAMATH_CALUDE_mikes_salary_increase_l3927_392762

theorem mikes_salary_increase (freds_salary_then : ℝ) (mikes_salary_now : ℝ) :
  freds_salary_then = 1000 →
  mikes_salary_now = 15400 →
  let mikes_salary_then := 10 * freds_salary_then
  (mikes_salary_now - mikes_salary_then) / mikes_salary_then * 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_mikes_salary_increase_l3927_392762


namespace NUMINAMATH_CALUDE_starting_number_proof_l3927_392752

theorem starting_number_proof (n : ℕ) (h1 : n > 0) (h2 : n ≤ 79) (h3 : n % 11 = 0)
  (h4 : ∀ k, k ∈ Finset.range 6 → (n - k * 11) % 11 = 0)
  (h5 : ∀ m, m < n - 5 * 11 → ¬(∃ l, l ∈ Finset.range 6 ∧ m = n - l * 11)) :
  n - 5 * 11 = 22 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l3927_392752


namespace NUMINAMATH_CALUDE_cages_needed_cages_needed_is_five_l3927_392781

def initial_puppies : ℕ := 45
def sold_puppies : ℕ := 11
def puppies_per_cage : ℕ := 7

theorem cages_needed : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage

theorem cages_needed_is_five : cages_needed = 5 := by
  sorry

end NUMINAMATH_CALUDE_cages_needed_cages_needed_is_five_l3927_392781
