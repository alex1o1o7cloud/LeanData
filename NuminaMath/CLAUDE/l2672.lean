import Mathlib

namespace NUMINAMATH_CALUDE_sports_competition_team_sizes_l2672_267203

theorem sports_competition_team_sizes :
  ∀ (boys girls : ℕ),
  (boys + 48 : ℚ) / 6 + (girls + 50 : ℚ) / 7 = 48 - (boys : ℚ) / 6 + 50 - (girls : ℚ) / 7 →
  boys - 48 = (girls - 50) / 2 →
  boys = 72 ∧ girls = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_competition_team_sizes_l2672_267203


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2672_267227

/-- The sum of an infinite geometric series with first term 1 and common ratio 1/3 is 3/2 -/
theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/3
  let S : ℝ := ∑' n, a * r^n
  S = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2672_267227


namespace NUMINAMATH_CALUDE_vector_problem_l2672_267297

theorem vector_problem (a b : ℝ × ℝ) :
  a + b = (2, 3) → a - b = (-2, 1) → a - 2 * b = (-4, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2672_267297


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l2672_267274

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the domain
def domain (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 5

-- Theorem: The curve is a line segment
theorem curve_is_line_segment :
  ∃ (a b : ℝ), ∀ (t : ℝ), domain t →
    ∃ (k : ℝ), 0 ≤ k ∧ k ≤ 1 ∧
      x t = a + k * (b - a) ∧
      y t = (x t - 2) / 3 ∧
      -1 ≤ y t ∧ y t ≤ 24 :=
by sorry


end NUMINAMATH_CALUDE_curve_is_line_segment_l2672_267274


namespace NUMINAMATH_CALUDE_quadratic_properties_l2672_267287

-- Define the quadratic function
def f (x : ℝ) := -x^2 + 9*x - 20

-- Theorem statement
theorem quadratic_properties :
  (∃ (max : ℝ), ∀ (x : ℝ), f x ≥ 0 → x ≤ max) ∧
  (∃ (max : ℝ), f max ≥ 0 ∧ max = 5) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x^2 - 9*x + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2672_267287


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2672_267204

/-- The magnitude of the complex number z = (1-i)/i is √2 -/
theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((1 - i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2672_267204


namespace NUMINAMATH_CALUDE_charity_race_total_amount_l2672_267269

def total_students : ℕ := 30
def group1_students : ℕ := 10
def group1_amount : ℕ := 20
def group2_amount : ℕ := 30

theorem charity_race_total_amount :
  (group1_students * group1_amount) + 
  ((total_students - group1_students) * group2_amount) = 800 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_total_amount_l2672_267269


namespace NUMINAMATH_CALUDE_integer_solutions_l2672_267295

def satisfies_inequalities (x : ℤ) : Prop :=
  (x + 8 : ℚ) / (x + 2 : ℚ) > 2 ∧ Real.log (x - 1 : ℝ) < 1

theorem integer_solutions :
  {x : ℤ | satisfies_inequalities x} = {2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_l2672_267295


namespace NUMINAMATH_CALUDE_amp_composition_l2672_267280

-- Define the & operation (postfix)
def postAmp (x : ℝ) : ℝ := 9 - x

-- Define the & operation (prefix)
def preAmp (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : preAmp (postAmp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l2672_267280


namespace NUMINAMATH_CALUDE_point_symmetric_range_l2672_267296

/-- 
Given a point P(a+1, 2a-3) that is symmetric about the x-axis and lies in the first quadrant,
prove that the range of a is (-1, 3/2).
-/
theorem point_symmetric_range (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 1 ∧ P.2 = 2*a - 3 ∧ P.1 > 0 ∧ P.2 > 0) ↔ 
  -1 < a ∧ a < 3/2 := by sorry

end NUMINAMATH_CALUDE_point_symmetric_range_l2672_267296


namespace NUMINAMATH_CALUDE_bill_eric_age_difference_l2672_267207

/-- The age difference between two brothers, given their total age and the older brother's age. -/
def age_difference (total_age : ℕ) (older_brother_age : ℕ) : ℕ :=
  older_brother_age - (total_age - older_brother_age)

/-- Theorem stating the age difference between Bill and Eric -/
theorem bill_eric_age_difference :
  let total_age : ℕ := 28
  let bill_age : ℕ := 16
  age_difference total_age bill_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_bill_eric_age_difference_l2672_267207


namespace NUMINAMATH_CALUDE_number_of_divisors_2310_l2672_267218

theorem number_of_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_2310_l2672_267218


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2672_267242

/-- An ellipse passing through (1,0) with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eq : 1 / a^2 + 0 / b^2 = 1

/-- A parabola with focus at (1,0) and vertex at (m,0) -/
structure Parabola where
  m : ℝ

/-- The theorem statement -/
theorem ellipse_parabola_intersection (ε : Ellipse) (ρ : Parabola) :
  let e := Real.sqrt (1 - ε.b^2)
  (Real.sqrt (2/3) < e ∧ e < 1) →
  (1 < ρ.m ∧ ρ.m < (3 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2672_267242


namespace NUMINAMATH_CALUDE_additional_cakes_is_21_l2672_267219

/-- Represents the number of cakes baked in a week -/
structure CakeQuantities where
  cheesecakes : ℕ
  muffins : ℕ
  redVelvet : ℕ
  chocolateMoist : ℕ
  fruitcakes : ℕ
  carrotCakes : ℕ

/-- Carter's usual cake quantities -/
def usualQuantities : CakeQuantities := {
  cheesecakes := 6,
  muffins := 5,
  redVelvet := 8,
  chocolateMoist := 0,
  fruitcakes := 0,
  carrotCakes := 0
}

/-- Calculate the new quantities based on the given rates -/
def newQuantities (usual : CakeQuantities) : CakeQuantities := {
  cheesecakes := (usual.cheesecakes * 3 + 1) / 2,
  muffins := (usual.muffins * 6 + 2) / 5,
  redVelvet := (usual.redVelvet * 9 + 2) / 5,
  chocolateMoist := ((usual.redVelvet * 9 + 2) / 5) / 2,
  fruitcakes := (((usual.muffins * 6 + 2) / 5) * 2) / 3,
  carrotCakes := 0
}

/-- Calculate the total additional cakes -/
def additionalCakes (usual new : CakeQuantities) : ℕ :=
  (new.cheesecakes - usual.cheesecakes) +
  (new.muffins - usual.muffins) +
  (new.redVelvet - usual.redVelvet) +
  (new.chocolateMoist - usual.chocolateMoist) +
  (new.fruitcakes - usual.fruitcakes) +
  (new.carrotCakes - usual.carrotCakes)

theorem additional_cakes_is_21 :
  additionalCakes usualQuantities (newQuantities usualQuantities) = 21 := by
  sorry

end NUMINAMATH_CALUDE_additional_cakes_is_21_l2672_267219


namespace NUMINAMATH_CALUDE_sin_225_plus_alpha_l2672_267253

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (π/4 + α) = 5/13) :
  Real.sin (5*π/4 + α) = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_plus_alpha_l2672_267253


namespace NUMINAMATH_CALUDE_h_x_equality_l2672_267236

theorem h_x_equality (x : ℝ) (h : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 + h x = 7 * x^3 - 5 * x^2 + 9 * x + 3) → 
  (h x = -2 * x^5 + 3 * x^3 - 5 * x^2 + 9 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_h_x_equality_l2672_267236


namespace NUMINAMATH_CALUDE_crayons_left_in_drawer_l2672_267290

theorem crayons_left_in_drawer (initial_crayons : ℕ) (crayons_taken : ℕ) : 
  initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by sorry

end NUMINAMATH_CALUDE_crayons_left_in_drawer_l2672_267290


namespace NUMINAMATH_CALUDE_fuel_food_ratio_l2672_267260

theorem fuel_food_ratio 
  (fuel_cost : ℝ) 
  (distance_per_tank : ℝ) 
  (total_distance : ℝ) 
  (total_spent : ℝ) 
  (h1 : fuel_cost = 45)
  (h2 : distance_per_tank = 500)
  (h3 : total_distance = 2000)
  (h4 : total_spent = 288) :
  (total_spent - (total_distance / distance_per_tank * fuel_cost)) / 
  (total_distance / distance_per_tank * fuel_cost) = 3 / 5 := by
sorry


end NUMINAMATH_CALUDE_fuel_food_ratio_l2672_267260


namespace NUMINAMATH_CALUDE_closest_approximation_l2672_267282

def x_values : List ℝ := [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]

def y (x : ℝ) : ℝ := x^2 - x

def distance_to_target (x : ℝ) : ℝ := |y x - 1.4|

theorem closest_approximation :
  ∀ x ∈ x_values, distance_to_target 1.8 ≤ distance_to_target x := by
  sorry

end NUMINAMATH_CALUDE_closest_approximation_l2672_267282


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2672_267257

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  q > 0 →
  a 3 + a 4 = a 5 →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2672_267257


namespace NUMINAMATH_CALUDE_function_max_at_zero_l2672_267270

/-- The function f(x) = x^3 - 3x^2 + 1 reaches its maximum value at x = 0 -/
theorem function_max_at_zero (f : ℝ → ℝ) (h : f = λ x => x^3 - 3*x^2 + 1) :
  ∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a ∧ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_max_at_zero_l2672_267270


namespace NUMINAMATH_CALUDE_expected_pairs_for_given_deck_l2672_267229

/-- Represents a deck of cards with numbered pairs and Joker pairs -/
structure Deck :=
  (num_pairs : ℕ)
  (joker_pairs : ℕ)

/-- Calculates the expected number of complete pairs when drawing until a Joker pair is found -/
def expected_complete_pairs (d : Deck) : ℚ :=
  (d.num_pairs : ℚ) / 3 + 1

theorem expected_pairs_for_given_deck :
  let d : Deck := ⟨7, 2⟩
  expected_complete_pairs d = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_pairs_for_given_deck_l2672_267229


namespace NUMINAMATH_CALUDE_expression_value_l2672_267208

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : abs m = 2)  -- |m| = 2
  : (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 35/8 ∨ 
    (3 * c * d) / (4 * m) + m^2 - 5 * (a + b) = 29/8 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2672_267208


namespace NUMINAMATH_CALUDE_complex_magnitude_l2672_267206

/-- Given a real number a, if (a^2 * i) / (1 + i) is imaginary, then |a + i| = √5 -/
theorem complex_magnitude (a : ℝ) : 
  (((a^2 * Complex.I) / (1 + Complex.I)).im ≠ 0 ∧ ((a^2 * Complex.I) / (1 + Complex.I)).re = 0) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2672_267206


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2672_267200

/-- A function satisfying the given functional equation. -/
def SatisfyingFunction (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f (f n)) = -f (f (m + 1)) - n

/-- The theorem stating that the only function satisfying the equation is f(p) = 1 - p. -/
theorem unique_satisfying_function :
  ∀ f : ℤ → ℤ, SatisfyingFunction f ↔ (∀ p : ℤ, f p = 1 - p) :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2672_267200


namespace NUMINAMATH_CALUDE_bus_children_count_l2672_267256

/-- The total number of children on a bus after more children got on is equal to the sum of the initial number of children and the additional children who got on. -/
theorem bus_children_count (initial_children additional_children : ℕ) :
  initial_children + additional_children = initial_children + additional_children :=
by sorry

end NUMINAMATH_CALUDE_bus_children_count_l2672_267256


namespace NUMINAMATH_CALUDE_distance_y_to_earth_l2672_267254

-- Define the distances
def distance_earth_to_x : ℝ := 0.5
def distance_x_to_y : ℝ := 0.1
def total_distance : ℝ := 0.7

-- Theorem to prove
theorem distance_y_to_earth : 
  total_distance - (distance_earth_to_x + distance_x_to_y) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_distance_y_to_earth_l2672_267254


namespace NUMINAMATH_CALUDE_total_squares_is_55_l2672_267223

/-- The number of squares of a given size that can be traced in a 5x5 grid -/
def squares_of_size (n : Nat) : Nat :=
  (6 - n) ^ 2

/-- The total number of squares that can be traced in a 5x5 grid -/
def total_squares : Nat :=
  (List.range 5).map (λ i => squares_of_size (i + 1)) |>.sum

/-- Theorem: The total number of different squares that can be traced in a 5x5 grid is 55 -/
theorem total_squares_is_55 : total_squares = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_is_55_l2672_267223


namespace NUMINAMATH_CALUDE_weightlifter_total_weight_l2672_267240

/-- The weight a weightlifter can lift in each hand, in pounds. -/
def weight_per_hand : ℕ := 10

/-- The total weight a weightlifter can lift at once, in pounds. -/
def total_weight : ℕ := weight_per_hand * 2

/-- Theorem stating that the total weight a weightlifter can lift at once is 20 pounds. -/
theorem weightlifter_total_weight : total_weight = 20 := by sorry

end NUMINAMATH_CALUDE_weightlifter_total_weight_l2672_267240


namespace NUMINAMATH_CALUDE_square_expression_is_perfect_square_l2672_267220

theorem square_expression_is_perfect_square (n k l : ℕ) 
  (h : n^2 + k^2 = 2 * l^2) : 
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 :=
sorry

end NUMINAMATH_CALUDE_square_expression_is_perfect_square_l2672_267220


namespace NUMINAMATH_CALUDE_m_range_l2672_267268

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≥ m

def q (m : ℝ) : Prop := ∀ x : ℝ, (-(7 - 3*m))^(x+1) < (-(7 - 3*m))^x

-- State the theorem
theorem m_range (m : ℝ) : 
  (p m ∧ ¬(q m)) ∨ (¬(p m) ∧ q m) → 1 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2672_267268


namespace NUMINAMATH_CALUDE_bries_slacks_count_l2672_267294

/-- Proves that Brie has 8 slacks given the conditions of the problem -/
theorem bries_slacks_count :
  ∀ (total_blouses total_skirts total_slacks : ℕ)
    (blouses_in_hamper skirts_in_hamper slacks_in_hamper : ℕ)
    (clothes_to_wash : ℕ),
  total_blouses = 12 →
  total_skirts = 6 →
  blouses_in_hamper = (75 * total_blouses) / 100 →
  skirts_in_hamper = (50 * total_skirts) / 100 →
  slacks_in_hamper = (25 * total_slacks) / 100 →
  clothes_to_wash = 14 →
  clothes_to_wash = blouses_in_hamper + skirts_in_hamper + slacks_in_hamper →
  total_slacks = 8 := by
sorry

end NUMINAMATH_CALUDE_bries_slacks_count_l2672_267294


namespace NUMINAMATH_CALUDE_hydrolysis_weight_change_l2672_267222

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Molecular weight of acetylsalicylic acid (C9H8O4) in g/mol -/
def acetylsalicylic_acid_weight : ℝ := 9 * C_weight + 8 * H_weight + 4 * O_weight

/-- Molecular weight of sodium hydroxide (NaOH) in g/mol -/
def sodium_hydroxide_weight : ℝ := Na_weight + O_weight + H_weight

/-- Molecular weight of salicylic acid (C7H6O3) in g/mol -/
def salicylic_acid_weight : ℝ := 7 * C_weight + 6 * H_weight + 3 * O_weight

/-- Molecular weight of sodium acetate (CH3COONa) in g/mol -/
def sodium_acetate_weight : ℝ := 2 * C_weight + 3 * H_weight + 2 * O_weight + Na_weight

/-- Theorem stating that the overall molecular weight change during the hydrolysis reaction is 0 g/mol -/
theorem hydrolysis_weight_change :
  acetylsalicylic_acid_weight + sodium_hydroxide_weight = salicylic_acid_weight + sodium_acetate_weight :=
by sorry

end NUMINAMATH_CALUDE_hydrolysis_weight_change_l2672_267222


namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l2672_267216

/-- The equation of the tangent line to y = -x^3 + 3x^2 at (1, 2) is y = 3x - 1 -/
theorem tangent_line_at_one_two (x : ℝ) :
  let f (x : ℝ) := -x^3 + 3*x^2
  let tangent_line (x : ℝ) := 3*x - 1
  f 1 = 2 ∧ 
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) ≠ tangent_line x - tangent_line 1) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |(f x - f 1) / (x - 1) - (tangent_line x - tangent_line 1) / (x - 1)| < ε) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l2672_267216


namespace NUMINAMATH_CALUDE_trajectory_equation_l2672_267228

/-- The trajectory of point M(x, y) such that the ratio of its distance from the line x = 25/4
    to its distance from the point (4, 0) is 5/4 -/
theorem trajectory_equation (x y : ℝ) :
  (|x - 25/4| / Real.sqrt ((x - 4)^2 + y^2) = 5/4) →
  (x^2 / 25 + y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2672_267228


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_four_l2672_267231

def binary_number : ℕ := 110110111101

theorem remainder_of_binary_div_four :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_four_l2672_267231


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2672_267212

theorem sqrt_equality_implies_specific_integers :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (1 + Real.sqrt (33 + 16 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 17 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2672_267212


namespace NUMINAMATH_CALUDE_badminton_team_combinations_l2672_267292

theorem badminton_team_combinations : 
  ∀ (male_players female_players : ℕ), 
    male_players = 6 → 
    female_players = 7 → 
    (male_players.choose 1) * (female_players.choose 1) = 42 := by
sorry

end NUMINAMATH_CALUDE_badminton_team_combinations_l2672_267292


namespace NUMINAMATH_CALUDE_april_price_achieves_profit_l2672_267241

/-- Represents the sales and pricing data for a desk lamp over several months -/
structure LampSalesData where
  cost_price : ℝ
  selling_price_jan_mar : ℝ
  sales_jan : ℕ
  sales_mar : ℕ
  price_reduction_sales_increase : ℝ
  desired_profit_apr : ℝ

/-- Calculates the selling price in April that achieves the desired profit -/
def calculate_april_price (data : LampSalesData) : ℝ :=
  sorry

/-- Theorem stating that the calculated April price achieves the desired profit -/
theorem april_price_achieves_profit (data : LampSalesData) 
  (h1 : data.cost_price = 25)
  (h2 : data.selling_price_jan_mar = 40)
  (h3 : data.sales_jan = 256)
  (h4 : data.sales_mar = 400)
  (h5 : data.price_reduction_sales_increase = 4)
  (h6 : data.desired_profit_apr = 4200) :
  let april_price := calculate_april_price data
  let april_sales := data.sales_mar + data.price_reduction_sales_increase * (data.selling_price_jan_mar - april_price)
  (april_price - data.cost_price) * april_sales = data.desired_profit_apr ∧ april_price = 35 :=
sorry

end NUMINAMATH_CALUDE_april_price_achieves_profit_l2672_267241


namespace NUMINAMATH_CALUDE_sine_function_properties_l2672_267283

/-- The function f we're analyzing -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- The angular frequency ω -/
noncomputable def ω : ℝ := sorry

/-- The phase shift φ -/
noncomputable def φ : ℝ := sorry

/-- The constant M -/
noncomputable def M : ℝ := sorry

/-- Theorem stating the properties of f and the conclusion -/
theorem sine_function_properties :
  (∃ A : ℝ, ∀ x : ℝ, f x ≤ f A) ∧  -- A is a highest point
  (ω > 0) ∧
  (0 < φ ∧ φ < 2 * Real.pi) ∧
  (∃ B C : ℝ, B < C ∧  -- B and C are adjacent centers of symmetry
    (∀ x : ℝ, f (B + x) = f (B - x)) ∧
    (∀ x : ℝ, f (C + x) = f (C - x)) ∧
    (C - B = Real.pi / ω)) ∧
  ((C - B) * (f A) / 2 = 1 / 2) ∧  -- Area of triangle ABC is 1/2
  (M > 0 ∧ ∀ x : ℝ, f (x + M) = M * f (-x)) →  -- Functional equation
  (∀ x : ℝ, f x = -Real.sin (Real.pi * x)) := by
sorry

end NUMINAMATH_CALUDE_sine_function_properties_l2672_267283


namespace NUMINAMATH_CALUDE_average_income_Q_R_l2672_267259

/-- The average monthly income of P and Q is Rs. 5050, 
    the average monthly income of P and R is Rs. 5200, 
    and the monthly income of P is Rs. 4000. 
    Prove that the average monthly income of Q and R is Rs. 6250. -/
theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 → 
  (P + R) / 2 = 5200 → 
  P = 4000 → 
  (Q + R) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_Q_R_l2672_267259


namespace NUMINAMATH_CALUDE_non_egg_laying_hens_l2672_267243

/-- Proves that the number of non-egg-laying hens is 20 given the total number of chickens,
    number of roosters, and number of egg-laying hens. -/
theorem non_egg_laying_hens (total_chickens roosters egg_laying_hens : ℕ) : 
  total_chickens = 325 →
  roosters = 28 →
  egg_laying_hens = 277 →
  total_chickens - roosters - egg_laying_hens = 20 := by
sorry

end NUMINAMATH_CALUDE_non_egg_laying_hens_l2672_267243


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l2672_267279

theorem geometric_progression_solution :
  ∃! x : ℚ, ((-10 + x)^2 = (-30 + x) * (40 + x)) ∧ x = 130/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l2672_267279


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2672_267248

/-- Given a cube of side length n, painted blue on all faces and split into unit cubes,
    if exactly one-third of the total faces of the unit cubes are blue, then n = 3 -/
theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2672_267248


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2672_267224

/-- Given a hyperbola C with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    its right focus F, and point B as a vertex on the imaginary axis,
    prove that the equation of C is x²/4 - y²/6 = 1 under the given conditions. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (B : ℝ × ℝ) (A : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c, F = (c, 0) ∧ c > 0) →
  B = (0, b) →
  A ∈ C →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ A = (1 - t) • B + t • F) →
  dist B A = 2 * dist A F →
  dist B F = 4 →
  (∀ x y, (x, y) ∈ C ↔ x^2 / 4 - y^2 / 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2672_267224


namespace NUMINAMATH_CALUDE_josies_calculation_l2672_267213

theorem josies_calculation (a b c d e : ℤ) : 
  a = 2 → b = 1 → c = -1 → d = 3 → 
  (a - b + c^2 - d + e = a - (b - (c^2 - (d + e)))) → e = 0 := by
sorry

end NUMINAMATH_CALUDE_josies_calculation_l2672_267213


namespace NUMINAMATH_CALUDE_smallest_fraction_l2672_267278

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) ((x^2+1)/8)))) = 8/(x+2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l2672_267278


namespace NUMINAMATH_CALUDE_triangle_proof_l2672_267272

open Real

theorem triangle_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  -- Given condition
  a * cos C - c / 2 = b →
  -- Part I
  A = 2 * π / 3 ∧
  -- Part II
  a = 3 →
  -- Perimeter range
  let l := a + b + c
  6 < l ∧ l ≤ 3 + 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l2672_267272


namespace NUMINAMATH_CALUDE_weed_ratio_l2672_267247

/-- Represents the number of weeds pulled on each day -/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℚ
  friday : ℚ

/-- The problem of Sarah's weed pulling -/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = (1 : ℚ) / 5 * w.wednesday ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem stating the ratio of weeds pulled on Thursday to Wednesday -/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.thursday / w.wednesday = (1 : ℚ) / 5 := by
  sorry


end NUMINAMATH_CALUDE_weed_ratio_l2672_267247


namespace NUMINAMATH_CALUDE_abc_sign_sum_l2672_267232

theorem abc_sign_sum (a b c : ℚ) (h : |a*b*c| / (a*b*c) = 1) :
  |a| / a + |b| / b + |c| / c = -1 ∨ |a| / a + |b| / b + |c| / c = 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_sign_sum_l2672_267232


namespace NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l2672_267246

/-- 
Given a sector with a central angle of 120°, 
the ratio of the area of the sector to the area of its inscribed circle is (7 + 4√3) / 9.
-/
theorem sector_inscribed_circle_area_ratio :
  ∀ R r : ℝ,
  R > 0 → r > 0 →
  r / (R - r) = Real.sqrt 3 / 2 →
  (1/3 * π * R^2) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_sector_inscribed_circle_area_ratio_l2672_267246


namespace NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2672_267263

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := (ones n) * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := (ones (n+1)) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : (1987 : ℕ) ∣ ones n) :
  (1987 : ℕ) ∣ p n ∧ (1987 : ℕ) ∣ q n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2672_267263


namespace NUMINAMATH_CALUDE_circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l2672_267284

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Theorem: Circumradius of triangle with arithmetic sequence angles -/
theorem circumradius_arithmetic_angles (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.B = t.A + k ∧ t.C = t.B + k) : 
  circumradius t = 2 * Real.sqrt 3 / 3 := by sorry

/-- Theorem: Maximum inradius of triangle with arithmetic sequence sides -/
theorem max_inradius_arithmetic_sides (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  inradius t ≤ Real.sqrt 3 / 3 := by sorry

/-- Corollary: The maximum inradius is achieved -/
theorem max_inradius_achieved (t : Triangle) 
  (h1 : t.b = 2)
  (h2 : ∃ k : ℝ, t.b = t.a + k ∧ t.c = t.b + k) :
  ∃ t' : Triangle, t'.b = 2 ∧ (∃ k : ℝ, t'.b = t'.a + k ∧ t'.c = t'.b + k) ∧ 
  inradius t' = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_circumradius_arithmetic_angles_max_inradius_arithmetic_sides_max_inradius_achieved_l2672_267284


namespace NUMINAMATH_CALUDE_lambda_n_lower_bound_l2672_267286

/-- The ratio of the longest to the shortest distance between any two of n points in the plane -/
def lambda_n (n : ℕ) : ℝ := sorry

/-- Theorem: For n ≥ 4, λ_n ≥ 2 * sin((n-2)π/(2n)) -/
theorem lambda_n_lower_bound (n : ℕ) (h : n ≥ 4) : 
  lambda_n n ≥ 2 * Real.sin ((n - 2) * Real.pi / (2 * n)) := by sorry

end NUMINAMATH_CALUDE_lambda_n_lower_bound_l2672_267286


namespace NUMINAMATH_CALUDE_rachels_math_homework_l2672_267221

/-- Rachel's homework problem -/
theorem rachels_math_homework (reading_pages : ℕ) (extra_math_pages : ℕ) : 
  reading_pages = 4 → extra_math_pages = 3 → reading_pages + extra_math_pages = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rachels_math_homework_l2672_267221


namespace NUMINAMATH_CALUDE_incorrect_number_value_l2672_267244

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg correct_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 20)
  (h3 : correct_avg = 26)
  (h4 : correct_value = 86) :
  ∃ x : ℚ, n * correct_avg - n * initial_avg = correct_value - x ∧ x = 26 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l2672_267244


namespace NUMINAMATH_CALUDE_part_one_part_two_l2672_267258

-- Define the sets A and B
def A : Set ℝ := {x | -2 + 3*x - x^2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Part 1: Prove that when a = 1, (∁A) ∩ B = (1, 2)
theorem part_one : (Set.compl A) ∩ (B 1) = Set.Ioo 1 2 := by sorry

-- Part 2: Prove that (∁A) ∩ B = ∅ if and only if a ≤ -1 or a ≥ 2
theorem part_two (a : ℝ) : (Set.compl A) ∩ (B a) = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2672_267258


namespace NUMINAMATH_CALUDE_sine_fraction_equals_three_l2672_267261

theorem sine_fraction_equals_three (d : ℝ) (h : d = π / 7) :
  (3 * Real.sin (2 * d) * Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d)) /
  (Real.sin d * Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_fraction_equals_three_l2672_267261


namespace NUMINAMATH_CALUDE_mountain_height_theorem_l2672_267267

-- Define the measurement points and the peak
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the measurement setup
structure MountainMeasurement where
  A : Point3D
  B : Point3D
  C : Point3D
  peak : Point3D
  AB : ℝ
  BC : ℝ
  angle_ABC : ℝ
  elevation_A : ℝ
  elevation_C : ℝ
  angle_BAT : ℝ

-- Define the theorem
theorem mountain_height_theorem (m : MountainMeasurement) 
  (h_AB : m.AB = 100)
  (h_BC : m.BC = 150)
  (h_angle_ABC : m.angle_ABC = 130 * π / 180)
  (h_elevation_A : m.elevation_A = 20 * π / 180)
  (h_elevation_C : m.elevation_C = 22 * π / 180)
  (h_angle_BAT : m.angle_BAT = 93 * π / 180) :
  ∃ (h1 h2 : ℝ), 
    (abs (h1 - 93.4) < 0.1 ∧ abs (h2 - 390.9) < 0.1) ∧
    ((m.peak.z - m.A.z = h1) ∨ (m.peak.z - m.A.z = h2)) := by
  sorry

end NUMINAMATH_CALUDE_mountain_height_theorem_l2672_267267


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l2672_267266

-- Define the function f(x) = -|x|
def f (x : ℝ) : ℝ := -|x|

-- State the theorem
theorem f_increasing_on_negative_reals :
  ∀ (x₁ x₂ : ℝ), x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_reals_l2672_267266


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2672_267225

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2672_267225


namespace NUMINAMATH_CALUDE_sum_inequality_l2672_267293

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2672_267293


namespace NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l2672_267265

/-- Represents a repeating decimal -/
def RepeatingDecimal (whole : ℕ) (repeating : List ℕ) : ℚ :=
  sorry

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 0 [1] + RepeatingDecimal 0 [1, 2] + RepeatingDecimal 0 [1, 2, 3] =
  RepeatingDecimal 0 [3, 5, 5, 4, 4, 6] :=
sorry

end NUMINAMATH_CALUDE_sum_of_specific_repeating_decimals_l2672_267265


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2672_267251

/-- Given vectors a and b in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-2, 4) →
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2672_267251


namespace NUMINAMATH_CALUDE_hemisphere_volume_l2672_267288

/-- Given a hemisphere with surface area (excluding the base) of 256π cm²,
    prove that its volume is (2048√2)/3 π cm³. -/
theorem hemisphere_volume (r : ℝ) (h : 2 * Real.pi * r^2 = 256 * Real.pi) :
  (2/3) * Real.pi * r^3 = (2048 * Real.sqrt 2 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l2672_267288


namespace NUMINAMATH_CALUDE_stock_price_increase_percentage_l2672_267298

theorem stock_price_increase_percentage (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = 1080)
  (h3 : higher_price_stocks > total_stocks - higher_price_stocks) :
  let lower_price_stocks := total_stocks - higher_price_stocks
  (higher_price_stocks - lower_price_stocks) / lower_price_stocks * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_percentage_l2672_267298


namespace NUMINAMATH_CALUDE_max_cut_length_30x30_225parts_l2672_267275

/-- Represents a square board -/
structure Board :=
  (size : ℕ)

/-- Represents a division of the board -/
structure Division :=
  (num_parts : ℕ)
  (equal_area : Bool)

/-- Calculates the maximum possible total length of cuts for a given board and division -/
def max_cut_length (b : Board) (d : Division) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_cut_length_30x30_225parts (b : Board) (d : Division) :
  b.size = 30 ∧ d.num_parts = 225 ∧ d.equal_area = true →
  max_cut_length b d = 1065 :=
sorry

end NUMINAMATH_CALUDE_max_cut_length_30x30_225parts_l2672_267275


namespace NUMINAMATH_CALUDE_extremum_and_minimum_l2672_267215

-- Define the function f(x) = x³ - 3ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem extremum_and_minimum (a : ℝ) :
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| ∧ |h| < ε → f a (-1 + h) ≤ f a (-1) ∨ f a (-1 + h) ≥ f a (-1)) →
  a = 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) (1 : ℝ) → f a x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_minimum_l2672_267215


namespace NUMINAMATH_CALUDE_parameterization_validity_l2672_267205

def line (x : ℝ) : ℝ := -3 * x + 4

def is_valid_parameterization (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line (p.1 + t * v.1) = p.2 + t * v.2

theorem parameterization_validity :
  (is_valid_parameterization (0, 4) (1, -3)) ∧
  (is_valid_parameterization (-4, 16) (1/3, -1)) ∧
  ¬(is_valid_parameterization (1/3, 0) (3, -1)) ∧
  ¬(is_valid_parameterization (2, -2) (4, -12)) ∧
  ¬(is_valid_parameterization (1, 1) (0.5, -1.5)) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l2672_267205


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l2672_267214

/-- Proves that the cost price of a computer table is 6625 when the selling price is 8215 with a 24% markup -/
theorem computer_table_cost_price (selling_price : ℕ) (markup_percentage : ℕ) (cost_price : ℕ) :
  selling_price = 8215 →
  markup_percentage = 24 →
  selling_price = cost_price + (cost_price * markup_percentage) / 100 →
  cost_price = 6625 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l2672_267214


namespace NUMINAMATH_CALUDE_other_diagonal_length_l2672_267210

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  diag1 : ℝ
  diag2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diag1 * r.diag2) / 2

theorem other_diagonal_length (r : Rhombus) 
  (h1 : r.diag1 = 14)
  (h2 : r.area = 140) : 
  r.diag2 = 20 := by
sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l2672_267210


namespace NUMINAMATH_CALUDE_power_24_in_terms_of_P_l2672_267281

theorem power_24_in_terms_of_P (a b : ℕ) (P : ℝ) (h_P : P = 2^a) : 24^(a*b) = P^(3*b) * 3^(a*b) := by
  sorry

end NUMINAMATH_CALUDE_power_24_in_terms_of_P_l2672_267281


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l2672_267271

theorem prime_equation_solutions :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  (p^(2*q) + q^(2*p)) / (p^3 - p*q + q^3) = r →
  ((p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 3 ∧ q = 2 ∧ r = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l2672_267271


namespace NUMINAMATH_CALUDE_divisibility_condition_l2672_267237

theorem divisibility_condition (n : ℕ+) : 
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2672_267237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_implication_l2672_267250

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_implication
  (a b : ℕ → ℝ)
  (h : ∀ n : ℕ, b n = a n + a (n + 1)) :
  is_arithmetic_sequence a → is_arithmetic_sequence b ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_implication_l2672_267250


namespace NUMINAMATH_CALUDE_system_solution_l2672_267239

theorem system_solution : ∃! (x y : ℝ), 
  (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
  (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
  x = (1 : ℝ) / 2 ∧ y = -(3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2672_267239


namespace NUMINAMATH_CALUDE_least_n_for_phi_cube_l2672_267233

def phi (n : ℕ) : ℕ := sorry

theorem least_n_for_phi_cube (n : ℕ) : 
  (∀ m < n, phi (phi (phi m)) * phi (phi m) * phi m ≠ 64000) ∧ 
  (phi (phi (phi n)) * phi (phi n) * phi n = 64000) → 
  n = 41 := by sorry

end NUMINAMATH_CALUDE_least_n_for_phi_cube_l2672_267233


namespace NUMINAMATH_CALUDE_perpendicular_lines_condition_l2672_267285

theorem perpendicular_lines_condition (m : ℝ) :
  (m = -1) ↔ (∀ x y : ℝ, mx + y - 3 = 0 → 2*x + m*(m-1)*y + 2 = 0 → m*2 + 1*m*(m-1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_condition_l2672_267285


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2672_267226

/-- Counts 4-digit numbers beginning with 2 that have exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 10 -- Total number of digits (0-9)
  let first_digit := 2 -- First digit is always 2
  let remaining_digits := digits - 1 -- Excluding 2
  let configurations := 2 -- Two main configurations: 2 is repeated or not

  -- When 2 is one of the repeated digits
  let case1 := 3 * remaining_digits * remaining_digits

  -- When 2 is not one of the repeated digits
  let case2 := 3 * remaining_digits * remaining_digits

  case1 + case2

theorem count_special_numbers_eq_384 : count_special_numbers = 384 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_384_l2672_267226


namespace NUMINAMATH_CALUDE_tan_five_pi_over_four_l2672_267230

theorem tan_five_pi_over_four : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_over_four_l2672_267230


namespace NUMINAMATH_CALUDE_parity_of_expression_l2672_267238

theorem parity_of_expression (o₁ o₂ n : ℤ) 
  (h₁ : ∃ k : ℤ, o₁ = 2*k + 1) 
  (h₂ : ∃ m : ℤ, o₂ = 2*m + 1) :
  (o₁^2 + n*o₁*o₂) % 2 = 1 ↔ n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parity_of_expression_l2672_267238


namespace NUMINAMATH_CALUDE_z_equals_negative_four_l2672_267234

theorem z_equals_negative_four (x y z : ℤ) : x = 2 → y = x^2 - 5 → z = y^2 - 5 → z = -4 := by
  sorry

end NUMINAMATH_CALUDE_z_equals_negative_four_l2672_267234


namespace NUMINAMATH_CALUDE_billion_yuan_scientific_notation_l2672_267262

/-- Represents the value of 209.6 billion yuan in standard form -/
def billion_yuan : ℝ := 209.6 * (10^9)

/-- Represents the scientific notation of 209.6 billion yuan -/
def scientific_notation : ℝ := 2.096 * (10^10)

/-- Theorem stating that the standard form equals the scientific notation -/
theorem billion_yuan_scientific_notation : billion_yuan = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_yuan_scientific_notation_l2672_267262


namespace NUMINAMATH_CALUDE_square_plot_poles_l2672_267209

/-- The number of fence poles needed for a square plot -/
def total_poles (poles_per_side : ℕ) : ℕ :=
  poles_per_side * 4 - 4

/-- Theorem: A square plot with 27 poles per side requires 104 poles in total -/
theorem square_plot_poles :
  total_poles 27 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_poles_l2672_267209


namespace NUMINAMATH_CALUDE_winning_team_arrangements_winning_team_groupings_winning_team_selections_l2672_267273

/-- A debate team with male and female members -/
structure DebateTeam where
  male_members : ℕ
  female_members : ℕ

/-- The national winning debate team -/
def winning_team : DebateTeam :=
  { male_members := 3, female_members := 5 }

/-- Number of arrangements with male members not adjacent -/
def non_adjacent_arrangements (team : DebateTeam) : ℕ := sorry

/-- Number of ways to divide into pairs for classes -/
def pair_groupings (team : DebateTeam) (num_classes : ℕ) : ℕ := sorry

/-- Number of ways to select debaters with at least one male -/
def debater_selections (team : DebateTeam) (num_debaters : ℕ) : ℕ := sorry

theorem winning_team_arrangements :
  non_adjacent_arrangements winning_team = 14400 := by sorry

theorem winning_team_groupings :
  pair_groupings winning_team 4 = 2520 := by sorry

theorem winning_team_selections :
  debater_selections winning_team 4 = 1560 := by sorry

end NUMINAMATH_CALUDE_winning_team_arrangements_winning_team_groupings_winning_team_selections_l2672_267273


namespace NUMINAMATH_CALUDE_equation_solution_l2672_267264

theorem equation_solution : ∃ x : ℚ, (2 * x + 1) / 5 - x / 10 = 2 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2672_267264


namespace NUMINAMATH_CALUDE_decimal_6_to_binary_l2672_267211

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_6_to_binary :
  decimal_to_binary 6 = [1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_decimal_6_to_binary_l2672_267211


namespace NUMINAMATH_CALUDE_f_property_l2672_267289

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1)^2 - a * Real.log x

theorem f_property (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → y₁ > 0 → y₂ > 0 →
      (f a (y₁ + 1) - f a (y₂ + 1)) / (y₁ - y₂) > 1) →
  a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_f_property_l2672_267289


namespace NUMINAMATH_CALUDE_least_k_value_l2672_267291

theorem least_k_value (p q k : ℕ) : 
  p > 1 → 
  q > 1 → 
  p + q = 36 → 
  17 * (p + 1) = k * (q + 1) → 
  k ≥ 2 ∧ (∃ (p' q' : ℕ), p' > 1 ∧ q' > 1 ∧ p' + q' = 36 ∧ 17 * (p' + 1) = 2 * (q' + 1)) :=
by sorry

end NUMINAMATH_CALUDE_least_k_value_l2672_267291


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l2672_267217

theorem set_equality_implies_sum (a b : ℝ) : 
  ({1, 2} : Set ℝ) = {a, b} → (2 * a + b = 4 ∨ 2 * a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l2672_267217


namespace NUMINAMATH_CALUDE_thousands_digit_of_common_remainder_l2672_267255

theorem thousands_digit_of_common_remainder (n : ℕ) 
  (h1 : n > 1000000)
  (h2 : n % 40 = n % 625) : 
  (n / 1000) % 10 = 0 ∨ (n / 1000) % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_thousands_digit_of_common_remainder_l2672_267255


namespace NUMINAMATH_CALUDE_five_letter_words_with_at_least_two_vowels_l2672_267201

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def count_words (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n

def count_words_with_exactly_k_vowels (k : Nat) : Nat :=
  Nat.choose word_length k * vowels.card ^ k * (alphabet.card - vowels.card) ^ (word_length - k)

theorem five_letter_words_with_at_least_two_vowels : 
  count_words alphabet word_length - 
  (count_words_with_exactly_k_vowels 0 + count_words_with_exactly_k_vowels 1) = 4192 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_with_at_least_two_vowels_l2672_267201


namespace NUMINAMATH_CALUDE_max_food_per_guest_l2672_267276

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ) (max_food : ℚ) : 
  total_food = 337 → min_guests = 169 → max_food = 2 → 
  max_food = (total_food : ℚ) / min_guests :=
by sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l2672_267276


namespace NUMINAMATH_CALUDE_cart_distance_l2672_267277

/-- The distance traveled by a cart with three wheels of different circumferences -/
theorem cart_distance (front_circ rear_circ third_circ : ℕ)
  (h1 : front_circ = 30)
  (h2 : rear_circ = 32)
  (h3 : third_circ = 34)
  (rev_rear : ℕ)
  (h4 : front_circ * (rev_rear + 5) = rear_circ * rev_rear)
  (h5 : third_circ * (rev_rear - 8) = rear_circ * rev_rear) :
  rear_circ * rev_rear = 2400 :=
sorry

end NUMINAMATH_CALUDE_cart_distance_l2672_267277


namespace NUMINAMATH_CALUDE_min_square_side_for_9x21_l2672_267235

/-- The minimum side length of a square that can contain 9x21 rectangles without rotation and overlap -/
def min_square_side (width : ℕ) (length : ℕ) : ℕ :=
  Nat.lcm width length

/-- Theorem stating that the minimum side length for 9x21 rectangles is 63 -/
theorem min_square_side_for_9x21 :
  min_square_side 9 21 = 63 := by sorry

end NUMINAMATH_CALUDE_min_square_side_for_9x21_l2672_267235


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l2672_267252

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 40 →
  (sarah_score + greg_score) / 2 = 102 →
  sarah_score = 122 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l2672_267252


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l2672_267202

/-- Given a cylinder with volume 72π cm³, prove that a sphere with the same radius has volume 96π cm³ -/
theorem sphere_volume_from_cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → (4/3) * π * r^3 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l2672_267202


namespace NUMINAMATH_CALUDE_marie_erasers_l2672_267249

/-- The number of erasers Marie loses -/
def erasers_lost : ℕ := 42

/-- The number of erasers Marie ends up with -/
def erasers_left : ℕ := 53

/-- The initial number of erasers Marie had -/
def initial_erasers : ℕ := erasers_left + erasers_lost

theorem marie_erasers : initial_erasers = 95 := by sorry

end NUMINAMATH_CALUDE_marie_erasers_l2672_267249


namespace NUMINAMATH_CALUDE_regression_analysis_l2672_267299

-- Define the data points
def data : List (ℝ × ℝ) := [(5, 17), (6, 20), (8, 25), (9, 28), (12, 35)]

-- Define the regression equation
def regression_equation (x : ℝ) (a : ℝ) : ℝ := 2.6 * x + a

-- Theorem statement
theorem regression_analysis :
  -- 1. Center point
  (let x_mean := (data.map Prod.fst).sum / data.length
   let y_mean := (data.map Prod.snd).sum / data.length
   (x_mean, y_mean) = (8, 25)) ∧
  -- 2. Y-intercept
  (∃ a : ℝ, a = 4.2 ∧
    regression_equation 8 a = 25) ∧
  -- 3. Residual when x = 5
  (let a := 4.2
   let y_pred := regression_equation 5 a
   let y_actual := 17
   y_actual - y_pred = -0.2) := by
  sorry


end NUMINAMATH_CALUDE_regression_analysis_l2672_267299


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l2672_267245

/-- Proves that given a tank with 100 gallons of pure water and 66.67 gallons of saline solution
    added to create a 10% salt solution, the original saline solution must have contained 25% salt. -/
theorem saline_solution_concentration
  (pure_water : ℝ)
  (saline_added : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water = 100)
  (h2 : saline_added = 66.67)
  (h3 : final_concentration = 0.1)
  : (final_concentration * (pure_water + saline_added)) / saline_added = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l2672_267245
