import Mathlib

namespace smallest_interesting_number_l3443_344365

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- The smallest interesting number is 1800. -/
theorem smallest_interesting_number : 
  (is_interesting 1800 ∧ ∀ m < 1800, ¬ is_interesting m) :=
sorry

end smallest_interesting_number_l3443_344365


namespace truck_count_l3443_344343

theorem truck_count (tanks trucks : ℕ) : 
  tanks = 5 * trucks →
  tanks + trucks = 140 →
  trucks = 23 := by
sorry

end truck_count_l3443_344343


namespace kay_age_is_32_l3443_344396

-- Define Kay's age and the number of siblings
def kay_age : ℕ := sorry
def num_siblings : ℕ := 14

-- Define the ages of the youngest and oldest siblings
def youngest_sibling_age : ℕ := kay_age / 2 - 5
def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

-- State the theorem
theorem kay_age_is_32 :
  num_siblings = 14 ∧
  youngest_sibling_age = kay_age / 2 - 5 ∧
  oldest_sibling_age = 4 * youngest_sibling_age ∧
  oldest_sibling_age = 44 →
  kay_age = 32 :=
by sorry

end kay_age_is_32_l3443_344396


namespace roots_sum_theorem_l3443_344375

theorem roots_sum_theorem (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^5*q + p*q^5 + q^3 = 617 := by
  sorry

end roots_sum_theorem_l3443_344375


namespace vanessa_score_l3443_344353

theorem vanessa_score (team_score : ℕ) (other_players : ℕ) (other_avg : ℚ) : 
  team_score = 48 → 
  other_players = 6 → 
  other_avg = 3.5 → 
  team_score - (other_players : ℚ) * other_avg = 27 := by
  sorry

end vanessa_score_l3443_344353


namespace composite_division_l3443_344395

def first_four_composites : List Nat := [4, 6, 8, 9]
def next_four_composites : List Nat := [10, 12, 14, 15]

theorem composite_division :
  (first_four_composites.prod : ℚ) / (next_four_composites.prod : ℚ) = 12 / 175 := by
  sorry

end composite_division_l3443_344395


namespace max_value_theorem_l3443_344338

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 + x * y - y^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 2 / 4 ∧ 
  ∀ (z : ℝ), z = (x - 2*y) / (5*x^2 - 2*x*y + 2*y^2) → z ≤ M :=
by sorry

end max_value_theorem_l3443_344338


namespace expression_evaluation_l3443_344393

theorem expression_evaluation : 
  0.064^(-1/3) - (-7/9)^0 + ((-2)^3)^(1/3) - 16^(-0.75) = -5/8 := by
  sorry

end expression_evaluation_l3443_344393


namespace passengers_boarded_in_north_carolina_l3443_344310

/-- Represents the number of passengers at different stages of the flight --/
structure FlightPassengers where
  initial : Nat
  afterTexas : Nat
  afterNorthCarolina : Nat
  final : Nat

/-- Represents the changes in passenger numbers during layovers --/
structure LayoverChanges where
  texasOff : Nat
  texasOn : Nat
  northCarolinaOff : Nat

/-- The main theorem about the flight --/
theorem passengers_boarded_in_north_carolina 
  (fp : FlightPassengers) 
  (lc : LayoverChanges) 
  (crew : Nat) 
  (h1 : fp.initial = 124)
  (h2 : lc.texasOff = 58)
  (h3 : lc.texasOn = 24)
  (h4 : lc.northCarolinaOff = 47)
  (h5 : crew = 10)
  (h6 : fp.final + crew = 67)
  (h7 : fp.afterTexas = fp.initial - lc.texasOff + lc.texasOn)
  (h8 : fp.afterNorthCarolina = fp.afterTexas - lc.northCarolinaOff)
  : fp.final - fp.afterNorthCarolina = 14 := by
  sorry

#check passengers_boarded_in_north_carolina

end passengers_boarded_in_north_carolina_l3443_344310


namespace food_distribution_l3443_344371

theorem food_distribution (initial_men : ℕ) (initial_days : ℕ) (additional_men : ℕ) (days_before_increase : ℕ) : 
  initial_men = 760 →
  initial_days = 22 →
  additional_men = 40 →
  days_before_increase = 2 →
  (initial_men * initial_days - initial_men * days_before_increase) / (initial_men + additional_men) = 19 :=
by sorry

end food_distribution_l3443_344371


namespace least_integer_absolute_value_l3443_344387

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → |3 * y + 4| > 18) ∧ |3 * x + 4| ≤ 18 → x = -7 :=
sorry

end least_integer_absolute_value_l3443_344387


namespace carton_height_theorem_l3443_344359

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the area of a rectangle given its length and width -/
def rectangleArea (length width : ℕ) : ℕ :=
  length * width

/-- Calculates the number of smaller rectangles that can fit in a larger rectangle -/
def fitRectangles (largeLength largeWidth smallLength smallWidth : ℕ) : ℕ :=
  (largeLength / smallLength) * (largeWidth / smallWidth)

/-- The main theorem about the carton height -/
theorem carton_height_theorem 
  (cartonLength cartonWidth : ℕ)
  (soapBox : BoxDimensions)
  (maxSoapBoxes : ℕ) :
  cartonLength = 30 →
  cartonWidth = 42 →
  soapBox.length = 7 →
  soapBox.width = 6 →
  soapBox.height = 5 →
  maxSoapBoxes = 360 →
  ∃ (cartonHeight : ℕ), cartonHeight = 60 ∧
    cartonHeight * fitRectangles cartonLength cartonWidth soapBox.length soapBox.width = 
    maxSoapBoxes * soapBox.height :=
sorry

end carton_height_theorem_l3443_344359


namespace system_solution_inequality_solution_l3443_344399

theorem system_solution :
  ∃! (x y : ℝ), (6 * x - 2 * y = 1 ∧ 2 * x + y = 2) ∧
  x = (1/2 : ℝ) ∧ y = 1 := by sorry

theorem inequality_solution :
  ∀ x : ℝ, (2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1) ↔ (2 < x ∧ x < 5) := by sorry

end system_solution_inequality_solution_l3443_344399


namespace unique_determination_of_polynomial_minimality_of_points_l3443_344394

/-- A polynomial of degree 2017 with integer coefficients and leading coefficient 1 -/
def IntPolynomial2017 : Type := 
  {p : Polynomial ℤ // p.degree = 2017 ∧ p.leadingCoeff = 1}

/-- The minimum number of points needed to uniquely determine the polynomial -/
def minPointsForUniqueness : ℕ := 2017

theorem unique_determination_of_polynomial (p q : IntPolynomial2017) 
  (points : Fin minPointsForUniqueness → ℤ) :
  (∀ i : Fin minPointsForUniqueness, p.val.eval (points i) = q.val.eval (points i)) →
  p = q :=
sorry

theorem minimality_of_points :
  ∀ k : ℕ, k < minPointsForUniqueness →
  ∃ (p q : IntPolynomial2017) (points : Fin k → ℤ),
    (∀ i : Fin k, p.val.eval (points i) = q.val.eval (points i)) ∧
    p ≠ q :=
sorry

end unique_determination_of_polynomial_minimality_of_points_l3443_344394


namespace chef_potato_problem_l3443_344363

theorem chef_potato_problem (cooked : ℕ) (cook_time : ℕ) (remaining_time : ℕ) : 
  cooked = 7 → 
  cook_time = 5 → 
  remaining_time = 45 → 
  cooked + remaining_time / cook_time = 16 := by
sorry

end chef_potato_problem_l3443_344363


namespace parabola_point_D_l3443_344379

/-- A parabola passing through three given points -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  point_A : a * (-0.8)^2 + b * (-0.8) + c = 4.132
  point_B : a * 1.2^2 + b * 1.2 + c = -1.948
  point_C : a * 2.8^2 + b * 2.8 + c = -3.932

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def y_coordinate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_point_D (p : Parabola) : y_coordinate p 1.8 = -2.992 := by
  sorry

end parabola_point_D_l3443_344379


namespace total_cleaner_needed_l3443_344398

/-- Amount of cleaner needed for a dog stain in ounces -/
def dog_cleaner : ℕ := 6

/-- Amount of cleaner needed for a cat stain in ounces -/
def cat_cleaner : ℕ := 4

/-- Amount of cleaner needed for a rabbit stain in ounces -/
def rabbit_cleaner : ℕ := 1

/-- Number of dogs -/
def num_dogs : ℕ := 6

/-- Number of cats -/
def num_cats : ℕ := 3

/-- Number of rabbits -/
def num_rabbits : ℕ := 1

/-- Theorem stating the total amount of cleaner needed -/
theorem total_cleaner_needed : 
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = 49 := by
  sorry

end total_cleaner_needed_l3443_344398


namespace jane_calculation_l3443_344354

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - 2 * (y - 3 * z) = 25)
  (h2 : x - 2 * y - 3 * z = 7) :
  x - 2 * y = 13 := by
sorry

end jane_calculation_l3443_344354


namespace staplers_remaining_l3443_344316

/-- The number of staplers left after stapling reports -/
def staplers_left (initial_staplers : ℕ) (dozens_stapled : ℕ) : ℕ :=
  initial_staplers - dozens_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_remaining : staplers_left 50 3 = 14 := by
  sorry

end staplers_remaining_l3443_344316


namespace model_comparison_theorem_l3443_344361

/-- A model for fitting data -/
structure Model where
  /-- The sum of squared residuals for this model -/
  sumSquaredResiduals : ℝ
  /-- Whether the residual points are uniformly distributed in a horizontal band -/
  uniformResiduals : Prop

/-- Compares the fitting effects of two models -/
def betterFit (m1 m2 : Model) : Prop :=
  m1.sumSquaredResiduals < m2.sumSquaredResiduals

/-- Indicates whether a model is appropriate based on its residual plot -/
def appropriateModel (m : Model) : Prop :=
  m.uniformResiduals

theorem model_comparison_theorem :
  ∀ (m1 m2 : Model),
    (betterFit m1 m2 → m1.sumSquaredResiduals < m2.sumSquaredResiduals) ∧
    (appropriateModel m1 ↔ m1.uniformResiduals) :=
by sorry

end model_comparison_theorem_l3443_344361


namespace sequence_property_l3443_344389

theorem sequence_property (a : ℕ → ℝ) 
  (h : ∀ m : ℕ, m > 1 → a (m + 1) * a (m - 1) = a m ^ 2 - a 1 ^ 2) :
  ∀ m n : ℕ, m > n ∧ n > 1 → a (m + n) * a (m - n) = a m ^ 2 - a n ^ 2 :=
by sorry

end sequence_property_l3443_344389


namespace repeating_56_equals_fraction_l3443_344345

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

/-- The repeating decimal 0.56̄ -/
def repeating_56 : ℚ := RepeatingDecimal 5 6

theorem repeating_56_equals_fraction : repeating_56 = 56 / 99 := by
  sorry

end repeating_56_equals_fraction_l3443_344345


namespace power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l3443_344308

theorem power_two_eq_square_plus_one_solutions (x n : ℕ) :
  2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := by sorry

theorem power_two_plus_one_eq_square_solution (x n : ℕ) :
  2^n + 1 = x^2 ↔ x = 3 ∧ n = 3 := by sorry

end power_two_eq_square_plus_one_solutions_power_two_plus_one_eq_square_solution_l3443_344308


namespace truck_wheels_l3443_344372

/-- Toll calculation function -/
def toll (x : ℕ) : ℚ :=
  0.5 + 0.5 * (x - 2)

/-- Number of wheels on the front axle -/
def frontWheels : ℕ := 2

/-- Number of wheels on each non-front axle -/
def otherWheels : ℕ := 4

/-- Theorem stating the total number of wheels on the truck -/
theorem truck_wheels (x : ℕ) (h1 : toll x = 2) (h2 : x > 0) : 
  frontWheels + (x - 1) * otherWheels = 18 := by
  sorry

#check truck_wheels

end truck_wheels_l3443_344372


namespace n_has_21_digits_l3443_344337

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect fourth power -/
axiom n_sq_fourth_power : ∃ k : ℕ, n^2 = k^4

/-- n^3 is a perfect fifth power -/
axiom n_cube_fifth_power : ∃ k : ℕ, n^3 = k^5

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (30 ∣ m) → (∃ k : ℕ, m^2 = k^4) → (∃ k : ℕ, m^3 = k^5) → n ≤ m

/-- The number of digits in a natural number -/
def num_digits (x : ℕ) : ℕ := sorry

/-- The main theorem: n has 21 digits -/
theorem n_has_21_digits : num_digits n = 21 := by sorry

end n_has_21_digits_l3443_344337


namespace mermaid_seashell_age_l3443_344344

/-- Converts a base-9 number to base-10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The mermaid's seashell collection age conversion theorem --/
theorem mermaid_seashell_age :
  base9_to_base10 3 6 2 = 299 := by
  sorry

end mermaid_seashell_age_l3443_344344


namespace circle_center_distance_l3443_344348

theorem circle_center_distance (x y : ℝ) :
  x^2 + y^2 = 8*x - 2*y + 23 →
  Real.sqrt ((4 - (-3))^2 + (-1 - 4)^2) = Real.sqrt 74 :=
by sorry

end circle_center_distance_l3443_344348


namespace min_value_a_l3443_344340

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
sorry

end min_value_a_l3443_344340


namespace sequence_inequality_l3443_344302

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n, n ≥ 2 ∧ n ≤ 100 → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 := by
  sorry

end sequence_inequality_l3443_344302


namespace smallest_top_block_l3443_344386

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : ℕ)

/-- The rule for assigning numbers to upper blocks -/
def upper_block_rule (a b c : ℕ) : ℕ := 2 * (a + b + c)

/-- The pyramid satisfies the numbering rules -/
def valid_pyramid (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16) ∧
  (∀ i : Fin 10, ∃ a b c : Fin 15, p.layer2 i = upper_block_rule (p.layer1 a) (p.layer1 b) (p.layer1 c)) ∧
  (∀ i : Fin 6, ∃ a b c : Fin 10, p.layer3 i = upper_block_rule (p.layer2 a) (p.layer2 b) (p.layer2 c)) ∧
  (∃ a b c : Fin 6, p.layer4 = upper_block_rule (p.layer3 a) (p.layer3 b) (p.layer3 c))

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_block (p : Pyramid) (h : valid_pyramid p) : p.layer4 ≥ 48 := by
  sorry

end smallest_top_block_l3443_344386


namespace tan_22_5_degree_decomposition_l3443_344377

theorem tan_22_5_degree_decomposition :
  ∃ (a b c : ℕ+), 
    (a.val ≥ b.val ∧ b.val ≥ c.val) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a.val - 1 + Real.sqrt b.val - Real.sqrt c.val) ∧
    (a.val + b.val + c.val = 12) := by sorry

end tan_22_5_degree_decomposition_l3443_344377


namespace hcf_problem_l3443_344370

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 45276) (h2 : Nat.lcm a b = 2058) :
  Nat.gcd a b = 22 := by sorry

end hcf_problem_l3443_344370


namespace expression_simplification_and_evaluation_l3443_344327

theorem expression_simplification_and_evaluation :
  let a : ℤ := -1
  let b : ℤ := -2
  let original_expression := (2*a + b)*(b - 2*a) - (a - 3*b)^2
  let simplified_expression := -5*a^2 + 6*a*b - 8*b^2
  original_expression = simplified_expression ∧ simplified_expression = -25 := by
sorry

end expression_simplification_and_evaluation_l3443_344327


namespace largest_s_value_largest_s_value_is_121_l3443_344373

/-- The largest possible value of s for regular polygons Q1 (r-gon) and Q2 (s-gon) 
    satisfying the given conditions -/
theorem largest_s_value : ℕ :=
  let r : ℕ → ℕ := fun s => 120 * s / (122 - s)
  let interior_angle : ℕ → ℚ := fun n => (n - 2 : ℚ) * 180 / n
  let s_max := 121
  have h1 : ∀ s : ℕ, s ≥ 3 → s ≤ s_max → 
    (interior_angle (r s)) / (interior_angle s) = 61 / 60 := by sorry
  have h2 : ∀ s : ℕ, s > s_max → ¬(∃ r : ℕ, r ≥ s ∧ 
    (interior_angle r) / (interior_angle s) = 61 / 60) := by sorry
  s_max

/-- Proof that the largest possible value of s is indeed 121 -/
theorem largest_s_value_is_121 : largest_s_value = 121 := by sorry

end largest_s_value_largest_s_value_is_121_l3443_344373


namespace exists_natural_sqrt_nested_root_l3443_344323

theorem exists_natural_sqrt_nested_root : ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (n : ℝ)^(7/8) = m := by
  sorry

end exists_natural_sqrt_nested_root_l3443_344323


namespace tangent_line_hyperbola_l3443_344300

/-- The equation of the tangent line to the hyperbola x^2 - y^2/2 = 1 at the point (√2, √2) is 2x - y - √2 = 0 -/
theorem tangent_line_hyperbola (x y : ℝ) :
  (x^2 - y^2/2 = 1) →
  let P : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  let tangent_line := fun (x y : ℝ) ↦ 2*x - y - Real.sqrt 2 = 0
  (x = P.1 ∧ y = P.2) →
  tangent_line x y :=
by sorry

end tangent_line_hyperbola_l3443_344300


namespace subset_condition_l3443_344380

def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 := by sorry

end subset_condition_l3443_344380


namespace fraction_subtraction_l3443_344311

theorem fraction_subtraction (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a - x) / (b - x) = c / d) 
  (h4 : d ≠ c) : 
  x = (b * c - a * d) / (d - c) := by
sorry

end fraction_subtraction_l3443_344311


namespace parabola_slope_theorem_l3443_344309

/-- A parabola with equation y² = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and three points on it, prove that the slopes of the lines
    formed by these points satisfy a specific equation -/
theorem parabola_slope_theorem (C : Parabola) (A B P M N : Point) 
  (h_A : A.y^2 = 2 * C.p * A.x) 
  (h_A_x : A.x = 1)
  (h_B : B.y = 0 ∧ B.x = -C.p/2)
  (h_AB : (A.x - B.x)^2 + (A.y - B.y)^2 = 8)
  (h_P : P.y^2 = 2 * C.p * P.x ∧ P.y = 2)
  (h_M : M.y^2 = 2 * C.p * M.x)
  (h_N : N.y^2 = 2 * C.p * N.x)
  (k₁ k₂ k₃ : ℝ)
  (h_k₁ : k₁ = (M.y - P.y) / (M.x - P.x))
  (h_k₂ : k₂ = (N.y - P.y) / (N.x - P.x))
  (h_k₃ : k₃ = (N.y - M.y) / (N.x - M.x)) :
  1/k₁ + 1/k₂ - 1/k₃ = 1 := by
  sorry

end parabola_slope_theorem_l3443_344309


namespace tailwind_speed_l3443_344350

/-- Given a plane's ground speeds with and against a tailwind, calculate the speed of the tailwind. -/
theorem tailwind_speed (speed_with_wind speed_against_wind : ℝ) 
  (h1 : speed_with_wind = 460)
  (h2 : speed_against_wind = 310) :
  ∃ (plane_speed wind_speed : ℝ),
    plane_speed + wind_speed = speed_with_wind ∧
    plane_speed - wind_speed = speed_against_wind ∧
    wind_speed = 75 := by
  sorry

end tailwind_speed_l3443_344350


namespace total_nut_weight_l3443_344320

/-- Represents the weight of nuts in kilograms -/
structure NutWeight where
  almonds : Float
  pecans : Float
  walnuts : Float
  cashews : Float
  pistachios : Float
  brazilNuts : Float
  macadamiaNuts : Float
  hazelnuts : Float

/-- Conversion rate from ounces to kilograms -/
def ounceToKgRate : Float := 0.0283495

/-- Weights of nuts bought by the chef -/
def chefNuts : NutWeight where
  almonds := 0.14
  pecans := 0.38
  walnuts := 0.22
  cashews := 0.47
  pistachios := 0.29
  brazilNuts := 6 * ounceToKgRate
  macadamiaNuts := 4.5 * ounceToKgRate
  hazelnuts := 7.3 * ounceToKgRate

/-- Theorem stating the total weight of nuts bought by the chef -/
theorem total_nut_weight : 
  chefNuts.almonds + chefNuts.pecans + chefNuts.walnuts + chefNuts.cashews + 
  chefNuts.pistachios + chefNuts.brazilNuts + chefNuts.macadamiaNuts + 
  chefNuts.hazelnuts = 2.1128216 := by
  sorry

end total_nut_weight_l3443_344320


namespace min_lateral_surface_area_cone_l3443_344321

/-- Given a cone with volume 4π/3, its minimum lateral surface area is 2√3π. -/
theorem min_lateral_surface_area_cone (r h : ℝ) (h_volume : (1/3) * π * r^2 * h = (4/3) * π) :
  ∃ (S_min : ℝ), S_min = 2 * Real.sqrt 3 * π ∧ 
  ∀ (S : ℝ), S = π * r * Real.sqrt (r^2 + h^2) → S ≥ S_min :=
sorry

end min_lateral_surface_area_cone_l3443_344321


namespace payment_per_task_l3443_344384

/-- Calculates the payment per task for a contractor given their work schedule and total earnings -/
theorem payment_per_task (hours_per_task : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ) (total_earnings : ℝ) :
  hours_per_task = 2 →
  hours_per_day = 10 →
  days_per_week = 5 →
  total_earnings = 1400 →
  (total_earnings / (days_per_week * (hours_per_day / hours_per_task))) = 56 := by
sorry

end payment_per_task_l3443_344384


namespace root_triple_relation_l3443_344301

theorem root_triple_relation (p q r : ℝ) (h : ∃ x y : ℝ, p * x^2 + q * x + r = 0 ∧ p * y^2 + q * y + r = 0 ∧ y = 3 * x) :
  3 * q^2 = 8 * p * r := by
sorry

end root_triple_relation_l3443_344301


namespace negative_1234_mod_9_l3443_344326

theorem negative_1234_mod_9 : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end negative_1234_mod_9_l3443_344326


namespace probability_of_winning_pair_l3443_344328

/-- A card in the deck -/
structure Card where
  color : Bool  -- True for red, False for green
  label : Fin 5 -- Labels A, B, C, D, E represented as 0, 1, 2, 3, 4

/-- The deck of cards -/
def deck : Finset Card := sorry

/-- A pair of cards is winning if they have the same color or the same label -/
def is_winning_pair (c1 c2 : Card) : Bool :=
  c1.color = c2.color ∨ c1.label = c2.label

/-- The set of all possible pairs of cards -/
def all_pairs : Finset (Card × Card) := sorry

/-- The set of winning pairs -/
def winning_pairs : Finset (Card × Card) := sorry

/-- The probability of drawing a winning pair -/
theorem probability_of_winning_pair :
  (winning_pairs.card : ℚ) / all_pairs.card = 35 / 66 := by sorry

end probability_of_winning_pair_l3443_344328


namespace f_prime_at_i_l3443_344305

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^4 - x^2

-- State the theorem
theorem f_prime_at_i : 
  (deriv f) i = -6 * i := by sorry

end f_prime_at_i_l3443_344305


namespace fraction_powers_equality_l3443_344364

theorem fraction_powers_equality : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end fraction_powers_equality_l3443_344364


namespace isosceles_trapezoid_leg_length_l3443_344374

/-- An isosceles trapezoid circumscribed around a circle with area S and acute base angle π/6 has leg length √(2S) -/
theorem isosceles_trapezoid_leg_length (S : ℝ) (h_pos : S > 0) :
  ∃ (x : ℝ),
    x > 0 ∧
    x = Real.sqrt (2 * S) ∧
    ∃ (a b h : ℝ),
      a > 0 ∧ b > 0 ∧ h > 0 ∧
      a + b = 2 * x ∧
      h = x * Real.sin (π / 6) ∧
      S = (a + b) * h / 2 :=
by sorry

end isosceles_trapezoid_leg_length_l3443_344374


namespace find_S_l3443_344369

theorem find_S : ∃ S : ℚ, (1/4 : ℚ) * (1/6 : ℚ) * S = (1/5 : ℚ) * (1/8 : ℚ) * 160 ∧ S = 96 := by
  sorry

end find_S_l3443_344369


namespace cube_nested_square_root_l3443_344351

theorem cube_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^3 = 8 := by
  sorry

end cube_nested_square_root_l3443_344351


namespace union_of_A_and_B_l3443_344358

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 0}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end union_of_A_and_B_l3443_344358


namespace marble_remainder_l3443_344360

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end marble_remainder_l3443_344360


namespace set_intersection_union_l3443_344368

theorem set_intersection_union (M N P : Set ℕ) : 
  M = {1} → N = {1, 2} → P = {1, 2, 3} → (M ∪ N) ∩ P = {1, 2} := by
  sorry

end set_intersection_union_l3443_344368


namespace star_properties_l3443_344342

/-- Custom multiplication operation for rational numbers -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the star operation -/
theorem star_properties :
  (star 2 3 = 7) ∧
  (star (star 1 4) (-1/2) = -3/2) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) := by
  sorry

end star_properties_l3443_344342


namespace integer_fraction_characterization_l3443_344356

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨ 
             (a = l ∧ b = 2 * l) ∨ 
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
sorry

end integer_fraction_characterization_l3443_344356


namespace butterfly_development_time_l3443_344355

/-- The time (in days) a butterfly spends in a cocoon -/
def cocoon_time : ℕ := 30

/-- The time (in days) a butterfly spends as a larva -/
def larva_time : ℕ := 3 * cocoon_time

/-- The total time (in days) from butterfly egg to butterfly -/
def total_time : ℕ := larva_time + cocoon_time

theorem butterfly_development_time : total_time = 120 := by
  sorry

end butterfly_development_time_l3443_344355


namespace newspaper_pieces_l3443_344304

theorem newspaper_pieces (petya_tears : ℕ) (vasya_tears : ℕ) (found_pieces : ℕ) :
  petya_tears = 5 →
  vasya_tears = 9 →
  found_pieces = 1988 →
  ∃ n : ℕ, (1 + n * (petya_tears - 1) + m * (vasya_tears - 1)) ≠ found_pieces :=
by sorry

end newspaper_pieces_l3443_344304


namespace negation_of_existence_negation_of_quadratic_equation_l3443_344391

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) := by sorry

end negation_of_existence_negation_of_quadratic_equation_l3443_344391


namespace plane_contains_line_and_parallel_to_intersection_l3443_344303

-- Define the line L
def L : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x - 1) / 2 = -y / 3 ∧ (x - 1) / 2 = 3 - z}

-- Define the two planes
def plane1 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 4*x + 5*z - 3 = 0}
def plane2 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x + y + 2*z = 0}

-- Define the plane P we want to prove
def P : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x - y + 7*z - 23 = 0}

-- Theorem statement
theorem plane_contains_line_and_parallel_to_intersection :
  (∀ p ∈ L, p ∈ P) ∧
  (∃ v : ℝ × ℝ × ℝ, v ≠ 0 ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ plane1 ∧ q ∈ plane1 ∧ p ∈ plane2 ∧ q ∈ plane2 → 
      ∃ t : ℝ, q - p = t • v) ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ P ∧ q ∈ P → 
      ∃ u : ℝ × ℝ × ℝ, u ≠ 0 ∧ q - p = u • v)) :=
by
  sorry

end plane_contains_line_and_parallel_to_intersection_l3443_344303


namespace intersection_of_A_and_B_l3443_344339

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < 3/2} := by sorry

end intersection_of_A_and_B_l3443_344339


namespace percentage_division_problem_l3443_344318

theorem percentage_division_problem : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end percentage_division_problem_l3443_344318


namespace reciprocal_of_negative_2023_l3443_344376

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end reciprocal_of_negative_2023_l3443_344376


namespace scooter_gain_percent_l3443_344385

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by sorry

end scooter_gain_percent_l3443_344385


namespace coefficient_x4_in_expansion_l3443_344347

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (1 - x)^5 * (2*x + 1))
  ∃ (a b c d e f : ℚ), 
    ∀ x, expansion x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = -15 := by
  sorry

end coefficient_x4_in_expansion_l3443_344347


namespace rat_count_proof_l3443_344317

def total_rats (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) : ℕ :=
  kenia_rats + hunter_rats + elodie_rats

theorem rat_count_proof (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) :
  kenia_rats = 3 * (hunter_rats + elodie_rats) →
  elodie_rats = 30 →
  elodie_rats = hunter_rats + 10 →
  total_rats kenia_rats hunter_rats elodie_rats = 200 :=
by
  sorry

end rat_count_proof_l3443_344317


namespace sqrt_180_simplification_l3443_344319

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_180_simplification_l3443_344319


namespace victor_stickers_l3443_344349

theorem victor_stickers (flower_stickers : ℕ) (total_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 → 
  total_stickers = 14 → 
  animal_stickers < flower_stickers → 
  flower_stickers + animal_stickers = total_stickers →
  animal_stickers = 6 := by
sorry

end victor_stickers_l3443_344349


namespace scientific_notation_of_400000_l3443_344322

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to represent in scientific notation -/
def number : ℝ := 400000

/-- The expected scientific notation representation -/
def expected : ScientificNotation :=
  { coefficient := 4
  , exponent := 5
  , is_valid := by sorry }

theorem scientific_notation_of_400000 :
  toScientificNotation number = expected := by sorry

end scientific_notation_of_400000_l3443_344322


namespace paint_mixture_intensity_l3443_344312

/-- Calculates the intensity of a paint mixture when a fraction of the original paint is replaced with a different paint. -/
def mixedPaintIntensity (originalIntensity addedIntensity : ℝ) (fractionReplaced : ℝ) : ℝ :=
  originalIntensity * (1 - fractionReplaced) + addedIntensity * fractionReplaced

/-- Theorem stating that mixing 45% intensity paint with 25% intensity paint in a 3:1 ratio results in 40% intensity paint. -/
theorem paint_mixture_intensity :
  let originalIntensity : ℝ := 0.45
  let addedIntensity : ℝ := 0.25
  let fractionReplaced : ℝ := 0.25
  mixedPaintIntensity originalIntensity addedIntensity fractionReplaced = 0.40 := by
  sorry

end paint_mixture_intensity_l3443_344312


namespace harry_book_count_l3443_344335

/-- The number of books Harry has -/
def harry_books : ℕ := 50

/-- The number of books Flora has -/
def flora_books : ℕ := 2 * harry_books

/-- The number of books Gary has -/
def gary_books : ℕ := harry_books / 2

/-- The total number of books -/
def total_books : ℕ := 175

theorem harry_book_count : 
  harry_books + flora_books + gary_books = total_books ∧ harry_books = 50 := by
  sorry

end harry_book_count_l3443_344335


namespace trig_identity_proof_l3443_344333

theorem trig_identity_proof : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 / (Real.cos (20 * π / 180))^2 := by
  sorry

end trig_identity_proof_l3443_344333


namespace plan_A_rate_correct_l3443_344331

/-- The per-minute charge after the first 5 minutes under plan A -/
def plan_A_rate : ℝ := 0.06

/-- The fixed charge for the first 5 minutes under plan A -/
def plan_A_fixed_charge : ℝ := 0.60

/-- The per-minute charge under plan B -/
def plan_B_rate : ℝ := 0.08

/-- The duration at which both plans cost the same -/
def equal_cost_duration : ℝ := 14.999999999999996

theorem plan_A_rate_correct :
  plan_A_rate * (equal_cost_duration - 5) + plan_A_fixed_charge =
  plan_B_rate * equal_cost_duration := by sorry

end plan_A_rate_correct_l3443_344331


namespace triangle_third_angle_l3443_344307

theorem triangle_third_angle (A B C : ℝ) (h : A + B = 90) : C = 90 :=
  by
  sorry

end triangle_third_angle_l3443_344307


namespace exists_good_number_in_interval_l3443_344388

/-- A function that checks if a natural number is a "good number" (all digits ≤ 5) -/
def is_good_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≤ 5

/-- The main theorem: For any natural number x, there exists a "good number" y in [x, 9/5x) -/
theorem exists_good_number_in_interval (x : ℕ) : 
  ∃ y : ℕ, x ≤ y ∧ y < (9 * x) / 5 ∧ is_good_number y :=
sorry

end exists_good_number_in_interval_l3443_344388


namespace uncool_parents_count_l3443_344367

theorem uncool_parents_count (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total_students = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total_students - (cool_dads - both_cool + cool_moms - both_cool + both_cool) = 10 :=
by sorry

end uncool_parents_count_l3443_344367


namespace dog_tricks_conversion_l3443_344334

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem dog_tricks_conversion :
  base7ToBase10 [3, 5, 6] = 332 := by
  sorry

end dog_tricks_conversion_l3443_344334


namespace sum_of_x_and_y_on_circle_l3443_344390

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 16*x - 8*y - 60) : x + y = 4 := by
  sorry

end sum_of_x_and_y_on_circle_l3443_344390


namespace flu_transmission_rate_l3443_344336

/-- 
Given two rounds of flu transmission where a total of 121 people are infected,
prove that on average, one person infects 10 others in each round.
-/
theorem flu_transmission_rate : 
  ∃ x : ℕ, 
    (1 + x + x * (1 + x) = 121) ∧ 
    (x = 10) := by
  sorry

end flu_transmission_rate_l3443_344336


namespace rectangular_prism_diagonal_l3443_344397

/-- Given a rectangular prism with sides a, b, and c, if its surface area is 11
    and the sum of its edges is 24, then the length of its diagonal is 5. -/
theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h_surface : 2 * (a * b + a * c + b * c) = 11) 
  (h_edges : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end rectangular_prism_diagonal_l3443_344397


namespace solution_difference_l3443_344346

theorem solution_difference (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ^ 3 = 7 - x₁^2 / 4) ∧ (x₂ ^ 3 = 7 - x₂^2 / 4) ∧ x₁ ≠ x₂) → 
  (∃ x₁ x₂ : ℝ, (x₁ ^ 3 = 7 - x₁^2 / 4) ∧ (x₂ ^ 3 = 7 - x₂^2 / 4) ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 4 * Real.sqrt 6) :=
by
  sorry

end solution_difference_l3443_344346


namespace workers_days_per_week_l3443_344306

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 5505

/-- The number of toys produced per day -/
def toys_per_day : ℕ := 1101

/-- The number of days worked in a week -/
def days_worked : ℕ := toys_per_week / toys_per_day

theorem workers_days_per_week :
  days_worked = 5 :=
by sorry

end workers_days_per_week_l3443_344306


namespace unfilled_holes_l3443_344392

theorem unfilled_holes (total : ℕ) (filled_percentage : ℚ) : 
  total = 8 → filled_percentage = 75 / 100 → total - (filled_percentage * total).floor = 2 := by
sorry

end unfilled_holes_l3443_344392


namespace line_point_distance_l3443_344341

/-- Given a point M(a,b) on the line 4x - 3y + c = 0, 
    if the minimum value of (a-1)² + (b-1)² is 4, 
    then c = -11 or c = 9 -/
theorem line_point_distance (a b c : ℝ) : 
  (4 * a - 3 * b + c = 0) → 
  (∃ (d : ℝ), d = 4 ∧ ∀ (x y : ℝ), 4 * x - 3 * y + c = 0 → (x - 1)^2 + (y - 1)^2 ≥ d) →
  (c = -11 ∨ c = 9) :=
sorry

end line_point_distance_l3443_344341


namespace sqrt_division_equality_l3443_344352

theorem sqrt_division_equality : Real.sqrt 2 / Real.sqrt 3 = Real.sqrt 6 / 3 := by
  sorry

end sqrt_division_equality_l3443_344352


namespace locus_is_hyperbola_l3443_344324

/-- Given the coordinates of point P(x, y) satisfying the following conditions,
    prove that the locus of P is a hyperbola. -/
theorem locus_is_hyperbola
  (a c : ℝ)
  (x y θ₁ θ₂ : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (h4 : c > 1) :
  ∃ (k m n : ℝ), y^2 = k * x^2 + m * x + n ∧ k > 1 :=
sorry

end locus_is_hyperbola_l3443_344324


namespace largest_divisor_of_n_squared_divisible_by_72_l3443_344382

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) (h_div : 72 ∣ n^2) :
  ∃ m : ℕ, m = 12 ∧ m ∣ n ∧ ∀ k : ℕ, k ∣ n → k ≤ m :=
by sorry

end largest_divisor_of_n_squared_divisible_by_72_l3443_344382


namespace arithmetic_sequence_part1_arithmetic_sequence_part2_l3443_344357

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  seq_def : ∀ n : ℕ, a n = a 1 + (n - 1) * d

/-- Part 1 of the problem -/
theorem arithmetic_sequence_part1 (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = -1) (h2 : seq.a 8 = 2) : 
  seq.a 1 = -5 ∧ seq.d = 1 := by
  sorry

/-- Part 2 of the problem -/
theorem arithmetic_sequence_part2 (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 6 = 12) (h2 : seq.a 4 = 7) :
  seq.a 9 = 17 := by
  sorry

end arithmetic_sequence_part1_arithmetic_sequence_part2_l3443_344357


namespace triangle_is_equilateral_l3443_344313

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C, 
    if b^2 + c^2 - bc = a^2 and b/c = tan(B) / tan(C), 
    then the triangle is equilateral. -/
theorem triangle_is_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 + c^2 - b*c = a^2) 
  (h2 : b/c = Real.tan B / Real.tan C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) :
  a = b ∧ b = c := by
  sorry


end triangle_is_equilateral_l3443_344313


namespace pig_farm_area_l3443_344330

/-- Represents a rectangular pig farm with specific properties -/
structure PigFarm where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  area : ℝ

/-- Creates a PigFarm given the length of the shorter side -/
def make_pig_farm (x : ℝ) : PigFarm :=
  { short_side := x
  , long_side := 2 * x
  , fence_length := 4 * x
  , area := 2 * x * x
  }

/-- Theorem stating the area of the pig farm with given conditions -/
theorem pig_farm_area :
  ∃ (farm : PigFarm), farm.fence_length = 150 ∧ farm.area = 2812.5 := by
  sorry


end pig_farm_area_l3443_344330


namespace quadratic_inequality_solution_l3443_344325

/-- Given a quadratic inequality ax^2 + (a-1)x - 1 > 0 with solution set (-1, -1/2), prove that a = -2 -/
theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + (a-1)*x - 1 > 0 ↔ -1 < x ∧ x < -1/2) → 
  a = -2 := by
  sorry

end quadratic_inequality_solution_l3443_344325


namespace university_visit_probability_l3443_344366

theorem university_visit_probability : 
  let n : ℕ := 4  -- number of students
  let k : ℕ := 2  -- number of universities
  let p : ℚ := (k^n - 2) / k^n  -- probability formula
  p = 7/8 := by sorry

end university_visit_probability_l3443_344366


namespace max_value_properties_l3443_344383

noncomputable def f (s : ℝ) (x : ℝ) : ℝ := (Real.log s) / (1 + x) - Real.log s

theorem max_value_properties (s : ℝ) (x₀ : ℝ) 
  (h_max : ∀ x, f s x ≤ f s x₀) :
  f s x₀ = x₀ ∧ f s x₀ < (1/2 : ℝ) := by
  sorry

end max_value_properties_l3443_344383


namespace monday_loaves_l3443_344381

/-- Represents the number of loaves baked on a given day -/
def loaves : Fin 6 → ℕ
  | 0 => 5  -- Wednesday
  | 1 => 7  -- Thursday
  | 2 => 10 -- Friday
  | 3 => 14 -- Saturday
  | 4 => 19 -- Sunday
  | 5 => 25 -- Monday (to be proven)

/-- The pattern of increase in loaves from one day to the next -/
def increase (n : Fin 5) : ℕ := loaves (n + 1) - loaves n

/-- The theorem stating that the number of loaves baked on Monday is 25 -/
theorem monday_loaves :
  (∀ n : Fin 4, increase (n + 1) = increase n + 1) →
  loaves 5 = 25 := by
  sorry


end monday_loaves_l3443_344381


namespace billy_homework_questions_l3443_344362

/-- Represents the number of questions solved in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Theorem: Given the conditions, Billy solved 132 questions in the third hour -/
theorem billy_homework_questions (q : HourlyQuestions) : 
  q.third = 2 * q.second ∧ 
  q.third = 3 * q.first ∧ 
  q.first + q.second + q.third = 242 → 
  q.third = 132 := by
  sorry

end billy_homework_questions_l3443_344362


namespace sevenPeopleArrangementCount_l3443_344314

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := (arrangements n) / ((arrangements k) * (arrangements (n - k)))

/-- The number of ways to arrange seven people in a row with two specific people not next to each other. -/
def sevenPeopleArrangement : ℕ :=
  (arrangements 5) * (choose 6 2)

theorem sevenPeopleArrangementCount :
  sevenPeopleArrangement = 3600 := by
  sorry

end sevenPeopleArrangementCount_l3443_344314


namespace floor_ceil_sum_l3443_344378

theorem floor_ceil_sum : ⌊(-3.01 : ℝ)⌋ + ⌈(24.99 : ℝ)⌉ = 21 := by sorry

end floor_ceil_sum_l3443_344378


namespace earth_inhabitable_fraction_l3443_344332

theorem earth_inhabitable_fraction :
  let land_fraction : ℚ := 2/3
  let inhabitable_land_fraction : ℚ := 3/4
  (land_fraction * inhabitable_land_fraction : ℚ) = 1/2 := by sorry

end earth_inhabitable_fraction_l3443_344332


namespace smallest_sum_of_four_odds_divisible_by_five_l3443_344329

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def consecutive_odds (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

theorem smallest_sum_of_four_odds_divisible_by_five :
  ∃ a b c d : ℕ,
    consecutive_odds a b c d ∧
    not_divisible_by_three a ∧
    not_divisible_by_three b ∧
    not_divisible_by_three c ∧
    not_divisible_by_three d ∧
    (a + b + c + d) % 5 = 0 ∧
    a + b + c + d = 40 ∧
    (∀ w x y z : ℕ,
      consecutive_odds w x y z →
      not_divisible_by_three w →
      not_divisible_by_three x →
      not_divisible_by_three y →
      not_divisible_by_three z →
      (w + x + y + z) % 5 = 0 →
      w + x + y + z ≥ 40) :=
by
  sorry

end smallest_sum_of_four_odds_divisible_by_five_l3443_344329


namespace last_three_digits_of_2005_power_l3443_344315

theorem last_three_digits_of_2005_power (A : ℕ) :
  A = 2005^2005 →
  A % 1000 = 125 := by
sorry

end last_three_digits_of_2005_power_l3443_344315
