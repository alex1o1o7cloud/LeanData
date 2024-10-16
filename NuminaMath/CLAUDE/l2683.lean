import Mathlib

namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l2683_268310

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 7 + 9 →
  7 + x > 9 →
  9 + x > 7 →
  ∀ y : ℕ,
  y > 0 →
  y < 7 + 9 →
  7 + y > 9 →
  9 + y > 7 →
  7 + 9 + x ≥ 7 + 9 + y →
  7 + 9 + x = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l2683_268310


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l2683_268387

/-- The upper limit of Arun's weight according to his own opinion -/
def arun_upper_limit : ℝ := 69

/-- Arun's lower weight limit -/
def arun_lower_limit : ℝ := 66

/-- The average of Arun's probable weights -/
def arun_average_weight : ℝ := 68

/-- Brother's upper limit for Arun's weight -/
def brother_upper_limit : ℝ := 70

/-- Mother's upper limit for Arun's weight -/
def mother_upper_limit : ℝ := 69

theorem arun_weight_upper_limit :
  arun_upper_limit = 69 ∧
  arun_lower_limit < arun_upper_limit ∧
  arun_lower_limit < brother_upper_limit ∧
  arun_upper_limit ≤ mother_upper_limit ∧
  arun_upper_limit ≤ brother_upper_limit ∧
  (arun_lower_limit + arun_upper_limit) / 2 = arun_average_weight :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l2683_268387


namespace NUMINAMATH_CALUDE_maximum_value_implies_ratio_l2683_268341

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem maximum_value_implies_ratio (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧  -- f(x) reaches maximum at x = 1
  (f a b 1 = 10) ∧            -- The maximum value is 10
  (f_deriv a b 1 = 0)         -- Derivative is zero at x = 1
  → a / b = -2 / 3 := by sorry

end NUMINAMATH_CALUDE_maximum_value_implies_ratio_l2683_268341


namespace NUMINAMATH_CALUDE_tangent_beta_l2683_268398

theorem tangent_beta (a b : ℝ) (α β γ : Real) 
  (h1 : (a + b) / (a - b) = Real.tan ((α + β) / 2) / Real.tan ((α - β) / 2))
  (h2 : (α + β) / 2 = π / 2 - γ / 2)
  (h3 : (α - β) / 2 = π / 2 - (β + γ / 2)) :
  Real.tan β = (2 * b * Real.tan (γ / 2)) / ((a + b) * Real.tan (γ / 2)^2 + (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_beta_l2683_268398


namespace NUMINAMATH_CALUDE_generatrix_angle_is_60_degrees_l2683_268381

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemiCircleCone where
  /-- The radius of the semicircle (equal to the generatrix of the cone) -/
  radius : ℝ
  /-- Assumption that the lateral surface unfolds into a semicircle -/
  lateral_surface_is_semicircle : True

/-- The angle between the two generatrices in the axial section of a cone
    whose lateral surface unfolds into a semicircle is 60 degrees -/
theorem generatrix_angle_is_60_degrees (cone : SemiCircleCone) :
  let angle_rad := Real.pi / 3
  angle_rad = Real.arccos (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_generatrix_angle_is_60_degrees_l2683_268381


namespace NUMINAMATH_CALUDE_hens_and_cows_problem_l2683_268352

theorem hens_and_cows_problem (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_animals = 46 →
  total_feet = 136 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_hens_and_cows_problem_l2683_268352


namespace NUMINAMATH_CALUDE_gcd_24_36_l2683_268359

theorem gcd_24_36 : Nat.gcd 24 36 = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_24_36_l2683_268359


namespace NUMINAMATH_CALUDE_circle_C_equation_l2683_268336

/-- The standard equation of a circle with center (h, k) and radius r -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Circle C with center (1, 2) and radius 3 -/
def circle_C (x y : ℝ) : Prop :=
  standard_circle_equation x y 1 2 3

theorem circle_C_equation :
  ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_C_equation_l2683_268336


namespace NUMINAMATH_CALUDE_quadratic_zeros_imply_range_bound_l2683_268391

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_zeros_imply_range_bound (b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    quadratic_function b c x₁ = 0 ∧ quadratic_function b c x₂ = 0) →
  0 < (1 + b) * c + c^2 ∧ (1 + b) * c + c^2 < 1/16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zeros_imply_range_bound_l2683_268391


namespace NUMINAMATH_CALUDE_parabola_point_value_l2683_268329

theorem parabola_point_value (k : ℝ) : 
  let line1 : ℝ → ℝ := λ x => -x + 3
  let line2 : ℝ → ℝ := λ x => (x - 6) / 2
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let x_intercept1 : ℝ := 3
  let x_intercept2 : ℝ := 6
  let parabola : ℝ → ℝ := λ x => (1/2) * (x - x_intercept1) * (x - x_intercept2)
  (parabola intersection_x = intersection_y) ∧
  (parabola x_intercept1 = 0) ∧
  (parabola x_intercept2 = 0) ∧
  (parabola 10 = k)
  → k = 14 := by
sorry


end NUMINAMATH_CALUDE_parabola_point_value_l2683_268329


namespace NUMINAMATH_CALUDE_perfect_square_product_l2683_268384

theorem perfect_square_product (a b c d : ℤ) (h : a + b + c + d = 0) :
  ∃ k : ℤ, (a * b - c * d) * (b * c - a * d) * (c * a - b * d) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_product_l2683_268384


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_zero_l2683_268356

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {-1, 1, 2, 4}
def B : Finset Int := {-1, 0, 2}

theorem intersection_complement_equals_singleton_zero :
  B ∩ (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_zero_l2683_268356


namespace NUMINAMATH_CALUDE_almost_every_graph_chromatic_number_l2683_268322

-- Define the random graph model
structure RandomGraph (n : ℕ) (p : ℝ) where
  -- Add necessary fields here

-- Define the chromatic number
def chromaticNumber (G : RandomGraph n p) : ℝ := sorry

-- Main theorem
theorem almost_every_graph_chromatic_number 
  (p : ℝ) (ε : ℝ) (n : ℕ) (h_p : 0 < p ∧ p < 1) (h_ε : ε > 0) :
  ∃ (G : RandomGraph n p), 
    chromaticNumber G > (Real.log (1 / (1 - p))) / (2 + ε) * (n / Real.log n) := by
  sorry

end NUMINAMATH_CALUDE_almost_every_graph_chromatic_number_l2683_268322


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2683_268353

theorem consecutive_odd_numbers_sum (n k : ℕ) : n > 0 ∧ k > 0 → 
  (∃ (seq : List ℕ), 
    (∀ i ∈ seq, ∃ j, i = n + 2 * j ∧ j ≤ k) ∧ 
    (seq.length = k + 1) ∧
    (seq.sum = 20 * (n + 2 * k)) ∧
    (seq.sum = 60 * n)) →
  n = 29 ∧ k = 29 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l2683_268353


namespace NUMINAMATH_CALUDE_concentric_circles_properties_l2683_268307

/-- Given two concentric circles where a chord is tangent to the smaller circle -/
structure ConcentricCircles where
  /-- Radius of the smaller circle -/
  r₁ : ℝ
  /-- Length of the chord tangent to the smaller circle -/
  chord_length : ℝ
  /-- The chord is tangent to the smaller circle -/
  tangent_chord : True

/-- Theorem about the radius of the larger circle and the area between the circles -/
theorem concentric_circles_properties (c : ConcentricCircles) 
  (h₁ : c.r₁ = 30)
  (h₂ : c.chord_length = 120) :
  ∃ (r₂ : ℝ) (area : ℝ),
    r₂ = 30 * Real.sqrt 5 ∧ 
    area = 3600 * Real.pi ∧
    r₂ > c.r₁ ∧
    area = Real.pi * (r₂^2 - c.r₁^2) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_properties_l2683_268307


namespace NUMINAMATH_CALUDE_orange_juice_profit_l2683_268349

-- Define the tree types and their properties
structure TreeType where
  name : String
  trees : ℕ
  orangesPerTree : ℕ
  pricePerCup : ℚ

-- Define the additional costs
def additionalCosts : ℚ := 180

-- Define the number of oranges needed to make one cup of juice
def orangesPerCup : ℕ := 3

-- Define the tree types
def valencia : TreeType := ⟨"Valencia", 150, 400, 4⟩
def navel : TreeType := ⟨"Navel", 120, 650, 9/2⟩
def bloodOrange : TreeType := ⟨"Blood Orange", 160, 500, 5⟩

-- Calculate profit for a single tree type
def calculateProfit (t : TreeType) : ℚ :=
  let totalOranges := t.trees * t.orangesPerTree
  let totalCups := totalOranges / orangesPerCup
  totalCups * t.pricePerCup - additionalCosts

-- Calculate total profit
def totalProfit : ℚ :=
  calculateProfit valencia + calculateProfit navel + calculateProfit bloodOrange

-- Theorem statement
theorem orange_juice_profit : totalProfit = 329795 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_profit_l2683_268349


namespace NUMINAMATH_CALUDE_three_in_range_of_quadratic_l2683_268323

theorem three_in_range_of_quadratic (b : ℝ) : ∃ x : ℝ, x^2 + b*x - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_in_range_of_quadratic_l2683_268323


namespace NUMINAMATH_CALUDE_airplane_seating_l2683_268305

/-- A proof problem about airplane seating --/
theorem airplane_seating (first_class business_class economy_class : ℕ) 
  (h1 : first_class = 10)
  (h2 : business_class = 30)
  (h3 : economy_class = 50)
  (h4 : economy_class / 2 = first_class + (business_class - (business_class - x)))
  (h5 : first_class - 7 = 3)
  (x : ℕ) :
  x = 8 := by sorry

end NUMINAMATH_CALUDE_airplane_seating_l2683_268305


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2683_268319

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2683_268319


namespace NUMINAMATH_CALUDE_function_value_comparison_l2683_268315

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem function_value_comparison
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f a b c (x + 1) = f a b c (1 - x)) :
  f a b c (Real.arcsin (1/3)) > f a b c (Real.arcsin (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_function_value_comparison_l2683_268315


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2683_268338

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2683_268338


namespace NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l2683_268388

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l2683_268388


namespace NUMINAMATH_CALUDE_supply_duration_l2683_268368

/-- Represents the number of pills in one supply -/
def supply : ℕ := 90

/-- Represents the fraction of a pill consumed in one dose -/
def dose : ℚ := 3/4

/-- Represents the number of days between doses -/
def interval : ℕ := 3

/-- Represents the number of days in a month (assumed average) -/
def days_per_month : ℕ := 30

/-- Theorem stating that the given supply lasts 12 months -/
theorem supply_duration :
  (supply : ℚ) * interval / dose / days_per_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_l2683_268368


namespace NUMINAMATH_CALUDE_sqrt_625_div_5_l2683_268300

theorem sqrt_625_div_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_625_div_5_l2683_268300


namespace NUMINAMATH_CALUDE_bad_carrots_l2683_268301

theorem bad_carrots (vanessa_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  vanessa_carrots = 17 → mom_carrots = 14 → good_carrots = 24 → 
  vanessa_carrots + mom_carrots - good_carrots = 7 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_l2683_268301


namespace NUMINAMATH_CALUDE_polar_bear_trout_consumption_l2683_268365

/-- The amount of fish eaten daily by the polar bear -/
def total_fish : ℝ := 0.6

/-- The amount of salmon eaten daily by the polar bear -/
def salmon : ℝ := 0.4

/-- The amount of trout eaten daily by the polar bear -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption : trout = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_trout_consumption_l2683_268365


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l2683_268366

/-- Calculates the total weight of carrots harvested given the number of carrots in each bed and the weight ratio --/
def total_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℕ :=
  ((bed1 + bed2 + bed3) / carrots_per_pound : ℕ)

/-- Theorem stating that Kelly harvested 39 pounds of carrots --/
theorem kelly_carrot_harvest :
  total_carrot_weight 55 101 78 6 = 39 := by
  sorry

#eval total_carrot_weight 55 101 78 6

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l2683_268366


namespace NUMINAMATH_CALUDE_polynomial_identity_l2683_268383

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2683_268383


namespace NUMINAMATH_CALUDE_first_number_a10_l2683_268334

def first_number (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1) / 2)

theorem first_number_a10 : first_number 10 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_number_a10_l2683_268334


namespace NUMINAMATH_CALUDE_max_surrounding_squares_l2683_268331

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents an arrangement of squares around a central square -/
structure SquareArrangement where
  centralSquare : Square
  surroundingSquare : Square
  numSurroundingSquares : ℕ

/-- The condition that the surrounding squares fit perfectly around the central square -/
def perfectFit (arrangement : SquareArrangement) : Prop :=
  arrangement.centralSquare.sideLength = arrangement.surroundingSquare.sideLength * (arrangement.numSurroundingSquares / 4 : ℝ)

/-- The theorem stating the maximum number of surrounding squares -/
theorem max_surrounding_squares (centralSquare : Square) (surroundingSquare : Square) 
    (h_central : centralSquare.sideLength = 4)
    (h_surrounding : surroundingSquare.sideLength = 1) :
    ∃ (arrangement : SquareArrangement), 
      arrangement.centralSquare = centralSquare ∧ 
      arrangement.surroundingSquare = surroundingSquare ∧
      arrangement.numSurroundingSquares = 16 ∧
      perfectFit arrangement ∧
      ∀ (otherArrangement : SquareArrangement), 
        otherArrangement.centralSquare = centralSquare → 
        otherArrangement.surroundingSquare = surroundingSquare → 
        perfectFit otherArrangement → 
        otherArrangement.numSurroundingSquares ≤ 16 :=
  sorry

end NUMINAMATH_CALUDE_max_surrounding_squares_l2683_268331


namespace NUMINAMATH_CALUDE_right_triangle_area_l2683_268374

/-- The area of a right triangle with vertices at (0, 0), (0, 7), and (-7, 0) is 24.5 square units. -/
theorem right_triangle_area : 
  let vertex1 : ℝ × ℝ := (0, 0)
  let vertex2 : ℝ × ℝ := (0, 7)
  let vertex3 : ℝ × ℝ := (-7, 0)
  let base : ℝ := 7
  let height : ℝ := 7
  let area : ℝ := (1 / 2) * base * height
  area = 24.5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2683_268374


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_parallel_l2683_268364

/-- A type representing lines in three-dimensional space -/
structure Line3D where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Parallel relation between two lines in 3D space -/
def parallel (l m : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines in 3D space -/
def perpendicular (l m : Line3D) : Prop :=
  sorry

/-- Theorem: If two lines are parallel to the same line, they are parallel to each other -/
theorem parallel_transitive (l m n : Line3D) :
  parallel l m → parallel m n → parallel l n :=
sorry

/-- Theorem: If a line is perpendicular to one of two parallel lines, it is perpendicular to the other -/
theorem perpendicular_to_parallel (l m n : Line3D) :
  perpendicular l m → parallel m n → perpendicular l n :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_parallel_l2683_268364


namespace NUMINAMATH_CALUDE_total_oranges_proof_l2683_268327

def initial_purchase : ℕ := 10
def additional_purchase : ℕ := 5
def weeks : ℕ := 3

def total_oranges : ℕ :=
  let week1_purchase := initial_purchase + additional_purchase
  let subsequent_weeks_purchase := 2 * week1_purchase
  week1_purchase + (weeks - 1) * subsequent_weeks_purchase

theorem total_oranges_proof :
  total_oranges = 75 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_proof_l2683_268327


namespace NUMINAMATH_CALUDE_twenty_sixth_digit_of_N_l2683_268367

def N (d : ℕ) : ℕ := 
  (10^49 - 1) / 9 + d * 10^24 - 10^25 + 1

theorem twenty_sixth_digit_of_N (d : ℕ) : 
  d < 10 → N d % 13 = 0 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_twenty_sixth_digit_of_N_l2683_268367


namespace NUMINAMATH_CALUDE_min_hours_to_reach_55_people_l2683_268328

/-- The number of people who have received the message after n hours -/
def people_reached (n : ℕ) : ℕ := 2^(n + 1) - 2

/-- The proposition that 6 hours is the minimum time needed to reach at least 55 people -/
theorem min_hours_to_reach_55_people : 
  (∀ k < 6, people_reached k ≤ 55) ∧ people_reached 6 > 55 :=
sorry

end NUMINAMATH_CALUDE_min_hours_to_reach_55_people_l2683_268328


namespace NUMINAMATH_CALUDE_smallest_a_value_l2683_268335

/-- Given a polynomial x^3 - ax^2 + bx - 3003 with three positive integer roots,
    the smallest possible value of a is 45 -/
theorem smallest_a_value (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 3003 = (x - r₁)*(x - r₂)*(x - r₃)) →
  a ≥ 45 ∧ ∃ a₀ b₀ r₁₀ r₂₀ r₃₀, 
    a₀ = 45 ∧ 
    (∀ x, x^3 - a₀*x^2 + b₀*x - 3003 = (x - r₁₀)*(x - r₂₀)*(x - r₃₀)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_a_value_l2683_268335


namespace NUMINAMATH_CALUDE_problem_solution_l2683_268378

theorem problem_solution (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2683_268378


namespace NUMINAMATH_CALUDE_log_cube_difference_l2683_268373

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_cube_difference 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (h_diff : f a x₁ - f a x₂ = 2) : 
  f a (x₁^3) - f a (x₂^3) = 6 := by
sorry

end NUMINAMATH_CALUDE_log_cube_difference_l2683_268373


namespace NUMINAMATH_CALUDE_card_distribution_convergence_l2683_268325

/-- Represents a person in the circular arrangement -/
structure Person where
  id : Nat
  cards : Nat

/-- Represents the state of the card distribution -/
structure CardState where
  people : List Person
  total_cards : Nat

/-- Defines a valid move in the card game -/
def valid_move (state : CardState) (giver : Nat) : Prop :=
  ∃ (p : Person), p ∈ state.people ∧ p.id = giver ∧ p.cards ≥ 2

/-- Defines the result of a move -/
def move_result (state : CardState) (giver : Nat) : CardState :=
  sorry

/-- Defines a sequence of moves -/
def move_sequence (initial : CardState) : List Nat → CardState
  | [] => initial
  | (m :: ms) => move_result (move_sequence initial ms) m

/-- The main theorem to be proved -/
theorem card_distribution_convergence 
  (n : Nat) 
  (h : n > 1) :
  ∃ (initial : CardState) (moves : List Nat),
    (initial.people.length = n) ∧ 
    (initial.total_cards = n - 1) ∧
    (∀ (p : Person), p ∈ (move_sequence initial moves).people → p.cards ≤ 1) :=
  sorry

end NUMINAMATH_CALUDE_card_distribution_convergence_l2683_268325


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2683_268389

theorem binomial_coefficient_congruence (n : ℕ+) :
  ∃ σ : Fin (2^(n.val-1)) ≃ Fin (2^(n.val-1)),
    ∀ k : Fin (2^(n.val-1)),
      (Nat.choose (2^n.val - 1) k) ≡ (2 * σ k + 1) [MOD 2^n.val] := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2683_268389


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2683_268302

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The x-coordinate of one focus -/
  focus_x : ℝ
  /-- The y-coordinate of one focus -/
  focus_y : ℝ
  /-- The slope of one asymptote -/
  asymptote_slope : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with one focus at (5,0) and one asymptote with slope 3/4 is 5/4 -/
theorem hyperbola_eccentricity :
  let h : Hyperbola := { focus_x := 5, focus_y := 0, asymptote_slope := 3/4 }
  eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2683_268302


namespace NUMINAMATH_CALUDE_a_3_eq_35_l2683_268347

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℕ := 5 * n ^ 2 + 10 * n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n - 1)

/-- Theorem: The third term of the sequence is 35 -/
theorem a_3_eq_35 : a 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_a_3_eq_35_l2683_268347


namespace NUMINAMATH_CALUDE_banana_theorem_l2683_268394

/-- The number of pounds of bananas purchased by a grocer -/
def banana_problem (buy_rate : ℚ) (sell_rate : ℚ) (total_profit : ℚ) : ℚ :=
  total_profit / (sell_rate - buy_rate)

theorem banana_theorem :
  banana_problem (1/6) (1/4) 11 = 132 := by
  sorry

#eval banana_problem (1/6) (1/4) 11

end NUMINAMATH_CALUDE_banana_theorem_l2683_268394


namespace NUMINAMATH_CALUDE_weighted_average_markup_percentage_l2683_268308

-- Define the fruit types
inductive Fruit
| Apple
| Orange
| Banana

-- Define the properties for each fruit
def cost (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 30
  | Fruit.Orange => 40
  | Fruit.Banana => 50

def markup_percentage (f : Fruit) : ℝ :=
  match f with
  | Fruit.Apple => 0.10
  | Fruit.Orange => 0.15
  | Fruit.Banana => 0.20

def quantity (f : Fruit) : ℕ :=
  match f with
  | Fruit.Apple => 25
  | Fruit.Orange => 20
  | Fruit.Banana => 15

-- Calculate the markup amount for a fruit
def markup_amount (f : Fruit) : ℝ :=
  cost f * markup_percentage f

-- Calculate the selling price for a fruit
def selling_price (f : Fruit) : ℝ :=
  cost f + markup_amount f

-- Calculate the total selling price for all fruits
def total_selling_price : ℝ :=
  selling_price Fruit.Apple + selling_price Fruit.Orange + selling_price Fruit.Banana

-- Calculate the total cost for all fruits
def total_cost : ℝ :=
  cost Fruit.Apple + cost Fruit.Orange + cost Fruit.Banana

-- Calculate the total markup for all fruits
def total_markup : ℝ :=
  markup_amount Fruit.Apple + markup_amount Fruit.Orange + markup_amount Fruit.Banana

-- Theorem: The weighted average markup percentage is 15.83%
theorem weighted_average_markup_percentage :
  (total_markup / total_cost) * 100 = 15.83 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_markup_percentage_l2683_268308


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2683_268311

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x^2 - x - 2 > 0}

theorem union_of_A_and_B : A ∪ B = {x | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2683_268311


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2683_268321

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 →
    (6 * x + 15) / (x^2 - 8*x - 48) = (87/16) / (x - 12) + (9/16) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2683_268321


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2683_268361

theorem unique_solution_quadratic (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) ↔ (j = 0 ∨ j = -36) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2683_268361


namespace NUMINAMATH_CALUDE_no_intersection_l2683_268339

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l2683_268339


namespace NUMINAMATH_CALUDE_symmetry_across_origin_l2683_268304

/-- Given two points A and B in a 2D plane, where B is symmetrical to A with respect to the origin,
    this theorem proves that if A has coordinates (2, -6), then B has coordinates (-2, 6). -/
theorem symmetry_across_origin (A B : ℝ × ℝ) :
  A = (2, -6) → B = (-A.1, -A.2) → B = (-2, 6) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_across_origin_l2683_268304


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2683_268317

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (p : ℝ), (p = 13 ∨ p = 14) ∧ (p = a + 2*b ∨ p = b + 2*a) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2683_268317


namespace NUMINAMATH_CALUDE_total_non_hot_peppers_l2683_268357

-- Define the types of peppers
inductive PepperType
| Hot
| Sweet
| Mild

-- Define a structure for daily pepper counts
structure DailyPeppers where
  hot : Nat
  sweet : Nat
  mild : Nat

-- Define the week's pepper counts
def weekPeppers : List DailyPeppers := [
  ⟨7, 10, 13⟩,  -- Sunday
  ⟨12, 8, 10⟩,  -- Monday
  ⟨14, 19, 7⟩,  -- Tuesday
  ⟨12, 5, 23⟩,  -- Wednesday
  ⟨5, 20, 5⟩,   -- Thursday
  ⟨18, 15, 12⟩, -- Friday
  ⟨12, 8, 30⟩   -- Saturday
]

-- Function to calculate non-hot peppers for a day
def nonHotPeppers (day : DailyPeppers) : Nat :=
  day.sweet + day.mild

-- Theorem: The sum of non-hot peppers throughout the week is 185
theorem total_non_hot_peppers :
  (weekPeppers.map nonHotPeppers).sum = 185 := by
  sorry


end NUMINAMATH_CALUDE_total_non_hot_peppers_l2683_268357


namespace NUMINAMATH_CALUDE_hector_gumballs_l2683_268320

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := 4 * alisha_gumballs - 5

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The total number of gumballs Hector purchased -/
def total_gumballs : ℕ := todd_gumballs + alisha_gumballs + bobby_gumballs + remaining_gumballs

theorem hector_gumballs : total_gumballs = 45 := by
  sorry

end NUMINAMATH_CALUDE_hector_gumballs_l2683_268320


namespace NUMINAMATH_CALUDE_function_identity_l2683_268355

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_identity_l2683_268355


namespace NUMINAMATH_CALUDE_cabbage_sales_proof_l2683_268386

def price_per_kg : ℝ := 2
def earnings_wednesday : ℝ := 30
def earnings_friday : ℝ := 24
def earnings_today : ℝ := 42

theorem cabbage_sales_proof :
  (earnings_wednesday + earnings_friday + earnings_today) / price_per_kg = 48 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_sales_proof_l2683_268386


namespace NUMINAMATH_CALUDE_intersection_distance_l2683_268375

theorem intersection_distance : 
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1^2 + p1.2^2 = 13) ∧ 
    (p1.1 + p1.2 = 4) ∧
    (p2.1^2 + p2.2^2 = 13) ∧ 
    (p2.1 + p2.2 = 4) ∧
    (p1 ≠ p2) ∧
    ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2683_268375


namespace NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2683_268330

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_quadratic (x : ℤ) : ℕ := Int.natAbs (8 * x^2 - 53 * x + 21)

theorem greatest_integer_prime_quadratic :
  ∀ x : ℤ, x > 1 → ¬(is_prime (abs_quadratic x)) ∧
  (is_prime (abs_quadratic 1)) ∧
  (∀ y : ℤ, y ≤ 1 → is_prime (abs_quadratic y) → y = 1) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_prime_quadratic_l2683_268330


namespace NUMINAMATH_CALUDE_xiaoming_savings_l2683_268362

/-- Represents the number of coins in each pile -/
structure CoinCount where
  pile1_2cent : ℕ
  pile1_5cent : ℕ
  pile2_2cent : ℕ
  pile2_5cent : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  2 * (coins.pile1_2cent + coins.pile2_2cent) + 5 * (coins.pile1_5cent + coins.pile2_5cent)

theorem xiaoming_savings (coins : CoinCount) :
  coins.pile1_2cent = coins.pile1_5cent →
  2 * coins.pile2_2cent = 5 * coins.pile2_5cent →
  2 * coins.pile1_2cent + 5 * coins.pile1_5cent = 2 * coins.pile2_2cent + 5 * coins.pile2_5cent →
  500 ≤ totalValue coins →
  totalValue coins ≤ 600 →
  totalValue coins = 560 := by
  sorry

#check xiaoming_savings

end NUMINAMATH_CALUDE_xiaoming_savings_l2683_268362


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2683_268372

-- Problem 1
theorem problem_1 : Real.sqrt 8 - 2 * Real.sin (π / 4) + |1 - Real.sqrt 2| + (1 / 2)⁻¹ = 2 * Real.sqrt 2 + 1 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ ∀ x : ℝ, x^2 + 4*x - 5 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2683_268372


namespace NUMINAMATH_CALUDE_words_with_A_count_l2683_268396

def letter_set : Finset Char := {'A', 'B', 'C', 'D', 'E'}

/-- The number of 4-letter words using letters A, B, C, D, E with repetition allowed -/
def total_words : ℕ := (Finset.card letter_set) ^ 4

/-- The number of 4-letter words using only B, C, D, E with repetition allowed -/
def words_without_A : ℕ := ((Finset.card letter_set) - 1) ^ 4

/-- The number of 4-letter words using A, B, C, D, E with repetition, containing at least one A -/
def words_with_A : ℕ := total_words - words_without_A

theorem words_with_A_count : words_with_A = 369 := by sorry

end NUMINAMATH_CALUDE_words_with_A_count_l2683_268396


namespace NUMINAMATH_CALUDE_expression_evaluation_l2683_268316

theorem expression_evaluation :
  let x : ℚ := -1/2
  let expr := 2*x^2 + 6*x - 6 - (-2*x^2 + 4*x + 1)
  expr = -7 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2683_268316


namespace NUMINAMATH_CALUDE_cost_price_determination_l2683_268351

theorem cost_price_determination (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) :
  loss_percentage = 0.1 →
  gain_percentage = 0.1 →
  price_increase = 50 →
  ∃ (cost_price : Real),
    cost_price * (1 - loss_percentage) + price_increase = cost_price * (1 + gain_percentage) ∧
    cost_price = 250 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_determination_l2683_268351


namespace NUMINAMATH_CALUDE_parabola_passes_through_origin_l2683_268380

/-- A parabola defined by y = 3x^2 passes through the point (0, 0) -/
theorem parabola_passes_through_origin :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2
  f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_through_origin_l2683_268380


namespace NUMINAMATH_CALUDE_stream_speed_l2683_268314

/-- 
Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr, 
this theorem proves that the speed of the stream is 2 km/hr.
-/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  let canoe_speed := (upstream_speed + downstream_speed) / 2
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 2 := by sorry

end NUMINAMATH_CALUDE_stream_speed_l2683_268314


namespace NUMINAMATH_CALUDE_second_year_interest_rate_problem_solution_l2683_268392

/-- Given an initial investment, interest rates, and final value, calculate the second year's interest rate -/
theorem second_year_interest_rate 
  (initial_investment : ℝ) 
  (first_year_rate : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_rate := (final_value / first_year_value) - 1
  second_year_rate * 100

/-- Prove that the second year's interest rate is 4% given the problem conditions -/
theorem problem_solution :
  second_year_interest_rate 15000 0.05 16380 = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_year_interest_rate_problem_solution_l2683_268392


namespace NUMINAMATH_CALUDE_bill_calculation_correct_l2683_268337

/-- Calculates the final bill amount after late charges and fees --/
def finalBillAmount (originalBill : ℝ) (firstLateChargeRate : ℝ) (secondLateChargeRate : ℝ) (flatFee : ℝ) : ℝ :=
  ((originalBill * (1 + firstLateChargeRate)) * (1 + secondLateChargeRate)) + flatFee

/-- Proves that the final bill amount is correct given the specified conditions --/
theorem bill_calculation_correct :
  finalBillAmount 500 0.01 0.02 5 = 520.1 := by
  sorry

#eval finalBillAmount 500 0.01 0.02 5

end NUMINAMATH_CALUDE_bill_calculation_correct_l2683_268337


namespace NUMINAMATH_CALUDE_trader_profit_above_goal_l2683_268348

theorem trader_profit_above_goal 
  (total_profit : ℕ) 
  (goal_amount : ℕ) 
  (donation_amount : ℕ) 
  (h1 : total_profit = 960)
  (h2 : goal_amount = 610)
  (h3 : donation_amount = 310) :
  (total_profit / 2 + donation_amount) - goal_amount = 180 :=
by sorry

end NUMINAMATH_CALUDE_trader_profit_above_goal_l2683_268348


namespace NUMINAMATH_CALUDE_bobby_pancakes_l2683_268360

theorem bobby_pancakes (total : ℕ) (dog_ate : ℕ) (left : ℕ) (bobby_ate : ℕ) : 
  total = 21 → dog_ate = 7 → left = 9 → bobby_ate = total - dog_ate - left → bobby_ate = 5 := by
  sorry

end NUMINAMATH_CALUDE_bobby_pancakes_l2683_268360


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l2683_268358

/-- Given odds of 3:4 for pulling a prize, the probability of not pulling the prize is 4/7 -/
theorem probability_not_pulling_prize (odds_for : ℚ) (odds_against : ℚ) 
  (h_odds : odds_for = 3 ∧ odds_against = 4) :
  (odds_against / (odds_for + odds_against)) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_probability_not_pulling_prize_l2683_268358


namespace NUMINAMATH_CALUDE_square_area_error_l2683_268385

theorem square_area_error (s : ℝ) (s' : ℝ) (h : s' = 1.04 * s) :
  (s' ^ 2 - s ^ 2) / s ^ 2 = 0.0816 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l2683_268385


namespace NUMINAMATH_CALUDE_exists_tangent_region_l2683_268369

noncomputable section

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := x - x^2
def parabola2 (a : ℝ) (x : ℝ) : ℝ := a * (x - x^2)

-- Define the tangent lines
def tangent1 (b : ℝ) (x : ℝ) : ℝ := (1 - 2*b)*x + b^2
def tangent2 (a : ℝ) (c : ℝ) (x : ℝ) : ℝ := a*(1 - 2*c)*x + a*c^2

-- Define the intersection point of tangent lines
def intersection (a b c : ℝ) : ℝ := (a*c^2 - b^2) / ((1 - 2*b) - a*(1 - 2*c))

-- Define the condition for the third point
def third_point (x b : ℝ) : ℝ := 2*x - b

-- Theorem statement
theorem exists_tangent_region (a : ℝ) (h : a ≥ 2) :
  ∃ (b c : ℝ), 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
  let x := intersection a b c
  let d := third_point x b
  0 < d ∧ d < 1 ∧
  (tangent1 b x = tangent2 a c x ∨ tangent1 d x = tangent1 b x ∨ tangent2 a d x = tangent2 a c x) :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_region_l2683_268369


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2683_268343

-- Define the function f
def f (x : ℝ) : ℝ := 25 * x^3 + 13 * x^2 + 2016 * x - 5

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 2016 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2683_268343


namespace NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2683_268344

theorem fraction_reciprocal_difference (x : ℚ) : 
  0 < x → x < 1 → (1 / x - x = 9 / 20) → x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2683_268344


namespace NUMINAMATH_CALUDE_lollipop_cost_is_two_l2683_268377

/-- The cost of a single lollipop in dollars -/
def lollipop_cost : ℝ := 2

/-- The number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- The number of chocolate packs bought -/
def num_chocolate_packs : ℕ := 6

/-- The number of $10 bills used for payment -/
def num_ten_dollar_bills : ℕ := 6

/-- The amount of change received in dollars -/
def change_received : ℝ := 4

theorem lollipop_cost_is_two :
  lollipop_cost = 2 ∧
  num_lollipops * lollipop_cost + num_chocolate_packs * (4 * lollipop_cost) = 
    num_ten_dollar_bills * 10 - change_received :=
by sorry

end NUMINAMATH_CALUDE_lollipop_cost_is_two_l2683_268377


namespace NUMINAMATH_CALUDE_expression_evaluation_l2683_268382

theorem expression_evaluation :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2683_268382


namespace NUMINAMATH_CALUDE_area_between_tangent_circles_l2683_268333

/-- The area of the region between two tangent circles -/
theorem area_between_tangent_circles
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ)
  (h₁ : r₁ = 4)
  (h₂ : r₂ = 7)
  (h₃ : d = 3)
  (h₄ : d = r₂ - r₁) :
  π * (r₂^2 - r₁^2) = 33 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_tangent_circles_l2683_268333


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l2683_268346

theorem complex_sum_of_parts (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 : ℂ) + 2 * i = a + b * i) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l2683_268346


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2683_268312

/-- Proves that the actual distance traveled is 100 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_diff : ℝ) 
  (h1 : speed_slow = 10)
  (h2 : speed_fast = 12)
  (h3 : distance_diff = 20)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ),
    actual_distance / speed_slow = (actual_distance + distance_diff) / speed_fast ∧
    actual_distance = 100 :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2683_268312


namespace NUMINAMATH_CALUDE_diagonal_length_from_offsets_and_area_l2683_268390

/-- The length of a diagonal of a quadrilateral, given its offsets and area -/
theorem diagonal_length_from_offsets_and_area 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  offset1 = 7 →
  offset2 = 3 →
  area = 50 →
  ∃ (d : ℝ), d = 10 ∧ area = (1/2) * d * (offset1 + offset2) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_from_offsets_and_area_l2683_268390


namespace NUMINAMATH_CALUDE_sharp_triple_30_l2683_268397

-- Define the function #
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_triple_30 : sharp (sharp (sharp 30)) = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_sharp_triple_30_l2683_268397


namespace NUMINAMATH_CALUDE_custom_operation_result_l2683_268393

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- State the theorem
theorem custom_operation_result : star (star 2 4) 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l2683_268393


namespace NUMINAMATH_CALUDE_line_through_point_l2683_268303

/-- Given a line ax - y - 1 = 0 passing through the point (1, 3), prove that a = 4 -/
theorem line_through_point (a : ℝ) : (a * 1 - 3 - 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2683_268303


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2683_268399

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_transitive 
  (a b c : Line) :
  perpendicular a b → parallel_line b c → perpendicular a c :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l2683_268399


namespace NUMINAMATH_CALUDE_election_result_l2683_268379

/-- Represents an election with three candidates -/
structure Election where
  total_votes : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Conditions for the specific election scenario -/
def election_conditions (e : Election) : Prop :=
  e.votes_a = (32 * e.total_votes) / 100 ∧
  e.votes_b = (42 * e.total_votes) / 100 ∧
  e.votes_c = e.votes_b - 1908 ∧
  e.total_votes = e.votes_a + e.votes_b + e.votes_c

/-- The theorem to be proved -/
theorem election_result (e : Election) (h : election_conditions e) :
  e.votes_c = (26 * e.total_votes) / 100 ∧ e.total_votes = 11925 := by
  sorry

#check election_result

end NUMINAMATH_CALUDE_election_result_l2683_268379


namespace NUMINAMATH_CALUDE_inverse_proposition_l2683_268306

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, x > y → x^3 > y^3) →
  (a^3 > b^3 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2683_268306


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2683_268332

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 9) (h2 : x * y = 10) : x^3 + y^3 = 459 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l2683_268332


namespace NUMINAMATH_CALUDE_chess_game_probability_l2683_268376

theorem chess_game_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2683_268376


namespace NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l2683_268350

theorem sixth_grade_homework_forgetfulness (students_A : ℕ) (students_B : ℕ) 
  (forgot_A_percent : ℚ) (forgot_B_percent : ℚ) (total_forgot_percent : ℚ) :
  students_A = 20 →
  forgot_A_percent = 20 / 100 →
  forgot_B_percent = 15 / 100 →
  total_forgot_percent = 16 / 100 →
  (students_A : ℚ) * forgot_A_percent + (students_B : ℚ) * forgot_B_percent = 
    total_forgot_percent * ((students_A : ℚ) + (students_B : ℚ)) →
  students_B = 80 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l2683_268350


namespace NUMINAMATH_CALUDE_exists_non_illuminating_rotation_l2683_268326

/-- Represents a three-dimensional cube --/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a projector that illuminates an octant --/
structure Projector where
  position : ℝ × ℝ × ℝ
  illumination : Set (ℝ × ℝ × ℝ)

/-- Represents a rotation in three-dimensional space --/
structure Rotation where
  matrix : Matrix (Fin 3) (Fin 3) ℝ

/-- Function to check if a point is illuminated by the projector --/
def is_illuminated (p : Projector) (point : ℝ × ℝ × ℝ) : Prop :=
  point ∈ p.illumination

/-- Function to apply a rotation to a projector --/
def rotate_projector (r : Rotation) (p : Projector) : Projector :=
  sorry

/-- Theorem stating that there exists a rotation such that no vertices are illuminated --/
theorem exists_non_illuminating_rotation (c : Cube) (p : Projector) :
  p.position = (0, 0, 0) →  -- Projector is at the center of the cube
  ∃ (r : Rotation), ∀ (v : Fin 8), ¬is_illuminated (rotate_projector r p) (c.vertices v) :=
sorry

end NUMINAMATH_CALUDE_exists_non_illuminating_rotation_l2683_268326


namespace NUMINAMATH_CALUDE_octagon_diagonals_l2683_268340

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l2683_268340


namespace NUMINAMATH_CALUDE_inequalities_proof_l2683_268395

theorem inequalities_proof (a b c d : ℝ) : 
  ((a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧ 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2683_268395


namespace NUMINAMATH_CALUDE_expression_simplification_l2683_268324

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2683_268324


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l2683_268345

theorem student_multiplication_problem (initial_number : ℕ) (final_result : ℕ) : 
  initial_number = 48 → final_result = 102 → ∃ (x : ℕ), initial_number * x - 138 = final_result ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l2683_268345


namespace NUMINAMATH_CALUDE_dana_earnings_l2683_268363

def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

theorem dana_earnings : 
  hourly_rate * (friday_hours + saturday_hours + sunday_hours) = 286 := by
  sorry

end NUMINAMATH_CALUDE_dana_earnings_l2683_268363


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2683_268342

theorem sum_of_fractions : (3 : ℚ) / 7 + (5 : ℚ) / 14 = (11 : ℚ) / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2683_268342


namespace NUMINAMATH_CALUDE_age_problem_l2683_268370

/-- Represents the ages of Sandy, Molly, and Kim -/
structure Ages where
  sandy : ℝ
  molly : ℝ
  kim : ℝ

/-- The problem statement -/
theorem age_problem (current : Ages) (future : Ages) : 
  -- Current ratio condition
  (current.sandy / current.molly = 4 / 3) ∧
  (current.sandy / current.kim = 4 / 5) ∧
  -- Future age condition
  (future.sandy = current.sandy + 8) ∧
  (future.molly = current.molly + 8) ∧
  (future.kim = current.kim + 8) ∧
  -- Future Sandy's age
  (future.sandy = 74) ∧
  -- Future ratio condition
  (future.sandy / future.molly = 9 / 7) ∧
  (future.sandy / future.kim = 9 / 10) →
  -- Conclusion
  current.molly = 49.5 ∧ current.kim = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l2683_268370


namespace NUMINAMATH_CALUDE_congruence_problem_l2683_268313

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 16 [ZMOD 44]) (h2 : b ≡ 77 [ZMOD 44]) :
  (a - b ≡ 159 [ZMOD 44]) ∧
  (∀ n : ℤ, 120 ≤ n ∧ n ≤ 161 → (a - b ≡ n [ZMOD 44] ↔ n = 159)) :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l2683_268313


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_23_l2683_268354

theorem greatest_four_digit_multiple_of_23 : ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 23 ∣ n → n ≤ 9978 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_23_l2683_268354


namespace NUMINAMATH_CALUDE_second_class_average_marks_l2683_268309

theorem second_class_average_marks (n1 n2 : ℕ) (avg1 avg_total : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg_total = 51.25 →
  (n1 * avg1 + n2 * (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2) / (n1 + n2) = avg_total →
  (n1 * avg1 + n2 * avg_total - n1 * avg1) / n2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_marks_l2683_268309


namespace NUMINAMATH_CALUDE_inequality_solution_l2683_268371

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 (13/2) ∪ {13/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2683_268371


namespace NUMINAMATH_CALUDE_total_germs_count_l2683_268318

/-- The number of petri dishes in the biology lab. -/
def num_dishes : ℕ := 10800

/-- The number of germs in a single petri dish. -/
def germs_per_dish : ℕ := 500

/-- The total number of germs in the biology lab. -/
def total_germs : ℕ := num_dishes * germs_per_dish

/-- Theorem stating that the total number of germs is 5,400,000. -/
theorem total_germs_count : total_germs = 5400000 := by
  sorry

end NUMINAMATH_CALUDE_total_germs_count_l2683_268318
