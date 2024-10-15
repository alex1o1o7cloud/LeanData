import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l756_75604

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4))
  (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l756_75604


namespace NUMINAMATH_CALUDE_nathan_baseball_weeks_l756_75687

/-- Nathan's baseball playing problem -/
theorem nathan_baseball_weeks (nathan_daily_hours tobias_daily_hours : ℕ) 
  (total_hours : ℕ) (tobias_weeks : ℕ) :
  nathan_daily_hours = 3 →
  tobias_daily_hours = 5 →
  tobias_weeks = 1 →
  total_hours = 77 →
  ∃ w : ℕ, w * (7 * nathan_daily_hours) + tobias_weeks * (7 * tobias_daily_hours) = total_hours ∧ w = 2 := by
  sorry

#check nathan_baseball_weeks

end NUMINAMATH_CALUDE_nathan_baseball_weeks_l756_75687


namespace NUMINAMATH_CALUDE_certain_number_problem_l756_75697

theorem certain_number_problem : ∃ x : ℚ, (x * 30 + (12 + 8) * 3) / 5 = 1212 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l756_75697


namespace NUMINAMATH_CALUDE_hole_perimeter_formula_l756_75670

/-- Represents an isosceles trapezium -/
structure IsoscelesTrapezium where
  a : ℝ  -- Length of non-parallel sides
  b : ℝ  -- Length of longer parallel side

/-- Represents an equilateral triangle with a hole formed by three congruent isosceles trapeziums -/
structure TriangleWithHole where
  trapezium : IsoscelesTrapezium
  -- Assumption that three of these trapeziums form an equilateral triangle with a hole

/-- The perimeter of the hole in a TriangleWithHole -/
def holePerimeter (t : TriangleWithHole) : ℝ :=
  6 * t.trapezium.a - 3 * t.trapezium.b

/-- Theorem stating that the perimeter of the hole is 6a - 3b -/
theorem hole_perimeter_formula (t : TriangleWithHole) :
  holePerimeter t = 6 * t.trapezium.a - 3 * t.trapezium.b :=
by
  sorry

end NUMINAMATH_CALUDE_hole_perimeter_formula_l756_75670


namespace NUMINAMATH_CALUDE_max_colors_without_monochromatic_trapezium_l756_75694

/-- Regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Sequence of regular n-gons where each subsequent polygon's vertices are midpoints of the previous polygon's edges -/
def NGonSequence (n : ℕ) (m : ℕ) : ℕ → RegularNGon n
  | 0 => sorry
  | i + 1 => sorry

/-- A coloring of vertices of m n-gons using k colors -/
def Coloring (n m k : ℕ) := Fin m → Fin n → Fin k

/-- Predicate to check if four points form an isosceles trapezium -/
def IsIsoscelesTrapezium (a b c d : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a coloring contains a monochromatic isosceles trapezium -/
def HasMonochromaticIsoscelesTrapezium (n m k : ℕ) (coloring : Coloring n m k) : Prop := sorry

/-- The maximum number of colors that can be used without forming a monochromatic isosceles trapezium -/
theorem max_colors_without_monochromatic_trapezium 
  (n : ℕ) (m : ℕ) (h : m ≥ n^2 - n + 1) :
  (∃ (k : ℕ), k = n - 1 ∧ 
    (∀ (coloring : Coloring n m k), HasMonochromaticIsoscelesTrapezium n m k coloring) ∧
    (∃ (coloring : Coloring n m (k + 1)), ¬HasMonochromaticIsoscelesTrapezium n m (k + 1) coloring)) :=
sorry

end NUMINAMATH_CALUDE_max_colors_without_monochromatic_trapezium_l756_75694


namespace NUMINAMATH_CALUDE_negative_odd_number_representation_l756_75685

theorem negative_odd_number_representation (x : ℤ) :
  (x < 0 ∧ x % 2 = 1) → ∃ n : ℕ+, x = -2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_odd_number_representation_l756_75685


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l756_75628

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.depth + solid.width * solid.depth)

theorem rectangular_solid_length 
  (solid : RectangularSolid) 
  (h1 : solid.width = 5)
  (h2 : solid.depth = 2)
  (h3 : surfaceArea solid = 104) : 
  solid.length = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l756_75628


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l756_75631

-- Define the total number of balls and the number of each color
def totalBalls : ℕ := 6
def redBalls : ℕ := 3
def whiteBalls : ℕ := 3

-- Define the number of balls drawn
def ballsDrawn : ℕ := 3

-- Define the events
def atLeastTwoWhite (w : ℕ) : Prop := w ≥ 2
def allRed (r : ℕ) : Prop := r = 3

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Prop) : Prop :=
  ¬(e1 ∧ e2)

-- Define opposite events
def oppositeEvents (e1 e2 : Prop) : Prop :=
  ∀ (outcome : ℕ × ℕ), (e1 ∨ e2) ∧ ¬(e1 ∧ e2)

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  (mutuallyExclusive (atLeastTwoWhite whiteBalls) (allRed redBalls)) ∧
  ¬(oppositeEvents (atLeastTwoWhite whiteBalls) (allRed redBalls)) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l756_75631


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l756_75634

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p / q = 3 / 2 ∧ 
   p + q = -10 ∧ p * q = k ∧ 
   ∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l756_75634


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l756_75674

theorem diophantine_equation_solutions (x y : ℤ) :
  x^6 - y^2 = 648 ↔ (x = 3 ∧ y = 9) ∨ (x = -3 ∧ y = 9) ∨ (x = 3 ∧ y = -9) ∨ (x = -3 ∧ y = -9) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l756_75674


namespace NUMINAMATH_CALUDE_city_rental_rate_proof_l756_75660

/-- The cost per mile for City Rentals -/
def city_rental_rate : ℝ := 0.31

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def safety_rental_rate : ℝ := 0.29

/-- The number of miles at which the costs are equal -/
def equal_miles : ℝ := 150.0

theorem city_rental_rate_proof :
  city_base_cost + equal_miles * city_rental_rate =
  safety_base_cost + equal_miles * safety_rental_rate :=
by sorry

end NUMINAMATH_CALUDE_city_rental_rate_proof_l756_75660


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l756_75605

/-- A hexagon ABCDEF with specific side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon ABCDEF is 13 -/
theorem hexagon_perimeter :
  ∃ (h : Hexagon),
    h.AB = 2 ∧ h.BC = 2 ∧ h.CD = 2 ∧ h.DE = 2 ∧ h.EF = 2 ∧ h.FA = 3 ∧
    perimeter h = 13 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l756_75605


namespace NUMINAMATH_CALUDE_tan_equality_implies_160_degrees_l756_75619

theorem tan_equality_implies_160_degrees (x : Real) :
  0 ≤ x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
                                   (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 160 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_implies_160_degrees_l756_75619


namespace NUMINAMATH_CALUDE_repeating_base_k_representation_l756_75647

theorem repeating_base_k_representation (k : ℕ) (h1 : k > 0) : 
  (4 * k + 5) / (k^2 - 1) = 11 / 143 → k = 52 :=
by sorry

end NUMINAMATH_CALUDE_repeating_base_k_representation_l756_75647


namespace NUMINAMATH_CALUDE_modulus_of_z_l756_75649

theorem modulus_of_z (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l756_75649


namespace NUMINAMATH_CALUDE_solve_for_a_l756_75616

-- Define the equation for all a, b, and c
def equation (a b c : ℝ) : Prop :=
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, (∀ b c : ℝ, equation a b c) → a * 15 * 2 = 4 → a = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_for_a_l756_75616


namespace NUMINAMATH_CALUDE_negation_of_proposition_l756_75623

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*|x| ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*|x| < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l756_75623


namespace NUMINAMATH_CALUDE_shore_distance_l756_75622

/-- The distance between two shores A and B, given the movement of two boats --/
theorem shore_distance (d : ℝ) : d = 800 :=
  -- Define the meeting points
  let first_meeting : ℝ := 500
  let second_meeting : ℝ := d - 300

  -- Define the distances traveled by each boat at the first meeting
  let boat_m_first : ℝ := first_meeting
  let boat_b_first : ℝ := d - first_meeting

  -- Define the distances traveled by each boat at the second meeting
  let boat_m_second : ℝ := second_meeting
  let boat_b_second : ℝ := 300

  -- The ratio of distances traveled should be equal for both meetings
  have h : boat_m_first / boat_b_first = boat_m_second / boat_b_second := by sorry

  -- The distance d satisfies the equation derived from the equal ratios
  have eq : d * d - 800 * d = 0 := by sorry

  -- The only positive solution to this equation is 800
  sorry


end NUMINAMATH_CALUDE_shore_distance_l756_75622


namespace NUMINAMATH_CALUDE_four_digit_greater_than_three_digit_l756_75672

theorem four_digit_greater_than_three_digit :
  ∀ (a b : ℕ), (1000 ≤ a ∧ a < 10000) → (100 ≤ b ∧ b < 1000) → a > b :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_greater_than_three_digit_l756_75672


namespace NUMINAMATH_CALUDE_group_size_proof_l756_75681

theorem group_size_proof (n : ℕ) 
  (h1 : (40 - 20 : ℝ) / n = 2.5) : n = 8 := by
  sorry

#check group_size_proof

end NUMINAMATH_CALUDE_group_size_proof_l756_75681


namespace NUMINAMATH_CALUDE_sufficient_necessary_condition_l756_75639

-- Define the interval (1, 4]
def OpenClosedInterval := { x : ℝ | 1 < x ∧ x ≤ 4 }

-- Define the inequality function
def InequalityFunction (m : ℝ) (x : ℝ) := x^2 - m*x + m > 0

-- State the theorem
theorem sufficient_necessary_condition :
  ∀ m : ℝ, (∀ x ∈ OpenClosedInterval, InequalityFunction m x) ↔ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_necessary_condition_l756_75639


namespace NUMINAMATH_CALUDE_first_episode_length_l756_75657

/-- Given a series with four episodes, where the second episode is 62 minutes long,
    the third episode is 65 minutes long, the fourth episode is 55 minutes long,
    and the total duration of all four episodes is 4 hours,
    prove that the first episode is 58 minutes long. -/
theorem first_episode_length :
  ∀ (episode1 episode2 episode3 episode4 : ℕ),
  episode2 = 62 →
  episode3 = 65 →
  episode4 = 55 →
  episode1 + episode2 + episode3 + episode4 = 4 * 60 →
  episode1 = 58 :=
by
  sorry


end NUMINAMATH_CALUDE_first_episode_length_l756_75657


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l756_75621

theorem inverse_variation_problem (x w : ℝ) (k : ℝ) :
  (∀ x w, x^4 * w^(1/4) = k) →
  (3^4 * 16^(1/4) = k) →
  (6^4 * w^(1/4) = k) →
  w = 1 / 4096 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l756_75621


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l756_75627

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y + 2)^2) - Real.sqrt ((x - 6)^2 + (y + 2)^2) = 4

-- Define the distance between foci
def distance_between_foci : ℝ := 5

-- Define the semi-major axis
def semi_major_axis : ℝ := 2

-- Define the positive slope of an asymptote
def positive_asymptote_slope : ℝ := 0.75

-- Theorem statement
theorem hyperbola_asymptote_slope :
  positive_asymptote_slope = (Real.sqrt (((distance_between_foci / 2)^2) - semi_major_axis^2)) / semi_major_axis :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l756_75627


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l756_75658

theorem arithmetic_mean_of_fractions :
  let a := 8 / 11
  let b := 9 / 11
  let c := 7 / 11
  a = (b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l756_75658


namespace NUMINAMATH_CALUDE_pizza_combinations_l756_75632

theorem pizza_combinations (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l756_75632


namespace NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l756_75630

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let july_born_scientists : ℕ := 15
  (july_born_scientists : ℚ) / total_scientists * 100 = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_scientists_born_in_july_percentage_l756_75630


namespace NUMINAMATH_CALUDE_cookies_remaining_batches_l756_75662

/-- Given the following conditions:
  * Each batch of cookies requires 2 cups of flour
  * 3 batches of cookies were baked
  * The initial amount of flour was 20 cups
  Prove that 7 additional batches of cookies can be made with the remaining flour -/
theorem cookies_remaining_batches 
  (flour_per_batch : ℕ) 
  (batches_baked : ℕ) 
  (initial_flour : ℕ) : 
  flour_per_batch = 2 →
  batches_baked = 3 →
  initial_flour = 20 →
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 :=
by sorry

end NUMINAMATH_CALUDE_cookies_remaining_batches_l756_75662


namespace NUMINAMATH_CALUDE_weight_vest_savings_l756_75698

theorem weight_vest_savings (weight_vest_cost plate_weight plate_cost_per_pound
                             weight_vest_200_cost weight_vest_200_discount : ℕ) :
  weight_vest_cost = 250 →
  plate_weight = 200 →
  plate_cost_per_pound = 12 / 10 →
  weight_vest_200_cost = 700 →
  weight_vest_200_discount = 100 →
  (weight_vest_200_cost - weight_vest_200_discount) - 
  (weight_vest_cost + plate_weight * plate_cost_per_pound) = 110 := by
sorry

end NUMINAMATH_CALUDE_weight_vest_savings_l756_75698


namespace NUMINAMATH_CALUDE_sum_squares_range_l756_75654

/-- Given a positive constant k and a sequence of positive real numbers x_i whose sum equals k,
    the sum of x_i^2 can take any value in the open interval (0, k^2). -/
theorem sum_squares_range (k : ℝ) (x : ℕ → ℝ) (h_k_pos : k > 0) (h_x_pos : ∀ n, x n > 0)
    (h_sum_x : ∑' n, x n = k) :
  ∀ y, 0 < y ∧ y < k^2 → ∃ x : ℕ → ℝ,
    (∀ n, x n > 0) ∧ (∑' n, x n = k) ∧ (∑' n, (x n)^2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_range_l756_75654


namespace NUMINAMATH_CALUDE_prob_two_s_is_one_tenth_l756_75617

/-- The set of tiles containing letters G, A, U, S, and S -/
def tiles : Finset Char := {'G', 'A', 'U', 'S', 'S'}

/-- The number of S tiles in the set -/
def num_s_tiles : Nat := (tiles.filter (· = 'S')).card

/-- The probability of selecting two S tiles when choosing 2 tiles at random -/
def prob_two_s : ℚ := (num_s_tiles.choose 2 : ℚ) / (tiles.card.choose 2)

/-- Theorem stating that the probability of selecting two S tiles is 1/10 -/
theorem prob_two_s_is_one_tenth : prob_two_s = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_prob_two_s_is_one_tenth_l756_75617


namespace NUMINAMATH_CALUDE_percentage_calculation_l756_75609

theorem percentage_calculation (x : ℝ) (p : ℝ) (h1 : x = 60) (h2 : x = (p / 100) * x + 52.8) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l756_75609


namespace NUMINAMATH_CALUDE_exponent_product_equality_l756_75679

theorem exponent_product_equality : 
  (10 ^ 0.4) * (10 ^ 0.1) * (10 ^ 0.7) * (10 ^ 0.2) * (10 ^ 0.6) * (5 ^ 2) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_equality_l756_75679


namespace NUMINAMATH_CALUDE_triangle_properties_l756_75668

-- Define the triangle ABC
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  b = 4 → c = 5 → A = π / 3 →
  -- Properties to prove
  a = Real.sqrt 21 ∧ Real.sin (2 * B) = 4 * Real.sqrt 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l756_75668


namespace NUMINAMATH_CALUDE_F_fraction_difference_l756_75671

def F : ℚ := 925 / 999

theorem F_fraction_difference : ∃ (a b : ℕ), 
  F = a / b ∧ 
  (∀ (c d : ℕ), F = c / d → b ≤ d) ∧
  b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_F_fraction_difference_l756_75671


namespace NUMINAMATH_CALUDE_rectangle_vertical_length_l756_75645

/-- Given a rectangle with perimeter 50 cm and horizontal length 13 cm, prove its vertical length is 12 cm -/
theorem rectangle_vertical_length (perimeter : ℝ) (horizontal_length : ℝ) (vertical_length : ℝ) : 
  perimeter = 50 ∧ horizontal_length = 13 → 
  perimeter = 2 * (horizontal_length + vertical_length) →
  vertical_length = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_vertical_length_l756_75645


namespace NUMINAMATH_CALUDE_intersection_point_l756_75682

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 3) is the intersection of the line y = x + 3 and the y-axis -/
theorem intersection_point :
  line_equation 0 3 ∧ on_y_axis 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l756_75682


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l756_75664

/-- Proves that a 45% increase in breadth and 88.5% increase in area results in a 30% increase in length for a rectangle -/
theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.45 * B ∧ L' * B' = 1.885 * (L * B) → L' = 1.3 * L := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l756_75664


namespace NUMINAMATH_CALUDE_quadratic_factorization_l756_75665

theorem quadratic_factorization (b : ℤ) : 
  (∃ (m n p q : ℤ), 15 * x^2 + b * x + 15 = (m * x + n) * (p * x + q)) →
  (∃ k : ℤ, b = 2 * k) ∧ 
  ¬(∀ k : ℤ, ∃ (m n p q : ℤ), 15 * x^2 + (2 * k) * x + 15 = (m * x + n) * (p * x + q)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l756_75665


namespace NUMINAMATH_CALUDE_fifteen_more_than_two_thirds_of_120_l756_75644

theorem fifteen_more_than_two_thirds_of_120 : (2 / 3 : ℚ) * 120 + 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_more_than_two_thirds_of_120_l756_75644


namespace NUMINAMATH_CALUDE_sum_in_quadrant_IV_l756_75683

/-- Given complex numbers z₁ and z₂, prove that their sum lies in Quadrant IV -/
theorem sum_in_quadrant_IV (z₁ z₂ : ℂ) : 
  z₁ = 1 - 3*I ∧ z₂ = 3 + 2*I → (z₁ + z₂).re > 0 ∧ (z₁ + z₂).im < 0 :=
by
  sorry

#check sum_in_quadrant_IV

end NUMINAMATH_CALUDE_sum_in_quadrant_IV_l756_75683


namespace NUMINAMATH_CALUDE_remainder_calculation_l756_75613

theorem remainder_calculation (P Q R D Q' R' D' D'' Q'' R'' : ℤ)
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : D'' = D' + 1)
  (h4 : P = D'' * Q'' + R'') :
  R'' = R + D * R' - Q'' := by sorry

end NUMINAMATH_CALUDE_remainder_calculation_l756_75613


namespace NUMINAMATH_CALUDE_workshop_technicians_salary_l756_75641

/-- Represents the average salary of technicians in a workshop -/
def average_salary_technicians (total_workers : ℕ) (technicians : ℕ) (avg_salary_all : ℚ) (avg_salary_others : ℚ) : ℚ :=
  let other_workers := total_workers - technicians
  let total_salary := (avg_salary_all * total_workers : ℚ)
  let other_salary := (avg_salary_others * other_workers : ℚ)
  let technicians_salary := total_salary - other_salary
  technicians_salary / technicians

/-- Theorem stating that the average salary of technicians is 1000 given the workshop conditions -/
theorem workshop_technicians_salary :
  average_salary_technicians 22 7 850 780 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_technicians_salary_l756_75641


namespace NUMINAMATH_CALUDE_range_of_S_3_l756_75693

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem range_of_S_3 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h_a2 : a 2 = 2) :
  ∀ x : ℝ, (x ∈ Set.Iic (-2) ∪ Set.Ici 6) ↔ ∃ q : ℝ, q ≠ 0 ∧ S_3 a = x :=
sorry

end NUMINAMATH_CALUDE_range_of_S_3_l756_75693


namespace NUMINAMATH_CALUDE_polynomial_condition_l756_75625

/-- A polynomial P satisfying the given condition for all real a, b, c is of the form ax² + bx -/
theorem polynomial_condition (P : ℝ → ℝ) : 
  (∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_condition_l756_75625


namespace NUMINAMATH_CALUDE_unique_integer_congruence_l756_75608

theorem unique_integer_congruence :
  ∃! n : ℤ, 6 ≤ n ∧ n ≤ 12 ∧ n ≡ 10403 [ZMOD 7] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_congruence_l756_75608


namespace NUMINAMATH_CALUDE_remaining_budget_theorem_l756_75635

/-- Represents the annual budget of Centerville --/
def annual_budget : ℝ := 20000

/-- Represents the percentage of the budget spent on the public library --/
def library_percentage : ℝ := 0.15

/-- Represents the amount spent on the public library --/
def library_spending : ℝ := 3000

/-- Represents the percentage of the budget spent on public parks --/
def parks_percentage : ℝ := 0.24

/-- Theorem stating the remaining amount of the budget after library and parks spending --/
theorem remaining_budget_theorem :
  annual_budget * (1 - library_percentage - parks_percentage) = 12200 := by
  sorry


end NUMINAMATH_CALUDE_remaining_budget_theorem_l756_75635


namespace NUMINAMATH_CALUDE_correct_grooming_time_l756_75656

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  nailClipTime : ℕ  -- Time to clip one nail in seconds
  earCleanTime : ℕ  -- Time to clean one ear in seconds
  totalTime : ℕ     -- Total grooming time in seconds

/-- Calculates the total grooming time for a cat -/
def calculateGroomingTime (gt : GroomingTime) (numClaws numFeet numEars : ℕ) : ℕ :=
  (gt.nailClipTime * numClaws * numFeet) + (gt.earCleanTime * numEars) + 
  (gt.totalTime - (gt.nailClipTime * numClaws * numFeet) - (gt.earCleanTime * numEars))

/-- Theorem stating that the total grooming time is correct -/
theorem correct_grooming_time (gt : GroomingTime) :
  gt.nailClipTime = 10 → 
  gt.earCleanTime = 90 → 
  gt.totalTime = 640 → 
  calculateGroomingTime gt 4 4 2 = 640 := by
  sorry

#eval calculateGroomingTime { nailClipTime := 10, earCleanTime := 90, totalTime := 640 } 4 4 2

end NUMINAMATH_CALUDE_correct_grooming_time_l756_75656


namespace NUMINAMATH_CALUDE_exponential_function_max_min_sum_l756_75680

theorem exponential_function_max_min_sum (a : ℝ) (f : ℝ → ℝ) :
  a > 1 →
  (∀ x, f x = a^x) →
  (∃ max min : ℝ, (∀ x ∈ Set.Icc 0 1, f x ≤ max) ∧
                  (∀ x ∈ Set.Icc 0 1, min ≤ f x) ∧
                  max + min = 3) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_max_min_sum_l756_75680


namespace NUMINAMATH_CALUDE_four_engine_safer_than_two_engine_l756_75646

-- Define the success rate of an engine
variable (P : ℝ) 

-- Define the probability of successful flight for a 2-engine airplane
def prob_success_2engine (P : ℝ) : ℝ := P^2 + 2*P*(1-P)

-- Define the probability of successful flight for a 4-engine airplane
def prob_success_4engine (P : ℝ) : ℝ := P^4 + 4*P^3*(1-P) + 6*P^2*(1-P)^2

-- Theorem statement
theorem four_engine_safer_than_two_engine :
  ∀ P, 2/3 < P ∧ P < 1 → prob_success_4engine P > prob_success_2engine P :=
sorry

end NUMINAMATH_CALUDE_four_engine_safer_than_two_engine_l756_75646


namespace NUMINAMATH_CALUDE_function_value_at_shifted_point_l756_75640

/-- Given a function f(x) = a * tan³(x) + b * sin(x) + 1 where f(4) = 5, prove that f(2π - 4) = -3 -/
theorem function_value_at_shifted_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x ^ 3 + b * Real.sin x + 1) 
  (h2 : f 4 = 5) : 
  f (2 * Real.pi - 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_shifted_point_l756_75640


namespace NUMINAMATH_CALUDE_gcd_problem_l756_75686

theorem gcd_problem (n : ℕ) : 
  30 ≤ n ∧ n ≤ 40 ∧ Nat.gcd 15 n = 5 → n = 35 ∨ n = 40 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l756_75686


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_sum_exact_l756_75677

theorem min_value_sum (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  ∀ x y : ℝ, x > 1 → y > 1 → x * y - (x + y) = 1 → a + b ≤ x + y :=
by sorry

theorem min_value_sum_exact (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) :
  a + b = 2 * (Real.sqrt 2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_sum_exact_l756_75677


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l756_75690

theorem probability_at_least_one_defective (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 23) (h2 : defective_bulbs = 4) :
  let non_defective := total_bulbs - defective_bulbs
  let prob_both_non_defective := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 164 / 506 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l756_75690


namespace NUMINAMATH_CALUDE_work_completion_time_l756_75637

/-- Given two workers a and b, where a is twice as fast as b, and b can complete a work in 24 days,
    prove that a and b together can complete the work in 8 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : b * 24 = 1) :
  1 / (a + b) = 8 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l756_75637


namespace NUMINAMATH_CALUDE_sum_digits_888_base_8_l756_75661

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base 8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base 8 representation -/
def sumDigits (repr : BaseEightRepresentation) : Nat :=
  sorry

theorem sum_digits_888_base_8 :
  sumDigits (toBaseEight 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_888_base_8_l756_75661


namespace NUMINAMATH_CALUDE_grid_transformation_impossible_l756_75624

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 1 | 0, 1 => 2 | 0, 2 => 3
    | 1, 0 => 4 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 7 | 2, 1 => 8 | 2, 2 => 9

/-- The target grid configuration -/
def target_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 7 | 0, 1 => 9 | 0, 2 => 2
    | 1, 0 => 3 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 1 | 2, 1 => 4 | 2, 2 => 8

/-- Calculates the invariant of a grid -/
def grid_invariant (g : Grid) : ℤ :=
  (g 0 0 + g 0 2 + g 1 1 + g 2 0 + g 2 2) - (g 0 1 + g 1 0 + g 1 2 + g 2 1)

/-- Theorem stating the impossibility of transforming the initial grid to the target grid -/
theorem grid_transformation_impossible : 
  ¬∃ (f : Grid → Grid), (f initial_grid = target_grid ∧ 
    ∀ g : Grid, grid_invariant g = grid_invariant (f g)) :=
by
  sorry


end NUMINAMATH_CALUDE_grid_transformation_impossible_l756_75624


namespace NUMINAMATH_CALUDE_min_value_theorem_l756_75695

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  a + 4 * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ + 4 * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l756_75695


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l756_75659

/-- The diagonal of a rectangle with length 30√3 cm and width 30 cm is 60 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 30 * Real.sqrt 3
  let width : ℝ := 30
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 60 := by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l756_75659


namespace NUMINAMATH_CALUDE_union_intersection_result_intersection_complement_result_l756_75653

-- Define the universe set U
def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

-- Theorem for the first part
theorem union_intersection_result : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

-- Theorem for the second part
theorem intersection_complement_result : A ∩ (U \ (B ∪ C)) = {3} := by
  sorry

end NUMINAMATH_CALUDE_union_intersection_result_intersection_complement_result_l756_75653


namespace NUMINAMATH_CALUDE_circle_tangent_sum_of_radii_l756_75618

theorem circle_tangent_sum_of_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  ∃ r' : ℝ,
  (r' > 0) ∧
  ((r' - 4)^2 + r'^2 = (r' + 2)^2) ∧
  (r + r' = 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_of_radii_l756_75618


namespace NUMINAMATH_CALUDE_circle_tangent_triangle_l756_75633

/-- Given a circle with radius R externally tangent to triangle ABC, 
    prove that angle C is π/6 and the maximum area is (√3 + 2)/4 * R^2 -/
theorem circle_tangent_triangle (R a b c : ℝ) (A B C : ℝ) :
  R > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 3 * a - b) * Real.sin B →
  C = π / 6 ∧ 
  ∃ (S : ℝ), S ≤ (Real.sqrt 3 + 2) / 4 * R^2 ∧ 
    (∀ (A' B' C' : ℝ), A' + B' + C' = π → 
      1 / 2 * 2 * R * Real.sin A' * 2 * R * Real.sin B' * Real.sin C' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_triangle_l756_75633


namespace NUMINAMATH_CALUDE_sector_max_area_angle_l756_75655

/-- Given a sector with circumference 4 cm, the central angle that maximizes the area is π radians. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) :
  r * θ + 2 * r = 4 →  -- Circumference condition
  (∀ r' θ', r' * θ' + 2 * r' = 4 → 
    (1/2) * r^2 * θ ≥ (1/2) * r'^2 * θ') →  -- Area maximization condition
  θ = π :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_angle_l756_75655


namespace NUMINAMATH_CALUDE_f_2017_eq_cos_l756_75636

open Real

/-- Recursive definition of the function sequence f_n -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

/-- The 2017th function in the sequence equals cosine -/
theorem f_2017_eq_cos : f 2017 = cos := by
  sorry

end NUMINAMATH_CALUDE_f_2017_eq_cos_l756_75636


namespace NUMINAMATH_CALUDE_range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l756_75626

noncomputable section

-- Define the function f
def f (m a x : ℝ) : ℝ := -m * (a * x + 1) * Real.log x + x - a

-- Part 1
theorem range_of_m_when_a_is_zero (m : ℝ) :
  (∀ x > 1, f m 0 x ≥ 0) ↔ m ∈ Set.Iic (Real.exp 1) :=
sorry

-- Part 2
theorem x_minus_one_times_f_nonpositive (x : ℝ) (hx : x > 0) :
  (x - 1) * f 1 1 x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l756_75626


namespace NUMINAMATH_CALUDE_reflections_count_l756_75696

/-- Number of reflections Sarah sees in tall mirrors -/
def sarah_tall : ℕ := 10

/-- Number of reflections Sarah sees in wide mirrors -/
def sarah_wide : ℕ := 5

/-- Number of reflections Sarah sees in narrow mirrors -/
def sarah_narrow : ℕ := 8

/-- Number of reflections Ellie sees in tall mirrors -/
def ellie_tall : ℕ := 6

/-- Number of reflections Ellie sees in wide mirrors -/
def ellie_wide : ℕ := 3

/-- Number of reflections Ellie sees in narrow mirrors -/
def ellie_narrow : ℕ := 4

/-- Number of times they pass through tall mirrors -/
def times_tall : ℕ := 3

/-- Number of times they pass through wide mirrors -/
def times_wide : ℕ := 5

/-- Number of times they pass through narrow mirrors -/
def times_narrow : ℕ := 4

/-- The total number of reflections seen by Sarah and Ellie -/
def total_reflections : ℕ :=
  (sarah_tall * times_tall + sarah_wide * times_wide + sarah_narrow * times_narrow) +
  (ellie_tall * times_tall + ellie_wide * times_wide + ellie_narrow * times_narrow)

theorem reflections_count : total_reflections = 136 := by
  sorry

end NUMINAMATH_CALUDE_reflections_count_l756_75696


namespace NUMINAMATH_CALUDE_fish_length_difference_l756_75652

theorem fish_length_difference :
  let fish1_length : ℝ := 0.3
  let fish2_length : ℝ := 0.2
  fish1_length - fish2_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_difference_l756_75652


namespace NUMINAMATH_CALUDE_initial_workers_correct_l756_75650

/-- The number of initial workers required to complete a job -/
def initialWorkers : ℕ := 6

/-- The total amount of work for the job -/
def totalWork : ℕ := initialWorkers * 8

/-- Proves that the initial number of workers is correct given the problem conditions -/
theorem initial_workers_correct :
  totalWork = initialWorkers * 3 + (initialWorkers + 4) * 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l756_75650


namespace NUMINAMATH_CALUDE_scarf_price_reduction_l756_75648

/-- Calculates the final price of a scarf after two successive price reductions -/
theorem scarf_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 10 ∧ first_reduction = 0.3 ∧ second_reduction = 0.5 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_scarf_price_reduction_l756_75648


namespace NUMINAMATH_CALUDE_min_value_theorem_l756_75699

theorem min_value_theorem (x y a b : ℝ) (h1 : x - 2*y - 2 ≤ 0) 
  (h2 : x + y - 2 ≤ 0) (h3 : 2*x - y + 2 ≥ 0) (ha : a > 0) (hb : b > 0)
  (h4 : ∀ (x' y' : ℝ), x' - 2*y' - 2 ≤ 0 → x' + y' - 2 ≤ 0 → 2*x' - y' + 2 ≥ 0 
    → a*x' + b*y' + 5 ≥ a*x + b*y + 5)
  (h5 : a*x + b*y + 5 = 2) :
  (2/a + 3/b : ℝ) ≥ (10 + 4*Real.sqrt 6)/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l756_75699


namespace NUMINAMATH_CALUDE_jacob_guarantee_sheep_l756_75615

/-- The maximum square number in the list -/
def max_square : Nat := 2021^2

/-- The list of square numbers from 1^2 to 2021^2 -/
def square_list : List Nat := List.range 2021 |>.map (λ x => (x + 1)^2)

/-- The game state, including the current sum on the whiteboard and the remaining numbers -/
structure GameState where
  sum : Nat
  remaining : List Nat

/-- A player's strategy for choosing a number from the list -/
def Strategy := GameState → Nat

/-- The result of playing the game, counting the number of times the sum is divisible by 4 after Jacob's turn -/
def play_game (jacob_strategy : Strategy) (laban_strategy : Strategy) : Nat :=
  sorry

/-- The theorem stating that Jacob can guarantee at least 506 sheep -/
theorem jacob_guarantee_sheep :
  ∃ (jacob_strategy : Strategy),
    ∀ (laban_strategy : Strategy),
      play_game jacob_strategy laban_strategy ≥ 506 := by
  sorry

end NUMINAMATH_CALUDE_jacob_guarantee_sheep_l756_75615


namespace NUMINAMATH_CALUDE_correct_operation_l756_75688

theorem correct_operation (m : ℝ) : 3 * m^2 * (2 * m^3) = 6 * m^5 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l756_75688


namespace NUMINAMATH_CALUDE_anns_shopping_problem_l756_75676

theorem anns_shopping_problem (total_spent : ℝ) (shorts_price : ℝ) (shorts_count : ℕ) 
  (shoes_price : ℝ) (shoes_count : ℕ) (tops_count : ℕ) :
  total_spent = 75 →
  shorts_price = 7 →
  shorts_count = 5 →
  shoes_price = 10 →
  shoes_count = 2 →
  tops_count = 4 →
  (total_spent - (shorts_price * shorts_count + shoes_price * shoes_count)) / tops_count = 5 := by
sorry

end NUMINAMATH_CALUDE_anns_shopping_problem_l756_75676


namespace NUMINAMATH_CALUDE_equation_solutions_l756_75614

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (14*x - x^2)/(x + 2) * (x + (14 - x)/(x + 2))
  ∃ (a b c : ℝ), 
    (f a = 48 ∧ f b = 48 ∧ f c = 48) ∧
    (a = 4 ∧ b = (1 + Real.sqrt 193)/2 ∧ c = (1 - Real.sqrt 193)/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l756_75614


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l756_75612

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt 32 + Real.sqrt x = Real.sqrt 50 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l756_75612


namespace NUMINAMATH_CALUDE_hannahs_appliance_cost_l756_75606

/-- The total cost of a washing machine and dryer after applying a discount -/
def total_cost_after_discount (washing_machine_cost : ℝ) (dryer_cost_difference : ℝ) (discount_rate : ℝ) : ℝ :=
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating the total cost after discount for Hannah's purchase -/
theorem hannahs_appliance_cost :
  total_cost_after_discount 100 30 0.1 = 153 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_appliance_cost_l756_75606


namespace NUMINAMATH_CALUDE_max_value_rational_function_l756_75600

theorem max_value_rational_function : 
  ∃ (M : ℤ), M = 57 ∧ 
  (∀ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) ≤ M) ∧
  (∃ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) > M - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_rational_function_l756_75600


namespace NUMINAMATH_CALUDE_picture_distance_l756_75601

/-- Proves that for a wall of width 24 feet and a picture of width 4 feet hung in the center,
    the distance from the end of the wall to the nearest edge of the picture is 10 feet. -/
theorem picture_distance (wall_width picture_width : ℝ) (h1 : wall_width = 24) (h2 : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_picture_distance_l756_75601


namespace NUMINAMATH_CALUDE_distance_between_B_and_C_l756_75678

/-- The distance between two locations in kilometers -/
def distance_between (x y : ℝ) : ℝ := |x - y|

/-- The position of an individual after traveling for a given time -/
def position_after_time (initial_position velocity time : ℝ) : ℝ :=
  initial_position + velocity * time

/-- Arithmetic sequence of four speeds -/
structure ArithmeticSpeedSequence (v₁ v₂ v₃ v₄ : ℝ) : Prop where
  decreasing : v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄
  arithmetic : ∃ d : ℝ, v₁ - v₂ = d ∧ v₂ - v₃ = d ∧ v₃ - v₄ = d

theorem distance_between_B_and_C
  (vA vB vC vD : ℝ)  -- Speeds of individuals A, B, C, and D
  (n : ℝ)            -- Time when B and C meet
  (h1 : ArithmeticSpeedSequence vA vB vC vD)
  (h2 : position_after_time 0 vB n = position_after_time 60 (-vC) n)  -- B and C meet after n hours
  (h3 : position_after_time 0 vA (2*n) = position_after_time 60 vD (2*n))  -- A catches up with D after 2n hours
  : distance_between 60 (position_after_time 60 (-vC) n) = 30 :=
sorry

end NUMINAMATH_CALUDE_distance_between_B_and_C_l756_75678


namespace NUMINAMATH_CALUDE_root_not_sufficient_for_bisection_l756_75642

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

-- Define the condition for a function to have a root
def HasRoot (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

-- Define the conditions for the bisection method to be applicable
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

-- Theorem statement
theorem root_not_sufficient_for_bisection :
  ∃ f : ℝ → ℝ, HasRoot f ∧ ¬(∃ a b, BisectionApplicable f a b) :=
sorry

end NUMINAMATH_CALUDE_root_not_sufficient_for_bisection_l756_75642


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l756_75673

theorem smallest_number_divisible (n : ℕ) : n ≥ 1012 ∧ 
  (∀ m : ℕ, m < 1012 → 
    ¬(((m - 4) % 12 = 0) ∧ 
      ((m - 4) % 16 = 0) ∧ 
      ((m - 4) % 18 = 0) ∧ 
      ((m - 4) % 21 = 0) ∧ 
      ((m - 4) % 28 = 0))) →
  ((n - 4) % 12 = 0) ∧ 
  ((n - 4) % 16 = 0) ∧ 
  ((n - 4) % 18 = 0) ∧ 
  ((n - 4) % 21 = 0) ∧ 
  ((n - 4) % 28 = 0) :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l756_75673


namespace NUMINAMATH_CALUDE_sum_c_d_eq_eight_l756_75666

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  c : ℝ
  d : ℝ
  h : (2 * 4 + c = 16) ∧ (4 * 4 + d = 16)

/-- The sum of c and d for intersecting lines -/
def sum_c_d (lines : IntersectingLines) : ℝ := lines.c + lines.d

/-- Theorem: The sum of c and d equals 8 -/
theorem sum_c_d_eq_eight (lines : IntersectingLines) : sum_c_d lines = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_d_eq_eight_l756_75666


namespace NUMINAMATH_CALUDE_power_function_m_value_l756_75689

/-- A function f is a power function if it has the form f(x) = ax^n where a is a non-zero constant and n is a real number -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- Given that f(x) = (2m+3)x^(m^2-3) is a power function, prove that m = -1 -/
theorem power_function_m_value (m : ℝ) 
    (h : IsPowerFunction (fun x ↦ (2*m+3) * x^(m^2-3))) : 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_m_value_l756_75689


namespace NUMINAMATH_CALUDE_parabola_translation_l756_75663

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical translation
def vertical_translation (y : ℝ) : ℝ := y + 3

-- Define the horizontal translation
def horizontal_translation (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem parabola_translation :
  ∀ x : ℝ, vertical_translation (original_parabola (horizontal_translation x)) = (x - 1)^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l756_75663


namespace NUMINAMATH_CALUDE_derivative_x_minus_inverse_x_l756_75602

open Real

theorem derivative_x_minus_inverse_x (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x - 1 / x) x = 1 + 1 / x^2 :=
sorry

end NUMINAMATH_CALUDE_derivative_x_minus_inverse_x_l756_75602


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l756_75607

-- Define atomic weights
def atomic_weight_C : ℝ := 12
def atomic_weight_H : ℝ := 1
def atomic_weight_O : ℝ := 16

-- Define the compound properties
def hydrogen_atoms : ℕ := 6
def oxygen_atoms : ℕ := 2
def molecular_weight : ℝ := 122

-- Theorem to prove
theorem carbon_atoms_in_compound :
  ∃ (carbon_atoms : ℕ),
    (carbon_atoms : ℝ) * atomic_weight_C +
    (hydrogen_atoms : ℝ) * atomic_weight_H +
    (oxygen_atoms : ℝ) * atomic_weight_O =
    molecular_weight ∧
    carbon_atoms = 7 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l756_75607


namespace NUMINAMATH_CALUDE_gcf_of_120_180_300_l756_75638

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_300_l756_75638


namespace NUMINAMATH_CALUDE_circle_square_radius_l756_75620

theorem circle_square_radius (s : ℝ) (r : ℝ) : 
  s^2 = 9/16 →                  -- Area of the square is 9/16
  π * r^2 = 9/16 →              -- Area of the circle is 9/16
  2 * r = s →                   -- Diameter of circle equals side length of square
  r = 3/8 := by                 -- Radius of the circle is 3/8
sorry

end NUMINAMATH_CALUDE_circle_square_radius_l756_75620


namespace NUMINAMATH_CALUDE_additional_buffaloes_count_l756_75692

/-- Represents the daily fodder consumption of one buffalo -/
def buffalo_consumption : ℚ := 1

/-- Represents the daily fodder consumption of one cow -/
def cow_consumption : ℚ := 3/4 * buffalo_consumption

/-- Represents the daily fodder consumption of one ox -/
def ox_consumption : ℚ := 3/2 * buffalo_consumption

/-- Represents the initial number of buffaloes -/
def initial_buffaloes : ℕ := 15

/-- Represents the initial number of oxen -/
def initial_oxen : ℕ := 8

/-- Represents the initial number of cows -/
def initial_cows : ℕ := 24

/-- Represents the initial duration of fodder in days -/
def initial_duration : ℕ := 24

/-- Represents the number of additional cows -/
def additional_cows : ℕ := 60

/-- Represents the new duration of fodder in days -/
def new_duration : ℕ := 9

/-- Theorem stating that the number of additional buffaloes is 30 -/
theorem additional_buffaloes_count : 
  ∃ (x : ℕ), 
    (initial_buffaloes * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     initial_cows * cow_consumption) * initial_duration =
    ((initial_buffaloes + x) * buffalo_consumption + 
     initial_oxen * ox_consumption + 
     (initial_cows + additional_cows) * cow_consumption) * new_duration ∧
    x = 30 := by sorry

end NUMINAMATH_CALUDE_additional_buffaloes_count_l756_75692


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l756_75603

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧
  (a * (a + 2) = -1) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l756_75603


namespace NUMINAMATH_CALUDE_siblings_age_sum_l756_75629

/-- The age difference between siblings -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age_now : ℕ := 20

/-- The number of years into the future we're considering -/
def years_forward : ℕ := 10

/-- The total age of three siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_age_now + years) + (eldest_age_now - age_gap + years) + (eldest_age_now - 2 * age_gap + years)

theorem siblings_age_sum :
  total_age_after years_forward = 75 :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l756_75629


namespace NUMINAMATH_CALUDE_system_solution_l756_75691

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 37 ∧ x^4 + x^2*y^2 + y^4 = 481) ↔ 
  ((x = -4 ∧ y = -3) ∨ (x = -3 ∧ y = -4) ∨ (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l756_75691


namespace NUMINAMATH_CALUDE_sum_of_squares_219_l756_75651

theorem sum_of_squares_219 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^2 + b^2 + c^2 = 219 →
  (a : ℕ) + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_219_l756_75651


namespace NUMINAMATH_CALUDE_nested_radical_value_l756_75667

/-- Given a continuous nested radical X = √(x√(y√(z√(x√(y√(z...)))))), 
    prove that X = ∛(x^4 * y^2 * z) -/
theorem nested_radical_value (x y z : ℝ) (X : ℝ) 
  (h : X = Real.sqrt (x * Real.sqrt (y * Real.sqrt (z * X)))) :
  X = (x^4 * y^2 * z)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_nested_radical_value_l756_75667


namespace NUMINAMATH_CALUDE_sum_abcd_equals_21_l756_75643

theorem sum_abcd_equals_21 
  (a b c d : ℝ) 
  (h1 : a * c + a * d + b * c + b * d = 68) 
  (h2 : c + d = 4) : 
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_21_l756_75643


namespace NUMINAMATH_CALUDE_coins_probability_theorem_l756_75669

def total_coins : ℕ := 15
def num_quarters : ℕ := 3
def num_dimes : ℕ := 5
def num_nickels : ℕ := 7
def coins_drawn : ℕ := 8

def value_quarter : ℚ := 25 / 100
def value_dime : ℚ := 10 / 100
def value_nickel : ℚ := 5 / 100

def target_value : ℚ := 3 / 2

def probability_at_least_target : ℚ := 316 / 6435

theorem coins_probability_theorem :
  let total_outcomes := Nat.choose total_coins coins_drawn
  let successful_outcomes := 
    Nat.choose num_quarters 3 * Nat.choose num_dimes 5 +
    Nat.choose num_quarters 2 * Nat.choose num_dimes 4 * Nat.choose num_nickels 2
  (successful_outcomes : ℚ) / total_outcomes = probability_at_least_target :=
sorry

end NUMINAMATH_CALUDE_coins_probability_theorem_l756_75669


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l756_75675

theorem cubic_sum_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r) : 
  p^3 + q^3 + r^3 = -21 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l756_75675


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l756_75610

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence a_n, if a_3 + a_8 = 22 and a_6 = 7, then a_5 = 15 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_a6 : a 6 = 7) : 
  a 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l756_75610


namespace NUMINAMATH_CALUDE_max_profit_at_84_l756_75611

/-- Defective rate as a function of daily output -/
def defective_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- Daily profit as a function of daily output and profit per qualified instrument -/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 then
    (x * (1 - defective_rate x) * A) - (x * defective_rate x * (A/2))
  else 0

theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≠ 0 → daily_profit 84 A ≥ daily_profit x A :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_84_l756_75611


namespace NUMINAMATH_CALUDE_no_stop_probability_theorem_l756_75684

/-- Represents the probability of a green light at a traffic point -/
def greenLightProbability (duration : ℕ) : ℚ := duration / 60

/-- The probability that a car doesn't stop at all three points -/
def noStopProbability (durationA durationB durationC : ℕ) : ℚ :=
  (greenLightProbability durationA) * (greenLightProbability durationB) * (greenLightProbability durationC)

theorem no_stop_probability_theorem (durationA durationB durationC : ℕ) 
  (hA : durationA = 25) (hB : durationB = 35) (hC : durationC = 45) :
  noStopProbability durationA durationB durationC = 35 / 192 := by
  sorry

end NUMINAMATH_CALUDE_no_stop_probability_theorem_l756_75684
