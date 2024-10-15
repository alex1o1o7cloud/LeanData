import Mathlib

namespace NUMINAMATH_CALUDE_english_spanish_difference_l2839_283900

def hours_english : ℕ := 7
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 4

theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_spanish_difference_l2839_283900


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l2839_283937

theorem sum_of_three_squares (n : ℕ+) : ¬ ∃ x y z : ℤ, x^2 + y^2 + z^2 = 8 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l2839_283937


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2839_283984

/-- The equation of a line passing through two points. -/
def line_equation (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let m := (y2 - y1) / (x2 - x1)
  λ x y => y - y1 = m * (x - x1)

/-- Theorem: The equation of the line passing through P(-2, 5) and Q(4, 1/2) is 3x + 4y - 14 = 0. -/
theorem line_equation_through_points :
  let p1 : ℝ × ℝ := (-2, 5)
  let p2 : ℝ × ℝ := (4, 1/2)
  ∀ x y : ℝ, line_equation p1 p2 x y ↔ 3 * x + 4 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2839_283984


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_pairs_l2839_283926

theorem infinitely_many_coprime_pairs (m : ℤ) :
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ,
    let (x, y) := f n
    -- Condition 1: x and y are coprime
    Int.gcd x y = 1 ∧
    -- Condition 2: y divides x^2 + m
    (x^2 + m) % y = 0 ∧
    -- Condition 3: x divides y^2 + m
    (y^2 + m) % x = 0 ∧
    -- Ensure infinitely many distinct pairs
    (∀ k < n, f k ≠ f n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_pairs_l2839_283926


namespace NUMINAMATH_CALUDE_original_rectangle_area_l2839_283922

theorem original_rectangle_area (original_area new_area : ℝ) : 
  (∀ (length width : ℝ), 
    length > 0 → width > 0 → 
    original_area = length * width → 
    new_area = (2 * length) * (2 * width)) →
  new_area = 32 →
  original_area = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_original_rectangle_area_l2839_283922


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2839_283979

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 2]

-- State the theorem
theorem vector_sum_magnitude :
  ‖(a + b)‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2839_283979


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l2839_283903

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 4, -2]

theorem inverse_as_linear_combination :
  N⁻¹ = (1 / 10 : ℚ) • N + (-1 / 10 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l2839_283903


namespace NUMINAMATH_CALUDE_problem_solution_l2839_283901

theorem problem_solution (x : ℝ) (h1 : x ≠ 0) : Real.sqrt ((5 * x) / 7) = x → x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2839_283901


namespace NUMINAMATH_CALUDE_old_soldiers_participation_l2839_283971

/-- Represents the distribution of soldiers in different age groups for a parade. -/
structure ParadeDistribution where
  total_soldiers : ℕ
  young_soldiers : ℕ
  middle_soldiers : ℕ
  old_soldiers : ℕ
  parade_spots : ℕ
  young_soldiers_le : young_soldiers ≤ total_soldiers
  middle_soldiers_le : middle_soldiers ≤ total_soldiers
  old_soldiers_le : old_soldiers ≤ total_soldiers
  total_sum : young_soldiers + middle_soldiers + old_soldiers = total_soldiers

/-- The number of soldiers aged over 23 participating in the parade. -/
def old_soldiers_in_parade (d : ParadeDistribution) : ℕ :=
  min d.old_soldiers (d.parade_spots - (d.parade_spots / 3 * 2))

/-- Theorem stating that for the given distribution, 2 soldiers aged over 23 will participate. -/
theorem old_soldiers_participation (d : ParadeDistribution) 
  (h1 : d.total_soldiers = 45)
  (h2 : d.young_soldiers = 15)
  (h3 : d.middle_soldiers = 20)
  (h4 : d.old_soldiers = 10)
  (h5 : d.parade_spots = 9) :
  old_soldiers_in_parade d = 2 := by
  sorry


end NUMINAMATH_CALUDE_old_soldiers_participation_l2839_283971


namespace NUMINAMATH_CALUDE_spring_deformation_l2839_283988

/-- A uniform spring with two attached weights -/
structure Spring :=
  (k : ℝ)  -- Spring constant
  (m₁ : ℝ) -- Mass of the top weight
  (m₂ : ℝ) -- Mass of the bottom weight

/-- The gravitational acceleration constant -/
def g : ℝ := 9.81

/-- Deformation when the spring is held vertically at its midpoint -/
def vertical_deformation (s : Spring) (x₁ x₂ : ℝ) : Prop :=
  2 * s.k * x₁ = s.m₁ * g ∧ x₁ = 0.08 ∧ x₂ = 0.15

/-- Deformation when the spring is laid horizontally -/
def horizontal_deformation (s : Spring) (x : ℝ) : Prop :=
  s.k * x = s.m₁ * g

/-- Theorem stating the relationship between vertical and horizontal deformations -/
theorem spring_deformation (s : Spring) (x₁ x₂ x : ℝ) :
  vertical_deformation s x₁ x₂ → horizontal_deformation s x → x = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_spring_deformation_l2839_283988


namespace NUMINAMATH_CALUDE_skylar_donation_amount_l2839_283992

theorem skylar_donation_amount (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : 
  start_age = 13 →
  current_age = 33 →
  total_donation = 105000 →
  (total_donation : ℚ) / ((current_age - start_age) : ℚ) = 5250 := by
  sorry

end NUMINAMATH_CALUDE_skylar_donation_amount_l2839_283992


namespace NUMINAMATH_CALUDE_median_is_55_l2839_283952

/-- A set of consecutive integers with a specific property --/
structure ConsecutiveIntegerSet where
  first : ℤ  -- The first integer in the set
  count : ℕ  -- The number of integers in the set
  sum_property : ∀ n : ℕ, n ≤ count → first + (n - 1) + (first + (count - n)) = 110

/-- The median of a set of consecutive integers --/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + (s.first + (s.count - 1))) / 2

/-- Theorem: The median of the ConsecutiveIntegerSet is always 55 --/
theorem median_is_55 (s : ConsecutiveIntegerSet) : median s = 55 := by
  sorry


end NUMINAMATH_CALUDE_median_is_55_l2839_283952


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l2839_283921

/-- Represents a configuration of seven circles tangent to each other and two lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  is_geometric_sequence : ∃ r : ℝ, ∀ i : Fin 6, radii i.succ = radii i * r
  smallest_radius : radii 0 = 6
  largest_radius : radii 6 = 24

/-- The theorem stating that the radius of the fourth circle is 12 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l2839_283921


namespace NUMINAMATH_CALUDE_height_derivative_at_one_l2839_283955

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

-- State the theorem
theorem height_derivative_at_one :
  (deriv h) 1 = 0.2 := by sorry

end NUMINAMATH_CALUDE_height_derivative_at_one_l2839_283955


namespace NUMINAMATH_CALUDE_minimum_cost_for_25_apples_l2839_283907

/-- Represents a group of apples with its cost -/
structure AppleGroup where
  count : Nat
  cost : Nat

/-- Calculates the total number of apples from a list of apple groups -/
def totalApples (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.count) 0

/-- Calculates the total cost from a list of apple groups -/
def totalCost (groups : List AppleGroup) : Nat :=
  groups.foldl (fun sum group => sum + group.cost) 0

/-- Represents the store's apple pricing policy -/
def applePricing : List AppleGroup := [
  { count := 4, cost := 15 },
  { count := 7, cost := 25 }
]

theorem minimum_cost_for_25_apples :
  ∃ (purchase : List AppleGroup),
    totalApples purchase = 25 ∧
    purchase.length ≥ 3 ∧
    (∀ group ∈ purchase, group ∈ applePricing) ∧
    totalCost purchase = 90 ∧
    (∀ (other : List AppleGroup),
      totalApples other = 25 →
      other.length ≥ 3 →
      (∀ group ∈ other, group ∈ applePricing) →
      totalCost purchase ≤ totalCost other) :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_cost_for_25_apples_l2839_283907


namespace NUMINAMATH_CALUDE_fraction_simplification_l2839_283958

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2839_283958


namespace NUMINAMATH_CALUDE_ellipse_right_triangle_x_coordinate_l2839_283928

/-- The x-coordinate of a point on an ellipse forming a right triangle with the foci -/
theorem ellipse_right_triangle_x_coordinate 
  (x y : ℝ) 
  (h_ellipse : x^2/16 + y^2/25 = 1) 
  (h_on_ellipse : ∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = y)
  (h_foci : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0)
  (h_right_triangle : ∃ (F₁ F₂ : ℝ × ℝ), F₁.1 = 0 ∧ F₂.1 = 0 ∧ 
    (F₁.2 - y)^2 + x^2 + (F₂.2 - y)^2 + x^2 = (F₂.2 - F₁.2)^2) :
  x = 16/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_right_triangle_x_coordinate_l2839_283928


namespace NUMINAMATH_CALUDE_tangent_speed_l2839_283993

/-- Given the equation (a * T) / (a * T - R) = (L + x) / x, where x represents a distance,
    prove that the speed of a point determined by x is equal to a * L / R. -/
theorem tangent_speed (a R L T : ℝ) (x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  (x / T) = a * L / R := by
  sorry

end NUMINAMATH_CALUDE_tangent_speed_l2839_283993


namespace NUMINAMATH_CALUDE_tangent_length_right_triangle_l2839_283917

/-- Given a right triangle with legs a and b, and hypotenuse c,
    the length of the tangent to the circumcircle drawn parallel
    to the hypotenuse is equal to c(a + b)²/(2ab) -/
theorem tangent_length_right_triangle (a b c : ℝ) 
  (h_right : c^2 = a^2 + b^2) (h_pos : a > 0 ∧ b > 0) :
  let x := c * (a + b)^2 / (2 * a * b)
  ∃ (m : ℝ), m > 0 ∧ 
    (c / x = m / (m + c/2)) ∧
    (m * c = a * b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_right_triangle_l2839_283917


namespace NUMINAMATH_CALUDE_inverse_implies_negation_l2839_283989

theorem inverse_implies_negation (p : Prop) : 
  (¬p → p) → ¬p :=
sorry

end NUMINAMATH_CALUDE_inverse_implies_negation_l2839_283989


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2839_283985

theorem six_digit_numbers_with_zero (total_six_digit : Nat) (six_digit_no_zero : Nat) : Nat :=
  by
  have h1 : total_six_digit = 900000 := by sorry
  have h2 : six_digit_no_zero = 531441 := by sorry
  have h3 : total_six_digit ≥ six_digit_no_zero := by sorry
  exact total_six_digit - six_digit_no_zero

theorem six_digit_numbers_with_zero_count :
    six_digit_numbers_with_zero 900000 531441 = 368559 :=
  by sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_six_digit_numbers_with_zero_count_l2839_283985


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2839_283991

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d,
    prove that its area A can be expressed as A = (10/29) * d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) :
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l^2 + w^2 = d^2 ∧ l * w = (10/29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2839_283991


namespace NUMINAMATH_CALUDE_equation_solutions_l2839_283995

noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

def is_solution (x : ℝ) : Prop :=
  x ≠ 0.5 ∧ (floor x : ℝ) - Real.sqrt ((floor x : ℝ) / (x - 0.5)) - 6 / (x - 0.5) = 0

theorem equation_solutions :
  ∀ x : ℝ, is_solution x ↔ (x = -1.5 ∨ x = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2839_283995


namespace NUMINAMATH_CALUDE_complex_number_parts_opposite_l2839_283942

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (3 + Complex.I)
  (z.re = -z.im) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_parts_opposite_l2839_283942


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2839_283925

/-- The value of a cow in dollars -/
def cow_value : ℕ := 500

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 350

/-- The smallest positive debt that can be resolved using cows and sheep -/
def smallest_debt : ℕ := 50

/-- Theorem stating that the smallest_debt is the smallest positive value that can be expressed as a linear combination of cow_value and sheep_value with integer coefficients -/
theorem smallest_resolvable_debt : 
  smallest_debt = Nat.gcd cow_value sheep_value ∧
  ∃ (c s : ℤ), smallest_debt = c * cow_value + s * sheep_value ∧
  ∀ (d : ℕ), d > 0 → (∃ (x y : ℤ), d = x * cow_value + y * sheep_value) → d ≥ smallest_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2839_283925


namespace NUMINAMATH_CALUDE_total_salaries_is_4000_l2839_283923

/-- The total amount of A and B's salaries is $4000 -/
theorem total_salaries_is_4000 
  (a_salary : ℝ) 
  (b_salary : ℝ) 
  (h1 : a_salary = 3000)
  (h2 : 0.05 * a_salary = 0.15 * b_salary) : 
  a_salary + b_salary = 4000 := by
  sorry

#check total_salaries_is_4000

end NUMINAMATH_CALUDE_total_salaries_is_4000_l2839_283923


namespace NUMINAMATH_CALUDE_eight_dice_probability_l2839_283909

theorem eight_dice_probability : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 4  -- number of dice showing even numbers
  let p : ℚ := 1/2  -- probability of a single die showing an even number
  Nat.choose n k * p^n = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_eight_dice_probability_l2839_283909


namespace NUMINAMATH_CALUDE_infinite_triples_exist_l2839_283961

theorem infinite_triples_exist : 
  ∀ y : ℝ, ∃ x z : ℝ, 
    (x^2 + y = y^2 + z) ∧ 
    (y^2 + z = z^2 + x) ∧ 
    (z^2 + x = x^2 + y) ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_exist_l2839_283961


namespace NUMINAMATH_CALUDE_square_side_properties_l2839_283969

theorem square_side_properties (a : ℝ) (h : a > 0) (area_eq : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_properties_l2839_283969


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l2839_283964

/-- The configuration of five identical balls and a circumscribing square pyramid. -/
structure BallPyramidConfig where
  -- Radius of each ball
  ball_radius : ℝ
  -- Distance between centers of adjacent bottom balls
  bottom_center_distance : ℝ
  -- Height from floor to center of top ball
  top_ball_height : ℝ
  -- Edge length of the square base of the pyramid
  pyramid_base_edge : ℝ

/-- The theorem stating the edge length of the square base of the pyramid. -/
theorem pyramid_base_edge_length 
  (config : BallPyramidConfig) 
  (h1 : config.ball_radius = 2)
  (h2 : config.bottom_center_distance = 2 * config.ball_radius)
  (h3 : config.top_ball_height = 3 * config.ball_radius)
  (h4 : config.pyramid_base_edge = config.bottom_center_distance * Real.sqrt 2) :
  config.pyramid_base_edge = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l2839_283964


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l2839_283978

theorem sum_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x*y/(z+x*y)) + Real.sqrt (y*z/(x+y*z)) + Real.sqrt (z*x/(y+z*x)) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l2839_283978


namespace NUMINAMATH_CALUDE_starters_count_l2839_283970

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of starters to be chosen
def num_starters : ℕ := 6

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (triplets : ℕ) (twins : ℕ) (starters : ℕ) : ℕ :=
  -- No triplets and no twins
  Nat.choose (total - triplets - twins) starters +
  -- One triplet and no twins
  triplets * Nat.choose (total - triplets - twins) (starters - 1) +
  -- No triplets and one twin
  twins * Nat.choose (total - triplets - twins) (starters - 1) +
  -- One triplet and one twin
  triplets * twins * Nat.choose (total - triplets - twins) (starters - 2)

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_triplets num_twins num_starters = 4752 :=
by sorry

end NUMINAMATH_CALUDE_starters_count_l2839_283970


namespace NUMINAMATH_CALUDE_remaining_pieces_count_l2839_283948

/-- Represents the number of pieces in a standard chess set -/
def standard_set : Nat := 32

/-- Represents the total number of missing pieces -/
def missing_pieces : Nat := 12

/-- Represents the number of missing kings -/
def missing_kings : Nat := 1

/-- Represents the number of missing queens -/
def missing_queens : Nat := 2

/-- Represents the number of missing knights -/
def missing_knights : Nat := 3

/-- Represents the number of missing pawns -/
def missing_pawns : Nat := 6

/-- Theorem stating that the number of remaining pieces is 20 -/
theorem remaining_pieces_count :
  standard_set - missing_pieces = 20 :=
by
  sorry

#check remaining_pieces_count

end NUMINAMATH_CALUDE_remaining_pieces_count_l2839_283948


namespace NUMINAMATH_CALUDE_no_coprime_natural_solution_l2839_283976

theorem no_coprime_natural_solution :
  ¬ ∃ (x y : ℕ), 
    (x ≠ 0) ∧ (y ≠ 0) ∧ 
    (Nat.gcd x y = 1) ∧ 
    (y^2 + y = x^3 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_coprime_natural_solution_l2839_283976


namespace NUMINAMATH_CALUDE_equation_solutions_l2839_283914

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2) ∧
  (∀ x : ℝ, 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2839_283914


namespace NUMINAMATH_CALUDE_additional_spend_needed_l2839_283934

-- Define the minimum spend for free delivery
def min_spend : ℝ := 35

-- Define the prices and quantities of items
def chicken_price : ℝ := 6
def chicken_quantity : ℝ := 1.5
def lettuce_price : ℝ := 3
def cherry_tomatoes_price : ℝ := 2.5
def sweet_potato_price : ℝ := 0.75
def sweet_potato_quantity : ℕ := 4
def broccoli_price : ℝ := 2
def broccoli_quantity : ℕ := 2
def brussel_sprouts_price : ℝ := 2.5

-- Calculate the total cost of items in the cart
def total_cost : ℝ :=
  chicken_price * chicken_quantity +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_quantity +
  broccoli_price * broccoli_quantity +
  brussel_sprouts_price

-- Theorem: The difference between min_spend and total_cost is 11
theorem additional_spend_needed : min_spend - total_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_additional_spend_needed_l2839_283934


namespace NUMINAMATH_CALUDE_mollys_age_condition_mollys_age_proof_l2839_283902

/-- Molly's present age -/
def mollys_present_age : ℕ := 12

/-- Condition: Molly's age in 18 years will be 5 times her age 6 years ago -/
theorem mollys_age_condition : 
  mollys_present_age + 18 = 5 * (mollys_present_age - 6) :=
by sorry

/-- Proof that Molly's present age is 12 years old -/
theorem mollys_age_proof : 
  mollys_present_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_mollys_age_condition_mollys_age_proof_l2839_283902


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_probability_l2839_283965

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  r : ℝ  -- radius of inscribed sphere
  R : ℝ  -- radius of circumscribed sphere
  h : R = 3 * r  -- relationship between R and r

/-- External sphere tangent to a face of the tetrahedron and the circumscribed sphere -/
structure ExternalSphere (t : RegularTetrahedron) where
  radius : ℝ
  h : radius = 1.5 * t.r

/-- The probability theorem for the tetrahedron and spheres setup -/
theorem tetrahedron_sphere_probability (t : RegularTetrahedron) 
  (e : ExternalSphere t) (n : ℕ) (h_n : n = 4) :
  let v_external := n * (4 / 3 * Real.pi * e.radius ^ 3)
  let v_circumscribed := 4 / 3 * Real.pi * t.R ^ 3
  v_external ≤ v_circumscribed ∧ 
  v_external / v_circumscribed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_probability_l2839_283965


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2839_283938

theorem simple_interest_problem (P R : ℝ) (h : P * (R + 10) * 8 / 100 - P * R * 8 / 100 = 150) : P = 187.50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2839_283938


namespace NUMINAMATH_CALUDE_sam_winning_probability_l2839_283936

theorem sam_winning_probability :
  let hit_prob : ℚ := 2/5
  let miss_prob : ℚ := 3/5
  let p : ℚ := hit_prob + miss_prob * miss_prob * p
  p = 5/8 := by sorry

end NUMINAMATH_CALUDE_sam_winning_probability_l2839_283936


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l2839_283913

theorem price_reduction_sales_increase 
  (price_reduction : Real) 
  (revenue_increase : Real) 
  (sales_increase : Real) : 
  price_reduction = 0.35 → 
  revenue_increase = 0.17 → 
  (1 - price_reduction) * (1 + sales_increase) = 1 + revenue_increase → 
  sales_increase = 0.8 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l2839_283913


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2839_283996

theorem pure_imaginary_complex_number (θ : Real) : 
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ (y : Real), (Complex.cos θ + Complex.I) * (2 * Complex.sin θ - Complex.I) = Complex.I * y) →
  θ = 3 * Real.pi / 4 ∨ θ = 7 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2839_283996


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l2839_283982

theorem smallest_solution_quadratic (x : ℝ) :
  (8 * x^2 - 38 * x + 35 = 0) → x ≥ 1.25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l2839_283982


namespace NUMINAMATH_CALUDE_total_bottle_caps_l2839_283956

theorem total_bottle_caps (bottle_caps_per_child : ℕ) (number_of_children : ℕ) 
  (h1 : bottle_caps_per_child = 5) 
  (h2 : number_of_children = 9) : 
  bottle_caps_per_child * number_of_children = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_bottle_caps_l2839_283956


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2839_283960

/-- Given constants a, b, c, where P(a,c) is in the fourth quadrant,
    prove that ax^2 + bx + c = 0 has two distinct real roots -/
theorem quadratic_equation_roots (a b c : ℝ) 
  (h1 : a > 0) (h2 : c < 0) : 
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2839_283960


namespace NUMINAMATH_CALUDE_largest_angle_is_80_l2839_283963

-- Define a right angle in degrees
def right_angle : ℝ := 90

-- Define the triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.angle1 + t.angle2 = (4/3) * right_angle ∧
  t.angle2 = t.angle1 + 40 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem largest_angle_is_80 (t : Triangle) :
  triangle_conditions t → (max t.angle1 (max t.angle2 t.angle3) = 80) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_80_l2839_283963


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2839_283941

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (((2 * x + b) / x + (2 * x - b) / x) / 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2839_283941


namespace NUMINAMATH_CALUDE_wire_length_from_sphere_l2839_283983

/-- The length of a wire drawn from a metallic sphere -/
theorem wire_length_from_sphere (r_sphere r_wire : ℝ) (h : r_sphere = 24 ∧ r_wire = 0.16) :
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let l_wire := v_sphere / (Real.pi * r_wire ^ 2)
  l_wire = 675000 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_from_sphere_l2839_283983


namespace NUMINAMATH_CALUDE_inequality_proof_l2839_283966

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2839_283966


namespace NUMINAMATH_CALUDE_surface_is_cone_l2839_283932

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The equation of the surface in spherical coordinates -/
def surface_equation (c : ℝ) (p : SphericalPoint) : Prop :=
  p.ρ = c * Real.sin p.φ

/-- Definition of a cone in spherical coordinates -/
def is_cone (S : Set SphericalPoint) : Prop :=
  ∃ c > 0, ∀ p ∈ S, surface_equation c p

theorem surface_is_cone (c : ℝ) (hc : c > 0) :
    is_cone {p : SphericalPoint | surface_equation c p} := by
  sorry

end NUMINAMATH_CALUDE_surface_is_cone_l2839_283932


namespace NUMINAMATH_CALUDE_oliver_tickets_l2839_283973

def carnival_tickets (ferris_wheel_rides : ℕ) (bumper_car_rides : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride

theorem oliver_tickets : carnival_tickets 5 4 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_oliver_tickets_l2839_283973


namespace NUMINAMATH_CALUDE_trajectory_equation_l2839_283940

theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (x, y)
  let PA : ℝ := Real.sqrt ((x + 2)^2 + y^2)
  let PB : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  PA = 2 * PB → x^2 + y^2 - 4*x = 0 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2839_283940


namespace NUMINAMATH_CALUDE_triangle_sin_C_l2839_283949

theorem triangle_sin_C (a c : ℝ) (A : ℝ) :
  a = 7 →
  c = 3 →
  A = π / 3 →
  Real.sin (Real.arcsin ((c * Real.sin A) / a)) = 3 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_C_l2839_283949


namespace NUMINAMATH_CALUDE_ratio_problem_l2839_283999

theorem ratio_problem (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 5 / a) 
  (h4 : (x + 12) / (y + 12) = 3 / 4) : y - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2839_283999


namespace NUMINAMATH_CALUDE_game_outcome_for_similar_numbers_l2839_283945

/-- The game outcome for a given number -/
inductive Outcome
| Good
| Bad

/-- Definition of the game -/
def game (k : ℕ) (n : ℕ) : Outcome :=
  sorry

/-- Two numbers are similar if they are divisible by the same primes up to k -/
def similar (k : ℕ) (n n' : ℕ) : Prop :=
  ∀ p, p.Prime → p ≤ k → (p ∣ n ↔ p ∣ n')

theorem game_outcome_for_similar_numbers (k : ℕ) (n n' : ℕ) (h_k : k ≥ 2) (h_n : n ≥ k) (h_n' : n' ≥ k) (h_similar : similar k n n') :
  game k n = game k n' :=
sorry

end NUMINAMATH_CALUDE_game_outcome_for_similar_numbers_l2839_283945


namespace NUMINAMATH_CALUDE_find_b_l2839_283981

/-- Given two functions p and q, prove that b = 7 when p(q(5)) = 11 -/
theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x - 5)
  (hq : ∀ x, q x = 3 * x - b)
  (h_pq : p (q 5) = 11) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_find_b_l2839_283981


namespace NUMINAMATH_CALUDE_unique_real_root_of_cubic_l2839_283980

theorem unique_real_root_of_cubic (α : Real) (h : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃! x : Real, x^3 + x^2 * Real.cos α + x * Real.sin α + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_root_of_cubic_l2839_283980


namespace NUMINAMATH_CALUDE_sausage_division_ratio_l2839_283972

/-- Represents the length of the sausage after each bite -/
def remaining_length (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then 3/4 * remaining_length n else 2/3 * remaining_length n

/-- Theorem stating that the sausage should be divided in a 1:1 ratio -/
theorem sausage_division_ratio :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |remaining_length n - 1/2| < ε :=
sorry

end NUMINAMATH_CALUDE_sausage_division_ratio_l2839_283972


namespace NUMINAMATH_CALUDE_simplify_square_root_l2839_283954

theorem simplify_square_root (x : ℝ) : 
  Real.sqrt (9 * x^6 + 3 * x^4) = Real.sqrt 3 * x^2 * Real.sqrt (3 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_l2839_283954


namespace NUMINAMATH_CALUDE_delta_eight_four_l2839_283924

/-- The Δ operation for non-zero integers -/
def delta (a b : ℤ) : ℚ :=
  a - a / b

/-- Theorem stating that 8 Δ 4 = 6 -/
theorem delta_eight_four : delta 8 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_delta_eight_four_l2839_283924


namespace NUMINAMATH_CALUDE_amy_avocado_business_l2839_283946

/-- Proves that given Amy's avocado business conditions, n = 50 --/
theorem amy_avocado_business (n : ℕ+) : 
  (15 * n : ℕ) = (15 * n : ℕ) ∧  -- Amy bought and sold 15n avocados
  (12 * n - 10 * n : ℕ) = 100 ∧  -- She made a profit of $100
  (2 : ℕ) = (2 : ℕ) ∧            -- She paid $2 for every 3 avocados
  (4 : ℕ) = (4 : ℕ)              -- She sold every 5 avocados for $4
  → n = 50 := by
sorry

end NUMINAMATH_CALUDE_amy_avocado_business_l2839_283946


namespace NUMINAMATH_CALUDE_min_real_floor_power_inequality_l2839_283968

theorem min_real_floor_power_inequality :
  ∃ (x : ℝ), x = Real.rpow 3 (1/3) ∧
  (∀ (n : ℕ), ⌊x^n⌋ < ⌊x^(n+1)⌋) ∧
  (∀ (y : ℝ), y < x → ∃ (m : ℕ), ⌊y^m⌋ ≥ ⌊y^(m+1)⌋) :=
by sorry

end NUMINAMATH_CALUDE_min_real_floor_power_inequality_l2839_283968


namespace NUMINAMATH_CALUDE_f_expression_for_x_gt_1_l2839_283944

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (-x + 1)

theorem f_expression_for_x_gt_1 (f : ℝ → ℝ) 
  (h1 : is_even_shifted f) 
  (h2 : ∀ x, x < 1 → f x = x^2 + 1) :
  ∀ x, x > 1 → f x = x^2 - 4*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_for_x_gt_1_l2839_283944


namespace NUMINAMATH_CALUDE_min_sum_of_products_l2839_283975

/-- A permutation of the numbers 1 to 12 -/
def Permutation12 := Fin 12 → Fin 12

/-- The sum of products for a given permutation -/
def sumOfProducts (p : Permutation12) : ℕ :=
  (p 0 + 1) * (p 1 + 1) * (p 2 + 1) +
  (p 3 + 1) * (p 4 + 1) * (p 5 + 1) +
  (p 6 + 1) * (p 7 + 1) * (p 8 + 1) +
  (p 9 + 1) * (p 10 + 1) * (p 11 + 1)

theorem min_sum_of_products :
  (∀ p : Permutation12, Function.Bijective p → sumOfProducts p ≥ 646) ∧
  (∃ p : Permutation12, Function.Bijective p ∧ sumOfProducts p = 646) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_products_l2839_283975


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2839_283929

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 40) (h2 : B = 80) : C = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2839_283929


namespace NUMINAMATH_CALUDE_train_speed_with_stoppages_l2839_283998

/-- Given a train that travels at 80 km/h without stoppages and stops for 15 minutes every hour,
    its average speed with stoppages is 60 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 80
  let stop_time_per_hour : ℝ := 15/60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (1 - stop_time_per_hour)
  speed_with_stoppages = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_with_stoppages_l2839_283998


namespace NUMINAMATH_CALUDE_surface_area_of_rearranged_cube_l2839_283962

-- Define the cube and its properties
def cube_volume : ℝ := 8
def cube_side_length : ℝ := 2

-- Define the cuts
def first_cut_distance : ℝ := 1
def second_cut_distance : ℝ := 0.5

-- Define the heights of the pieces
def height_X : ℝ := first_cut_distance
def height_Y : ℝ := second_cut_distance
def height_Z : ℝ := cube_side_length - (first_cut_distance + second_cut_distance)

-- Define the total width of the rearranged pieces
def total_width : ℝ := height_X + height_Y + height_Z

-- Theorem statement
theorem surface_area_of_rearranged_cube :
  cube_volume = cube_side_length ^ 3 →
  (2 * cube_side_length * cube_side_length +    -- Top and bottom surfaces
   2 * total_width * cube_side_length +         -- Side surfaces
   2 * cube_side_length * cube_side_length) = 46 := by
sorry

end NUMINAMATH_CALUDE_surface_area_of_rearranged_cube_l2839_283962


namespace NUMINAMATH_CALUDE_valid_arrangements_eq_48_l2839_283953

/-- The number of people in the lineup -/
def n : ℕ := 5

/-- A function that calculates the number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of valid arrangements for 5 people is 48 -/
theorem valid_arrangements_eq_48 : validArrangements n = 48 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_eq_48_l2839_283953


namespace NUMINAMATH_CALUDE_signal_count_is_324_l2839_283931

/-- Represents the number of indicator lights in a row -/
def total_lights : Nat := 6

/-- Represents the number of lights displayed at a time -/
def displayed_lights : Nat := 3

/-- Represents the number of possible colors for each light -/
def color_options : Nat := 3

/-- Calculates the number of different signals that can be displayed -/
def signal_count : Nat :=
  let adjacent_pair_positions := total_lights - 1
  let non_adjacent_positions := total_lights - 2
  (adjacent_pair_positions * non_adjacent_positions) * color_options^displayed_lights

/-- Theorem stating that the number of different signals is 324 -/
theorem signal_count_is_324 : signal_count = 324 := by
  sorry

end NUMINAMATH_CALUDE_signal_count_is_324_l2839_283931


namespace NUMINAMATH_CALUDE_blue_balls_in_jar_l2839_283986

theorem blue_balls_in_jar (total : ℕ) (blue : ℕ) (prob : ℚ) : 
  total = 12 →
  blue ≤ total →
  prob = 1 / 55 →
  (blue.choose 3 : ℚ) / (total.choose 3 : ℚ) = prob →
  blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_in_jar_l2839_283986


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l2839_283939

/-- The amount Grant spends yearly on newspaper delivery -/
def grant_yearly_cost : ℝ := 200

/-- The amount Juanita spends on newspapers Monday through Saturday -/
def juanita_weekday_cost : ℝ := 0.5

/-- The amount Juanita spends on newspapers on Sunday -/
def juanita_sunday_cost : ℝ := 2

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of weekdays (Monday through Saturday) -/
def weekdays : ℕ := 6

theorem newspaper_cost_difference : 
  (weekdays * juanita_weekday_cost + juanita_sunday_cost) * weeks_per_year - grant_yearly_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l2839_283939


namespace NUMINAMATH_CALUDE_delta_computation_l2839_283947

-- Define the custom operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 2)) (4^(delta 7 3)) = 5^68 - 4^46 := by
  sorry

end NUMINAMATH_CALUDE_delta_computation_l2839_283947


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2839_283905

theorem quadratic_equation_result (x : ℝ) (h : 6 * x^2 + 9 = 4 * x + 16) : (12 * x - 4)^2 = 188 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2839_283905


namespace NUMINAMATH_CALUDE_fraction_equality_l2839_283957

theorem fraction_equality : (1722^2 - 1715^2) / (1729^2 - 1708^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2839_283957


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2839_283994

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  altitude : ℝ
  perimeter : ℝ
  base_to_side_ratio : ℚ
  is_isosceles : base ≠ side
  altitude_value : altitude = 10
  perimeter_value : perimeter = 40
  ratio_value : base_to_side_ratio = 2 / 3

/-- The area of an isosceles triangle with the given properties is 80 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : t.base * t.altitude / 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2839_283994


namespace NUMINAMATH_CALUDE_cell_division_genetic_info_l2839_283904

-- Define the types for cells and genetic information
variable (Cell GeneticInfo : Type)

-- Define the functions for getting genetic information from a cell
variable (genetic_info : Cell → GeneticInfo)

-- Define the cells
variable (C₁ C₂ S₁ S₂ : Cell)

-- Define the property of being daughter cells from mitosis
variable (mitosis_daughter_cells : Cell → Cell → Prop)

-- Define the property of being secondary spermatocytes from meiosis I
variable (meiosis_I_secondary_spermatocytes : Cell → Cell → Prop)

-- State the theorem
theorem cell_division_genetic_info :
  mitosis_daughter_cells C₁ C₂ →
  meiosis_I_secondary_spermatocytes S₁ S₂ →
  (genetic_info C₁ = genetic_info C₂) ∧
  (genetic_info S₁ ≠ genetic_info S₂) :=
by sorry

end NUMINAMATH_CALUDE_cell_division_genetic_info_l2839_283904


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2839_283915

theorem power_of_three_mod_five : 3^2000 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2839_283915


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequences_l2839_283910

theorem smallest_sum_of_sequences (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C = (7 * B) / 4) →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (7 * B') / 4) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequences_l2839_283910


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2839_283906

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Proof that a chess tournament with 25 players, where each player plays 
    four times against every opponent, results in 1200 games total -/
theorem chess_tournament_games :
  num_games 25 4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2839_283906


namespace NUMINAMATH_CALUDE_fish_size_difference_l2839_283943

/-- The size difference between Seongjun's and Sungwoo's fish given the conditions -/
theorem fish_size_difference (S J W : ℝ) 
  (h1 : S = J + 21.52)
  (h2 : J = W - 12.64) :
  S - W = 8.88 := by
  sorry

end NUMINAMATH_CALUDE_fish_size_difference_l2839_283943


namespace NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l2839_283919

/-- Given a cone with radius 2 inches and height 8 inches, 
    prove that a sphere with twice the volume of this cone 
    has a radius of 2^(4/3) inches. -/
theorem sphere_radius_from_cone_volume 
  (cone_radius : ℝ) 
  (cone_height : ℝ) 
  (sphere_radius : ℝ) :
  cone_radius = 2 ∧ 
  cone_height = 8 ∧ 
  (4/3) * π * sphere_radius^3 = 2 * ((1/3) * π * cone_radius^2 * cone_height) →
  sphere_radius = 2^(4/3) :=
by
  sorry

#check sphere_radius_from_cone_volume

end NUMINAMATH_CALUDE_sphere_radius_from_cone_volume_l2839_283919


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2839_283990

theorem quadratic_factorization (a b : ℕ) (h1 : a ≥ b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  4*b - a = 27 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2839_283990


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l2839_283967

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 60)
  (h2 : discount_percentage = 0.20)
  (h3 : gross_profit = 4)
  (h4 : 0 ≤ markup_percentage ∧ markup_percentage < 1)
  (h5 : let selling_price := purchase_price / (1 - markup_percentage);
        gross_profit = selling_price * (1 - discount_percentage) - purchase_price) :
  markup_percentage = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l2839_283967


namespace NUMINAMATH_CALUDE_algorithm_can_contain_all_structures_l2839_283920

/-- Represents the types of logical structures in algorithms -/
inductive LogicalStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List LogicalStructure

/-- Theorem stating that an algorithm can contain all three types of logical structures -/
theorem algorithm_can_contain_all_structures :
  ∃ (a : Algorithm), (LogicalStructure.Sequential ∈ a.structures) ∧
                     (LogicalStructure.Conditional ∈ a.structures) ∧
                     (LogicalStructure.Loop ∈ a.structures) :=
by sorry


end NUMINAMATH_CALUDE_algorithm_can_contain_all_structures_l2839_283920


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2839_283997

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 * a 12 = 64 → a 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2839_283997


namespace NUMINAMATH_CALUDE_constant_term_proof_l2839_283977

/-- Given an equation (ax + w)(cx + d) = 6x^2 + x - 12, where a, w, c, and d are real numbers
    whose absolute values sum to 12, prove that the constant term in the expanded form is -12. -/
theorem constant_term_proof (a w c d : ℝ) 
    (eq : ∀ x, (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
    (sum_abs : |a| + |w| + |c| + |d| = 12) :
    w * d = -12 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l2839_283977


namespace NUMINAMATH_CALUDE_gcf_lcm_product_l2839_283916

def numbers : List Nat := [6, 18, 24]

theorem gcf_lcm_product (A B : Nat) 
  (h1 : A = Nat.gcd 6 (Nat.gcd 18 24))
  (h2 : B = Nat.lcm 6 (Nat.lcm 18 24)) :
  A * B = 432 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_l2839_283916


namespace NUMINAMATH_CALUDE_carly_to_lisa_tshirt_ratio_l2839_283911

def lisa_tshirts : ℚ := 40
def lisa_jeans : ℚ := lisa_tshirts / 2
def lisa_coats : ℚ := lisa_tshirts * 2

def carly_jeans : ℚ := lisa_jeans * 3
def carly_coats : ℚ := lisa_coats / 4

def total_spending : ℚ := 230

theorem carly_to_lisa_tshirt_ratio :
  ∃ (carly_tshirts : ℚ),
    lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats = total_spending ∧
    carly_tshirts / lisa_tshirts = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_carly_to_lisa_tshirt_ratio_l2839_283911


namespace NUMINAMATH_CALUDE_value_of_a_l2839_283987

theorem value_of_a : 
  let a := Real.sqrt ((19.19^2) + (39.19^2) - (38.38 * 39.19))
  a = 20 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l2839_283987


namespace NUMINAMATH_CALUDE_diagonal_planes_increment_l2839_283959

/-- The number of diagonal planes in a prism with k edges -/
def f (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem: The number of diagonal planes in a prism with k+1 edges
    is equal to the number of diagonal planes in a prism with k edges plus k-1 -/
theorem diagonal_planes_increment (k : ℕ) :
  f (k + 1) = f k + k - 1 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_planes_increment_l2839_283959


namespace NUMINAMATH_CALUDE_sphere_volume_for_maximized_tetrahedron_l2839_283951

theorem sphere_volume_for_maximized_tetrahedron (r : ℝ) (h : r = (3 * Real.sqrt 3) / 2) :
  (4 / 3) * Real.pi * r^3 = (27 * Real.sqrt 3 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_for_maximized_tetrahedron_l2839_283951


namespace NUMINAMATH_CALUDE_remaining_money_is_29_l2839_283974

/-- Calculates the remaining money after spending on a novel and lunch -/
def remaining_money (initial_amount novel_cost : ℕ) : ℕ :=
  initial_amount - (novel_cost + 2 * novel_cost)

/-- Theorem: Given $50 initial amount and $7 novel cost, the remaining money is $29 -/
theorem remaining_money_is_29 :
  remaining_money 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_29_l2839_283974


namespace NUMINAMATH_CALUDE_negative_quartic_count_l2839_283927

theorem negative_quartic_count : ∃ (S : Finset ℤ), (∀ x ∈ S, x^4 - 62*x^2 + 60 < 0) ∧ S.card = 12 ∧ 
  ∀ x : ℤ, x^4 - 62*x^2 + 60 < 0 → x ∈ S :=
sorry

end NUMINAMATH_CALUDE_negative_quartic_count_l2839_283927


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l2839_283933

/-- The number of different arrangements for 5 students and 2 teachers in a row,
    with exactly 2 students between the teachers. -/
def photo_arrangements : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students between the teachers -/
def students_between : ℕ := 2

theorem photo_arrangement_count :
  photo_arrangements = 960 ∧
  num_students = 5 ∧
  num_teachers = 2 ∧
  students_between = 2 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l2839_283933


namespace NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2839_283950

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r ≠ 0 ∧ ∀ n, b n = b₁ * r^(n - 1)

/-- The main theorem -/
theorem arithmetic_geometric_relation
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_relation : a 1 - (a 7)^2 + a 13 = 0)
  (h_equal : b 7 = a 7) :
  b 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2839_283950


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2839_283912

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice * 100

/-- Proves that the percentage loss is 10% given the conditions of the problem -/
theorem book_sale_loss_percentage
  (sellingPrice : ℚ)
  (gainPrice : ℚ)
  (gainPercentage : ℚ)
  (h1 : sellingPrice = 540)
  (h2 : gainPrice = 660)
  (h3 : gainPercentage = 10)
  (h4 : gainPrice = (100 + gainPercentage) / 100 * (gainPrice / (1 + gainPercentage / 100))) :
  percentageLoss (gainPrice / (1 + gainPercentage / 100)) sellingPrice = 10 := by
  sorry

#eval percentageLoss (660 / (1 + 10 / 100)) 540

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2839_283912


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2839_283918

/-- The function f(x) = x^2 + 12x - 15 -/
def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem root_exists_in_interval :
  (f 1.1 < 0) → (f 1.2 > 0) → ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2839_283918


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2839_283935

/-- Given two lines l₁ and l₂ in the plane, prove that a=2 is a necessary and sufficient condition for l₁ to be parallel to l₂. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, 2*x - a*y + 1 = 0 ↔ (a-1)*x - y + a = 0) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2839_283935


namespace NUMINAMATH_CALUDE_final_apartments_can_be_less_l2839_283930

/-- Represents the structure of an apartment building project -/
structure ApartmentProject where
  entrances : ℕ
  floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in a project -/
def total_apartments (p : ApartmentProject) : ℕ :=
  p.entrances * p.floors * p.apartments_per_floor

/-- Applies the architect's adjustments to a project -/
def adjust_project (p : ApartmentProject) (removed_entrances floors_added : ℕ) : ApartmentProject :=
  { entrances := p.entrances - removed_entrances,
    floors := p.floors + floors_added,
    apartments_per_floor := p.apartments_per_floor }

/-- The main theorem stating that the final number of apartments can be less than the initial number -/
theorem final_apartments_can_be_less :
  ∃ (initial : ApartmentProject)
    (removed_entrances1 floors_added1 removed_entrances2 floors_added2 : ℕ),
    initial.entrances = 5 ∧
    initial.floors = 2 ∧
    initial.apartments_per_floor = 1 ∧
    removed_entrances1 = 2 ∧
    floors_added1 = 3 ∧
    removed_entrances2 = 2 ∧
    floors_added2 = 3 ∧
    let first_adjustment := adjust_project initial removed_entrances1 floors_added1
    let final_project := adjust_project first_adjustment removed_entrances2 floors_added2
    total_apartments final_project < total_apartments initial :=
by
  sorry

end NUMINAMATH_CALUDE_final_apartments_can_be_less_l2839_283930


namespace NUMINAMATH_CALUDE_trig_equality_l2839_283908

theorem trig_equality (α β γ : Real) 
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
       (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) : 
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  |Real.cos α * Real.cos β * Real.cos γ| := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l2839_283908
