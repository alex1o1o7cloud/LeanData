import Mathlib

namespace NUMINAMATH_CALUDE_det_of_matrix_is_one_l2979_297900

theorem det_of_matrix_is_one : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_is_one_l2979_297900


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2979_297910

theorem sufficient_condition_for_inequality (x : ℝ) :
  0 < x ∧ x < 2 → x^2 - 3*x < 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2979_297910


namespace NUMINAMATH_CALUDE_intersection_forms_right_triangle_l2979_297997

/-- Given a line and a parabola that intersect at two points, prove that these points and the origin form a right triangle -/
theorem intersection_forms_right_triangle (m : ℝ) :
  ∃ (x₁ x₂ : ℝ),
    -- The points satisfy the line equation
    (m * x₁ - x₁^2 + 1 = 0) ∧ (m * x₂ - x₂^2 + 1 = 0) ∧
    -- The points are distinct
    (x₁ ≠ x₂) →
    -- The triangle formed by (0,0), (x₁, x₁^2), and (x₂, x₂^2) is right-angled
    (x₁ * x₂ + x₁^2 * x₂^2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_intersection_forms_right_triangle_l2979_297997


namespace NUMINAMATH_CALUDE_remainder_problem_l2979_297933

theorem remainder_problem (k : ℕ+) (h : ∃ q : ℕ, 120 = k^2 * q + 12) :
  ∃ r : ℕ, 160 = k * (160 / k) + r ∧ r < k ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2979_297933


namespace NUMINAMATH_CALUDE_max_a_value_l2979_297992

theorem max_a_value (x y : ℝ) (hx : 0 < x ∧ x ≤ 2) (hy : 0 < y ∧ y ≤ 2) (hxy : x * y = 2) :
  (∀ a : ℝ, (∀ x y : ℝ, 0 < x ∧ x ≤ 2 → 0 < y ∧ y ≤ 2 → x * y = 2 → 
    6 - 2*x - y ≥ a*(2 - x)*(4 - y)) → a ≤ 1) ∧ 
  (∃ a : ℝ, a = 1 ∧ ∀ x y : ℝ, 0 < x ∧ x ≤ 2 → 0 < y ∧ y ≤ 2 → x * y = 2 → 
    6 - 2*x - y ≥ a*(2 - x)*(4 - y)) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2979_297992


namespace NUMINAMATH_CALUDE_xy_sum_product_l2979_297911

theorem xy_sum_product (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 196/25 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_product_l2979_297911


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l2979_297990

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 18) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 54 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l2979_297990


namespace NUMINAMATH_CALUDE_basketball_score_proof_l2979_297931

theorem basketball_score_proof (junior_score : ℕ) (percentage_increase : ℚ) : 
  junior_score = 260 → percentage_increase = 20/100 →
  junior_score + (junior_score + junior_score * percentage_increase) = 572 :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l2979_297931


namespace NUMINAMATH_CALUDE_genuine_product_probability_l2979_297958

theorem genuine_product_probability 
  (p_second : ℝ) 
  (p_third : ℝ) 
  (h1 : p_second = 0.03) 
  (h2 : p_third = 0.01) 
  : 1 - (p_second + p_third) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_genuine_product_probability_l2979_297958


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l2979_297924

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l2979_297924


namespace NUMINAMATH_CALUDE_extended_segment_endpoint_l2979_297947

-- Define points in 2D space
def Point := ℝ × ℝ

-- Define the given points
def A : Point := (-3, 5)
def B : Point := (9, -1)

-- Define vector addition
def vadd (p q : Point) : Point := (p.1 + q.1, p.2 + q.2)

-- Define scalar multiplication
def smul (k : ℝ) (p : Point) : Point := (k * p.1, k * p.2)

-- Define vector from two points
def vec (p q : Point) : Point := (q.1 - p.1, q.2 - p.2)

-- Theorem statement
theorem extended_segment_endpoint (C : Point) :
  vec A B = smul 3 (vec B C) → C = (15, -4) := by
  sorry

end NUMINAMATH_CALUDE_extended_segment_endpoint_l2979_297947


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l2979_297919

theorem long_furred_brown_dogs
  (total : ℕ)
  (long_furred : ℕ)
  (brown : ℕ)
  (neither : ℕ)
  (h_total : total = 45)
  (h_long_furred : long_furred = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_furred + brown - (total - neither) = 11 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l2979_297919


namespace NUMINAMATH_CALUDE_least_six_digit_binary_l2979_297978

theorem least_six_digit_binary : ∃ n : ℕ, n = 32 ∧ 
  (∀ m : ℕ, m < n → (Nat.log 2 m).succ < 6) ∧
  (Nat.log 2 n).succ = 6 :=
sorry

end NUMINAMATH_CALUDE_least_six_digit_binary_l2979_297978


namespace NUMINAMATH_CALUDE_factor_polynomial_l2979_297988

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2979_297988


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2979_297929

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2979_297929


namespace NUMINAMATH_CALUDE_maintenance_interval_increase_l2979_297989

/-- The combined percentage increase in maintenance interval when using three additives -/
theorem maintenance_interval_increase (increase_a increase_b increase_c : ℝ) :
  increase_a = 0.2 →
  increase_b = 0.3 →
  increase_c = 0.4 →
  ((1 + increase_a) * (1 + increase_b) * (1 + increase_c) - 1) * 100 = 118.4 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_increase_l2979_297989


namespace NUMINAMATH_CALUDE_g_geq_f_implies_t_leq_one_l2979_297983

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := exp x - x * log x
def g (t : ℝ) (x : ℝ) : ℝ := exp x - t * x^2 + x

-- State the theorem
theorem g_geq_f_implies_t_leq_one (t : ℝ) :
  (∀ x > 0, g t x ≥ f x) → t ≤ 1 := by
  sorry

end

end NUMINAMATH_CALUDE_g_geq_f_implies_t_leq_one_l2979_297983


namespace NUMINAMATH_CALUDE_line_transformation_l2979_297961

open Matrix

/-- The matrix representing the linear transformation -/
def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

/-- The original line equation: x + y + 2 = 0 -/
def original_line (x y : ℝ) : Prop := x + y + 2 = 0

/-- The transformed line equation: x + 2y + 2 = 0 -/
def transformed_line (x y : ℝ) : Prop := x + 2*y + 2 = 0

/-- Theorem stating that the linear transformation maps the original line to the transformed line -/
theorem line_transformation :
  ∀ (x y : ℝ), original_line x y → 
  ∃ (x' y' : ℝ), M.mulVec ![x', y'] = ![x, y] ∧ transformed_line x' y' := by
sorry

end NUMINAMATH_CALUDE_line_transformation_l2979_297961


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l2979_297951

theorem baker_remaining_cakes 
  (initial_cakes : ℝ) 
  (additional_cakes : ℝ) 
  (sold_cakes : ℝ) 
  (h1 : initial_cakes = 62.5)
  (h2 : additional_cakes = 149.25)
  (h3 : sold_cakes = 144.75) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l2979_297951


namespace NUMINAMATH_CALUDE_cos_alpha_plus_beta_l2979_297974

theorem cos_alpha_plus_beta (α β : Real) 
  (h1 : Real.sin (3 * Real.pi / 4 + α) = 5 / 13)
  (h2 : Real.cos (Real.pi / 4 - β) = 3 / 5)
  (h3 : 0 < α) (h4 : α < Real.pi / 4) (h5 : Real.pi / 4 < β) (h6 : β < 3 * Real.pi / 4) :
  Real.cos (α + β) = -33 / 65 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_beta_l2979_297974


namespace NUMINAMATH_CALUDE_f_equality_f_explicit_formula_l2979_297930

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x - x^2

-- State the theorem
theorem f_equality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : 
  f (1 - Real.cos x) = Real.sin x ^ 2 := by
  sorry

-- Prove that f(x) = 2x - x^2 for 0 ≤ x ≤ 2
theorem f_explicit_formula (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  f x = 2 * x - x^2 := by
  sorry

end NUMINAMATH_CALUDE_f_equality_f_explicit_formula_l2979_297930


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l2979_297962

theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 + c^2 = a^2 + b*c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l2979_297962


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l2979_297964

/-- Given a rectangle with width w, length 10, and perimeter 30, 
    prove that the ratio of width to length is 1:2 -/
theorem rectangle_width_length_ratio 
  (w : ℝ) -- width of the rectangle
  (h1 : w > 0) -- width is positive
  (h2 : 2 * w + 2 * 10 = 30) -- perimeter formula
  : w / 10 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l2979_297964


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2979_297918

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 / x

theorem tangent_line_sum (a b m : ℝ) : 
  (∀ x : ℝ, 3 * x + f a 1 = b) →  -- Tangent line equation
  (∀ x : ℝ, f a x = a * Real.log x + 2 / x) →  -- Function definition
  (f a 1 = m) →  -- Point of tangency
  a + b = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2979_297918


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2979_297955

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that m + n = p + q implies a_m + a_n = a_p + a_q -/
def SufficientCondition (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q

/-- The property that a_m + a_n = a_p + a_q does not always imply m + n = p + q -/
def NotNecessaryCondition (a : ℕ → ℝ) : Prop :=
  ∃ m n p q : ℕ, a m + a n = a p + a q ∧ m + n ≠ p + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  ArithmeticSequence a →
  SufficientCondition a ∧ NotNecessaryCondition a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2979_297955


namespace NUMINAMATH_CALUDE_ralph_wild_animal_pictures_l2979_297934

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference between Derrick's and Ralph's picture count -/
def picture_difference : ℕ := 8

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := derrick_pictures - picture_difference

theorem ralph_wild_animal_pictures : ralph_pictures = 26 := by sorry

end NUMINAMATH_CALUDE_ralph_wild_animal_pictures_l2979_297934


namespace NUMINAMATH_CALUDE_inverse_inequality_iff_inequality_l2979_297922

theorem inverse_inequality_iff_inequality (a b : ℝ) (h : a * b > 0) :
  (1 / a < 1 / b) ↔ (a > b) := by sorry

end NUMINAMATH_CALUDE_inverse_inequality_iff_inequality_l2979_297922


namespace NUMINAMATH_CALUDE_quadratic_inequalities_solutions_l2979_297982

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x ≤ -5 ∨ x ≥ 2}
def solution_set2 : Set ℝ := {x | (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2}

-- State the theorem
theorem quadratic_inequalities_solutions :
  (∀ x : ℝ, x^2 + 3*x - 10 ≥ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x^2 - 3*x - 2 ≤ 0 ↔ x ∈ solution_set2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_solutions_l2979_297982


namespace NUMINAMATH_CALUDE_negation_equivalence_l2979_297936

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Teenager : U → Prop)
variable (Responsible : U → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ x, Teenager x ∧ ¬Responsible x) ↔ ¬(∀ x, Teenager x → Responsible x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2979_297936


namespace NUMINAMATH_CALUDE_characterization_of_satisfying_functions_l2979_297987

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2

/-- The main theorem stating the form of functions satisfying the equation -/
theorem characterization_of_satisfying_functions :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 := by
sorry

end NUMINAMATH_CALUDE_characterization_of_satisfying_functions_l2979_297987


namespace NUMINAMATH_CALUDE_equal_chords_implies_tangential_l2979_297967

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- Add necessary fields

/-- A circle -/
structure Circle where
  -- Add necessary fields

/-- Represents the property that a circle intersects each side of a quadrilateral at two points forming equal chords -/
def has_equal_chords_intersection (q : ConvexQuadrilateral) (c : Circle) : Prop :=
  sorry

/-- A quadrilateral is tangential if it has an inscribed circle -/
def is_tangential (q : ConvexQuadrilateral) : Prop :=
  ∃ c : Circle, sorry -- c is inscribed in q

/-- If a convex quadrilateral has the property that a circle intersects each of its sides 
    at two points forming equal chords, then the quadrilateral is tangential -/
theorem equal_chords_implies_tangential (q : ConvexQuadrilateral) (c : Circle) :
  has_equal_chords_intersection q c → is_tangential q :=
by
  sorry

end NUMINAMATH_CALUDE_equal_chords_implies_tangential_l2979_297967


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2979_297916

def polynomial (x : ℝ) : ℝ := (x^2 - 3*x + 2) * x * (x - 4)

theorem roots_of_polynomial :
  {x : ℝ | polynomial x = 0} = {0, 1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2979_297916


namespace NUMINAMATH_CALUDE_birds_in_tree_l2979_297995

theorem birds_in_tree (initial_birds final_birds : ℕ) (h1 : initial_birds = 29) (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2979_297995


namespace NUMINAMATH_CALUDE_prob_basket_A_given_white_l2979_297928

/-- Represents a basket with white and black balls -/
structure Basket where
  white : ℕ
  black : ℕ

/-- The probability of choosing a specific basket -/
def choose_probability : ℚ := 1/2

/-- Calculates the probability of picking a white ball from a given basket -/
def white_probability (b : Basket) : ℚ :=
  b.white / (b.white + b.black)

/-- Theorem: Probability of choosing Basket A given a white ball was picked -/
theorem prob_basket_A_given_white 
  (basket_A basket_B : Basket)
  (h_A : basket_A = ⟨2, 3⟩)
  (h_B : basket_B = ⟨1, 3⟩) :
  let p_A := choose_probability
  let p_B := choose_probability
  let p_W_A := white_probability basket_A
  let p_W_B := white_probability basket_B
  let p_W := p_A * p_W_A + p_B * p_W_B
  p_A * p_W_A / p_W = 8/13 := by
    sorry

end NUMINAMATH_CALUDE_prob_basket_A_given_white_l2979_297928


namespace NUMINAMATH_CALUDE_unique_k_for_coplanarity_l2979_297939

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (D - A) = a • (B - A) + b • (C - A) + c • (A - A)

-- State the theorem
theorem unique_k_for_coplanarity :
  ∃! k : ℝ, ∀ (A B C D : V),
    (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = 0) →
    coplanar A B C D :=
by sorry

end NUMINAMATH_CALUDE_unique_k_for_coplanarity_l2979_297939


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2979_297999

/-- A Mersenne number is of the form 2^n - 1 where n is a positive integer. -/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A Mersenne prime is a Mersenne number that is also prime. -/
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = mersenne_number n ∧ Prime p

/-- The largest Mersenne prime less than 500 is 127. -/
theorem largest_mersenne_prime_under_500 :
  (∀ p : ℕ, p < 500 → is_mersenne_prime p → p ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2979_297999


namespace NUMINAMATH_CALUDE_wednesday_rainfall_calculation_l2979_297920

/-- Calculates the rainfall on Wednesday given the conditions of the problem -/
def wednesday_rainfall (monday : ℝ) (tuesday_difference : ℝ) : ℝ :=
  2 * (monday + (monday - tuesday_difference))

/-- Theorem stating that given the specific conditions, Wednesday's rainfall is 2.2 inches -/
theorem wednesday_rainfall_calculation :
  wednesday_rainfall 0.9 0.7 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_rainfall_calculation_l2979_297920


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_7_l2979_297977

/-- The number of hour marks on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're interested in -/
def target_hour : ℕ := 7

/-- The angle between each hour mark on the clock -/
def hour_angle : ℕ := full_circle_degrees / clock_hours

/-- The smaller angle formed by the clock hands at 7 o'clock -/
def smaller_angle_at_7 : ℕ := target_hour * hour_angle

theorem clock_hands_angle_at_7 :
  smaller_angle_at_7 = 150 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_7_l2979_297977


namespace NUMINAMATH_CALUDE_small_cheese_slices_l2979_297921

/-- The number of slices in a pizza order --/
structure PizzaOrder where
  small_cheese : ℕ
  large_pepperoni : ℕ
  eaten_per_person : ℕ
  left_per_person : ℕ

/-- Theorem: Given the conditions, the small cheese pizza has 8 slices --/
theorem small_cheese_slices (order : PizzaOrder)
  (h1 : order.large_pepperoni = 14)
  (h2 : order.eaten_per_person = 9)
  (h3 : order.left_per_person = 2)
  : order.small_cheese = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_cheese_slices_l2979_297921


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l2979_297943

theorem rhombus_diagonal (side_length square_area rhombus_area diagonal1 diagonal2 : ℝ) :
  square_area = side_length * side_length →
  rhombus_area = square_area →
  rhombus_area = (diagonal1 * diagonal2) / 2 →
  side_length = 8 →
  diagonal1 = 16 →
  diagonal2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l2979_297943


namespace NUMINAMATH_CALUDE_f_max_value_l2979_297996

/-- The quadratic function f(x) = -3x^2 + 18x - 4 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 4

/-- The maximum value of f(x) is 77 -/
theorem f_max_value : ∃ (M : ℝ), M = 77 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l2979_297996


namespace NUMINAMATH_CALUDE_constant_polar_angle_forms_cone_l2979_297912

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPolarAngleSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Statement: The set of points with constant polar angle forms a cone
theorem constant_polar_angle_forms_cone (c : ℝ) :
  ∃ (cone : Set SphericalCoord), ConstantPolarAngleSet c = cone :=
sorry

end NUMINAMATH_CALUDE_constant_polar_angle_forms_cone_l2979_297912


namespace NUMINAMATH_CALUDE_modulus_one_minus_i_to_eight_l2979_297972

theorem modulus_one_minus_i_to_eight : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_modulus_one_minus_i_to_eight_l2979_297972


namespace NUMINAMATH_CALUDE_tom_payment_multiple_l2979_297953

def original_price : ℝ := 3.00
def tom_payment : ℝ := 9.00

theorem tom_payment_multiple : tom_payment / original_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_multiple_l2979_297953


namespace NUMINAMATH_CALUDE_abigail_cookies_l2979_297969

theorem abigail_cookies (grayson_boxes : ℚ) (olivia_boxes : ℕ) (cookies_per_box : ℕ) (total_cookies : ℕ) :
  grayson_boxes = 3/4 →
  olivia_boxes = 3 →
  cookies_per_box = 48 →
  total_cookies = 276 →
  (total_cookies - (grayson_boxes * cookies_per_box + olivia_boxes * cookies_per_box)) / cookies_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_abigail_cookies_l2979_297969


namespace NUMINAMATH_CALUDE_campers_third_week_l2979_297944

/-- Proves the number of campers in the third week given conditions about three consecutive weeks of camping. -/
theorem campers_third_week
  (total : ℕ)
  (second_week : ℕ)
  (h_total : total = 150)
  (h_second : second_week = 40)
  (h_difference : second_week = (second_week - 10) + 10) :
  total - (second_week - 10) - second_week = 80 :=
by sorry

end NUMINAMATH_CALUDE_campers_third_week_l2979_297944


namespace NUMINAMATH_CALUDE_combined_population_is_8000_l2979_297959

/-- The total population of five towns -/
def total_population : ℕ := 120000

/-- The population of Gordonia -/
def gordonia_population : ℕ := total_population / 3

/-- The population of Toadon -/
def toadon_population : ℕ := (gordonia_population * 3) / 4

/-- The population of Riverbank -/
def riverbank_population : ℕ := toadon_population + (toadon_population * 2) / 5

/-- The combined population of Lake Bright and Sunshine Hills -/
def lake_bright_sunshine_hills_population : ℕ := 
  total_population - (gordonia_population + toadon_population + riverbank_population)

theorem combined_population_is_8000 : 
  lake_bright_sunshine_hills_population = 8000 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_is_8000_l2979_297959


namespace NUMINAMATH_CALUDE_crease_lines_equivalence_l2979_297902

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the ellipse
structure Ellipse where
  focus1 : Point
  focus2 : Point
  majorAxis : ℝ

-- Define the set of points on crease lines
def CreaseLines (c : Circle) (a : Point) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ (a' : Point), a'.x^2 + a'.y^2 = c.radius^2 ∧ 
    (p.1 - (a.x + a'.x)/2)^2 + (p.2 - (a.y + a'.y)/2)^2 = ((a.x - a'.x)^2 + (a.y - a'.y)^2) / 4 }

-- Define the set of points not on the ellipse
def NotOnEllipse (e : Ellipse) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (Real.sqrt ((p.1 - e.focus1.x)^2 + (p.2 - e.focus1.y)^2) + 
                 Real.sqrt ((p.1 - e.focus2.x)^2 + (p.2 - e.focus2.y)^2)) ≠ e.majorAxis }

-- Theorem statement
theorem crease_lines_equivalence 
  (c : Circle) (a : Point) (e : Ellipse) 
  (h1 : (a.x - c.center.1)^2 + (a.y - c.center.2)^2 < c.radius^2)  -- A is inside the circle
  (h2 : e.focus1 = Point.mk c.center.1 c.center.2)  -- O is a focus of the ellipse
  (h3 : e.focus2 = a)  -- A is the other focus of the ellipse
  (h4 : e.majorAxis = c.radius) :  -- The major axis of the ellipse is R
  CreaseLines c a = NotOnEllipse e := by
  sorry

end NUMINAMATH_CALUDE_crease_lines_equivalence_l2979_297902


namespace NUMINAMATH_CALUDE_prob_heads_win_value_l2979_297942

/-- The probability of getting heads in a fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails in a fair coin flip -/
def p_tails : ℚ := 1/2

/-- The number of consecutive heads needed to win -/
def heads_to_win : ℕ := 6

/-- The number of consecutive tails needed to lose -/
def tails_to_lose : ℕ := 3

/-- The probability of encountering a run of 6 heads before a run of 3 tails 
    when repeatedly flipping a fair coin -/
def prob_heads_win : ℚ := 32/63

/-- Theorem stating that the probability of encountering a run of 6 heads 
    before a run of 3 tails when repeatedly flipping a fair coin is 32/63 -/
theorem prob_heads_win_value : 
  prob_heads_win = 32/63 :=
sorry

end NUMINAMATH_CALUDE_prob_heads_win_value_l2979_297942


namespace NUMINAMATH_CALUDE_max_abc_value_l2979_297905

theorem max_abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + c = (a + c) * (b + c)) :
  a * b * c ≤ 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l2979_297905


namespace NUMINAMATH_CALUDE_valid_four_digit_numbers_l2979_297986

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (∀ d : ℕ, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → is_prime d) ∧
  (∀ p : ℕ, p ∈ [n / 100, n / 10 % 100, n % 100] → is_prime p)

theorem valid_four_digit_numbers :
  {n : ℕ | is_valid_number n} = {2373, 3737, 5373, 7373} :=
by sorry

end NUMINAMATH_CALUDE_valid_four_digit_numbers_l2979_297986


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l2979_297975

theorem cara_seating_arrangements (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l2979_297975


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2979_297903

theorem inequality_solution_set (x : ℝ) :
  (-x^2 + 3*x - 2 > 0) ↔ (1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2979_297903


namespace NUMINAMATH_CALUDE_tissues_per_box_l2979_297926

theorem tissues_per_box (boxes : ℕ) (used : ℕ) (left : ℕ) : 
  boxes = 3 → used = 210 → left = 270 → (used + left) / boxes = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_tissues_per_box_l2979_297926


namespace NUMINAMATH_CALUDE_point_on_line_not_perpendicular_to_y_axis_l2979_297908

-- Define a line l with equation x + my - 2 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y - 2 = 0

-- Theorem stating that (2,0) always lies on line l
theorem point_on_line (m : ℝ) : line_l m 2 0 := by sorry

-- Theorem stating that line l is not perpendicular to the y-axis
theorem not_perpendicular_to_y_axis (m : ℝ) : m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_point_on_line_not_perpendicular_to_y_axis_l2979_297908


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l2979_297938

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b : ℕ, b < a → (Nat.gcd b 72 = 1 ∨ Nat.gcd b 45 = 1)) ∧ 
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 45 > 1 → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l2979_297938


namespace NUMINAMATH_CALUDE_function_value_implies_input_l2979_297966

/-- Given a function f(x) = (2x + 1) / (x - 1) and f(p) = 4, prove that p = 5/2 -/
theorem function_value_implies_input (f : ℝ → ℝ) (p : ℝ) 
  (h1 : ∀ x, x ≠ 1 → f x = (2 * x + 1) / (x - 1))
  (h2 : f p = 4) :
  p = 5/2 := by
sorry

end NUMINAMATH_CALUDE_function_value_implies_input_l2979_297966


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_660_l2979_297963

theorem consecutive_numbers_with_lcm_660 (a b c : ℕ) : 
  b = a + 1 ∧ c = b + 1 ∧ Nat.lcm (Nat.lcm a b) c = 660 → 
  a = 10 ∧ b = 11 ∧ c = 12 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_lcm_660_l2979_297963


namespace NUMINAMATH_CALUDE_function_upper_bound_l2979_297904

/-- Given a function f(x) = ax - x ln x - a, prove that if f(x) ≤ 0 for all x ≥ 2, 
    then a ≤ 2ln 2 -/
theorem function_upper_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → a * x - x * Real.log x - a ≤ 0) → 
  a ≤ 2 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2979_297904


namespace NUMINAMATH_CALUDE_abc_sum_product_l2979_297915

theorem abc_sum_product (x : ℝ) : ∃ a b c : ℝ, a + b + c = 1 ∧ a * b + a * c + b * c = x := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_l2979_297915


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2979_297960

theorem min_value_complex_expression (p q r : ℤ) (ξ : ℂ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 5 ∧ 
    (∀ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) ≥ m) ∧
    (∃ (p' q' r' : ℤ) (h_distinct' : p' ≠ q' ∧ q' ≠ r' ∧ p' ≠ r'),
      Complex.abs (p' + q' * ξ + r' * ξ^3) = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2979_297960


namespace NUMINAMATH_CALUDE_sneakers_cost_proof_l2979_297993

/-- The cost of a wallet in dollars -/
def wallet_cost : ℝ := 50

/-- The cost of a backpack in dollars -/
def backpack_cost : ℝ := 100

/-- The cost of a pair of jeans in dollars -/
def jeans_cost : ℝ := 50

/-- The total amount spent by Leonard and Michael in dollars -/
def total_spent : ℝ := 450

/-- The number of pairs of sneakers bought -/
def num_sneakers : ℕ := 2

/-- The number of pairs of jeans bought -/
def num_jeans : ℕ := 2

/-- The cost of each pair of sneakers in dollars -/
def sneakers_cost : ℝ := 100

theorem sneakers_cost_proof :
  wallet_cost + num_sneakers * sneakers_cost + backpack_cost + num_jeans * jeans_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sneakers_cost_proof_l2979_297993


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l2979_297940

/-- The total sleep John got in a week given specific sleep patterns --/
def totalSleepInWeek (daysWithLowSleep : ℕ) (hoursLowSleep : ℕ) 
  (recommendedSleep : ℕ) (percentageNormalSleep : ℚ) : ℚ :=
  (daysWithLowSleep * hoursLowSleep : ℚ) + 
  ((7 - daysWithLowSleep) * (recommendedSleep * percentageNormalSleep))

/-- Theorem stating that John's total sleep for the week is 30 hours --/
theorem johns_weekly_sleep : 
  totalSleepInWeek 2 3 8 (60 / 100) = 30 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_sleep_l2979_297940


namespace NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l2979_297937

/-- The probability of success for each event -/
def p : ℝ := 0.7

/-- The probability of at least one success in two independent events -/
def prob_at_least_one (p : ℝ) : ℝ := 1 - (1 - p) * (1 - p)

/-- Theorem stating that the probability of at least one success is 0.91 -/
theorem prob_at_least_one_is_correct : prob_at_least_one p = 0.91 := by
  sorry

#eval prob_at_least_one p

end NUMINAMATH_CALUDE_prob_at_least_one_is_correct_l2979_297937


namespace NUMINAMATH_CALUDE_dog_human_years_ratio_l2979_297957

theorem dog_human_years_ratio : 
  (∀ (dog_age human_age : ℝ), dog_age = 7 * human_age) → 
  (∃ (x : ℝ), x * 3 = 21 ∧ 7 / x = 7 / 6) :=
by sorry

end NUMINAMATH_CALUDE_dog_human_years_ratio_l2979_297957


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2979_297971

theorem distance_from_origin_to_point : Real.sqrt ((-12)^2 + 9^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2979_297971


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l2979_297906

theorem product_of_five_consecutive_integers (n : ℕ) : 
  n = 3 → (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l2979_297906


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2979_297970

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles (n : ℕ) : ℕ := (n.choose 2) ^ 2

/-- Theorem: The number of different rectangles with sides parallel to the grid
    that can be formed by connecting four dots in a 5x5 square array of dots is 100. -/
theorem rectangles_in_5x5_grid :
  num_rectangles 5 = 100 := by
  sorry

#eval num_rectangles 5  -- Should output 100

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2979_297970


namespace NUMINAMATH_CALUDE_perpendicular_planes_line_l2979_297917

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_line 
  (a : Line) (α β : Plane) (l : Line)
  (h1 : perp_planes α β)
  (h2 : l = intersect α β)
  (h3 : perp_line_plane a β) :
  subset a α ∧ perp_lines a l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_line_l2979_297917


namespace NUMINAMATH_CALUDE_baker_cakes_l2979_297998

theorem baker_cakes (initial_cakes : ℕ) : 
  (initial_cakes - 75 + 76 = 111) → initial_cakes = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_l2979_297998


namespace NUMINAMATH_CALUDE_line_points_l2979_297901

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨1, 2⟩
  let p3 : Point := ⟨3, 6⟩
  let p4 : Point := ⟨2, 4⟩
  let p5 : Point := ⟨5, 10⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 ∧ collinear p1 p2 p5 := by
  sorry

end NUMINAMATH_CALUDE_line_points_l2979_297901


namespace NUMINAMATH_CALUDE_decimal_division_equals_forty_l2979_297981

theorem decimal_division_equals_forty : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_equals_forty_l2979_297981


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2979_297941

/-- Given z = 1 + i and (z^2 + az + b) / (z^2 - z + 1) = 1 - i, where a and b are real numbers, 
    then a = -1 and b = 2. -/
theorem complex_fraction_equality (a b : ℝ) : 
  let z : ℂ := 1 + I
  ((z^2 + a*z + b) / (z^2 - z + 1) = 1 - I) → (a = -1 ∧ b = 2) := by
sorry


end NUMINAMATH_CALUDE_complex_fraction_equality_l2979_297941


namespace NUMINAMATH_CALUDE_mean_temperature_is_84_l2979_297985

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length : ℚ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_84_l2979_297985


namespace NUMINAMATH_CALUDE_square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l2979_297968

theorem square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven
  (a b : ℝ) (h : a^2 - 3*b = 5) : 2*a^2 - 6*b + 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_triple_eq_five_implies_double_square_minus_sextuple_plus_one_eq_eleven_l2979_297968


namespace NUMINAMATH_CALUDE_complex_cube_root_problem_l2979_297923

theorem complex_cube_root_problem : ∃ (c : ℤ), (1 + 3*I : ℂ)^3 = -26 + c*I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_problem_l2979_297923


namespace NUMINAMATH_CALUDE_cross_pentominoes_fit_on_chessboard_l2979_297954

/-- A "cross" pentomino consists of 5 unit squares -/
def cross_pentomino_area : ℝ := 5

/-- The chessboard is 8x8 units -/
def chessboard_side : ℝ := 8

/-- The number of cross pentominoes to be cut -/
def num_crosses : ℕ := 9

/-- The area of half-rectangles between crosses -/
def half_rectangle_area : ℝ := 1

/-- The number of half-rectangles -/
def num_half_rectangles : ℕ := 8

/-- The maximum area of corner pieces -/
def max_corner_piece_area : ℝ := 1.5

/-- The number of corner pieces -/
def num_corner_pieces : ℕ := 4

theorem cross_pentominoes_fit_on_chessboard :
  (num_crosses : ℝ) * cross_pentomino_area +
  (num_half_rectangles : ℝ) * half_rectangle_area +
  (num_corner_pieces : ℝ) * max_corner_piece_area ≤ chessboard_side ^ 2 :=
sorry

end NUMINAMATH_CALUDE_cross_pentominoes_fit_on_chessboard_l2979_297954


namespace NUMINAMATH_CALUDE_parabola_properties_l2979_297965

/-- Parabola with vertex at origin and focus at (1,0) -/
structure Parabola where
  vertex : ℝ × ℝ := (0, 0)
  focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus of the parabola -/
structure Line (p : Parabola) where
  slope : ℝ

/-- Intersection points of the line with the parabola -/
def intersection_points (p : Parabola) (l : Line p) : Set (ℝ × ℝ) :=
  sorry

/-- Area of triangle formed by origin, focus, and two intersection points -/
def triangle_area (p : Parabola) (l : Line p) : ℝ :=
  sorry

theorem parabola_properties (p : Parabola) :
  (∀ x y : ℝ, (x, y) ∈ {(x, y) | y^2 = 4*x}) ∧
  (∃ min_area : ℝ, min_area = 2 ∧ 
    ∀ l : Line p, triangle_area p l ≥ min_area) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2979_297965


namespace NUMINAMATH_CALUDE_range_of_m_l2979_297927

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 5
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, p x ∧ ¬(q x m)) →
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2979_297927


namespace NUMINAMATH_CALUDE_medication_frequency_l2979_297980

/-- The number of times Kara takes her medication per day -/
def medication_times_per_day : ℕ := sorry

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of days Kara followed her medication schedule -/
def days_followed : ℕ := 14

/-- The number of doses Kara missed in the two-week period -/
def doses_missed : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks in ounces -/
def total_water_consumed : ℕ := 160

theorem medication_frequency :
  medication_times_per_day = 3 :=
by
  have h1 : water_per_dose * (days_followed * medication_times_per_day - doses_missed) = total_water_consumed := sorry
  sorry

end NUMINAMATH_CALUDE_medication_frequency_l2979_297980


namespace NUMINAMATH_CALUDE_existence_of_opposite_colors_l2979_297945

/-- Represents a piece on the circle -/
inductive Piece
| White
| Black

/-- Represents the circle with pieces placed on it -/
structure Circle :=
  (pieces : Fin 40 → Piece)
  (white_count : Nat)
  (black_count : Nat)
  (white_count_eq : white_count = 25)
  (black_count_eq : black_count = 15)
  (total_count : white_count + black_count = 40)

/-- Two points are diametrically opposite if their indices differ by 20 (mod 40) -/
def diametricallyOpposite (i j : Fin 40) : Prop :=
  (i.val + 20) % 40 = j.val ∨ (j.val + 20) % 40 = i.val

/-- Main theorem: There exist diametrically opposite white and black pieces -/
theorem existence_of_opposite_colors (c : Circle) :
  ∃ (i j : Fin 40), diametricallyOpposite i j ∧ 
    c.pieces i = Piece.White ∧ c.pieces j = Piece.Black :=
sorry

end NUMINAMATH_CALUDE_existence_of_opposite_colors_l2979_297945


namespace NUMINAMATH_CALUDE_r_daily_earnings_l2979_297914

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 60 per day. -/
theorem r_daily_earnings (P Q R : ℚ) 
  (h1 : P + Q + R = 190) 
  (h2 : P + R = 120)
  (h3 : Q + R = 130) : 
  R = 60 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earnings_l2979_297914


namespace NUMINAMATH_CALUDE_divisibility_property_l2979_297948

theorem divisibility_property (A B : ℤ) 
  (h : ∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (A + B) % k = 0) : 
  ((A + B) % 66 = 0) ∧ ¬(∀ C D : ℤ, (∀ k : ℤ, 1 ≤ k ∧ k ≤ 65 → (C + D) % k = 0) → (C + D) % 67 = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l2979_297948


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l2979_297909

/-- A coloring of vertices using three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Checks if four vertices form an isosceles trapezoid in a regular n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop :=
  sorry

/-- Checks if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n),
    c v1 = c v2 ∧ c v2 = c v3 ∧ c v3 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l2979_297909


namespace NUMINAMATH_CALUDE_school_trip_buses_l2979_297935

/-- The number of buses for a school trip, given the number of supervisors per bus and the total number of supervisors. -/
def number_of_buses (supervisors_per_bus : ℕ) (total_supervisors : ℕ) : ℕ :=
  total_supervisors / supervisors_per_bus

/-- Theorem stating that the number of buses is 7, given the conditions from the problem. -/
theorem school_trip_buses : number_of_buses 3 21 = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_buses_l2979_297935


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l2979_297976

theorem triangle_cosine_theorem (a b c : ℝ) (h1 : b^2 = a*c) (h2 : c = 2*a) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C = -Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l2979_297976


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2979_297984

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k : ℕ, a k > 0) →  -- Positive sequence
  (∃ q : ℝ, q > 0 ∧ ∀ k : ℕ, a (k + 1) = q * a k) →  -- Geometric sequence
  a 2018 = a 2017 + 2 * a 2016 →  -- Given condition
  (a m * a n = 16 * (a 1)^2) →  -- Derived from √(a_m * a_n) = 4a_1
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ a i * a j = 16 * (a 1)^2 → 1/i + 5/j ≥ 7/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2979_297984


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2979_297949

theorem sum_of_reciprocals (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x - y) = a)
  (h2 : x * z / (x - z) = b)
  (h3 : y * z / (y - z) = c) :
  1 / x + 1 / y + 1 / z = (1 / a + 1 / b + 1 / c) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2979_297949


namespace NUMINAMATH_CALUDE_song_book_cost_l2979_297979

/-- The cost of the song book given the total amount spent and the cost of the trumpet -/
theorem song_book_cost (total_spent : ℚ) (trumpet_cost : ℚ) (h1 : total_spent = 151) (h2 : trumpet_cost = 145.16) :
  total_spent - trumpet_cost = 5.84 := by
  sorry

end NUMINAMATH_CALUDE_song_book_cost_l2979_297979


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l2979_297946

/-- Given a right triangular prism where:
    - The lateral edge is equal to the height of its base
    - The area of the cross-section passing through this lateral edge and the height of the base is Q
    Prove that the volume of the prism is Q √(3Q) -/
theorem right_triangular_prism_volume (Q : ℝ) (Q_pos : Q > 0) :
  ∃ (V : ℝ), V = Q * Real.sqrt (3 * Q) ∧
  (∃ (a h : ℝ) (a_pos : a > 0) (h_pos : h > 0),
    h = a * Real.sqrt 5 / 2 ∧
    Q = a * Real.sqrt 5 / 2 * h ∧
    V = Real.sqrt 3 / 4 * a^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l2979_297946


namespace NUMINAMATH_CALUDE_log_x2y2_value_l2979_297925

theorem log_x2y2_value (x y : ℝ) (hxy4 : Real.log (x * y^4) = 1) (hx3y : Real.log (x^3 * y) = 1) :
  Real.log (x^2 * y^2) = 10/11 := by
sorry

end NUMINAMATH_CALUDE_log_x2y2_value_l2979_297925


namespace NUMINAMATH_CALUDE_decimal_as_fraction_l2979_297994

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.73864864864

/-- The denominator of the target fraction -/
def denominator : ℕ := 999900

/-- The theorem stating that our decimal equals the target fraction -/
theorem decimal_as_fraction : decimal = 737910 / denominator := by sorry

end NUMINAMATH_CALUDE_decimal_as_fraction_l2979_297994


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2979_297950

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2979_297950


namespace NUMINAMATH_CALUDE_min_value_expression_l2979_297907

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (Real.sqrt 10 - 1)^2 ∧
  (∃ (a b c : ℝ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
    (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = 4 * (Real.sqrt 10 - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2979_297907


namespace NUMINAMATH_CALUDE_expression_simplification_l2979_297973

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a^2 - 6*a + 9) / (a^2 - 2*a) / (1 - 1/(a - 2)) = 2020 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2979_297973


namespace NUMINAMATH_CALUDE_circle_C_equation_and_OP_not_parallel_AB_l2979_297956

-- Define the circle M
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define circle C
def circle_C (r : ℝ) (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = r^2

-- Define point P
def point_P : ℝ × ℝ := (1, 1)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the slope of line OP
def slope_OP : ℝ := 1

-- Define the slope of line AB
def slope_AB : ℝ := 0

theorem circle_C_equation_and_OP_not_parallel_AB (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, circle_C r x y ↔ (x - 2)^2 + (y - 2)^2 = r^2) ∧ 
  slope_OP ≠ slope_AB :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_and_OP_not_parallel_AB_l2979_297956


namespace NUMINAMATH_CALUDE_bridge_length_l2979_297913

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 140)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 235 :=
sorry

end NUMINAMATH_CALUDE_bridge_length_l2979_297913


namespace NUMINAMATH_CALUDE_absolute_value_of_one_plus_i_squared_l2979_297991

theorem absolute_value_of_one_plus_i_squared : Complex.abs ((1 : ℂ) + Complex.I) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_one_plus_i_squared_l2979_297991


namespace NUMINAMATH_CALUDE_solution_set_2x_plus_y_eq_9_l2979_297952

theorem solution_set_2x_plus_y_eq_9 :
  {(x, y) : ℕ × ℕ | 2 * x + y = 9} = {(0, 9), (1, 7), (2, 5), (3, 3), (4, 1)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_2x_plus_y_eq_9_l2979_297952


namespace NUMINAMATH_CALUDE_cooper_savings_l2979_297932

/-- Calculates the total savings for a given daily savings amount and number of days. -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Proves that saving $34 daily for 365 days results in a total savings of $12,410. -/
theorem cooper_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end NUMINAMATH_CALUDE_cooper_savings_l2979_297932
