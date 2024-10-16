import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l4_417

theorem percentage_problem (p : ℝ) : 
  (p / 100) * 180 - (1 / 3) * ((p / 100) * 180) = 18 ↔ p = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l4_417


namespace NUMINAMATH_CALUDE_triangle_properties_l4_431

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b * Real.cos A + a * Real.cos B = 2 * c * Real.cos A →
  D.1 + D.2 = 1 →
  D.1 = 3 * D.2 →
  (D.1 * a + D.2 * b)^2 + (D.1 * c)^2 - 2 * (D.1 * a + D.2 * b) * (D.1 * c) * Real.cos A = 9 →
  A = π / 3 ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    (D.1 * a' + D.2 * b')^2 + (D.1 * c')^2 - 2 * (D.1 * a' + D.2 * b') * (D.1 * c') * Real.cos A = 9 →
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4_431


namespace NUMINAMATH_CALUDE_number_of_children_l4_405

theorem number_of_children (total : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 42 → 
  children = 2 * adults → 
  total = adults + children →
  children = 28 := by
sorry

end NUMINAMATH_CALUDE_number_of_children_l4_405


namespace NUMINAMATH_CALUDE_tees_per_member_l4_499

/-- The number of people in Bill's golfing group -/
def group_size : ℕ := 4

/-- The number of tees in a generic package -/
def generic_package_size : ℕ := 12

/-- The number of tees in an aero flight package -/
def aero_package_size : ℕ := 2

/-- The maximum number of generic packages Bill can buy -/
def max_generic_packages : ℕ := 2

/-- The number of aero flight packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- Theorem stating that the number of golf tees per member is 20 -/
theorem tees_per_member :
  (max_generic_packages * generic_package_size + aero_packages * aero_package_size) / group_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_tees_per_member_l4_499


namespace NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l4_494

theorem quadratic_trinomial_factorization (p q : ℝ) (x : ℝ) :
  x^2 + (p + q)*x + p*q = (x + p)*(x + q) := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l4_494


namespace NUMINAMATH_CALUDE_function_properties_l4_444

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 / 4 - a / (x^2) - 1 / x

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    -- The tangent at (1, f(1)) is perpendicular to y = (1/2)x
    f_derivative a 1 = -2 ∧
    -- a = 5/4
    a = 5 / 4 ∧
    -- f(x) is decreasing on (0, 5) and increasing on (5, +∞)
    (∀ x ∈ Set.Ioo 0 5, f_derivative a x < 0) ∧
    (∀ x ∈ Set.Ioi 5, f_derivative a x > 0) ∧
    -- The minimum value of f(x) is -ln(5) at x = 5
    (∀ x > 0, f a x ≥ f a 5) ∧
    f a 5 = -Real.log 5 :=
by
  sorry

end

end NUMINAMATH_CALUDE_function_properties_l4_444


namespace NUMINAMATH_CALUDE_triangle_problem_l4_412

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that B = π/3 and AD = (2√13)/3 under certain conditions. -/
theorem triangle_problem (A B C : Real) (a b c : Real) (D : Real) :
  0 < A ∧ A < π/2 →  -- Triangle ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.sin A = a * Real.cos (B - π/6) →  -- Given condition
  b = Real.sqrt 13 →  -- Given condition
  a = 4 →  -- Given condition
  0 ≤ D ∧ D ≤ c →  -- D is on AC
  (1/2) * a * D * Real.sin B = 2 * Real.sqrt 3 →  -- Area of ABD
  B = π/3 ∧ D = (2 * Real.sqrt 13) / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l4_412


namespace NUMINAMATH_CALUDE_find_a_value_l4_489

-- Define the variables
variable (x y z a : ℚ)

-- Define the theorem
theorem find_a_value (h1 : y = 4 * x) (h2 : z = 5 * x) (h3 : y = 15 * a - 5) (h4 : y = 60) : a = 13/3 := by
  sorry


end NUMINAMATH_CALUDE_find_a_value_l4_489


namespace NUMINAMATH_CALUDE_adam_cat_food_packages_l4_475

/-- The number of packages of cat food Adam bought -/
def cat_food_packages : ℕ := sorry

/-- The number of packages of dog food Adam bought -/
def dog_food_packages : ℕ := 7

/-- The number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := 10

/-- The number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- The difference between the number of cans of cat food and dog food -/
def cans_difference : ℕ := 55

theorem adam_cat_food_packages : 
  cat_food_packages * cans_per_cat_package = 
  dog_food_packages * cans_per_dog_package + cans_difference ∧
  cat_food_packages = 9 := by sorry

end NUMINAMATH_CALUDE_adam_cat_food_packages_l4_475


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4_466

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ gcd n d = 1 ∧ (n : ℚ) / d = 0.4 + (36 : ℚ) / 99 :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l4_466


namespace NUMINAMATH_CALUDE_complex_number_theorem_l4_424

-- Define the complex number z
variable (z : ℂ)

-- Define the real number m
variable (m : ℝ)

-- State the theorem
theorem complex_number_theorem (h1 : (z + 2*I).im = 0) 
                                (h2 : (z / (2 - I)).im = 0)
                                (h3 : ((z + m*I)^2).re > 0)
                                (h4 : ((z + m*I)^2).im > 0) :
  z = 4 - 2*I ∧ 2 < m ∧ m < 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l4_424


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l4_429

theorem negative_fraction_comparison : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l4_429


namespace NUMINAMATH_CALUDE_meaningful_fraction_iff_x_gt_three_l4_458

theorem meaningful_fraction_iff_x_gt_three (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_iff_x_gt_three_l4_458


namespace NUMINAMATH_CALUDE_smallest_a_value_l4_460

/-- The smallest possible value of a given the conditions -/
theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 2 * Real.pi - 17 ∧ ∃ (a₀ : ℝ), a₀ = 2 * Real.pi - 17 ∧ 
  (∃ b₀ : ℝ, 0 ≤ b₀ ∧ ∀ x : ℤ, Real.sin (a₀ * ↑x + b₀) = Real.sin (17 * ↑x)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l4_460


namespace NUMINAMATH_CALUDE_nori_crayons_l4_467

theorem nori_crayons (initial_boxes : ℕ) (crayons_per_box : ℕ) (crayons_left : ℕ) (extra_to_lea : ℕ) :
  initial_boxes = 4 →
  crayons_per_box = 8 →
  crayons_left = 15 →
  extra_to_lea = 7 →
  ∃ (crayons_to_mae : ℕ),
    initial_boxes * crayons_per_box = crayons_left + crayons_to_mae + (crayons_to_mae + extra_to_lea) ∧
    crayons_to_mae = 5 :=
by sorry

end NUMINAMATH_CALUDE_nori_crayons_l4_467


namespace NUMINAMATH_CALUDE_interest_rate_proof_l4_493

def simple_interest (P r t : ℝ) : ℝ := P * (1 + r * t)

theorem interest_rate_proof (P : ℝ) (h1 : P > 0) :
  ∃ r : ℝ, r > 0 ∧ simple_interest P r 2 = 100 ∧ simple_interest P r 6 = 200 →
  r = 0.5 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l4_493


namespace NUMINAMATH_CALUDE_companion_numbers_example_companion_numbers_expression_l4_495

/-- Two numbers are companion numbers if their sum equals their product. -/
def CompanionNumbers (a b : ℝ) : Prop := a + b = a * b

theorem companion_numbers_example : CompanionNumbers (-1) (1/2) := by sorry

theorem companion_numbers_expression (m n : ℝ) (h : CompanionNumbers m n) :
  -2 * m * n + 1/2 * (3 * m + 2 * (1/2 * n - m) + 3 * m * n - 6) = -3 := by sorry

end NUMINAMATH_CALUDE_companion_numbers_example_companion_numbers_expression_l4_495


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l4_409

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l4_409


namespace NUMINAMATH_CALUDE_fraction_equality_l4_457

theorem fraction_equality (x : ℝ) : (2 + x) / (4 + x) = (3 + x) / (7 + x) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4_457


namespace NUMINAMATH_CALUDE_find_other_number_l4_483

theorem find_other_number (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 21)
  (h3 : a = 210) :
  b = 462 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l4_483


namespace NUMINAMATH_CALUDE_min_difference_gcd3_lcm135_l4_484

def min_difference_with_gcd_lcm : ℕ → ℕ → ℕ → ℕ → ℕ := sorry

theorem min_difference_gcd3_lcm135 :
  min_difference_with_gcd_lcm 3 135 = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_gcd3_lcm135_l4_484


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l4_435

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l4_435


namespace NUMINAMATH_CALUDE_abs_minus_three_minus_three_eq_zero_l4_449

theorem abs_minus_three_minus_three_eq_zero : |(-3 : ℤ)| - 3 = 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_three_minus_three_eq_zero_l4_449


namespace NUMINAMATH_CALUDE_race_initial_distance_l4_488

/-- The initial distance between two speed walkers in a race --/
def initial_distance (john_speed steve_speed : ℝ) (duration : ℝ) (final_lead : ℝ) : ℝ :=
  john_speed * duration - (steve_speed * duration + final_lead)

/-- Theorem stating the initial distance between John and Steve --/
theorem race_initial_distance :
  initial_distance 4.2 3.7 34 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_race_initial_distance_l4_488


namespace NUMINAMATH_CALUDE_common_point_polar_coords_l4_450

-- Define the circle O in polar coordinates
def circle_O (ρ θ : ℝ) : Prop := ρ = Real.cos θ + Real.sin θ

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2 / 2

-- Theorem statement
theorem common_point_polar_coords :
  ∃ (ρ θ : ℝ), 
    circle_O ρ θ ∧ 
    line_l ρ θ ∧ 
    0 < θ ∧ 
    θ < Real.pi ∧ 
    ρ = 1 ∧ 
    θ = Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_common_point_polar_coords_l4_450


namespace NUMINAMATH_CALUDE_age_difference_l4_486

/-- Given information about Jacob and Michael's ages, prove their current age difference -/
theorem age_difference (jacob_current : ℕ) (michael_current : ℕ) : 
  (jacob_current + 4 = 13) → 
  (michael_current + 3 = 2 * (jacob_current + 3)) →
  (michael_current - jacob_current = 12) := by
sorry

end NUMINAMATH_CALUDE_age_difference_l4_486


namespace NUMINAMATH_CALUDE_parabola_point_relation_l4_406

theorem parabola_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = x₁^2 - 4*x₁ + 3 →
  y₂ = x₂^2 - 4*x₂ + 3 →
  x₁ > x₂ →
  x₂ > 2 →
  y₁ > y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l4_406


namespace NUMINAMATH_CALUDE_inverse_proportion_l4_439

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 54) (h3 : x = 3 * y) :
  ∃ (y' : ℝ), 5 * y' = k ∧ y' = 109.35 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l4_439


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l4_425

def is_valid_rearrangement (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ),
    (∀ i, i < n → f i < n) ∧
    (∀ i, i < n → f i ≠ i) ∧
    (∀ i, i < n → (f i + 2) % n = f ((i + 1) % n) ∨ (f i + n - 2) % n = f ((i + 1) % n))

theorem rearrangement_theorem :
  (is_valid_rearrangement 15 = false) ∧ (is_valid_rearrangement 20 = true) :=
sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l4_425


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l4_461

def M (m : ℤ) : Set ℤ := {m, -3}

def N : Set ℤ := {x : ℤ | 2*x^2 + 7*x + 3 < 0}

theorem intersection_implies_m_value (m : ℤ) :
  (M m ∩ N).Nonempty → m = -2 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l4_461


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l4_430

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_coefficient (f : QuadraticFunction) 
  (vertex_x : f.eval 2 = 3)
  (point : f.eval 0 = 7) : 
  f.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l4_430


namespace NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l4_419

-- Define the set M
def M : Set ℝ := {x | |2*x - 1| < 1}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: Inequality for elements in M
theorem inequality_for_M_elements (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  a * b + 1 > a + b := by sorry

end NUMINAMATH_CALUDE_characterization_of_M_inequality_for_M_elements_l4_419


namespace NUMINAMATH_CALUDE_algebraic_expression_theorem_l4_476

-- Define the algebraic expression
def algebraic_expression (a b x : ℝ) : ℝ :=
  (a*x - 3) * (2*x + 4) - x^2 - b

-- Define the condition for no x^2 term
def no_x_squared_term (a : ℝ) : Prop :=
  2*a - 1 = 0

-- Define the condition for no constant term
def no_constant_term (b : ℝ) : Prop :=
  -12 - b = 0

-- Define the final expression to be calculated
def final_expression (a b : ℝ) : ℝ :=
  (2*a + b)^2 - (2 - 2*b)*(2 + 2*b) - 3*a*(a - b)

-- Theorem statement
theorem algebraic_expression_theorem (a b : ℝ) :
  no_x_squared_term a ∧ no_constant_term b →
  a = 1/2 ∧ b = -12 ∧ final_expression a b = 678 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_theorem_l4_476


namespace NUMINAMATH_CALUDE_total_wall_length_divisible_by_four_l4_445

/-- Represents a partition of a square room into smaller square rooms -/
structure RoomPartition where
  size : ℕ  -- Size of the original square room
  partitions : List (ℕ × ℕ × ℕ)  -- List of (x, y, size) for each smaller room

/-- The sum of all partition wall lengths in a room partition -/
def totalWallLength (rp : RoomPartition) : ℕ :=
  sorry

/-- Theorem: The total wall length of any valid room partition is divisible by 4 -/
theorem total_wall_length_divisible_by_four (rp : RoomPartition) :
  4 ∣ totalWallLength rp :=
sorry

end NUMINAMATH_CALUDE_total_wall_length_divisible_by_four_l4_445


namespace NUMINAMATH_CALUDE_certain_number_proof_l4_420

theorem certain_number_proof (x y C : ℝ) : 
  (2 * x - y = C) → (6 * x - 3 * y = 12) → C = 4 := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4_420


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l4_403

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l4_403


namespace NUMINAMATH_CALUDE_f_symmetric_l4_492

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x + 1

theorem f_symmetric (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_l4_492


namespace NUMINAMATH_CALUDE_parabola_translation_l4_485

/-- Given a parabola y = x^2 + bx + c that is translated 3 units right and 4 units down
    to become y = x^2 - 2x + 2, prove that b = 4 and c = 9 -/
theorem parabola_translation (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 + b*x + c ↔ y + 4 = (x - 3)^2 - 2*(x - 3) + 2) →
  b = 4 ∧ c = 9 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4_485


namespace NUMINAMATH_CALUDE_equation_solution_l4_471

theorem equation_solution (x : ℝ) : 
  (21 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 5 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4_471


namespace NUMINAMATH_CALUDE_goats_and_hens_total_amount_l4_433

/-- The total amount spent on goats and hens -/
def total_amount (num_goats num_hens goat_price hen_price : ℕ) : ℕ :=
  num_goats * goat_price + num_hens * hen_price

/-- Theorem: The total amount spent on 5 goats at Rs. 400 each and 10 hens at Rs. 50 each is Rs. 2500 -/
theorem goats_and_hens_total_amount :
  total_amount 5 10 400 50 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_goats_and_hens_total_amount_l4_433


namespace NUMINAMATH_CALUDE_trig_expression_equality_l4_423

theorem trig_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 
  Real.sin (5 * π / 180) / Real.sin (8 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l4_423


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4_496

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4_496


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4_421

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h2 : seq.a 1 * seq.a 2 * seq.a 3 = 80) :
  seq.a 11 + seq.a 12 + seq.a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4_421


namespace NUMINAMATH_CALUDE_no_integer_solution_l4_401

theorem no_integer_solution :
  ¬ ∃ (A B C : ℤ),
    (A - B = 1620) ∧
    ((75 : ℚ) / 1000 * A = (125 : ℚ) / 1000 * B) ∧
    (A + B = (1 : ℚ) / 2 * C^4) ∧
    (A^2 + B^2 = C^2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l4_401


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4_427

theorem trigonometric_identity :
  Real.cos (17 * π / 180) * Real.sin (43 * π / 180) +
  Real.sin (163 * π / 180) * Real.sin (47 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4_427


namespace NUMINAMATH_CALUDE_polygon_triangulation_l4_459

/-- Theorem: For any polygon with n sides divided into k triangles, k ≥ n - 2 -/
theorem polygon_triangulation (n k : ℕ) (h_n : n ≥ 3) (h_k : k > 0) : k ≥ n - 2 := by
  sorry


end NUMINAMATH_CALUDE_polygon_triangulation_l4_459


namespace NUMINAMATH_CALUDE_line_points_k_value_l4_410

/-- A line contains the points (4, 10), (-4, k), and (-12, 6). Prove that k = 8. -/
theorem line_points_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 4 + b) ∧ 
    (k = m * (-4) + b) ∧ 
    (6 = m * (-12) + b)) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l4_410


namespace NUMINAMATH_CALUDE_f_min_max_l4_432

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem f_min_max :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 0) ∧
  (∀ (y : ℝ), y ∈ domain → f y ≥ 0) ∧
  (∃ (z : ℝ), z ∈ domain ∧ f z = 4) ∧
  (∀ (w : ℝ), w ∈ domain → f w ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_f_min_max_l4_432


namespace NUMINAMATH_CALUDE_richsWalkDistance_l4_415

/-- Calculates the total distance Rich walks given his walking pattern --/
def richsWalk (houseToSidewalk : ℕ) (sidewalkToRoadEnd : ℕ) : ℕ :=
  let initialDistance := houseToSidewalk + sidewalkToRoadEnd
  let toIntersection := initialDistance * 2
  let toEndOfRoute := (initialDistance + toIntersection) / 2
  let oneWayDistance := initialDistance + toIntersection + toEndOfRoute
  oneWayDistance * 2

theorem richsWalkDistance :
  richsWalk 20 200 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_richsWalkDistance_l4_415


namespace NUMINAMATH_CALUDE_vector_simplification_l4_498

/-- Given four points A, B, C, and D in a vector space, 
    prove that the vector AB minus DC minus CB equals AD -/
theorem vector_simplification (V : Type*) [AddCommGroup V] 
  (A B C D : V) : 
  (B - A) - (C - D) - (B - C) = D - A := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l4_498


namespace NUMINAMATH_CALUDE_exists_digit_satisfying_equation_l4_468

theorem exists_digit_satisfying_equation : ∃ a : ℕ, 
  0 ≤ a ∧ a ≤ 9 ∧ 1111 * a - 1 = (a - 1) ^ (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_digit_satisfying_equation_l4_468


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_77pi_l4_451

/-- Represents a triangular pyramid with vertices P, A, B, C -/
structure TriangularPyramid where
  PA : ℝ
  BC : ℝ
  AC : ℝ
  BP : ℝ
  CP : ℝ
  AB : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (t : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 77π -/
theorem circumscribed_sphere_surface_area_is_77pi :
  let t : TriangularPyramid := {
    PA := 2 * Real.sqrt 13,
    BC := 2 * Real.sqrt 13,
    AC := Real.sqrt 41,
    BP := Real.sqrt 41,
    CP := Real.sqrt 61,
    AB := Real.sqrt 61
  }
  circumscribedSphereSurfaceArea t = 77 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_77pi_l4_451


namespace NUMINAMATH_CALUDE_shopping_spending_l4_426

/-- The total spending of Elizabeth, Emma, and Elsa given their spending relationships -/
theorem shopping_spending (emma_spending : ℕ) : emma_spending = 58 →
  (emma_spending + 2 * emma_spending + 4 * (2 * emma_spending) = 638) := by
  sorry

#check shopping_spending

end NUMINAMATH_CALUDE_shopping_spending_l4_426


namespace NUMINAMATH_CALUDE_envelope_width_l4_480

/-- Given a rectangular envelope with length 4 inches and area 16 square inches, prove its width is 4 inches. -/
theorem envelope_width (length : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : length = 4)
  (h2 : area = 16)
  (h3 : area = length * width) : 
  width = 4 := by
  sorry

end NUMINAMATH_CALUDE_envelope_width_l4_480


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l4_478

theorem five_integers_sum_20_product_420 :
  ∃! (a b c d e : ℕ+),
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l4_478


namespace NUMINAMATH_CALUDE_count_factors_of_eight_squared_nine_cubed_seven_fifth_l4_491

theorem count_factors_of_eight_squared_nine_cubed_seven_fifth (n : Nat) :
  n = 8^2 * 9^3 * 7^5 →
  (Finset.filter (λ m : Nat => n % m = 0) (Finset.range (n + 1))).card = 294 :=
by sorry

end NUMINAMATH_CALUDE_count_factors_of_eight_squared_nine_cubed_seven_fifth_l4_491


namespace NUMINAMATH_CALUDE_min_distance_sum_l4_411

-- Define the parabola E
def E (x y : ℝ) : Prop := x^2 = 4*y

-- Define the circle F
def F (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define a line passing through F(0,1)
def line_through_F (m : ℝ) (x y : ℝ) : Prop := x = m*(y-1)

-- Define the theorem
theorem min_distance_sum (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    E x1 y1 ∧ E x2 y2 ∧
    line_through_F m x1 y1 ∧ line_through_F m x2 y2 ∧
    y1 + 2*y2 ≥ 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l4_411


namespace NUMINAMATH_CALUDE_solve_system_l4_474

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l4_474


namespace NUMINAMATH_CALUDE_last_two_digits_zero_l4_470

theorem last_two_digits_zero (x y : ℕ) : 
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_zero_l4_470


namespace NUMINAMATH_CALUDE_least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l4_482

theorem least_11_heavy_three_digit : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 106 → n % 11 ≤ 6 :=
by sorry

theorem is_11_heavy_106 : 106 % 11 > 6 :=
by sorry

theorem least_11_heavy_three_digit_is_106 : 
  ∀ n : ℕ, 100 ≤ n ∧ n % 11 > 6 → n ≥ 106 :=
by sorry

end NUMINAMATH_CALUDE_least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l4_482


namespace NUMINAMATH_CALUDE_sum_abc_equals_109610_l4_408

/-- Proves that given the conditions, the sum of a, b, and c is 109610 rupees -/
theorem sum_abc_equals_109610 (a b c : ℕ) : 
  (0.5 / 100 : ℚ) * a = 95 / 100 →  -- 0.5% of a equals 95 paise
  b = 3 * a - 50 →                  -- b is three times the amount of a minus 50
  c = (a - b) ^ 2 →                 -- c is the difference between a and b squared
  a > 0 →                           -- a is a positive integer
  c > 0 →                           -- c is a positive integer
  a + b + c = 109610 := by           -- The sum of a, b, and c is 109610 rupees
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_109610_l4_408


namespace NUMINAMATH_CALUDE_commodity_profit_optimization_l4_469

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 10)

theorem commodity_profit_optimization (cost_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : sales_quantity 20 = 360)
  (h3 : sales_quantity 30 = 60) :
  (∃ (optimal_price max_profit : ℝ),
    (∀ x, monthly_profit x ≤ monthly_profit optimal_price) ∧
    monthly_profit optimal_price = max_profit ∧
    optimal_price = 21 ∧
    max_profit = 3630) := by
  sorry

end NUMINAMATH_CALUDE_commodity_profit_optimization_l4_469


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l4_404

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x | x ∉ A}

-- State the theorem
theorem complement_A_intersect_B :
  complementA ∩ B = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l4_404


namespace NUMINAMATH_CALUDE_helen_cookies_baked_this_morning_l4_456

theorem helen_cookies_baked_this_morning (total_cookies : ℕ) (yesterday_cookies : ℕ) 
  (h1 : total_cookies = 574)
  (h2 : yesterday_cookies = 435) :
  total_cookies - yesterday_cookies = 139 := by
sorry

end NUMINAMATH_CALUDE_helen_cookies_baked_this_morning_l4_456


namespace NUMINAMATH_CALUDE_percent_problem_l4_440

theorem percent_problem (x : ℝ) : (0.25 * x = 0.12 * 1500 - 15) → x = 660 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l4_440


namespace NUMINAMATH_CALUDE_sticker_distribution_l4_463

/-- The number of ways to distribute n indistinguishable objects among k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

/-- Theorem stating that there are 935 ways to distribute 10 stickers among 5 sheets -/
theorem sticker_distribution : distribute num_stickers num_sheets = 935 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l4_463


namespace NUMINAMATH_CALUDE_student_arrangement_count_l4_447

theorem student_arrangement_count : ℕ := by
  -- Define the total number of students
  let total_students : ℕ := 7
  
  -- Define the condition that A and B are adjacent
  let adjacent_pair : ℕ := 1
  
  -- Define the condition that C and D are not adjacent
  let non_adjacent_pair : ℕ := 2
  
  -- Define the number of entities to arrange after bundling A and B
  let entities : ℕ := total_students - adjacent_pair
  
  -- Define the number of gaps after arranging the entities
  let gaps : ℕ := entities + 1
  
  -- Calculate the total number of arrangements
  let arrangements : ℕ := 
    (Nat.factorial entities) *    -- Arrange entities
    (gaps * (gaps - 1)) *         -- Place C and D in gaps
    2                             -- Arrange A and B within their bundle
  
  -- Prove that the number of arrangements is 960
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l4_447


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l4_437

def systematicSample (n : ℕ) (k : ℕ) (start : ℕ) : List ℕ :=
  List.range k |>.map (fun i => start + i * (n / k))

theorem correct_systematic_sample :
  systematicSample 20 4 5 = [5, 10, 15, 20] := by
  sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l4_437


namespace NUMINAMATH_CALUDE_inverse_g_at_505_l4_473

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem inverse_g_at_505 : g⁻¹ 505 = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_505_l4_473


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4_472

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_decreasing : is_decreasing_sequence a)
  (h_third_term : a 3 = 18)
  (h_fourth_term : a 4 = 12) :
  a 1 = 40.5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4_472


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l4_455

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, 2 * n = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, 2 * m = 5 * k) ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l4_455


namespace NUMINAMATH_CALUDE_planned_pigs_addition_l4_400

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The initial number of animals on the farm -/
def initial_animals : FarmAnimals := ⟨2, 3, 6⟩

/-- The number of cows and goats to be added -/
def planned_additions : FarmAnimals := ⟨3, 0, 2⟩

/-- The final total number of animals -/
def final_total : ℕ := 21

/-- The number of pigs to be added -/
def pigs_to_add : ℕ := final_total - (initial_animals.cows + initial_animals.goats + 
                        planned_additions.cows + planned_additions.goats) - initial_animals.pigs

theorem planned_pigs_addition : pigs_to_add = 8 := by
  sorry

end NUMINAMATH_CALUDE_planned_pigs_addition_l4_400


namespace NUMINAMATH_CALUDE_min_holiday_days_l4_452

/-- Represents a day during the holiday -/
structure Day where
  morning_sunny : Bool
  afternoon_sunny : Bool

/-- Conditions for the holiday weather -/
def valid_holiday (days : List Day) : Prop :=
  let total_days := days.length
  let rainy_days := days.filter (fun d => ¬d.morning_sunny ∨ ¬d.afternoon_sunny)
  let sunny_afternoons := days.filter (fun d => d.afternoon_sunny)
  let sunny_mornings := days.filter (fun d => d.morning_sunny)
  rainy_days.length = 7 ∧
  days.all (fun d => ¬d.afternoon_sunny → d.morning_sunny) ∧
  sunny_afternoons.length = 5 ∧
  sunny_mornings.length = 6

/-- The theorem to be proved -/
theorem min_holiday_days :
  ∃ (days : List Day), valid_holiday days ∧
    ∀ (other_days : List Day), valid_holiday other_days → days.length ≤ other_days.length :=
by
  sorry

end NUMINAMATH_CALUDE_min_holiday_days_l4_452


namespace NUMINAMATH_CALUDE_matts_writing_speed_l4_446

/-- Matt's writing speed problem -/
theorem matts_writing_speed (right_hand_speed : ℕ) (time : ℕ) (difference : ℕ) : 
  right_hand_speed = 10 →
  time = 5 →
  difference = 15 →
  ∃ (left_hand_speed : ℕ), 
    right_hand_speed * time = left_hand_speed * time + difference ∧
    left_hand_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_matts_writing_speed_l4_446


namespace NUMINAMATH_CALUDE_divisible_count_theorem_l4_479

def count_divisible (n : ℕ) : ℕ :=
  let div2 := n / 2
  let div3 := n / 3
  let div5 := n / 5
  let div6 := n / 6
  let div10 := n / 10
  let div15 := n / 15
  let div30 := n / 30
  (div2 + div3 + div5 - div6 - div10 - div15 + div30) - div6

theorem divisible_count_theorem :
  count_divisible 1000 = 568 := by sorry

end NUMINAMATH_CALUDE_divisible_count_theorem_l4_479


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l4_434

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x² = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l4_434


namespace NUMINAMATH_CALUDE_tank_volume_proof_l4_453

def inletRate : ℝ := 5
def outletRate1 : ℝ := 9
def outletRate2 : ℝ := 8
def emptyTime : ℝ := 2880
def inchesPerFoot : ℝ := 12

def tankVolume : ℝ := 20

theorem tank_volume_proof :
  let netEmptyRate := outletRate1 + outletRate2 - inletRate
  let volumeInCubicInches := netEmptyRate * emptyTime
  let cubicInchesPerCubicFoot := inchesPerFoot ^ 3
  volumeInCubicInches / cubicInchesPerCubicFoot = tankVolume := by
  sorry

#check tank_volume_proof

end NUMINAMATH_CALUDE_tank_volume_proof_l4_453


namespace NUMINAMATH_CALUDE_projectile_max_height_l4_442

/-- The height of a projectile as a function of time -/
def h (t : ℝ) : ℝ := -18 * t^2 + 72 * t + 25

/-- Theorem: The maximum height reached by the projectile is 97 feet -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 97 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l4_442


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l4_402

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f) (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m > 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l4_402


namespace NUMINAMATH_CALUDE_square_area_ratio_l4_462

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 / b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l4_462


namespace NUMINAMATH_CALUDE_steps_to_eleventh_floor_l4_464

/-- Given that there are 42 steps between the 3rd and 5th floors of a building,
    prove that there are 210 steps from the ground floor to the 11th floor. -/
theorem steps_to_eleventh_floor :
  let steps_between_3_and_5 : ℕ := 42
  let floor_xiao_dong_lives : ℕ := 11
  let ground_floor : ℕ := 1
  let steps_to_xiao_dong : ℕ := (floor_xiao_dong_lives - ground_floor) * 
    (steps_between_3_and_5 / (5 - 3))
  steps_to_xiao_dong = 210 := by
  sorry


end NUMINAMATH_CALUDE_steps_to_eleventh_floor_l4_464


namespace NUMINAMATH_CALUDE_jackson_missed_wednesdays_l4_448

/-- The number of missed Wednesdays in Jackson's school year --/
def missed_wednesdays (weeks : ℕ) (total_sandwiches : ℕ) (missed_fridays : ℕ) : ℕ :=
  weeks * 2 - total_sandwiches - missed_fridays

theorem jackson_missed_wednesdays :
  missed_wednesdays 36 69 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_jackson_missed_wednesdays_l4_448


namespace NUMINAMATH_CALUDE_pages_to_read_on_third_day_l4_497

/-- Given a book with 100 pages and Lance's reading progress over three days,
    prove that he needs to read 35 pages on the third day to finish the book. -/
theorem pages_to_read_on_third_day (pages_day1 pages_day2 : ℕ) 
  (h1 : pages_day1 = 35)
  (h2 : pages_day2 = pages_day1 - 5) :
  100 - (pages_day1 + pages_day2) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_on_third_day_l4_497


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l4_487

theorem right_triangle_squares_area (x : ℝ) : 
  let small_square_area := (3 * x)^2
  let large_square_area := (6 * x)^2
  let triangle_area := (1/2) * (3 * x) * (6 * x)
  small_square_area + large_square_area + triangle_area = 1000 →
  x = 10 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l4_487


namespace NUMINAMATH_CALUDE_rotation_cycle_implies_equilateral_l4_443

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A₁ : Point
  A₂ : Point
  A₃ : Point

/-- Rotation of a point around a center by an angle -/
def rotate (center : Point) (angle : ℝ) (p : Point) : Point :=
  sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  sorry

/-- The sequence of points A_s -/
def A (s : ℕ) : Point :=
  sorry

/-- The sequence of points P_k -/
def P (k : ℕ) : Point :=
  sorry

/-- The main theorem -/
theorem rotation_cycle_implies_equilateral 
  (t : Triangle) (P₀ : Point) : 
  (P 1986 = P₀) → Triangle.isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_rotation_cycle_implies_equilateral_l4_443


namespace NUMINAMATH_CALUDE_linear_system_sum_theorem_l4_436

theorem linear_system_sum_theorem (a b c x y z : ℝ) 
  (eq1 : 23*x + b*y + c*z = 0)
  (eq2 : a*x + 33*y + c*z = 0)
  (eq3 : a*x + b*y + 52*z = 0)
  (ha : a ≠ 23)
  (hx : x ≠ 0) :
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_system_sum_theorem_l4_436


namespace NUMINAMATH_CALUDE_distinct_roots_root_one_k_values_l4_414

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 - (2*k + 1)*x + k^2 + k

-- Theorem 1: The equation has two distinct real roots for all k
theorem distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 :=
sorry

-- Theorem 2: When one root is 1, k is either 0 or 1
theorem root_one_k_values : 
  ∀ k : ℝ, quadratic k 1 = 0 → k = 0 ∨ k = 1 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_root_one_k_values_l4_414


namespace NUMINAMATH_CALUDE_right_triangle_median_property_l4_477

theorem right_triangle_median_property (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_median : (c/2)^2 = a*b) : c/2 = (c/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_median_property_l4_477


namespace NUMINAMATH_CALUDE_max_x_is_one_l4_413

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x + a else x^2 + 1 + a

-- State the theorem
theorem max_x_is_one (a : ℝ) :
  (∀ x : ℝ, f a (2 - x) ≥ f a x) →
  (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_x_is_one_l4_413


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l4_418

theorem roots_sum_of_powers (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^4 + p^3*q^2 + p^2*q^3 + q^4 = 241 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l4_418


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l4_441

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : 
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  senior_tickets = 348 →
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 9745 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l4_441


namespace NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l4_490

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((9 : ℝ) / 16 + 16 / 81) = Real.sqrt 985 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l4_490


namespace NUMINAMATH_CALUDE_inequality_proof_l4_481

theorem inequality_proof (a b c : ℝ) :
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) - (a*b + b*c + c*a - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4_481


namespace NUMINAMATH_CALUDE_pool_filling_time_l4_454

/-- Proves that filling a 30,000-gallon pool with 5 hoses, each supplying 2.5 gallons per minute, takes 40 hours. -/
theorem pool_filling_time :
  let pool_capacity : ℝ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 2.5
  let minutes_per_hour : ℕ := 60
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose * minutes_per_hour
  pool_capacity / total_flow_rate = 40 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l4_454


namespace NUMINAMATH_CALUDE_min_contestants_solved_all_l4_428

theorem min_contestants_solved_all (total : ℕ) (solved1 solved2 solved3 solved4 : ℕ) 
  (h_total : total = 100)
  (h_solved1 : solved1 = 90)
  (h_solved2 : solved2 = 85)
  (h_solved3 : solved3 = 80)
  (h_solved4 : solved4 = 75) :
  ∃ (min_solved_all : ℕ), 
    min_solved_all ≤ solved1 ∧
    min_solved_all ≤ solved2 ∧
    min_solved_all ≤ solved3 ∧
    min_solved_all ≤ solved4 ∧
    min_solved_all ≥ solved1 + solved2 + solved3 + solved4 - 3 * total ∧
    min_solved_all = 30 :=
sorry

end NUMINAMATH_CALUDE_min_contestants_solved_all_l4_428


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l4_465

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else Int.floor (1 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l4_465


namespace NUMINAMATH_CALUDE_common_tangent_sum_l4_438

/-- Parabola Q₁ -/
def Q₁ (x y : ℝ) : Prop := y = x^2 + 2

/-- Parabola Q₂ -/
def Q₂ (x y : ℝ) : Prop := x = y^2 + 8

/-- Common tangent line M -/
def M (d e f : ℤ) (x y : ℝ) : Prop := d * x + e * y = f

/-- M has nonzero integer slope -/
def nonzero_integer_slope (d e : ℤ) : Prop := d ≠ 0 ∧ e ≠ 0

/-- d, e, f are coprime -/
def coprime (d e f : ℤ) : Prop := Nat.gcd (Nat.gcd d.natAbs e.natAbs) f.natAbs = 1

/-- Main theorem -/
theorem common_tangent_sum (d e f : ℤ) :
  (∃ x y : ℝ, Q₁ x y ∧ Q₂ x y ∧ M d e f x y) →
  nonzero_integer_slope d e →
  coprime d e f →
  d + e + f = 8 := by sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l4_438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l4_407

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -1 → a 2 = 1 →
  ∀ n : ℕ, a n = 2 * n - 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l4_407


namespace NUMINAMATH_CALUDE_jason_stored_23_bales_l4_416

/-- The number of bales Jason stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Jason stored 23 bales in the barn -/
theorem jason_stored_23_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73) 
  (h2 : final_bales = 96) : 
  bales_stored initial_bales final_bales = 23 := by
  sorry

end NUMINAMATH_CALUDE_jason_stored_23_bales_l4_416


namespace NUMINAMATH_CALUDE_fuji_ratio_l4_422

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The conditions of the orchard as described in the problem -/
def orchardConditions (o : Orchard) : Prop :=
  o.crossPollinated = o.totalTrees / 10 ∧
  o.pureFuji + o.crossPollinated = 204 ∧
  o.pureGala = 36 ∧
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated

/-- The theorem stating the ratio of pure Fuji trees to all trees -/
theorem fuji_ratio (o : Orchard) (h : orchardConditions o) :
  3 * o.totalTrees = 4 * o.pureFuji := by
  sorry


end NUMINAMATH_CALUDE_fuji_ratio_l4_422
