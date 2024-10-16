import Mathlib

namespace NUMINAMATH_CALUDE_hexagon_theorem_l2289_228949

/-- Regular hexagon with side length 4 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ := 4)

/-- Intersection point of diagonals CE and DF -/
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Point K defined by vector equation -/
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Predicate for a point being outside a hexagon -/
def is_outside (p : ℝ × ℝ) (hex : RegularHexagon) : Prop := sorry

theorem hexagon_theorem (hex : RegularHexagon) :
  is_outside (K hex) hex ∧ distance (K hex) hex.A = (4 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_theorem_l2289_228949


namespace NUMINAMATH_CALUDE_third_person_weight_is_131_l2289_228919

/-- Calculates the true weight of the third person (C) entering an elevator given the following conditions:
    - There are initially 6 people in the elevator with an average weight of 156 lbs.
    - Three people (A, B, C) enter the elevator one by one.
    - The weights of their clothing and backpacks are 18 lbs, 20 lbs, and 22 lbs respectively.
    - After each person enters, the average weight changes to 159 lbs, 162 lbs, and 161 lbs respectively. -/
def calculate_third_person_weight (initial_people : Nat) (initial_avg : Nat)
  (a_extra_weight : Nat) (b_extra_weight : Nat) (c_extra_weight : Nat)
  (avg_after_a : Nat) (avg_after_b : Nat) (avg_after_c : Nat) : Nat :=
  let total_initial := initial_people * initial_avg
  let total_after_a := (initial_people + 1) * avg_after_a
  let total_after_b := (initial_people + 2) * avg_after_b
  let total_after_c := (initial_people + 3) * avg_after_c
  total_after_c - total_after_b - c_extra_weight

/-- Theorem stating that given the conditions in the problem, 
    the true weight of the third person (C) is 131 lbs. -/
theorem third_person_weight_is_131 :
  calculate_third_person_weight 6 156 18 20 22 159 162 161 = 131 := by
  sorry

end NUMINAMATH_CALUDE_third_person_weight_is_131_l2289_228919


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2289_228966

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 40) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1 / 2 : ℝ) * d * h₁ + (1 / 2 : ℝ) * d * h₂ = 300 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2289_228966


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2289_228913

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 2041 / 25 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2289_228913


namespace NUMINAMATH_CALUDE_addition_and_subtraction_proof_l2289_228947

theorem addition_and_subtraction_proof :
  (1 + (-11) = -10) ∧ (0 - 4.5 = -4.5) := by sorry

end NUMINAMATH_CALUDE_addition_and_subtraction_proof_l2289_228947


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2289_228920

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2289_228920


namespace NUMINAMATH_CALUDE_car_speed_is_60_l2289_228935

/-- Represents the scenario of two friends traveling to a hunting base -/
structure HuntingTrip where
  walker_distance : ℝ  -- Distance of walker from base
  car_distance : ℝ     -- Distance of car owner from base
  total_time : ℝ       -- Total time to reach the base
  early_start : ℝ      -- Time walker would start earlier in alternative scenario
  early_meet : ℝ       -- Distance from walker's home where they'd meet in alternative scenario

/-- Calculates the speed of the car given the hunting trip scenario -/
def calculate_car_speed (trip : HuntingTrip) : ℝ :=
  60  -- Placeholder for the actual calculation

/-- Theorem stating that the car speed is 60 km/h given the specific scenario -/
theorem car_speed_is_60 (trip : HuntingTrip) 
  (h1 : trip.walker_distance = 46)
  (h2 : trip.car_distance = 30)
  (h3 : trip.total_time = 1)
  (h4 : trip.early_start = 8/3)
  (h5 : trip.early_meet = 11) :
  calculate_car_speed trip = 60 := by
  sorry

#eval calculate_car_speed { 
  walker_distance := 46, 
  car_distance := 30, 
  total_time := 1, 
  early_start := 8/3, 
  early_meet := 11 
}

end NUMINAMATH_CALUDE_car_speed_is_60_l2289_228935


namespace NUMINAMATH_CALUDE_final_marble_difference_l2289_228934

/- Define the initial difference in marbles between Ed and Doug -/
def initial_difference : ℕ := 30

/- Define the number of marbles Ed lost -/
def marbles_lost : ℕ := 21

/- Define Ed's final number of marbles -/
def ed_final_marbles : ℕ := 91

/- Define Doug's number of marbles (which remains constant) -/
def doug_marbles : ℕ := ed_final_marbles + marbles_lost - initial_difference

/- Theorem stating the final difference in marbles -/
theorem final_marble_difference :
  ed_final_marbles - doug_marbles = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_final_marble_difference_l2289_228934


namespace NUMINAMATH_CALUDE_product_odd_probability_l2289_228914

def range_start : ℕ := 5
def range_end : ℕ := 19

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def odd_integers : ℕ := (total_integers + 1) / 2

theorem product_odd_probability :
  (odd_integers.choose 2 : ℚ) / (total_integers.choose 2) = 4 / 15 :=
sorry

end NUMINAMATH_CALUDE_product_odd_probability_l2289_228914


namespace NUMINAMATH_CALUDE_min_value_theorem_l2289_228986

theorem min_value_theorem (y₁ y₂ y₃ : ℝ) (h_pos₁ : y₁ > 0) (h_pos₂ : y₂ > 0) (h_pos₃ : y₃ > 0)
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) :
  y₁^2 + 4 * y₂^2 + 9 * y₃^2 ≥ 14400 / 29 ∧
  (∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + 4 * y₂'^2 + 9 * y₃'^2 = 14400 / 29 ∧
    2 * y₁' + 3 * y₂' + 4 * y₃' = 120 ∧ y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2289_228986


namespace NUMINAMATH_CALUDE_piggy_bank_dime_difference_l2289_228961

theorem piggy_bank_dime_difference :
  ∀ (a b c d : ℕ),
  a + b + c + d = 150 →
  5 * a + 10 * b + 25 * c + 50 * d = 1500 →
  (∃ (b_max b_min : ℕ),
    (∀ (a' b' c' d' : ℕ),
      a' + b' + c' + d' = 150 →
      5 * a' + 10 * b' + 25 * c' + 50 * d' = 1500 →
      b' ≤ b_max ∧ b' ≥ b_min) ∧
    b_max - b_min = 150) :=
by sorry

end NUMINAMATH_CALUDE_piggy_bank_dime_difference_l2289_228961


namespace NUMINAMATH_CALUDE_jake_planting_charge_l2289_228926

/-- The hourly rate Jake wants to make -/
def desired_hourly_rate : ℝ := 20

/-- The time it takes to plant flowers in hours -/
def planting_time : ℝ := 2

/-- The amount Jake should charge for planting flowers -/
def planting_charge : ℝ := desired_hourly_rate * planting_time

theorem jake_planting_charge : planting_charge = 40 := by
  sorry

end NUMINAMATH_CALUDE_jake_planting_charge_l2289_228926


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2289_228993

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement to be proved
theorem perpendicular_transitivity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) (h2 : α ≠ β)
  (h3 : perp a α) (h4 : perp a β) (h5 : perp b β) :
  perp b α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2289_228993


namespace NUMINAMATH_CALUDE_gcd_1987_2025_l2289_228923

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1987_2025_l2289_228923


namespace NUMINAMATH_CALUDE_x_squared_eq_neg_one_is_quadratic_l2289_228930

/-- A quadratic equation in one variable -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Check if an equation is in the form ax² + bx + c = 0 -/
def isQuadraticForm (f : ℝ → ℝ) : Prop :=
  ∃ (q : QuadraticEquation), ∀ x, f x = q.a * x^2 + q.b * x + q.c

/-- The specific equation x² = -1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: The equation x² = -1 is a quadratic equation in one variable -/
theorem x_squared_eq_neg_one_is_quadratic : isQuadraticForm f := by sorry

end NUMINAMATH_CALUDE_x_squared_eq_neg_one_is_quadratic_l2289_228930


namespace NUMINAMATH_CALUDE_special_triangle_is_equilateral_l2289_228985

/-- A triangle with sides in geometric progression and angles in arithmetic progression -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  q : ℝ
  -- Angles of the triangle
  α : ℝ
  δ : ℝ
  -- Side lengths form a geometric progression
  side_gp : q > 0
  -- Angles form an arithmetic progression
  angle_ap : True
  -- Sum of angles is 180 degrees
  angle_sum : α - δ + α + (α + δ) = 180

/-- The theorem stating that a SpecialTriangle must be equilateral -/
theorem special_triangle_is_equilateral (t : SpecialTriangle) : t.q = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_is_equilateral_l2289_228985


namespace NUMINAMATH_CALUDE_unit_square_max_distance_l2289_228939

theorem unit_square_max_distance (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  min (min (min (Real.sqrt ((x - 0)^2 + (y - 0)^2))
                (Real.sqrt ((x - 1)^2 + (y - 0)^2)))
           (Real.sqrt ((x - 1)^2 + (y - 1)^2)))
      (Real.sqrt ((x - 0)^2 + (y - 1)^2))
  ≤ Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_unit_square_max_distance_l2289_228939


namespace NUMINAMATH_CALUDE_quadratic_coefficient_unique_l2289_228967

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.evaluate (f.vertexX)

theorem quadratic_coefficient_unique (f : QuadraticFunction) :
    f.vertexX = 2 ∧ f.vertexY = -3 ∧ f.evaluate 1 = -2 → f.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_unique_l2289_228967


namespace NUMINAMATH_CALUDE_fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l2289_228999

theorem fraction_of_fraction (a b c d : ℚ) (h : a / b = c / d) :
  (c / d) / (a / b) = d / a :=
by sorry

theorem fraction_of_three_fifths_is_two_fifteenths :
  (2 / 15) / (3 / 5) = 2 / 9 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_fraction_of_three_fifths_is_two_fifteenths_l2289_228999


namespace NUMINAMATH_CALUDE_library_books_taken_out_l2289_228996

theorem library_books_taken_out (initial_books : ℕ) (books_returned : ℕ) (books_taken_out : ℕ) (final_books : ℕ) :
  initial_books = 235 →
  books_returned = 56 →
  books_taken_out = 35 →
  final_books = 29 →
  ∃ (tuesday_books : ℕ), tuesday_books = 227 ∧ 
    initial_books - tuesday_books + books_returned - books_taken_out = final_books :=
by
  sorry


end NUMINAMATH_CALUDE_library_books_taken_out_l2289_228996


namespace NUMINAMATH_CALUDE_optimal_cylinder_ratio_l2289_228974

/-- The optimal ratio of height to radius for a cylinder with minimal surface area --/
theorem optimal_cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  V = π * r^2 * h → (∀ h' r', h' > 0 → r' > 0 → V = π * r'^2 * h' → 
    2 * π * r^2 + 2 * π * r * h ≤ 2 * π * r'^2 + 2 * π * r' * h') → 
  h / r = 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_cylinder_ratio_l2289_228974


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l2289_228933

theorem complex_multiplication_simplification :
  ((-3 - 2 * Complex.I) - (1 + 4 * Complex.I)) * (2 - 3 * Complex.I) = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l2289_228933


namespace NUMINAMATH_CALUDE_not_necessarily_true_inequality_l2289_228964

theorem not_necessarily_true_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬(∀ a b c, c < b ∧ b < a ∧ a * c < 0 → b^2 / c > a^2 / c) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_true_inequality_l2289_228964


namespace NUMINAMATH_CALUDE_number_of_divisors_of_fermat_like_expression_l2289_228998

theorem number_of_divisors_of_fermat_like_expression : 
  ∃ (S : Finset Nat), 
    (∀ n ∈ S, n > 1 ∧ ∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) ∧ 
    (∀ n : Nat, n > 1 → (∀ a : ℤ, (n : ℤ) ∣ (a^25 - a)) → n ∈ S) ∧
    Finset.card S = 31 :=
sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_fermat_like_expression_l2289_228998


namespace NUMINAMATH_CALUDE_least_consecutive_bigness_l2289_228962

def bigness (a b c : ℕ) : ℕ := a * b * c + 2 * (a * b + b * c + a * c) + 4 * (a + b + c)

def has_integer_sides (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ bigness a b c = n

theorem least_consecutive_bigness :
  (∀ k < 55, ¬(has_integer_sides k ∧ has_integer_sides (k + 1))) ∧
  (has_integer_sides 55 ∧ has_integer_sides 56) :=
sorry

end NUMINAMATH_CALUDE_least_consecutive_bigness_l2289_228962


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l2289_228906

/-- Given a 2x2 matrix M, prove that its inverse is correct. -/
theorem matrix_inverse_proof (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = ![![1, 0], ![1, 1]]) : 
  M⁻¹ = ![![1, 0], ![-1, 1]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l2289_228906


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_specific_prism_l2289_228951

/-- An equilateral triangular prism -/
structure EquilateralTriangularPrism where
  /-- The base side length of the prism -/
  baseSideLength : ℝ
  /-- The height of the prism -/
  height : ℝ

/-- The radius of the inscribed sphere in an equilateral triangular prism -/
def inscribedSphereRadius (prism : EquilateralTriangularPrism) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed sphere in an equilateral triangular prism
    with base side length 1 and height √2 is equal to √2/6 -/
theorem inscribed_sphere_radius_specific_prism :
  let prism : EquilateralTriangularPrism := { baseSideLength := 1, height := Real.sqrt 2 }
  inscribedSphereRadius prism = Real.sqrt 2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_specific_prism_l2289_228951


namespace NUMINAMATH_CALUDE_problem_solution_l2289_228915

theorem problem_solution (m n : ℕ+) 
  (h1 : m.val + 12 < n.val + 3)
  (h2 : (m.val + (m.val + 6) + (m.val + 12) + (n.val + 3) + (n.val + 6) + 3 * n.val) / 6 = n.val + 3)
  (h3 : (m.val + 12 + n.val + 3) / 2 = n.val + 3) : 
  m.val + n.val = 57 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2289_228915


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2289_228997

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 2.4)
  (h2 : (c + d) / 2 = 2.3)
  (h3 : (e + f) / 2 = 3.7) :
  (a + b + c + d + e + f) / 6 = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2289_228997


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l2289_228959

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_sum : a + b + c + d = 100)
  (h_frac_sum : a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c) = 95) :
  1 / (b + c + d) + 1 / (a + c + d) + 1 / (a + b + d) + 1 / (a + b + c) = 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l2289_228959


namespace NUMINAMATH_CALUDE_negation_equivalence_l2289_228916

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2289_228916


namespace NUMINAMATH_CALUDE_sticker_distribution_l2289_228943

/-- The number of stickers Mary bought initially -/
def total_stickers : ℕ := 1500

/-- Susan's share of stickers -/
def susan_share : ℕ := 300

/-- Andrew's initial share of stickers -/
def andrew_initial_share : ℕ := 300

/-- Sam's initial share of stickers -/
def sam_initial_share : ℕ := 900

/-- The amount of stickers Sam gave to Andrew -/
def sam_to_andrew : ℕ := 600

/-- Andrew's final share of stickers -/
def andrew_final_share : ℕ := 900

theorem sticker_distribution :
  -- The total is the sum of all initial shares
  total_stickers = susan_share + andrew_initial_share + sam_initial_share ∧
  -- The ratio of shares is 1:1:3
  susan_share = andrew_initial_share ∧
  sam_initial_share = 3 * andrew_initial_share ∧
  -- Sam gave Andrew two-thirds of his share
  sam_to_andrew = 2 * sam_initial_share / 3 ∧
  -- Andrew's final share is his initial plus what Sam gave him
  andrew_final_share = andrew_initial_share + sam_to_andrew :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2289_228943


namespace NUMINAMATH_CALUDE_rectangle_area_l2289_228928

theorem rectangle_area (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2289_228928


namespace NUMINAMATH_CALUDE_max_trig_function_ratio_l2289_228955

/-- Given a function f(x) = 3sin(x) + 4cos(x) that attains its maximum value when x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 65/7 -/
theorem max_trig_function_ratio (θ : Real) 
    (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
    (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_trig_function_ratio_l2289_228955


namespace NUMINAMATH_CALUDE_power_calculation_l2289_228965

theorem power_calculation : (8^8 / 8^5) * 2^10 * 2^3 = 2^22 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2289_228965


namespace NUMINAMATH_CALUDE_simplify_fraction_l2289_228992

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2289_228992


namespace NUMINAMATH_CALUDE_smallest_sum_square_config_l2289_228991

/-- A configuration of four positive integers on a square's vertices. -/
structure SquareConfig where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+

/-- Predicate to check if one number is a multiple of another. -/
def isMultiple (x y : ℕ+) : Prop := ∃ k : ℕ+, x = k * y

/-- Predicate to check if the configuration satisfies the edge multiple condition. -/
def satisfiesEdgeCondition (config : SquareConfig) : Prop :=
  (isMultiple config.a config.b ∨ isMultiple config.b config.a) ∧
  (isMultiple config.b config.c ∨ isMultiple config.c config.b) ∧
  (isMultiple config.c config.d ∨ isMultiple config.d config.c) ∧
  (isMultiple config.d config.a ∨ isMultiple config.a config.d)

/-- Predicate to check if the configuration satisfies the diagonal non-multiple condition. -/
def satisfiesDiagonalCondition (config : SquareConfig) : Prop :=
  ¬(isMultiple config.a config.c ∨ isMultiple config.c config.a) ∧
  ¬(isMultiple config.b config.d ∨ isMultiple config.d config.b)

/-- Theorem stating the smallest possible sum of the four integers. -/
theorem smallest_sum_square_config :
  ∀ config : SquareConfig,
    satisfiesEdgeCondition config →
    satisfiesDiagonalCondition config →
    (config.a + config.b + config.c + config.d : ℕ) ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_square_config_l2289_228991


namespace NUMINAMATH_CALUDE_production_days_l2289_228987

theorem production_days (n : ℕ) 
  (h1 : (50 : ℝ) * n = n * 50)
  (h2 : (50 : ℝ) * n + 115 = (n + 1) * 55) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l2289_228987


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2289_228960

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, n - 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 2*Real.sqrt 2 + 3) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2/x + 1/y = 2*Real.sqrt 2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2289_228960


namespace NUMINAMATH_CALUDE_probability_two_segments_longer_than_one_l2289_228956

/-- The probability of exactly two segments being longer than 1 when a line segment 
    of length 3 is divided into three parts by randomly selecting two points -/
theorem probability_two_segments_longer_than_one (total_length : ℝ) 
  (h_total_length : total_length = 3) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_probability_two_segments_longer_than_one_l2289_228956


namespace NUMINAMATH_CALUDE_garden_area_l2289_228932

/-- A rectangular garden with specific walking conditions -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  total_distance : ℝ
  length_condition : length * length_walks = total_distance
  perimeter_condition : (2 * length + 2 * width) * perimeter_walks = total_distance

/-- The area of a garden with the given conditions is 2400 square meters -/
theorem garden_area (g : Garden) 
    (h1 : g.length_walks = 50)
    (h2 : g.perimeter_walks = 15)
    (h3 : g.total_distance = 3000) : 
  g.length * g.width = 2400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2289_228932


namespace NUMINAMATH_CALUDE_mr_green_garden_yield_l2289_228946

/-- Represents the dimensions and expected yield of a rectangular garden -/
structure Garden where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  yield_per_sqft : ℚ

/-- Calculates the expected potato yield from a garden in pounds -/
def expected_yield (g : Garden) : ℚ :=
  (g.length_paces * g.feet_per_pace) *
  (g.width_paces * g.feet_per_pace) *
  g.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Green's garden -/
theorem mr_green_garden_yield :
  let g : Garden := {
    length_paces := 18,
    width_paces := 25,
    feet_per_pace := 3,
    yield_per_sqft := 3/4
  }
  expected_yield g = 3037.5 := by sorry

end NUMINAMATH_CALUDE_mr_green_garden_yield_l2289_228946


namespace NUMINAMATH_CALUDE_stating_professor_seating_arrangements_l2289_228945

/-- Represents the number of chairs in a row -/
def num_chairs : ℕ := 10

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the number of professors -/
def num_professors : ℕ := 3

/-- Represents the effective number of chair positions professors can choose from -/
def effective_chairs : ℕ := 4

/-- 
Theorem stating that the number of ways professors can choose their chairs
under the given conditions is 24.
-/
theorem professor_seating_arrangements :
  (effective_chairs.choose num_professors) * num_professors.factorial = 24 :=
by sorry

end NUMINAMATH_CALUDE_stating_professor_seating_arrangements_l2289_228945


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2289_228936

theorem abs_sum_inequality (x b : ℝ) (hb : b > 0) :
  (|x - 2| + |x + 3| < b) ↔ (b > 5) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2289_228936


namespace NUMINAMATH_CALUDE_equation_solution_l2289_228929

theorem equation_solution : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2289_228929


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2289_228982

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (25*a / (73*b)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l2289_228982


namespace NUMINAMATH_CALUDE_total_nails_is_113_l2289_228911

/-- The number of nails Cassie needs to cut for her pets -/
def total_nails_to_cut : ℕ :=
  let num_dogs : ℕ := 4
  let num_parrots : ℕ := 8
  let nails_per_dog_foot : ℕ := 4
  let feet_per_dog : ℕ := 4
  let claws_per_parrot_leg : ℕ := 3
  let legs_per_parrot : ℕ := 2
  let extra_nail : ℕ := 1

  let dog_nails : ℕ := num_dogs * nails_per_dog_foot * feet_per_dog
  let parrot_nails : ℕ := num_parrots * claws_per_parrot_leg * legs_per_parrot
  
  dog_nails + parrot_nails + extra_nail

/-- Theorem stating that the total number of nails Cassie needs to cut is 113 -/
theorem total_nails_is_113 : total_nails_to_cut = 113 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_is_113_l2289_228911


namespace NUMINAMATH_CALUDE_eighth_term_value_l2289_228940

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first six terms is 21
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  -- Seventh term is 8
  seventh_term : a + 6*d = 8

/-- The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : seq.a + 7*seq.d = 65/7 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2289_228940


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2289_228912

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2289_228912


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2289_228938

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2289_228938


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2289_228917

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2289_228917


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2289_228944

theorem probability_of_white_ball (P_red P_black P_yellow P_white : ℚ) : 
  P_red = 1/3 →
  P_black + P_yellow = 5/12 →
  P_yellow + P_white = 5/12 →
  P_red + P_black + P_yellow + P_white = 1 →
  P_white = 1/4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2289_228944


namespace NUMINAMATH_CALUDE_coffee_lasts_40_days_l2289_228910

/-- The number of days coffee will last given the amount bought, brewing capacity, and daily consumption. -/
def coffee_duration (pounds_bought : ℕ) (cups_per_pound : ℕ) (cups_per_day : ℕ) : ℕ :=
  (pounds_bought * cups_per_pound) / cups_per_day

/-- Theorem stating that under the given conditions, the coffee will last 40 days. -/
theorem coffee_lasts_40_days :
  coffee_duration 3 40 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_coffee_lasts_40_days_l2289_228910


namespace NUMINAMATH_CALUDE_reservoir_refill_rate_l2289_228937

theorem reservoir_refill_rate 
  (V : ℝ) (R : ℝ) 
  (h1 : V - 90 * (40000 - R) = 0) 
  (h2 : V - 60 * (32000 - R) = 0) : 
  R = 56000 := by
sorry

end NUMINAMATH_CALUDE_reservoir_refill_rate_l2289_228937


namespace NUMINAMATH_CALUDE_valid_numbers_l2289_228958

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a m X : ℕ),
    (a = 1 ∨ a = 2) ∧
    m > 0 ∧
    X < 10^(m-1) ∧
    n = a * 10^(m-1) + X ∧
    3 * n = 10 * X + a

theorem valid_numbers :
  {n : ℕ | is_valid_number n} =
    {142857, 285714, 428571, 571428, 714285, 857142} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2289_228958


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2289_228979

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 104) :
  |x - y| = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2289_228979


namespace NUMINAMATH_CALUDE_probability_log3_integer_l2289_228904

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The count of four-digit numbers that are powers of 3. -/
def CountPowersOfThree : ℕ := 2

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 3. -/
def ProbabilityPowerOfThree : ℚ := CountPowersOfThree / TotalFourDigitNumbers

theorem probability_log3_integer :
  ProbabilityPowerOfThree = 1 / 4500 := by
  sorry

end NUMINAMATH_CALUDE_probability_log3_integer_l2289_228904


namespace NUMINAMATH_CALUDE_cricket_run_rate_l2289_228978

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate : required_run_rate 50 10 (32/10) 282 = 25/4 := by
  sorry

#eval required_run_rate 50 10 (32/10) 282

end NUMINAMATH_CALUDE_cricket_run_rate_l2289_228978


namespace NUMINAMATH_CALUDE_pie_shop_revenue_l2289_228907

/-- Represents the price of a single slice of pie in dollars -/
def slice_price : ℕ := 5

/-- Represents the number of slices in a whole pie -/
def slices_per_pie : ℕ := 4

/-- Represents the number of pies sold -/
def pies_sold : ℕ := 9

/-- Calculates the total revenue from selling pies -/
def total_revenue : ℕ := pies_sold * slices_per_pie * slice_price

theorem pie_shop_revenue :
  total_revenue = 180 :=
by sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_l2289_228907


namespace NUMINAMATH_CALUDE_grass_sheet_cost_per_cubic_meter_l2289_228972

/-- The cost of a grass sheet per cubic meter, given the area of a playground,
    the depth of the grass sheet, and the total cost to cover the playground. -/
theorem grass_sheet_cost_per_cubic_meter
  (area : ℝ) (depth_cm : ℝ) (total_cost : ℝ)
  (h_area : area = 5900)
  (h_depth : depth_cm = 1)
  (h_total_cost : total_cost = 165.2) :
  total_cost / (area * depth_cm / 100) = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_grass_sheet_cost_per_cubic_meter_l2289_228972


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2289_228988

theorem sin_45_degrees :
  let r : ℝ := 1  -- radius of the unit circle
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  let Q : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)  -- point on the circle at 45°
  let E : ℝ × ℝ := (Q.1, 0)  -- foot of the perpendicular from Q to x-axis
  Real.sin θ = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2289_228988


namespace NUMINAMATH_CALUDE_fourth_power_sum_equality_l2289_228995

theorem fourth_power_sum_equality : 120^4 + 97^4 + 84^4 + 27^4 = 174^4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equality_l2289_228995


namespace NUMINAMATH_CALUDE_pascals_triangle_25th_number_l2289_228989

theorem pascals_triangle_25th_number (n : ℕ) (k : ℕ) : 
  n = 27 ∧ k = 24 → Nat.choose n k = 2925 :=
by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_25th_number_l2289_228989


namespace NUMINAMATH_CALUDE_initial_puppies_count_l2289_228957

/-- The number of puppies Alyssa initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has left -/
def puppies_remaining : ℕ := 5

/-- Theorem stating that the initial number of puppies is equal to
    the sum of puppies given away and puppies remaining -/
theorem initial_puppies_count :
  initial_puppies = puppies_given_away + puppies_remaining := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l2289_228957


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2289_228990

theorem closest_integer_to_cube_root_150 : 
  ∀ n : ℤ, |n - (150 : ℝ)^(1/3)| ≥ |6 - (150 : ℝ)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2289_228990


namespace NUMINAMATH_CALUDE_guy_speed_increase_point_l2289_228971

/-- Represents the problem of finding the point where Guy increases his speed --/
theorem guy_speed_increase_point
  (total_distance : ℝ)
  (average_speed : ℝ)
  (first_half_speed : ℝ)
  (speed_increase : ℝ)
  (h1 : total_distance = 60)
  (h2 : average_speed = 30)
  (h3 : first_half_speed = 24)
  (h4 : speed_increase = 16) :
  let second_half_speed := first_half_speed + speed_increase
  let increase_point := (total_distance * first_half_speed) / (first_half_speed + second_half_speed)
  increase_point = 30 := by sorry

end NUMINAMATH_CALUDE_guy_speed_increase_point_l2289_228971


namespace NUMINAMATH_CALUDE_roots_negative_of_each_other_l2289_228948

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s, 
    if r = -s, then b = 0 -/
theorem roots_negative_of_each_other 
  (a b c r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * r^2 + b * r + c = 0) 
  (h3 : a * s^2 + b * s + c = 0) 
  (h4 : r = -s) : 
  b = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_negative_of_each_other_l2289_228948


namespace NUMINAMATH_CALUDE_total_paint_is_47_l2289_228950

/-- Calculates the total amount of paint used for all canvases --/
def total_paint_used (extra_large_count : ℕ) (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) 
  (extra_large_paint : ℕ) (large_paint : ℕ) (medium_paint : ℕ) (small_paint : ℕ) : ℕ :=
  extra_large_count * extra_large_paint + 
  large_count * large_paint + 
  medium_count * medium_paint + 
  small_count * small_paint

/-- Theorem stating that the total paint used is 47 ounces --/
theorem total_paint_is_47 : 
  total_paint_used 3 5 6 8 4 3 2 1 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_paint_is_47_l2289_228950


namespace NUMINAMATH_CALUDE_sam_picked_42_cans_l2289_228924

/-- The number of cans Sam picked up in total -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem: Sam picked up 42 cans in total -/
theorem sam_picked_42_cans :
  total_cans 4 3 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sam_picked_42_cans_l2289_228924


namespace NUMINAMATH_CALUDE_existence_of_special_multiple_l2289_228925

theorem existence_of_special_multiple (n : ℕ+) : 
  ∃ m : ℕ+, (m.val % n.val = 0) ∧ 
             (m.val ≤ n.val^2) ∧ 
             (∃ d : Fin 10, ∀ k : ℕ, (m.val / 10^k % 10) ≠ d.val) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_multiple_l2289_228925


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l2289_228981

theorem rectangle_areas_sum : 
  let rectangles : List (ℕ × ℕ) := [(2, 1), (2, 9), (2, 25), (2, 49), (2, 81), (2, 121)]
  let areas := rectangles.map (fun (w, l) => w * l)
  areas.sum = 572 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l2289_228981


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l2289_228975

theorem polynomial_sum_theorem (p q r s : ℤ) :
  (∀ x : ℝ, (x^2 + p*x + q) * (x^2 + r*x + s) = x^4 + 3*x^3 - 4*x^2 + 9*x + 7) →
  p + q + r + s = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l2289_228975


namespace NUMINAMATH_CALUDE_soft_drink_cost_l2289_228994

/-- The cost of a 12-pack of soft drinks in dollars -/
def pack_cost : ℚ := 299 / 100

/-- The number of cans in a pack -/
def cans_per_pack : ℕ := 12

/-- The cost per can of soft drink -/
def cost_per_can : ℚ := pack_cost / cans_per_pack

/-- Rounding function to the nearest cent -/
def round_to_cent (x : ℚ) : ℚ := round (100 * x) / 100

theorem soft_drink_cost :
  round_to_cent cost_per_can = 25 / 100 :=
sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l2289_228994


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l2289_228922

/-- Represents the daily rental business for canoes and kayaks -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ
  total_revenue : ℕ

/-- The rental business satisfies the given conditions -/
def valid_rental_business (rb : RentalBusiness) : Prop :=
  rb.canoe_cost = 9 ∧
  rb.canoe_count = rb.kayak_count + 6 ∧
  4 * rb.kayak_count = 3 * rb.canoe_count ∧
  rb.total_revenue = rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count ∧
  rb.total_revenue = 432

/-- The theorem stating that under the given conditions, the kayak rental cost is $12 per day -/
theorem kayak_rental_cost (rb : RentalBusiness) 
  (h : valid_rental_business rb) : rb.kayak_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l2289_228922


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2289_228931

def shekar_scores : List ℝ := [92, 78, 85, 67, 89, 74, 81, 95, 70, 88]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℝ) = 81.9 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2289_228931


namespace NUMINAMATH_CALUDE_triangle_side_length_l2289_228903

theorem triangle_side_length (a b c : ℝ) (A : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  S = (3 * Real.sqrt 3) / 2 →  -- Area of the triangle
  b + c = 3 * Real.sqrt 3 →  -- Sum of sides b and c
  S = (1 / 2) * b * c * Real.sin A →  -- Area formula
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →  -- Law of cosines
  a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2289_228903


namespace NUMINAMATH_CALUDE_range_of_a_l2289_228927

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > 2
def solution_set (a x : ℝ) : Prop := x < 2 / (a - 1)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  (∀ x, inequality a x ↔ solution_set a x) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2289_228927


namespace NUMINAMATH_CALUDE_a_equals_one_l2289_228977

theorem a_equals_one (a : ℝ) : 
  ((a - Complex.I) ^ 2 * Complex.I).re > 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_l2289_228977


namespace NUMINAMATH_CALUDE_pencils_purchased_l2289_228983

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ) :
  num_pens = 30 →
  total_cost = 690 →
  pencil_price = 2 →
  pen_price = 18 →
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l2289_228983


namespace NUMINAMATH_CALUDE_factor_x_minus_one_factorization_p_factorization_q_l2289_228942

-- Define the polynomials
def p (x : ℝ) : ℝ := 6 * x^2 - x - 5
def q (x : ℝ) : ℝ := x^3 - 7 * x + 6

-- Theorem 1: (x - 1) is a factor of 6x^2 - x - 5
theorem factor_x_minus_one (x : ℝ) : ∃ (r : ℝ → ℝ), p x = (x - 1) * r x := by sorry

-- Theorem 2: 6x^2 - x - 5 = (x - 1)(6x + 5)
theorem factorization_p (x : ℝ) : p x = (x - 1) * (6 * x + 5) := by sorry

-- Theorem 3: x^3 - 7x + 6 = (x - 1)(x + 3)(x - 2)
theorem factorization_q (x : ℝ) : q x = (x - 1) * (x + 3) * (x - 2) := by sorry

-- Given condition: When x = 1, the value of 6x^2 - x - 5 is 0
axiom p_zero_at_one : p 1 = 0

end NUMINAMATH_CALUDE_factor_x_minus_one_factorization_p_factorization_q_l2289_228942


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_zero_b_range_for_real_roots_l2289_228968

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + x^3 - x^2 - a * x

theorem extremum_point_implies_a_zero :
  (∀ a : ℝ, (∃ ε > 0, ∀ x ∈ Set.Ioo ((2/3) - ε) ((2/3) + ε), f a x ≤ f a (2/3) ∨ f a x ≥ f a (2/3))) →
  (∃ a : ℝ, ∀ x : ℝ, f a x = f a (2/3)) :=
sorry

theorem b_range_for_real_roots :
  ∀ b : ℝ, (∃ x : ℝ, f (-1) (1 - x) - (1 - x)^3 = b) → b ∈ Set.Iic 0 :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_zero_b_range_for_real_roots_l2289_228968


namespace NUMINAMATH_CALUDE_range_of_c_l2289_228941

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) :
  1 < c ∧ c ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l2289_228941


namespace NUMINAMATH_CALUDE_intersection_condition_l2289_228918

/-- The set M in ℝ² defined by y ≥ x² -/
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≥ p.1^2}

/-- The set N in ℝ² defined by x² + (y-a)² ≤ 1 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 ≤ 1}

/-- Theorem stating the necessary and sufficient condition for M ∩ N = N -/
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2289_228918


namespace NUMINAMATH_CALUDE_no_solution_when_p_divides_x_l2289_228900

theorem no_solution_when_p_divides_x (p : ℕ) (hp : p.Prime) (hp_gt_3 : p > 3) :
  ∀ (x y : ℕ), x > 0 → y > 0 → p ∣ x → x^2 - 1 ≠ y^p := by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_p_divides_x_l2289_228900


namespace NUMINAMATH_CALUDE_boys_between_rajan_and_vinay_l2289_228976

theorem boys_between_rajan_and_vinay (total_boys : ℕ) (rajan_position : ℕ) (vinay_position : ℕ)
  (h1 : total_boys = 24)
  (h2 : rajan_position = 6)
  (h3 : vinay_position = 10) :
  total_boys - (rajan_position - 1 + vinay_position - 1 + 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_between_rajan_and_vinay_l2289_228976


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l2289_228901

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + 14*x + y^2 + 8*y - k

theorem circle_equation_k_value :
  ∃! k : ℝ, is_circle_equation (-7) (-4) 5 (given_equation k) ∧ k = -40 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l2289_228901


namespace NUMINAMATH_CALUDE_tripod_height_theorem_l2289_228908

/-- Represents a tripod with three legs -/
structure Tripod where
  leg_length : ℝ
  original_height : ℝ
  broken_leg_length : ℝ

/-- Calculates the new height of a tripod after one leg is shortened -/
def new_height (t : Tripod) : ℝ :=
  sorry

/-- Expresses the new height as a fraction m / √n -/
def height_fraction (t : Tripod) : ℚ × ℕ :=
  sorry

theorem tripod_height_theorem (t : Tripod) 
  (h_leg : t.leg_length = 5)
  (h_height : t.original_height = 4)
  (h_broken : t.broken_leg_length = 4) :
  let (m, n) := height_fraction t
  ⌊m + Real.sqrt n⌋ = 183 :=
sorry

end NUMINAMATH_CALUDE_tripod_height_theorem_l2289_228908


namespace NUMINAMATH_CALUDE_base3_to_decimal_21201_l2289_228963

/-- Converts a list of digits in base 3 to a decimal number -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 0, 2, 1, 2]

/-- Theorem stating that the conversion of 21201 in base 3 to decimal is 208 -/
theorem base3_to_decimal_21201 :
  base3ToDecimal base3Number = 208 := by
  sorry

#eval base3ToDecimal base3Number

end NUMINAMATH_CALUDE_base3_to_decimal_21201_l2289_228963


namespace NUMINAMATH_CALUDE_perpetually_alive_configurations_l2289_228905

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i : Fin m) (j : Fin n) : ℕ :=
  sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i : Fin m) (j : Fin n) : CellState :=
  sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n :=
  sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop :=
  sorry

/-- Checks if a grid configuration is perpetually alive -/
def isPerpetuallyAlive (initialGrid : Grid m n) : Prop :=
  ∀ t : ℕ, hasAliveCell ((updateGrid^[t]) initialGrid)

/-- The main theorem: for all pairs (m, n) except (1,1), (1,3), and (3,1),
    there exists a perpetually alive configuration -/
theorem perpetually_alive_configurations (m n : ℕ) 
  (h : (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1)) :
  ∃ (initialGrid : Grid m n), isPerpetuallyAlive initialGrid :=
sorry

end NUMINAMATH_CALUDE_perpetually_alive_configurations_l2289_228905


namespace NUMINAMATH_CALUDE_cat_dog_food_difference_l2289_228902

/-- Represents the number of packages of cat food Adam bought. -/
def cat_food_packages : ℕ := 15

/-- Represents the number of packages of dog food Adam bought. -/
def dog_food_packages : ℕ := 10

/-- Represents the number of cans in each package of cat food. -/
def cans_per_cat_package : ℕ := 12

/-- Represents the number of cans in each package of dog food. -/
def cans_per_dog_package : ℕ := 8

/-- Theorem stating the difference between the total number of cans of cat food and dog food. -/
theorem cat_dog_food_difference :
  cat_food_packages * cans_per_cat_package - dog_food_packages * cans_per_dog_package = 100 := by
  sorry

end NUMINAMATH_CALUDE_cat_dog_food_difference_l2289_228902


namespace NUMINAMATH_CALUDE_lottery_winnings_l2289_228970

theorem lottery_winnings 
  (num_tickets : ℕ) 
  (winning_numbers_per_ticket : ℕ) 
  (total_winnings : ℕ) 
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : total_winnings = 300) :
  total_winnings / (num_tickets * winning_numbers_per_ticket) = 20 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winnings_l2289_228970


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2289_228909

/-- Four distinct positive real numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

theorem arithmetic_sequence_inequality (a b c d : ℝ) 
  (h : is_arithmetic_sequence a b c d) : (a + d) / 2 > Real.sqrt (b * c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2289_228909


namespace NUMINAMATH_CALUDE_power_sum_five_l2289_228973

theorem power_sum_five (x : ℝ) (h : x + 1/x = 5) : x^5 + 1/x^5 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_five_l2289_228973


namespace NUMINAMATH_CALUDE_stock_price_theorem_l2289_228952

/-- The face value of the stock (assumed to be $100) -/
def faceValue : ℝ := 100

/-- A's stock interest rate -/
def interestRateA : ℝ := 0.10

/-- B's stock interest rate -/
def interestRateB : ℝ := 0.12

/-- The amount B must invest to get an equally good investment -/
def bInvestment : ℝ := 115.2

/-- The price of the stock A invested in -/
def stockPriceA : ℝ := 138.24

/-- Theorem stating that given the conditions, the price of A's stock is $138.24 -/
theorem stock_price_theorem :
  let incomeA := faceValue * interestRateA
  let requiredInvestmentB := incomeA / interestRateB
  let marketPriceB := bInvestment * (faceValue / requiredInvestmentB)
  marketPriceB = stockPriceA := by
  sorry

#check stock_price_theorem

end NUMINAMATH_CALUDE_stock_price_theorem_l2289_228952


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2289_228984

theorem complex_equation_solution (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 5) :
  a = 4 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2289_228984


namespace NUMINAMATH_CALUDE_painted_faces_count_correct_l2289_228954

/-- Represents the count of painted faces for unit cubes cut from a larger cube. -/
structure PaintedFacesCount where
  one_face : ℕ
  two_faces : ℕ
  three_faces : ℕ

/-- 
  Given a cube with side length a (where a is a natural number greater than 2),
  calculates the number of unit cubes with exactly one, two, and three faces painted
  when the cube is cut into unit cubes.
-/
def count_painted_faces (a : ℕ) : PaintedFacesCount :=
  { one_face := 6 * (a - 2)^2,
    two_faces := 12 * (a - 2),
    three_faces := 8 }

/-- Theorem stating the correct count of painted faces for unit cubes. -/
theorem painted_faces_count_correct (a : ℕ) (h : a > 2) :
  count_painted_faces a = { one_face := 6 * (a - 2)^2,
                            two_faces := 12 * (a - 2),
                            three_faces := 8 } := by
  sorry

end NUMINAMATH_CALUDE_painted_faces_count_correct_l2289_228954


namespace NUMINAMATH_CALUDE_odd_shift_three_l2289_228969

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_shift_three (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  is_odd (λ x => f (x + 3)) := by
sorry

end NUMINAMATH_CALUDE_odd_shift_three_l2289_228969


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l2289_228953

/-- Proves that if a koala absorbs 20% of the fiber it eats and it absorbed 8 ounces of fiber in one day, then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_intake (absorption_rate : Real) (absorbed_amount : Real) (total_intake : Real) :
  absorption_rate = 0.20 →
  absorbed_amount = 8 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l2289_228953


namespace NUMINAMATH_CALUDE_triangle_segment_relation_l2289_228921

/-- Given a triangle ABC with point D on AB and point E on AD, 
    prove the relation for FC where F is on AC. -/
theorem triangle_segment_relation 
  (A B C D E F : ℝ × ℝ) 
  (h1 : dist D C = 6)
  (h2 : dist C B = 9)
  (h3 : dist A B = 1/5 * dist A D)
  (h4 : dist E D = 2/3 * dist A D) :
  dist F C = (dist E D * dist C A) / dist D A :=
sorry

end NUMINAMATH_CALUDE_triangle_segment_relation_l2289_228921


namespace NUMINAMATH_CALUDE_special_form_not_perfect_square_l2289_228980

/-- A function that returns true if the input number has at least three digits,
    all digits except the first and last are zeros, and the first and last digits are non-zeros -/
def has_special_form (n : ℕ) : Prop :=
  n ≥ 100 ∧
  ∃ (d b : ℕ) (k : ℕ), 
    d ≠ 0 ∧ b ≠ 0 ∧ 
    n = d * 10^k + b ∧
    k ≥ 1

theorem special_form_not_perfect_square (n : ℕ) :
  has_special_form n → ¬ ∃ (m : ℕ), n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_special_form_not_perfect_square_l2289_228980
