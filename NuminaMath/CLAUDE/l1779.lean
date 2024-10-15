import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l1779_177936

theorem inequality_proof (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1779_177936


namespace NUMINAMATH_CALUDE_unique_square_board_state_l1779_177995

/-- Represents the state of numbers on the board -/
def BoardState := List Nat

/-- The process of replacing a number with its proper divisors -/
def replace_with_divisors (a : Nat) : BoardState :=
  sorry

/-- The full process of repeatedly replacing numbers until no more replacements are possible -/
def process (initial : BoardState) : BoardState :=
  sorry

/-- Theorem: The only natural number N for which the described process
    can result in exactly N^2 numbers on the board is 1 -/
theorem unique_square_board_state (N : Nat) :
  (∃ (final : BoardState), process [N] = final ∧ final.length = N^2) ↔ N = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_square_board_state_l1779_177995


namespace NUMINAMATH_CALUDE_slope_implies_y_coordinate_l1779_177973

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -3/2, then the y-coordinate of Q is -2. -/
theorem slope_implies_y_coordinate (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -2 →
  y₁ = 7 →
  x₂ = 4 →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 →
  y₂ = -2 :=
by sorry

end NUMINAMATH_CALUDE_slope_implies_y_coordinate_l1779_177973


namespace NUMINAMATH_CALUDE_average_height_is_141_l1779_177967

def student_heights : List ℝ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height_is_141 :
  (student_heights.sum / student_heights.length : ℝ) = 141 := by
  sorry

end NUMINAMATH_CALUDE_average_height_is_141_l1779_177967


namespace NUMINAMATH_CALUDE_larger_number_problem_l1779_177993

theorem larger_number_problem (x y : ℕ) 
  (h1 : x * y = 40)
  (h2 : x + y = 13)
  (h3 : Even x ∨ Even y) :
  max x y = 8 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1779_177993


namespace NUMINAMATH_CALUDE_transformed_roots_l1779_177960

theorem transformed_roots (p : ℝ) (α β : ℝ) : 
  (3 * α^2 + 4 * α + p = 0) → 
  (3 * β^2 + 4 * β + p = 0) → 
  ((α / 3 - 2)^2 + 16 * (α / 3 - 2) + (60 + 3 * p) = 0) ∧
  ((β / 3 - 2)^2 + 16 * (β / 3 - 2) + (60 + 3 * p) = 0) := by
  sorry

end NUMINAMATH_CALUDE_transformed_roots_l1779_177960


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1779_177968

/-- The length of the major axis of the ellipse 2x^2 + y^2 = 8 is 4√2 -/
theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + y^2 = 8}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ ellipse ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    2 * a = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1779_177968


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l1779_177997

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 221 → ∀ a b : ℕ, a^2 - b^2 = 221 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 229 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l1779_177997


namespace NUMINAMATH_CALUDE_martins_trip_distance_l1779_177922

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that Martin's trip distance is 72.0 miles -/
theorem martins_trip_distance :
  let speed : ℝ := 12.0
  let time : ℝ := 6.0
  distance speed time = 72.0 := by
sorry

end NUMINAMATH_CALUDE_martins_trip_distance_l1779_177922


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_in_set_l1779_177915

-- Define the set of numbers
def number_set : Set ℝ := {0, 1.414, Real.sqrt 2, 1/3}

-- Define irrationality
def is_irrational (x : ℝ) : Prop := ∀ p q : ℤ, q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- Theorem statement
theorem sqrt_two_irrational_in_set : 
  ∃ x ∈ number_set, is_irrational x ∧ x = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_in_set_l1779_177915


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1779_177904

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2/3 ∧ x₂ = 2 ∧ 3*x₁^2 - 8*x₁ + 4 = 0 ∧ 3*x₂^2 - 8*x₂ + 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 4/3 ∧ y₂ = -2 ∧ (2*y₁ - 1)^2 = (y₁ - 3)^2 ∧ (2*y₂ - 1)^2 = (y₂ - 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1779_177904


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1779_177964

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x^2 + 1/x^2 = 2) : x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l1779_177964


namespace NUMINAMATH_CALUDE_sum_remainder_nine_l1779_177942

theorem sum_remainder_nine (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_nine_l1779_177942


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l1779_177962

theorem opposite_reciprocal_expression_value :
  ∀ (a b c : ℤ) (m n : ℚ),
    a = -b →                          -- a and b are opposite numbers
    c = -1 →                          -- c is the smallest negative integer in absolute value
    m * n = 1 →                       -- m and n are reciprocal numbers
    (a + b) / 3 + c^2 - 4 * m * n = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l1779_177962


namespace NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_l1779_177980

def abcd_plus_dcba (a : ℕ) : ℕ := 2222 * a + 12667

theorem gcd_abcd_plus_dcba : 
  Nat.gcd (abcd_plus_dcba 0) (Nat.gcd (abcd_plus_dcba 1) (Nat.gcd (abcd_plus_dcba 2) (abcd_plus_dcba 3))) = 2222 := by
  sorry

end NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_l1779_177980


namespace NUMINAMATH_CALUDE_gcd_of_45_75_105_l1779_177908

theorem gcd_of_45_75_105 : Nat.gcd 45 (Nat.gcd 75 105) = 15 := by sorry

end NUMINAMATH_CALUDE_gcd_of_45_75_105_l1779_177908


namespace NUMINAMATH_CALUDE_m_range_l1779_177931

-- Define the propositions p and q
def p (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) < 0
def q (x : ℝ) : Prop := 1/2 < x ∧ x < 2/3

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x)

-- State the theorem
theorem m_range :
  ∀ m : ℝ, (necessary_but_not_sufficient (p m) q) ↔ (-1/3 ≤ m ∧ m ≤ 3/2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1779_177931


namespace NUMINAMATH_CALUDE_inner_circle_radius_l1779_177900

theorem inner_circle_radius : 
  ∀ r : ℝ,
  (r > 0) →
  (π * (9^2) - π * ((0.75 * r)^2) = 3.6 * (π * 6^2 - π * r^2)) →
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l1779_177900


namespace NUMINAMATH_CALUDE_ceva_theorem_l1779_177946

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (X Y Z : Point)

/-- Represents a line segment -/
structure LineSegment :=
  (A B : Point)

/-- Represents the intersection point of three lines -/
def intersectionPoint (l1 l2 l3 : LineSegment) : Point := sorry

/-- Calculates the ratio of distances from a point to two other points -/
def distanceRatio (P A B : Point) : ℝ := sorry

theorem ceva_theorem (T : Triangle) (X' Y' Z' : Point) (P : Point) :
  let XX' := LineSegment.mk T.X X'
  let YY' := LineSegment.mk T.Y Y'
  let ZZ' := LineSegment.mk T.Z Z'
  P = intersectionPoint XX' YY' ZZ' →
  (distanceRatio P T.X X' + distanceRatio P T.Y Y' + distanceRatio P T.Z Z' = 100) →
  (distanceRatio P T.X X' * distanceRatio P T.Y Y' * distanceRatio P T.Z Z' = 102) := by
  sorry

end NUMINAMATH_CALUDE_ceva_theorem_l1779_177946


namespace NUMINAMATH_CALUDE_probability_not_face_card_l1779_177969

theorem probability_not_face_card (total_cards : ℕ) (red_cards : ℕ) (spades_cards : ℕ)
  (red_face_cards : ℕ) (spades_face_cards : ℕ) :
  total_cards = 52 →
  red_cards = 26 →
  spades_cards = 13 →
  red_face_cards = 6 →
  spades_face_cards = 3 →
  (red_cards + spades_cards - (red_face_cards + spades_face_cards)) / (red_cards + spades_cards) = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_face_card_l1779_177969


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l1779_177951

theorem complex_power_magnitude : Complex.abs ((2 + Complex.I * Real.sqrt 11) ^ 4) = 225 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l1779_177951


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l1779_177996

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 2*x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (0, 1/2]
def open_closed_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1/2}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = open_closed_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l1779_177996


namespace NUMINAMATH_CALUDE_soda_difference_l1779_177975

def julio_orange : ℕ := 4
def julio_grape : ℕ := 7
def mateo_orange : ℕ := 1
def mateo_grape : ℕ := 3
def sophia_orange : ℕ := 6
def sophia_strawberry : ℕ := 5

def orange_soda_volume : ℚ := 2
def grape_soda_volume : ℚ := 2
def sophia_orange_volume : ℚ := 1.5
def sophia_strawberry_volume : ℚ := 2.5

def julio_total : ℚ := julio_orange * orange_soda_volume + julio_grape * grape_soda_volume
def mateo_total : ℚ := mateo_orange * orange_soda_volume + mateo_grape * grape_soda_volume
def sophia_total : ℚ := sophia_orange * sophia_orange_volume + sophia_strawberry * sophia_strawberry_volume

theorem soda_difference :
  (max julio_total (max mateo_total sophia_total)) - (min julio_total (min mateo_total sophia_total)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l1779_177975


namespace NUMINAMATH_CALUDE_trees_to_plant_l1779_177956

/-- The number of trees chopped down in the first half of the year -/
def first_half_trees : ℕ := 200

/-- The number of trees chopped down in the second half of the year -/
def second_half_trees : ℕ := 300

/-- The number of trees to be planted for each tree chopped down -/
def trees_to_plant_ratio : ℕ := 3

/-- Theorem stating the number of trees the company needs to plant -/
theorem trees_to_plant : 
  (first_half_trees + second_half_trees) * trees_to_plant_ratio = 1500 := by
  sorry

end NUMINAMATH_CALUDE_trees_to_plant_l1779_177956


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l1779_177913

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l1779_177913


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l1779_177991

def IsValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, m ≠ n → a m ≠ a n) ∧
  (∀ n : ℕ, a n % a (a n) = 0)

theorem unique_valid_sequence :
  ∀ a : ℕ → ℕ, IsValidSequence a → (∀ n : ℕ, a n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_sequence_l1779_177991


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l1779_177947

/-- The expression is a perfect square when p equals 0.28 -/
theorem expression_is_perfect_square : 
  ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l1779_177947


namespace NUMINAMATH_CALUDE_kindergartners_with_orange_shirts_l1779_177932

-- Define the constants from the problem
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

def yellow_shirt_cost : ℚ := 5
def blue_shirt_cost : ℚ := 56/10
def green_shirt_cost : ℚ := 21/4
def orange_shirt_cost : ℚ := 29/5

def total_spent : ℚ := 2317

-- Theorem to prove
theorem kindergartners_with_orange_shirts :
  (total_spent -
   (first_graders * yellow_shirt_cost +
    second_graders * blue_shirt_cost +
    third_graders * green_shirt_cost)) / orange_shirt_cost = 101 := by
  sorry

end NUMINAMATH_CALUDE_kindergartners_with_orange_shirts_l1779_177932


namespace NUMINAMATH_CALUDE_number_of_possible_a_values_l1779_177981

theorem number_of_possible_a_values : ∃ (S : Finset ℕ),
  (∀ a ∈ S, ∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) ∧
  (∀ a : ℕ, (∃ b c d : ℕ,
    a > b ∧ b > c ∧ c > d ∧
    a + b + c + d = 2060 ∧
    a^2 - b^2 + c^2 - d^2 = 1987) → a ∈ S) ∧
  Finset.card S = 513 :=
sorry

end NUMINAMATH_CALUDE_number_of_possible_a_values_l1779_177981


namespace NUMINAMATH_CALUDE_greater_solution_of_quadratic_l1779_177992

theorem greater_solution_of_quadratic (x : ℝ) : 
  x^2 + 20*x - 96 = 0 → x ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greater_solution_of_quadratic_l1779_177992


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1779_177925

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1779_177925


namespace NUMINAMATH_CALUDE_eleventh_sum_14_l1779_177909

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number has digits that sum to 14 -/
def sum_to_14 (n : ℕ) : Prop := digit_sum n = 14

/-- Returns the nth positive integer whose digits sum to 14 -/
def nth_sum_14 (n : ℕ) : ℕ := sorry

theorem eleventh_sum_14 : nth_sum_14 11 = 149 := by sorry

end NUMINAMATH_CALUDE_eleventh_sum_14_l1779_177909


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1779_177952

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if S_5, S_4, and S_6 form an arithmetic sequence, then q = -2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  2 * S 4 = S 5 + S 6 →  -- S_5, S_4, and S_6 form an arithmetic sequence
  q = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1779_177952


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1779_177984

/-- Calculates the simple interest rate given the principal, interest, and time. -/
def calculate_interest_rate (principal interest : ℕ) (time : ℕ) : ℚ :=
  (interest * 100 : ℚ) / (principal * time)

/-- Proves that the interest rate is 12% given the specified conditions. -/
theorem interest_rate_is_twelve_percent 
  (principal : ℕ) 
  (interest : ℕ) 
  (time : ℕ) 
  (h1 : principal = 875)
  (h2 : interest = 2100)
  (h3 : time = 20) :
  calculate_interest_rate principal interest time = 12 := by
  sorry

#eval calculate_interest_rate 875 2100 20

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1779_177984


namespace NUMINAMATH_CALUDE_roots_product_l1779_177918

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  (lg x)^2 + (lg 5 + lg 7) * (lg x) + (lg 5) * (lg 7) = 0

-- Theorem statement
theorem roots_product (m n : ℝ) : 
  equation m ∧ equation n ∧ m ≠ n → m * n = 1/35 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_l1779_177918


namespace NUMINAMATH_CALUDE_cos_supplementary_angles_l1779_177934

theorem cos_supplementary_angles (α β : Real) (h : α + β = Real.pi) : 
  Real.cos α = Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cos_supplementary_angles_l1779_177934


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1779_177917

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove the values of a, b, and the area of the triangle.
-/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  (a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) ∧
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1779_177917


namespace NUMINAMATH_CALUDE_container_capacity_l1779_177903

theorem container_capacity : ∀ (C : ℝ), 
  (0.3 * C + 18 = 0.75 * C) → C = 40 :=
fun C h => by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1779_177903


namespace NUMINAMATH_CALUDE_prime_factor_count_l1779_177989

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_factor_count (x : ℕ) : 
  is_prime x → 
  (∃ (n : ℕ), 2^22 * x^7 * 11^2 = n ∧ (Nat.factors n).length = 31) → 
  x = 7 :=
sorry

end NUMINAMATH_CALUDE_prime_factor_count_l1779_177989


namespace NUMINAMATH_CALUDE_inverse_composition_l1779_177939

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Condition: f⁻¹(g(x)) = 7x - 4
axiom condition : ∀ x, f_inv (g x) = 7 * x - 4

-- Theorem to prove
theorem inverse_composition :
  g_inv (f 2) = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_l1779_177939


namespace NUMINAMATH_CALUDE_expression_one_l1779_177926

theorem expression_one : 5 * (-2)^2 - (-2)^3 / 4 = 22 := by sorry

end NUMINAMATH_CALUDE_expression_one_l1779_177926


namespace NUMINAMATH_CALUDE_distance_between_4th_and_30th_red_l1779_177953

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the cyclic pattern of lights -/
def lightPattern : List LightColor := 
  [LightColor.Red, LightColor.Red, LightColor.Red, 
   LightColor.Green, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The distance between each light in inches -/
def lightDistance : ℕ := 8

/-- Calculates the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ := sorry

/-- Calculates the distance between two positions in feet -/
def distanceInFeet (pos1 pos2 : ℕ) : ℚ := sorry

/-- Theorem: The distance between the 4th and 30th red light is 41.33 feet -/
theorem distance_between_4th_and_30th_red : 
  distanceInFeet (nthRedLightPosition 4) (nthRedLightPosition 30) = 41.33 := by sorry

end NUMINAMATH_CALUDE_distance_between_4th_and_30th_red_l1779_177953


namespace NUMINAMATH_CALUDE_stuffed_animals_ratio_l1779_177935

/-- Proves the ratio of Kenley's stuffed animals to McKenna's is 2:1 --/
theorem stuffed_animals_ratio :
  let mcKenna : ℕ := 34
  let total : ℕ := 175
  let kenley : ℕ := (total - mcKenna - 5) / 2
  (kenley : ℚ) / mcKenna = 2 := by sorry

end NUMINAMATH_CALUDE_stuffed_animals_ratio_l1779_177935


namespace NUMINAMATH_CALUDE_intersection_right_triangle_l1779_177930

/-- Given a line and a circle that intersect, and the triangle formed by the
    intersection points and the circle's center is right-angled, prove the value of a. -/
theorem intersection_right_triangle (a : ℝ) : 
  -- Line equation
  (∃ x y : ℝ, a * x - y + 6 = 0) →
  -- Circle equation
  (∃ x y : ℝ, (x + 1)^2 + (y - a)^2 = 16) →
  -- Circle center
  let C : ℝ × ℝ := (-1, a)
  -- Intersection points exist
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - A.2 + 6 = 0) ∧ ((A.1 + 1)^2 + (A.2 - a)^2 = 16) ∧
    (a * B.1 - B.2 + 6 = 0) ∧ ((B.1 + 1)^2 + (B.2 - a)^2 = 16)) →
  -- Triangle ABC is right-angled
  (∃ A B : ℝ × ℝ, (A - C) • (B - C) = 0) →
  -- Conclusion
  a = 3 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_right_triangle_l1779_177930


namespace NUMINAMATH_CALUDE_fraction_equality_l1779_177955

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 5) : (a - b) / b = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1779_177955


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1779_177933

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 5 * x + 1 = 0
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 - x^2 = 0

-- Theorem for the solutions of equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 17) / 4 ∧
              x₂ = (5 - Real.sqrt 17) / 4 ∧
              equation1 x₁ ∧
              equation1 x₂ ∧
              ∀ x : ℝ, equation1 x → (x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the solutions of equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧
              x₂ = 1 ∧
              equation2 x₁ ∧
              equation2 x₂ ∧
              ∀ x : ℝ, equation2 x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1779_177933


namespace NUMINAMATH_CALUDE_five_objects_two_groups_l1779_177985

/-- The number of ways to partition n indistinguishable objects into k indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to partition 5 indistinguishable objects into 2 indistinguishable groups -/
theorem five_objects_two_groups : partition_count 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_five_objects_two_groups_l1779_177985


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1779_177958

theorem quadratic_equation_solution (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1779_177958


namespace NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l1779_177907

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle (n : ℕ) : (4^n) % 10 = if n % 2 = 0 then 6 else 4 := by sorry

theorem seventeen_power_seventeen_odd : 17^17 % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l1779_177907


namespace NUMINAMATH_CALUDE_min_value_problem_l1779_177976

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 6*x*y - 1 = 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 6*a*b - 1 = 0 → x + 2*y ≤ a + 2*b) ∧ 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 6*a*b - 1 = 0 ∧ x + 2*y = a + 2*b) ∧
  x + 2*y = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1779_177976


namespace NUMINAMATH_CALUDE_chocolate_box_pieces_l1779_177920

theorem chocolate_box_pieces (initial_boxes : ℕ) (given_away : ℕ) (remaining_pieces : ℕ) :
  initial_boxes = 7 →
  given_away = 3 →
  remaining_pieces = 16 →
  ∃ (pieces_per_box : ℕ), 
    pieces_per_box * (initial_boxes - given_away) = remaining_pieces ∧
    pieces_per_box = 4 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_box_pieces_l1779_177920


namespace NUMINAMATH_CALUDE_irrational_plus_five_iff_irrational_l1779_177957

theorem irrational_plus_five_iff_irrational (a : ℝ) :
  Irrational a ↔ Irrational (a + 5) :=
sorry

end NUMINAMATH_CALUDE_irrational_plus_five_iff_irrational_l1779_177957


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l1779_177983

variable (A B C a b c : ℝ)

-- Define the triangle ABC
def is_triangle (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given condition
def given_condition (A B C a b c : ℝ) : Prop :=
  (2 * a - c) / b = Real.cos C / Real.cos B

-- Theorem statement
theorem max_sum_of_sides 
  (h_triangle : is_triangle A B C a b c)
  (h_condition : given_condition A B C a b c)
  (h_b : b = 4) :
  ∃ (max : ℝ), max = 8 ∧ a + c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l1779_177983


namespace NUMINAMATH_CALUDE_twenty_bulb_series_string_possibilities_l1779_177972

/-- Represents a string of decorative lights -/
structure LightString where
  num_bulbs : ℕ
  is_series : Bool

/-- Calculates the number of ways a light string can be non-functioning -/
def non_functioning_possibilities (ls : LightString) : ℕ :=
  if ls.is_series then 2^ls.num_bulbs - 1 else 0

/-- Theorem stating the number of non-functioning possibilities for a specific light string -/
theorem twenty_bulb_series_string_possibilities :
  ∃ (ls : LightString), ls.num_bulbs = 20 ∧ ls.is_series = true ∧ non_functioning_possibilities ls = 2^20 - 1 :=
sorry

end NUMINAMATH_CALUDE_twenty_bulb_series_string_possibilities_l1779_177972


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l1779_177986

/-- Given a man's speed against a current and the speed of the current,
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed with the current is 15 km/hr. -/
theorem mans_speed_with_current_is_15
  (speed_against_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_against_current = 8.6)
  (h2 : current_speed = 3.2) :
  mans_speed_with_current speed_against_current current_speed = 15 := by
  sorry

#eval mans_speed_with_current 8.6 3.2

end NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l1779_177986


namespace NUMINAMATH_CALUDE_polygon_diagonals_l1779_177924

theorem polygon_diagonals (n : ℕ) (h1 : n * 10 = 360) : n * (n - 3) / 2 = 594 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l1779_177924


namespace NUMINAMATH_CALUDE_subtraction_problem_sum_l1779_177994

theorem subtraction_problem_sum (K L M N : ℕ) : 
  K < 10 → L < 10 → M < 10 → N < 10 →
  6000 + 100 * K + L - (900 + N) = 2011 →
  K + L + M + N = 17 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_sum_l1779_177994


namespace NUMINAMATH_CALUDE_exact_power_pair_l1779_177970

theorem exact_power_pair : 
  ∀ (a b : ℕ), 
  (∀ (n : ℕ), ∃ (c : ℕ), a^n + b^n = c^(n+1)) → 
  (a = 2 ∧ b = 2) := by
sorry

end NUMINAMATH_CALUDE_exact_power_pair_l1779_177970


namespace NUMINAMATH_CALUDE_max_discarded_grapes_l1779_177982

theorem max_discarded_grapes (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 7 → n ≠ 7 * (q + 1) + r :=
by sorry

end NUMINAMATH_CALUDE_max_discarded_grapes_l1779_177982


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1779_177949

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_power (n : ℤ) : i^n = 1 ↔ 4 ∣ n :=
sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1779_177949


namespace NUMINAMATH_CALUDE_x_value_l1779_177902

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 21 ∧ x = 49 := by sorry

end NUMINAMATH_CALUDE_x_value_l1779_177902


namespace NUMINAMATH_CALUDE_total_donation_is_375_l1779_177948

/- Define the donation amounts for each company -/
def foster_farms_donation : ℕ := 45
def american_summits_donation : ℕ := 2 * foster_farms_donation
def hormel_donation : ℕ := 3 * foster_farms_donation
def boudin_butchers_donation : ℕ := hormel_donation / 3
def del_monte_foods_donation : ℕ := american_summits_donation - 30

/- Define the total donation -/
def total_donation : ℕ := 
  foster_farms_donation + 
  american_summits_donation + 
  hormel_donation + 
  boudin_butchers_donation + 
  del_monte_foods_donation

/- Theorem stating that the total donation is 375 -/
theorem total_donation_is_375 : total_donation = 375 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_is_375_l1779_177948


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l1779_177945

theorem largest_prime_divisor_to_test (n : ℕ) (h : 800 ≤ n ∧ n ≤ 850) :
  (∀ p : ℕ, p ≤ 29 → Prime p → ¬(p ∣ n)) → Prime n :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l1779_177945


namespace NUMINAMATH_CALUDE_set_difference_M_N_l1779_177921

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem set_difference_M_N : M \ N = {1, 5} := by sorry

end NUMINAMATH_CALUDE_set_difference_M_N_l1779_177921


namespace NUMINAMATH_CALUDE_trig_system_solution_l1779_177971

theorem trig_system_solution (x y : ℝ) 
  (h1 : Real.tan x * Real.tan y = 1/6)
  (h2 : Real.sin x * Real.sin y = 1/(5 * Real.sqrt 2)) :
  Real.cos (x + y) = 1/Real.sqrt 2 ∧ 
  Real.cos (x - y) = 7/(5 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_trig_system_solution_l1779_177971


namespace NUMINAMATH_CALUDE_remainder_is_zero_l1779_177914

-- Define the given binary number
def binary_num : Nat := 857  -- 1101011001₂ in decimal

-- Theorem statement
theorem remainder_is_zero : (binary_num + 3) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_zero_l1779_177914


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l1779_177999

-- Define the shapes
inductive Shape
  | RegularPentagon
  | Square
  | Trapezoid
  | IsoscelesTriangle

-- Define the stability property
def is_stable (s : Shape) : Prop :=
  match s with
  | Shape.RegularPentagon => false
  | Shape.Square => false
  | Shape.Trapezoid => false
  | Shape.IsoscelesTriangle => true

-- Theorem statement
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, is_stable s → s = Shape.IsoscelesTriangle :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l1779_177999


namespace NUMINAMATH_CALUDE_cube_root_of_neg_125_l1779_177943

theorem cube_root_of_neg_125 : ∃ x : ℝ, x^3 = -125 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_neg_125_l1779_177943


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1779_177998

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^2 = y^2 + 2*y + 13 ↔ (x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5) := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1779_177998


namespace NUMINAMATH_CALUDE_blue_balls_in_jar_l1779_177940

theorem blue_balls_in_jar (total_balls : ℕ) (removed_blue : ℕ) (prob_after : ℚ) : 
  total_balls = 25 → 
  removed_blue = 5 → 
  prob_after = 1/5 → 
  ∃ initial_blue : ℕ, 
    initial_blue = 9 ∧ 
    (initial_blue - removed_blue : ℚ) / (total_balls - removed_blue : ℚ) = prob_after :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_in_jar_l1779_177940


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1779_177928

-- Define the interest rates for each year
def r1 : ℝ := 0.08
def r2 : ℝ := 0.10
def r3 : ℝ := 0.12
def r4 : ℝ := 0.09
def r5 : ℝ := 0.11

-- Define the total compound interest
def total_interest : ℝ := 4016.25

-- Define the compound interest factor
def compound_factor : ℝ := (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- State the theorem
theorem principal_amount_proof :
  ∃ P : ℝ, P * (compound_factor - 1) = total_interest ∧ 
  abs (P - 7065.84) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l1779_177928


namespace NUMINAMATH_CALUDE_smallest_n_is_25_l1779_177901

/-- Represents a student's answers as a 5-tuple of integers from 1 to 4 -/
def Answer := Fin 5 → Fin 4

/-- The set of all possible answer patterns satisfying the modular constraint -/
def S : Set Answer :=
  {a | (a 0).val + (a 1).val + (a 2).val + (a 3).val + (a 4).val ≡ 0 [MOD 4]}

/-- The number of students -/
def num_students : ℕ := 2000

/-- The function that checks if two answers differ in at least two positions -/
def differ_in_two (a b : Answer) : Prop :=
  ∃ i j, i ≠ j ∧ a i ≠ b i ∧ a j ≠ b j

/-- The theorem to be proved -/
theorem smallest_n_is_25 :
  ∀ f : Fin num_students → Answer,
  ∃ n : ℕ, n = 25 ∧
  (∀ subset : Fin n → Fin num_students,
   ∃ a b c d : Fin n,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) ∧
  (∀ m : ℕ, m < 25 →
   ∃ f : Fin num_students → Answer,
   ∀ subset : Fin m → Fin num_students,
   ¬∃ a b c d : Fin m,
   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   differ_in_two (f (subset a)) (f (subset b)) ∧
   differ_in_two (f (subset a)) (f (subset c)) ∧
   differ_in_two (f (subset a)) (f (subset d)) ∧
   differ_in_two (f (subset b)) (f (subset c)) ∧
   differ_in_two (f (subset b)) (f (subset d)) ∧
   differ_in_two (f (subset c)) (f (subset d))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_25_l1779_177901


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_l1779_177988

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_l1779_177988


namespace NUMINAMATH_CALUDE_collision_count_l1779_177966

/-- The number of collisions between two groups of balls moving in opposite directions in a trough with a wall -/
def totalCollisions (n m : ℕ) : ℕ :=
  n * m + n * (n - 1) / 2

/-- Theorem stating that the total number of collisions for 20 balls moving towards a wall
    and 16 balls moving in the opposite direction is 510 -/
theorem collision_count : totalCollisions 20 16 = 510 := by
  sorry

end NUMINAMATH_CALUDE_collision_count_l1779_177966


namespace NUMINAMATH_CALUDE_crayons_per_pack_l1779_177959

theorem crayons_per_pack (num_packs : ℕ) (extra_crayons : ℕ) (total_crayons : ℕ) 
  (h1 : num_packs = 4)
  (h2 : extra_crayons = 6)
  (h3 : total_crayons = 46) :
  ∃ (crayons_per_pack : ℕ), 
    crayons_per_pack * num_packs + extra_crayons = total_crayons ∧ 
    crayons_per_pack = 10 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l1779_177959


namespace NUMINAMATH_CALUDE_product_pqr_l1779_177912

theorem product_pqr (p q r : ℤ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 26)
  (h_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 360 / (p * q * r) = 1) :
  p * q * r = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_pqr_l1779_177912


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l1779_177916

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male or female -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_grandchildren_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l1779_177916


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1779_177990

theorem fraction_evaluation : 
  (⌈(21 / 8 : ℚ) - ⌈(35 / 21 : ℚ)⌉⌉ : ℚ) / 
  (⌈(35 / 8 : ℚ) + ⌈(8 * 21 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1779_177990


namespace NUMINAMATH_CALUDE_total_miles_four_weeks_eq_272_l1779_177963

/-- Calculates the total miles Vins rides in a four-week period -/
def total_miles_four_weeks : ℕ :=
  let library_distance : ℕ := 6
  let school_distance : ℕ := 5
  let friend_distance : ℕ := 8
  let extra_return_distance : ℕ := 1
  let friend_shortcut : ℕ := 2
  let library_days_per_week : ℕ := 3
  let school_days_per_week : ℕ := 2
  let friend_visits_per_four_weeks : ℕ := 2
  let weeks : ℕ := 4

  let library_miles_per_week := (library_distance + library_distance + extra_return_distance) * library_days_per_week
  let school_miles_per_week := (school_distance + school_distance + extra_return_distance) * school_days_per_week
  let friend_miles_per_four_weeks := (friend_distance + friend_distance - friend_shortcut) * friend_visits_per_four_weeks

  (library_miles_per_week + school_miles_per_week) * weeks + friend_miles_per_four_weeks

theorem total_miles_four_weeks_eq_272 : total_miles_four_weeks = 272 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_four_weeks_eq_272_l1779_177963


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1779_177987

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * z = 1 + Complex.I) :
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1779_177987


namespace NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1779_177910

/-- The measure of an interior angle of a regular pentagon in degrees -/
def regular_pentagon_angle : ℝ := 108

/-- The measure of an interior angle of a regular triangle in degrees -/
def regular_triangle_angle : ℝ := 60

/-- Theorem: The sum of angles formed by two adjacent sides of a regular pentagon 
    and one side of a regular triangle that share a vertex is 168 degrees -/
theorem pentagon_triangle_angle_sum : 
  regular_pentagon_angle + regular_triangle_angle = 168 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_angle_sum_l1779_177910


namespace NUMINAMATH_CALUDE_function_minimum_value_l1779_177950

/-- Given a function f(x) = (ax + b) / (x^2 + 4) that attains a maximum value of 1 at x = -1,
    prove that the minimum value of f(x) is -1/4 -/
theorem function_minimum_value (a b : ℝ) :
  let f := fun x : ℝ => (a * x + b) / (x^2 + 4)
  (f (-1) = 1) →
  (∃ x₀, ∀ x, f x ≥ f x₀) →
  (∃ x₁, f x₁ = -1/4 ∧ ∀ x, f x ≥ -1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_value_l1779_177950


namespace NUMINAMATH_CALUDE_triangle_condition_right_triangle_condition_l1779_177977

/-- Given vectors in 2D space -/
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -(4 + m))

/-- Vector subtraction -/
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Condition for three points to form a triangle -/
def forms_triangle (m : ℝ) : Prop :=
  let AB := vec_sub OB OA
  let AC := vec_sub (OC m) OA
  let BC := vec_sub (OC m) OB
  AB.1 / AB.2 ≠ AC.1 / AC.2 ∧ AB.1 / AB.2 ≠ BC.1 / BC.2 ∧ AC.1 / AC.2 ≠ BC.1 / BC.2

/-- Theorem: Condition for A, B, and C to form a triangle -/
theorem triangle_condition : ∀ m : ℝ, forms_triangle m ↔ m ≠ -1 := by sorry

/-- Theorem: Condition for ABC to be a right triangle with angle A as the right angle -/
theorem right_triangle_condition : 
  ∀ m : ℝ, dot_product (vec_sub OB OA) (vec_sub (OC m) OA) = 0 ↔ m = 3/2 := by sorry

end NUMINAMATH_CALUDE_triangle_condition_right_triangle_condition_l1779_177977


namespace NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l1779_177961

/-- A cube is represented by its 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → Point)

/-- A tetrahedron is represented by its 4 vertices -/
structure Tetrahedron :=
  (vertices : Fin 4 → Point)

/-- Function to check if a set of 4 vertices forms a valid tetrahedron -/
def is_valid_tetrahedron (c : Cube) (t : Tetrahedron) : Prop :=
  sorry

/-- The number of valid tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons (c : Cube) : ℕ :=
  sorry

/-- Theorem stating that the number of tetrahedrons formed from a cube's vertices is 58 -/
theorem num_tetrahedrons_in_cube (c : Cube) : num_tetrahedrons c = 58 :=
  sorry

end NUMINAMATH_CALUDE_num_tetrahedrons_in_cube_l1779_177961


namespace NUMINAMATH_CALUDE_total_amount_l1779_177965

-- Define the amounts received by A, B, and C
variable (A B C : ℝ)

-- Define the conditions
axiom condition1 : A = (1/3) * (B + C)
axiom condition2 : B = (2/7) * (A + C)
axiom condition3 : A = B + 20

-- Define the total amount
def total : ℝ := A + B + C

-- Theorem statement
theorem total_amount : total A B C = 720 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_l1779_177965


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_range_l1779_177944

/-- An isosceles triangle with perimeter 20 and base length x -/
structure IsoscelesTriangle where
  base : ℝ
  perimeter : ℝ
  is_isosceles : perimeter = 20
  base_definition : base > 0

/-- The range of possible base lengths for an isosceles triangle with perimeter 20 -/
theorem isosceles_triangle_base_range (t : IsoscelesTriangle) :
  5 < t.base ∧ t.base < 10 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_base_range_l1779_177944


namespace NUMINAMATH_CALUDE_fraction_sum_lower_bound_sum_lower_bound_l1779_177923

-- Part 1
theorem fraction_sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  1 / a + 1 / (b + 1) ≥ 4 / 5 := by sorry

-- Part 2
theorem sum_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) :
  a + b ≥ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_lower_bound_sum_lower_bound_l1779_177923


namespace NUMINAMATH_CALUDE_probability_two_white_balls_is_one_fifth_l1779_177954

/-- The probability of drawing two white balls from a box containing 7 white balls
    and 8 black balls, when drawing two balls at random without replacement. -/
def probability_two_white_balls : ℚ :=
  let total_balls : ℕ := 7 + 8
  let white_balls : ℕ := 7
  (Nat.choose white_balls 2 : ℚ) / (Nat.choose total_balls 2 : ℚ)

/-- Theorem stating that the probability of drawing two white balls is 1/5. -/
theorem probability_two_white_balls_is_one_fifth :
  probability_two_white_balls = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_white_balls_is_one_fifth_l1779_177954


namespace NUMINAMATH_CALUDE_class_size_l1779_177941

/-- The position of Xiao Ming from the front of the line -/
def position_from_front : ℕ := 23

/-- The position of Xiao Ming from the back of the line -/
def position_from_back : ℕ := 23

/-- The total number of students in the class -/
def total_students : ℕ := position_from_front + position_from_back - 1

theorem class_size :
  total_students = 45 :=
sorry

end NUMINAMATH_CALUDE_class_size_l1779_177941


namespace NUMINAMATH_CALUDE_prob_sum_gt_8_is_correct_l1779_177937

/-- The probability of getting a sum greater than 8 when tossing two dice -/
def prob_sum_gt_8 : ℚ :=
  5 / 18

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to get a sum of 8 or less when tossing two dice -/
def ways_sum_le_8 : ℕ := 26

/-- Theorem: The probability of getting a sum greater than 8 when tossing two dice is 5/18 -/
theorem prob_sum_gt_8_is_correct : prob_sum_gt_8 = 1 - (ways_sum_le_8 : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_gt_8_is_correct_l1779_177937


namespace NUMINAMATH_CALUDE_kitchen_width_proof_l1779_177929

/-- Proves that the width of a rectangular kitchen floor is 8 inches, given the specified conditions. -/
theorem kitchen_width_proof (tile_area : ℝ) (kitchen_length : ℝ) (total_tiles : ℕ) 
  (h1 : tile_area = 6)
  (h2 : kitchen_length = 72)
  (h3 : total_tiles = 96) :
  (tile_area * total_tiles) / kitchen_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_width_proof_l1779_177929


namespace NUMINAMATH_CALUDE_f_value_at_2_l1779_177919

def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1779_177919


namespace NUMINAMATH_CALUDE_unique_number_property_l1779_177906

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 2 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1779_177906


namespace NUMINAMATH_CALUDE_red_apples_count_l1779_177927

/-- The number of apples produced by each tree -/
def applesPerTree : ℕ := 20

/-- The percentage of red apples on the first tree -/
def firstTreeRedPercentage : ℚ := 40 / 100

/-- The percentage of red apples on the second tree -/
def secondTreeRedPercentage : ℚ := 50 / 100

/-- The total number of red apples from both trees -/
def totalRedApples : ℕ := 18

theorem red_apples_count :
  ⌊(firstTreeRedPercentage * applesPerTree : ℚ)⌋ +
  ⌊(secondTreeRedPercentage * applesPerTree : ℚ)⌋ = totalRedApples :=
sorry

end NUMINAMATH_CALUDE_red_apples_count_l1779_177927


namespace NUMINAMATH_CALUDE_checkers_inequality_l1779_177911

theorem checkers_inequality (n : ℕ) (A B : ℕ) : A ≤ 3 * B :=
  by
  -- Assume n is the number of black checkers (equal to the number of white checkers)
  -- A is the number of triples with white majority
  -- B is the number of triples with black majority
  sorry

end NUMINAMATH_CALUDE_checkers_inequality_l1779_177911


namespace NUMINAMATH_CALUDE_polynomial_root_comparison_l1779_177905

theorem polynomial_root_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≤ a₂) (h2 : a₂ ≤ a₃) 
  (h3 : b₁ ≤ b₂) (h4 : b₂ ≤ b₃) 
  (h5 : a₁ + a₂ + a₃ = b₁ + b₂ + b₃) 
  (h6 : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃) 
  (h7 : a₁ ≤ b₁) : 
  a₃ ≤ b₃ := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_comparison_l1779_177905


namespace NUMINAMATH_CALUDE_bridge_length_l1779_177979

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1779_177979


namespace NUMINAMATH_CALUDE_five_students_three_colleges_l1779_177974

/-- The number of ways for students to apply to colleges -/
def applicationWays (numStudents : ℕ) (numColleges : ℕ) : ℕ :=
  numColleges ^ numStudents

/-- Theorem: 5 students applying to 3 colleges results in 3^5 different ways -/
theorem five_students_three_colleges : 
  applicationWays 5 3 = 3^5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_colleges_l1779_177974


namespace NUMINAMATH_CALUDE_cd_length_calculation_l1779_177938

theorem cd_length_calculation : 
  let first_cd_length : ℝ := 1.5
  let second_cd_length : ℝ := 1.5
  let third_cd_length : ℝ := 2 * first_cd_length
  first_cd_length + second_cd_length + third_cd_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_length_calculation_l1779_177938


namespace NUMINAMATH_CALUDE_salary_ratio_proof_l1779_177978

/-- Proves that the ratio of Shyam's monthly salary to Abhinav's monthly salary is 2:1 -/
theorem salary_ratio_proof (ram_salary shyam_salary abhinav_annual_salary : ℕ) : 
  ram_salary = 25600 →
  abhinav_annual_salary = 192000 →
  10 * ram_salary = 8 * shyam_salary →
  ∃ (k : ℕ), shyam_salary = k * (abhinav_annual_salary / 12) →
  shyam_salary / (abhinav_annual_salary / 12) = 2 := by
  sorry

end NUMINAMATH_CALUDE_salary_ratio_proof_l1779_177978
