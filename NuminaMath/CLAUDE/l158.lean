import Mathlib

namespace NUMINAMATH_CALUDE_concatNaturalsDecimal_irrational_l158_15863

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point -/
def concatNaturalsDecimal : ℝ :=
  sorry  -- Definition of the decimal (implementation details omitted)

/-- The infinite decimal formed by concatenating all natural numbers in order after the decimal point is irrational -/
theorem concatNaturalsDecimal_irrational : Irrational concatNaturalsDecimal := by
  sorry

end NUMINAMATH_CALUDE_concatNaturalsDecimal_irrational_l158_15863


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l158_15804

/-- The parabola y² = 4x in the cartesian plane -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- A line in the cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The intersection points of a line with the parabola -/
def intersection (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ Parabola ∧ p.1 = l.slope * p.2 + l.intercept}

/-- The dot product of two points in ℝ² -/
def dot_product (p q : ℝ × ℝ) : ℝ :=
  p.1 * q.1 + p.2 * q.2

theorem line_passes_through_fixed_point (l : Line) 
    (h_distinct : ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ intersection l ∧ B ∈ intersection l)
    (h_dot_product : ∃ A B : ℝ × ℝ, A ∈ intersection l ∧ B ∈ intersection l ∧ 
                     dot_product A B = -4) :
    (2, 0) ∈ {p : ℝ × ℝ | p.1 = l.slope * p.2 + l.intercept} :=
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l158_15804


namespace NUMINAMATH_CALUDE_mode_of_data_l158_15864

def data : List Nat := [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data :
  mode data = 5 := by sorry

end NUMINAMATH_CALUDE_mode_of_data_l158_15864


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l158_15898

-- Define the displacement function
def h (t : ℝ) : ℝ := 1.5 * t - 0.1 * t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 1.5 - 0.2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3s : v 3 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3s_l158_15898


namespace NUMINAMATH_CALUDE_problem_statement_l158_15866

theorem problem_statement (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l158_15866


namespace NUMINAMATH_CALUDE_city_population_theorem_l158_15810

/-- Given three cities with populations H, L, and C, where H > L and C = H - 5000,
    prove that if L + C = H + C - 5000, then L = H - 5000. -/
theorem city_population_theorem (H L C : ℕ) 
    (h1 : H > L) 
    (h2 : C = H - 5000) 
    (h3 : L + C = H + C - 5000) : 
  L = H - 5000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_theorem_l158_15810


namespace NUMINAMATH_CALUDE_log_greater_than_square_near_zero_l158_15877

theorem log_greater_than_square_near_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x → x < δ → Real.log (1 + x) > x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_greater_than_square_near_zero_l158_15877


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l158_15835

/-- Given a mixture of alcohol and water, prove that adding 10 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 20 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 20
  let initial_ratio : ℝ := 4 / 3
  let final_ratio : ℝ := 4 / 5
  let water_added : ℝ := 10
  let initial_water : ℝ := initial_alcohol / initial_ratio
  let final_water : ℝ := initial_water + water_added
  initial_alcohol / initial_water = initial_ratio ∧
  initial_alcohol / final_water = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l158_15835


namespace NUMINAMATH_CALUDE_inequality_solution_set_l158_15857

theorem inequality_solution_set (x : ℝ) : 
  x^6 - (x + 2) > (x + 2)^3 - x^2 ↔ x < -1 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l158_15857


namespace NUMINAMATH_CALUDE_steven_apple_count_l158_15846

/-- The number of apples Jake has -/
def jake_apples : ℕ := 11

/-- The difference between Jake's and Steven's apple count -/
def apple_difference : ℕ := 3

/-- Proves that Steven has 8 apples given the conditions -/
theorem steven_apple_count : ∃ (steven_apples : ℕ), steven_apples = jake_apples - apple_difference :=
  sorry

end NUMINAMATH_CALUDE_steven_apple_count_l158_15846


namespace NUMINAMATH_CALUDE_compute_expression_l158_15895

theorem compute_expression : 20 * ((150 / 5) - (40 / 8) + (16 / 32) + 3) = 570 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l158_15895


namespace NUMINAMATH_CALUDE_pet_shop_theorem_l158_15887

def pet_shop_problem (parakeet_cost : ℕ) : Prop :=
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  parakeet_cost = 10 → total_cost = 130

theorem pet_shop_theorem : pet_shop_problem 10 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_theorem_l158_15887


namespace NUMINAMATH_CALUDE_prob_event_l158_15861

/-- Represents a standard deck of cards -/
structure Deck :=
  (total : Nat)
  (queens : Nat)
  (jacks : Nat)
  (red : Nat)

/-- Calculates the probability of drawing two queens -/
def prob_two_queens (d : Deck) : Rat :=
  (d.queens * (d.queens - 1)) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing at least one jack -/
def prob_at_least_one_jack (d : Deck) : Rat :=
  1 - (d.total - d.jacks) * (d.total - d.jacks - 1) / (d.total * (d.total - 1))

/-- Calculates the probability of drawing two red cards -/
def prob_two_red (d : Deck) : Rat :=
  (d.red * (d.red - 1)) / (d.total * (d.total - 1))

/-- Theorem stating the probability of the given event -/
theorem prob_event (d : Deck) (h1 : d.total = 52) (h2 : d.queens = 4) (h3 : d.jacks = 4) (h4 : d.red = 26) :
  prob_two_queens d + prob_at_least_one_jack d + prob_two_red d = 89 / 221 := by
  sorry

end NUMINAMATH_CALUDE_prob_event_l158_15861


namespace NUMINAMATH_CALUDE_mean_median_difference_l158_15889

-- Define the frequency distribution of days missed
def days_missed : List (Nat × Nat) := [
  (0, 2),  -- 2 students missed 0 days
  (1, 3),  -- 3 students missed 1 day
  (2, 6),  -- 6 students missed 2 days
  (3, 5),  -- 5 students missed 3 days
  (4, 2),  -- 2 students missed 4 days
  (5, 2)   -- 2 students missed 5 days
]

-- Define the total number of students
def total_students : Nat := 20

-- Theorem statement
theorem mean_median_difference :
  let mean := (days_missed.map (λ (d, f) => d * f)).sum / total_students
  let median := 2  -- The median is 2 days (10th and 11th students both missed 2 days)
  mean - median = 2 / 5 := by sorry


end NUMINAMATH_CALUDE_mean_median_difference_l158_15889


namespace NUMINAMATH_CALUDE_nine_valid_numbers_l158_15831

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≥ 0 ∧ units ≤ 9

/-- Reverses the digits of a two-digit number -/
def reverse (n : TwoDigitNumber) : TwoDigitNumber :=
  ⟨n.units, n.tens, by sorry⟩

/-- Converts a TwoDigitNumber to a natural number -/
def to_nat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- Checks if a natural number is a positive perfect square -/
def is_positive_perfect_square (n : Nat) : Prop :=
  ∃ m : Nat, m > 0 ∧ m * m = n

/-- The main theorem to prove -/
theorem nine_valid_numbers :
  ∃ (S : Finset TwoDigitNumber),
    S.card = 9 ∧
    (∀ n : TwoDigitNumber, n ∈ S ↔
      is_positive_perfect_square (to_nat n - to_nat (reverse n))) ∧
    (∀ n : TwoDigitNumber,
      is_positive_perfect_square (to_nat n - to_nat (reverse n)) →
      n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_nine_valid_numbers_l158_15831


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l158_15815

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 3 = 1

-- Define the focus of the hyperbola
def focus (F : ℝ × ℝ) : Prop := 
  F.1^2 - F.2^2 = 6 ∧ F.2 = 0

-- Define an asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = x ∨ y = -x

-- Theorem statement
theorem distance_focus_to_asymptote :
  ∀ (F : ℝ × ℝ) (x y : ℝ),
  focus F → hyperbola x y → asymptote x y →
  ∃ (d : ℝ), d = Real.sqrt 3 ∧ 
  d = Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) := by sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l158_15815


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l158_15873

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- The number of vertices in a regular decagon -/
def decagonVertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangleVertices : ℕ := 3

/-- Theorem: The number of triangles that can be formed by selecting 3 vertices
    from a 10-vertex polygon is equal to 120 -/
theorem triangles_in_decagon_count :
  Nat.choose decagonVertices triangleVertices = trianglesInDecagon := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l158_15873


namespace NUMINAMATH_CALUDE_existence_of_n_l158_15845

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hcd : c * d = 1) :
  ∃ n : ℕ, (a * b ≤ n^2) ∧ (n^2 ≤ (a + c) * (b + d)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l158_15845


namespace NUMINAMATH_CALUDE_jims_gross_pay_l158_15839

theorem jims_gross_pay (G : ℝ) : 
  G - 0.25 * G - 100 = 740 → G = 1120 := by
  sorry

end NUMINAMATH_CALUDE_jims_gross_pay_l158_15839


namespace NUMINAMATH_CALUDE_sine_of_sum_angle_l158_15867

theorem sine_of_sum_angle (θ : Real) :
  (∃ (x y : Real), x = -3 ∧ y = 4 ∧ 
   x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.sin (θ + π/4) = Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_sine_of_sum_angle_l158_15867


namespace NUMINAMATH_CALUDE_special_triangle_properties_l158_15830

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1 ∧
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C ∧
  (1/2) * t.a * t.b * Real.sin t.C = (1/5) * Real.sin t.C

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.c = 1 ∧ Real.cos t.C = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_properties_l158_15830


namespace NUMINAMATH_CALUDE_M_mod_1000_l158_15883

/-- The number of characters in the string -/
def n : ℕ := 15

/-- The number of A's in the string -/
def a : ℕ := 3

/-- The number of B's in the string -/
def b : ℕ := 5

/-- The number of C's in the string -/
def c : ℕ := 4

/-- The number of D's in the string -/
def d : ℕ := 3

/-- The length of the first section where A's are not allowed -/
def first_section : ℕ := 3

/-- The length of the middle section where B's are not allowed -/
def middle_section : ℕ := 5

/-- The length of the last section where C's are not allowed -/
def last_section : ℕ := 7

/-- The function that calculates the number of permutations -/
def M : ℕ := sorry

theorem M_mod_1000 : M % 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l158_15883


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l158_15884

theorem largest_sum_and_simplification : 
  let sums := [1/3 + 1/7, 1/3 + 1/8, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/4]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l158_15884


namespace NUMINAMATH_CALUDE_complex_modulus_l158_15842

theorem complex_modulus (z : ℂ) (h : z^2 = -4) : Complex.abs (1 + z) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l158_15842


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l158_15872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem no_prime_sum_53 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l158_15872


namespace NUMINAMATH_CALUDE_two_digit_multiplication_l158_15829

theorem two_digit_multiplication (a b c : ℕ) : 
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_l158_15829


namespace NUMINAMATH_CALUDE_intersection_M_N_l158_15843

-- Define set M
def M : Set ℝ := {x | x * (x - 3) < 0}

-- Define set N
def N : Set ℝ := {x | |x| < 2}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l158_15843


namespace NUMINAMATH_CALUDE_base_eight_132_equals_90_l158_15890

def base_eight_to_ten (a b c : Nat) : Nat :=
  a * 8^2 + b * 8^1 + c * 8^0

theorem base_eight_132_equals_90 : base_eight_to_ten 1 3 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_132_equals_90_l158_15890


namespace NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l158_15854

theorem cubic_root_reciprocal_sum (a b c d : ℝ) (p q r : ℝ) : 
  a ≠ 0 → d ≠ 0 →
  (a * p^3 + b * p^2 + c * p + d = 0) →
  (a * q^3 + b * q^2 + c * q + d = 0) →
  (a * r^3 + b * r^2 + c * r + d = 0) →
  (1 / p^2 + 1 / q^2 + 1 / r^2) = (c^2 - 2 * b * d) / d^2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_reciprocal_sum_l158_15854


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l158_15812

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4 * x - 2)^6 = b₆ * x^6 + b₅ * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l158_15812


namespace NUMINAMATH_CALUDE_sylvia_incorrect_fraction_l158_15825

/-- Proves that Sylvia's fraction of incorrect answers is 1/5 given the conditions -/
theorem sylvia_incorrect_fraction (total_questions : ℕ) (sergio_incorrect : ℕ) (difference : ℕ) :
  total_questions = 50 →
  sergio_incorrect = 4 →
  difference = 6 →
  (total_questions - (total_questions - sergio_incorrect - difference)) / total_questions = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sylvia_incorrect_fraction_l158_15825


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l158_15868

theorem inequality_and_equality_condition (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x^2 + y^2 + z^2 + 3 = 2*(x*y + y*z + z*x)) :
  Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) ≥ 3 ∧
  (Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l158_15868


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l158_15874

theorem intersection_point_x_coordinate :
  ∀ (x y : ℝ),
  (y = 3 * x - 15) →
  (3 * x + y = 120) →
  (x = 22.5) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l158_15874


namespace NUMINAMATH_CALUDE_people_in_virginia_l158_15849

/-- The number of people landing in Virginia given the initial passengers, layover changes, and crew members. -/
def peopleInVirginia (initialPassengers : ℕ) (texasOff texasOn ncOff ncOn crewMembers : ℕ) : ℕ :=
  initialPassengers - texasOff + texasOn - ncOff + ncOn + crewMembers

/-- Theorem stating that the number of people landing in Virginia is 67. -/
theorem people_in_virginia :
  peopleInVirginia 124 58 24 47 14 10 = 67 := by
  sorry

end NUMINAMATH_CALUDE_people_in_virginia_l158_15849


namespace NUMINAMATH_CALUDE_max_distance_squared_l158_15847

theorem max_distance_squared (x y : ℝ) : 
  (x + 2)^2 + (y - 5)^2 = 9 → 
  ∃ (max : ℝ), max = 64 ∧ ∀ (x' y' : ℝ), (x' + 2)^2 + (y' - 5)^2 = 9 → (x' - 1)^2 + (y' - 1)^2 ≤ max := by
sorry

end NUMINAMATH_CALUDE_max_distance_squared_l158_15847


namespace NUMINAMATH_CALUDE_earth_surface_utilization_l158_15809

theorem earth_surface_utilization (
  exposed_land : ℚ)
  (inhabitable_land : ℚ)
  (utilized_land : ℚ)
  (h1 : exposed_land = 1 / 3)
  (h2 : inhabitable_land = 2 / 5 * exposed_land)
  (h3 : utilized_land = 3 / 4 * inhabitable_land) :
  utilized_land = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_earth_surface_utilization_l158_15809


namespace NUMINAMATH_CALUDE_conference_games_count_l158_15832

/-- Calculates the number of games in a sports conference season. -/
def conference_games (total_teams : ℕ) (division_size : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let teams_per_division := total_teams / 2
  let intra_games := teams_per_division * (division_size - 1) * intra_division_games
  let inter_games := total_teams * division_size * inter_division_games
  (intra_games + inter_games) / 2

/-- Theorem stating the number of games in the specific conference setup. -/
theorem conference_games_count : 
  conference_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l158_15832


namespace NUMINAMATH_CALUDE_opposite_teal_is_violet_l158_15817

-- Define the colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Violet
  | Teal
  | Pink

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def left : Fin 6 := 2
def right : Fin 6 := 3
def front : Fin 6 := 4
def back : Fin 6 := 5

-- Define the theorem
theorem opposite_teal_is_violet (cube : Cube) :
  (∀ (view : Fin 3), cube top = Color.Violet) →
  (∀ (view : Fin 3), cube left = Color.Orange) →
  (cube front = Color.Blue ∨ cube front = Color.Yellow ∨ cube front = Color.Pink) →
  (∃ (face : Fin 6), cube face = Color.Teal) →
  (∀ (face1 face2 : Fin 6), face1 ≠ face2 → cube face1 ≠ cube face2) →
  (cube bottom = Color.Teal → cube top = Color.Violet) :=
by sorry

end NUMINAMATH_CALUDE_opposite_teal_is_violet_l158_15817


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l158_15865

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (h_prime : Nat.Prime p) (h_gt_three : p > 3) : 
  12 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l158_15865


namespace NUMINAMATH_CALUDE_meal_capacity_for_children_l158_15853

theorem meal_capacity_for_children (adult_capacity : ℕ) (child_capacity : ℕ) (adults_fed : ℕ) : 
  adult_capacity = 70 → child_capacity = 90 → adults_fed = 7 →
  child_capacity - (adults_fed * child_capacity / adult_capacity) = 81 :=
by sorry

end NUMINAMATH_CALUDE_meal_capacity_for_children_l158_15853


namespace NUMINAMATH_CALUDE_crayon_difference_l158_15813

theorem crayon_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  red = 14 → 
  yellow = 32 → 
  yellow = 2 * blue - 6 → 
  blue - red = 5 := by
sorry

end NUMINAMATH_CALUDE_crayon_difference_l158_15813


namespace NUMINAMATH_CALUDE_quadratic_intersection_condition_l158_15852

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 - 4 * m * x + 2 * m - 6

-- Define the condition for intersection with negative x-axis
def intersects_negative_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ f m x = 0

-- Theorem statement
theorem quadratic_intersection_condition :
  ∀ m : ℝ, intersects_negative_x_axis m ↔ (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_condition_l158_15852


namespace NUMINAMATH_CALUDE_valid_drawing_probability_l158_15837

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of red balls in the box -/
def red_balls : ℕ := 1

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls + red_balls

/-- The number of valid drawing sequences -/
def valid_sequences : ℕ := 2

/-- The probability of drawing the balls in a valid sequence -/
def probability : ℚ := valid_sequences / (Nat.factorial total_balls / (Nat.factorial white_balls * Nat.factorial black_balls * Nat.factorial red_balls))

theorem valid_drawing_probability : probability = 1 / 231 := by
  sorry

end NUMINAMATH_CALUDE_valid_drawing_probability_l158_15837


namespace NUMINAMATH_CALUDE_min_value_expression_l158_15888

theorem min_value_expression (x y : ℝ) :
  Real.sqrt (4 + y^2) + Real.sqrt (x^2 + y^2 - 4*x - 4*y + 8) + Real.sqrt (x^2 - 8*x + 17) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l158_15888


namespace NUMINAMATH_CALUDE_water_tower_capacity_l158_15870

/-- The capacity of a water tower serving four neighborhoods --/
theorem water_tower_capacity :
  let first_neighborhood : ℕ := 150
  let second_neighborhood : ℕ := 2 * first_neighborhood
  let third_neighborhood : ℕ := second_neighborhood + 100
  let fourth_neighborhood : ℕ := 350
  first_neighborhood + second_neighborhood + third_neighborhood + fourth_neighborhood = 1200 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_capacity_l158_15870


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l158_15859

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + y ≤ (a : ℕ) + b :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l158_15859


namespace NUMINAMATH_CALUDE_equation_solution_l158_15820

theorem equation_solution : ∃ (q r : ℝ), 
  q ≠ r ∧ 
  q > r ∧
  ((5 * q - 15) / (q^2 + q - 20) = q + 3) ∧
  ((5 * r - 15) / (r^2 + r - 20) = r + 3) ∧
  q - r = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l158_15820


namespace NUMINAMATH_CALUDE_immediate_prepayment_better_l158_15833

variable (S T r : ℝ)

-- S: initial loan balance
-- T: monthly payment amount
-- r: interest rate for the period

-- Assumption: All variables are positive and r is between 0 and 1
axiom S_pos : S > 0
axiom T_pos : T > 0
axiom r_pos : r > 0
axiom r_lt_one : r < 1

-- Define the final balance for immediate prepayment
def final_balance_immediate (S T r : ℝ) : ℝ :=
  S - 2*T + r*S - 0.5*r*T + (0.5*r*S)^2

-- Define the final balance for waiting until the end of the period
def final_balance_waiting (S T r : ℝ) : ℝ :=
  S - 2*T + r*S

-- Theorem: Immediate prepayment results in a lower final balance
theorem immediate_prepayment_better :
  final_balance_immediate S T r < final_balance_waiting S T r :=
sorry

end NUMINAMATH_CALUDE_immediate_prepayment_better_l158_15833


namespace NUMINAMATH_CALUDE_spider_count_l158_15836

theorem spider_count (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 40) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 5 := by
  sorry

end NUMINAMATH_CALUDE_spider_count_l158_15836


namespace NUMINAMATH_CALUDE_miranda_goose_feathers_l158_15891

/-- The number of feathers needed for one pillow -/
def feathers_per_pillow : ℕ := 2 * 300

/-- The number of pillows Miranda can stuff -/
def pillows_stuffed : ℕ := 6

/-- The number of feathers on Miranda's goose -/
def goose_feathers : ℕ := feathers_per_pillow * pillows_stuffed

theorem miranda_goose_feathers : goose_feathers = 3600 := by
  sorry

end NUMINAMATH_CALUDE_miranda_goose_feathers_l158_15891


namespace NUMINAMATH_CALUDE_olivias_dad_spending_l158_15896

theorem olivias_dad_spending (cost_per_meal : ℕ) (number_of_meals : ℕ) (total_cost : ℕ) : 
  cost_per_meal = 7 → number_of_meals = 3 → total_cost = cost_per_meal * number_of_meals → total_cost = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_olivias_dad_spending_l158_15896


namespace NUMINAMATH_CALUDE_sample_size_is_192_l158_15850

/-- Represents the total population in the school survey --/
def total_population : ℕ := 2400

/-- Represents the number of female students in the school --/
def female_students : ℕ := 1000

/-- Represents the number of female students in the sample --/
def female_sample : ℕ := 80

/-- Calculates the sample size based on the given information --/
def sample_size : ℕ := (total_population * female_sample) / female_students

/-- Theorem stating that the sample size is 192 --/
theorem sample_size_is_192 : sample_size = 192 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_192_l158_15850


namespace NUMINAMATH_CALUDE_oliver_good_games_l158_15848

theorem oliver_good_games (total_games bad_games : ℕ) 
  (h1 : total_games = 11) 
  (h2 : bad_games = 5) : 
  total_games - bad_games = 6 := by
  sorry

end NUMINAMATH_CALUDE_oliver_good_games_l158_15848


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_9999_l158_15822

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The sum of factorials from 1 to n -/
def sumFactorials (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + sumFactorials n

theorem units_digit_sum_factorials_9999 :
  unitsDigit (sumFactorials 9999) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_9999_l158_15822


namespace NUMINAMATH_CALUDE_jackson_school_supplies_cost_l158_15821

/-- Calculates the total cost of school supplies for a class, given the number of students,
    item quantities per student, item costs, and a teacher discount. -/
def totalCostOfSupplies (students : ℕ) 
                        (penPerStudent notebookPerStudent binderPerStudent highlighterPerStudent : ℕ)
                        (penCost notebookCost binderCost highlighterCost : ℚ)
                        (teacherDiscount : ℚ) : ℚ :=
  let totalPens := students * penPerStudent
  let totalNotebooks := students * notebookPerStudent
  let totalBinders := students * binderPerStudent
  let totalHighlighters := students * highlighterPerStudent
  let totalCost := totalPens * penCost + totalNotebooks * notebookCost + 
                   totalBinders * binderCost + totalHighlighters * highlighterCost
  totalCost - teacherDiscount

/-- Theorem stating that the total cost of school supplies for Jackson's class is $858.25 -/
theorem jackson_school_supplies_cost : 
  totalCostOfSupplies 45 6 4 2 3 (65/100) (145/100) (480/100) (85/100) 125 = 85825/100 := by
  sorry

end NUMINAMATH_CALUDE_jackson_school_supplies_cost_l158_15821


namespace NUMINAMATH_CALUDE_equation_solution_l158_15878

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ 9*x - 3 = k*x + 14) ↔ (k = 8 ∨ k = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l158_15878


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l158_15869

/-- The minimum distance between two complex numbers on specific circles -/
theorem min_distance_complex_circles :
  ∀ (z w : ℂ),
  Complex.abs (z - (2 + Complex.I)) = 2 →
  Complex.abs (w + (3 + 4 * Complex.I)) = 4 →
  (∀ (z' w' : ℂ),
    Complex.abs (z' - (2 + Complex.I)) = 2 →
    Complex.abs (w' + (3 + 4 * Complex.I)) = 4 →
    Complex.abs (z - w) ≤ Complex.abs (z' - w')) →
  Complex.abs (z - w) = 5 * Real.sqrt 2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l158_15869


namespace NUMINAMATH_CALUDE_parabola_reflection_l158_15885

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -5 * x^2 + 2

-- Define the new parabola
def new_parabola (x : ℝ) : ℝ := 5 * x^2 - 2

-- Theorem statement
theorem parabola_reflection :
  ∀ x : ℝ, new_parabola x = -(original_parabola x) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_reflection_l158_15885


namespace NUMINAMATH_CALUDE_alices_age_l158_15858

theorem alices_age (alice : ℕ) (eve : ℕ) 
  (h1 : alice = 2 * eve) 
  (h2 : alice = eve + 10) : 
  alice = 20 := by
sorry

end NUMINAMATH_CALUDE_alices_age_l158_15858


namespace NUMINAMATH_CALUDE_investment_problem_l158_15893

/-- Proves that the initial investment is $698 given the conditions of Peter and David's investments -/
theorem investment_problem (peter_amount : ℝ) (david_amount : ℝ) (peter_years : ℝ) (david_years : ℝ)
  (h_peter : peter_amount = 815)
  (h_david : david_amount = 854)
  (h_peter_years : peter_years = 3)
  (h_david_years : david_years = 4)
  (h_same_principal : ∃ (P : ℝ), P > 0 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years)) :
  ∃ (P : ℝ), P = 698 ∧ 
    ∃ (r : ℝ), r > 0 ∧ 
      peter_amount = P * (1 + r * peter_years) ∧
      david_amount = P * (1 + r * david_years) :=
sorry


end NUMINAMATH_CALUDE_investment_problem_l158_15893


namespace NUMINAMATH_CALUDE_perfect_square_sum_of_powers_l158_15851

theorem perfect_square_sum_of_powers (x y z : ℕ+) :
  ∃ (k : ℕ), (4:ℕ)^(x:ℕ) + (4:ℕ)^(y:ℕ) + (4:ℕ)^(z:ℕ) = k^2 ↔
  ∃ (b z' : ℕ+), x = 2*b - 1 + z' ∧ y = b + z' ∧ z = z' :=
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_of_powers_l158_15851


namespace NUMINAMATH_CALUDE_three_male_students_probability_l158_15834

theorem three_male_students_probability 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (selection_size : ℕ) 
  (prob_at_least_one_female : ℚ) : 
  total_male = 4 → 
  total_female = 2 → 
  selection_size = 3 → 
  prob_at_least_one_female = 4/5 → 
  (1 : ℚ) - prob_at_least_one_female = 1/5 := by
sorry

end NUMINAMATH_CALUDE_three_male_students_probability_l158_15834


namespace NUMINAMATH_CALUDE_wood_burning_problem_l158_15814

/-- Wood burning problem -/
theorem wood_burning_problem (initial_bundles morning_burned end_bundles : ℕ) 
  (h1 : initial_bundles = 10)
  (h2 : morning_burned = 4)
  (h3 : end_bundles = 3) :
  initial_bundles - morning_burned - end_bundles = 3 :=
by sorry

end NUMINAMATH_CALUDE_wood_burning_problem_l158_15814


namespace NUMINAMATH_CALUDE_min_brilliant_product_l158_15823

/-- A triple of integers (a, b, c) is brilliant if:
    1. a > b > c are prime numbers
    2. a = b + 2c
    3. a + b + c is a perfect square number -/
def is_brilliant (a b c : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
  a > b ∧ b > c ∧
  a = b + 2 * c ∧
  ∃ k, a + b + c = k * k

/-- The minimum value of abc for a brilliant triple (a, b, c) is 35651 -/
theorem min_brilliant_product :
  (∀ a b c : ℕ, is_brilliant a b c → a * b * c ≥ 35651) ∧
  ∃ a b c : ℕ, is_brilliant a b c ∧ a * b * c = 35651 :=
sorry

end NUMINAMATH_CALUDE_min_brilliant_product_l158_15823


namespace NUMINAMATH_CALUDE_plane_points_distance_l158_15876

theorem plane_points_distance (n : ℕ) (P : Fin n → ℝ × ℝ) (Q : ℝ × ℝ) 
  (h_n : n ≥ 12)
  (h_distinct : ∀ i j, i ≠ j → P i ≠ P j ∧ P i ≠ Q) :
  ∃ i : Fin n, ∃ S : Finset (Fin n), 
    S.card ≥ (n / 6 : ℕ) - 1 ∧ 
    (∀ j ∈ S, j ≠ i → dist (P j) (P i) < dist (P i) Q) :=
by sorry

end NUMINAMATH_CALUDE_plane_points_distance_l158_15876


namespace NUMINAMATH_CALUDE_determinant_is_zero_l158_15824

-- Define the polynomial and its roots
variable (p q r : ℝ)
variable (a b c d : ℝ)

-- Define the condition that a, b, c, d are roots of the polynomial
def are_roots (a b c d p q r : ℝ) : Prop :=
  a^4 + 2*a^3 + p*a^2 + q*a + r = 0 ∧
  b^4 + 2*b^3 + p*b^2 + q*b + r = 0 ∧
  c^4 + 2*c^3 + p*c^2 + q*c + r = 0 ∧
  d^4 + 2*d^3 + p*d^2 + q*d + r = 0

-- Define the matrix
def matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  !![a, b, c, d;
     b, c, d, a;
     c, d, a, b;
     d, a, b, c]

-- State the theorem
theorem determinant_is_zero (p q r : ℝ) (a b c d : ℝ) 
  (h : are_roots a b c d p q r) : 
  Matrix.det (matrix a b c d) = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_is_zero_l158_15824


namespace NUMINAMATH_CALUDE_two_integers_sum_l158_15805

theorem two_integers_sum (a b : ℕ) : 
  a > 0 → b > 0 → 
  a * b + a + b = 255 → 
  (Odd a ∨ Odd b) → 
  a < 30 → b < 30 → 
  a + b = 30 := by sorry

end NUMINAMATH_CALUDE_two_integers_sum_l158_15805


namespace NUMINAMATH_CALUDE_relationship_abc_l158_15840

noncomputable def a : ℝ := Real.rpow 0.6 0.6
noncomputable def b : ℝ := Real.rpow 0.6 1.5
noncomputable def c : ℝ := Real.rpow 1.5 0.6

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l158_15840


namespace NUMINAMATH_CALUDE_polynomial_simplification_l158_15818

theorem polynomial_simplification (x : ℝ) :
  3 + 5*x - 7*x^2 - 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 = 9 - x - x^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l158_15818


namespace NUMINAMATH_CALUDE_right_triangle_sides_l158_15807

theorem right_triangle_sides : ∃ (a b c : ℕ), a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l158_15807


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l158_15844

/-- Proves that for a regular polygon with an exterior angle of 36°, the sum of its interior angles is 1440°. -/
theorem regular_polygon_interior_angle_sum (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 36 → 
  n * ext_angle = 360 →
  (n - 2) * 180 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l158_15844


namespace NUMINAMATH_CALUDE_product_one_sum_squares_and_products_inequality_l158_15826

theorem product_one_sum_squares_and_products_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_product_one_sum_squares_and_products_inequality_l158_15826


namespace NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l158_15881

theorem and_false_necessary_not_sufficient_for_or_false (p q : Prop) :
  (¬(p ∧ q) → ¬(p ∨ q)) ∧ ¬(¬(p ∧ q) ↔ ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_and_false_necessary_not_sufficient_for_or_false_l158_15881


namespace NUMINAMATH_CALUDE_x_varies_as_square_of_sin_z_l158_15802

/-- Given that x is directly proportional to the square of y, and y is directly proportional to sin(z),
    prove that x varies as the 2nd power of sin(z). -/
theorem x_varies_as_square_of_sin_z
  (x y z : ℝ)
  (hxy : ∃ k : ℝ, x = k * y^2)
  (hyz : ∃ j : ℝ, y = j * Real.sin z) :
  ∃ m : ℝ, x = m * (Real.sin z)^2 :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_square_of_sin_z_l158_15802


namespace NUMINAMATH_CALUDE_rebus_solution_l158_15894

theorem rebus_solution : ∃! (K I S : Nat), 
  K < 10 ∧ I < 10 ∧ S < 10 ∧
  K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
  100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K := by
  sorry

end NUMINAMATH_CALUDE_rebus_solution_l158_15894


namespace NUMINAMATH_CALUDE_powerjet_pump_volume_l158_15880

/-- The Powerjet pump rate in gallons per hour -/
def pump_rate : ℝ := 350

/-- The time period in hours -/
def time_period : ℝ := 1.5

/-- The total volume of water pumped -/
def total_volume : ℝ := pump_rate * time_period

theorem powerjet_pump_volume : total_volume = 525 := by
  sorry

end NUMINAMATH_CALUDE_powerjet_pump_volume_l158_15880


namespace NUMINAMATH_CALUDE_negation_of_proposition_l158_15892

theorem negation_of_proposition :
  (∀ a b : ℝ, ab > 0 → a > 0) ↔ (∀ a b : ℝ, ab ≤ 0 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l158_15892


namespace NUMINAMATH_CALUDE_find_A_l158_15882

theorem find_A (A B C D : ℝ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (eq1 : 2 * B + B = 12)
  (eq2 : C - B = 5)
  (eq3 : D + C = 12)
  (eq4 : A - D = 5) :
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_find_A_l158_15882


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l158_15879

theorem concentric_circles_area_ratio 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : (π * R^2) / (π * r^2) = 4) : 
  R = 2 * r ∧ R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l158_15879


namespace NUMINAMATH_CALUDE_problem_1_proof_l158_15862

theorem problem_1_proof : (1 : ℝ) - 1^2 + (64 : ℝ)^(1/3) - (-2) * (9 : ℝ)^(1/2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_proof_l158_15862


namespace NUMINAMATH_CALUDE_a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l158_15841

theorem a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq :
  ∃ (a b c d : ℝ),
    (a < b ∧ ¬(a^2 < b^2)) ∧
    (c^2 < d^2 ∧ ¬(c < d)) :=
sorry

end NUMINAMATH_CALUDE_a_lt_b_neither_sufficient_nor_necessary_for_a_sq_lt_b_sq_l158_15841


namespace NUMINAMATH_CALUDE_square_sum_given_condition_l158_15800

theorem square_sum_given_condition (x y : ℝ) :
  (2*x + 1)^2 + |y - 1| = 0 → x^2 + y^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_condition_l158_15800


namespace NUMINAMATH_CALUDE_trevor_eggs_end_wednesday_l158_15827

/-- Represents the number of eggs laid by a chicken on a given day -/
structure ChickenEggs :=
  (monday : ℕ)
  (tuesday : ℕ)
  (wednesday : ℕ)

/-- Represents the egg-laying data for all chickens -/
def chicken_data : List ChickenEggs := [
  ⟨4, 6, 4⟩,  -- Gertrude
  ⟨3, 3, 3⟩,  -- Blanche
  ⟨2, 1, 2⟩,  -- Nancy
  ⟨3, 4, 3⟩,  -- Martha
  ⟨5, 3, 5⟩,  -- Ophelia
  ⟨1, 3, 1⟩,  -- Penelope
  ⟨3, 1, 3⟩,  -- Quinny
  ⟨4, 0, 4⟩   -- Rosie
]

def eggs_eaten_per_day : ℕ := 2
def eggs_dropped_monday : ℕ := 3
def eggs_dropped_wednesday : ℕ := 3

def total_eggs_collected (data : List ChickenEggs) : ℕ :=
  (data.map (·.monday)).sum + (data.map (·.tuesday)).sum + (data.map (·.wednesday)).sum

def eggs_eaten_total (days : ℕ) : ℕ :=
  eggs_eaten_per_day * days

def eggs_dropped_total : ℕ :=
  eggs_dropped_monday + eggs_dropped_wednesday

def eggs_sold (data : List ChickenEggs) : ℕ :=
  (data.map (·.tuesday)).sum / 2

theorem trevor_eggs_end_wednesday :
  total_eggs_collected chicken_data - 
  eggs_eaten_total 3 - 
  eggs_dropped_total - 
  eggs_sold chicken_data = 49 := by
  sorry

end NUMINAMATH_CALUDE_trevor_eggs_end_wednesday_l158_15827


namespace NUMINAMATH_CALUDE_inequality_proof_l158_15801

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ((a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 1) ∧ (a * b + b * c + a * c ≤ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l158_15801


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l158_15855

theorem square_difference_of_sum_and_diff (a b : ℕ+) 
  (h_sum : a + b = 60) 
  (h_diff : a - b = 14) : 
  a^2 - b^2 = 840 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l158_15855


namespace NUMINAMATH_CALUDE_team_selection_count_l158_15808

/-- Represents the number of ways to select a team under given conditions -/
def selectTeam (totalMale totalFemale teamSize : ℕ) 
               (maleCaptains femaleCaptains : ℕ) : ℕ := 
  Nat.choose (totalMale + totalFemale - 1) (teamSize - 1) + 
  Nat.choose (totalMale + totalFemale - maleCaptains - 1) (teamSize - 1) - 
  Nat.choose (totalMale - maleCaptains) (teamSize - 1)

/-- Theorem stating the number of ways to select a team of 5 from 6 male (1 captain) 
    and 4 female (1 captain) athletes, including at least 1 female and a captain -/
theorem team_selection_count : 
  selectTeam 6 4 5 1 1 = 191 := by sorry

end NUMINAMATH_CALUDE_team_selection_count_l158_15808


namespace NUMINAMATH_CALUDE_diagonal_passes_through_800_cubes_l158_15838

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 240 × 360 × 400 rectangular solid passes through 800 unit cubes -/
theorem diagonal_passes_through_800_cubes :
  cubes_passed_by_diagonal 240 360 400 = 800 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_800_cubes_l158_15838


namespace NUMINAMATH_CALUDE_jake_ball_count_l158_15806

/-- The number of balls each person has -/
structure BallCount where
  jake : ℕ
  audrey : ℕ
  charlie : ℕ

/-- The conditions of the problem -/
def problem_conditions (bc : BallCount) : Prop :=
  bc.audrey = bc.jake + 34 ∧
  bc.audrey = 2 * bc.charlie ∧
  bc.charlie + 7 = 41

/-- The theorem to be proved -/
theorem jake_ball_count (bc : BallCount) : 
  problem_conditions bc → bc.jake = 62 := by
  sorry

end NUMINAMATH_CALUDE_jake_ball_count_l158_15806


namespace NUMINAMATH_CALUDE_symmetry_implies_values_and_minimum_l158_15899

/-- A function f(x) that is symmetric about the line x = -1 -/
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-(x + 2)) = f x

/-- The function f(x) = (x^2 - 4)(x^2 + ax + b) -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  (x^2 - 4) * (x^2 + a*x + b)

theorem symmetry_implies_values_and_minimum (a b : ℝ) :
  symmetric_about_neg_one (f a b) →
  (a = 4 ∧ b = 0) ∧
  (∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f a b x ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_and_minimum_l158_15899


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l158_15860

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on (-∞, 0]
def increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Theorem statement
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_inc : increasing_on_neg f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l158_15860


namespace NUMINAMATH_CALUDE_distinct_reals_with_integer_differences_are_integers_l158_15871

theorem distinct_reals_with_integer_differences_are_integers 
  (a b : ℝ) 
  (distinct : a ≠ b) 
  (int_diff : ∀ k : ℕ, ∃ n : ℤ, a^k - b^k = n) : 
  ∃ m n : ℤ, (a : ℝ) = m ∧ (b : ℝ) = n := by
  sorry

end NUMINAMATH_CALUDE_distinct_reals_with_integer_differences_are_integers_l158_15871


namespace NUMINAMATH_CALUDE_characterize_valid_common_differences_l158_15811

/-- A number is interesting if 2018 divides its number of positive divisors -/
def IsInteresting (n : ℕ) : Prop :=
  2018 ∣ (Nat.divisors n).card

/-- An arithmetic progression with first term a and common difference k -/
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ :=
  fun i => a + i * k

/-- The property of k being a valid common difference for an infinite
    arithmetic progression of interesting numbers -/
def IsValidCommonDifference (k : ℕ) : Prop :=
  ∃ a : ℕ, ∀ i : ℕ, IsInteresting (ArithmeticProgression a k i)

/-- The main theorem characterizing valid common differences -/
theorem characterize_valid_common_differences :
  ∀ k : ℕ, k > 0 →
  (IsValidCommonDifference k ↔
    (∃ (m : ℕ) (p : ℕ), m > 0 ∧ Nat.Prime p ∧ k = m * p^1009) ∧
    k ≠ 2^2009) :=
  sorry

end NUMINAMATH_CALUDE_characterize_valid_common_differences_l158_15811


namespace NUMINAMATH_CALUDE_cats_after_sale_l158_15803

/-- The number of cats left after a sale in a pet store -/
theorem cats_after_sale (siamese_cats house_cats sold_cats : ℕ) :
  siamese_cats = 12 →
  house_cats = 20 →
  sold_cats = 20 →
  siamese_cats + house_cats - sold_cats = 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l158_15803


namespace NUMINAMATH_CALUDE_steve_has_dimes_l158_15875

/-- Represents the types of coins in US currency --/
inductive USCoin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a US coin in cents --/
def coin_value (c : USCoin) : ℕ :=
  match c with
  | USCoin.Penny => 1
  | USCoin.Nickel => 5
  | USCoin.Dime => 10
  | USCoin.Quarter => 25

/-- Theorem: Given the conditions, Steve must have 26 dimes --/
theorem steve_has_dimes (total_coins : ℕ) (total_value : ℕ) (majority_coin_count : ℕ)
    (h_total_coins : total_coins = 36)
    (h_total_value : total_value = 310)
    (h_majority_coin_count : majority_coin_count = 26)
    (h_two_types : ∃ (c1 c2 : USCoin), c1 ≠ c2 ∧
      ∃ (n1 n2 : ℕ), n1 + n2 = total_coins ∧
        n1 * coin_value c1 + n2 * coin_value c2 = total_value ∧
        (n1 = majority_coin_count ∨ n2 = majority_coin_count)) :
    ∃ (other_coin : USCoin), other_coin ≠ USCoin.Dime ∧
      majority_coin_count * coin_value USCoin.Dime +
      (total_coins - majority_coin_count) * coin_value other_coin = total_value :=
  sorry

end NUMINAMATH_CALUDE_steve_has_dimes_l158_15875


namespace NUMINAMATH_CALUDE_transformer_min_current_load_l158_15856

def number_of_units : ℕ := 3
def running_current_per_unit : ℕ := 40
def starting_current_multiplier : ℕ := 2

theorem transformer_min_current_load :
  let total_running_current := number_of_units * running_current_per_unit
  let min_starting_current := starting_current_multiplier * total_running_current
  min_starting_current = 240 := by
  sorry

end NUMINAMATH_CALUDE_transformer_min_current_load_l158_15856


namespace NUMINAMATH_CALUDE_jesses_room_width_l158_15819

/-- Proves that the width of Jesse's room is 12 feet -/
theorem jesses_room_width (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) :
  length = 2 →
  tile_area = 4 →
  num_tiles = 6 →
  (length * (tile_area * num_tiles / length : ℝ) = length * 12) :=
by
  sorry

end NUMINAMATH_CALUDE_jesses_room_width_l158_15819


namespace NUMINAMATH_CALUDE_centerville_snail_count_l158_15886

/-- The number of snails removed from Centerville -/
def snails_removed : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def snails_remaining : ℕ := 8278

/-- The original number of snails in Centerville -/
def original_snails : ℕ := snails_removed + snails_remaining

theorem centerville_snail_count : original_snails = 11760 := by
  sorry

end NUMINAMATH_CALUDE_centerville_snail_count_l158_15886


namespace NUMINAMATH_CALUDE_jacket_cost_l158_15828

theorem jacket_cost (shorts_cost total_cost : ℚ) 
  (shorts_eq : shorts_cost = 14.28)
  (total_eq : total_cost = 19.02) :
  total_cost - shorts_cost = 4.74 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_l158_15828


namespace NUMINAMATH_CALUDE_lcm_16_35_l158_15897

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end NUMINAMATH_CALUDE_lcm_16_35_l158_15897


namespace NUMINAMATH_CALUDE_total_daily_allowance_l158_15816

theorem total_daily_allowance (total_students : ℕ) 
  (high_allowance : ℚ) (low_allowance : ℚ) :
  total_students = 60 →
  high_allowance = 6 →
  low_allowance = 4 →
  (2 : ℚ) / 3 * total_students * high_allowance + 
  (1 : ℚ) / 3 * total_students * low_allowance = 320 := by
sorry

end NUMINAMATH_CALUDE_total_daily_allowance_l158_15816
