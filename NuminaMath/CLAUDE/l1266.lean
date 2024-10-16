import Mathlib

namespace NUMINAMATH_CALUDE_xy_fraction_sum_l1266_126695

theorem xy_fraction_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_fraction_sum_l1266_126695


namespace NUMINAMATH_CALUDE_expression_simplification_l1266_126683

theorem expression_simplification (a b : ℝ) (h : a * b ≠ 0) :
  (3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b = -a^2 + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1266_126683


namespace NUMINAMATH_CALUDE_pi_greater_than_314_l1266_126665

theorem pi_greater_than_314 : π > 3.14 := by
  sorry

end NUMINAMATH_CALUDE_pi_greater_than_314_l1266_126665


namespace NUMINAMATH_CALUDE_crow_eating_time_l1266_126677

/-- The time it takes for a crow to eat a fraction of nuts -/
def eat_time (total_fraction : ℚ) (time : ℚ) : ℚ := total_fraction / time

theorem crow_eating_time :
  let quarter_time : ℚ := 5
  let quarter_fraction : ℚ := 1/4
  let fifth_fraction : ℚ := 1/5
  let rate := eat_time quarter_fraction quarter_time
  eat_time fifth_fraction rate = 4 := by sorry

end NUMINAMATH_CALUDE_crow_eating_time_l1266_126677


namespace NUMINAMATH_CALUDE_cube_number_sum_l1266_126643

theorem cube_number_sum : 
  ∀ (n : ℤ),
  (∀ (i : Fin 6), i.val < 6 → ∃ (face : ℤ), face = n + i.val) →
  (∃ (s : ℤ), s % 2 = 1 ∧ 
    (n + (n + 5) = s) ∧ 
    ((n + 1) + (n + 4) = s) ∧ 
    ((n + 2) + (n + 3) = s)) →
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 27) :=
by sorry


end NUMINAMATH_CALUDE_cube_number_sum_l1266_126643


namespace NUMINAMATH_CALUDE_hugo_prime_given_win_l1266_126669

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The set of prime numbers on the die -/
def prime_rolls : Set ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number -/
def prob_prime : ℚ := 1/2

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1/num_players

/-- The probability that all other players roll non-prime or smaller prime -/
def prob_others_smaller : ℚ := (1/2)^(num_players - 1)

/-- The main theorem: probability of Hugo's first roll being prime given he won -/
theorem hugo_prime_given_win : 
  (prob_prime * prob_others_smaller) / prob_hugo_wins = 5/32 := by sorry

end NUMINAMATH_CALUDE_hugo_prime_given_win_l1266_126669


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l1266_126676

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the athlete's heart beats 28800 times during the race --/
theorem athlete_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l1266_126676


namespace NUMINAMATH_CALUDE_button_problem_l1266_126614

/-- Proof of the button problem -/
theorem button_problem (green : ℕ) (yellow : ℕ) (blue : ℕ) (total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  total = 275 →
  total = green + yellow + blue →
  green - blue = 5 := by sorry

end NUMINAMATH_CALUDE_button_problem_l1266_126614


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l1266_126694

/-- A regular hexagon with a diagonal of 18 inches has sides of 9 inches. -/
theorem regular_hexagon_side_length :
  ∀ (diagonal side : ℝ),
  diagonal = 18 →
  diagonal = 2 * side →
  side = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l1266_126694


namespace NUMINAMATH_CALUDE_equation_always_has_real_root_l1266_126648

theorem equation_always_has_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_always_has_real_root_l1266_126648


namespace NUMINAMATH_CALUDE_sequence_contains_intermediate_value_l1266_126639

theorem sequence_contains_intermediate_value 
  (n : ℕ) 
  (a : ℕ → ℤ) 
  (A B : ℤ) 
  (h1 : a 1 < A ∧ A < B ∧ B < a n) 
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < n → a (i + 1) - a i ≤ 1) :
  ∀ C : ℤ, A ≤ C ∧ C ≤ B → ∃ i₀ : ℕ, 1 < i₀ ∧ i₀ < n ∧ a i₀ = C := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_intermediate_value_l1266_126639


namespace NUMINAMATH_CALUDE_population_after_two_years_l1266_126608

def initial_population : ℕ := 1000
def year1_increase : ℚ := 20 / 100
def year2_increase : ℚ := 30 / 100

theorem population_after_two_years :
  let year1_population := initial_population * (1 + year1_increase)
  let year2_population := year1_population * (1 + year2_increase)
  ↑(round year2_population) = 1560 := by sorry

end NUMINAMATH_CALUDE_population_after_two_years_l1266_126608


namespace NUMINAMATH_CALUDE_second_tree_height_l1266_126640

/-- Given two trees casting shadows under the same conditions, 
    this theorem calculates the height of the second tree. -/
theorem second_tree_height
  (h1 : ℝ) -- Height of the first tree
  (s1 : ℝ) -- Shadow length of the first tree
  (s2 : ℝ) -- Shadow length of the second tree
  (h1_positive : h1 > 0)
  (s1_positive : s1 > 0)
  (s2_positive : s2 > 0)
  (h1_value : h1 = 28)
  (s1_value : s1 = 30)
  (s2_value : s2 = 45) :
  ∃ (h2 : ℝ), h2 = 42 ∧ h2 / s2 = h1 / s1 := by
  sorry


end NUMINAMATH_CALUDE_second_tree_height_l1266_126640


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1266_126680

theorem parabola_line_intersection (α : Real) : 
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) → α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1266_126680


namespace NUMINAMATH_CALUDE_integer_pair_existence_l1266_126602

theorem integer_pair_existence : ∃ (x y : ℤ), 
  (x * y + (x + y) = 95) ∧ 
  (x * y - (x + y) = 59) ∧ 
  ((x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11)) := by
  sorry

end NUMINAMATH_CALUDE_integer_pair_existence_l1266_126602


namespace NUMINAMATH_CALUDE_solution_existence_conditions_l1266_126691

theorem solution_existence_conditions (a b : ℝ) :
  (∃ x y : ℝ, (Real.tan x) * (Real.tan y) = a ∧ (Real.sin x)^2 + (Real.sin y)^2 = b^2) ↔
  (1 < b^2 ∧ b^2 < 2*a/(a+1)) ∨ (1 < b^2 ∧ b^2 < 2*a/(a-1)) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_conditions_l1266_126691


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_S_l1266_126659

def S : Set ℕ := {n : ℕ | ∃ (d₁ d₂ d₃ : ℕ), d₁ > d₂ ∧ d₂ > d₃ ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₁ ≠ n ∧ d₂ ≠ n ∧ d₃ ≠ n ∧ d₁ + d₂ + d₃ > n}

theorem largest_common_divisor_of_S : ∀ n ∈ S, 6 ∣ n ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_S_l1266_126659


namespace NUMINAMATH_CALUDE_fraction_simplification_l1266_126606

theorem fraction_simplification : (2 + 4) / (1 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1266_126606


namespace NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l1266_126644

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Theorem statement
theorem cubic_inches_in_cubic_foot :
  (inches_per_foot ^ 3 : ℕ) = 1728 :=
sorry

end NUMINAMATH_CALUDE_cubic_inches_in_cubic_foot_l1266_126644


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l1266_126674

theorem opposite_of_negative_six : -(-(6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l1266_126674


namespace NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1266_126667

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x^(5/2) + Real.sqrt 32) * (x^(5/2) - Real.sqrt 32) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_10_minus_1024_l1266_126667


namespace NUMINAMATH_CALUDE_cooler_capacity_sum_l1266_126635

theorem cooler_capacity_sum (c1 c2 c3 : ℝ) : 
  c1 = 100 →
  c2 = c1 + c1 * 0.5 →
  c3 = c2 / 2 →
  c1 + c2 + c3 = 325 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_sum_l1266_126635


namespace NUMINAMATH_CALUDE_point_transformation_l1266_126685

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Transformation from x-axis coordinates to y-axis coordinates -/
def transformToYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

/-- Theorem stating the transformation of point P -/
theorem point_transformation :
  ∃ (P : Point2D), P.x = 1 ∧ P.y = -2 → (transformToYAxis P).x = -1 ∧ (transformToYAxis P).y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1266_126685


namespace NUMINAMATH_CALUDE_M₄_is_mutually_orthogonal_l1266_126657

/-- A set M is a mutually orthogonal point set if for all (x₁, y₁) in M,
    there exists (x₂, y₂) in M such that x₁x₂ + y₁y₂ = 0 -/
def MutuallyOrthogonalPointSet (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- The set M₄ defined as {(x, y) | y = sin(x) + 1} -/
def M₄ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sin p.1 + 1}

/-- Theorem stating that M₄ is a mutually orthogonal point set -/
theorem M₄_is_mutually_orthogonal : MutuallyOrthogonalPointSet M₄ := by
  sorry

end NUMINAMATH_CALUDE_M₄_is_mutually_orthogonal_l1266_126657


namespace NUMINAMATH_CALUDE_no_parallel_solution_perpendicular_solutions_l1266_126668

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := (2*m^2 + m - 3)*x + (m^2 - m)*y = 2*m
def line2 (x y : ℝ) : Prop := x - y = 1

def line3 (a : ℝ) (x y : ℝ) : Prop := a*x + (1 - a)*y = 3
def line4 (a : ℝ) (x y : ℝ) : Prop := (a - 1)*x + (2*a + 3)*y = 2

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ 2*m^2 + m - 3 = k ∧ m^2 - m = -k

def perpendicular (a : ℝ) : Prop := a*(a - 1) + (1 - a)*(2*a + 3) = 0

-- State the theorems
theorem no_parallel_solution : ¬∃ m : ℝ, parallel m := sorry

theorem perpendicular_solutions : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -3) := sorry

end NUMINAMATH_CALUDE_no_parallel_solution_perpendicular_solutions_l1266_126668


namespace NUMINAMATH_CALUDE_pencils_left_l1266_126630

def initial_pencils : ℕ := 4527
def pencils_to_dorothy : ℕ := 1896
def pencils_to_samuel : ℕ := 754
def pencils_to_alina : ℕ := 307

theorem pencils_left : 
  initial_pencils - (pencils_to_dorothy + pencils_to_samuel + pencils_to_alina) = 1570 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_l1266_126630


namespace NUMINAMATH_CALUDE_vacation_cost_per_person_l1266_126682

theorem vacation_cost_per_person (num_people : ℕ) (airbnb_cost car_cost : ℚ) :
  num_people = 8 ∧ airbnb_cost = 3200 ∧ car_cost = 800 →
  (airbnb_cost + car_cost) / num_people = 500 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_per_person_l1266_126682


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1266_126692

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 180) (h2 : leg = 30) :
  ∃ (other_leg : ℝ) (hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1266_126692


namespace NUMINAMATH_CALUDE_fraction_sum_20_equals_10_9_l1266_126613

def fraction_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 2 / ((i + 1) * (i + 4)))

theorem fraction_sum_20_equals_10_9 : fraction_sum 20 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_20_equals_10_9_l1266_126613


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1266_126652

theorem salary_increase_percentage 
  (original_salary : ℝ) 
  (current_salary : ℝ) 
  (decrease_percentage : ℝ) 
  (increase_percentage : ℝ) :
  original_salary = 2000 →
  current_salary = 2090 →
  decrease_percentage = 5 →
  current_salary = (1 - decrease_percentage / 100) * (original_salary * (1 + increase_percentage / 100)) →
  increase_percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l1266_126652


namespace NUMINAMATH_CALUDE_hd_ha_ratio_specific_triangle_l1266_126631

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from a vertex to the opposite side -/
def altitude_foot (t : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

/-- The vertex of a triangle -/
def vertex (t : Triangle) (v : ℕ) : ℝ × ℝ := sorry

/-- The ratio of distances HD:HA in the triangle -/
def hd_ha_ratio (t : Triangle) : ℝ × ℝ := sorry

theorem hd_ha_ratio_specific_triangle :
  let t : Triangle := ⟨11, 13, 20, sorry, sorry, sorry, sorry, sorry, sorry⟩
  let h := orthocenter t
  let d := altitude_foot t 0  -- Assuming 0 represents vertex A
  let a := vertex t 0
  hd_ha_ratio t = (0, 6.6) := by sorry

end NUMINAMATH_CALUDE_hd_ha_ratio_specific_triangle_l1266_126631


namespace NUMINAMATH_CALUDE_triangle_properties_l1266_126666

/-- Properties of a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about triangle properties -/
theorem triangle_properties (t : Triangle) :
  (t.c = 2 ∧ t.C = π / 3 ∧ (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) →
  (Real.cos (t.A + t.B) = -1 / 2 ∧ t.a = 2 ∧ t.b = 2) ∧
  (t.B > π / 2 ∧ Real.cos t.A = 3 / 5 ∧ Real.sin t.B = 12 / 13) →
  Real.sin t.C = 16 / 65 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1266_126666


namespace NUMINAMATH_CALUDE_final_result_l1266_126633

def alternateOperations (start : ℚ) (n : ℕ) : ℚ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 
             then alternateOperations start n * 3 
             else alternateOperations start n / 2

theorem final_result : alternateOperations 1458 5 = 3^9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l1266_126633


namespace NUMINAMATH_CALUDE_addition_puzzle_solution_l1266_126603

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct_digits (a b c d e : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def number_from_digits (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem addition_puzzle_solution (a b c d e : ℕ) :
  distinct_digits a b c d e →
  number_from_digits a b c d + number_from_digits c b d a = number_from_digits d e e a →
  (∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x, x ∈ s ↔ (∃ a b c d, distinct_digits a b c d x ∧
    number_from_digits a b c d + number_from_digits c b d a = number_from_digits d x x a)) :=
sorry

end NUMINAMATH_CALUDE_addition_puzzle_solution_l1266_126603


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1266_126662

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) :
  let original_circumference := 2 * π * r
  let removed_sector_angle := π  -- 180° in radians
  let full_circle_angle := 2 * π  -- 360° in radians
  let base_circumference := (removed_sector_angle / full_circle_angle) * original_circumference
  base_circumference = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1266_126662


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l1266_126620

theorem product_mod_seventeen : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l1266_126620


namespace NUMINAMATH_CALUDE_three_white_balls_probability_l1266_126651

/-- The number of white balls in the urn -/
def white_balls : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := 21

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Probability of drawing 3 white balls without replacement -/
def prob_without_replacement : ℚ := 2 / 133

/-- Probability of drawing 3 white balls with replacement -/
def prob_with_replacement : ℚ := 8 / 343

/-- Probability of drawing 3 white balls simultaneously -/
def prob_simultaneous : ℚ := 2 / 133

/-- Theorem stating the probabilities of drawing 3 white balls under different conditions -/
theorem three_white_balls_probability :
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_without_replacement ∧
  ((white_balls : ℚ) / total_balls) ^ drawn_balls = prob_with_replacement ∧
  (Nat.choose white_balls drawn_balls / Nat.choose total_balls drawn_balls : ℚ) = prob_simultaneous :=
sorry

end NUMINAMATH_CALUDE_three_white_balls_probability_l1266_126651


namespace NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l1266_126615

theorem ceiling_sum_of_square_roots : 
  ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l1266_126615


namespace NUMINAMATH_CALUDE_dime_count_in_collection_l1266_126686

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of a coin collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

theorem dime_count_in_collection (c : CoinCollection) :
  totalCoins c = 13 ∧
  totalValue c = 141 ∧
  c.pennies ≥ 2 ∧
  c.nickels ≥ 2 ∧
  c.dimes ≥ 2 ∧
  c.quarters ≥ 2 →
  c.dimes = 3 := by
  sorry

end NUMINAMATH_CALUDE_dime_count_in_collection_l1266_126686


namespace NUMINAMATH_CALUDE_max_unique_sums_l1266_126670

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

def coin_set : List ℕ := [nickel, nickel, nickel, dime, dime, dime, quarter, quarter, half_dollar, half_dollar]

def unique_sums (coins : List ℕ) : Finset ℕ :=
  (do
    let c1 <- coins
    let c2 <- coins
    pure (c1 + c2)
  ).toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_set) = 10 := by sorry

end NUMINAMATH_CALUDE_max_unique_sums_l1266_126670


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1266_126675

def polynomial (x : ℝ) : ℝ := 3 * (3 * x^7 + 8 * x^4 - 7) + 7 * (x^5 - 7 * x^2 + 5)

theorem sum_of_coefficients : polynomial 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1266_126675


namespace NUMINAMATH_CALUDE_only_negative_number_l1266_126642

theorem only_negative_number (a b c d : ℚ) : 
  a = 0 → b = -(-3) → c = -1/2 → d = 3.2 → 
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ 
  (a ≥ 0 ∧ b ≥ 0 ∧ d ≥ 0) ∧ 
  c < 0 := by
sorry

end NUMINAMATH_CALUDE_only_negative_number_l1266_126642


namespace NUMINAMATH_CALUDE_subset_implies_a_geq_two_l1266_126624

def A : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a ≤ 0}

theorem subset_implies_a_geq_two (a : ℝ) (h : A ⊆ B a) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_geq_two_l1266_126624


namespace NUMINAMATH_CALUDE_corresponding_angles_are_equal_l1266_126698

-- Define the concept of angles
def Angle : Type := sorry

-- Define the property of being corresponding angles
def are_corresponding (a b : Angle) : Prop := sorry

-- Theorem statement
theorem corresponding_angles_are_equal (a b : Angle) : 
  are_corresponding a b → a = b := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_are_equal_l1266_126698


namespace NUMINAMATH_CALUDE_count_integers_in_list_integers_in_list_D_l1266_126634

def consecutive_integers (start : Int) (count : Nat) : List Int :=
  List.range count |>.map (fun i => start + i)

theorem count_integers_in_list (start : Int) (positive_range : Nat) : 
  let list := consecutive_integers start (positive_range + start.natAbs + 1)
  list.length = positive_range + start.natAbs + 1 :=
by sorry

-- The main theorem
theorem integers_in_list_D : 
  let start := -4
  let positive_range := 6
  let list_D := consecutive_integers start (positive_range + start.natAbs + 1)
  list_D.length = 12 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_in_list_integers_in_list_D_l1266_126634


namespace NUMINAMATH_CALUDE_sarah_pizza_consumption_l1266_126660

theorem sarah_pizza_consumption (total_slices : ℕ) (eaten_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 20 →
  eaten_slices = 3 →
  shared_slice = 1/3 →
  (eaten_slices : ℚ) / total_slices + shared_slice / total_slices = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pizza_consumption_l1266_126660


namespace NUMINAMATH_CALUDE_range_of_a_for_second_quadrant_l1266_126655

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (1 - Complex.I) * (a + Complex.I)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem range_of_a_for_second_quadrant :
  ∀ a : ℝ, in_second_quadrant (z a) ↔ a < -1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_second_quadrant_l1266_126655


namespace NUMINAMATH_CALUDE_kenny_friday_jacks_l1266_126622

/-- The number of jumping jacks Kenny did last week -/
def last_week_total : ℕ := 324

/-- The number of jumping jacks Kenny did on Sunday -/
def sunday_jacks : ℕ := 34

/-- The number of jumping jacks Kenny did on Monday -/
def monday_jacks : ℕ := 20

/-- The number of jumping jacks Kenny did on Tuesday -/
def tuesday_jacks : ℕ := 0

/-- The number of jumping jacks Kenny did on Wednesday -/
def wednesday_jacks : ℕ := 123

/-- The number of jumping jacks Kenny did on Thursday -/
def thursday_jacks : ℕ := 64

/-- The number of jumping jacks Kenny did on some unspecified day -/
def some_day_jacks : ℕ := 61

/-- The number of jumping jacks Kenny did on Friday -/
def friday_jacks : ℕ := 23

/-- Theorem stating that Kenny did 23 jumping jacks on Friday -/
theorem kenny_friday_jacks : 
  friday_jacks = 23 ∧ 
  friday_jacks + sunday_jacks + monday_jacks + tuesday_jacks + wednesday_jacks + thursday_jacks + some_day_jacks > last_week_total :=
by sorry

end NUMINAMATH_CALUDE_kenny_friday_jacks_l1266_126622


namespace NUMINAMATH_CALUDE_rabbit_position_after_ten_exchanges_l1266_126699

-- Define the seats
inductive Seat
| one
| two
| three
| four

-- Define the animals
inductive Animal
| mouse
| monkey
| rabbit
| cat

-- Define the seating arrangement
def Arrangement := Seat → Animal

-- Define the initial arrangement
def initial_arrangement : Arrangement := fun seat =>
  match seat with
  | Seat.one => Animal.mouse
  | Seat.two => Animal.monkey
  | Seat.three => Animal.rabbit
  | Seat.four => Animal.cat

-- Define a single exchange operation
def exchange (arr : Arrangement) (n : ℕ) : Arrangement := 
  if n % 2 = 0 then
    fun seat =>
      match seat with
      | Seat.one => arr Seat.three
      | Seat.two => arr Seat.four
      | Seat.three => arr Seat.one
      | Seat.four => arr Seat.two
  else
    fun seat =>
      match seat with
      | Seat.one => arr Seat.two
      | Seat.two => arr Seat.one
      | Seat.three => arr Seat.four
      | Seat.four => arr Seat.three

-- Define multiple exchanges
def multiple_exchanges (arr : Arrangement) (n : ℕ) : Arrangement :=
  match n with
  | 0 => arr
  | n+1 => exchange (multiple_exchanges arr n) n

-- Theorem statement
theorem rabbit_position_after_ten_exchanges :
  ∃ (seat : Seat), (multiple_exchanges initial_arrangement 10) seat = Animal.rabbit ∧ seat = Seat.two :=
sorry

end NUMINAMATH_CALUDE_rabbit_position_after_ten_exchanges_l1266_126699


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l1266_126616

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 - 2*p*x + p^2 - 5*p - 1 = 0 ∧ 
    y^2 - 2*p*y + p^2 - 5*p - 1 = 0) → 
  p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l1266_126616


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1266_126604

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1266_126604


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l1266_126617

/-- A square with side length 10 -/
structure Square :=
  (side : ℝ)
  (is_ten : side = 10)

/-- A circle passing through two opposite vertices of the square and tangent to one side -/
structure Circle (s : Square) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through_vertices : True)  -- This is a placeholder for the actual condition
  (tangent_to_side : True)  -- This is a placeholder for the actual condition

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (s : Square) (c : Circle s) : c.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l1266_126617


namespace NUMINAMATH_CALUDE_total_breakfast_cost_l1266_126658

def breakfast_cost (muffin_price fruit_cup_price francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins * muffin_price + francis_fruit_cups * fruit_cup_price) +
  (kiera_muffins * muffin_price + kiera_fruit_cups * fruit_cup_price)

theorem total_breakfast_cost :
  breakfast_cost 2 3 2 2 2 1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_total_breakfast_cost_l1266_126658


namespace NUMINAMATH_CALUDE_jill_weekly_earnings_l1266_126673

/-- Calculates Jill's earnings as a waitress for a week --/
def jill_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) 
                  (shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ) : ℝ :=
  let total_hours := shifts * hours_per_shift
  let wage_earnings := hourly_wage * total_hours
  let total_orders := avg_orders_per_hour * total_hours
  let orders_with_tax := total_orders * (1 + sales_tax)
  let tip_earnings := orders_with_tax * tip_rate
  wage_earnings + tip_earnings

/-- Theorem stating Jill's earnings for the week --/
theorem jill_weekly_earnings : 
  jill_earnings 4 0.15 0.1 3 8 40 = 254.4 := by
  sorry

end NUMINAMATH_CALUDE_jill_weekly_earnings_l1266_126673


namespace NUMINAMATH_CALUDE_box_difference_b_and_d_l1266_126671

/-- Represents the number of boxes of table tennis balls taken by each person. -/
structure BoxCount where
  a : ℕ  -- Number of boxes taken by A
  b : ℕ  -- Number of boxes taken by B
  c : ℕ  -- Number of boxes taken by C
  d : ℕ  -- Number of boxes taken by D

/-- Represents the money owed between individuals. -/
structure MoneyOwed where
  a_to_c : ℕ  -- Amount A owes to C
  b_to_d : ℕ  -- Amount B owes to D

/-- Theorem stating the difference in boxes between B and D is 18. -/
theorem box_difference_b_and_d (boxes : BoxCount) (money : MoneyOwed) : 
  boxes.b = boxes.a + 4 →  -- A took 4 boxes less than B
  boxes.d = boxes.c + 8 →  -- C took 8 boxes less than D
  money.a_to_c = 112 →     -- A owes C 112 yuan
  money.b_to_d = 72 →      -- B owes D 72 yuan
  boxes.b - boxes.d = 18 := by
  sorry

#check box_difference_b_and_d

end NUMINAMATH_CALUDE_box_difference_b_and_d_l1266_126671


namespace NUMINAMATH_CALUDE_consecutive_integers_base_equation_l1266_126612

/-- Given two consecutive positive integers A and B that satisfy the equation
    132_A + 43_B = 69_(A+B), prove that A + B = 13 -/
theorem consecutive_integers_base_equation (A B : ℕ) : 
  A > 0 ∧ B > 0 ∧ (B = A + 1 ∨ A = B + 1) →
  (A^2 + 3*A + 2) + (4*B + 3) = 6*(A + B) + 9 →
  A + B = 13 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_base_equation_l1266_126612


namespace NUMINAMATH_CALUDE_tundra_electrification_l1266_126618

theorem tundra_electrification (x y : ℝ) : 
  x + y = 1 →                 -- Initial parts sum to 1
  2*x + 0.75*y = 1 →          -- Condition after changes
  0 ≤ x ∧ x ≤ 1 →             -- x is a fraction
  0 ≤ y ∧ y ≤ 1 →             -- y is a fraction
  y = 4/5 :=                  -- Conclusion: non-electrified part was 4/5
by sorry

end NUMINAMATH_CALUDE_tundra_electrification_l1266_126618


namespace NUMINAMATH_CALUDE_sufficient_condition_problem_l1266_126681

theorem sufficient_condition_problem (p q r s : Prop) 
  (h1 : p → q)
  (h2 : s → q)
  (h3 : q → r)
  (h4 : r → s) :
  p → s := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_problem_l1266_126681


namespace NUMINAMATH_CALUDE_lemonade_glasses_l1266_126664

/-- Calculates the total number of glasses of lemonade that can be served -/
def total_glasses (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem: Given 5 glasses per pitcher and 6 pitchers, the total glasses served is 30 -/
theorem lemonade_glasses : total_glasses 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_l1266_126664


namespace NUMINAMATH_CALUDE_circumcircle_theorem_tangent_circles_theorem_l1266_126645

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 3)
def C : ℝ × ℝ := (0, 0)

-- Define the circumcircle equation
def circumcircle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 3*y = 0

-- Define the circles with center on y-axis and radius 5
def circle_eq_1 (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 25

def circle_eq_2 (x y : ℝ) : Prop :=
  x^2 + (y - 11)^2 = 25

-- Theorem for the circumcircle
theorem circumcircle_theorem :
  circumcircle_eq A.1 A.2 ∧
  circumcircle_eq B.1 B.2 ∧
  circumcircle_eq C.1 C.2 :=
sorry

-- Theorem for the circles tangent to y = 6
theorem tangent_circles_theorem :
  (∃ x y : ℝ, circle_eq_1 x y ∧ y = 6) ∧
  (∃ x y : ℝ, circle_eq_2 x y ∧ y = 6) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_theorem_tangent_circles_theorem_l1266_126645


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1266_126649

theorem arithmetic_sequence_tenth_term
  (a₁ a₁₇ : ℚ)
  (h₁ : a₁ = 2 / 3)
  (h₂ : a₁₇ = 3 / 2)
  (h_arith : ∀ n : ℕ, n > 0 → ∃ d : ℚ, a₁₇ = a₁ + (17 - 1) * d ∧ ∀ k : ℕ, k > 0 → a₁ + (k - 1) * d = a₁ + (k - 1) * ((a₁₇ - a₁) / 16)) :
  a₁ + 9 * ((a₁₇ - a₁) / 16) = 109 / 96 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1266_126649


namespace NUMINAMATH_CALUDE_equilateral_triangle_splitting_l1266_126687

/-- An equilateral triangle with side length 111 -/
def EquilateralTriangle : ℕ := 111

/-- The number of marked points in the triangle -/
def MarkedPoints : ℕ := 6216

/-- The number of linear sets -/
def LinearSets : ℕ := 111

/-- The number of ways to split the marked points into linear sets -/
def SplittingWays : ℕ := 2^4107

theorem equilateral_triangle_splitting (T : ℕ) (points : ℕ) (sets : ℕ) (ways : ℕ) :
  T = EquilateralTriangle →
  points = MarkedPoints →
  sets = LinearSets →
  ways = SplittingWays →
  ways = 2^(points / 3 * 2) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_splitting_l1266_126687


namespace NUMINAMATH_CALUDE_circle_max_area_center_l1266_126600

/-- Given a circle represented by the equation x^2 + y^2 + kx + 2y + k^2 = 0 in the Cartesian 
coordinate system, this theorem states that when the circle has maximum area, its center 
coordinates are (-k/2, -1). -/
theorem circle_max_area_center (k : ℝ) :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-k/2, -1)
  let is_max_area := ∀ k' : ℝ, 
    (∃ x y, circle_equation x y) → 
    (∃ x' y', x'^2 + y'^2 + k'*x' + 2*y' + k'^2 = 0 ∧ 
              (x' - (-k'/2))^2 + (y' - (-1))^2 ≤ (x - (-k/2))^2 + (y - (-1))^2)
  is_max_area → 
  ∃ x y, circle_equation x y ∧ 
         (x - center.1)^2 + (y - center.2)^2 = 
         (x - (-k/2))^2 + (y - (-1))^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_max_area_center_l1266_126600


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1266_126632

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  ((∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1266_126632


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1266_126625

theorem polynomial_divisibility (x : ℝ) (m : ℝ) : 
  (5 * x^3 - 3 * x^2 - 12 * x + m) % (x - 4) = 0 ↔ m = -224 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1266_126625


namespace NUMINAMATH_CALUDE_city_area_most_reliable_xiao_liang_most_reliable_l1266_126623

/-- Represents a survey method for assessing elderly health conditions -/
inductive SurveyMethod
  | Hospital
  | SquareDancing
  | CityArea

/-- Represents the reliability of a survey method -/
def reliability (method : SurveyMethod) : ℕ :=
  match method with
  | .Hospital => 1
  | .SquareDancing => 2
  | .CityArea => 3

/-- Theorem stating that the CityArea survey method is the most reliable -/
theorem city_area_most_reliable :
  ∀ (method : SurveyMethod), method ≠ SurveyMethod.CityArea →
    reliability method < reliability SurveyMethod.CityArea :=
by sorry

/-- Corollary: Xiao Liang's survey (CityArea) is the most reliable -/
theorem xiao_liang_most_reliable :
  reliability SurveyMethod.CityArea = max (reliability SurveyMethod.Hospital)
    (max (reliability SurveyMethod.SquareDancing) (reliability SurveyMethod.CityArea)) :=
by sorry

end NUMINAMATH_CALUDE_city_area_most_reliable_xiao_liang_most_reliable_l1266_126623


namespace NUMINAMATH_CALUDE_max_cart_length_l1266_126679

/-- The maximum length of a rectangular cart that can navigate through a right-angled corridor -/
theorem max_cart_length (corridor_width : ℝ) (cart_width : ℝ) :
  corridor_width = 1.5 →
  cart_width = 1 →
  ∃ (max_length : ℝ), max_length = 3 * Real.sqrt 2 - 2 ∧
    ∀ (cart_length : ℝ), cart_length ≤ max_length →
      ∃ (θ : ℝ), 0 < θ ∧ θ < Real.pi / 2 ∧
        cart_length ≤ (3 * (Real.sin θ + Real.cos θ) - 2) / (2 * Real.sin θ * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_max_cart_length_l1266_126679


namespace NUMINAMATH_CALUDE_dance_pairing_l1266_126626

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dancing relation
variable (danced_with : Boy → Girl → Prop)

-- State the theorem
theorem dance_pairing
  (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced_with b g)
  (h2 : ∀ g : Girl, ∃ b : Boy, danced_with b g)
  : ∃ (g g' : Boy) (f f' : Girl),
    danced_with g f ∧ ¬danced_with g f' ∧
    danced_with g' f' ∧ ¬danced_with g' f :=
by sorry

end NUMINAMATH_CALUDE_dance_pairing_l1266_126626


namespace NUMINAMATH_CALUDE_equation_solution_l1266_126653

theorem equation_solution : ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
  by
    use -30
    constructor
    · -- Prove that x = -30 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

end NUMINAMATH_CALUDE_equation_solution_l1266_126653


namespace NUMINAMATH_CALUDE_blackboard_numbers_l1266_126636

theorem blackboard_numbers (n : ℕ) (h1 : n = 2004) (h2 : (List.range n).sum % 167 = 0)
  (x : ℕ) (h3 : x ≤ 166) (h4 : (x + 999) % 167 = 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l1266_126636


namespace NUMINAMATH_CALUDE_point_distributive_l1266_126650

/-- Addition of two points in the plane -/
noncomputable def point_add (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Multiplication (midpoint) of two points in the plane -/
noncomputable def point_mul (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: A × (B + C) = (B + A) × (A + C) for any three points A, B, C in the plane -/
theorem point_distributive (A B C : ℝ × ℝ) :
  point_mul A (point_add B C) = point_mul (point_add B A) (point_add A C) :=
sorry

end NUMINAMATH_CALUDE_point_distributive_l1266_126650


namespace NUMINAMATH_CALUDE_january_salary_l1266_126690

theorem january_salary (feb mar apr may : ℕ) 
  (h1 : (feb + mar + apr + may) / 4 = 8300)
  (h2 : may = 6500)
  (h3 : ∃ jan, (jan + feb + mar + apr) / 4 = 8000) :
  ∃ jan, (jan + feb + mar + apr) / 4 = 8000 ∧ jan = 5300 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l1266_126690


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1266_126627

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = c * d) (h2 : a + b > c + d) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1266_126627


namespace NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_l1266_126647

theorem regular_polygon_with_144_degree_angle (n : ℕ) :
  n > 2 →
  (n - 2) * 180 = 144 * n →
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_144_degree_angle_l1266_126647


namespace NUMINAMATH_CALUDE_function_inequality_implies_b_bound_l1266_126611

open Real

theorem function_inequality_implies_b_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, exp x * (x - b) + x * exp x * (x + 1 - b) > 0) →
  b < 8/3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_b_bound_l1266_126611


namespace NUMINAMATH_CALUDE_fixed_point_on_circle_l1266_126696

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The parabola x^2 = 12y -/
def on_parabola (p : Point) : Prop :=
  p.x^2 = 12 * p.y

/-- The line y = -3 -/
def on_line (p : Point) : Prop :=
  p.y = -3

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The circle is tangent to the line y = -3 -/
def tangent_to_line (c : Circle) : Prop :=
  c.center.y + c.radius = -3

/-- Main theorem -/
theorem fixed_point_on_circle :
  ∀ (c : Circle),
    on_parabola c.center →
    tangent_to_line c →
    on_circle ⟨0, 3⟩ c :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_circle_l1266_126696


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_l1266_126637

-- Define the set of all substances
def Substance : Type := String

-- Define the property of conducting electricity
def conductsElectricity : Substance → Prop := sorry

-- Define the property of being a metal
def isMetal : Substance → Prop := sorry

-- Define iron as a substance
def iron : Substance := "iron"

-- Define the concept of deductive reasoning
def isDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Theorem statement
theorem iron_conducts_electricity_is_deductive :
  (∀ x, isMetal x → conductsElectricity x) →  -- All metals conduct electricity
  isMetal iron →                              -- Iron is a metal
  isDeductiveReasoning 
    (∀ x, isMetal x → conductsElectricity x)
    (isMetal iron)
    (conductsElectricity iron) :=
by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_is_deductive_l1266_126637


namespace NUMINAMATH_CALUDE_cubic_root_product_theorem_l1266_126672

/-- The cubic polynomial x^3 - 2x^2 + x + k -/
def cubic (k : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + x + k

/-- The condition that the product of roots equals the square of the difference between max and min real roots -/
def root_product_condition (k : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, cubic k x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    a * b * c = (max a (max b c) - min a (min b c))^2

theorem cubic_root_product_theorem : 
  ∀ k : ℝ, root_product_condition k ↔ k = -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_product_theorem_l1266_126672


namespace NUMINAMATH_CALUDE_distribute_5_3_l1266_126619

/-- The number of ways to distribute n identical objects into k identical containers,
    where at least one container must remain empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 26 ways to distribute 5 identical objects into 3 identical containers,
    where at least one container must remain empty -/
theorem distribute_5_3 : distribute 5 3 = 26 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1266_126619


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l1266_126641

theorem tangent_and_trigonometric_identity (α : Real) 
  (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2*π/3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43/52) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l1266_126641


namespace NUMINAMATH_CALUDE_efficient_elimination_of_y_l1266_126684

theorem efficient_elimination_of_y (x y : ℝ) :
  (3 * x - 2 * y = 3) →
  (4 * x + y = 15) →
  ∃ k : ℝ, (2 * (4 * x + y) + (3 * x - 2 * y) = k) ∧ (11 * x = 33) :=
by
  sorry

end NUMINAMATH_CALUDE_efficient_elimination_of_y_l1266_126684


namespace NUMINAMATH_CALUDE_triangle_side_length_l1266_126605

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = Real.pi / 3 →  -- 60° in radians
  a^2 + c^2 = 3 * a * c →  -- Given equation
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1266_126605


namespace NUMINAMATH_CALUDE_fraction_comparison_l1266_126610

theorem fraction_comparison (a b m : ℝ) (ha : a > b) (hb : b > 0) (hm : m > 0) :
  b / a < (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1266_126610


namespace NUMINAMATH_CALUDE_product_63_57_l1266_126697

theorem product_63_57 : 63 * 57 = 3591 := by
  sorry

end NUMINAMATH_CALUDE_product_63_57_l1266_126697


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1266_126609

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day. -/
def pagesPerDay (totalPages : ℕ) (days : ℕ) : ℕ :=
  totalPages / days

theorem stacy_paper_pages_per_day :
  pagesPerDay 33 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l1266_126609


namespace NUMINAMATH_CALUDE_estimate_total_students_l1266_126678

/-- Represents the survey data and estimated total students -/
structure SurveyData where
  total_students : ℕ  -- Estimated total number of first-year students
  first_survey : ℕ    -- Number of students in the first survey
  second_survey : ℕ   -- Number of students in the second survey
  overlap : ℕ         -- Number of students in both surveys

/-- The theorem states that given the survey conditions, 
    the estimated total number of first-year students is 400 -/
theorem estimate_total_students (data : SurveyData) :
  data.first_survey = 80 →
  data.second_survey = 100 →
  data.overlap = 20 →
  data.total_students = 400 :=
by sorry

end NUMINAMATH_CALUDE_estimate_total_students_l1266_126678


namespace NUMINAMATH_CALUDE_batsman_average_l1266_126693

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) :
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 3.5 →
  (∃ (previous_average : ℚ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = 
    previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 11) :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l1266_126693


namespace NUMINAMATH_CALUDE_problem_solution_l1266_126654

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | (x-1)*(x-a+1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_solution (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ 
  (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1266_126654


namespace NUMINAMATH_CALUDE_expression_equals_eight_l1266_126688

theorem expression_equals_eight : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l1266_126688


namespace NUMINAMATH_CALUDE_circular_mat_radius_increase_l1266_126621

theorem circular_mat_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) : 
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circular_mat_radius_increase_l1266_126621


namespace NUMINAMATH_CALUDE_negation_of_root_existence_l1266_126656

theorem negation_of_root_existence :
  ¬(∀ a : ℝ, a > 0 → a ≠ 1 → ∃ x : ℝ, a^x - x - a = 0) ↔
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a^x - x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_root_existence_l1266_126656


namespace NUMINAMATH_CALUDE_textbook_order_cost_l1266_126628

/-- Calculates the total cost of a textbook order with discounts applied --/
def calculate_order_cost (quantities : List Nat) (prices : List Float) (discount_threshold : Nat) (discount_rate : Float) : Float :=
  let total_cost := List.sum (List.zipWith (λ q p => q.toFloat * p) quantities prices)
  let discounted_cost := List.sum (List.zipWith 
    (λ q p => 
      if q ≥ discount_threshold then
        q.toFloat * p * (1 - discount_rate)
      else
        q.toFloat * p
    ) quantities prices)
  discounted_cost

theorem textbook_order_cost : 
  let quantities := [35, 35, 20, 30, 25, 15]
  let prices := [7.50, 10.50, 12.00, 9.50, 11.25, 6.75]
  let discount_threshold := 30
  let discount_rate := 0.1
  calculate_order_cost quantities prices discount_threshold discount_rate = 1446.00 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l1266_126628


namespace NUMINAMATH_CALUDE_remaining_seeds_l1266_126607

theorem remaining_seeds (initial_seeds : ℕ) (seeds_per_zone : ℕ) (num_zones : ℕ) : 
  initial_seeds = 54000 →
  seeds_per_zone = 3123 →
  num_zones = 7 →
  initial_seeds - (seeds_per_zone * num_zones) = 32139 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_seeds_l1266_126607


namespace NUMINAMATH_CALUDE_min_value_theorem_l1266_126663

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem min_value_theorem (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc 0 1, f a x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) ↔ 
  a = 0 ∨ a = 3 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1266_126663


namespace NUMINAMATH_CALUDE_unique_integers_sum_l1266_126601

theorem unique_integers_sum (x : ℝ) : x = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (a b c : ℕ+), 
    x^100 = 4*x^98 + 18*x^96 + 19*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^40 ∧
    (a : ℕ) + (b : ℕ) + (c : ℕ) = 534 := by
  sorry

end NUMINAMATH_CALUDE_unique_integers_sum_l1266_126601


namespace NUMINAMATH_CALUDE_vector_dot_product_problem_l1266_126638

theorem vector_dot_product_problem (a b : ℝ × ℝ) : 
  a = (0, 1) → b = (-1, 1) → (3 • a + 2 • b) • b = 7 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_problem_l1266_126638


namespace NUMINAMATH_CALUDE_min_value_abs_2a_minus_b_l1266_126646

theorem min_value_abs_2a_minus_b (a b : ℝ) (h : 2 * a^2 - b^2 = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y : ℝ), 2 * x^2 - y^2 = 1 → |2 * x - y| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_2a_minus_b_l1266_126646


namespace NUMINAMATH_CALUDE_max_value_of_roots_expression_l1266_126689

theorem max_value_of_roots_expression (a : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + a*x₁ + a = 2 →
  x₂^2 + a*x₂ + a = 2 →
  x₁ ≠ x₂ →
  ∀ b : ℝ, (x₁ - 2*x₂)*(x₂ - 2*x₁) ≤ -63/8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_expression_l1266_126689


namespace NUMINAMATH_CALUDE_anthonys_remaining_pencils_l1266_126661

/-- Represents the number of pencils Anthony has initially -/
def initial_pencils : ℝ := 56.0

/-- Represents the number of pencils Anthony gives to Kathryn -/
def pencils_given : ℝ := 9.5

/-- Theorem stating that Anthony's remaining pencils equal the initial amount minus the amount given away -/
theorem anthonys_remaining_pencils : 
  initial_pencils - pencils_given = 46.5 := by sorry

end NUMINAMATH_CALUDE_anthonys_remaining_pencils_l1266_126661


namespace NUMINAMATH_CALUDE_sampling_probabilities_equal_l1266_126629

/-- The total number of parts -/
def total_parts : ℕ := 160

/-- The number of first-class products -/
def first_class : ℕ := 48

/-- The number of second-class products -/
def second_class : ℕ := 64

/-- The number of third-class products -/
def third_class : ℕ := 32

/-- The number of substandard products -/
def substandard : ℕ := 16

/-- The sample size -/
def sample_size : ℕ := 20

/-- The probability of selection in simple random sampling -/
def p₁ : ℚ := sample_size / total_parts

/-- The probability of selection in stratified sampling -/
def p₂ : ℚ := sample_size / total_parts

/-- The probability of selection in systematic sampling -/
def p₃ : ℚ := sample_size / total_parts

theorem sampling_probabilities_equal :
  p₁ = p₂ ∧ p₂ = p₃ ∧ p₁ = 1/8 :=
sorry

end NUMINAMATH_CALUDE_sampling_probabilities_equal_l1266_126629
