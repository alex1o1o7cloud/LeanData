import Mathlib

namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1241_124158

/-- 
Given a rectangular box with face areas of 36, 18, and 8 square inches,
prove that its volume is 72 cubic inches.
-/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 36)
  (area2 : w * h = 18)
  (area3 : l * h = 8) :
  l * w * h = 72 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1241_124158


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l1241_124138

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (female_in_sample : ℕ) 
  (h1 : total_students = 1200) 
  (h2 : sample_size = 30) 
  (h3 : female_in_sample = 14) :
  let male_in_sample := sample_size - female_in_sample
  let male_in_grade := (male_in_sample : ℚ) / sample_size * total_students
  male_in_grade = 640 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l1241_124138


namespace NUMINAMATH_CALUDE_sqrt_3_sum_square_l1241_124160

theorem sqrt_3_sum_square (x y : ℝ) : x = Real.sqrt 3 + 1 → y = Real.sqrt 3 - 1 → x^2 + x*y + y^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_sum_square_l1241_124160


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1241_124181

/-- The area of a circle with circumference 24 cm is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1241_124181


namespace NUMINAMATH_CALUDE_sequence_errors_l1241_124159

-- Part (a)
def sequence_a (x y z : ℝ) : Prop :=
  (225 / 25 + 75 = 100 - 16) ∧
  (25 * (9 / (1 + 3)) = 84) ∧
  (25 * 12 = 84) ∧
  (25 = 7)

-- Part (b)
def sequence_b (x y z : ℝ) : Prop :=
  (5005 - 2002 = 35 * 143 - 143 * 14) ∧
  (5005 - 35 * 143 = 2002 - 143 * 14) ∧
  (5 * (1001 - 7 * 143) = 2 * (1001 - 7 * 143)) ∧
  (5 = 2)

theorem sequence_errors :
  ¬(∃ x y z : ℝ, sequence_a x y z) ∧
  ¬(∃ x y z : ℝ, sequence_b x y z) :=
sorry

end NUMINAMATH_CALUDE_sequence_errors_l1241_124159


namespace NUMINAMATH_CALUDE_evaluate_g_l1241_124166

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g : 3 * g 2 - 4 * g (-2) = -89 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l1241_124166


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1241_124125

theorem product_of_sum_and_cube_sum (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (cube_sum_eq : x^3 + y^3 = 172) : 
  x * y = 41.4 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1241_124125


namespace NUMINAMATH_CALUDE_lower_limit_x_l1241_124100

/-- The function f(x) = x - 5 -/
def f (x : ℝ) : ℝ := x - 5

/-- The lower limit of x for which f(x) ≤ 8 is 13 -/
theorem lower_limit_x (x : ℝ) : f x ≤ 8 ↔ x ≤ 13 := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_x_l1241_124100


namespace NUMINAMATH_CALUDE_cricket_team_win_percentage_l1241_124110

theorem cricket_team_win_percentage (matches_in_august : ℕ) (total_wins : ℕ) (overall_win_rate : ℚ) :
  matches_in_august = 120 →
  total_wins = 75 →
  overall_win_rate = 52/100 →
  (total_wins : ℚ) / (matches_in_august + (total_wins / overall_win_rate - matches_in_august)) = overall_win_rate →
  (matches_in_august * overall_win_rate - (total_wins - matches_in_august)) / matches_in_august = 17/40 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_win_percentage_l1241_124110


namespace NUMINAMATH_CALUDE_tree_planting_equation_correct_l1241_124116

/-- Represents the tree planting scenario -/
structure TreePlanting where
  totalTrees : ℕ
  originalRate : ℝ
  actualRateFactor : ℝ
  daysAhead : ℕ

/-- The equation representing the tree planting scenario is correct -/
theorem tree_planting_equation_correct (tp : TreePlanting)
  (h1 : tp.totalTrees = 960)
  (h2 : tp.originalRate > 0)
  (h3 : tp.actualRateFactor = 4/3)
  (h4 : tp.daysAhead = 4) :
  (tp.totalTrees : ℝ) / tp.originalRate - (tp.totalTrees : ℝ) / (tp.actualRateFactor * tp.originalRate) = tp.daysAhead :=
sorry

end NUMINAMATH_CALUDE_tree_planting_equation_correct_l1241_124116


namespace NUMINAMATH_CALUDE_f_of_2_equals_2_l1241_124190

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

-- State the theorem
theorem f_of_2_equals_2 : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_2_l1241_124190


namespace NUMINAMATH_CALUDE_at_least_one_third_l1241_124134

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l1241_124134


namespace NUMINAMATH_CALUDE_contractor_payment_proof_l1241_124185

/-- Calculates the total amount received by a contractor given the contract terms and absences. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_payment := working_days * payment_per_day
  let total_fine := absent_days * fine_per_day
  total_payment - total_fine

/-- Proves that the contractor receives Rs. 425 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_proof_l1241_124185


namespace NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l1241_124183

/-- An equilateral triangle with rotational symmetry -/
class EquilateralTriangle :=
  (rotation_symmetry : Bool)
  (is_equilateral : Bool)

/-- The minimum rotation angle (in degrees) for a shape with rotational symmetry -/
def min_rotation_angle (shape : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem: The minimum rotation angle for an equilateral triangle with rotational symmetry is 120 degrees -/
theorem equilateral_triangle_min_rotation_angle (t : EquilateralTriangle)
  (h1 : t.rotation_symmetry = true)
  (h2 : t.is_equilateral = true) :
  min_rotation_angle t = 120 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l1241_124183


namespace NUMINAMATH_CALUDE_opposite_of_five_l1241_124164

theorem opposite_of_five : 
  -(5 : ℤ) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_five_l1241_124164


namespace NUMINAMATH_CALUDE_hockey_games_in_season_l1241_124135

theorem hockey_games_in_season (games_per_month : ℕ) (months_in_season : ℕ) 
  (h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
  games_per_month * months_in_season = 182 :=
by sorry

end NUMINAMATH_CALUDE_hockey_games_in_season_l1241_124135


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l1241_124114

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l1241_124114


namespace NUMINAMATH_CALUDE_man_son_age_difference_l1241_124127

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
  sonAge = 14 →
  manAge + 2 = 2 * (sonAge + 2) →
  ageDifference manAge sonAge = 16 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l1241_124127


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_five_halves_l1241_124119

theorem sum_of_solutions_eq_five_halves :
  let f : ℝ → ℝ := λ x => (4*x + 6)*(3*x - 12)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_five_halves_l1241_124119


namespace NUMINAMATH_CALUDE_apple_difference_l1241_124155

theorem apple_difference (martha_apples harry_apples : ℕ) 
  (h1 : martha_apples = 68)
  (h2 : harry_apples = 19)
  (h3 : ∃ tim_apples : ℕ, tim_apples = 2 * harry_apples ∧ tim_apples < martha_apples) :
  martha_apples - (2 * harry_apples) = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l1241_124155


namespace NUMINAMATH_CALUDE_coin_count_proof_l1241_124161

/-- Represents the total number of coins given the following conditions:
  - There are coins of 20 paise and 25 paise denominations
  - The total value of all coins is 7100 paise (71 Rs)
  - There are 200 coins of 20 paise denomination
-/
def totalCoins (totalValue : ℕ) (value20p : ℕ) (value25p : ℕ) (count20p : ℕ) : ℕ :=
  count20p + (totalValue - count20p * value20p) / value25p

theorem coin_count_proof :
  totalCoins 7100 20 25 200 = 324 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_proof_l1241_124161


namespace NUMINAMATH_CALUDE_root_range_l1241_124144

theorem root_range (k : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 + (k-5)*x + 9 = 0) ↔ 
  (-5 < k ∧ k < -3/2) := by
sorry

end NUMINAMATH_CALUDE_root_range_l1241_124144


namespace NUMINAMATH_CALUDE_min_distance_line_to_log_curve_l1241_124175

/-- The minimum distance between a point on y = x and a point on y = ln x is √2/2 -/
theorem min_distance_line_to_log_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = P.1) → 
    (Q.2 = Real.log Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_to_log_curve_l1241_124175


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l1241_124150

/-- Given a line segment connecting (1,3) and (4,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,3), prove that p^2 + q^2 + r^2 + s^2 = 55 -/
theorem line_segment_param_sum_squares (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) → -- parameterization
  (q = 1 ∧ s = 3) → -- t = 0 corresponds to (1,3)
  (p + q = 4 ∧ r + s = 9) → -- endpoint (4,9)
  p^2 + q^2 + r^2 + s^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l1241_124150


namespace NUMINAMATH_CALUDE_team_score_l1241_124162

/-- Given a basketball team where each person scores 2 points and there are 9 people playing,
    the total points scored by the team is 18. -/
theorem team_score (points_per_person : ℕ) (num_players : ℕ) (total_points : ℕ) :
  points_per_person = 2 →
  num_players = 9 →
  total_points = points_per_person * num_players →
  total_points = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_score_l1241_124162


namespace NUMINAMATH_CALUDE_A_equals_nine_l1241_124151

/-- Represents the positions in the diagram --/
inductive Position
| A | B | C | D | E | F | G

/-- Represents the assignment of numbers to positions --/
def Assignment := Position → Fin 10

/-- Checks if all numbers from 1 to 10 are used exactly once --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∀ n : Fin 10, ∃! p : Position, a p = n

/-- Checks if the square condition is satisfied --/
def square_condition (a : Assignment) : Prop :=
  a Position.F = |a Position.A - a Position.B|

/-- Checks if the circle condition is satisfied --/
def circle_condition (a : Assignment) : Prop :=
  a Position.G = a Position.D + a Position.E

/-- Main theorem: A equals 9 --/
theorem A_equals_nine :
  ∃ (a : Assignment),
    is_valid_assignment a ∧
    square_condition a ∧
    circle_condition a ∧
    a Position.A = 9 := by
  sorry

end NUMINAMATH_CALUDE_A_equals_nine_l1241_124151


namespace NUMINAMATH_CALUDE_john_finishes_ahead_l1241_124131

/-- The distance John finishes ahead of Steve in a race --/
def distance_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time) - (steve_speed * push_time + initial_distance)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 15
  let push_time : ℝ := 34
  distance_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end NUMINAMATH_CALUDE_john_finishes_ahead_l1241_124131


namespace NUMINAMATH_CALUDE_remainder_theorem_l1241_124123

/-- The polynomial of degree 2023 --/
def f (z : ℂ) : ℂ := z^2023 + 2

/-- The cubic polynomial --/
def g (z : ℂ) : ℂ := z^3 + z^2 + 1

/-- The theorem statement --/
theorem remainder_theorem :
  ∃ (P S : ℂ → ℂ), (∀ z, f z = g z * P z + S z) ∧
                   (∃ a b c, ∀ z, S z = a * z^2 + b * z + c) →
  ∀ z, S z = z + 2 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1241_124123


namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l1241_124168

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- Represents the number 412 in base 5 -/
def num_base5 : ℕ := 412

/-- Represents the number 212 in base 7 -/
def num_base7 : ℕ := 212

theorem base5_to_base7_conversion :
  base10ToBase7 (base5ToBase10 num_base5) = num_base7 := by
  sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l1241_124168


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1241_124186

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1241_124186


namespace NUMINAMATH_CALUDE_power_difference_equals_one_third_l1241_124117

def is_greatest_power_of_2_factor (x : ℕ) : Prop :=
  2^x ∣ 200 ∧ ∀ k > x, ¬(2^k ∣ 200)

def is_greatest_power_of_5_factor (y : ℕ) : Prop :=
  5^y ∣ 200 ∧ ∀ k > y, ¬(5^k ∣ 200)

theorem power_difference_equals_one_third
  (x y : ℕ)
  (h2 : is_greatest_power_of_2_factor x)
  (h5 : is_greatest_power_of_5_factor y) :
  (1/3 : ℚ)^(x - y) = 1/3 := by sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_third_l1241_124117


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1241_124141

theorem fractional_equation_solution_range (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ x ≠ 1/2 ∧ 2/(x-1) = m/(2*x-1)) ↔ 
  (m > 4 ∨ m < 2) ∧ m ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1241_124141


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l1241_124124

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallelLines : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallelLinePlane a α) 
  (h2 : parallelLines a b) 
  (h3 : ¬ containedIn b α) : 
  parallelLinePlane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l1241_124124


namespace NUMINAMATH_CALUDE_complex_power_sum_l1241_124171

/-- Given that z = (i - 1) / √2, prove that z^100 + z^50 + 1 = -i -/
theorem complex_power_sum (z : ℂ) : z = (Complex.I - 1) / Real.sqrt 2 → z^100 + z^50 + 1 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1241_124171


namespace NUMINAMATH_CALUDE_updated_mean_calculation_l1241_124101

theorem updated_mean_calculation (n : ℕ) (original_mean : ℝ) (decrement : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : decrement = 34) :
  (n : ℝ) * original_mean - n * decrement = n * 166 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_calculation_l1241_124101


namespace NUMINAMATH_CALUDE_diamond_royalty_sanity_undetermined_l1241_124128

/-- Represents the sanity status of a person -/
inductive SanityStatus
  | Sane
  | Insane
  | Unknown

/-- Represents a royal person -/
structure RoyalPerson where
  name : String
  status : SanityStatus

/-- Represents a rumor about a person's sanity -/
structure Rumor where
  subject : RoyalPerson
  content : SanityStatus

/-- Represents the reliability of information -/
inductive Reliability
  | Reliable
  | Unreliable
  | Unknown

/-- The problem setup -/
def diamondRoyalty : Prop := ∃ (king queen : RoyalPerson) 
  (rumor : Rumor) (rumorReliability : Reliability),
  king.name = "King of Diamonds" ∧
  queen.name = "Queen of Diamonds" ∧
  rumor.subject = queen ∧
  rumor.content = SanityStatus.Insane ∧
  rumorReliability = Reliability.Unknown ∧
  (king.status = SanityStatus.Unknown ∨ 
   king.status = SanityStatus.Insane) ∧
  queen.status = SanityStatus.Unknown

/-- The theorem to be proved -/
theorem diamond_royalty_sanity_undetermined : 
  diamondRoyalty → 
  ∃ (king queen : RoyalPerson), 
    king.name = "King of Diamonds" ∧
    queen.name = "Queen of Diamonds" ∧
    king.status = SanityStatus.Unknown ∧
    queen.status = SanityStatus.Unknown :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_royalty_sanity_undetermined_l1241_124128


namespace NUMINAMATH_CALUDE_triangle_third_angle_l1241_124167

theorem triangle_third_angle (a b : ℝ) (ha : a = 37) (hb : b = 75) :
  180 - a - b = 68 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l1241_124167


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_l1241_124198

theorem purely_imaginary_complex (m : ℝ) : 
  (Complex.mk (m^2 - m) m).im ≠ 0 ∧ (Complex.mk (m^2 - m) m).re = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_l1241_124198


namespace NUMINAMATH_CALUDE_part_one_part_two_l1241_124195

/-- The function f(x) as defined in the problem -/
def f (a c x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

/-- Part 1 of the theorem -/
theorem part_one (a : ℝ) :
  f a 19 1 > 0 ↔ -2 < a ∧ a < 8 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a c : ℝ) :
  (∀ x : ℝ, f a c x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1241_124195


namespace NUMINAMATH_CALUDE_paula_and_olive_spend_twenty_l1241_124157

/-- The total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains : ℕ)
  (olive_coloring_books olive_bracelets : ℕ) : ℕ :=
  (paula_bracelets * bracelet_price + paula_keychains * keychain_price) +
  (olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price)

/-- Theorem stating that Paula and Olive spend $20 in total -/
theorem paula_and_olive_spend_twenty :
  total_spent 4 5 3 2 1 1 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_paula_and_olive_spend_twenty_l1241_124157


namespace NUMINAMATH_CALUDE_two_students_not_invited_l1241_124115

/-- Represents the social network of students in Mia's class -/
structure ClassNetwork where
  total_students : ℕ
  mia_friends : ℕ
  friends_of_friends : ℕ

/-- Calculates the number of students not invited to Mia's study session -/
def students_not_invited (network : ClassNetwork) : ℕ :=
  network.total_students - (1 + network.mia_friends + network.friends_of_friends)

/-- Theorem stating that 2 students will not be invited to Mia's study session -/
theorem two_students_not_invited (network : ClassNetwork) 
  (h1 : network.total_students = 15)
  (h2 : network.mia_friends = 4)
  (h3 : network.friends_of_friends = 8) : 
  students_not_invited network = 2 := by
  sorry

#eval students_not_invited ⟨15, 4, 8⟩

end NUMINAMATH_CALUDE_two_students_not_invited_l1241_124115


namespace NUMINAMATH_CALUDE_solution_set_solution_characterization_solution_equals_intervals_l1241_124142

theorem solution_set : Set ℝ := by
  sorry

theorem solution_characterization :
  solution_set = {x : ℝ | 2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ∧ (x - 3)^2 ≤ 16} := by
  sorry

theorem solution_equals_intervals :
  solution_set = Set.Icc (-1) 1 ∪ Set.Icc 5 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_solution_characterization_solution_equals_intervals_l1241_124142


namespace NUMINAMATH_CALUDE_double_age_in_three_years_l1241_124146

/-- The number of years from now when Tully will be twice as old as Kate -/
def years_until_double_age (tully_age_last_year : ℕ) (kate_age_now : ℕ) : ℕ :=
  3

theorem double_age_in_three_years (tully_age_last_year kate_age_now : ℕ) 
  (h1 : tully_age_last_year = 60) (h2 : kate_age_now = 29) :
  years_until_double_age tully_age_last_year kate_age_now = 3 := by
  sorry

end NUMINAMATH_CALUDE_double_age_in_three_years_l1241_124146


namespace NUMINAMATH_CALUDE_bundle_sheets_value_l1241_124147

/-- The number of sheets in a bundle -/
def bundle_sheets : ℕ := 2

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def bunch_sheets : ℕ := 4

/-- The number of sheets in a heap -/
def heap_sheets : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets : ℕ := 114

theorem bundle_sheets_value :
  colored_bundles * bundle_sheets + white_bunches * bunch_sheets + scrap_heaps * heap_sheets = total_sheets :=
by sorry

end NUMINAMATH_CALUDE_bundle_sheets_value_l1241_124147


namespace NUMINAMATH_CALUDE_blue_balls_count_l1241_124192

def probability_two_red (red green blue : ℕ) : ℚ :=
  (red.choose 2 : ℚ) / ((red + green + blue).choose 2 : ℚ)

theorem blue_balls_count (red green : ℕ) (prob : ℚ) :
  red = 7 →
  green = 4 →
  probability_two_red red green (blue : ℕ) = (175 : ℚ) / 1000 →
  blue = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1241_124192


namespace NUMINAMATH_CALUDE_min_value_given_max_l1241_124179

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_given_max (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_given_max_l1241_124179


namespace NUMINAMATH_CALUDE_cubic_function_coefficients_l1241_124165

/-- Given a cubic function f(x) = ax³ - bx + 4, prove that if f(2) = -4/3 and f'(2) = 0,
    then a = 1/3 and b = 4 -/
theorem cubic_function_coefficients (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 4
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - b
  f 2 = -4/3 ∧ f' 2 = 0 → a = 1/3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficients_l1241_124165


namespace NUMINAMATH_CALUDE_shortest_side_length_l1241_124184

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the shortest side (opposite to 30° angle) -/
  short : ℝ
  /-- The length of the middle side (opposite to 60° angle) -/
  middle : ℝ
  /-- The length of the hypotenuse (opposite to 90° angle) -/
  hypotenuse : ℝ
  /-- The ratio of sides in a 30-60-90 triangle -/
  ratio_prop : short = middle / Real.sqrt 3 ∧ middle = hypotenuse / 2

/-- Theorem: In a 30-60-90 triangle with hypotenuse 30 units, the shortest side is 15 units -/
theorem shortest_side_length (t : Triangle30_60_90) (h : t.hypotenuse = 30) : t.short = 15 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1241_124184


namespace NUMINAMATH_CALUDE_trihedral_angle_inequality_l1241_124111

/-- Represents a trihedral angle with three planar angles -/
structure TrihedralAngle where
  α : ℝ
  β : ℝ
  γ : ℝ
  α_pos : 0 < α
  β_pos : 0 < β
  γ_pos : 0 < γ
  α_lt_pi : α < π
  β_lt_pi : β < π
  γ_lt_pi : γ < π

/-- The sum of any two planar angles in a trihedral angle is greater than the third angle -/
theorem trihedral_angle_inequality (t : TrihedralAngle) :
  t.α + t.β > t.γ ∧ t.α + t.γ > t.β ∧ t.β + t.γ > t.α := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_inequality_l1241_124111


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l1241_124120

/-- Given a square with sides measured to the nearest centimeter as 7 cm,
    the least possible actual area of the square is 42.25 cm². -/
theorem least_possible_area_of_square (side_length : ℝ) : 
  (6.5 ≤ side_length) ∧ (side_length < 7.5) → side_length ^ 2 ≥ 42.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l1241_124120


namespace NUMINAMATH_CALUDE_frank_weekly_spending_l1241_124152

theorem frank_weekly_spending (lawn_money weed_money weeks : ℕ) 
  (h1 : lawn_money = 5)
  (h2 : weed_money = 58)
  (h3 : weeks = 9) :
  (lawn_money + weed_money) / weeks = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_weekly_spending_l1241_124152


namespace NUMINAMATH_CALUDE_average_weight_of_children_l1241_124173

theorem average_weight_of_children (num_boys num_girls : ℕ) 
  (avg_weight_boys avg_weight_girls : ℚ) :
  num_boys = 8 →
  num_girls = 5 →
  avg_weight_boys = 160 →
  avg_weight_girls = 130 →
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l1241_124173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1241_124182

/-- An arithmetic sequence with common ratio q ≠ 1 -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) - a n = q * (a n - a (n - 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  ArithmeticSequence a q →
  (a 1 + a 2 + a 3 + a 4 + a 5 = 6) →
  (a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18) →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1241_124182


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1241_124104

/-- Given a geometric sequence {aₙ} where a₂a₃a₄ = 1 and a₆a₇a₈ = 64, prove that a₅ = 2 -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h1 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q)  -- a is a geometric sequence
  (h2 : a 2 * a 3 * a 4 = 1)  -- Condition 1
  (h3 : a 6 * a 7 * a 8 = 64)  -- Condition 2
  : a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1241_124104


namespace NUMINAMATH_CALUDE_string_longest_piece_fraction_l1241_124180

theorem string_longest_piece_fraction (L : ℝ) (x : ℝ) (h1 : x > 0) : 
  x + 2*x + 4*x + 8*x = L → 8*x / L = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_string_longest_piece_fraction_l1241_124180


namespace NUMINAMATH_CALUDE_vector_AB_l1241_124140

/-- Given points A(1, -1) and B(1, 2), prove that the vector AB is (0, 3) -/
theorem vector_AB (A B : ℝ × ℝ) (hA : A = (1, -1)) (hB : B = (1, 2)) :
  B.1 - A.1 = 0 ∧ B.2 - A.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_l1241_124140


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l1241_124163

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l1241_124163


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1241_124178

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l1241_124178


namespace NUMINAMATH_CALUDE_product_increase_by_2022_l1241_124133

theorem product_increase_by_2022 : ∃ (a b c : ℕ),
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 := by
  sorry

end NUMINAMATH_CALUDE_product_increase_by_2022_l1241_124133


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_l1241_124105

/-- Represents a 12-hour digital clock with a glitch that displays 7 instead of 5 -/
structure GlitchedClock where
  hours : Fin 12
  minutes : Fin 60

/-- Checks if a given hour is displayed correctly -/
def correctHour (h : Fin 12) : Bool :=
  h ≠ 5

/-- Checks if a given minute is displayed correctly -/
def correctMinute (m : Fin 60) : Bool :=
  m % 10 ≠ 5 ∧ m / 10 ≠ 5

/-- Calculates the fraction of the day the clock shows the correct time -/
def fractionCorrect : ℚ :=
  (11 : ℚ) / 12 * (54 : ℚ) / 60

theorem glitched_clock_correct_time :
  fractionCorrect = 33 / 40 := by
  sorry

#eval fractionCorrect

end NUMINAMATH_CALUDE_glitched_clock_correct_time_l1241_124105


namespace NUMINAMATH_CALUDE_possible_r_value_l1241_124118

-- Define sets A and B
def A (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 * (p.1 - 1) + p.2 * (p.2 - 1) ≤ r}

def B (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ r^2}

-- Theorem statement
theorem possible_r_value :
  ∃ (r : ℝ), (A r ⊆ B r) ∧ (r = Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_possible_r_value_l1241_124118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1241_124107

/-- Given an arithmetic sequence where the fifth term is 15 and the common difference is 2,
    prove that the product of the first two terms is 63. -/
theorem arithmetic_sequence_product (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n + 2) →  -- Common difference is 2
  a 5 = 15 →                    -- Fifth term is 15
  a 1 * a 2 = 63 :=              -- Product of first two terms is 63
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1241_124107


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1241_124176

theorem inequality_solution_set (x : ℝ) :
  (-6 * x^2 - x + 2 < 0) ↔ (x < -2/3 ∨ x > 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1241_124176


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l1241_124103

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (Bᶜ) = {x : ℝ | -5 < x ∧ x < 3}) := by sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l1241_124103


namespace NUMINAMATH_CALUDE_complex_trajectory_l1241_124174

theorem complex_trajectory (z : ℂ) (x y : ℝ) :
  z = x + y * Complex.I →
  Complex.abs z ^ 2 - 2 * Complex.abs z - 3 = 0 →
  x ^ 2 + y ^ 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_complex_trajectory_l1241_124174


namespace NUMINAMATH_CALUDE_new_person_weight_l1241_124106

theorem new_person_weight (n : ℕ) (initial_avg : ℝ) (weight_decrease : ℝ) :
  n = 20 →
  initial_avg = 58 →
  weight_decrease = 5 →
  let total_weight := n * initial_avg
  let new_avg := initial_avg - weight_decrease
  let new_person_weight := total_weight - (n + 1) * new_avg
  new_person_weight = 47 := by
sorry

end NUMINAMATH_CALUDE_new_person_weight_l1241_124106


namespace NUMINAMATH_CALUDE_time_difference_walk_bike_l1241_124154

/-- The number of blocks between Youseff's home and office -/
def B : ℕ := 21

/-- The time in minutes it takes to walk one block -/
def walk_time_per_block : ℚ := 1

/-- The time in minutes it takes to bike one block -/
def bike_time_per_block : ℚ := 20 / 60

/-- The total time in minutes it takes to walk to work -/
def total_walk_time : ℚ := B * walk_time_per_block

/-- The total time in minutes it takes to bike to work -/
def total_bike_time : ℚ := B * bike_time_per_block

theorem time_difference_walk_bike : 
  total_walk_time - total_bike_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_walk_bike_l1241_124154


namespace NUMINAMATH_CALUDE_partition_equal_product_l1241_124172

def numbers : List Nat := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

def target_product : Nat := 349188840

theorem partition_equal_product :
  ∃ (A B : List Nat),
    A.length = 5 ∧
    B.length = 5 ∧
    A ∪ B = numbers ∧
    A ∩ B = [] ∧
    A.prod = target_product ∧
    B.prod = target_product :=
by
  sorry

end NUMINAMATH_CALUDE_partition_equal_product_l1241_124172


namespace NUMINAMATH_CALUDE_equation_solution_difference_l1241_124143

theorem equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ + 3)^2 / (2*x₁ + 15) = 3 ∧
  (x₂ + 3)^2 / (2*x₂ + 15) = 3 ∧
  x₁ ≠ x₂ ∧
  x₂ - x₁ = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l1241_124143


namespace NUMINAMATH_CALUDE_die_roll_sequences_l1241_124170

/-- The number of sides on the die -/
def num_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 6

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := num_sides ^ num_rolls

theorem die_roll_sequences :
  num_sequences = 46656 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_sequences_l1241_124170


namespace NUMINAMATH_CALUDE_solve_F_equation_l1241_124149

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_F_equation :
  ∃ a : ℚ, F a 3 10 = F a 5 20 ∧ a = -5/49 := by
  sorry

end NUMINAMATH_CALUDE_solve_F_equation_l1241_124149


namespace NUMINAMATH_CALUDE_carpentry_job_cost_l1241_124196

/-- Calculates the total cost of a carpentry job -/
theorem carpentry_job_cost 
  (hourly_rate : ℕ) 
  (material_cost : ℕ) 
  (estimated_hours : ℕ) 
  (h1 : hourly_rate = 28)
  (h2 : material_cost = 560)
  (h3 : estimated_hours = 15) :
  hourly_rate * estimated_hours + material_cost = 980 := by
sorry

end NUMINAMATH_CALUDE_carpentry_job_cost_l1241_124196


namespace NUMINAMATH_CALUDE_andrea_sod_rectangles_l1241_124148

/-- Calculates the number of sod rectangles needed for a given area -/
def sodRectanglesNeeded (length width : ℕ) : ℕ :=
  (length * width + 11) / 12

/-- The total number of sod rectangles needed for Andrea's backyard -/
def totalSodRectangles : ℕ :=
  sodRectanglesNeeded 35 42 +
  sodRectanglesNeeded 55 86 +
  sodRectanglesNeeded 20 50 +
  sodRectanglesNeeded 48 66

theorem andrea_sod_rectangles :
  totalSodRectangles = 866 := by
  sorry

end NUMINAMATH_CALUDE_andrea_sod_rectangles_l1241_124148


namespace NUMINAMATH_CALUDE_seahawks_score_is_37_l1241_124153

/-- The final score of the Seattle Seahawks in the football game -/
def seahawks_final_score : ℕ :=
  let touchdowns : ℕ := 4
  let field_goals : ℕ := 3
  let touchdown_points : ℕ := 7
  let field_goal_points : ℕ := 3
  touchdowns * touchdown_points + field_goals * field_goal_points

/-- Theorem stating that the Seattle Seahawks' final score is 37 points -/
theorem seahawks_score_is_37 : seahawks_final_score = 37 := by
  sorry

end NUMINAMATH_CALUDE_seahawks_score_is_37_l1241_124153


namespace NUMINAMATH_CALUDE_gcd_xyz_square_l1241_124113

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_xyz_square_l1241_124113


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1241_124188

theorem cousins_ages_sum : ∃ (a b c d : ℕ), 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 24 ∧ c * d = 35) ∨ (a * c = 24 ∧ b * d = 35) ∨ 
   (a * d = 24 ∧ b * c = 35) ∨ (b * c = 24 ∧ a * d = 35) ∨ 
   (b * d = 24 ∧ a * c = 35) ∨ (c * d = 24 ∧ a * b = 35)) →
  a + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1241_124188


namespace NUMINAMATH_CALUDE_polygon_distance_inequality_l1241_124109

/-- A polygon in a plane -/
structure Polygon where
  vertices : List (Real × Real)
  is_closed : vertices.length > 2

/-- Calculate the perimeter of a polygon -/
def perimeter (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the vertices of a polygon -/
def sum_distances_to_vertices (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- Calculate the sum of distances from a point to the sidelines of a polygon -/
def sum_distances_to_sidelines (X : Real × Real) (F : Polygon) : Real :=
  sorry

/-- The main theorem -/
theorem polygon_distance_inequality (X : Real × Real) (F : Polygon) :
  let p := perimeter F
  let d := sum_distances_to_vertices X F
  let h := sum_distances_to_sidelines X F
  d^2 - h^2 ≥ p^2 / 4 := by
    sorry

end NUMINAMATH_CALUDE_polygon_distance_inequality_l1241_124109


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1241_124137

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure it's a two-digit number
  (n % 5 = 0) →     -- Divisible by 5
  ((n / 10) % 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n / 10) % 10 * (n % 10) = 35) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1241_124137


namespace NUMINAMATH_CALUDE_odd_function_extension_l1241_124199

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = x^2 - 2*x) : 
  ∀ x ≤ 0, f x = -x^2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1241_124199


namespace NUMINAMATH_CALUDE_circle_equation_l1241_124139

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersectsXAxisAt (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = 0 ∧ p2.2 = 0 ∧
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2

def isTangentToLine (c : Circle) : Prop :=
  let d := |c.center.1 - c.center.2 + 1| / Real.sqrt 2
  d = c.radius

-- Theorem statement
theorem circle_equation (c : Circle) :
  isInFirstQuadrant c.center →
  intersectsXAxisAt c (1, 0) (3, 0) →
  isTangentToLine c →
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1241_124139


namespace NUMINAMATH_CALUDE_problem_solution_l1241_124197

-- Define the ⊕ operation
def circleplus (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem problem_solution :
  (circleplus 7 4 - 12) * 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1241_124197


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1241_124145

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1241_124145


namespace NUMINAMATH_CALUDE_min_tiles_needed_l1241_124108

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 4⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the number of tiles needed to cover the floor -/
def tilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed :
  tilesNeeded = 173 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_needed_l1241_124108


namespace NUMINAMATH_CALUDE_graph_regions_count_l1241_124132

/-- A line in the coordinate plane defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The set of lines defining the graph -/
def graph_lines : Set Line := {⟨3, 0⟩, ⟨1/3, 0⟩}

/-- The number of regions created by the graph lines -/
def num_regions : ℕ := 4

/-- Theorem stating that the number of regions created by the graph lines is 4 -/
theorem graph_regions_count :
  num_regions = 4 :=
sorry

end NUMINAMATH_CALUDE_graph_regions_count_l1241_124132


namespace NUMINAMATH_CALUDE_largest_solution_is_two_l1241_124187

theorem largest_solution_is_two :
  ∃ (x : ℝ), x > 0 ∧ (x / 4 + 2 / (3 * x) = 5 / 6) ∧
  (∀ (y : ℝ), y > 0 → y / 4 + 2 / (3 * y) = 5 / 6 → y ≤ x) ∧
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_is_two_l1241_124187


namespace NUMINAMATH_CALUDE_eleventh_grade_sample_l1241_124193

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
| 0 => 3  -- 10th grade
| 1 => 3  -- 11th grade
| 2 => 4  -- 12th grade

/-- The total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students to be sampled from a specific grade -/
def students_to_sample (grade : Fin 3) : ℕ :=
  (grade_ratio grade * sample_size) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

theorem eleventh_grade_sample :
  students_to_sample 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_grade_sample_l1241_124193


namespace NUMINAMATH_CALUDE_max_quadrilateral_area_l1241_124126

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Points on the x-axis that the ellipse passes through -/
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (-1, 0)

/-- A function representing parallel lines passing through a point with slope k -/
def parallelLine (p : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - p.1) + p.2

/-- The quadrilateral formed by the intersection of parallel lines and the ellipse -/
def quadrilateral (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, ellipse x y ∧ (parallelLine P k x y ∨ parallelLine Q k x y)}

/-- The area of the quadrilateral as a function of the slope k -/
noncomputable def quadrilateralArea (k : ℝ) : ℝ :=
  sorry  -- Actual computation of area

theorem max_quadrilateral_area :
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 3 ∧
    ∀ k, quadrilateralArea k ≤ max_area :=
  sorry

#check max_quadrilateral_area

end NUMINAMATH_CALUDE_max_quadrilateral_area_l1241_124126


namespace NUMINAMATH_CALUDE_postcard_selection_ways_l1241_124122

theorem postcard_selection_ways : 
  let total_teachers : ℕ := 4
  let type_a_cards : ℕ := 2
  let type_b_cards : ℕ := 3
  let total_cards_to_select : ℕ := 4
  ∃ (ways : ℕ), ways = 10 ∧ 
    ways = (Nat.choose total_teachers type_a_cards) + 
           (Nat.choose total_teachers (total_teachers - type_a_cards)) :=
by sorry

end NUMINAMATH_CALUDE_postcard_selection_ways_l1241_124122


namespace NUMINAMATH_CALUDE_no_real_solutions_l1241_124189

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 4) ∧ (x * y - z^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1241_124189


namespace NUMINAMATH_CALUDE_carson_seed_fertilizer_problem_l1241_124130

/-- The problem of calculating the total amount of seed and fertilizer used by Carson. -/
theorem carson_seed_fertilizer_problem :
  ∀ (seed fertilizer : ℝ),
  seed = 45 →
  seed = 3 * fertilizer →
  seed + fertilizer = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_carson_seed_fertilizer_problem_l1241_124130


namespace NUMINAMATH_CALUDE_tim_one_dollar_bills_l1241_124112

/-- Represents the number of bills of a certain denomination -/
structure BillCount where
  count : ℕ
  denomination : ℕ

/-- Represents Tim's wallet -/
structure Wallet where
  tenDollarBills : BillCount
  fiveDollarBills : BillCount
  oneDollarBills : BillCount

def Wallet.totalValue (w : Wallet) : ℕ :=
  w.tenDollarBills.count * w.tenDollarBills.denomination +
  w.fiveDollarBills.count * w.fiveDollarBills.denomination +
  w.oneDollarBills.count * w.oneDollarBills.denomination

def Wallet.totalBills (w : Wallet) : ℕ :=
  w.tenDollarBills.count + w.fiveDollarBills.count + w.oneDollarBills.count

theorem tim_one_dollar_bills 
  (w : Wallet)
  (h1 : w.tenDollarBills = ⟨13, 10⟩)
  (h2 : w.fiveDollarBills = ⟨11, 5⟩)
  (h3 : w.totalValue = 128)
  (h4 : w.totalBills ≥ 16) :
  w.oneDollarBills.count = 57 := by
  sorry

end NUMINAMATH_CALUDE_tim_one_dollar_bills_l1241_124112


namespace NUMINAMATH_CALUDE_triangle_inequality_l1241_124121

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  let s := (a + b + c) / 2
  (2*a*(2*a - s))/(b + c) + (2*b*(2*b - s))/(c + a) + (2*c*(2*c - s))/(a + b) ≥ s := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1241_124121


namespace NUMINAMATH_CALUDE_weighted_average_combined_class_l1241_124191

/-- Given two classes of students, prove that the weighted average of the combined class
    is equal to the sum of the products of each class's student count and average mark,
    divided by the total number of students. -/
theorem weighted_average_combined_class
  (n₁ : ℕ) (n₂ : ℕ) (x₁ : ℚ) (x₂ : ℚ)
  (h₁ : n₁ = 58)
  (h₂ : n₂ = 52)
  (h₃ : x₁ = 67)
  (h₄ : x₂ = 82) :
  (n₁ * x₁ + n₂ * x₂) / (n₁ + n₂ : ℚ) = (58 * 67 + 52 * 82) / (58 + 52 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_combined_class_l1241_124191


namespace NUMINAMATH_CALUDE_mode_estimate_is_tallest_rectangle_midpoint_l1241_124129

/-- Represents a rectangle in a frequency distribution histogram --/
structure HistogramRectangle where
  height : ℝ
  base_midpoint : ℝ

/-- Represents a sample frequency distribution histogram --/
structure FrequencyHistogram where
  rectangles : List HistogramRectangle

/-- Finds the tallest rectangle in a frequency histogram --/
def tallestRectangle (h : FrequencyHistogram) : HistogramRectangle :=
  sorry

/-- Estimates the mode of a dataset from a frequency histogram --/
def estimateMode (h : FrequencyHistogram) : ℝ :=
  (tallestRectangle h).base_midpoint

theorem mode_estimate_is_tallest_rectangle_midpoint (h : FrequencyHistogram) :
  estimateMode h = (tallestRectangle h).base_midpoint :=
sorry

end NUMINAMATH_CALUDE_mode_estimate_is_tallest_rectangle_midpoint_l1241_124129


namespace NUMINAMATH_CALUDE_triangle_altitude_tangent_relation_l1241_124136

-- Define a triangle with its properties
structure Triangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles
  A : ℝ
  B : ℝ
  C : ℝ
  -- Altitudes
  DA' : ℝ
  EB' : ℝ
  FC' : ℝ
  -- Conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_altitudes : 0 < DA' ∧ 0 < EB' ∧ 0 < FC'
  angle_sum : A + B + C = π
  -- Additional conditions may be needed to fully define a valid triangle

-- Theorem statement
theorem triangle_altitude_tangent_relation (t : Triangle) :
  t.a / t.DA' + t.b / t.EB' + t.c / t.FC' = 2 * Real.tan t.A * Real.tan t.B * Real.tan t.C := by
  sorry


end NUMINAMATH_CALUDE_triangle_altitude_tangent_relation_l1241_124136


namespace NUMINAMATH_CALUDE_tomato_seeds_planted_l1241_124102

/-- The total number of tomato seeds planted by Mike, Ted, and Sarah -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon + sarah_morning + sarah_afternoon

theorem tomato_seeds_planted :
  ∃ (mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon : ℕ),
    mike_morning = 50 ∧
    ted_morning = 2 * mike_morning ∧
    sarah_morning = mike_morning + 30 ∧
    mike_afternoon = 60 ∧
    ted_afternoon = mike_afternoon - 20 ∧
    sarah_afternoon = sarah_morning + 20 ∧
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon sarah_morning sarah_afternoon = 430 :=
by sorry

end NUMINAMATH_CALUDE_tomato_seeds_planted_l1241_124102


namespace NUMINAMATH_CALUDE_fuel_in_truck_is_38_l1241_124156

/-- Calculates the amount of fuel already in a truck given the total capacity,
    amount spent, change received, and cost per liter. -/
def fuel_already_in_truck (total_capacity : ℕ) (amount_spent : ℕ) (change : ℕ) (cost_per_liter : ℕ) : ℕ :=
  total_capacity - (amount_spent - change) / cost_per_liter

/-- Proves that given the specific conditions, the amount of fuel already in the truck is 38 liters. -/
theorem fuel_in_truck_is_38 :
  fuel_already_in_truck 150 350 14 3 = 38 := by
  sorry

#eval fuel_already_in_truck 150 350 14 3

end NUMINAMATH_CALUDE_fuel_in_truck_is_38_l1241_124156


namespace NUMINAMATH_CALUDE_student_travel_distance_l1241_124177

/-- Proves that given a total distance of 105.00000000000003 km, where 1/5 is traveled by foot
    and 2/3 is traveled by bus, the remaining distance traveled by car is 14.000000000000002 km. -/
theorem student_travel_distance (total_distance : ℝ) 
    (h1 : total_distance = 105.00000000000003) 
    (foot_fraction : ℝ) (h2 : foot_fraction = 1/5)
    (bus_fraction : ℝ) (h3 : bus_fraction = 2/3) : 
    total_distance - (foot_fraction * total_distance + bus_fraction * total_distance) = 14.000000000000002 := by
  sorry

end NUMINAMATH_CALUDE_student_travel_distance_l1241_124177


namespace NUMINAMATH_CALUDE_final_value_is_four_l1241_124169

def increment_sequence (initial : ℕ) : ℕ :=
  let step1 := initial + 1
  let step2 := step1 + 2
  step2

theorem final_value_is_four :
  increment_sequence 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_final_value_is_four_l1241_124169


namespace NUMINAMATH_CALUDE_rectangle_waste_area_l1241_124194

theorem rectangle_waste_area (x y : ℝ) (h1 : x + 2*y = 7) (h2 : 2*x + 3*y = 11) : 
  let a := Real.sqrt (x^2 + y^2)
  let total_area := 11 * 7
  let waste_area := total_area - 4 * a^2
  let waste_percentage := (waste_area / total_area) * 100
  ∃ ε > 0, abs (waste_percentage - 48) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_waste_area_l1241_124194
