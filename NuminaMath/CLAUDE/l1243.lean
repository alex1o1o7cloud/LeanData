import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l1243_124314

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1243_124314


namespace NUMINAMATH_CALUDE_sailing_speed_calculation_l1243_124388

/-- The sailing speed of a ship in still water, given the following conditions:
  * Two ships (Knight and Warrior) depart from ports A and B at 8 a.m.
  * They travel towards each other, turn around at the opposite port, and return to their starting points.
  * Both ships return to their starting points at 10 a.m.
  * The time it takes for the ships to travel in the same direction is 10 minutes.
  * The speed of the current is 0.5 meters per second.
-/
def sailing_speed : ℝ := 6

/-- The speed of the current in meters per second. -/
def current_speed : ℝ := 0.5

/-- The time it takes for the ships to travel in the same direction, in seconds. -/
def same_direction_time : ℝ := 600

/-- The total travel time for each ship, in seconds. -/
def total_travel_time : ℝ := 7200

theorem sailing_speed_calculation :
  let v := sailing_speed
  let c := current_speed
  let t := same_direction_time
  let T := total_travel_time
  (v + c) * t + (v - c) * t = v * T ∧
  2 * ((v + c)⁻¹ + (v - c)⁻¹) * (v * t) = T :=
by sorry

#check sailing_speed_calculation

end NUMINAMATH_CALUDE_sailing_speed_calculation_l1243_124388


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1243_124370

theorem perfect_square_polynomial (n : ℤ) : 
  ∃ (m : ℤ), n^4 + 6*n^3 + 11*n^2 + 3*n + 31 = m^2 ↔ n = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1243_124370


namespace NUMINAMATH_CALUDE_vector_collinearity_l1243_124311

def a (k : ℝ) : Fin 2 → ℝ := ![1, k]
def b : Fin 2 → ℝ := ![2, 2]

theorem vector_collinearity (k : ℝ) :
  (∀ (i : Fin 2), (a k + b) i = (a k) i * (3 : ℝ)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1243_124311


namespace NUMINAMATH_CALUDE_cosine_function_vertical_shift_l1243_124350

/-- Given a cosine function y = a * cos(b * x + c) + d that oscillates between 5 and -3,
    prove that the vertical shift d equals 1. -/
theorem cosine_function_vertical_shift
  (a b c d : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_oscillation : ∀ x, -3 ≤ a * Real.cos (b * x + c) + d ∧ 
                        a * Real.cos (b * x + c) + d ≤ 5) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_cosine_function_vertical_shift_l1243_124350


namespace NUMINAMATH_CALUDE_third_guinea_pig_eats_more_l1243_124363

/-- The number of guinea pigs Rollo has -/
def num_guinea_pigs : ℕ := 3

/-- The amount of food the first guinea pig eats in cups -/
def first_guinea_pig_food : ℕ := 2

/-- The amount of food the second guinea pig eats in cups -/
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food

/-- The total amount of food for all guinea pigs in cups -/
def total_food : ℕ := 13

/-- The amount of food the third guinea pig eats in cups -/
def third_guinea_pig_food : ℕ := total_food - first_guinea_pig_food - second_guinea_pig_food

theorem third_guinea_pig_eats_more :
  third_guinea_pig_food = second_guinea_pig_food + 3 := by
  sorry

end NUMINAMATH_CALUDE_third_guinea_pig_eats_more_l1243_124363


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l1243_124352

theorem real_part_of_complex_number (z : ℂ) : z = (Complex.I^3) / (1 + 2 * Complex.I) → Complex.re z = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l1243_124352


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1243_124386

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

def is_center (h k : ℝ) : Prop := ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4

theorem circle_center_and_radius :
  is_center 2 0 ∧ ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + y^2 ≤ 4 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1243_124386


namespace NUMINAMATH_CALUDE_ball_probabilities_l1243_124338

/-- The number of red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of white balls in the bag -/
def num_white_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- The probability of picking a red ball in a single draw -/
def prob_red : ℚ := num_red_balls / total_balls

/-- The probability of picking a white ball in a single draw -/
def prob_white : ℚ := num_white_balls / total_balls

theorem ball_probabilities :
  -- Statement A
  (Nat.choose 2 1 * Nat.choose 4 2 : ℚ) / Nat.choose 6 3 = 3 / 5 ∧
  -- Statement B
  (6 : ℚ) * prob_red * (1 - prob_red) = 4 / 3 ∧
  -- Statement C
  (4 : ℚ) / 6 * 3 / 5 = 2 / 5 ∧
  -- Statement D
  1 - (1 - prob_red) ^ 3 = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1243_124338


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1243_124329

theorem imaginary_part_of_z (m : ℝ) :
  let z := (2 + m * Complex.I) / (1 + Complex.I)
  (z.re = 0) → z.im = -2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1243_124329


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l1243_124381

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), 
  (x^2 + y = a^2) ∧ (y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l1243_124381


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l1243_124345

-- Define the sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | x^2 - x < 0}

-- State the theorem
theorem union_M_complement_N_equals_R : M ∪ (Set.univ \ N) = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_R_l1243_124345


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1243_124336

theorem smallest_three_digit_multiple_of_13 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 104 → ¬(∃ k : ℕ, n = 13 * k) :=
by
  sorry

#check smallest_three_digit_multiple_of_13

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_13_l1243_124336


namespace NUMINAMATH_CALUDE_snail_square_exists_l1243_124347

/-- A natural number is a "snail" number if it can be formed by concatenating
    three consecutive natural numbers in some order. -/
def is_snail (n : ℕ) : Prop :=
  ∃ a b c : ℕ, b = a + 1 ∧ c = b + 1 ∧
  (n.repr = a.repr ++ b.repr ++ c.repr ∨
   n.repr = a.repr ++ c.repr ++ b.repr ∨
   n.repr = b.repr ++ a.repr ++ c.repr ∨
   n.repr = b.repr ++ c.repr ++ a.repr ∨
   n.repr = c.repr ++ a.repr ++ b.repr ∨
   n.repr = c.repr ++ b.repr ++ a.repr)

theorem snail_square_exists :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ is_snail n ∧ ∃ m : ℕ, n = m^2 :=
by
  use 1089
  sorry

end NUMINAMATH_CALUDE_snail_square_exists_l1243_124347


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l1243_124353

/-- Given two points on the parabola y = 2x^2 with perpendicular tangents, 
    their intersection point has y-coordinate -1/2 -/
theorem intersection_of_perpendicular_tangents_on_parabola 
  (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 2 * a^2)
  let B : ℝ × ℝ := (b, 2 * b^2)
  let tangent_A (x : ℝ) := 4 * a * x - 2 * a^2
  let tangent_B (x : ℝ) := 4 * b * x - 2 * b^2
  -- Condition: A and B are on the parabola y = 2x^2
  -- Condition: Tangents at A and B are perpendicular
  4 * a * 4 * b = -1 →
  -- Conclusion: The y-coordinate of the intersection point P is -1/2
  ∃ x, tangent_A x = tangent_B x ∧ tangent_A x = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_tangents_on_parabola_l1243_124353


namespace NUMINAMATH_CALUDE_max_x_value_l1243_124310

theorem max_x_value (x : ℝ) : 
  (((5*x - 20) / (4*x - 5))^2 + ((5*x - 20) / (4*x - 5)) = 20) → 
  x ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l1243_124310


namespace NUMINAMATH_CALUDE_constant_sign_of_root_combination_l1243_124328

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ

/-- The roots of P(x) - t for a given t -/
def roots (P : Polynomial4) (t : ℝ) : Fin 4 → ℝ := sorry

/-- Predicate to check if P(x) - t has four distinct real roots -/
def has_four_distinct_real_roots (P : Polynomial4) (t : ℝ) : Prop := sorry

theorem constant_sign_of_root_combination (P : Polynomial4) :
  ∀ t₁ t₂ : ℝ, has_four_distinct_real_roots P t₁ → has_four_distinct_real_roots P t₂ →
  (roots P t₁ 0 + roots P t₁ 3 - roots P t₁ 1 - roots P t₁ 2) *
  (roots P t₂ 0 + roots P t₂ 3 - roots P t₂ 1 - roots P t₂ 2) > 0 :=
sorry

end NUMINAMATH_CALUDE_constant_sign_of_root_combination_l1243_124328


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1243_124377

theorem least_sum_of_bases (a b : ℕ+) : 
  (6 * a.val + 3 = 3 * b.val + 6) →
  (∀ (a' b' : ℕ+), (6 * a'.val + 3 = 3 * b'.val + 6) → (a'.val + b'.val ≥ a.val + b.val)) →
  a.val + b.val = 20 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1243_124377


namespace NUMINAMATH_CALUDE_dog_bones_problem_l1243_124396

theorem dog_bones_problem (initial_bones final_bones : ℕ) 
  (h1 : initial_bones = 493)
  (h2 : final_bones = 860) :
  final_bones - initial_bones = 367 := by
  sorry

end NUMINAMATH_CALUDE_dog_bones_problem_l1243_124396


namespace NUMINAMATH_CALUDE_base_n_representation_of_d_l1243_124348

/-- Represents a number in base n --/
structure BaseN (n : ℕ) where
  digits : List ℕ
  all_less : ∀ d ∈ digits, d < n

/-- Convert a base-n number to its decimal representation --/
def toDecimal (n : ℕ) (b : BaseN n) : ℕ :=
  b.digits.enum.foldl (fun acc (i, d) => acc + d * n ^ i) 0

theorem base_n_representation_of_d (n : ℕ) (c d : ℕ) :
  n > 8 →
  n ^ 2 - c * n + d = 0 →
  toDecimal n ⟨[2, 1], by sorry⟩ = c →
  toDecimal n ⟨[0, 1, 1], by sorry⟩ = d :=
by sorry

end NUMINAMATH_CALUDE_base_n_representation_of_d_l1243_124348


namespace NUMINAMATH_CALUDE_gwen_math_problems_l1243_124330

theorem gwen_math_problems 
  (science_problems : ℕ) 
  (finished_problems : ℕ) 
  (remaining_problems : ℕ) 
  (h1 : science_problems = 11)
  (h2 : finished_problems = 24)
  (h3 : remaining_problems = 5)
  (h4 : finished_problems + remaining_problems = science_problems + math_problems) :
  math_problems = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_gwen_math_problems_l1243_124330


namespace NUMINAMATH_CALUDE_work_completion_time_l1243_124394

theorem work_completion_time (b_time : ℕ) (joint_time : ℕ) (b_remaining_time : ℕ) :
  b_time = 40 → joint_time = 9 → b_remaining_time = 23 →
  ∃ (a_time : ℕ),
    (joint_time : ℚ) * ((1 : ℚ) / a_time + (1 : ℚ) / b_time) + 
    (b_remaining_time : ℚ) * ((1 : ℚ) / b_time) = 1 ∧
    a_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1243_124394


namespace NUMINAMATH_CALUDE_james_total_matches_l1243_124379

/-- The number of boxes in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def james_dozens : ℕ := 5

/-- The number of matches in each box -/
def matches_per_box : ℕ := 20

/-- Theorem: James has 1200 matches in total -/
theorem james_total_matches :
  james_dozens * dozen * matches_per_box = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_total_matches_l1243_124379


namespace NUMINAMATH_CALUDE_shooter_probability_l1243_124303

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l1243_124303


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1243_124318

/-- Given a geometric sequence {a_n} where each term is positive and 2a_1 + a_2 = a_3,
    prove that (a_4 + a_5) / (a_3 + a_4) = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_cond : 2 * a 1 + a 2 = a 3) :
  (a 4 + a 5) / (a 3 + a 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1243_124318


namespace NUMINAMATH_CALUDE_club_simplifier_probability_l1243_124399

def probability_more_wins_than_losses (num_matches : ℕ) 
  (prob_win prob_lose prob_tie : ℚ) : ℚ :=
  sorry

theorem club_simplifier_probability :
  probability_more_wins_than_losses 3 (1/2) (1/4) (1/4) = 25/64 :=
by sorry

end NUMINAMATH_CALUDE_club_simplifier_probability_l1243_124399


namespace NUMINAMATH_CALUDE_siblings_average_age_l1243_124354

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) :
  youngest_age = 17 →
  age_differences = [4, 5, 7] →
  (youngest_age + (age_differences.map (λ d => youngest_age + d)).sum) / 4 = 21 :=
by sorry

end NUMINAMATH_CALUDE_siblings_average_age_l1243_124354


namespace NUMINAMATH_CALUDE_wang_shifu_not_yuan_dramatist_l1243_124341

/-- The set of four great dramatists of the Yuan Dynasty -/
def YuanDramatists : Set String :=
  {"Guan Hanqing", "Zheng Guangzu", "Bai Pu", "Ma Zhiyuan"}

/-- Wang Shifu -/
def WangShifu : String := "Wang Shifu"

/-- Theorem stating that Wang Shifu is not one of the four great dramatists of the Yuan Dynasty -/
theorem wang_shifu_not_yuan_dramatist :
  WangShifu ∉ YuanDramatists := by
  sorry

end NUMINAMATH_CALUDE_wang_shifu_not_yuan_dramatist_l1243_124341


namespace NUMINAMATH_CALUDE_lock_settings_count_l1243_124375

/-- The number of digits on each dial of the lock -/
def numDigits : ℕ := 10

/-- The number of dials on the lock -/
def numDials : ℕ := 4

/-- Calculates the number of different settings for the lock -/
def lockSettings : ℕ := numDigits * (numDigits - 1) * (numDigits - 2) * (numDigits - 3)

/-- Theorem stating that the number of different settings for the lock is 5040 -/
theorem lock_settings_count : lockSettings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_lock_settings_count_l1243_124375


namespace NUMINAMATH_CALUDE_intersection_length_l1243_124371

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The line l passing through (0, 1) -/
def l (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

/-- Point A on the intersection of C and l -/
def A (k x₁ y₁ : ℝ) : Prop := C x₁ y₁ ∧ l k x₁ y₁

/-- Point B on the intersection of C and l -/
def B (k x₂ y₂ : ℝ) : Prop := C x₂ y₂ ∧ l k x₂ y₂

/-- The condition that OA · AB = 0 -/
def orthogonal (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem intersection_length 
  (k x₁ y₁ x₂ y₂ : ℝ) 
  (hA : A k x₁ y₁) 
  (hB : B k x₂ y₂) 
  (hO : orthogonal x₁ y₁ x₂ y₂) : 
  ((x₂ - x₁)^2 + (y₂ - y₁)^2) = (4*Real.sqrt 65/17)^2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_length_l1243_124371


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1243_124376

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1243_124376


namespace NUMINAMATH_CALUDE_triangle_radius_equality_l1243_124380

/-- For a triangle with sides a, b, c, circumradius R, inradius r, and semi-perimeter p,
    prove that ab + bc + ac = r² + p² + 4Rr -/
theorem triangle_radius_equality (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  a * b + b * c + a * c = r^2 + p^2 + 4 * R * r := by
sorry

end NUMINAMATH_CALUDE_triangle_radius_equality_l1243_124380


namespace NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l1243_124374

/-- Given a 2x2 matrix B with real entries such that B^4 = 0, prove that B^3 = 0 -/
theorem nilpotent_matrix_cube_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_cube_zero_l1243_124374


namespace NUMINAMATH_CALUDE_odd_then_even_probability_l1243_124322

/-- Represents a ball with a number -/
structure Ball :=
  (number : Nat)

/-- Represents the bag of balls -/
def Bag : Finset Ball := sorry

/-- The bag contains 5 balls numbered 1 to 5 -/
axiom bag_content : Bag.card = 5 ∧ ∀ n : Nat, 1 ≤ n ∧ n ≤ 5 → ∃! b : Ball, b ∈ Bag ∧ b.number = n

/-- A ball is odd-numbered if its number is odd -/
def is_odd (b : Ball) : Prop := b.number % 2 = 1

/-- A ball is even-numbered if its number is even -/
def is_even (b : Ball) : Prop := b.number % 2 = 0

/-- The probability of drawing an odd-numbered ball first and an even-numbered ball second -/
def prob_odd_then_even : ℚ := sorry

/-- The main theorem to prove -/
theorem odd_then_even_probability : prob_odd_then_even = 3 / 10 := sorry

end NUMINAMATH_CALUDE_odd_then_even_probability_l1243_124322


namespace NUMINAMATH_CALUDE_monroe_made_200_granola_bars_l1243_124337

/-- The number of granola bars Monroe made -/
def total_granola_bars : ℕ := sorry

/-- The number of granola bars eaten by Monroe and her husband -/
def eaten_by_parents : ℕ := 80

/-- The number of children in Monroe's family -/
def number_of_children : ℕ := 6

/-- The number of granola bars each child received -/
def bars_per_child : ℕ := 20

/-- Theorem stating that Monroe made 200 granola bars -/
theorem monroe_made_200_granola_bars :
  total_granola_bars = eaten_by_parents + number_of_children * bars_per_child :=
sorry

end NUMINAMATH_CALUDE_monroe_made_200_granola_bars_l1243_124337


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1243_124327

theorem inequality_solution_set : 
  {x : ℝ | x^2 - x - 6 < 0} = Set.Ioo (-2) 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1243_124327


namespace NUMINAMATH_CALUDE_total_ebook_readers_l1243_124308

/-- The number of eBook readers Anna bought -/
def anna_readers : ℕ := 50

/-- The number of eBook readers John bought initially -/
def john_initial_readers : ℕ := anna_readers - 15

/-- The number of eBook readers John lost -/
def john_lost_readers : ℕ := 3

/-- The number of eBook readers John has after losing some -/
def john_final_readers : ℕ := john_initial_readers - john_lost_readers

/-- The total number of eBook readers John and Anna have together -/
def total_readers : ℕ := anna_readers + john_final_readers

theorem total_ebook_readers :
  total_readers = 82 :=
by sorry

end NUMINAMATH_CALUDE_total_ebook_readers_l1243_124308


namespace NUMINAMATH_CALUDE_factor_72x3_minus_252x7_l1243_124361

theorem factor_72x3_minus_252x7 (x : ℝ) : 72 * x^3 - 252 * x^7 = 36 * x^3 * (2 - 7 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_72x3_minus_252x7_l1243_124361


namespace NUMINAMATH_CALUDE_sum_of_square_roots_l1243_124326

theorem sum_of_square_roots : 
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_l1243_124326


namespace NUMINAMATH_CALUDE_unique_M_condition_l1243_124356

theorem unique_M_condition (M : ℝ) : 
  (M > 0 ∧ 
   (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
     (a + M / (a * b) ≥ 1 + M ∨ 
      b + M / (b * c) ≥ 1 + M ∨ 
      c + M / (c * a) ≥ 1 + M))) ↔ 
  M = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_unique_M_condition_l1243_124356


namespace NUMINAMATH_CALUDE_max_diff_inequality_l1243_124333

open Function Set

variable {n : ℕ}

/-- Two strictly increasing finite sequences of real numbers -/
def StrictlyIncreasingSeq (a b : Fin n → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j ∧ b i < b j

theorem max_diff_inequality
  (a b : Fin n → ℝ)
  (h_inc : StrictlyIncreasingSeq a b)
  (f : Fin n → Fin n)
  (h_bij : Bijective f) :
  (⨆ i, |a i - b i|) ≤ (⨆ i, |a i - b (f i)|) :=
sorry

end NUMINAMATH_CALUDE_max_diff_inequality_l1243_124333


namespace NUMINAMATH_CALUDE_sarahs_age_proof_l1243_124313

theorem sarahs_age_proof (ana billy mark sarah : ℕ) : 
  ana + 8 = 40 → 
  billy = ana / 2 → 
  mark = billy + 4 → 
  sarah = 3 * mark - 4 → 
  sarah = 56 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_age_proof_l1243_124313


namespace NUMINAMATH_CALUDE_joyce_apples_l1243_124305

/-- The number of apples Joyce ends up with after giving some away -/
def apples_remaining (starting_apples given_away : ℕ) : ℕ :=
  starting_apples - given_away

/-- Theorem stating that Joyce ends up with 23 apples -/
theorem joyce_apples : apples_remaining 75 52 = 23 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l1243_124305


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l1243_124397

theorem triangle_angle_inequality (X Y Z : ℝ) 
  (h_positive : X > 0 ∧ Y > 0 ∧ Z > 0) 
  (h_sum : 2 * X + 2 * Y + 2 * Z = π) : 
  (Real.sin X / Real.cos (Y - Z)) + 
  (Real.sin Y / Real.cos (Z - X)) + 
  (Real.sin Z / Real.cos (X - Y)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l1243_124397


namespace NUMINAMATH_CALUDE_negative_reciprocal_inequality_l1243_124383

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -1/a < -1/b := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_inequality_l1243_124383


namespace NUMINAMATH_CALUDE_proportion_solution_l1243_124365

theorem proportion_solution (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1243_124365


namespace NUMINAMATH_CALUDE_power_equation_solution_l1243_124359

theorem power_equation_solution : 2^5 - 7 = 3^3 + (-2) := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1243_124359


namespace NUMINAMATH_CALUDE_complex_subtraction_l1243_124325

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 + I) (h2 : b = 2 - 3 * I) :
  a - 3 * b = 11 - 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1243_124325


namespace NUMINAMATH_CALUDE_toys_after_game_purchase_l1243_124351

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 63 → game_cost = 48 → toy_cost = 3 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_after_game_purchase_l1243_124351


namespace NUMINAMATH_CALUDE_last_digit_periodicity_last_digits_first_five_l1243_124358

def a (n : ℕ+) : ℕ := (n - 1) * n

theorem last_digit_periodicity (n : ℕ+) :
  ∃ (k : ℕ+), a (n + 5 * k) % 10 = a n % 10 :=
sorry

theorem last_digits_first_five :
  (a 1 % 10 = 0) ∧
  (a 2 % 10 = 2) ∧
  (a 3 % 10 = 6) ∧
  (a 4 % 10 = 2) ∧
  (a 5 % 10 = 0) :=
sorry

end NUMINAMATH_CALUDE_last_digit_periodicity_last_digits_first_five_l1243_124358


namespace NUMINAMATH_CALUDE_odd_function_value_l1243_124346

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : IsOdd f)
    (h_periodic : ∀ x, f (x + 4) = f x + f 2) (h_f_neg_one : f (-1) = -2) :
    f 2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l1243_124346


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1243_124321

theorem ratio_of_numbers (a b : ℕ) (h1 : a > b) (h2 : a + b = 96) (h3 : a = 64) (h4 : b = 32) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1243_124321


namespace NUMINAMATH_CALUDE_ruined_tomatoes_percentage_l1243_124364

/-- The percentage of ruined and discarded tomatoes -/
def ruined_percentage : ℝ := 15

/-- The purchase price per pound of tomatoes -/
def purchase_price : ℝ := 0.80

/-- The desired profit percentage on the cost of tomatoes -/
def profit_percentage : ℝ := 8

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 1.0165

/-- Theorem stating that given the purchase price, profit percentage, and selling price,
    the percentage of ruined and discarded tomatoes is approximately 15% -/
theorem ruined_tomatoes_percentage :
  ∀ (W : ℝ), W > 0 →
  selling_price * (100 - ruined_percentage) / 100 * W - purchase_price * W =
  profit_percentage / 100 * purchase_price * W :=
by sorry

end NUMINAMATH_CALUDE_ruined_tomatoes_percentage_l1243_124364


namespace NUMINAMATH_CALUDE_trig_identity_l1243_124390

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  2 * (Real.sin α)^2 + 4 * (Real.sin α) * (Real.cos α) - 9 * (Real.cos α)^2 = 21/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1243_124390


namespace NUMINAMATH_CALUDE_max_value_x_plus_7y_l1243_124319

theorem max_value_x_plus_7y :
  ∀ x y : ℝ,
  0 ≤ x ∧ x ≤ 1 →
  0 ≤ y ∧ y ≤ 1 →
  Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + Real.sqrt (y * (1 - x)) / Real.sqrt 7 →
  (∀ z w : ℝ, 0 ≤ z ∧ z ≤ 1 → 0 ≤ w ∧ w ≤ 1 → z + 7 * w ≤ 57 / 8) ∧
  ∃ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ a + 7 * b = 57 / 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_7y_l1243_124319


namespace NUMINAMATH_CALUDE_consecutive_triangle_altitude_l1243_124331

/-- Represents a triangle with consecutive integer side lengths -/
structure ConsecutiveTriangle where
  a : ℕ
  is_acute : a > 0

/-- The altitude from the vertex opposite the middle side -/
def altitude (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The length of the shorter segment of the middle side -/
def shorter_segment (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The length of the longer segment of the middle side -/
def longer_segment (t : ConsecutiveTriangle) : ℝ :=
  sorry

/-- The theorem stating the properties of the triangle -/
theorem consecutive_triangle_altitude (t : ConsecutiveTriangle) :
  (longer_segment t - shorter_segment t = 4) →
  (∃ n : ℕ, altitude t = n) →
  t.a + 1 ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_consecutive_triangle_altitude_l1243_124331


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1243_124384

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1243_124384


namespace NUMINAMATH_CALUDE_increasing_quadratic_implies_a_ge_five_l1243_124307

/-- Given a function f(x) = -x^2 + 2(a-1)x + 2 that is increasing on the interval (-∞, 4),
    prove that a ≥ 5. -/
theorem increasing_quadratic_implies_a_ge_five (a : ℝ) :
  (∀ x < 4, Monotone (fun x => -x^2 + 2*(a-1)*x + 2)) →
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_implies_a_ge_five_l1243_124307


namespace NUMINAMATH_CALUDE_mapping_preimage_property_l1243_124309

theorem mapping_preimage_property (A B : Type) (f : A → B) :
  ∃ (b : B), ∃ (a1 a2 : A), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end NUMINAMATH_CALUDE_mapping_preimage_property_l1243_124309


namespace NUMINAMATH_CALUDE_dave_toy_tickets_l1243_124362

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys (initial_tickets clothes_tickets toy_extra : ℕ) : ℕ :=
  clothes_tickets + toy_extra

/-- Proof that Dave used 12 tickets to buy toys -/
theorem dave_toy_tickets :
  let initial_tickets : ℕ := 19
  let clothes_tickets : ℕ := 7
  let toy_extra : ℕ := 5
  tickets_for_toys initial_tickets clothes_tickets toy_extra = 12 := by
  sorry

end NUMINAMATH_CALUDE_dave_toy_tickets_l1243_124362


namespace NUMINAMATH_CALUDE_julia_watch_collection_l1243_124387

theorem julia_watch_collection :
  let silver_watches : ℕ := 20
  let bronze_watches : ℕ := 3 * silver_watches
  let platinum_watches : ℕ := 2 * bronze_watches
  let gold_watches : ℕ := (silver_watches + platinum_watches) / 5
  let total_watches : ℕ := silver_watches + bronze_watches + platinum_watches + gold_watches
  total_watches = 228 :=
by sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l1243_124387


namespace NUMINAMATH_CALUDE_bryan_mineral_samples_l1243_124301

/-- The number of mineral samples per shelf -/
def samples_per_shelf : ℕ := 65

/-- The number of shelves -/
def number_of_shelves : ℕ := 7

/-- The total number of mineral samples -/
def total_samples : ℕ := samples_per_shelf * number_of_shelves

theorem bryan_mineral_samples :
  total_samples = 455 :=
sorry

end NUMINAMATH_CALUDE_bryan_mineral_samples_l1243_124301


namespace NUMINAMATH_CALUDE_simplify_tan_product_l1243_124385

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_product_l1243_124385


namespace NUMINAMATH_CALUDE_integral_x_squared_sqrt_25_minus_x_squared_l1243_124349

theorem integral_x_squared_sqrt_25_minus_x_squared : 
  ∫ x in (0)..(5), x^2 * Real.sqrt (25 - x^2) = (625 * Real.pi) / 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_sqrt_25_minus_x_squared_l1243_124349


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1243_124335

theorem unique_solution_lcm_gcd_equation :
  ∃! n : ℕ+, n.val > 0 ∧ Nat.lcm n.val 180 = Nat.gcd n.val 180 + 360 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l1243_124335


namespace NUMINAMATH_CALUDE_inverse_function_b_value_l1243_124368

theorem inverse_function_b_value (f : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 5 - b * x) →
  (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g (-3) = 3) →
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_inverse_function_b_value_l1243_124368


namespace NUMINAMATH_CALUDE_min_value_of_f_l1243_124304

/-- The function f(x) = 3x^2 - 6x + 9 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 9

/-- The minimum value of f(x) is 6 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1243_124304


namespace NUMINAMATH_CALUDE_half_oz_mixture_bubbles_l1243_124320

/-- The number of bubbles that can be made from one ounce of Dawn liquid soap -/
def dawn_bubbles_per_oz : ℕ := 200000

/-- The number of bubbles that can be made from one ounce of Dr. Bronner's liquid soap -/
def bronner_bubbles_per_oz : ℕ := 2 * dawn_bubbles_per_oz

/-- The number of bubbles that can be made from one ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps -/
def mixture_bubbles_per_oz : ℕ := (dawn_bubbles_per_oz + bronner_bubbles_per_oz) / 2

/-- Theorem: One half ounce of an equal mixture of Dawn and Dr. Bronner's liquid soaps can make 150,000 bubbles -/
theorem half_oz_mixture_bubbles : mixture_bubbles_per_oz / 2 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_half_oz_mixture_bubbles_l1243_124320


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l1243_124323

def tank_capacity (initial_loss_rate : ℕ) (initial_loss_hours : ℕ) 
  (secondary_loss_rate : ℕ) (secondary_loss_hours : ℕ)
  (fill_rate : ℕ) (fill_hours : ℕ) (remaining_to_fill : ℕ) : ℕ :=
  (initial_loss_rate * initial_loss_hours) + 
  (secondary_loss_rate * secondary_loss_hours) + 
  (fill_rate * fill_hours) + 
  remaining_to_fill

theorem tank_capacity_proof : 
  tank_capacity 32000 5 10000 10 40000 3 140000 = 520000 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l1243_124323


namespace NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l1243_124391

/-- Given a square and a regular octagon with equal perimeters, 
    if the area of the square is 16, then the area of the octagon is 8 + 4√2 -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 → 4 * a = 8 * b → a^2 = 16 → 
  2 * (1 + Real.sqrt 2) * b^2 = 8 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_given_equal_perimeter_and_square_area_l1243_124391


namespace NUMINAMATH_CALUDE_max_table_sum_l1243_124389

def primes : List Nat := [2, 3, 5, 7, 17, 19]

def is_valid_arrangement (top bottom : List Nat) : Prop :=
  top.length = 3 ∧ bottom.length = 3 ∧ 
  (∀ x ∈ top, x ∈ primes) ∧ (∀ x ∈ bottom, x ∈ primes) ∧
  (∀ x ∈ top, x ∉ bottom) ∧ (∀ x ∈ bottom, x ∉ top)

def table_sum (top bottom : List Nat) : Nat :=
  (top.sum * bottom.sum)

theorem max_table_sum :
  ∀ top bottom : List Nat,
  is_valid_arrangement top bottom →
  table_sum top bottom ≤ 682 :=
sorry

end NUMINAMATH_CALUDE_max_table_sum_l1243_124389


namespace NUMINAMATH_CALUDE_number_of_students_l1243_124355

theorem number_of_students : ∃ (x : ℕ), 
  (∃ (total : ℕ), total = 3 * x + 8) ∧ 
  (5 * (x - 1) + 3 > 3 * x + 8) ∧
  (3 * x + 8 ≥ 5 * (x - 1)) ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_l1243_124355


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1243_124373

-- Define the number of sibling pairs
def num_sibling_pairs : ℕ := 4

-- Define the number of seats in each row
def seats_per_row : ℕ := 4

-- Define the number of rows in the van
def num_rows : ℕ := 2

-- Define the derangement function for 4 objects
def derangement_4 : ℕ := 9

-- Theorem statement
theorem seating_arrangements_count :
  (seats_per_row.factorial) * derangement_4 * (2^num_sibling_pairs) = 3456 := by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_count_l1243_124373


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1243_124392

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, 0, 1}
def B : Set Int := {-1, 0, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {-2, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1243_124392


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1243_124334

theorem smallest_n_congruence (n : ℕ+) : 
  (∀ k : ℕ+, k < n → (7 : ℤ)^(k : ℕ) % 5 ≠ (k : ℤ)^7 % 5) ∧ 
  (7 : ℤ)^(n : ℕ) % 5 = (n : ℤ)^7 % 5 → 
  n = 7 := by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1243_124334


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_l1243_124369

/-- Given an arithmetic sequence {a_n} with a_1 = 20, a_n = 54, and S_n = 999, prove that n = 27 -/
theorem arithmetic_sequence_n (a : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : 
  (∀ k, a (k + 1) - a k = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 20 →
  a n = 54 →
  S_n = 999 →
  n = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_l1243_124369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1243_124366

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  common_difference a = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1243_124366


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l1243_124302

-- Define a parallelogram
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_property : (C - B) = (D - A) ∧ (D - C) = (B - A))

-- Theorem statement
theorem parallelogram_diagonal_sum 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.B - ABCD.A + (ABCD.D - ABCD.A) = ABCD.C - ABCD.A :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_sum_l1243_124302


namespace NUMINAMATH_CALUDE_clearance_sale_prices_l1243_124382

/-- Calculates the final price after applying two successive discounts --/
def finalPrice (initialPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initialPrice * (1 - discount1) * (1 - discount2)

/-- Proves that the final prices of the hat and gloves are correct --/
theorem clearance_sale_prices 
  (hatInitialPrice : ℝ) 
  (hatDiscount1 : ℝ) 
  (hatDiscount2 : ℝ)
  (glovesInitialPrice : ℝ) 
  (glovesDiscount1 : ℝ) 
  (glovesDiscount2 : ℝ)
  (hatInitialPrice_eq : hatInitialPrice = 15)
  (hatDiscount1_eq : hatDiscount1 = 0.20)
  (hatDiscount2_eq : hatDiscount2 = 0.40)
  (glovesInitialPrice_eq : glovesInitialPrice = 8)
  (glovesDiscount1_eq : glovesDiscount1 = 0.25)
  (glovesDiscount2_eq : glovesDiscount2 = 0.30) :
  finalPrice hatInitialPrice hatDiscount1 hatDiscount2 = 7.20 ∧
  finalPrice glovesInitialPrice glovesDiscount1 glovesDiscount2 = 4.20 := by
  sorry

#check clearance_sale_prices

end NUMINAMATH_CALUDE_clearance_sale_prices_l1243_124382


namespace NUMINAMATH_CALUDE_triangle_side_length_l1243_124324

/-- Given a triangle ABC with the following properties:
  - f(x) = 2sin(2x + π/6) + 1
  - f(A) = 2
  - b = 1
  - Area of triangle ABC is √3/2
  Prove that a = √3 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6) + 1) →
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 3 := by 
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1243_124324


namespace NUMINAMATH_CALUDE_election_votes_proof_l1243_124340

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 360) : 
  (62 * total_votes) / 100 = 930 := by
sorry

end NUMINAMATH_CALUDE_election_votes_proof_l1243_124340


namespace NUMINAMATH_CALUDE_natural_number_pairs_l1243_124393

theorem natural_number_pairs : ∀ a b : ℕ,
  (90 < a + b ∧ a + b < 100) ∧ (9/10 < (a : ℚ) / (b : ℚ) ∧ (a : ℚ) / (b : ℚ) < 91/100) ↔
  ((a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52)) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l1243_124393


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1243_124339

theorem triangle_isosceles (A B C : ℝ) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : A = B :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1243_124339


namespace NUMINAMATH_CALUDE_lindys_speed_l1243_124360

/-- Proves that Lindy's speed is 9 feet per second given the problem conditions --/
theorem lindys_speed (initial_distance : ℝ) (jack_speed christina_speed : ℝ) 
  (lindy_distance : ℝ) : ℝ :=
by
  -- Define the given conditions
  have h1 : initial_distance = 240 := by sorry
  have h2 : jack_speed = 5 := by sorry
  have h3 : christina_speed = 3 := by sorry
  have h4 : lindy_distance = 270 := by sorry

  -- Calculate the time it takes for Jack and Christina to meet
  let total_speed := jack_speed + christina_speed
  let time_to_meet := initial_distance / total_speed

  -- Calculate Lindy's speed
  let lindy_speed := lindy_distance / time_to_meet

  -- Prove that Lindy's speed is 9 feet per second
  have h5 : lindy_speed = 9 := by sorry

  exact lindy_speed

end NUMINAMATH_CALUDE_lindys_speed_l1243_124360


namespace NUMINAMATH_CALUDE_a_range_for_unique_positive_zero_l1243_124332

/-- The function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The statement that f has only one zero point -/
def has_unique_zero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem a_range_for_unique_positive_zero (a : ℝ) :
  (has_unique_zero (f a)) ∧ (∃ x₀ > 0, f a x₀ = 0) → a < -2 :=
sorry

end NUMINAMATH_CALUDE_a_range_for_unique_positive_zero_l1243_124332


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l1243_124306

/-- Calculates the perimeter of a rectangular field enclosed by evenly spaced posts -/
def field_perimeter (num_posts : ℕ) (post_width : ℚ) (gap_width : ℚ) : ℚ :=
  let short_side_posts := num_posts / 3
  let short_side_gaps := short_side_posts - 1
  let short_side_length := short_side_gaps * gap_width + short_side_posts * (post_width / 12)
  let long_side_length := 2 * short_side_length
  2 * (short_side_length + long_side_length)

theorem rectangular_field_perimeter :
  field_perimeter 36 (4 / 1) (7 / 2) = 238 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l1243_124306


namespace NUMINAMATH_CALUDE_galia_number_transformation_l1243_124343

theorem galia_number_transformation (k : ℝ) :
  (∃ N : ℝ, ((k * N + N) / N - N = k - 100)) → (∃ N : ℝ, N = 101) :=
by sorry

end NUMINAMATH_CALUDE_galia_number_transformation_l1243_124343


namespace NUMINAMATH_CALUDE_parameterization_validity_l1243_124342

/-- The slope of the line -/
def m : ℚ := 7/4

/-- The y-intercept of the line -/
def b : ℚ := -14/4

/-- The line equation -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- Vector parameterization A -/
def param_A (t : ℚ) : ℚ × ℚ := (3 + 4*t, -5/4 + 7*t)

/-- Vector parameterization B -/
def param_B (t : ℚ) : ℚ × ℚ := (7 + 8*t, 7/4 + 14*t)

/-- Vector parameterization C -/
def param_C (t : ℚ) : ℚ × ℚ := (2 + 14*t, 1/2 + 7*t)

/-- Vector parameterization D -/
def param_D (t : ℚ) : ℚ × ℚ := (-1 + 8*t, -27/4 - 15*t)

/-- Vector parameterization E -/
def param_E (t : ℚ) : ℚ × ℚ := (4 - 7*t, 9/2 + 5*t)

theorem parameterization_validity :
  (∀ t, line_eq (param_A t).1 (param_A t).2) ∧
  (∀ t, line_eq (param_B t).1 (param_B t).2) ∧
  ¬(∀ t, line_eq (param_C t).1 (param_C t).2) ∧
  ¬(∀ t, line_eq (param_D t).1 (param_D t).2) ∧
  ¬(∀ t, line_eq (param_E t).1 (param_E t).2) :=
by sorry

end NUMINAMATH_CALUDE_parameterization_validity_l1243_124342


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_103_l1243_124367

theorem units_digit_of_7_to_103 : ∃ n : ℕ, 7^103 ≡ 3 [ZMOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_103_l1243_124367


namespace NUMINAMATH_CALUDE_sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l1243_124395

-- Define the polynomial coefficients
def a₀ : ℝ := sorry
def a₁ : ℝ := sorry
def a₂ : ℝ := sorry
def a₃ : ℝ := sorry
def a₄ : ℝ := sorry

-- Define the polynomial equation
axiom polynomial_eq : ∀ x : ℝ, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4

-- Define S
def S : ℕ := (Finset.range 27).sum (fun k => Nat.choose 27 (k + 1))

-- Theorem statements
theorem sum_of_all_coeff : a₀ + a₁ + a₂ + a₃ + a₄ = 16 := by sorry

theorem sum_of_even_coeff : a₀ + a₂ + a₄ = 136 := by sorry

theorem sum_of_coeff_except_a₀ : a₁ + a₂ + a₃ + a₄ = 15 := by sorry

theorem S_mod_9 : S % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_all_coeff_sum_of_even_coeff_sum_of_coeff_except_a₀_S_mod_9_l1243_124395


namespace NUMINAMATH_CALUDE_particle_in_semicircle_probability_l1243_124315

theorem particle_in_semicircle_probability (AB BC : Real) (h1 : AB = 2) (h2 : BC = 1) :
  let rectangle_area := AB * BC
  let semicircle_radius := AB / 2
  let semicircle_area := π * semicircle_radius^2 / 2
  semicircle_area / rectangle_area = π / 4 := by sorry

end NUMINAMATH_CALUDE_particle_in_semicircle_probability_l1243_124315


namespace NUMINAMATH_CALUDE_new_person_weight_l1243_124316

theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 8 →
  replaced_weight = 35 →
  avg_increase = 2.5 →
  (initial_count * avg_increase + replaced_weight : Real) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1243_124316


namespace NUMINAMATH_CALUDE_bounded_recurrence_periodic_l1243_124300

def is_bounded (x : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n, |x n| ≤ M

def recurrence_relation (x : ℕ → ℤ) : Prop :=
  ∀ n, x (n + 5) = (5 * (x (n + 4))^3 + x (n + 3) - 3 * x (n + 2) + x n) /
                   (2 * x (n + 2) + (x (n + 1))^2 + x (n + 1) * x n)

def eventually_periodic (x : ℕ → ℤ) : Prop :=
  ∃ k p : ℕ, p > 0 ∧ ∀ n ≥ k, x (n + p) = x n

theorem bounded_recurrence_periodic
  (x : ℕ → ℤ) (h_bounded : is_bounded x) (h_recurrence : recurrence_relation x) :
  eventually_periodic x :=
sorry

end NUMINAMATH_CALUDE_bounded_recurrence_periodic_l1243_124300


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1243_124357

theorem unique_solution_cube_difference_square :
  ∀ x y z : ℕ+,
  y.val.Prime → 
  (¬ 3 ∣ z.val) → 
  (¬ y.val ∣ z.val) → 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_square_l1243_124357


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l1243_124317

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    Y = 4 * F →
    F = 2 * F →
    Z = 7 * G →
    G = 7 * G →
    X = Y * Z →
    X = 392 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l1243_124317


namespace NUMINAMATH_CALUDE_school_student_count_l1243_124378

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 24

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 5

/-- The total number of students in the school -/
def total_students : ℕ := num_classrooms * students_per_classroom

theorem school_student_count : total_students = 120 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l1243_124378


namespace NUMINAMATH_CALUDE_correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l1243_124398

theorem correct_sqrt_product : ∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b) := by sorry

theorem incorrect_sqrt_sum : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b) := by sorry

theorem incorrect_sqrt_diff : ∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3 := by sorry

theorem incorrect_sqrt_div : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2 := by sorry

theorem only_sqrt_product_correct :
  (∀ a b : ℝ, a > 0 → b > 0 → (Real.sqrt a) * (Real.sqrt b) = Real.sqrt (a * b)) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) + (Real.sqrt b) ≠ Real.sqrt (a + b)) ∧
  (∃ a : ℝ, a > 0 ∧ 3 * (Real.sqrt a) - (Real.sqrt a) ≠ 3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (Real.sqrt a) / (Real.sqrt b) ≠ 2) := by sorry

end NUMINAMATH_CALUDE_correct_sqrt_product_incorrect_sqrt_sum_incorrect_sqrt_diff_incorrect_sqrt_div_only_sqrt_product_correct_l1243_124398


namespace NUMINAMATH_CALUDE_triangle_area_l1243_124344

/-- Given a triangle ABC with side lengths AB = 1 and BC = 3, and the dot product of vectors AB and BC equal to -1, 
    prove that the area of the triangle is √2. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let AB := ((B.1 - A.1), (B.2 - A.2))
  let BC := ((C.1 - B.1), (C.2 - B.2))
  (AB.1^2 + AB.2^2 = 1) →
  (BC.1^2 + BC.2^2 = 9) →
  (AB.1 * BC.1 + AB.2 * BC.2 = -1) →
  (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_l1243_124344


namespace NUMINAMATH_CALUDE_m_intersect_n_eq_m_l1243_124372

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem m_intersect_n_eq_m : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_eq_m_l1243_124372


namespace NUMINAMATH_CALUDE_probability_alice_has_ball_after_three_turns_l1243_124312

-- Define the probabilities
def alice_pass : ℚ := 1/3
def alice_keep : ℚ := 2/3
def bob_pass : ℚ := 1/4
def bob_keep : ℚ := 3/4

-- Define the game state after three turns
def alice_has_ball_after_three_turns : ℚ :=
  alice_keep^3 + alice_pass * bob_pass * alice_keep + alice_keep * alice_pass * bob_pass

-- Theorem statement
theorem probability_alice_has_ball_after_three_turns :
  alice_has_ball_after_three_turns = 11/27 := by sorry

end NUMINAMATH_CALUDE_probability_alice_has_ball_after_three_turns_l1243_124312
