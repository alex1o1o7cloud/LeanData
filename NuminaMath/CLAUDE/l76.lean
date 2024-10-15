import Mathlib

namespace NUMINAMATH_CALUDE_triangle_tan_c_l76_7697

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S satisfies 2S = (a + b)² - c², then tan C = -4/3 -/
theorem triangle_tan_c (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1 / 2) * a * b * Real.sin C
  2 * S = (a + b)^2 - c^2 →
  Real.tan C = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tan_c_l76_7697


namespace NUMINAMATH_CALUDE_diamonds_10th_pattern_l76_7677

/-- The number of diamonds in the n-th pattern of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 4
  else diamonds (n - 1) + 4 * (2 * n - 1)

/-- The theorem stating that the 10th pattern has 400 diamonds -/
theorem diamonds_10th_pattern : diamonds 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_10th_pattern_l76_7677


namespace NUMINAMATH_CALUDE_intersection_sum_is_eight_l76_7657

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 2)^2
def parabola2 (x y : ℝ) : Prop := x + 5 = (y - 4)^2

-- Define the set of intersection points
def intersectionPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem intersection_sum_is_eight :
  ∃ (points : Finset (ℝ × ℝ)), points.toSet = intersectionPoints ∧
  points.card = 4 ∧
  (points.sum (λ p => p.1) + points.sum (λ p => p.2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_is_eight_l76_7657


namespace NUMINAMATH_CALUDE_relationship_proof_l76_7699

theorem relationship_proof (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end NUMINAMATH_CALUDE_relationship_proof_l76_7699


namespace NUMINAMATH_CALUDE_builder_problem_l76_7696

/-- Calculate the minimum number of packs needed given the total items and items per pack -/
def minPacks (total : ℕ) (perPack : ℕ) : ℕ :=
  (total + perPack - 1) / perPack

/-- The problem statement -/
theorem builder_problem :
  let totalBrackets := 42
  let bracketsPerPack := 5
  minPacks totalBrackets bracketsPerPack = 9 := by
  sorry

end NUMINAMATH_CALUDE_builder_problem_l76_7696


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l76_7617

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (5 : ℚ) / 8 → q' ≥ q) →
  q - p = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l76_7617


namespace NUMINAMATH_CALUDE_appended_number_divisible_by_seven_l76_7627

theorem appended_number_divisible_by_seven (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) 
  (hb : 100 ≤ b ∧ b < 1000) (h_rem : a % 7 = b % 7) :
  ∃ k : ℕ, 1000 * a + b = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_appended_number_divisible_by_seven_l76_7627


namespace NUMINAMATH_CALUDE_paul_weed_eating_earnings_l76_7646

/-- The amount of money Paul made mowing lawns -/
def money_mowing : ℕ := 68

/-- The number of weeks Paul's money would last -/
def weeks : ℕ := 9

/-- The amount Paul would spend per week -/
def spend_per_week : ℕ := 9

/-- The total amount of money Paul had -/
def total_money : ℕ := weeks * spend_per_week

/-- The amount of money Paul made weed eating -/
def money_weed_eating : ℕ := total_money - money_mowing

theorem paul_weed_eating_earnings : money_weed_eating = 13 := by
  sorry

end NUMINAMATH_CALUDE_paul_weed_eating_earnings_l76_7646


namespace NUMINAMATH_CALUDE_work_solution_l76_7601

def work_problem (a b : ℝ) : Prop :=
  b = 15 ∧
  (3 / a + 5 * (1 / a + 1 / b) = 1) →
  a = 12

theorem work_solution : ∃ a b : ℝ, work_problem a b := by
  sorry

end NUMINAMATH_CALUDE_work_solution_l76_7601


namespace NUMINAMATH_CALUDE_incorrect_height_correction_l76_7673

theorem incorrect_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) :
  n = 35 →
  initial_avg = 180 →
  wrong_height = 166 →
  actual_avg = 178 →
  (n * initial_avg - wrong_height + (n * actual_avg - n * initial_avg + wrong_height)) / n = 236 :=
by sorry

end NUMINAMATH_CALUDE_incorrect_height_correction_l76_7673


namespace NUMINAMATH_CALUDE_power_23_2023_mod_29_l76_7643

theorem power_23_2023_mod_29 : 23^2023 % 29 = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_23_2023_mod_29_l76_7643


namespace NUMINAMATH_CALUDE_existence_of_indices_with_inequalities_l76_7690

theorem existence_of_indices_with_inequalities 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_existence_of_indices_with_inequalities_l76_7690


namespace NUMINAMATH_CALUDE_tv_horizontal_length_l76_7614

/-- Calculates the horizontal length of a TV given its aspect ratio and diagonal length -/
theorem tv_horizontal_length 
  (aspect_width : ℝ) 
  (aspect_height : ℝ) 
  (diagonal_length : ℝ) 
  (aspect_width_pos : 0 < aspect_width)
  (aspect_height_pos : 0 < aspect_height)
  (diagonal_length_pos : 0 < diagonal_length) :
  let horizontal_length := aspect_width * diagonal_length / Real.sqrt (aspect_width^2 + aspect_height^2)
  horizontal_length = 16 * diagonal_length / Real.sqrt 337 :=
by sorry

end NUMINAMATH_CALUDE_tv_horizontal_length_l76_7614


namespace NUMINAMATH_CALUDE_sequence_is_decreasing_l76_7616

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem sequence_is_decreasing (a : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, a n - a (n + 1) = 10) : 
  is_decreasing a :=
sorry

end NUMINAMATH_CALUDE_sequence_is_decreasing_l76_7616


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l76_7652

/-- The y-coordinate of the point on the y-axis that is equidistant from A(3, 0) and B(-4, 5) -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, (y = 16/5) ∧ 
  ((0 - 3)^2 + (y - 0)^2 = (0 - (-4))^2 + (y - 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l76_7652


namespace NUMINAMATH_CALUDE_solution_equality_l76_7653

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x - 1| + |x + 2| < 5}

-- State the theorem
theorem solution_equality : solution_set = Set.Ioo (-3) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l76_7653


namespace NUMINAMATH_CALUDE_sum_of_fractions_l76_7647

theorem sum_of_fractions : 
  (2 : ℚ) / 10 + 4 / 10 + 6 / 10 + 8 / 10 + 10 / 10 + 12 / 10 + 14 / 10 + 16 / 10 + 18 / 10 + 20 / 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l76_7647


namespace NUMINAMATH_CALUDE_alternating_arrangement_count_l76_7612

/-- The number of ways to arrange n elements from a set of m elements. -/
def A (m n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 4 boys and 4 girls in a row,
    such that no two girls are adjacent and no two boys are adjacent. -/
def alternating_arrangement : ℕ := sorry

/-- Theorem stating that the number of alternating arrangements
    of 4 boys and 4 girls is equal to 2A₄⁴A₄⁴. -/
theorem alternating_arrangement_count :
  alternating_arrangement = 2 * A 4 4 * A 4 4 := by sorry

end NUMINAMATH_CALUDE_alternating_arrangement_count_l76_7612


namespace NUMINAMATH_CALUDE_rectangle_shading_l76_7619

theorem rectangle_shading (length width : ℕ) (initial_shaded_fraction final_shaded_fraction : ℚ) :
  length = 15 →
  width = 20 →
  initial_shaded_fraction = 1 / 4 →
  final_shaded_fraction = 1 / 5 →
  (initial_shaded_fraction * final_shaded_fraction : ℚ) = 1 / 20 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_shading_l76_7619


namespace NUMINAMATH_CALUDE_count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l76_7622

/-- A function that returns the count of four-digit numbers with all even digits in descending order -/
def count_descending_even_digits : ℕ :=
  5

/-- A function that returns the count of four-digit numbers with all even digits in ascending order -/
def count_ascending_even_digits : ℕ :=
  1

/-- Theorem stating the count of four-digit numbers with all even digits in descending order -/
theorem count_descending_even_digits_is_five :
  count_descending_even_digits = 5 := by sorry

/-- Theorem stating the count of four-digit numbers with all even digits in ascending order -/
theorem count_ascending_even_digits_is_one :
  count_ascending_even_digits = 1 := by sorry

end NUMINAMATH_CALUDE_count_descending_even_digits_is_five_count_ascending_even_digits_is_one_l76_7622


namespace NUMINAMATH_CALUDE_infinite_factorial_solutions_l76_7611

theorem infinite_factorial_solutions :
  ∃ f : ℕ → ℕ × ℕ × ℕ, ∀ n : ℕ,
    let (x, y, z) := f n
    x > 1 ∧ y > 1 ∧ z > 1 ∧ Nat.factorial x * Nat.factorial y = Nat.factorial z :=
by sorry

end NUMINAMATH_CALUDE_infinite_factorial_solutions_l76_7611


namespace NUMINAMATH_CALUDE_valid_coloring_exists_l76_7693

/-- Represents a 9x9 grid where each cell can be colored or uncolored -/
def Grid := Fin 9 → Fin 9 → Bool

/-- Check if two cells are adjacent (by side or corner) -/
def adjacent (x1 y1 x2 y2 : Fin 9) : Bool :=
  (x1 = x2 ∧ y1.val = y2.val + 1) ∨
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1 = y2) ∨
  (x1.val + 1 = x2.val ∧ y1 = y2) ∨
  (x1.val = x2.val + 1 ∧ y1.val = y2.val + 1) ∨
  (x1.val + 1 = x2.val ∧ y1.val + 1 = y2.val) ∨
  (x1.val = x2.val + 1 ∧ y1.val + 1 = y2.val) ∨
  (x1.val + 1 = x2.val ∧ y1.val = y2.val + 1)

/-- Check if a grid coloring is valid -/
def valid_coloring (g : Grid) : Prop :=
  -- Center is not colored
  ¬g 4 4 ∧
  -- No adjacent cells are colored
  (∀ x1 y1 x2 y2, adjacent x1 y1 x2 y2 → ¬(g x1 y1 ∧ g x2 y2)) ∧
  -- Any ray from center intersects a colored cell
  (∀ dx dy, dx ≠ 0 ∨ dy ≠ 0 →
    ∃ t : ℚ, t > 0 ∧ g ⌊4 + t * dx⌋ ⌊4 + t * dy⌋)

/-- Theorem: There exists a valid coloring of the 9x9 grid -/
theorem valid_coloring_exists : ∃ g : Grid, valid_coloring g :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_exists_l76_7693


namespace NUMINAMATH_CALUDE_inverse_proportional_symmetry_axis_l76_7621

theorem inverse_proportional_symmetry_axis (k : ℝ) (h1 : k ≠ 0) (h2 : k ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧
  (∀ x ≠ 0, ∃ y, f y = f x ∧ (y + x) * (-k / |k|) = y - x) :=
sorry

end NUMINAMATH_CALUDE_inverse_proportional_symmetry_axis_l76_7621


namespace NUMINAMATH_CALUDE_exponent_division_thirteen_eleven_div_thirteen_four_l76_7667

theorem exponent_division (a : ℕ) (m n : ℕ) (h : a > 0) : a^m / a^n = a^(m - n) := by sorry

theorem thirteen_eleven_div_thirteen_four :
  (13 : ℕ)^11 / (13 : ℕ)^4 = (13 : ℕ)^7 := by sorry

end NUMINAMATH_CALUDE_exponent_division_thirteen_eleven_div_thirteen_four_l76_7667


namespace NUMINAMATH_CALUDE_equation_solution_l76_7650

theorem equation_solution : 
  ∃ y : ℝ, (4 : ℝ) * 8^3 = 4^y ∧ y = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l76_7650


namespace NUMINAMATH_CALUDE_single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l76_7620

-- Define the basic events
def hits_9_rings : Prop := sorry
def hits_8_rings : Prop := sorry
def A_hits_10_rings : Prop := sorry
def B_hits_8_rings : Prop := sorry
def A_hits_target : Prop := sorry
def B_hits_target : Prop := sorry

-- Define compound events
def both_hit_target : Prop := A_hits_target ∧ B_hits_target
def neither_hit_target : Prop := ¬A_hits_target ∧ ¬B_hits_target
def at_least_one_hits : Prop := A_hits_target ∨ B_hits_target
def A_misses_B_hits : Prop := ¬A_hits_target ∧ B_hits_target

-- Define mutual exclusivity
def mutually_exclusive (p q : Prop) : Prop := ¬(p ∧ q)

-- Theorem statements
theorem single_shot_exclusivity : 
  mutually_exclusive hits_9_rings hits_8_rings := by sorry

theorem two_shooters_not_exclusive : 
  ¬(mutually_exclusive A_hits_10_rings B_hits_8_rings) := by sorry

theorem hit_or_miss_exclusivity : 
  mutually_exclusive both_hit_target neither_hit_target := by sorry

theorem at_least_one_not_exclusive : 
  ¬(mutually_exclusive at_least_one_hits A_misses_B_hits) := by sorry

end NUMINAMATH_CALUDE_single_shot_exclusivity_two_shooters_not_exclusive_hit_or_miss_exclusivity_at_least_one_not_exclusive_l76_7620


namespace NUMINAMATH_CALUDE_quartic_roots_sum_product_l76_7666

theorem quartic_roots_sum_product (p q : ℝ) : 
  (p^4 - 6*p - 1 = 0) → 
  (q^4 - 6*q - 1 = 0) → 
  (p ≠ q) →
  (∀ x : ℝ, x^4 - 6*x - 1 = 0 → x = p ∨ x = q) →
  p*q + p + q = 1 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_product_l76_7666


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l76_7661

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l76_7661


namespace NUMINAMATH_CALUDE_room_length_calculation_l76_7637

/-- The length of a rectangular room given its width, paving cost, and paving rate. -/
theorem room_length_calculation (width : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 ∧ paving_cost = 34200 ∧ paving_rate = 900 →
  paving_cost / paving_rate / width = 8 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l76_7637


namespace NUMINAMATH_CALUDE_power_division_simplification_quadratic_expression_simplification_l76_7604

-- Problem 1
theorem power_division_simplification :
  10^7 / (10^3 / 10^2) = 10^6 := by sorry

-- Problem 2
theorem quadratic_expression_simplification (x : ℝ) :
  (x + 2)^2 - (x + 1) * (x - 1) = 4 * x + 5 := by sorry

end NUMINAMATH_CALUDE_power_division_simplification_quadratic_expression_simplification_l76_7604


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l76_7613

theorem cube_root_equation_solution :
  ∃! x : ℝ, (10 - 2*x)^(1/3 : ℝ) = -2 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l76_7613


namespace NUMINAMATH_CALUDE_money_distribution_l76_7638

/-- Given three people A, B, and C with money, prove that B and C together have 360 Rs. -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  a + c = 200 →      -- Money A and C have together
  c = 60 →           -- Money C has
  b + c = 360        -- Money B and C have together
  := by sorry

end NUMINAMATH_CALUDE_money_distribution_l76_7638


namespace NUMINAMATH_CALUDE_truck_travel_distance_l76_7605

/-- 
Given a truck that travels:
- x miles north
- 30 miles east
- x miles north again
And ends up 50 miles from the starting point,
prove that x must equal 20.
-/
theorem truck_travel_distance (x : ℝ) : 
  (2 * x)^2 + 30^2 = 50^2 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l76_7605


namespace NUMINAMATH_CALUDE_question_paper_combinations_l76_7656

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem question_paper_combinations : choose 10 8 * choose 10 5 = 11340 := by
  sorry

end NUMINAMATH_CALUDE_question_paper_combinations_l76_7656


namespace NUMINAMATH_CALUDE_davids_remaining_money_is_19_90_l76_7651

/-- Calculates David's remaining money after expenses and taxes -/
def davidsRemainingMoney (rate1 rate2 rate3 : ℝ) (hours : ℝ) (shoePrice : ℝ) 
  (shoeDiscount taxRate giftFraction : ℝ) : ℝ :=
  let totalEarnings := (rate1 + rate2 + rate3) * hours
  let taxAmount := totalEarnings * taxRate
  let discountedShoePrice := shoePrice * (1 - shoeDiscount)
  let remainingAfterShoes := totalEarnings - taxAmount - discountedShoePrice
  remainingAfterShoes * (1 - giftFraction)

/-- Theorem stating that David's remaining money is $19.90 -/
theorem davids_remaining_money_is_19_90 :
  davidsRemainingMoney 14 18 20 2 75 0.15 0.1 (1/3) = 19.90 := by
  sorry

end NUMINAMATH_CALUDE_davids_remaining_money_is_19_90_l76_7651


namespace NUMINAMATH_CALUDE_min_value_greater_than_five_l76_7684

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 1)^2 + |x + a - 1|

/-- The theorem statement -/
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f a x > 5) ↔ a < (1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_min_value_greater_than_five_l76_7684


namespace NUMINAMATH_CALUDE_plane_equation_l76_7609

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space represented by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space represented by Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

def line_on_plane (l : Line3D) (plane : Plane) : Prop :=
  ∀ t : ℝ, point_on_plane ⟨l.x t, l.y t, l.z t⟩ plane

def is_solution (plane : Plane) : Prop :=
  let p1 : Point3D := ⟨1, 4, -3⟩
  let p2 : Point3D := ⟨0, -3, 0⟩
  let l : Line3D := { x := λ t => 4 * t + 2, y := λ t => -t - 2, z := λ t => 5 * t + 1 }
  point_on_plane p1 plane ∧
  point_on_plane p2 plane ∧
  line_on_plane l plane ∧
  plane.A > 0 ∧
  Nat.gcd (Int.natAbs plane.A) (Nat.gcd (Int.natAbs plane.B) (Nat.gcd (Int.natAbs plane.C) (Int.natAbs plane.D))) = 1

theorem plane_equation : is_solution ⟨10, 9, -13, 27⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l76_7609


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l76_7669

theorem quadratic_two_zeros (b : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + b*x₁ - 3 = 0) ∧ 
    (x₂^2 + b*x₂ - 3 = 0) ∧ 
    (∀ x : ℝ, x^2 + b*x - 3 = 0 → (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l76_7669


namespace NUMINAMATH_CALUDE_product_multiple_of_three_probability_l76_7685

/-- The probability of rolling a multiple of 3 on a standard die -/
def prob_multiple_of_three : ℚ := 1/3

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability that the product of all rolls is a multiple of 3 -/
def prob_product_multiple_of_three : ℚ :=
  1 - (1 - prob_multiple_of_three) ^ num_rolls

theorem product_multiple_of_three_probability :
  prob_product_multiple_of_three = 6305/6561 := by
  sorry

end NUMINAMATH_CALUDE_product_multiple_of_three_probability_l76_7685


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l76_7692

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The theorem statement -/
theorem coin_flip_probability_difference : 
  |prob_k_heads 6 4 - prob_k_heads 6 6| = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l76_7692


namespace NUMINAMATH_CALUDE_five_ounce_letter_cost_l76_7641

/-- Postage fee structure -/
structure PostageFee where
  baseRate : ℚ  -- Base rate in dollars
  additionalRate : ℚ  -- Additional rate per ounce in dollars
  handlingFee : ℚ  -- Handling fee in dollars
  handlingFeeThreshold : ℕ  -- Threshold in ounces for applying handling fee

/-- Calculate the total postage fee for a given weight -/
def calculatePostageFee (fee : PostageFee) (weight : ℕ) : ℚ :=
  fee.baseRate +
  fee.additionalRate * (weight - 1) +
  if weight > fee.handlingFeeThreshold then fee.handlingFee else 0

/-- Theorem: The cost to send a 5-ounce letter is $1.45 -/
theorem five_ounce_letter_cost :
  let fee : PostageFee := {
    baseRate := 35 / 100,
    additionalRate := 25 / 100,
    handlingFee := 10 / 100,
    handlingFeeThreshold := 2
  }
  calculatePostageFee fee 5 = 145 / 100 := by
  sorry

end NUMINAMATH_CALUDE_five_ounce_letter_cost_l76_7641


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l76_7658

/-- Represents the base length of an isosceles triangle -/
def BaseLengthIsosceles (area : ℝ) (equalSide : ℝ) : Set ℝ :=
  {x | x > 0 ∧ (x * (equalSide ^ 2 - (x / 2) ^ 2).sqrt / 2 = area)}

/-- Theorem: The base length of an isosceles triangle with area 3 cm² and equal side 25 cm is either 14 cm or 48 cm -/
theorem isosceles_triangle_base_length :
  BaseLengthIsosceles 3 25 = {14, 48} := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l76_7658


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l76_7694

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (-1, x) (-2, 4) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l76_7694


namespace NUMINAMATH_CALUDE_equation_describes_cone_l76_7633

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the equation z = kr
def coneEquation (k : ℝ) (p : CylindricalCoord) : Prop :=
  p.z = k * p.r

-- Define a cone in cylindrical coordinates
def isCone (S : Set CylindricalCoord) : Prop :=
  ∃ k : ℝ, ∀ p ∈ S, coneEquation k p

-- Theorem statement
theorem equation_describes_cone (k : ℝ) :
  isCone { p : CylindricalCoord | coneEquation k p } :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l76_7633


namespace NUMINAMATH_CALUDE_cooper_fence_length_l76_7628

/-- The length of each wall in Cooper's fence --/
def wall_length : ℕ := 20

/-- The number of walls in Cooper's fence --/
def num_walls : ℕ := 4

/-- The height of each wall in bricks --/
def wall_height : ℕ := 5

/-- The depth of each wall in bricks --/
def wall_depth : ℕ := 2

/-- The total number of bricks needed for the fence --/
def total_bricks : ℕ := 800

theorem cooper_fence_length :
  wall_length * num_walls * wall_height * wall_depth = total_bricks :=
by sorry

end NUMINAMATH_CALUDE_cooper_fence_length_l76_7628


namespace NUMINAMATH_CALUDE_cosine_power_expansion_sum_of_squares_l76_7672

open Real

theorem cosine_power_expansion_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
    (∀ θ : ℝ, (cos θ)^7 = b₁ * cos θ + b₂ * cos (2*θ) + b₃ * cos (3*θ) + 
                          b₄ * cos (4*θ) + b₅ * cos (5*θ) + b₆ * cos (6*θ) + 
                          b₇ * cos (7*θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_expansion_sum_of_squares_l76_7672


namespace NUMINAMATH_CALUDE_valid_placement_exists_l76_7645

/-- Represents the configuration of the circles --/
inductive Position
| TopLeft | TopMiddle | TopRight
| MiddleLeft | MiddleRight
| BottomLeft | BottomMiddle | BottomRight
| Center

/-- A function type that maps positions to numbers --/
def Placement := Position → Fin 9

/-- Checks if two numbers are adjacent in the configuration --/
def are_adjacent (p1 p2 : Position) : Bool :=
  match p1, p2 with
  | Position.TopLeft, Position.TopMiddle => true
  | Position.TopLeft, Position.MiddleLeft => true
  | Position.TopMiddle, Position.TopRight => true
  | Position.TopMiddle, Position.Center => true
  | Position.TopRight, Position.MiddleRight => true
  | Position.MiddleLeft, Position.BottomLeft => true
  | Position.MiddleLeft, Position.Center => true
  | Position.MiddleRight, Position.BottomRight => true
  | Position.MiddleRight, Position.Center => true
  | Position.BottomLeft, Position.BottomMiddle => true
  | Position.BottomMiddle, Position.BottomRight => true
  | Position.BottomMiddle, Position.Center => true
  | _, _ => false

/-- The main theorem stating the existence of a valid placement --/
theorem valid_placement_exists : ∃ (p : Placement),
  (∀ pos1 pos2, pos1 ≠ pos2 → p pos1 ≠ p pos2) ∧
  (∀ pos1 pos2, are_adjacent pos1 pos2 → Nat.gcd (p pos1).val.succ (p pos2).val.succ = 1) :=
sorry

end NUMINAMATH_CALUDE_valid_placement_exists_l76_7645


namespace NUMINAMATH_CALUDE_smallest_N_l76_7631

/-- Represents a point in the square array -/
structure Point where
  row : Fin 4
  col : Nat

/-- The first numbering scheme (left to right, top to bottom) -/
def x (p : Point) (N : Nat) : Nat :=
  p.row.val * N + p.col

/-- The second numbering scheme (top to bottom, left to right) -/
def y (p : Point) : Nat :=
  (p.col - 1) * 4 + p.row.val + 1

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N : ∃ (N : Nat) (p₁ p₂ p₃ p₄ : Point),
  N > 0 ∧
  p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧
  p₁.col > 0 ∧ p₂.col > 0 ∧ p₃.col > 0 ∧ p₄.col > 0 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧
  x p₁ N = y p₃ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₂ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ : Point),
    M > 0 ∧
    q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧
    q₁.col > 0 ∧ q₂.col > 0 ∧ q₃.col > 0 ∧ q₄.col > 0 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧
    x q₁ M = y q₃ ∧
    x q₂ M = y q₁ ∧
    x q₃ M = y q₄ ∧
    x q₄ M = y q₂ →
    N ≤ M) ∧
  N = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_l76_7631


namespace NUMINAMATH_CALUDE_circle_areas_in_right_triangle_l76_7683

theorem circle_areas_in_right_triangle (a b c : Real) (r : Real) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ r = 1 →
  a^2 + b^2 = c^2 →
  let α := Real.arctan (a / b)
  let β := Real.arctan (b / a)
  let γ := π / 2
  (α + β + γ = π) →
  (α / 2 + β / 2 + γ / 2) * r^2 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_areas_in_right_triangle_l76_7683


namespace NUMINAMATH_CALUDE_pizza_toppings_l76_7679

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : pepperoni_slices = 8)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ≤ total_slices → (slice ≤ pepperoni_slices ∨ slice ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧
    both_toppings = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l76_7679


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l76_7680

theorem smallest_solution_congruence (x : ℕ) :
  (x > 0 ∧ 5 * x ≡ 17 [MOD 31]) ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l76_7680


namespace NUMINAMATH_CALUDE_equivalent_proposition_l76_7689

/-- Represents the quality of goods -/
inductive Quality
| High
| NotHigh

/-- Represents the price of goods -/
inductive Price
| Cheap
| NotCheap

/-- Translates a Chinese phrase to its logical meaning -/
def translate : String → (Quality → Price → Prop)
| "好货不便宜" => λ q p => q = Quality.High → p ≠ Price.Cheap
| "便宜没好货" => λ q p => p = Price.Cheap → q ≠ Quality.High
| _ => λ _ _ => False

theorem equivalent_proposition : 
  ∀ (q : Quality) (p : Price), 
    (translate "好货不便宜" q p) ↔ (translate "便宜没好货" q p) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_proposition_l76_7689


namespace NUMINAMATH_CALUDE_jill_makes_30_trips_l76_7678

/-- Represents the water-carrying problem with Jack and Jill --/
structure WaterProblem where
  tank_capacity : ℕ
  bucket_capacity : ℕ
  jack_buckets_per_trip : ℕ
  jill_buckets_per_trip : ℕ
  jack_trips_ratio : ℕ
  jill_trips_ratio : ℕ

/-- Calculates the number of trips Jill makes to fill the tank --/
def jill_trips (wp : WaterProblem) : ℕ :=
  let jack_water_per_trip := wp.jack_buckets_per_trip * wp.bucket_capacity
  let jill_water_per_trip := wp.jill_buckets_per_trip * wp.bucket_capacity
  let water_per_cycle := jack_water_per_trip * wp.jack_trips_ratio + jill_water_per_trip * wp.jill_trips_ratio
  let cycles := wp.tank_capacity / water_per_cycle
  cycles * wp.jill_trips_ratio

/-- Theorem stating that Jill makes 30 trips to fill the tank under the given conditions --/
theorem jill_makes_30_trips :
  let wp : WaterProblem := {
    tank_capacity := 600,
    bucket_capacity := 5,
    jack_buckets_per_trip := 2,
    jill_buckets_per_trip := 1,
    jack_trips_ratio := 3,
    jill_trips_ratio := 2
  }
  jill_trips wp = 30 := by
  sorry


end NUMINAMATH_CALUDE_jill_makes_30_trips_l76_7678


namespace NUMINAMATH_CALUDE_linear_equation_solution_l76_7655

theorem linear_equation_solution (k : ℝ) (x : ℝ) :
  k - 2 = 0 →        -- Condition for linearity
  4 * k ≠ 0 →        -- Ensure non-trivial equation
  (k - 2) * x^2 + 4 * k * x - 5 = 0 →
  x = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l76_7655


namespace NUMINAMATH_CALUDE_always_quadratic_radical_l76_7688

theorem always_quadratic_radical (a : ℝ) : 0 ≤ a^2 + 1 := by sorry

end NUMINAMATH_CALUDE_always_quadratic_radical_l76_7688


namespace NUMINAMATH_CALUDE_mod_nine_equivalence_l76_7610

theorem mod_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_mod_nine_equivalence_l76_7610


namespace NUMINAMATH_CALUDE_rachel_earnings_l76_7660

/-- Rachel's earnings as a waitress in one hour -/
theorem rachel_earnings (hourly_wage : ℝ) (people_served : ℕ) (tip_per_person : ℝ) 
  (h1 : hourly_wage = 12)
  (h2 : people_served = 20)
  (h3 : tip_per_person = 1.25) :
  hourly_wage + (people_served : ℝ) * tip_per_person = 37 := by
  sorry

end NUMINAMATH_CALUDE_rachel_earnings_l76_7660


namespace NUMINAMATH_CALUDE_chinese_math_problem_l76_7691

-- Define the system of equations
def equation_system (x y : ℝ) : Prop :=
  5 * x + 2 * y = 19 ∧ 2 * x + 5 * y = 16

-- Define the profit function
def profit_function (m : ℝ) : ℝ := 0.5 * m + 5

-- Theorem statement
theorem chinese_math_problem :
  (∃ (x y : ℝ), equation_system x y ∧ x = 3 ∧ y = 2) ∧
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 5 → profit_function m ≤ profit_function 5) :=
by sorry

end NUMINAMATH_CALUDE_chinese_math_problem_l76_7691


namespace NUMINAMATH_CALUDE_train_speed_proof_l76_7600

/-- Proves that a train crossing a 280-meter platform in 30 seconds and passing a stationary man in 16 seconds has a speed of 72 km/h -/
theorem train_speed_proof (platform_length : Real) (platform_crossing_time : Real) 
  (man_passing_time : Real) (speed_kmh : Real) : 
  platform_length = 280 ∧ 
  platform_crossing_time = 30 ∧ 
  man_passing_time = 16 ∧
  speed_kmh = (platform_length / (platform_crossing_time - man_passing_time)) * 3.6 →
  speed_kmh = 72 := by
sorry

end NUMINAMATH_CALUDE_train_speed_proof_l76_7600


namespace NUMINAMATH_CALUDE_five_teachers_three_classes_l76_7615

/-- The number of ways to assign n teachers to k distinct classes, 
    with at least one teacher per class -/
def teacher_assignments (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 150 ways to assign 5 teachers to 3 classes -/
theorem five_teachers_three_classes : 
  teacher_assignments 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_teachers_three_classes_l76_7615


namespace NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l76_7681

/-- Conversion from yards to feet -/
def yards_to_feet (yards : ℝ) : ℝ := 3 * yards

/-- Conversion from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (cubic_yards : ℝ) : ℝ := 27 * cubic_yards

/-- Theorem: 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_equals_135_cubic_feet :
  cubic_yards_to_cubic_feet 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_equals_135_cubic_feet_l76_7681


namespace NUMINAMATH_CALUDE_common_chord_equation_length_AB_l76_7644

-- Define the circles C and M
def C (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def M (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 2*y - 40 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2, 6)
def B : ℝ × ℝ := (4, -2)

-- Theorem for the equation of the common chord
theorem common_chord_equation : 
  ∀ (x y : ℝ), C x y ∧ M x y → 4*x + 2*y - 10 = 0 :=
sorry

-- Theorem for the length of AB
theorem length_AB : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_length_AB_l76_7644


namespace NUMINAMATH_CALUDE_nested_cubes_properties_l76_7642

/-- Represents a cube with an inscribed sphere, which contains another inscribed cube. -/
structure NestedCubes where
  outer_surface_area : ℝ
  outer_side_length : ℝ
  sphere_diameter : ℝ
  inner_side_length : ℝ

/-- The surface area of a cube given its side length. -/
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

/-- The volume of a cube given its side length. -/
def cube_volume (side_length : ℝ) : ℝ := side_length^3

/-- Theorem stating the properties of the nested cubes structure. -/
theorem nested_cubes_properties (nc : NestedCubes) 
  (h1 : nc.outer_surface_area = 54)
  (h2 : nc.outer_side_length^2 = 54 / 6)
  (h3 : nc.sphere_diameter = nc.outer_side_length)
  (h4 : nc.inner_side_length * Real.sqrt 3 = nc.sphere_diameter) :
  cube_surface_area nc.inner_side_length = 18 ∧ 
  cube_volume nc.inner_side_length = 3 * Real.sqrt 3 := by
  sorry

#check nested_cubes_properties

end NUMINAMATH_CALUDE_nested_cubes_properties_l76_7642


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l76_7665

theorem arithmetic_mean_of_fractions :
  let a := 3 / 8
  let b := 5 / 9
  (a + b) / 2 = 67 / 144 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l76_7665


namespace NUMINAMATH_CALUDE_expression_value_l76_7671

theorem expression_value : (36 + 9)^2 - (9^2 + 36^2) = -1894224 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l76_7671


namespace NUMINAMATH_CALUDE_fifteen_children_pencil_count_l76_7654

/-- Given a number of children and pencils per child, calculates the total number of pencils -/
def total_pencils (num_children : ℕ) (pencils_per_child : ℕ) : ℕ :=
  num_children * pencils_per_child

/-- Proves that 15 children with 2 pencils each have 30 pencils in total -/
theorem fifteen_children_pencil_count :
  total_pencils 15 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_children_pencil_count_l76_7654


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l76_7629

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l76_7629


namespace NUMINAMATH_CALUDE_value_range_equivalence_l76_7625

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem value_range_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ∈ Set.Icc (f a a) (f a 4)) ↔ 
  a ∈ Set.Icc (-2 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_value_range_equivalence_l76_7625


namespace NUMINAMATH_CALUDE_vector_sum_proof_l76_7675

theorem vector_sum_proof :
  let v1 : Fin 3 → ℝ := ![5, -3, 8]
  let v2 : Fin 3 → ℝ := ![-2, 4, 1]
  let v3 : Fin 3 → ℝ := ![3, -6, -9]
  v1 + v2 + v3 = ![6, -5, 0] :=
by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l76_7675


namespace NUMINAMATH_CALUDE_division_and_addition_l76_7602

theorem division_and_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l76_7602


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l76_7662

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 ∧ 
  b^3 - 3*b - 2 = 0 ∧ 
  c^3 - 3*c - 2 = 0 → 
  a^2*(b - c)^2 + b^2*(c - a)^2 + c^2*(a - b)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l76_7662


namespace NUMINAMATH_CALUDE_smallest_c_value_l76_7635

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  c ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l76_7635


namespace NUMINAMATH_CALUDE_grapes_filling_days_l76_7670

/-- The number of days required to fill a certain number of drums of grapes -/
def days_to_fill_grapes (pickers : ℕ) (drums_per_day : ℕ) (total_drums : ℕ) : ℕ :=
  total_drums / drums_per_day

/-- Theorem stating that it takes 77 days to fill 17017 drums of grapes -/
theorem grapes_filling_days :
  days_to_fill_grapes 235 221 17017 = 77 := by
  sorry

end NUMINAMATH_CALUDE_grapes_filling_days_l76_7670


namespace NUMINAMATH_CALUDE_iron_to_steel_ratio_l76_7626

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of an alloy -/
structure Alloy where
  iron : ℕ
  steel : ℕ

/-- Simplifies a ratio by dividing both numerator and denominator by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd, denominator := r.denominator / gcd }

/-- Theorem: The ratio of iron to steel in the alloy is 2:5 -/
theorem iron_to_steel_ratio (alloy : Alloy) (h : alloy = { iron := 14, steel := 35 }) :
  simplifyRatio { numerator := alloy.iron, denominator := alloy.steel } = { numerator := 2, denominator := 5 } := by
  sorry

end NUMINAMATH_CALUDE_iron_to_steel_ratio_l76_7626


namespace NUMINAMATH_CALUDE_only_135_and_144_satisfy_l76_7649

/-- Represents a 3-digit positive integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The decimal representation of abc --/
def decimal_rep (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The product of digits multiplied by their sum --/
def digit_product_sum (n : ThreeDigitInt) : Nat :=
  n.a * n.b * n.c * (n.a + n.b + n.c)

/-- The theorem stating that only 135 and 144 satisfy the equation --/
theorem only_135_and_144_satisfy :
  ∀ n : ThreeDigitInt, decimal_rep n = digit_product_sum n ↔ decimal_rep n = 135 ∨ decimal_rep n = 144 := by
  sorry

end NUMINAMATH_CALUDE_only_135_and_144_satisfy_l76_7649


namespace NUMINAMATH_CALUDE_primality_test_upper_bound_l76_7663

theorem primality_test_upper_bound :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 1100 →
  (∀ p : ℕ, p.Prime ∧ p ≤ 31 → ¬(p ∣ n)) →
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_primality_test_upper_bound_l76_7663


namespace NUMINAMATH_CALUDE_M_mod_500_l76_7636

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 9 ones -/
def T : ℕ → ℕ := sorry

/-- M is the 500th number in the sequence T -/
def M : ℕ := T 500

theorem M_mod_500 : M % 500 = 281 := by sorry

end NUMINAMATH_CALUDE_M_mod_500_l76_7636


namespace NUMINAMATH_CALUDE_local_min_condition_l76_7603

/-- The function f(x) = (x-1)e^x - ax has a local minimum point less than 0 
    if and only if a is in the open interval (-1/e, 0) -/
theorem local_min_condition (a : ℝ) : 
  (∃ x₀ < 0, IsLocalMin (fun x => (x - 1) * Real.exp x - a * x) x₀) ↔ 
  -1 / Real.exp 1 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_local_min_condition_l76_7603


namespace NUMINAMATH_CALUDE_no_cyclic_prime_divisibility_l76_7695

theorem no_cyclic_prime_divisibility : ¬∃ (p : Fin 2007 → ℕ), 
  (∀ i, Nat.Prime (p i)) ∧ 
  (∀ i : Fin 2006, (p i)^2 - 1 ∣ p (i + 1)) ∧
  ((p 2006)^2 - 1 ∣ p 0) := by
  sorry

end NUMINAMATH_CALUDE_no_cyclic_prime_divisibility_l76_7695


namespace NUMINAMATH_CALUDE_back_squat_increase_calculation_l76_7659

/-- Represents the increase in John's back squat in kg -/
def back_squat_increase : ℝ := sorry

/-- John's original back squat weight in kg -/
def original_back_squat : ℝ := 200

/-- The ratio of John's front squat to his back squat -/
def front_squat_ratio : ℝ := 0.8

/-- The ratio of a triple to John's front squat -/
def triple_ratio : ℝ := 0.9

/-- The total weight moved in three triples in kg -/
def total_triple_weight : ℝ := 540

theorem back_squat_increase_calculation :
  3 * (triple_ratio * front_squat_ratio * (original_back_squat + back_squat_increase)) = total_triple_weight ∧
  back_squat_increase = 50 := by sorry

end NUMINAMATH_CALUDE_back_squat_increase_calculation_l76_7659


namespace NUMINAMATH_CALUDE_triangle_properties_l76_7630

/-- Triangle with sides a, b, c opposite to angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about properties of a specific triangle --/
theorem triangle_properties (t : Triangle) 
  (h_acute : t.A > 0 ∧ t.A < π/2 ∧ t.B > 0 ∧ t.B < π/2 ∧ t.C > 0 ∧ t.C < π/2)
  (h_cosine : t.a * Real.cos t.A + t.b * Real.cos t.B = t.c) :
  (t.a = t.b) ∧ 
  (∀ (circumcircle_area : ℝ), circumcircle_area = π → 
    7 < (3 * t.b^2 + t.b + 4 * t.c) / t.a ∧ 
    (3 * t.b^2 + t.b + 4 * t.c) / t.a < 7 * Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l76_7630


namespace NUMINAMATH_CALUDE_sqrt_difference_of_squares_l76_7686

theorem sqrt_difference_of_squares : (Real.sqrt 5 + Real.sqrt 3) * (Real.sqrt 5 - Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_of_squares_l76_7686


namespace NUMINAMATH_CALUDE_quadratic_inequality_l76_7640

theorem quadratic_inequality (x : ℝ) : x^2 - 6*x > 15 ↔ x < -1.5 ∨ x > 7.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l76_7640


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l76_7698

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) : 
  Complex.abs (4 * i / (1 - i)) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l76_7698


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l76_7608

theorem earliest_meeting_time (david_lap_time maria_lap_time leo_lap_time : ℕ) 
  (h1 : david_lap_time = 5)
  (h2 : maria_lap_time = 8)
  (h3 : leo_lap_time = 10) :
  Nat.lcm (Nat.lcm david_lap_time maria_lap_time) leo_lap_time = 40 := by
sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l76_7608


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l76_7632

theorem infinitely_many_solutions (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l76_7632


namespace NUMINAMATH_CALUDE_rectangle_length_fraction_l76_7634

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 200)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_fraction_l76_7634


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l76_7624

theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2 : ℚ) / Nat.choose total_marbles 4 = 1848 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l76_7624


namespace NUMINAMATH_CALUDE_problem_statement_l76_7607

/-- The problem statement --/
theorem problem_statement (x₀ y₀ r : ℝ) : 
  -- P(x₀, y₀) lies on both curves
  y₀ = 2 * Real.log x₀ ∧ 
  (x₀ - 3)^2 + y₀^2 = r^2 ∧ 
  -- Tangent lines are identical
  (2 / x₀ = -x₀ / y₀) ∧ 
  (2 / x₀ = x₀ * (y₀ - 2) / (9 - 3*x₀ - r^2)) ∧
  -- Quadratic function passes through (0,0), P(x₀, y₀), and (3,0)
  ∃ (a b c : ℝ), ∀ x, 
    (a*x^2 + b*x + c = 0) ∧
    (a*x₀^2 + b*x₀ + c = y₀) ∧
    (9*a + 3*b + c = 0) →
  -- The maximum value of the quadratic function is 9/8
  ∃ (f : ℝ → ℝ), (∀ x, f x ≤ 9/8) ∧ (∃ x, f x = 9/8) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l76_7607


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l76_7606

theorem pure_imaginary_condition (m : ℝ) : 
  (((2 : ℂ) - m * Complex.I) / (1 + Complex.I)).re = 0 → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l76_7606


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l76_7668

theorem modular_inverse_28_mod_29 : ∃ x : ℕ, x ≤ 28 ∧ (28 * x) % 29 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l76_7668


namespace NUMINAMATH_CALUDE_circle_circumference_from_area_l76_7639

theorem circle_circumference_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 225 * π → 2 * π * r = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_from_area_l76_7639


namespace NUMINAMATH_CALUDE_fred_final_cards_l76_7674

-- Define the initial number of cards Fred has
def initial_cards : ℕ := 5

-- Define the number of cards Fred gives to Melanie
def cards_to_melanie : ℕ := 2

-- Define the number of cards Fred trades with Sam
def cards_traded_with_sam : ℕ := 1

-- Define the number of cards Sam gives to Fred
def cards_from_sam : ℕ := 4

-- Define the number of cards Lisa has
def lisa_cards : ℕ := 3

-- Theorem to prove
theorem fred_final_cards : 
  initial_cards - cards_to_melanie - cards_traded_with_sam + cards_from_sam + 2 * lisa_cards = 12 :=
by sorry

end NUMINAMATH_CALUDE_fred_final_cards_l76_7674


namespace NUMINAMATH_CALUDE_unique_three_digit_factorion_l76_7648

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumOfDigitFactorials (n : ℕ) : ℕ :=
  (n.digits 10).map factorial |>.sum

def isFactorion (n : ℕ) : Prop :=
  n = sumOfDigitFactorials n

theorem unique_three_digit_factorion :
  ∀ n : ℕ, 100 ≤ n → n < 1000 → isFactorion n → n = 145 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorion_l76_7648


namespace NUMINAMATH_CALUDE_sum_outside_layers_l76_7676

/-- Represents a 3D cube with specific properties -/
structure Cube3D where
  size : Nat
  total_units : Nat
  sum_per_line : ℝ
  special_value : ℝ

/-- Theorem stating the sum of numbers outside three layers in a specific cube -/
theorem sum_outside_layers (c : Cube3D) 
  (h_size : c.size = 20)
  (h_units : c.total_units = 8000)
  (h_sum : c.sum_per_line = 1)
  (h_special : c.special_value = 10) :
  let total_sum := c.size * c.size * c.sum_per_line
  let layer_sum := 3 * c.sum_per_line - 2 * c.sum_per_line + c.special_value
  total_sum - layer_sum = 392 := by
  sorry

end NUMINAMATH_CALUDE_sum_outside_layers_l76_7676


namespace NUMINAMATH_CALUDE_circle_m_range_l76_7618

/-- A circle equation with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 2*x + m = 0

/-- Condition for the equation to represent a circle -/
def is_circle (m : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The range of m for which the equation represents a circle -/
theorem circle_m_range :
  ∀ m : ℝ, is_circle m ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l76_7618


namespace NUMINAMATH_CALUDE_jill_bouncy_balls_difference_l76_7623

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := balls_per_pack * red_packs

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := balls_per_pack * yellow_packs

/-- The difference between the number of red and yellow bouncy balls -/
def difference : ℕ := total_red_balls - total_yellow_balls

theorem jill_bouncy_balls_difference :
  difference = 18 := by sorry

end NUMINAMATH_CALUDE_jill_bouncy_balls_difference_l76_7623


namespace NUMINAMATH_CALUDE_intersection_of_intervals_l76_7682

open Set

-- Define the sets A and B
def A : Set ℝ := Ioo (-1) 2
def B : Set ℝ := Ioi 0

-- State the theorem
theorem intersection_of_intervals : A ∩ B = Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_intervals_l76_7682


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l76_7664

theorem polynomial_root_sum (p q r s : ℝ) : 
  let g : ℂ → ℂ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (g (-3*I) = 0 ∧ g (1 + I) = 0) → p + q + r + s = 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l76_7664


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l76_7687

theorem polynomial_product_equality (x a : ℝ) : (x - a) * (x^2 + a*x + a^2) = x^3 - a^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l76_7687
