import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_simplification_l2776_277628

theorem polynomial_simplification (x : ℝ) :
  (2*x^6 + 3*x^5 + 4*x^4 + x^3 + x^2 + x + 20) - (x^6 + 4*x^5 + 2*x^4 - x^3 + 2*x^2 + x + 5) =
  x^6 - x^5 + 2*x^4 + 2*x^3 - x^2 + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2776_277628


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2776_277647

theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets →
  total_cost = 308 := by
sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2776_277647


namespace NUMINAMATH_CALUDE_cube_square_difference_property_l2776_277670

theorem cube_square_difference_property (x : ℝ) : 
  x^3 - x^2 = (x^2 - x)^2 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_square_difference_property_l2776_277670


namespace NUMINAMATH_CALUDE_smallest_divisible_by_million_l2776_277649

/-- A geometric sequence with first term a₁ and common ratio r -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) : ℕ → ℚ :=
  λ n => a₁ * r^(n - 1)

/-- The nth term of the sequence is divisible by m -/
def is_divisible_by (seq : ℕ → ℚ) (n : ℕ) (m : ℕ) : Prop :=
  ∃ k : ℤ, seq n = k * (m : ℚ)

theorem smallest_divisible_by_million :
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := 15
  let r : ℚ := a₂ / a₁
  let seq := geometric_sequence a₁ r
  (∀ n < 7, ¬ is_divisible_by seq n 1000000) ∧
  is_divisible_by seq 7 1000000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_million_l2776_277649


namespace NUMINAMATH_CALUDE_prime_power_plus_144_square_l2776_277699

theorem prime_power_plus_144_square (p : ℕ) (n : ℕ) (m : ℤ) : 
  p.Prime → n > 0 → (p : ℤ)^n + 144 = m^2 → 
  ((p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 15)) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_144_square_l2776_277699


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2776_277661

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2776_277661


namespace NUMINAMATH_CALUDE_blue_parrots_count_l2776_277648

theorem blue_parrots_count (total : ℕ) (green_fraction : ℚ) : 
  total = 108 →
  green_fraction = 5/6 →
  (total : ℚ) * (1 - green_fraction) = 18 := by
  sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l2776_277648


namespace NUMINAMATH_CALUDE_thirtieth_term_value_l2776_277682

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the specific arithmetic sequence -/
def thirtieth_term : ℝ := arithmetic_sequence 3 4 30

theorem thirtieth_term_value : thirtieth_term = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_value_l2776_277682


namespace NUMINAMATH_CALUDE_parameterized_line_solution_l2776_277612

/-- The line y = 4x - 7 parameterized by (x, y) = (s, -3) + t(3, m) -/
def parameterized_line (s m t : ℝ) : ℝ × ℝ :=
  (s + 3*t, -3 + m*t)

/-- The line y = 4x - 7 -/
def line (x y : ℝ) : Prop :=
  y = 4*x - 7

theorem parameterized_line_solution :
  ∃ (s m : ℝ), ∀ (t : ℝ),
    let (x, y) := parameterized_line s m t
    line x y ∧ s = 1 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_solution_l2776_277612


namespace NUMINAMATH_CALUDE_star_property_l2776_277619

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.two
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.one
  | Element.two, Element.four => Element.three
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.three
  | Element.four, Element.three => Element.two
  | Element.four, Element.four => Element.one

theorem star_property : 
  star (star Element.two Element.four) (star Element.one Element.three) = Element.four := by
  sorry

end NUMINAMATH_CALUDE_star_property_l2776_277619


namespace NUMINAMATH_CALUDE_k_at_negative_eight_l2776_277675

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x - 2

-- Define the property that k is a cubic polynomial with the given conditions
def is_valid_k (k : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, h x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, k x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    k 0 = 2

-- Theorem statement
theorem k_at_negative_eight (k : ℝ → ℝ) (hk : is_valid_k k) : k (-8) = -20 := by
  sorry

end NUMINAMATH_CALUDE_k_at_negative_eight_l2776_277675


namespace NUMINAMATH_CALUDE_square_perimeter_l2776_277680

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2776_277680


namespace NUMINAMATH_CALUDE_solve_q_l2776_277623

theorem solve_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_q_l2776_277623


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l2776_277659

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_three (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = 3 := by sorry

-- Theorem 2: z is in the first quadrant iff m < -2 or m > 3
theorem first_quadrant_iff_m_lt_neg_two_or_gt_three (m : ℝ) :
  (z m).re > 0 ∧ (z m).im > 0 ↔ m < -2 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l2776_277659


namespace NUMINAMATH_CALUDE_not_first_year_percentage_l2776_277656

/-- Represents the percentage of associates in each category at a law firm -/
structure LawFirmAssociates where
  secondYear : ℝ
  moreThanTwoYears : ℝ

/-- Theorem stating the percentage of associates who are not first-year associates -/
theorem not_first_year_percentage (firm : LawFirmAssociates) 
  (h1 : firm.secondYear = 25)
  (h2 : firm.moreThanTwoYears = 50) :
  100 - (100 - firm.moreThanTwoYears - firm.secondYear) = 75 := by
  sorry

#check not_first_year_percentage

end NUMINAMATH_CALUDE_not_first_year_percentage_l2776_277656


namespace NUMINAMATH_CALUDE_money_division_l2776_277600

theorem money_division (p q r : ℕ) (total : ℚ) : 
  p + q + r = 22 →  -- ratio sum: 3 + 7 + 12 = 22
  (12 / 22) * total - (7 / 22) * total = 4000 →
  (7 / 22) * total - (3 / 22) * total = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_money_division_l2776_277600


namespace NUMINAMATH_CALUDE_budget_allocation_circle_graph_l2776_277671

theorem budget_allocation_circle_graph (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  (360 : ℝ) * (100 - (microphotonics + home_electronics + food_additives + 
    genetically_modified_microorganisms + industrial_lubricants)) / 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_circle_graph_l2776_277671


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2776_277654

theorem sum_of_three_consecutive_cubes_divisible_by_nine (k : ℕ) :
  ∃ m : ℤ, k^3 + (k+1)^3 + (k+2)^3 = 9*m := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2776_277654


namespace NUMINAMATH_CALUDE_max_value_under_constraints_l2776_277632

/-- Given real numbers x and y satisfying the given conditions, 
    the maximum value of 2x - y is 8. -/
theorem max_value_under_constraints (x y : ℝ) 
  (h1 : x + y - 7 ≤ 0) 
  (h2 : x - 3*y + 1 ≤ 0) 
  (h3 : 3*x - y - 5 ≥ 0) : 
  (∀ a b : ℝ, a + b - 7 ≤ 0 → a - 3*b + 1 ≤ 0 → 3*a - b - 5 ≥ 0 → 2*a - b ≤ 2*x - y) ∧ 
  2*x - y = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_under_constraints_l2776_277632


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l2776_277651

/-- Given two vectors in R², prove that the cosine of the angle between them is 4/5 -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (4, 2) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l2776_277651


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2776_277609

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = -1 - 3*Real.sqrt 2) →
  a = 19/17 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2776_277609


namespace NUMINAMATH_CALUDE_hannah_games_played_l2776_277604

def total_points : ℕ := 312
def average_points : ℕ := 13

theorem hannah_games_played : 
  (total_points / average_points : ℕ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_hannah_games_played_l2776_277604


namespace NUMINAMATH_CALUDE_triangle_height_l2776_277638

/-- Given a triangle with area 615 m² and one side 123 meters, 
    the perpendicular height to that side is 10 meters. -/
theorem triangle_height (area : ℝ) (side : ℝ) (height : ℝ) : 
  area = 615 ∧ side = 123 → height = (2 * area) / side → height = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2776_277638


namespace NUMINAMATH_CALUDE_part_one_part_two_l2776_277630

-- Define the functions f and g
def f (x : ℝ) := |x - 2|
def g (m : ℝ) (x : ℝ) := -|x + 3| + m

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x, g m x ≥ 0 ↔ -5 ≤ x ∧ x ≤ -1) → m = 2 := by sorry

-- Part II
theorem part_two (m : ℝ) :
  (∀ x, f x > g m x) → m < 5 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2776_277630


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2776_277610

/-- Given a line with equation 3x - 6y = 12, prove that its slope (and the slope of any parallel line) is 1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2776_277610


namespace NUMINAMATH_CALUDE_min_m_plus_n_range_of_a_l2776_277642

-- Part I
theorem min_m_plus_n (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = |x + 1| + (1/2) * |2*x - 1|) →
  (m > 0 ∧ n > 0) →
  (∀ x, f x ≥ 1/m + 1/n) →
  m + n ≥ 8/3 :=
sorry

-- Part II
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x + 1| + a * |2*x - 1|) →
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ |x - 2|) →
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_range_of_a_l2776_277642


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l2776_277633

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (4 * x^2 + 3 = 3 * x - 9) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 207/64 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l2776_277633


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l2776_277673

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l2776_277673


namespace NUMINAMATH_CALUDE_a_alone_finish_time_l2776_277624

/-- Represents the time taken by A alone to finish the job -/
def time_a : ℝ := 16

/-- Represents the time taken by A and B together to finish the job -/
def time_ab : ℝ := 40

/-- Represents the number of days A and B worked together -/
def days_together : ℝ := 10

/-- Represents the number of days A worked alone after B left -/
def days_a_alone : ℝ := 12

/-- Theorem stating that given the conditions, A alone can finish the job in 16 days -/
theorem a_alone_finish_time :
  (1 / time_a + 1 / time_ab) * days_together + (1 / time_a) * days_a_alone = 1 :=
sorry

end NUMINAMATH_CALUDE_a_alone_finish_time_l2776_277624


namespace NUMINAMATH_CALUDE_average_tree_height_height_pattern_known_heights_l2776_277664

def tree_heights : List ℝ := [8, 4, 16, 8, 32, 16]

theorem average_tree_height : 
  (tree_heights.sum / tree_heights.length : ℝ) = 14 :=
by
  sorry

theorem height_pattern (i : Fin 5) : 
  tree_heights[i] = 2 * tree_heights[i.succ] ∨ 
  tree_heights[i] = tree_heights[i.succ] / 2 :=
by
  sorry

theorem known_heights : 
  tree_heights[0] = 8 ∧ tree_heights[2] = 16 ∧ tree_heights[4] = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_average_tree_height_height_pattern_known_heights_l2776_277664


namespace NUMINAMATH_CALUDE_power_one_third_five_l2776_277681

theorem power_one_third_five : (1/3 : ℚ)^5 = 1/243 := by sorry

end NUMINAMATH_CALUDE_power_one_third_five_l2776_277681


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_3_closed_2_l2776_277625

-- Define the sets A and B
def A : Set ℝ := {t : ℝ | ∀ x : ℝ, x^2 + 2*t*x - 4*t - 3 ≠ 0}
def B : Set ℝ := {t : ℝ | ∃ x : ℝ, x^2 + 2*t*x - 2*t = 0}

-- State the theorem
theorem A_intersect_B_eq_open_3_closed_2 : A ∩ B = Set.Ioc (-3) (-2) := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_3_closed_2_l2776_277625


namespace NUMINAMATH_CALUDE_monkey_climb_proof_l2776_277658

/-- The height of the tree in feet -/
def tree_height : ℝ := 19

/-- The number of hours the monkey climbs -/
def climbing_hours : ℕ := 17

/-- The distance the monkey slips back each hour in feet -/
def slip_distance : ℝ := 2

/-- The distance the monkey hops each hour in feet -/
def hop_distance : ℝ := 3

theorem monkey_climb_proof :
  tree_height = (climbing_hours - 1) * (hop_distance - slip_distance) + hop_distance :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_proof_l2776_277658


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_2160_l2776_277663

theorem least_k_cube_divisible_by_2160 : 
  ∃ k : ℕ+, (k : ℕ)^3 % 2160 = 0 ∧ ∀ m : ℕ+, (m : ℕ)^3 % 2160 = 0 → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_2160_l2776_277663


namespace NUMINAMATH_CALUDE_mice_without_coins_l2776_277646

theorem mice_without_coins (total_mice : ℕ) (total_coins : ℕ) 
  (h1 : total_mice = 40)
  (h2 : total_coins = 40)
  (h3 : ∃ (y z : ℕ), 
    2 * 2 + 7 * y + 4 * z = total_coins ∧
    2 + y + z + (total_mice - (2 + y + z)) = total_mice) :
  total_mice - (2 + y + z) = 32 :=
by sorry

end NUMINAMATH_CALUDE_mice_without_coins_l2776_277646


namespace NUMINAMATH_CALUDE_equation_equivalence_l2776_277684

theorem equation_equivalence (x : ℝ) : 
  (1 / 2 - (x - 1) / 3 = 1) ↔ (3 - 2 * (x - 1) = 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2776_277684


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_lengths_l2776_277669

/-- A quadrilateral with side lengths 7, 9, 15, and 10 has 10 possible whole number lengths for its diagonal. -/
theorem quadrilateral_diagonal_lengths :
  ∃ (lengths : Finset ℕ),
    (Finset.card lengths = 10) ∧
    (∀ x ∈ lengths,
      (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) ∧
      (x + 10 > 15) ∧ (x + 15 > 10) ∧ (10 + 15 > x) ∧
      (x ≥ 6) ∧ (x ≤ 15)) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_lengths_l2776_277669


namespace NUMINAMATH_CALUDE_prob_3_red_in_5_draws_eq_8_81_l2776_277634

/-- The probability of drawing a red ball from the bag -/
def prob_red : ℚ := 1 / 3

/-- The probability of drawing a white ball from the bag -/
def prob_white : ℚ := 2 / 3

/-- The number of ways to choose 2 red balls from 4 draws -/
def ways_to_choose_2_from_4 : ℕ := 6

/-- The probability of drawing exactly 3 red balls in 5 draws, with the last draw being red -/
def prob_3_red_in_5_draws : ℚ :=
  ways_to_choose_2_from_4 * prob_red^2 * prob_white^2 * prob_red

theorem prob_3_red_in_5_draws_eq_8_81 : 
  prob_3_red_in_5_draws = 8 / 81 := by sorry

end NUMINAMATH_CALUDE_prob_3_red_in_5_draws_eq_8_81_l2776_277634


namespace NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_l2776_277674

theorem largest_integer_with_four_digit_square : ∃ N : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10000 → N ≤ n) ∧ 
  (N^2 < 10000) ∧
  (N^2 ≥ 1000) ∧
  N = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_integer_with_four_digit_square_l2776_277674


namespace NUMINAMATH_CALUDE_cos_double_angle_l2776_277690

theorem cos_double_angle (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((Real.cos α)^2 + (1 / 4)^2) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_l2776_277690


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2776_277607

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_of_A_and_B : A ∪ B = {-3, -2, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2776_277607


namespace NUMINAMATH_CALUDE_erin_curlers_count_l2776_277688

/-- Represents the number of curlers Erin put in her hair -/
def total_curlers : ℕ := 16

/-- Represents the number of small pink curlers -/
def pink_curlers : ℕ := total_curlers / 4

/-- Represents the number of medium blue curlers -/
def blue_curlers : ℕ := 2 * pink_curlers

/-- Represents the number of large green curlers -/
def green_curlers : ℕ := 4

/-- Proves that the total number of curlers is 16 -/
theorem erin_curlers_count :
  total_curlers = pink_curlers + blue_curlers + green_curlers :=
by sorry

end NUMINAMATH_CALUDE_erin_curlers_count_l2776_277688


namespace NUMINAMATH_CALUDE_g_is_zero_l2776_277666

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.sin x ^ 4 + 3 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end NUMINAMATH_CALUDE_g_is_zero_l2776_277666


namespace NUMINAMATH_CALUDE_set_equality_l2776_277627

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2776_277627


namespace NUMINAMATH_CALUDE_sector_area_l2776_277697

theorem sector_area (angle : Real) (radius : Real) (area : Real) : 
  angle = 120 * (π / 180) →  -- Convert 120° to radians
  radius = 10 →
  area = (angle / (2 * π)) * π * radius^2 →
  area = 100 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l2776_277697


namespace NUMINAMATH_CALUDE_tuesday_max_available_l2776_277644

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Dana
  | Evan

-- Define a function to represent availability
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => true
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => true
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Monday => true
  | Person.Dana, Day.Tuesday => true
  | Person.Dana, Day.Wednesday => false
  | Person.Dana, Day.Thursday => false
  | Person.Dana, Day.Friday => true
  | Person.Evan, Day.Monday => false
  | Person.Evan, Day.Tuesday => true
  | Person.Evan, Day.Wednesday => false
  | Person.Evan, Day.Thursday => true
  | Person.Evan, Day.Friday => true

-- Count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dana, Person.Evan])

-- Theorem: Tuesday has the maximum number of available people
theorem tuesday_max_available :
  ∀ d : Day, countAvailable Day.Tuesday ≥ countAvailable d :=
by
  sorry

end NUMINAMATH_CALUDE_tuesday_max_available_l2776_277644


namespace NUMINAMATH_CALUDE_circle_through_points_equation_l2776_277636

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The circle passing through three given points -/
def CircleThroughThreePoints (A B C : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = radius^2

/-- The standard equation of a circle -/
def StandardCircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_points_equation :
  let A : Point := ⟨1, 12⟩
  let B : Point := ⟨7, 10⟩
  let C : Point := ⟨-9, 2⟩
  CircleThroughThreePoints A B C →
  ∃ (x y : ℝ), StandardCircleEquation ⟨1, 2⟩ 10 x y :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_equation_l2776_277636


namespace NUMINAMATH_CALUDE_art_show_sales_l2776_277655

theorem art_show_sales (total : ℕ) (ratio_remaining : ℕ) (ratio_sold : ℕ) (sold : ℕ) : 
  total = 153 →
  ratio_remaining = 9 →
  ratio_sold = 8 →
  (total - sold) * ratio_sold = sold * ratio_remaining →
  sold = 72 := by
sorry

end NUMINAMATH_CALUDE_art_show_sales_l2776_277655


namespace NUMINAMATH_CALUDE_triangle_angles_l2776_277643

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a = 2 * b * Real.cos C ∧
  Real.sin A * Real.sin (B / 2 + C) = Real.sin C * (Real.sin (B / 2) + Real.sin A) →
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l2776_277643


namespace NUMINAMATH_CALUDE_equation_solution_l2776_277602

theorem equation_solution (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2776_277602


namespace NUMINAMATH_CALUDE_target_line_is_correct_l2776_277606

/-- Given two lines in the xy-plane -/
def line1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
def line2 : ℝ → Prop := λ x => x - 2 = 0
def line3 : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0

/-- The intersection point of line2 and line3 -/
def intersection_point : ℝ × ℝ := (2, -1)

/-- The equation of the line we want to prove -/
def target_line : ℝ → ℝ → Prop := λ x y => x - y - 3 = 0

/-- Main theorem -/
theorem target_line_is_correct : 
  (∀ x y, line1 x y ↔ ∃ k, target_line (x + k) (y + k)) ∧ 
  target_line intersection_point.1 intersection_point.2 :=
sorry

end NUMINAMATH_CALUDE_target_line_is_correct_l2776_277606


namespace NUMINAMATH_CALUDE_max_value_abc_l2776_277601

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a^4 * b^2 * c ≤ 1024/117649 := by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2776_277601


namespace NUMINAMATH_CALUDE_jeongyeon_height_is_142_57_l2776_277692

/-- Jeongyeon's height in centimeters -/
def jeongyeon_height : ℝ := 1.06 * 134.5

/-- Theorem stating that Jeongyeon's height is 142.57 cm -/
theorem jeongyeon_height_is_142_57 : 
  jeongyeon_height = 142.57 := by sorry

end NUMINAMATH_CALUDE_jeongyeon_height_is_142_57_l2776_277692


namespace NUMINAMATH_CALUDE_john_total_weight_l2776_277683

/-- The total weight moved by John during his workout -/
def total_weight_moved (weight_per_rep : ℕ) (reps_per_set : ℕ) (num_sets : ℕ) : ℕ :=
  weight_per_rep * reps_per_set * num_sets

/-- Theorem stating that John moves 450 pounds in total -/
theorem john_total_weight :
  total_weight_moved 15 10 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_john_total_weight_l2776_277683


namespace NUMINAMATH_CALUDE_unique_factorial_solution_l2776_277693

theorem unique_factorial_solution : ∃! n : ℕ, n * n.factorial + 2 * n.factorial = 5040 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_solution_l2776_277693


namespace NUMINAMATH_CALUDE_frog_corner_probability_l2776_277687

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up | Down | Left | Right | UpLeft | UpRight | DownLeft | DownRight

/-- The grid on which Frieda moves -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Calculates the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  sorry

/-- Calculates the probability of reaching a corner from a given position in n hops -/
def cornerProbability (grid : Grid) (p : Position) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of reaching any corner within 3 hops from (2,2) is 27/64 -/
theorem frog_corner_probability :
  let initialGrid : Grid := λ _ _ => 0
  let startPos : Position := ⟨1, 1⟩  -- (2,2) in 0-based indexing
  cornerProbability initialGrid startPos 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l2776_277687


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l2776_277686

/-- Given a square carpet with side length 12 feet, containing one large shaded square
    with side length S and eight smaller congruent shaded squares with side length T,
    where 12:S = S:T = 4, prove that the total shaded area is 13.5 square feet. -/
theorem carpet_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  S^2 + 8 * T^2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l2776_277686


namespace NUMINAMATH_CALUDE_complex_modulus_l2776_277660

theorem complex_modulus (z : ℂ) : z * (1 + Complex.I) = Complex.I → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2776_277660


namespace NUMINAMATH_CALUDE_product_of_eights_place_values_l2776_277608

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 780.38

/-- The theorem stating that the product of place values of two 8's in 780.38 is 6.4 -/
theorem product_of_eights_place_values :
  (place_value 8 1) * (place_value 8 (-2)) = 6.4 := by sorry

end NUMINAMATH_CALUDE_product_of_eights_place_values_l2776_277608


namespace NUMINAMATH_CALUDE_briefcase_pen_price_ratio_l2776_277652

/-- Given a pen price of 4 and a total cost of 24 for the pen and a briefcase,
    where the briefcase's price is some multiple of the pen's price,
    prove that the ratio of the briefcase's price to the pen's price is 5. -/
theorem briefcase_pen_price_ratio :
  ∀ (briefcase_price : ℝ),
  briefcase_price > 0 →
  ∃ (multiple : ℝ), multiple > 0 ∧ briefcase_price = 4 * multiple →
  4 + briefcase_price = 24 →
  briefcase_price / 4 = 5 := by
sorry

end NUMINAMATH_CALUDE_briefcase_pen_price_ratio_l2776_277652


namespace NUMINAMATH_CALUDE_siblings_average_age_l2776_277653

theorem siblings_average_age (youngest_age : ℕ) (age_differences : List ℕ) : 
  youngest_age = 20 → 
  age_differences = [2, 7, 11] →
  (youngest_age + youngest_age + 2 + youngest_age + 7 + youngest_age + 11) / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_siblings_average_age_l2776_277653


namespace NUMINAMATH_CALUDE_envelope_area_l2776_277641

/-- The area of a rectangular envelope with width and length both equal to 4 inches is 16 square inches. -/
theorem envelope_area (width : ℝ) (length : ℝ) (h_width : width = 4) (h_length : length = 4) :
  width * length = 16 := by
  sorry

end NUMINAMATH_CALUDE_envelope_area_l2776_277641


namespace NUMINAMATH_CALUDE_dot_product_bound_l2776_277657

theorem dot_product_bound (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 4) 
  (h2 : x^2 + y^2 + z^2 = 9) : 
  -6 ≤ a * x + b * y + c * z ∧ a * x + b * y + c * z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_bound_l2776_277657


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2776_277685

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 25)
  (h2 : rate = 25 / 4)
  (h3 : time = 73 / 365)
  : (interest * 100) / (rate * time) = 2000 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2776_277685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2776_277678

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 2 →                    -- a_1 = 2
  a 2 + a 3 = 13 →             -- a_2 + a_3 = 13
  a 4 + a 5 + a 6 = 42 :=      -- conclusion: a_4 + a_5 + a_6 = 42
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2776_277678


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2776_277695

/-- Given a trapezoid PQRS with specified dimensions, prove the length of QR -/
theorem trapezoid_side_length (area : ℝ) (altitude PQ RS : ℝ) (h1 : area = 210)
  (h2 : altitude = 10) (h3 : PQ = 12) (h4 : RS = 21) :
  ∃ QR : ℝ, QR = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2776_277695


namespace NUMINAMATH_CALUDE_intersection_kth_element_l2776_277645

-- Define set A
def A : Set ℕ := {n | ∃ m : ℕ, n = m * (m + 1)}

-- Define set B
def B : Set ℕ := {n | ∃ m : ℕ, n = 3 * m - 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℕ := A ∩ B

-- Define the kth element of the intersection
def a (k : ℕ) : ℕ := 9 * k^2 - 9 * k + 2

-- Theorem statement
theorem intersection_kth_element (k : ℕ) : 
  a k ∈ A_intersect_B ∧ 
  (∀ n ∈ A_intersect_B, n < a k → 
    ∃ j < k, n = a j) ∧
  (∀ n ∈ A_intersect_B, n ≠ a k → 
    ∃ j ≠ k, n = a j) :=
sorry

end NUMINAMATH_CALUDE_intersection_kth_element_l2776_277645


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2776_277694

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2776_277694


namespace NUMINAMATH_CALUDE_square_49_equals_square_50_minus_99_l2776_277650

theorem square_49_equals_square_50_minus_99 : 49^2 = 50^2 - 99 := by
  sorry

end NUMINAMATH_CALUDE_square_49_equals_square_50_minus_99_l2776_277650


namespace NUMINAMATH_CALUDE_square_of_linear_cyclic_l2776_277611

variable (a b c A B C : ℝ)

/-- Two linear polynomials sum to a square of a linear polynomial iff their coefficients satisfy this condition -/
def is_square_of_linear (α β γ δ : ℝ) : Prop :=
  α * δ = β * γ

/-- The main theorem: if two pairs of expressions are squares of linear polynomials, 
    then the third pair is also a square of a linear polynomial -/
theorem square_of_linear_cyclic 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0)
  (h1 : is_square_of_linear a b A B)
  (h2 : is_square_of_linear b c B C) :
  is_square_of_linear c a C A :=
sorry

end NUMINAMATH_CALUDE_square_of_linear_cyclic_l2776_277611


namespace NUMINAMATH_CALUDE_proportion_equality_l2776_277618

theorem proportion_equality (x y : ℝ) (h1 : 3 * x = 2 * y) (h2 : y ≠ 0) : x / 2 = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2776_277618


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l2776_277617

-- Define a type for lines
variable (Line : Type)

-- Define a property for line intersection
variable (intersect : Line → Line → Prop)

-- Define a property for lines passing through a common point
variable (pass_through_common_point : (Set Line) → Prop)

-- Define a property for lines lying in one plane
variable (lie_in_one_plane : (Set Line) → Prop)

-- The main theorem
theorem line_intersection_theorem (S : Set Line) :
  (∀ l1 l2 : Line, l1 ∈ S → l2 ∈ S → l1 ≠ l2 → intersect l1 l2) →
  (pass_through_common_point S ∨ lie_in_one_plane S) :=
by sorry


end NUMINAMATH_CALUDE_line_intersection_theorem_l2776_277617


namespace NUMINAMATH_CALUDE_line_equation_given_ellipse_midpoint_l2776_277629

/-- The equation of a line that intersects an ellipse, given the midpoint of the intersection -/
theorem line_equation_given_ellipse_midpoint (x y : ℝ → ℝ) (l : Set (ℝ × ℝ)) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, (x t₁, y t₁) ∈ l ∧ (x t₂, y t₂) ∈ l ∧ t₁ ≠ t₂) →  -- Line intersects ellipse at two points
  ((x t₁ + x t₂) / 2 = 4 ∧ (y t₁ + y t₂) / 2 = 2) →  -- Midpoint is (4,2)
  (∀ p, p ∈ l ↔ p.1 + 2 * p.2 - 8 = 0) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_line_equation_given_ellipse_midpoint_l2776_277629


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2776_277667

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2776_277667


namespace NUMINAMATH_CALUDE_smallest_group_sample_size_l2776_277639

def stratified_sampling (total_sample_size : ℕ) (group_ratios : List ℕ) : List ℕ :=
  let total_ratio := group_ratios.sum
  group_ratios.map (λ ratio => (total_sample_size * ratio) / total_ratio)

theorem smallest_group_sample_size 
  (total_sample_size : ℕ) 
  (group_ratios : List ℕ) :
  total_sample_size = 20 →
  group_ratios = [5, 4, 1] →
  (stratified_sampling total_sample_size group_ratios).getLast! = 2 :=
by
  sorry

#eval stratified_sampling 20 [5, 4, 1]

end NUMINAMATH_CALUDE_smallest_group_sample_size_l2776_277639


namespace NUMINAMATH_CALUDE_article_cost_price_l2776_277620

theorem article_cost_price : ∃ C : ℝ, C = 400 ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 2) := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2776_277620


namespace NUMINAMATH_CALUDE_large_bottle_price_calculation_l2776_277622

-- Define the variables
def large_bottles : ℕ := 1300
def small_bottles : ℕ := 750
def small_bottle_price : ℚ := 138 / 100
def average_price : ℚ := 17034 / 10000

-- Define the theorem
theorem large_bottle_price_calculation :
  ∃ (large_price : ℚ),
    (large_bottles * large_price + small_bottles * small_bottle_price) / (large_bottles + small_bottles) = average_price ∧
    abs (large_price - 189 / 100) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_large_bottle_price_calculation_l2776_277622


namespace NUMINAMATH_CALUDE_modulus_of_one_over_one_plus_i_l2776_277677

open Complex

theorem modulus_of_one_over_one_plus_i : 
  let z : ℂ := 1 / (1 + I)
  abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_over_one_plus_i_l2776_277677


namespace NUMINAMATH_CALUDE_yolas_past_weight_l2776_277637

/-- Yola's past weight given current weights and differences -/
theorem yolas_past_weight
  (yola_current : ℝ)
  (wanda_yola_current_diff : ℝ)
  (wanda_yola_past_diff : ℝ)
  (h1 : yola_current = 220)
  (h2 : wanda_yola_current_diff = 30)
  (h3 : wanda_yola_past_diff = 80) :
  yola_current - (wanda_yola_past_diff - wanda_yola_current_diff) = 170 :=
by
  sorry

end NUMINAMATH_CALUDE_yolas_past_weight_l2776_277637


namespace NUMINAMATH_CALUDE_one_third_of_recipe_flour_l2776_277676

theorem one_third_of_recipe_flour (original_flour : ℚ) (reduced_flour : ℚ) : 
  original_flour = 17/3 → reduced_flour = original_flour / 3 → reduced_flour = 17/9 := by
  sorry

#check one_third_of_recipe_flour

end NUMINAMATH_CALUDE_one_third_of_recipe_flour_l2776_277676


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l2776_277613

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 40, 40, and 36, the distance between two adjacent parallel lines is 7.8. -/
theorem parallel_lines_distance (r : ℝ) : 
  let d := (4336 : ℝ) / 71
  (40 : ℝ) * r^2 = 16000 + 10 * d ∧ 
  (36 : ℝ) * r^2 = 11664 + 81 * d → 
  Real.sqrt d = 7.8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l2776_277613


namespace NUMINAMATH_CALUDE_sum_rational_irrational_not_rational_l2776_277626

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sum_rational_irrational_not_rational :
  ∀ (r i : ℝ), IsRational r → IsIrrational i → IsIrrational (r + i) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_rational_irrational_not_rational_l2776_277626


namespace NUMINAMATH_CALUDE_bottle_production_l2776_277603

/-- Given that 5 identical machines produce 270 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2160 bottles in 4 minutes. -/
theorem bottle_production 
  (machines_5 : ℕ) 
  (bottles_per_minute_5 : ℕ) 
  (machines_10 : ℕ) 
  (minutes : ℕ) 
  (h1 : machines_5 = 5) 
  (h2 : bottles_per_minute_5 = 270) 
  (h3 : machines_10 = 10) 
  (h4 : minutes = 4) :
  (machines_10 * (bottles_per_minute_5 / machines_5) * minutes) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l2776_277603


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l2776_277679

theorem greatest_prime_factor_of_sum (n : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^8 + 10^7) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^8 + 10^7) → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^8 + 10^7) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^8 + 10^7) → q ≤ p ∧ p = 19) :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l2776_277679


namespace NUMINAMATH_CALUDE_fraction_value_l2776_277605

theorem fraction_value (x : ℤ) : 
  (∃ (n : ℕ+), (2 : ℚ) / (x + 1 : ℚ) = n) → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2776_277605


namespace NUMINAMATH_CALUDE_bashers_win_probability_l2776_277665

def probability_at_least_4_out_of_5 (p : ℝ) : ℝ :=
  5 * p^4 * (1 - p) + p^5

theorem bashers_win_probability :
  probability_at_least_4_out_of_5 (4/5) = 3072/3125 := by
  sorry

end NUMINAMATH_CALUDE_bashers_win_probability_l2776_277665


namespace NUMINAMATH_CALUDE_number_equation_solution_l2776_277640

theorem number_equation_solution : ∃ x : ℝ, 33 + 3 * x = 48 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2776_277640


namespace NUMINAMATH_CALUDE_triangle_theorem_l2776_277621

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  (t.c * Real.cos t.A + t.a * Real.cos t.C = 2 * t.b * Real.cos t.A) →
  Real.cos t.A = 1 / 2 ∧
  (t.a = Real.sqrt 7 ∧ t.b + t.c = 4) →
  (1 / 2 : ℝ) * t.b * t.c * Real.sqrt (1 - (1 / 2)^2) = 3 * Real.sqrt 3 / 4 :=
by sorry

-- Note: The area formula is expanded as (1/2) * b * c * sin A,
-- where sin A is written as sqrt(1 - cos^2 A)

end NUMINAMATH_CALUDE_triangle_theorem_l2776_277621


namespace NUMINAMATH_CALUDE_investment_problem_l2776_277631

/-- Given a sum P invested at a rate R for 20 years, if investing at a rate (R + 10)%
    yields Rs. 3000 more in interest, then P = 1500. -/
theorem investment_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  P * (R + 10) * 20 / 100 = P * R * 20 / 100 + 3000 →
  P = 1500 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l2776_277631


namespace NUMINAMATH_CALUDE_church_attendance_l2776_277662

/-- The total number of people in the church is the sum of children, male adults, and female adults. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (female_adults : ℕ) 
  (h1 : children = 80) 
  (h2 : male_adults = 60) 
  (h3 : female_adults = 60) : 
  children + male_adults + female_adults = 200 := by
  sorry

#check church_attendance

end NUMINAMATH_CALUDE_church_attendance_l2776_277662


namespace NUMINAMATH_CALUDE_rs_length_l2776_277614

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧ 
  t.pq ∈ t.edges ∧
  ∀ e ∈ t.edges, e > 0

theorem rs_length (t : Tetrahedron) 
  (h_valid : valid_tetrahedron t)
  (h_edges : t.edges = {9, 16, 22, 31, 39, 48})
  (h_pq : t.pq = 48) :
  ∃ rs ∈ t.edges, rs = 9 :=
sorry

end NUMINAMATH_CALUDE_rs_length_l2776_277614


namespace NUMINAMATH_CALUDE_babysitting_earnings_l2776_277696

/-- Calculates the amount earned for a given hour of babysitting -/
def hourlyRate (hour : ℕ) : ℕ :=
  2 * (hour % 6 + 1)

/-- Calculates the total amount earned for a given number of hours -/
def totalEarned (hours : ℕ) : ℕ :=
  (List.range hours).map hourlyRate |>.sum

theorem babysitting_earnings : totalEarned 48 = 288 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l2776_277696


namespace NUMINAMATH_CALUDE_select_books_count_l2776_277615

/-- The number of ways to select 5 books from 8 books, where 3 of the books form a trilogy that must be selected together. -/
def select_books : ℕ := 
  Nat.choose 5 2 + Nat.choose 5 5

/-- Theorem stating that the number of ways to select the books is 11. -/
theorem select_books_count : select_books = 11 := by
  sorry

end NUMINAMATH_CALUDE_select_books_count_l2776_277615


namespace NUMINAMATH_CALUDE_kittens_from_friends_proof_l2776_277698

/-- The number of kittens Joan's cat had initially -/
def initial_kittens : ℕ := 8

/-- The total number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := total_kittens - initial_kittens

theorem kittens_from_friends_proof :
  kittens_from_friends = total_kittens - initial_kittens :=
by sorry

end NUMINAMATH_CALUDE_kittens_from_friends_proof_l2776_277698


namespace NUMINAMATH_CALUDE_two_distinct_roots_l2776_277691

/-- The equation has exactly two distinct real roots if and only if a > 0 or a = -2 -/
theorem two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    x^2 - 6*x + (a-2)*|x-3| + 9 - 2*a = 0 ∧
    y^2 - 6*y + (a-2)*|y-3| + 9 - 2*a = 0 ∧
    (∀ z : ℝ, z^2 - 6*z + (a-2)*|z-3| + 9 - 2*a = 0 → z = x ∨ z = y)) ↔
  (a > 0 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l2776_277691


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2776_277616

theorem absolute_value_inequality (x : ℝ) : |5 - 2*x| ≥ 3 ↔ x < 1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2776_277616


namespace NUMINAMATH_CALUDE_cello_practice_time_l2776_277635

/-- Calculates the remaining practice time in minutes for a cellist -/
theorem cello_practice_time (total_hours : ℝ) (daily_minutes : ℕ) (practice_days : ℕ) :
  total_hours = 7.5 ∧ daily_minutes = 86 ∧ practice_days = 2 →
  (total_hours * 60 : ℝ) - (daily_minutes * practice_days : ℕ) = 278 := by
  sorry

#check cello_practice_time

end NUMINAMATH_CALUDE_cello_practice_time_l2776_277635


namespace NUMINAMATH_CALUDE_lcm_of_16_and_24_l2776_277668

theorem lcm_of_16_and_24 :
  let n : ℕ := 16
  let m : ℕ := 24
  let g : ℕ := 8
  Nat.gcd n m = g →
  Nat.lcm n m = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_16_and_24_l2776_277668


namespace NUMINAMATH_CALUDE_bike_speed_calculation_l2776_277689

/-- Proves that given the conditions, the bike speed is 15 km/h -/
theorem bike_speed_calculation (distance : ℝ) (car_speed_multiplier : ℝ) (time_difference : ℝ) :
  distance = 15 →
  car_speed_multiplier = 4 →
  time_difference = 45 / 60 →
  ∃ (bike_speed : ℝ), 
    bike_speed > 0 ∧
    distance / bike_speed - distance / (car_speed_multiplier * bike_speed) = time_difference ∧
    bike_speed = 15 :=
by sorry

end NUMINAMATH_CALUDE_bike_speed_calculation_l2776_277689


namespace NUMINAMATH_CALUDE_mat_weaving_in_12_days_l2776_277672

/-- Represents a group of mat weavers -/
structure WeaverGroup where
  weavers : ℕ
  mats : ℕ
  days : ℕ

/-- Calculates the number of mats a group can weave in a given number of days -/
def mats_in_days (group : WeaverGroup) (target_days : ℕ) : ℕ :=
  (group.mats * target_days) / group.days

/-- Group A of mat weavers -/
def group_A : WeaverGroup :=
  { weavers := 4, mats := 4, days := 4 }

/-- Group B of mat weavers -/
def group_B : WeaverGroup :=
  { weavers := 6, mats := 9, days := 3 }

/-- Group C of mat weavers -/
def group_C : WeaverGroup :=
  { weavers := 8, mats := 16, days := 4 }

theorem mat_weaving_in_12_days :
  mats_in_days group_A 12 = 12 ∧
  mats_in_days group_B 12 = 36 ∧
  mats_in_days group_C 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_mat_weaving_in_12_days_l2776_277672
