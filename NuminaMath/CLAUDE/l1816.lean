import Mathlib

namespace NUMINAMATH_CALUDE_algebraic_identity_l1816_181608

theorem algebraic_identity (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l1816_181608


namespace NUMINAMATH_CALUDE_johns_age_l1816_181668

def johns_age_problem (j d : ℕ) : Prop :=
  (j = d - 30) ∧ (j + d = 80)

theorem johns_age : ∃ j d : ℕ, johns_age_problem j d ∧ j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l1816_181668


namespace NUMINAMATH_CALUDE_a1_range_l1816_181659

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

theorem a1_range (a : ℕ → ℝ) :
  is_monotonically_increasing a →
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 3 * n - 6) →
  a 2 = (1 / 2) * a 1 →
  ∃ x : ℝ, -12/5 < x ∧ x < -3/2 ∧ a 1 = x :=
by
  sorry

end NUMINAMATH_CALUDE_a1_range_l1816_181659


namespace NUMINAMATH_CALUDE_point_n_coordinates_l1816_181626

/-- Given point M(5, -6) and vector a = (1, -2), if MN = -3a, then N has coordinates (2, 0) -/
theorem point_n_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) :
  M = (5, -6) →
  a = (1, -2) →
  N.1 - M.1 = -3 * a.1 ∧ N.2 - M.2 = -3 * a.2 →
  N = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_n_coordinates_l1816_181626


namespace NUMINAMATH_CALUDE_base_b_121_is_perfect_square_l1816_181610

/-- Represents a number in base b as a list of digits --/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * b + digit) 0

/-- Checks if a number is a perfect square --/
def IsPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base_b_121_is_perfect_square (b : Nat) :
  (b > 2) ↔ IsPerfectSquare (BaseRepresentation [1, 2, 1] b) :=
by sorry

end NUMINAMATH_CALUDE_base_b_121_is_perfect_square_l1816_181610


namespace NUMINAMATH_CALUDE_multiple_properties_l1816_181665

theorem multiple_properties (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  (∃ k : ℤ, x - y = 3 * k) ∧ 
  (∃ a b : ℤ, (∃ m : ℤ, a = 6 * m) ∧ (∃ n : ℤ, b = 9 * n) ∧ (∃ l : ℤ, a - b = 9 * l)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l1816_181665


namespace NUMINAMATH_CALUDE_min_value_of_f_l1816_181603

/-- The function f(n) = n^2 - 8n + 5 -/
def f (n : ℝ) : ℝ := n^2 - 8*n + 5

/-- The minimum value of f(n) is -11 -/
theorem min_value_of_f : ∀ n : ℝ, f n ≥ -11 ∧ ∃ n₀ : ℝ, f n₀ = -11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1816_181603


namespace NUMINAMATH_CALUDE_f_has_two_roots_l1816_181696

/-- The function f(x) = x^4 + 5x^3 + 6x^2 - 4x - 16 -/
def f (x : ℝ) : ℝ := x^4 + 5*x^3 + 6*x^2 - 4*x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4*x^3 + 15*x^2 + 12*x - 4

theorem f_has_two_roots :
  ∃! (a b : ℝ), a < b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_roots_l1816_181696


namespace NUMINAMATH_CALUDE_x_fourth_minus_inverse_x_fourth_l1816_181690

theorem x_fourth_minus_inverse_x_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_inverse_x_fourth_l1816_181690


namespace NUMINAMATH_CALUDE_unique_solution_l1816_181632

/-- Represents a number in a given base --/
def baseRepresentation (n : ℕ) (base : ℕ) : ℕ → ℕ
| 0 => n % base
| k + 1 => baseRepresentation (n / base) base k

/-- The equation to be solved --/
def equationHolds (x : ℕ) : Prop :=
  baseRepresentation 2016 x 3 * x^3 +
  baseRepresentation 2016 x 2 * x^2 +
  baseRepresentation 2016 x 1 * x +
  baseRepresentation 2016 x 0 = x^3 + 2*x + 342

theorem unique_solution :
  ∃! x : ℕ, x > 0 ∧ equationHolds x ∧ x = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1816_181632


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_108_l1816_181629

theorem percentage_of_360_equals_108 : 
  ∃ (p : ℝ), p * 360 / 100 = 108.0 ∧ p = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_108_l1816_181629


namespace NUMINAMATH_CALUDE_necklace_price_l1816_181664

theorem necklace_price (bracelet_price earring_price ensemble_price : ℕ)
                       (necklaces bracelets earrings ensembles : ℕ)
                       (total_revenue : ℕ) :
  bracelet_price = 15 →
  earring_price = 10 →
  ensemble_price = 45 →
  necklaces = 5 →
  bracelets = 10 →
  earrings = 20 →
  ensembles = 2 →
  total_revenue = 565 →
  ∃ (necklace_price : ℕ),
    necklace_price = 25 ∧
    necklace_price * necklaces + bracelet_price * bracelets + 
    earring_price * earrings + ensemble_price * ensembles = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_necklace_price_l1816_181664


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1816_181672

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(4, -3) is -8/3 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, 
    (3 - 0)^2 + (0 - y)^2 = (4 - 0)^2 + (-3 - y)^2 ∧ 
    y = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l1816_181672


namespace NUMINAMATH_CALUDE_result_of_operation_l1816_181657

theorem result_of_operation (n : ℕ) (h : n = 95) : (n / 5 + 23 : ℚ) = 42 := by
  sorry

end NUMINAMATH_CALUDE_result_of_operation_l1816_181657


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_specific_evaluation_l1816_181655

theorem expression_simplification_and_evaluation (a b : ℚ) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = 4*a*b :=
by sorry

theorem specific_evaluation :
  let a : ℚ := -1
  let b : ℚ := 1/2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) - 2*a*a = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_specific_evaluation_l1816_181655


namespace NUMINAMATH_CALUDE_sine_cosine_unique_pair_l1816_181613

open Real

theorem sine_cosine_unique_pair :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    sin (cos c) = c ∧ cos (sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_unique_pair_l1816_181613


namespace NUMINAMATH_CALUDE_other_divisor_is_57_l1816_181619

theorem other_divisor_is_57 : 
  ∃ (x : ℕ), x ≠ 38 ∧ 
  114 % x = 0 ∧ 
  115 % x = 1 ∧
  115 % 38 = 1 ∧
  (∀ y : ℕ, y > x → 114 % y = 0 → y = 38 ∨ y = 114) :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_is_57_l1816_181619


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_tenth_squared_l1816_181667

theorem decimal_equivalent_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_tenth_squared_l1816_181667


namespace NUMINAMATH_CALUDE_afternoon_campers_l1816_181618

theorem afternoon_campers (morning_campers : ℕ) (afternoon_difference : ℕ) : 
  morning_campers = 52 → 
  afternoon_difference = 9 → 
  morning_campers + afternoon_difference = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_l1816_181618


namespace NUMINAMATH_CALUDE_system_solutions_l1816_181607

-- Define the system of equations
def equation1 (y z : ℚ) : Prop := y * z = 3 * y + 2 * z - 8
def equation2 (z x : ℚ) : Prop := z * x = 4 * z + 3 * x - 8
def equation3 (x y : ℚ) : Prop := x * y = 2 * x + y - 1

-- Define the solutions
def solution1 : (ℚ × ℚ × ℚ) := (2, 3, 1)
def solution2 : (ℚ × ℚ × ℚ) := (3, 5/2, -1)

-- Theorem statement
theorem system_solutions :
  (equation1 solution1.2.1 solution1.2.2 ∧ 
   equation2 solution1.2.2 solution1.1 ∧ 
   equation3 solution1.1 solution1.2.1) ∧
  (equation1 solution2.2.1 solution2.2.2 ∧ 
   equation2 solution2.2.2 solution2.1 ∧ 
   equation3 solution2.1 solution2.2.1) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1816_181607


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1816_181678

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ x, a*x^2 + b*x + c = 0 ∧ ∀ y, a*y^2 + b*y + c = 0 → y = x

/-- The problem statement -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, y^2 = 32*x → (4*x + 6*y + k = 0 ↔ 
    ∃! t, 4*t + 6*(32*t)^(1/2) + k = 0 ∨ 4*t + 6*(-32*t)^(1/2) + k = 0)) →
  k = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1816_181678


namespace NUMINAMATH_CALUDE_unique_rectangle_dimensions_l1816_181640

theorem unique_rectangle_dimensions (a b : ℕ) : 
  a < b ∧ 
  (∃ k : ℕ, 49 * 51 = k * (a * b)) ∧ 
  (∃ m : ℕ, 99 * 101 = m * (a * b)) →
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_rectangle_dimensions_l1816_181640


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l1816_181627

theorem cubic_function_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 - b*x + a^2
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x - b
  (f 1 = 10 ∧ f' 1 = 0) → ((a = -4 ∧ b = 11) ∨ (a = 3 ∧ b = -3)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_l1816_181627


namespace NUMINAMATH_CALUDE_special_function_property_l1816_181623

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) :
  (g 7 - g 3) / g 4 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1816_181623


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l1816_181675

/-- The number of times Terrell lifts the 40-pound weight -/
def original_lifts : ℕ := 12

/-- The weight of the original weight in pounds -/
def original_weight : ℕ := 40

/-- The weight of the new weight in pounds -/
def new_weight : ℕ := 30

/-- The total weight lifted with the original weight -/
def total_weight : ℕ := original_weight * original_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def new_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  new_lifts = 16 :=
sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l1816_181675


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l1816_181652

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  team_avg_age = 23 →
  ∃ (wicket_keeper_age : ℕ),
    wicket_keeper_age > captain_age ∧
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = team_avg_age ∧
    wicket_keeper_age - captain_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l1816_181652


namespace NUMINAMATH_CALUDE_ones_count_l1816_181653

theorem ones_count (hundreds tens total : ℕ) (h1 : hundreds = 3) (h2 : tens = 8) (h3 : total = 383) :
  total - (hundreds * 100 + tens * 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_count_l1816_181653


namespace NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_property_l1816_181601

-- Define a structure for a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (A B C : ℝ)  -- angles in radians

-- Define the property of angles forming an arithmetic sequence
def arithmeticSequence (t : Triangle) : Prop :=
  ∃ d : ℝ, t.B - t.A = d ∧ t.C - t.B = d

-- State the theorem
theorem triangle_angle_arithmetic_sequence_property (t : Triangle) 
  (h1 : t.a > 0) (h2 : t.b > 0) (h3 : t.c > 0)  -- positive sides
  (h4 : arithmeticSequence t)  -- angles form arithmetic sequence
  : 1 / (t.a + t.b) + 1 / (t.b + t.c) = 3 / (t.a + t.b + t.c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_property_l1816_181601


namespace NUMINAMATH_CALUDE_notebook_cost_theorem_l1816_181646

def total_money : ℕ := 40
def num_posters : ℕ := 2
def poster_cost : ℕ := 5
def num_notebooks : ℕ := 3
def num_bookmarks : ℕ := 2
def bookmark_cost : ℕ := 2
def money_left : ℕ := 14

theorem notebook_cost_theorem (notebook_cost : ℕ) : 
  total_money - (num_posters * poster_cost + num_notebooks * notebook_cost + num_bookmarks * bookmark_cost) = money_left →
  notebook_cost = 4 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_theorem_l1816_181646


namespace NUMINAMATH_CALUDE_smallest_b_value_l1816_181680

theorem smallest_b_value (a c d : ℤ) (x₁ x₂ x₃ x₄ : ℝ) :
  (∀ x, x^4 + a*x^3 + (x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄)*x^2 + c*x + d = 0 → x > 0) →
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 →
  d = x₁ * x₂ * x₃ * x₄ →
  x₁*x₂ + x₁*x₃ + x₁*x₄ + x₂*x₃ + x₂*x₄ + x₃*x₄ ≥ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1816_181680


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1816_181645

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d equals 2 when (S_3 / 3) - (S_2 / 2) = 1 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_condition : S 3 / 3 - S 2 / 2 = 1) :
  a 2 - a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1816_181645


namespace NUMINAMATH_CALUDE_circular_track_length_circular_track_length_is_280_l1816_181648

/-- The length of a circular track given specific running conditions -/
theorem circular_track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed jim_speed : ℝ),
      brenda_speed > 0 ∧ jim_speed > 0 →
      ∃ (first_meet_time second_meet_time : ℝ),
        first_meet_time > 0 ∧ second_meet_time > first_meet_time ∧
        brenda_speed * first_meet_time = 120 ∧
        jim_speed * second_meet_time = 300 ∧
        (brenda_speed * first_meet_time + jim_speed * first_meet_time = track_length / 2) ∧
        (brenda_speed * second_meet_time + jim_speed * second_meet_time = track_length) →
        track_length = 280

/-- The circular track length is 280 meters -/
theorem circular_track_length_is_280 : circular_track_length 280 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_length_circular_track_length_is_280_l1816_181648


namespace NUMINAMATH_CALUDE_gcd_78_36_l1816_181639

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l1816_181639


namespace NUMINAMATH_CALUDE_home_run_difference_l1816_181671

theorem home_run_difference (aaron_hr winfield_hr : ℕ) : 
  aaron_hr = 755 → winfield_hr = 465 → 2 * winfield_hr - aaron_hr = 175 := by
  sorry

end NUMINAMATH_CALUDE_home_run_difference_l1816_181671


namespace NUMINAMATH_CALUDE_heathers_remaining_blocks_l1816_181679

theorem heathers_remaining_blocks
  (initial_blocks : ℕ)
  (shared_with_jose : ℕ)
  (shared_with_emily : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_with_jose = 41)
  (h3 : shared_with_emily = 15) :
  initial_blocks - (shared_with_jose + shared_with_emily) = 30 :=
by sorry

end NUMINAMATH_CALUDE_heathers_remaining_blocks_l1816_181679


namespace NUMINAMATH_CALUDE_azalea_profit_l1816_181612

/-- Calculates the profit from a sheep farm given the number of sheep, shearing cost, wool per sheep, and price per pound of wool. -/
def sheep_farm_profit (num_sheep : ℕ) (shearing_cost : ℕ) (wool_per_sheep : ℕ) (price_per_pound : ℕ) : ℕ :=
  num_sheep * wool_per_sheep * price_per_pound - shearing_cost

/-- Proves that Azalea's profit from her sheep farm is $38,000 -/
theorem azalea_profit : sheep_farm_profit 200 2000 10 20 = 38000 := by
  sorry

end NUMINAMATH_CALUDE_azalea_profit_l1816_181612


namespace NUMINAMATH_CALUDE_m_range_l1816_181654

def p (x m : ℝ) : Prop := x^2 + 2*x - m > 0

theorem m_range :
  (∀ m : ℝ, ¬(p 1 m) ∧ (p 2 m)) ↔ (∀ m : ℝ, 3 ≤ m ∧ m < 8) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1816_181654


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1816_181683

-- Define set A
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define set B
def B : Set ℝ := {a : ℝ | ∀ x : ℝ, ¬(|x - 4| + |x - 3| < a)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1816_181683


namespace NUMINAMATH_CALUDE_number_equation_solution_l1816_181642

theorem number_equation_solution : ∃ (x : ℝ), x + 3 * x = 20 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1816_181642


namespace NUMINAMATH_CALUDE_cannot_reach_goal_l1816_181620

/-- Represents the types of donuts --/
inductive DonutType
  | Plain
  | Glazed
  | Chocolate

/-- Represents the cost and price information for donuts --/
structure DonutInfo where
  costPerDozen : ℝ
  sellingPrice : ℝ

/-- The goal amount to be raised --/
def goalAmount : ℝ := 96

/-- The maximum number of dozens that can be bought --/
def maxDozens : ℕ := 6

/-- The number of donut types --/
def numTypes : ℕ := 3

/-- The donut information for each type --/
def donutData : DonutType → DonutInfo
  | DonutType.Plain => { costPerDozen := 2.4, sellingPrice := 1 }
  | DonutType.Glazed => { costPerDozen := 3.6, sellingPrice := 1.5 }
  | DonutType.Chocolate => { costPerDozen := 4.8, sellingPrice := 2 }

/-- Calculate the profit for a given number of dozens of a specific donut type --/
def profitForType (t : DonutType) (dozens : ℝ) : ℝ :=
  let info := donutData t
  dozens * (12 * info.sellingPrice - info.costPerDozen)

/-- The main theorem stating that the goal cannot be reached --/
theorem cannot_reach_goal :
  ∀ x : ℝ, x > 0 → x ≤ (maxDozens / numTypes : ℝ) →
  (profitForType DonutType.Plain x +
   profitForType DonutType.Glazed x +
   profitForType DonutType.Chocolate x) < goalAmount :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_goal_l1816_181620


namespace NUMINAMATH_CALUDE_cube_sum_of_distinct_reals_l1816_181647

theorem cube_sum_of_distinct_reals (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_distinct_reals_l1816_181647


namespace NUMINAMATH_CALUDE_alice_ice_cream_l1816_181685

/-- The number of pints of ice cream Alice bought on Sunday -/
def sunday_pints : ℕ := sorry

/-- The number of pints Alice had on Wednesday after returning expired ones -/
def wednesday_pints : ℕ := 18

theorem alice_ice_cream :
  sunday_pints = 4 ∧
  3 * sunday_pints + sunday_pints + sunday_pints - sunday_pints / 2 = wednesday_pints :=
by sorry

end NUMINAMATH_CALUDE_alice_ice_cream_l1816_181685


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1816_181687

theorem quadratic_roots_properties (b c x₁ x₂ : ℝ) 
  (h_eq : ∀ x, x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂)
  (h_distinct : x₁ ≠ x₂)
  (h_order : x₁ < x₂)
  (h_x₁_range : -1 < x₁ ∧ x₁ < 0) :
  (x₂ > 0 → c < 0) ∧
  (|x₂ - x₁| = 2 → |1 - b + c| - |1 + b + c| > 2*|4 + 2*b + c| - 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1816_181687


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_with_square_root_l1816_181617

theorem unique_solution_quadratic_with_square_root :
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 4) = 31 :=
by
  -- The unique solution is (11 - 3√5) / 2
  use (11 - 3 * Real.sqrt 5) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_with_square_root_l1816_181617


namespace NUMINAMATH_CALUDE_factorization_problem_l1816_181637

theorem factorization_problem (A B : ℤ) :
  (∀ x : ℝ, 10 * x^2 - 31 * x + 21 = (A * x - 7) * (B * x - 3)) →
  A * B + A = 15 := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_l1816_181637


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1816_181649

theorem min_value_quadratic_form (x₁ x₂ x₃ x₄ : ℝ) 
  (h : 5*x₁ + 6*x₂ - 7*x₃ + 4*x₄ = 1) : 
  3*x₁^2 + 2*x₂^2 + 5*x₃^2 + x₄^2 ≥ 15/782 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1816_181649


namespace NUMINAMATH_CALUDE_goose_survival_rate_l1816_181689

theorem goose_survival_rate (total_eggs : ℝ) (hatched_fraction : ℝ) (first_year_survival_fraction : ℝ) (first_year_survivors : ℕ) : 
  total_eggs = 550 →
  hatched_fraction = 2/3 →
  first_year_survival_fraction = 2/5 →
  first_year_survivors = 110 →
  ∃ (first_month_survival_fraction : ℝ),
    first_month_survival_fraction * hatched_fraction * first_year_survival_fraction * total_eggs = first_year_survivors ∧
    first_month_survival_fraction = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_goose_survival_rate_l1816_181689


namespace NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l1816_181663

theorem no_solutions_lcm_gcd_equation : 
  ¬∃ (n : ℕ+), Nat.lcm n 120 = Nat.gcd n 120 + 360 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_lcm_gcd_equation_l1816_181663


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1816_181609

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x^2 - x - 2 < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(x^2 - x - 2 < 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1816_181609


namespace NUMINAMATH_CALUDE_candies_remaining_l1816_181633

/-- Calculates the number of candies remaining after Carlos ate all yellow candies -/
theorem candies_remaining (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow = 3 * red - 20)
  (h3 : blue = yellow / 2) :
  red + blue = 90 := by
  sorry

#check candies_remaining

end NUMINAMATH_CALUDE_candies_remaining_l1816_181633


namespace NUMINAMATH_CALUDE_expression_sum_equals_one_l1816_181691

theorem expression_sum_equals_one (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hprod : x * y * z = 1) :
  1 / (1 + x + x * y) + y / (1 + y + y * z) + x * z / (1 + z + x * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_sum_equals_one_l1816_181691


namespace NUMINAMATH_CALUDE_bruce_books_purchased_l1816_181638

def bruce_purchase (num_books : ℕ) : Prop :=
  let crayon_cost : ℕ := 5 * 5
  let calculator_cost : ℕ := 3 * 5
  let total_cost : ℕ := crayon_cost + calculator_cost + num_books * 5
  let remaining_money : ℕ := 200 - total_cost
  remaining_money = 11 * 10

theorem bruce_books_purchased : ∃ (num_books : ℕ), bruce_purchase num_books ∧ num_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_bruce_books_purchased_l1816_181638


namespace NUMINAMATH_CALUDE_largest_divisor_of_cube_difference_l1816_181697

theorem largest_divisor_of_cube_difference (n : ℤ) (h : 5 ∣ n) :
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) → 
  (∃ (m : ℤ), m ∣ (n^3 - n) ∧ ∀ (k : ℤ), k ∣ (n^3 - n) → k ≤ m) ∧ m = 10 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_cube_difference_l1816_181697


namespace NUMINAMATH_CALUDE_oil_drilling_probability_l1816_181630

/-- The probability of drilling into an oil layer in a sea area -/
theorem oil_drilling_probability (total_area oil_area : ℝ) (h1 : total_area = 10000) (h2 : oil_area = 40) :
  oil_area / total_area = 0.004 := by
sorry

end NUMINAMATH_CALUDE_oil_drilling_probability_l1816_181630


namespace NUMINAMATH_CALUDE_equation_solution_l1816_181695

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 2 ∧
  (∀ x : ℝ, 2*x^2 - 4*x = 6 - 3*x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1816_181695


namespace NUMINAMATH_CALUDE_trapezoid_area_is_half_sq_dm_l1816_181634

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  smallBase : ℝ
  adjacentAngle : ℝ
  diagonalAngle : ℝ

/-- The area of a trapezoid with given measurements -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  0.5

/-- Theorem stating that a trapezoid with the given measurements has an area of 0.5 square decimeters -/
theorem trapezoid_area_is_half_sq_dm (t : Trapezoid) 
    (h1 : t.smallBase = 1)
    (h2 : t.adjacentAngle = 135 * π / 180)
    (h3 : t.diagonalAngle = 150 * π / 180) :
    trapezoidArea t = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_half_sq_dm_l1816_181634


namespace NUMINAMATH_CALUDE_multiple_of_twenty_day_after_power_of_three_l1816_181681

-- Part 1
theorem multiple_of_twenty (n : ℕ+) : ∃ k : ℤ, 4 * 6^n.val + 5^(n.val + 1) - 9 = 20 * k := by sorry

-- Part 2
theorem day_after_power_of_three : (3^100 % 7 : ℕ) + 1 = 5 := by sorry

end NUMINAMATH_CALUDE_multiple_of_twenty_day_after_power_of_three_l1816_181681


namespace NUMINAMATH_CALUDE_carl_watermelons_left_l1816_181673

/-- Calculates the number of watermelons left after a day of selling -/
def watermelons_left (price : ℕ) (profit : ℕ) (initial : ℕ) : ℕ :=
  initial - (profit / price)

/-- Theorem: Given the conditions, Carl has 18 watermelons left -/
theorem carl_watermelons_left :
  let price : ℕ := 3
  let profit : ℕ := 105
  let initial : ℕ := 53
  watermelons_left price profit initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_carl_watermelons_left_l1816_181673


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1816_181621

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := (x + Complex.I) / (y - Complex.I)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1816_181621


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_shared_foci_l1816_181635

-- Define the ellipse equation
def ellipse (x y a : ℝ) : Prop := x^2 / 6 + y^2 / a^2 = 1

-- Define the hyperbola equation
def hyperbola (x y a : ℝ) : Prop := x^2 / a - y^2 / 4 = 1

-- Define the property of shared foci
def shared_foci (a : ℝ) : Prop :=
  ∀ x y : ℝ, ellipse x y a ∧ hyperbola x y a → 
    (6 - a^2).sqrt = (a + 4).sqrt

-- Theorem statement
theorem ellipse_hyperbola_shared_foci :
  ∃ a : ℝ, a > 0 ∧ shared_foci a ∧ a = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_shared_foci_l1816_181635


namespace NUMINAMATH_CALUDE_range_of_a_l1816_181650

-- Define proposition p
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 
  0 ≤ a ∧ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1816_181650


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1816_181628

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1816_181628


namespace NUMINAMATH_CALUDE_exists_A_square_diff_two_l1816_181636

/-- The ceiling function -/
noncomputable def ceil (x : ℝ) : ℤ :=
  Int.floor x + 1

/-- Main theorem -/
theorem exists_A_square_diff_two :
  ∃ A : ℝ, ∀ n : ℕ, ∃ m : ℤ, |A^n - m^2| = 2 :=
sorry

end NUMINAMATH_CALUDE_exists_A_square_diff_two_l1816_181636


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1816_181670

/-- The inclination angle of a line given by the equation x*cos(140°) + y*sin(40°) + 1 = 0 is 50°. -/
theorem line_inclination_angle (x y : ℝ) :
  x * Real.cos (140 * π / 180) + y * Real.sin (40 * π / 180) + 1 = 0 →
  Real.arctan (Real.tan (50 * π / 180)) = 50 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1816_181670


namespace NUMINAMATH_CALUDE_only_C_not_like_terms_l1816_181611

-- Define a structure for a term
structure Term where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ
  m_exponent : ℕ
  n_exponent : ℕ
  deriving Repr

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_exponent = t2.x_exponent ∧
  t1.y_exponent = t2.y_exponent ∧
  t1.m_exponent = t2.m_exponent ∧
  t1.n_exponent = t2.n_exponent

-- Define the terms from the problem
def term_A1 : Term := ⟨-1, 2, 1, 0, 0⟩  -- -x²y
def term_A2 : Term := ⟨2, 2, 1, 0, 0⟩   -- 2yx²
def term_B1 : Term := ⟨2, 0, 0, 0, 0⟩   -- 2πR (treating π and R as constants)
def term_B2 : Term := ⟨1, 0, 0, 0, 0⟩   -- π²R (treating π and R as constants)
def term_C1 : Term := ⟨-1, 0, 0, 2, 1⟩  -- -m²n
def term_C2 : Term := ⟨1/2, 0, 0, 1, 2⟩ -- 1/2mn²
def term_D1 : Term := ⟨1, 0, 0, 0, 0⟩   -- 2³ (8)
def term_D2 : Term := ⟨1, 0, 0, 0, 0⟩   -- 3² (9)

-- Theorem stating that only pair C contains terms that are not like terms
theorem only_C_not_like_terms :
  are_like_terms term_A1 term_A2 ∧
  are_like_terms term_B1 term_B2 ∧
  ¬(are_like_terms term_C1 term_C2) ∧
  are_like_terms term_D1 term_D2 :=
sorry

end NUMINAMATH_CALUDE_only_C_not_like_terms_l1816_181611


namespace NUMINAMATH_CALUDE_factorial_calculation_l1816_181604

theorem factorial_calculation : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l1816_181604


namespace NUMINAMATH_CALUDE_rogers_wife_is_anne_l1816_181688

-- Define the set of people
inductive Person : Type
  | Henry | Peter | Louis | Roger | Elizabeth | Jeanne | Mary | Anne

-- Define the relationship of being married
def married : Person → Person → Prop := sorry

-- Define the action of dancing
def dancing : Person → Prop := sorry

-- Define the action of playing an instrument
def playing : Person → String → Prop := sorry

theorem rogers_wife_is_anne :
  -- Conditions
  (∀ p : Person, ∃! q : Person, married p q) →
  (∃ p : Person, married Person.Henry p ∧ dancing p ∧ 
    ∃ q : Person, married q Person.Elizabeth ∧ dancing q) →
  (¬ dancing Person.Roger) →
  (¬ dancing Person.Anne) →
  (playing Person.Peter "trumpet") →
  (playing Person.Mary "piano") →
  (¬ married Person.Anne Person.Peter) →
  -- Conclusion
  married Person.Roger Person.Anne :=
by sorry

end NUMINAMATH_CALUDE_rogers_wife_is_anne_l1816_181688


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1816_181674

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ), x^5 - 24*x^3 + 12*x^2 - x + 20 = (x - 3) * (x^4 + 3*x^3 - 15*x^2 - 33*x - 100) + (-280) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1816_181674


namespace NUMINAMATH_CALUDE_solution_implies_result_l1816_181682

theorem solution_implies_result (a b x y : ℝ) 
  (h1 : x = 1)
  (h2 : y = -2)
  (h3 : 2*a*x - 3*y = 10 - b)
  (h4 : a*x - b*y = -1) :
  (b - a)^3 = -125 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_result_l1816_181682


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1816_181644

/-- Given an arithmetic sequence with non-zero common difference,
    if a_5, a_9, and a_15 form a geometric sequence,
    then a_15 / a_9 = 3/2 -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_d_nonzero : d ≠ 0)
  (h_geom : (a 9) ^ 2 = (a 5) * (a 15)) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1816_181644


namespace NUMINAMATH_CALUDE_car_speed_adjustment_l1816_181658

/-- Given a car traveling a fixed distance D at 2 mph for T hours,
    prove that to cover the same distance in 5.0 hours, its speed S should be (2T)/5 mph. -/
theorem car_speed_adjustment (T : ℝ) (h : T > 0) : 
  let D := 2 * T  -- Distance covered at 2 mph for T hours
  let S := (2 * T) / 5  -- New speed to cover the same distance in 5 hours
  D = S * 5 := by sorry

end NUMINAMATH_CALUDE_car_speed_adjustment_l1816_181658


namespace NUMINAMATH_CALUDE_monkey_climbing_l1816_181693

/-- Monkey's tree climbing problem -/
theorem monkey_climbing (tree_height : ℝ) (hop_distance : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 21)
  (h2 : hop_distance = 3)
  (h3 : total_time = 19) :
  ∃ (slip_distance : ℝ), 
    slip_distance = 2 ∧ 
    (hop_distance - slip_distance) * (total_time - 1 : ℝ) + hop_distance = tree_height :=
by sorry

end NUMINAMATH_CALUDE_monkey_climbing_l1816_181693


namespace NUMINAMATH_CALUDE_right_triangle_area_l1816_181684

theorem right_triangle_area (longer_leg : ℝ) (angle : ℝ) :
  longer_leg = 10 →
  angle = 30 * (π / 180) →
  ∃ (area : ℝ), area = (50 * Real.sqrt 3) / 3 ∧
  area = (1 / 2) * longer_leg * (longer_leg / Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1816_181684


namespace NUMINAMATH_CALUDE_max_value_f_monotonic_condition_inequality_condition_l1816_181614

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := (-x^2 + 2*a*x) * Real.exp x
def g (x : ℝ) : ℝ := (x - 1) * Real.exp (2*x)

-- Theorem for part (I)
theorem max_value_f (a : ℝ) (h : a ≥ 0) :
  ∃ x : ℝ, x = a - 1 + Real.sqrt (a^2 + 1) ∨ x = a - 1 - Real.sqrt (a^2 + 1) ∧
  ∀ y : ℝ, f a y ≤ f a x :=
sorry

-- Theorem for part (II)
theorem monotonic_condition (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ↔ a ≥ 3/4 :=
sorry

-- Theorem for part (III)
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≤ g x) ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_value_f_monotonic_condition_inequality_condition_l1816_181614


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1816_181616

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0

def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, proposition_B a → proposition_A a) ∧
  (∃ a : ℝ, proposition_A a ∧ ¬proposition_B a) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1816_181616


namespace NUMINAMATH_CALUDE_coin_value_difference_l1816_181698

theorem coin_value_difference (p n d : ℕ) : 
  p + n + d = 3030 →
  p ≥ 1 →
  n ≥ 1 →
  d ≥ 1 →
  (∀ p' n' d' : ℕ, p' + n' + d' = 3030 ∧ p' ≥ 1 ∧ n' ≥ 1 ∧ d' ≥ 1 →
    p' + 5 * n' + 10 * d' ≤ 30286 ∧
    p' + 5 * n' + 10 * d' ≥ 3043) →
  30286 - 3043 = 27243 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1816_181698


namespace NUMINAMATH_CALUDE_luca_pizza_ingredients_l1816_181643

/-- Calculates the required amount of milk and oil for a given amount of flour in Luca's pizza dough recipe. -/
def pizza_ingredients (flour : ℚ) : ℚ × ℚ :=
  let milk_ratio : ℚ := 70 / 350
  let oil_ratio : ℚ := 30 / 350
  (flour * milk_ratio, flour * oil_ratio)

/-- Proves that for 1050 mL of flour, Luca needs 210 mL of milk and 90 mL of oil. -/
theorem luca_pizza_ingredients : pizza_ingredients 1050 = (210, 90) := by
  sorry

end NUMINAMATH_CALUDE_luca_pizza_ingredients_l1816_181643


namespace NUMINAMATH_CALUDE_power_inequality_l1816_181677

theorem power_inequality (n : ℕ) (h : n > 2) : n^(n+1) > (n+1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1816_181677


namespace NUMINAMATH_CALUDE_customers_left_l1816_181676

theorem customers_left (initial : Nat) (remaining : Nat) : initial - remaining = 11 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_customers_left_l1816_181676


namespace NUMINAMATH_CALUDE_complement_of_A_l1816_181651

def A : Set ℝ := {x | |x - 1| ≤ 2}

theorem complement_of_A :
  Aᶜ = {x : ℝ | x < -1 ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1816_181651


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_difference_l1816_181699

/-- Given two inverse proportion functions and points satisfying certain conditions, 
    prove that the difference of their constants is 4. -/
theorem inverse_proportion_constant_difference 
  (k₁ k₂ : ℝ) 
  (f₁ : ℝ → ℝ) 
  (f₂ : ℝ → ℝ) 
  (a b : ℝ) 
  (h₁ : ∀ x, f₁ x = k₁ / x) 
  (h₂ : ∀ x, f₂ x = k₂ / x) 
  (h₃ : |f₁ a - f₂ a| = 2) 
  (h₄ : |f₂ b - f₁ b| = 3) 
  (h₅ : |b - a| = 10/3) : 
  k₂ - k₁ = 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_difference_l1816_181699


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1816_181625

theorem subtraction_multiplication_equality : ((3.54 - 1.32) * 2) = 4.44 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1816_181625


namespace NUMINAMATH_CALUDE_inequality_proof_l1816_181641

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1816_181641


namespace NUMINAMATH_CALUDE_sales_tax_difference_l1816_181666

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) 
  (h1 : price = 50)
  (h2 : tax_rate1 = 0.0725)
  (h3 : tax_rate2 = 0.0675) : 
  (tax_rate1 - tax_rate2) * price = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l1816_181666


namespace NUMINAMATH_CALUDE_bus_capacity_l1816_181605

/-- The number of rows in a bus -/
def rows : ℕ := 9

/-- The number of children that can be accommodated in each row -/
def children_per_row : ℕ := 4

/-- The total number of children a bus can accommodate -/
def total_children : ℕ := rows * children_per_row

theorem bus_capacity : total_children = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l1816_181605


namespace NUMINAMATH_CALUDE_relationship_abc_l1816_181631

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 0.6 → 
  b = Real.rpow 0.6 (1/3) → 
  c = Real.log 3 / Real.log 0.6 → 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1816_181631


namespace NUMINAMATH_CALUDE_divisible_by_42_l1816_181622

theorem divisible_by_42 (n : ℕ) : ∃ k : ℤ, (n ^ 3 * (n ^ 6 - 1) : ℤ) = 42 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_42_l1816_181622


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l1816_181660

/-- Proves that the original sales tax percentage was 3.5% given the conditions -/
theorem original_sales_tax_percentage
  (new_tax_rate : ℚ)
  (market_price : ℚ)
  (tax_difference : ℚ)
  (h1 : new_tax_rate = 10 / 3)
  (h2 : market_price = 6600)
  (h3 : tax_difference = 10.999999999999991)
  : ∃ (original_tax_rate : ℚ), original_tax_rate = 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l1816_181660


namespace NUMINAMATH_CALUDE_cube_dimensions_l1816_181600

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the side length and diagonal of the cube
theorem cube_dimensions (s d : ℝ) : 
  (6 * s^2 = surface_area) → 
  (s = 12) ∧ 
  (d = 12 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_cube_dimensions_l1816_181600


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_binomials_l1816_181686

theorem infinitely_many_coprime_binomials (k l : ℕ+) :
  ∃ (S : Set ℕ), (∀ (m : ℕ), m ∈ S → m ≥ k) ∧
                 (Set.Infinite S) ∧
                 (∀ (m : ℕ), m ∈ S → Nat.gcd (Nat.choose m k) l = 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_binomials_l1816_181686


namespace NUMINAMATH_CALUDE_no_large_squares_in_H_l1816_181624

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 6 and 2 ≤ |y| ≤ 6 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 6 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 6}

/-- A square with side length at least 8 -/
def IsValidSquare (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (a b c d : ℤ × ℤ), s = {a, b, c, d} ∧
  (a.1 - b.1)^2 + (a.2 - b.2)^2 ≥ 64 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 ≥ 64 ∧
  (c.1 - d.1)^2 + (c.2 - d.2)^2 ≥ 64 ∧
  (d.1 - a.1)^2 + (d.2 - a.2)^2 ≥ 64

theorem no_large_squares_in_H :
  ¬∃ s : Set (ℤ × ℤ), (∀ p ∈ s, p ∈ H) ∧ IsValidSquare s := by
  sorry

end NUMINAMATH_CALUDE_no_large_squares_in_H_l1816_181624


namespace NUMINAMATH_CALUDE_triangle_angle_A_l1816_181661

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_A (t : Triangle) : 
  t.a = 4 * Real.sqrt 3 → 
  t.c = 12 → 
  t.C = π / 3 → 
  t.A = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_A_l1816_181661


namespace NUMINAMATH_CALUDE_hexahedron_faces_l1816_181602

/-- A hexahedron is a polyhedron with six faces -/
structure Hexahedron where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces of a hexahedron -/
def num_faces (h : Hexahedron) : ℕ := sorry

/-- Theorem: The number of faces of a hexahedron is 6 -/
theorem hexahedron_faces (h : Hexahedron) : num_faces h = 6 := by
  sorry

end NUMINAMATH_CALUDE_hexahedron_faces_l1816_181602


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1816_181694

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n < 900 → is_divisible_by_digits n → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l1816_181694


namespace NUMINAMATH_CALUDE_systematic_sampling_elimination_l1816_181606

theorem systematic_sampling_elimination (population : Nat) (sample_size : Nat) 
    (h1 : population = 1252) 
    (h2 : sample_size = 50) : 
  population % sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_elimination_l1816_181606


namespace NUMINAMATH_CALUDE_no_integer_solution_for_book_club_l1816_181692

theorem no_integer_solution_for_book_club : 
  ¬ ∃ (x : ℤ), x + (2 * x + 33) + (3 * x - 24) = 100 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_book_club_l1816_181692


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l1816_181669

theorem combined_tennis_percentage :
  let north_students : ℕ := 1800
  let south_students : ℕ := 2200
  let north_tennis_percentage : ℚ := 25 / 100
  let south_tennis_percentage : ℚ := 35 / 100
  let total_students := north_students + south_students
  let north_tennis_students := (north_students : ℚ) * north_tennis_percentage
  let south_tennis_students := (south_students : ℚ) * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let combined_percentage := total_tennis_students / (total_students : ℚ) * 100
  ⌊combined_percentage⌋ = 31 := by sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l1816_181669


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1816_181662

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 → 
  n ∈ Finset.range 1982 → 
  (n^2 - m*n - m^2)^2 = 1 → 
  m^2 + n^2 ≤ 3524578 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1816_181662


namespace NUMINAMATH_CALUDE_clowns_in_mobiles_l1816_181656

/-- Given a number of clown mobiles and a total number of clowns,
    calculate the number of clowns in each mobile assuming even distribution -/
def clowns_per_mobile (num_mobiles : ℕ) (total_clowns : ℕ) : ℕ :=
  total_clowns / num_mobiles

/-- Theorem stating that with 5 clown mobiles and 140 clowns in total,
    there are 28 clowns in each mobile -/
theorem clowns_in_mobiles :
  clowns_per_mobile 5 140 = 28 := by
  sorry


end NUMINAMATH_CALUDE_clowns_in_mobiles_l1816_181656


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1816_181615

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 1)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1816_181615
