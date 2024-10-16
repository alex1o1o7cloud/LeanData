import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l1333_133312

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (3^3) = 2^2 % (3^3))
  (h2 : (6 + x) % (5^3) = 3^2 % (5^3))
  (h3 : (8 + x) % (7^3) = 5^2 % (7^3)) :
  x % 105 = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1333_133312


namespace NUMINAMATH_CALUDE_inequality_theorem_l1333_133327

-- Define the inequality condition
def inequality_condition (x : ℝ) : Prop :=
  |x^2 - 2*x - 8| ≤ 2 * |x - 4| * |x + 2|

-- Define the second condition for x > 1
def second_condition (x m : ℝ) : Prop :=
  x > 1 → x^2 - 2*x - 8 ≥ (m + 2)*x - m - 15

-- Theorem statement
theorem inequality_theorem :
  (∀ x : ℝ, inequality_condition x) ∧
  (∀ m : ℝ, (∀ x : ℝ, second_condition x m) → m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1333_133327


namespace NUMINAMATH_CALUDE_expand_product_l1333_133302

theorem expand_product (x : ℝ) : (3*x + 4) * (2*x + 7) = 6*x^2 + 29*x + 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1333_133302


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1333_133316

theorem quadratic_equation_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + m*y - 2 = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1333_133316


namespace NUMINAMATH_CALUDE_max_min_difference_c_l1333_133354

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 12) :
  ∃ (c_max c_min : ℝ),
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c' ≤ c_max) ∧
    (∀ c', (∃ a' b', a' + b' + c' = 2 ∧ a'^2 + b'^2 + c'^2 = 12) → c_min ≤ c') ∧
    c_max - c_min = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l1333_133354


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1333_133380

theorem at_least_one_leq_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_two_l1333_133380


namespace NUMINAMATH_CALUDE_f_sum_constant_l1333_133317

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem f_sum_constant (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_constant_l1333_133317


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1333_133318

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The perimeter of the trapezoid -/
  perimeter : ℝ
  /-- The diagonal bisects the obtuse angle -/
  diagonalBisectsObtuseAngle : Prop

/-- The area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific isosceles trapezoid is 96 -/
theorem area_of_specific_trapezoid :
  ∀ (t : IsoscelesTrapezoid),
    t.smallerBase = 3 ∧
    t.perimeter = 42 ∧
    t.diagonalBisectsObtuseAngle →
    areaOfTrapezoid t = 96 :=
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1333_133318


namespace NUMINAMATH_CALUDE_event_probability_l1333_133329

theorem event_probability (P_B P_AB P_AorB : ℝ) 
  (hB : P_B = 0.4)
  (hAB : P_AB = 0.25)
  (hAorB : P_AorB = 0.6) :
  ∃ P_A : ℝ, P_A = 0.45 ∧ P_AorB = P_A + P_B - P_AB :=
sorry

end NUMINAMATH_CALUDE_event_probability_l1333_133329


namespace NUMINAMATH_CALUDE_inverse_f_composition_l1333_133315

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

theorem inverse_f_composition : 
  ∃ (f_inv : ℤ → ℤ), 
    (∀ (y : ℤ), f (f_inv y) = y) ∧ 
    (∀ (x : ℤ), f_inv (f x) = x) ∧
    f_inv (f_inv 122 / f_inv 18 + f_inv 50) = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_composition_l1333_133315


namespace NUMINAMATH_CALUDE_percentage_of_blue_flowers_l1333_133300

/-- Given a set of flowers with specific colors, calculate the percentage of blue flowers. -/
theorem percentage_of_blue_flowers (total : ℕ) (red : ℕ) (white : ℕ) (h1 : total = 10) (h2 : red = 4) (h3 : white = 2) :
  (total - red - white) / total * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_blue_flowers_l1333_133300


namespace NUMINAMATH_CALUDE_f_of_g_of_3_l1333_133319

/-- Given two functions f and g, prove that f(g(3)) = 97 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 1) 
  (hg : ∀ x, g x = x + 3) : 
  f (g 3) = 97 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_l1333_133319


namespace NUMINAMATH_CALUDE_lilith_cap_collection_l1333_133364

/-- Calculates the total number of caps Lilith has collected over 5 years -/
def total_caps_collected : ℕ :=
  let caps_first_year := 3 * 12
  let caps_after_first_year := 5 * 12 * 4
  let caps_from_christmas := 40 * 5
  let caps_lost := 15 * 5
  caps_first_year + caps_after_first_year + caps_from_christmas - caps_lost

/-- Theorem stating that the total number of caps Lilith has collected is 401 -/
theorem lilith_cap_collection : total_caps_collected = 401 := by
  sorry

end NUMINAMATH_CALUDE_lilith_cap_collection_l1333_133364


namespace NUMINAMATH_CALUDE_orlans_rope_problem_l1333_133347

theorem orlans_rope_problem (total_length : ℝ) (allan_portion : ℝ) (jack_portion : ℝ) (remaining : ℝ) :
  total_length = 20 →
  jack_portion = (2/3) * (total_length - allan_portion) →
  remaining = 5 →
  total_length = allan_portion + jack_portion + remaining →
  allan_portion / total_length = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_orlans_rope_problem_l1333_133347


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_630_l1333_133310

theorem sin_n_equals_cos_630 (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  (Real.sin (n * π / 180) = Real.cos (630 * π / 180) ↔ n = 0 ∨ n = 180 ∨ n = -180) :=
by sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_630_l1333_133310


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1333_133374

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r)^2 = 240 → 
  P * (1 + r) = 217.68707482993196 → 
  r = 0.1025 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1333_133374


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1333_133396

def C : Set Nat := {37, 39, 42, 43, 47}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → Nat.minFac n ≤ Nat.minFac m) ∧ n = 42 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1333_133396


namespace NUMINAMATH_CALUDE_point_on_line_equal_intercepts_l1333_133303

/-- A line passing through (-2, -3) with equal x and y intercepts -/
def line_with_equal_intercepts (x y : ℝ) : Prop :=
  x + y = 5

/-- The point (-2, -3) lies on the line -/
theorem point_on_line : line_with_equal_intercepts (-2) (-3) := by sorry

/-- The line has equal intercepts on x and y axes -/
theorem equal_intercepts :
  ∃ a : ℝ, a > 0 ∧ line_with_equal_intercepts a 0 ∧ line_with_equal_intercepts 0 a := by sorry

end NUMINAMATH_CALUDE_point_on_line_equal_intercepts_l1333_133303


namespace NUMINAMATH_CALUDE_cow_fraction_sold_l1333_133397

/-- Represents the number of animals on a petting farm. -/
structure PettingFarm where
  cows : ℕ
  dogs : ℕ

/-- Represents the state of the petting farm before and after selling animals. -/
structure FarmState where
  initial : PettingFarm
  final : PettingFarm

theorem cow_fraction_sold (farm : FarmState) : 
  farm.initial.cows = 184 →
  farm.initial.cows = 2 * farm.initial.dogs →
  farm.final.dogs = farm.initial.dogs / 4 →
  farm.final.cows + farm.final.dogs = 161 →
  (farm.initial.cows - farm.final.cows) / farm.initial.cows = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cow_fraction_sold_l1333_133397


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_l1333_133375

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2*x + 2

theorem tangent_line_at_point_one :
  (f' 1 = 4) ∧
  (∀ x y : ℝ, y = f 1 → (4*x - y - 3 = 0 ↔ y - f 1 = f' 1 * (x - 1))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_l1333_133375


namespace NUMINAMATH_CALUDE_square_value_l1333_133389

theorem square_value (x : ℚ) : 
  10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1 → x = 28 := by
sorry

end NUMINAMATH_CALUDE_square_value_l1333_133389


namespace NUMINAMATH_CALUDE_expression_equals_one_l1333_133306

theorem expression_equals_one (x : ℝ) 
  (h1 : x^4 + 2*x + 2 ≠ 0) 
  (h2 : x^4 - 2*x + 2 ≠ 0) : 
  ((((x+2)^3 * (x^3-2*x+2)^3) / (x^4+2*x+2)^3)^3 * 
   (((x-2)^3 * (x^3+2*x+2)^3) / (x^4-2*x+2)^3)^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1333_133306


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l1333_133393

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l1333_133393


namespace NUMINAMATH_CALUDE_total_students_l1333_133377

/-- Represents the setup of students in lines -/
structure StudentLines where
  total_lines : ℕ
  students_per_line : ℕ
  left_position : ℕ
  right_position : ℕ

/-- Theorem stating the total number of students given the conditions -/
theorem total_students (setup : StudentLines) 
  (h1 : setup.total_lines = 5)
  (h2 : setup.left_position = 4)
  (h3 : setup.right_position = 9)
  (h4 : setup.students_per_line = setup.left_position + setup.right_position - 1) :
  setup.total_lines * setup.students_per_line = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1333_133377


namespace NUMINAMATH_CALUDE_bag_of_balls_l1333_133309

theorem bag_of_balls (total : ℕ) (blue : ℕ) (green : ℕ) : 
  blue = 6 →
  blue + green = total →
  (blue : ℚ) / total = 1 / 4 →
  green = 18 := by
sorry

end NUMINAMATH_CALUDE_bag_of_balls_l1333_133309


namespace NUMINAMATH_CALUDE_intersection_A_B_l1333_133378

-- Define set A
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1333_133378


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1333_133381

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 5 + a 8 = 15 → a 3 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1333_133381


namespace NUMINAMATH_CALUDE_side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l1333_133328

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  t.b * Real.cos t.A + t.a * Real.cos t.B = t.a * t.c

-- Theorem for part 1
theorem side_a_is_one (t : Triangle) (h : satisfiesCondition t) : t.a = 1 := by
  sorry

-- Theorem for part 2
theorem max_perimeter_is_three (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  t.a + t.b + t.c ≤ 3 := by
  sorry

-- Theorem for the maximum perimeter occurring when b = c
theorem max_perimeter_when_b_equals_c (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.A = Real.pi / 3) :
  ∃ (t' : Triangle), satisfiesCondition t' ∧ t'.A = Real.pi / 3 ∧ t'.b = t'.c ∧ t'.a + t'.b + t'.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_side_a_is_one_max_perimeter_is_three_max_perimeter_when_b_equals_c_l1333_133328


namespace NUMINAMATH_CALUDE_paper_strip_sequence_l1333_133311

theorem paper_strip_sequence : ∃ (a : Fin 10 → ℝ), 
  a 0 = 9 ∧ 
  a 8 = 5 ∧ 
  ∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_paper_strip_sequence_l1333_133311


namespace NUMINAMATH_CALUDE_xiao_ding_distance_to_school_l1333_133348

/-- Proof that Xiao Ding's distance to school is 60 meters -/
theorem xiao_ding_distance_to_school : 
  ∀ (xw xd xc xz : ℝ),
  xw + xd + xc + xz = 705 →  -- Total distance condition
  xw = 4 * xd →              -- Xiao Wang's distance condition
  xc = xw / 2 + 20 →         -- Xiao Chen's distance condition
  xz = 2 * xc - 15 →         -- Xiao Zhang's distance condition
  xd = 60 := by              -- Conclusion: Xiao Ding's distance is 60 meters
sorry

end NUMINAMATH_CALUDE_xiao_ding_distance_to_school_l1333_133348


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l1333_133341

theorem quadratic_equivalence :
  ∀ x y : ℝ, y = x^2 + 2*x + 4 ↔ y = (x + 1)^2 + 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l1333_133341


namespace NUMINAMATH_CALUDE_season_games_calculation_l1333_133326

/-- Represents the number of games played by a team in a season -/
def total_games : ℕ := 125

/-- Represents the number of games in the first part of the season -/
def first_games : ℕ := 100

/-- Represents the win percentage for the first part of the season -/
def first_win_percentage : ℚ := 75 / 100

/-- Represents the win percentage for the remaining games -/
def remaining_win_percentage : ℚ := 50 / 100

/-- Represents the overall win percentage for the entire season -/
def overall_win_percentage : ℚ := 70 / 100

theorem season_games_calculation :
  let remaining_games := total_games - first_games
  (first_win_percentage * first_games + remaining_win_percentage * remaining_games) / total_games = overall_win_percentage :=
by sorry

end NUMINAMATH_CALUDE_season_games_calculation_l1333_133326


namespace NUMINAMATH_CALUDE_modified_mindmaster_codes_l1333_133330

/-- The number of different colors available for pegs -/
def num_colors : ℕ := 6

/-- The number of slots in the secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the modified Mindmaster game -/
def num_secret_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the number of possible secret codes is 7776 -/
theorem modified_mindmaster_codes : num_secret_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_modified_mindmaster_codes_l1333_133330


namespace NUMINAMATH_CALUDE_range_of_a_l1333_133333

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Iic 6, StrictMonoOn (f a) (Set.Iic x)) →
  a ∈ Set.Iic (-5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1333_133333


namespace NUMINAMATH_CALUDE_caterer_order_underdetermined_l1333_133353

/-- Represents the caterer's order problem -/
def CatererOrder (x y z m : ℝ) : Prop :=
  0.60 * x + 1.20 * y + m * z = 425 ∧ x + y + z = 350

/-- The caterer's order problem is underdetermined -/
theorem caterer_order_underdetermined :
  ∃ (x₁ y₁ z₁ m₁ x₂ y₂ z₂ m₂ : ℝ),
    x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∨ z₁ ≠ z₂ ∨ m₁ ≠ m₂ ∧
    CatererOrder x₁ y₁ z₁ m₁ ∧
    CatererOrder x₂ y₂ z₂ m₂ :=
by
  sorry

#check caterer_order_underdetermined

end NUMINAMATH_CALUDE_caterer_order_underdetermined_l1333_133353


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l1333_133340

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-3, 4)

theorem reflection_across_y_axis :
  reflect_y P = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l1333_133340


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_x_in_open_zero_one_l1333_133379

theorem arccos_lt_arcsin_iff_x_in_open_zero_one (x : ℝ) :
  x ∈ Set.Icc (-1) 1 →
  (Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_x_in_open_zero_one_l1333_133379


namespace NUMINAMATH_CALUDE_product_base_conversion_l1333_133370

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat := sorry

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat := sorry

theorem product_base_conversion :
  let binary := [true, true, false, true]  -- 1101 in base 2
  let ternary := [2, 0, 2]  -- 202 in base 3
  (base2To10 binary) * (base3To10 ternary) = 260 := by sorry

end NUMINAMATH_CALUDE_product_base_conversion_l1333_133370


namespace NUMINAMATH_CALUDE_quadratic_and_optimization_l1333_133361

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint equation
def constraint (x y : ℝ) : Prop :=
  (1 / (x + 1)) + (2 / (y + 1)) = 1

-- Define the objective function
def objective (x y : ℝ) : ℝ := 2 * x + y + 3

-- State the theorem
theorem quadratic_and_optimization :
  ∃ a b : ℝ,
    solution_set a b ∧
    (a = 1 ∧ b = 2) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → constraint x y →
      objective x y ≥ 8 ∧
      ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ constraint x₀ y₀ ∧ objective x₀ y₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_and_optimization_l1333_133361


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l1333_133336

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for (¬ᵤA) ∩ (¬ᵤB)
theorem intersection_complement_A_B : (Aᶜ) ∩ (Bᶜ) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_l1333_133336


namespace NUMINAMATH_CALUDE_triangle_point_movement_l1333_133308

theorem triangle_point_movement (AB BC : ℝ) (v_P v_Q : ℝ) (area_PBQ : ℝ) : 
  AB = 6 →
  BC = 8 →
  v_P = 1 →
  v_Q = 2 →
  area_PBQ = 5 →
  ∃ t : ℝ, t = 1 ∧ 
    (1/2) * (AB - t * v_P) * (t * v_Q) = area_PBQ ∧
    t * v_Q ≤ BC :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_movement_l1333_133308


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l1333_133314

/-- Given two vectors a and b in R², where a = (3, 2) and b = (m, -1),
    if a and b are orthogonal, then m = 2/3 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (m, -1)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l1333_133314


namespace NUMINAMATH_CALUDE_product_of_solutions_l1333_133366

theorem product_of_solutions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1905) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1910)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1905) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1910)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1905) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1910) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/191 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1333_133366


namespace NUMINAMATH_CALUDE_prob_both_blue_is_25_64_l1333_133331

/-- Represents the contents of a jar --/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar --/
def prob_blue (jar : JarContents) : ℚ :=
  jar.blue / (jar.red + jar.blue)

/-- The initial contents of Jar C --/
def initial_jar_c : JarContents :=
  { red := 6, blue := 10 }

/-- The number of buttons removed from Jar C --/
def removed : JarContents :=
  { red := 3, blue := 5 }

/-- The contents of Jar C after removal --/
def final_jar_c : JarContents :=
  { red := initial_jar_c.red - removed.red,
    blue := initial_jar_c.blue - removed.blue }

/-- The contents of Jar D after removal --/
def jar_d : JarContents := removed

theorem prob_both_blue_is_25_64 :
  (prob_blue final_jar_c * prob_blue jar_d = 25 / 64) ∧
  (final_jar_c.red + final_jar_c.blue = (initial_jar_c.red + initial_jar_c.blue) / 2) :=
sorry

end NUMINAMATH_CALUDE_prob_both_blue_is_25_64_l1333_133331


namespace NUMINAMATH_CALUDE_junk_food_ratio_l1333_133349

theorem junk_food_ratio (weekly_allowance sweets_cost savings : ℚ)
  (h1 : weekly_allowance = 30)
  (h2 : sweets_cost = 8)
  (h3 : savings = 12)
  (h4 : weekly_allowance = sweets_cost + savings + (weekly_allowance - sweets_cost - savings)) :
  (weekly_allowance - sweets_cost - savings) / weekly_allowance = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_junk_food_ratio_l1333_133349


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1333_133384

/-- Calculate the number of games in a chess tournament -/
theorem chess_tournament_games (n : ℕ) (h : n = 20) : n * (n - 1) = 760 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l1333_133384


namespace NUMINAMATH_CALUDE_robins_hair_length_l1333_133360

theorem robins_hair_length (initial_length cut_length growth_length final_length : ℕ) :
  cut_length = 11 →
  growth_length = 12 →
  final_length = 17 →
  final_length = initial_length - cut_length + growth_length →
  initial_length = 16 :=
by sorry

end NUMINAMATH_CALUDE_robins_hair_length_l1333_133360


namespace NUMINAMATH_CALUDE_ice_cream_fraction_l1333_133352

theorem ice_cream_fraction (initial_amount : ℚ) (lunch_cost : ℚ) (ice_cream_cost : ℚ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_cost = 5 →
  ice_cream_cost / (initial_amount - lunch_cost) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_fraction_l1333_133352


namespace NUMINAMATH_CALUDE_calculation_proof_l1333_133392

theorem calculation_proof :
  ((-1/4 + 5/6 - 2/9) * (-36) = -13) ∧
  (-1^4 - 1/6 - (3 + (-3)^2) / (-1 - 1/2) = 6 + 5/6) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1333_133392


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1333_133335

theorem condition_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 1 → x^2 + y^2 ≥ 2) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 2 ∧ ¬(x ≥ 1 ∧ y ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1333_133335


namespace NUMINAMATH_CALUDE_abby_and_damon_weight_l1333_133363

theorem abby_and_damon_weight (a b c d : ℝ)
  (h1 : a + b = 265)
  (h2 : b + c = 250)
  (h3 : c + d = 280) :
  a + d = 295 := by
  sorry

end NUMINAMATH_CALUDE_abby_and_damon_weight_l1333_133363


namespace NUMINAMATH_CALUDE_determinant_cubic_roots_l1333_133385

theorem determinant_cubic_roots (p q k : ℝ) (a b c : ℝ) : 
  a^3 + p*a + q = 0 → 
  b^3 + p*b + q = 0 → 
  c^3 + p*c + q = 0 → 
  Matrix.det !![k + a, 1, 1; 1, k + b, 1; 1, 1, k + c] = k^3 + k*p - q := by
  sorry

end NUMINAMATH_CALUDE_determinant_cubic_roots_l1333_133385


namespace NUMINAMATH_CALUDE_dessert_combinations_eq_twelve_l1333_133398

/-- The number of dessert options available -/
def num_desserts : ℕ := 4

/-- The number of courses in the meal -/
def num_courses : ℕ := 2

/-- Function to calculate the number of ways to order the dessert -/
def dessert_combinations : ℕ := num_desserts * (num_desserts - 1)

/-- Theorem stating that the number of ways to order the dessert is 12 -/
theorem dessert_combinations_eq_twelve : dessert_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_dessert_combinations_eq_twelve_l1333_133398


namespace NUMINAMATH_CALUDE_sam_recycling_cans_l1333_133372

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := 3

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 9

/-- The total number of cans Sam picked up -/
def total_cans : ℕ := (saturday_bags + sunday_bags) * cans_per_bag

theorem sam_recycling_cans : total_cans = 63 := by
  sorry

end NUMINAMATH_CALUDE_sam_recycling_cans_l1333_133372


namespace NUMINAMATH_CALUDE_trip_cost_difference_l1333_133321

def trip_cost_sharing (alice_paid bob_paid charlie_paid dex_paid : ℚ) : ℚ :=
  let total_paid := alice_paid + bob_paid + charlie_paid + dex_paid
  let fair_share := total_paid / 4
  let alice_owes := max (fair_share - alice_paid) 0
  let charlie_owes := max (fair_share - charlie_paid) 0
  let bob_receives := max (bob_paid - fair_share) 0
  min alice_owes bob_receives - min charlie_owes (bob_receives - min alice_owes bob_receives)

theorem trip_cost_difference :
  trip_cost_sharing 160 220 190 95 = -35/2 :=
by sorry

end NUMINAMATH_CALUDE_trip_cost_difference_l1333_133321


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l1333_133382

def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g :
  ∃ (r : ℝ), r = -Real.sqrt (3/7) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → |x| ≥ |r| :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l1333_133382


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1333_133301

theorem inequality_solution_set :
  ∀ x : ℝ, (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1333_133301


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1333_133399

theorem container_volume_ratio : 
  ∀ (v1 v2 v3 : ℝ), 
    v1 > 0 → v2 > 0 → v3 > 0 →
    (2/3 : ℝ) * v1 = (1/2 : ℝ) * v2 →
    (1/2 : ℝ) * v2 = (3/5 : ℝ) * v3 →
    v1 / v3 = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1333_133399


namespace NUMINAMATH_CALUDE_event_probability_l1333_133344

theorem event_probability (P_A_and_B P_A_or_B P_B : ℝ) 
  (h1 : P_A_and_B = 0.25)
  (h2 : P_A_or_B = 0.8)
  (h3 : P_B = 0.65) :
  ∃ P_A : ℝ, P_A = 0.4 ∧ P_A_or_B = P_A + P_B - P_A_and_B := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l1333_133344


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l1333_133324

theorem digit_sum_puzzle :
  ∀ (B E D H : ℕ),
    B < 10 → E < 10 → D < 10 → H < 10 →
    B ≠ E → B ≠ D → B ≠ H → E ≠ D → E ≠ H → D ≠ H →
    (10 * B + E) * (10 * D + E) = 111 * H →
    E + B + D + H = 17 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l1333_133324


namespace NUMINAMATH_CALUDE_B_elements_l1333_133304

def A : Set ℤ := {-1, 0, 1}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_elements : B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_B_elements_l1333_133304


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_cube_l1333_133320

theorem smallest_sum_of_squares_cube (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x ≠ y → y ≠ z → x ≠ z →
  x^2 + y^2 = z^3 →
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → 
    a ≠ b → b ≠ c → a ≠ c →
    a^2 + b^2 = c^3 →
    x + y + z ≤ a + b + c →
  x + y + z = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_cube_l1333_133320


namespace NUMINAMATH_CALUDE_cos_210_degrees_l1333_133332

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l1333_133332


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1333_133334

def initial_purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 500
def price_increase : ℝ → ℝ := λ x => x
def sales_volume : ℝ → ℝ := λ x => initial_sales_volume - 10 * price_increase x
def selling_price : ℝ → ℝ := λ x => initial_selling_price + price_increase x
def profit : ℝ → ℝ := λ x => (selling_price x * sales_volume x) - (initial_purchase_price * sales_volume x)

theorem optimal_selling_price :
  ∃ x : ℝ, (∀ y : ℝ, profit y ≤ profit x) ∧ selling_price x = 70 := by
  sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l1333_133334


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1333_133351

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a and b, 
    and f(x) > 0 for all real x, with f(1) = 1/3 -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ a b : ℝ, f (a + b) = f a * f b) ∧ 
  (∀ x : ℝ, f x > 0) ∧
  (f 1 = 1/3)

/-- If f satisfies the functional equation, then f(-2) = 9 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : f (-2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1333_133351


namespace NUMINAMATH_CALUDE_flea_treatment_ratio_l1333_133394

theorem flea_treatment_ratio (F : ℕ) (p : ℚ) : 
  F - 14 = 210 → 
  F * (1 - p)^4 = 14 → 
  ∃ (n : ℕ), n * (F * p) = F ∧ n = 448 := by
sorry

end NUMINAMATH_CALUDE_flea_treatment_ratio_l1333_133394


namespace NUMINAMATH_CALUDE_random_walk_2d_properties_l1333_133388

-- Define the random walk on a 2D grid
def RandomWalk2D := ℕ × ℕ → ℝ

-- Probability of reaching a specific x-coordinate
def prob_reach_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Expected y-coordinate when reaching a specific x-coordinate
def expected_y_at_x (walk : RandomWalk2D) (x : ℕ) : ℝ := sorry

-- Theorem statement
theorem random_walk_2d_properties (walk : RandomWalk2D) :
  (∀ x : ℕ, prob_reach_x walk x = 1) ∧
  (∀ n : ℕ, expected_y_at_x walk n = n) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_2d_properties_l1333_133388


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l1333_133339

/-- Sine of 30 degrees is 1/2 -/
theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l1333_133339


namespace NUMINAMATH_CALUDE_arc_length_300_degrees_l1333_133368

/-- The length of an arc with radius 2 and central angle 300° is 10π/3 -/
theorem arc_length_300_degrees (r : Real) (θ : Real) : 
  r = 2 → θ = 300 * Real.pi / 180 → r * θ = 10 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_300_degrees_l1333_133368


namespace NUMINAMATH_CALUDE_polynomial_root_behavior_l1333_133367

def Q (x : ℝ) : ℝ := x^6 - 6*x^5 + 10*x^4 - x^3 - x + 12

theorem polynomial_root_behavior :
  (∀ x < 0, Q x ≠ 0) ∧ (∃ x > 0, Q x = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_behavior_l1333_133367


namespace NUMINAMATH_CALUDE_stu_book_count_l1333_133359

theorem stu_book_count (stu_books : ℕ) (albert_books : ℕ) : 
  albert_books = 4 * stu_books →
  stu_books + albert_books = 45 →
  stu_books = 9 := by
sorry

end NUMINAMATH_CALUDE_stu_book_count_l1333_133359


namespace NUMINAMATH_CALUDE_ladder_slide_approx_l1333_133395

noncomputable def ladder_slide (ladder_length : Real) (initial_distance : Real) (slip_distance : Real) : Real :=
  let initial_height := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height := initial_height - slip_distance
  let new_distance := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance

theorem ladder_slide_approx :
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |ladder_slide 30 11 5 - 3.7| < ε :=
sorry

end NUMINAMATH_CALUDE_ladder_slide_approx_l1333_133395


namespace NUMINAMATH_CALUDE_todd_total_gum_l1333_133356

-- Define the initial number of gum pieces Todd had
def initial_gum : ℕ := 38

-- Define the number of gum pieces Steve gave to Todd
def steve_gum : ℕ := 16

-- Theorem statement
theorem todd_total_gum : initial_gum + steve_gum = 54 := by
  sorry

end NUMINAMATH_CALUDE_todd_total_gum_l1333_133356


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1333_133345

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

theorem sufficient_not_necessary_condition (m : ℝ) :
  (m < 1 → ∃ x, f m x = 0) ∧
  ¬(∀ m, (∃ x, f m x = 0) → m < 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1333_133345


namespace NUMINAMATH_CALUDE_no_consecutive_fourth_powers_l1333_133322

theorem no_consecutive_fourth_powers (n : ℤ) : 
  n^4 + (n+1)^4 + (n+2)^4 + (n+3)^4 ≠ (n+4)^4 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_fourth_powers_l1333_133322


namespace NUMINAMATH_CALUDE_circle_radius_isosceles_right_triangle_l1333_133325

/-- The radius of a circle tangent to both axes and the hypotenuse of an isosceles right triangle -/
theorem circle_radius_isosceles_right_triangle (O : ℝ × ℝ) (P Q R S T U : ℝ × ℝ) (r : ℝ) :
  -- PQR is an isosceles right triangle
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 →
  (P.1 - R.1)^2 + (P.2 - R.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 →
  (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0 →
  -- S is on PQ
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2) →
  -- Circle with center O is tangent to coordinate axes
  O.1 = r ∧ O.2 = r →
  -- Circle is tangent to PQ at T
  (T.1 - O.1)^2 + (T.2 - O.2)^2 = r^2 →
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (s * P.1 + (1 - s) * Q.1, s * P.2 + (1 - s) * Q.2) →
  -- U is on x-axis and circle is tangent at U
  U.2 = 0 ∧ (U.1 - O.1)^2 + (U.2 - O.2)^2 = r^2 →
  -- The radius of the circle is 2 + √2
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_isosceles_right_triangle_l1333_133325


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_fourth_l1333_133357

/-- Represents a square board with shaded regions -/
structure Board :=
  (size : ℕ)
  (shaded_area : ℚ)

/-- Calculates the fraction of shaded area on the board -/
def shaded_fraction (b : Board) : ℚ :=
  b.shaded_area / (b.size * b.size : ℚ)

/-- Represents the specific board configuration described in the problem -/
def problem_board : Board :=
  { size := 4,
    shaded_area := 4 }

/-- Theorem stating that the shaded fraction of the problem board is 1/4 -/
theorem shaded_fraction_is_one_fourth :
  shaded_fraction problem_board = 1/4 := by
  sorry

#check shaded_fraction_is_one_fourth

end NUMINAMATH_CALUDE_shaded_fraction_is_one_fourth_l1333_133357


namespace NUMINAMATH_CALUDE_fly_probabilities_l1333_133346

def fly_walk (n m : ℕ) : ℚ := (Nat.choose (n + m) n : ℚ) / (2 ^ (n + m))

def fly_walk_through (n₁ m₁ n₂ m₂ : ℕ) : ℚ :=
  (Nat.choose (n₁ + m₁) n₁ : ℚ) * (Nat.choose (n₂ + m₂) n₂) / (2 ^ (n₁ + m₁ + n₂ + m₂))

def fly_walk_circle : ℚ :=
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18

theorem fly_probabilities :
  (fly_walk 8 10 = (Nat.choose 18 8 : ℚ) / 2^18) ∧
  (fly_walk_through 5 6 2 4 = ((Nat.choose 11 5 : ℚ) * Nat.choose 6 2) / 2^18) ∧
  (fly_walk_circle = (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + Nat.choose 9 4 * Nat.choose 9 4 : ℚ) / 2^18) := by
  sorry

end NUMINAMATH_CALUDE_fly_probabilities_l1333_133346


namespace NUMINAMATH_CALUDE_intersection_sufficient_not_necessary_for_union_l1333_133365

-- Define the sets M and P
def M : Set ℝ := {x | x > 1}
def P : Set ℝ := {x | x < 4}

-- State the theorem
theorem intersection_sufficient_not_necessary_for_union :
  (∀ x, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  (∃ x, x ∈ M ∪ P ∧ x ∉ M ∩ P) := by
  sorry

end NUMINAMATH_CALUDE_intersection_sufficient_not_necessary_for_union_l1333_133365


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l1333_133387

theorem isosceles_right_triangle (A B C : ℝ) (a b c : ℝ) : 
  (Real.sin (A - B))^2 + (Real.cos C)^2 = 0 → 
  (A = B ∧ C = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l1333_133387


namespace NUMINAMATH_CALUDE_cut_cube_properties_l1333_133373

/-- A cube with one corner cut off -/
structure CutCube where
  vertices : Finset (ℝ × ℝ × ℝ)
  faces : Finset (Finset (ℝ × ℝ × ℝ))

/-- Properties of a cube with one corner cut off -/
def is_valid_cut_cube (c : CutCube) : Prop :=
  c.vertices.card = 10 ∧ c.faces.card = 9

/-- Theorem: A cube with one corner cut off has 10 vertices and 9 faces -/
theorem cut_cube_properties (c : CutCube) (h : is_valid_cut_cube c) :
  c.vertices.card = 10 ∧ c.faces.card = 9 := by
  sorry


end NUMINAMATH_CALUDE_cut_cube_properties_l1333_133373


namespace NUMINAMATH_CALUDE_business_investment_problem_l1333_133390

theorem business_investment_problem (y_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) (x_investment : ℕ) :
  y_investment = 15000 →
  total_profit = 1600 →
  x_profit_share = 400 →
  x_profit_share * y_investment = (total_profit - x_profit_share) * x_investment →
  x_investment = 5000 := by
sorry

end NUMINAMATH_CALUDE_business_investment_problem_l1333_133390


namespace NUMINAMATH_CALUDE_max_value_of_function_l1333_133307

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  (∀ z : ℝ, z < 5/4 → 4*z - 2 + 1/(4*z - 5) ≤ 4*x - 2 + 1/(4*x - 5)) →
  4*x - 2 + 1/(4*x - 5) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1333_133307


namespace NUMINAMATH_CALUDE_train_passing_time_l1333_133343

/-- Proves the time it takes for a train to pass a stationary point given its speed and time to cross a platform of known length -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) : 
  train_speed_kmph = 72 → 
  platform_length = 260 → 
  platform_crossing_time = 30 → 
  (platform_length + (train_speed_kmph * 1000 / 3600 * platform_crossing_time)) / (train_speed_kmph * 1000 / 3600) = 17 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1333_133343


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l1333_133355

theorem matrix_multiplication_example : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l1333_133355


namespace NUMINAMATH_CALUDE_decimal_to_binary_98_l1333_133376

theorem decimal_to_binary_98 : 
  (98 : ℕ).digits 2 = [0, 1, 0, 0, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_98_l1333_133376


namespace NUMINAMATH_CALUDE_investment_problem_l1333_133386

theorem investment_problem (initial_investment : ℝ) (growth_rate_year1 : ℝ) (growth_rate_year2 : ℝ) (final_value : ℝ) (amount_added : ℝ) : 
  initial_investment = 80 →
  growth_rate_year1 = 0.15 →
  growth_rate_year2 = 0.10 →
  final_value = 132 →
  amount_added = 28 →
  final_value = (initial_investment * (1 + growth_rate_year1) + amount_added) * (1 + growth_rate_year2) :=
by sorry

#check investment_problem

end NUMINAMATH_CALUDE_investment_problem_l1333_133386


namespace NUMINAMATH_CALUDE_kamal_math_marks_l1333_133362

/-- Proves that given Kamal's marks in English, Physics, Chemistry, and Biology,
    with a specific average for all 5 subjects, his marks in Mathematics can be determined. -/
theorem kamal_math_marks
  (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ)
  (h_english : english = 76)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_biology : biology = 85)
  (h_average : average = 75)
  (h_subjects : 5 * average = english + physics + chemistry + biology + mathematics) :
  mathematics = 65 :=
by sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l1333_133362


namespace NUMINAMATH_CALUDE_tim_bought_three_dozens_l1333_133313

/-- The number of dozens of eggs Tim bought -/
def dozens_bought (egg_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (12 * egg_price)

/-- Theorem stating that Tim bought 3 dozens of eggs -/
theorem tim_bought_three_dozens :
  dozens_bought (1/2) 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tim_bought_three_dozens_l1333_133313


namespace NUMINAMATH_CALUDE_prob_three_odd_less_than_eighth_l1333_133350

def total_integers : ℕ := 2020
def odd_integers : ℕ := total_integers / 2

theorem prob_three_odd_less_than_eighth :
  let p := (odd_integers : ℚ) / total_integers *
           ((odd_integers - 1) : ℚ) / (total_integers - 1) *
           ((odd_integers - 2) : ℚ) / (total_integers - 2)
  p < 1 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_three_odd_less_than_eighth_l1333_133350


namespace NUMINAMATH_CALUDE_no_curious_numbers_l1333_133337

def CuriousNumber (f : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, f x = f (a - x)

theorem no_curious_numbers (f : ℤ → ℤ) 
  (h1 : ∀ x : ℤ, f x ≠ x) :
  ¬ (∃ a ∈ ({60, 62, 823} : Set ℤ), CuriousNumber f a) :=
sorry

end NUMINAMATH_CALUDE_no_curious_numbers_l1333_133337


namespace NUMINAMATH_CALUDE_coffee_ounces_per_cup_l1333_133358

/-- Proves that the number of ounces of coffee per cup is 0.5 --/
theorem coffee_ounces_per_cup : 
  ∀ (people : ℕ) (cups_per_person_per_day : ℕ) (cost_per_ounce : ℚ) (weekly_expenditure : ℚ),
    people = 4 →
    cups_per_person_per_day = 2 →
    cost_per_ounce = 1.25 →
    weekly_expenditure = 35 →
    (weekly_expenditure / cost_per_ounce) / (people * cups_per_person_per_day * 7 : ℚ) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_coffee_ounces_per_cup_l1333_133358


namespace NUMINAMATH_CALUDE_smallest_product_l1333_133369

def digits : List Nat := [1, 2, 3, 4]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : Nat) : Nat :=
  (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    product a b c d ≥ 312 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l1333_133369


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1333_133342

/-- A hyperbola with the given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let e := c / a
  -- Hyperbola equation
  (P.1 / a) ^ 2 - (P.2 / b) ^ 2 = 1 ∧
  -- Line through F₁ at 30° inclination
  (P.2 + c * Real.tan (30 * π / 180)) / (P.1 + c) = Real.tan (30 * π / 180) ∧
  -- Circle with diameter PF₁ passes through F₂
  (P.1 - (-c)) ^ 2 + P.2 ^ 2 = (2 * c) ^ 2 ∧
  -- Standard hyperbola relations
  c ^ 2 = a ^ 2 + b ^ 2 ∧
  P.1 > 0 -- P is on the right branch
  →
  e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1333_133342


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1333_133323

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1333_133323


namespace NUMINAMATH_CALUDE_distance_AB_l1333_133338

-- Define the points and distances
structure Points where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

-- Define the speeds
structure Speeds where
  vA : ℝ
  vB : ℝ

-- Define the problem conditions
structure Conditions where
  points : Points
  speeds : Speeds
  CD_distance : ℝ
  B_remaining_distance : ℝ
  speed_ratio : ℝ
  speed_reduction : ℝ

-- Theorem statement
theorem distance_AB (c : Conditions) : 
  c.CD_distance = 900 ∧ 
  c.B_remaining_distance = 720 ∧ 
  c.speed_ratio = 5/4 ∧ 
  c.speed_reduction = 4/5 →
  c.points.B - c.points.A = 5265 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_l1333_133338


namespace NUMINAMATH_CALUDE_boys_in_class_l1333_133305

theorem boys_in_class (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 56) (h2 : girl_ratio = 4) (h3 : boy_ratio = 3) :
  (total * boy_ratio) / (girl_ratio + boy_ratio) = 24 :=
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l1333_133305


namespace NUMINAMATH_CALUDE_inequality_proof_l1333_133391

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1333_133391


namespace NUMINAMATH_CALUDE_two_to_700_gt_five_to_300_l1333_133383

theorem two_to_700_gt_five_to_300 : 2^700 > 5^300 := by
  sorry

end NUMINAMATH_CALUDE_two_to_700_gt_five_to_300_l1333_133383


namespace NUMINAMATH_CALUDE_triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l1333_133371

-- Define the new magnitude
def new_magnitude (x y : ℝ) : ℝ := |x + y| + |x - y|

-- Theorem for proposition (1)
theorem triangle_inequality_new_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  new_magnitude (x₁ - x₂) (y₁ - y₂) ≤ new_magnitude x₁ y₁ + new_magnitude x₂ y₂ := by
  sorry

-- Theorem for proposition (2)
theorem min_magnitude_on_line :
  ∃ (t : ℝ), ∀ (s : ℝ), new_magnitude t (t - 1) ≤ new_magnitude s (s - 1) ∧ new_magnitude t (t - 1) = 1 := by
  sorry

-- Theorem for proposition (3)
theorem max_magnitude_on_circle :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ new_magnitude x y = 2 ∧ 
  ∀ (a b : ℝ), a^2 + b^2 = 1 → new_magnitude a b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l1333_133371
