import Mathlib

namespace NUMINAMATH_CALUDE_max_prism_volume_in_hexagonal_pyramid_l3524_352476

/-- Represents a regular hexagonal pyramid -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_leg_length : ℝ

/-- Represents a right square prism -/
structure SquarePrism where
  side_length : ℝ

/-- Calculates the volume of a right square prism -/
def prism_volume (p : SquarePrism) : ℝ := p.side_length ^ 3

/-- Theorem stating the maximum volume of the square prism within the hexagonal pyramid -/
theorem max_prism_volume_in_hexagonal_pyramid 
  (pyramid : HexagonalPyramid) 
  (prism : SquarePrism) 
  (h1 : pyramid.base_side_length = 2) 
  (h2 : prism.side_length ≤ pyramid.base_side_length) 
  (h3 : prism.side_length > 0) :
  prism_volume prism ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_prism_volume_in_hexagonal_pyramid_l3524_352476


namespace NUMINAMATH_CALUDE_linear_equation_condition_l3524_352434

theorem linear_equation_condition (m : ℤ) : 
  (∃ a b : ℝ, ∀ x : ℝ, (m + 1 : ℝ) * x^(|m|) + 3 = a * x + b) ↔ m = 1 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l3524_352434


namespace NUMINAMATH_CALUDE_sector_arc_length_ratio_l3524_352474

theorem sector_arc_length_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let sector_radius := 2 * r / 3
  let sector_area := 5 * circle_area / 27
  let circle_circumference := 2 * π * r
  ∃ α : ℝ, 
    sector_area = α * sector_radius^2 / 2 ∧ 
    (α * sector_radius) / circle_circumference = 5 / 18 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_ratio_l3524_352474


namespace NUMINAMATH_CALUDE_longest_side_length_l3524_352465

-- Define a triangle with angle ratio 1:2:3 and shortest side 5 cm
structure SpecialTriangle where
  -- a, b, c are the side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle A is opposite to side a, B to b, C to c
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Conditions
  angle_ratio : angleA / angleB = 1/2 ∧ angleB / angleC = 2/3
  shortest_side : min a (min b c) = 5
  -- Triangle properties
  sum_angles : angleA + angleB + angleC = π
  -- Law of sines
  law_of_sines : a / (Real.sin angleA) = b / (Real.sin angleB)
                 ∧ b / (Real.sin angleB) = c / (Real.sin angleC)

-- Theorem statement
theorem longest_side_length (t : SpecialTriangle) : max t.a (max t.b t.c) = 10 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_length_l3524_352465


namespace NUMINAMATH_CALUDE_remaining_gasoline_l3524_352407

/-- Calculates the remaining gasoline in a car's tank after a journey -/
theorem remaining_gasoline
  (initial_gasoline : ℝ)
  (distance : ℝ)
  (fuel_consumption : ℝ)
  (h1 : initial_gasoline = 47)
  (h2 : distance = 275)
  (h3 : fuel_consumption = 12)
  : initial_gasoline - (distance * fuel_consumption / 100) = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_gasoline_l3524_352407


namespace NUMINAMATH_CALUDE_range_of_m_l3524_352488

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3524_352488


namespace NUMINAMATH_CALUDE_project_hours_difference_l3524_352430

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 135) 
  (pat kate mark : ℕ) 
  (h_pat_kate : pat = 2 * kate) 
  (h_pat_mark : pat * 3 = mark) 
  (h_sum : pat + kate + mark = total_hours) : 
  mark - kate = 75 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3524_352430


namespace NUMINAMATH_CALUDE_max_weight_difference_is_0_6_l3524_352470

/-- Represents the weight range of a flour bag -/
structure FlourBag where
  center : ℝ
  tolerance : ℝ

/-- Calculates the maximum weight of a flour bag -/
def max_weight (bag : FlourBag) : ℝ := bag.center + bag.tolerance

/-- Calculates the minimum weight of a flour bag -/
def min_weight (bag : FlourBag) : ℝ := bag.center - bag.tolerance

/-- Theorem: The maximum difference in weights between any two bags is 0.6 kg -/
theorem max_weight_difference_is_0_6 (bag1 bag2 bag3 : FlourBag)
  (h1 : bag1 = ⟨25, 0.1⟩)
  (h2 : bag2 = ⟨25, 0.2⟩)
  (h3 : bag3 = ⟨25, 0.3⟩) :
  (max_weight bag3 - min_weight bag3) = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_max_weight_difference_is_0_6_l3524_352470


namespace NUMINAMATH_CALUDE_no_unchanged_sum_l3524_352438

theorem no_unchanged_sum : ¬∃ (A B : ℕ), A + B = 2022 ∧ A / 2 + 3 * B = A + B := by
  sorry

end NUMINAMATH_CALUDE_no_unchanged_sum_l3524_352438


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3524_352482

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ (Set.Icc 0 1), f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3524_352482


namespace NUMINAMATH_CALUDE_f_root_and_positivity_l3524_352464

noncomputable def f (x : ℝ) : ℝ := 2^x - 2/x

theorem f_root_and_positivity :
  (∃! x : ℝ, f x = 0 ∧ x = 1) ∧
  (∀ x : ℝ, x ≠ 0 → (f x > 0 ↔ x < 0 ∨ x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_root_and_positivity_l3524_352464


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l3524_352446

theorem positive_numbers_inequalities (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≥ 1 ∧
  a^2/(b+c) + b^2/(a+c) + c^2/(a+b) ≥ 1/2 := by
sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l3524_352446


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3524_352459

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ (a : ℕ) + (b : ℕ) = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → (c : ℕ) + (d : ℕ) ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l3524_352459


namespace NUMINAMATH_CALUDE_triangle_probability_l3524_352458

def stick_lengths : List ℕ := [1, 4, 6, 8, 9, 10, 12, 15]

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_perimeter_gt_20 (a b c : ℕ) : Prop :=
  a + b + c > 20

def valid_triangle_count : ℕ := 16

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability : 
  (valid_triangle_count : ℚ) / total_combinations = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_triangle_probability_l3524_352458


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3524_352426

/-- Given a right triangle with sides 5, 12, and 13 (13 being the hypotenuse),
    a square of side length x inscribed with one side along the leg of length 12,
    and another square of side length y inscribed with one side along the hypotenuse,
    the ratio of x to y is 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 → y > 0 →
  x^2 + x^2 = 5 * x →
  y^2 + y^2 = 13 * y →
  x / y = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3524_352426


namespace NUMINAMATH_CALUDE_point_5_neg1_in_fourth_quadrant_l3524_352494

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_5_neg1_in_fourth_quadrant :
  point_in_fourth_quadrant 5 (-1) := by
  sorry

end NUMINAMATH_CALUDE_point_5_neg1_in_fourth_quadrant_l3524_352494


namespace NUMINAMATH_CALUDE_dvd_packs_cost_l3524_352462

/-- Proves that given the cost of each DVD pack and the number of packs that can be bought,
    the total amount of money available is correct. -/
theorem dvd_packs_cost (cost_per_pack : ℕ) (num_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 26 → num_packs = 4 → total_money = cost_per_pack * num_packs →
  total_money = 104 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_cost_l3524_352462


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l3524_352467

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 3)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_obtuse (v w : ℝ × ℝ) : Prop := dot_product v w < 0

theorem obtuse_angle_range (t : ℝ) :
  is_obtuse vector_a (vector_b t) →
  t ∈ (Set.Iio (-6) ∪ Set.Ioo (-6) (3/2)) :=
sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l3524_352467


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3524_352497

/-- Represents the number of players selected from a class using stratified sampling -/
def stratified_sample (total_players : ℕ) (class_players : ℕ) (sample_size : ℕ) : ℕ :=
  (class_players * sample_size) / total_players

theorem basketball_team_selection (class5_players class16_players class33_players : ℕ) 
  (h1 : class5_players = 6)
  (h2 : class16_players = 8)
  (h3 : class33_players = 10)
  (h4 : class5_players + class16_players + class33_players = 24)
  (sample_size : ℕ)
  (h5 : sample_size = 12) :
  stratified_sample (class5_players + class16_players + class33_players) class5_players sample_size = 3 ∧
  stratified_sample (class5_players + class16_players + class33_players) class16_players sample_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3524_352497


namespace NUMINAMATH_CALUDE_smallest_k_for_error_bound_l3524_352441

def u : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem smallest_k_for_error_bound :
  ∃ (k : ℕ), (∀ (n : ℕ), n < k → |u n - L| > 1/2^1000) ∧
             |u k - L| ≤ 1/2^1000 ∧
             k = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_error_bound_l3524_352441


namespace NUMINAMATH_CALUDE_non_adjacent_white_balls_arrangements_select_balls_with_min_score_l3524_352415

/-- Represents the number of red balls in the bag -/
def num_red_balls : ℕ := 5

/-- Represents the number of white balls in the bag -/
def num_white_balls : ℕ := 4

/-- Represents the score for taking out a red ball -/
def red_ball_score : ℕ := 2

/-- Represents the score for taking out a white ball -/
def white_ball_score : ℕ := 1

/-- Represents the minimum required score -/
def min_score : ℕ := 8

/-- Represents the number of balls to be taken out -/
def balls_to_take : ℕ := 5

/-- Theorem for the number of ways to arrange balls with non-adjacent white balls -/
theorem non_adjacent_white_balls_arrangements : ℕ := by sorry

/-- Theorem for the number of ways to select balls with a minimum score -/
theorem select_balls_with_min_score : ℕ := by sorry

end NUMINAMATH_CALUDE_non_adjacent_white_balls_arrangements_select_balls_with_min_score_l3524_352415


namespace NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l3524_352433

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 5)

theorem reflection_of_P_across_x_axis :
  reflect_x P = (-2, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_P_across_x_axis_l3524_352433


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_75_l3524_352490

theorem distinct_prime_factors_of_75 : Nat.card (Nat.factors 75).toFinset = 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_75_l3524_352490


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3524_352427

theorem geometric_sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * r) →
  a 3 = 256 →
  a 5 = 4 →
  a 3 + a 4 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3524_352427


namespace NUMINAMATH_CALUDE_calculation_proof_l3524_352400

theorem calculation_proof :
  (- (1 : ℤ) ^ 2023 + 8 * (-(1/2 : ℚ))^3 + |(-3 : ℤ)| = 1) ∧
  ((-25 : ℤ) * (3/2 : ℚ) - (-25 : ℤ) * (5/8 : ℚ) + (-25 : ℤ) / 8 = -25) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3524_352400


namespace NUMINAMATH_CALUDE_sum_1984_consecutive_not_square_l3524_352498

theorem sum_1984_consecutive_not_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (992 * (2 * n + 1985) : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_1984_consecutive_not_square_l3524_352498


namespace NUMINAMATH_CALUDE_evaluate_expression_l3524_352418

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 1 = 188 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3524_352418


namespace NUMINAMATH_CALUDE_subtraction_decimal_l3524_352483

theorem subtraction_decimal : 7.42 - 2.09 = 5.33 := by sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l3524_352483


namespace NUMINAMATH_CALUDE_unique_factors_of_135135_l3524_352456

theorem unique_factors_of_135135 :
  ∃! (a b c d e f : ℕ),
    1 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    a * b * c * d * e * f = 135135 ∧
    a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 ∧ e = 11 ∧ f = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_factors_of_135135_l3524_352456


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3524_352452

open Real

/-- Given a function g: ℝ → ℝ satisfying the functional equation
    (g(x) * g(y) - g(x*y)) / 5 = x + y + 4 for all x, y ∈ ℝ,
    prove that g(x) = x + 5 for all x ∈ ℝ. -/
theorem functional_equation_solution (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  ∀ x : ℝ, g x = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3524_352452


namespace NUMINAMATH_CALUDE_difference_of_variables_l3524_352417

theorem difference_of_variables (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_variables_l3524_352417


namespace NUMINAMATH_CALUDE_correlation_analysis_l3524_352495

-- Define the types of variables
def TaxiFare : Type := ℝ
def Distance : Type := ℝ
def HouseSize : Type := ℝ
def HousePrice : Type := ℝ
def Height : Type := ℝ
def Weight : Type := ℝ
def IronBlockSize : Type := ℝ
def IronBlockMass : Type := ℝ

-- Define the relationship between variables
def functionalRelationship (α β : Type) : Prop := ∃ f : α → β, ∀ x : α, ∃! y : β, f x = y

-- Define correlation
def correlated (α β : Type) : Prop := 
  ¬(functionalRelationship α β) ∧ ¬(functionalRelationship β α) ∧ 
  ∃ f : α → β, ∀ x y : α, x ≠ y → f x ≠ f y

-- State the theorem
theorem correlation_analysis :
  functionalRelationship TaxiFare Distance ∧
  functionalRelationship HouseSize HousePrice ∧
  correlated Height Weight ∧
  functionalRelationship IronBlockSize IronBlockMass :=
by sorry

end NUMINAMATH_CALUDE_correlation_analysis_l3524_352495


namespace NUMINAMATH_CALUDE_bean_sprouts_and_dried_tofu_problem_l3524_352449

/-- Bean sprouts and dried tofu problem -/
theorem bean_sprouts_and_dried_tofu_problem 
  (bean_sprouts_price dried_tofu_price : ℚ)
  (bean_sprouts_sell_price dried_tofu_sell_price : ℚ)
  (total_units : ℕ)
  (max_cost : ℚ) :
  bean_sprouts_price = 60 →
  dried_tofu_price = 40 →
  bean_sprouts_sell_price = 80 →
  dried_tofu_sell_price = 55 →
  total_units = 200 →
  max_cost = 10440 →
  2 * bean_sprouts_price + 3 * dried_tofu_price = 240 →
  3 * bean_sprouts_price + 4 * dried_tofu_price = 340 →
  ∃ (bean_sprouts_units dried_tofu_units : ℕ),
    bean_sprouts_units + dried_tofu_units = total_units ∧
    bean_sprouts_price * bean_sprouts_units + dried_tofu_price * dried_tofu_units ≤ max_cost ∧
    (bean_sprouts_units : ℚ) ≥ (3/2) * dried_tofu_units ∧
    bean_sprouts_units = 122 ∧
    dried_tofu_units = 78 ∧
    (bean_sprouts_sell_price - bean_sprouts_price) * bean_sprouts_units +
    (dried_tofu_sell_price - dried_tofu_price) * dried_tofu_units = 3610 ∧
    ∀ (other_bean_sprouts_units other_dried_tofu_units : ℕ),
      other_bean_sprouts_units + other_dried_tofu_units = total_units →
      bean_sprouts_price * other_bean_sprouts_units + dried_tofu_price * other_dried_tofu_units ≤ max_cost →
      (other_bean_sprouts_units : ℚ) ≥ (3/2) * other_dried_tofu_units →
      (bean_sprouts_sell_price - bean_sprouts_price) * other_bean_sprouts_units +
      (dried_tofu_sell_price - dried_tofu_price) * other_dried_tofu_units ≤ 3610 :=
by
  sorry

end NUMINAMATH_CALUDE_bean_sprouts_and_dried_tofu_problem_l3524_352449


namespace NUMINAMATH_CALUDE_table_length_proof_l3524_352444

theorem table_length_proof (table_width : ℕ) (sheet_width sheet_height : ℕ) :
  table_width = 80 ∧ 
  sheet_width = 8 ∧ 
  sheet_height = 5 ∧ 
  (∃ n : ℕ, table_width = sheet_width + n ∧ n + 1 = sheet_width - sheet_height + 1) →
  ∃ table_length : ℕ, table_length = 77 ∧ table_length = table_width - (sheet_width - sheet_height) :=
by sorry

end NUMINAMATH_CALUDE_table_length_proof_l3524_352444


namespace NUMINAMATH_CALUDE_max_element_of_A_l3524_352420

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- Define the set A
def A (x y : ℝ) : Set ℝ := {log x, log y, log (x + y/x)}

-- Define the theorem
theorem max_element_of_A :
  ∀ x y : ℝ, x > 0 → y > 0 → {0, 1} ⊆ A x y →
  ∃ z ∈ A x y, ∀ w ∈ A x y, w ≤ z ∧ z = log 11 :=
sorry

end NUMINAMATH_CALUDE_max_element_of_A_l3524_352420


namespace NUMINAMATH_CALUDE_linear_dependence_condition_l3524_352408

def vector1 : Fin 2 → ℝ := ![2, 3]
def vector2 (k : ℝ) : Fin 2 → ℝ := ![4, k]

def is_linearly_dependent (v1 v2 : Fin 2 → ℝ) : Prop :=
  ∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ c1 • v1 + c2 • v2 = 0

theorem linear_dependence_condition (k : ℝ) :
  is_linearly_dependent vector1 (vector2 k) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_linear_dependence_condition_l3524_352408


namespace NUMINAMATH_CALUDE_card_sending_probability_l3524_352480

def num_senders : ℕ := 3
def num_recipients : ℕ := 2

theorem card_sending_probability :
  let total_outcomes := num_recipients ^ num_senders
  let favorable_outcomes := num_recipients
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_card_sending_probability_l3524_352480


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3524_352437

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = x}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3524_352437


namespace NUMINAMATH_CALUDE_five_solutions_l3524_352451

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    Finset.card solutions = 5 ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ solutions ↔
      x = z + w + 2*z*w*x ∧
      y = w + x + w*x*y ∧
      z = x + y + x*y*z ∧
      w = y + z + 2*y*z*w := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l3524_352451


namespace NUMINAMATH_CALUDE_exponential_function_implies_a_eq_three_l3524_352468

/-- A function f: ℝ → ℝ is exponential if there exist constants b > 0, b ≠ 1, and c such that f(x) = c * b^x for all x ∈ ℝ. -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ ∀ x, f x = c * b^x

/-- If f(x) = (a-2) * a^x is an exponential function, then a = 3. -/
theorem exponential_function_implies_a_eq_three (a : ℝ) :
  IsExponentialFunction (fun x ↦ (a - 2) * a^x) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_implies_a_eq_three_l3524_352468


namespace NUMINAMATH_CALUDE_m_range_l3524_352443

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  m * x₁^2 - x₁ + m - 4 = 0 ∧ 
  m * x₂^2 - x₂ + m - 4 = 0 ∧ 
  x₁ > 0 ∧ x₂ < 0

-- Define the theorem
theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) → ¬(p m) → m ≥ 1 + Real.sqrt 2 ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3524_352443


namespace NUMINAMATH_CALUDE_sum_of_odd_three_digit_numbers_l3524_352478

/-- The set of odd digits -/
def OddDigits : Finset ℕ := {1, 3, 5, 7, 9}

/-- A three-digit number with odd digits -/
structure OddThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_in_odd_digits : hundreds ∈ OddDigits
  tens_in_odd_digits : tens ∈ OddDigits
  units_in_odd_digits : units ∈ OddDigits

/-- The set of all possible odd three-digit numbers -/
def AllOddThreeDigitNumbers : Finset OddThreeDigitNumber := sorry

/-- The value of an odd three-digit number -/
def value (n : OddThreeDigitNumber) : ℕ := 100 * n.hundreds + 10 * n.tens + n.units

/-- The theorem stating the sum of all odd three-digit numbers -/
theorem sum_of_odd_three_digit_numbers :
  (AllOddThreeDigitNumbers.sum value) = 69375 := by sorry

end NUMINAMATH_CALUDE_sum_of_odd_three_digit_numbers_l3524_352478


namespace NUMINAMATH_CALUDE_monica_reading_ratio_l3524_352469

/-- The number of books Monica read last year -/
def last_year : ℕ := 16

/-- The number of books Monica will read next year -/
def next_year : ℕ := 69

/-- The number of books Monica read this year -/
def this_year : ℕ := last_year * 2

theorem monica_reading_ratio :
  (this_year / last_year : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_monica_reading_ratio_l3524_352469


namespace NUMINAMATH_CALUDE_eldest_child_age_l3524_352447

theorem eldest_child_age 
  (n : ℕ) 
  (d : ℕ) 
  (sum : ℕ) 
  (h1 : n = 5) 
  (h2 : d = 2) 
  (h3 : sum = 50) : 
  (sum - (n * (n - 1) / 2) * d) / n + (n - 1) * d = 14 := by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l3524_352447


namespace NUMINAMATH_CALUDE_basketball_shots_l3524_352414

theorem basketball_shots (total_points : ℕ) (total_shots : ℕ) 
  (h_points : total_points = 26) (h_shots : total_shots = 11) :
  ∃ (three_pointers two_pointers : ℕ),
    three_pointers + two_pointers = total_shots ∧
    3 * three_pointers + 2 * two_pointers = total_points ∧
    three_pointers = 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_shots_l3524_352414


namespace NUMINAMATH_CALUDE_g_pi_third_equals_one_l3524_352461

/-- Given a function f and a constant w, φ, prove that g(π/3) = 1 -/
theorem g_pi_third_equals_one 
  (f : ℝ → ℝ) 
  (w φ : ℝ) 
  (h1 : ∀ x, f x = 5 * Real.cos (w * x + φ))
  (h2 : ∀ x, f (π/3 + x) = f (π/3 - x))
  (g : ℝ → ℝ)
  (h3 : ∀ x, g x = 4 * Real.sin (w * x + φ) + 1) :
  g (π/3) = 1 := by
sorry

end NUMINAMATH_CALUDE_g_pi_third_equals_one_l3524_352461


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3524_352406

theorem cubic_equation_solution :
  let f (x : ℂ) := (x - 2)^3 + (x - 6)^3
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∀ x, f x = 0 → x ∈ s) ∧
    (4 ∈ s) ∧ 
    (4 + 2 * Complex.I * Real.sqrt 3 ∈ s) ∧ 
    (4 - 2 * Complex.I * Real.sqrt 3 ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3524_352406


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l3524_352450

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l3524_352450


namespace NUMINAMATH_CALUDE_product_from_lcm_gcf_l3524_352496

theorem product_from_lcm_gcf (a b c : ℕ+) 
  (h1 : Nat.lcm (Nat.lcm a.val b.val) c.val = 2310)
  (h2 : Nat.gcd (Nat.gcd a.val b.val) c.val = 30) :
  a * b * c = 69300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcf_l3524_352496


namespace NUMINAMATH_CALUDE_merry_and_brother_lambs_l3524_352402

/-- The number of lambs Merry and her brother have in total -/
def total_lambs (merry_lambs : ℕ) (brother_extra : ℕ) : ℕ :=
  merry_lambs + (merry_lambs + brother_extra)

/-- Theorem stating the total number of lambs Merry and her brother have -/
theorem merry_and_brother_lambs :
  total_lambs 10 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_merry_and_brother_lambs_l3524_352402


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3524_352409

def total_cars : ℕ := 250
def mercedes_percent : ℚ := 18 / 100
def toyota_percent : ℚ := 25 / 100
def acura_percent : ℚ := 15 / 100

theorem bmw_sales_count :
  (total_cars : ℚ) * (1 - (mercedes_percent + toyota_percent + acura_percent)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3524_352409


namespace NUMINAMATH_CALUDE_alpha_gamma_relation_l3524_352471

theorem alpha_gamma_relation (α β γ : ℝ) 
  (h1 : β = 10^(1 / (1 - Real.log α)))
  (h2 : γ = 10^(1 / (1 - Real.log β))) :
  α = 10^(1 / (1 - Real.log γ)) := by
  sorry

end NUMINAMATH_CALUDE_alpha_gamma_relation_l3524_352471


namespace NUMINAMATH_CALUDE_exists_x0_abs_fx0_plus_a_nonneg_l3524_352499

theorem exists_x0_abs_fx0_plus_a_nonneg (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |((x₀^2 : ℝ) + a * x₀ + b) + a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_abs_fx0_plus_a_nonneg_l3524_352499


namespace NUMINAMATH_CALUDE_stirling_number_second_kind_formula_l3524_352440

def stirling_number_second_kind (n r : ℕ) : ℚ :=
  (1 / r.factorial) *
    (Finset.sum (Finset.range (r + 1)) (fun k => 
      ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n))))

theorem stirling_number_second_kind_formula (n r : ℕ) (h : n ≥ r) (hr : r > 0) :
  stirling_number_second_kind n r =
    (1 / r.factorial : ℚ) *
      (Finset.sum (Finset.range (r + 1)) (fun k => 
        ((-1 : ℚ) ^ k * (r.choose k) * ((r - k) ^ n)))) :=
by sorry

end NUMINAMATH_CALUDE_stirling_number_second_kind_formula_l3524_352440


namespace NUMINAMATH_CALUDE_change_calculation_l3524_352466

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def bills_given : ℕ := 20 * 2
def coins_given : ℕ := 3

def total_cost : ℕ := flour_cost + cake_stand_cost
def total_paid : ℕ := bills_given + coins_given

theorem change_calculation (change : ℕ) : 
  change = total_paid - total_cost := by sorry

end NUMINAMATH_CALUDE_change_calculation_l3524_352466


namespace NUMINAMATH_CALUDE_dog_speed_l3524_352445

/-- Proves that a dog catching a rabbit with given parameters runs at 24 miles per hour -/
theorem dog_speed (rabbit_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  rabbit_speed = 15 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  let dog_distance := rabbit_speed * catch_up_time + head_start
  dog_distance / catch_up_time = 24 := by sorry

end NUMINAMATH_CALUDE_dog_speed_l3524_352445


namespace NUMINAMATH_CALUDE_larger_number_proof_l3524_352436

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 24672)
  (h2 : L = 13 * S + 257) :
  L = 26706 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3524_352436


namespace NUMINAMATH_CALUDE_rooster_earnings_l3524_352493

/-- Calculates the total earnings from selling roosters -/
def total_earnings (price_per_kg : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  price_per_kg * (weight1 + weight2)

/-- Theorem: The total earnings from selling two roosters weighing 30 kg and 40 kg at $0.50 per kg is $35 -/
theorem rooster_earnings : total_earnings 0.5 30 40 = 35 := by
  sorry

end NUMINAMATH_CALUDE_rooster_earnings_l3524_352493


namespace NUMINAMATH_CALUDE_symmetry_sum_l3524_352439

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = -p2.1 ∧ p1.2 = -p2.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (a, -3) (4, b) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l3524_352439


namespace NUMINAMATH_CALUDE_smallest_n_greater_than_threshold_l3524_352404

/-- The first term of the arithmetic sequence -/
def a₁ : ℕ := 11

/-- The common difference of the arithmetic sequence -/
def d : ℕ := 6

/-- The threshold value -/
def threshold : ℕ := 2017

/-- The n-th term of the arithmetic sequence -/
def aₙ (n : ℕ) : ℕ := a₁ + (n - 1) * d

/-- The proposition to be proved -/
theorem smallest_n_greater_than_threshold :
  (∀ k ≥ 336, aₙ k > threshold) ∧
  (∀ m < 336, ∃ l ≥ m, aₙ l ≤ threshold) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_greater_than_threshold_l3524_352404


namespace NUMINAMATH_CALUDE_percentage_problem_l3524_352492

theorem percentage_problem (x : ℝ) (P : ℝ) : 
  x = 680 →
  (P / 100) * x = 0.20 * 1000 - 30 →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3524_352492


namespace NUMINAMATH_CALUDE_sum_of_fractions_and_decimal_l3524_352403

theorem sum_of_fractions_and_decimal : 
  (1 : ℚ) / 3 + 5 / 24 + (816 : ℚ) / 100 + 1 / 8 = 5296 / 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_and_decimal_l3524_352403


namespace NUMINAMATH_CALUDE_cos_4theta_from_exp_l3524_352475

theorem cos_4theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 3) / 2) :
  Real.cos (4 * θ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_4theta_from_exp_l3524_352475


namespace NUMINAMATH_CALUDE_min_value_cubic_function_l3524_352401

theorem min_value_cubic_function (y : ℝ) (h : y > 0) :
  y^2 + 10*y + 100/y^3 ≥ 50^(2/3) + 10 * 50^(1/3) + 2 ∧
  (y^2 + 10*y + 100/y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 ↔ y = 50^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_function_l3524_352401


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3524_352473

theorem consecutive_integers_product_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) * (n + 3) = 3024 → n + (n + 1) + (n + 2) + (n + 3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3524_352473


namespace NUMINAMATH_CALUDE_consecutive_palindrome_diff_l3524_352457

/-- A function that checks if a number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The set of all five-digit palindromes -/
def five_digit_palindromes : Set ℕ :=
  {n : ℕ | is_five_digit_palindrome n}

/-- The theorem stating the possible differences between consecutive five-digit palindromes -/
theorem consecutive_palindrome_diff 
  (a b : ℕ) 
  (ha : a ∈ five_digit_palindromes) 
  (hb : b ∈ five_digit_palindromes)
  (hless : a < b)
  (hconsec : ∀ x, x ∈ five_digit_palindromes → a < x → x < b → False) :
  b - a = 100 ∨ b - a = 110 ∨ b - a = 11 :=
sorry

end NUMINAMATH_CALUDE_consecutive_palindrome_diff_l3524_352457


namespace NUMINAMATH_CALUDE_ball_box_arrangement_l3524_352454

/-- The number of ways to arrange n balls in n boxes with exactly k balls in their corresponding boxes. -/
def arrangeWithExactMatches (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of derangements of n objects. -/
def derangement (n : ℕ) : ℕ :=
  sorry

theorem ball_box_arrangement :
  arrangeWithExactMatches 5 2 = 20 :=
sorry

end NUMINAMATH_CALUDE_ball_box_arrangement_l3524_352454


namespace NUMINAMATH_CALUDE_jon_points_l3524_352484

theorem jon_points (jon jack tom : ℕ) : 
  (jack = jon + 5) →
  (tom = jon + jack - 4) →
  (jon + jack + tom = 18) →
  (jon = 3) := by
sorry

end NUMINAMATH_CALUDE_jon_points_l3524_352484


namespace NUMINAMATH_CALUDE_sequence_a_10_l3524_352411

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p * a q)

theorem sequence_a_10 (a : ℕ → ℝ) 
  (h_prop : sequence_property a) 
  (h_a8 : a 8 = 16) : 
  a 10 = 32 := by
sorry

end NUMINAMATH_CALUDE_sequence_a_10_l3524_352411


namespace NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l3524_352432

/-- The number of integers from 1 to 60 inclusive -/
def total_numbers : ℕ := 60

/-- The number of multiples of 4 from 1 to 60 inclusive -/
def multiples_of_four : ℕ := 15

/-- The probability of choosing a number that is not a multiple of 4 -/
def prob_not_multiple_of_four : ℚ := (total_numbers - multiples_of_four) / total_numbers

theorem prob_at_least_one_multiple_of_four :
  1 - prob_not_multiple_of_four ^ 2 = 7 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_four_l3524_352432


namespace NUMINAMATH_CALUDE_coin_difference_is_eight_l3524_352431

/-- Represents the available coin denominations in cents -/
def coin_denominations : List Nat := [5, 10, 20, 25]

/-- The amount to be paid in cents -/
def amount_to_pay : Nat := 50

/-- Calculates the minimum number of coins needed to make the given amount -/
def min_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Calculates the maximum number of coins needed to make the given amount -/
def max_coins (amount : Nat) (denominations : List Nat) : Nat :=
  sorry

/-- Proves that the difference between the maximum and minimum number of coins
    needed to make 50 cents using the given denominations is 8 -/
theorem coin_difference_is_eight :
  max_coins amount_to_pay coin_denominations - min_coins amount_to_pay coin_denominations = 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_difference_is_eight_l3524_352431


namespace NUMINAMATH_CALUDE_sophie_joe_marbles_l3524_352481

theorem sophie_joe_marbles (sophie_initial : ℕ) (joe_initial : ℕ) (marbles_given : ℕ) :
  sophie_initial = 120 →
  joe_initial = 19 →
  marbles_given = 16 →
  sophie_initial - marbles_given = 3 * (joe_initial + marbles_given) :=
by
  sorry

end NUMINAMATH_CALUDE_sophie_joe_marbles_l3524_352481


namespace NUMINAMATH_CALUDE_sequence_general_term_l3524_352463

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →
  (a 1 = 1) →
  (∀ n : ℕ, (n + 1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0) →
  (∀ n : ℕ, a n = 1 / n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3524_352463


namespace NUMINAMATH_CALUDE_difference_x_y_l3524_352423

theorem difference_x_y : ∀ (x y : ℤ), x + y = 250 → y = 225 → |x - y| = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l3524_352423


namespace NUMINAMATH_CALUDE_johns_labor_cost_johns_specific_labor_cost_l3524_352442

/-- Represents the problem of calculating labor costs for John's table-making business --/
theorem johns_labor_cost (trees : ℕ) (planks_per_tree : ℕ) (planks_per_table : ℕ) 
  (price_per_table : ℕ) (total_profit : ℕ) : ℕ :=
  let total_planks := trees * planks_per_tree
  let total_tables := total_planks / planks_per_table
  let total_revenue := total_tables * price_per_table
  let labor_cost := total_revenue - total_profit
  labor_cost

/-- The specific instance of John's labor cost calculation --/
theorem johns_specific_labor_cost : 
  johns_labor_cost 30 25 15 300 12000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_johns_labor_cost_johns_specific_labor_cost_l3524_352442


namespace NUMINAMATH_CALUDE_not_right_angled_triangle_l3524_352413

theorem not_right_angled_triangle : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 3) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 2)) ∧
  (a^2 + b^2 ≠ c^2) := by
sorry

end NUMINAMATH_CALUDE_not_right_angled_triangle_l3524_352413


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l3524_352472

theorem isosceles_triangle_condition (A B C : Real) :
  (A > 0) → (B > 0) → (C > 0) → (A + B + C = π) →
  (Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) →
  ∃ (x y : Real), (x = y) ∧ 
  ((A = x ∧ B = y ∧ C = y) ∨ (A = y ∧ B = x ∧ C = y) ∨ (A = y ∧ B = y ∧ C = x)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l3524_352472


namespace NUMINAMATH_CALUDE_discount_restoration_l3524_352421

theorem discount_restoration (original_price : ℝ) (discount_rate : ℝ) (restoration_rate : ℝ) : 
  discount_rate = 0.2 ∧ restoration_rate = 0.25 →
  original_price * (1 - discount_rate) * (1 + restoration_rate) = original_price :=
by sorry

end NUMINAMATH_CALUDE_discount_restoration_l3524_352421


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l3524_352477

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l3524_352477


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_of_2018_l3524_352416

theorem sum_of_prime_factors_of_2018 :
  ∀ p q : ℕ, 
  Prime p → Prime q → p * q = 2018 → p + q = 1011 := by
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_of_2018_l3524_352416


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3524_352487

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n) →
  a 1 + a 2 + a 3 = 2 →
  a 3 + a 4 + a 5 = 8 →
  a 4 + a 5 + a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3524_352487


namespace NUMINAMATH_CALUDE_rational_sqrt_one_minus_xy_l3524_352489

theorem rational_sqrt_one_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ (q : ℚ), q^2 = 1 - x*y := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_one_minus_xy_l3524_352489


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l3524_352412

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c) = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l3524_352412


namespace NUMINAMATH_CALUDE_rotation_result_l3524_352428

-- Define a type for the shapes
inductive Shape
  | Square
  | Pentagon
  | Ellipse

-- Define a type for the positions
inductive Position
  | X
  | Y
  | Z

-- Define a function to represent the initial configuration
def initial_config : Shape → Position
  | Shape.Square => Position.X
  | Shape.Pentagon => Position.Y
  | Shape.Ellipse => Position.Z

-- Define a function to represent the rotation
def rotate_180 (p : Position) : Position :=
  match p with
  | Position.X => Position.Y
  | Position.Y => Position.X
  | Position.Z => Position.Z

-- Theorem statement
theorem rotation_result :
  ∀ (s : Shape),
    rotate_180 (initial_config s) =
      match s with
      | Shape.Square => Position.Y
      | Shape.Pentagon => Position.X
      | Shape.Ellipse => Position.Z
  := by sorry

end NUMINAMATH_CALUDE_rotation_result_l3524_352428


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3524_352479

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 330 = 1) ∧ 
  (∃ (k : ℕ), m^(m : ℕ) = k * n^(n : ℕ)) ∧ 
  (¬∃ (l : ℕ), m = l * n) ∧
  (∀ (p q : ℕ+), 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ (k : ℕ), p^(p : ℕ) = k * q^(q : ℕ)) → 
    (¬∃ (l : ℕ), p = l * q) → 
    (m + n ≤ p + q)) ∧
  (m + n = 182) := by
sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3524_352479


namespace NUMINAMATH_CALUDE_power_of_square_l3524_352491

theorem power_of_square (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l3524_352491


namespace NUMINAMATH_CALUDE_irrational_partner_is_one_l3524_352486

theorem irrational_partner_is_one (a b : ℝ) : 
  (∃ (q : ℚ), a ≠ (q : ℝ)) → -- a is irrational
  (a * b - a - b + 1 = 0) →  -- given equation
  b = 1 :=                   -- conclusion: b equals 1
by sorry

end NUMINAMATH_CALUDE_irrational_partner_is_one_l3524_352486


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_specific_roots_l3524_352419

/-- The quadratic equation x^2 - (k+2)x + 2k - 1 = 0 has two distinct real roots for any real k -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - (k+2)*x₁ + 2*k - 1 = 0 ∧ 
    x₂^2 - (k+2)*x₂ + 2*k - 1 = 0 :=
sorry

/-- When one root of the equation x^2 - (k+2)x + 2k - 1 = 0 is 3, k = 2 and the other root is 1 -/
theorem specific_roots : 
  ∃ k : ℝ, 3^2 - (k+2)*3 + 2*k - 1 = 0 ∧ 
    k = 2 ∧
    1^2 - (k+2)*1 + 2*k - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_specific_roots_l3524_352419


namespace NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3524_352485

theorem original_price_of_discounted_shoes 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : purchase_price = 51)
  (h2 : discount_percentage = 75) : 
  purchase_price / (1 - discount_percentage / 100) = 204 := by
  sorry

end NUMINAMATH_CALUDE_original_price_of_discounted_shoes_l3524_352485


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3524_352435

theorem no_two_cubes_between_squares : ¬ ∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l3524_352435


namespace NUMINAMATH_CALUDE_email_sample_not_representative_l3524_352429

/-- Represents the urban population -/
def UrbanPopulation : Type := Unit

/-- Represents a person in the urban population -/
def Person : Type := Unit

/-- Predicate for whether a person owns an email address -/
def has_email_address (p : Person) : Prop := sorry

/-- Predicate for whether a person uses the internet -/
def uses_internet (p : Person) : Prop := sorry

/-- Predicate for whether a person gets news from the internet -/
def gets_news_from_internet (p : Person) : Prop := sorry

/-- The sample of email address owners -/
def email_sample (n : ℕ) : Set Person := sorry

/-- A sample is representative if it accurately reflects the population characteristics -/
def is_representative (s : Set Person) : Prop := sorry

/-- Theorem stating that the email sample is not representative -/
theorem email_sample_not_representative (n : ℕ) : 
  ¬(is_representative (email_sample n)) := by sorry

end NUMINAMATH_CALUDE_email_sample_not_representative_l3524_352429


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3524_352424

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ α = γ) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 70°
  (α = 70 ∨ β = 70 ∨ γ = 70) →
  -- The vertex angle (the one that's not equal to the other two) is either 70° or 40°
  (((α ≠ β ∧ α ≠ γ) → α = 70 ∨ α = 40) ∧
   ((β ≠ α ∧ β ≠ γ) → β = 70 ∨ β = 40) ∧
   ((γ ≠ α ∧ γ ≠ β) → γ = 70 ∨ γ = 40)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3524_352424


namespace NUMINAMATH_CALUDE_range_of_z_l3524_352425

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  let z := x^2 + 4*y^2
  4 ≤ z ∧ z ≤ 12 := by
sorry

end NUMINAMATH_CALUDE_range_of_z_l3524_352425


namespace NUMINAMATH_CALUDE_inequalities_proof_l3524_352410

theorem inequalities_proof (a b : ℝ) (h : a > b) (h0 : b > 0) :
  (Real.sqrt a > Real.sqrt b) ∧ (a - 1/a > b - 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3524_352410


namespace NUMINAMATH_CALUDE_race_probability_l3524_352422

theorem race_probability (pX pY pZ : ℚ) : 
  pX = 1/4 → pY = 1/12 → pZ = 1/7 → 
  (pX + pY + pZ : ℚ) = 10/21 := by sorry

end NUMINAMATH_CALUDE_race_probability_l3524_352422


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3524_352453

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) : 
  total_votes = 600 → vote_majority = 240 → 
  (70 : ℚ) / 100 * total_votes = (total_votes + vote_majority) / 2 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3524_352453


namespace NUMINAMATH_CALUDE_max_value_of_a_l3524_352405

theorem max_value_of_a (a b c d : ℕ) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ), a' = 8924 ∧ a' < 3 * b' ∧ b' < 4 * c' ∧ c' < 5 * d' ∧ d' < 150 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3524_352405


namespace NUMINAMATH_CALUDE_min_split_links_for_all_weights_l3524_352455

/-- Represents a chain of links -/
structure Chain where
  totalLinks : Nat
  linkWeight : Nat

/-- Represents the result of splitting a chain -/
structure SplitChain where
  originalChain : Chain
  splitLinks : Nat

/-- Checks if all weights from 1 to the total weight can be assembled -/
def canAssembleAllWeights (sc : SplitChain) : Prop :=
  ∀ w : Nat, 1 ≤ w ∧ w ≤ sc.originalChain.totalLinks → 
    ∃ (subset : Finset Nat), subset.card ≤ sc.splitLinks + 1 ∧ 
      (subset.sum (λ i => sc.originalChain.linkWeight)) = w

/-- The main theorem -/
theorem min_split_links_for_all_weights 
  (c : Chain) 
  (h1 : c.totalLinks = 60) 
  (h2 : c.linkWeight = 1) :
  (∃ (k : Nat), k = 3 ∧ 
    canAssembleAllWeights ⟨c, k⟩ ∧
    (∀ (m : Nat), m < k → ¬canAssembleAllWeights ⟨c, m⟩)) :=
  sorry

end NUMINAMATH_CALUDE_min_split_links_for_all_weights_l3524_352455


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3524_352460

theorem sqrt_equation_solution :
  ∀ x : ℝ, x ≥ 0 → x + 4 ≥ 0 → Real.sqrt x + Real.sqrt (x + 4) = 12 → x = 1225 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3524_352460


namespace NUMINAMATH_CALUDE_exists_number_satisfying_condition_l3524_352448

theorem exists_number_satisfying_condition : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_condition_l3524_352448
