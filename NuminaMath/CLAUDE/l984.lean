import Mathlib

namespace NUMINAMATH_CALUDE_opposite_silver_is_black_l984_98445

-- Define the colors
inductive Color
  | Yellow
  | Orange
  | Blue
  | Black
  | Silver
  | Pink

-- Define a face of the cube
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  left : Face
  right : Face

-- Define a view of the cube
structure CubeView where
  top : Face
  front : Face
  right : Face

-- Define the theorem
theorem opposite_silver_is_black (c : Cube) 
  (view1 view2 view3 : CubeView)
  (h1 : c.top.color = Color.Black ∧ 
        c.right.color = Color.Blue)
  (h2 : view1.top.color = Color.Black ∧ 
        view1.front.color = Color.Pink ∧ 
        view1.right.color = Color.Blue)
  (h3 : view2.top.color = Color.Black ∧ 
        view2.front.color = Color.Orange ∧ 
        view2.right.color = Color.Blue)
  (h4 : view3.top.color = Color.Black ∧ 
        view3.front.color = Color.Yellow ∧ 
        view3.right.color = Color.Blue)
  (h5 : c.bottom.color = Color.Silver) :
  c.top.color = Color.Black :=
sorry

end NUMINAMATH_CALUDE_opposite_silver_is_black_l984_98445


namespace NUMINAMATH_CALUDE_inverse_24_mod_53_l984_98453

theorem inverse_24_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (24⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_24_mod_53_l984_98453


namespace NUMINAMATH_CALUDE_min_amount_for_equal_distribution_l984_98474

/-- Given initial sheets of paper, number of students, and cost per sheet,
    calculate the minimum amount needed to buy additional sheets for equal distribution. -/
def min_amount_needed (initial_sheets : ℕ) (num_students : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_sheets_needed := (num_students * ((initial_sheets + num_students - 1) / num_students))
  let additional_sheets := total_sheets_needed - initial_sheets
  additional_sheets * cost_per_sheet

/-- Theorem stating that given 98 sheets of paper, 12 students, and a cost of 450 won per sheet,
    the minimum amount needed to buy additional sheets for equal distribution is 4500 won. -/
theorem min_amount_for_equal_distribution :
  min_amount_needed 98 12 450 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_min_amount_for_equal_distribution_l984_98474


namespace NUMINAMATH_CALUDE_job_fair_theorem_l984_98437

/-- Represents a candidate in the job fair --/
structure Candidate where
  correct_answers : ℕ
  prob_correct : ℚ

/-- The job fair scenario --/
structure JobFair where
  total_questions : ℕ
  selected_questions : ℕ
  candidate_a : Candidate
  candidate_b : Candidate

/-- Calculates the probability of a specific sequence of answers for candidate A --/
def prob_sequence (jf : JobFair) : ℚ :=
  (1 - jf.candidate_a.correct_answers / jf.total_questions) *
  (jf.candidate_a.correct_answers / (jf.total_questions - 1)) *
  ((jf.candidate_a.correct_answers - 1) / (jf.total_questions - 2))

/-- Calculates the variance of correct answers for candidate A --/
def variance_a (jf : JobFair) : ℚ := sorry

/-- Calculates the variance of correct answers for candidate B --/
def variance_b (jf : JobFair) : ℚ := sorry

/-- The main theorem to be proved --/
theorem job_fair_theorem (jf : JobFair)
    (h1 : jf.total_questions = 8)
    (h2 : jf.selected_questions = 3)
    (h3 : jf.candidate_a.correct_answers = 6)
    (h4 : jf.candidate_b.prob_correct = 3/4) :
    prob_sequence jf = 5/7 ∧ variance_a jf < variance_b jf := by
  sorry

end NUMINAMATH_CALUDE_job_fair_theorem_l984_98437


namespace NUMINAMATH_CALUDE_flower_baskets_count_l984_98419

/-- The number of baskets used to hold flowers --/
def num_baskets (initial_flowers_per_daughter : ℕ) (additional_flowers : ℕ) (dead_flowers : ℕ) (flowers_per_basket : ℕ) : ℕ :=
  ((2 * initial_flowers_per_daughter + additional_flowers - dead_flowers) / flowers_per_basket)

/-- Theorem stating the number of baskets in the given scenario --/
theorem flower_baskets_count : num_baskets 5 20 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_flower_baskets_count_l984_98419


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l984_98484

theorem triangle_angle_measure (A B C : ℝ) : 
  -- Triangle ABC exists
  -- Angle C is three times angle B
  C = 3 * B →
  -- Angle B is 15°
  B = 15 →
  -- Sum of angles in a triangle is 180°
  A + B + C = 180 →
  -- Prove that angle A is 120°
  A = 120 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l984_98484


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l984_98473

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r h l : ℝ) (S : ℝ),
    r = 3 →
    h = 4 →
    l^2 = r^2 + h^2 →
    S = π * r * l →
    S = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l984_98473


namespace NUMINAMATH_CALUDE_largest_two_digit_number_from_3_and_6_l984_98413

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ((n / 10 = 3 ∧ n % 10 = 6) ∨ (n / 10 = 6 ∧ n % 10 = 3))

theorem largest_two_digit_number_from_3_and_6 :
  ∀ n : ℕ, is_valid_number n → n ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_from_3_and_6_l984_98413


namespace NUMINAMATH_CALUDE_range_of_a_l984_98455

-- Define set A
def A : Set ℝ := {x | x * (4 - x) ≥ 3}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = A → a < 1 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_range_of_a_l984_98455


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l984_98435

theorem greatest_three_digit_divisible_by_3_6_5 :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧
  n = 990 ∧
  ∀ (m : ℕ), 100 ≤ m ∧ m ≤ 999 ∧ 
  m % 3 = 0 ∧ m % 6 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l984_98435


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l984_98422

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l984_98422


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l984_98438

theorem rectangle_perimeter (L W : ℝ) (h1 : L * W = (L + 6) * (W - 2)) (h2 : L * W = (L - 12) * (W + 6)) : 
  2 * (L + W) = 132 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l984_98438


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l984_98441

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x > m) ∧ 
  (∃ x : ℝ, x > m ∧ x^2 - 3*x + 2 ≥ 0) → 
  m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l984_98441


namespace NUMINAMATH_CALUDE_equation_solution_l984_98471

theorem equation_solution : ∃! x : ℝ, Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2 :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 3
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l984_98471


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l984_98401

theorem arithmetic_calculations :
  ((-8) - 5 + (-4) - (-10) = -7) ∧
  (18 - 6 / (-2) * (-1/3) = 17) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l984_98401


namespace NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l984_98447

theorem no_solution_for_x_equals_one (a : ℝ) (h : a ≠ 0) :
  ¬∃ x : ℝ, x = 1 ∧ a^2 * x^2 + (a + 1) * x + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_x_equals_one_l984_98447


namespace NUMINAMATH_CALUDE_race_head_start_l984_98493

/-- Given two runners A and B, where A's speed is 20/15 times B's speed,
    the head start A should give B for a dead heat is 1/4 of the race length. -/
theorem race_head_start (speed_a speed_b race_length head_start : ℝ) :
  speed_a = (20 / 15) * speed_b →
  race_length > 0 →
  speed_a > 0 →
  speed_b > 0 →
  (race_length / speed_a = (race_length - head_start) / speed_b ↔ head_start = (1 / 4) * race_length) :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l984_98493


namespace NUMINAMATH_CALUDE_find_p_l984_98440

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 20

-- State the theorem
theorem find_p : ∃ p : ℝ, f (f (f p)) = 6 ∧ p = 18.25 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l984_98440


namespace NUMINAMATH_CALUDE_johns_allowance_l984_98496

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 92/100 = A) →
  A = 345/100 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l984_98496


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l984_98443

/-- Represents a rectangular piece of paper --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle when folded along its width --/
def perimeterFoldedWidth (r : Rectangle) : ℝ := 2 * r.length + r.width

/-- The perimeter of the rectangle when folded along its length --/
def perimeterFoldedLength (r : Rectangle) : ℝ := 2 * r.width + r.length

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem folded_rectangle_perimeter 
  (r : Rectangle) 
  (h1 : area r = 140)
  (h2 : perimeterFoldedWidth r = 34) :
  perimeterFoldedLength r = 38 := by
  sorry

#check folded_rectangle_perimeter

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l984_98443


namespace NUMINAMATH_CALUDE_numbers_with_seven_from_1_to_800_l984_98459

def contains_seven (n : ℕ) : Bool :=
  sorry

def count_numbers_with_seven (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem numbers_with_seven_from_1_to_800 :
  count_numbers_with_seven 1 800 = 62 :=
sorry

end NUMINAMATH_CALUDE_numbers_with_seven_from_1_to_800_l984_98459


namespace NUMINAMATH_CALUDE_condition_satisfies_equation_l984_98476

theorem condition_satisfies_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_condition_satisfies_equation_l984_98476


namespace NUMINAMATH_CALUDE_min_sum_of_tangent_product_l984_98475

theorem min_sum_of_tangent_product (x y : ℝ) :
  (Real.tan x - 2) * (Real.tan y - 2) = 5 →
  ∃ (min_sum : ℝ), min_sum = Real.pi - Real.arctan (1 / 2) ∧
    ∀ (a b : ℝ), (Real.tan a - 2) * (Real.tan b - 2) = 5 →
      a + b ≥ min_sum := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_tangent_product_l984_98475


namespace NUMINAMATH_CALUDE_cricket_bat_price_l984_98457

theorem cricket_bat_price (profit_A_to_B profit_B_to_C profit_C_to_D final_price : ℝ)
  (h1 : profit_A_to_B = 0.2)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.3)
  (h4 : final_price = 400) :
  ∃ (original_price : ℝ),
    original_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l984_98457


namespace NUMINAMATH_CALUDE_base_6_divisibility_l984_98444

def base_6_to_10 (a b c d : ℕ) : ℕ := a * 6^3 + b * 6^2 + c * 6 + d

theorem base_6_divisibility :
  ∃! (d : ℕ), d < 6 ∧ (base_6_to_10 3 d d 7) % 13 = 0 :=
sorry

end NUMINAMATH_CALUDE_base_6_divisibility_l984_98444


namespace NUMINAMATH_CALUDE_total_paths_is_nine_l984_98478

/-- A graph representing the paths between points A, B, C, and D -/
structure PathGraph where
  paths_AB : ℕ
  paths_BD : ℕ
  paths_DC : ℕ
  direct_AC : ℕ

/-- The total number of paths from A to C in the given graph -/
def total_paths (g : PathGraph) : ℕ :=
  g.paths_AB * g.paths_BD * g.paths_DC + g.direct_AC

/-- Theorem stating that the total number of paths from A to C is 9 -/
theorem total_paths_is_nine (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BD = 2)
  (h3 : g.paths_DC = 2)
  (h4 : g.direct_AC = 1) :
  total_paths g = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_is_nine_l984_98478


namespace NUMINAMATH_CALUDE_a_plus_b_value_l984_98497

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |a| = 4)
  (h2 : Real.sqrt (b^2) = 3)
  (h3 : a + b > 0) :
  a + b = 1 ∨ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l984_98497


namespace NUMINAMATH_CALUDE_line_intersecting_ellipse_l984_98416

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define a line by its slope and y-intercept
def line_equation (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

-- Define what it means for a point to be a midpoint of two other points
def is_midpoint (x₁ y₁ x₂ y₂ x y : ℝ) : Prop := 
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

theorem line_intersecting_ellipse (x₁ y₁ x₂ y₂ : ℝ) :
  is_on_ellipse x₁ y₁ → 
  is_on_ellipse x₂ y₂ → 
  is_midpoint x₁ y₁ x₂ y₂ 1 (1/2) →
  ∃ k b, line_equation k b x₁ y₁ ∧ line_equation k b x₂ y₂ ∧ k = -1 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersecting_ellipse_l984_98416


namespace NUMINAMATH_CALUDE_salary_restoration_l984_98489

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  let restoration_factor := reduced_salary * (1 + 0.25)
  restoration_factor = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l984_98489


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l984_98439

theorem square_area_equal_perimeter_triangle (a b c s : ℝ) : 
  a = 7.5 ∧ b = 9.3 ∧ c = 12.2 → -- triangle side lengths
  s * 4 = a + b + c →           -- equal perimeters
  s * s = 52.5625 :=            -- square area
by sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l984_98439


namespace NUMINAMATH_CALUDE_frank_remaining_money_l984_98469

def calculate_remaining_money (initial_amount : ℕ) 
                              (action_figure_cost : ℕ) (action_figure_count : ℕ)
                              (board_game_cost : ℕ) (board_game_count : ℕ)
                              (puzzle_set_cost : ℕ) (puzzle_set_count : ℕ) : ℕ :=
  initial_amount - 
  (action_figure_cost * action_figure_count + 
   board_game_cost * board_game_count + 
   puzzle_set_cost * puzzle_set_count)

theorem frank_remaining_money :
  calculate_remaining_money 100 12 3 11 2 6 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l984_98469


namespace NUMINAMATH_CALUDE_total_markers_l984_98436

theorem total_markers (red_markers blue_markers : ℕ) :
  red_markers = 41 → blue_markers = 64 → red_markers + blue_markers = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_total_markers_l984_98436


namespace NUMINAMATH_CALUDE_remainder_problem_l984_98450

theorem remainder_problem (L S R : ℕ) (h1 : L - S = 1365) (h2 : S = 270) (h3 : L = 6 * S + R) : R = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l984_98450


namespace NUMINAMATH_CALUDE_product_expansion_sum_l984_98462

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (4 * x^2 - 6 * x + 3) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l984_98462


namespace NUMINAMATH_CALUDE_problem_solution_l984_98442

-- Define the set of possible values
def S : Set ℕ := {0, 1, 3}

-- Define the properties
def prop1 (a b c : ℕ) : Prop := a ≠ 3
def prop2 (a b c : ℕ) : Prop := b = 3
def prop3 (a b c : ℕ) : Prop := c ≠ 0

theorem problem_solution (a b c : ℕ) :
  {a, b, c} = S →
  (prop1 a b c ∨ prop2 a b c ∨ prop3 a b c) →
  (¬(prop1 a b c ∧ prop2 a b c) ∧ ¬(prop1 a b c ∧ prop3 a b c) ∧ ¬(prop2 a b c ∧ prop3 a b c)) →
  100 * a + 10 * b + c = 301 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l984_98442


namespace NUMINAMATH_CALUDE_wax_needed_l984_98492

theorem wax_needed (current_wax total_wax_required : ℕ) 
  (h1 : current_wax = 11)
  (h2 : total_wax_required = 492) : 
  total_wax_required - current_wax = 481 :=
by sorry

end NUMINAMATH_CALUDE_wax_needed_l984_98492


namespace NUMINAMATH_CALUDE_average_monthly_balance_l984_98420

def monthly_balances : List ℕ := [200, 250, 300, 350, 400]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 300 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l984_98420


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l984_98454

/-- Given two hyperbolas with equations x^2/9 - y^2/16 = 1 and y^2/25 - x^2/M = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ x y : ℝ, |y| = (4/3) * |x| ↔ |y| = (5/Real.sqrt M) * |x|) →
  M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l984_98454


namespace NUMINAMATH_CALUDE_remainder_problem_l984_98448

theorem remainder_problem (a b : ℕ) (h1 : a - b = 1311) (h2 : a / b = 11) (h3 : a = 1430) :
  a % b = 121 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l984_98448


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l984_98424

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), (-2 : ℝ) * (x - 5) + 4 * (y + 6) = 0 ↔ y = (1/2 : ℝ) * x - (17/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l984_98424


namespace NUMINAMATH_CALUDE_line_slope_one_m_value_l984_98414

/-- Given a line passing through points P (-2, m) and Q (m, 4) with a slope of 1,
    prove that the value of m is 1. -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_one_m_value_l984_98414


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l984_98423

/-- The mass of a man who causes a boat to sink by a certain amount -/
def man_mass (boat_length boat_width boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_width * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 140 kg -/
theorem man_mass_on_boat :
  let boat_length : ℝ := 7
  let boat_width : ℝ := 2
  let boat_sink_depth : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000    -- kg/m³
  man_mass boat_length boat_width boat_sink_depth water_density = 140 := by
  sorry

#eval man_mass 7 2 0.01 1000

end NUMINAMATH_CALUDE_man_mass_on_boat_l984_98423


namespace NUMINAMATH_CALUDE_arithmetic_operations_possible_l984_98406

noncomputable section

-- Define the machine operation
def machine_op (a b : ℝ) : ℝ := 1 - a / b

-- Define arithmetic operations using the machine operation
def division (a b : ℝ) : ℝ := 1 - machine_op a b

def multiplication (a b : ℝ) : ℝ := 1 - machine_op a (1 / b)

def subtraction (a b : ℝ) : ℝ := machine_op (machine_op a b) (1 / b)

def negation (a b : ℝ) : ℝ := subtraction b (subtraction b a)

def addition (a b : ℝ) : ℝ := subtraction b (negation a b)

-- Theorem stating that all arithmetic operations can be performed
theorem arithmetic_operations_possible :
  ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 →
    (division a b = a / b) ∧
    (multiplication a b = a * b) ∧
    (subtraction b a = b - a) ∧
    (addition a b = a + b) :=
by sorry

end

end NUMINAMATH_CALUDE_arithmetic_operations_possible_l984_98406


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l984_98449

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 70 ∧ 
  Q = 100 ∧ 
  R = S ∧ 
  T = 2 * R + 20 ∧ 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 195 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l984_98449


namespace NUMINAMATH_CALUDE_bernie_selection_probability_l984_98466

theorem bernie_selection_probability 
  (p_carol : ℝ) 
  (p_both : ℝ) 
  (h1 : p_carol = 4/5)
  (h2 : p_both = 0.48)
  (h3 : p_both = p_carol * p_bernie)
  : p_bernie = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_bernie_selection_probability_l984_98466


namespace NUMINAMATH_CALUDE_bananas_per_box_l984_98421

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l984_98421


namespace NUMINAMATH_CALUDE_dave_tickets_l984_98480

theorem dave_tickets (won lost used : ℕ) (h1 : won = 14) (h2 : lost = 2) (h3 : used = 10) :
  won - lost - used = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_l984_98480


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l984_98483

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025 →
  a 3 + a 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l984_98483


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l984_98472

/-- Given two lines that intersect at a specific point, prove the sum of their y-intercepts. -/
theorem intersection_y_intercept_sum (a b : ℝ) : 
  (∀ x y : ℝ, x = 3 * y + a ∧ y = 3 * x + b → x = 4 ∧ y = 1) →
  a + b = -10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l984_98472


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l984_98412

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_gt_3 : n > 3) : 
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l984_98412


namespace NUMINAMATH_CALUDE_time_capsule_depth_relation_l984_98495

/-- Represents the relationship between the depths of time capsules buried by Southton and Northton -/
theorem time_capsule_depth_relation (x y z : ℝ) : 
  (y = 4 * x + z) ↔ (y - 4 * x = z) :=
by sorry

end NUMINAMATH_CALUDE_time_capsule_depth_relation_l984_98495


namespace NUMINAMATH_CALUDE_exponent_division_l984_98432

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l984_98432


namespace NUMINAMATH_CALUDE_quadratic_max_value_l984_98411

theorem quadratic_max_value :
  ∃ (max : ℝ), max = 216 ∧ ∀ (s : ℝ), -3 * s^2 + 54 * s - 27 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l984_98411


namespace NUMINAMATH_CALUDE_wood_sawing_time_l984_98460

/-- Time to saw wood into segments -/
def saw_time (segments : ℕ) (time : ℕ) : Prop :=
  segments > 1 ∧ time = (segments - 1) * (12 / 3)

theorem wood_sawing_time :
  saw_time 4 12 →
  saw_time 8 28 ∧ ¬saw_time 8 24 := by
sorry

end NUMINAMATH_CALUDE_wood_sawing_time_l984_98460


namespace NUMINAMATH_CALUDE_divide_ten_with_difference_five_l984_98468

theorem divide_ten_with_difference_five :
  ∀ x y : ℝ, x + y = 10 ∧ y - x = 5 → x = (5 : ℝ) / 2 ∧ y = (15 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_ten_with_difference_five_l984_98468


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l984_98456

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, 
    a > 0 ∧ 
    a ≠ 1 ∧ 
    (∀ x y : ℝ, x < y → f a x < f a y) →
    a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l984_98456


namespace NUMINAMATH_CALUDE_colored_isosceles_triangle_exists_l984_98415

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A colored vertex in a polygon -/
def ColoredVertex (n : ℕ) (p : RegularPolygon n) := Fin n

/-- Three vertices form an isosceles triangle -/
def IsIsoscelesTriangle (n : ℕ) (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop := sorry

theorem colored_isosceles_triangle_exists 
  (p : RegularPolygon 5000) 
  (colored : Finset (ColoredVertex 5000 p)) 
  (h : colored.card = 2001) : 
  ∃ (v1 v2 v3 : ColoredVertex 5000 p), 
    v1 ∈ colored ∧ v2 ∈ colored ∧ v3 ∈ colored ∧ 
    IsIsoscelesTriangle 5000 p v1 v2 v3 :=
  sorry

end NUMINAMATH_CALUDE_colored_isosceles_triangle_exists_l984_98415


namespace NUMINAMATH_CALUDE_area_of_triangle_AEB_main_theorem_l984_98479

/-- Rectangle ABCD with given dimensions and points -/
structure Rectangle :=
  (A B C D F G E : ℝ × ℝ)
  (ab_length : ℝ)
  (bc_length : ℝ)
  (df_length : ℝ)
  (gc_length : ℝ)

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.ab_length = 7 ∧
  rect.bc_length = 4 ∧
  rect.df_length = 2 ∧
  rect.gc_length = 1 ∧
  rect.F.2 = rect.C.2 ∧
  rect.G.2 = rect.C.2 ∧
  rect.A.1 = rect.D.1 ∧
  rect.B.1 = rect.C.1 ∧
  rect.A.2 = rect.B.2 ∧
  rect.C.2 = rect.D.2 ∧
  (rect.E.1 - rect.A.1) / (rect.B.1 - rect.A.1) = (rect.F.1 - rect.D.1) / (rect.C.1 - rect.D.1) ∧
  (rect.E.1 - rect.B.1) / (rect.A.1 - rect.B.1) = (rect.G.1 - rect.C.1) / (rect.D.1 - rect.C.1)

/-- Theorem: The area of triangle AEB is 22.4 -/
theorem area_of_triangle_AEB (rect : Rectangle) 
  (h : rectangle_conditions rect) : ℝ :=
  22.4

/-- Main theorem: If the rectangle satisfies the given conditions, 
    then the area of triangle AEB is 22.4 -/
theorem main_theorem (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  area_of_triangle_AEB rect h = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_AEB_main_theorem_l984_98479


namespace NUMINAMATH_CALUDE_touching_x_axis_with_max_value_l984_98463

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem touching_x_axis_with_max_value (a b m : ℝ) :
  m ≠ 0 →
  f a b m = 0 →
  f' a b m = 0 →
  (∀ x, f a b x ≤ 1/2) →
  (∃ x, f a b x = 1/2) →
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_touching_x_axis_with_max_value_l984_98463


namespace NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l984_98417

theorem sqrt_x_plus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 1) ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_one_real_l984_98417


namespace NUMINAMATH_CALUDE_basketball_scoring_ratio_l984_98430

/-- The basketball scoring problem -/
theorem basketball_scoring_ratio :
  ∀ (first_away second_away third_away last_home next_game : ℕ),
  second_away = first_away + 18 →
  third_away = second_away + 2 →
  last_home = 62 →
  first_away + second_away + third_away + last_home + next_game = 4 * last_home →
  next_game = 55 →
  (last_home : ℚ) / first_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_scoring_ratio_l984_98430


namespace NUMINAMATH_CALUDE_final_amoeba_type_l984_98482

/-- Represents the type of a Martian amoeba -/
inductive AmoebaTy
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebaPop where
  a : Nat
  b : Nat
  c : Nat

/-- Merges two amoebas of different types into the third type -/
def merge (pop : AmoebaPop) : AmoebaPop :=
  sorry

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop :=
  n % 2 = 1

/-- The initial population of amoebas -/
def initialPop : AmoebaPop :=
  { a := 20, b := 21, c := 22 }

theorem final_amoeba_type (finalPop : AmoebaPop)
    (h : ∃ n : Nat, finalPop = (merge^[n] initialPop))
    (hTotal : finalPop.a + finalPop.b + finalPop.c = 1) :
    isOdd finalPop.b ∧ ¬isOdd finalPop.a ∧ ¬isOdd finalPop.c :=
  sorry

end NUMINAMATH_CALUDE_final_amoeba_type_l984_98482


namespace NUMINAMATH_CALUDE_man_pants_count_l984_98426

theorem man_pants_count (t_shirts : ℕ) (total_ways : ℕ) (pants : ℕ) : 
  t_shirts = 8 → 
  total_ways = 72 → 
  total_ways = t_shirts * pants → 
  pants = 9 := by sorry

end NUMINAMATH_CALUDE_man_pants_count_l984_98426


namespace NUMINAMATH_CALUDE_sunglasses_profit_ratio_l984_98428

theorem sunglasses_profit_ratio (selling_price cost_price : ℚ) (pairs_sold : ℕ) (sign_cost : ℚ) :
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  sign_cost = 20 →
  sign_cost / ((selling_price - cost_price) * pairs_sold) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_profit_ratio_l984_98428


namespace NUMINAMATH_CALUDE_horner_method_f_3_l984_98409

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ :=
  horner [7, 6, 5, 4, 3, 2, 1, 0] x

theorem horner_method_f_3 :
  f 3 = 21324 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l984_98409


namespace NUMINAMATH_CALUDE_square_remainder_l984_98467

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l984_98467


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l984_98403

theorem modulus_of_complex_number (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 + 1 / i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l984_98403


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l984_98408

/-- Represents the probability of a team member answering correctly -/
structure TeamMember where
  prob_correct : ℝ
  prob_correct_nonneg : 0 ≤ prob_correct
  prob_correct_le_one : prob_correct ≤ 1

/-- Represents a team in the knowledge competition -/
structure Team where
  member_a : TeamMember
  member_b : TeamMember

/-- The total score of a team in the competition -/
inductive TotalScore
  | zero
  | ten
  | twenty
  | thirty

def prob_first_correct (team : Team) : ℝ :=
  team.member_a.prob_correct + (1 - team.member_a.prob_correct) * team.member_b.prob_correct

def prob_distribution (team : Team) : TotalScore → ℝ
  | TotalScore.zero => (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.ten => prob_first_correct team * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.twenty => (prob_first_correct team)^2 * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.thirty => (prob_first_correct team)^3

theorem knowledge_competition_probabilities (team : Team)
  (h_a : team.member_a.prob_correct = 2/5)
  (h_b : team.member_b.prob_correct = 2/3) :
  prob_first_correct team = 4/5 ∧
  prob_distribution team TotalScore.zero = 1/5 ∧
  prob_distribution team TotalScore.ten = 4/25 ∧
  prob_distribution team TotalScore.twenty = 16/125 ∧
  prob_distribution team TotalScore.thirty = 64/125 := by
  sorry

#check knowledge_competition_probabilities

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l984_98408


namespace NUMINAMATH_CALUDE_z_minimum_l984_98490

/-- The function z(x, y) defined in the problem -/
def z (x y : ℝ) : ℝ := x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3

/-- Theorem stating the minimum value of z and where it occurs -/
theorem z_minimum :
  (∀ x y : ℝ, z x y ≥ 1) ∧ (z 0 (-1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_z_minimum_l984_98490


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l984_98488

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l984_98488


namespace NUMINAMATH_CALUDE_new_students_average_age_l984_98425

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 15)
  (h3 : new_students = 15)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_strength * original_average
  let total_new := (original_strength + new_students) * new_average - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end NUMINAMATH_CALUDE_new_students_average_age_l984_98425


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l984_98470

/-- Given that Kelly has a certain number of Nintendo games and wants to keep a specific number,
    prove that the number of games she needs to give away is the difference between these two numbers. -/
theorem kelly_games_to_give_away (initial_nintendo_games kept_nintendo_games : ℕ) :
  initial_nintendo_games ≥ kept_nintendo_games →
  initial_nintendo_games - kept_nintendo_games =
  initial_nintendo_games - kept_nintendo_games :=
by
  sorry

#check kelly_games_to_give_away 20 12

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l984_98470


namespace NUMINAMATH_CALUDE_license_plate_count_l984_98446

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 20

/-- The number of vowels (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits -/
def num_digits : ℕ := 10

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_consonants * num_digits * num_vowels * (num_consonants - 1)

/-- Theorem stating the number of possible license plates -/
theorem license_plate_count : num_license_plates = 22800 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l984_98446


namespace NUMINAMATH_CALUDE_max_sum_removed_numbers_l984_98491

theorem max_sum_removed_numbers (n : ℕ) (m k : ℕ) 
  (h1 : n > 2) 
  (h2 : 1 < m ∧ m < n) 
  (h3 : 1 < k ∧ k < n) 
  (h4 : (n * (n + 1) / 2 - m - k) / (n - 2) = 17) :
  m + k ≤ 51 ∧ ∃ (m' k' : ℕ), 1 < m' ∧ m' < n ∧ 1 < k' ∧ k' < n ∧ m' + k' = 51 := by
  sorry

#check max_sum_removed_numbers

end NUMINAMATH_CALUDE_max_sum_removed_numbers_l984_98491


namespace NUMINAMATH_CALUDE_savings_distribution_l984_98465

/-- Represents the savings and debt problem of Tamara, Nora, and Lulu -/
theorem savings_distribution (debt : ℕ) (lulu_savings : ℕ) : 
  debt = 40 →
  lulu_savings = 6 →
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  let remainder := total_savings - debt
  remainder / 3 = 2 := by sorry

end NUMINAMATH_CALUDE_savings_distribution_l984_98465


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l984_98405

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9 -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → 
  circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l984_98405


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_value_l984_98452

theorem quadratic_roots_imply_a_value (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 3) * x^2 + 5 * x - 2 = 0 ↔ (x = 1/2 ∨ x = 2)) →
  (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_value_l984_98452


namespace NUMINAMATH_CALUDE_range_of_H_l984_98404

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H :
  Set.range H = {-4, 4} := by sorry

end NUMINAMATH_CALUDE_range_of_H_l984_98404


namespace NUMINAMATH_CALUDE_find_d_l984_98494

theorem find_d : ∃ d : ℝ, 
  (∃ x : ℤ, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧ 
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0) ∧
  d = 3.2 := by
sorry

end NUMINAMATH_CALUDE_find_d_l984_98494


namespace NUMINAMATH_CALUDE_equation_solution_l984_98485

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l984_98485


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l984_98461

theorem cos_alpha_plus_pi_fourth (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/4) = 7 * Real.sqrt 2 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l984_98461


namespace NUMINAMATH_CALUDE_coopers_age_l984_98458

/-- Given the ages of four people with specific relationships, prove Cooper's age --/
theorem coopers_age (cooper dante maria emily : ℕ) : 
  cooper + dante + maria + emily = 62 →
  dante = 2 * cooper →
  maria = dante + 1 →
  emily = 3 * cooper →
  cooper = 8 :=
by sorry

end NUMINAMATH_CALUDE_coopers_age_l984_98458


namespace NUMINAMATH_CALUDE_safari_count_l984_98498

theorem safari_count (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 710) : ∃ (birds mammals tripeds : ℕ),
  birds + mammals + tripeds = total_heads ∧
  2 * birds + 4 * mammals + 3 * tripeds = total_legs ∧
  birds = 139 := by
  sorry

end NUMINAMATH_CALUDE_safari_count_l984_98498


namespace NUMINAMATH_CALUDE_max_intersections_8_6_l984_98486

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 8 x-axis points and 6 y-axis points -/
theorem max_intersections_8_6 :
  max_intersections 8 6 = 420 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_8_6_l984_98486


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l984_98407

theorem cylinder_height_relation (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l984_98407


namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l984_98481

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2*p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l984_98481


namespace NUMINAMATH_CALUDE_bad_carrots_count_l984_98499

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) 
  (h1 : olivia_carrots = 20)
  (h2 : mom_carrots = 14)
  (h3 : good_carrots = 19) :
  olivia_carrots + mom_carrots - good_carrots = 15 :=
by sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l984_98499


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l984_98431

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  (x^2 - 17*x + 8 = 0) → 
  ∃ r₁ r₂ : ℝ, (r₁ ≠ 0 ∧ r₂ ≠ 0) ∧ 
             (x - r₁) * (x - r₂) = x^2 - 17*x + 8 ∧ 
             1/r₁ + 1/r₂ = 17/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l984_98431


namespace NUMINAMATH_CALUDE_average_hours_upside_down_per_month_l984_98402

/-- The number of inches Alex needs to grow to ride the roller coaster -/
def height_difference : ℚ := 54 - 48

/-- Alex's normal growth rate in inches per month -/
def normal_growth_rate : ℚ := 1 / 3

/-- Alex's growth rate in inches per hour when hanging upside down -/
def upside_down_growth_rate : ℚ := 1 / 12

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating the average number of hours Alex needs to hang upside down per month -/
theorem average_hours_upside_down_per_month :
  (height_difference - normal_growth_rate * months_per_year) / (upside_down_growth_rate * months_per_year) = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_upside_down_per_month_l984_98402


namespace NUMINAMATH_CALUDE_number_system_existence_l984_98418

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) (value : ℕ) : Prop :=
  value < base

/-- Addition in a given base --/
def BaseAdd (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a + b) % base = c

/-- Multiplication in a given base --/
def BaseMult (base : ℕ) (a b c : ℕ) : Prop :=
  BaseNumber base a ∧ BaseNumber base b ∧ BaseNumber base c ∧
  (a * b) % base = c

theorem number_system_existence :
  (∃ b : ℕ, BaseAdd b 3 4 10 ∧ BaseMult b 3 4 15) ∧
  (¬ ∃ b : ℕ, BaseAdd b 2 3 5 ∧ BaseMult b 2 3 11) := by
  sorry

end NUMINAMATH_CALUDE_number_system_existence_l984_98418


namespace NUMINAMATH_CALUDE_M_intersect_N_is_empty_l984_98464

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (2*x - x^2)}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_is_empty_l984_98464


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l984_98429

/-- Calculates the alcohol percentage in a new mixture after adding water to an alcohol solution -/
theorem alcohol_percentage_after_dilution
  (original_volume : ℝ)
  (original_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (h1 : original_volume = 9)
  (h2 : original_alcohol_percentage = 57)
  (h3 : added_water = 3) :
  let original_alcohol_volume := original_volume * (original_alcohol_percentage / 100)
  let new_total_volume := original_volume + added_water
  let new_alcohol_percentage := (original_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 42.75 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l984_98429


namespace NUMINAMATH_CALUDE_function_inequality_l984_98410

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x - 2) * deriv f x ≥ 0) : 
  f 1 + f 3 ≥ 2 * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l984_98410


namespace NUMINAMATH_CALUDE_max_area_rectangle_l984_98451

/-- A rectangle with a given perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The rectangle with maximum area among all rectangles with perimeter 40 is a square with sides 10 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ area { length := 10, width := 10, perimeter_constraint := by norm_num } :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l984_98451


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l984_98427

/-- Calculates the total cost of fruits with applied discounts -/
def total_cost_with_discounts (apple_kg : ℝ) (apple_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
  (orange_kg : ℝ) (orange_price : ℝ) (banana_kg : ℝ) (banana_price : ℝ)
  (apple_discount : ℝ) (orange_discount : ℝ) : ℝ :=
  let apple_cost := apple_kg * apple_price * (1 - apple_discount)
  let mango_cost := mango_kg * mango_price
  let orange_cost := orange_kg * orange_price * (1 - orange_discount)
  let banana_cost := banana_kg * banana_price
  apple_cost + mango_cost + orange_cost + banana_cost

/-- Theorem stating that the total cost of Tom's fruit purchase is $1391.5 -/
theorem toms_fruit_purchase_cost :
  total_cost_with_discounts 8 70 9 65 5 50 3 30 0.1 0.15 = 1391.5 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l984_98427


namespace NUMINAMATH_CALUDE_mike_pens_l984_98487

theorem mike_pens (initial_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ) :
  initial_pens = 7 →
  sharon_pens = 19 →
  final_pens = 39 →
  ∃ M : ℕ, 2 * (initial_pens + M) - sharon_pens = final_pens ∧ M = 22 :=
by sorry

end NUMINAMATH_CALUDE_mike_pens_l984_98487


namespace NUMINAMATH_CALUDE_third_month_sale_l984_98400

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 7991
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l984_98400


namespace NUMINAMATH_CALUDE_x_twelve_percent_greater_than_seventy_l984_98434

theorem x_twelve_percent_greater_than_seventy (x : ℝ) : 
  x = 70 * (1 + 12 / 100) → x = 78.4 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_percent_greater_than_seventy_l984_98434


namespace NUMINAMATH_CALUDE_dave_race_walking_time_l984_98477

theorem dave_race_walking_time 
  (total_time : ℕ) 
  (jogging_ratio : ℕ) 
  (walking_ratio : ℕ) 
  (h1 : total_time = 21)
  (h2 : jogging_ratio = 4)
  (h3 : walking_ratio = 3) :
  (walking_ratio * total_time) / (jogging_ratio + walking_ratio) = 9 := by
sorry


end NUMINAMATH_CALUDE_dave_race_walking_time_l984_98477


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l984_98433

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l984_98433
