import Mathlib

namespace NUMINAMATH_CALUDE_hospital_transfer_l2348_234868

theorem hospital_transfer (x : ℝ) (x_pos : x > 0) : 
  let wing_a := x
  let wing_b := 2 * x
  let wing_c := 3 * x
  let occupied_a := (1/3) * wing_a
  let occupied_b := (1/2) * wing_b
  let occupied_c := (1/4) * wing_c
  let max_capacity_b := (3/4) * wing_b
  let max_capacity_c := (5/6) * wing_c
  occupied_a + occupied_b ≤ max_capacity_b →
  (occupied_a + occupied_b) / wing_b = 2/3 ∧ occupied_c / wing_c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_hospital_transfer_l2348_234868


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2348_234866

theorem system_of_equations_solution :
  ∃! (x y z u : ℤ),
    x + y + z = 15 ∧
    x + y + u = 16 ∧
    x + z + u = 18 ∧
    y + z + u = 20 ∧
    x = 3 ∧ y = 5 ∧ z = 7 ∧ u = 8 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2348_234866


namespace NUMINAMATH_CALUDE_fourth_divisor_l2348_234849

theorem fourth_divisor (n : Nat) (h1 : n = 9600) (h2 : n % 15 = 0) (h3 : n % 25 = 0) (h4 : n % 40 = 0) :
  ∃ m : Nat, m = 16 ∧ n % m = 0 ∧ ∀ k : Nat, k > m → n % k = 0 → (k % 15 = 0 ∨ k % 25 = 0 ∨ k % 40 = 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_divisor_l2348_234849


namespace NUMINAMATH_CALUDE_number_accuracy_l2348_234862

-- Define a function to represent the accuracy of a number
def accuracy_place (n : ℝ) : ℕ :=
  sorry

-- Define the number in scientific notation
def number : ℝ := 2.3 * (10 ^ 4)

-- Theorem stating that the number is accurate to the thousands place
theorem number_accuracy :
  accuracy_place number = 3 :=
sorry

end NUMINAMATH_CALUDE_number_accuracy_l2348_234862


namespace NUMINAMATH_CALUDE_equation_solution_l2348_234899

theorem equation_solution (x : ℝ) : (x^2 - 1) / (x + 1) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2348_234899


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l2348_234823

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 640 →
  absent_children = 320 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l2348_234823


namespace NUMINAMATH_CALUDE_total_questions_is_60_l2348_234812

/-- Represents the citizenship test study problem --/
def CitizenshipTestStudy : Prop :=
  let multipleChoice : ℕ := 30
  let fillInBlank : ℕ := 30
  let multipleChoiceTime : ℕ := 15
  let fillInBlankTime : ℕ := 25
  let totalStudyTime : ℕ := 20 * 60

  (multipleChoice * multipleChoiceTime + fillInBlank * fillInBlankTime = totalStudyTime) →
  (multipleChoice + fillInBlank = 60)

/-- Theorem stating that the total number of questions on the test is 60 --/
theorem total_questions_is_60 : CitizenshipTestStudy := by
  sorry

end NUMINAMATH_CALUDE_total_questions_is_60_l2348_234812


namespace NUMINAMATH_CALUDE_hexagonal_prism_intersection_area_l2348_234833

-- Define the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)

-- Define the plane
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

-- Define the area of intersection
def area_of_intersection (prism : HexagonalPrism) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem hexagonal_prism_intersection_area 
  (prism : HexagonalPrism) 
  (plane : Plane) 
  (h1 : prism.height = 5) 
  (h2 : prism.side_length = 6) 
  (h3 : plane.point = (6, 0, 0)) 
  (h4 : (∃ (t : ℝ), plane.point = (-3, 3 * Real.sqrt 3, 5))) 
  (h5 : (∃ (t : ℝ), plane.point = (-3, -3 * Real.sqrt 3, 0))) : 
  area_of_intersection prism plane = 6 * Real.sqrt 399 := by sorry

end NUMINAMATH_CALUDE_hexagonal_prism_intersection_area_l2348_234833


namespace NUMINAMATH_CALUDE_obstacle_course_probability_l2348_234855

def pass_rate_1 : ℝ := 0.8
def pass_rate_2 : ℝ := 0.7
def pass_rate_3 : ℝ := 0.6

theorem obstacle_course_probability :
  let prob_pass_two := pass_rate_1 * pass_rate_2 * (1 - pass_rate_3)
  prob_pass_two = 0.224 := by
sorry

end NUMINAMATH_CALUDE_obstacle_course_probability_l2348_234855


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2348_234821

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_in_first_overs := game.first_run_rate * game.first_overs
  let remaining_runs := game.target - runs_in_first_overs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 3.6)
  (h4 : game.target = 282) :
  required_run_rate game = 6.15 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2348_234821


namespace NUMINAMATH_CALUDE_angle_of_inclination_negative_sqrt_three_line_l2348_234809

theorem angle_of_inclination_negative_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ -Real.sqrt 3 * x + 1
  let slope : ℝ := -Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan (-Real.sqrt 3)
  (0 ≤ angle_of_inclination) ∧ (angle_of_inclination < π) →
  angle_of_inclination = 2 * π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_angle_of_inclination_negative_sqrt_three_line_l2348_234809


namespace NUMINAMATH_CALUDE_count_juggling_sequences_l2348_234836

/-- The number of juggling sequences of length n with exactly 1 ball -/
def jugglingSequences (n : ℕ) : ℕ := 2^n - 1

/-- Theorem: The number of juggling sequences of length n with exactly 1 ball is 2^n - 1 -/
theorem count_juggling_sequences (n : ℕ) : 
  jugglingSequences n = 2^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_count_juggling_sequences_l2348_234836


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_square_roots_l2348_234857

theorem fourth_power_of_nested_square_roots : 
  (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_square_roots_l2348_234857


namespace NUMINAMATH_CALUDE_drawings_on_last_page_is_sixty_l2348_234804

/-- Represents the problem of rearranging drawings in notebooks --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  filled_notebooks : ℕ
  filled_pages_in_last_notebook : ℕ

/-- Calculate the number of drawings on the last page of the partially filled notebook --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let filled_pages := p.filled_notebooks * p.pages_per_notebook + p.filled_pages_in_last_notebook
  let drawings_on_filled_pages := filled_pages * p.new_drawings_per_page
  total_drawings - drawings_on_filled_pages

/-- The main theorem stating that for the given problem, there are 60 drawings on the last page --/
theorem drawings_on_last_page_is_sixty :
  let p : NotebookProblem := {
    initial_notebooks := 5,
    pages_per_notebook := 60,
    initial_drawings_per_page := 8,
    new_drawings_per_page := 12,
    filled_notebooks := 3,
    filled_pages_in_last_notebook := 45
  }
  drawings_on_last_page p = 60 := by
  sorry


end NUMINAMATH_CALUDE_drawings_on_last_page_is_sixty_l2348_234804


namespace NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_is_seven_l2348_234863

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let adult_ticket_cost := (total_revenue - child_ticket_cost * child_tickets) / adult_tickets
  adult_ticket_cost

#check adult_ticket_cost 4 900 5100 400 = 7

theorem adult_ticket_cost_is_seven :
  adult_ticket_cost 4 900 5100 400 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_is_seven_l2348_234863


namespace NUMINAMATH_CALUDE_range_of_sum_reciprocals_l2348_234800

theorem range_of_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_reciprocals_l2348_234800


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2348_234814

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2348_234814


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2348_234892

theorem smallest_positive_integer_congruence : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (5 * x ≡ 18 [MOD 33]) ∧ 
  (x ≡ 4 [MOD 7]) ∧ 
  (∀ (y : ℕ), y > 0 → (5 * y ≡ 18 [MOD 33]) → (y ≡ 4 [MOD 7]) → x ≤ y) ∧
  x = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2348_234892


namespace NUMINAMATH_CALUDE_ab_plus_one_gt_a_plus_b_l2348_234847

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem ab_plus_one_gt_a_plus_b (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_one_gt_a_plus_b_l2348_234847


namespace NUMINAMATH_CALUDE_regression_properties_l2348_234807

def unit_prices : List ℝ := [4, 5, 6, 7, 8, 9]
def sales_volumes : List ℝ := [90, 84, 83, 80, 75, 68]

def empirical_regression (x : ℝ) (a : ℝ) : ℝ := -4 * x + a

theorem regression_properties :
  let avg_sales := (List.sum sales_volumes) / (List.length sales_volumes)
  let slope := -4
  let a := 106
  (avg_sales = 80) ∧
  (∀ x₁ x₂, empirical_regression x₂ a - empirical_regression x₁ a = slope * (x₂ - x₁)) ∧
  (empirical_regression 10 a = 66) := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l2348_234807


namespace NUMINAMATH_CALUDE_min_horizontal_distance_l2348_234811

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

-- Define the set of x-coordinates for points P
def P : Set ℝ := {x | f x = 5}

-- Define the set of x-coordinates for points Q
def Q : Set ℝ := {x | f x = -2}

-- State the theorem
theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ P ∧ q ∈ Q ∧
  ∀ (p' q' : ℝ), p' ∈ P → q' ∈ Q →
  |p - q| ≤ |p' - q'| ∧
  |p - q| = |Real.sqrt 6 - Real.sqrt 3| :=
sorry

end NUMINAMATH_CALUDE_min_horizontal_distance_l2348_234811


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l2348_234870

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l2348_234870


namespace NUMINAMATH_CALUDE_largest_number_in_block_l2348_234864

/-- Represents a 2x3 block of numbers in a 10-column table -/
structure NumberBlock where
  first_number : ℕ
  deriving Repr

/-- The sum of numbers in a 2x3 block -/
def block_sum (block : NumberBlock) : ℕ :=
  6 * block.first_number + 36

theorem largest_number_in_block (block : NumberBlock) 
  (h1 : block.first_number ≥ 1)
  (h2 : block.first_number + 12 ≤ 100)
  (h3 : block_sum block = 480) :
  (block.first_number + 12 = 86) :=
sorry

end NUMINAMATH_CALUDE_largest_number_in_block_l2348_234864


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2348_234856

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2348_234856


namespace NUMINAMATH_CALUDE_simplify_fraction_l2348_234867

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) :
  (x^2 / (x - 2)) - (4 / (x - 2)) = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2348_234867


namespace NUMINAMATH_CALUDE_pump_time_correct_l2348_234896

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 6

/-- The time it takes to fill the tank with both pump and leak -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

/-- Theorem stating that the pump time is correct given the conditions -/
theorem pump_time_correct : 
  (1 / pump_time - 1 / leak_empty_time) = 1 / fill_time_with_leak := by sorry

end NUMINAMATH_CALUDE_pump_time_correct_l2348_234896


namespace NUMINAMATH_CALUDE_decimal_93_to_binary_l2348_234897

def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_93_to_binary :
  decimalToBinary 93 = [1, 0, 1, 1, 1, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_93_to_binary_l2348_234897


namespace NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l2348_234885

/-- Calculates the sum of ages of baby animals in 5 years -/
def sum_of_baby_ages_in_5_years (lioness_age : ℕ) : ℕ :=
  let hyena_age := lioness_age / 2
  let baby_lioness_age := lioness_age / 2
  let baby_hyena_age := hyena_age / 2
  (baby_lioness_age + 5) + (baby_hyena_age + 5)

/-- Theorem stating that the sum of ages of baby animals in 5 years is 19 -/
theorem sum_of_baby_ages_theorem :
  sum_of_baby_ages_in_5_years 12 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_baby_ages_theorem_l2348_234885


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2348_234839

theorem max_value_of_sum_products (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  w + x + y + z = 200 →
  w * x + x * y + y * z + w * z ≤ 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2348_234839


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2348_234802

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2348_234802


namespace NUMINAMATH_CALUDE_ratio_x_to_2y_l2348_234843

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 5 * y) / (x - 2 * y) = 26) : 
  x / (2 * y) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_x_to_2y_l2348_234843


namespace NUMINAMATH_CALUDE_range_of_f_l2348_234889

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.Icc 1 10, ∃ x ∈ Set.Icc 1 5, f x = y ∧
  ∀ x ∈ Set.Icc 1 5, f x ∈ Set.Icc 1 10 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2348_234889


namespace NUMINAMATH_CALUDE_alien_gems_count_l2348_234873

/-- Converts a number from base 6 to base 10 --/
def base6To10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 6^2 + tens * 6^1 + ones * 6^0

/-- The number of gems the alien has --/
def alienGems : ℕ := base6To10 2 5 6

theorem alien_gems_count : alienGems = 108 := by
  sorry

end NUMINAMATH_CALUDE_alien_gems_count_l2348_234873


namespace NUMINAMATH_CALUDE_hero_qin_equivalence_l2348_234876

theorem hero_qin_equivalence (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  let p := (a + b + c) / 2
  Real.sqrt (p * (p - a) * (p - b) * (p - c)) =
  Real.sqrt ((1 / 4) * (a^2 * b^2 - ((a^2 + b^2 + c^2) / 2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_hero_qin_equivalence_l2348_234876


namespace NUMINAMATH_CALUDE_line_proof_l2348_234858

-- Define the lines
def line1 (x y : ℝ) : Prop := 4 * x + 2 * y + 5 = 0
def line2 (x y : ℝ) : Prop := 3 * x - 2 * y + 9 = 0
def line3 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def result_line (x y : ℝ) : Prop := 4 * x - 2 * y + 11 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    result_line x y ∧
    perpendicular
      ((4 : ℝ) / 2) -- Slope of result_line
      ((-1 : ℝ) / 2) -- Slope of line3
  := by sorry

end NUMINAMATH_CALUDE_line_proof_l2348_234858


namespace NUMINAMATH_CALUDE_log_equality_ratio_l2348_234831

theorem log_equality_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧ 
       Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32) : 
  b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) := by
sorry

end NUMINAMATH_CALUDE_log_equality_ratio_l2348_234831


namespace NUMINAMATH_CALUDE_rotated_square_base_vertex_on_line_l2348_234813

/-- Represents a square with side length 2 inches -/
structure Square :=
  (side : ℝ)
  (is_two_inch : side = 2)

/-- Represents the configuration of three squares -/
structure SquareConfiguration :=
  (left : Square)
  (center : Square)
  (right : Square)
  (rotation_angle : ℝ)
  (is_thirty_degrees : rotation_angle = π / 6)

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The base vertex of the rotated square after lowering -/
def base_vertex (config : SquareConfiguration) : Point :=
  sorry

theorem rotated_square_base_vertex_on_line (config : SquareConfiguration) :
  (base_vertex config).y = 0 := by
  sorry

end NUMINAMATH_CALUDE_rotated_square_base_vertex_on_line_l2348_234813


namespace NUMINAMATH_CALUDE_max_red_socks_l2348_234850

theorem max_red_socks (r g : ℕ) : 
  let t := r + g
  (t ≤ 3000) → 
  (r * (r - 1) + g * (g - 1)) / (t * (t - 1)) = 3/5 →
  r ≤ 1199 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l2348_234850


namespace NUMINAMATH_CALUDE_point_on_circle_l2348_234841

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle (P Q : ℝ × ℝ) :
  P = (1, 0) →
  unit_circle P.1 P.2 →
  unit_circle Q.1 Q.2 →
  arc_length (4 * π / 3) = abs (Real.arccos P.1 - Real.arccos Q.1) →
  Q = (-1/2, Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_l2348_234841


namespace NUMINAMATH_CALUDE_chocolate_box_count_l2348_234819

theorem chocolate_box_count : ∀ (total caramels nougats truffles peanut_clusters : ℕ),
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + 6 →
  peanut_clusters = total - (caramels + nougats + truffles) →
  (peanut_clusters : ℚ) / total = 64 / 100 →
  total = 50 := by sorry

end NUMINAMATH_CALUDE_chocolate_box_count_l2348_234819


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2348_234852

/-- Represents the stratified sampling problem --/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  elderly_count : ℕ
  middle_aged_count : ℕ
  young_count : ℕ

/-- Calculates the sample size for a specific group --/
def group_sample_size (s : StratifiedSample) (group_count : ℕ) : ℕ :=
  (group_count * s.sample_size) / s.total_population

/-- Theorem statement for the stratified sampling problem --/
theorem stratified_sampling_theorem (s : StratifiedSample)
  (h1 : s.total_population = s.elderly_count + s.middle_aged_count + s.young_count)
  (h2 : s.total_population = 162)
  (h3 : s.sample_size = 36)
  (h4 : s.elderly_count = 27)
  (h5 : s.middle_aged_count = 54)
  (h6 : s.young_count = 81) :
  group_sample_size s s.elderly_count = 6 ∧
  group_sample_size s s.middle_aged_count = 12 ∧
  group_sample_size s s.young_count = 18 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2348_234852


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2348_234827

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2348_234827


namespace NUMINAMATH_CALUDE_second_derivative_zero_l2348_234871

open Real

/-- Given a differentiable function f and a point x₀ such that 
    the limit of (f(x₀) - f(x₀ + 2Δx)) / Δx as Δx approaches 0 is 2,
    prove that the second derivative of f at x₀ is 0. -/
theorem second_derivative_zero (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f x₀ - f (x₀ + 2*Δx)) / Δx) - 2| < ε) :
  deriv (deriv f) x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_zero_l2348_234871


namespace NUMINAMATH_CALUDE_joe_team_wins_l2348_234898

/-- Represents the number of points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the number of points awarded for a tie -/
def tie_points : ℕ := 1

/-- Represents the number of draws Joe's team had -/
def joe_team_draws : ℕ := 3

/-- Represents the number of wins the first-place team had -/
def first_place_wins : ℕ := 2

/-- Represents the number of ties the first-place team had -/
def first_place_ties : ℕ := 2

/-- Represents the point difference between the first-place team and Joe's team -/
def point_difference : ℕ := 2

/-- Theorem stating that Joe's team won exactly one game -/
theorem joe_team_wins : ℕ := by
  sorry

end NUMINAMATH_CALUDE_joe_team_wins_l2348_234898


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2348_234806

/-- An angle in the fourth quadrant -/
structure FourthQuadrantAngle where
  α : Real
  in_fourth_quadrant : α > -π/2 ∧ α < 0

/-- A point on the terminal side of an angle -/
structure TerminalPoint where
  x : Real
  y : Real

/-- Properties of the angle α -/
structure AngleProperties (α : FourthQuadrantAngle) where
  terminal_point : TerminalPoint
  x_coord : terminal_point.x = 4
  sin_value : Real.sin α.α = terminal_point.y / 5

theorem tan_value_fourth_quadrant (α : FourthQuadrantAngle) 
  (props : AngleProperties α) : Real.tan α.α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2348_234806


namespace NUMINAMATH_CALUDE_elizabeth_stickers_l2348_234854

/-- Represents the number of stickers Elizabeth placed on each water bottle. -/
def stickers_per_bottle (initial_bottles : ℕ) (lost_bottles : ℕ) (stolen_bottles : ℕ) (total_stickers : ℕ) : ℕ :=
  total_stickers / (initial_bottles - lost_bottles - stolen_bottles)

/-- Theorem: Elizabeth placed 3 stickers on each remaining water bottle. -/
theorem elizabeth_stickers :
  stickers_per_bottle 10 2 1 21 = 3 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_stickers_l2348_234854


namespace NUMINAMATH_CALUDE_probability_heart_joker_value_l2348_234893

/-- A deck of cards with 54 cards total, including 13 hearts and 2 jokers -/
structure Deck :=
  (total : Nat)
  (hearts : Nat)
  (jokers : Nat)
  (h_total : total = 54)
  (h_hearts : hearts = 13)
  (h_jokers : jokers = 2)

/-- The probability of drawing a heart first and a joker second from the deck -/
def probability_heart_joker (d : Deck) : ℚ :=
  (d.hearts : ℚ) / d.total * d.jokers / (d.total - 1)

/-- Theorem stating the probability of drawing a heart first and a joker second -/
theorem probability_heart_joker_value (d : Deck) :
  probability_heart_joker d = 13 / 1419 := by
  sorry

#eval (13 : ℚ) / 1419

end NUMINAMATH_CALUDE_probability_heart_joker_value_l2348_234893


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l2348_234883

theorem ceiling_floor_expression : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l2348_234883


namespace NUMINAMATH_CALUDE_dividend_calculation_l2348_234880

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 10 * quotient →
  divisor = 5 * remainder →
  remainder = 46 →
  divisor * quotient + remainder = 5336 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2348_234880


namespace NUMINAMATH_CALUDE_identify_counterfeit_pile_l2348_234888

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : Nat)
  (hasRealCoin : Bool)

/-- Represents the result of weighing two sets of coins -/
inductive WeighResult
  | Equal
  | Unequal

/-- Function to weigh two sets of coins -/
def weigh (pile1 : CoinPile) (pile2 : CoinPile) (count : Nat) : WeighResult :=
  sorry

/-- Theorem stating that it's possible to identify the all-counterfeit pile -/
theorem identify_counterfeit_pile 
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)
  (h1 : pile1.count = 15)
  (h2 : pile2.count = 19)
  (h3 : pile3.count = 25)
  (h4 : pile1.hasRealCoin ∨ pile2.hasRealCoin ∨ pile3.hasRealCoin)
  (h5 : ¬(pile1.hasRealCoin ∧ pile2.hasRealCoin) ∧ 
        ¬(pile1.hasRealCoin ∧ pile3.hasRealCoin) ∧ 
        ¬(pile2.hasRealCoin ∧ pile3.hasRealCoin)) :
  ∃ (p : CoinPile), p ∈ [pile1, pile2, pile3] ∧ ¬p.hasRealCoin :=
sorry

end NUMINAMATH_CALUDE_identify_counterfeit_pile_l2348_234888


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2348_234874

-- Problem 1
theorem problem_1 : -7 - |(-9)| - (-11) - 3 = -8 := by sorry

-- Problem 2
theorem problem_2 : 5.6 + (-0.9) + 4.4 + (-8.1) = 1 := by sorry

-- Problem 3
theorem problem_3 : (-1/6 : ℚ) + (1/3 : ℚ) + (-1/12 : ℚ) = 1/12 := by sorry

-- Problem 4
theorem problem_4 : (2/5 : ℚ) - |(-1.5 : ℚ)| - (2.25 : ℚ) - (-2.75 : ℚ) = -0.6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2348_234874


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2348_234853

/-- A line that always passes through a fixed point regardless of the parameter m -/
def line (m x y : ℝ) : Prop :=
  (m - 1) * x - y + 2 * m + 1 = 0

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line m (fixed_point.1) (fixed_point.2) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2348_234853


namespace NUMINAMATH_CALUDE_hamster_ratio_l2348_234828

/-- Proves that the ratio of male hamsters to total hamsters is 1:3 given the specified conditions --/
theorem hamster_ratio (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) 
  (h1 : total_pets = 92)
  (h2 : total_gerbils = 68)
  (h3 : total_males = 25)
  (h4 : total_gerbils * 1/4 = total_gerbils / 4) -- One-quarter of gerbils are male
  (h5 : total_pets = total_gerbils + (total_pets - total_gerbils)) -- Total pets consist of gerbils and hamsters
  : (total_males - total_gerbils / 4) / (total_pets - total_gerbils) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_hamster_ratio_l2348_234828


namespace NUMINAMATH_CALUDE_prev_geng_yin_year_is_1950_l2348_234882

/-- The number of Heavenly Stems in the Ganzhi system -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches in the Ganzhi system -/
def earthly_branches : ℕ := 12

/-- The year we know to be a Geng-Yin year -/
def known_geng_yin_year : ℕ := 2010

/-- The function to calculate the previous Geng-Yin year -/
def prev_geng_yin_year (current_year : ℕ) : ℕ :=
  current_year - Nat.lcm heavenly_stems earthly_branches

theorem prev_geng_yin_year_is_1950 :
  prev_geng_yin_year known_geng_yin_year = 1950 := by
  sorry

#eval prev_geng_yin_year known_geng_yin_year

end NUMINAMATH_CALUDE_prev_geng_yin_year_is_1950_l2348_234882


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_l2348_234835

theorem contrapositive_square_sum (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_l2348_234835


namespace NUMINAMATH_CALUDE_discount_sales_income_increase_l2348_234879

/-- Proves that a 10% discount with 15% increase in sales volume results in 3.5% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.15) : 
  let new_price := original_price * (1 - discount_rate)
  let new_quantity := original_quantity * (1 + sales_increase_rate)
  let original_income := original_price * original_quantity
  let new_income := new_price * new_quantity
  (new_income - original_income) / original_income = 0.035 := by
sorry

end NUMINAMATH_CALUDE_discount_sales_income_increase_l2348_234879


namespace NUMINAMATH_CALUDE_base6_divisible_by_13_l2348_234801

def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

theorem base6_divisible_by_13 (d : Nat) :
  d ≤ 5 → (base6_to_base10 d % 13 = 0 ↔ d = 5) := by
  sorry

end NUMINAMATH_CALUDE_base6_divisible_by_13_l2348_234801


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l2348_234838

theorem no_solutions_for_equation (x : ℝ) : 
  x > 6 → 
  ¬(Real.sqrt (x + 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l2348_234838


namespace NUMINAMATH_CALUDE_factorial_product_not_perfect_power_l2348_234875

-- Define the factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the product of factorials from 1 to n
def factorial_product (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define a function to check if a number is a perfect power greater than 1
def is_perfect_power (n : ℕ) : Prop :=
  ∃ (base exponent : ℕ), base > 1 ∧ exponent > 1 ∧ base ^ exponent = n

-- State the theorem
theorem factorial_product_not_perfect_power :
  ¬ (is_perfect_power (factorial_product 2022)) :=
sorry

end NUMINAMATH_CALUDE_factorial_product_not_perfect_power_l2348_234875


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2348_234846

theorem four_digit_divisible_by_eleven (B : ℕ) : 
  (4000 + 100 * B + 10 * B + 6) % 11 = 0 → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l2348_234846


namespace NUMINAMATH_CALUDE_product_eleven_one_seventeenth_thirtyfour_l2348_234820

theorem product_eleven_one_seventeenth_thirtyfour : 11 * (1 / 17) * 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_one_seventeenth_thirtyfour_l2348_234820


namespace NUMINAMATH_CALUDE_goods_selection_theorem_l2348_234818

def total_goods : ℕ := 35
def counterfeit_goods : ℕ := 15
def selection_size : ℕ := 3

theorem goods_selection_theorem :
  (Nat.choose (total_goods - 1) (selection_size - 1) = 561) ∧
  (Nat.choose (total_goods - 1) selection_size = 5984) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 = 2100) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 + 
   Nat.choose counterfeit_goods 3 = 2555) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 + 
   Nat.choose counterfeit_goods 1 * Nat.choose (total_goods - counterfeit_goods) 2 + 
   Nat.choose (total_goods - counterfeit_goods) 3 = 6090) := by
  sorry


end NUMINAMATH_CALUDE_goods_selection_theorem_l2348_234818


namespace NUMINAMATH_CALUDE_chord_equation_l2348_234844

/-- Given a parabola and a chord, prove the equation of the line containing the chord -/
theorem chord_equation (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 = -2*y₁) →  -- Point A on parabola
  (x₂^2 = -2*y₂) →  -- Point B on parabola
  (x₁ + x₂ = -2) →  -- Sum of x-coordinates
  ((x₁ + x₂)/2 = -1) →  -- x-coordinate of midpoint
  ((y₁ + y₂)/2 = -5) →  -- y-coordinate of midpoint
  ∃ (m b : ℝ), ∀ x y, (y = m*x + b) ↔ (y - y₁)*(x₂ - x₁) = (x - x₁)*(y₂ - y₁) ∧ m = 1 ∧ b = -4 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l2348_234844


namespace NUMINAMATH_CALUDE_equation_solution_l2348_234895

theorem equation_solution : 
  ∃ (n : ℚ), (2 / (n + 2) + 3 / (n + 2) + n / (n + 2) + 1 / (n + 2) = 4) ∧ (n = -2/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2348_234895


namespace NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l2348_234830

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (∀ x, HasDerivAt (f a) ((3 * a * x^2) + 8 * x + 3) x) →
  HasDerivAt (f a) 2 1 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l2348_234830


namespace NUMINAMATH_CALUDE_inequality_proof_l2348_234861

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2348_234861


namespace NUMINAMATH_CALUDE_total_flowers_sold_l2348_234881

/-- Represents the number of flowers in a bouquet -/
def bouquet_size : ℕ := 12

/-- Represents the total number of bouquets sold -/
def total_bouquets : ℕ := 20

/-- Represents the number of rose bouquets sold -/
def rose_bouquets : ℕ := 10

/-- Represents the number of daisy bouquets sold -/
def daisy_bouquets : ℕ := 10

/-- Theorem stating that the total number of flowers sold is 240 -/
theorem total_flowers_sold : 
  bouquet_size * rose_bouquets + bouquet_size * daisy_bouquets = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_sold_l2348_234881


namespace NUMINAMATH_CALUDE_hoodie_price_l2348_234805

/-- Proves that the price of the hoodie is $80 given the conditions of Celina's hiking equipment purchase. -/
theorem hoodie_price (total_spent : ℝ) (boots_original : ℝ) (boots_discount : ℝ) (flashlight_ratio : ℝ) 
  (h_total : total_spent = 195)
  (h_boots_original : boots_original = 110)
  (h_boots_discount : boots_discount = 0.1)
  (h_flashlight : flashlight_ratio = 0.2) : 
  ∃ (hoodie_price : ℝ), 
    hoodie_price = 80 ∧ 
    (boots_original * (1 - boots_discount) + flashlight_ratio * hoodie_price + hoodie_price = total_spent) :=
by
  sorry


end NUMINAMATH_CALUDE_hoodie_price_l2348_234805


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2348_234815

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n → perpendicular n β → subset m α → perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2348_234815


namespace NUMINAMATH_CALUDE_students_liking_both_channels_l2348_234845

theorem students_liking_both_channels
  (total : ℕ)
  (sports : ℕ)
  (arts : ℕ)
  (neither : ℕ)
  (h1 : total = 100)
  (h2 : sports = 68)
  (h3 : arts = 55)
  (h4 : neither = 3)
  : (sports + arts) - (total - neither) = 26 :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_channels_l2348_234845


namespace NUMINAMATH_CALUDE_cosine_tangent_ratio_equals_two_l2348_234803

theorem cosine_tangent_ratio_equals_two : 
  (Real.cos (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.cos (50 * π / 180)) = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_tangent_ratio_equals_two_l2348_234803


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_minus_2_bounds_l2348_234878

theorem sqrt_3_times_sqrt_6_minus_2_bounds : 2 < Real.sqrt 3 * Real.sqrt 6 - 2 ∧ Real.sqrt 3 * Real.sqrt 6 - 2 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_minus_2_bounds_l2348_234878


namespace NUMINAMATH_CALUDE_intersection_slope_inequality_l2348_234810

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + Real.log x)

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x + 2

-- Theorem statement
theorem intersection_slope_inequality (x₁ x₂ k : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) 
  (h3 : k = (f' x₂ - f' x₁) / (x₂ - x₁)) : 
  x₁ < 1 / k ∧ 1 / k < x₂ := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_slope_inequality_l2348_234810


namespace NUMINAMATH_CALUDE_fraction_inequality_l2348_234817

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2348_234817


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2348_234869

/-- Theorem: For a rectangle with length L and width W, if L/W = 5/2 and L * W = 4000, 
    then the perimeter 2L + 2W = 280. -/
theorem rectangle_perimeter (L W : ℝ) 
    (h1 : L / W = 5 / 2) 
    (h2 : L * W = 4000) : 
  2 * L + 2 * W = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2348_234869


namespace NUMINAMATH_CALUDE_sum_is_zero_l2348_234832

theorem sum_is_zero (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_is_zero_l2348_234832


namespace NUMINAMATH_CALUDE_probability_all_red_fourth_draw_correct_l2348_234840

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing all red balls exactly after the 4th draw -/
def probability_all_red_fourth_draw : ℝ := 0.0434

/-- Theorem stating the probability of drawing all red balls exactly after the 4th draw -/
theorem probability_all_red_fourth_draw_correct :
  probability_all_red_fourth_draw = 
    (initial_red_balls / total_balls) * 
    ((initial_white_balls + 1) / total_balls) * 
    (initial_red_balls / total_balls) * 
    (1 / (initial_white_balls + 1)) := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_fourth_draw_correct_l2348_234840


namespace NUMINAMATH_CALUDE_log_problem_l2348_234834

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_problem (x : ℝ) (h : log 8 (3 * x) = 3) :
  log x 125 = 3 / (9 * log 5 2 - log 5 3) := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2348_234834


namespace NUMINAMATH_CALUDE_sum_of_angles_three_triangles_l2348_234877

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the property that the sum of angles in a triangle is 180°
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define three non-overlapping triangles
variable (A B C : Triangle)

-- Assume each triangle is valid
variable (hA : is_valid_triangle A)
variable (hB : is_valid_triangle B)
variable (hC : is_valid_triangle C)

-- Theorem: The sum of all angles in the three triangles is 540°
theorem sum_of_angles_three_triangles :
  A.angle1 + A.angle2 + A.angle3 +
  B.angle1 + B.angle2 + B.angle3 +
  C.angle1 + C.angle2 + C.angle3 = 540 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_angles_three_triangles_l2348_234877


namespace NUMINAMATH_CALUDE_complex_calculation_l2348_234808

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = 2 - 3*I) :
  3*a + 4*b = 17 - 6*I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2348_234808


namespace NUMINAMATH_CALUDE_point_translation_proof_l2348_234886

def translate_point (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem point_translation_proof :
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := translate_point (translate_point P 0 (-3)) 2 0
  Q = (-1, 1) := by sorry

end NUMINAMATH_CALUDE_point_translation_proof_l2348_234886


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2348_234837

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2348_234837


namespace NUMINAMATH_CALUDE_equation_solution_l2348_234822

theorem equation_solution :
  let x : ℝ := -Real.sqrt 3
  let y : ℝ := 4
  x^2 + 2 * Real.sqrt 3 * x + y - 4 * Real.sqrt y + 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2348_234822


namespace NUMINAMATH_CALUDE_cubic_root_function_l2348_234890

theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * (8 : ℝ)^(1/3) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_function_l2348_234890


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2348_234816

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def possible_roots : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (y : ℤ), polynomial b₂ b₁ x = 0} = possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2348_234816


namespace NUMINAMATH_CALUDE_unique_k_for_inequality_l2348_234848

theorem unique_k_for_inequality :
  ∃! k : ℝ, ∀ t : ℝ, t ∈ Set.Ioo (-1) 1 →
    (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_k_for_inequality_l2348_234848


namespace NUMINAMATH_CALUDE_coeff_x6_eq_30_implies_a_eq_2_l2348_234884

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 -/
def coeff_x6 (a : ℚ) : ℚ := (binomial 10 3 : ℚ) - a * (binomial 10 2 : ℚ)

/-- Theorem: If the coefficient of x^6 in the expansion of (x^2 - a)(x + 1/x)^10 is 30, then a = 2 -/
theorem coeff_x6_eq_30_implies_a_eq_2 :
  coeff_x6 2 = 30 :=
by sorry

end NUMINAMATH_CALUDE_coeff_x6_eq_30_implies_a_eq_2_l2348_234884


namespace NUMINAMATH_CALUDE_total_insects_l2348_234872

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (stones : ℕ) (ants_per_stone : ℕ) 
  (bees : ℕ) (flowers : ℕ) : 
  leaves = 345 → 
  ladybugs_per_leaf = 267 → 
  stones = 178 → 
  ants_per_stone = 423 → 
  bees = 498 → 
  flowers = 6 → 
  leaves * ladybugs_per_leaf + stones * ants_per_stone + bees = 167967 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_l2348_234872


namespace NUMINAMATH_CALUDE_triangle_properties_l2348_234829

/-- Theorem about a triangle ABC with specific angle and side properties -/
theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sqrt 2 * Real.sin A * Real.sin B →
  -- Additional conditions
  Real.cos B = 3 / 5 →
  0 < D ∧ D < 1 →  -- Representing CD = 4BD as D = 4/(1+4) = 4/5
  -- Area condition (using scaled version to avoid square root)
  a * c * D * Real.sin A = 14 / 5 →
  -- Conclusions
  C = π / 4 ∧ a = 2 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2348_234829


namespace NUMINAMATH_CALUDE_abs_eq_self_iff_nonneg_l2348_234887

theorem abs_eq_self_iff_nonneg (x : ℝ) : |x| = x ↔ x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_iff_nonneg_l2348_234887


namespace NUMINAMATH_CALUDE_heptagon_side_sum_l2348_234825

/-- Represents a polygon with 7 vertices --/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- Calculates the area of a polygon --/
def area (p : Heptagon) : ℝ := sorry

/-- Calculates the distance between two points --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem heptagon_side_sum (p : Heptagon) :
  area p = 120 ∧
  distance p.A p.B = 10 ∧
  distance p.B p.C = 15 ∧
  distance p.G p.A = 7 →
  distance p.D p.E + distance p.E p.F = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_side_sum_l2348_234825


namespace NUMINAMATH_CALUDE_zombies_less_than_threshold_days_l2348_234860

/-- The number of zombies in the mall today -/
def current_zombies : ℕ := 480

/-- The threshold number of zombies -/
def threshold : ℕ := 50

/-- The function that calculates the number of zombies n days ago -/
def zombies_n_days_ago (n : ℕ) : ℚ :=
  current_zombies / (2 ^ n : ℚ)

/-- The theorem stating that 4 days ago is when there were less than 50 zombies -/
theorem zombies_less_than_threshold_days : 
  (∃ (n : ℕ), zombies_n_days_ago n < threshold) ∧ 
  (∀ (m : ℕ), m < 4 → zombies_n_days_ago m ≥ threshold) ∧
  zombies_n_days_ago 4 < threshold :=
sorry

end NUMINAMATH_CALUDE_zombies_less_than_threshold_days_l2348_234860


namespace NUMINAMATH_CALUDE_school_sections_l2348_234826

/-- Given a school with 408 boys and 216 girls, prove that when divided into equal sections
    of either boys or girls alone, the total number of sections formed is 26. -/
theorem school_sections (num_boys num_girls : ℕ) 
    (h_boys : num_boys = 408) 
    (h_girls : num_girls = 216) : 
    (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l2348_234826


namespace NUMINAMATH_CALUDE_initial_shells_count_l2348_234824

/-- The number of shells Ed found at the beach -/
def ed_shells : ℕ := 13

/-- The number of shells Jacob found at the beach -/
def jacob_shells : ℕ := ed_shells + 2

/-- The total number of shells after collecting -/
def total_shells : ℕ := 30

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := total_shells - (ed_shells + jacob_shells)

theorem initial_shells_count : initial_shells = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_count_l2348_234824


namespace NUMINAMATH_CALUDE_project_completion_time_l2348_234865

theorem project_completion_time (team_a_time team_b_time team_c_time total_time : ℝ) 
  (h1 : team_a_time = 10)
  (h2 : team_b_time = 15)
  (h3 : team_c_time = 20)
  (h4 : total_time = 6) :
  (1 - (1 / team_b_time + 1 / team_c_time) * total_time) / (1 / team_a_time) = 3 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l2348_234865


namespace NUMINAMATH_CALUDE_shirt_sale_price_l2348_234894

/-- Given a shirt with a cost price, profit margin, and discount percentage,
    calculate the final sale price. -/
def final_sale_price (cost_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_margin)
  selling_price * (1 - discount)

/-- Theorem stating that for a shirt with a cost price of $20, a profit margin of 30%,
    and a discount of 50%, the final sale price is $13. -/
theorem shirt_sale_price :
  final_sale_price 20 0.3 0.5 = 13 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l2348_234894


namespace NUMINAMATH_CALUDE_block_size_correct_l2348_234842

/-- The number of squares on a standard chessboard -/
def standardChessboardSize : Nat := 64

/-- The number of squares removed from the chessboard -/
def removedSquares : Nat := 2

/-- The number of rectangular blocks that can be placed on the modified chessboard -/
def numberOfBlocks : Nat := 30

/-- The size of the rectangular block in squares -/
def blockSize : Nat := 2

/-- Theorem stating that the given block size is correct for the modified chessboard -/
theorem block_size_correct :
  blockSize * numberOfBlocks ≤ standardChessboardSize - removedSquares ∧
  (blockSize + 1) * numberOfBlocks > standardChessboardSize - removedSquares :=
sorry

end NUMINAMATH_CALUDE_block_size_correct_l2348_234842


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2348_234851

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2348_234851


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l2348_234859

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the equation of the ellipse we want to prove
def target_ellipse (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- State the theorem
theorem shortest_major_axis_ellipse :
  ∃ (M : ℝ × ℝ), 
    line_l M.1 M.2 ∧ 
    (∀ (E : ℝ × ℝ → Prop), 
      (∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
        (∀ (x y : ℝ), E (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
        E M ∧ 
        (∀ (x y : ℝ), E (x, y) → 
          Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) + 
          Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2 * a)) →
      (∀ (x y : ℝ), E (x, y) → target_ellipse x y)) :=
sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l2348_234859


namespace NUMINAMATH_CALUDE_circle_tangent_and_point_condition_l2348_234891

-- Define the given points and lines
def point_A : ℝ × ℝ := (0, 3)
def line_l (x : ℝ) : ℝ := 2 * x - 4

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that the center of C is on line l
def center_on_line_l (C : Circle) : Prop :=
  C.center.2 = line_l C.center.1

-- Define the condition that the center of C is on y = x - 1
def center_on_diagonal (C : Circle) : Prop :=
  C.center.2 = C.center.1 - 1

-- Define the tangent line
def is_tangent_line (k b : ℝ) (C : Circle) : Prop :=
  let (cx, cy) := C.center
  (k * cx - cy + b)^2 = (k^2 + 1) * C.radius^2

-- Define the condition |MA| = 2|MO|
def condition_MA_MO (M : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  (mx^2 + (my - 3)^2) = 4 * (mx^2 + my^2)

-- Main theorem
theorem circle_tangent_and_point_condition (C : Circle) :
  C.radius = 1 →
  center_on_line_l C →
  (center_on_diagonal C →
    (∃ k b, is_tangent_line k b C ∧ (k = 0 ∨ (k = -3/4 ∧ b = 3)))) ∧
  (∃ M, condition_MA_MO M → 
    C.center.1 ≥ 0 ∧ C.center.1 ≤ 12/5) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_and_point_condition_l2348_234891
