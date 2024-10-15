import Mathlib

namespace NUMINAMATH_CALUDE_unique_cube_labeling_l1635_163510

/-- A cube labeling is a function from vertices to integers -/
def CubeLabeling := Fin 8 → Fin 8

/-- A face of the cube is a set of four vertices -/
def CubeFace := Finset (Fin 8)

/-- The set of all faces of a cube -/
def allFaces : Finset CubeFace := sorry

/-- A labeling is valid if it's a bijection (each number used once) -/
def isValidLabeling (l : CubeLabeling) : Prop :=
  Function.Bijective l

/-- The sum of labels on a face equals 22 -/
def faceSum22 (l : CubeLabeling) (face : CubeFace) : Prop :=
  (face.sum (λ v => (l v).val + 1) : ℕ) = 22

/-- All faces of a labeling sum to 22 -/
def allFacesSum22 (l : CubeLabeling) : Prop :=
  ∀ face ∈ allFaces, faceSum22 l face

/-- Two labelings are equivalent if they can be obtained by flipping the cube -/
def equivalentLabelings (l₁ l₂ : CubeLabeling) : Prop := sorry

/-- The main theorem: there is only one unique labeling up to equivalence -/
theorem unique_cube_labeling :
  ∃! l : CubeLabeling, isValidLabeling l ∧ allFacesSum22 l := by sorry

end NUMINAMATH_CALUDE_unique_cube_labeling_l1635_163510


namespace NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l1635_163557

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Theorem: Removing a 1-foot cube from a 4×3×5 feet rectangular solid increases surface area by 2 sq ft -/
theorem surface_area_increase_after_cube_removal 
  (original : RectangularSolid) 
  (removed : Cube) 
  (h1 : original.length = 4)
  (h2 : original.width = 3)
  (h3 : original.height = 5)
  (h4 : removed.side = 1)
  (h5 : removed.side < original.length ∧ removed.side < original.width ∧ removed.side < original.height) :
  surfaceArea original + 2 = surfaceArea original + 
    (removed.side * removed.side + 2 * removed.side * removed.side) - removed.side * removed.side := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l1635_163557


namespace NUMINAMATH_CALUDE_paper_tearing_impossibility_l1635_163580

theorem paper_tearing_impossibility : ¬ ∃ (n : ℕ), 1 + 3 * n = 2007 := by
  sorry

end NUMINAMATH_CALUDE_paper_tearing_impossibility_l1635_163580


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l1635_163547

theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  angle_CBD = 90 →
  angle_ABC + angle_ABD + angle_CBD = 270 →
  angle_ABD = 100 →
  angle_ABC = 80 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l1635_163547


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1635_163572

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (correct_values wrong_values : List ℚ) :
  n = 30 ∧ 
  incorrect_mean = 170 ∧
  correct_values = [190, 200, 175] ∧
  wrong_values = [150, 195, 160] →
  (n * incorrect_mean - wrong_values.sum + correct_values.sum) / n = 172 :=
by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1635_163572


namespace NUMINAMATH_CALUDE_max_children_count_max_children_is_26_l1635_163582

def initial_apples : ℕ := 55
def initial_cookies : ℕ := 114
def initial_chocolates : ℕ := 83

def remaining_apples : ℕ := 3
def remaining_cookies : ℕ := 10
def remaining_chocolates : ℕ := 5

def distributed_apples : ℕ := initial_apples - remaining_apples
def distributed_cookies : ℕ := initial_cookies - remaining_cookies
def distributed_chocolates : ℕ := initial_chocolates - remaining_chocolates

theorem max_children_count : ℕ → Prop :=
  fun n =>
    n > 0 ∧
    distributed_apples % n = 0 ∧
    distributed_cookies % n = 0 ∧
    distributed_chocolates % n = 0 ∧
    ∀ m : ℕ, m > n →
      (distributed_apples % m ≠ 0 ∨
       distributed_cookies % m ≠ 0 ∨
       distributed_chocolates % m ≠ 0)

theorem max_children_is_26 : max_children_count 26 := by sorry

end NUMINAMATH_CALUDE_max_children_count_max_children_is_26_l1635_163582


namespace NUMINAMATH_CALUDE_paper_cutting_game_l1635_163521

theorem paper_cutting_game (n : ℕ) : 
  (8 * n + 1 = 2009) ↔ (n = 251) :=
by sorry

#check paper_cutting_game

end NUMINAMATH_CALUDE_paper_cutting_game_l1635_163521


namespace NUMINAMATH_CALUDE_work_duration_problem_l1635_163507

/-- The problem of determining how long a worker worked on a task before another worker finished it. -/
theorem work_duration_problem 
  (W : ℝ) -- Total work
  (x_rate : ℝ) -- x's work rate per day
  (y_rate : ℝ) -- y's work rate per day
  (y_finish_time : ℝ) -- Time y took to finish the remaining work
  (hx : x_rate = W / 40) -- x's work rate condition
  (hy : y_rate = W / 20) -- y's work rate condition
  (h_finish : y_finish_time = 16) -- y's finish time condition
  : ∃ (d : ℝ), d * x_rate + y_finish_time * y_rate = W ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_duration_problem_l1635_163507


namespace NUMINAMATH_CALUDE_minimum_excellence_rate_l1635_163577

theorem minimum_excellence_rate (total : ℕ) (math_rate : ℚ) (chinese_rate : ℚ) 
  (h_math : math_rate = 70 / 100)
  (h_chinese : chinese_rate = 75 / 100)
  (h_total : total > 0) :
  ∃ (both_rate : ℚ), 
    both_rate ≥ 45 / 100 ∧ 
    both_rate * total ≤ math_rate * total ∧ 
    both_rate * total ≤ chinese_rate * total :=
sorry

end NUMINAMATH_CALUDE_minimum_excellence_rate_l1635_163577


namespace NUMINAMATH_CALUDE_tip_fraction_is_55_93_l1635_163505

/-- Represents the waiter's salary structure over four weeks -/
structure WaiterSalary where
  base : ℚ  -- Base salary
  tips1 : ℚ := 5/3 * base  -- Tips in week 1
  tips2 : ℚ := 3/2 * base  -- Tips in week 2
  tips3 : ℚ := base        -- Tips in week 3
  tips4 : ℚ := 4/3 * base  -- Tips in week 4
  expenses : ℚ := 2/5 * base  -- Total expenses over 4 weeks (10% per week)

/-- Calculates the fraction of total income after expenses that came from tips -/
def tipFraction (s : WaiterSalary) : ℚ :=
  let totalTips := s.tips1 + s.tips2 + s.tips3 + s.tips4
  let totalIncome := 4 * s.base + totalTips
  let incomeAfterExpenses := totalIncome - s.expenses
  totalTips / incomeAfterExpenses

/-- Theorem stating that the fraction of total income after expenses that came from tips is 55/93 -/
theorem tip_fraction_is_55_93 (s : WaiterSalary) : tipFraction s = 55/93 := by
  sorry


end NUMINAMATH_CALUDE_tip_fraction_is_55_93_l1635_163505


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1635_163560

theorem sqrt_x_div_sqrt_y (x y : ℝ) : 
  (((1/3)^2 + (1/4)^2) / ((1/5)^2 + (1/6)^2) = 25*x/(61*y)) → 
  Real.sqrt x / Real.sqrt y = 5/2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1635_163560


namespace NUMINAMATH_CALUDE_fathers_age_when_sum_is_100_l1635_163544

/-- Given a mother aged 42 and a father aged 44, prove that the father will be 51 years old when the sum of their ages is 100. -/
theorem fathers_age_when_sum_is_100 (mother_age father_age : ℕ) 
  (h1 : mother_age = 42) 
  (h2 : father_age = 44) : 
  ∃ (years : ℕ), mother_age + years + (father_age + years) = 100 ∧ father_age + years = 51 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_when_sum_is_100_l1635_163544


namespace NUMINAMATH_CALUDE_right_triangle_c_squared_l1635_163586

theorem right_triangle_c_squared (a b c : ℝ) : 
  a = 9 → b = 12 → (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2) → c^2 = 225 ∨ c^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_c_squared_l1635_163586


namespace NUMINAMATH_CALUDE_equation_positive_root_m_value_l1635_163515

theorem equation_positive_root_m_value (m x : ℝ) : 
  (m / (x^2 - 9) + 2 / (x + 3) = 1 / (x - 3)) → 
  (x > 0) → 
  (m = 6 ∨ m = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_positive_root_m_value_l1635_163515


namespace NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l1635_163566

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 400 * x % 576 = 0 → x ≥ 36 :=
  sorry

theorem thirty_six_satisfies : 400 * 36 % 576 = 0 :=
  sorry

theorem thirty_six_is_smallest : ∃ (x : ℕ), x > 0 ∧ 400 * x % 576 = 0 ∧ x = 36 :=
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_thirty_six_satisfies_thirty_six_is_smallest_l1635_163566


namespace NUMINAMATH_CALUDE_angle_ratio_l1635_163587

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the conditions
axiom trisect : angle A C P = angle P C Q ∧ angle P C Q = angle Q C B
axiom bisect : angle P C M = angle M C Q

-- State the theorem
theorem angle_ratio : 
  (angle M C Q) / (angle A C Q) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_l1635_163587


namespace NUMINAMATH_CALUDE_digit_sum_s_99_l1635_163527

/-- s(n) is the number formed by concatenating the first n perfect squares -/
def s (n : ℕ) : ℕ := sorry

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: The digit sum of s(99) is 4 -/
theorem digit_sum_s_99 : digitSum (s 99) = 4 := by sorry

end NUMINAMATH_CALUDE_digit_sum_s_99_l1635_163527


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1635_163550

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 4) → 
  b = 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1635_163550


namespace NUMINAMATH_CALUDE_total_football_games_l1635_163561

/-- Calculates the total number of football games in a season -/
theorem total_football_games 
  (games_per_month : ℝ) 
  (season_duration : ℝ) 
  (h1 : games_per_month = 323.0)
  (h2 : season_duration = 17.0) :
  games_per_month * season_duration = 5491.0 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l1635_163561


namespace NUMINAMATH_CALUDE_mollys_gift_cost_l1635_163514

/-- Represents the cost and family structure for Molly's gift-sending problem -/
structure GiftSendingProblem where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_sisters : ℕ
  children_per_brother : ℕ
  children_of_sister : ℕ
  num_grandparents : ℕ
  num_cousins : ℕ

/-- Calculates the total number of packages to be sent -/
def total_packages (p : GiftSendingProblem) : ℕ :=
  p.num_parents + p.num_brothers + p.num_sisters +
  (p.num_brothers * p.children_per_brother) +
  p.children_of_sister + p.num_grandparents + p.num_cousins

/-- Calculates the total cost of sending all packages -/
def total_cost (p : GiftSendingProblem) : ℕ :=
  p.cost_per_package * total_packages p

/-- Theorem stating that the total cost for Molly's specific situation is $182 -/
theorem mollys_gift_cost :
  let p : GiftSendingProblem := {
    cost_per_package := 7,
    num_parents := 2,
    num_brothers := 4,
    num_sisters := 1,
    children_per_brother := 3,
    children_of_sister := 2,
    num_grandparents := 2,
    num_cousins := 3
  }
  total_cost p = 182 := by sorry

end NUMINAMATH_CALUDE_mollys_gift_cost_l1635_163514


namespace NUMINAMATH_CALUDE_incorrect_calculation_l1635_163516

theorem incorrect_calculation (h1 : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6)
  (h2 : Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3)
  (h3 : (-Real.sqrt 2)^2 = 2) :
  Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l1635_163516


namespace NUMINAMATH_CALUDE_robbery_participants_l1635_163539

-- Define the suspects
variable (Alexey Boris Veniamin Grigory : Prop)

-- Define the conditions
axiom condition1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom condition2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom condition3 : Grigory → Boris
axiom condition4 : Boris → (Alexey ∨ Veniamin)

-- Theorem to prove
theorem robbery_participants :
  Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin :=
sorry

end NUMINAMATH_CALUDE_robbery_participants_l1635_163539


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1635_163509

theorem quadratic_equation_roots (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let sum_roots := -b / a
  let prod_roots := c / a
  let new_sum := sum_roots + prod_roots
  let new_prod := sum_roots * prod_roots
  f 0 = 0 →
  (∃ x y : ℝ, x + y = new_sum ∧ x * y = new_prod) →
  ∃ k : ℝ, k ≠ 0 ∧ f = λ x => k * (x^2 - new_sum * x + new_prod) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1635_163509


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1635_163576

theorem degree_to_radian_conversion (angle_deg : ℝ) : 
  angle_deg * (π / 180) = -5 * π / 3 ↔ angle_deg = -300 :=
sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1635_163576


namespace NUMINAMATH_CALUDE_pentagon_shaded_probability_l1635_163595

/-- A regular pentagon game board with shaded regions -/
structure PentagonBoard where
  /-- The total number of regions formed by the diagonals -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- Proof that the number of shaded regions is less than or equal to the total regions -/
  h_shaded_le_total : shaded_regions ≤ total_regions

/-- The probability of landing in a shaded region -/
def shaded_probability (board : PentagonBoard) : ℚ :=
  board.shaded_regions / board.total_regions

/-- Theorem stating the probability of landing in a shaded region for the specific game board -/
theorem pentagon_shaded_probability :
  ∃ (board : PentagonBoard),
    board.total_regions = 10 ∧
    board.shaded_regions = 3 ∧
    shaded_probability board = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_shaded_probability_l1635_163595


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l1635_163529

theorem solution_set_of_equations (x y z : ℝ) : 
  (3 * (x^2 + y^2 + z^2) = 1 ∧ 
   x^2*y^2 + y^2*z^2 + z^2*x^2 = x*y*z*(x+y+z)^2) ↔ 
  ((x = 0 ∧ y = 0 ∧ z = Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = 0 ∧ z = -Real.sqrt 3 / 3) ∨
   (x = 0 ∧ y = Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = 0 ∧ y = -Real.sqrt 3 / 3 ∧ z = 0) ∨
   (x = Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = -Real.sqrt 3 / 3 ∧ y = 0 ∧ z = 0) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨
   (x = -1/3 ∧ y = -1/3 ∧ z = -1/3)) := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_equations_l1635_163529


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_gt_zero_l1635_163551

theorem x_eq_one_sufficient_not_necessary_for_x_gt_zero :
  (∃ x : ℝ, x = 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_gt_zero_l1635_163551


namespace NUMINAMATH_CALUDE_wendy_uses_six_products_l1635_163517

/-- The number of facial products Wendy uses -/
def num_products : ℕ := sorry

/-- The time Wendy waits between each product (in minutes) -/
def wait_time : ℕ := 5

/-- The additional time Wendy spends on make-up (in minutes) -/
def makeup_time : ℕ := 30

/-- The total time for Wendy's "full face" routine (in minutes) -/
def total_time : ℕ := 55

/-- Theorem stating that Wendy uses 6 facial products -/
theorem wendy_uses_six_products : num_products = 6 :=
  by sorry

end NUMINAMATH_CALUDE_wendy_uses_six_products_l1635_163517


namespace NUMINAMATH_CALUDE_exam_average_l1635_163581

theorem exam_average (students_group1 : ℕ) (average_group1 : ℚ) 
                      (students_group2 : ℕ) (average_group2 : ℚ) : 
  students_group1 = 15 →
  average_group1 = 73 / 100 →
  students_group2 = 10 →
  average_group2 = 88 / 100 →
  let total_students := students_group1 + students_group2
  let total_score := students_group1 * average_group1 + students_group2 * average_group2
  let overall_average := total_score / total_students
  overall_average = 79 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l1635_163581


namespace NUMINAMATH_CALUDE_find_p_l1635_163528

/-- Given a system of equations with a known solution, prove the value of p. -/
theorem find_p (p q : ℝ) (h1 : p * 2 + q * (-4) = 8) (h2 : 3 * 2 - q * (-4) = 38) : p = 20 := by
  sorry

#check find_p

end NUMINAMATH_CALUDE_find_p_l1635_163528


namespace NUMINAMATH_CALUDE_divides_n_squared_plus_2n_plus_27_l1635_163567

theorem divides_n_squared_plus_2n_plus_27 (n : ℕ) :
  n ∣ (n^2 + 2*n + 27) ↔ n = 1 ∨ n = 3 ∨ n = 9 ∨ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_divides_n_squared_plus_2n_plus_27_l1635_163567


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1635_163573

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 3668 → 
    ¬((y + 7) % 25 = 0 ∧ (y + 7) % 49 = 0 ∧ (y + 7) % 15 = 0 ∧ (y + 7) % 21 = 0)) ∧
  ((3668 + 7) % 25 = 0 ∧ (3668 + 7) % 49 = 0 ∧ (3668 + 7) % 15 = 0 ∧ (3668 + 7) % 21 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1635_163573


namespace NUMINAMATH_CALUDE_inequalities_solution_l1635_163511

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x - 3 * (x - 2) > 4
def inequality2 (x : ℝ) : Prop := (2 * x - 1) / 3 ≤ (x + 1) / 2

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem inequalities_solution :
  ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l1635_163511


namespace NUMINAMATH_CALUDE_max_servings_emily_l1635_163522

/-- Represents the recipe for the smoothie --/
structure Recipe :=
  (servings : ℕ)
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)
  (honey : ℕ)

/-- Represents Emily's available ingredients --/
structure Available :=
  (bananas : ℕ)
  (strawberries : ℕ)
  (yogurt : ℕ)

def recipe : Recipe :=
  { servings := 8
  , bananas := 3
  , strawberries := 2
  , yogurt := 1
  , honey := 4 }

def emily : Available :=
  { bananas := 9
  , strawberries := 8
  , yogurt := 3 }

/-- Calculates the maximum number of servings that can be made --/
def maxServings (r : Recipe) (a : Available) : ℕ :=
  min (a.bananas * r.servings / r.bananas)
      (min (a.strawberries * r.servings / r.strawberries)
           (a.yogurt * r.servings / r.yogurt))

theorem max_servings_emily :
  maxServings recipe emily = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l1635_163522


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l1635_163559

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop := r = 1 / (Real.sin θ + Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ → 
    y = r * Real.sin θ → 
    line_equation x y :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l1635_163559


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1635_163564

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : 
  r^4 + 1/r^4 = 7 := by
sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l1635_163564


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l1635_163558

/-- Given that point P(x₀, y₀) and point A(1, 2) are on opposite sides of the line 3x + 2y - 8 = 0,
    then 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (∃ (ε : ℝ), (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) = -ε ∧ ε > 0) →
  3*x₀ + 2*y₀ > 8 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l1635_163558


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1635_163519

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatingLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / ((10 ^ x.repeatingLength - 1) : ℚ)

/-- The repeating decimal 7.036036036... -/
def number : RepeatingDecimal :=
  { integerPart := 7
    repeatingPart := 36
    repeatingLength := 3 }

theorem repeating_decimal_equals_fraction :
  toRational number = 781 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1635_163519


namespace NUMINAMATH_CALUDE_polar_to_circle_l1635_163555

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (2 * Real.sin θ - Real.cos θ)

/-- The equation of a circle in Cartesian coordinates -/
def circle_equation (x y : ℝ) (h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that the polar equation represents a circle -/
theorem polar_to_circle :
  ∃ h k r, ∀ x y θ,
    polar_equation (Real.sqrt (x^2 + y^2)) θ →
    x = (Real.sqrt (x^2 + y^2)) * Real.cos θ →
    y = (Real.sqrt (x^2 + y^2)) * Real.sin θ →
    circle_equation x y h k r :=
sorry

end NUMINAMATH_CALUDE_polar_to_circle_l1635_163555


namespace NUMINAMATH_CALUDE_product_equals_sum_and_difference_l1635_163575

theorem product_equals_sum_and_difference :
  ∀ a b : ℤ, (a * b = a + b ∧ a * b = a - b) → (a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_equals_sum_and_difference_l1635_163575


namespace NUMINAMATH_CALUDE_petya_winning_strategy_l1635_163562

/-- Represents the state of cups on a 2n-gon -/
def CupState (n : ℕ) := Fin (2 * n) → Bool

/-- Checks if two positions are adjacent on a 2n-gon -/
def adjacent (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + 1) % (2 * n) = j.val ∨ (j.val + 1) % (2 * n) = i.val

/-- Checks if two positions are symmetric with respect to the center of a 2n-gon -/
def symmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + n) % (2 * n) = j.val

/-- Checks if a move is valid in the tea-pouring game -/
def valid_move (n : ℕ) (state : CupState n) (i j : Fin (2 * n)) : Prop :=
  ¬state i ∧ ¬state j ∧ (adjacent n i j ∨ symmetric n i j)

/-- Represents a winning strategy for Petya in the tea-pouring game -/
def petya_wins (n : ℕ) : Prop :=
  ∀ (state : CupState n),
    (∃ (i j : Fin (2 * n)), valid_move n state i j) →
    ∃ (i j : Fin (2 * n)), valid_move n state i j ∧
      ¬(∃ (k l : Fin (2 * n)), valid_move n (Function.update (Function.update state i true) j true) k l)

/-- The main theorem: Petya has a winning strategy if and only if n is odd -/
theorem petya_winning_strategy (n : ℕ) : petya_wins n ↔ Odd n := by
  sorry

end NUMINAMATH_CALUDE_petya_winning_strategy_l1635_163562


namespace NUMINAMATH_CALUDE_equal_cost_at_40_bookshelves_l1635_163588

/-- The number of bookcases to be purchased -/
def num_bookcases : ℕ := 20

/-- The cost of a bookcase in dollars -/
def bookcase_cost : ℕ := 300

/-- The cost of a bookshelf in dollars -/
def bookshelf_cost : ℕ := 100

/-- The discount rate at supermarket B as a fraction -/
def discount_rate : ℚ := 1/5

/-- Calculate the cost at supermarket A -/
def cost_A (x : ℕ) : ℕ := num_bookcases * bookcase_cost + bookshelf_cost * (x - num_bookcases)

/-- Calculate the cost at supermarket B -/
def cost_B (x : ℕ) : ℚ := (1 - discount_rate) * (num_bookcases * bookcase_cost + x * bookshelf_cost)

theorem equal_cost_at_40_bookshelves :
  ∃ x : ℕ, x ≥ num_bookcases ∧ (cost_A x : ℚ) = cost_B x ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_40_bookshelves_l1635_163588


namespace NUMINAMATH_CALUDE_pens_sold_is_226_l1635_163534

/-- Represents the profit and cost structure of a store promotion -/
structure StorePromotion where
  penProfit : ℕ        -- Profit from selling one pen (in yuan)
  bearCost : ℕ         -- Cost of one teddy bear (in yuan)
  pensPerBundle : ℕ    -- Number of pens in a promotion bundle
  totalProfit : ℕ      -- Total profit from the promotion (in yuan)

/-- Calculates the number of pens sold during a store promotion -/
def pensSold (promo : StorePromotion) : ℕ :=
  -- Implementation details are omitted as per instructions
  sorry

/-- Theorem stating that the number of pens sold is 226 for the given promotion -/
theorem pens_sold_is_226 (promo : StorePromotion) 
  (h1 : promo.penProfit = 9)
  (h2 : promo.bearCost = 2)
  (h3 : promo.pensPerBundle = 4)
  (h4 : promo.totalProfit = 1922) : 
  pensSold promo = 226 := by
  sorry

end NUMINAMATH_CALUDE_pens_sold_is_226_l1635_163534


namespace NUMINAMATH_CALUDE_draw_probability_l1635_163513

/-- The probability of player A winning a chess game -/
def prob_A_wins : ℝ := 0.4

/-- The probability that player A does not lose a chess game -/
def prob_A_not_lose : ℝ := 0.9

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem draw_probability :
  prob_draw = 0.5 :=
sorry

end NUMINAMATH_CALUDE_draw_probability_l1635_163513


namespace NUMINAMATH_CALUDE_eulers_formula_l1635_163546

/-- A polyhedron with S vertices, A edges, and F faces, where no four vertices are coplanar. -/
structure Polyhedron where
  S : ℕ  -- number of vertices
  A : ℕ  -- number of edges
  F : ℕ  -- number of faces
  no_four_coplanar : True  -- represents the condition that no four vertices are coplanar

/-- Euler's formula for polyhedra -/
theorem eulers_formula (p : Polyhedron) : p.S + p.F = p.A + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1635_163546


namespace NUMINAMATH_CALUDE_number_of_fives_l1635_163531

theorem number_of_fives (x y : ℕ) : 
  x + y = 20 →
  3 * x + 5 * y = 94 →
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_number_of_fives_l1635_163531


namespace NUMINAMATH_CALUDE_carries_profit_l1635_163540

/-- Carrie's profit from making and decorating a wedding cake -/
theorem carries_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  supply_cost = 54 →
  (hours_per_day * days_worked * hourly_rate - supply_cost : ℕ) = 122 := by
  sorry

end NUMINAMATH_CALUDE_carries_profit_l1635_163540


namespace NUMINAMATH_CALUDE_cos_value_for_special_angle_l1635_163501

theorem cos_value_for_special_angle (θ : Real) 
  (h1 : 6 * Real.tan θ = 2 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_for_special_angle_l1635_163501


namespace NUMINAMATH_CALUDE_tan_5460_deg_equals_sqrt_3_l1635_163523

theorem tan_5460_deg_equals_sqrt_3 : Real.tan (5460 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_5460_deg_equals_sqrt_3_l1635_163523


namespace NUMINAMATH_CALUDE_expression_value_l1635_163579

theorem expression_value (x y z : ℝ) : 
  (abs (x - 2) + (y + 3)^2 = 0) → 
  (z = -1) → 
  (2 * (x^2 * y + x * y * z) - 3 * (x^2 * y - x * y * z) - 4 * x^2 * y = 90) :=
by sorry


end NUMINAMATH_CALUDE_expression_value_l1635_163579


namespace NUMINAMATH_CALUDE_xy_power_2023_l1635_163526

theorem xy_power_2023 (x y : ℝ) (h : |x + 1| + Real.sqrt (y - 1) = 0) : 
  (x * y) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_xy_power_2023_l1635_163526


namespace NUMINAMATH_CALUDE_dogs_equal_initial_l1635_163598

/-- Calculates the remaining number of dogs in an animal rescue center after a series of events. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that the number of remaining dogs equals the initial number under specific conditions. -/
theorem dogs_equal_initial 
  (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) 
  (h1 : initial = 200) 
  (h2 : moved_in = 100) 
  (h3 : first_adoption = 40) 
  (h4 : second_adoption = 60) : 
  remaining_dogs initial moved_in first_adoption second_adoption = initial :=
by sorry

end NUMINAMATH_CALUDE_dogs_equal_initial_l1635_163598


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_square_digits_l1635_163518

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by the square of each of its digits -/
def divisible_by_square_of_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % (d * d) = 0

theorem smallest_four_digit_divisible_by_square_digits :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    has_different_digits n →
    divisible_by_square_of_digits n →
    2268 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_square_digits_l1635_163518


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1635_163525

theorem algebraic_expression_equality (x : ℝ) (h : x = 5) :
  3 / (x - 4) - 24 / (x^2 - 16) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1635_163525


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1635_163530

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l1635_163530


namespace NUMINAMATH_CALUDE_candy_bars_per_box_l1635_163548

/-- Proves that the number of candy bars in each box is 10 given the specified conditions --/
theorem candy_bars_per_box 
  (num_boxes : ℕ) 
  (selling_price buying_price : ℚ)
  (total_profit : ℚ)
  (h1 : num_boxes = 5)
  (h2 : selling_price = 3/2)
  (h3 : buying_price = 1)
  (h4 : total_profit = 25) :
  (total_profit / (num_boxes * (selling_price - buying_price))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_candy_bars_per_box_l1635_163548


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1635_163541

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b in R², if a is perpendicular to b, 
    and a = (1, 2) and b = (x, 1), then x = -2 -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  perpendicular a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1635_163541


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l1635_163565

/-- A positive arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = q * a n

theorem min_value_of_sequence (a : ℕ → ℝ) (m n : ℕ) :
  ArithmeticGeometricSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (1 : ℝ) / m + 4 / n ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l1635_163565


namespace NUMINAMATH_CALUDE_line_parameterization_l1635_163574

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 20*t - 14) → 
  (∀ t : ℝ, g t = 10*t + 13) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l1635_163574


namespace NUMINAMATH_CALUDE_largest_divisor_is_24_l1635_163594

/-- The set of all integer tuples (a, b, c, d, e, f) satisfying a^2 + b^2 + c^2 + d^2 + e^2 = f^2 -/
def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

/-- The property that k divides the product of all elements in a tuple -/
def DividesTuple (k : ℤ) (t : ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) : Prop :=
  let (a, b, c, d, e, f) := t
  k ∣ (a * b * c * d * e * f)

theorem largest_divisor_is_24 :
  ∃ (k : ℤ), k = 24 ∧ (∀ t ∈ S, DividesTuple k t) ∧
  (∀ m : ℤ, (∀ t ∈ S, DividesTuple m t) → m ≤ k) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_is_24_l1635_163594


namespace NUMINAMATH_CALUDE_consecutive_three_digit_prime_factors_l1635_163592

theorem consecutive_three_digit_prime_factors :
  ∀ n : ℕ, 
    100 ≤ n ∧ n + 9 ≤ 999 →
    ∃ (S : Finset ℕ),
      (∀ p ∈ S, Nat.Prime p) ∧
      (Finset.card S ≤ 23) ∧
      (∀ k : ℕ, n ≤ k ∧ k ≤ n + 9 → ∀ p : ℕ, Nat.Prime p → p ∣ k → p ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_three_digit_prime_factors_l1635_163592


namespace NUMINAMATH_CALUDE_margin_calculation_l1635_163597

-- Define the sheet dimensions and side margin
def sheet_width : ℝ := 20
def sheet_length : ℝ := 30
def side_margin : ℝ := 2

-- Define the percentage of the page used for typing
def typing_percentage : ℝ := 0.64

-- Define the function to calculate the typing area
def typing_area (top_bottom_margin : ℝ) : ℝ :=
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin)

-- Define the theorem
theorem margin_calculation :
  ∃ (top_bottom_margin : ℝ),
    typing_area top_bottom_margin = typing_percentage * sheet_width * sheet_length ∧
    top_bottom_margin = 3 := by
  sorry

end NUMINAMATH_CALUDE_margin_calculation_l1635_163597


namespace NUMINAMATH_CALUDE_fraction_equality_l1635_163545

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 14) :
  m / q = 3 / 14 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1635_163545


namespace NUMINAMATH_CALUDE_expression_equality_l1635_163569

theorem expression_equality : 
  4 * (Real.sin (π / 3)) + (1 / 2)⁻¹ - Real.sqrt 12 + |(-3)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1635_163569


namespace NUMINAMATH_CALUDE_max_groups_is_nine_l1635_163590

/-- Represents the number of singers for each voice type -/
structure ChoirComposition :=
  (sopranos : ℕ)
  (altos : ℕ)
  (tenors : ℕ)
  (basses : ℕ)

/-- Represents the ratio of voice types required in each group -/
structure GroupRatio :=
  (soprano_ratio : ℕ)
  (alto_ratio : ℕ)
  (tenor_ratio : ℕ)
  (bass_ratio : ℕ)

/-- Function to calculate the maximum number of complete groups -/
def maxCompleteGroups (choir : ChoirComposition) (ratio : GroupRatio) : ℕ :=
  min (choir.sopranos / ratio.soprano_ratio)
      (min (choir.altos / ratio.alto_ratio)
           (min (choir.tenors / ratio.tenor_ratio)
                (choir.basses / ratio.bass_ratio)))

/-- Theorem stating that the maximum number of complete groups is 9 -/
theorem max_groups_is_nine :
  let choir := ChoirComposition.mk 10 15 12 18
  let ratio := GroupRatio.mk 1 1 1 2
  maxCompleteGroups choir ratio = 9 :=
by
  sorry

#check max_groups_is_nine

end NUMINAMATH_CALUDE_max_groups_is_nine_l1635_163590


namespace NUMINAMATH_CALUDE_rectangle_exists_l1635_163556

/-- A list of the given square side lengths -/
def square_sides : List ℕ := [2, 5, 7, 9, 16, 25, 28, 33, 36]

/-- The total area covered by all squares -/
def total_area : ℕ := (square_sides.map (λ x => x * x)).sum

/-- Proposition: There exists a rectangle with integer dimensions that can be tiled by the given squares -/
theorem rectangle_exists : ∃ (length width : ℕ), 
  length * width = total_area ∧ 
  length > 0 ∧ 
  width > 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_exists_l1635_163556


namespace NUMINAMATH_CALUDE_team_selection_proof_l1635_163570

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 5 players from a team of 9 players, 
    where 2 seeded players must be included -/
def teamSelection : ℕ := sorry

theorem team_selection_proof :
  let totalPlayers : ℕ := 9
  let seededPlayers : ℕ := 2
  let selectCount : ℕ := 5
  teamSelection = choose (totalPlayers - seededPlayers) (selectCount - seededPlayers) := by
  sorry

end NUMINAMATH_CALUDE_team_selection_proof_l1635_163570


namespace NUMINAMATH_CALUDE_certain_number_l1635_163524

theorem certain_number (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_l1635_163524


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1635_163512

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, (a^3 * b - 1) = k * (a + 1)) ∧ 
  (∃ m : ℤ, (a * b^3 + 1) = m * (b - 1)) ↔ 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1635_163512


namespace NUMINAMATH_CALUDE_line_obtuse_angle_a_range_l1635_163552

/-- Given a line passing through points K(1-a, 1+a) and Q(3, 2a),
    if the line forms an obtuse angle, then a is in the open interval (-2, 1). -/
theorem line_obtuse_angle_a_range (a : ℝ) :
  let K : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let m : ℝ := (Q.2 - K.2) / (Q.1 - K.1)
  (m < 0) → a ∈ Set.Ioo (-2 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_line_obtuse_angle_a_range_l1635_163552


namespace NUMINAMATH_CALUDE_coffee_drinkers_possible_values_l1635_163502

def round_table_coffee_problem (n : ℕ) (coffee_drinkers : ℕ) : Prop :=
  n = 14 ∧
  0 < coffee_drinkers ∧
  coffee_drinkers < n ∧
  ∃ (k : ℕ), k > 0 ∧ k < n/2 ∧ coffee_drinkers = n - 2*k

theorem coffee_drinkers_possible_values :
  ∀ (n : ℕ) (coffee_drinkers : ℕ),
    round_table_coffee_problem n coffee_drinkers →
    coffee_drinkers = 6 ∨ coffee_drinkers = 8 ∨ coffee_drinkers = 10 ∨ coffee_drinkers = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_drinkers_possible_values_l1635_163502


namespace NUMINAMATH_CALUDE_find_k_value_l1635_163537

/-- Given two functions f and g, prove that if f(5) - g(5) = 12, then k = -53/5 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 8)
  (hg : ∀ x, g x = x^2 - k * x + 3)
  (h_diff : f 5 - g 5 = 12) : 
  k = -53/5 := by sorry

end NUMINAMATH_CALUDE_find_k_value_l1635_163537


namespace NUMINAMATH_CALUDE_size_and_precision_difference_l1635_163504

/-- Represents the precision of a number -/
inductive Precision
  | Ones
  | Tenths

/-- Represents a number with its value and precision -/
structure NumberWithPrecision where
  value : ℝ
  precision : Precision

/-- The statement that the size and precision of 3.0 and 3 are the same is false -/
theorem size_and_precision_difference : ∃ (a b : NumberWithPrecision), 
  a.value = b.value ∧ a.precision ≠ b.precision := by
  sorry

/-- The numerical value of 3.0 equals 3 -/
axiom value_equality : ∃ (a b : NumberWithPrecision), 
  a.value = 3 ∧ b.value = 3 ∧ a.value = b.value

/-- The precision of 3.0 is to the tenth -/
axiom precision_three_point_zero : ∃ (a : NumberWithPrecision), 
  a.value = 3 ∧ a.precision = Precision.Tenths

/-- The precision of 3 is to 1 -/
axiom precision_three : ∃ (b : NumberWithPrecision), 
  b.value = 3 ∧ b.precision = Precision.Ones

end NUMINAMATH_CALUDE_size_and_precision_difference_l1635_163504


namespace NUMINAMATH_CALUDE_odd_function_property_l1635_163599

-- Define an odd function on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : odd_function f) (h2 : f 2 = 2) :
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1635_163599


namespace NUMINAMATH_CALUDE_total_exercise_time_l1635_163543

def natasha_daily_exercise : ℕ := 30
def natasha_days : ℕ := 7
def esteban_daily_exercise : ℕ := 10
def esteban_days : ℕ := 9
def minutes_per_hour : ℕ := 60

theorem total_exercise_time :
  (natasha_daily_exercise * natasha_days + esteban_daily_exercise * esteban_days) / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_exercise_time_l1635_163543


namespace NUMINAMATH_CALUDE_rainfall_problem_l1635_163563

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_problem_l1635_163563


namespace NUMINAMATH_CALUDE_wire_length_difference_l1635_163503

theorem wire_length_difference (total_length piece1 piece2 : ℝ) : 
  total_length = 30 →
  piece1 = 14 →
  piece2 = 16 →
  |piece2 - piece1| = 2 := by sorry

end NUMINAMATH_CALUDE_wire_length_difference_l1635_163503


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_powers_of_three_l1635_163596

def powers_of_three : List ℕ := [3, 9, 27, 81, 243, 729, 2187, 6561, 19683]

theorem arithmetic_mean_of_powers_of_three :
  (List.sum powers_of_three) / powers_of_three.length = 2970 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_powers_of_three_l1635_163596


namespace NUMINAMATH_CALUDE_plastic_rings_weight_l1635_163583

/-- The weight of the orange ring in ounces -/
def orange_weight : ℝ := 0.08333333333333333

/-- The weight of the purple ring in ounces -/
def purple_weight : ℝ := 0.3333333333333333

/-- The weight of the white ring in ounces -/
def white_weight : ℝ := 0.4166666666666667

/-- The total weight of all rings in ounces -/
def total_weight : ℝ := orange_weight + purple_weight + white_weight

theorem plastic_rings_weight : total_weight = 0.8333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_plastic_rings_weight_l1635_163583


namespace NUMINAMATH_CALUDE_derivative_not_always_constant_l1635_163549

-- Define a real-valued function
def f : ℝ → ℝ := sorry

-- Define the derivative of f at a point x
def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ := sorry

-- Theorem stating that the derivative is not always a constant
theorem derivative_not_always_constant :
  ∃ (f : ℝ → ℝ) (x y : ℝ), x ≠ y → derivative_at f x ≠ derivative_at f y :=
sorry

end NUMINAMATH_CALUDE_derivative_not_always_constant_l1635_163549


namespace NUMINAMATH_CALUDE_middle_number_is_four_or_five_l1635_163553

/-- Represents a triple of positive integers -/
structure IntTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if the triple satisfies all given conditions -/
def satisfiesConditions (t : IntTriple) : Prop :=
  t.a < t.b ∧ t.b < t.c ∧ t.a + t.b + t.c = 15

/-- Represents the set of all possible triples satisfying the conditions -/
def possibleTriples : Set IntTriple :=
  {t : IntTriple | satisfiesConditions t}

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.a = t.a ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.c = t.c ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : IntTriple) : Prop :=
  ∃ t' ∈ possibleTriples, t'.b = t.b ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 or 5 -/
theorem middle_number_is_four_or_five :
  ∀ t ∈ possibleTriples,
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.b = 4 ∨ t.b = 5 :=
sorry

end NUMINAMATH_CALUDE_middle_number_is_four_or_five_l1635_163553


namespace NUMINAMATH_CALUDE_point_coordinates_l1635_163589

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the fourth quadrant -/
def is_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Distance of a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance of a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (P : Point) 
  (h1 : is_fourth_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 5) :
  P.x = 5 ∧ P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1635_163589


namespace NUMINAMATH_CALUDE_uma_income_l1635_163584

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℝ
  expenditure : ℝ

/-- The problem setup -/
def problem_setup (uma bala : Person) : Prop :=
  -- Income ratio
  uma.income / bala.income = 8 / 7 ∧
  -- Expenditure ratio
  uma.expenditure / bala.expenditure = 7 / 6 ∧
  -- Savings
  uma.income - uma.expenditure = 2000 ∧
  bala.income - bala.expenditure = 2000

/-- The theorem to prove -/
theorem uma_income (uma bala : Person) :
  problem_setup uma bala → uma.income = 8000 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_uma_income_l1635_163584


namespace NUMINAMATH_CALUDE_cookie_theorem_l1635_163533

def cookie_problem (initial_cookies eaten_cookies bought_cookies : ℕ) : Prop :=
  eaten_cookies - bought_cookies = 2

theorem cookie_theorem (initial_cookies : ℕ) : 
  cookie_problem initial_cookies 5 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l1635_163533


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l1635_163536

/-- Proves that the equation for average annual growth rate of housing prices is correct -/
theorem housing_price_growth_equation (initial_price final_price : ℝ) (growth_rate : ℝ) 
  (h1 : initial_price = 8100)
  (h2 : final_price = 12500)
  (h3 : growth_rate ≥ 0)
  (h4 : growth_rate < 1) :
  initial_price * (1 + growth_rate)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l1635_163536


namespace NUMINAMATH_CALUDE_roller_coaster_problem_l1635_163520

/-- The number of times a roller coaster must run to accommodate all people in line -/
def roller_coaster_runs (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

/-- Theorem stating that for 84 people in line, 7 cars, and 2 people per car, 6 runs are needed -/
theorem roller_coaster_problem : roller_coaster_runs 84 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_problem_l1635_163520


namespace NUMINAMATH_CALUDE_intersection_theorem_l1635_163578

def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | |x| > 2}

theorem intersection_theorem : A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1635_163578


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l1635_163585

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ a b, a < b → f a < f b

-- Theorem statement
theorem f_is_odd_and_increasing : is_odd f ∧ is_increasing f := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l1635_163585


namespace NUMINAMATH_CALUDE_four_bb_two_divisible_by_nine_l1635_163500

theorem four_bb_two_divisible_by_nine :
  ∃! (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_four_bb_two_divisible_by_nine_l1635_163500


namespace NUMINAMATH_CALUDE_factor_expression_l1635_163554

theorem factor_expression (x : ℝ) : 35 * x^13 + 245 * x^26 = 35 * x^13 * (1 + 7 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1635_163554


namespace NUMINAMATH_CALUDE_one_fourth_more_than_32_5_l1635_163568

theorem one_fourth_more_than_32_5 : (1 / 4 : ℚ) + 32.5 = 32.75 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_more_than_32_5_l1635_163568


namespace NUMINAMATH_CALUDE_smallest_m_for_candies_l1635_163538

theorem smallest_m_for_candies : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < m → ¬(10 ∣ 15*k ∧ 18 ∣ 15*k ∧ 20 ∣ 15*k)) ∧
  (10 ∣ 15*m ∧ 18 ∣ 15*m ∧ 20 ∣ 15*m) ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_for_candies_l1635_163538


namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l1635_163535

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bob_start_delay : ℝ)
  (bob_rate : ℝ)
  (bob_distance : ℝ)
  (h1 : total_distance = 52)
  (h2 : bob_start_delay = 1)
  (h3 : bob_rate = 4)
  (h4 : bob_distance = 28) :
  ∃ (yolanda_rate : ℝ),
    yolanda_rate = 3 ∧
    yolanda_rate * (bob_distance / bob_rate + bob_start_delay) + bob_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_yolandas_walking_rate_l1635_163535


namespace NUMINAMATH_CALUDE_jerry_weller_votes_l1635_163571

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes + john_votes = total_votes ∧
    jerry_votes = john_votes + vote_difference ∧
    jerry_votes = 108375 := by
sorry

end NUMINAMATH_CALUDE_jerry_weller_votes_l1635_163571


namespace NUMINAMATH_CALUDE_green_toads_in_shrublands_l1635_163591

/-- Represents the different types of toads -/
inductive ToadType
| Green
| Brown
| Blue
| Red

/-- Represents the different habitats -/
inductive Habitat
| Wetlands
| Forests
| Grasslands
| Marshlands
| Shrublands

/-- The population ratio of toads -/
def populationRatio : ToadType → ℕ
| ToadType.Green => 1
| ToadType.Brown => 25
| ToadType.Blue => 10
| ToadType.Red => 20

/-- The proportion of brown toads that are spotted -/
def spottedBrownProportion : ℚ := 1/4

/-- The proportion of blue toads that are striped -/
def stripedBlueProportion : ℚ := 1/3

/-- The proportion of red toads with star pattern -/
def starPatternRedProportion : ℚ := 1/2

/-- The density of specific toad types in each habitat -/
def specificToadDensity : Habitat → ℚ
| Habitat.Wetlands => 60  -- spotted brown toads
| Habitat.Forests => 45   -- camouflaged blue toads
| Habitat.Grasslands => 100  -- star pattern red toads
| Habitat.Marshlands => 120  -- plain brown toads
| Habitat.Shrublands => 35   -- striped blue toads

/-- Theorem: The number of green toads per acre in Shrublands is 10.5 -/
theorem green_toads_in_shrublands :
  let totalBlueToads : ℚ := specificToadDensity Habitat.Shrublands / stripedBlueProportion
  let greenToads : ℚ := totalBlueToads / populationRatio ToadType.Blue
  greenToads = 10.5 := by sorry

end NUMINAMATH_CALUDE_green_toads_in_shrublands_l1635_163591


namespace NUMINAMATH_CALUDE_function_range_function_range_with_condition_l1635_163506

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_range (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Ioo 0 1 :=
by sorry

theorem function_range_with_condition (a : ℝ) (h : a ≠ 0) :
  a ≥ 2 → (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_function_range_function_range_with_condition_l1635_163506


namespace NUMINAMATH_CALUDE_five_player_tournament_l1635_163542

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players where each player plays against every other player
    exactly once, the total number of games played is 10. -/
theorem five_player_tournament : tournament_games 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_player_tournament_l1635_163542


namespace NUMINAMATH_CALUDE_seashell_difference_l1635_163532

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of cracked seashells -/
def cracked_seashells : ℕ := 29

/-- Theorem stating the difference between Fred's and Tom's seashell counts -/
theorem seashell_difference : fred_seashells - tom_seashells = 28 := by
  sorry

end NUMINAMATH_CALUDE_seashell_difference_l1635_163532


namespace NUMINAMATH_CALUDE_min_value_sum_of_products_l1635_163508

theorem min_value_sum_of_products (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_products_l1635_163508


namespace NUMINAMATH_CALUDE_point_same_side_as_origin_l1635_163593

def same_side_as_origin (x y : ℝ) : Prop :=
  (3 * x + 2 * y + 5) * (3 * 0 + 2 * 0 + 5) > 0

theorem point_same_side_as_origin :
  same_side_as_origin (-3) 4 := by sorry

end NUMINAMATH_CALUDE_point_same_side_as_origin_l1635_163593
