import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_forall_not_equal_l3157_315798

theorem negation_of_forall_not_equal (x : ℝ) :
  (¬ ∀ x > 0, Real.log x ≠ x - 1) ↔ (∃ x > 0, Real.log x = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_not_equal_l3157_315798


namespace NUMINAMATH_CALUDE_series_convergence_l3157_315713

def series_term (n : ℕ) : ℚ :=
  (4 * n + 3) / ((4 * n - 2)^2 * (4 * n + 2)^2)

def series_sum : ℚ := 1 / 128

theorem series_convergence : 
  (∑' n, series_term n) = series_sum :=
sorry

end NUMINAMATH_CALUDE_series_convergence_l3157_315713


namespace NUMINAMATH_CALUDE_expression_equals_two_l3157_315737

theorem expression_equals_two :
  (-1)^2023 - 2 * Real.sin (π / 3) + |(-Real.sqrt 3)| + (1/3)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l3157_315737


namespace NUMINAMATH_CALUDE_sum_of_roots_l3157_315716

theorem sum_of_roots (x : ℝ) : x + 16 / x = 12 → ∃ y : ℝ, y + 16 / y = 12 ∧ x + y = 12 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3157_315716


namespace NUMINAMATH_CALUDE_polygon_sides_l3157_315747

/-- A polygon with side length 4 and perimeter 24 has 6 sides -/
theorem polygon_sides (side_length : ℝ) (perimeter : ℝ) (num_sides : ℕ) : 
  side_length = 4 → perimeter = 24 → num_sides * side_length = perimeter → num_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3157_315747


namespace NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_l3157_315712

theorem two_numbers_with_difference_and_quotient :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a - b = 157 ∧ a / b = 2 ∧ a = 314 ∧ b = 157 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_l3157_315712


namespace NUMINAMATH_CALUDE_system_solution_l3157_315792

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 1) (eq2 : 2*x + y = 2) : x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3157_315792


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l3157_315731

/-- The sum of distinct powers of 2 that equals 700 -/
def sum_of_powers (powers : List ℕ) : Prop :=
  (powers.map (λ x => 2^x)).sum = 700 ∧ powers.Nodup

/-- The proposition that 30 is the least possible sum of exponents -/
theorem least_sum_of_exponents :
  ∀ powers : List ℕ,
    sum_of_powers powers →
    powers.length ≥ 3 →
    powers.sum ≥ 30 ∧
    ∃ optimal_powers : List ℕ,
      sum_of_powers optimal_powers ∧
      optimal_powers.length ≥ 3 ∧
      optimal_powers.sum = 30 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l3157_315731


namespace NUMINAMATH_CALUDE_radio_show_ad_break_duration_l3157_315707

theorem radio_show_ad_break_duration 
  (total_show_time : ℕ) 
  (talking_segment_duration : ℕ) 
  (num_talking_segments : ℕ) 
  (num_ad_breaks : ℕ) 
  (song_duration : ℕ) 
  (h1 : total_show_time = 3 * 60) 
  (h2 : talking_segment_duration = 10)
  (h3 : num_talking_segments = 3)
  (h4 : num_ad_breaks = 5)
  (h5 : song_duration = 125) : 
  (total_show_time - (num_talking_segments * talking_segment_duration) - song_duration) / num_ad_breaks = 5 := by
sorry

end NUMINAMATH_CALUDE_radio_show_ad_break_duration_l3157_315707


namespace NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_l3157_315791

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (removedSquares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of our specific chessboard -/
def ourChessboard : ModifiedChessboard :=
  { size := 8, removedSquares := 2 }

/-- Defines the properties of our domino -/
def ourDomino : Domino :=
  { length := 2, width := 1 }

/-- Function to check if a chessboard can be tiled with dominoes -/
def canBeTiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  ∃ (tiling : Nat), 
    (board.size * board.size - board.removedSquares) = tiling * tile.length * tile.width

/-- Theorem stating that our specific chessboard cannot be tiled with our specific dominoes -/
theorem chessboard_cannot_be_tiled : 
  ¬(canBeTiled ourChessboard ourDomino) := by
  sorry


end NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_l3157_315791


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l3157_315736

/-- The quadratic equation kx^2 - 2(3k - 1)x + 9k - 1 = 0 has at least one integer root
    if and only if k is -3 or -7. -/
theorem quadratic_integer_root (k : ℤ) : 
  (∃ x : ℤ, k * x^2 - 2*(3*k - 1)*x + 9*k - 1 = 0) ↔ (k = -3 ∨ k = -7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l3157_315736


namespace NUMINAMATH_CALUDE_cookies_per_child_l3157_315769

theorem cookies_per_child (total_cookies : ℕ) (num_adults num_children : ℕ) (adult_fraction : ℚ) : 
  total_cookies = 240 →
  num_adults = 4 →
  num_children = 6 →
  adult_fraction = 1/4 →
  (total_cookies - (adult_fraction * total_cookies).num) / num_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_child_l3157_315769


namespace NUMINAMATH_CALUDE_fraction_equality_l3157_315741

theorem fraction_equality : 
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3157_315741


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3157_315717

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3157_315717


namespace NUMINAMATH_CALUDE_viewers_of_program_A_l3157_315783

theorem viewers_of_program_A (total_viewers : ℕ) (ratio_both ratio_A ratio_B : ℕ) : 
  total_viewers = 560 →
  ratio_both = 1 →
  ratio_A = 2 →
  ratio_B = 3 →
  (ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A) * total_viewers / ((ratio_both + ratio_A + ratio_B) * (ratio_both + ratio_A + ratio_B)) = 280 :=
by sorry

end NUMINAMATH_CALUDE_viewers_of_program_A_l3157_315783


namespace NUMINAMATH_CALUDE_cookies_per_bag_l3157_315708

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) 
  (h1 : chocolate_chip = 23)
  (h2 : oatmeal = 25)
  (h3 : baggies = 8) :
  (chocolate_chip + oatmeal) / baggies = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l3157_315708


namespace NUMINAMATH_CALUDE_chess_proficiency_multiple_chess_proficiency_multiple_proof_l3157_315765

theorem chess_proficiency_multiple : ℕ → Prop :=
  fun x =>
    let time_learn_rules : ℕ := 2
    let time_get_proficient : ℕ := time_learn_rules * x
    let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
    let total_time : ℕ := 10100
    total_time = time_learn_rules + time_get_proficient + time_become_master →
    x = 49

theorem chess_proficiency_multiple_proof : chess_proficiency_multiple 49 := by
  sorry

end NUMINAMATH_CALUDE_chess_proficiency_multiple_chess_proficiency_multiple_proof_l3157_315765


namespace NUMINAMATH_CALUDE_tech_company_work_hours_l3157_315751

/-- Calculates the total hours worked in a day for a tech company's help desk -/
theorem tech_company_work_hours :
  let total_hours : ℝ := 24
  let software_hours : ℝ := 24
  let user_help_hours : ℝ := 17
  let maintenance_percent : ℝ := 35
  let research_dev_percent : ℝ := 27
  let marketing_percent : ℝ := 15
  let multitasking_employees : ℕ := 3
  let additional_employees : ℕ := 4
  let additional_hours : ℝ := 12
  
  (maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours +
  software_hours + user_help_hours ≥ total_hours →
  
  (max software_hours (max user_help_hours ((maintenance_percent + research_dev_percent + marketing_percent) / 100 * total_hours))) +
  additional_hours = 36 :=
by sorry

end NUMINAMATH_CALUDE_tech_company_work_hours_l3157_315751


namespace NUMINAMATH_CALUDE_expression_one_equality_l3157_315723

theorem expression_one_equality : 2 - (-4) + 8 / (-2) + (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_expression_one_equality_l3157_315723


namespace NUMINAMATH_CALUDE_light_path_length_in_cube_l3157_315782

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  cubeSideLength : ℝ

/-- Calculates the length of the light path -/
def lightPathLength (path : LightPath) : ℝ :=
  sorry

theorem light_path_length_in_cube (c : Cube) (path : LightPath) :
  c.sideLength = 10 ∧
  path.start = Point3D.mk 0 0 0 ∧
  path.reflection = Point3D.mk 10 3 4 ∧
  path.cubeSideLength = c.sideLength →
  lightPathLength path = 50 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_light_path_length_in_cube_l3157_315782


namespace NUMINAMATH_CALUDE_fran_red_macaroons_l3157_315742

/-- Represents the number of macaroons in various states --/
structure MacaroonCounts where
  green_baked : ℕ
  green_eaten : ℕ
  red_eaten : ℕ
  total_remaining : ℕ

/-- The theorem stating the number of red macaroons Fran baked --/
theorem fran_red_macaroons (m : MacaroonCounts) 
  (h1 : m.green_baked = 40)
  (h2 : m.green_eaten = 15)
  (h3 : m.red_eaten = 2 * m.green_eaten)
  (h4 : m.total_remaining = 45) :
  ∃ red_baked : ℕ, red_baked = 50 ∧ 
    red_baked = m.red_eaten + (m.total_remaining - (m.green_baked - m.green_eaten)) :=
by sorry

end NUMINAMATH_CALUDE_fran_red_macaroons_l3157_315742


namespace NUMINAMATH_CALUDE_wages_theorem_l3157_315728

/-- Given a sum of money that can pay B's wages for 12 days and C's wages for 24 days,
    prove that it can pay both B and C's wages together for 8 days -/
theorem wages_theorem (S : ℝ) (W_B W_C : ℝ) (h1 : S = 12 * W_B) (h2 : S = 24 * W_C) :
  S = 8 * (W_B + W_C) := by
  sorry

end NUMINAMATH_CALUDE_wages_theorem_l3157_315728


namespace NUMINAMATH_CALUDE_arianna_daily_chores_l3157_315745

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def sleep_hours : ℕ := 13

theorem arianna_daily_chores : 
  hours_in_day - (work_hours + sleep_hours) = 5 := by
  sorry

end NUMINAMATH_CALUDE_arianna_daily_chores_l3157_315745


namespace NUMINAMATH_CALUDE_rice_dumpling_max_profit_l3157_315784

/-- A structure representing the rice dumpling problem -/
structure RiceDumplingProblem where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ
  total_purchase_cost : ℝ
  total_boxes : ℕ

/-- The profit function for the rice dumpling problem -/
def profit (p : RiceDumplingProblem) (x y : ℕ) : ℝ :=
  (p.selling_price_A - p.purchase_price_A) * x + (p.selling_price_B - p.purchase_price_B) * y

/-- Theorem stating the maximum profit for the rice dumpling problem -/
theorem rice_dumpling_max_profit (p : RiceDumplingProblem) 
  (h1 : p.purchase_price_A = 25)
  (h2 : p.purchase_price_B = 30)
  (h3 : p.selling_price_A = 32)
  (h4 : p.selling_price_B = 40)
  (h5 : p.total_purchase_cost = 1500)
  (h6 : p.total_boxes = 60) :
  ∃ (x y : ℕ), x + y = p.total_boxes ∧ x ≥ 2 * y ∧ profit p x y = 480 ∧ 
  ∀ (a b : ℕ), a + b = p.total_boxes → a ≥ 2 * b → profit p a b ≤ 480 :=
by sorry

#check rice_dumpling_max_profit

end NUMINAMATH_CALUDE_rice_dumpling_max_profit_l3157_315784


namespace NUMINAMATH_CALUDE_diophantus_problem_l3157_315794

theorem diophantus_problem (x y z t : ℤ) : 
  x = 11 ∧ y = 4 ∧ z = 7 ∧ t = 9 →
  x + y + z = 22 ∧
  x + y + t = 24 ∧
  x + z + t = 27 ∧
  y + z + t = 20 := by
sorry

end NUMINAMATH_CALUDE_diophantus_problem_l3157_315794


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l3157_315773

theorem max_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 2| + |x - 8| ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 2| + |x - 8| ≥ b) → b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l3157_315773


namespace NUMINAMATH_CALUDE_max_value_of_f_l3157_315757

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2

-- Define the interval
def interval : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 4 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3157_315757


namespace NUMINAMATH_CALUDE_symmetry_probability_l3157_315706

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨5, 5⟩

/-- Checks if a point is on a line of symmetry -/
def isOnLineOfSymmetry (p : GridPoint) : Bool :=
  p.x = 5 ∨ p.y = 5 ∨ p.x = p.y ∨ p.x + p.y = 10

/-- The total number of points in the grid -/
def totalPoints : Nat := 121

/-- The number of points on lines of symmetry, excluding the center -/
def symmetryPoints : Nat := 40

/-- Theorem stating the probability of selecting a point on a line of symmetry -/
theorem symmetry_probability :
  (symmetryPoints : ℚ) / (totalPoints - 1 : ℚ) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_probability_l3157_315706


namespace NUMINAMATH_CALUDE_joey_age_is_12_l3157_315720

def ages : List ℕ := [4, 6, 8, 10, 12, 14]

def went_to_movies (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def went_to_soccer (a b : ℕ) : Prop := a < 12 ∧ b < 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

def stayed_home (joey_age : ℕ) : Prop := joey_age ∈ ages ∧ 6 ∈ ages

theorem joey_age_is_12 :
  ∃! (joey_age : ℕ),
    (∃ (a b c d : ℕ),
      went_to_movies a b ∧
      went_to_soccer c d ∧
      stayed_home joey_age ∧
      {a, b, c, d, joey_age, 6} = ages.toFinset) ∧
    joey_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_joey_age_is_12_l3157_315720


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l3157_315714

theorem factorization_x_squared_minus_2x (x : ℝ) : x^2 - 2*x = x*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2x_l3157_315714


namespace NUMINAMATH_CALUDE_novel_pages_calculation_l3157_315776

/-- Calculates the total number of pages in a novel based on a specific reading pattern -/
theorem novel_pages_calculation : 
  let first_four_days := 4
  let next_two_days := 2
  let last_day := 1
  let pages_per_day_first_four := 42
  let pages_per_day_next_two := 50
  let pages_last_day := 30
  
  (first_four_days * pages_per_day_first_four) + 
  (next_two_days * pages_per_day_next_two) + 
  pages_last_day = 298 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_calculation_l3157_315776


namespace NUMINAMATH_CALUDE_circle_condition_l3157_315770

-- Define the equation of the curve
def curve_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*m*x - 2*y + 5*m = 0

-- Define the condition for m
def m_condition (m : ℝ) : Prop :=
  m < 1/4 ∨ m > 1

-- Theorem statement
theorem circle_condition (m : ℝ) :
  (∃ h k r, ∀ x y, curve_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ m_condition m :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l3157_315770


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_over_3_l3157_315729

theorem cos_2alpha_plus_4pi_over_3 (α : Real) 
  (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) : 
  Real.cos (2 * α + 4 * π / 3) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_over_3_l3157_315729


namespace NUMINAMATH_CALUDE_tabithas_final_amount_l3157_315721

/-- Calculates Tabitha's remaining money after various transactions --/
def tabithas_remaining_money (initial_amount : ℚ) (given_to_mom : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let after_mom := initial_amount - given_to_mom
  let after_investment := after_mom / 2
  let spent_on_items := num_items * item_cost
  after_investment - spent_on_items

/-- Theorem stating that Tabitha's remaining money is 6 dollars --/
theorem tabithas_final_amount :
  tabithas_remaining_money 25 8 5 (1/2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_tabithas_final_amount_l3157_315721


namespace NUMINAMATH_CALUDE_equation_has_one_integral_root_l3157_315771

theorem equation_has_one_integral_root :
  ∃! (x : ℤ), x - 5 / (x - 4 : ℚ) = 2 - 5 / (x - 4 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_integral_root_l3157_315771


namespace NUMINAMATH_CALUDE_rectangle_width_l3157_315705

/-- Given a rectangle with specific properties, prove its width is 6 meters -/
theorem rectangle_width (area perimeter length width : ℝ) 
  (h_area : area = 50)
  (h_perimeter : perimeter = 30)
  (h_ratio : length = (3/2) * width)
  (h_area_def : area = length * width)
  (h_perimeter_def : perimeter = 2 * (length + width)) :
  width = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l3157_315705


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3157_315739

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3157_315739


namespace NUMINAMATH_CALUDE_simplify_fraction_l3157_315772

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3157_315772


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l3157_315711

/-- Calculates the total expenditure for covering the interior of a rectangular hall with mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := floor_area + wall_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for the given hall dimensions and mat cost is 19500 -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 30 = 19500 := by
  sorry

#eval total_expenditure 20 15 5 30

end NUMINAMATH_CALUDE_hall_mat_expenditure_l3157_315711


namespace NUMINAMATH_CALUDE_boxed_divisibility_boxed_27_divisibility_l3157_315767

def boxed (n : ℕ+) : ℕ := (10^n.val - 1) / 9

theorem boxed_divisibility (m : ℕ) :
  ∃ k : ℕ, boxed (3^m : ℕ+) = k * 3^m ∧ 
  ∀ l : ℕ, boxed (3^m : ℕ+) ≠ l * 3^(m+1) :=
sorry

theorem boxed_27_divisibility (n : ℕ+) :
  27 ∣ n ↔ 27 ∣ boxed n :=
sorry

end NUMINAMATH_CALUDE_boxed_divisibility_boxed_27_divisibility_l3157_315767


namespace NUMINAMATH_CALUDE_distance_to_larger_section_l3157_315730

/-- Right pentagonal pyramid with two parallel cross sections -/
structure RightPentagonalPyramid where
  /-- Area of smaller cross section in square feet -/
  area_small : ℝ
  /-- Area of larger cross section in square feet -/
  area_large : ℝ
  /-- Distance between the two cross sections in feet -/
  distance_between : ℝ

/-- Theorem: Distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : RightPentagonalPyramid) 
  (h_area_small : pyramid.area_small = 100 * Real.sqrt 3)
  (h_area_large : pyramid.area_large = 225 * Real.sqrt 3)
  (h_distance : pyramid.distance_between = 5) :
  ∃ (d : ℝ), d = 15 ∧ d * d * pyramid.area_small = (d - 5) * (d - 5) * pyramid.area_large :=
by sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_l3157_315730


namespace NUMINAMATH_CALUDE_total_people_in_program_l3157_315726

theorem total_people_in_program (parents pupils teachers staff volunteers : ℕ) 
  (h1 : parents = 105)
  (h2 : pupils = 698)
  (h3 : teachers = 35)
  (h4 : staff = 20)
  (h5 : volunteers = 50) :
  parents + pupils + teachers + staff + volunteers = 908 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l3157_315726


namespace NUMINAMATH_CALUDE_complex_multiplication_l3157_315719

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3157_315719


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3157_315718

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |m - (9^3 + 7^3)^(1/3)| ≥ |n - (9^3 + 7^3)^(1/3)| :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3157_315718


namespace NUMINAMATH_CALUDE_penny_theorem_l3157_315796

def penny_problem (initial_amount : ℕ) (sock_pairs : ℕ) (sock_price : ℕ) (hat_price : ℕ) : Prop :=
  let total_spent := sock_pairs * sock_price + hat_price
  initial_amount - total_spent = 5

theorem penny_theorem : 
  penny_problem 20 4 2 7 := by
  sorry

end NUMINAMATH_CALUDE_penny_theorem_l3157_315796


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3157_315732

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.tan (5 * π / 2 + x) - 3 * Real.tan x ^ 2 = (Real.cos (2 * x) - 1) / Real.cos x ^ 2) →
  ∃ k : ℤ, x = π / 4 * (4 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3157_315732


namespace NUMINAMATH_CALUDE_largest_root_cubic_bounded_l3157_315764

theorem largest_root_cubic_bounded (b₂ b₁ b₀ : ℝ) 
  (h₂ : |b₂| ≤ 1) (h₁ : |b₁| ≤ 1) (h₀ : |b₀| ≤ 1) :
  ∃ r : ℝ, r > 0 ∧ r^3 + b₂*r^2 + b₁*r + b₀ = 0 ∧
  (∀ s : ℝ, s > 0 ∧ s^3 + b₂*s^2 + b₁*s + b₀ = 0 → s ≤ r) ∧
  1.5 < r ∧ r < 2 :=
sorry

end NUMINAMATH_CALUDE_largest_root_cubic_bounded_l3157_315764


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3157_315779

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- The unique solution is x = -4
  use -4
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3157_315779


namespace NUMINAMATH_CALUDE_marthas_cat_rats_l3157_315704

theorem marthas_cat_rats (R : ℕ) : 
  (5 * (R + 7) - 3 = 47) → R = 3 := by
  sorry

end NUMINAMATH_CALUDE_marthas_cat_rats_l3157_315704


namespace NUMINAMATH_CALUDE_candy_distribution_l3157_315740

theorem candy_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2013 →
  a = 2 * b + 10 →
  a = 3 * c + 18 →
  a = 5 * d - 55 →
  a + b + c + d = total →
  a = 990 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3157_315740


namespace NUMINAMATH_CALUDE_next_occurrence_sqrt_l3157_315762

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 60 * 60

/-- The time difference in seconds between two consecutive occurrences -/
def time_difference : ℕ := seconds_per_day + seconds_per_hour

theorem next_occurrence_sqrt (S : ℕ) (h : S = time_difference) : 
  Real.sqrt (S : ℝ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_next_occurrence_sqrt_l3157_315762


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l3157_315780

def is_three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

def digits_consecutive (n : ℕ) : Prop :=
  ∃ d₁ d₂ d₃ : ℕ, n = d₁ * 100 + d₂ * 10 + d₃ ∧
  ((d₁ + 1 = d₂ ∧ d₂ + 1 = d₃) ∨
   (d₁ + 1 = d₃ ∧ d₃ + 1 = d₂) ∨
   (d₂ + 1 = d₁ ∧ d₁ + 1 = d₃) ∨
   (d₂ + 1 = d₃ ∧ d₃ + 1 = d₁) ∨
   (d₃ + 1 = d₁ ∧ d₁ + 1 = d₂) ∨
   (d₃ + 1 = d₂ ∧ d₂ + 1 = d₁))

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ, is_three_digit a → is_three_digit b →
  digits_consecutive a → digits_consecutive b →
  ∃ S : ℕ, S = a + b ∧ sum_of_digits S ≥ 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l3157_315780


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l3157_315754

theorem trig_fraction_equality (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1/2) :
  Real.cos a / (Real.sin a - 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l3157_315754


namespace NUMINAMATH_CALUDE_daragh_favorite_bears_l3157_315701

/-- The number of stuffed bears Daragh had initially -/
def initial_bears : ℕ := 20

/-- The number of sisters Daragh divided bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_initial_bears : ℕ := 10

/-- The number of bears Eden has after receiving more -/
def eden_final_bears : ℕ := 14

/-- The number of favorite stuffed bears Daragh took out -/
def favorite_bears : ℕ := initial_bears - (eden_final_bears - eden_initial_bears) * num_sisters

theorem daragh_favorite_bears :
  favorite_bears = 8 :=
by sorry

end NUMINAMATH_CALUDE_daragh_favorite_bears_l3157_315701


namespace NUMINAMATH_CALUDE_image_of_four_six_l3157_315778

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

theorem image_of_four_six : f (4, 6) = (10, -2) := by
  sorry

end NUMINAMATH_CALUDE_image_of_four_six_l3157_315778


namespace NUMINAMATH_CALUDE_initial_acorns_l3157_315738

def acorns_given_away : ℕ := 7
def acorns_left : ℕ := 9

theorem initial_acorns : 
  acorns_given_away + acorns_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_acorns_l3157_315738


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l3157_315793

theorem simplify_fourth_root : 
  (2^5 * 5^3 : ℝ)^(1/4) = 2 * (250 : ℝ)^(1/4) := by sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l3157_315793


namespace NUMINAMATH_CALUDE_spinsters_and_cats_l3157_315768

theorem spinsters_and_cats (spinsters : ℕ) (cats : ℕ) : 
  spinsters = 12 →
  (spinsters : ℚ) / cats = 2 / 9 →
  cats > spinsters →
  cats - spinsters = 42 := by
sorry

end NUMINAMATH_CALUDE_spinsters_and_cats_l3157_315768


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3157_315715

def M : Set ℝ := {y | ∃ x, y = Real.sin x}
def N : Set ℝ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3157_315715


namespace NUMINAMATH_CALUDE_trees_needed_for_road_l3157_315781

/-- The number of trees needed to plant on one side of a road -/
def num_trees (road_length : ℕ) (interval : ℕ) : ℕ :=
  road_length / interval + 1

/-- Theorem: The number of trees needed for a 1500m road with 25m intervals is 61 -/
theorem trees_needed_for_road : num_trees 1500 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_trees_needed_for_road_l3157_315781


namespace NUMINAMATH_CALUDE_not_p_and_not_q_l3157_315786

-- Define proposition p
def p : Prop := ∃ t : ℝ, t > 0 ∧ t^2 - 2*t + 2 = 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x - x - 1 ≥ -1

-- Theorem statement
theorem not_p_and_not_q : (¬p) ∧ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_l3157_315786


namespace NUMINAMATH_CALUDE_minimum_value_quadratic_l3157_315748

theorem minimum_value_quadratic (a : ℝ) : 
  (∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ 
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → x^2 - 2*a*x + a - 1 ≤ y^2 - 2*a*y + a - 1) ∧
    x^2 - 2*a*x + a - 1 = -2) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_quadratic_l3157_315748


namespace NUMINAMATH_CALUDE_log_base_8_equals_3_l3157_315722

theorem log_base_8_equals_3 (y : ℝ) (h : Real.log y / Real.log 8 = 3) : y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_base_8_equals_3_l3157_315722


namespace NUMINAMATH_CALUDE_tim_score_is_38000_l3157_315703

/-- The value of a single line in points -/
def single_line_value : ℕ := 1000

/-- The value of a tetris in points -/
def tetris_value : ℕ := 8 * single_line_value

/-- The number of singles Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Tim's total score -/
def tim_total_score : ℕ := tim_singles * single_line_value + tim_tetrises * tetris_value

theorem tim_score_is_38000 : tim_total_score = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_score_is_38000_l3157_315703


namespace NUMINAMATH_CALUDE_range_of_a_l3157_315753

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 ≥ 0 ∧ a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (-1 ≤ a ∧ a ≤ 1) ∨ a > 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3157_315753


namespace NUMINAMATH_CALUDE_plane_speed_l3157_315756

theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (still_air_speed : ℝ) (time : ℝ),
    time > 0 ∧
    distance_with_wind = (still_air_speed + wind_speed) * time ∧
    distance_against_wind = (still_air_speed - wind_speed) * time ∧
    still_air_speed = 180 :=
by sorry

end NUMINAMATH_CALUDE_plane_speed_l3157_315756


namespace NUMINAMATH_CALUDE_teal_more_blue_l3157_315725

/-- The number of people surveyed -/
def total_surveyed : ℕ := 150

/-- The number of people who believe teal is "more green" -/
def more_green : ℕ := 90

/-- The number of people who believe teal is both "more green" and "more blue" -/
def both : ℕ := 35

/-- The number of people who believe teal is neither "more green" nor "more blue" -/
def neither : ℕ := 25

/-- The theorem stating that 70 people believe teal is "more blue" -/
theorem teal_more_blue : 
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
  more_blue + (more_green - both) + both + neither = total_surveyed :=
sorry

end NUMINAMATH_CALUDE_teal_more_blue_l3157_315725


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3157_315700

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3157_315700


namespace NUMINAMATH_CALUDE_proportion_equality_l3157_315750

theorem proportion_equality (x : ℝ) (h : (3/4) / x = 7/8) : x = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3157_315750


namespace NUMINAMATH_CALUDE_min_value_theorem_l3157_315744

def arithmeticSequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  arithmeticSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (∃ min : ℝ, min = 3/2 ∧ ∀ p q : ℕ, 1/p + 4/q ≥ min) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3157_315744


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l3157_315795

-- Define a function to check if a number is composite
def isComposite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

-- Define a function to check if a sequence of numbers is all composite
def allComposite (start : ℕ) (length : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range length → isComposite (start + i)

-- Theorem statement
theorem consecutive_composites_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ allComposite start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ allComposite start 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l3157_315795


namespace NUMINAMATH_CALUDE_remainder_theorem_l3157_315761

theorem remainder_theorem : (104 * 106 - 8) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3157_315761


namespace NUMINAMATH_CALUDE_binomial_expansion_103_l3157_315787

theorem binomial_expansion_103 : 102^3 + 3*(102^2) + 3*102 + 1 = 103^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_l3157_315787


namespace NUMINAMATH_CALUDE_percentage_of_female_cows_l3157_315735

theorem percentage_of_female_cows (total_cows : ℕ) (pregnant_cows : ℕ) 
  (h1 : total_cows = 44)
  (h2 : pregnant_cows = 11)
  (h3 : pregnant_cows = (female_cows / 2 : ℚ)) :
  (female_cows : ℚ) / total_cows * 100 = 50 :=
by
  sorry

#check percentage_of_female_cows

end NUMINAMATH_CALUDE_percentage_of_female_cows_l3157_315735


namespace NUMINAMATH_CALUDE_female_democrats_count_l3157_315743

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 660 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total : ℚ) / 3 →
  female / 2 = 110 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l3157_315743


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3157_315709

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s + 2) * (s + 2) * (s - 2) = s^3 - 10 →
  s^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3157_315709


namespace NUMINAMATH_CALUDE_function_properties_l3157_315788

/-- The function f(x) = ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g(x) = f(x) - kx -/
def g (a b k x : ℝ) : ℝ := f a b x - k * x

theorem function_properties (a b k : ℝ) :
  (∀ x, f a b x ≥ 0) ∧  -- Range of f(x) is [0, +∞)
  (f a b (-1) = 0) ∧    -- f(x) has a zero point at x = -1
  (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) -- g(x) is monotonic on [-2, 2]
  →
  (f a b = fun x ↦ x^2 + 2*x + 1) ∧  -- f(x) = x² + 2x + 1
  (k ≥ 6 ∨ k ≤ -2)                   -- Range of k
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l3157_315788


namespace NUMINAMATH_CALUDE_charles_chocolate_milk_l3157_315760

/-- The amount of chocolate milk Charles can drink given his supplies -/
def chocolate_milk_total (milk_per_glass : ℚ) (syrup_per_glass : ℚ) 
  (total_milk : ℚ) (total_syrup : ℚ) : ℚ :=
  let glasses_from_milk := total_milk / milk_per_glass
  let glasses_from_syrup := total_syrup / syrup_per_glass
  let glasses := min glasses_from_milk glasses_from_syrup
  glasses * (milk_per_glass + syrup_per_glass)

/-- Theorem stating that Charles will drink 160 ounces of chocolate milk -/
theorem charles_chocolate_milk :
  chocolate_milk_total (6.5) (1.5) (130) (60) = 160 := by
  sorry

end NUMINAMATH_CALUDE_charles_chocolate_milk_l3157_315760


namespace NUMINAMATH_CALUDE_power_of_six_l3157_315775

theorem power_of_six : (6 : ℕ) ^ ((6 : ℕ) / 2) = 216 := by sorry

end NUMINAMATH_CALUDE_power_of_six_l3157_315775


namespace NUMINAMATH_CALUDE_equation_solution_l3157_315766

theorem equation_solution :
  let S : Set ℂ := {x | (x - 1)^4 + (x - 1) = 0}
  S = {1, 0, Complex.mk 1 (Real.sqrt 3 / 2), Complex.mk 1 (-Real.sqrt 3 / 2)} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3157_315766


namespace NUMINAMATH_CALUDE_negation_of_exp_inequality_l3157_315774

theorem negation_of_exp_inequality :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exp_inequality_l3157_315774


namespace NUMINAMATH_CALUDE_resort_group_combinations_l3157_315777

theorem resort_group_combinations : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_resort_group_combinations_l3157_315777


namespace NUMINAMATH_CALUDE_white_mailbox_houses_l3157_315759

theorem white_mailbox_houses (total_mail : ℕ) (total_houses : ℕ) (red_houses : ℕ) (mail_per_house : ℕ)
  (h1 : total_mail = 48)
  (h2 : total_houses = 8)
  (h3 : red_houses = 3)
  (h4 : mail_per_house = 6) :
  total_houses - red_houses = 5 := by
sorry

end NUMINAMATH_CALUDE_white_mailbox_houses_l3157_315759


namespace NUMINAMATH_CALUDE_gcd_20586_58768_l3157_315752

theorem gcd_20586_58768 : Nat.gcd 20586 58768 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_20586_58768_l3157_315752


namespace NUMINAMATH_CALUDE_james_delivery_capacity_l3157_315799

/-- Given that James takes 20 trips a day and delivers 1000 bags in 5 days,
    prove that he can carry 10 bags on each trip. -/
theorem james_delivery_capacity
  (trips_per_day : ℕ)
  (total_bags : ℕ)
  (total_days : ℕ)
  (h1 : trips_per_day = 20)
  (h2 : total_bags = 1000)
  (h3 : total_days = 5) :
  total_bags / (trips_per_day * total_days) = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_delivery_capacity_l3157_315799


namespace NUMINAMATH_CALUDE_square_formation_theorem_l3157_315789

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Bool :=
  sum_of_naturals n % 4 = 0

def min_breaks_to_square (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else
    let total := sum_of_naturals n
    let target := (total + 3) / 4 * 4
    (target - total + 1) / 2

theorem square_formation_theorem :
  (min_breaks_to_square 12 = 2) ∧ (can_form_square 15 = true) := by
  sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l3157_315789


namespace NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l3157_315790

/-- Calculates the cost per serving of an apple pie --/
theorem apple_pie_cost_per_serving 
  (num_servings : ℕ)
  (apple_pounds : ℝ)
  (apple_cost_per_pound : ℝ)
  (crust_cost : ℝ)
  (lemon_cost : ℝ)
  (butter_cost : ℝ)
  (h1 : num_servings = 8)
  (h2 : apple_pounds = 2)
  (h3 : apple_cost_per_pound = 2)
  (h4 : crust_cost = 2)
  (h5 : lemon_cost = 0.5)
  (h6 : butter_cost = 1.5) :
  (apple_pounds * apple_cost_per_pound + crust_cost + lemon_cost + butter_cost) / num_servings = 1 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l3157_315790


namespace NUMINAMATH_CALUDE_specific_grid_rhombuses_l3157_315763

/-- A grid composed of equilateral triangles -/
structure TriangleGrid where
  num_triangles : ℕ
  num_rows : ℕ
  num_cols : ℕ

/-- The number of rhombuses that can be formed from two adjacent triangles in the grid -/
def count_rhombuses (grid : TriangleGrid) : ℕ :=
  sorry

/-- Theorem stating that a specific grid with 25 triangles has 30 rhombuses -/
theorem specific_grid_rhombuses :
  ∃ (grid : TriangleGrid), 
    grid.num_triangles = 25 ∧ 
    grid.num_rows = 5 ∧ 
    grid.num_cols = 5 ∧ 
    count_rhombuses grid = 30 := by
  sorry

end NUMINAMATH_CALUDE_specific_grid_rhombuses_l3157_315763


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l3157_315724

theorem min_product_of_reciprocal_sum (a b : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → 
  ∀ c d : ℕ+, (1 : ℚ) / c + (1 : ℚ) / (3 * d) = (1 : ℚ) / 6 → 
  a * b ≤ c * d ∧ a * b = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l3157_315724


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l3157_315746

/-- A parabola with directrix x = 1 has the standard equation y² = -4x -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p / 2 = 1 ∧ y^2 = -2 * p * x) → y^2 = -4 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l3157_315746


namespace NUMINAMATH_CALUDE_set_intersection_empty_iff_complement_subset_l3157_315797

universe u

theorem set_intersection_empty_iff_complement_subset {U : Type u} (A B : Set U) :
  A ∩ B = ∅ ↔ ∃ C : Set U, A ⊆ C ∧ B ⊆ Cᶜ :=
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_iff_complement_subset_l3157_315797


namespace NUMINAMATH_CALUDE_town_population_problem_l3157_315733

theorem town_population_problem (initial_population : ℕ) : 
  let after_changes := initial_population + 100 - 400
  let after_year_1 := after_changes / 2
  let after_year_2 := after_year_1 / 2
  let after_year_3 := after_year_2 / 2
  let after_year_4 := after_year_3 / 2
  after_year_4 = 60 → initial_population = 780 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3157_315733


namespace NUMINAMATH_CALUDE_carpet_length_l3157_315702

/-- Given a rectangular carpet with width 4 feet covering an entire room floor of area 60 square feet, 
    prove that the length of the carpet is 15 feet. -/
theorem carpet_length (carpet_width : ℝ) (room_area : ℝ) (h1 : carpet_width = 4) (h2 : room_area = 60) :
  room_area / carpet_width = 15 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_l3157_315702


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tan_a7_l3157_315785

theorem arithmetic_sequence_tan_a7 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 5 + a 9 = 8 * Real.pi / 3 →                         -- given condition
  Real.tan (a 7) = Real.sqrt 3 :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tan_a7_l3157_315785


namespace NUMINAMATH_CALUDE_square_difference_l3157_315734

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) :
  (x - y)^2 = 24 := by sorry

end NUMINAMATH_CALUDE_square_difference_l3157_315734


namespace NUMINAMATH_CALUDE_bonnets_theorem_l3157_315727

def bonnets_problem (monday thursday friday : ℕ) : Prop :=
  let tuesday_wednesday := 2 * monday
  let total_mon_to_thu := monday + tuesday_wednesday + thursday
  let total_sent := 11 * 5
  thursday = monday + 5 ∧
  total_sent = total_mon_to_thu + friday ∧
  thursday - friday = 5

theorem bonnets_theorem : 
  ∃ (monday thursday friday : ℕ), 
    monday = 10 ∧ 
    bonnets_problem monday thursday friday :=
sorry

end NUMINAMATH_CALUDE_bonnets_theorem_l3157_315727


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3157_315758

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3157_315758


namespace NUMINAMATH_CALUDE_custom_op_example_l3157_315710

/-- Custom binary operation ⊗ defined as a ⊗ b = a - |b| -/
def custom_op (a b : ℝ) : ℝ := a - abs b

/-- Theorem stating that 2 ⊗ (-3) = -1 -/
theorem custom_op_example : custom_op 2 (-3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3157_315710


namespace NUMINAMATH_CALUDE_book_chapters_l3157_315749

/-- A problem about determining the number of chapters in a book based on reading rate. -/
theorem book_chapters (chapters_read : ℕ) (hours_read : ℕ) (hours_remaining : ℕ) : 
  chapters_read = 2 →
  hours_read = 3 →
  hours_remaining = 9 →
  ∃ (total_chapters : ℕ), 
    total_chapters = chapters_read + (chapters_read * hours_remaining / hours_read) ∧
    total_chapters = 8 := by
  sorry


end NUMINAMATH_CALUDE_book_chapters_l3157_315749


namespace NUMINAMATH_CALUDE_vector_dot_product_l3157_315755

/-- Given vectors a and b in ℝ², prove that (2a + b) · a = 6 -/
theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3157_315755
