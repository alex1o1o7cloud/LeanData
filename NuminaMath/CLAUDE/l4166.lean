import Mathlib

namespace NUMINAMATH_CALUDE_max_tuesdays_in_63_days_l4166_416675

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days we are considering -/
def total_days : ℕ := 63

/-- Each week has one Tuesday -/
axiom one_tuesday_per_week : ℕ

/-- The maximum number of Tuesdays in the first 63 days of a year -/
def max_tuesdays : ℕ := total_days / days_in_week

theorem max_tuesdays_in_63_days : max_tuesdays = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tuesdays_in_63_days_l4166_416675


namespace NUMINAMATH_CALUDE_remaining_money_l4166_416606

def initial_amount : ℚ := 10
def candy_bar_cost : ℚ := 2
def chocolate_cost : ℚ := 3
def soda_cost : ℚ := 1.5
def gum_cost : ℚ := 1.25

theorem remaining_money :
  initial_amount - candy_bar_cost - chocolate_cost - soda_cost - gum_cost = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l4166_416606


namespace NUMINAMATH_CALUDE_smallest_factor_l4166_416622

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 36 → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 12^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * 36) ∧ 3^3 ∣ (936 * 36) ∧ 12^2 ∣ (936 * 36)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_l4166_416622


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4166_416679

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 11) 
  (eq2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4166_416679


namespace NUMINAMATH_CALUDE_inequality_solution_count_l4166_416674

theorem inequality_solution_count : 
  (Finset.filter (fun x => (x - 2)^2 ≤ 4) (Finset.range 100)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l4166_416674


namespace NUMINAMATH_CALUDE_puppy_group_arrangements_eq_2520_l4166_416646

/-- The number of ways to divide 12 puppies into groups of 4, 6, and 2,
    with Coco in the 4-puppy group and Rocky in the 6-puppy group. -/
def puppy_group_arrangements : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

/-- Theorem stating that the number of puppy group arrangements is 2520. -/
theorem puppy_group_arrangements_eq_2520 :
  puppy_group_arrangements = 2520 := by
  sorry

#eval puppy_group_arrangements

end NUMINAMATH_CALUDE_puppy_group_arrangements_eq_2520_l4166_416646


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l4166_416631

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | sub {f} : S f → S (λ x => x - f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : ℝ → ℝ), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l4166_416631


namespace NUMINAMATH_CALUDE_max_distance_PQ_l4166_416650

noncomputable section

-- Define the real parameters m and n
variables (m n : ℝ)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := m * x - n * y - 5 * m + n = 0
def l₂ (x y : ℝ) : Prop := n * x + m * y - 5 * m - n = 0

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the intersection point P
def P (x y : ℝ) : Prop := l₁ m n x y ∧ l₂ m n x y

-- Define point Q on circle C
def Q (x y : ℝ) : Prop := C x y

-- State the theorem
theorem max_distance_PQ (hm : m^2 + n^2 ≠ 0) :
  ∃ (px py qx qy : ℝ), P m n px py ∧ Q qx qy ∧
  ∀ (px' py' qx' qy' : ℝ), P m n px' py' → Q qx' qy' →
  (px - qx)^2 + (py - qy)^2 ≤ (6 + 2 * Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_PQ_l4166_416650


namespace NUMINAMATH_CALUDE_vessel_base_length_l4166_416641

/-- Given a cube and a rectangular vessel, proves the length of the vessel's base --/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 16) 
  (h2 : vessel_width = 15) 
  (h3 : water_rise = 13.653333333333334) : 
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l4166_416641


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l4166_416658

theorem no_infinite_sequence_exists : 
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = k n - 1 / k n) ∧ 
    (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l4166_416658


namespace NUMINAMATH_CALUDE_father_son_age_sum_l4166_416670

/-- Given the ages of a father and son 25 years ago and their current age ratio,
    prove that the sum of their present ages is 300 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (s f : ℕ) =>
    (f = 4 * s) →                  -- 25 years ago, father was 4 times as old as son
    (f + 25 = 3 * (s + 25)) →      -- Now, father is 3 times as old as son
    ((s + 25) + (f + 25) = 300)    -- Sum of their present ages is 300

/-- Proof of the theorem -/
lemma prove_father_son_age_sum : ∃ (s f : ℕ), father_son_age_sum s f := by
  sorry

#check prove_father_son_age_sum

end NUMINAMATH_CALUDE_father_son_age_sum_l4166_416670


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4166_416655

/-- The equation of a line passing through (-2, 3) with a 45° angle of inclination -/
theorem line_equation_through_point_with_inclination :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + 2) = (y - 3)) ∧ 
    m = Real.tan (45 * π / 180) ∧
    (x - y + 5 = 0) = (y = m * x + b) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l4166_416655


namespace NUMINAMATH_CALUDE_simplify_fraction_l4166_416669

theorem simplify_fraction : (180 / 16) * (5 / 120) * (8 / 3) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4166_416669


namespace NUMINAMATH_CALUDE_nth_monomial_formula_l4166_416677

/-- Represents the coefficient of the nth monomial in the sequence -/
def coefficient (n : ℕ) : ℕ := 3 * n + 2

/-- Represents the exponent of 'a' in the nth monomial of the sequence -/
def exponent (n : ℕ) : ℕ := n

/-- Represents the nth monomial in the sequence as a function of 'a' -/
def nthMonomial (n : ℕ) (a : ℝ) : ℝ := (coefficient n : ℝ) * a ^ (exponent n)

/-- The sequence of monomials follows the pattern 5a, 8a^2, 11a^3, 14a^4, ... -/
axiom sequence_pattern (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n

/-- Theorem: The nth monomial in the sequence is equal to (3n+2)a^n -/
theorem nth_monomial_formula (n : ℕ) (a : ℝ) : 
  n ≥ 1 → nthMonomial n a = (3 * n + 2 : ℝ) * a ^ n := by
  sorry

end NUMINAMATH_CALUDE_nth_monomial_formula_l4166_416677


namespace NUMINAMATH_CALUDE_solution_characterization_l4166_416657

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1/3, 1/3, 1/3), (0, 0, 1), (2/3, -1/3, 2/3), (0, 1, 0), (1, 0, 0), (-1, 1, 1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 1 ∧
  x^2*y + y^2*z + z^2*x = x*y^2 + y*z^2 + z*x^2 ∧
  x^3 + y^2 + z = y^3 + z^2 + x

theorem solution_characterization :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l4166_416657


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l4166_416628

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of segment ER -/
  er : ℝ
  /-- Length of segment RF -/
  rf : ℝ
  /-- Length of segment GS -/
  gs : ℝ
  /-- Length of segment SH -/
  sh : ℝ
  /-- The circle is tangent to EF at R and to GH at S -/
  tangent_condition : True

/-- The theorem stating that the square of the radius of the inscribed circle is (3225/118)^2 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.er = 22)
  (h2 : c.rf = 21)
  (h3 : c.gs = 40)
  (h4 : c.sh = 35) :
  c.r^2 = (3225/118)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l4166_416628


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l4166_416611

theorem multiplication_subtraction_equality : (5 * 3) - 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l4166_416611


namespace NUMINAMATH_CALUDE_parabola_transformation_transformation_is_right_shift_2_l4166_416645

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ := (x + 5) * (x - 3)

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ := (x + 3) * (x - 5)

-- Define the transformation
def transformation (x : ℝ) : ℝ := x + 2

-- Theorem stating the transformation between the two parabolas
theorem parabola_transformation :
  ∀ x : ℝ, parabola1 x = parabola2 (transformation x) :=
by
  sorry

-- Theorem stating that the transformation is a shift of 2 units to the right
theorem transformation_is_right_shift_2 :
  ∀ x : ℝ, transformation x = x + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_transformation_is_right_shift_2_l4166_416645


namespace NUMINAMATH_CALUDE_commission_calculation_l4166_416667

def base_salary : ℚ := 370
def past_incomes : List ℚ := [406, 413, 420, 436, 395]
def desired_average : ℚ := 500
def total_weeks : ℕ := 7
def past_weeks : ℕ := 5

theorem commission_calculation (base_salary : ℚ) (past_incomes : List ℚ) 
  (desired_average : ℚ) (total_weeks : ℕ) (past_weeks : ℕ) :
  (desired_average * total_weeks - past_incomes.sum - base_salary * total_weeks) / (total_weeks - past_weeks) = 345 :=
by sorry

end NUMINAMATH_CALUDE_commission_calculation_l4166_416667


namespace NUMINAMATH_CALUDE_water_tank_capacity_l4166_416661

theorem water_tank_capacity : ∃ T : ℝ, T > 0 ∧ T = 72 ∧ 0.4 * T = 0.9 * T - 36 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l4166_416661


namespace NUMINAMATH_CALUDE_total_easter_eggs_l4166_416626

def clubHouseEggs : ℕ := 40
def parkEggs : ℕ := 25
def townHallEggs : ℕ := 15

theorem total_easter_eggs : 
  clubHouseEggs + parkEggs + townHallEggs = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_easter_eggs_l4166_416626


namespace NUMINAMATH_CALUDE_total_packs_sold_is_51_l4166_416634

/-- The total number of cookie packs sold in two villages -/
def total_packs_sold (village1_packs : ℕ) (village2_packs : ℕ) : ℕ :=
  village1_packs + village2_packs

/-- Theorem: The total number of packs sold in two villages is 51 -/
theorem total_packs_sold_is_51 :
  total_packs_sold 23 28 = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_sold_is_51_l4166_416634


namespace NUMINAMATH_CALUDE_einstein_soda_sales_l4166_416604

def goal : ℝ := 500
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def remaining : ℝ := 258

theorem einstein_soda_sales :
  ∃ (soda_sold : ℕ),
    goal = pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold + remaining ∧
    soda_sold = 25 := by
  sorry

end NUMINAMATH_CALUDE_einstein_soda_sales_l4166_416604


namespace NUMINAMATH_CALUDE_sculpture_cost_equivalence_l4166_416698

-- Define exchange rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem to prove
theorem sculpture_cost_equivalence :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_equivalence_l4166_416698


namespace NUMINAMATH_CALUDE_checkerboard_coverage_unsolvable_boards_l4166_416637

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)
  (removed_squares : ℕ)

/-- Determines if a checkerboard can be completely covered by dominoes -/
def can_cover (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

theorem checkerboard_coverage (board : Checkerboard) :
  can_cover board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x7 board -/
def board_5x7 : Checkerboard := ⟨5, 7, 0⟩

/-- 7x3 board with two removed squares -/
def board_7x3_modified : Checkerboard := ⟨7, 3, 2⟩

theorem unsolvable_boards :
  ¬(can_cover board_5x7) ∧ ¬(can_cover board_7x3_modified) := by
  sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_unsolvable_boards_l4166_416637


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4166_416600

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {2,5,8}

theorem complement_intersection_theorem : 
  (U \ M) ∩ N = {2,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4166_416600


namespace NUMINAMATH_CALUDE_menu_choices_l4166_416623

/-- The number of ways to choose one menu for lunch and one for dinner -/
def choose_menus (lunch_chinese : Nat) (lunch_japanese : Nat) (dinner_chinese : Nat) (dinner_japanese : Nat) : Nat :=
  (lunch_chinese + lunch_japanese) * (dinner_chinese + dinner_japanese)

theorem menu_choices : choose_menus 5 4 3 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_menu_choices_l4166_416623


namespace NUMINAMATH_CALUDE_divisibility_n_plus_seven_l4166_416687

theorem divisibility_n_plus_seven (n : ℕ+) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_n_plus_seven_l4166_416687


namespace NUMINAMATH_CALUDE_triangle_angle_sum_rounded_l4166_416640

-- Define a structure for a triangle with actual and rounded angles
structure Triangle where
  P' : ℝ
  Q' : ℝ
  R' : ℝ
  P : ℤ
  Q : ℤ
  R : ℤ

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.P' + t.Q' + t.R' = 180 ∧
  t.P' > 0 ∧ t.Q' > 0 ∧ t.R' > 0 ∧
  (t.P' - 0.5 ≤ t.P ∧ t.P ≤ t.P' + 0.5) ∧
  (t.Q' - 0.5 ≤ t.Q ∧ t.Q ≤ t.Q' + 0.5) ∧
  (t.R' - 0.5 ≤ t.R ∧ t.R ≤ t.R' + 0.5)

-- Theorem statement
theorem triangle_angle_sum_rounded (t : Triangle) :
  is_valid_triangle t → (t.P + t.Q + t.R = 179 ∨ t.P + t.Q + t.R = 180 ∨ t.P + t.Q + t.R = 181) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_rounded_l4166_416640


namespace NUMINAMATH_CALUDE_missing_number_proof_l4166_416653

theorem missing_number_proof : ∃ x : ℚ, (3/4 * 60 - 8/5 * 60 + x = 12) ∧ (x = 63) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l4166_416653


namespace NUMINAMATH_CALUDE_expression_value_l4166_416659

theorem expression_value (x y z : ℝ) 
  (hx : x = -5/4) 
  (hy : y = -3/2) 
  (hz : z = Real.sqrt 2) : 
  -2 * x^3 - y^2 + Real.sin z = 53/32 + Real.sin (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4166_416659


namespace NUMINAMATH_CALUDE_range_of_H_l4166_416696

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2|^2 - |x - 2|^2

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, ∃ x : ℝ, H x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l4166_416696


namespace NUMINAMATH_CALUDE_implicit_function_derivative_specific_point_derivative_l4166_416694

noncomputable section

/-- The implicit function defined by 10x^3 + 4x^2y + y^2 = 0 -/
def f (x y : ℝ) : ℝ := 10 * x^3 + 4 * x^2 * y + y^2

/-- The derivative of the implicit function -/
def f_derivative (x y : ℝ) : ℝ := (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

theorem implicit_function_derivative (x y : ℝ) (h : f x y = 0) :
  deriv (fun y => f x y) y = f_derivative x y := by sorry

theorem specific_point_derivative :
  f_derivative (-2) 4 = -7/3 := by sorry

end

end NUMINAMATH_CALUDE_implicit_function_derivative_specific_point_derivative_l4166_416694


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4166_416630

theorem complex_fraction_equality : 50 / (8 - 3/7) = 350/53 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4166_416630


namespace NUMINAMATH_CALUDE_cube_division_theorem_l4166_416625

/-- Represents the volume of the remaining solid after removal of marked cubes --/
def remaining_volume (k : ℕ) : ℚ :=
  if k % 2 = 0 then 1/2
  else (k+1)^2 * (2*k-1) / (4*k^3)

/-- Represents the surface area of the remaining solid after removal of marked cubes --/
def remaining_surface_area (k : ℕ) : ℚ :=
  if k % 2 = 0 then 3*(k+1) / 2
  else 3*(k+1)^2 / (2*k)

theorem cube_division_theorem (k : ℕ) :
  (k ≥ 65 → remaining_surface_area k > 100) ∧
  (k % 2 = 0 → remaining_volume k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_cube_division_theorem_l4166_416625


namespace NUMINAMATH_CALUDE_num_divisors_1386_l4166_416662

/-- The number of positive divisors of a natural number n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- Theorem: The number of positive divisors of 1386 is 24 -/
theorem num_divisors_1386 : num_divisors 1386 = 24 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_1386_l4166_416662


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4166_416678

theorem complex_fraction_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^(4/3))^(3/(2*5)) / (a^4)^(3/5) / ((a * (a^2 * b)^(1/3))^(1/2))^4 * ((a * b^(1/2))^(1/4))^6 = 1 / (a^2 * b)^(1/12) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4166_416678


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l4166_416691

/-- Represents a hexagonal tile figure --/
structure HexFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Adds a border of yellow tiles to a hexagonal figure --/
def add_yellow_border (fig : HexFigure) : HexFigure :=
  { red_tiles := fig.red_tiles,
    yellow_tiles := fig.yellow_tiles + 18 }

/-- The initial hexagonal figure --/
def initial_figure : HexFigure :=
  { red_tiles := 15,
    yellow_tiles := 10 }

theorem yellow_red_difference_after_border :
  (add_yellow_border initial_figure).yellow_tiles - (add_yellow_border initial_figure).red_tiles = 13 :=
by sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l4166_416691


namespace NUMINAMATH_CALUDE_school_referendum_non_voters_l4166_416618

theorem school_referendum_non_voters (total : ℝ) (yes_votes : ℝ) (no_votes : ℝ)
  (h1 : yes_votes = (3 / 5) * total)
  (h2 : no_votes = 0.28 * total)
  (h3 : total > 0) :
  (total - (yes_votes + no_votes)) / total = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_non_voters_l4166_416618


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4166_416685

theorem min_value_quadratic (x y : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 9 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4166_416685


namespace NUMINAMATH_CALUDE_g_inequality_l4166_416620

def g (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem g_inequality : g (3/2) < g 0 ∧ g 0 < g 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inequality_l4166_416620


namespace NUMINAMATH_CALUDE_consecutive_binomial_coefficient_ratio_l4166_416616

theorem consecutive_binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 1 / 2 →
  n + k = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_coefficient_ratio_l4166_416616


namespace NUMINAMATH_CALUDE_henri_total_miles_l4166_416656

-- Define the variables
def gervais_average_miles : ℕ := 315
def gervais_days : ℕ := 3
def additional_miles : ℕ := 305

-- Define the theorem
theorem henri_total_miles :
  let gervais_total := gervais_average_miles * gervais_days
  let henri_total := gervais_total + additional_miles
  henri_total = 1250 := by
  sorry

end NUMINAMATH_CALUDE_henri_total_miles_l4166_416656


namespace NUMINAMATH_CALUDE_rectangle_problem_l4166_416602

-- Define the rectangle EFGH
structure Rectangle (EFGH : Type) where
  is_rectangle : EFGH → Prop

-- Define point M on FG
def M_on_FG (EFGH : Type) (M : EFGH) : Prop := sorry

-- Define angle EMH as 90°
def angle_EMH_90 (EFGH : Type) (E M H : EFGH) : Prop := sorry

-- Define UV perpendicular to FG
def UV_perp_FG (EFGH : Type) (U V F G : EFGH) : Prop := sorry

-- Define FU = UM
def FU_eq_UM (EFGH : Type) (F U M : EFGH) : Prop := sorry

-- Define MH intersects UV at N
def MH_intersect_UV_at_N (EFGH : Type) (M H U V N : EFGH) : Prop := sorry

-- Define S on GH such that SE passes through N
def S_on_GH_SE_through_N (EFGH : Type) (S G H E N : EFGH) : Prop := sorry

-- Define triangle MNE with given measurements
def triangle_MNE (EFGH : Type) (M N E : EFGH) : Prop :=
  let ME := 25
  let EN := 20
  let MN := 20
  sorry

-- Theorem statement
theorem rectangle_problem (EFGH : Type) 
  (E F G H M U V N S : EFGH) 
  (rect : Rectangle EFGH) 
  (h1 : M_on_FG EFGH M)
  (h2 : angle_EMH_90 EFGH E M H)
  (h3 : UV_perp_FG EFGH U V F G)
  (h4 : FU_eq_UM EFGH F U M)
  (h5 : MH_intersect_UV_at_N EFGH M H U V N)
  (h6 : S_on_GH_SE_through_N EFGH S G H E N)
  (h7 : triangle_MNE EFGH M N E) :
  ∃ (FM NV : ℝ), FM = 15 ∧ NV = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l4166_416602


namespace NUMINAMATH_CALUDE_negation_constant_arithmetic_sequence_l4166_416647

theorem negation_constant_arithmetic_sequence :
  ¬(∀ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) → (∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) ↔
  (∃ s : ℕ → ℝ, (∀ n : ℕ, s n = s 0) ∧ ¬(∃ d : ℝ, ∀ n : ℕ, s (n + 1) = s n + d)) :=
by sorry

end NUMINAMATH_CALUDE_negation_constant_arithmetic_sequence_l4166_416647


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4166_416699

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = 0 → x^2 - 2*x = 0) ∧
  (∃ x : ℝ, x^2 - 2*x = 0 ∧ x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4166_416699


namespace NUMINAMATH_CALUDE_gardening_time_ratio_l4166_416627

/-- Proves that the ratio of time to plant one flower to time to mow one line is 1/4 --/
theorem gardening_time_ratio :
  ∀ (total_time mow_time plant_time : ℕ) 
    (lines flowers_per_row rows : ℕ) 
    (time_per_line : ℕ),
  total_time = 108 →
  lines = 40 →
  time_per_line = 2 →
  flowers_per_row = 7 →
  rows = 8 →
  mow_time = lines * time_per_line →
  plant_time = total_time - mow_time →
  (plant_time : ℚ) / (rows * flowers_per_row : ℚ) / (time_per_line : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gardening_time_ratio_l4166_416627


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4166_416609

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : x - y = 1) :
  x + 1/y ≥ 3 ∧ ∃ (x0 y0 : ℝ), x0 > 1 ∧ x0 - y0 = 1 ∧ x0 + 1/y0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4166_416609


namespace NUMINAMATH_CALUDE_perpendicular_unit_vector_l4166_416654

theorem perpendicular_unit_vector (a : ℝ × ℝ) (v : ℝ × ℝ) : 
  a = (1, 1) → 
  v = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) → 
  (a.1 * v.1 + a.2 * v.2 = 0) ∧ 
  (v.1^2 + v.2^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vector_l4166_416654


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_neg_two_l4166_416636

theorem at_least_one_not_less_than_neg_two
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 1/b ≥ -2) ∨ (b + 1/c ≥ -2) ∨ (c + 1/a ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_neg_two_l4166_416636


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l4166_416668

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 12 →
  original_list.sum / original_list.length = 75 →
  (original_list.sum + x + y + z) / (original_list.length + 3) = 90 →
  (x + y + z) / 3 = 150 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l4166_416668


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l4166_416689

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem stating that the opposite of -2 is 2
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l4166_416689


namespace NUMINAMATH_CALUDE_tim_cabinet_price_l4166_416632

/-- The price Tim paid for a cabinet after discount -/
theorem tim_cabinet_price (original_price : ℝ) (discount_percentage : ℝ) 
  (h1 : original_price = 1200)
  (h2 : discount_percentage = 15) : 
  original_price * (1 - discount_percentage / 100) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_tim_cabinet_price_l4166_416632


namespace NUMINAMATH_CALUDE_abs_two_set_l4166_416684

theorem abs_two_set : {x : ℝ | |x| = 2} = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_abs_two_set_l4166_416684


namespace NUMINAMATH_CALUDE_solution_sets_correct_l4166_416672

/-- The function f(x) = x^2 - 3ax + 2a^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*a*x + 2*a^2

/-- The solution set for f(x) ≤ 0 when a = 1 -/
def solution_set_1 : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

/-- The solution set for f(x) < 0 when a = 0 -/
def solution_set_2_zero : Set ℝ := ∅

/-- The solution set for f(x) < 0 when a > 0 -/
def solution_set_2_pos (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a}

/-- The solution set for f(x) < 0 when a < 0 -/
def solution_set_2_neg (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a}

theorem solution_sets_correct :
  (∀ x, x ∈ solution_set_1 ↔ f 1 x ≤ 0) ∧
  (∀ x, x ∈ solution_set_2_zero ↔ f 0 x < 0) ∧
  (∀ a > 0, ∀ x, x ∈ solution_set_2_pos a ↔ f a x < 0) ∧
  (∀ a < 0, ∀ x, x ∈ solution_set_2_neg a ↔ f a x < 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_correct_l4166_416672


namespace NUMINAMATH_CALUDE_sequence_inequality_l4166_416605

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) :
  n ≥ 2 →
  (∀ k, a k > 0) →
  (∀ k ∈ Finset.range (n - 1), (a (k - 1) + a k) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) →
  a n < 1 / (n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l4166_416605


namespace NUMINAMATH_CALUDE_log_equation_implies_p_zero_l4166_416690

theorem log_equation_implies_p_zero (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q + 2 > 0) :
  Real.log p - Real.log q = Real.log (p / (q + 2)) → p = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_p_zero_l4166_416690


namespace NUMINAMATH_CALUDE_not_all_points_follow_linear_relation_l4166_416651

-- Define the type for our data points
structure DataPoint where
  n : Nat
  w : Nat

-- Define our dataset
def dataset : List DataPoint := [
  { n := 1, w := 55 },
  { n := 2, w := 110 },
  { n := 3, w := 160 },
  { n := 4, w := 200 },
  { n := 5, w := 254 },
  { n := 6, w := 300 },
  { n := 7, w := 350 }
]

-- Theorem statement
theorem not_all_points_follow_linear_relation :
  ∃ point : DataPoint, point ∈ dataset ∧ point.w ≠ 55 * point.n := by
  sorry


end NUMINAMATH_CALUDE_not_all_points_follow_linear_relation_l4166_416651


namespace NUMINAMATH_CALUDE_some_number_value_l4166_416613

theorem some_number_value (n : ℝ) : 9 / (1 + n / 0.5) = 1 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l4166_416613


namespace NUMINAMATH_CALUDE_intersection_product_l4166_416642

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 21 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 52 = 0

/-- Intersection point of the two circles -/
def intersection_point (x y : ℝ) : Prop :=
  circle1 x y ∧ circle2 x y

/-- The theorem stating that the product of all coordinates of intersection points is 189 -/
theorem intersection_product : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁ * y₁ * x₂ * y₂ = 189 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_l4166_416642


namespace NUMINAMATH_CALUDE_math_team_selection_l4166_416633

theorem math_team_selection (boys : ℕ) (girls : ℕ) (team_size : ℕ) : 
  boys = 10 → girls = 12 → team_size = 8 → 
  Nat.choose (boys + girls) team_size = 319770 := by
sorry

end NUMINAMATH_CALUDE_math_team_selection_l4166_416633


namespace NUMINAMATH_CALUDE_smallest_number_l4166_416610

theorem smallest_number (a b c d : ℚ) (ha : a = -2) (hb : b = -5/2) (hc : c = 0) (hd : d = 1/5) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l4166_416610


namespace NUMINAMATH_CALUDE_temperature_difference_l4166_416648

theorem temperature_difference (N : ℝ) : 
  (∃ (M L : ℝ),
    -- Noon conditions
    M = L + N ∧
    -- 6:00 PM conditions
    (M - 11) - (L + 5) = 6 ∨ (M - 11) - (L + 5) = -6) →
  N = 22 ∨ N = 10 :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l4166_416648


namespace NUMINAMATH_CALUDE_tickets_per_box_l4166_416673

theorem tickets_per_box (total_tickets : ℕ) (num_boxes : ℕ) (h1 : total_tickets = 45) (h2 : num_boxes = 9) :
  total_tickets / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_tickets_per_box_l4166_416673


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4166_416643

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ 
  (a > -2 ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4166_416643


namespace NUMINAMATH_CALUDE_impossible_arrangement_l4166_416664

/-- Represents a 3x3 grid filled with digits --/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of available digits --/
def AvailableDigits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Check if a list of cells contains three different digits --/
def hasThreeDifferentDigits (g : Grid) (cells : List (Fin 3 × Fin 3)) : Prop :=
  (cells.map (fun (i, j) => g i j)).toFinset.card = 3

/-- Check if all rows, columns, and diagonals have three different digits --/
def isValidArrangement (g : Grid) : Prop :=
  (∀ i : Fin 3, hasThreeDifferentDigits g [(i, 0), (i, 1), (i, 2)]) ∧
  (∀ j : Fin 3, hasThreeDifferentDigits g [(0, j), (1, j), (2, j)]) ∧
  hasThreeDifferentDigits g [(0, 0), (1, 1), (2, 2)] ∧
  hasThreeDifferentDigits g [(0, 2), (1, 1), (2, 0)]

/-- Main theorem: It's impossible to arrange the digits as described --/
theorem impossible_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry


end NUMINAMATH_CALUDE_impossible_arrangement_l4166_416664


namespace NUMINAMATH_CALUDE_triangle_side_length_l4166_416615

theorem triangle_side_length (a b c : ℝ) (angle_C : ℝ) : 
  a = 3 → b = 5 → angle_C = 2 * π / 3 → c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4166_416615


namespace NUMINAMATH_CALUDE_elite_cheaper_at_min_shirts_l4166_416639

/-- Elite T-Shirt Company's pricing structure -/
def elite_cost (n : ℕ) : ℚ := 30 + 8 * n

/-- Omega T-Shirt Company's pricing structure -/
def omega_cost (n : ℕ) : ℚ := 10 + 12 * n

/-- The minimum number of shirts for which Elite is cheaper than Omega -/
def min_shirts_for_elite : ℕ := 6

theorem elite_cheaper_at_min_shirts :
  elite_cost min_shirts_for_elite < omega_cost min_shirts_for_elite ∧
  ∀ k : ℕ, k < min_shirts_for_elite → elite_cost k ≥ omega_cost k :=
by sorry

end NUMINAMATH_CALUDE_elite_cheaper_at_min_shirts_l4166_416639


namespace NUMINAMATH_CALUDE_same_color_socks_probability_l4166_416612

def total_red_socks : ℕ := 12
def total_blue_socks : ℕ := 10

theorem same_color_socks_probability :
  let total_socks := total_red_socks + total_blue_socks
  let same_color_combinations := (total_red_socks.choose 2) + (total_blue_socks.choose 2)
  let total_combinations := total_socks.choose 2
  (same_color_combinations : ℚ) / total_combinations = 37 / 77 := by
  sorry

end NUMINAMATH_CALUDE_same_color_socks_probability_l4166_416612


namespace NUMINAMATH_CALUDE_ribbon_fraction_l4166_416695

theorem ribbon_fraction (total_fraction : ℚ) (num_packages : ℕ) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_packages = 5) :
  total_fraction / num_packages = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_fraction_l4166_416695


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l4166_416663

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 1 / a + 2 / b ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l4166_416663


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4166_416688

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (abs m < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → abs m < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4166_416688


namespace NUMINAMATH_CALUDE_yangtze_farm_grass_consumption_l4166_416619

/-- Represents the grass consumption scenario on Yangtze Farm -/
structure GrassConsumption where
  /-- The amount of grass one cow eats in one day -/
  b : ℝ
  /-- The initial amount of grass -/
  g : ℝ
  /-- The rate of grass growth per day -/
  r : ℝ

/-- Given the conditions, proves that 36 cows will eat the grass in 3 days -/
theorem yangtze_farm_grass_consumption (gc : GrassConsumption) 
  (h1 : gc.g + 6 * gc.r = 24 * 6 * gc.b)  -- 24 cows eat the grass in 6 days
  (h2 : gc.g + 8 * gc.r = 21 * 8 * gc.b)  -- 21 cows eat the grass in 8 days
  : gc.g + 3 * gc.r = 36 * 3 * gc.b := by
  sorry


end NUMINAMATH_CALUDE_yangtze_farm_grass_consumption_l4166_416619


namespace NUMINAMATH_CALUDE_little_twelve_games_l4166_416682

/-- Represents a basketball conference with divisions and teams. -/
structure BasketballConference where
  num_divisions : ℕ
  teams_per_division : ℕ
  intra_division_games : ℕ
  inter_division_games : ℕ

/-- Calculates the total number of scheduled games in the conference. -/
def total_games (conf : BasketballConference) : ℕ :=
  let intra_games := conf.num_divisions * (conf.teams_per_division.choose 2) * conf.intra_division_games
  let inter_games := (conf.num_divisions * (conf.num_divisions - 1) / 2) * (conf.teams_per_division ^ 2) * conf.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the Little Twelve Basketball Conference has 102 scheduled games. -/
theorem little_twelve_games :
  let conf : BasketballConference := {
    num_divisions := 3,
    teams_per_division := 4,
    intra_division_games := 3,
    inter_division_games := 2
  }
  total_games conf = 102 := by
  sorry


end NUMINAMATH_CALUDE_little_twelve_games_l4166_416682


namespace NUMINAMATH_CALUDE_least_positive_angle_solution_l4166_416614

theorem least_positive_angle_solution (θ : Real) : 
  (θ > 0 ∧ θ < 360 ∧ Real.cos (10 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)) →
  θ = 80 := by
sorry

end NUMINAMATH_CALUDE_least_positive_angle_solution_l4166_416614


namespace NUMINAMATH_CALUDE_savings_calculation_l4166_416692

/-- Calculates a person's savings given their income and the ratio of income to expenditure. -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that for a person with an income of 36000 and an income to expenditure ratio of 9:8, their savings are 4000. -/
theorem savings_calculation :
  calculate_savings 36000 9 8 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l4166_416692


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l4166_416638

theorem fixed_point_on_graph (k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 9 * x^2 + 3 * k * x - 5 * k
  f 5 = 225 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l4166_416638


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4166_416676

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -2]

theorem perpendicular_vectors (x : ℝ) : 
  (∀ i : Fin 2, (a i) * ((a i) - (b x i)) = 0) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4166_416676


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l4166_416624

theorem no_infinite_sequence_exists : 
  ¬ (∃ (x : ℕ → ℝ), (∀ n : ℕ, x n > 0) ∧ 
    (∀ n : ℕ, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n))) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l4166_416624


namespace NUMINAMATH_CALUDE_specific_pyramid_side_edge_l4166_416683

/-- Regular square pyramid with given base edge length and volume -/
structure RegularSquarePyramid where
  base_edge : ℝ
  volume : ℝ

/-- The side edge length of a regular square pyramid -/
def side_edge_length (p : RegularSquarePyramid) : ℝ :=
  sorry

/-- Theorem stating the side edge length of a specific regular square pyramid -/
theorem specific_pyramid_side_edge :
  let p : RegularSquarePyramid := ⟨4 * Real.sqrt 2, 32⟩
  side_edge_length p = 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_side_edge_l4166_416683


namespace NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l4166_416665

/-- A linear function f(x) = mx + b does not pass through the second quadrant
    if its slope m is positive and its y-intercept b is negative. -/
theorem linear_function_not_in_second_quadrant 
  (f : ℝ → ℝ) (m b : ℝ) (h1 : ∀ x, f x = m * x + b) (h2 : m > 0) (h3 : b < 0) :
  ∃ x y, f x = y ∧ (x ≤ 0 ∨ y ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_second_quadrant_l4166_416665


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l4166_416660

theorem multiplication_addition_equality : 24 * 44 + 56 * 24 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l4166_416660


namespace NUMINAMATH_CALUDE_carols_age_l4166_416649

theorem carols_age (bob_age carol_age : ℕ) : 
  bob_age + carol_age = 66 →
  carol_age = 3 * bob_age + 2 →
  carol_age = 50 := by
sorry

end NUMINAMATH_CALUDE_carols_age_l4166_416649


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l4166_416629

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - 1) (a - 2)
  (z.re = 0 ∧ z.im ≠ 0) → (a = -1 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l4166_416629


namespace NUMINAMATH_CALUDE_marys_hourly_rate_l4166_416617

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  maxEarnings : ℚ

/-- Calculates the regular hourly rate given a work schedule --/
def regularHourlyRate (schedule : WorkSchedule) : ℚ :=
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let x := schedule.maxEarnings / (schedule.regularHours + overtimeHours * schedule.overtimeRate)
  x

/-- Theorem stating that Mary's regular hourly rate is $8 --/
theorem marys_hourly_rate :
  let schedule := WorkSchedule.mk 80 20 1.25 760
  regularHourlyRate schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_marys_hourly_rate_l4166_416617


namespace NUMINAMATH_CALUDE_quadratic_root_value_l4166_416666

theorem quadratic_root_value (a : ℝ) : 
  ((a + 1) * 1^2 - 1 + a^2 - 2*a - 2 = 0) → a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l4166_416666


namespace NUMINAMATH_CALUDE_problem_1_l4166_416635

theorem problem_1 : -7 - (-10) + (-8) = -5 := by sorry

end NUMINAMATH_CALUDE_problem_1_l4166_416635


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l4166_416607

theorem line_segment_endpoint (x : ℝ) :
  x < 0 ∧
  ((x - 1)^2 + (8 - 3)^2).sqrt = 15 →
  x = 1 - 10 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l4166_416607


namespace NUMINAMATH_CALUDE_negation_of_exists_exponential_l4166_416681

theorem negation_of_exists_exponential (x : ℝ) :
  (¬ ∃ x₀ : ℝ, (2 : ℝ) ^ x₀ ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_exponential_l4166_416681


namespace NUMINAMATH_CALUDE_larger_segment_is_70_l4166_416652

/-- A triangle with sides 40, 50, and 90 units -/
structure Triangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  ha : side_a = 40
  hb : side_b = 50
  hc : side_c = 90

/-- The altitude dropped on the side of length 90 -/
def altitude (t : Triangle) : ℝ := sorry

/-- The larger segment cut off on the side of length 90 -/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is 70 units -/
theorem larger_segment_is_70 (t : Triangle) : larger_segment t = 70 := by sorry

end NUMINAMATH_CALUDE_larger_segment_is_70_l4166_416652


namespace NUMINAMATH_CALUDE_library_wall_length_proof_l4166_416680

/-- The length of the library wall given the specified conditions -/
def library_wall_length : ℝ := 8

/-- Represents the number of desks (which is equal to the number of bookcases) -/
def num_furniture : ℕ := 2

theorem library_wall_length_proof :
  (∃ n : ℕ, 
    n = num_furniture ∧ 
    2 * n + 1.5 * n + 1 = library_wall_length ∧
    ∀ m : ℕ, m > n → 2 * m + 1.5 * m + 1 > library_wall_length) := by
  sorry

#check library_wall_length_proof

end NUMINAMATH_CALUDE_library_wall_length_proof_l4166_416680


namespace NUMINAMATH_CALUDE_ratio_of_sums_l4166_416621

theorem ratio_of_sums (p q r u v w : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_u : 0 < u) (pos_v : 0 < v) (pos_w : 0 < w)
  (sum_squares_pqr : p^2 + q^2 + r^2 = 49)
  (sum_squares_uvw : u^2 + v^2 + w^2 = 64)
  (sum_products : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l4166_416621


namespace NUMINAMATH_CALUDE_range_of_a_l4166_416644

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 2)*x + 1 ≥ 0) → 0 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4166_416644


namespace NUMINAMATH_CALUDE_green_ball_probability_l4166_416603

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The game setup -/
def game : List Container := [
  { red := 8, green := 4 },
  { red := 7, green := 4 },
  { red := 7, green := 4 }
]

/-- The probability of selecting each container -/
def containerProb : Rat := 1 / 3

/-- Calculates the probability of drawing a green ball from a given container -/
def greenProbFromContainer (c : Container) : Rat :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of drawing a green ball -/
def totalGreenProb : Rat :=
  (game.map greenProbFromContainer).sum / game.length

/-- The main theorem: the probability of drawing a green ball is 35/99 -/
theorem green_ball_probability : totalGreenProb = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l4166_416603


namespace NUMINAMATH_CALUDE_plot_length_is_58_l4166_416671

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculate_plot_length (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.length_breadth_difference

/-- Theorem stating that the length of the plot is 58 meters under given conditions -/
theorem plot_length_is_58 (plot : RectangularPlot) 
  (h1 : plot.length = plot.breadth + 16)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.length_breadth_difference = 16) : 
  calculate_plot_length plot = 58 := by
  sorry

#eval calculate_plot_length { breadth := 42, length := 58, fencing_cost_per_meter := 26.5, total_fencing_cost := 5300, length_breadth_difference := 16 }

end NUMINAMATH_CALUDE_plot_length_is_58_l4166_416671


namespace NUMINAMATH_CALUDE_compound_statement_properties_l4166_416608

/-- Given two propositions p and q, prove the compound statement properties --/
theorem compound_statement_properties (p q : Prop) 
  (hp : p ↔ (8 + 7 = 16)) 
  (hq : q ↔ (π > 3)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p := by sorry

end NUMINAMATH_CALUDE_compound_statement_properties_l4166_416608


namespace NUMINAMATH_CALUDE_quadratic_minimum_l4166_416697

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

-- State the theorem
theorem quadratic_minimum :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l4166_416697


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l4166_416686

def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - 3

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x ≤ 1) ∧
  (∃ x ∈ Set.Icc (-3/2 : ℝ) 2, f a x = 1) →
  a = 3/4 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l4166_416686


namespace NUMINAMATH_CALUDE_vector_from_origin_to_line_l4166_416601

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 4 * t + 2
  y := λ t => t + 2

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line :
  liesOnLine (6, 3) ∧ isParallel (6, 3) (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_from_origin_to_line_l4166_416601


namespace NUMINAMATH_CALUDE_same_heads_probability_l4166_416693

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 3

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem same_heads_probability :
  probability = 3 / 32 := by sorry

end NUMINAMATH_CALUDE_same_heads_probability_l4166_416693
