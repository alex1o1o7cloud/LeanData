import Mathlib

namespace NUMINAMATH_CALUDE_basketball_players_l127_12754

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 3)
  (h3 : total = 12) :
  ∃ basketball : ℕ, basketball = total - cricket + both :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l127_12754


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l127_12798

/-- The perimeter of a semi-circle with radius 21.977625925131363 cm is approximately 113.024 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 21.977625925131363
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 113.024) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l127_12798


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l127_12732

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 * t.c * (t.a - t.b) + t.b^2 * t.a * (t.b - t.c) + t.c^2 * t.b * (t.c - t.a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l127_12732


namespace NUMINAMATH_CALUDE_inequality_solution_set_l127_12734

theorem inequality_solution_set (x : ℝ) : -x^2 - 2*x + 3 > 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l127_12734


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l127_12747

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l127_12747


namespace NUMINAMATH_CALUDE_complex_power_result_l127_12768

theorem complex_power_result : (((Complex.I * Real.sqrt 2) / (1 + Complex.I)) ^ 100 : ℂ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l127_12768


namespace NUMINAMATH_CALUDE_second_digit_prime_in_powers_l127_12733

-- Define four-digit powers of 2 and 5
def fourDigitPowersOf2 : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 2^m ∧ 1000 ≤ n ∧ n < 10000}
def fourDigitPowersOf5 : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 5^m ∧ 1000 ≤ n ∧ n < 10000}

-- Function to get the second digit of a number
def secondDigit (n : ℕ) : ℕ := (n / 100) % 10

-- The theorem to prove
theorem second_digit_prime_in_powers :
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    (∃ n ∈ fourDigitPowersOf2, secondDigit n = p) ∧
    (∃ n ∈ fourDigitPowersOf5, secondDigit n = p) :=
  sorry

end NUMINAMATH_CALUDE_second_digit_prime_in_powers_l127_12733


namespace NUMINAMATH_CALUDE_correct_control_group_setup_l127_12770

/-- Represents the different media types used in the experiment -/
inductive Medium
| BeefExtractPeptone
| SelectiveUreaDecomposing

/-- Represents the different inoculation methods -/
inductive InoculationMethod
| SoilSample
| SterileWater
| NoInoculation

/-- Represents a control group setup -/
structure ControlGroup :=
  (medium : Medium)
  (inoculation : InoculationMethod)

/-- The correct control group setup for the experiment -/
def correctControlGroup : ControlGroup :=
  { medium := Medium.BeefExtractPeptone,
    inoculation := InoculationMethod.SoilSample }

/-- The experiment setup -/
structure Experiment :=
  (name : String)
  (goal : String)
  (controlGroup : ControlGroup)

/-- Theorem stating that the correct control group is the one that inoculates
    the same soil sample liquid on beef extract peptone medium -/
theorem correct_control_group_setup
  (exp : Experiment)
  (h1 : exp.name = "Separating Bacteria that Decompose Urea in Soil")
  (h2 : exp.goal = "judge whether the separation effect has been achieved")
  : exp.controlGroup = correctControlGroup := by
  sorry

end NUMINAMATH_CALUDE_correct_control_group_setup_l127_12770


namespace NUMINAMATH_CALUDE_inequality_proof_l127_12716

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l127_12716


namespace NUMINAMATH_CALUDE_complement_of_A_l127_12707

def U : Set Nat := {1, 3, 5, 7, 9}
def A : Set Nat := {1, 5, 7}

theorem complement_of_A :
  (U \ A) = {3, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l127_12707


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l127_12790

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - 8 * x + 3 < 0) ↔ (1/3 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l127_12790


namespace NUMINAMATH_CALUDE_exists_non_intersecting_line_l127_12780

/-- Represents a domino on a chessboard -/
structure Domino where
  x : Fin 6
  y : Fin 6
  horizontal : Bool

/-- Represents a 6x6 chessboard covered by 18 dominoes -/
structure ChessboardCovering where
  dominoes : Finset Domino
  count : dominoes.card = 18
  valid_placement : ∀ d ∈ dominoes, 
    if d.horizontal
    then d.x < 5
    else d.y < 5
  covers_board : ∀ x y : Fin 6, ∃ d ∈ dominoes,
    (d.x = x ∧ d.y = y) ∨
    (d.horizontal ∧ d.x = x - 1 ∧ d.y = y) ∨
    (¬d.horizontal ∧ d.x = x ∧ d.y = y - 1)

/-- Main theorem: There exists a horizontal or vertical line that doesn't intersect any domino -/
theorem exists_non_intersecting_line (c : ChessboardCovering) :
  (∃ x : Fin 5, ∀ d ∈ c.dominoes, d.x ≠ x ∧ d.x ≠ x + 1) ∨
  (∃ y : Fin 5, ∀ d ∈ c.dominoes, d.y ≠ y ∧ d.y ≠ y + 1) :=
sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_line_l127_12780


namespace NUMINAMATH_CALUDE_dinner_attendees_l127_12731

theorem dinner_attendees (total_clinks : ℕ) : total_clinks = 45 → ∃ x : ℕ, x = 10 ∧ x * (x - 1) / 2 = total_clinks := by
  sorry

end NUMINAMATH_CALUDE_dinner_attendees_l127_12731


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l127_12730

theorem polar_to_rectangular_transformation (x y : ℝ) (r θ : ℝ) 
  (h1 : x = 12 ∧ y = 5)
  (h2 : r = (x^2 + y^2).sqrt)
  (h3 : θ = Real.arctan (y / x)) :
  let new_r := 2 * r^2
  let new_θ := 3 * θ
  (new_r * Real.cos new_θ = 338 * 828 / 2197) ∧
  (new_r * Real.sin new_θ = 338 * 2035 / 2197) := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l127_12730


namespace NUMINAMATH_CALUDE_units_digit_37_power_l127_12773

/-- The units digit of 37^(5*(14^14)) is 1 -/
theorem units_digit_37_power : ∃ k : ℕ, 37^(5*(14^14)) ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_power_l127_12773


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l127_12713

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l127_12713


namespace NUMINAMATH_CALUDE_jim_can_bake_two_loaves_l127_12765

/-- The amount of flour Jim has in the cupboard (in grams) -/
def flour_cupboard : ℕ := 200

/-- The amount of flour Jim has on the kitchen counter (in grams) -/
def flour_counter : ℕ := 100

/-- The amount of flour Jim has in the pantry (in grams) -/
def flour_pantry : ℕ := 100

/-- The amount of flour required for one loaf of bread (in grams) -/
def flour_per_loaf : ℕ := 200

/-- The total amount of flour Jim has (in grams) -/
def total_flour : ℕ := flour_cupboard + flour_counter + flour_pantry

/-- The number of loaves Jim can bake -/
def loaves_baked : ℕ := total_flour / flour_per_loaf

theorem jim_can_bake_two_loaves : loaves_baked = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_can_bake_two_loaves_l127_12765


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l127_12750

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  ∃ (M : ℝ), M = 27 ∧ x^3 / y^4 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 
    3 ≤ x₀ * y₀^2 ∧ x₀ * y₀^2 ≤ 8 ∧ 
    4 ≤ x₀^2 / y₀ ∧ x₀^2 / y₀ ≤ 9 ∧ 
    x₀^3 / y₀^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l127_12750


namespace NUMINAMATH_CALUDE_dress_price_discount_l127_12752

theorem dress_price_discount (P : ℝ) : P > 0 → 
  (1 - 0.35) * (1 - 0.30) * P = 0.455 * P :=
by
  sorry

end NUMINAMATH_CALUDE_dress_price_discount_l127_12752


namespace NUMINAMATH_CALUDE_orange_straws_count_l127_12784

/-- The number of orange straws needed for each mat -/
def orange_straws : ℕ := 30

/-- The number of red straws needed for each mat -/
def red_straws : ℕ := 20

/-- The number of green straws needed for each mat -/
def green_straws : ℕ := orange_straws / 2

/-- The total number of mats -/
def total_mats : ℕ := 10

/-- The total number of straws needed for all mats -/
def total_straws : ℕ := 650

theorem orange_straws_count :
  orange_straws = 30 ∧
  red_straws = 20 ∧
  green_straws = orange_straws / 2 ∧
  total_mats * (red_straws + orange_straws + green_straws) = total_straws :=
by sorry

end NUMINAMATH_CALUDE_orange_straws_count_l127_12784


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_65_degrees_l127_12722

theorem supplement_of_complement_of_65_degrees : 
  let α : ℝ := 65
  let complement_of_α : ℝ := 90 - α
  let supplement_of_complement : ℝ := 180 - complement_of_α
  supplement_of_complement = 155 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_65_degrees_l127_12722


namespace NUMINAMATH_CALUDE_t_perimeter_is_14_l127_12789

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a T-shaped figure formed by two rectangles -/
def t_perimeter (top : Rectangle) (bottom : Rectangle) : ℝ :=
  let exposed_top := top.width
  let exposed_sides := (top.width - bottom.width) + 2 * bottom.height
  let exposed_bottom := bottom.width
  exposed_top + exposed_sides + exposed_bottom

/-- Theorem stating that the perimeter of the T-shaped figure is 14 inches -/
theorem t_perimeter_is_14 :
  let top := Rectangle.mk 6 1
  let bottom := Rectangle.mk 3 4
  t_perimeter top bottom = 14 := by
  sorry

end NUMINAMATH_CALUDE_t_perimeter_is_14_l127_12789


namespace NUMINAMATH_CALUDE_max_students_distribution_max_students_is_184_l127_12709

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem max_students_distribution (pens pencils markers : ℕ) : Prop :=
  pens = 1080 →
  pencils = 920 →
  markers = 680 →
  ∃ (students : ℕ) (pens_per_student pencils_per_student markers_per_student : ℕ),
    students > 0 ∧
    students * pens_per_student = pens ∧
    students * pencils_per_student = pencils ∧
    students * markers_per_student = markers ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student ∧
    ∀ (n : ℕ), n > students →
      ¬(∃ (p q r : ℕ),
        p > 0 ∧ q > 0 ∧ r > 0 ∧
        is_prime q ∧
        n * p = pens ∧
        n * q = pencils ∧
        n * r = markers)

theorem max_students_is_184 : max_students_distribution 1080 920 680 → 
  ∃ (pens_per_student pencils_per_student markers_per_student : ℕ),
    184 * pens_per_student = 1080 ∧
    184 * pencils_per_student = 920 ∧
    184 * markers_per_student = 680 ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student :=
by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_max_students_is_184_l127_12709


namespace NUMINAMATH_CALUDE_fourth_row_from_bottom_sum_l127_12742

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid and its properties -/
structure Grid :=
  (size : ℕ)
  (start : Position)
  (max_num : ℕ)

/-- Represents the spiral filling of the grid -/
def spiral_fill (g : Grid) : Position → ℕ := sorry

/-- The sum of the greatest and least number in a given row -/
def row_sum (g : Grid) (row : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fourth_row_from_bottom_sum :
  let g : Grid := {
    size := 16,
    start := { row := 8, col := 8 },
    max_num := 256
  }
  row_sum g 4 = 497 := by sorry

end NUMINAMATH_CALUDE_fourth_row_from_bottom_sum_l127_12742


namespace NUMINAMATH_CALUDE_domain_condition_implies_m_range_l127_12729

theorem domain_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_domain_condition_implies_m_range_l127_12729


namespace NUMINAMATH_CALUDE_red_balls_count_l127_12760

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  yellow = 8 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 9 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l127_12760


namespace NUMINAMATH_CALUDE_students_just_passed_l127_12769

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) (third_div_percent : ℚ) 
  (h_total : total = 500)
  (h_first : first_div_percent = 30 / 100)
  (h_second : second_div_percent = 45 / 100)
  (h_third : third_div_percent = 20 / 100)
  (h_sum_lt_1 : first_div_percent + second_div_percent + third_div_percent < 1) :
  total - (total * (first_div_percent + second_div_percent + third_div_percent)).floor = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l127_12769


namespace NUMINAMATH_CALUDE_isosceles_diagonal_implies_two_equal_among_four_l127_12799

-- Define a convex n-gon
structure ConvexNGon where
  n : ℕ
  sides : Fin n → ℝ
  is_convex : Bool
  n_gt_4 : n > 4

-- Define the isosceles triangle property
def isosceles_diagonal_property (polygon : ConvexNGon) : Prop :=
  ∀ (i j : Fin polygon.n), i ≠ j → 
    ∃ (k : Fin polygon.n), k ≠ i ∧ k ≠ j ∧ 
      (polygon.sides i = polygon.sides k ∨ polygon.sides j = polygon.sides k)

-- Define the property of having at least two equal sides among any four
def two_equal_among_four (polygon : ConvexNGon) : Prop :=
  ∀ (i j k l : Fin polygon.n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i →
    polygon.sides i = polygon.sides j ∨ polygon.sides i = polygon.sides k ∨
    polygon.sides i = polygon.sides l ∨ polygon.sides j = polygon.sides k ∨
    polygon.sides j = polygon.sides l ∨ polygon.sides k = polygon.sides l

-- The theorem to be proved
theorem isosceles_diagonal_implies_two_equal_among_four 
  (polygon : ConvexNGon) (h : isosceles_diagonal_property polygon) :
  two_equal_among_four polygon := by
  sorry

end NUMINAMATH_CALUDE_isosceles_diagonal_implies_two_equal_among_four_l127_12799


namespace NUMINAMATH_CALUDE_max_profit_theorem_l127_12724

-- Define the cost and profit for each pen type
def cost_A : ℝ := 5
def cost_B : ℝ := 10
def profit_A : ℝ := 2
def profit_B : ℝ := 3

-- Define the total number of pens and the constraint
def total_pens : ℕ := 300
def constraint (x : ℕ) : Prop := x ≥ 4 * (total_pens - x)

-- Define the profit function
def profit (x : ℕ) : ℝ := profit_A * x + profit_B * (total_pens - x)

theorem max_profit_theorem :
  ∃ x : ℕ, x ≤ total_pens ∧ constraint x ∧
  profit x = 660 ∧
  ∀ y : ℕ, y ≤ total_pens → constraint y → profit y ≤ profit x :=
by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l127_12724


namespace NUMINAMATH_CALUDE_not_perfect_square_sum_l127_12712

theorem not_perfect_square_sum (x y : ℤ) : 
  ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 ≠ n^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_sum_l127_12712


namespace NUMINAMATH_CALUDE_lives_lost_l127_12727

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the number of lives Kaleb lost is 25 -/
theorem lives_lost : initial_lives - remaining_lives = 25 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l127_12727


namespace NUMINAMATH_CALUDE_abc_values_l127_12753

theorem abc_values (A B C : ℝ) 
  (sum_eq : A + B = 10)
  (relation : 2 * A = 3 * B + 5)
  (product : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end NUMINAMATH_CALUDE_abc_values_l127_12753


namespace NUMINAMATH_CALUDE_chris_breath_holding_goal_l127_12796

def breath_holding_sequence (n : ℕ) : ℕ :=
  10 * n

theorem chris_breath_holding_goal :
  breath_holding_sequence 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chris_breath_holding_goal_l127_12796


namespace NUMINAMATH_CALUDE_vanessa_savings_time_l127_12714

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + weekly_savings - 1) / weekly_savings

/-- Proof that Vanessa needs exactly 3 weeks to save for the dress -/
theorem vanessa_savings_time : 
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_savings_time_l127_12714


namespace NUMINAMATH_CALUDE_min_value_theorem_l127_12719

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x - 1/y)^2 = 16*y/x) :
  (∀ x' y', x' > 0 → y' > 0 → (x' - 1/y')^2 = 16*y'/x' → x + 1/y ≤ x' + 1/y') →
  x^2 + 1/y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l127_12719


namespace NUMINAMATH_CALUDE_wheel_marking_theorem_l127_12777

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧ 
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 → 
      ∃ (n : ℕ), 0 ≤ n - R * θ / 360 ∧ n - R * θ / 360 < R / 360 := by
  sorry

end NUMINAMATH_CALUDE_wheel_marking_theorem_l127_12777


namespace NUMINAMATH_CALUDE_monica_books_l127_12739

/-- The number of books Monica read last year -/
def books_last_year : ℕ := sorry

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_books : books_last_year = 16 ∧ books_next_year = 69 := by
  sorry

end NUMINAMATH_CALUDE_monica_books_l127_12739


namespace NUMINAMATH_CALUDE_jump_rope_record_time_l127_12749

theorem jump_rope_record_time (record : ℕ) (jumps_per_second : ℕ) : 
  record = 54000 → jumps_per_second = 3 → 
  ∃ (hours : ℕ), hours = 5 ∧ hours * (jumps_per_second * 3600) > record :=
by sorry

end NUMINAMATH_CALUDE_jump_rope_record_time_l127_12749


namespace NUMINAMATH_CALUDE_circumcenter_property_l127_12746

-- Define the basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def intersection_point (c1 c2 : Circle) : Point := sorry

def tangent_point (c : Circle) (p : Point) : Point := sorry

def is_parallelogram (p1 p2 p3 p4 : Point) : Prop := sorry

def is_circumcenter (p : Point) (t : Point × Point × Point) : Prop := sorry

-- State the theorem
theorem circumcenter_property 
  (X Y A B C P : Point) 
  (c1 c2 : Circle) :
  c1.center = X →
  c2.center = Y →
  A = intersection_point c1 c2 →
  B = tangent_point c1 A →
  C = tangent_point c2 A →
  is_parallelogram P X A Y →
  is_circumcenter P (B, C, A) :=
sorry

end NUMINAMATH_CALUDE_circumcenter_property_l127_12746


namespace NUMINAMATH_CALUDE_system_solution_l127_12710

theorem system_solution (n k m : ℕ+) 
  (eq1 : n + k = (Nat.gcd n k)^2)
  (eq2 : k + m = (Nat.gcd k m)^2) :
  n = 2 ∧ k = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l127_12710


namespace NUMINAMATH_CALUDE_median_circumradius_inequality_l127_12787

/-- A triangle with medians and circumradius -/
structure Triangle where
  m_a : ℝ
  m_b : ℝ
  m_c : ℝ
  R : ℝ

/-- Theorem about the relationship between medians and circumradius of a triangle -/
theorem median_circumradius_inequality (t : Triangle) :
  t.m_a^2 + t.m_b^2 + t.m_c^2 ≤ 27 * t.R^2 / 4 ∧
  t.m_a + t.m_b + t.m_c ≤ 9 * t.R / 2 :=
by sorry

end NUMINAMATH_CALUDE_median_circumradius_inequality_l127_12787


namespace NUMINAMATH_CALUDE_three_planes_division_l127_12795

/-- A plane in three-dimensional space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- Represents the configuration of three planes in space -/
inductive PlaneConfiguration
  | AllParallel
  | TwoParallelOneIntersecting
  | IntersectAlongLine
  | IntersectPairwiseParallelLines
  | IntersectPairwiseAtPoint

/-- Counts the number of parts that three planes divide space into -/
def countParts (config : PlaneConfiguration) : ℕ :=
  match config with
  | .AllParallel => 4
  | .TwoParallelOneIntersecting => 6
  | .IntersectAlongLine => 6
  | .IntersectPairwiseParallelLines => 7
  | .IntersectPairwiseAtPoint => 8

/-- The set of possible numbers of parts -/
def possiblePartCounts : Set ℕ := {4, 6, 7, 8}

theorem three_planes_division (p1 p2 p3 : Plane3D) 
  (h : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) : 
  ∃ (config : PlaneConfiguration), countParts config ∈ possiblePartCounts := by
  sorry

end NUMINAMATH_CALUDE_three_planes_division_l127_12795


namespace NUMINAMATH_CALUDE_female_average_score_l127_12755

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 84)
  (h3 : male_count = 8)
  (h4 : female_count = 24) :
  let total_count := male_count + female_count
  let total_sum := total_average * total_count
  let male_sum := male_average * male_count
  let female_sum := total_sum - male_sum
  female_sum / female_count = 92 := by sorry

end NUMINAMATH_CALUDE_female_average_score_l127_12755


namespace NUMINAMATH_CALUDE_rick_ironing_l127_12786

/-- The number of dress shirts Rick can iron in an hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in an hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing :
  total_pieces = 27 := by sorry

end NUMINAMATH_CALUDE_rick_ironing_l127_12786


namespace NUMINAMATH_CALUDE_unique_valid_list_l127_12758

def isValidList (l : List Nat) : Prop :=
  l.length = 10 ∧
  (∀ n ∈ l, n % 2 = 0 ∧ n > 0) ∧
  (∀ i ∈ l.enum.tail, 
    let (idx, n) := i
    if n > 2 then 
      l[idx-1]? = some (n - 2)
    else 
      true) ∧
  (∀ i ∈ l.enum.tail,
    let (idx, n) := i
    if n % 4 = 0 then
      l[idx-1]? = some (n - 1)
    else
      true)

theorem unique_valid_list : 
  ∃! l : List Nat, isValidList l :=
sorry

end NUMINAMATH_CALUDE_unique_valid_list_l127_12758


namespace NUMINAMATH_CALUDE_simplified_expression_value_l127_12726

theorem simplified_expression_value (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_value_l127_12726


namespace NUMINAMATH_CALUDE_total_games_five_months_l127_12771

def games_month1 : ℕ := 32
def games_month2 : ℕ := 24
def games_month3 : ℕ := 29
def games_month4 : ℕ := 19
def games_month5 : ℕ := 34

theorem total_games_five_months :
  games_month1 + games_month2 + games_month3 + games_month4 + games_month5 = 138 := by
  sorry

end NUMINAMATH_CALUDE_total_games_five_months_l127_12771


namespace NUMINAMATH_CALUDE_cos_F_value_l127_12791

-- Define the triangle
def Triangle (DE DF : ℝ) : Prop :=
  DE > 0 ∧ DF > 0 ∧ DE < DF

-- Define right triangle
def RightTriangle (DE DF : ℝ) : Prop :=
  Triangle DE DF ∧ DE^2 + (DF^2 - DE^2) = DF^2

-- Theorem statement
theorem cos_F_value (DE DF : ℝ) :
  RightTriangle DE DF → DE = 8 → DF = 17 → Real.cos (Real.arccos (DE / DF)) = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_F_value_l127_12791


namespace NUMINAMATH_CALUDE_complex_sum_problem_l127_12772

theorem complex_sum_problem (a b c d e f : ℝ) : 
  d = 2 →
  e = -a - 2*c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = -7*Complex.I →
  b + f = -9 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l127_12772


namespace NUMINAMATH_CALUDE_min_sum_with_geometric_mean_l127_12775

theorem min_sum_with_geometric_mean (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt (3^a * 3^b) = Real.sqrt (3^a * 3^b)) → 
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (3^x * 3^y) = Real.sqrt (3^x * 3^y) → x + y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_with_geometric_mean_l127_12775


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l127_12715

theorem product_of_sum_and_cube_sum (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l127_12715


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l127_12763

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (3 + Real.sqrt x) = 4 → x = 169 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l127_12763


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l127_12793

theorem quadratic_inequality_problem (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (a = -12 ∧ b = -2) ∧
  (∀ x : ℝ, (a*x + b) / (x - 2) ≥ 0 ↔ -1/6 ≤ x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l127_12793


namespace NUMINAMATH_CALUDE_multiply_divide_equation_l127_12794

theorem multiply_divide_equation : ∃ x : ℝ, (3.242 * x) / 100 = 0.04863 ∧ abs (x - 1.5) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_equation_l127_12794


namespace NUMINAMATH_CALUDE_unused_bricks_fraction_l127_12783

def bricks_used : ℝ := 20
def bricks_remaining : ℝ := 10

theorem unused_bricks_fraction :
  bricks_remaining / (bricks_used + bricks_remaining) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unused_bricks_fraction_l127_12783


namespace NUMINAMATH_CALUDE_prob_less_than_3_l127_12792

/-- A fair cubic die with faces labeled 1 to 6 -/
structure FairDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a number less than 3 -/
def LessThan3 (d : FairDie) : Finset Nat :=
  d.faces.filter (λ x => x < 3)

/-- The probability of an event for a fair die -/
def Probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event.card : Rat) / (d.faces.card : Rat)

theorem prob_less_than_3 (d : FairDie) :
  Probability d (LessThan3 d) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_3_l127_12792


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l127_12700

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 4^5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l127_12700


namespace NUMINAMATH_CALUDE_complex_equation_implies_product_l127_12764

theorem complex_equation_implies_product (x y : ℝ) : 
  (x + Complex.I) * (3 + y * Complex.I) = (2 : ℂ) + 4 * Complex.I → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_product_l127_12764


namespace NUMINAMATH_CALUDE_trig_identity_l127_12718

theorem trig_identity (α : ℝ) :
  3.410 * (Real.sin (2 * α))^3 * Real.cos (6 * α) + 
  (Real.cos (2 * α))^3 * Real.sin (6 * α) = 
  3/4 * Real.sin (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l127_12718


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l127_12788

theorem squirrels_and_nuts (num_squirrels num_nuts : ℕ) 
  (h1 : num_squirrels = 4) 
  (h2 : num_nuts = 2) : 
  num_squirrels - num_nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l127_12788


namespace NUMINAMATH_CALUDE_lcm_of_8_9_10_21_l127_12759

theorem lcm_of_8_9_10_21 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 21)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_9_10_21_l127_12759


namespace NUMINAMATH_CALUDE_desk_height_in_cm_mm_l127_12741

/-- The height of a chair in millimeters -/
def chair_height : ℕ := 537

/-- Dong-min's height when standing on the chair, in millimeters -/
def height_on_chair : ℕ := 1900

/-- Dong-min's height when standing on the desk, in millimeters -/
def height_on_desk : ℕ := 2325

/-- The height of the desk in millimeters -/
def desk_height : ℕ := height_on_desk - (height_on_chair - chair_height)

theorem desk_height_in_cm_mm : 
  desk_height = 96 * 10 + 2 := by sorry

end NUMINAMATH_CALUDE_desk_height_in_cm_mm_l127_12741


namespace NUMINAMATH_CALUDE_marble_203_is_blue_l127_12745

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue
| Green

/-- The length of one complete cycle of marbles -/
def cycleLength : Nat := 6 + 5 + 4

/-- The position of a marble within its cycle -/
def positionInCycle (n : Nat) : Nat :=
  n % cycleLength

/-- The color of a marble at a given position within a cycle -/
def colorInCycle (pos : Nat) : Color :=
  if pos ≤ 6 then Color.Red
  else if pos ≤ 11 then Color.Blue
  else Color.Green

/-- The color of the nth marble in the sequence -/
def marbleColor (n : Nat) : Color :=
  colorInCycle (positionInCycle n)

/-- Theorem: The 203rd marble is blue -/
theorem marble_203_is_blue : marbleColor 203 = Color.Blue := by
  sorry

end NUMINAMATH_CALUDE_marble_203_is_blue_l127_12745


namespace NUMINAMATH_CALUDE_circle_sum_center_radius_l127_12778

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 4 = -y^2 + 2*y

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 5 + Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_circle_sum_center_radius_l127_12778


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l127_12701

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l127_12701


namespace NUMINAMATH_CALUDE_complex_problem_l127_12737

def complex_operation (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_problem (x y : ℂ) : 
  x = (1 - I) / (1 + I) →
  y = complex_operation (4 * I) (1 + I) (3 - x * I) (x + I) →
  y = -2 - 2 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_problem_l127_12737


namespace NUMINAMATH_CALUDE_x_power_6_minus_6x_equals_711_l127_12703

theorem x_power_6_minus_6x_equals_711 (x : ℝ) (h : x = 3) : x^6 - 6*x = 711 := by
  sorry

end NUMINAMATH_CALUDE_x_power_6_minus_6x_equals_711_l127_12703


namespace NUMINAMATH_CALUDE_change_calculation_l127_12736

/-- Calculates the change in USD given the cost per cup in Euros, payment in Euros, and USD/Euro conversion rate -/
def calculate_change_usd (cost_per_cup_eur : ℝ) (payment_eur : ℝ) (usd_per_eur : ℝ) : ℝ :=
  (payment_eur - cost_per_cup_eur) * usd_per_eur

/-- Proves that the change received is 0.4956 USD given the specified conditions -/
theorem change_calculation :
  let cost_per_cup_eur : ℝ := 0.58
  let payment_eur : ℝ := 1
  let usd_per_eur : ℝ := 1.18
  calculate_change_usd cost_per_cup_eur payment_eur usd_per_eur = 0.4956 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l127_12736


namespace NUMINAMATH_CALUDE_x_range_when_ln_x_less_than_neg_one_l127_12706

theorem x_range_when_ln_x_less_than_neg_one (x : ℝ) (h : Real.log x < -1) : 0 < x ∧ x < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_x_range_when_ln_x_less_than_neg_one_l127_12706


namespace NUMINAMATH_CALUDE_sum_of_angles_in_four_intersecting_lines_l127_12743

-- Define the angles as real numbers
variable (p q r s : ℝ)

-- Define the property of four intersecting lines
def four_intersecting_lines (p q r s : ℝ) : Prop :=
  -- Add any additional properties that define four intersecting lines
  True

-- Theorem statement
theorem sum_of_angles_in_four_intersecting_lines 
  (h : four_intersecting_lines p q r s) : 
  p + q + r + s = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_four_intersecting_lines_l127_12743


namespace NUMINAMATH_CALUDE_quadratic_function_property_l127_12762

/-- Given a quadratic function f(x) = ax^2 + bx - 3 where a ≠ 0,
    if f(2) = f(4), then f(6) = -3 -/
theorem quadratic_function_property (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - 3
  f 2 = f 4 → f 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l127_12762


namespace NUMINAMATH_CALUDE_area_scientific_notation_l127_12704

/-- Represents the area in square meters -/
def area : ℝ := 216000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 2.16

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 5

/-- Theorem stating that the area is equal to its scientific notation representation -/
theorem area_scientific_notation : area = coefficient * (10 : ℝ) ^ exponent := by sorry

end NUMINAMATH_CALUDE_area_scientific_notation_l127_12704


namespace NUMINAMATH_CALUDE_connected_triangles_theorem_l127_12766

/-- A sequence of three connected right-angled triangles -/
structure TriangleSequence where
  -- First triangle
  AE : ℝ
  BE : ℝ
  -- Second triangle
  CE : ℝ
  -- Angles
  angleAEB : Real
  angleBEC : Real
  angleCED : Real

/-- The theorem statement -/
theorem connected_triangles_theorem (t : TriangleSequence) : 
  t.AE = 20 ∧ 
  t.angleAEB = 45 ∧ 
  t.angleBEC = 45 ∧ 
  t.angleCED = 45 → 
  t.CE = 10 := by
  sorry


end NUMINAMATH_CALUDE_connected_triangles_theorem_l127_12766


namespace NUMINAMATH_CALUDE_house_sale_buyback_loss_l127_12738

/-- Represents the financial outcome of a house sale and buyback transaction -/
def houseSaleBuybackOutcome (initialValue : ℝ) (profitPercentage : ℝ) (lossPercentage : ℝ) : ℝ :=
  let salePrice := initialValue * (1 + profitPercentage)
  let buybackPrice := salePrice * (1 - lossPercentage)
  buybackPrice - initialValue

/-- Theorem stating that the financial outcome for the given scenario results in a $240 loss -/
theorem house_sale_buyback_loss :
  houseSaleBuybackOutcome 12000 0.2 0.15 = -240 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_buyback_loss_l127_12738


namespace NUMINAMATH_CALUDE_pizza_toppings_l127_12723

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 9)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  mushroom_slices - (pepperoni_slices + mushroom_slices - total_slices) = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l127_12723


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l127_12708

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l127_12708


namespace NUMINAMATH_CALUDE_incorrect_value_at_three_l127_12797

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Calculates the y-value for a given x-value using the linear function -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.k * x + f.b

/-- Theorem: The value -2 for x = 3 is incorrect for the linear function passing through (-1, 3) and (0, 2) -/
theorem incorrect_value_at_three (f : LinearFunction) 
  (h1 : f.eval (-1) = 3)
  (h2 : f.eval 0 = 2) : 
  f.eval 3 ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_value_at_three_l127_12797


namespace NUMINAMATH_CALUDE_warehouse_shoes_l127_12717

/-- The number of pairs of shoes in a warehouse -/
def total_shoes (blue green purple : ℕ) : ℕ := blue + green + purple

/-- Theorem: The total number of shoes in the warehouse is 1250 -/
theorem warehouse_shoes : ∃ (green : ℕ), 
  let blue := 540
  let purple := 355
  (green = purple) ∧ (total_shoes blue green purple = 1250) := by
  sorry

end NUMINAMATH_CALUDE_warehouse_shoes_l127_12717


namespace NUMINAMATH_CALUDE_notebook_cost_l127_12779

theorem notebook_cost (initial_amount : ℕ) (poster_cost : ℕ) (bookmark_cost : ℕ)
  (num_posters : ℕ) (num_notebooks : ℕ) (num_bookmarks : ℕ) (remaining_amount : ℕ) :
  initial_amount = 40 →
  poster_cost = 5 →
  bookmark_cost = 2 →
  num_posters = 2 →
  num_notebooks = 3 →
  num_bookmarks = 2 →
  remaining_amount = 14 →
  ∃ (notebook_cost : ℕ),
    initial_amount = num_posters * poster_cost + num_notebooks * notebook_cost +
      num_bookmarks * bookmark_cost + remaining_amount ∧
    notebook_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l127_12779


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l127_12781

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 2) : 
  a / b = 0.6 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l127_12781


namespace NUMINAMATH_CALUDE_age_difference_l127_12756

/-- Given that the sum of A's and B's ages is 18 years more than the sum of B's and C's ages,
    prove that A is 18 years older than C. -/
theorem age_difference (a b c : ℕ) (h : a + b = b + c + 18) : a = c + 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l127_12756


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_unique_l127_12744

theorem quadratic_minimum (x : ℝ) : 
  2 * x^2 - 8 * x + 1 ≥ 2 * 2^2 - 8 * 2 + 1 := by
  sorry

theorem quadratic_minimum_unique (x : ℝ) : 
  (2 * x^2 - 8 * x + 1 = 2 * 2^2 - 8 * 2 + 1) → (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_unique_l127_12744


namespace NUMINAMATH_CALUDE_problem_statement_l127_12728

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sequence x_n
def x : ℕ → ℝ := sorry

-- State the theorem
theorem problem_statement :
  (∀ a b : ℝ, a < b → f a < f b) →  -- f is monotonically increasing
  (∀ a : ℝ, f (-a) = -f a) →  -- f is odd
  (∀ n : ℕ, x (n + 1) = x n + 2) →  -- x_n is arithmetic with common difference 2
  (f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0) →  -- given condition
  x 2012 = 4005 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l127_12728


namespace NUMINAMATH_CALUDE_team_ate_96_point_5_slices_l127_12774

/-- The total number of pizza slices initially bought -/
def total_slices : ℝ := 116

/-- The number of pizza slices left after eating -/
def slices_left : ℝ := 19.5

/-- The number of pizza slices eaten by the team -/
def slices_eaten : ℝ := total_slices - slices_left

theorem team_ate_96_point_5_slices : slices_eaten = 96.5 := by
  sorry

end NUMINAMATH_CALUDE_team_ate_96_point_5_slices_l127_12774


namespace NUMINAMATH_CALUDE_matrix_commute_special_case_l127_12735

open Matrix

theorem matrix_commute_special_case 
  (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![10, 3; -2, 5]) : 
  D * C = C * D := by
sorry

end NUMINAMATH_CALUDE_matrix_commute_special_case_l127_12735


namespace NUMINAMATH_CALUDE_equations_consistency_l127_12702

/-- Given a system of equations, prove its consistency -/
theorem equations_consistency 
  (r₁ r₂ r₃ s a b c : ℝ) 
  (eq1 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = s^2)
  (eq2 : (s - b) * (s - c) * r₁ + (s - c) * (s - a) * r₂ + (s - a) * (s - b) * r₃ = r₁ * r₂ * r₃) :
  ∃ (r₁' r₂' r₃' s' a' b' c' : ℝ),
    r₁' * r₂' + r₂' * r₃' + r₃' * r₁' = s'^2 ∧
    (s' - b') * (s' - c') * r₁' + (s' - c') * (s' - a') * r₂' + (s' - a') * (s' - b') * r₃' = r₁' * r₂' * r₃' :=
by
  sorry


end NUMINAMATH_CALUDE_equations_consistency_l127_12702


namespace NUMINAMATH_CALUDE_color_copies_comparison_l127_12757

/-- The cost per color copy at print shop X -/
def cost_X : ℚ := 1.25

/-- The cost per color copy at print shop Y -/
def cost_Y : ℚ := 2.75

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℚ := 60

/-- The number of color copies being compared -/
def n : ℚ := 40

theorem color_copies_comparison :
  cost_Y * n = cost_X * n + additional_charge := by
  sorry

end NUMINAMATH_CALUDE_color_copies_comparison_l127_12757


namespace NUMINAMATH_CALUDE_correct_sums_l127_12725

theorem correct_sums (R W : ℕ) : W = 5 * R → R + W = 180 → R = 30 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_l127_12725


namespace NUMINAMATH_CALUDE_linear_equation_solution_l127_12705

theorem linear_equation_solution : ∃ (x y : ℤ), x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l127_12705


namespace NUMINAMATH_CALUDE_perpendicular_lines_line_through_P_l127_12711

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m = -3)

-- Define point P on l₂
def P_on_l₂ (m : ℝ) : Prop := l₂ m 1 (2 * m)

-- Define line l passing through P with opposite intercepts
def line_l (x y : ℝ) : Prop := 
  (2 * x - y = 0) ∨ (x - y + 1 = 0)

-- Theorem statements
theorem perpendicular_lines (m : ℝ) : 
  (∀ x y, l₁ m x y ∧ l₂ m x y → perpendicular m) := sorry

theorem line_through_P (m : ℝ) : 
  P_on_l₂ m → (∀ x y, line_l x y) := sorry

end NUMINAMATH_CALUDE_perpendicular_lines_line_through_P_l127_12711


namespace NUMINAMATH_CALUDE_minimum_vases_for_70_flowers_l127_12767

/-- Represents the capacity of each vase type -/
structure VaseCapacity where
  a : Nat
  b : Nat
  c : Nat

/-- Represents the number of each vase type -/
structure VaseCount where
  a : Nat
  b : Nat
  c : Nat

/-- Calculates the total number of flowers that can be held by the given vases -/
def totalFlowers (capacity : VaseCapacity) (count : VaseCount) : Nat :=
  capacity.a * count.a + capacity.b * count.b + capacity.c * count.c

/-- Checks if the given vase count is sufficient to hold the total number of flowers -/
def isSufficient (capacity : VaseCapacity) (count : VaseCount) (total : Nat) : Prop :=
  totalFlowers capacity count ≥ total

/-- Checks if the given vase count is the minimum required to hold the total number of flowers -/
def isMinimum (capacity : VaseCapacity) (count : VaseCount) (total : Nat) : Prop :=
  isSufficient capacity count total ∧
  ∀ (other : VaseCount), isSufficient capacity other total →
    count.a + count.b + count.c ≤ other.a + other.b + other.c

/-- Theorem: The minimum number of vases required to hold 70 flowers is 8 vases C, 1 vase B, and 0 vases A -/
theorem minimum_vases_for_70_flowers :
  let capacity := VaseCapacity.mk 4 6 8
  let count := VaseCount.mk 0 1 8
  isMinimum capacity count 70 := by
  sorry

end NUMINAMATH_CALUDE_minimum_vases_for_70_flowers_l127_12767


namespace NUMINAMATH_CALUDE_fraction_equality_l127_12785

theorem fraction_equality : (35 : ℚ) / (6 - 2/5) = 25/4 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l127_12785


namespace NUMINAMATH_CALUDE_composite_divides_factorial_l127_12748

theorem composite_divides_factorial (k n : ℕ) (P_k : ℕ) : 
  k ≥ 14 →
  P_k < k →
  (∀ p, p < k ∧ Nat.Prime p → p ≤ P_k) →
  Nat.Prime P_k →
  P_k ≥ 3 * k / 4 →
  ¬Nat.Prime n →
  n > 2 * P_k →
  n ∣ Nat.factorial (n - k) :=
by sorry

end NUMINAMATH_CALUDE_composite_divides_factorial_l127_12748


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l127_12740

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ |x + 1| + 1} = {x : ℝ | x > 0.5} := by sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | {x : ℝ | x ≤ -1} ⊆ {x : ℝ | f a x + 3*x ≤ 0}} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l127_12740


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l127_12776

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℚ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 20

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℚ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $199 -/
theorem carries_tshirt_purchase : total_cost = 199 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l127_12776


namespace NUMINAMATH_CALUDE_sum_of_five_terms_positive_l127_12720

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isMonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b → f b ≤ f a

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf_odd : isOddFunction f)
  (hf_mono : ∀ x y, 0 ≤ x → 0 ≤ y → isMonotonicallyDecreasing f x y)
  (ha_arith : isArithmeticSequence a)
  (ha3_neg : a 3 < 0) :
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_terms_positive_l127_12720


namespace NUMINAMATH_CALUDE_count_primes_with_no_three_distinct_roots_l127_12761

theorem count_primes_with_no_three_distinct_roots : 
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (∀ p ∉ S, ¬Nat.Prime p ∨ 
      ∃ (x y z : Nat), x < p ∧ y < p ∧ z < p ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      (x^3 - 5*x^2 - 22*x + 56) % p = 0 ∧
      (y^3 - 5*y^2 - 22*y + 56) % p = 0 ∧
      (z^3 - 5*z^2 - 22*z + 56) % p = 0) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_no_three_distinct_roots_l127_12761


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l127_12751

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l127_12751


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l127_12782

theorem arithmetic_sequence_product (a b c d : ℝ) (m n p : ℕ+) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  b - a = Real.sqrt 2 →
  c - b = Real.sqrt 2 →
  d - c = Real.sqrt 2 →
  a * b * c * d = 2021 →
  d = (m + Real.sqrt n) / Real.sqrt p →
  ∀ (q : ℕ+), q * q ∣ m → q = 1 →
  ∀ (q : ℕ+), q * q ∣ n → q = 1 →
  ∀ (q : ℕ+), q * q ∣ p → q = 1 →
  m + n + p = 100 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l127_12782


namespace NUMINAMATH_CALUDE_max_type_B_bins_l127_12721

def unit_price_A : ℕ := 300
def unit_price_B : ℕ := 450
def total_budget : ℕ := 8000
def total_bins : ℕ := 20

theorem max_type_B_bins :
  ∀ y : ℕ,
    y ≤ 13 ∧
    y ≤ total_bins ∧
    unit_price_A * (total_bins - y) + unit_price_B * y ≤ total_budget ∧
    (∀ z : ℕ, z > y →
      z > 13 ∨
      z > total_bins ∨
      unit_price_A * (total_bins - z) + unit_price_B * z > total_budget) :=
by sorry

end NUMINAMATH_CALUDE_max_type_B_bins_l127_12721
