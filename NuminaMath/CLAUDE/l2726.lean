import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2726_272689

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_in_first_quadrant : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2726_272689


namespace NUMINAMATH_CALUDE_no_special_polynomials_l2726_272657

/-- Represents a polynomial of the form x^4 + ax^3 + bx^2 + cx + 2048 -/
def SpecialPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + 2048

/-- Predicate to check if a complex number is a root of the polynomial -/
def IsRoot (a b c : ℝ) (s : ℂ) : Prop :=
  SpecialPolynomial a b c s = 0

/-- Predicate to check if the polynomial satisfies the special root property -/
def HasSpecialRootProperty (a b c : ℝ) : Prop :=
  ∀ s : ℂ, IsRoot a b c s → IsRoot a b c (s^2) ∧ IsRoot a b c (s⁻¹)

theorem no_special_polynomials :
  ¬∃ a b c : ℝ, HasSpecialRootProperty a b c :=
sorry

end NUMINAMATH_CALUDE_no_special_polynomials_l2726_272657


namespace NUMINAMATH_CALUDE_odd_power_congruence_l2726_272698

theorem odd_power_congruence (a n : ℕ) (h_odd : Odd a) (h_pos : 0 < n) :
  (a ^ (2 ^ n)) ≡ 1 [MOD 2 ^ (n + 2)] := by
  sorry

end NUMINAMATH_CALUDE_odd_power_congruence_l2726_272698


namespace NUMINAMATH_CALUDE_tan_alpha_results_l2726_272666

theorem tan_alpha_results (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.sin (2*α) / (Real.sin α^2 + Real.sin α * Real.cos α - Real.cos (2*α) - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_results_l2726_272666


namespace NUMINAMATH_CALUDE_average_equation_solution_l2726_272629

theorem average_equation_solution (x : ℝ) : 
  ((x + 3) + (4 * x + 1) + (3 * x + 6)) / 3 = 3 * x - 8 → x = 34 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l2726_272629


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2726_272628

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : neither = 5)
  (h5 : badminton + tennis - (badminton + tennis - total + neither) = total - neither) :
  badminton + tennis - total + neither = 3 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2726_272628


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_planes_parallel_to_same_plane_are_parallel_l2726_272687

-- Define the basic types for our geometric objects
variable (Point Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Statement 1: Two lines parallel to the same line are parallel
theorem lines_parallel_to_same_line_are_parallel 
  (A B C : Line) 
  (h1 : parallel_lines A B) 
  (h2 : parallel_lines B C) : 
  parallel_lines A C :=
sorry

-- Statement 2: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_same_plane_are_parallel 
  (X Y Z : Plane) 
  (h1 : parallel_planes X Y) 
  (h2 : parallel_planes Y Z) : 
  parallel_planes X Z :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_planes_parallel_to_same_plane_are_parallel_l2726_272687


namespace NUMINAMATH_CALUDE_small_months_not_remainder_l2726_272612

/-- The number of months in a year -/
def total_months : ℕ := 12

/-- The number of big months in a year -/
def big_months : ℕ := 7

/-- The number of small months in a year -/
def small_months : ℕ := 4

/-- February is a special case and not counted as either big or small -/
def february_special : Prop := True

theorem small_months_not_remainder :
  small_months ≠ total_months - big_months :=
by sorry

end NUMINAMATH_CALUDE_small_months_not_remainder_l2726_272612


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l2726_272619

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := a * 10 + b / 99

/-- The main theorem stating that the ratio of two specific repeating decimals equals 9/4 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 8 1) / (RepeatingDecimal 3 6) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l2726_272619


namespace NUMINAMATH_CALUDE_arthur_baked_115_muffins_l2726_272694

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The number of muffins James baked -/
def james_muffins : ℕ := 1380

/-- James baked 12 times as many muffins as Arthur -/
axiom james_baked_12_times : james_muffins = 12 * arthur_muffins

theorem arthur_baked_115_muffins : arthur_muffins = 115 := by
  sorry

end NUMINAMATH_CALUDE_arthur_baked_115_muffins_l2726_272694


namespace NUMINAMATH_CALUDE_panda_bamboo_consumption_l2726_272669

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily * 7) + (baby_daily * 7)

/-- Theorem: The total amount of bamboo eaten by an adult panda and a baby panda in a week is 1316 pounds -/
theorem panda_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end NUMINAMATH_CALUDE_panda_bamboo_consumption_l2726_272669


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l2726_272637

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l2726_272637


namespace NUMINAMATH_CALUDE_triangle_inequality_l2726_272639

/-- Given a triangle with side lengths a, b, and c, 
    prove that a(b-c)^2 + b(c-a)^2 + c(a-b)^2 + 4abc > a^3 + b^3 + c^3 -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2726_272639


namespace NUMINAMATH_CALUDE_modulus_z_is_sqrt_2_l2726_272618

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The complex number z -/
noncomputable def z : ℂ := (2 * i) / (1 + i)

/-- Theorem: The modulus of z is √2 -/
theorem modulus_z_is_sqrt_2 : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_z_is_sqrt_2_l2726_272618


namespace NUMINAMATH_CALUDE_emma_sandwich_combinations_l2726_272632

def num_meat : ℕ := 12
def num_cheese : ℕ := 11

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2)

theorem emma_sandwich_combinations :
  sandwich_combinations = 3630 := by sorry

end NUMINAMATH_CALUDE_emma_sandwich_combinations_l2726_272632


namespace NUMINAMATH_CALUDE_min_weighings_nine_medals_l2726_272670

/-- Represents a set of medals with one heavier than the others -/
structure MedalSet :=
  (total : Nat)
  (heavier_exists : total > 0)

/-- Represents a balance scale used for weighing -/
structure BalanceScale

/-- The minimum number of weighings required to find the heavier medal -/
def min_weighings (medals : MedalSet) (scale : BalanceScale) : Nat :=
  sorry

/-- Theorem stating that for 9 medals, the minimum number of weighings is 2 -/
theorem min_weighings_nine_medals :
  ∀ (scale : BalanceScale),
  min_weighings ⟨9, by norm_num⟩ scale = 2 :=
sorry

end NUMINAMATH_CALUDE_min_weighings_nine_medals_l2726_272670


namespace NUMINAMATH_CALUDE_min_value_inequality_l2726_272604

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = Real.sqrt 3) :
  1/a^2 + 2/b^2 ≥ 2 ∧ ∀ x, f x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2726_272604


namespace NUMINAMATH_CALUDE_rectangle_with_equal_adjacent_sides_is_square_l2726_272616

-- Define a rectangle
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_positive : length > 0)
  (width_positive : width > 0)

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Theorem: A rectangle with one pair of adjacent sides equal is a square
theorem rectangle_with_equal_adjacent_sides_is_square (r : Rectangle) 
  (h : r.length = r.width) : ∃ (s : Square), s.side = r.length :=
sorry

end NUMINAMATH_CALUDE_rectangle_with_equal_adjacent_sides_is_square_l2726_272616


namespace NUMINAMATH_CALUDE_always_two_real_roots_integer_roots_condition_l2726_272645

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - a*x + (a - 1)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (a : ℝ) :
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x = 0 ∧ quadratic_equation a y = 0 :=
sorry

-- Theorem 2: When roots are integers and one is twice the other, a = 3
theorem integer_roots_condition (a : ℝ) :
  (∃ x y : ℤ, x ≠ y ∧ quadratic_equation a (x : ℝ) = 0 ∧ quadratic_equation a (y : ℝ) = 0 ∧ y = 2*x) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_integer_roots_condition_l2726_272645


namespace NUMINAMATH_CALUDE_red_balls_count_l2726_272630

/-- The number of red balls in a bag with given conditions -/
theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) (h1 : total = 100) (h2 : white = 50) (h3 : green = 30) 
    (h4 : yellow = 10) (h5 : purple = 3) (h6 : prob_not_red_purple = 9/10) : 
    total - (white + green + yellow + purple) = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2726_272630


namespace NUMINAMATH_CALUDE_right_triangle_sin_cos_l2726_272699

/-- In a right triangle XYZ with ∠Y = 90°, hypotenuse XZ = 15, and leg XY = 9, sin X = 4/5 and cos X = 3/5 -/
theorem right_triangle_sin_cos (X Y Z : ℝ) (h1 : X^2 + Y^2 = Z^2) (h2 : Z = 15) (h3 : X = 9) :
  Real.sin (Real.arccos (X / Z)) = 4/5 ∧ Real.cos (Real.arccos (X / Z)) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_cos_l2726_272699


namespace NUMINAMATH_CALUDE_cube_volume_increase_l2726_272660

theorem cube_volume_increase (a : ℝ) (ha : a > 0) :
  (2 * a)^3 - a^3 = 7 * a^3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l2726_272660


namespace NUMINAMATH_CALUDE_promotion_b_saves_more_l2726_272643

/-- The cost of a single shirt in dollars -/
def shirtCost : ℝ := 40

/-- The cost of two shirts under Promotion A -/
def promotionACost : ℝ := shirtCost + (shirtCost * 0.75)

/-- The cost of two shirts under Promotion B -/
def promotionBCost : ℝ := shirtCost + (shirtCost - 15)

/-- Theorem stating that Promotion B costs $5 less than Promotion A -/
theorem promotion_b_saves_more :
  promotionACost - promotionBCost = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_b_saves_more_l2726_272643


namespace NUMINAMATH_CALUDE_marathon_remainder_l2726_272605

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder (m : ℕ) (y : ℕ) 
    (h : Distance.mk m y = 
      { miles := num_marathons * marathon.miles + (num_marathons * marathon.yards) / yards_per_mile,
        yards := (num_marathons * marathon.yards) % yards_per_mile }) :
  y = 495 := by sorry

end NUMINAMATH_CALUDE_marathon_remainder_l2726_272605


namespace NUMINAMATH_CALUDE_power_of_seven_l2726_272667

theorem power_of_seven (k : ℕ) : (7 : ℝ) ^ (4 * k + 2) = 784 → (7 : ℝ) ^ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l2726_272667


namespace NUMINAMATH_CALUDE_cube_sum_of_conjugate_fractions_l2726_272613

theorem cube_sum_of_conjugate_fractions :
  let x := (2 + Real.sqrt 3) / (2 - Real.sqrt 3)
  let y := (2 - Real.sqrt 3) / (2 + Real.sqrt 3)
  x^3 + y^3 = 2702 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_conjugate_fractions_l2726_272613


namespace NUMINAMATH_CALUDE_race_result_l2726_272650

/-- The race between John and Steve --/
theorem race_result (initial_distance : ℝ) (john_speed steve_speed : ℝ) (time : ℝ) :
  initial_distance = 14 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  time = 32 →
  (john_speed * time + initial_distance) - (steve_speed * time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l2726_272650


namespace NUMINAMATH_CALUDE_buratino_spent_10_dollars_l2726_272679

/-- Represents a transaction at the exchange point -/
inductive Transaction
  | type1  -- Give 2 euros, receive 3 dollars and a candy
  | type2  -- Give 5 dollars, receive 3 euros and a candy

/-- Represents Buratino's exchange activities -/
structure ExchangeActivity where
  transactions : List Transaction
  initialDollars : ℕ
  finalDollars : ℕ
  finalEuros : ℕ
  candiesReceived : ℕ

/-- Calculates the net dollar change for a given transaction -/
def netDollarChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => 3
  | Transaction.type2 => -5

/-- Calculates the net euro change for a given transaction -/
def netEuroChange (t : Transaction) : ℤ :=
  match t with
  | Transaction.type1 => -2
  | Transaction.type2 => 3

/-- Theorem stating that Buratino spent 10 dollars -/
theorem buratino_spent_10_dollars (activity : ExchangeActivity) :
  activity.candiesReceived = 50 ∧
  activity.finalEuros = 0 ∧
  activity.finalDollars < activity.initialDollars →
  activity.initialDollars - activity.finalDollars = 10 := by
  sorry


end NUMINAMATH_CALUDE_buratino_spent_10_dollars_l2726_272679


namespace NUMINAMATH_CALUDE_fold_equilateral_triangle_l2726_272677

-- Define an equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the folding operation
def FoldTriangle (A B C P Q : ℝ × ℝ) : Prop :=
  P.1 = B.1 + 7 * (A.1 - B.1) / 10 ∧
  P.2 = B.2 + 7 * (A.2 - B.2) / 10 ∧
  Q.1 = C.1 + 7 * (A.1 - C.1) / 10 ∧
  Q.2 = C.2 + 7 * (A.2 - C.2) / 10

theorem fold_equilateral_triangle :
  ∀ (A B C P Q : ℝ × ℝ),
  EquilateralTriangle A B C →
  dist A B = 10 →
  FoldTriangle A B C P Q →
  (dist P Q)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fold_equilateral_triangle_l2726_272677


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l2726_272641

theorem complement_intersection_equals_set (U M N : Set ℕ) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 3, 4} →
  N = {2, 4, 5} →
  (U \ M) ∩ N = {2, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l2726_272641


namespace NUMINAMATH_CALUDE_inverse_fifty_l2726_272649

theorem inverse_fifty (x : ℝ) : (1 / x = 50) → (x = 1 / 50) := by
  sorry

end NUMINAMATH_CALUDE_inverse_fifty_l2726_272649


namespace NUMINAMATH_CALUDE_daves_hourly_wage_l2726_272610

/-- Dave's hourly wage calculation --/
theorem daves_hourly_wage (monday_hours tuesday_hours total_amount : ℕ) 
  (h1 : monday_hours = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_amount = 48) :
  total_amount / (monday_hours + tuesday_hours) = 6 := by
  sorry

end NUMINAMATH_CALUDE_daves_hourly_wage_l2726_272610


namespace NUMINAMATH_CALUDE_remainder_sum_l2726_272662

theorem remainder_sum (a b : ℤ) 
  (ha : a % 84 = 77) 
  (hb : b % 120 = 113) : 
  (a + b) % 42 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2726_272662


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2726_272668

theorem x_plus_y_values (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (h : x - y > 0) :
  x + y = 8 ∨ x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2726_272668


namespace NUMINAMATH_CALUDE_fraction_square_value_l2726_272648

theorem fraction_square_value (x y : ℚ) (hx : x = 3) (hy : y = 5) :
  ((1 / y) / (1 / x))^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_value_l2726_272648


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_area_is_twelve_l2726_272655

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5, and sides AB and AD of lengths 3 and 4 respectively. -/
structure RightQuadrilateral where
  AC : ℝ
  AB : ℝ
  AD : ℝ
  ac_eq : AC = 5
  ab_eq : AB = 3
  ad_eq : AD = 4

/-- The area of the RightQuadrilateral is 12. -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : ℝ :=
  12

/-- The area of a RightQuadrilateral is equal to 12. -/
theorem area_is_twelve (q : RightQuadrilateral) : right_quadrilateral_area q = 12 := by
  sorry

end NUMINAMATH_CALUDE_right_quadrilateral_area_area_is_twelve_l2726_272655


namespace NUMINAMATH_CALUDE_evaluate_expression_l2726_272608

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2726_272608


namespace NUMINAMATH_CALUDE_olivia_dvd_count_l2726_272678

theorem olivia_dvd_count (dvds_per_season : ℕ) (seasons_bought : ℕ) : 
  dvds_per_season = 8 → seasons_bought = 5 → dvds_per_season * seasons_bought = 40 :=
by sorry

end NUMINAMATH_CALUDE_olivia_dvd_count_l2726_272678


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2726_272652

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧ Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) ≤ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 2 ∧
  Real.sqrt x₀ + Real.sqrt (2 * y₀) + Real.sqrt (3 * z₀) = m :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2726_272652


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2726_272617

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x : ℝ, |x - 2| - |x + 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l2726_272617


namespace NUMINAMATH_CALUDE_books_left_over_l2726_272601

theorem books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ)
  (h1 : initial_boxes = 1575)
  (h2 : books_per_initial_box = 45)
  (h3 : books_per_new_box = 46) :
  (initial_boxes * books_per_initial_box) % books_per_new_box = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_left_over_l2726_272601


namespace NUMINAMATH_CALUDE_xyz_product_l2726_272600

/-- Given complex numbers x, y, and z satisfying the specified equations,
    prove that their product equals 260/3. -/
theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -25) :
  x * y * z = 260 / 3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l2726_272600


namespace NUMINAMATH_CALUDE_find_a_value_l2726_272695

theorem find_a_value (A B : Set ℝ) (a : ℝ) :
  A = {2^a, 3} →
  B = {2, 3} →
  A ∪ B = {1, 2, 3} →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l2726_272695


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2726_272692

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l2726_272692


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l2726_272607

/-- A right cone with a sphere inscribed inside it. -/
structure InscribedSphere where
  /-- The base radius of the cone in cm. -/
  base_radius : ℝ
  /-- The height of the cone in cm. -/
  cone_height : ℝ
  /-- The radius of the inscribed sphere in cm. -/
  sphere_radius : ℝ
  /-- The base radius is 9 cm. -/
  base_radius_eq : base_radius = 9
  /-- The cone height is 27 cm. -/
  cone_height_eq : cone_height = 27
  /-- The sphere is inscribed in the cone. -/
  inscribed : sphere_radius ≤ base_radius ∧ sphere_radius ≤ cone_height

/-- The radius of the inscribed sphere is 3√10 - 3 cm. -/
theorem inscribed_sphere_radius (s : InscribedSphere) : 
  s.sphere_radius = 3 * Real.sqrt 10 - 3 := by
  sorry

#check inscribed_sphere_radius

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l2726_272607


namespace NUMINAMATH_CALUDE_gails_wallet_l2726_272681

/-- Represents the contents of Gail's wallet -/
structure Wallet where
  total : ℕ
  five_dollar_bills : ℕ
  twenty_dollar_bills : ℕ
  ten_dollar_bills : ℕ

/-- Calculates the total amount in the wallet based on the bill counts -/
def wallet_total (w : Wallet) : ℕ :=
  5 * w.five_dollar_bills + 20 * w.twenty_dollar_bills + 10 * w.ten_dollar_bills

/-- Theorem stating that given the conditions, Gail has 2 ten-dollar bills -/
theorem gails_wallet :
  ∃ (w : Wallet),
    w.total = 100 ∧
    w.five_dollar_bills = 4 ∧
    w.twenty_dollar_bills = 3 ∧
    wallet_total w = w.total ∧
    w.ten_dollar_bills = 2 := by
  sorry


end NUMINAMATH_CALUDE_gails_wallet_l2726_272681


namespace NUMINAMATH_CALUDE_sangita_cross_country_hours_l2726_272661

/-- Calculates the cross-country flying hours completed by Sangita --/
def cross_country_hours (total_required : ℕ) (day_flying : ℕ) (night_flying : ℕ) 
  (hours_per_month : ℕ) (duration_months : ℕ) : ℕ :=
  total_required - (day_flying + night_flying)

/-- Theorem stating that Sangita's cross-country flying hours equal 1261 --/
theorem sangita_cross_country_hours : 
  cross_country_hours 1500 50 9 220 6 = 1261 := by
  sorry

#eval cross_country_hours 1500 50 9 220 6

end NUMINAMATH_CALUDE_sangita_cross_country_hours_l2726_272661


namespace NUMINAMATH_CALUDE_marc_total_spend_l2726_272686

/-- The total amount spent by Marc on his purchase of model cars, paint bottles, and paintbrushes. -/
def total_spent (num_cars num_paint num_brushes : ℕ) (price_car price_paint price_brush : ℚ) : ℚ :=
  num_cars * price_car + num_paint * price_paint + num_brushes * price_brush

/-- Theorem stating that Marc's total spend is $160 given his purchases. -/
theorem marc_total_spend :
  total_spent 5 5 5 20 10 2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_marc_total_spend_l2726_272686


namespace NUMINAMATH_CALUDE_flight_duration_sum_l2726_272602

/-- Represents a time with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem flight_duration_sum (departureLA : Time) (arrivalNY : Time) (h m : ℕ) :
  departureLA.hours = 9 →
  departureLA.minutes = 15 →
  arrivalNY.hours = 17 →
  arrivalNY.minutes = 40 →
  0 < m →
  m < 60 →
  timeDiffInMinutes 
    {hours := departureLA.hours + 3, minutes := departureLA.minutes, valid := sorry}
    arrivalNY = h * 60 + m →
  h + m = 30 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l2726_272602


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l2726_272658

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 6 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 12 * x

theorem unique_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l2726_272658


namespace NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l2726_272673

-- Problem 1
theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 4 / x + 5 ≥ 9 :=
sorry

-- Problem 2
theorem max_value_theorem (x : ℝ) (h1 : x > 0) (h2 : x < 1/2) :
  1/2 * x * (1 - 2*x) ≤ 1/16 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_max_value_theorem_l2726_272673


namespace NUMINAMATH_CALUDE_earl_owes_fred_l2726_272620

/-- Represents the financial state of Earl, Fred, and Greg -/
structure FinancialState where
  earl : Int
  fred : Int
  greg : Int

/-- Calculates the final financial state after debts are paid -/
def finalState (initial : FinancialState) (earlOwes : Int) : FinancialState :=
  { earl := initial.earl - earlOwes + 40,
    fred := initial.fred + earlOwes - 32,
    greg := initial.greg + 32 - 40 }

/-- The theorem to be proved -/
theorem earl_owes_fred (initial : FinancialState) :
  initial.earl = 90 →
  initial.fred = 48 →
  initial.greg = 36 →
  (let final := finalState initial 28
   final.earl + final.greg = 130) :=
by sorry

end NUMINAMATH_CALUDE_earl_owes_fred_l2726_272620


namespace NUMINAMATH_CALUDE_number_and_percentage_l2726_272675

theorem number_and_percentage (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 35) : 
  (40/100) * N = 420 := by
sorry

end NUMINAMATH_CALUDE_number_and_percentage_l2726_272675


namespace NUMINAMATH_CALUDE_first_number_in_second_set_l2726_272697

theorem first_number_in_second_set (x : ℝ) : 
  (24 + 35 + 58) / 3 = ((x + 51 + 29) / 3) + 6 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_number_in_second_set_l2726_272697


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_solve_unknown_blanket_rate_l2726_272622

/-- Proves that the unknown rate of two blankets is 275, given the conditions of the problem --/
theorem unknown_blanket_rate : ℕ → Prop := fun x =>
  let total_blankets : ℕ := 12
  let average_price : ℕ := 150
  let total_cost : ℕ := total_blankets * average_price
  let known_cost : ℕ := 5 * 100 + 5 * 150
  2 * x = total_cost - known_cost → x = 275

/-- Solution to the unknown_blanket_rate theorem --/
theorem solve_unknown_blanket_rate : unknown_blanket_rate 275 := by
  sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_solve_unknown_blanket_rate_l2726_272622


namespace NUMINAMATH_CALUDE_percentage_difference_l2726_272674

theorem percentage_difference (A B C : ℝ) (hpos : 0 < C ∧ C < B ∧ B < A) :
  let x := 100 * (A - B) / B
  let y := 100 * (A - C) / C
  (∃ k, A = B * (1 + k/100) → x = 100 * (A - B) / B) ∧
  (∃ m, A = C * (1 + m/100) → y = 100 * (A - C) / C) := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2726_272674


namespace NUMINAMATH_CALUDE_reimu_win_probability_l2726_272633

/-- Represents a coin with two sides that can be colored -/
structure Coin :=
  (side1 : Color)
  (side2 : Color)

/-- Possible colors for a coin side -/
inductive Color
  | White
  | Red
  | Green

/-- The game state -/
structure GameState :=
  (coins : List Coin)
  (currentPlayer : Player)

/-- The players in the game -/
inductive Player
  | Reimu
  | Sanae

/-- The result of the game -/
inductive GameResult
  | ReimuWins
  | SanaeWins
  | Tie

/-- Represents an optimal strategy for playing the game -/
def OptimalStrategy := GameState → Color

/-- The probability of a specific game result given optimal play -/
def resultProbability (strategy : OptimalStrategy) (result : GameResult) : ℚ :=
  sorry

/-- Theorem stating the probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (strategy : OptimalStrategy) :
  resultProbability strategy GameResult.ReimuWins = 5 / 16 :=
sorry

end NUMINAMATH_CALUDE_reimu_win_probability_l2726_272633


namespace NUMINAMATH_CALUDE_even_function_symmetry_l2726_272690

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def has_min_value_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem even_function_symmetry (f : ℝ → ℝ) :
  is_even_function f →
  is_increasing_on f 3 7 →
  has_min_value_on f 3 7 2 →
  is_decreasing_on f (-7) (-3) ∧ has_min_value_on f (-7) (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l2726_272690


namespace NUMINAMATH_CALUDE_max_abs_z_value_l2726_272611

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b) 
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : 2 * a * z^2 + b * z + c * z = 0) : 
  Complex.abs z ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l2726_272611


namespace NUMINAMATH_CALUDE_min_value_sum_l2726_272672

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 12 + 16 * Real.sqrt 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    1 / (a₀ + 3) + 1 / (b₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ = 12 + 16 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l2726_272672


namespace NUMINAMATH_CALUDE_digit_repetition_property_l2726_272665

def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

theorem digit_repetition_property (n : ℕ) (h : n > 0) :
  (repeat_digit 6 n)^2 + repeat_digit 8 n = repeat_digit 4 (2*n) :=
sorry

end NUMINAMATH_CALUDE_digit_repetition_property_l2726_272665


namespace NUMINAMATH_CALUDE_distance_between_axes_of_symmetry_l2726_272609

/-- The distance between two adjacent axes of symmetry in the graph of y = 3sin(2x + π/4) is π/2 -/
theorem distance_between_axes_of_symmetry :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (2 * x + π / 4)
  ∃ d : ℝ, d = π / 2 ∧ ∀ x : ℝ, f (x + d) = f x := by sorry

end NUMINAMATH_CALUDE_distance_between_axes_of_symmetry_l2726_272609


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2726_272688

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ / b₁ = a₂ / b₂

/-- Definition of line l₁ -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- Definition of line l₂ -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2726_272688


namespace NUMINAMATH_CALUDE_snail_distance_bound_l2726_272684

/-- Represents the crawling of a snail over time -/
structure SnailCrawl where
  -- The distance function of the snail over time
  distance : ℝ → ℝ
  -- The distance function is non-decreasing (snail doesn't move backward)
  monotone : Monotone distance
  -- The total observation time
  total_time : ℝ
  -- The total time is 6 minutes
  total_time_is_six : total_time = 6

/-- Represents an observation of the snail -/
structure Observation where
  -- Start time of the observation
  start_time : ℝ
  -- Duration of the observation (1 minute)
  duration : ℝ
  duration_is_one : duration = 1
  -- The observation starts within the total time
  start_within_total : start_time ≥ 0 ∧ start_time + duration ≤ 6

/-- The theorem stating that the snail's total distance is at most 10 meters -/
theorem snail_distance_bound (crawl : SnailCrawl) 
  (observations : List Observation) 
  (observed_distance : ∀ obs ∈ observations, 
    crawl.distance (obs.start_time + obs.duration) - crawl.distance obs.start_time = 1) :
  crawl.distance crawl.total_time - crawl.distance 0 ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_bound_l2726_272684


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2726_272651

theorem smallest_number_of_eggs (total_containers : ℕ) (eggs_per_full_container : ℕ) 
  (underfilled_containers : ℕ) (eggs_per_underfilled : ℕ) : ℕ :=
  let total_eggs := (total_containers - underfilled_containers) * eggs_per_full_container + 
                    underfilled_containers * eggs_per_underfilled
  have h1 : eggs_per_full_container = 15 := by sorry
  have h2 : underfilled_containers = 3 := by sorry
  have h3 : eggs_per_underfilled = 14 := by sorry
  have h4 : total_eggs > 150 := by sorry
  have h5 : ∀ n : ℕ, n < total_containers → 
            n * eggs_per_full_container - underfilled_containers ≤ 150 := by sorry
  total_eggs

theorem smallest_number_of_eggs_is_162 : 
  smallest_number_of_eggs 11 15 3 14 = 162 := by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_smallest_number_of_eggs_is_162_l2726_272651


namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l2726_272693

/-- Represents the e-commerce platform's T-shirt sales scenario -/
structure TShirtSales where
  cost : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  min_price : ℝ
  max_margin : ℝ

/-- Calculates the profit for a given selling price -/
def profit (s : TShirtSales) (price : ℝ) : ℝ :=
  (price - s.cost) * (s.initial_sales + s.price_sensitivity * (s.initial_price - price))

/-- Theorem stating the maximum profit and optimal price -/
theorem max_profit_at_optimal_price (s : TShirtSales) 
  (h_cost : s.cost = 40)
  (h_initial_price : s.initial_price = 60)
  (h_initial_sales : s.initial_sales = 500)
  (h_price_sensitivity : s.price_sensitivity = 50)
  (h_min_price : s.min_price = s.cost)
  (h_max_margin : s.max_margin = 0.3)
  (h_price_range : ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → 
    profit s p ≤ profit s 52) :
  profit s 52 = 10800 ∧ 
  ∀ p, s.min_price ≤ p ∧ p ≤ s.cost * (1 + s.max_margin) → profit s p ≤ 10800 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l2726_272693


namespace NUMINAMATH_CALUDE_larry_road_trip_money_l2726_272663

theorem larry_road_trip_money (initial_money : ℝ) : 
  initial_money * (1 - 0.04 - 0.30) - 52 = 368 → 
  initial_money = 636.36 := by
sorry

end NUMINAMATH_CALUDE_larry_road_trip_money_l2726_272663


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2726_272625

theorem no_integer_solutions : 
  ¬ ∃ (x : ℤ), (x^2 - 3*x + 2)^2 - 3*(x^2 - 3*x) - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2726_272625


namespace NUMINAMATH_CALUDE_multiplier_problem_l2726_272653

theorem multiplier_problem (n : ℝ) (m : ℝ) (h1 : n = 3) (h2 : m * n = 3 * n + 12) : m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l2726_272653


namespace NUMINAMATH_CALUDE_clothing_purchase_optimal_l2726_272696

/-- Represents the prices and quantities of clothing types A and B -/
structure ClothingPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- The conditions and solution for the clothing purchase problem -/
def clothing_problem (p : ClothingPrices) : Prop :=
  -- Conditions
  p.price_a + 2 * p.price_b = 110 ∧
  2 * p.price_a + 3 * p.price_b = 190 ∧
  p.quantity_a + p.quantity_b = 100 ∧
  p.quantity_a ≥ (p.quantity_b : ℝ) / 3 ∧
  -- Solution
  p.price_a = 50 ∧
  p.price_b = 30 ∧
  p.quantity_a = 25 ∧
  p.quantity_b = 75

/-- The total cost of purchasing the clothing with the discount -/
def total_cost (p : ClothingPrices) : ℝ :=
  (p.price_a - 5) * p.quantity_a + p.price_b * p.quantity_b

/-- Theorem stating that the given solution minimizes the cost -/
theorem clothing_purchase_optimal (p : ClothingPrices) :
  clothing_problem p →
  total_cost p = 3375 ∧
  (∀ q : ClothingPrices, clothing_problem q → total_cost q ≥ total_cost p) :=
sorry

end NUMINAMATH_CALUDE_clothing_purchase_optimal_l2726_272696


namespace NUMINAMATH_CALUDE_farmer_randy_planting_l2726_272606

/-- Calculates the number of acres each tractor needs to plant per day -/
def acres_per_tractor_per_day (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) : ℚ :=
  total_acres / (tractors_first_period * days_first_period + 
                 tractors_second_period * days_second_period)

theorem farmer_randy_planting (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) 
  (h1 : total_acres = 1700)
  (h2 : total_days = 5)
  (h3 : tractors_first_period = 2)
  (h4 : days_first_period = 2)
  (h5 : tractors_second_period = 7)
  (h6 : days_second_period = 3)
  (h7 : total_days = days_first_period + days_second_period) :
  acres_per_tractor_per_day total_acres total_days 
    tractors_first_period days_first_period
    tractors_second_period days_second_period = 68 := by
  sorry

#eval acres_per_tractor_per_day 1700 5 2 2 7 3

end NUMINAMATH_CALUDE_farmer_randy_planting_l2726_272606


namespace NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l2726_272640

theorem diophantine_equation_unique_solution :
  ∀ x y z t : ℤ, x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_unique_solution_l2726_272640


namespace NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l2726_272636

/-- Given two curves and their tangent lines, prove the range of parameter a -/
theorem tangent_lines_perpendicular_range (a : ℝ) : 
  ∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀ ≤ 3/2 ∧
  let f₁ (x : ℝ) := (a * x - 1) * Real.exp x
  let f₂ (x : ℝ) := (1 - x) * Real.exp (-x)
  let k₁ := (a * x₀ + a - 1) * Real.exp x₀
  let k₂ := (x₀ - 2) * Real.exp (-x₀)
  k₁ * k₂ = -1 →
  1 ≤ a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_perpendicular_range_l2726_272636


namespace NUMINAMATH_CALUDE_rectangle_dimension_solution_l2726_272634

theorem rectangle_dimension_solution :
  ∃! x : ℚ, (3 * x - 4 > 0) ∧ 
             (2 * x + 7 > 0) ∧ 
             ((3 * x - 4) * (2 * x + 7) = 18 * x - 10) ∧
             x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_solution_l2726_272634


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l2726_272635

theorem expand_polynomial_product : 
  ∀ x : ℝ, (3*x^2 - 2*x + 4) * (4*x^2 + 3*x - 6) = 12*x^4 + x^3 - 8*x^2 + 24*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l2726_272635


namespace NUMINAMATH_CALUDE_exactly_fifteen_numbers_l2726_272603

/-- Represents a three-digit positive integer in base 10 -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Converts a natural number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Checks if the two rightmost digits of two numbers are the same -/
def sameLastTwoDigits (a b : ℕ) : Prop :=
  a % 100 = b % 100

/-- The main theorem stating that there are exactly 15 numbers satisfying the condition -/
theorem exactly_fifteen_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 15 ∧
    ∀ n, n ∈ s ↔ 
      ThreeDigitInteger n ∧
      sameLastTwoDigits (toBase7 n * toBase8 n) (3 * n) :=
  sorry


end NUMINAMATH_CALUDE_exactly_fifteen_numbers_l2726_272603


namespace NUMINAMATH_CALUDE_square_and_cube_sum_l2726_272691

theorem square_and_cube_sum (p q : ℝ) (h1 : p * q = 8) (h2 : p + q = 7) :
  p^2 + q^2 = 33 ∧ p^3 + q^3 = 175 := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_sum_l2726_272691


namespace NUMINAMATH_CALUDE_optimal_tax_and_revenue_correct_l2726_272646

/-- Market model with linear supply and demand functions -/
structure MarketModel where
  -- Supply function coefficients
  supply_slope : ℝ
  supply_intercept : ℝ
  -- Demand function coefficient (slope)
  demand_slope : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax rate
  tax_rate : ℝ
  -- Consumer price after tax
  consumer_price : ℝ

/-- Calculate the optimal tax rate and maximum tax revenue -/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  -- Placeholder for the actual calculation
  (60, 8640)

/-- Theorem stating the optimal tax rate and maximum tax revenue -/
theorem optimal_tax_and_revenue_correct (model : MarketModel) :
  model.supply_slope = 6 ∧
  model.supply_intercept = -312 ∧
  model.demand_slope = -4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax_rate = 30 ∧
  model.consumer_price = 118 →
  optimal_tax_and_revenue model = (60, 8640) := by
  sorry

end NUMINAMATH_CALUDE_optimal_tax_and_revenue_correct_l2726_272646


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2726_272682

/-- The cost price of the bicycle for A -/
def cost_price_A : ℝ := sorry

/-- The selling price from A to B -/
def selling_price_B : ℝ := 1.20 * cost_price_A

/-- The selling price from B to C before tax -/
def selling_price_C : ℝ := 1.25 * selling_price_B

/-- The total cost for C including tax -/
def total_cost_C : ℝ := 1.15 * selling_price_C

/-- The selling price from C to D before discount -/
def selling_price_D1 : ℝ := 1.30 * total_cost_C

/-- The final selling price from C to D after discount -/
def selling_price_D2 : ℝ := 0.90 * selling_price_D1

/-- The final price D pays for the bicycle -/
def final_price_D : ℝ := 350

theorem bicycle_cost_price :
  cost_price_A = final_price_D / 2.01825 := by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2726_272682


namespace NUMINAMATH_CALUDE_bill_receives_26_l2726_272642

/-- Given a sum of money M to be divided among Allan, Bill, and Carol, prove that Bill receives $26 --/
theorem bill_receives_26 (M : ℚ) : 
  (∃ (allan_share bill_share carol_share : ℚ),
    -- Allan's share
    allan_share = 1 + (1/3) * (M - 1) ∧
    -- Bill's share
    bill_share = 6 + (1/3) * (M - allan_share - 6) ∧
    -- Carol's share
    carol_share = M - allan_share - bill_share ∧
    -- Carol receives $40
    carol_share = 40 ∧
    -- Bill's share is $26
    bill_share = 26) :=
by sorry

end NUMINAMATH_CALUDE_bill_receives_26_l2726_272642


namespace NUMINAMATH_CALUDE_lg_expression_equals_two_l2726_272664

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_two :
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_expression_equals_two_l2726_272664


namespace NUMINAMATH_CALUDE_max_initial_happy_citizens_l2726_272614

/-- Represents the state of happiness for a citizen --/
inductive MoodState
| Happy
| Unhappy

/-- Represents a citizen in Happy City --/
structure Citizen where
  id : Nat
  mood : MoodState

/-- Represents the state of Happy City --/
structure HappyCity where
  citizens : List Citizen
  day : Nat

/-- Function to simulate a day of smiling in Happy City --/
def smileDay (city : HappyCity) : HappyCity :=
  sorry

/-- Function to count happy citizens --/
def countHappy (city : HappyCity) : Nat :=
  sorry

/-- Theorem stating the maximum initial number of happy citizens --/
theorem max_initial_happy_citizens :
  ∀ (initialCity : HappyCity),
    initialCity.citizens.length = 2014 →
    (∃ (finalCity : HappyCity),
      finalCity = (smileDay ∘ smileDay ∘ smileDay ∘ smileDay) initialCity ∧
      countHappy finalCity = 2000) →
    countHappy initialCity ≤ 32 :=
  sorry

end NUMINAMATH_CALUDE_max_initial_happy_citizens_l2726_272614


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l2726_272631

theorem imaginary_part_of_z_squared (z : ℂ) (h : z * (1 - Complex.I) = 2) : 
  Complex.im (z^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l2726_272631


namespace NUMINAMATH_CALUDE_caterer_order_l2726_272671

theorem caterer_order (ice_cream_price sundae_price total_price : ℚ) 
  (h1 : ice_cream_price = 0.60)
  (h2 : sundae_price = 1.20)
  (h3 : total_price = 225.00)
  (h4 : ice_cream_price * x + sundae_price * x = total_price) :
  x = 125 :=
by
  sorry

#check caterer_order

end NUMINAMATH_CALUDE_caterer_order_l2726_272671


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2726_272626

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three :
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2726_272626


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2726_272654

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) : 
  Real.cos (a 2 * a 12) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2726_272654


namespace NUMINAMATH_CALUDE_min_difference_is_one_l2726_272683

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  xz : ℕ
  yz : ℕ
  xy : ℕ

/-- Checks if the given side lengths form a valid triangle --/
def isValidTriangle (t : TriangleSides) : Prop :=
  t.xz + t.yz > t.xy ∧ t.xz + t.xy > t.yz ∧ t.yz + t.xy > t.xz

/-- Checks if the given side lengths satisfy the problem conditions --/
def satisfiesConditions (t : TriangleSides) : Prop :=
  t.xz + t.yz + t.xy = 3001 ∧ t.xz < t.yz ∧ t.yz ≤ t.xy

theorem min_difference_is_one :
  ∀ t : TriangleSides,
    isValidTriangle t →
    satisfiesConditions t →
    ∀ u : TriangleSides,
      isValidTriangle u →
      satisfiesConditions u →
      t.yz - t.xz ≤ u.yz - u.xz →
      t.yz - t.xz = 1 :=
sorry

end NUMINAMATH_CALUDE_min_difference_is_one_l2726_272683


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2726_272638

/-- For a geometric sequence with common ratio 2, the ratio of the sum of the first 3 terms to the first term is 7 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 3 / a 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l2726_272638


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2726_272659

theorem pure_imaginary_complex_fraction (a : ℝ) : 
  (Complex.I * (((a + 3 * Complex.I) / (1 - 2 * Complex.I)).im) = ((a + 3 * Complex.I) / (1 - 2 * Complex.I))) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_fraction_l2726_272659


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2726_272656

theorem marble_fraction_after_tripling (total : ℝ) (h_total_pos : total > 0) :
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3/5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l2726_272656


namespace NUMINAMATH_CALUDE_sum_of_456_l2726_272621

/-- A geometric sequence with first term 3 and sum of first three terms 9 -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) * a n = (a (n + 1))^2
  first_term : a 1 = 3
  sum_first_three : a 1 + a 2 + a 3 = 9

/-- The sum of the 4th, 5th, and 6th terms is either 9 or -72 -/
theorem sum_of_456 (seq : GeometricSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 9 ∨ seq.a 4 + seq.a 5 + seq.a 6 = -72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_456_l2726_272621


namespace NUMINAMATH_CALUDE_students_with_a_l2726_272627

theorem students_with_a (total_students : ℕ) (ratio : ℚ) 
  (h1 : total_students = 30) 
  (h2 : ratio = 2 / 3) : 
  ∃ (a_students : ℕ) (percentage : ℚ), 
    a_students = 20 ∧ 
    percentage = 200 / 3 ∧
    (a_students : ℚ) / total_students = ratio ∧
    percentage = (a_students : ℚ) / total_students * 100 := by
  sorry

end NUMINAMATH_CALUDE_students_with_a_l2726_272627


namespace NUMINAMATH_CALUDE_simplify_expression_l2726_272623

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2726_272623


namespace NUMINAMATH_CALUDE_amalia_reading_time_l2726_272680

/-- Represents the time in minutes it takes Amalia to read a given number of pages -/
def reading_time (pages : ℕ) : ℚ :=
  (pages : ℚ) * 2 / 4

/-- Theorem stating that it takes Amalia 9 minutes to read 18 pages -/
theorem amalia_reading_time :
  reading_time 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_amalia_reading_time_l2726_272680


namespace NUMINAMATH_CALUDE_sock_pairs_same_color_l2726_272685

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem sock_pairs_same_color (red green yellow : ℕ) 
  (h_red : red = 5) (h_green : green = 6) (h_yellow : yellow = 4) :
  choose red 2 + choose green 2 + choose yellow 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_same_color_l2726_272685


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2726_272644

/-- Triangle ABC with given side lengths and angle -/
structure TriangleABC where
  AB : ℝ
  BC : ℝ
  cosC : ℝ
  h_AB : AB = Real.sqrt 2
  h_BC : BC = 1
  h_cosC : cosC = 3/4

/-- The main theorem about TriangleABC -/
theorem triangle_ABC_properties (t : TriangleABC) :
  let sinA := Real.sqrt (14) / 8
  let dot_product := -(3/2 : ℝ)
  (∃ (CA : ℝ), sinA = Real.sqrt (1 - t.cosC^2) * t.BC / t.AB) ∧
  (∃ (CA : ℝ), dot_product = t.BC * CA * (-t.cosC)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2726_272644


namespace NUMINAMATH_CALUDE_circle_radii_l2726_272676

theorem circle_radii (r R : ℝ) (hr : r > 0) (hR : R > 0) : 
  ∃ (circumscribed_radius inscribed_radius : ℝ),
    circumscribed_radius = Real.sqrt (r * R) ∧
    inscribed_radius = (Real.sqrt (r * R) * (Real.sqrt R + Real.sqrt r - Real.sqrt (R + r))) / Real.sqrt (R + r) := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_l2726_272676


namespace NUMINAMATH_CALUDE_missing_sale_is_7225_l2726_272647

/-- Calculates the missing month's sale given the sales of other months and the target average --/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 target_average : ℕ) : ℕ :=
  6 * target_average - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Proves that the missing month's sale is 7225 given the problem conditions --/
theorem missing_sale_is_7225 :
  let sale1 : ℕ := 6235
  let sale2 : ℕ := 6927
  let sale3 : ℕ := 6855
  let sale5 : ℕ := 6562
  let sale6 : ℕ := 5191
  let target_average : ℕ := 6500
  calculate_missing_sale sale1 sale2 sale3 sale5 sale6 target_average = 7225 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_sale_is_7225_l2726_272647


namespace NUMINAMATH_CALUDE_percentage_of_boy_scouts_l2726_272615

theorem percentage_of_boy_scouts (total_scouts : ℝ) (boy_scouts : ℝ) (girl_scouts : ℝ)
  (h1 : boy_scouts + girl_scouts = total_scouts)
  (h2 : 0.60 * total_scouts = 0.50 * boy_scouts + 0.6818 * girl_scouts)
  : boy_scouts / total_scouts = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boy_scouts_l2726_272615


namespace NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l2726_272624

/-- Given two pipes that can fill a tank in 10 and 20 hours respectively,
    prove that when both are opened simultaneously, the tank fills in 20/3 hours. -/
theorem simultaneous_pipe_filling_time :
  ∀ (tank_capacity : ℝ) (pipe_a_rate pipe_b_rate : ℝ),
    pipe_a_rate = tank_capacity / 10 →
    pipe_b_rate = tank_capacity / 20 →
    tank_capacity / (pipe_a_rate + pipe_b_rate) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_pipe_filling_time_l2726_272624
