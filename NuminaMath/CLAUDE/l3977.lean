import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l3977_397701

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x / (x - 2) - x / (x + 2)) / (4 * x / (x - 2)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3977_397701


namespace NUMINAMATH_CALUDE_home_electronics_budget_allocation_l3977_397731

theorem home_electronics_budget_allocation 
  (total_budget : ℝ)
  (microphotonics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_budget = 100)
  (h2 : microphotonics = 14)
  (h3 : food_additives = 20)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 18)
  (h7 : (basic_astrophysics_degrees / 360) * 100 + microphotonics + food_additives + genetically_modified_microorganisms + industrial_lubricants + home_electronics = total_budget) :
  home_electronics = 24 := by
  sorry

end NUMINAMATH_CALUDE_home_electronics_budget_allocation_l3977_397731


namespace NUMINAMATH_CALUDE_log_2_irrational_l3977_397708

theorem log_2_irrational : Irrational (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_2_irrational_l3977_397708


namespace NUMINAMATH_CALUDE_license_plate_difference_l3977_397784

def california_plates := 26^3 * 10^4
def texas_plates := 26^3 * 10^3

theorem license_plate_difference :
  california_plates - texas_plates = 4553200000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3977_397784


namespace NUMINAMATH_CALUDE_sum_a_b_range_m_solve_equation_l3977_397764

-- Define the system of equations
def system (a b m : ℝ) : Prop :=
  a + 2*b = 4 ∧ 2*a + b = 3 - m

-- Theorem 1: Express a + b in terms of m
theorem sum_a_b (a b m : ℝ) :
  system a b m → a + b = (7 - m) / 3 := by sorry

-- Theorem 2: Find the range of values for m
theorem range_m (a b m : ℝ) :
  system a b m → a - b > -4 → m < 3 := by sorry

-- Theorem 3: Solve the equation for positive integer m
theorem solve_equation (m : ℕ) (x : ℝ) :
  m < 3 → (m * x - (1 - x) / 2 = 5 ↔ x = 11/3 ∨ x = 2.2) := by sorry

end NUMINAMATH_CALUDE_sum_a_b_range_m_solve_equation_l3977_397764


namespace NUMINAMATH_CALUDE_no_infinite_set_with_perfect_square_property_l3977_397710

theorem no_infinite_set_with_perfect_square_property : 
  ¬ ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → ∃ k : ℕ, a * b * c + 1 = k * k) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_set_with_perfect_square_property_l3977_397710


namespace NUMINAMATH_CALUDE_min_value_implies_m_l3977_397797

/-- The function f(x) = -x^3 + 6x^2 + m -/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 + m

/-- Theorem: If f(x) has a minimum value of 23, then m = 23 -/
theorem min_value_implies_m (m : ℝ) : 
  (∃ (y : ℝ), ∀ (x : ℝ), f x m ≥ y ∧ ∃ (x₀ : ℝ), f x₀ m = y) ∧ 
  (∃ (x₀ : ℝ), f x₀ m = 23) → 
  m = 23 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_m_l3977_397797


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3977_397780

/-- Given:
A lends B Rs. 3500
B lends C Rs. 3500 at 11.5% per annum
B's gain over 3 years is Rs. 157.5
Prove: The interest rate at which A lent to B is 10% per annum
-/
theorem interest_rate_calculation (principal : ℝ) (rate_b_to_c : ℝ) (time : ℝ) (gain : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_b_to_c = 11.5)
  (h3 : time = 3)
  (h4 : gain = 157.5)
  (h5 : gain = principal * rate_b_to_c / 100 * time - principal * rate_a_to_b / 100 * time) :
  rate_a_to_b = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3977_397780


namespace NUMINAMATH_CALUDE_permutation_inequality_l3977_397793

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
    (2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end NUMINAMATH_CALUDE_permutation_inequality_l3977_397793


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3977_397792

theorem quadratic_equations_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = 1 + Real.sqrt 15 / 3 ∨ x = 1 - Real.sqrt 15 / 3) ∧
  (∀ x : ℝ, x^2 - 2 - 3 * x = 0 ↔ x = (3 + Real.sqrt 17) / 2 ∨ x = (3 - Real.sqrt 17) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3977_397792


namespace NUMINAMATH_CALUDE_factorization_equality_l3977_397744

theorem factorization_equality (a b : ℝ) : a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3977_397744


namespace NUMINAMATH_CALUDE_polynomial_roots_l3977_397760

theorem polynomial_roots : 
  let f : ℂ → ℂ := λ x => 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3
  let r1 := (-1 + Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r2 := (-1 - Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r3 := (-1 + Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  let r4 := (-1 - Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  (f r1 = 0) ∧ (f r2 = 0) ∧ (f r3 = 0) ∧ (f r4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3977_397760


namespace NUMINAMATH_CALUDE_number_1991_in_32nd_group_l3977_397789

/-- The function that gives the number of elements in the nth group of odd numbers -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- The function that gives the sum of elements in the first n groups -/
def sum_of_first_n_groups (n : ℕ) : ℕ := n^2

/-- The theorem stating that 1991 appears in the 32nd group -/
theorem number_1991_in_32nd_group :
  (∀ k < 32, sum_of_first_n_groups k < 1991) ∧
  sum_of_first_n_groups 32 ≥ 1991 := by
  sorry

end NUMINAMATH_CALUDE_number_1991_in_32nd_group_l3977_397789


namespace NUMINAMATH_CALUDE_parabola_directrix_l3977_397748

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the parabola g(x) = 2px^2
def g (p x : ℝ) : ℝ := 2 * p * x^2

-- Define the tangent line to f(x) at x = -1
def tangent_line (x : ℝ) : ℝ := 2 * x + 4

-- Theorem statement
theorem parabola_directrix :
  ∃ (p : ℝ),
    (∀ x, tangent_line x = g p x) ∧
    (∃ x, f x = tangent_line x ∧ x = -1) →
    (∀ y, y = 1 ↔ ∃ x, x^2 = -4 * y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3977_397748


namespace NUMINAMATH_CALUDE_race_course_length_is_correct_l3977_397711

/-- The length of a race course where two runners finish at the same time -/
def race_course_length (v_B : ℝ) : ℝ :=
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  100

theorem race_course_length_is_correct (v_B : ℝ) (h : v_B > 0) :
  let v_A : ℝ := 4 * v_B
  let head_start : ℝ := 75
  let L : ℝ := race_course_length v_B
  L / v_A = (L - head_start) / v_B :=
by
  sorry

#check race_course_length_is_correct

end NUMINAMATH_CALUDE_race_course_length_is_correct_l3977_397711


namespace NUMINAMATH_CALUDE_test_scores_l3977_397772

theorem test_scores (scores : List Nat) : 
  (scores.length > 0) →
  (scores.Pairwise (·≠·)) →
  (scores.sum = 119) →
  (scores.take 3).sum = 23 →
  (scores.reverse.take 3).sum = 49 →
  (scores.length = 10 ∧ scores.maximum = some 18) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_l3977_397772


namespace NUMINAMATH_CALUDE_banana_bunches_l3977_397799

theorem banana_bunches (total_bananas : ℕ) (bunches_of_seven : ℕ) (bananas_per_bunch_of_seven : ℕ) (bananas_per_bunch_of_eight : ℕ) :
  total_bananas = 83 →
  bunches_of_seven = 5 →
  bananas_per_bunch_of_seven = 7 →
  bananas_per_bunch_of_eight = 8 →
  ∃ (bunches_of_eight : ℕ), 
    total_bananas = bunches_of_seven * bananas_per_bunch_of_seven + bunches_of_eight * bananas_per_bunch_of_eight ∧
    bunches_of_eight = 6 :=
by sorry

end NUMINAMATH_CALUDE_banana_bunches_l3977_397799


namespace NUMINAMATH_CALUDE_largest_prime_to_test_primality_l3977_397727

theorem largest_prime_to_test_primality (n : ℕ) : 
  900 ≤ n ∧ n ≤ 950 → 
  (∀ (p : ℕ), p.Prime → p ≤ 29 → (p ∣ n ↔ ¬n.Prime)) ∧
  (∀ (p : ℕ), p.Prime → p > 29 → (p ∣ n → ¬n.Prime)) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_primality_l3977_397727


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l3977_397714

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the product of the slopes of its two asymptotes is -2
    and its focal length is 6, then a² = 3 and b² = 6 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 / a^2 = 2) →  -- product of slopes of asymptotes is -2
  (6^2 = 4 * (a^2 + b^2)) →  -- focal length is 6
  a^2 = 3 ∧ b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l3977_397714


namespace NUMINAMATH_CALUDE_min_value_range_l3977_397733

def f (x : ℝ) := x^2 - 6*x + 8

theorem min_value_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) →
  a ∈ Set.Ioo 1 3 ∪ {3} :=
by sorry

end NUMINAMATH_CALUDE_min_value_range_l3977_397733


namespace NUMINAMATH_CALUDE_x_greater_than_y_l3977_397766

theorem x_greater_than_y (a b x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : x = a + 1/a) 
  (h4 : y = b + 1/b) : 
  x > y := by
sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l3977_397766


namespace NUMINAMATH_CALUDE_forgotten_angle_measure_l3977_397702

theorem forgotten_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 2070 → 
  (n - 2) * 180 - sum_without_one = 90 := by
  sorry

end NUMINAMATH_CALUDE_forgotten_angle_measure_l3977_397702


namespace NUMINAMATH_CALUDE_percent_relation_l3977_397713

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 5 * b = (5/2) * a := by sorry

end NUMINAMATH_CALUDE_percent_relation_l3977_397713


namespace NUMINAMATH_CALUDE_courtyard_paving_l3977_397718

-- Define the courtyard dimensions in meters
def courtyard_length : ℝ := 42
def courtyard_width : ℝ := 22

-- Define the brick dimensions in centimeters
def brick_length : ℝ := 16
def brick_width : ℝ := 10

-- Define the conversion factor from square meters to square centimeters
def sq_m_to_sq_cm : ℝ := 10000

-- Theorem statement
theorem courtyard_paving (courtyard_length courtyard_width brick_length brick_width sq_m_to_sq_cm : ℝ) :
  courtyard_length = 42 →
  courtyard_width = 22 →
  brick_length = 16 →
  brick_width = 10 →
  sq_m_to_sq_cm = 10000 →
  (courtyard_length * courtyard_width * sq_m_to_sq_cm) / (brick_length * brick_width) = 57750 :=
by
  sorry


end NUMINAMATH_CALUDE_courtyard_paving_l3977_397718


namespace NUMINAMATH_CALUDE_evans_class_enrollment_l3977_397796

theorem evans_class_enrollment (q1 q2 both not_taken : ℕ) 
  (h1 : q1 = 19)
  (h2 : q2 = 24)
  (h3 : both = 19)
  (h4 : not_taken = 5) :
  q1 + q2 - both + not_taken = 29 := by
sorry

end NUMINAMATH_CALUDE_evans_class_enrollment_l3977_397796


namespace NUMINAMATH_CALUDE_nes_sale_price_l3977_397798

/-- The sale price of the NES given the trade-in value of SNES, additional payment, change, and game value. -/
theorem nes_sale_price
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_payment : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_payment = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_payment - change - game_value = 160 :=
sorry


end NUMINAMATH_CALUDE_nes_sale_price_l3977_397798


namespace NUMINAMATH_CALUDE_ellipse_properties_l3977_397712

/-- An ellipse with center at the origin, foci on the x-axis, left focus at (-2,0), and passing through (2,√2) -/
structure Ellipse :=
  (equation : ℝ → ℝ → Prop)
  (center_origin : equation 0 0)
  (foci_on_x_axis : ∀ y, y ≠ 0 → ¬ equation (-2) y ∧ ¬ equation 2 y)
  (left_focus : equation (-2) 0)
  (passes_through : equation 2 (Real.sqrt 2))

/-- The intersection points of a line y=kx with the ellipse -/
def intersect (C : Ellipse) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C.equation p.1 p.2 ∧ p.2 = k * p.1}

/-- The y-intercepts of lines from A to intersection points -/
def y_intercepts (C : Ellipse) (k : ℝ) : Set ℝ :=
  {y | ∃ p ∈ intersect C k, y = (p.2 / (p.1 + 2*Real.sqrt 2)) * (2*Real.sqrt 2)}

/-- The theorem to be proved -/
theorem ellipse_properties (C : Ellipse) :
  (∀ x y, C.equation x y ↔ x^2/8 + y^2/4 = 1) ∧
  (∀ k ≠ 0, ∀ y ∈ y_intercepts C k,
    (0^2 + y^2 + 2*Real.sqrt 2/k*y = 4) ∧
    (2^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4) ∧
    ((-2)^2 + 0^2 + 2*Real.sqrt 2/k*0 = 4)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3977_397712


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l3977_397778

def DigitSet : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

theorem digit_sum_puzzle (a b c x z : Nat) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ a ≠ z ∧
                b ≠ c ∧ b ≠ x ∧ b ≠ z ∧
                c ≠ x ∧ c ≠ z ∧
                x ≠ z)
  (h_in_set : a ∈ DigitSet ∧ b ∈ DigitSet ∧ c ∈ DigitSet ∧ x ∈ DigitSet ∧ z ∈ DigitSet)
  (h_vertical_sum : a + b + c = 17)
  (h_horizontal_sum : x + b + z = 14) :
  a + b + c + x + z = 26 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l3977_397778


namespace NUMINAMATH_CALUDE_disjoint_triangles_probability_disjoint_triangles_probability_proof_l3977_397742

/-- The probability that two triangles formed by six points chosen sequentially at random on a circle's circumference are disjoint -/
theorem disjoint_triangles_probability : ℚ :=
  3/10

/-- Total number of distinct arrangements with one point fixed -/
def total_arrangements : ℕ := 120

/-- Number of favorable outcomes where the triangles are disjoint -/
def favorable_outcomes : ℕ := 36

theorem disjoint_triangles_probability_proof :
  disjoint_triangles_probability = (favorable_outcomes : ℚ) / total_arrangements :=
by sorry

end NUMINAMATH_CALUDE_disjoint_triangles_probability_disjoint_triangles_probability_proof_l3977_397742


namespace NUMINAMATH_CALUDE_mike_picked_seven_apples_l3977_397771

/-- The number of apples picked by Mike, given the total number of apples and the number picked by Nancy and Keith. -/
def mike_apples (total : ℕ) (nancy : ℕ) (keith : ℕ) : ℕ :=
  total - (nancy + keith)

/-- Theorem stating that Mike picked 7 apples given the problem conditions. -/
theorem mike_picked_seven_apples :
  mike_apples 16 3 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_seven_apples_l3977_397771


namespace NUMINAMATH_CALUDE_square_configuration_angle_l3977_397781

/-- Theorem: In a configuration of three squares attached by their vertices to each other and to two vertical rods,
    where the sum of the white angles equals the sum of the gray angles, and given angles of 30°, 126°, 75°,
    and three 90° angles, the measure of the remaining angle x is 39°. -/
theorem square_configuration_angle (white_angles gray_angles : List ℝ)
  (h1 : white_angles.sum = gray_angles.sum)
  (h2 : white_angles.length = 4)
  (h3 : gray_angles.length = 3)
  (h4 : white_angles.take 3 = [30, 126, 75])
  (h5 : gray_angles = [90, 90, 90]) :
  white_angles[3] = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_configuration_angle_l3977_397781


namespace NUMINAMATH_CALUDE_catering_cost_comparison_l3977_397730

def cost_caterer1 (x : ℕ) : ℚ := 150 + 18 * x
def cost_caterer2 (x : ℕ) : ℚ := 250 + 15 * x

theorem catering_cost_comparison :
  (∀ x : ℕ, x < 34 → cost_caterer1 x ≤ cost_caterer2 x) ∧
  (∀ x : ℕ, x ≥ 34 → cost_caterer1 x > cost_caterer2 x) :=
by sorry

end NUMINAMATH_CALUDE_catering_cost_comparison_l3977_397730


namespace NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l3977_397705

-- Define the cost function C(n)
def C (n : ℕ) : ℕ :=
  if n ≤ 30 then 15 * n
  else if n ≤ 60 then 13 * n
  else 12 * n

-- Define a function that checks if buying n+1 books is cheaper than n books
def isCheaperToBuyMore (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem cheaper_to_buy_more_count :
  (∃ (S : Finset ℕ), S.card = 5 ∧ (∀ n, n ∈ S ↔ isCheaperToBuyMore n)) :=
by sorry

end NUMINAMATH_CALUDE_cheaper_to_buy_more_count_l3977_397705


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_bounds_l3977_397767

/-- Regular tetrahedron with side length and height 1 -/
structure Tetrahedron where
  side_length : ℝ
  height : ℝ
  is_regular : side_length = 1 ∧ height = 1

/-- Rectangular parallelepiped inscribed in the tetrahedron -/
structure Parallelepiped (t : Tetrahedron) where
  base_area : ℝ
  base_in_tetrahedron_base : Prop
  opposite_vertex_on_lateral_surface : Prop
  diagonal : ℝ

/-- Theorem stating the bounds of the parallelepiped's diagonal -/
theorem parallelepiped_diagonal_bounds (t : Tetrahedron) (p : Parallelepiped t) :
  (0 < p.base_area ∧ p.base_area ≤ 1/18 →
    Real.sqrt (2/3 - 2*p.base_area) ≤ p.diagonal ∧ p.diagonal < Real.sqrt (2 - 2*p.base_area)) ∧
  ((7 + 2*Real.sqrt 6)/25 ≤ p.base_area ∧ p.base_area < 1/2 →
    Real.sqrt (1 - 2*Real.sqrt (2*p.base_area) + 4*p.base_area) ≤ p.diagonal ∧
    p.diagonal < Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) ∧
  (1/2 ≤ p.base_area ∧ p.base_area < 1 →
    Real.sqrt (2*p.base_area) < p.diagonal ∧
    p.diagonal ≤ Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_bounds_l3977_397767


namespace NUMINAMATH_CALUDE_correct_calculation_l3977_397749

theorem correct_calculation (original : ℤ) (incorrect_result : ℤ) 
  (incorrect_addition : ℤ) (correct_addition : ℤ) : 
  incorrect_result = original + incorrect_addition →
  original + correct_addition = 97 :=
by
  intro h
  sorry

#check correct_calculation 35 61 26 62

end NUMINAMATH_CALUDE_correct_calculation_l3977_397749


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3977_397783

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3977_397783


namespace NUMINAMATH_CALUDE_mother_daughter_age_ratio_l3977_397739

/-- Given a mother who is 27 years older than her daughter and is currently 55 years old,
    prove that the ratio of their ages one year ago was 2:1. -/
theorem mother_daughter_age_ratio : 
  ∀ (mother_age daughter_age : ℕ),
  mother_age = 55 →
  mother_age = daughter_age + 27 →
  (mother_age - 1) / (daughter_age - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_mother_daughter_age_ratio_l3977_397739


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3977_397794

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((64 - 2*x) ^ (1/4) : ℝ) + ((48 + 2*x) ^ (1/4) : ℝ) = 6) ↔ (x = 32 ∨ x = -8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3977_397794


namespace NUMINAMATH_CALUDE_total_flowers_in_gardens_l3977_397715

/-- Given 10 gardens, each with 544 pots, and each pot containing 32 flowers,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_in_gardens_l3977_397715


namespace NUMINAMATH_CALUDE_x_approximates_one_l3977_397765

/-- The polynomial function P(x) = x^4 - 4x^3 + 4x^2 + 4 -/
def P (x : ℝ) : ℝ := x^4 - 4*x^3 + 4*x^2 + 4

/-- A small positive real number representing the tolerance for approximation -/
def ε : ℝ := 0.000000000000001

theorem x_approximates_one :
  ∃ x : ℝ, abs (P x - 4.999999999999999) < ε ∧ abs (x - 1) < ε :=
sorry

end NUMINAMATH_CALUDE_x_approximates_one_l3977_397765


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3977_397746

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 1 → 
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) →
  A = 1/16 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3977_397746


namespace NUMINAMATH_CALUDE_thabo_book_ratio_l3977_397788

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 200 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 35

/-- The ratio of paperback fiction to paperback nonfiction books is 2:1 -/
def hasRatioTwoToOne (books : BookCollection) : Prop :=
  2 * books.paperbackNonfiction = books.paperbackFiction

theorem thabo_book_ratio (books : BookCollection) :
  validCollection books → hasRatioTwoToOne books := by
  sorry

end NUMINAMATH_CALUDE_thabo_book_ratio_l3977_397788


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3977_397761

theorem complex_equation_solution (z : ℂ) : (1 - 3*I)*z = 2 + 4*I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3977_397761


namespace NUMINAMATH_CALUDE_angle_tangent_relation_l3977_397709

theorem angle_tangent_relation (θ : Real) :
  (-(π / 2) < θ ∧ θ < 0) →  -- θ is in the fourth quadrant
  (Real.sin (θ + π / 4) = 3 / 5) →
  (Real.tan (θ - π / 4) = -4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_angle_tangent_relation_l3977_397709


namespace NUMINAMATH_CALUDE_det_B_equals_two_l3977_397729

open Matrix

theorem det_B_equals_two (x y : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  (B + 2 * B⁻¹ = 0) → det B = 2 := by
  sorry

end NUMINAMATH_CALUDE_det_B_equals_two_l3977_397729


namespace NUMINAMATH_CALUDE_tan_two_pi_thirds_l3977_397775

theorem tan_two_pi_thirds : Real.tan (2 * Real.pi / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_pi_thirds_l3977_397775


namespace NUMINAMATH_CALUDE_thermodynamic_cycle_efficiency_l3977_397751

/-- Represents a thermodynamic cycle with three stages -/
structure ThermodynamicCycle where
  P₀ : ℝ
  ρ₀ : ℝ
  stage1_isochoric : ℝ → ℝ → Prop
  stage2_isobaric : ℝ → ℝ → Prop
  stage3_return : ℝ → ℝ → Prop

/-- Efficiency of a thermodynamic cycle -/
def efficiency (cycle : ThermodynamicCycle) : ℝ := sorry

/-- Maximum possible efficiency for given temperature range -/
def max_efficiency (T_min T_max : ℝ) : ℝ := sorry

/-- Theorem stating the efficiency of the described thermodynamic cycle -/
theorem thermodynamic_cycle_efficiency (cycle : ThermodynamicCycle) 
  (h1 : cycle.stage1_isochoric (3 * cycle.P₀) cycle.P₀)
  (h2 : cycle.stage2_isobaric cycle.ρ₀ (3 * cycle.ρ₀))
  (h3 : cycle.stage3_return 1 1)
  (h4 : ∃ T_min T_max, efficiency cycle = (1 / 8) * max_efficiency T_min T_max) :
  efficiency cycle = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_thermodynamic_cycle_efficiency_l3977_397751


namespace NUMINAMATH_CALUDE_three_heads_probability_l3977_397770

theorem three_heads_probability (p : ℝ) (h_fair : p = 1 / 2) :
  p * p * p = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_probability_l3977_397770


namespace NUMINAMATH_CALUDE_strawberry_weight_calculation_l3977_397717

def total_fruit_weight : ℕ := 10
def apple_weight : ℕ := 3
def orange_weight : ℕ := 1
def grape_weight : ℕ := 3

theorem strawberry_weight_calculation :
  total_fruit_weight - (apple_weight + orange_weight + grape_weight) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_weight_calculation_l3977_397717


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3977_397741

/-- Ellipse C with foci at (-2,0) and (2,0), passing through (0, √5) -/
def ellipse_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ x^2/9 + y^2/5 = 1}

/-- Line l passing through (-2,0) with slope 1 -/
def line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ y = x + 2}

/-- Theorem stating the standard equation of ellipse C and the length of PQ -/
theorem ellipse_and_line_intersection :
  (∀ (x y : ℝ), (x, y) ∈ ellipse_C ↔ x^2/9 + y^2/5 = 1) ∧
  (∃ (P Q : ℝ × ℝ), P ∈ ellipse_C ∧ Q ∈ ellipse_C ∧ P ∈ line_l ∧ Q ∈ line_l ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 30/7) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3977_397741


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3977_397759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5 * Real.sqrt 3)
  (h_mean2 : (a 3 + a 7) / 2 = 7 * Real.sqrt 3) :
  a 4 = 5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l3977_397759


namespace NUMINAMATH_CALUDE_digit_101_of_7_26_l3977_397752

def decimal_expansion (n d : ℕ) : ℕ → ℕ
  | 0 => (10 * n / d) % 10
  | k + 1 => decimal_expansion (10 * (n % d)) d k

theorem digit_101_of_7_26 : decimal_expansion 7 26 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_101_of_7_26_l3977_397752


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3977_397736

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 8 * x + 5 ∧ x = -17 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3977_397736


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l3977_397722

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (51.25 : ℚ) := by
  sorry

#eval (35 : ℚ) * 40 + 45 * 60
#eval (35 + 45 : ℚ) * (51.25 : ℚ)

end NUMINAMATH_CALUDE_average_marks_combined_classes_l3977_397722


namespace NUMINAMATH_CALUDE_num_factors_of_given_number_l3977_397745

/-- The number of distinct natural-number factors of 8^2 * 9^3 * 7^5 -/
def num_factors : ℕ :=
  (7 : ℕ) * (7 : ℕ) * (6 : ℕ)

/-- The given number 8^2 * 9^3 * 7^5 -/
def given_number : ℕ :=
  (8^2 : ℕ) * (9^3 : ℕ) * (7^5 : ℕ)

theorem num_factors_of_given_number :
  (Finset.filter (fun d => given_number % d = 0) (Finset.range (given_number + 1))).card = num_factors :=
by sorry

end NUMINAMATH_CALUDE_num_factors_of_given_number_l3977_397745


namespace NUMINAMATH_CALUDE_madam_arrangements_count_l3977_397734

/-- The number of unique arrangements of the letters in the word MADAM -/
def madam_arrangements : ℕ := 30

/-- The total number of letters in the word MADAM -/
def total_letters : ℕ := 5

/-- The number of times the letter M appears in MADAM -/
def m_count : ℕ := 2

/-- The number of times the letter A appears in MADAM -/
def a_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in MADAM is 30 -/
theorem madam_arrangements_count :
  madam_arrangements = Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial a_count) :=
by sorry

end NUMINAMATH_CALUDE_madam_arrangements_count_l3977_397734


namespace NUMINAMATH_CALUDE_circle_through_points_l3977_397735

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x - 2*y + 12 = 0

-- Define the points
def P : ℝ × ℝ := (2, 2)
def M : ℝ × ℝ := (5, 3)
def N : ℝ × ℝ := (3, -1)

-- Theorem statement
theorem circle_through_points :
  circle_equation P.1 P.2 ∧ circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 :=
sorry

end NUMINAMATH_CALUDE_circle_through_points_l3977_397735


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_pair_l3977_397726

/-- The number of ways to split 2n people into n pairs --/
def total_pairings (n : ℕ) : ℕ := (2 * n).factorial / (2^n * n.factorial)

/-- The number of ways to pair n boys with n girls --/
def boy_girl_pairings (n : ℕ) : ℕ := n.factorial

theorem probability_at_least_one_girl_pair (n : ℕ) (hn : n = 5) :
  (total_pairings n - boy_girl_pairings n) / total_pairings n = 23640 / 23760 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_pair_l3977_397726


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3977_397737

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 210) (h2 : b = 4620) :
  (Nat.gcd a b) * (3 * Nat.lcm a b) = 2910600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3977_397737


namespace NUMINAMATH_CALUDE_sum_equals_fraction_l3977_397754

/-- Given a real number k > 2 such that the infinite sum of (6n-2)/k^n from n=1 to infinity
    equals 31/9, prove that k = 147/62. -/
theorem sum_equals_fraction (k : ℝ) 
  (h1 : k > 2)
  (h2 : ∑' n, (6 * n - 2) / k^n = 31/9) : 
  k = 147/62 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fraction_l3977_397754


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l3977_397773

/-- Represents the juice extraction rate for a fruit -/
structure JuiceRate where
  fruit : String
  ounces : ℚ
  count : ℕ

/-- Calculates the percentage of one juice in a blend of two juices with equal volumes -/
def juicePercentage (rate1 rate2 : JuiceRate) : ℚ :=
  100 * (rate1.ounces * rate2.count) / (rate1.ounces * rate2.count + rate2.ounces * rate1.count)

theorem pear_juice_percentage (pearRate orangeRate : JuiceRate) 
  (h1 : pearRate.fruit = "pear")
  (h2 : orangeRate.fruit = "orange")
  (h3 : pearRate.ounces = 9)
  (h4 : pearRate.count = 4)
  (h5 : orangeRate.ounces = 10)
  (h6 : orangeRate.count = 3) :
  juicePercentage pearRate orangeRate = 50 := by
  sorry

#eval juicePercentage 
  { fruit := "pear", ounces := 9, count := 4 }
  { fruit := "orange", ounces := 10, count := 3 }

end NUMINAMATH_CALUDE_pear_juice_percentage_l3977_397773


namespace NUMINAMATH_CALUDE_age_problem_l3977_397774

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- A is two years older than B
  b = 2 * c →  -- B is twice as old as C
  a + b + c = 37 →  -- Total age is 37
  b = 14 :=  -- B's age is 14
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l3977_397774


namespace NUMINAMATH_CALUDE_jeans_to_janes_money_ratio_l3977_397782

theorem jeans_to_janes_money_ratio (total : ℕ) (jeans_money : ℕ) :
  total = 76 →
  jeans_money = 57 →
  (jeans_money : ℚ) / (total - jeans_money : ℚ) = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_jeans_to_janes_money_ratio_l3977_397782


namespace NUMINAMATH_CALUDE_bernardo_wins_l3977_397707

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 75 < 1000 ∧
  4 * N + 150 < 1000 ∧
  4 * N + 225 < 1000 ∧
  8 * N + 450 < 1000 ∧
  8 * N + 525 < 1000 ∧
  16 * N + 1050 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  (∃ N : ℕ, game_winner N ∧ 
    (∀ M : ℕ, M < N → ¬game_winner M) ∧
    N = 56 ∧
    sum_of_digits N = 11) :=
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l3977_397707


namespace NUMINAMATH_CALUDE_pizzeria_sales_l3977_397769

theorem pizzeria_sales (small_price large_price total_revenue small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_revenue = 40)
  (h4 : small_count = 8) :
  (total_revenue - small_price * small_count) / large_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l3977_397769


namespace NUMINAMATH_CALUDE_people_per_tent_l3977_397721

theorem people_per_tent 
  (total_people : ℕ) 
  (house_capacity : ℕ) 
  (num_tents : ℕ) 
  (h1 : total_people = 14) 
  (h2 : house_capacity = 4) 
  (h3 : num_tents = 5) :
  (total_people - house_capacity) / num_tents = 2 :=
by sorry

end NUMINAMATH_CALUDE_people_per_tent_l3977_397721


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_theorem_l3977_397755

def numbers : List Nat := [15, 20, 30]

theorem gcf_lcm_sum_theorem :
  (Nat.gcd (Nat.gcd 15 20) 30) + (Nat.lcm (Nat.lcm 15 20) 30) = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_theorem_l3977_397755


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l3977_397738

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  -- The lengths of the parallel bases
  AB : ℝ
  CD : ℝ
  -- Areas of the four triangles formed by the point
  area_PAB : ℝ
  area_PBC : ℝ
  area_PCD : ℝ
  area_PDA : ℝ
  -- Conditions
  AB_gt_CD : AB > CD
  areas_clockwise : area_PAB = 9 ∧ area_PBC = 7 ∧ area_PCD = 3 ∧ area_PDA = 5

/-- The ratio of the parallel bases in the isosceles trapezoid is 3 -/
theorem isosceles_trapezoid_ratio 
  (T : IsoscelesTrapezoidWithPoint) : T.AB / T.CD = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_ratio_l3977_397738


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_139_l3977_397762

/-- The first nonzero digit to the right of the decimal point in the decimal representation of 1/139 is 1. -/
theorem first_nonzero_digit_of_one_over_139 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 139 = (n : ℚ) / 10^(d + 1) + (1 : ℚ) / (10 * 10^(d + 1)) + (r : ℚ) / (100 * 10^(d + 1)) ∧ 
  0 ≤ r ∧ r < 10 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_139_l3977_397762


namespace NUMINAMATH_CALUDE_craft_corner_sales_l3977_397777

/-- The percentage of sales that are neither brushes nor paints -/
def other_sales_percentage (total : ℝ) (brushes : ℝ) (paints : ℝ) : ℝ :=
  total - (brushes + paints)

/-- Theorem stating that given the total sales is 100%, and brushes and paints account for 45% and 28% of sales respectively, the percentage of sales that are neither brushes nor paints is 27% -/
theorem craft_corner_sales :
  let total := 100
  let brushes := 45
  let paints := 28
  other_sales_percentage total brushes paints = 27 := by
sorry

end NUMINAMATH_CALUDE_craft_corner_sales_l3977_397777


namespace NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l3977_397743

theorem hamilton_marching_band_max_members :
  ∃ (m : ℕ),
    (30 * m) % 31 = 5 ∧
    30 * m < 1500 ∧
    ∀ (n : ℕ), (30 * n) % 31 = 5 ∧ 30 * n < 1500 → 30 * n ≤ 30 * m :=
by sorry

end NUMINAMATH_CALUDE_hamilton_marching_band_max_members_l3977_397743


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3977_397753

theorem sqrt_product_equality : Real.sqrt 125 * Real.sqrt 45 * Real.sqrt 10 = 75 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3977_397753


namespace NUMINAMATH_CALUDE_random_points_probability_l3977_397706

/-- The probability that a randomly chosen point y is greater than another randomly chosen point x
    but less than three times x, where both x and y are chosen uniformly from the interval [0, 1] -/
theorem random_points_probability : Real := by
  sorry

end NUMINAMATH_CALUDE_random_points_probability_l3977_397706


namespace NUMINAMATH_CALUDE_power_product_rule_l3977_397768

theorem power_product_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_product_rule_l3977_397768


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3977_397795

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 2) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l3977_397795


namespace NUMINAMATH_CALUDE_furniture_purchase_price_l3977_397757

theorem furniture_purchase_price :
  let marked_price : ℝ := 132
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.1
  let purchase_price : ℝ := 108
  marked_price * (1 - discount_rate) - purchase_price = profit_rate * purchase_price :=
by
  sorry

end NUMINAMATH_CALUDE_furniture_purchase_price_l3977_397757


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3977_397719

-- Define the possible operations
inductive Operation
  | mul
  | div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℕ) : ℚ :=
  match op with
  | Operation.mul => (a * b : ℚ)
  | Operation.div => (a / b : ℚ)

theorem equation_solution_exists : 
  ∃ (op1 op2 : Operation), 
    (apply_op op1 9 1307 = 100) ∧ 
    (∃ (n : ℕ), apply_op op2 14 2 = apply_op op2 n 5 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3977_397719


namespace NUMINAMATH_CALUDE_special_calculator_problem_l3977_397723

-- Define a function to reverse digits of a number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the calculator operation
def calculator_operation (x : ℕ) : ℕ := reverse_digits (2 * x) + 2

-- Theorem statement
theorem special_calculator_problem (x : ℕ) :
  x ≥ 10 ∧ x < 100 →  -- two-digit number condition
  calculator_operation x = 45 →
  x = 17 := by sorry

end NUMINAMATH_CALUDE_special_calculator_problem_l3977_397723


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3977_397790

theorem multiplication_mistake (x : ℕ) : x = 43 := by
  have h1 : 136 * x - 1224 = 136 * 34 := by sorry
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3977_397790


namespace NUMINAMATH_CALUDE_jenny_cat_expenses_l3977_397779

theorem jenny_cat_expenses (adoption_fee : ℝ) (vet_visits : ℝ) (monthly_food_cost : ℝ) (toy_cost : ℝ) :
  adoption_fee = 50 →
  vet_visits = 500 →
  monthly_food_cost = 25 →
  toy_cost = 200 →
  (adoption_fee + vet_visits + 12 * monthly_food_cost) / 2 + toy_cost = 625 := by
  sorry

end NUMINAMATH_CALUDE_jenny_cat_expenses_l3977_397779


namespace NUMINAMATH_CALUDE_a_range_l3977_397785

noncomputable def f (x : ℝ) : ℝ := 1 / Real.exp x - Real.exp x + 2 * x - (1 / 3) * x^3

theorem a_range (a : ℝ) : f (3 * a^2) + f (2 * a - 1) ≥ 0 → a ∈ Set.Icc (-1) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3977_397785


namespace NUMINAMATH_CALUDE_count_students_in_line_l3977_397756

/-- The number of students standing in a line, given that Yoojeong is at the back and there are some people in front of her. -/
def studentsInLine (peopleInFront : ℕ) : ℕ :=
  peopleInFront + 1

/-- Theorem stating that the number of students in the line is equal to the number of people in front of Yoojeong plus one. -/
theorem count_students_in_line (peopleInFront : ℕ) :
  studentsInLine peopleInFront = peopleInFront + 1 := by
  sorry

end NUMINAMATH_CALUDE_count_students_in_line_l3977_397756


namespace NUMINAMATH_CALUDE_square_roots_sum_equals_ten_l3977_397747

theorem square_roots_sum_equals_ten :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_sum_equals_ten_l3977_397747


namespace NUMINAMATH_CALUDE_quadratic_roots_of_nine_l3977_397763

theorem quadratic_roots_of_nine (x : ℝ) : x^2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_of_nine_l3977_397763


namespace NUMINAMATH_CALUDE_binomial_16_choose_12_l3977_397740

theorem binomial_16_choose_12 : Nat.choose 16 12 = 43680 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_choose_12_l3977_397740


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3977_397703

theorem isosceles_triangle_largest_angle (α β γ : ℝ) :
  -- The triangle is isosceles with two equal angles
  α = β ∧
  -- One of the equal angles is 30°
  α = 30 ∧
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- The largest angle is 120°
  max α (max β γ) = 120 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3977_397703


namespace NUMINAMATH_CALUDE_zero_of_f_l3977_397720

-- Define the function f(x) = 2x + 7
def f (x : ℝ) : ℝ := 2 * x + 7

-- Theorem stating that the zero of f(x) is -7/2
theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -7/2 := by
sorry

end NUMINAMATH_CALUDE_zero_of_f_l3977_397720


namespace NUMINAMATH_CALUDE_triangle_property_l3977_397732

open Real

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * cos B - b * cos A = c / 2 ∧
  B = π / 4 ∧
  b = sqrt 5 →
  tan A = 3 * tan B ∧
  (1 / 2) * a * b * sin C = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l3977_397732


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_800_l3977_397758

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∀ n : ℕ, n < 800 ∧ 5 ∣ n ∧ 6 ∣ n → n ≤ 780 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_800_l3977_397758


namespace NUMINAMATH_CALUDE_f_range_is_real_l3977_397787

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.cos x * Real.sin x + Real.sin x ^ 4 + Real.tan x

theorem f_range_is_real : Set.range f = Set.univ := by sorry

end NUMINAMATH_CALUDE_f_range_is_real_l3977_397787


namespace NUMINAMATH_CALUDE_seedling_problem_l3977_397750

theorem seedling_problem (x : ℕ) : 
  (x^2 + 39 = (x + 1)^2 - 50) → (x^2 + 39 = 1975) :=
by
  sorry

#check seedling_problem

end NUMINAMATH_CALUDE_seedling_problem_l3977_397750


namespace NUMINAMATH_CALUDE_set_union_condition_implies_p_bound_l3977_397786

/-- Given sets A and B, where A = {x | -2 < x < 5} and B = {x | p+1 < x < 2p-1},
    if A ∪ B = A, then p ≤ 3. -/
theorem set_union_condition_implies_p_bound (p : ℝ) :
  let A : Set ℝ := {x | -2 < x ∧ x < 5}
  let B : Set ℝ := {x | p + 1 < x ∧ x < 2*p - 1}
  (A ∪ B = A) → p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_set_union_condition_implies_p_bound_l3977_397786


namespace NUMINAMATH_CALUDE_lisa_borrowed_chairs_l3977_397776

/-- The number of chairs Lisa borrowed from Rodrigo's classroom -/
def chairs_borrowed (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ) : ℕ :=
  total_chairs_before - total_chairs_after

/-- Theorem stating the number of chairs Lisa borrowed -/
theorem lisa_borrowed_chairs : 
  ∀ (red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after : ℕ),
  red_chairs = 4 →
  yellow_chairs = 2 * red_chairs →
  blue_chairs = yellow_chairs - 2 →
  total_chairs_before = red_chairs + yellow_chairs + blue_chairs →
  total_chairs_after = 15 →
  chairs_borrowed red_chairs yellow_chairs blue_chairs total_chairs_before total_chairs_after = 3 :=
by sorry

end NUMINAMATH_CALUDE_lisa_borrowed_chairs_l3977_397776


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3977_397716

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (-3 + 2*i) / (1 + i) = -1/2 + 5/2*i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3977_397716


namespace NUMINAMATH_CALUDE_cube_root_existence_l3977_397791

theorem cube_root_existence : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by
  sorry

end NUMINAMATH_CALUDE_cube_root_existence_l3977_397791


namespace NUMINAMATH_CALUDE_set_equality_l3977_397725

theorem set_equality (x y : ℝ) : 
  (x^2 - y^2 = x / (x^2 + y^2) ∧ 2*x*y + y / (x^2 + y^2) = 3) ↔ 
  (x^3 - 3*x*y^2 + 3*y = 1 ∧ 3*x^2*y - 3*x - y^3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3977_397725


namespace NUMINAMATH_CALUDE_square_area_increase_l3977_397700

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.4 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.96 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l3977_397700


namespace NUMINAMATH_CALUDE_polygon_three_sides_l3977_397728

/-- A polygon with n sides where the sum of interior angles is less than the sum of exterior angles. -/
structure Polygon (n : ℕ) where
  interior_sum : ℝ
  exterior_sum : ℝ
  interior_less : interior_sum < exterior_sum
  exterior_360 : exterior_sum = 360

/-- Theorem: If a polygon's interior angle sum is less than its exterior angle sum (which is 360°), then it has 3 sides. -/
theorem polygon_three_sides {n : ℕ} (p : Polygon n) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_polygon_three_sides_l3977_397728


namespace NUMINAMATH_CALUDE_linear_equation_solutions_l3977_397704

theorem linear_equation_solutions : 
  {(x, y) : ℕ × ℕ | 5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0} = {(1, 10), (3, 5)} := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solutions_l3977_397704


namespace NUMINAMATH_CALUDE_binomial_divisibility_l3977_397724

theorem binomial_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (Nat.choose (2*p - 1) (p - 1) : ℤ) - 1 = k * p^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l3977_397724
