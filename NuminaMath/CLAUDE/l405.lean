import Mathlib

namespace problem_solution_l405_40592

theorem problem_solution : 
  (∃ n : ℕ, 140 * 5 = n * 100 ∧ n % 10 ≠ 0) ∧ 
  (4 * 150 - 7 = 593) := by
  sorry

end problem_solution_l405_40592


namespace max_n_for_consecutive_product_l405_40525

theorem max_n_for_consecutive_product (n : ℕ) : 
  (∃ k : ℕ, 9*n^2 + 5*n + 26 = k * (k + 1)) → n ≤ 6 :=
by sorry

end max_n_for_consecutive_product_l405_40525


namespace triangle_prime_angles_l405_40537

theorem triangle_prime_angles (a b c : ℕ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c →  -- All angles are prime
  a = 2 ∨ b = 2 ∨ c = 2 :=  -- One angle must be 2 degrees
by
  sorry

end triangle_prime_angles_l405_40537


namespace vasyas_numbers_l405_40564

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
  sorry

end vasyas_numbers_l405_40564


namespace no_function_satisfies_inequality_l405_40502

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), (∀ x > 0, f x > 0) ∧
    (∀ x y, x > 0 → y > 0 → f x ^ 2 ≥ f (x + y) * (f x + y)) := by
  sorry

end no_function_satisfies_inequality_l405_40502


namespace count_valid_lists_l405_40555

/-- A structure representing a list of five integers with the given properties -/
structure IntegerList :=
  (a b : ℕ+)
  (h1 : a < b)
  (h2 : 2 * a.val + 3 * b.val = 124)

/-- The number of valid integer lists -/
def validListCount : ℕ := sorry

/-- Theorem stating that there are exactly 8 valid integer lists -/
theorem count_valid_lists : validListCount = 8 := by sorry

end count_valid_lists_l405_40555


namespace computation_proof_l405_40585

theorem computation_proof : 8 * (250 / 3 + 50 / 6 + 16 / 32 + 2) = 2260 / 3 := by
  sorry

end computation_proof_l405_40585


namespace average_score_calculation_l405_40512

theorem average_score_calculation (T : ℝ) (h : T > 0) :
  let male_ratio : ℝ := 0.4
  let female_ratio : ℝ := 1 - male_ratio
  let male_avg : ℝ := 75
  let female_avg : ℝ := 80
  let total_score : ℝ := male_ratio * T * male_avg + female_ratio * T * female_avg
  total_score / T = 78 := by
  sorry

end average_score_calculation_l405_40512


namespace correspondence_proof_l405_40500

/-- Given sets A and B, and a mapping f from A to B defined as
    f(x, y) = (x + 2y, 2x - y), prove that (1, 1) in A
    corresponds to (3, 1) in B under this mapping. -/
theorem correspondence_proof (A B : Set (ℝ × ℝ)) (f : ℝ × ℝ → ℝ × ℝ)
    (hf : ∀ (x y : ℝ), f (x, y) = (x + 2*y, 2*x - y))
    (hA : (1, 1) ∈ A) (hB : (3, 1) ∈ B) :
    f (1, 1) = (3, 1) := by
  sorry

end correspondence_proof_l405_40500


namespace solve_equation_and_evaluate_l405_40526

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 7 = 15 * x + 21) → 3 * (x + 10) = 21.6 := by
  sorry

end solve_equation_and_evaluate_l405_40526


namespace division_problem_l405_40531

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 690 →
  divisor = 36 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end division_problem_l405_40531


namespace remainder_theorem_l405_40591

theorem remainder_theorem (n : ℤ) : n % 9 = 3 → (4 * n - 9) % 9 = 3 := by
  sorry

end remainder_theorem_l405_40591


namespace cat_monitored_area_percentage_l405_40598

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangular room -/
structure Room where
  width : ℝ
  height : ℝ

/-- Calculates the area of a room -/
def roomArea (r : Room) : ℝ := r.width * r.height

/-- Calculates the area that a cat can monitor in a room -/
noncomputable def monitoredArea (r : Room) (catPosition : Point) : ℝ := sorry

/-- Theorem stating that a cat at (3, 8) in a 10x8 room monitors 66.875% of the area -/
theorem cat_monitored_area_percentage (r : Room) (catPos : Point) :
  r.width = 10 ∧ r.height = 8 ∧ catPos.x = 3 ∧ catPos.y = 8 →
  monitoredArea r catPos / roomArea r = 66.875 / 100 := by sorry

end cat_monitored_area_percentage_l405_40598


namespace main_theorem_l405_40517

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent length
def tangent_length (p : Point_P) : ℝ := sorry

-- Define the circumcircle N
def circle_N (p : Point_P) (x y : ℝ) : Prop := sorry

-- Define the chord length AB
def chord_length (p : Point_P) : ℝ := sorry

theorem main_theorem :
  (∃ p1 p2 : Point_P, tangent_length p1 = 2*Real.sqrt 3 ∧ tangent_length p2 = 2*Real.sqrt 3 ∧
    ((p1.x = 0 ∧ p1.y = 0) ∨ (p1.x = 16/5 ∧ p1.y = 8/5)) ∧
    ((p2.x = 0 ∧ p2.y = 0) ∨ (p2.x = 16/5 ∧ p2.y = 8/5))) ∧
  (∀ p : Point_P, circle_N p 0 4 ∧ circle_N p (8/5) (4/5)) ∧
  (∃ p_min : Point_P, ∀ p : Point_P, chord_length p_min ≤ chord_length p ∧ chord_length p_min = Real.sqrt 11) :=
sorry

end main_theorem_l405_40517


namespace power_sum_equality_l405_40553

theorem power_sum_equality : 2^567 + 9^5 / 3^2 = 2^567 + 6561 := by sorry

end power_sum_equality_l405_40553


namespace specific_arc_rectangle_boundary_l405_40543

/-- Represents a rectangle with quarter-circle arcs on its corners -/
structure ArcRectangle where
  area : ℝ
  length_width_ratio : ℝ
  divisions : ℕ

/-- Calculates the boundary length of the ArcRectangle -/
def boundary_length (r : ArcRectangle) : ℝ :=
  sorry

/-- Theorem stating the boundary length of a specific ArcRectangle -/
theorem specific_arc_rectangle_boundary :
  let r : ArcRectangle := { area := 72, length_width_ratio := 2, divisions := 3 }
  boundary_length r = 4 * Real.pi + 24 := by
  sorry

end specific_arc_rectangle_boundary_l405_40543


namespace hurleys_age_l405_40509

theorem hurleys_age (hurley_age richard_age : ℕ) : 
  richard_age - hurley_age = 20 →
  (richard_age + 40) + (hurley_age + 40) = 128 →
  hurley_age = 14 := by
sorry

end hurleys_age_l405_40509


namespace m_equals_two_sufficient_not_necessary_l405_40583

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (m : ℝ) : Prop := A m ∩ B = {4}

-- Theorem statement
theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → property m) ∧
  (∃ m : ℝ, m ≠ 2 ∧ property m) :=
sorry

end m_equals_two_sufficient_not_necessary_l405_40583


namespace problem_1_problem_2_problem_3_l405_40593

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^3 * (y - x)^2 = (x - y)^5 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (-3 * a^2)^3 = -27 * a^6 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) (h : x ≠ 0) : x^10 / (2*x)^2 = x^8 / 4 := by sorry

end problem_1_problem_2_problem_3_l405_40593


namespace luke_rounds_played_l405_40545

/-- The number of points Luke scored in total -/
def total_points : ℕ := 154

/-- The number of points Luke gained in each round -/
def points_per_round : ℕ := 11

/-- The number of rounds Luke played -/
def rounds_played : ℕ := total_points / points_per_round

theorem luke_rounds_played :
  rounds_played = 14 :=
by sorry

end luke_rounds_played_l405_40545


namespace percent_within_one_std_dev_l405_40584

/-- A symmetric distribution about a mean -/
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

/-- Theorem: In a symmetric distribution where 80% is less than mean + std_dev,
    60% lies within one standard deviation of the mean -/
theorem percent_within_one_std_dev
  (dist : SymmetricDistribution)
  (h_symmetric : dist.is_symmetric = true)
  (h_eighty_percent : dist.percent_less_than_mean_plus_std = 80) :
  ∃ (percent_within : ℝ), percent_within = 60 :=
sorry

end percent_within_one_std_dev_l405_40584


namespace square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l405_40579

-- Part 1
theorem square_root_five_expansion (a b m n : ℤ) :
  a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2 →
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem square_root_three_expansion :
  ∃ (x m n : ℕ+), x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 ∧
  ((m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7)) :=
sorry

-- Part 3
theorem simplify_nested_square_root :
  Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l405_40579


namespace complex_equation_solution_l405_40519

theorem complex_equation_solution (a b : ℝ) (z : ℂ) 
  (hz : z = Complex.mk a b) 
  (heq : Complex.I / z = Complex.mk 2 (-1)) : 
  a - b = -3/5 := by
  sorry

end complex_equation_solution_l405_40519


namespace double_cone_is_cone_l405_40560

/-- Represents a point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Defines the set of points satisfying the given equations -/
def DoubleConeSurface (c : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.r = |p.z|}

/-- Defines a cone surface in cylindrical coordinates -/
def ConeSurface : Set CylindricalPoint :=
  {p : CylindricalPoint | ∃ (k : ℝ), p.r = k * |p.z|}

/-- Theorem stating that the surface defined by the equations is a cone -/
theorem double_cone_is_cone (c : ℝ) :
  ∃ (k : ℝ), DoubleConeSurface c ⊆ ConeSurface :=
sorry

end double_cone_is_cone_l405_40560


namespace f_greater_than_g_l405_40594

/-- The function f defined as f(x) = 3x^2 - x + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 - x + 1

/-- The function g defined as g(x) = 2x^2 + x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

/-- For all real x, f(x) > g(x) -/
theorem f_greater_than_g : ∀ x : ℝ, f x > g x := by sorry

end f_greater_than_g_l405_40594


namespace product_greater_than_sum_implies_sum_greater_than_four_l405_40530

theorem product_greater_than_sum_implies_sum_greater_than_four (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_product : x * y > x + y) : x + y > 4 := by
  sorry

end product_greater_than_sum_implies_sum_greater_than_four_l405_40530


namespace reggie_brother_long_shots_l405_40559

-- Define the point values for each shot type
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shots
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the point difference
def point_difference : ℕ := 2

-- Theorem to prove
theorem reggie_brother_long_shots :
  let reggie_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
  let brother_points := reggie_points + point_difference
  brother_points / long_shot_points = 4 :=
by sorry

end reggie_brother_long_shots_l405_40559


namespace simplify_sqrt_difference_l405_40561

theorem simplify_sqrt_difference : 
  (Real.sqrt 648 / Real.sqrt 81) - (Real.sqrt 294 / Real.sqrt 49) = 2 * Real.sqrt 2 - Real.sqrt 42 := by
  sorry

end simplify_sqrt_difference_l405_40561


namespace ball_drawing_probabilities_l405_40599

/-- The total number of balls -/
def total_balls : ℕ := 6

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of black balls -/
def black_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing two balls of the same color -/
def prob_same_color : ℚ := 2/5

/-- The probability of drawing two balls of different colors -/
def prob_diff_color : ℚ := 3/5

theorem ball_drawing_probabilities :
  (prob_same_color + prob_diff_color = 1) ∧
  (prob_same_color = 2/5) ∧
  (prob_diff_color = 3/5) :=
by sorry

end ball_drawing_probabilities_l405_40599


namespace solve_fraction_equation_l405_40515

theorem solve_fraction_equation :
  ∀ x : ℚ, (2 / 3 - 1 / 4 : ℚ) = 1 / x → x = 12 / 5 := by
  sorry

end solve_fraction_equation_l405_40515


namespace sqrt_point_five_equals_sqrt_two_over_two_l405_40547

theorem sqrt_point_five_equals_sqrt_two_over_two :
  Real.sqrt 0.5 = Real.sqrt 2 / 2 := by
  sorry

end sqrt_point_five_equals_sqrt_two_over_two_l405_40547


namespace simplify_sqrt_quadratic_l405_40554

theorem simplify_sqrt_quadratic (x : ℝ) (h : x < 2) : 
  Real.sqrt (x^2 - 4*x + 4) = 2 - x := by
sorry

end simplify_sqrt_quadratic_l405_40554


namespace symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l405_40580

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 2

-- Define the symmetric axis
def symmetric_axis : ℝ := -3

-- Theorem statement
theorem symmetric_axis_of_shifted_quadratic :
  ∀ x : ℝ, f (symmetric_axis + x) = f (symmetric_axis - x) := by
  sorry

-- The symmetric axis is unique
theorem unique_symmetric_axis :
  ∀ h : ℝ, h ≠ symmetric_axis →
  ∃ x : ℝ, f (h + x) ≠ f (h - x) := by
  sorry

end symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l405_40580


namespace three_billion_three_hundred_million_scientific_notation_l405_40523

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem three_billion_three_hundred_million_scientific_notation :
  to_scientific_notation 3300000000 = ScientificNotation.mk 3.3 9 sorry := by
  sorry

end three_billion_three_hundred_million_scientific_notation_l405_40523


namespace inverse_on_negative_T_to_0_l405_40572

-- Define a periodic function f with period T
def isPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define the smallest positive period
def isSmallestPositivePeriod (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ isPeriodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬isPeriodic f S

-- Define the inverse function on (0, T)
def inverseOn0T (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ) : Prop :=
  ∀ x ∈ D, 0 < x ∧ x < T → f (fInv x) = x ∧ fInv (f x) = x

-- Main theorem
theorem inverse_on_negative_T_to_0
  (f : ℝ → ℝ) (T : ℝ) (D : Set ℝ) (fInv : ℝ → ℝ)
  (h_periodic : isPeriodic f T)
  (h_smallest : isSmallestPositivePeriod f T)
  (h_inverse : inverseOn0T f T D fInv) :
  ∀ x ∈ D, -T < x ∧ x < 0 → f (fInv x - T) = x ∧ fInv x - T = f⁻¹ x :=
sorry

end inverse_on_negative_T_to_0_l405_40572


namespace cubic_function_property_l405_40529

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f (-2) = 2 → f 2 = -10 := by
sorry

end cubic_function_property_l405_40529


namespace inequality_system_solution_l405_40586

theorem inequality_system_solution (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
by sorry

end inequality_system_solution_l405_40586


namespace vermont_ads_l405_40510

def ads_problem (first_page_ads : ℕ) (total_pages : ℕ) (click_fraction : ℚ) : Prop :=
  let second_page_ads := 2 * first_page_ads
  let third_page_ads := second_page_ads + 24
  let fourth_page_ads := (3 : ℚ) / 4 * second_page_ads
  let total_ads := first_page_ads + second_page_ads + third_page_ads + fourth_page_ads
  let clicked_ads := click_fraction * total_ads
  
  first_page_ads = 12 ∧
  total_pages = 4 ∧
  click_fraction = 2 / 3 ∧
  clicked_ads = 68

theorem vermont_ads : ads_problem 12 4 (2/3) := by sorry

end vermont_ads_l405_40510


namespace quadratic_coefficient_l405_40556

/-- Given a quadratic equation 5 * x^2 + 14 * x + 5 = 0 with two reciprocal roots,
    the coefficient of the squared term is 5. -/
theorem quadratic_coefficient (x : ℝ) :
  (5 * x^2 + 14 * x + 5 = 0) →
  (∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (x = r₁ ∨ x = r₂) ∧ 5 * r₁^2 + 14 * r₁ + 5 = 0 ∧ 5 * r₂^2 + 14 * r₂ + 5 = 0) →
  5 = 5 :=
by sorry

end quadratic_coefficient_l405_40556


namespace sequence_identity_l405_40540

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n ≥ 1 ∧ a (a n) + a n = 2 * n

theorem sequence_identity (a : ℕ → ℕ) (h : is_valid_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = n :=
sorry

end sequence_identity_l405_40540


namespace qr_equals_b_l405_40568

def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

theorem qr_equals_b (a b c : ℝ) (h1 : curve c a b) (h2 : curve c 0 c) :
  let normal_slope := -1 / (Real.sinh (a / c) / c)
  let r_x := c * Real.sinh (a / c) / 2
  Real.sqrt ((r_x - 0)^2 + (0 - c)^2) = b := by sorry

end qr_equals_b_l405_40568


namespace A_star_B_equals_zero_three_l405_40589

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

def star (X Y : Set ℕ) : Set ℕ :=
  {x | x ∈ X ∨ x ∈ Y ∧ x ∉ X ∩ Y}

theorem A_star_B_equals_zero_three : star A B = {0, 3} := by
  sorry

end A_star_B_equals_zero_three_l405_40589


namespace cube_third_times_eighth_equals_one_over_216_l405_40582

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by sorry

end cube_third_times_eighth_equals_one_over_216_l405_40582


namespace isosceles_triangle_base_angle_l405_40533

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We represent angles in degrees as natural numbers
  base_angle₁ : ℕ
  base_angle₂ : ℕ
  vertex_angle : ℕ
  is_isosceles : base_angle₁ = base_angle₂
  angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180

theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : t.base_angle₁ = 50 ∨ t.vertex_angle = 50) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 := by
  sorry

end isosceles_triangle_base_angle_l405_40533


namespace smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l405_40539

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ↔ (x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 3) :=
by sorry

theorem smallest_solution_is_3_minus_sqrt_3 :
  ∃ x : ℝ, (1 / (x - 2) + 1 / (x - 4) = 3 / (x - 3)) ∧
           (∀ y : ℝ, (1 / (y - 2) + 1 / (y - 4) = 3 / (y - 3)) → x ≤ y) ∧
           x = 3 - Real.sqrt 3 :=
by sorry

end smallest_solution_of_equation_smallest_solution_is_3_minus_sqrt_3_l405_40539


namespace quadratic_equal_roots_l405_40503

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k - 3 = 0 ∧ (∀ y : ℝ, y^2 + k - 3 = 0 → y = x)) ↔ k = 3 := by
  sorry

end quadratic_equal_roots_l405_40503


namespace arithmetic_progression_formula_recursive_formula_initial_condition_l405_40507

/-- Arithmetic progression with first term 13.5 and common difference 4.2 -/
def arithmetic_progression (n : ℕ) : ℝ :=
  13.5 + (n - 1 : ℝ) * 4.2

/-- The nth term of the arithmetic progression -/
def nth_term (n : ℕ) : ℝ :=
  4.2 * n + 9.3

theorem arithmetic_progression_formula (n : ℕ) :
  arithmetic_progression n = nth_term n := by sorry

theorem recursive_formula (n : ℕ) (h : n > 0) :
  arithmetic_progression (n + 1) = arithmetic_progression n + 4.2 := by sorry

theorem initial_condition :
  arithmetic_progression 1 = 13.5 := by sorry

end arithmetic_progression_formula_recursive_formula_initial_condition_l405_40507


namespace smallest_base_for_inequality_l405_40562

theorem smallest_base_for_inequality (k : ℕ) (h : k = 7) : 
  (∃ (base : ℕ), base^k > 4^20 ∧ ∀ (b : ℕ), b < base → b^k ≤ 4^20) ↔ 64^k > 4^20 ∧ ∀ (b : ℕ), b < 64 → b^k ≤ 4^20 :=
by sorry

end smallest_base_for_inequality_l405_40562


namespace rhombus_area_l405_40563

/-- The area of a rhombus with side length 3 cm and an acute angle of 45 degrees is 9√2/2 square centimeters. -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) :
  side_length = 3 →
  acute_angle = 45 * π / 180 →
  let area := side_length * side_length * Real.sin acute_angle
  area = 9 * Real.sqrt 2 / 2 := by
  sorry

#check rhombus_area

end rhombus_area_l405_40563


namespace square_rearrangement_theorem_l405_40536

-- Define a type for square sheets of paper
def Square : Type := Unit

-- Define a function that represents the possibility of cutting and rearranging squares
def can_cut_and_rearrange (n : ℕ) : Prop :=
  ∀ (squares : Fin n → Square), ∃ (new_square : Square), True

-- State the theorem
theorem square_rearrangement_theorem (n : ℕ) (h : n > 1) :
  can_cut_and_rearrange n :=
sorry

end square_rearrangement_theorem_l405_40536


namespace equation_solution_l405_40535

theorem equation_solution (a b x : ℝ) :
  (a ≠ b ∧ a ≠ -b ∧ b ≠ 0 → x = a^2 - b^2 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) ∧
  (b = 0 ∧ a ≠ 0 ∧ x ≠ 0 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) :=
by sorry

end equation_solution_l405_40535


namespace fish_tank_problem_l405_40522

theorem fish_tank_problem (x : ℕ) : x + (x - 4) = 20 → x - 4 = 8 := by
  sorry

end fish_tank_problem_l405_40522


namespace nicki_total_miles_run_l405_40528

/-- Calculates the total miles run in a year given weekly mileage for each half -/
def total_miles_run (weeks_in_year : ℕ) (miles_first_half : ℕ) (miles_second_half : ℕ) : ℕ :=
  let half_year := weeks_in_year / 2
  (miles_first_half * half_year) + (miles_second_half * half_year)

theorem nicki_total_miles_run : total_miles_run 52 20 30 = 1300 := by
  sorry

#eval total_miles_run 52 20 30

end nicki_total_miles_run_l405_40528


namespace linda_savings_l405_40576

theorem linda_savings (savings : ℝ) : 
  (5 / 6 : ℝ) * savings + 500 = savings → savings = 3000 := by
  sorry

end linda_savings_l405_40576


namespace primes_not_sum_of_composites_l405_40596

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d, d ∣ n → d = 1 ∨ d = n

def cannot_be_sum_of_two_composites (p : ℕ) : Prop :=
  is_prime p ∧ ¬∃ a b, is_composite a ∧ is_composite b ∧ p = a + b

theorem primes_not_sum_of_composites :
  {p : ℕ | cannot_be_sum_of_two_composites p} = {2, 3, 5, 7, 11} :=
sorry

end primes_not_sum_of_composites_l405_40596


namespace problem_solutions_l405_40552

theorem problem_solutions :
  (∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0) ∧
  (∃ x : ℕ, x^2 ≤ x) ∧
  (∃ x : ℕ, 29 % x = 0) := by
  sorry

end problem_solutions_l405_40552


namespace eccentricity_ratio_for_common_point_l405_40534

/-- The eccentricity of an ellipse -/
def eccentricity_ellipse (a b : ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity_hyperbola (a b : ℝ) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem eccentricity_ratio_for_common_point 
  (F₁ F₂ P : ℝ × ℝ) 
  (e₁ : ℝ) 
  (e₂ : ℝ) 
  (h_ellipse : e₁ = eccentricity_ellipse (distance F₁ P) (distance F₂ P))
  (h_hyperbola : e₂ = eccentricity_hyperbola (distance F₁ P) (distance F₂ P))
  (h_common_point : distance P F₁ + distance P F₂ = distance F₁ F₂) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 := by
  sorry

end eccentricity_ratio_for_common_point_l405_40534


namespace inequality_proof_equality_condition_theorem_l405_40548

-- Define the theorem
theorem inequality_proof (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) ≥ Real.sqrt (a^2 + c^2 + a*c) := by
  sorry

-- Define the equality condition
def equality_condition (a b c : ℝ) : Prop :=
  (a * c = a * b + b * c) ∧ (a * b + a * c + b * c - 2 * b^2 ≥ 0)

-- Theorem for the equality condition
theorem equality_condition_theorem (a b c : ℝ) :
  (Real.sqrt (a^2 + b^2 - a*b) + Real.sqrt (b^2 + c^2 - b*c) = Real.sqrt (a^2 + c^2 + a*c)) ↔
  equality_condition a b c := by
  sorry

end inequality_proof_equality_condition_theorem_l405_40548


namespace no_clax_is_snapp_l405_40571

-- Define the sets
variable (U : Type) -- Universe set
variable (Clax Ell Snapp Plott : Set U)

-- Define the conditions
variable (h1 : Clax ⊆ Ellᶜ)
variable (h2 : ∃ x, x ∈ Ell ∩ Snapp)
variable (h3 : Snapp ∩ Plott = ∅)

-- State the theorem
theorem no_clax_is_snapp : Clax ∩ Snapp = ∅ := by
  sorry

end no_clax_is_snapp_l405_40571


namespace complement_of_M_in_U_l405_40550

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M_in_U : 
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end complement_of_M_in_U_l405_40550


namespace dress_cost_calculation_l405_40569

def dresses : ℕ := 5
def pants : ℕ := 3
def jackets : ℕ := 4
def pants_cost : ℕ := 12
def jackets_cost : ℕ := 30
def transportation_cost : ℕ := 5
def initial_money : ℕ := 400
def remaining_money : ℕ := 139

theorem dress_cost_calculation (dress_cost : ℕ) : 
  dress_cost * dresses + pants * pants_cost + jackets * jackets_cost + transportation_cost = initial_money - remaining_money → 
  dress_cost = 20 := by
sorry

end dress_cost_calculation_l405_40569


namespace constant_is_two_l405_40520

theorem constant_is_two (p c : ℕ) (n : ℕ) (hp : Prime p) (hp_gt_two : p > 2)
  (hn : n = c * p) (h_one_even_divisor : ∃! d : ℕ, d ∣ n ∧ Even d) : c = 2 := by
  sorry

end constant_is_two_l405_40520


namespace interior_angles_sum_l405_40541

theorem interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 2340) →
  (180 * ((n + 3) - 2) = 2880) :=
by sorry

end interior_angles_sum_l405_40541


namespace watermelons_last_two_weeks_l405_40504

/-- Represents the number of watermelons Jeremy eats in a given week -/
def jeremyEats (week : ℕ) : ℕ :=
  match week % 3 with
  | 0 => 3
  | 1 => 4
  | _ => 5

/-- Represents the number of watermelons Jeremy gives to his dad in a given week -/
def dadReceives (week : ℕ) : ℕ := week + 1

/-- Represents the number of watermelons Jeremy gives to his sister in a given week -/
def sisterReceives (week : ℕ) : ℕ := 2 * week - 1

/-- Represents the number of watermelons Jeremy gives to his neighbor in a given week -/
def neighborReceives (week : ℕ) : ℕ := max (2 - week) 0

/-- Represents the total number of watermelons consumed in a given week -/
def totalConsumed (week : ℕ) : ℕ :=
  jeremyEats week + dadReceives week + sisterReceives week + neighborReceives week

/-- The initial number of watermelons -/
def initialWatermelons : ℕ := 30

/-- Theorem stating that the watermelons will last for 2 complete weeks -/
theorem watermelons_last_two_weeks :
  initialWatermelons ≥ totalConsumed 1 + totalConsumed 2 ∧
  initialWatermelons < totalConsumed 1 + totalConsumed 2 + totalConsumed 3 :=
sorry

end watermelons_last_two_weeks_l405_40504


namespace inequality_proof_l405_40544

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) : 
  x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z) := by
  sorry

end inequality_proof_l405_40544


namespace discount_percentage_l405_40542

theorem discount_percentage (srp mp paid : ℝ) : 
  srp = 1.2 * mp →  -- SRP is 20% higher than MP
  paid = 0.6 * mp →  -- John paid 60% of MP (40% off)
  paid / srp = 0.5 :=  -- John paid 50% of SRP
by sorry

end discount_percentage_l405_40542


namespace triangle_ef_length_l405_40505

/-- Given a triangle DEF with the specified conditions, prove that EF = 3 -/
theorem triangle_ef_length (D E F : ℝ) (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2) (h2 : DE = 6) : EF = 3 := by
  sorry

end triangle_ef_length_l405_40505


namespace least_4_light_four_digit_l405_40597

def is_4_light (n : ℕ) : Prop := n % 9 < 4

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_4_light_four_digit : 
  (∀ n : ℕ, is_four_digit n → is_4_light n → 1000 ≤ n) ∧ is_four_digit 1000 ∧ is_4_light 1000 :=
sorry

end least_4_light_four_digit_l405_40597


namespace ellipse_parameter_sum_l405_40590

/-- Given two points F₁ and F₂ in the plane, we define an ellipse as the set of points P
    such that PF₁ + PF₂ is constant. This theorem proves that for the specific points
    F₁ = (2, 3) and F₂ = (8, 3), and the constant sum PF₁ + PF₂ = 10, 
    the resulting ellipse has parameters h, k, a, and b whose sum is 17. -/
theorem ellipse_parameter_sum : 
  let F₁ : ℝ × ℝ := (2, 3)
  let F₂ : ℝ × ℝ := (8, 3)
  let distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let is_on_ellipse (P : ℝ × ℝ) : Prop := distance P F₁ + distance P F₂ = 10
  let h : ℝ := (F₁.1 + F₂.1) / 2
  let k : ℝ := F₁.2  -- since F₁.2 = F₂.2
  let c : ℝ := distance F₁ ((F₁.1 + F₂.1) / 2, F₁.2) / 2
  let a : ℝ := 5  -- half of the constant sum
  let b : ℝ := Real.sqrt (a^2 - c^2)
  h + k + a + b = 17
  := by sorry

end ellipse_parameter_sum_l405_40590


namespace sum_of_reciprocals_l405_40518

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (sum_prod : x + y = 6 * x * y) (double : y = 2 * x) :
  1 / x + 1 / y = 6 := by sorry

end sum_of_reciprocals_l405_40518


namespace max_similar_triangle_lines_l405_40532

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is outside a triangle --/
def IsOutside (P : Point) (T : Triangle) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (L : Line) (P : Point) : Prop := sorry

/-- Predicate to check if a line cuts off a similar triangle --/
def CutsSimilarTriangle (L : Line) (T : Triangle) : Prop := sorry

/-- The main theorem --/
theorem max_similar_triangle_lines 
  (T : Triangle) (P : Point) (h : IsOutside P T) :
  ∃ (S : Finset Line), 
    (∀ L ∈ S, PassesThrough L P ∧ CutsSimilarTriangle L T) ∧ 
    S.card = 6 ∧
    (∀ S' : Finset Line, 
      (∀ L ∈ S', PassesThrough L P ∧ CutsSimilarTriangle L T) → 
      S'.card ≤ 6) := by
  sorry

end max_similar_triangle_lines_l405_40532


namespace parabola_translation_l405_40574

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally -/
def translate_x (p : Parabola) (dx : ℝ) : Parabola :=
  { f := fun x => p.f (x - dx) }

/-- Translates a parabola vertically -/
def translate_y (p : Parabola) (dy : ℝ) : Parabola :=
  { f := fun x => p.f x + dy }

/-- The original parabola y = x^2 + 3 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 + 3 }

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  { f := fun x => (x+3)^2 - 1 }

theorem parabola_translation :
  (translate_y (translate_x original_parabola 3) (-4)).f =
  resulting_parabola.f := by sorry

end parabola_translation_l405_40574


namespace hyperbola_equation_l405_40558

/-- Given a hyperbola and conditions on its asymptote and focus, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∃ (x y : ℝ), y = Real.sqrt 3 * x) ∧
  (∃ (x : ℝ), x^2 + b^2 = 144) →
  a^2 = 36 ∧ b^2 = 108 := by
  sorry

end hyperbola_equation_l405_40558


namespace right_triangle_sin_value_l405_40551

theorem right_triangle_sin_value (A B C : Real) (h1 : 0 < A) (h2 : A < π / 2) :
  (Real.cos B = 0) →
  (3 * Real.sin A = 4 * Real.cos A) →
  Real.sin A = 4 / 5 := by
sorry

end right_triangle_sin_value_l405_40551


namespace equal_perimeter_lines_concurrent_l405_40538

open Real

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line through a vertex
structure VertexLine :=
  (vertex : ℝ × ℝ)
  (point : ℝ × ℝ)

-- Function to check if a line divides a triangle into two triangles with equal perimeter
def divides_equal_perimeter (t : Triangle) (l : VertexLine) : Prop :=
  sorry

-- Function to check if three lines are concurrent
def are_concurrent (l1 l2 l3 : VertexLine) : Prop :=
  sorry

-- Theorem statement
theorem equal_perimeter_lines_concurrent (t : Triangle) :
  ∀ (l1 l2 l3 : VertexLine),
    (divides_equal_perimeter t l1 ∧ 
     divides_equal_perimeter t l2 ∧ 
     divides_equal_perimeter t l3) →
    are_concurrent l1 l2 l3 :=
sorry

end equal_perimeter_lines_concurrent_l405_40538


namespace quadrilateral_area_product_not_1988_l405_40516

/-- Represents a convex quadrilateral divided by its diagonals into four triangles -/
structure QuadrilateralWithDiagonals where
  S₁ : ℕ  -- Area of triangle AOB
  S₂ : ℕ  -- Area of triangle BOC
  S₃ : ℕ  -- Area of triangle COD
  S₄ : ℕ  -- Area of triangle DOA

/-- The product of the areas of the four triangles in a quadrilateral divided by its diagonals
    cannot end in 1988 -/
theorem quadrilateral_area_product_not_1988 (q : QuadrilateralWithDiagonals) :
  ∀ (n : ℕ), q.S₁ * q.S₂ * q.S₃ * q.S₄ ≠ 1988 + 10000 * n := by
  sorry

end quadrilateral_area_product_not_1988_l405_40516


namespace chess_tournament_proof_l405_40570

theorem chess_tournament_proof (i g n : ℕ) (I G : ℚ) :
  g = 10 * i →
  n = i + g →
  G = (9/2) * I →
  (n * (n - 1)) / 2 = I + G →
  i = 1 ∧ g = 10 ∧ (n * (n - 1)) / 2 = 55 :=
by sorry

end chess_tournament_proof_l405_40570


namespace money_division_l405_40546

theorem money_division (p q r : ℕ) (total : ℕ) : 
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  r = 400 →
  total = p + q + r →
  total = 1210 := by
sorry

end money_division_l405_40546


namespace equation_solutions_l405_40527

theorem equation_solutions : 
  {x : ℝ | x^4 + (3-x)^4 + x^3 = 82} = {3, -3} := by sorry

end equation_solutions_l405_40527


namespace expression_simplification_l405_40588

theorem expression_simplification (a b : ℚ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) + 4*b^2) / (-2*b) = 14 := by
  sorry

end expression_simplification_l405_40588


namespace jane_age_problem_l405_40508

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

-- Define what it means for a number to be a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem jane_age_problem :
  ∃! x : ℕ, x > 0 ∧ is_perfect_square (x - 1) ∧ is_perfect_cube (x + 1) ∧ x = 26 := by
  sorry

end jane_age_problem_l405_40508


namespace cube_root_of_negative_eight_l405_40595

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by sorry

end cube_root_of_negative_eight_l405_40595


namespace discount_calculation_l405_40565

/-- Calculates the percentage discount given the original price and sale price -/
def percentage_discount (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Proves that the percentage discount for an item with original price $25 and sale price $18.75 is 25% -/
theorem discount_calculation :
  let original_price : ℚ := 25
  let sale_price : ℚ := 37/2  -- Representing 18.75 as a rational number
  percentage_discount original_price sale_price = 25 := by
  sorry


end discount_calculation_l405_40565


namespace quadratic_inequality_solution_range_l405_40578

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ x^2 - 3*x - 2 - a > 0) → a < 2 := by
  sorry

end quadratic_inequality_solution_range_l405_40578


namespace gravel_weight_l405_40587

/-- Proves that the weight of gravel in a cement mixture is 10 pounds given the specified conditions. -/
theorem gravel_weight (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) :
  total_weight = 23.999999999999996 →
  sand_fraction = 1 / 3 →
  water_fraction = 1 / 4 →
  total_weight - (sand_fraction * total_weight + water_fraction * total_weight) = 10 := by
  sorry

end gravel_weight_l405_40587


namespace certain_number_proof_l405_40511

theorem certain_number_proof (x : ℝ) : 
  (x / 3 = 248.14814814814815 / 100 * 162) → x = 1206 := by
  sorry

end certain_number_proof_l405_40511


namespace isabellas_houses_l405_40566

theorem isabellas_houses (green yellow red : ℕ) : 
  green = 3 * yellow →
  yellow = red - 40 →
  green + red = 160 →
  green + red = 160 := by
sorry

end isabellas_houses_l405_40566


namespace integer_difference_l405_40581

theorem integer_difference (S L : ℤ) : 
  S = 10 → 
  S + L = 30 → 
  5 * S > 2 * L → 
  5 * S - 2 * L = 10 := by
sorry

end integer_difference_l405_40581


namespace pencil_cost_l405_40524

theorem pencil_cost (pen_price pencil_price : ℚ) 
  (eq1 : 5 * pen_price + 4 * pencil_price = 310)
  (eq2 : 3 * pen_price + 6 * pencil_price = 238) :
  pencil_price = 130 / 9 := by
  sorry

end pencil_cost_l405_40524


namespace percent_equivalence_l405_40573

theorem percent_equivalence (x : ℝ) (h : 0.3 * 0.05 * x = 18) : 0.05 * 0.3 * x = 18 := by
  sorry

end percent_equivalence_l405_40573


namespace total_video_time_l405_40549

def cat_video_length : ℕ := 4

def dog_video_length (cat : ℕ) : ℕ := 2 * cat

def gorilla_video_length (cat : ℕ) : ℕ := cat ^ 2

def elephant_video_length (cat dog gorilla : ℕ) : ℕ := cat + dog + gorilla

def penguin_video_length (cat dog gorilla elephant : ℕ) : ℕ := (cat + dog + gorilla + elephant) ^ 3

def dolphin_video_length (cat dog gorilla elephant penguin : ℕ) : ℕ :=
  cat + dog + gorilla + elephant + penguin

theorem total_video_time :
  let cat := cat_video_length
  let dog := dog_video_length cat
  let gorilla := gorilla_video_length cat
  let elephant := elephant_video_length cat dog gorilla
  let penguin := penguin_video_length cat dog gorilla elephant
  let dolphin := dolphin_video_length cat dog gorilla elephant penguin
  cat + dog + gorilla + elephant + penguin + dolphin = 351344 := by
  sorry

end total_video_time_l405_40549


namespace speech_competition_score_l405_40567

/-- Calculates the weighted average score for a speech competition --/
def weighted_average (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  (4 * content_score + 4 * delivery_score + 2 * effectiveness_score) / 10

/-- Theorem: The weighted average score for a student with scores 91, 94, and 90 is 92 --/
theorem speech_competition_score : weighted_average 91 94 90 = 92 := by
  sorry

end speech_competition_score_l405_40567


namespace min_n_is_correct_l405_40521

/-- The minimum positive integer n such that the expansion of (x^2 - 1/x^3)^n contains a constant term -/
def min_n : ℕ := 5

/-- The expansion of (x^2 - 1/x^3)^n contains a constant term if and only if
    there exists an r such that 2n - 5r = 0 -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 2 * n = 5 * r

theorem min_n_is_correct :
  (∀ k < min_n, ¬ has_constant_term k) ∧ has_constant_term min_n :=
sorry

end min_n_is_correct_l405_40521


namespace triangle_abc_theorem_l405_40557

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) →
  (a + b = 6) →
  (1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) →
  -- Conclusions
  (C = 2 * π / 3 ∧ c = 2 * Real.sqrt 7) :=
by sorry

end triangle_abc_theorem_l405_40557


namespace amount_after_two_years_l405_40514

/-- The amount after n years with a given initial value and annual increase rate -/
def amountAfterYears (initialValue : ℝ) (increaseRate : ℝ) (years : ℕ) : ℝ :=
  initialValue * (1 + increaseRate) ^ years

/-- Theorem: Given an initial value of 3200 and an annual increase rate of 1/8,
    the value after two years will be 4050 -/
theorem amount_after_two_years :
  amountAfterYears 3200 (1/8) 2 = 4050 := by
  sorry

end amount_after_two_years_l405_40514


namespace least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l405_40506

theorem least_pennies_count (n : ℕ) : n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 → n ≥ 11 :=
by sorry

theorem eleven_satisfies_conditions : 11 % 5 = 1 ∧ 11 % 3 = 2 :=
by sorry

theorem least_pennies_is_eleven : ∃ (n : ℕ), n > 0 ∧ n % 5 = 1 ∧ n % 3 = 2 ∧ ∀ m : ℕ, (m > 0 ∧ m % 5 = 1 ∧ m % 3 = 2) → m ≥ n :=
by sorry

end least_pennies_count_eleven_satisfies_conditions_least_pennies_is_eleven_l405_40506


namespace product_expansion_l405_40501

theorem product_expansion {R : Type*} [CommRing R] (x : R) :
  (3 * x + 4) * (2 * x^2 + x + 6) = 6 * x^3 + 11 * x^2 + 22 * x + 24 := by
  sorry

end product_expansion_l405_40501


namespace star_two_three_l405_40577

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b^3 - b + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 53 := by
  sorry

end star_two_three_l405_40577


namespace number_value_l405_40513

theorem number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 75 := by
  sorry

end number_value_l405_40513


namespace kath_group_admission_cost_l405_40575

/-- Calculates the total admission cost for a group watching a movie before 6 P.M. -/
def total_admission_cost (regular_price : ℕ) (discount : ℕ) (num_people : ℕ) : ℕ :=
  (regular_price - discount) * num_people

/-- The total admission cost for Kath's group is $30 -/
theorem kath_group_admission_cost :
  let regular_price := 8
  let discount := 3
  let num_people := 6
  total_admission_cost regular_price discount num_people = 30 := by
  sorry

end kath_group_admission_cost_l405_40575
