import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_segments_constant_l2157_215774

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- Checks if a point is inside a triangle -/
def isInterior (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Calculates the length of a segment from a vertex to the intersection
    of a parallel line through a point with the opposite side -/
def segmentLength (t : Triangle) (p : Point) (v : Point) : ℝ :=
  sorry

/-- Main theorem -/
theorem sum_of_segments_constant (t : Triangle) (p : Point) :
  isEquilateral t → isInterior p t →
  segmentLength t p t.A + segmentLength t p t.B + segmentLength t p t.C =
  distance t.A t.B :=
by sorry

end NUMINAMATH_CALUDE_sum_of_segments_constant_l2157_215774


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2157_215716

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ (m : ℝ), m = (1 : ℝ) / 2 ∧ ∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x + c := by
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2157_215716


namespace NUMINAMATH_CALUDE_books_ratio_proof_l2157_215721

/-- Proves the ratio of books to read this month to books read last month -/
theorem books_ratio_proof (total : ℕ) (last_month : ℕ) : 
  total = 12 → last_month = 4 → (total - last_month) / last_month = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_ratio_proof_l2157_215721


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l2157_215787

/-- A plane in Euclidean geometry -/
structure EuclideanPlane :=
  (points : Type*)
  (lines : Type*)
  (on_line : points → lines → Prop)

/-- Definition of perpendicular lines in a plane -/
def perpendicular (p : EuclideanPlane) (l1 l2 : p.lines) : Prop :=
  sorry

/-- Statement: In a plane, given a line and a point not on the line,
    there exists a unique line passing through the point
    that is perpendicular to the given line -/
theorem unique_perpendicular_line
  (p : EuclideanPlane) (l : p.lines) (pt : p.points)
  (h : ¬ p.on_line pt l) :
  ∃! l' : p.lines, p.on_line pt l' ∧ perpendicular p l l' :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l2157_215787


namespace NUMINAMATH_CALUDE_student_divisor_problem_l2157_215704

theorem student_divisor_problem (correct_divisor correct_quotient student_quotient : ℕ) 
  (h1 : correct_divisor = 36)
  (h2 : correct_quotient = 42)
  (h3 : student_quotient = 24)
  : ∃ student_divisor : ℕ, 
    student_divisor * student_quotient = correct_divisor * correct_quotient ∧ 
    student_divisor = 63 := by
  sorry

end NUMINAMATH_CALUDE_student_divisor_problem_l2157_215704


namespace NUMINAMATH_CALUDE_condition_property_l2157_215729

theorem condition_property : 
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) ∧ 
  ¬(∀ x : ℝ, x < 2 → x^2 - 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l2157_215729


namespace NUMINAMATH_CALUDE_triangle_with_seven_points_forms_fifteen_triangles_l2157_215706

/-- The number of smaller triangles formed in a triangle with interior points -/
def num_smaller_triangles (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem: A triangle with 7 interior points forms 15 smaller triangles -/
theorem triangle_with_seven_points_forms_fifteen_triangles :
  num_smaller_triangles 7 = 15 := by
  sorry

#eval num_smaller_triangles 7  -- Should output 15

end NUMINAMATH_CALUDE_triangle_with_seven_points_forms_fifteen_triangles_l2157_215706


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l2157_215705

open InnerProductSpace

theorem max_sum_squared_distances {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b c d : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hc : ‖c‖ = 1) (hd : ‖d‖ = 1) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖a - d‖^2 + ‖b - c‖^2 + ‖b - d‖^2 + ‖c - d‖^2 ≤ 16 ∧
  ∃ (a' b' c' d' : V), ‖a'‖ = 1 ∧ ‖b'‖ = 1 ∧ ‖c'‖ = 1 ∧ ‖d'‖ = 1 ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖a' - d'‖^2 + ‖b' - c'‖^2 + ‖b' - d'‖^2 + ‖c' - d'‖^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l2157_215705


namespace NUMINAMATH_CALUDE_intersection_point_l2157_215728

/-- The system of linear equations representing two lines -/
def system (x y : ℝ) : Prop :=
  8 * x + 5 * y = 40 ∧ 3 * x - 10 * y = 15

/-- The theorem stating that (5, 0) is the unique solution to the system -/
theorem intersection_point : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (5, 0) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2157_215728


namespace NUMINAMATH_CALUDE_simplify_fraction_l2157_215767

theorem simplify_fraction (a : ℝ) (h : a = 5) : 15 * a^4 / (75 * a^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2157_215767


namespace NUMINAMATH_CALUDE_twelfth_term_is_12_l2157_215792

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  let a₁ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

/-- The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_12 : arithmetic_sequence 12 = 12 := by
  sorry

#eval arithmetic_sequence 12  -- For verification

end NUMINAMATH_CALUDE_twelfth_term_is_12_l2157_215792


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_specific_roots_and_difference_l2157_215776

/-- Two monic cubic polynomials with specific roots and a constant difference -/
theorem cubic_polynomials_with_specific_roots_and_difference (f g : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = (x - (r + 1)) * (x - (r + 7)) * (x - (3 * r + 8))) →  -- f is monic cubic with roots r+1, r+7, and 3r+8
  (∀ x, g x = (x - (r + 3)) * (x - (r + 9)) * (x - (3 * r + 12))) →  -- g is monic cubic with roots r+3, r+9, and 3r+12
  (∀ x, f x - g x = r) →  -- constant difference between f and g
  r = 32 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_specific_roots_and_difference_l2157_215776


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l2157_215753

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem: Jason gave 9 Pokemon cards to his friends -/
theorem jason_gave_nine_cards : cards_given_to_friends 13 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l2157_215753


namespace NUMINAMATH_CALUDE_intersection_theorem_l2157_215779

/-- A permutation of {1, ..., n} is a bijective function from {1, ..., n} to itself. -/
def Permutation (n : ℕ) := {f : Fin n → Fin n // Function.Bijective f}

/-- Two permutations intersect if they have the same value at some position. -/
def intersect {n : ℕ} (p q : Permutation n) : Prop :=
  ∃ k : Fin n, p.val k = q.val k

/-- There exists a set of 1006 permutations of {1, ..., 2010} such that 
    any permutation of {1, ..., 2010} intersects with at least one of them. -/
theorem intersection_theorem : 
  ∃ (S : Finset (Permutation 2010)), S.card = 1006 ∧ 
    ∀ p : Permutation 2010, ∃ q ∈ S, intersect p q := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2157_215779


namespace NUMINAMATH_CALUDE_sufficient_condition_for_ellipse_l2157_215788

/-- The equation of a potential ellipse -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / m + y^2 / (2*m - 1) = 1

/-- Condition for the equation to represent an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- Theorem stating that m > 1 is a sufficient but not necessary condition for the equation to represent an ellipse -/
theorem sufficient_condition_for_ellipse :
  ∀ m : ℝ, m > 1 → is_ellipse m ∧ ∃ m₀ : ℝ, m₀ ≤ 1 ∧ is_ellipse m₀ :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_ellipse_l2157_215788


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l2157_215791

theorem sqrt_fourth_power_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l2157_215791


namespace NUMINAMATH_CALUDE_least_repeating_digits_eight_elevenths_l2157_215732

/-- The least number of digits in a repeating block of the decimal expansion of 8/11 is 2. -/
theorem least_repeating_digits_eight_elevenths : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (m : ℕ), m < n → ¬ (∃ (k : ℕ+), 8 * (10^m - 1) = 11 * k)) ∧
  (∃ (k : ℕ+), 8 * (10^n - 1) = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_least_repeating_digits_eight_elevenths_l2157_215732


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2157_215768

/-- Calculates the discount percentage on half of the bricks given the total number of bricks,
    full price per brick, and total amount spent. -/
theorem discount_percentage_calculation
  (total_bricks : ℕ)
  (full_price_per_brick : ℚ)
  (total_spent : ℚ)
  (h1 : total_bricks = 1000)
  (h2 : full_price_per_brick = 1/2)
  (h3 : total_spent = 375) :
  let half_bricks := total_bricks / 2
  let full_price_half := half_bricks * full_price_per_brick
  let discounted_price := total_spent - full_price_half
  let discount_amount := full_price_half - discounted_price
  let discount_percentage := (discount_amount / full_price_half) * 100
  discount_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l2157_215768


namespace NUMINAMATH_CALUDE_max_n_value_l2157_215760

theorem max_n_value (A B : ℤ) (h : A * B = 54) : 
  ∃ (n : ℤ), n = 3 * B + A ∧ ∀ (m : ℤ), m = 3 * B + A → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l2157_215760


namespace NUMINAMATH_CALUDE_roots_of_equation_l2157_215775

theorem roots_of_equation (x : ℝ) : 
  (x = 0 ∨ x = -3) ↔ -x * (x + 3) = x * (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2157_215775


namespace NUMINAMATH_CALUDE_mark_change_factor_l2157_215773

/-- Given a class of students, prove that if their marks are changed by a factor
    that doubles the average, then this factor must be 2. -/
theorem mark_change_factor
  (n : ℕ)                    -- number of students
  (initial_avg : ℝ)          -- initial average mark
  (final_avg : ℝ)            -- final average mark
  (h_n : n = 30)             -- there are 30 students
  (h_initial : initial_avg = 45)  -- initial average is 45
  (h_final : final_avg = 90)      -- final average is 90
  : (final_avg / initial_avg : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mark_change_factor_l2157_215773


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2157_215739

def polynomial (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 7*x^2 - 5*x - 30

def divisor (x : ℝ) : ℝ := 2*x - 4

theorem polynomial_division_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2157_215739


namespace NUMINAMATH_CALUDE_sum_reciprocals_l2157_215745

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -2) (hb : b ≠ -2) (hc : c ≠ -2) (hd : d ≠ -2)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω) :
  (1 / (a + 2)) + (1 / (b + 2)) + (1 / (c + 2)) + (1 / (d + 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l2157_215745


namespace NUMINAMATH_CALUDE_negation_of_neither_odd_l2157_215720

theorem negation_of_neither_odd (a b : ℤ) :
  ¬(¬(Odd a) ∧ ¬(Odd b)) ↔ Odd a ∨ Odd b := by sorry

end NUMINAMATH_CALUDE_negation_of_neither_odd_l2157_215720


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2157_215749

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C*(x + y + z)) ↔ C ≤ 2 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2157_215749


namespace NUMINAMATH_CALUDE_correct_num_children_l2157_215797

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := 16

/-- The number of children -/
def num_children : ℕ := total_pencils / pencils_per_child

theorem correct_num_children : num_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_children_l2157_215797


namespace NUMINAMATH_CALUDE_saturday_price_of_200_dollar_coat_l2157_215795

/-- Calculates the Saturday price of a coat at Ajax Outlet Store -/
def saturday_price (original_price : ℝ) : ℝ :=
  let regular_discount_rate : ℝ := 0.6
  let saturday_discount_rate : ℝ := 0.3
  let price_after_regular_discount := original_price * (1 - regular_discount_rate)
  price_after_regular_discount * (1 - saturday_discount_rate)

/-- Theorem stating that the Saturday price of a $200 coat is $56 -/
theorem saturday_price_of_200_dollar_coat :
  saturday_price 200 = 56 := by
  sorry

end NUMINAMATH_CALUDE_saturday_price_of_200_dollar_coat_l2157_215795


namespace NUMINAMATH_CALUDE_cookies_left_l2157_215712

def initial_cookies : ℕ := 32
def eaten_cookies : ℕ := 9

theorem cookies_left : initial_cookies - eaten_cookies = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l2157_215712


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l2157_215793

theorem a_gt_abs_b_sufficient_not_necessary :
  (∃ a b : ℝ, a > |b| ∧ a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > |b|)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l2157_215793


namespace NUMINAMATH_CALUDE_class_representatives_count_l2157_215766

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 3

/-- Represents the number of subjects needing representatives -/
def num_subjects : ℕ := 5

/-- Calculates the number of ways to select representatives with fewer girls than boys -/
def count_fewer_girls : ℕ := sorry

/-- Calculates the number of ways to select representatives with Boy A as a representative but not for mathematics -/
def count_boy_a_not_math : ℕ := sorry

/-- Calculates the number of ways to select representatives with Girl B for Chinese and Boy A as a representative but not for mathematics -/
def count_girl_b_chinese_boy_a_not_math : ℕ := sorry

/-- Theorem stating the correct number of ways for each condition -/
theorem class_representatives_count :
  count_fewer_girls = 5520 ∧
  count_boy_a_not_math = 3360 ∧
  count_girl_b_chinese_boy_a_not_math = 360 := by sorry

end NUMINAMATH_CALUDE_class_representatives_count_l2157_215766


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sixth_root_64_l2157_215748

theorem cube_root_27_times_fourth_root_81_times_sixth_root_64 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^6 = 64 ∧ a * b * c = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sixth_root_64_l2157_215748


namespace NUMINAMATH_CALUDE_triangle_existence_l2157_215714

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l2157_215714


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2157_215742

theorem triangle_trigonometric_identities 
  (a b c : ℝ) (α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_sum_angles : α + β + γ = π)
  (h_law_of_sines : a / Real.sin α = b / Real.sin β ∧ b / Real.sin β = c / Real.sin γ) :
  (a + b) / c = Real.cos ((α - β) / 2) / Real.sin (γ / 2) ∧
  (a - b) / c = Real.sin ((α - β) / 2) / Real.cos (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2157_215742


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_half_unit_l2157_215796

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral is 0.5 square units -/
theorem quadrilateral_area_is_half_unit : 
  let l1 : Line := { a := 3, b := 4, c := -12 }
  let l2 : Line := { a := 6, b := -4, c := -12 }
  let l3 : Line := { a := 1, b := 0, c := -3 }
  let l4 : Line := { a := 0, b := 1, c := -1 }
  let p1 := intersectionPoint l1 l2
  let p2 := intersectionPoint l1 l3
  let p3 := intersectionPoint l2 l3
  let p4 := intersectionPoint l1 l4
  quadrilateralArea p1 p2 p3 p4 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_half_unit_l2157_215796


namespace NUMINAMATH_CALUDE_max_salary_basketball_team_l2157_215780

/-- Represents the maximum possible salary for a single player in a basketball team. -/
def maxSalary (numPlayers : ℕ) (minSalary : ℕ) (totalSalaryCap : ℕ) : ℕ :=
  totalSalaryCap - (numPlayers - 1) * minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the team composition and salary constraints. -/
theorem max_salary_basketball_team :
  maxSalary 12 20000 500000 = 280000 := by
  sorry

#eval maxSalary 12 20000 500000

end NUMINAMATH_CALUDE_max_salary_basketball_team_l2157_215780


namespace NUMINAMATH_CALUDE_variance_of_linear_transform_l2157_215740

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- A linear transformation of a random variable -/
structure LinearTransform (X : BinomialRV) where
  a : ℝ
  b : ℝ
  Y : ℝ := a * X.n + b

theorem variance_of_linear_transform (X : BinomialRV) (Y : LinearTransform X) :
  X.n = 5 ∧ X.p = 1/4 ∧ Y.a = 4 ∧ Y.b = -3 →
  Y.a^2 * variance X = 15 :=
sorry

end NUMINAMATH_CALUDE_variance_of_linear_transform_l2157_215740


namespace NUMINAMATH_CALUDE_outfits_count_l2157_215743

/-- The number of different outfits that can be created with given clothing items. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) (blazers : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (blazers + 1)

/-- Theorem stating the number of outfits with specific clothing items. -/
theorem outfits_count : number_of_outfits 5 4 5 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2157_215743


namespace NUMINAMATH_CALUDE_imaginary_unit_sixth_power_l2157_215752

theorem imaginary_unit_sixth_power (i : ℂ) (hi : i * i = -1) : i^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sixth_power_l2157_215752


namespace NUMINAMATH_CALUDE_blood_donation_selection_count_l2157_215794

def male_teachers : ℕ := 3
def female_teachers : ℕ := 6
def total_teachers : ℕ := male_teachers + female_teachers
def selection_size : ℕ := 5

theorem blood_donation_selection_count :
  (Nat.choose total_teachers selection_size) - (Nat.choose female_teachers selection_size) = 120 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_count_l2157_215794


namespace NUMINAMATH_CALUDE_unique_base_representation_l2157_215717

theorem unique_base_representation :
  ∃! (x y z b : ℕ), 
    1987 = x * b^2 + y * b + z ∧
    b > 1 ∧
    x < b ∧ y < b ∧ z < b ∧
    x + y + z = 25 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l2157_215717


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l2157_215725

theorem geometric_arithmetic_sequence_sum (x y : ℝ) :
  0 < x ∧ 0 < y ∧
  (1 : ℝ) * x = x * y ∧  -- Geometric sequence condition
  y - x = 3 - y →        -- Arithmetic sequence condition
  x + y = 15/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l2157_215725


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2157_215769

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- Ensuring c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- Representing the repeating decimal
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2157_215769


namespace NUMINAMATH_CALUDE_min_distance_ABCD_l2157_215765

/-- Given four points A, B, C, and D on a line, with AB = 12, BC = 6, and CD = 5,
    the minimum possible distance between A and D is 1. -/
theorem min_distance_ABCD (A B C D : ℝ) : 
  abs (B - A) = 12 →
  abs (C - B) = 6 →
  abs (D - C) = 5 →
  ∃ (A' B' C' D' : ℝ), 
    abs (B' - A') = 12 ∧
    abs (C' - B') = 6 ∧
    abs (D' - C') = 5 ∧
    abs (D' - A') = 1 ∧
    ∀ (A'' B'' C'' D'' : ℝ),
      abs (B'' - A'') = 12 →
      abs (C'' - B'') = 6 →
      abs (D'' - C'') = 5 →
      abs (D'' - A'') ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_ABCD_l2157_215765


namespace NUMINAMATH_CALUDE_min_distance_point_to_y_axis_l2157_215709

/-- Given point A (-3, -2) and point B on the y-axis, the distance between A and B is minimized when B has coordinates (0, -2) -/
theorem min_distance_point_to_y_axis (A B : ℝ × ℝ) :
  A = (-3, -2) →
  B.1 = 0 →
  (∀ C : ℝ × ℝ, C.1 = 0 → dist A B ≤ dist A C) →
  B = (0, -2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_y_axis_l2157_215709


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l2157_215734

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (p q : ℝ) : ℕ → ℝ
  | 0 => p
  | 1 => 9
  | 2 => 3*p - q + 7
  | 3 => 3*p + q + 2
  | n + 4 => ArithmeticSequence p q 3 + (n + 1) * (ArithmeticSequence p q 3 - ArithmeticSequence p q 2)

/-- Theorem stating that the 1001st term of the sequence is 5004 -/
theorem arithmetic_sequence_1001st_term (p q : ℝ) :
  ArithmeticSequence p q 1000 = 5004 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l2157_215734


namespace NUMINAMATH_CALUDE_complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l2157_215757

-- (1) (3+2i)+(\sqrt{3}-2)i
theorem complex_addition : ℂ → Prop :=
  fun z ↦ (3 + 2*Complex.I) + (Real.sqrt 3 - 2)*Complex.I = 3 + Real.sqrt 3 * Complex.I

-- (2) (9+2i)/(2+i)
theorem complex_division : ℂ → Prop :=
  fun z ↦ (9 + 2*Complex.I) / (2 + Complex.I) = 4 - Complex.I

-- (3) ((-1+i)(2+i))/(i^3)
theorem complex_multiplication_division : ℂ → Prop :=
  fun z ↦ ((-1 + Complex.I) * (2 + Complex.I)) / (Complex.I^3) = -1 - 3*Complex.I

-- (4) Given vectors a⃗=(-1,2) and b⃗=(2,1), calculate 2a⃗+3b⃗ and a⃗•b⃗
theorem vector_operations (a b : ℝ × ℝ) : Prop :=
  let a := (-1, 2)
  let b := (2, 1)
  (2 • a + 3 • b = (4, 7)) ∧ (a.1 * b.1 + a.2 * b.2 = 0)

-- (5) Given vectors a⃗ and b⃗ satisfy |a⃗|=1 and a⃗•b⃗=-1, calculate a⃗•(2a⃗-b⃗)
theorem vector_dot_product (a b : ℝ × ℝ) : Prop :=
  (a.1^2 + a.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  a.1 * (2*a.1 - b.1) + a.2 * (2*a.2 - b.2) = 3

-- Proofs are omitted
theorem all_parts_combined : Prop :=
  complex_addition 0 ∧
  complex_division 0 ∧
  complex_multiplication_division 0 ∧
  vector_operations (0, 0) (0, 0) ∧
  vector_dot_product (0, 0) (0, 0)

-- Add sorry to skip the proof
theorem all_parts_combined_proof : all_parts_combined := by sorry

end NUMINAMATH_CALUDE_complex_addition_complex_division_complex_multiplication_division_vector_operations_vector_dot_product_all_parts_combined_all_parts_combined_proof_l2157_215757


namespace NUMINAMATH_CALUDE_lily_account_balance_l2157_215763

def initial_amount : ℕ := 55
def shirt_cost : ℕ := 7

theorem lily_account_balance :
  initial_amount - (shirt_cost + 3 * shirt_cost) = 27 :=
by sorry

end NUMINAMATH_CALUDE_lily_account_balance_l2157_215763


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l2157_215747

/-- The number of chairs in the row -/
def n : ℕ := 12

/-- The probability that Mary and James don't sit next to each other -/
def prob_not_adjacent : ℚ := 5/6

/-- The theorem stating the probability of Mary and James not sitting next to each other -/
theorem probability_not_adjacent :
  (1 - (n - 1 : ℚ) / (n.choose 2 : ℚ)) = prob_not_adjacent :=
sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l2157_215747


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l2157_215741

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ 72542 ≡ n [ZMOD 25] ∧ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l2157_215741


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2157_215746

theorem divisibility_theorem (m n : ℕ+) (h : 5 ∣ (2^n.val + 3^m.val)) :
  5 ∣ (2^m.val + 3^n.val) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2157_215746


namespace NUMINAMATH_CALUDE_triangle_area_angle_l2157_215758

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / 4
  S = (1/2) * a * b * Real.sin (π/4) →
  ∃ A B C : ℝ,
    A + B + C = π ∧
    a = BC ∧ b = AC ∧ c = AB ∧
    C = π/4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l2157_215758


namespace NUMINAMATH_CALUDE_square_side_length_l2157_215786

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 49) (h2 : side^2 = area) : side = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2157_215786


namespace NUMINAMATH_CALUDE_work_done_is_four_l2157_215726

-- Define the force vector
def F : Fin 2 → ℝ := ![2, 3]

-- Define points A and B
def A : Fin 2 → ℝ := ![2, 0]
def B : Fin 2 → ℝ := ![4, 0]

-- Define the displacement vector
def displacement : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

-- Define work as the dot product of force and displacement
def work : ℝ := (F 0 * displacement 0) + (F 1 * displacement 1)

-- Theorem statement
theorem work_done_is_four : work = 4 := by sorry

end NUMINAMATH_CALUDE_work_done_is_four_l2157_215726


namespace NUMINAMATH_CALUDE_parabola_vertices_distance_l2157_215713

/-- The equation of the parabolas -/
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 4

/-- The y-coordinate of the vertex for the upper parabola (y ≥ 2) -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the vertex for the lower parabola (y < 2) -/
def lower_vertex_y : ℝ := -1

/-- The distance between the vertices of the parabolas -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem parabola_vertices_distance :
  vertex_distance = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertices_distance_l2157_215713


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2157_215708

theorem regular_polygon_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 108) (h2 : side_length = 12) :
  (perimeter / side_length : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2157_215708


namespace NUMINAMATH_CALUDE_volume_removed_percent_l2157_215771

def box_length : ℝ := 15
def box_width : ℝ := 10
def box_height : ℝ := 8
def cube_side : ℝ := 3
def num_corners : ℕ := 8

def box_volume : ℝ := box_length * box_width * box_height
def removed_cube_volume : ℝ := cube_side ^ 3
def total_removed_volume : ℝ := num_corners * removed_cube_volume

theorem volume_removed_percent :
  (total_removed_volume / box_volume) * 100 = 18 := by sorry

end NUMINAMATH_CALUDE_volume_removed_percent_l2157_215771


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l2157_215722

/-- Represents the daily sales volume as a function of the selling price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 740

/-- Represents the daily profit as a function of the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The cost price of each book -/
def cost_price : ℝ := 40

/-- The minimum selling price -/
def min_price : ℝ := 44

/-- The maximum selling price based on the profit margin constraint -/
def max_price : ℝ := 52

theorem optimal_price_and_profit :
  ∀ x : ℝ, min_price ≤ x ∧ x ≤ max_price →
  daily_profit x ≤ daily_profit max_price ∧
  daily_profit max_price = 2640 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l2157_215722


namespace NUMINAMATH_CALUDE_sailboat_speed_at_max_power_l2157_215718

/-- The speed of a sailboat when the wind power is maximized -/
theorem sailboat_speed_at_max_power 
  (C S ρ : ℝ) 
  (v₀ : ℝ) 
  (h_positive : C > 0 ∧ S > 0 ∧ ρ > 0 ∧ v₀ > 0) :
  ∃ (v : ℝ), 
    v = v₀ / 3 ∧ 
    (∀ (u : ℝ), 
      u * (C * S * ρ * (v₀ - u)^2) / 2 ≤ v * (C * S * ρ * (v₀ - v)^2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sailboat_speed_at_max_power_l2157_215718


namespace NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l2157_215781

def is_lovely (n : ℕ+) : Prop :=
  ∃ (k : ℕ+) (d : Fin k → ℕ+),
    n = (Finset.range k).prod (λ i => d i) ∧
    ∀ i : Fin k, (d i)^2 ∣ (n + d i)

theorem infinitely_many_lovely_numbers :
  ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ is_lovely n :=
sorry

theorem no_lovely_square_greater_than_one :
  ¬∃ m : ℕ+, m > 1 ∧ is_lovely (m^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l2157_215781


namespace NUMINAMATH_CALUDE_journey_time_calculation_l2157_215798

/-- Given a journey with the following conditions:
    - Total distance is 224 km
    - Journey is divided into two equal halves
    - First half is traveled at 21 km/hr
    - Second half is traveled at 24 km/hr
    The total time taken to complete the journey is 10 hours. -/
theorem journey_time_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 224 →
  first_half_speed = 21 →
  second_half_speed = 24 →
  (total_distance / 2 / first_half_speed) + (total_distance / 2 / second_half_speed) = 10 := by
sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l2157_215798


namespace NUMINAMATH_CALUDE_f_negative_before_root_l2157_215790

-- Define the function f(x) = 2^x + log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- State the theorem
theorem f_negative_before_root (a : ℝ) (h1 : f a = 0) (x : ℝ) (h2 : 0 < x) (h3 : x < a) :
  f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_before_root_l2157_215790


namespace NUMINAMATH_CALUDE_initial_distance_proof_l2157_215777

/-- The initial distance between Tim and Élan -/
def initial_distance : ℝ := 30

/-- Tim's initial speed in mph -/
def tim_speed : ℝ := 10

/-- Élan's initial speed in mph -/
def elan_speed : ℝ := 5

/-- The distance Tim travels until meeting Élan -/
def tim_distance : ℝ := 20

/-- The time it takes for Tim and Élan to meet -/
def meeting_time : ℝ := 1.5

theorem initial_distance_proof :
  initial_distance = 
    tim_speed * 1 + 
    elan_speed * 1 + 
    (tim_speed * 2) * (meeting_time - 1) + 
    (elan_speed * 2) * (meeting_time - 1) :=
sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l2157_215777


namespace NUMINAMATH_CALUDE_problem_statement_l2157_215702

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a^2 / (b - c) + b^2 / (c - a) + c^2 / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2157_215702


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l2157_215731

theorem integer_solutions_of_system (x y z t : ℤ) : 
  (x * z - 2 * y * t = 3 ∧ x * t + y * z = 1) ↔ 
  ((x = 1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = 1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = 3 ∧ t = -1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = 1) ∨
   (x = -1 ∧ y = 0 ∧ z = -3 ∧ t = -1) ∨
   (x = 3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = 3 ∧ y = -1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = 1 ∧ z = -1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = 1 ∧ t = 0) ∨
   (x = -3 ∧ y = -1 ∧ z = -1 ∧ t = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l2157_215731


namespace NUMINAMATH_CALUDE_fraction_equality_implies_zero_l2157_215724

theorem fraction_equality_implies_zero (x : ℝ) :
  (1 / (x - 1) = 2 / (x - 2)) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_zero_l2157_215724


namespace NUMINAMATH_CALUDE_multiples_of_seven_l2157_215756

theorem multiples_of_seven (a b : ℕ) (q : Finset ℕ) : 
  (∃ k₁ k₂ : ℕ, a = 14 * k₁ ∧ b = 14 * k₂) →  -- a and b are multiples of 14
  (∀ x ∈ q, a ≤ x ∧ x ≤ b) →  -- q is the set of consecutive integers between a and b, inclusive
  (∀ x ∈ q, x + 1 ∈ q ∨ x = b) →  -- q contains consecutive integers
  (q.filter (λ x => x % 14 = 0)).card = 14 →  -- q contains 14 multiples of 14
  (q.filter (λ x => x % 7 = 0)).card = 27 :=  -- The number of multiples of 7 in q is 27
by sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l2157_215756


namespace NUMINAMATH_CALUDE_thirtieth_day_production_l2157_215750

/-- Represents the daily cloth production in feet -/
def cloth_sequence (n : ℕ) : ℚ :=
  5 + (n - 1) * ((390 - 30 * 5) / (30 * 29 / 2))

/-- The sum of the cloth_sequence for the first 30 days -/
def total_cloth : ℚ := 390

/-- The theorem states that the 30th term of the cloth_sequence is 21 -/
theorem thirtieth_day_production : cloth_sequence 30 = 21 := by sorry

end NUMINAMATH_CALUDE_thirtieth_day_production_l2157_215750


namespace NUMINAMATH_CALUDE_fabric_cost_theorem_l2157_215772

/-- Represents the cost in livres, sous, and deniers -/
structure Cost :=
  (livres : ℕ)
  (sous : ℕ)
  (deniers : ℚ)

/-- Converts a Cost to deniers -/
def cost_to_deniers (c : Cost) : ℚ :=
  c.livres * 20 * 12 + c.sous * 12 + c.deniers

/-- Converts deniers to a Cost -/
def deniers_to_cost (d : ℚ) : Cost :=
  let total_sous := d / 12
  let livres := (total_sous / 20).floor
  let remaining_sous := total_sous - livres * 20
  { livres := livres.toNat,
    sous := remaining_sous.floor.toNat,
    deniers := d - (livres * 20 * 12 + remaining_sous.floor * 12) }

def ell_cost : Cost := { livres := 42, sous := 17, deniers := 11 }

def fabric_length : ℚ := 15 + 13 / 16

theorem fabric_cost_theorem :
  deniers_to_cost (cost_to_deniers ell_cost * fabric_length) =
  { livres := 682, sous := 15, deniers := 9 + 11 / 16 } := by
  sorry

end NUMINAMATH_CALUDE_fabric_cost_theorem_l2157_215772


namespace NUMINAMATH_CALUDE_election_result_l2157_215751

theorem election_result (total_votes : ℕ) (majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  majority = 320 →
  winning_percentage = 70 →
  (winning_percentage / 100) * total_votes - ((100 - winning_percentage) / 100) * total_votes = majority :=
by sorry

end NUMINAMATH_CALUDE_election_result_l2157_215751


namespace NUMINAMATH_CALUDE_ratio_of_sums_l2157_215799

/-- Represents an arithmetic progression --/
structure ArithmeticProgression where
  firstTerm : ℕ
  difference : ℕ
  length : ℕ

/-- Calculates the sum of an arithmetic progression --/
def sumOfArithmeticProgression (ap : ArithmeticProgression) : ℕ :=
  ap.length * (2 * ap.firstTerm + (ap.length - 1) * ap.difference) / 2

/-- Generates a list of arithmetic progressions for the first group --/
def firstGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * (i + 1)) 10)

/-- Generates a list of arithmetic progressions for the second group --/
def secondGroup : List ArithmeticProgression :=
  List.range 15
    |> List.map (fun i => ArithmeticProgression.mk (i + 1) (2 * i + 1) 10)

/-- Calculates the sum of all elements in a group of arithmetic progressions --/
def sumOfGroup (group : List ArithmeticProgression) : ℕ :=
  group.map sumOfArithmeticProgression |> List.sum

theorem ratio_of_sums : 
  (sumOfGroup firstGroup : ℚ) / (sumOfGroup secondGroup : ℚ) = 160 / 151 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l2157_215799


namespace NUMINAMATH_CALUDE_tangent_slope_tan_at_pi_over_four_l2157_215762

theorem tangent_slope_tan_at_pi_over_four :
  let f : ℝ → ℝ := λ x ↦ Real.tan x
  let x₀ : ℝ := π / 4
  (deriv f) x₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_tan_at_pi_over_four_l2157_215762


namespace NUMINAMATH_CALUDE_positive_solution_condition_l2157_215715

theorem positive_solution_condition (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
    x₁ - x₂ = a ∧ x₃ - x₄ = b ∧ x₁ + x₂ + x₃ + x₄ = 1) ↔
  abs a + abs b < 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_solution_condition_l2157_215715


namespace NUMINAMATH_CALUDE_equal_product_sequence_characterization_l2157_215764

def is_equal_product_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k

theorem equal_product_sequence_characterization (a : ℕ → ℝ) :
  is_equal_product_sequence a ↔
    ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k :=
by sorry

end NUMINAMATH_CALUDE_equal_product_sequence_characterization_l2157_215764


namespace NUMINAMATH_CALUDE_expression_evaluation_l2157_215784

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2157_215784


namespace NUMINAMATH_CALUDE_probability_at_least_three_white_balls_l2157_215770

theorem probability_at_least_three_white_balls 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 8)
  (h3 : black_balls = 7)
  (h4 : drawn_balls = 5) :
  let favorable_outcomes := Nat.choose white_balls 3 * Nat.choose black_balls 2 +
                            Nat.choose white_balls 4 * Nat.choose black_balls 1 +
                            Nat.choose white_balls 5 * Nat.choose black_balls 0
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 1722 / 3003 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_three_white_balls_l2157_215770


namespace NUMINAMATH_CALUDE_even_function_property_l2157_215785

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f (2 + x) = -f (2 - x)) :
  f 2010 = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_property_l2157_215785


namespace NUMINAMATH_CALUDE_simple_interest_rate_example_l2157_215755

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

theorem simple_interest_rate_example :
  simple_interest_rate 750 900 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_example_l2157_215755


namespace NUMINAMATH_CALUDE_product_of_1001_2_and_121_3_l2157_215733

/-- Converts a number from base 2 to base 10 -/
def base2To10 (n : List Bool) : Nat :=
  n.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a number from base 3 to base 10 -/
def base3To10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The problem statement -/
theorem product_of_1001_2_and_121_3 :
  let n1 := base2To10 [true, false, false, true]
  let n2 := base3To10 [1, 2, 1]
  n1 * n2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1001_2_and_121_3_l2157_215733


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l2157_215737

/-- A quadratic function f(x) = x^2 + 4x + c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) : ℝ → ℝ := fun x ↦ f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_exactly_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
    (g x = 0 ∧ g y = 0 ∧ g z = 0) ∧
    (∀ w : ℝ, g w = 0 → w = x ∨ w = y ∨ w = z)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, has_exactly_three_distinct_real_roots (f_comp c) ↔ c = 1 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l2157_215737


namespace NUMINAMATH_CALUDE_all_positive_integers_are_dapper_l2157_215707

/-- A positive integer is dapper if at least one of its multiples begins with 2008. -/
def is_dapper (n : ℕ+) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℕ), k * n.val = 2008 * 10^m + m ∧ m < 10^m

/-- Every positive integer is dapper. -/
theorem all_positive_integers_are_dapper : ∀ (n : ℕ+), is_dapper n := by
  sorry

end NUMINAMATH_CALUDE_all_positive_integers_are_dapper_l2157_215707


namespace NUMINAMATH_CALUDE_solution_range_l2157_215782

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (2 * x + m) / (x - 1) = 1) → 
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l2157_215782


namespace NUMINAMATH_CALUDE_vector_dot_product_equality_l2157_215701

/-- Given vectors a and b in ℝ², and a scalar t, prove that if the dot product of a and c
    is equal to the dot product of b and c, where c = a + t*b, then t = 13/2. -/
theorem vector_dot_product_equality (a b : ℝ × ℝ) (t : ℝ) :
  a = (5, 12) →
  b = (2, 0) →
  let c := a + t • b
  (a.1 * c.1 + a.2 * c.2) = (b.1 * c.1 + b.2 * c.2) →
  t = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_equality_l2157_215701


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2157_215789

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2157_215789


namespace NUMINAMATH_CALUDE_coefficient_x4_in_product_l2157_215736

/-- The coefficient of x^4 in the expansion of (2x^3 + 5x^2 - 3x)(3x^3 - 8x^2 + 6x - 9) is -37 -/
theorem coefficient_x4_in_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 5 * X^2 - 3 * X
  let p₂ : Polynomial ℤ := 3 * X^3 - 8 * X^2 + 6 * X - 9
  (p₁ * p₂).coeff 4 = -37 := by
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_product_l2157_215736


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l2157_215778

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.3 * P) 
  (hN : N = 0.6 * (2 * P)) : 
  M / N = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l2157_215778


namespace NUMINAMATH_CALUDE_sin_cos_difference_77_47_l2157_215723

theorem sin_cos_difference_77_47 :
  Real.sin (77 * π / 180) * Real.cos (47 * π / 180) -
  Real.cos (77 * π / 180) * Real.sin (47 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_77_47_l2157_215723


namespace NUMINAMATH_CALUDE_remainder_2356912_div_8_l2157_215700

theorem remainder_2356912_div_8 : 2356912 % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2356912_div_8_l2157_215700


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l2157_215710

theorem cube_root_of_negative_eight_squared (x : ℝ) : x^3 = (-8)^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l2157_215710


namespace NUMINAMATH_CALUDE_salary_of_C_salary_C_is_11000_l2157_215730

-- Define the salaries as natural numbers (assuming whole rupees)
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

-- Define the average salary
def average_salary : ℕ := 8000

-- Theorem to prove
theorem salary_of_C : ℕ :=
  let total_salary := salary_A + salary_B + salary_D + salary_E
  let salary_C := 5 * average_salary - total_salary
  salary_C

-- Proof (skipped)
theorem salary_C_is_11000 : salary_of_C = 11000 := by
  sorry

end NUMINAMATH_CALUDE_salary_of_C_salary_C_is_11000_l2157_215730


namespace NUMINAMATH_CALUDE_electricity_cost_per_watt_l2157_215727

theorem electricity_cost_per_watt 
  (watts : ℕ) 
  (late_fee : ℕ) 
  (total_payment : ℕ) 
  (h1 : watts = 300)
  (h2 : late_fee = 150)
  (h3 : total_payment = 1350) :
  (total_payment - late_fee) / watts = 4 := by
  sorry

end NUMINAMATH_CALUDE_electricity_cost_per_watt_l2157_215727


namespace NUMINAMATH_CALUDE_two_digit_subtraction_pattern_l2157_215783

theorem two_digit_subtraction_pattern (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_pattern_l2157_215783


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_10_l2157_215719

theorem binomial_coefficient_19_10 (h1 : Nat.choose 17 7 = 19448) (h2 : Nat.choose 17 9 = 24310) :
  Nat.choose 19 10 = 92378 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_10_l2157_215719


namespace NUMINAMATH_CALUDE_mary_breeding_balls_l2157_215744

theorem mary_breeding_balls (snakes_per_ball : ℕ) (additional_pairs : ℕ) (total_snakes : ℕ) 
  (h1 : snakes_per_ball = 8)
  (h2 : additional_pairs = 6)
  (h3 : total_snakes = 36) :
  ∃ (num_balls : ℕ), 
    num_balls * snakes_per_ball + additional_pairs * 2 = total_snakes ∧ 
    num_balls = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_breeding_balls_l2157_215744


namespace NUMINAMATH_CALUDE_height_difference_calculation_l2157_215759

/-- The combined height difference between an uncle and his two relatives -/
def combined_height_difference (uncle_height james_initial_ratio growth_spurt younger_sibling_height : ℝ) : ℝ :=
  let james_new_height := uncle_height * james_initial_ratio + growth_spurt
  let diff_uncle_james := uncle_height - james_new_height
  let diff_uncle_younger := uncle_height - younger_sibling_height
  diff_uncle_james + diff_uncle_younger

/-- Theorem stating the combined height difference given specific measurements -/
theorem height_difference_calculation :
  combined_height_difference 72 (2/3) 10 38 = 48 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_calculation_l2157_215759


namespace NUMINAMATH_CALUDE_grocer_bananas_theorem_l2157_215761

/-- Represents the number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- Represents the purchase price in dollars per 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- Represents the selling price in dollars per 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- Represents the total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased by the grocer is 96 -/
theorem grocer_bananas_theorem :
  bananas_purchased = 96 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
sorry

end NUMINAMATH_CALUDE_grocer_bananas_theorem_l2157_215761


namespace NUMINAMATH_CALUDE_unique_sequence_solution_l2157_215754

/-- Represents a solution to the sequence problem -/
structure SequenceSolution where
  n : ℕ
  q : ℚ
  d : ℚ

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (sol : SequenceSolution) : Prop :=
  sol.n > 1 ∧
  1 + (sol.n - 1) * sol.d = 81 ∧
  1 * sol.q^(sol.n - 1) = 81 ∧
  sol.q / sol.d = 0.15

/-- The unique solution to the sequence problem -/
def unique_solution : SequenceSolution :=
  { n := 5, q := 3, d := 20 }

/-- Theorem stating that the unique_solution is the only valid solution -/
theorem unique_sequence_solution :
  is_valid_solution unique_solution ∧
  ∀ (sol : SequenceSolution), is_valid_solution sol → sol = unique_solution :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_solution_l2157_215754


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2157_215703

theorem imaginary_part_of_z_times_i :
  let z : ℂ := -1 + 2 * I
  Complex.im (z * I) = -1 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l2157_215703


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l2157_215738

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ (r₁ r₂ : ℝ), r₁ > r₂ ∧ 
    2 * r₁^2 + 5 * r₁ = k ∧
    2 * r₂^2 + 5 * r₂ = k ∧
    r₁ - r₂ = 5.5) →
  k = -28.875 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l2157_215738


namespace NUMINAMATH_CALUDE_assignment_increases_by_one_l2157_215711

-- Define the assignment operation
def assign (x : ℕ) : ℕ := x + 1

-- Theorem stating that the assignment n = n + 1 increases n by 1
theorem assignment_increases_by_one (n : ℕ) : assign n = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_assignment_increases_by_one_l2157_215711


namespace NUMINAMATH_CALUDE_a_equals_three_iff_parallel_not_coincident_l2157_215735

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 3 * a = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ 3 * x + (a - 1) * y + 7 - a = 0

/-- Condition for two lines to be parallel and not coincident -/
def parallel_not_coincident (l : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), l.line1 (x, y) ↔ l.line2 (k * x + l.a, k * y + 2))

/-- The main theorem -/
theorem a_equals_three_iff_parallel_not_coincident (l : TwoLines) :
  l.a = 3 ↔ parallel_not_coincident l :=
sorry

end NUMINAMATH_CALUDE_a_equals_three_iff_parallel_not_coincident_l2157_215735
