import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2906_290637

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2906_290637


namespace NUMINAMATH_CALUDE_video_voting_result_l2906_290611

/-- Represents the voting system for a video --/
structure VideoVoting where
  totalVotes : ℕ
  likePercentage : ℚ
  finalScore : ℤ

/-- Theorem stating the conditions and the result to be proved --/
theorem video_voting_result (v : VideoVoting) 
  (h1 : v.likePercentage = 3/4)
  (h2 : v.finalScore = 140) :
  v.totalVotes = 280 := by
  sorry

end NUMINAMATH_CALUDE_video_voting_result_l2906_290611


namespace NUMINAMATH_CALUDE_girls_percentage_approx_l2906_290627

/-- The percentage of girls in a school, given the total number of students and the number of boys -/
def percentage_of_girls (total : ℕ) (boys : ℕ) : ℚ :=
  ((total - boys : ℚ) / total) * 100

/-- Theorem stating that the percentage of girls is approximately 91.91% -/
theorem girls_percentage_approx (total : ℕ) (boys : ℕ) 
  (h1 : total = 1150) (h2 : boys = 92) : 
  91.9 < percentage_of_girls total boys ∧ percentage_of_girls total boys < 91.92 := by
  sorry

end NUMINAMATH_CALUDE_girls_percentage_approx_l2906_290627


namespace NUMINAMATH_CALUDE_expand_expression_l2906_290683

theorem expand_expression (x : ℝ) : (7 * x^2 - 3) * 5 * x^3 = 35 * x^5 - 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2906_290683


namespace NUMINAMATH_CALUDE_constant_q_value_l2906_290687

theorem constant_q_value (p q : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + p) = x^2 + q*x + 12) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_q_value_l2906_290687


namespace NUMINAMATH_CALUDE_alex_more_pens_than_jane_l2906_290616

def alex_pens (week : Nat) : Nat :=
  4 * 2^(week - 1)

def jane_pens : Nat := 16

theorem alex_more_pens_than_jane :
  alex_pens 4 - jane_pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_alex_more_pens_than_jane_l2906_290616


namespace NUMINAMATH_CALUDE_f_properties_l2906_290654

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / 4^x - 1 / 2^x else 2^x - 4^x

theorem f_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 0, f x = 1 / 4^x - 1 / 2^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x = 2^x - 4^x) ∧
  (∀ x ∈ Set.Icc 0 1, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc 0 1, f x = 0) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l2906_290654


namespace NUMINAMATH_CALUDE_gcf_40_48_l2906_290674

theorem gcf_40_48 : Nat.gcd 40 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_40_48_l2906_290674


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l2906_290620

/-- The probability of a specific arrangement of lava lamps -/
theorem lava_lamp_probability :
  let total_lamps : ℕ := 6
  let red_lamps : ℕ := 3
  let blue_lamps : ℕ := 3
  let lamps_on : ℕ := 3
  let color_arrangements := Nat.choose total_lamps red_lamps
  let on_arrangements := Nat.choose total_lamps lamps_on
  let remaining_lamps : ℕ := 4
  let remaining_red : ℕ := 2
  let remaining_color_arrangements := Nat.choose remaining_lamps remaining_red
  let remaining_on_arrangements := Nat.choose remaining_lamps remaining_red
  (remaining_color_arrangements * remaining_on_arrangements : ℚ) / (color_arrangements * on_arrangements) = 9 / 100 :=
by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l2906_290620


namespace NUMINAMATH_CALUDE_trapezoid_angle_bisector_inscribed_circle_l2906_290662

noncomputable section

/-- Represents a point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : Point) : ℝ := sorry

/-- The angle bisector of an angle -/
def angleBisector (p q r : Point) : Point := sorry

/-- Check if a point lies on a line segment -/
def onSegment (p q r : Point) : Prop := sorry

/-- Check if a circle is inscribed in a triangle -/
def isInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Check if a point is a tangent point of a circle on a line segment -/
def isTangentPoint (p : Point) (c : Circle) (q r : Point) : Prop := sorry

theorem trapezoid_angle_bisector_inscribed_circle 
  (ABCD : Trapezoid) (E M H : Point) (c : Circle) :
  onSegment E ABCD.B ABCD.C →
  angleBisector ABCD.B ABCD.A ABCD.D = E →
  isInscribed c (Triangle.mk ABCD.A ABCD.B E) →
  isTangentPoint M c ABCD.A ABCD.B →
  isTangentPoint H c ABCD.B E →
  distance ABCD.A ABCD.B = 2 →
  distance M H = 1 →
  angle ABCD.B ABCD.A ABCD.D = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_angle_bisector_inscribed_circle_l2906_290662


namespace NUMINAMATH_CALUDE_sum_of_squares_squared_l2906_290672

theorem sum_of_squares_squared (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) :
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_squared_l2906_290672


namespace NUMINAMATH_CALUDE_coefficient_x3_in_expansion_l2906_290659

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of coefficients
def sumCoefficients (n : ℕ) : ℕ := (3 : ℕ) ^ n

-- Define the function to calculate the coefficient of x³
def coefficientX3 (n : ℕ) : ℕ := 8 * binomial n 3

-- Theorem statement
theorem coefficient_x3_in_expansion :
  ∃ n : ℕ, sumCoefficients n = 243 ∧ coefficientX3 n = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_in_expansion_l2906_290659


namespace NUMINAMATH_CALUDE_common_root_of_three_equations_l2906_290641

/-- Given nonzero real numbers a, b, c, and the fact that any two of the equations
    ax^11 + bx^4 + c = 0, bx^11 + cx^4 + a = 0, cx^11 + ax^4 + b = 0 have a common root,
    prove that all three equations have a common root. -/
theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_common_12 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_common_23 : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_common_13 : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ c * x^11 + a * x^4 + b = 0) :
  ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_common_root_of_three_equations_l2906_290641


namespace NUMINAMATH_CALUDE_three_coins_same_probability_l2906_290668

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The number of specific coins we're interested in -/
def num_specific_coins : ℕ := 3

/-- The probability of three specific coins out of five all coming up the same -/
theorem three_coins_same_probability :
  (2^(num_specific_coins - 1) * 2^(num_coins - num_specific_coins)) / 2^num_coins = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_three_coins_same_probability_l2906_290668


namespace NUMINAMATH_CALUDE_projected_attendance_increase_l2906_290600

theorem projected_attendance_increase (A : ℝ) (h1 : A > 0) : 
  let actual_attendance := 0.8 * A
  let projected_attendance := (1 + P / 100) * A
  0.8 * A = 0.64 * ((1 + P / 100) * A) →
  P = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_projected_attendance_increase_l2906_290600


namespace NUMINAMATH_CALUDE_not_square_difference_l2906_290669

/-- The square difference formula -/
def square_difference (p q : ℝ) : ℝ := p^2 - q^2

/-- Expression that cannot be directly represented by the square difference formula -/
def problematic_expression (a : ℝ) : ℝ := (a - 1) * (-a + 1)

/-- Theorem stating that the problematic expression cannot be directly represented
    by the square difference formula for any real values of p and q -/
theorem not_square_difference :
  ∀ (a p q : ℝ), problematic_expression a ≠ square_difference p q :=
by sorry

end NUMINAMATH_CALUDE_not_square_difference_l2906_290669


namespace NUMINAMATH_CALUDE_total_raisins_l2906_290675

def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

theorem total_raisins : yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_l2906_290675


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_iff_perpendicular_two_lines_l2906_290602

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines are distinct -/
def distinct_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem to be proven false -/
theorem perpendicular_line_plane_iff_perpendicular_two_lines (l : Line3D) (p : Plane3D) :
  perpendicular_line_plane l p ↔ 
  ∃ (l1 l2 : Line3D), line_in_plane l1 p ∧ line_in_plane l2 p ∧ 
                      distinct_lines l1 l2 ∧ 
                      perpendicular_lines l l1 ∧ perpendicular_lines l l2 :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_iff_perpendicular_two_lines_l2906_290602


namespace NUMINAMATH_CALUDE_mode_of_student_ages_l2906_290618

def student_ages : List ℕ := [13, 14, 15, 14, 14, 15]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_ages :
  mode student_ages = 14 := by sorry

end NUMINAMATH_CALUDE_mode_of_student_ages_l2906_290618


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l2906_290606

theorem intersection_sum_zero (x₁ x₂ : ℝ) (h₁ : x₁^2 + 6^2 = 144) (h₂ : x₂^2 + 6^2 = 144) :
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l2906_290606


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2906_290693

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 < x ∧ x < 1) ↔ (x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2906_290693


namespace NUMINAMATH_CALUDE_incenter_coords_l2906_290635

/-- Triangle ABC with incenter I -/
structure TriangleWithIncenter where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Incenter I of the triangle -/
  I : ℝ × ℝ
  /-- Coordinates of incenter I as (x, y, z) where x⃗A + y⃗B + z⃗C = ⃗I -/
  coords : ℝ × ℝ × ℝ

/-- The theorem stating that the coordinates of the incenter are (2/9, 1/3, 4/9) -/
theorem incenter_coords (t : TriangleWithIncenter) 
  (h1 : t.AB = 6)
  (h2 : t.BC = 8)
  (h3 : t.CA = 4)
  (h4 : t.coords.1 + t.coords.2.1 + t.coords.2.2 = 1) :
  t.coords = (2/9, 1/3, 4/9) := by
  sorry

end NUMINAMATH_CALUDE_incenter_coords_l2906_290635


namespace NUMINAMATH_CALUDE_product_less_than_2400_l2906_290615

theorem product_less_than_2400 : 817 * 3 < 2400 := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_2400_l2906_290615


namespace NUMINAMATH_CALUDE_system_equation_ratio_l2906_290663

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l2906_290663


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2906_290679

/-- The area of a square inscribed in the ellipse x²/4 + y²/9 = 1, with sides parallel to the coordinate axes. -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, x^2 / 4 + y^2 / 9 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 144 / 13) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2906_290679


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2906_290676

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right-angled triangle condition (Pythagorean theorem)
  c^2 = a^2 + b^2 →
  -- Sum of squares of all sides is 2500
  a^2 + b^2 + c^2 = 2500 →
  -- Difference between hypotenuse and one side is 10
  c - a = 10 →
  -- Prove that the hypotenuse length is 25√2
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2906_290676


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_remainder_is_z_minus_one_l2906_290661

/-- The polynomial division theorem for this specific case -/
theorem polynomial_division_theorem (z : ℂ) :
  ∃ (Q R : ℂ → ℂ), z^2023 + 1 = (z^2 - z + 1) * Q z + R z ∧ 
  (∀ x, ∃ (a b : ℂ), R x = a * x + b) := by sorry

/-- The main theorem proving R(z) = z - 1 -/
theorem remainder_is_z_minus_one :
  ∃ (Q R : ℂ → ℂ), 
    (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) ∧
    (∀ x, ∃ (a b : ℂ), R x = a * x + b) ∧
    (∀ z, R z = z - 1) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_remainder_is_z_minus_one_l2906_290661


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2906_290636

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d > 0) :
  let decimal := (n : ℚ) / d
  let whole_part := (decimal.floor : ℤ)
  let fractional_part := decimal - whole_part
  let expanded := fractional_part * (10 ^ 10 : ℚ)  -- Multiply by a large power of 10 to see digits
  ∃ k, 0 < k ∧ k ≤ 10 ∧ (expanded.floor : ℤ) % (10 ^ k) ≠ 0 ∧
      ∀ j, 0 < j ∧ j < k → (expanded.floor : ℤ) % (10 ^ j) = 0 →
  (n = 5 ∧ d = 3125) → k - 1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2906_290636


namespace NUMINAMATH_CALUDE_marble_probability_l2906_290691

theorem marble_probability (green yellow white : ℕ) 
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow : ℚ) / (green + yellow + white) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l2906_290691


namespace NUMINAMATH_CALUDE_congruence_solution_l2906_290644

theorem congruence_solution (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 201) (h3 : 200 * n ≡ 144 [ZMOD 101]) :
  n ≡ 29 [ZMOD 101] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2906_290644


namespace NUMINAMATH_CALUDE_burger_cost_proof_l2906_290698

def total_cost : ℝ := 15
def fries_cost : ℝ := 2
def fries_quantity : ℕ := 2
def salad_cost_multiplier : ℕ := 3

theorem burger_cost_proof :
  let salad_cost := salad_cost_multiplier * fries_cost
  let fries_total_cost := fries_quantity * fries_cost
  let burger_cost := total_cost - (salad_cost + fries_total_cost)
  burger_cost = 5 := by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l2906_290698


namespace NUMINAMATH_CALUDE_triangle_area_l2906_290647

theorem triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (BC = 8 ∧ AB = 10 ∧ AC^2 + BC^2 = AB^2) →
  (1/2 * BC * AC = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2906_290647


namespace NUMINAMATH_CALUDE_middle_number_proof_l2906_290617

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22) : 
  y = 8 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2906_290617


namespace NUMINAMATH_CALUDE_min_area_rectangle_l2906_290625

theorem min_area_rectangle (l w : ℤ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 120 → l * w ≥ 59 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l2906_290625


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2906_290682

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2906_290682


namespace NUMINAMATH_CALUDE_shelly_keychain_thread_l2906_290629

def inches_per_keychain (class_friends : ℕ) (club_friends : ℕ) (total_thread : ℕ) : ℚ :=
  total_thread / (class_friends + club_friends)

theorem shelly_keychain_thread : 
  let class_friends : ℕ := 6
  let club_friends : ℕ := class_friends / 2
  let total_thread : ℕ := 108
  inches_per_keychain class_friends club_friends total_thread = 12 := by
  sorry

end NUMINAMATH_CALUDE_shelly_keychain_thread_l2906_290629


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2906_290695

theorem polynomial_simplification (x : ℝ) :
  4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6) =
  x^3 + 12 * x^2 - 2 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2906_290695


namespace NUMINAMATH_CALUDE_prime_fraction_equality_l2906_290670

theorem prime_fraction_equality (A B : ℕ) : 
  Nat.Prime A → 
  Nat.Prime B → 
  A > 0 → 
  B > 0 → 
  (1 : ℚ) / A - (1 : ℚ) / B = 192 / (2005^2 - 2004^2) → 
  B = 211 := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_equality_l2906_290670


namespace NUMINAMATH_CALUDE_inverse_of_49_mod_89_l2906_290648

theorem inverse_of_49_mod_89 (h : (7⁻¹ : ZMod 89) = 55) : (49⁻¹ : ZMod 89) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_49_mod_89_l2906_290648


namespace NUMINAMATH_CALUDE_bubble_theorem_l2906_290601

/-- Given a hemisphere with radius 4∛2 cm and volume double that of an initial spherical bubble,
    prove the radius of the original bubble and the volume of a new sphere with doubled radius. -/
theorem bubble_theorem (r : ℝ) (h1 : r = 4 * Real.rpow 2 (1/3)) :
  let R := Real.rpow 4 (1/3)
  let V_new := (64/3) * Real.pi * Real.rpow 4 (1/3)
  (2/3) * Real.pi * r^3 = 2 * ((4/3) * Real.pi * R^3) ∧ 
  (4/3) * Real.pi * (2*R)^3 = V_new := by
  sorry

end NUMINAMATH_CALUDE_bubble_theorem_l2906_290601


namespace NUMINAMATH_CALUDE_f_intersects_y_axis_l2906_290609

-- Define the function f(x) = 4x - 4
def f (x : ℝ) : ℝ := 4 * x - 4

-- Theorem: f intersects the y-axis at (0, -4)
theorem f_intersects_y_axis :
  f 0 = -4 := by sorry

end NUMINAMATH_CALUDE_f_intersects_y_axis_l2906_290609


namespace NUMINAMATH_CALUDE_tank_capacity_l2906_290613

/-- Proves that a tank's full capacity is 270/7 gallons, given initial and final fill levels -/
theorem tank_capacity (initial_fill : Rat) (final_fill : Rat) (used_gallons : Rat) 
  (h1 : initial_fill = 4/5)
  (h2 : final_fill = 1/3)
  (h3 : used_gallons = 18)
  (h4 : initial_fill * full_capacity - final_fill * full_capacity = used_gallons) :
  full_capacity = 270/7 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l2906_290613


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2906_290640

/-- If the equation x²/(4-m) - y²/(2+m) = 1 represents a hyperbola, 
    then the range of m is (-2, 4) -/
theorem hyperbola_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (4 - m) - y^2 / (2 + m) = 1) → 
  -2 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2906_290640


namespace NUMINAMATH_CALUDE_square_difference_l2906_290652

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 5) : (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2906_290652


namespace NUMINAMATH_CALUDE_number_solution_l2906_290684

theorem number_solution : ∃ x : ℝ, 2 * x - 2.6 * 4 = 10 ∧ x = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l2906_290684


namespace NUMINAMATH_CALUDE_exist_decreasing_gcd_sequence_l2906_290639

theorem exist_decreasing_gcd_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j : Fin 100, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.gcd (a i) (a (i + 1)) > Nat.gcd (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_exist_decreasing_gcd_sequence_l2906_290639


namespace NUMINAMATH_CALUDE_inequality_proof_l2906_290603

theorem inequality_proof (n : ℕ+) (k : ℝ) (hk : k > 0) :
  1 - 1/k ≤ n * (k^(1/n : ℝ) - 1) ∧ n * (k^(1/n : ℝ) - 1) ≤ k - 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2906_290603


namespace NUMINAMATH_CALUDE_equation_solution_l2906_290678

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = 13/4 ∧ 
  (∀ x : ℝ, x - 3 = 4 * (x - 3)^2 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2906_290678


namespace NUMINAMATH_CALUDE_inequality_proof_l2906_290630

theorem inequality_proof (a : ℝ) (h : -1 < a ∧ a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2906_290630


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2906_290634

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem tangent_line_equation (P : ℝ × ℝ) (h₁ : P = (-2, -2)) :
  ∃ (m b : ℝ), (∀ x, (m * x + b = 9 * x + 16) ∨ (m * x + b = -2)) ∧
  (∃ x₀, f x₀ = m * x₀ + b ∧ 
         ∀ x, f x ≥ m * x + b ∧ 
         (f x = m * x + b ↔ x = x₀)) ∧
  (m * P.1 + b = P.2) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2906_290634


namespace NUMINAMATH_CALUDE_representatives_selection_theorem_l2906_290658

/-- The number of ways to select 3 representatives from 3 different companies -/
def selectRepresentatives (totalCompanies : ℕ) (companiesWithOneRep : ℕ) (repsFromSpecialCompany : ℕ) : ℕ :=
  Nat.choose repsFromSpecialCompany 1 * Nat.choose companiesWithOneRep 2 +
  Nat.choose companiesWithOneRep 3

/-- Theorem stating that the number of ways to select 3 representatives from 3 different companies
    out of 5 companies (where one company has 2 representatives and the others have 1 each) is 16 -/
theorem representatives_selection_theorem :
  selectRepresentatives 5 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_representatives_selection_theorem_l2906_290658


namespace NUMINAMATH_CALUDE_max_books_borrowed_l2906_290688

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (min_three_books : Nat) (avg_books : Nat) :
  total_students = 20 →
  zero_books = 2 →
  one_book = 10 →
  two_books = 5 →
  min_three_books = total_students - zero_books - one_book - two_books →
  avg_books = 2 →
  ∃ (max_books : Nat), 
    max_books = (total_students * avg_books) - 
      (one_book * 1 + two_books * 2 + (min_three_books - 1) * 3) ∧
    max_books ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l2906_290688


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_leg_length_l2906_290614

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The acute angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The length of the trapezoid's leg -/
  leg_length : ℝ

/-- Theorem stating the relationship between the area, base angle, and leg length of a circumscribed trapezoid -/
theorem circumscribed_trapezoid_leg_length 
  (t : CircumscribedTrapezoid) 
  (h1 : t.area = 32 * Real.sqrt 3)
  (h2 : t.base_angle = π / 3) :
  t.leg_length = 8 := by
  sorry

#check circumscribed_trapezoid_leg_length

end NUMINAMATH_CALUDE_circumscribed_trapezoid_leg_length_l2906_290614


namespace NUMINAMATH_CALUDE_select_students_equality_l2906_290646

/-- The number of ways to select 5 students from a class of 50, including one president and one 
    vice-president, with at least one of the president or vice-president attending. -/
def select_students (n : ℕ) (k : ℕ) (total : ℕ) (leaders : ℕ) : ℕ :=
  Nat.choose leaders 1 * Nat.choose (total - leaders) (k - 1) +
  Nat.choose leaders 2 * Nat.choose (total - leaders) (k - 2)

theorem select_students_equality :
  select_students 5 5 50 2 = Nat.choose 50 5 - Nat.choose 48 5 :=
sorry

end NUMINAMATH_CALUDE_select_students_equality_l2906_290646


namespace NUMINAMATH_CALUDE_husband_towel_usage_l2906_290651

/-- The number of bath towels used by Kylie in a month -/
def kylie_towels : ℕ := 3

/-- The number of bath towels used by Kylie's daughters in a month -/
def daughters_towels : ℕ := 6

/-- The number of bath towels that fit in one load of laundry -/
def towels_per_load : ℕ := 4

/-- The number of loads of laundry needed to clean all used towels -/
def loads_of_laundry : ℕ := 3

/-- The number of bath towels used by the husband in a month -/
def husband_towels : ℕ := 3

theorem husband_towel_usage :
  kylie_towels + daughters_towels + husband_towels = towels_per_load * loads_of_laundry :=
by sorry

end NUMINAMATH_CALUDE_husband_towel_usage_l2906_290651


namespace NUMINAMATH_CALUDE_custom_op_solution_l2906_290605

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem: Given the custom operation and x9 = 160, x must equal 21 -/
theorem custom_op_solution : ∃ x : ℤ, customOp x 9 = 160 ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l2906_290605


namespace NUMINAMATH_CALUDE_probability_two_blue_buttons_l2906_290653

/-- Represents a jar with buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of an event -/
def Probability := ℚ

/-- Initial state of Jar C -/
def initial_jar_c : Jar := ⟨5, 10⟩

/-- Number of buttons removed from each color -/
def removed_buttons : ℕ := 2

/-- Final state of Jar C after removal -/
def final_jar_c : Jar := ⟨initial_jar_c.red - removed_buttons, initial_jar_c.blue - 2 * removed_buttons⟩

/-- State of Jar D after receiving removed buttons -/
def jar_d : Jar := ⟨removed_buttons, 2 * removed_buttons⟩

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue_buttons : 
  (final_jar_c.red + final_jar_c.blue : ℚ) = 3/5 * (initial_jar_c.red + initial_jar_c.blue) →
  (final_jar_c.blue : ℚ) / (final_jar_c.red + final_jar_c.blue) * 
  (jar_d.blue : ℚ) / (jar_d.red + jar_d.blue) = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_blue_buttons_l2906_290653


namespace NUMINAMATH_CALUDE_only_B_in_region_l2906_290692

def point_A : ℝ × ℝ := (1, -1)
def point_B : ℝ × ℝ := (0, 1)
def point_C : ℝ × ℝ := (1, 0)
def point_D : ℝ × ℝ := (-2, 0)

def in_region (p : ℝ × ℝ) : Prop := p.1 + 2 * p.2 - 1 > 0

theorem only_B_in_region :
  ¬(in_region point_A) ∧
  (in_region point_B) ∧
  ¬(in_region point_C) ∧
  ¬(in_region point_D) :=
by sorry

end NUMINAMATH_CALUDE_only_B_in_region_l2906_290692


namespace NUMINAMATH_CALUDE_race_outcomes_six_participants_l2906_290697

/-- The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties -/
def race_outcomes (n : ℕ) : ℕ :=
  if n ≥ 4 then n * (n - 1) * (n - 2) * (n - 3) else 0

/-- Theorem: The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties is 360 -/
theorem race_outcomes_six_participants : race_outcomes 6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_six_participants_l2906_290697


namespace NUMINAMATH_CALUDE_right_triangle_area_l2906_290610

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : 
  (1/2) * a * b = 30 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2906_290610


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l2906_290685

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_div : 360 ∣ n^3) :
  ∃ (w : ℕ), w = 30 ∧ w ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ w :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l2906_290685


namespace NUMINAMATH_CALUDE_iterative_average_difference_l2906_290677

def iterative_average (seq : List ℚ) : ℚ :=
  seq.foldl (λ acc x => (acc + x) / 2) (seq.head!)

def max_sequence : List ℚ := [6, 5, 4, 3, 2, 1]
def min_sequence : List ℚ := [1, 2, 3, 4, 5, 6]

theorem iterative_average_difference :
  iterative_average max_sequence - iterative_average min_sequence = 1 := by
  sorry

end NUMINAMATH_CALUDE_iterative_average_difference_l2906_290677


namespace NUMINAMATH_CALUDE_monkey_rope_system_length_l2906_290643

/-- Represents the age and weight of a monkey and its mother, and the properties of a rope system -/
structure MonkeyRopeSystem where
  monkey_age : ℝ
  mother_age : ℝ
  rope_weight_per_foot : ℝ
  weight : ℝ

/-- The conditions of the monkey-rope system problem -/
def monkey_rope_system_conditions (s : MonkeyRopeSystem) : Prop :=
  s.monkey_age + s.mother_age = 4 ∧
  s.monkey_age = s.mother_age / 2 ∧
  s.rope_weight_per_foot = 1/4 ∧
  s.weight = s.mother_age

/-- The theorem stating that under the given conditions, the rope length is 5 feet -/
theorem monkey_rope_system_length
  (s : MonkeyRopeSystem)
  (h : monkey_rope_system_conditions s) :
  (s.weight + s.weight) / (3/4) = 5 :=
sorry

end NUMINAMATH_CALUDE_monkey_rope_system_length_l2906_290643


namespace NUMINAMATH_CALUDE_monomial_division_l2906_290624

theorem monomial_division (x : ℝ) : 2 * x^3 / x^2 = 2 * x := by sorry

end NUMINAMATH_CALUDE_monomial_division_l2906_290624


namespace NUMINAMATH_CALUDE_frog_climb_time_l2906_290690

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ

/-- Calculates the time taken for the frog to climb the well -/
def climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to climb the well -/
theorem frog_climb_time :
  let f : FrogClimb := {
    well_depth := 12,
    climb_distance := 3,
    slip_distance := 1,
    slip_time_ratio := 1/3
  }
  climb_time f = 22 := by sorry

end NUMINAMATH_CALUDE_frog_climb_time_l2906_290690


namespace NUMINAMATH_CALUDE_B_pow_101_eq_B_l2906_290633

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_pow_101_eq_B : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_pow_101_eq_B_l2906_290633


namespace NUMINAMATH_CALUDE_will_remaining_candy_l2906_290626

/-- Calculates the remaining pieces of candy after giving some away. -/
def remaining_candy (
  chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (chocolate_pieces mint_pieces caramel_pieces : ℕ)
  (chocolate_given mint_given caramel_given : ℕ) : ℕ :=
  (chocolate_boxes * chocolate_pieces + 
   mint_boxes * mint_pieces + 
   caramel_boxes * caramel_pieces) - 
  (chocolate_given * chocolate_pieces + 
   mint_given * mint_pieces + 
   caramel_given * caramel_pieces)

/-- Proves that Will has 123 pieces of candy remaining. -/
theorem will_remaining_candy : 
  remaining_candy 7 5 4 12 15 10 3 2 1 = 123 := by
  sorry

end NUMINAMATH_CALUDE_will_remaining_candy_l2906_290626


namespace NUMINAMATH_CALUDE_number_division_problem_l2906_290673

theorem number_division_problem (x y : ℚ) : 
  (x - 5) / y = 7 → (x - 2) / 13 = 4 → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2906_290673


namespace NUMINAMATH_CALUDE_no_time_left_after_student_council_l2906_290604

/-- Represents the journey to school with various stops -/
structure SchoolJourney where
  totalTimeAvailable : ℕ
  travelTimeWithTraffic : ℕ
  timeToLibrary : ℕ
  timeToReturnBooks : ℕ
  extraTimeForManyBooks : ℕ
  timeToStudentCouncil : ℕ
  timeToSubmitProject : ℕ
  timeToClassroom : ℕ

/-- Calculates the time left after leaving the student council room -/
def timeLeftAfterStudentCouncil (journey : SchoolJourney) : Int :=
  journey.totalTimeAvailable - (journey.travelTimeWithTraffic + journey.timeToLibrary +
  journey.timeToReturnBooks + journey.extraTimeForManyBooks + journey.timeToStudentCouncil +
  journey.timeToSubmitProject)

/-- Theorem stating that in the worst-case scenario, there's no time left after leaving the student council room -/
theorem no_time_left_after_student_council (journey : SchoolJourney)
  (h1 : journey.totalTimeAvailable = 30)
  (h2 : journey.travelTimeWithTraffic = 25)
  (h3 : journey.timeToLibrary = 3)
  (h4 : journey.timeToReturnBooks = 2)
  (h5 : journey.extraTimeForManyBooks = 2)
  (h6 : journey.timeToStudentCouncil = 5)
  (h7 : journey.timeToSubmitProject = 3)
  (h8 : journey.timeToClassroom = 6) :
  timeLeftAfterStudentCouncil journey ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_time_left_after_student_council_l2906_290604


namespace NUMINAMATH_CALUDE_equation_solution_set_l2906_290681

theorem equation_solution_set : ∃ (S : Set ℝ), 
  S = {x : ℝ | (1 / (x^2 + 8*x - 12) + 1 / (x^2 + 5*x - 12) + 1 / (x^2 - 10*x - 12) = 0)} ∧ 
  S = {Real.sqrt 12, -Real.sqrt 12, 4, 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l2906_290681


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2906_290619

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Length of the shorter diagonal -/
  shorter_diagonal : ℝ
  /-- Length of the longer diagonal -/
  longer_diagonal : ℝ
  /-- Perimeter of the rhombus -/
  perimeter : ℝ
  /-- The shorter diagonal is 30 cm -/
  shorter_diagonal_length : shorter_diagonal = 30
  /-- The perimeter is 156 cm -/
  perimeter_length : perimeter = 156
  /-- The longer diagonal is longer than the shorter diagonal -/
  diagonal_order : longer_diagonal ≥ shorter_diagonal

/-- Theorem: In a rhombus with one diagonal of 30 cm and a perimeter of 156 cm, 
    the length of the other diagonal is 72 cm -/
theorem rhombus_longer_diagonal (r : Rhombus) : r.longer_diagonal = 72 := by
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2906_290619


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l2906_290689

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h1 : failed_english = 45)
  (h2 : failed_both = 20)
  (h3 : passed_both = 40) :
  ∃ (failed_hindi : ℝ), failed_hindi = 35 := by
sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l2906_290689


namespace NUMINAMATH_CALUDE_tan_range_proof_l2906_290645

theorem tan_range_proof (x : ℝ) (hx : x ∈ Set.Icc (-π/4) (π/4) ∧ x ≠ 0) :
  ∃ y, y = Real.tan (π/2 - x) ↔ y ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_tan_range_proof_l2906_290645


namespace NUMINAMATH_CALUDE_rational_number_statements_l2906_290612

theorem rational_number_statements (a b : ℚ) : 
  (∃! n : ℕ, n = 2 ∧ 
    (((a + b > 0 ∧ (a > 0 ↔ b > 0)) → (a > 0 ∧ b > 0)) = true) ∧
    ((a + b < 0 → ¬(a > 0 ↔ b > 0)) = false) ∧
    (((abs a > abs b ∧ ¬(a > 0 ↔ b > 0)) → a + b > 0) = false) ∧
    ((abs a < b → a + b > 0) = true)) :=
sorry

end NUMINAMATH_CALUDE_rational_number_statements_l2906_290612


namespace NUMINAMATH_CALUDE_emu_count_correct_l2906_290696

/-- Represents the number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 60

/-- Represents the number of parts (head + legs) per emu -/
def parts_per_emu : ℕ := 3

/-- Theorem stating that the number of emus is correct given the total number of heads and legs -/
theorem emu_count_correct : num_emus * parts_per_emu = total_heads_and_legs := by
  sorry

end NUMINAMATH_CALUDE_emu_count_correct_l2906_290696


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2906_290632

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l2906_290632


namespace NUMINAMATH_CALUDE_optimal_range_golden_section_l2906_290680

theorem optimal_range_golden_section (m : ℝ) : 
  (1000 ≤ m) →  -- The optimal range starts at 1000
  (1000 + (m - 1000) * 0.618 = 1618) →  -- The good point is determined by the golden ratio
  (m = 2000) :=  -- We want to prove that m = 2000
by
  sorry

end NUMINAMATH_CALUDE_optimal_range_golden_section_l2906_290680


namespace NUMINAMATH_CALUDE_number_of_children_l2906_290622

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) (h1 : crayons_per_child = 3) (h2 : total_crayons = 18) :
  total_crayons / crayons_per_child = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2906_290622


namespace NUMINAMATH_CALUDE_point_opposite_sides_line_value_range_l2906_290666

/-- Given that the points (3,1) and (-4,6) lie on opposite sides of the line 3x - 2y + a = 0,
    prove that the value range of a is -7 < a < 24. -/
theorem point_opposite_sides_line_value_range (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 → -7 < a ∧ a < 24 := by
  sorry

end NUMINAMATH_CALUDE_point_opposite_sides_line_value_range_l2906_290666


namespace NUMINAMATH_CALUDE_red_candy_count_l2906_290638

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end NUMINAMATH_CALUDE_red_candy_count_l2906_290638


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2906_290642

/-- The parabola function -/
def f (x : ℝ) : ℝ := (2 - x) * x

/-- The axis of symmetry of the parabola -/
def axis_of_symmetry : ℝ := 1

/-- Theorem: The axis of symmetry of the parabola y = (2-x)x is the line x = 1 -/
theorem parabola_axis_of_symmetry :
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2906_290642


namespace NUMINAMATH_CALUDE_sum_max_min_cubes_l2906_290699

/-- Represents a view of the geometric figure -/
structure View where
  (front : Set (ℕ × ℕ))
  (left : Set (ℕ × ℕ))
  (top : Set (ℕ × ℕ))

/-- Counts the number of cubes in a valid configuration -/
def count_cubes (v : View) : ℕ → Bool := sorry

/-- The maximum number of cubes that can form the figure -/
def max_cubes (v : View) : ℕ := sorry

/-- The minimum number of cubes that can form the figure -/
def min_cubes (v : View) : ℕ := sorry

/-- The theorem stating that the sum of max and min cubes is 20 -/
theorem sum_max_min_cubes (v : View) : max_cubes v + min_cubes v = 20 := by sorry

end NUMINAMATH_CALUDE_sum_max_min_cubes_l2906_290699


namespace NUMINAMATH_CALUDE_range_of_t_l2906_290665

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_t (f : ℝ → ℝ) (h1 : MonoDecreasing f) :
  {t : ℝ | f (t^2) - f t < 0} = {t : ℝ | t < 0 ∨ t > 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l2906_290665


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_12_l2906_290664

/-- The function f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3 -/
def has_extremum_at_3 (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2 - 10*x
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 3| ∧ |x - 3| < ε → 
    (f x - f 3) * (x - 3) ≤ 0

/-- Given that f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3, prove that a = 12 -/
theorem extremum_implies_a_equals_12 : 
  has_extremum_at_3 a → a = 12 := by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_12_l2906_290664


namespace NUMINAMATH_CALUDE_unique_solution_l2906_290656

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, f (a^2) (f b c + 1) = a^2 * (b * c + 1)

/-- Theorem stating that the only function satisfying the equation is f(a,b) = a*b -/
theorem unique_solution {f : ℝ → ℝ → ℝ} (hf : SatisfiesEquation f) :
  ∀ a b : ℝ, f a b = a * b := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2906_290656


namespace NUMINAMATH_CALUDE_mango_problem_l2906_290649

theorem mango_problem (alexis dilan ashley : ℕ) : 
  alexis = 4 * (dilan + ashley) →
  ashley = 2 * dilan →
  alexis = 60 →
  alexis + dilan + ashley = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_mango_problem_l2906_290649


namespace NUMINAMATH_CALUDE_keith_pears_given_away_l2906_290671

/-- The number of pears Keith gave away -/
def pears_given_away (keith_initial : ℕ) (mike_initial : ℕ) (remaining : ℕ) : ℕ :=
  keith_initial + mike_initial - remaining

theorem keith_pears_given_away :
  pears_given_away 47 12 13 = 46 := by
  sorry

end NUMINAMATH_CALUDE_keith_pears_given_away_l2906_290671


namespace NUMINAMATH_CALUDE_quiz_homework_difference_l2906_290694

/-- Represents the points distribution in Paul's biology class -/
structure PointsDistribution where
  total : ℕ
  homework : ℕ
  quiz : ℕ
  test : ℕ

/-- The conditions for Paul's point distribution -/
def paulsDistribution (p : PointsDistribution) : Prop :=
  p.total = 265 ∧
  p.homework = 40 ∧
  p.test = 4 * p.quiz ∧
  p.total = p.homework + p.quiz + p.test

/-- Theorem stating the difference between quiz and homework points -/
theorem quiz_homework_difference (p : PointsDistribution) 
  (h : paulsDistribution p) : p.quiz - p.homework = 5 := by
  sorry

end NUMINAMATH_CALUDE_quiz_homework_difference_l2906_290694


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l2906_290686

/-- The number of green marbles Dan has after Mike took some -/
def remaining_green_marbles (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Proof that Dan has 9 green marbles after Mike took 23 -/
theorem dans_remaining_marbles :
  remaining_green_marbles 32 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l2906_290686


namespace NUMINAMATH_CALUDE_bridget_weight_l2906_290623

/-- Given that Martha weighs 2 pounds and Bridget is 37 pounds heavier than Martha,
    prove that Bridget weighs 39 pounds. -/
theorem bridget_weight (martha_weight : ℕ) (weight_difference : ℕ) :
  martha_weight = 2 →
  weight_difference = 37 →
  martha_weight + weight_difference = 39 :=
by sorry

end NUMINAMATH_CALUDE_bridget_weight_l2906_290623


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l2906_290660

theorem power_of_power_of_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l2906_290660


namespace NUMINAMATH_CALUDE_quadratic_sets_equal_or_disjoint_l2906_290608

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The set of f(2n) where n is an integer -/
def M (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n)}

/-- The set of f(2n+1) where n is an integer -/
def N (f : ℝ → ℝ) : Set ℝ := {y | ∃ n : ℤ, y = f (2 * ↑n + 1)}

/-- Theorem: For any quadratic function, M and N are either equal or disjoint -/
theorem quadratic_sets_equal_or_disjoint (a b c : ℝ) :
  let f := QuadraticFunction a b c
  (M f = N f) ∨ (M f ∩ N f = ∅) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sets_equal_or_disjoint_l2906_290608


namespace NUMINAMATH_CALUDE_first_hour_rate_l2906_290667

def shift_duration : ℕ := 4 -- hours
def masks_per_shift : ℕ := 45
def later_rate : ℕ := 6 -- minutes per mask after the first hour

-- x is the time (in minutes) to make one mask in the first hour
theorem first_hour_rate (x : ℕ) : x = 4 ↔ 
  (60 / x : ℚ) + (shift_duration - 1) * (60 / later_rate : ℚ) = masks_per_shift :=
by sorry

end NUMINAMATH_CALUDE_first_hour_rate_l2906_290667


namespace NUMINAMATH_CALUDE_dora_receives_two_packs_l2906_290657

/-- The number of packs of stickers Dora receives --/
def dora_sticker_packs (allowance : ℕ) (card_price : ℕ) (sticker_price : ℕ) (num_people : ℕ) : ℕ :=
  let total_money := allowance * num_people
  let remaining_money := total_money - card_price
  let total_sticker_packs := remaining_money / sticker_price
  total_sticker_packs / num_people

/-- Theorem stating that Dora receives 2 packs of stickers --/
theorem dora_receives_two_packs :
  dora_sticker_packs 9 10 2 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dora_receives_two_packs_l2906_290657


namespace NUMINAMATH_CALUDE_total_colored_pencils_l2906_290631

theorem total_colored_pencils (madeline_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : ∃ cheryl_pencils : ℕ, cheryl_pencils = 2 * madeline_pencils)
  (h3 : ∃ cyrus_pencils : ℕ, 3 * cyrus_pencils = cheryl_pencils) :
  ∃ total_pencils : ℕ, total_pencils = madeline_pencils + cheryl_pencils + cyrus_pencils ∧ total_pencils = 231 :=
by
  sorry


end NUMINAMATH_CALUDE_total_colored_pencils_l2906_290631


namespace NUMINAMATH_CALUDE_triangle_side_value_l2906_290628

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 - c^2 = 2*b →
  Real.sin A * Real.cos C = 3 * Real.cos A * Real.sin A →
  b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l2906_290628


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2906_290621

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (abc : Triangle) :
  abc.a = 5 →
  Real.cos abc.B = 4/5 →
  (1/2) * abc.a * abc.c * Real.sin abc.B = 12 →
  (abc.a + abc.c) / (Real.sin abc.A + Real.sin abc.C) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2906_290621


namespace NUMINAMATH_CALUDE_triangle_perimeter_sum_specific_triangle_perimeter_sum_l2906_290607

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem triangle_perimeter_sum (initial_perimeter : ℝ) :
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by sorry

/-- The specific case where the initial triangle has a perimeter of 90 cm -/
theorem specific_triangle_perimeter_sum :
  (∑' n, 90 * (1/2)^n) = 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_sum_specific_triangle_perimeter_sum_l2906_290607


namespace NUMINAMATH_CALUDE_jerry_total_games_l2906_290655

/-- Calculates the total number of games Jerry has after his birthday and trade --/
def total_games_after (initial_action : ℕ) (initial_strategy : ℕ) 
  (action_increase_percent : ℕ) (strategy_increase_percent : ℕ) 
  (action_traded : ℕ) (sports_received : ℕ) : ℕ :=
  let action_increase := (initial_action * action_increase_percent) / 100
  let strategy_increase := (initial_strategy * strategy_increase_percent) / 100
  let final_action := initial_action + action_increase - action_traded
  let final_strategy := initial_strategy + strategy_increase
  final_action + final_strategy + sports_received

/-- Theorem stating that Jerry's total games after birthday and trade is 16 --/
theorem jerry_total_games : 
  total_games_after 7 5 30 20 2 3 = 16 := by sorry

end NUMINAMATH_CALUDE_jerry_total_games_l2906_290655


namespace NUMINAMATH_CALUDE_lawrence_walk_l2906_290650

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that given a speed of 3 km/h and a time of 1.33 hours, 
    the distance traveled is 3.99 km -/
theorem lawrence_walk : distance 3 1.33 = 3.99 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_walk_l2906_290650
