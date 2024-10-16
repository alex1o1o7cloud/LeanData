import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l2206_220654

-- Part 1
theorem simplify_expression (x y : ℝ) : 3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = -x * y := by
  sorry

-- Part 2
theorem simplify_and_evaluate (a b : ℝ) (h1 : a = 2) (h2 : b = -3) : 
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l2206_220654


namespace NUMINAMATH_CALUDE_inequality_proof_l2206_220656

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (c + d) + b / (d + a) + c / (a + b) + d / (b + c) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2206_220656


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2206_220661

theorem election_votes_theorem (total_votes : ℕ) : 
  (75 : ℝ) / 100 * ((100 : ℝ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2206_220661


namespace NUMINAMATH_CALUDE_tan_half_sum_l2206_220644

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3) 
  (h2 : Real.sin p + Real.sin q = 8/17) : 
  Real.tan ((p + q) / 2) = 24/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l2206_220644


namespace NUMINAMATH_CALUDE_different_signs_larger_negative_l2206_220635

theorem different_signs_larger_negative (a b : ℝ) : 
  a + b < 0 → a * b < 0 → 
  ((a < 0 ∧ b > 0 ∧ abs a > abs b) ∨ (a > 0 ∧ b < 0 ∧ abs b > abs a)) := by
  sorry

end NUMINAMATH_CALUDE_different_signs_larger_negative_l2206_220635


namespace NUMINAMATH_CALUDE_smallest_quotient_smallest_quotient_achievable_l2206_220688

def card_set : Set ℤ := {-5, -4, 0, 4, 6}

theorem smallest_quotient (a b : ℤ) (ha : a ∈ card_set) (hb : b ∈ card_set) (hab : a ≠ b) (hb_nonzero : b ≠ 0) :
  (a : ℚ) / b ≥ -3/2 :=
sorry

theorem smallest_quotient_achievable :
  ∃ (a b : ℤ), a ∈ card_set ∧ b ∈ card_set ∧ a ≠ b ∧ b ≠ 0 ∧ (a : ℚ) / b = -3/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_smallest_quotient_achievable_l2206_220688


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2206_220652

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

/-- Theorem stating that the ratio of Sandy's age to Molly's age is 7/9 -/
theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 63
  let molly_age : ℕ := sandy_age + 18
  age_ratio sandy_age molly_age = 7/9 := by
sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l2206_220652


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l2206_220666

theorem nonnegative_solutions_count : ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x := by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l2206_220666


namespace NUMINAMATH_CALUDE_log_equation_sum_l2206_220669

theorem log_equation_sum (A B C : ℕ+) (h_coprime : Nat.Coprime A B ∧ Nat.Coprime A C ∧ Nat.Coprime B C)
  (h_eq : A * Real.log 5 / Real.log 180 + B * Real.log 3 / Real.log 180 + C * Real.log 2 / Real.log 180 = 1) :
  A + B + C = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l2206_220669


namespace NUMINAMATH_CALUDE_min_stamps_theorem_l2206_220624

/-- The minimum number of stamps needed to make 48 cents using only 5 cent and 7 cent stamps -/
def min_stamps : ℕ := 8

/-- The value of stamps in cents -/
def total_value : ℕ := 48

/-- Represents a combination of 5 cent and 7 cent stamps -/
structure StampCombination where
  five_cent : ℕ
  seven_cent : ℕ

/-- Calculates the total value of a stamp combination -/
def combination_value (c : StampCombination) : ℕ :=
  5 * c.five_cent + 7 * c.seven_cent

/-- Calculates the total number of stamps in a combination -/
def total_stamps (c : StampCombination) : ℕ :=
  c.five_cent + c.seven_cent

/-- Predicate for a valid stamp combination that sums to the total value -/
def is_valid_combination (c : StampCombination) : Prop :=
  combination_value c = total_value

theorem min_stamps_theorem :
  ∃ (c : StampCombination), is_valid_combination c ∧
  (∀ (d : StampCombination), is_valid_combination d → total_stamps c ≤ total_stamps d) ∧
  total_stamps c = min_stamps :=
sorry

end NUMINAMATH_CALUDE_min_stamps_theorem_l2206_220624


namespace NUMINAMATH_CALUDE_largest_number_game_l2206_220604

theorem largest_number_game (a b c d : ℤ) : 
  (let game := λ (x y z w : ℤ) => (x + y + z) / 3 + w
   ({game a b c d, game a b d c, game a c d b, game b c d a} : Set ℤ) = {17, 21, 23, 29}) →
  (max a (max b (max c d)) = 21) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_game_l2206_220604


namespace NUMINAMATH_CALUDE_cube_root_of_eight_l2206_220687

theorem cube_root_of_eight : ∃ x : ℝ, x^3 = 8 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_eight_l2206_220687


namespace NUMINAMATH_CALUDE_expansion_properties_l2206_220697

/-- Given the expansion of (2x + 3/∛x)^n, where the ratio of the binomial coefficient
    of the third term to that of the second term is 5:2, prove the following: -/
theorem expansion_properties (n : ℕ) (x : ℝ) :
  (Nat.choose n 2 : ℚ) / (Nat.choose n 1 : ℚ) = 5 / 2 →
  (n = 6 ∧
   (∃ (r : ℕ), Nat.choose 6 r * 2^(6-r) * 3^r * x^(6 - 4/3*r) = 4320 * x^2) ∧
   (∃ (k : ℕ), Nat.choose 6 k * 2^(6-k) * 3^k * x^((2:ℝ)/3) = 4860 * x^((2:ℝ)/3) ∧
               ∀ (j : ℕ), j ≠ k → Nat.choose 6 j * 2^(6-j) * 3^j ≤ Nat.choose 6 k * 2^(6-k) * 3^k)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l2206_220697


namespace NUMINAMATH_CALUDE_beads_per_bracelet_l2206_220668

/-- The number of beaded necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used by Kylie -/
def total_beads : ℕ := 325

/-- Theorem stating that 10 beads are needed to make one beaded bracelet -/
theorem beads_per_bracelet : 
  (total_beads - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / wednesday_bracelets = 10 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_bracelet_l2206_220668


namespace NUMINAMATH_CALUDE_special_polynomial_form_l2206_220633

/-- A polynomial in two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  cyclic_sum : ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∀ (x y : ℝ), sp.P x y = (x + y) ^ sp.n.val * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l2206_220633


namespace NUMINAMATH_CALUDE_cylinder_radius_l2206_220695

structure Cone where
  diameter : ℚ
  altitude : ℚ

structure Cylinder where
  radius : ℚ

def inscribed_cylinder (cone : Cone) (cyl : Cylinder) : Prop :=
  cyl.radius * 2 = cyl.radius * 2 ∧  -- cylinder's diameter equals its height
  cone.diameter = 10 ∧
  cone.altitude = 12 ∧
  -- The axes of the cylinder and cone coincide (implicit in the problem setup)
  true

theorem cylinder_radius (cone : Cone) (cyl : Cylinder) :
  inscribed_cylinder cone cyl → cyl.radius = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_l2206_220695


namespace NUMINAMATH_CALUDE_euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l2206_220660

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (x * Complex.I)

-- Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- Theorem 1
theorem euler_identity : cexp π + 1 = 0 := by sorry

-- Theorem 2
theorem complex_power : (Complex.ofReal (1/2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2)) ^ 2022 = 1 := by sorry

-- Theorem 3
theorem cexp_sum_bound (x : ℝ) : Complex.abs (cexp x + cexp (-x)) ≤ 2 := by sorry

-- Theorem 4
theorem cexp_diff_not_always_bounded :
  ¬ (∀ x : ℝ, -2 ≤ (cexp x - cexp (-x)).re ∧ (cexp x - cexp (-x)).re ≤ 2 ∧
               -2 ≤ (cexp x - cexp (-x)).im ∧ (cexp x - cexp (-x)).im ≤ 2) := by sorry

end NUMINAMATH_CALUDE_euler_identity_complex_power_cexp_sum_bound_cexp_diff_not_always_bounded_l2206_220660


namespace NUMINAMATH_CALUDE_tangerine_count_l2206_220672

theorem tangerine_count (initial : ℕ) (added : ℕ) (total : ℕ) : 
  initial = 10 → added = 6 → total = initial + added → total = 16 := by
sorry

end NUMINAMATH_CALUDE_tangerine_count_l2206_220672


namespace NUMINAMATH_CALUDE_avg_age_coaches_is_23_l2206_220664

/-- Represents a sports club with members of different categories. -/
structure SportsClub where
  total_members : ℕ
  men : ℕ
  women : ℕ
  coaches : ℕ
  avg_age : ℕ
  avg_age_men : ℕ
  avg_age_women : ℕ

/-- Calculates the average age of coaches in a sports club. -/
def avg_age_coaches (club : SportsClub) : ℚ :=
  let total_age := club.total_members * club.avg_age
  let men_age := club.men * club.avg_age_men
  let women_age := club.women * club.avg_age_women
  (total_age - men_age - women_age) / club.coaches

/-- Theorem stating that for the given sports club, the average age of coaches is 23. -/
theorem avg_age_coaches_is_23 (club : SportsClub)
  (h1 : club.total_members = 50)
  (h2 : club.avg_age = 20)
  (h3 : club.men = 30)
  (h4 : club.women = 15)
  (h5 : club.coaches = 5)
  (h6 : club.avg_age_men = 19)
  (h7 : club.avg_age_women = 21) :
  avg_age_coaches club = 23 := by
  sorry

#eval avg_age_coaches {
  total_members := 50,
  men := 30,
  women := 15,
  coaches := 5,
  avg_age := 20,
  avg_age_men := 19,
  avg_age_women := 21
}

end NUMINAMATH_CALUDE_avg_age_coaches_is_23_l2206_220664


namespace NUMINAMATH_CALUDE_largest_quantity_l2206_220657

theorem largest_quantity (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5) :
  c > a ∧ c > b ∧ c > d :=
by sorry

end NUMINAMATH_CALUDE_largest_quantity_l2206_220657


namespace NUMINAMATH_CALUDE_square_of_difference_of_square_roots_l2206_220699

theorem square_of_difference_of_square_roots : 
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3))^2 = 10 + 2 * Complex.I * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_square_of_difference_of_square_roots_l2206_220699


namespace NUMINAMATH_CALUDE_sugar_for_muffins_sugar_for_muffins_proof_l2206_220682

/-- Given that 45 muffins require 3 cups of sugar, 
    prove that 135 muffins require 9 cups of sugar. -/
theorem sugar_for_muffins : ℝ → ℝ → ℝ → Prop :=
  fun muffins_base sugar_base muffins_target =>
    (muffins_base = 45 ∧ sugar_base = 3) →
    (muffins_target = 135) →
    (muffins_target * sugar_base / muffins_base = 9)

/-- Proof of the theorem -/
theorem sugar_for_muffins_proof : sugar_for_muffins 45 3 135 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_muffins_sugar_for_muffins_proof_l2206_220682


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l2206_220622

-- Define the point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (m + 3, m - 1)

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_fourth_quadrant_m_range :
  ∀ m : ℝ, in_fourth_quadrant (P m) ↔ -3 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_m_range_l2206_220622


namespace NUMINAMATH_CALUDE_problem_solution_l2206_220600

theorem problem_solution (x y : ℝ) 
  (h1 : 5^2 = x - 5)
  (h2 : (x + y)^(1/3) = 3) :
  x = 30 ∧ y = -3 ∧ Real.sqrt (x + 2*y) = 2 * Real.sqrt 6 ∨ Real.sqrt (x + 2*y) = -2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2206_220600


namespace NUMINAMATH_CALUDE_divisor_with_remainder_one_l2206_220655

theorem divisor_with_remainder_one (n : ℕ) : 
  ∃ k : ℕ, 2^200 - 3 = k * (2^100 - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_with_remainder_one_l2206_220655


namespace NUMINAMATH_CALUDE_number_of_men_l2206_220665

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 20 : ℝ) = W / ((M - 4) * 25 : ℝ)) → M = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_l2206_220665


namespace NUMINAMATH_CALUDE_ellipse_k_value_l2206_220659

-- Define the ellipse equation
def ellipse_equation (k : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + k * y^2 = 4

-- Define the focus point
def focus : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem ellipse_k_value :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), ellipse_equation k x y → 
      ∃ (c : ℝ), c^2 = (4/k) - 1 ∧ c = 1) →
    k = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l2206_220659


namespace NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l2206_220673

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There are infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l2206_220673


namespace NUMINAMATH_CALUDE_product_evaluation_l2206_220690

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2206_220690


namespace NUMINAMATH_CALUDE_min_value_theorem_l2206_220634

theorem min_value_theorem (x : ℝ) (h : x > 5) : x + 1 / (x - 5) ≥ 7 ∧ ∃ y > 5, y + 1 / (y - 5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2206_220634


namespace NUMINAMATH_CALUDE_tangent_perpendicular_condition_l2206_220696

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0
def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

-- Define the condition for perpendicular tangent lines
def perpendicular_tangents (a m n : ℝ) : Prop :=
  (n + 2) / m * (n + 1) / (m - (1 - a)) = -1

-- Define the theorem
theorem tangent_perpendicular_condition :
  ∃ (a : ℝ), a = -2 ∧
  ∀ (m n : ℝ), circle1 m n → circle2 a m n → perpendicular_tangents a m n :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_condition_l2206_220696


namespace NUMINAMATH_CALUDE_inverse_of_A_l2206_220649

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2206_220649


namespace NUMINAMATH_CALUDE_henry_seed_growth_l2206_220601

/-- Given that Henry starts with 5 seeds and triples his seeds each day, 
    this theorem proves that it takes 6 days to exceed 500 seeds. -/
theorem henry_seed_growth (n : ℕ) : n > 0 ∧ 5 * 3^(n-1) > 500 ↔ n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_henry_seed_growth_l2206_220601


namespace NUMINAMATH_CALUDE_factorization_problems_l2206_220648

theorem factorization_problems (m x n : ℝ) : 
  (m * x^2 - 2 * m^2 * x + m^3 = m * (x - m)^2) ∧ 
  (8 * m^2 * n + 2 * m * n = 2 * m * n * (4 * m + 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2206_220648


namespace NUMINAMATH_CALUDE_basketball_shooting_probability_unique_shot_probability_l2206_220685

/-- The probability of passing a basketball shooting test -/
def pass_probability : ℝ := 0.784

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.4

/-- The number of shooting opportunities -/
def max_attempts : ℕ := 3

/-- Theorem stating that the given shot probability results in the specified pass probability -/
theorem basketball_shooting_probability :
  shot_probability + (1 - shot_probability) * shot_probability + 
  (1 - shot_probability)^2 * shot_probability = pass_probability := by
  sorry

/-- Theorem stating that the shot probability is the unique solution -/
theorem unique_shot_probability :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 →
  (p + (1 - p) * p + (1 - p)^2 * p = pass_probability) →
  p = shot_probability := by
  sorry

end NUMINAMATH_CALUDE_basketball_shooting_probability_unique_shot_probability_l2206_220685


namespace NUMINAMATH_CALUDE_entire_square_shaded_l2206_220698

/-- The fraction of area shaded in the first step -/
def initial_shaded : ℚ := 5 / 9

/-- The fraction of area remaining unshaded after each step -/
def unshaded_fraction : ℚ := 4 / 9

/-- The sum of the infinite geometric series representing the total shaded area -/
def total_shaded_area : ℚ := initial_shaded / (1 - unshaded_fraction)

/-- Theorem stating that the entire square is shaded in the limit -/
theorem entire_square_shaded : total_shaded_area = 1 := by sorry

end NUMINAMATH_CALUDE_entire_square_shaded_l2206_220698


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l2206_220618

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l2206_220618


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2206_220647

theorem quadratic_roots_relation (m n p : ℝ) : 
  (∀ r : ℝ, (r^2 + p*r + m = 0) → ((3*r)^2 + m*(3*r) + n = 0)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2206_220647


namespace NUMINAMATH_CALUDE_line_y_intercept_l2206_220602

/-- A line with slope 3 and x-intercept (-4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = 3 ∧ x_intercept = (-4, 0) →
  ∃ (y : ℝ), (∀ (x : ℝ), y = slope * x + (slope * x_intercept.1 + x_intercept.2)) ∧
              y = slope * 0 + (slope * x_intercept.1 + x_intercept.2) ∧
              (0, y) = (0, 12) :=
by sorry


end NUMINAMATH_CALUDE_line_y_intercept_l2206_220602


namespace NUMINAMATH_CALUDE_tenth_student_problems_l2206_220677

theorem tenth_student_problems (n : ℕ) : 
  -- Total number of students
  (10 : ℕ) > 0 →
  -- Each problem is solved by exactly 7 students
  ∃ p : ℕ, p > 0 ∧ (7 * p = 36 + n) →
  -- First 9 students each solved 4 problems
  (9 * 4 = 36) →
  -- The number of problems solved by the tenth student is n
  n ≤ p →
  -- Conclusion: The tenth student solved 6 problems
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_tenth_student_problems_l2206_220677


namespace NUMINAMATH_CALUDE_unique_base_solution_l2206_220627

/-- Convert a base-6 number to decimal --/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Convert a number in base b to decimal --/
def baseBToDecimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The unique positive solution to 35₆ = 151ᵦ --/
theorem unique_base_solution : 
  ∃! (b : ℕ), b > 0 ∧ base6ToDecimal 35 = baseBToDecimal 151 b := by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l2206_220627


namespace NUMINAMATH_CALUDE_exactly_three_imply_l2206_220608

open Classical

variables (p q r : Prop)

def statement1 : Prop := ¬p ∧ q ∧ ¬r
def statement2 : Prop := p ∧ ¬q ∧ ¬r
def statement3 : Prop := ¬p ∧ ¬q ∧ r
def statement4 : Prop := p ∧ q ∧ ¬r

def implication : Prop := (¬p → ¬q) → ¬r

theorem exactly_three_imply :
  ∃! (n : Nat), n = 3 ∧
  (n = (if statement1 p q r → implication p q r then 1 else 0) +
       (if statement2 p q r → implication p q r then 1 else 0) +
       (if statement3 p q r → implication p q r then 1 else 0) +
       (if statement4 p q r → implication p q r then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_imply_l2206_220608


namespace NUMINAMATH_CALUDE_egg_roll_ratio_l2206_220683

def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4
def patrick_egg_rolls : ℕ := alvin_egg_rolls / 2

theorem egg_roll_ratio :
  matthew_egg_rolls / patrick_egg_rolls = 3 := by
  sorry

end NUMINAMATH_CALUDE_egg_roll_ratio_l2206_220683


namespace NUMINAMATH_CALUDE_condition_analysis_l2206_220679

theorem condition_analysis (x : ℝ) :
  (∀ x, -1 < x ∧ x < 3 → x^2 - 2*x < 8) ∧
  (∃ x, x^2 - 2*x < 8 ∧ ¬(-1 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l2206_220679


namespace NUMINAMATH_CALUDE_mn_value_l2206_220689

theorem mn_value (m n : ℕ+) (h : m.val^2 + n.val^2 + 4*m.val - 46 = 0) :
  m.val * n.val = 5 ∨ m.val * n.val = 15 := by
sorry

end NUMINAMATH_CALUDE_mn_value_l2206_220689


namespace NUMINAMATH_CALUDE_watch_payment_in_dimes_l2206_220693

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of cents in a dime -/
def cents_per_dime : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- Calculates the number of dimes needed to pay for an item given its cost in dollars -/
def dimes_needed (cost : ℕ) : ℕ :=
  (cost * cents_per_dollar) / cents_per_dime

theorem watch_payment_in_dimes :
  dimes_needed watch_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_watch_payment_in_dimes_l2206_220693


namespace NUMINAMATH_CALUDE_fraction_comparison_and_absolute_value_inequality_l2206_220658

theorem fraction_comparison_and_absolute_value_inequality :
  (-3 : ℚ) / 7 < (-2 : ℚ) / 5 ∧
  ∃ (a b : ℚ), |a + b| ≠ |a| + |b| :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_absolute_value_inequality_l2206_220658


namespace NUMINAMATH_CALUDE_cnc_machine_profit_l2206_220639

/-- Represents the profit function for a CNC machine -/
def profit_function (x : ℕ+) : ℤ := -2 * x.val ^ 2 + 40 * x.val - 98

/-- Represents when the machine starts generating profit -/
def profit_start : ℕ+ := 3

/-- Represents the year of maximum average annual profit -/
def max_avg_profit_year : ℕ+ := 7

/-- Represents the year of maximum total profit -/
def max_total_profit_year : ℕ+ := 10

/-- Theorem stating the properties of the CNC machine profit -/
theorem cnc_machine_profit :
  (∀ x : ℕ+, profit_function x = -2 * x.val ^ 2 + 40 * x.val - 98) ∧
  (∀ x : ℕ+, x < profit_start → profit_function x ≤ 0) ∧
  (∀ x : ℕ+, x ≥ profit_start → profit_function x > 0) ∧
  (∀ x : ℕ+, x ≠ max_avg_profit_year → 
    (profit_function x : ℚ) / x.val ≤ (profit_function max_avg_profit_year : ℚ) / max_avg_profit_year.val) ∧
  (∀ x : ℕ+, profit_function x ≤ profit_function max_total_profit_year) :=
by sorry

end NUMINAMATH_CALUDE_cnc_machine_profit_l2206_220639


namespace NUMINAMATH_CALUDE_range_of_b_l2206_220636

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ Real.sqrt (x^2 + y^2) + |x - 4| = 5}

-- Define the point B
def B (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the symmetry condition
def symmetricPoints (b : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 : ℝ × ℝ),
    p1 ∈ C ∧ p2 ∈ C ∧ p3 ∈ C ∧ p4 ∈ C ∧ p5 ∈ C ∧ p6 ∈ C ∧
    p1 ≠ p2 ∧ p3 ≠ p4 ∧ p5 ≠ p6 ∧
    (p1.1 + p2.1) / 2 = b ∧ (p3.1 + p4.1) / 2 = b ∧ (p5.1 + p6.1) / 2 = b

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, (∀ p ∈ C, Real.sqrt ((p.1)^2 + (p.2)^2) + |p.1 - 4| = 5) →
            symmetricPoints b →
            2 < b ∧ b < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l2206_220636


namespace NUMINAMATH_CALUDE_uncle_bob_parking_probability_l2206_220674

def parking_spaces : ℕ := 18
def parked_cars : ℕ := 15
def rv_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_arrangements := Nat.choose parking_spaces parked_cars
  let blocked_arrangements := Nat.choose (parking_spaces - rv_spaces + 1) (parked_cars - rv_spaces + 1)
  (total_arrangements - blocked_arrangements : ℚ) / total_arrangements = 16 / 51 := by
  sorry

end NUMINAMATH_CALUDE_uncle_bob_parking_probability_l2206_220674


namespace NUMINAMATH_CALUDE_star_composition_l2206_220684

-- Define the * operation
def star (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

-- Theorem statement
theorem star_composition (A B : Set α) : star A (star A B) = A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_star_composition_l2206_220684


namespace NUMINAMATH_CALUDE_truck_distance_l2206_220641

theorem truck_distance (truck_time car_time : ℝ) (speed_difference : ℝ) :
  truck_time = 8 →
  car_time = 5 →
  speed_difference = 18 →
  ∃ (truck_speed : ℝ),
    truck_speed * truck_time = (truck_speed + speed_difference) * car_time ∧
    truck_speed * truck_time = 240 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l2206_220641


namespace NUMINAMATH_CALUDE_taxi_journey_theorem_l2206_220653

def itinerary : List Int := [-15, 4, -5, 10, -12, 5, 8, -7]
def gasoline_consumption : Rat := 10 / 100  -- 10 liters per 100 km
def gasoline_price : Rat := 8  -- 8 yuan per liter

def total_distance (route : List Int) : Int :=
  route.map (Int.natAbs) |>.sum

theorem taxi_journey_theorem :
  let distance := total_distance itinerary
  let cost := (distance : Rat) * gasoline_consumption * gasoline_price
  distance = 66 ∧ cost = 52.8 := by sorry

end NUMINAMATH_CALUDE_taxi_journey_theorem_l2206_220653


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2206_220611

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2206_220611


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2206_220617

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 20 15 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2206_220617


namespace NUMINAMATH_CALUDE_cycle_gain_percentage_l2206_220694

def cycleA_cp : ℚ := 1000
def cycleB_cp : ℚ := 3000
def cycleC_cp : ℚ := 6000

def cycleB_discount_rate : ℚ := 10 / 100
def cycleC_tax_rate : ℚ := 5 / 100

def cycleA_sp : ℚ := 2000
def cycleB_sp : ℚ := 4500
def cycleC_sp : ℚ := 8000

def cycleA_sales_tax_rate : ℚ := 5 / 100
def cycleB_selling_discount_rate : ℚ := 8 / 100

def total_cp : ℚ := cycleA_cp + cycleB_cp * (1 - cycleB_discount_rate) + cycleC_cp * (1 + cycleC_tax_rate)
def total_sp : ℚ := cycleA_sp * (1 + cycleA_sales_tax_rate) + cycleB_sp * (1 - cycleB_selling_discount_rate) + cycleC_sp

def overall_gain : ℚ := total_sp - total_cp
def gain_percentage : ℚ := (overall_gain / total_cp) * 100

theorem cycle_gain_percentage :
  gain_percentage = 42.4 := by sorry

end NUMINAMATH_CALUDE_cycle_gain_percentage_l2206_220694


namespace NUMINAMATH_CALUDE_number_of_teachers_l2206_220605

/-- Represents the total number of people (teachers and students) in the school -/
def total : ℕ := 2400

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school based on the given information -/
def teachers : ℕ := total - (total * students_in_sample / sample_size)

/-- Theorem stating that the number of teachers in the school is 150 -/
theorem number_of_teachers : teachers = 150 := by sorry

end NUMINAMATH_CALUDE_number_of_teachers_l2206_220605


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l2206_220623

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l2206_220623


namespace NUMINAMATH_CALUDE_radio_station_survey_l2206_220626

theorem radio_station_survey (males_dont_listen : ℕ) (females_listen : ℕ) 
  (total_listeners : ℕ) (total_non_listeners : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listeners = 180)
  (h4 : total_non_listeners = 120) :
  total_listeners - females_listen = 105 := by
  sorry

end NUMINAMATH_CALUDE_radio_station_survey_l2206_220626


namespace NUMINAMATH_CALUDE_sample_size_calculation_l2206_220680

theorem sample_size_calculation (total_population : ℕ) (sampling_rate : ℚ) :
  total_population = 2000 →
  sampling_rate = 1/10 →
  (total_population : ℚ) * sampling_rate = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l2206_220680


namespace NUMINAMATH_CALUDE_median_triangle_theorem_l2206_220646

/-- Given a triangle ABC with area 1 and medians s_a, s_b, s_c, there exists a triangle
    with sides s_a, s_b, s_c, and its area is 4/3 times the area of triangle ABC. -/
theorem median_triangle_theorem (A B C : ℝ × ℝ) (s_a s_b s_c : ℝ) :
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  let median_a := ((B.1 + C.1) / 2 - A.1, (B.2 + C.2) / 2 - A.2)
  let median_b := ((A.1 + C.1) / 2 - B.1, (A.2 + C.2) / 2 - B.2)
  let median_c := ((A.1 + B.1) / 2 - C.1, (A.2 + B.2) / 2 - C.2)
  triangle_area = 1 ∧
  s_a = Real.sqrt (median_a.1^2 + median_a.2^2) ∧
  s_b = Real.sqrt (median_b.1^2 + median_b.2^2) ∧
  s_c = Real.sqrt (median_c.1^2 + median_c.2^2) →
  ∃ (D E F : ℝ × ℝ),
    let new_triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
    Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = s_a ∧
    Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = s_b ∧
    Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = s_c ∧
    new_triangle_area = 4/3 * triangle_area := by
  sorry


end NUMINAMATH_CALUDE_median_triangle_theorem_l2206_220646


namespace NUMINAMATH_CALUDE_sum_difference_is_60_l2206_220625

def sum_even_2_to_120 : ℕ := (Finset.range 60).sum (fun i => 2 * (i + 1))

def sum_odd_1_to_119 : ℕ := (Finset.range 60).sum (fun i => 2 * i + 1)

theorem sum_difference_is_60 : sum_even_2_to_120 - sum_odd_1_to_119 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_60_l2206_220625


namespace NUMINAMATH_CALUDE_fruit_baskets_count_l2206_220607

/-- The number of non-empty fruit baskets -/
def num_fruit_baskets (num_apples num_oranges : ℕ) : ℕ :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem: The number of non-empty fruit baskets with 6 apples and 12 oranges is 90 -/
theorem fruit_baskets_count :
  num_fruit_baskets 6 12 = 90 := by
sorry

end NUMINAMATH_CALUDE_fruit_baskets_count_l2206_220607


namespace NUMINAMATH_CALUDE_students_playing_neither_l2206_220671

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 38 →
  football = 26 →
  tennis = 20 →
  both = 17 →
  total - (football + tennis - both) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l2206_220671


namespace NUMINAMATH_CALUDE_sandwich_percentage_l2206_220645

theorem sandwich_percentage (total_weight : ℝ) (condiment_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_percentage_l2206_220645


namespace NUMINAMATH_CALUDE_monkey_percentage_after_eating_l2206_220610

/-- The percentage of monkeys among animals after two monkeys each eat one bird -/
theorem monkey_percentage_after_eating (initial_monkeys initial_birds : ℕ) 
  (h1 : initial_monkeys = 6)
  (h2 : initial_birds = 6)
  (h3 : initial_monkeys > 0)
  (h4 : initial_birds ≥ 2) : 
  (initial_monkeys : ℚ) / (initial_monkeys + initial_birds - 2 : ℚ) = 3/5 := by
  sorry

#check monkey_percentage_after_eating

end NUMINAMATH_CALUDE_monkey_percentage_after_eating_l2206_220610


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l2206_220621

theorem simplify_sqrt_sum (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) = 2 * (|x - 1| + |x + 1|) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l2206_220621


namespace NUMINAMATH_CALUDE_triangle_area_l2206_220603

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem triangle_area (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  0 < C ∧ C < π / 2 →
  f 1 C = Real.sqrt 3 →
  c = 3 →
  Real.sin B = 2 * Real.sin A →
  (1 / 2 : ℝ) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2206_220603


namespace NUMINAMATH_CALUDE_complex_dot_product_l2206_220642

theorem complex_dot_product (z : ℂ) (h1 : Complex.abs z = Real.sqrt 2) (h2 : Complex.im (z^2) = 2) :
  (z + z^2) • (z - z^2) = -2 ∨ (z + z^2) • (z - z^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_dot_product_l2206_220642


namespace NUMINAMATH_CALUDE_coin_stacking_arrangements_l2206_220615

/-- Represents the number of ways to arrange n indistinguishable objects in k positions -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Represents the number of valid orientation arrangements -/
def validOrientations : ℕ := 9

/-- Represents the total number of gold coins -/
def goldCoins : ℕ := 5

/-- Represents the total number of silver coins -/
def silverCoins : ℕ := 3

/-- Represents the total number of coins -/
def totalCoins : ℕ := goldCoins + silverCoins

/-- The number of distinguishable arrangements for stacking coins -/
def distinguishableArrangements : ℕ := 
  binomial totalCoins goldCoins * validOrientations

theorem coin_stacking_arrangements :
  distinguishableArrangements = 504 := by sorry

end NUMINAMATH_CALUDE_coin_stacking_arrangements_l2206_220615


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2206_220667

/-- The lateral area of a cone with specific properties -/
theorem cone_lateral_area (base_diameter : ℝ) (slant_height : ℝ) :
  base_diameter = 6 →
  slant_height = 6 →
  (1 / 2) * (2 * Real.pi) * (base_diameter / 2) * slant_height = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l2206_220667


namespace NUMINAMATH_CALUDE_second_machine_rate_l2206_220630

/-- Represents a copy machine with a constant rate of copies per minute -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Represents two copy machines working together -/
structure TwoMachines where
  machine1 : CopyMachine
  machine2 : CopyMachine

/-- The total number of copies produced by two machines in a given time -/
def total_copies (machines : TwoMachines) (minutes : ℕ) : ℕ :=
  (machines.machine1.copies_per_minute + machines.machine2.copies_per_minute) * minutes

theorem second_machine_rate (machines : TwoMachines) 
  (h1 : machines.machine1.copies_per_minute = 25)
  (h2 : total_copies machines 30 = 2400) :
  machines.machine2.copies_per_minute = 55 := by
sorry

end NUMINAMATH_CALUDE_second_machine_rate_l2206_220630


namespace NUMINAMATH_CALUDE_sixteen_points_configuration_unique_configuration_l2206_220678

/-- Represents a configuration of points on a line -/
structure LineConfiguration where
  totalPoints : ℕ
  pointA : ℕ
  pointB : ℕ

/-- Counts the number of segments that contain a given point -/
def segmentsContainingPoint (config : LineConfiguration) (point : ℕ) : ℕ :=
  (point - 1) * (config.totalPoints - point)

/-- The main theorem stating that the configuration with 16 points satisfies the given conditions -/
theorem sixteen_points_configuration :
  ∃ (config : LineConfiguration),
    config.totalPoints = 16 ∧
    segmentsContainingPoint config config.pointA = 50 ∧
    segmentsContainingPoint config config.pointB = 56 := by
  sorry

/-- Uniqueness theorem: there is only one configuration satisfying the conditions -/
theorem unique_configuration (config1 config2 : LineConfiguration) :
  segmentsContainingPoint config1 config1.pointA = 50 →
  segmentsContainingPoint config1 config1.pointB = 56 →
  segmentsContainingPoint config2 config2.pointA = 50 →
  segmentsContainingPoint config2 config2.pointB = 56 →
  config1.totalPoints = config2.totalPoints := by
  sorry

end NUMINAMATH_CALUDE_sixteen_points_configuration_unique_configuration_l2206_220678


namespace NUMINAMATH_CALUDE_adult_meal_cost_l2206_220614

theorem adult_meal_cost 
  (total_people : ℕ) 
  (kids : ℕ) 
  (total_cost : ℚ) 
  (h1 : total_people = 11) 
  (h2 : kids = 2) 
  (h3 : total_cost = 72) : 
  (total_cost / (total_people - kids) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_adult_meal_cost_l2206_220614


namespace NUMINAMATH_CALUDE_gcd_product_l2206_220681

theorem gcd_product (a b A B : ℕ) (d D : ℕ+) :
  d = Nat.gcd a b →
  D = Nat.gcd A B →
  (d * D : ℕ) = Nat.gcd (a * A) (Nat.gcd (a * B) (Nat.gcd (b * A) (b * B))) :=
by sorry

end NUMINAMATH_CALUDE_gcd_product_l2206_220681


namespace NUMINAMATH_CALUDE_division_relation_l2206_220643

theorem division_relation : 
  (29.94 / 1.45 = 17.9) → (2994 / 14.5 = 1790) := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l2206_220643


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2206_220650

/-- An arithmetic sequence with a_4 = 3 and a_12 = 19 has a common difference of 2 -/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence condition
  a 4 = 3 →
  a 12 = 19 →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2206_220650


namespace NUMINAMATH_CALUDE_machine_present_value_l2206_220691

/-- The present value of a machine given its future value and depreciation rate -/
theorem machine_present_value (future_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) 
  (h1 : future_value = 810)
  (h2 : depreciation_rate = 0.1)
  (h3 : years = 2) :
  future_value = 1000 * (1 - depreciation_rate) ^ years := by
  sorry

end NUMINAMATH_CALUDE_machine_present_value_l2206_220691


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2206_220620

theorem arithmetic_computation : (143 + 29) * 2 + 25 + 13 = 382 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2206_220620


namespace NUMINAMATH_CALUDE_sum_first_six_primes_gt_10_l2206_220670

def first_six_primes_gt_10 : List Nat :=
  [11, 13, 17, 19, 23, 29]

theorem sum_first_six_primes_gt_10 :
  first_six_primes_gt_10.sum = 112 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_gt_10_l2206_220670


namespace NUMINAMATH_CALUDE_probability_four_students_same_group_l2206_220686

theorem probability_four_students_same_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 800) 
  (h2 : num_groups = 4) 
  (h3 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups^3) = 1/64 :=
sorry

end NUMINAMATH_CALUDE_probability_four_students_same_group_l2206_220686


namespace NUMINAMATH_CALUDE_travel_ways_count_l2206_220692

/-- The number of highways from A to B -/
def num_highways : ℕ := 3

/-- The number of railways from A to B -/
def num_railways : ℕ := 2

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := num_highways + num_railways

theorem travel_ways_count : total_ways = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_count_l2206_220692


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l2206_220629

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_sum_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l2206_220629


namespace NUMINAMATH_CALUDE_cone_slant_height_l2206_220619

/-- Given a cone with base radius 1 and lateral surface that unfolds into a semicircle,
    prove that its slant height is 2. -/
theorem cone_slant_height (r : ℝ) (s : ℝ) (h1 : r = 1) (h2 : π * s = 2 * π * r) : s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2206_220619


namespace NUMINAMATH_CALUDE_negation_of_existence_square_plus_one_positive_negation_l2206_220628

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem square_plus_one_positive_negation :
  (¬∃ x : ℝ, x^2 + 1 > 0) ↔ (∀ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_plus_one_positive_negation_l2206_220628


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l2206_220675

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l2206_220675


namespace NUMINAMATH_CALUDE_intersection_quadratic_equations_l2206_220663

theorem intersection_quadratic_equations (p q : ℝ) : 
  let M := {x : ℝ | x^2 - p*x + 6 = 0}
  let N := {x : ℝ | x^2 + 6*x - q = 0}
  (M ∩ N = {2}) → p + q = 21 := by
sorry

end NUMINAMATH_CALUDE_intersection_quadratic_equations_l2206_220663


namespace NUMINAMATH_CALUDE_cost_per_minute_advertising_l2206_220616

/-- The cost of one minute of advertising during a race, given the number of advertisements,
    duration of each advertisement, and total cost of transmission. -/
theorem cost_per_minute_advertising (num_ads : ℕ) (duration_per_ad : ℕ) (total_cost : ℕ) :
  num_ads = 5 →
  duration_per_ad = 3 →
  total_cost = 60000 →
  total_cost / (num_ads * duration_per_ad) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_minute_advertising_l2206_220616


namespace NUMINAMATH_CALUDE_unique_solution_l2206_220606

theorem unique_solution : ∀ a b : ℕ+,
  (¬ (7 ∣ (a * b * (a + b)))) →
  ((7^7) ∣ ((a + b)^7 - a^7 - b^7)) →
  (a = 18 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2206_220606


namespace NUMINAMATH_CALUDE_some_trinks_not_zorbs_l2206_220638

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Zorb, Glarb, and Trink
variable (Zorb Glarb Trink : U → Prop)

-- Hypothesis I: All Zorbs are not Glarbs
variable (h1 : ∀ x, Zorb x → ¬Glarb x)

-- Hypothesis II: Some Glarbs are Trinks
variable (h2 : ∃ x, Glarb x ∧ Trink x)

-- Theorem: Some Trinks are not Zorbs
theorem some_trinks_not_zorbs :
  ∃ x, Trink x ∧ ¬Zorb x :=
sorry

end NUMINAMATH_CALUDE_some_trinks_not_zorbs_l2206_220638


namespace NUMINAMATH_CALUDE_largest_choir_size_l2206_220609

theorem largest_choir_size : 
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2 + 11) ∧ 
    (∃ (m : ℕ), n = m * (m + 5)) ∧
    (∀ (x : ℕ), 
      ((∃ (k : ℕ), x = k^2 + 11) ∧ 
       (∃ (m : ℕ), x = m * (m + 5))) → 
      x ≤ n) ∧
    n = 325 :=
by sorry

end NUMINAMATH_CALUDE_largest_choir_size_l2206_220609


namespace NUMINAMATH_CALUDE_inequality_range_l2206_220612

theorem inequality_range (x y : ℝ) :
  y - x^2 < Real.sqrt (x^2) →
  ((x ≥ 0 → y < x + x^2) ∧ (x < 0 → y < -x + x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2206_220612


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_product_l2206_220662

theorem gcd_lcm_sum_product (a b : ℕ) (ha : a = 8) (hb : b = 12) :
  (Nat.gcd a b + Nat.lcm a b) * Nat.gcd a b = 112 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_product_l2206_220662


namespace NUMINAMATH_CALUDE_inner_automorphism_is_automorphism_l2206_220676

variable {G : Type*} [Group G]

def inner_automorphism (x : G) (y : G) : G := x⁻¹ * y * x

theorem inner_automorphism_is_automorphism (x : G) :
  Function.Bijective (inner_automorphism x) ∧
  ∀ y z : G, inner_automorphism x (y * z) = inner_automorphism x y * inner_automorphism x z :=
sorry

end NUMINAMATH_CALUDE_inner_automorphism_is_automorphism_l2206_220676


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l2206_220640

/-- The number of ways to arrange 3 girls and 5 boys in a row -/
def arrangement_count (n_girls : ℕ) (n_boys : ℕ) : ℕ × ℕ × ℕ :=
  let total := n_girls + n_boys
  let adjacent := (Nat.factorial n_girls) * (Nat.factorial (total - n_girls + 1))
  let not_adjacent := (Nat.factorial n_boys) * (Nat.choose (n_boys + 1) n_girls) * (Nat.factorial n_girls)
  let boys_fixed := Nat.choose total n_girls
  (adjacent, not_adjacent, boys_fixed)

/-- Theorem stating the correct number of arrangements for 3 girls and 5 boys -/
theorem correct_arrangement_count :
  arrangement_count 3 5 = (4320, 14400, 336) := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l2206_220640


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l2206_220631

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + 1| - |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 4 x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Part II
theorem range_of_a_part_ii :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 2 3, f a x ≥ |x - 4|) → a ∈ Set.Icc (-1) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l2206_220631


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2206_220651

/-- Given a solution with initial volume and salt percentage, prove that
    adding water to reach a specific salt percentage results in the correct
    initial salt percentage. -/
theorem salt_solution_problem (initial_volume : ℝ) (water_added : ℝ) 
    (final_salt_percentage : ℝ) (initial_salt_percentage : ℝ) : 
    initial_volume = 64 →
    water_added = 16 →
    final_salt_percentage = 0.08 →
    initial_salt_percentage * initial_volume = 
      final_salt_percentage * (initial_volume + water_added) →
    initial_salt_percentage = 0.1 := by
  sorry

#check salt_solution_problem

end NUMINAMATH_CALUDE_salt_solution_problem_l2206_220651


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2206_220613

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -5 →
    1 / (x^3 + 2*x^2 - 25*x - 50) = A / (x - 2) + B / (x + 5) + C / ((x + 5)^2)) →
  B = -11/490 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l2206_220613


namespace NUMINAMATH_CALUDE_strawberries_picked_l2206_220637

/-- Given that Paul started with 28 strawberries and ended up with 63 strawberries,
    prove that he picked 35 strawberries. -/
theorem strawberries_picked (initial : ℕ) (final : ℕ) (h1 : initial = 28) (h2 : final = 63) :
  final - initial = 35 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_picked_l2206_220637


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l2206_220632

theorem sqrt_four_squared : Real.sqrt (4^2) = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l2206_220632
