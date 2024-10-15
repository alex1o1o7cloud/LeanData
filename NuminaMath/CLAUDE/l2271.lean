import Mathlib

namespace NUMINAMATH_CALUDE_madeline_pencils_l2271_227109

theorem madeline_pencils (cyrus_pencils : ℕ) 
  (h1 : cyrus_pencils + 3 * cyrus_pencils + 3 * cyrus_pencils / 2 = 231) : 
  3 * cyrus_pencils / 2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_madeline_pencils_l2271_227109


namespace NUMINAMATH_CALUDE_company_profit_l2271_227146

theorem company_profit (a : ℝ) : 
  let october_profit := a
  let november_decrease := 0.06
  let december_increase := 0.10
  let november_profit := october_profit * (1 - november_decrease)
  let december_profit := november_profit * (1 + december_increase)
  december_profit = (1 - 0.06) * (1 + 0.10) * a := by
sorry

end NUMINAMATH_CALUDE_company_profit_l2271_227146


namespace NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2271_227186

/-- An ellipse intersecting with a line -/
structure EllipseIntersection where
  m : ℝ
  n : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  ellipse_eq₁ : m * x₁^2 + n * y₁^2 = 1
  ellipse_eq₂ : m * x₂^2 + n * y₂^2 = 1
  line_eq₁ : y₁ = 1 - x₁
  line_eq₂ : y₂ = 1 - x₂

/-- The theorem stating the relationship between m and n -/
theorem ellipse_intersection_ratio (e : EllipseIntersection) 
  (h : (y₁ + y₂) / (x₁ + x₂) = Real.sqrt 2) : 
  e.m / e.n = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_ratio_l2271_227186


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2271_227176

def f (x : ℝ) : ℝ := x^2

def B : Set ℝ := {1, 4}

theorem intersection_of_A_and_B (A : Set ℝ) (h : ∀ x ∈ A, f x ∈ B) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2271_227176


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l2271_227178

/-- Given points A and B, and a point C on the line extension of AB such that BC = 1/2 * AB, 
    prove that C has the specified coordinates. -/
theorem extended_segment_coordinates (A B C : ℝ × ℝ) : 
  A = (3, -3) → 
  B = (15, 3) → 
  C - B = (1/2 : ℝ) • (B - A) → 
  C = (21, 6) := by
sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l2271_227178


namespace NUMINAMATH_CALUDE_problem_solution_l2271_227171

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2271_227171


namespace NUMINAMATH_CALUDE_matrix_inverse_problem_l2271_227173

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 2, 1]

theorem matrix_inverse_problem :
  (∃ (B : Matrix (Fin 2) (Fin 2) ℚ), A * B = 1 ∧ B * A = 1) →
  (A⁻¹ = !![1/11, 3/11; -2/11, 5/11]) ∨
  (¬∃ (B : Matrix (Fin 2) (Fin 2) ℚ), A * B = 1 ∧ B * A = 1) →
  (A⁻¹ = 0) :=
sorry

end NUMINAMATH_CALUDE_matrix_inverse_problem_l2271_227173


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2271_227158

theorem inequality_solution_set (x : ℝ) :
  (x + 5) / (x - 1) ≥ 2 ↔ x ∈ Set.Ioo 1 7 ∪ {7} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2271_227158


namespace NUMINAMATH_CALUDE_sum_less_than_sum_of_roots_l2271_227136

theorem sum_less_than_sum_of_roots (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : |a - b| < 2) (hbc : |b - c| < 2) (hca : |c - a| < 2) :
  a + b + c < Real.sqrt (a * b + 1) + Real.sqrt (a * c + 1) + Real.sqrt (b * c + 1) := by
  sorry


end NUMINAMATH_CALUDE_sum_less_than_sum_of_roots_l2271_227136


namespace NUMINAMATH_CALUDE_friends_total_points_l2271_227117

/-- The total points scored by four friends in table football games -/
def total_points (darius matt marius sofia : ℕ) : ℕ :=
  darius + matt + marius + sofia

/-- Theorem stating the total points scored by the four friends -/
theorem friends_total_points :
  ∀ (darius matt marius sofia : ℕ),
    darius = 10 →
    marius = darius + 3 →
    darius = matt - 5 →
    sofia = 2 * matt →
    total_points darius matt marius sofia = 68 := by
  sorry

#check friends_total_points

end NUMINAMATH_CALUDE_friends_total_points_l2271_227117


namespace NUMINAMATH_CALUDE_complex_equation_l2271_227166

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given that m/(1+i) = 1 - ni, where m and n are real numbers,
    prove that m + ni = 2 + i -/
theorem complex_equation (m n : ℝ) (h : (m : ℂ) / (1 + i) = 1 - n * i) :
  (m : ℂ) + n * i = 2 + i := by sorry

end NUMINAMATH_CALUDE_complex_equation_l2271_227166


namespace NUMINAMATH_CALUDE_largest_angle_in_quadrilateral_with_ratio_l2271_227107

/-- 
Given a quadrilateral divided into two triangles by a diagonal,
with the measures of the angles around this diagonal in the ratio 2:3:4:5,
prove that the largest of these angles is 900°/7.
-/
theorem largest_angle_in_quadrilateral_with_ratio (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Angles are positive
  a + b + c + d = 360 →  -- Sum of angles around a point is 360°
  ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 4*x ∧ d = 5*x →  -- Angles are in ratio 2:3:4:5
  (max a (max b (max c d))) = 900 / 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_quadrilateral_with_ratio_l2271_227107


namespace NUMINAMATH_CALUDE_min_q_value_l2271_227151

/-- The number of cards in the deck -/
def num_cards : ℕ := 52

/-- The probability function q(a) -/
def q (a : ℕ) : ℚ :=
  let total_combinations := (num_cards - 2).choose 2
  let lower_team := (num_cards - (a + 11) - 1).choose 2
  let higher_team := (a - 1).choose 2
  (lower_team + higher_team : ℚ) / total_combinations

/-- The minimum value of a for which q(a) ≥ 1/2 -/
def min_a : ℕ := 4

/-- The theorem stating the minimum value of q(a) ≥ 1/2 -/
theorem min_q_value :
  q min_a = 91 / 175 ∧ 
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ num_cards - 11 → q a ≥ 1 / 2 → q a ≥ q min_a :=
sorry

end NUMINAMATH_CALUDE_min_q_value_l2271_227151


namespace NUMINAMATH_CALUDE_spending_recorded_as_negative_l2271_227185

/-- Represents a WeChat payment record -/
structure WeChatPayment where
  amount : ℝ

/-- Records a receipt in WeChat payment system -/
def recordReceipt (x : ℝ) : WeChatPayment :=
  ⟨x⟩

/-- Records spending in WeChat payment system -/
def recordSpending (x : ℝ) : WeChatPayment :=
  ⟨-x⟩

theorem spending_recorded_as_negative :
  recordSpending 10.6 = WeChatPayment.mk (-10.6) := by
  sorry

end NUMINAMATH_CALUDE_spending_recorded_as_negative_l2271_227185


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2271_227181

theorem arithmetic_sequence_proof :
  ∀ (a n d : ℤ),
  a = -7 →
  n = 3 →
  d = 5 →
  (a + (n - 1) * d = n) ∧
  (n * (2 * a + (n - 1) * d) / 2 = -6) :=
λ a n d h1 h2 h3 =>
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2271_227181


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2271_227194

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 1 = 0 →
  q^3 - 8*q^2 + 10*q - 1 = 0 →
  r^3 - 8*r^2 + 10*r - 1 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 113/20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2271_227194


namespace NUMINAMATH_CALUDE_sin_315_degrees_l2271_227139

theorem sin_315_degrees : Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l2271_227139


namespace NUMINAMATH_CALUDE_cyclists_meet_time_l2271_227196

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 C : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : C = 600) :
  C / (v1 + v2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meet_time_l2271_227196


namespace NUMINAMATH_CALUDE_not_complete_residue_sum_l2271_227137

theorem not_complete_residue_sum (n : ℕ) (a b : Fin n → ℕ) : 
  Even n →
  (∀ k : Fin n, ∃ i : Fin n, a i ≡ k [ZMOD n]) →
  (∀ k : Fin n, ∃ i : Fin n, b i ≡ k [ZMOD n]) →
  ¬(∀ k : Fin n, ∃ i : Fin n, (a i + b i) ≡ k [ZMOD n]) :=
by sorry

end NUMINAMATH_CALUDE_not_complete_residue_sum_l2271_227137


namespace NUMINAMATH_CALUDE_smallest_num_prime_factors_l2271_227199

/-- Given a list of positive integers, returns true if the GCDs of all nonempty subsets are pairwise distinct -/
def has_distinct_gcds (nums : List Nat) : Prop := sorry

/-- Returns the number of prime factors of a natural number -/
def num_prime_factors (n : Nat) : Nat := sorry

theorem smallest_num_prime_factors (N : Nat) (nums : List Nat) 
  (h1 : nums.length = N)
  (h2 : ∀ n ∈ nums, n > 0)
  (h3 : has_distinct_gcds nums) :
  (N = 1 ∧ num_prime_factors (nums.prod) = 0) ∨
  (N ≥ 2 ∧ num_prime_factors (nums.prod) = N) := by sorry

end NUMINAMATH_CALUDE_smallest_num_prime_factors_l2271_227199


namespace NUMINAMATH_CALUDE_solve_equation_l2271_227150

theorem solve_equation (x : ℝ) : 3 - 1 / (2 + x) = 2 * (1 / (2 + x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2271_227150


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2271_227174

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2271_227174


namespace NUMINAMATH_CALUDE_inequality_proof_l2271_227156

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2271_227156


namespace NUMINAMATH_CALUDE_alpha_range_l2271_227154

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ π) 
  (h3 : ∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) :
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l2271_227154


namespace NUMINAMATH_CALUDE_polyhedron_inequality_l2271_227115

-- Define a convex polyhedron
structure ConvexPolyhedron where
  edges : List ℝ
  dihedralAngles : List ℝ

-- Define the theorem
theorem polyhedron_inequality (R : ℝ) (P : ConvexPolyhedron) :
  R > 0 →
  P.edges.length = P.dihedralAngles.length →
  (List.sum (List.zipWith (λ l φ => l * (Real.pi - φ)) P.edges P.dihedralAngles)) ≤ 8 * Real.pi * R :=
by sorry

end NUMINAMATH_CALUDE_polyhedron_inequality_l2271_227115


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2271_227153

theorem oil_price_reduction (original_price : ℝ) : 
  (∃ (quantity : ℝ), 
    original_price * quantity = 800 ∧ 
    (0.75 * original_price) * (quantity + 5) = 800) →
  0.75 * original_price = 30 := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2271_227153


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2271_227125

/-- The probability of selecting a red ball from a bag with white and red balls -/
theorem probability_of_red_ball (white_balls red_balls : ℕ) : 
  white_balls = 3 → red_balls = 7 → 
  (red_balls : ℚ) / (white_balls + red_balls) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2271_227125


namespace NUMINAMATH_CALUDE_negative_two_m_cubed_squared_l2271_227131

theorem negative_two_m_cubed_squared (m : ℝ) : (-2 * m^3)^2 = 4 * m^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_m_cubed_squared_l2271_227131


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2271_227144

def total_players : ℕ := 14
def triplets : ℕ := 3
def starters : ℕ := 6

theorem volleyball_team_selection :
  (Nat.choose total_players starters) -
  ((Nat.choose triplets 2) * (Nat.choose (total_players - triplets) (starters - 2)) +
   (Nat.choose triplets 3) * (Nat.choose (total_players - triplets) (starters - 3))) = 1848 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2271_227144


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2271_227165

theorem system_of_equations_solution (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 20) 
  (eq2 : 4 * x + y = 25) : 
  (x + y)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2271_227165


namespace NUMINAMATH_CALUDE_boys_average_age_l2271_227116

theorem boys_average_age 
  (total_students : ℕ) 
  (girls_avg_age : ℝ) 
  (school_avg_age : ℝ) 
  (num_girls : ℕ) 
  (h1 : total_students = 604)
  (h2 : girls_avg_age = 11)
  (h3 : school_avg_age = 11.75)
  (h4 : num_girls = 151) :
  let num_boys : ℕ := total_students - num_girls
  let boys_total_age : ℝ := school_avg_age * total_students - girls_avg_age * num_girls
  boys_total_age / num_boys = 5411 / 453 :=
by sorry

end NUMINAMATH_CALUDE_boys_average_age_l2271_227116


namespace NUMINAMATH_CALUDE_oplus_inequality_l2271_227192

def oplus (x y : ℝ) : ℝ := (x - y)^2

theorem oplus_inequality : ∃ x y : ℝ, 2 * (oplus x y) ≠ oplus (2*x) (2*y) := by
  sorry

end NUMINAMATH_CALUDE_oplus_inequality_l2271_227192


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2271_227187

-- Define an isosceles triangle with one angle of 150°
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a = b ∧ c = 150

-- Theorem statement
theorem isosceles_triangle_base_angles 
  (a b c : ℝ) (h : IsoscelesTriangle a b c) : a = 15 ∧ b = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angles_l2271_227187


namespace NUMINAMATH_CALUDE_point_c_satisfies_inequalities_l2271_227161

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the given system of inequalities -/
def satisfiesInequalities (p : Point2D) : Prop :=
  p.x + p.y - 1 < 0 ∧ p.x - p.y + 1 > 0

theorem point_c_satisfies_inequalities :
  satisfiesInequalities ⟨0, -2⟩ ∧
  ¬satisfiesInequalities ⟨0, 2⟩ ∧
  ¬satisfiesInequalities ⟨-2, 0⟩ ∧
  ¬satisfiesInequalities ⟨2, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_point_c_satisfies_inequalities_l2271_227161


namespace NUMINAMATH_CALUDE_exactly_one_defective_probability_l2271_227168

theorem exactly_one_defective_probability
  (pass_rate_1 : ℝ)
  (pass_rate_2 : ℝ)
  (h1 : pass_rate_1 = 0.90)
  (h2 : pass_rate_2 = 0.95)
  : (pass_rate_1 * (1 - pass_rate_2)) + ((1 - pass_rate_1) * pass_rate_2) = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_defective_probability_l2271_227168


namespace NUMINAMATH_CALUDE_product_of_sines_equals_2_25_l2271_227113

theorem product_of_sines_equals_2_25 :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_2_25_l2271_227113


namespace NUMINAMATH_CALUDE_triangle_theorem_l2271_227127

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  -- Part 1: Angle C is 60 degrees (π/3 radians)
  t.C = π / 3 ∧
  -- Part 2: If c = 2, the maximum area is √3
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2271_227127


namespace NUMINAMATH_CALUDE_regression_sum_of_squares_l2271_227122

theorem regression_sum_of_squares 
  (SST : ℝ) (SSE : ℝ) (SSR : ℝ) 
  (h1 : SST = 256) 
  (h2 : SSE = 32) 
  (h3 : SSR = SST - SSE) : 
  SSR = 224 := by
  sorry

end NUMINAMATH_CALUDE_regression_sum_of_squares_l2271_227122


namespace NUMINAMATH_CALUDE_jean_spots_l2271_227101

/-- Represents the distribution of spots on a jaguar -/
structure JaguarSpots where
  total : ℕ
  upperTorso : ℕ
  backAndHindquarters : ℕ
  sides : ℕ

/-- Checks if the spot distribution is valid according to the given conditions -/
def isValidDistribution (spots : JaguarSpots) : Prop :=
  spots.upperTorso = spots.total / 2 ∧
  spots.backAndHindquarters = spots.total / 3 ∧
  spots.sides = spots.total - spots.upperTorso - spots.backAndHindquarters

theorem jean_spots (spots : JaguarSpots) 
  (h_valid : isValidDistribution spots) 
  (h_upperTorso : spots.upperTorso = 30) : 
  spots.sides = 10 := by
  sorry

#check jean_spots

end NUMINAMATH_CALUDE_jean_spots_l2271_227101


namespace NUMINAMATH_CALUDE_election_majority_l2271_227177

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 700 →
  winning_percentage = 84 / 100 →
  (winning_percentage * total_votes : ℚ).floor - 
  ((1 - winning_percentage) * total_votes : ℚ).floor = 476 := by
sorry

end NUMINAMATH_CALUDE_election_majority_l2271_227177


namespace NUMINAMATH_CALUDE_square_sum_and_difference_l2271_227123

/-- Given a = √3 - 2 and b = √3 + 2, prove that (a + b)² = 12 and a² - b² = -8√3 -/
theorem square_sum_and_difference (a b : ℝ) (ha : a = Real.sqrt 3 - 2) (hb : b = Real.sqrt 3 + 2) :
  (a + b)^2 = 12 ∧ a^2 - b^2 = -8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_square_sum_and_difference_l2271_227123


namespace NUMINAMATH_CALUDE_combined_salaries_l2271_227149

theorem combined_salaries (salary_C : ℕ) (average_salary : ℕ) (num_individuals : ℕ) :
  salary_C = 16000 →
  average_salary = 9000 →
  num_individuals = 5 →
  (average_salary * num_individuals) - salary_C = 29000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l2271_227149


namespace NUMINAMATH_CALUDE_playground_to_landscape_ratio_l2271_227121

def rectangular_landscape (length breadth : ℝ) : Prop :=
  length = 8 * breadth ∧ length = 240

def playground_area : ℝ := 1200

theorem playground_to_landscape_ratio 
  (length breadth : ℝ) 
  (h : rectangular_landscape length breadth) : 
  playground_area / (length * breadth) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_playground_to_landscape_ratio_l2271_227121


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_of_complement_implies_m_range_l2271_227191

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1: If A ∩ B = [1, 3], then m = 3
theorem intersection_implies_m_equals_three (m : ℝ) :
  A ∩ B m = Set.Icc 1 3 → m = 3 := by sorry

-- Theorem 2: If A ⊆ (ℝ \ B), then m > 5 or m < -3
theorem subset_of_complement_implies_m_range (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_of_complement_implies_m_range_l2271_227191


namespace NUMINAMATH_CALUDE_inverse_function_property_l2271_227163

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Given condition: the graph of y = x - f(x) passes through (1, 2)
axiom condition : 1 - f 1 = 2

-- Theorem to prove
theorem inverse_function_property : f_inv (-1) - (-1) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2271_227163


namespace NUMINAMATH_CALUDE_fraction_example_l2271_227145

/-- A fraction is defined as an expression with a variable in the denominator. -/
def is_fraction (f : ℝ → ℝ) : Prop :=
  ∃ (g h : ℝ → ℝ), ∀ x, f x = g x / h x ∧ h x ≠ 0 ∧ ∃ y, h y ≠ h 0

/-- The expression 1 / (1 - x) is a fraction. -/
theorem fraction_example : is_fraction (λ x => 1 / (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_example_l2271_227145


namespace NUMINAMATH_CALUDE_y_value_at_x_8_l2271_227160

theorem y_value_at_x_8 (k : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (∃ y : ℝ, y = 4 * Real.sqrt 3 ∧ 64^(1/3) * k = y) →
  ∃ y : ℝ, y = 2 * Real.sqrt 3 ∧ 8^(1/3) * k = y :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_x_8_l2271_227160


namespace NUMINAMATH_CALUDE_language_study_difference_l2271_227142

def total_students : ℕ := 2500

def german_min : ℕ := 1750
def german_max : ℕ := 1875

def russian_min : ℕ := 625
def russian_max : ℕ := 875

theorem language_study_difference : 
  let m := german_min + russian_min - total_students
  let M := german_max + russian_max - total_students
  M - m = 375 := by sorry

end NUMINAMATH_CALUDE_language_study_difference_l2271_227142


namespace NUMINAMATH_CALUDE_ace_of_hearts_or_diamonds_probability_l2271_227182

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Definition of a standard deck -/
def standardDeck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4 }

/-- Number of Aces of ♥ or ♦ in a standard deck -/
def redAces : Nat := 2

/-- Probability of drawing a specific card from a randomly arranged deck -/
def drawProbability (d : Deck) (favorableOutcomes : Nat) : Rat :=
  favorableOutcomes / d.cards

/-- Theorem: Probability of drawing an Ace of ♥ or ♦ from a standard deck is 1/26 -/
theorem ace_of_hearts_or_diamonds_probability :
  drawProbability standardDeck redAces = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ace_of_hearts_or_diamonds_probability_l2271_227182


namespace NUMINAMATH_CALUDE_defective_shipped_less_than_one_percent_l2271_227110

/-- Represents the production and shipping process of units --/
structure ProductionProcess where
  initial_units : ℝ
  prod_stage1_defect_rate : ℝ
  prod_stage2_defect_rate : ℝ
  prod_stage3_defect_rate : ℝ
  ship_stage1_rate : ℝ
  ship_stage2_rate : ℝ
  ship_stage3_rate : ℝ

/-- Calculates the percentage of defective units shipped for sale --/
def defective_shipped_percentage (p : ProductionProcess) : ℝ :=
  sorry

/-- Theorem stating that the percentage of defective units shipped is less than 1% --/
theorem defective_shipped_less_than_one_percent (p : ProductionProcess)
  (h1 : p.initial_units > 0)
  (h2 : p.prod_stage1_defect_rate = 0.06)
  (h3 : p.prod_stage2_defect_rate = 0.03)
  (h4 : p.prod_stage3_defect_rate = 0.02)
  (h5 : p.ship_stage1_rate = 0.04)
  (h6 : p.ship_stage2_rate = 0.03)
  (h7 : p.ship_stage3_rate = 0.02) :
  defective_shipped_percentage p < 0.01 :=
sorry

end NUMINAMATH_CALUDE_defective_shipped_less_than_one_percent_l2271_227110


namespace NUMINAMATH_CALUDE_faculty_size_l2271_227102

-- Define the number of students in each category
def numeric_methods : ℕ := 230
def automatic_control : ℕ := 423
def both_subjects : ℕ := 134

-- Define the percentage of students in these subjects compared to total faculty
def percentage : ℚ := 80 / 100

-- Theorem statement
theorem faculty_size :
  ∃ (total : ℕ), 
    (numeric_methods + automatic_control - both_subjects : ℚ) = percentage * total ∧
    total = 649 := by sorry

end NUMINAMATH_CALUDE_faculty_size_l2271_227102


namespace NUMINAMATH_CALUDE_m_value_l2271_227111

def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {1, m}

theorem m_value (m : ℕ) : A ∩ B m = B m → m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l2271_227111


namespace NUMINAMATH_CALUDE_cube_regions_tetrahedron_regions_l2271_227138

/-- Represents a set of planes in 3D space -/
structure PlaneSet where
  num_planes : ℕ

/-- Calculates the number of regions created by a set of planes -/
def num_regions (planes : PlaneSet) : ℕ := sorry

/-- A cube's faces represented as 6 planes -/
def cube_faces : PlaneSet := { num_planes := 6 }

/-- A tetrahedron's faces represented as 4 planes -/
def tetrahedron_faces : PlaneSet := { num_planes := 4 }

/-- Theorem: The number of regions created by a cube's faces is 27 -/
theorem cube_regions :
  num_regions cube_faces = 27 := by sorry

/-- Theorem: The number of regions created by a tetrahedron's faces is 15 -/
theorem tetrahedron_regions :
  num_regions tetrahedron_faces = 15 := by sorry

end NUMINAMATH_CALUDE_cube_regions_tetrahedron_regions_l2271_227138


namespace NUMINAMATH_CALUDE_min_value_x_plus_9y_l2271_227167

theorem min_value_x_plus_9y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  x + 9*y ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_9y_l2271_227167


namespace NUMINAMATH_CALUDE_star_arrangements_l2271_227112

/-- The number of points on a regular six-pointed star -/
def num_points : ℕ := 12

/-- The number of rotational symmetries of a regular six-pointed star -/
def num_rotations : ℕ := 6

/-- The number of reflectional symmetries of a regular six-pointed star -/
def num_reflections : ℕ := 2

/-- The total number of symmetries of a regular six-pointed star -/
def total_symmetries : ℕ := num_rotations * num_reflections

/-- The number of distinct arrangements of objects on a regular six-pointed star -/
def distinct_arrangements : ℕ := Nat.factorial num_points / total_symmetries

theorem star_arrangements :
  distinct_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l2271_227112


namespace NUMINAMATH_CALUDE_magazine_cost_is_one_l2271_227106

/-- The cost of a magazine in dollars -/
def magazine_cost : ℝ := sorry

/-- The cost of a chocolate bar in dollars -/
def chocolate_cost : ℝ := sorry

/-- Theorem stating the cost of one magazine is $1 -/
theorem magazine_cost_is_one :
  (4 * chocolate_cost = 8 * magazine_cost) →
  (12 * chocolate_cost = 24) →
  magazine_cost = 1 := by sorry

end NUMINAMATH_CALUDE_magazine_cost_is_one_l2271_227106


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2271_227129

theorem polynomial_expansion (z : ℂ) :
  (z^2 - 3*z + 1) * (4*z^4 + z^3 - 2*z^2 + 3) = 4*z^6 - 12*z^5 + 3*z^4 + 4*z^3 - z^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2271_227129


namespace NUMINAMATH_CALUDE_f_min_value_l2271_227170

/-- The function f(x) = -x³ + 3x² + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

/-- The theorem stating that if f(x) has a maximum value of 2 on [-2, -1], 
    then its minimum value on this interval is -5 -/
theorem f_min_value (a : ℝ) : 
  (∃ x ∈ Set.Icc (-2) (-1), ∀ y ∈ Set.Icc (-2) (-1), f a y ≤ f a x) ∧ 
  (∃ x ∈ Set.Icc (-2) (-1), f a x = 2) →
  (∃ x ∈ Set.Icc (-2) (-1), ∀ y ∈ Set.Icc (-2) (-1), f a x ≤ f a y) ∧
  (∃ x ∈ Set.Icc (-2) (-1), f a x = -5) :=
by sorry


end NUMINAMATH_CALUDE_f_min_value_l2271_227170


namespace NUMINAMATH_CALUDE_min_broken_pastries_correct_l2271_227152

/-- The number of different fillings -/
def num_fillings : ℕ := 10

/-- The total number of pastries -/
def total_pastries : ℕ := 45

/-- Predicate to check if a number of broken pastries is sufficient for the trick -/
def is_sufficient (n : ℕ) : Prop :=
  ∀ (remaining : Finset (Fin 2 → Fin num_fillings)),
    remaining.card = total_pastries - n →
    ∀ pastry ∈ remaining, ∃ filling, ∀ other ∈ remaining, other ≠ pastry → pastry filling ≠ other filling

/-- The smallest number of pastries that need to be broken for the trick to work -/
def min_broken_pastries : ℕ := 36

/-- Theorem stating that min_broken_pastries is the smallest number for which the trick works -/
theorem min_broken_pastries_correct :
  is_sufficient min_broken_pastries ∧ ∀ k < min_broken_pastries, ¬is_sufficient k := by sorry

end NUMINAMATH_CALUDE_min_broken_pastries_correct_l2271_227152


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l2271_227119

theorem complex_fraction_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i) / (1 - 2*i) - (1 - 2*i) / (1 + 2*i) = 8/5 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l2271_227119


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2271_227188

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2271_227188


namespace NUMINAMATH_CALUDE_sum_and_fraction_difference_l2271_227126

theorem sum_and_fraction_difference (x y : ℝ) 
  (sum_eq : x + y = 450)
  (fraction_eq : x / y = 0.8) : 
  y - x = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_and_fraction_difference_l2271_227126


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2271_227135

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a * b = k

/-- Given two inversely proportional numbers a and b, 
    if a + b = 60 and a = 3b, then when a = 12, b = 56.25 -/
theorem inverse_proportion_problem (a b : ℝ) 
  (h_inv : InverselyProportional a b) 
  (h_sum : a + b = 60) 
  (h_prop : a = 3 * b) : 
  ∃ b' : ℝ, InverselyProportional 12 b' ∧ b' = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_problem_l2271_227135


namespace NUMINAMATH_CALUDE_kit_price_difference_l2271_227195

-- Define the prices
def kit_price : ℚ := 145.75
def filter_price_1 : ℚ := 9.50
def filter_price_2 : ℚ := 15.30
def filter_price_3 : ℚ := 20.75
def filter_price_4 : ℚ := 25.80

-- Define the quantities
def quantity_1 : ℕ := 3
def quantity_2 : ℕ := 2
def quantity_3 : ℕ := 1
def quantity_4 : ℕ := 2

-- Calculate the total price of individual filters
def total_individual_price : ℚ :=
  filter_price_1 * quantity_1 +
  filter_price_2 * quantity_2 +
  filter_price_3 * quantity_3 +
  filter_price_4 * quantity_4

-- Define the theorem
theorem kit_price_difference :
  kit_price - total_individual_price = 14.30 := by
  sorry

end NUMINAMATH_CALUDE_kit_price_difference_l2271_227195


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2271_227184

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x = 2 * y ∧ 3 * x - 2 * y = 8 → x = 4 ∧ y = 2 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  3 * x + 2 * y = 4 ∧ x / 2 - (y + 1) / 3 = 1 → x = 2 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2271_227184


namespace NUMINAMATH_CALUDE_square_and_hexagon_symmetric_l2271_227128

-- Define the set of polygons
inductive Polygon
| EquilateralTriangle
| Square
| RegularPentagon
| RegularHexagon

-- Define properties
def isAxiSymmetric : Polygon → Prop
| Polygon.EquilateralTriangle => True
| Polygon.Square => True
| Polygon.RegularPentagon => True
| Polygon.RegularHexagon => True

def isCentrallySymmetric : Polygon → Prop
| Polygon.EquilateralTriangle => False
| Polygon.Square => True
| Polygon.RegularPentagon => False
| Polygon.RegularHexagon => True

-- Theorem statement
theorem square_and_hexagon_symmetric :
  ∀ p : Polygon, (isAxiSymmetric p ∧ isCentrallySymmetric p) ↔ (p = Polygon.Square ∨ p = Polygon.RegularHexagon) :=
by sorry

end NUMINAMATH_CALUDE_square_and_hexagon_symmetric_l2271_227128


namespace NUMINAMATH_CALUDE_unit_circle_angle_properties_l2271_227155

theorem unit_circle_angle_properties (α : Real) :
  (∃ P : Real × Real, P.1^2 + P.2^2 = 1 ∧ P.1 = 3/5 ∧ P.2 = 4/5 ∧ 
   Real.cos α = P.1 ∧ Real.sin α = P.2) →
  Real.sin (π - α) = 4/5 ∧ Real.tan (π/4 + α) = -7 := by
  sorry

end NUMINAMATH_CALUDE_unit_circle_angle_properties_l2271_227155


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2271_227157

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l2271_227157


namespace NUMINAMATH_CALUDE_train_speed_train_speed_result_l2271_227162

/-- The speed of a train given its length, the time to cross a man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed + man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 63.0036 km/hr --/
theorem train_speed_result :
  ∃ ε > 0, abs (train_speed 250 14.998800095992321 3 - 63.0036) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_result_l2271_227162


namespace NUMINAMATH_CALUDE_coffee_stock_solution_l2271_227133

/-- Represents the coffee stock problem --/
def coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (new_batch_decaf_percent : ℝ) (final_decaf_percent : ℝ) (new_batch : ℝ) : Prop :=
  let initial_decaf := initial_stock * initial_decaf_percent
  let new_batch_decaf := new_batch * new_batch_decaf_percent
  let total_stock := initial_stock + new_batch
  let total_decaf := initial_decaf + new_batch_decaf
  (total_decaf / total_stock) = final_decaf_percent

/-- Theorem stating the solution to the coffee stock problem --/
theorem coffee_stock_solution :
  coffee_stock_problem 400 0.25 0.60 0.32 100 := by
  sorry

#check coffee_stock_solution

end NUMINAMATH_CALUDE_coffee_stock_solution_l2271_227133


namespace NUMINAMATH_CALUDE_water_wave_area_increase_rate_l2271_227189

/-- The rate of increase of the area of a circular water wave -/
theorem water_wave_area_increase_rate 
  (v : ℝ) -- velocity of radius expansion
  (r : ℝ) -- current radius
  (h1 : v = 50) -- given velocity
  (h2 : r = 250) -- given radius
  : (π * v * r * 2) = 25000 * π := by
  sorry

end NUMINAMATH_CALUDE_water_wave_area_increase_rate_l2271_227189


namespace NUMINAMATH_CALUDE_distinct_scores_count_l2271_227179

/-- Represents the possible scores for a basketball player who made 7 baskets,
    where each basket is worth either 2 or 3 points. -/
def basketball_scores : Finset ℕ :=
  Finset.image (fun x => x + 14) (Finset.range 8)

/-- The number of distinct possible scores for the basketball player. -/
theorem distinct_scores_count : basketball_scores.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_scores_count_l2271_227179


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2271_227105

theorem solution_set_equivalence (m n : ℝ) 
  (h : ∀ x : ℝ, m * x + n > 0 ↔ x < 1/3) : 
  ∀ x : ℝ, n * x - m < 0 ↔ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2271_227105


namespace NUMINAMATH_CALUDE_neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l2271_227143

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_increasing_nonneg : ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → f x₁ < f x₂
axiom f_one_eq_two : f 1 = 2

-- State the theorems to be proved
theorem neg_f_squared_increasing_nonpos :
  ∀ x₁ x₂ : ℝ, x₁ ≤ 0 ∧ x₂ ≤ 0 ∧ x₁ < x₂ → -(f x₁)^2 < -(f x₂)^2 := by sorry

theorem neg_f_squared_decreasing_nonneg :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2 := by sorry

theorem a_range :
  ∀ a : ℝ, f (2*a^2 - 1) + 2*f a - 6 < 0 ↔ -2 < a ∧ a < 1 := by sorry

end

end NUMINAMATH_CALUDE_neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l2271_227143


namespace NUMINAMATH_CALUDE_expand_product_l2271_227175

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2271_227175


namespace NUMINAMATH_CALUDE_research_paper_word_count_l2271_227148

/-- Calculates the number of words typed given a typing speed and duration. -/
def words_typed (typing_speed : ℕ) (duration_hours : ℕ) : ℕ :=
  typing_speed * (duration_hours * 60)

/-- Proves that given a typing speed of 38 words per minute and a duration of 2 hours,
    the total number of words typed is 4560. -/
theorem research_paper_word_count :
  words_typed 38 2 = 4560 := by
  sorry

end NUMINAMATH_CALUDE_research_paper_word_count_l2271_227148


namespace NUMINAMATH_CALUDE_tournament_300_players_l2271_227134

/-- A single-elimination tournament with initial players and power-of-2 rounds -/
structure Tournament :=
  (initial_players : ℕ)
  (is_power_of_two : ℕ → Prop)

/-- Calculate the number of byes and total games in a tournament -/
def tournament_results (t : Tournament) : ℕ × ℕ :=
  sorry

theorem tournament_300_players 
  (t : Tournament) 
  (h1 : t.initial_players = 300) 
  (h2 : ∀ n, t.is_power_of_two n ↔ ∃ k, n = 2^k) : 
  tournament_results t = (44, 255) :=
sorry

end NUMINAMATH_CALUDE_tournament_300_players_l2271_227134


namespace NUMINAMATH_CALUDE_new_average_age_l2271_227132

-- Define the initial conditions
def initial_people : ℕ := 8
def initial_average_age : ℚ := 28
def leaving_person_age : ℕ := 20
def entering_person_age : ℕ := 25

-- Define the theorem
theorem new_average_age :
  let initial_total_age : ℚ := initial_people * initial_average_age
  let after_leaving_age : ℚ := initial_total_age - leaving_person_age
  let final_total_age : ℚ := after_leaving_age + entering_person_age
  final_total_age / initial_people = 229 / 8 := by sorry

end NUMINAMATH_CALUDE_new_average_age_l2271_227132


namespace NUMINAMATH_CALUDE_infinite_solutions_l2271_227120

/-- A system of linear equations -/
structure LinearSystem where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  eq3 : ℝ → ℝ → ℝ

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem where
  eq1 := fun x y => 3 * x - 4 * y - 5
  eq2 := fun x y => 6 * x - 8 * y - 10
  eq3 := fun x y => 9 * x - 12 * y - 15

/-- A solution to the system is a pair (x, y) that satisfies all equations -/
def isSolution (system : LinearSystem) (x y : ℝ) : Prop :=
  system.eq1 x y = 0 ∧ system.eq2 x y = 0 ∧ system.eq3 x y = 0

/-- The theorem stating that the system has infinitely many solutions -/
theorem infinite_solutions (system : LinearSystem) 
  (h1 : ∀ x y, system.eq2 x y = 2 * system.eq1 x y)
  (h2 : ∀ x y, system.eq3 x y = 3 * system.eq1 x y) :
  ∃ f : ℝ → ℝ × ℝ, ∀ t, isSolution system (f t).1 (f t).2 :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2271_227120


namespace NUMINAMATH_CALUDE_parallel_lines_length_l2271_227147

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents a geometric figure with parallel lines -/
structure GeometricFigure where
  AB : Segment
  CD : Segment
  EF : Segment
  GH : Segment
  parallel : AB.length / CD.length = CD.length / EF.length ∧ 
             CD.length / EF.length = EF.length / GH.length

theorem parallel_lines_length (fig : GeometricFigure) 
  (h1 : fig.AB.length = 180)
  (h2 : fig.CD.length = 120)
  (h3 : fig.GH.length = 80) :
  fig.EF.length = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_length_l2271_227147


namespace NUMINAMATH_CALUDE_min_distance_exponential_linear_l2271_227103

theorem min_distance_exponential_linear (t : ℝ) : 
  let f (x : ℝ) := Real.exp x
  let g (x : ℝ) := 2 * x
  let distance (t : ℝ) := |f t - g t|
  ∃ (min_dist : ℝ), ∀ (t : ℝ), distance t ≥ min_dist ∧ min_dist = 2 - 2 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_exponential_linear_l2271_227103


namespace NUMINAMATH_CALUDE_calvins_weight_after_one_year_l2271_227124

/-- Represents the weight loss from gym training for each month --/
def gym_training_loss : List Nat := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]

/-- Represents the weight loss from additional exercise routines for each month --/
def exercise_routines_loss : List Nat := [2, 3, 4, 3, 2, 4, 3, 2, 1, 3, 2, 4]

/-- Calculates Calvin's weight after one year --/
def calculate_final_weight (initial_weight : Nat) (gym_loss : List Nat) (diet_loss_per_month : Nat) (exercise_loss : List Nat) : Nat :=
  initial_weight - (gym_loss.sum + diet_loss_per_month * 12 + exercise_loss.sum)

/-- Theorem stating Calvin's weight after one year --/
theorem calvins_weight_after_one_year :
  calculate_final_weight 250 gym_training_loss 3 exercise_routines_loss = 106 := by
  sorry


end NUMINAMATH_CALUDE_calvins_weight_after_one_year_l2271_227124


namespace NUMINAMATH_CALUDE_smallest_cube_with_four_8s_l2271_227104

/-- A function that returns the first k digits of a natural number n -/
def firstKDigits (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A function that checks if the first k digits of n are all 8 -/
def startsWithK8s (n : ℕ) (k : ℕ) : Prop := 
  firstKDigits n k = (8 : ℕ) * (10^k - 1) / 9

theorem smallest_cube_with_four_8s :
  (∀ m : ℕ, m < 9615 → ¬ startsWithK8s (m^3) 4) ∧ startsWithK8s (9615^3) 4 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_with_four_8s_l2271_227104


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2271_227140

theorem quadratic_inequality_empty_solution_set (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - m * x + m ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2271_227140


namespace NUMINAMATH_CALUDE_quadratic_roots_are_integers_l2271_227159

theorem quadratic_roots_are_integers
  (a b : ℤ)
  (h : ∃ (p q : ℤ), q ≠ 0 ∧ a^2 - 4*b = (p / q : ℚ)^2) :
  ∃ (x : ℤ), x^2 - a*x + b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_are_integers_l2271_227159


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l2271_227193

open BigOperators

theorem sum_even_coefficients (n : ℕ) (a : ℕ → ℝ) :
  (∀ x, (1 + x + x^2)^n = ∑ i in Finset.range (2*n + 1), a i * x^i) →
  ∑ i in Finset.range (n + 1), a (2*i) = (3^n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l2271_227193


namespace NUMINAMATH_CALUDE_system_solution_implies_a_minus_b_l2271_227198

theorem system_solution_implies_a_minus_b (a b : ℤ) : 
  (a * (-2) + b * 1 = 1) → 
  (b * (-2) + a * 1 = 7) → 
  (a - b = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_a_minus_b_l2271_227198


namespace NUMINAMATH_CALUDE_parallelepiped_plane_ratio_l2271_227114

/-- A parallelepiped in 3D space -/
structure Parallelepiped where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The number of distinct planes passing through any three vertices of a parallelepiped -/
def num_distinct_planes (V : Parallelepiped) : ℕ := sorry

/-- The number of planes that bisect the volume of a parallelepiped -/
def num_bisecting_planes (V : Parallelepiped) : ℕ := sorry

/-- Theorem stating the ratio of bisecting planes to total distinct planes -/
theorem parallelepiped_plane_ratio (V : Parallelepiped) : 
  (num_bisecting_planes V : ℚ) / (num_distinct_planes V : ℚ) = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_plane_ratio_l2271_227114


namespace NUMINAMATH_CALUDE_y_satisfies_equation_l2271_227183

/-- The function y defined as the cube root of a quadratic expression -/
def y (x : ℝ) : ℝ := (2 + 3*x - 3*x^2)^(1/3)

/-- The statement that y satisfies the given differential equation -/
theorem y_satisfies_equation :
  ∀ x : ℝ, (y x) * (deriv y x) = (1 - 2*x) / (y x) := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_equation_l2271_227183


namespace NUMINAMATH_CALUDE_amount_to_leave_in_till_l2271_227169

/-- Represents the number of bills of each denomination in the till -/
structure TillContents where
  hundred_bills : Nat
  fifty_bills : Nat
  twenty_bills : Nat
  ten_bills : Nat
  five_bills : Nat
  one_bills : Nat

/-- Calculates the total value of bills in the till -/
def total_in_notes (till : TillContents) : Nat :=
  till.hundred_bills * 100 +
  till.fifty_bills * 50 +
  till.twenty_bills * 20 +
  till.ten_bills * 10 +
  till.five_bills * 5 +
  till.one_bills

/-- Calculates the amount to leave in the till -/
def amount_to_leave (till : TillContents) (amount_to_hand_in : Nat) : Nat :=
  total_in_notes till - amount_to_hand_in

/-- Jack's till contents -/
def jacks_till : TillContents :=
  { hundred_bills := 2
  , fifty_bills := 1
  , twenty_bills := 5
  , ten_bills := 3
  , five_bills := 7
  , one_bills := 27 }

theorem amount_to_leave_in_till :
  amount_to_leave jacks_till 142 = 300 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_leave_in_till_l2271_227169


namespace NUMINAMATH_CALUDE_three_pairs_square_product_400_l2271_227100

/-- The number of pairs of positive integers whose squares multiply to 400 -/
def count_pairs : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 * p.2^2 = 400)
    (Finset.product (Finset.range 21) (Finset.range 21))).card

/-- Theorem stating that there are exactly 3 pairs of positive integers
    whose squares multiply to 400 -/
theorem three_pairs_square_product_400 :
  count_pairs = 3 := by sorry

end NUMINAMATH_CALUDE_three_pairs_square_product_400_l2271_227100


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2271_227164

/-- Given a complex number z satisfying |z - 3 + 2i| = 4,
    the minimum value of |z + 1 - i|^2 + |z - 7 + 5i|^2 is 36. -/
theorem min_value_complex_expression (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  36 ≤ Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 5*I))^2 ∧
  ∃ w : ℂ, Complex.abs (w - (3 - 2*I)) = 4 ∧
          Complex.abs (w + (1 - I))^2 + Complex.abs (w - (7 - 5*I))^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2271_227164


namespace NUMINAMATH_CALUDE_political_test_analysis_l2271_227190

def class_A_scores : List ℝ := [41, 47, 43, 45, 50, 49, 48, 50, 50, 49, 48, 47, 44, 50, 43, 50, 50, 50, 49, 47]

structure FrequencyDistribution :=
  (range1 : ℕ)
  (range2 : ℕ)
  (range3 : ℕ)
  (range4 : ℕ)
  (range5 : ℕ)

def class_B_dist : FrequencyDistribution :=
  { range1 := 1
  , range2 := 1
  , range3 := 3  -- This is 'a' in the original problem
  , range4 := 6
  , range5 := 9 }

def class_B_46_to_48 : List ℝ := [47, 48, 48, 47, 48, 48]

structure ClassStats :=
  (average : ℝ)
  (median : ℝ)
  (mode : ℝ)

def class_A_stats : ClassStats :=
  { average := 47.5
  , median := 48.5
  , mode := 50 }  -- This is 'c' in the original problem

def class_B_stats : ClassStats :=
  { average := 47.5
  , median := 48  -- This is 'b' in the original problem
  , mode := 49 }

def total_students : ℕ := 800

theorem political_test_analysis :
  (class_B_dist.range3 = 3) ∧
  (class_B_stats.median = 48) ∧
  (class_A_stats.mode = 50) ∧
  (((List.filter (λ x => x ≥ 49) class_A_scores).length +
    class_B_dist.range5) / 40 * total_students = 380) := by
  sorry


end NUMINAMATH_CALUDE_political_test_analysis_l2271_227190


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2271_227197

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 1) ∧ 
  (a % 5 = 3) ∧
  (∀ (b : ℕ), b > 0 ∧ b % 3 = 2 ∧ b % 4 = 1 ∧ b % 5 = 3 → a ≤ b) ∧
  (a = 53) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2271_227197


namespace NUMINAMATH_CALUDE_star_five_three_l2271_227172

def star (a b : ℤ) : ℤ := a^2 + a*b - b^2

theorem star_five_three : star 5 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l2271_227172


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l2271_227141

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d →
  (c + d) + (b + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 2015 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l2271_227141


namespace NUMINAMATH_CALUDE_max_digit_sum_is_24_l2271_227180

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  h_range : hours ≤ 23
  m_range : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumDigits (n / 10)

/-- Calculates the sum of all digits in a Time24 -/
def totalDigitSum (t : Time24) : Nat :=
  sumDigits t.hours + sumDigits t.minutes

/-- The maximum sum of digits possible in a 24-hour format display -/
def maxDigitSum : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format display is 24 -/
theorem max_digit_sum_is_24 : 
  ∀ t : Time24, totalDigitSum t ≤ maxDigitSum :=
by sorry

end NUMINAMATH_CALUDE_max_digit_sum_is_24_l2271_227180


namespace NUMINAMATH_CALUDE_part_one_part_two_l2271_227108

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (a : ℝ) : (∀ x, f a x ≤ 3 ↔ x ∈ Set.Icc (-1) 5) → a = 2 := by sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) ∧ 
  (∀ m : ℝ, (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ m) → m ≤ 5) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2271_227108


namespace NUMINAMATH_CALUDE_grandmas_red_bacon_bits_l2271_227130

/-- Calculates the number of red bacon bits on Grandma's salad --/
def red_bacon_bits (mushrooms : ℕ) : ℕ :=
  let cherry_tomatoes := 2 * mushrooms
  let pickles := 4 * cherry_tomatoes
  let bacon_bits := 4 * pickles
  bacon_bits / 3

/-- Theorem stating that the number of red bacon bits on Grandma's salad is 32 --/
theorem grandmas_red_bacon_bits :
  red_bacon_bits 3 = 32 := by
  sorry

#eval red_bacon_bits 3

end NUMINAMATH_CALUDE_grandmas_red_bacon_bits_l2271_227130


namespace NUMINAMATH_CALUDE_triangle_problem_l2271_227118

open Real

theorem triangle_problem (A B C : ℝ) (s t : ℝ × ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Condition on dot products
  (B - C) * (C - A) = (C - A) * (A - B) →
  -- Definitions of vectors s and t
  s = (2 * sin C, -Real.sqrt 3) ∧
  t = (sin (2 * C), 2 * (cos (C / 2))^2 - 1) →
  -- Vectors s and t are parallel
  ∃ (k : ℝ), s.1 * t.2 = s.2 * t.1 →
  -- Given value of sin A
  sin A = 1 / 3 →
  -- Conclusion
  sin (π / 3 - B) = (2 * Real.sqrt 6 - 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2271_227118
