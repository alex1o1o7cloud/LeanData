import Mathlib

namespace neg_cube_eq_cube_of_neg_l2749_274998

theorem neg_cube_eq_cube_of_neg (x : ℚ) : -x^3 = (-x)^3 := by
  sorry

end neg_cube_eq_cube_of_neg_l2749_274998


namespace tangent_slope_angle_at_zero_l2749_274963

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_slope_angle_at_zero :
  let slope := (deriv f) 0
  Real.arctan slope = π / 4 := by
  sorry

end tangent_slope_angle_at_zero_l2749_274963


namespace lcm_gcd_relation_l2749_274933

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b + Nat.gcd a b = a * b / 5) ↔ 
  ((a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6)) :=
sorry

end lcm_gcd_relation_l2749_274933


namespace arithmetic_seq_problem_l2749_274931

/-- Define an arithmetic sequence {aₙ/n} with common difference d -/
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) :=
  ∀ n m : ℕ, a m / m - a n / n = d * (m - n)

theorem arithmetic_seq_problem (a : ℕ → ℚ) (d : ℚ) 
  (h_seq : arithmetic_seq a d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1/9 ∧ a 12 = 20 := by
  sorry


end arithmetic_seq_problem_l2749_274931


namespace f_max_min_on_interval_l2749_274930

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ Set.Icc 0 2 ∧ b ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f a) ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≥ f b) ∧
  f a = 5 ∧ f b = -15 :=
sorry


end f_max_min_on_interval_l2749_274930


namespace field_trip_group_size_l2749_274978

/-- Calculates the number of students in each group excluding the student themselves -/
def students_per_group (total_bread : ℕ) (num_groups : ℕ) (sandwiches_per_student : ℕ) (bread_per_sandwich : ℕ) : ℕ :=
  (total_bread / (num_groups * sandwiches_per_student * bread_per_sandwich)) - 1

/-- Theorem: Given the specified conditions, there are 5 students in each group excluding the student themselves -/
theorem field_trip_group_size :
  students_per_group 120 5 2 2 = 5 := by
  sorry

end field_trip_group_size_l2749_274978


namespace smallest_z_value_l2749_274966

theorem smallest_z_value (w x y z : ℕ) : 
  w^3 + x^3 + y^3 = z^3 →
  w < x ∧ x < y ∧ y < z →
  Odd w ∧ Odd x ∧ Odd y ∧ Odd z →
  (∀ a b c d : ℕ, a < b ∧ b < c ∧ c < d ∧ 
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧
    a^3 + b^3 + c^3 = d^3 → z ≤ d) →
  z = 9 :=
by sorry

end smallest_z_value_l2749_274966


namespace pentagonal_gcd_one_l2749_274935

theorem pentagonal_gcd_one (n : ℕ+) : 
  let P : ℕ+ → ℕ := fun m => (m * (3 * m - 1)) / 2
  Nat.gcd (5 * P n) (n + 1) = 1 := by
  sorry

end pentagonal_gcd_one_l2749_274935


namespace rectangular_hyperbola_real_axis_length_l2749_274904

/-- Given a rectangular hyperbola C centered at the origin with foci on the x-axis,
    if C intersects the line x = -4 at two points with a vertical distance of 4√3,
    then the length of the real axis of C is 4. -/
theorem rectangular_hyperbola_real_axis_length
  (C : Set (ℝ × ℝ))
  (h1 : ∃ (a : ℝ), a > 0 ∧ C = {(x, y) | x^2 - y^2 = a^2})
  (h2 : ∃ (y1 y2 : ℝ), ((-4, y1) ∈ C ∧ (-4, y2) ∈ C ∧ |y1 - y2| = 4 * Real.sqrt 3)) :
  ∃ (a : ℝ), a > 0 ∧ C = {(x, y) | x^2 - y^2 = a^2} ∧ 2 * a = 4 :=
sorry

end rectangular_hyperbola_real_axis_length_l2749_274904


namespace min_value_of_f_l2749_274957

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 :=
sorry

end min_value_of_f_l2749_274957


namespace ways_to_express_114_l2749_274934

/-- Represents the number of ways to express a given number as the sum of ones and threes with a minimum number of ones -/
def waysToExpress (total : ℕ) (minOnes : ℕ) : ℕ :=
  (total - minOnes) / 3 + 1

/-- The theorem stating that there are 35 ways to express 114 as the sum of ones and threes with at least 10 ones -/
theorem ways_to_express_114 : waysToExpress 114 10 = 35 := by
  sorry

#eval waysToExpress 114 10

end ways_to_express_114_l2749_274934


namespace power_sum_equality_l2749_274929

theorem power_sum_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ a b c d : ℝ, (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) :=
by sorry

end power_sum_equality_l2749_274929


namespace parallelogram_D_coordinates_l2749_274952

structure Point where
  x : ℝ
  y : ℝ

def Parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) ∧
  (C.x - B.x, C.y - B.y) = (A.x - D.x, A.y - D.y)

theorem parallelogram_D_coordinates :
  let A : Point := ⟨-1, 2⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨1, 7⟩
  let D : Point := ⟨0, 9⟩
  Parallelogram A B C D → D = ⟨0, 9⟩ := by
  sorry

end parallelogram_D_coordinates_l2749_274952


namespace quadratic_equation_completion_square_l2749_274942

theorem quadratic_equation_completion_square (x m n : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) → 
  ((x + m)^2 = n) →
  (m + n = 11) := by
sorry

end quadratic_equation_completion_square_l2749_274942


namespace larger_solid_volume_is_4_point_5_l2749_274941

-- Define the rectangular prism
def rectangular_prism (length width height : ℝ) := length * width * height

-- Define a plane that cuts the prism
structure cutting_plane (length width height : ℝ) :=
  (passes_through_vertex : Bool)
  (passes_through_midpoint_edge1 : Bool)
  (passes_through_midpoint_edge2 : Bool)

-- Define the volume of the larger solid resulting from the cut
def larger_solid_volume (length width height : ℝ) (plane : cutting_plane length width height) : ℝ :=
  sorry

-- Theorem statement
theorem larger_solid_volume_is_4_point_5 :
  ∀ (plane : cutting_plane 2 1 3),
    plane.passes_through_vertex = true ∧
    plane.passes_through_midpoint_edge1 = true ∧
    plane.passes_through_midpoint_edge2 = true →
    larger_solid_volume 2 1 3 plane = 4.5 :=
by sorry

end larger_solid_volume_is_4_point_5_l2749_274941


namespace cos_leq_half_range_l2749_274943

theorem cos_leq_half_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) :=
by sorry

end cos_leq_half_range_l2749_274943


namespace max_value_implies_m_eq_two_l2749_274919

/-- The function f(x) = x^3 - 3x^2 + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + m

/-- Theorem: If the maximum value of f(x) in [-1, 1] is 2, then m = 2 -/
theorem max_value_implies_m_eq_two (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f m x ≤ 2) ∧ (∃ x ∈ Set.Icc (-1) 1, f m x = 2) → m = 2 := by
  sorry

#check max_value_implies_m_eq_two

end max_value_implies_m_eq_two_l2749_274919


namespace share_distribution_l2749_274903

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 364 →
  a = (1/2) * b →
  b = (1/2) * c →
  a + b + c = total →
  c = 208 := by
sorry

end share_distribution_l2749_274903


namespace initial_average_production_l2749_274962

theorem initial_average_production 
  (n : ℕ) 
  (today_production : ℕ) 
  (new_average : ℚ) 
  (h1 : n = 8) 
  (h2 : today_production = 95) 
  (h3 : new_average = 55) : 
  (n : ℚ) * (n * new_average - today_production) / (n * (n + 1)) = 50 := by
  sorry

end initial_average_production_l2749_274962


namespace largest_five_digit_congruent_to_15_mod_17_l2749_274901

theorem largest_five_digit_congruent_to_15_mod_17 :
  ∀ n : ℕ, n < 100000 → n ≡ 15 [MOD 17] → n ≤ 99977 :=
by sorry

end largest_five_digit_congruent_to_15_mod_17_l2749_274901


namespace line_l_equation_l2749_274977

-- Define the intersection point of the two given lines
def intersection_point : ℝ × ℝ := (2, 1)

-- Define point A
def point_A : ℝ × ℝ := (5, 0)

-- Define the distance from point A to line l
def distance_to_l : ℝ := 3

-- Define the two possible equations for line l
def line_eq1 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line_eq2 (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem line_l_equation : 
  ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ (line_eq1 x y ∨ line_eq2 x)) ∧
    (l (intersection_point.1) (intersection_point.2)) ∧
    (∀ x y, l x y → 
      (|4 * point_A.1 - 3 * point_A.2 - 5| / Real.sqrt (4^2 + 3^2) = distance_to_l ∨
       |point_A.1 - 2| = distance_to_l)) :=
sorry

end line_l_equation_l2749_274977


namespace smallest_covering_triangular_number_l2749_274923

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def all_remainders_covered (m : ℕ) : Prop :=
  ∀ r : Fin 7, ∃ k : ℕ, k ≤ m ∧ triangular_number k % 7 = r.val

theorem smallest_covering_triangular_number :
  (all_remainders_covered 10) ∧
  (∀ n < 10, ¬ all_remainders_covered n) :=
sorry

end smallest_covering_triangular_number_l2749_274923


namespace prime_condition_equivalence_l2749_274959

/-- For a prime number p, this function returns true if for each integer a 
    such that 1 < a < p/2, there exists an integer b such that p/2 < b < p 
    and p divides ab - 1 -/
def satisfies_condition (p : ℕ) : Prop :=
  ∀ a : ℕ, 1 < a → a < p / 2 → ∃ b : ℕ, p / 2 < b ∧ b < p ∧ p ∣ (a * b - 1)

theorem prime_condition_equivalence (p : ℕ) (hp : Nat.Prime p) : 
  satisfies_condition p ↔ p ∈ ({5, 7, 13} : Set ℕ) := by
  sorry

end prime_condition_equivalence_l2749_274959


namespace box_dimensions_sum_l2749_274979

/-- Given a rectangular box with dimensions A, B, and C, prove that if the surface areas of its faces
    are 30, 30, 60, 60, 90, and 90 square units, then A + B + C = 24. -/
theorem box_dimensions_sum (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A * B = 30 →
  A * C = 60 →
  B * C = 90 →
  A + B + C = 24 := by
  sorry

end box_dimensions_sum_l2749_274979


namespace product_of_differences_l2749_274932

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2010) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2009)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2010) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2009)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2010) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2009) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/2010 := by
  sorry

end product_of_differences_l2749_274932


namespace dividend_calculation_l2749_274996

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 36 → quotient = 19 → remainder = 6 → 
  divisor * quotient + remainder = 690 := by
sorry

end dividend_calculation_l2749_274996


namespace distance_to_felix_l2749_274989

/-- The vertical distance David and Emma walk together to reach Felix -/
theorem distance_to_felix (david_x david_y emma_x emma_y felix_x felix_y : ℝ) 
  (h1 : david_x = 2 ∧ david_y = -25)
  (h2 : emma_x = -3 ∧ emma_y = 19)
  (h3 : felix_x = -1/2 ∧ felix_y = -6) :
  let midpoint_y := (david_y + emma_y) / 2
  |(midpoint_y - felix_y)| = 3 := by
  sorry

end distance_to_felix_l2749_274989


namespace rectangle_length_l2749_274964

/-- Proves that a rectangle with area 6 m² and width 150 cm has length 400 cm -/
theorem rectangle_length (area : ℝ) (width_cm : ℝ) (length_cm : ℝ) : 
  area = 6 → 
  width_cm = 150 → 
  area = (width_cm / 100) * (length_cm / 100) → 
  length_cm = 400 := by
sorry

end rectangle_length_l2749_274964


namespace pond_length_proof_l2749_274988

def field_length : ℝ := 80

theorem pond_length_proof (field_width : ℝ) (pond_side : ℝ) : 
  field_length = 2 * field_width →
  pond_side^2 = (field_length * field_width) / 50 →
  pond_side = 8 := by
sorry

end pond_length_proof_l2749_274988


namespace lines_parallel_iff_m_eq_neg_seven_l2749_274902

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2 ∧ a1 / b1 ≠ c1 / c2

/-- Definition of line l1 -/
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (3 + m) * x + 4 * y = 5 - 3 * m

/-- Definition of line l2 -/
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + (5 + m) * y = 8

/-- Theorem: Lines l1 and l2 are parallel if and only if m = -7 -/
theorem lines_parallel_iff_m_eq_neg_seven :
  ∀ m : ℝ, parallel_lines (3 + m) 4 (5 - 3 * m) 2 (5 + m) 8 ↔ m = -7 := by sorry

end lines_parallel_iff_m_eq_neg_seven_l2749_274902


namespace parabola_comparison_l2749_274987

theorem parabola_comparison :
  ∀ x : ℝ, x^2 - 3/4*x + 3 ≥ x^2 + 1/4*x + 1 := by
  sorry

end parabola_comparison_l2749_274987


namespace brick_width_calculation_l2749_274949

/-- Proves that given a courtyard of 25 meters by 15 meters, to be paved with 18750 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 15 →
  brick_length = 0.2 →
  total_bricks = 18750 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * 100) * (courtyard_width * 100) = 
      total_bricks * brick_length * 100 * brick_width * 100 :=
by sorry

end brick_width_calculation_l2749_274949


namespace not_prime_n_pow_n_minus_6n_plus_5_l2749_274912

theorem not_prime_n_pow_n_minus_6n_plus_5 (n : ℕ) : ¬ Prime (n^n - 6*n + 5) := by
  sorry

end not_prime_n_pow_n_minus_6n_plus_5_l2749_274912


namespace four_meetings_theorem_l2749_274938

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Bool -- True for clockwise, False for counterclockwise

/-- Calculates the number of meetings between two runners on a circular track -/
def number_of_meetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with speeds 2 m/s and 3 m/s in opposite directions meet 4 times -/
theorem four_meetings_theorem (track_length : ℝ) (h : track_length > 0) :
  let runner1 : Runner := ⟨2, true⟩
  let runner2 : Runner := ⟨3, false⟩
  number_of_meetings runner1 runner2 = 4 :=
sorry

end four_meetings_theorem_l2749_274938


namespace isosceles_triangle_exists_l2749_274990

/-- Represents an isosceles triangle with base a and leg b -/
structure IsoscelesTriangle where
  a : ℝ  -- base
  b : ℝ  -- leg
  ma : ℝ  -- height corresponding to base
  mb : ℝ  -- height corresponding to leg
  h_isosceles : b > 0 ∧ ma > 0 ∧ mb > 0 ∧ a * ma = b * mb

/-- Given the sums and differences of sides and heights, 
    prove the existence of an isosceles triangle -/
theorem isosceles_triangle_exists 
  (sum_sides : ℝ) 
  (sum_heights : ℝ) 
  (diff_sides : ℝ) 
  (diff_heights : ℝ) 
  (h_positive : sum_sides > 0 ∧ sum_heights > 0 ∧ diff_sides > 0 ∧ diff_heights > 0) :
  ∃ t : IsoscelesTriangle, 
    t.a + t.b = sum_sides ∧ 
    t.ma + t.mb = sum_heights ∧
    t.b - t.a = diff_sides ∧
    t.ma - t.mb = diff_heights :=
sorry

end isosceles_triangle_exists_l2749_274990


namespace problem_statement_l2749_274913

def A : Set ℝ := {x | x^2 - x - 2 < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (2*a+6)*x + a^2 + 6*a ≤ 0}

theorem problem_statement :
  (∀ a : ℝ, (A ⊂ B a ∧ A ≠ B a) → -4 ≤ a ∧ a ≤ -1) ∧
  (∀ a : ℝ, (A ∩ B a = ∅) → a ≤ -7 ∨ a ≥ 2) :=
sorry

end problem_statement_l2749_274913


namespace simplify_expression_l2749_274969

theorem simplify_expression (x : ℝ) (h1 : 1 < x) (h2 : x < 4) :
  Real.sqrt ((1 - x)^2) + |x - 4| = 3 := by
  sorry

end simplify_expression_l2749_274969


namespace sum_of_equal_expressions_l2749_274954

theorem sum_of_equal_expressions 
  (a b c d e f g h i : ℤ) 
  (eq1 : a + b + c + d = d + e + f + g) 
  (eq2 : d + e + f + g = g + h + i) 
  (ha : a = 4) 
  (hg : g = 13) 
  (hh : h = 6) : 
  ∃ S : ℤ, (a + b + c + d = S) ∧ (d + e + f + g = S) ∧ (g + h + i = S) ∧ (S = 19 + i) :=
sorry

end sum_of_equal_expressions_l2749_274954


namespace smallest_A_with_triple_factors_l2749_274973

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_A_with_triple_factors : 
  ∃ (A : ℕ), A > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < A → number_of_factors (6 * k) ≠ 3 * number_of_factors k) ∧
  number_of_factors (6 * A) = 3 * number_of_factors A ∧
  A = 2 := by
  sorry

end smallest_A_with_triple_factors_l2749_274973


namespace football_progress_l2749_274915

/-- Calculates the net progress of a football team given a loss and a gain in yards. -/
def net_progress (loss : ℤ) (gain : ℤ) : ℤ := -loss + gain

/-- Theorem stating that a loss of 5 yards followed by a gain of 8 yards results in a net progress of 3 yards. -/
theorem football_progress : net_progress 5 8 = 3 := by
  sorry

end football_progress_l2749_274915


namespace comic_book_problem_l2749_274927

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 2 + 6 = 17) → initial_books = 22 :=
by
  sorry

end comic_book_problem_l2749_274927


namespace bank_transaction_decrease_fraction_l2749_274908

/-- Represents a bank account transaction --/
structure BankTransaction where
  initialBalance : ℚ
  withdrawal : ℚ
  depositFraction : ℚ
  finalBalance : ℚ

/-- Calculates the fraction by which the account balance decreased after withdrawal --/
def decreaseFraction (t : BankTransaction) : ℚ :=
  t.withdrawal / t.initialBalance

/-- Theorem stating the conditions and the result to be proved --/
theorem bank_transaction_decrease_fraction 
  (t : BankTransaction)
  (h1 : t.withdrawal = 200)
  (h2 : t.depositFraction = 1/5)
  (h3 : t.finalBalance = 360)
  (h4 : t.finalBalance = t.initialBalance - t.withdrawal + t.depositFraction * (t.initialBalance - t.withdrawal)) :
  decreaseFraction t = 2/5 := by sorry


end bank_transaction_decrease_fraction_l2749_274908


namespace min_value_of_ab_l2749_274984

theorem min_value_of_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + 4 * b + 5) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + 4 * y + 5 → a * b ≤ x * y :=
by sorry

end min_value_of_ab_l2749_274984


namespace greatest_n_less_than_200_l2749_274965

theorem greatest_n_less_than_200 :
  ∃ (n : ℕ), n < 200 ∧ 
  (∃ (k : ℕ), n = 9 * k - 2) ∧
  (∃ (l : ℕ), n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 200 ∧ 
    (∃ (p : ℕ), m = 9 * p - 2) ∧ 
    (∃ (q : ℕ), m = 6 * q - 4) → 
    m ≤ n) ∧
  n = 194 := by
sorry

end greatest_n_less_than_200_l2749_274965


namespace parabola_ellipse_intersection_l2749_274991

theorem parabola_ellipse_intersection (p : ℝ) (m n k : ℝ) : 
  p > 0 → m > n → n > 0 → 
  ∃ (x₀ y₀ : ℝ), 
    y₀^2 = 2*p*x₀ ∧ 
    (x₀ + p/2)^2 + y₀^2 = 3^2 ∧ 
    x₀^2 + y₀^2 = 9 →
  ∃ (c : ℝ), 
    c = 2 ∧ 
    m^2 - n^2 = c^2 ∧
    2/m = 1/2 →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/m^2 + y₁^2/n^2 = 1 ∧
    x₂^2/m^2 + y₂^2/n^2 = 1 ∧
    y₁ = k*x₁ - 4 ∧
    y₂ = k*x₂ - 4 ∧
    x₁ ≠ x₂ ∧
    x₁*x₂ + y₁*y₂ > 0 →
  (-2*Real.sqrt 3/3 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < 2*Real.sqrt 3/3) :=
by sorry

end parabola_ellipse_intersection_l2749_274991


namespace purely_imaginary_complex_l2749_274997

/-- A complex number is purely imaginary if its real part is zero -/
def PurelyImaginary (z : ℂ) : Prop := z.re = 0

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The condition that (z+2)/(1-i) + z is a real number -/
def IsRealCondition (z : ℂ) : Prop := ((z + 2) / (1 - i) + z).im = 0

theorem purely_imaginary_complex : 
  ∀ z : ℂ, PurelyImaginary z → IsRealCondition z → z = -2/3 * i :=
sorry

end purely_imaginary_complex_l2749_274997


namespace power_division_multiplication_l2749_274953

theorem power_division_multiplication (x : ℕ) : (3^18 / 27^2) * 7 = 3720087 := by
  sorry

end power_division_multiplication_l2749_274953


namespace binomial_expansion_example_l2749_274972

theorem binomial_expansion_example : 50^4 + 4*(50^3) + 6*(50^2) + 4*50 + 1 = 6765201 := by
  sorry

end binomial_expansion_example_l2749_274972


namespace quarters_spent_l2749_274975

def initial_quarters : ℕ := 760
def remaining_quarters : ℕ := 342

theorem quarters_spent : initial_quarters - remaining_quarters = 418 := by
  sorry

end quarters_spent_l2749_274975


namespace min_value_of_function_l2749_274937

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (4 * x^2 + 8 * x + 13) / (6 * (1 + x)) ≥ 2 ∧
  ∃ y > 0, (4 * y^2 + 8 * y + 13) / (6 * (1 + y)) = 2 :=
by sorry

end min_value_of_function_l2749_274937


namespace parkway_elementary_girls_not_soccer_l2749_274985

theorem parkway_elementary_girls_not_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 500 →
  boys = 350 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 115 :=
by sorry

end parkway_elementary_girls_not_soccer_l2749_274985


namespace expression_value_l2749_274910

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 2 * y) / (x - 2 * y) = 23 := by
  sorry

end expression_value_l2749_274910


namespace positive_integer_solutions_l2749_274956

theorem positive_integer_solutions :
  ∀ x y z : ℕ+,
    x < y →
    2 * (x + 1) * (y + 1) - 1 = x * y * z →
    ((x = 1 ∧ y = 3 ∧ z = 5) ∨ (x = 3 ∧ y = 7 ∧ z = 3)) :=
by sorry

end positive_integer_solutions_l2749_274956


namespace min_vertical_distance_l2749_274946

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vert_distance (x : ℝ) : ℝ := |abs_func x - quad_func x|

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 3 ∧ ∀ (x : ℝ), vert_distance x ≥ min_dist :=
sorry

end min_vertical_distance_l2749_274946


namespace intersection_point_property_l2749_274999

theorem intersection_point_property (α : ℝ) (h1 : α ≠ 0) (h2 : Real.tan α = -α) :
  (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := by
  sorry

end intersection_point_property_l2749_274999


namespace reciprocal_of_negative_2023_l2749_274911

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end reciprocal_of_negative_2023_l2749_274911


namespace die_game_first_player_win_probability_l2749_274980

def game_win_probability : ℚ := 5/11

theorem die_game_first_player_win_probability :
  let n := 6  -- number of sides on the die
  let m := 7  -- winning condition (multiple of m)
  ∀ (k : ℕ), k < m →
    let p : ℚ := game_win_probability  -- probability of winning starting from state k
    (p = n / (2*n - 1) ∧
     p = (n-1) * (1 - p) / n + 1/n) :=
by sorry

end die_game_first_player_win_probability_l2749_274980


namespace consecutive_integers_sum_l2749_274981

theorem consecutive_integers_sum (x y z : ℤ) (w : ℤ) : 
  y = x + 1 → 
  z = x + 2 → 
  x + y + z = 150 → 
  w = 2*z - x → 
  x + y + z + w = 203 := by sorry

end consecutive_integers_sum_l2749_274981


namespace rhombus_area_l2749_274916

/-- The area of a rhombus with diagonals of length 6 and 10 is 30. -/
theorem rhombus_area (d₁ d₂ : ℝ) (h₁ : d₁ = 6) (h₂ : d₂ = 10) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 30 := by
  sorry

end rhombus_area_l2749_274916


namespace complex_equation_solution_l2749_274992

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2749_274992


namespace liquid_volume_range_l2749_274918

-- Define the cube
def cube_volume : ℝ := 6

-- Define the liquid volume as a real number between 0 and the cube volume
def liquid_volume : ℝ := sorry

-- Define the condition that the liquid surface is not a triangle
def not_triangle_surface : Prop := sorry

-- Theorem statement
theorem liquid_volume_range (h : not_triangle_surface) : 
  1 < liquid_volume ∧ liquid_volume < 5 := by sorry

end liquid_volume_range_l2749_274918


namespace lulu_cupcakes_count_l2749_274951

/-- Represents the number of pastries baked by Lola and Lulu -/
structure Pastries where
  lola_cupcakes : ℕ
  lola_poptarts : ℕ
  lola_pies : ℕ
  lulu_cupcakes : ℕ
  lulu_poptarts : ℕ
  lulu_pies : ℕ

/-- The total number of pastries baked by Lola and Lulu -/
def total_pastries (p : Pastries) : ℕ :=
  p.lola_cupcakes + p.lola_poptarts + p.lola_pies +
  p.lulu_cupcakes + p.lulu_poptarts + p.lulu_pies

/-- Theorem stating that Lulu baked 16 mini cupcakes -/
theorem lulu_cupcakes_count (p : Pastries) 
  (h1 : p.lola_cupcakes = 13)
  (h2 : p.lola_poptarts = 10)
  (h3 : p.lola_pies = 8)
  (h4 : p.lulu_poptarts = 12)
  (h5 : p.lulu_pies = 14)
  (h6 : total_pastries p = 73) :
  p.lulu_cupcakes = 16 := by
  sorry

end lulu_cupcakes_count_l2749_274951


namespace inscriptions_exist_l2749_274993

/-- Represents the maker of a casket -/
inductive Maker
| Bellini
| Cellini

/-- Represents a casket with its inscription -/
structure Casket where
  maker : Maker
  inscription : Prop

/-- The pair of caskets satisfies the given conditions -/
def satisfies_conditions (golden silver : Casket) : Prop :=
  let P := (golden.maker = Maker.Bellini ∧ silver.maker = Maker.Cellini) ∨
           (golden.maker = Maker.Cellini ∧ silver.maker = Maker.Bellini)
  let Q := silver.maker = Maker.Cellini
  
  -- Condition 1: One can conclude that one casket is made by Bellini and the other by Cellini
  (golden.inscription ∧ silver.inscription → P) ∧
  
  -- Condition 1 (continued): But it's impossible to determine which casket is whose work
  (golden.inscription ∧ silver.inscription → ¬(golden.maker = Maker.Bellini ∨ golden.maker = Maker.Cellini)) ∧
  
  -- Condition 2: The inscription on either casket alone doesn't allow concluding about the makers
  (golden.inscription → ¬P) ∧
  (silver.inscription → ¬P)

/-- There exist inscriptions that satisfy the given conditions -/
theorem inscriptions_exist : ∃ (golden silver : Casket), satisfies_conditions golden silver := by
  sorry

end inscriptions_exist_l2749_274993


namespace common_chord_equation_l2749_274955

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, C1 x y ∧ C2 x y → common_chord x y :=
by sorry

end common_chord_equation_l2749_274955


namespace highest_power_divisibility_l2749_274968

theorem highest_power_divisibility (n : ℕ) : 
  (∃ k : ℕ, (1991 : ℕ)^k ∣ 1990^(1991^1002) + 1992^(1501^1901)) ∧ 
  (∀ m : ℕ, m > 1001 → ¬((1991 : ℕ)^m ∣ 1990^(1991^1002) + 1992^(1501^1901))) :=
by sorry

end highest_power_divisibility_l2749_274968


namespace scribes_expenditure_change_l2749_274917

/-- Proves that reducing the number of scribes by 50% and increasing the salaries
    of the remaining scribes by 50% results in a 25% decrease in total expenditure. -/
theorem scribes_expenditure_change
  (initial_allocation : ℝ)
  (n : ℕ)
  (h1 : initial_allocation > 0)
  (h2 : n > 0) :
  let reduced_scribes := n / 2
  let initial_salary := initial_allocation / n
  let new_salary := initial_salary * 1.5
  let new_expenditure := reduced_scribes * new_salary
  new_expenditure / initial_allocation = 0.75 := by
  sorry

end scribes_expenditure_change_l2749_274917


namespace factorization_of_2x_cubed_minus_8x_l2749_274922

theorem factorization_of_2x_cubed_minus_8x (x : ℝ) : 2*x^3 - 8*x = 2*x*(x+2)*(x-2) := by
  sorry

end factorization_of_2x_cubed_minus_8x_l2749_274922


namespace no_natural_solution_l2749_274970

theorem no_natural_solution :
  ¬∃ (a b c : ℕ), (a^b - b^c) * (b^c - c^a) * (c^a - a^b) = 11713 := by
  sorry

end no_natural_solution_l2749_274970


namespace complex_equation_solution_l2749_274909

theorem complex_equation_solution :
  ∃ (z : ℂ), 3 - 3 * Complex.I * z = -2 + 5 * Complex.I * z + (1 - 2 * Complex.I) ∧
             z = (1 / 4 : ℂ) - (3 / 8 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l2749_274909


namespace marble_weight_l2749_274940

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  (9 * marble_weight = 4 * car_weight) →
  (3 * car_weight = 36) →
  marble_weight = 16 / 3 := by
sorry

end marble_weight_l2749_274940


namespace smallest_marble_count_l2749_274994

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a specific combination of marbles -/
def probability (m : MarbleCount) (r w b : ℕ) : ℚ :=
  (m.red.choose r) * (m.white.choose w) * (m.blue.choose b) /
  ((m.red + m.white + m.blue).choose 4)

/-- Checks if the three specified events are equally likely -/
def events_equally_likely (m : MarbleCount) : Prop :=
  probability m 3 1 0 = probability m 2 1 1 ∧
  probability m 3 1 0 = probability m 2 1 1

/-- The theorem stating that 8 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count :
  ∃ (m : MarbleCount),
    m.red + m.white + m.blue = 8 ∧
    events_equally_likely m ∧
    ∀ (n : MarbleCount),
      n.red + n.white + n.blue < 8 →
      ¬(events_equally_likely n) :=
sorry

end smallest_marble_count_l2749_274994


namespace king_of_diamonds_in_top_two_l2749_274921

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (ranks : ℕ)
  (jokers : ℕ)

/-- The probability of an event occurring -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of the King of Diamonds being one of the top two cards -/
theorem king_of_diamonds_in_top_two (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.ranks = 13)
  (h4 : d.jokers = 2) :
  probability 2 d.total_cards = 1 / 27 := by
  sorry

#check king_of_diamonds_in_top_two

end king_of_diamonds_in_top_two_l2749_274921


namespace geometric_sum_half_five_l2749_274961

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_half_five :
  geometric_sum (1/2) (1/2) 5 = 31/32 := by
  sorry

end geometric_sum_half_five_l2749_274961


namespace total_cost_is_543_l2749_274958

/-- Calculates the total amount John has to pay for earbuds and a smartwatch, including tax and discount. -/
def totalCost (earbudsCost smartwatchCost : ℝ) (earbudsTaxRate smartwatchTaxRate earbusDiscountRate : ℝ) : ℝ :=
  let discountedEarbudsCost := earbudsCost * (1 - earbusDiscountRate)
  let earbudsTax := discountedEarbudsCost * earbudsTaxRate
  let smartwatchTax := smartwatchCost * smartwatchTaxRate
  discountedEarbudsCost + earbudsTax + smartwatchCost + smartwatchTax

/-- Theorem stating that given the specific costs, tax rates, and discount, the total cost is $543. -/
theorem total_cost_is_543 :
  totalCost 200 300 0.15 0.12 0.10 = 543 := by
  sorry

end total_cost_is_543_l2749_274958


namespace hyperbola_focus_m_value_l2749_274971

/-- Given a hyperbola with equation 3mx^2 - my^2 = 3 and one focus at (0, 2), prove that m = -1 -/
theorem hyperbola_focus_m_value (m : ℝ) : 
  (∃ (x y : ℝ), 3 * m * x^2 - m * y^2 = 3) →  -- Hyperbola equation
  (∃ (a b : ℝ), a^2 / (3/m) + b^2 / (1/m) = 1) →  -- Standard form of hyperbola
  (2 : ℝ)^2 = (3/m) + (1/m) →  -- Focus property
  m = -1 := by
sorry

end hyperbola_focus_m_value_l2749_274971


namespace midpoint_chain_l2749_274982

/-- Given a line segment XY with midpoints defined as follows:
    G is the midpoint of XY
    H is the midpoint of XG
    I is the midpoint of XH
    J is the midpoint of XI
    If XJ = 4, then XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →
  (H = (X + G) / 2) →
  (I = (X + H) / 2) →
  (J = (X + I) / 2) →
  (J - X = 4) →
  (Y - X = 64) := by
  sorry

end midpoint_chain_l2749_274982


namespace greatest_x_value_l2749_274924

theorem greatest_x_value (x : ℝ) : 
  x ≠ 2 → 
  (x^2 - 5*x - 14) / (x - 2) = 4 / (x + 4) → 
  x ≤ -2 :=
by sorry

end greatest_x_value_l2749_274924


namespace point_groups_theorem_l2749_274950

theorem point_groups_theorem (n₁ n₂ : ℕ) : 
  n₁ + n₂ = 28 → 
  (n₁ * (n₁ - 1)) / 2 - (n₂ * (n₂ - 1)) / 2 = 81 → 
  (n₁ = 17 ∧ n₂ = 11) ∨ (n₁ = 11 ∧ n₂ = 17) := by
  sorry

end point_groups_theorem_l2749_274950


namespace dice_cube_properties_l2749_274967

/-- Represents a cube formed from 27 dice in a 3x3x3 configuration -/
structure DiceCube where
  size : Nat
  visible_dice : Nat
  faces_per_die : Nat

/-- Calculates the probability of exactly 25 sixes on the surface of the cube -/
def prob_25_sixes (cube : DiceCube) : ℚ :=
  31 / (2^13 * 3^18)

/-- Calculates the probability of at least one "one" on the surface of the cube -/
def prob_at_least_one_one (cube : DiceCube) : ℚ :=
  1 - (5^6 / (2^2 * 3^18))

/-- Calculates the expected number of sixes showing on the surface of the cube -/
def expected_sixes (cube : DiceCube) : ℚ :=
  9

/-- Calculates the expected sum of the numbers on the surface of the cube -/
def expected_sum (cube : DiceCube) : ℚ :=
  6 - (5^6 / (2 * 3^17))

/-- Main theorem stating the properties of the dice cube -/
theorem dice_cube_properties (cube : DiceCube) 
    (h1 : cube.size = 27) 
    (h2 : cube.visible_dice = 26) 
    (h3 : cube.faces_per_die = 6) : 
  (prob_25_sixes cube = 31 / (2^13 * 3^18)) ∧ 
  (prob_at_least_one_one cube = 1 - 5^6 / (2^2 * 3^18)) ∧ 
  (expected_sixes cube = 9) ∧ 
  (expected_sum cube = 6 - 5^6 / (2 * 3^17)) := by
  sorry

#check dice_cube_properties

end dice_cube_properties_l2749_274967


namespace jogger_multiple_l2749_274986

/-- The number of joggers bought by each person -/
structure JoggerPurchase where
  tyson : ℕ
  alexander : ℕ
  christopher : ℕ

/-- The conditions of the jogger purchase problem -/
def JoggerProblem (jp : JoggerPurchase) : Prop :=
  jp.alexander = jp.tyson + 22 ∧
  jp.christopher = 80 ∧
  jp.christopher = jp.alexander + 54 ∧
  ∃ m : ℕ, jp.christopher = m * jp.tyson

theorem jogger_multiple (jp : JoggerPurchase) (h : JoggerProblem jp) :
  ∃ m : ℕ, jp.christopher = m * jp.tyson ∧ m = 20 := by
  sorry

#check jogger_multiple

end jogger_multiple_l2749_274986


namespace arithmetic_sequence_squares_l2749_274995

theorem arithmetic_sequence_squares (x : ℚ) :
  (∃ (a d : ℚ), 
    (5 + x)^2 = a - d ∧
    (7 + x)^2 = a ∧
    (10 + x)^2 = a + d ∧
    d ≠ 0) →
  x = -31/8 ∧ (∃ d : ℚ, d^2 = 1/2) :=
by sorry

end arithmetic_sequence_squares_l2749_274995


namespace phoenix_hike_length_l2749_274936

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
theorem phoenix_hike_length 
  (day1 day2 day3 day4 : ℝ) 
  (first_two_days : day1 + day2 = 22)
  (second_third_avg : (day2 + day3) / 2 = 13)
  (last_two_days : day3 + day4 = 30)
  (first_third_days : day1 + day3 = 26) :
  day1 + day2 + day3 + day4 = 52 :=
by
  sorry


end phoenix_hike_length_l2749_274936


namespace least_coins_coins_exist_l2749_274926

theorem least_coins (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 → n ≥ 24 := by
  sorry

theorem coins_exist : ∃ n : ℕ, n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 ∧ n = 24 := by
  sorry

end least_coins_coins_exist_l2749_274926


namespace geometric_sequence_sum_l2749_274905

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21, 
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 1 + a 3 + a 5 = 21) :
  a 3 + a 5 + a 7 = 42 := by
sorry

end geometric_sequence_sum_l2749_274905


namespace min_xy_value_l2749_274907

theorem min_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 5/x + 3/y = 1) :
  ∀ z : ℝ, x * y ≤ z → 60 ≤ z :=
sorry

end min_xy_value_l2749_274907


namespace correct_reasoning_definitions_l2749_274945

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define the relationship between reasoning types and directions
def reasoningDirection (t : ReasoningType) : ReasoningDirection :=
  match t with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating the correct definitions of reasoning types
theorem correct_reasoning_definitions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end correct_reasoning_definitions_l2749_274945


namespace appears_in_31st_equation_l2749_274925

/-- The first term of the nth equation in the sequence -/
def first_term (n : ℕ) : ℕ := 2 * n^2

/-- The proposition that 2016 appears in the 31st equation -/
theorem appears_in_31st_equation : ∃ k : ℕ, k ≥ first_term 31 ∧ k ≤ first_term 32 ∧ k = 2016 :=
sorry

end appears_in_31st_equation_l2749_274925


namespace math_problems_l2749_274974

theorem math_problems :
  (32 * 3 = 96) ∧
  (43 / 9 = 4 ∧ 43 % 9 = 7) ∧
  (630 / 9 = 70) ∧
  (125 * 47 * 8 = 125 * 8 * 47) := by
  sorry

end math_problems_l2749_274974


namespace complex_square_plus_one_zero_l2749_274960

theorem complex_square_plus_one_zero (x : ℂ) : x^2 + 1 = 0 → x = Complex.I ∨ x = -Complex.I := by
  sorry

end complex_square_plus_one_zero_l2749_274960


namespace fraction_nonnegative_l2749_274947

theorem fraction_nonnegative (x : ℝ) : 
  (x^4 - 4*x^3 + 4*x^2) / (1 - x^3) ≥ 0 ↔ x ∈ Set.Ici 0 :=
by sorry

end fraction_nonnegative_l2749_274947


namespace f_max_is_k_max_b_ac_l2749_274976

/-- The function f(x) = |x-1| - 2|x+1| --/
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

/-- The maximum value of f(x) --/
def k : ℝ := 2

/-- Theorem stating that k is the maximum value of f(x) --/
theorem f_max_is_k : ∀ x : ℝ, f x ≤ k :=
sorry

/-- Theorem for the maximum value of b(a+c) given the conditions --/
theorem max_b_ac (a b c : ℝ) (h : (a^2 + c^2) / 2 + b^2 = k) :
  b * (a + c) ≤ 2 :=
sorry

end f_max_is_k_max_b_ac_l2749_274976


namespace hyperbola_focal_length_l2749_274948

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → 2 * Real.sqrt 10 = 2 * Real.sqrt (7 + 3) :=
by sorry

end hyperbola_focal_length_l2749_274948


namespace geometric_sequence_12th_term_l2749_274906

/-- Given a geometric sequence where the 5th term is 5 and the 8th term is 40, 
    the 12th term is 640. -/
theorem geometric_sequence_12th_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_5th : a 5 = 5) 
  (h_8th : a 8 = 40) : 
  a 12 = 640 := by
sorry


end geometric_sequence_12th_term_l2749_274906


namespace sum_of_factors_l2749_274928

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120 →
  a + b + c + d + e = 25 := by
sorry

end sum_of_factors_l2749_274928


namespace binomial_expansion_coefficient_coefficient_x4_eq_neg35_l2749_274920

theorem binomial_expansion_coefficient (a : ℝ) : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * a^k * a^(7-k)) = (1 + a)^7 :=
sorry

theorem coefficient_x4_eq_neg35 (a : ℝ) : 
  (Nat.choose 7 3) * a^3 = -35 → a = -1 :=
sorry

end binomial_expansion_coefficient_coefficient_x4_eq_neg35_l2749_274920


namespace system_solution_l2749_274983

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 4 - x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (y_pos : y > 0) : y = 2 := by
  sorry

end system_solution_l2749_274983


namespace product_maximization_second_factor_expression_analogous_product_maximization_l2749_274900

theorem product_maximization (a b : ℝ) :
  a ≥ 0 → b ≥ 0 → a + b = 10 → a * b ≤ 25 := by sorry

theorem second_factor_expression (a b : ℝ) :
  a + b = 10 → b = 10 - a := by sorry

theorem analogous_product_maximization (x y : ℝ) :
  x ≥ 0 → y ≥ 0 → x + y = 36 → x * y ≤ 324 := by sorry

end product_maximization_second_factor_expression_analogous_product_maximization_l2749_274900


namespace sum_is_integer_four_or_negative_four_l2749_274939

theorem sum_is_integer_four_or_negative_four 
  (x y z t : ℝ) 
  (h : x / (y + z + t) = y / (z + t + x) ∧ 
       y / (z + t + x) = z / (t + x + y) ∧ 
       z / (t + x + y) = t / (x + y + z)) : 
  (x + y) / (z + t) + (y + z) / (t + x) + (z + t) / (x + y) + (t + x) / (y + z) = 4 ∨
  (x + y) / (z + t) + (y + z) / (t + x) + (z + t) / (x + y) + (t + x) / (y + z) = -4 :=
by sorry

end sum_is_integer_four_or_negative_four_l2749_274939


namespace smallest_side_difference_l2749_274914

theorem smallest_side_difference (P Q R : ℕ) (h_perimeter : P + Q + R = 3010)
  (h_order : P < Q ∧ Q ≤ R) : ∃ (P' Q' R' : ℕ), 
  P' + Q' + R' = 3010 ∧ P' < Q' ∧ Q' ≤ R' ∧ Q' - P' = 1 ∧ 
  ∀ (X Y Z : ℕ), X + Y + Z = 3010 → X < Y → Y ≤ Z → Y - X ≥ 1 := by
  sorry


end smallest_side_difference_l2749_274914


namespace remaining_cube_volume_l2749_274944

/-- Calculates the remaining volume of a cube after removing a cylindrical section. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side^3 - π * cylinder_radius^2 * cylinder_height = 216 - 54 * π :=
by
  sorry

#check remaining_cube_volume

end remaining_cube_volume_l2749_274944
