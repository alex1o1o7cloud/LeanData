import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l1045_104521

theorem imaginary_part_of_complex (z : ℂ) (h : z = 3 - 4 * I) : z.im = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l1045_104521


namespace NUMINAMATH_CALUDE_correct_num_kids_l1045_104581

/-- The number of kids in a group that can wash whiteboards -/
def num_kids : ℕ := 4

/-- The number of whiteboards the group can wash in 20 minutes -/
def group_whiteboards : ℕ := 3

/-- The time in minutes it takes the group to wash their whiteboards -/
def group_time : ℕ := 20

/-- The number of whiteboards one kid can wash -/
def one_kid_whiteboards : ℕ := 6

/-- The time in minutes it takes one kid to wash their whiteboards -/
def one_kid_time : ℕ := 160

/-- Theorem stating that the number of kids in the group is correct -/
theorem correct_num_kids :
  num_kids * (group_whiteboards * one_kid_time) = (one_kid_whiteboards * group_time) :=
by sorry

end NUMINAMATH_CALUDE_correct_num_kids_l1045_104581


namespace NUMINAMATH_CALUDE_complex_exp_conversion_l1045_104515

theorem complex_exp_conversion : Complex.exp (13 * π * Complex.I / 4) * (Real.sqrt 2) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_conversion_l1045_104515


namespace NUMINAMATH_CALUDE_worker_distribution_l1045_104527

theorem worker_distribution (total_workers : ℕ) (male_days female_days : ℝ) 
  (h_total : total_workers = 20)
  (h_male_days : male_days = 2)
  (h_female_days : female_days = 3)
  (h_work_rate : ∀ (x y : ℕ), x + y = total_workers → 
    (x : ℝ) / male_days + (y : ℝ) / female_days = 1) :
  ∃ (male_workers female_workers : ℕ),
    male_workers + female_workers = total_workers ∧
    male_workers = 12 ∧
    female_workers = 8 :=
sorry

end NUMINAMATH_CALUDE_worker_distribution_l1045_104527


namespace NUMINAMATH_CALUDE_team_selection_count_l1045_104546

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 10

-- Define the team size and minimum number of girls required
def team_size : ℕ := 6
def min_girls : ℕ := 3

-- Define the function to calculate the number of ways to select the team
def select_team : ℕ := 
  (Nat.choose num_girls 3 * Nat.choose num_boys 3) +
  (Nat.choose num_girls 4 * Nat.choose num_boys 2) +
  (Nat.choose num_girls 5 * Nat.choose num_boys 1) +
  (Nat.choose num_girls 6)

-- Theorem statement
theorem team_selection_count : select_team = 4770 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l1045_104546


namespace NUMINAMATH_CALUDE_vegetable_field_area_l1045_104536

theorem vegetable_field_area (V W : ℝ) 
  (h1 : (1/2) * V + (1/3) * W = 13)
  (h2 : (1/2) * W + (1/3) * V = 12) : 
  V = 18 := by
sorry

end NUMINAMATH_CALUDE_vegetable_field_area_l1045_104536


namespace NUMINAMATH_CALUDE_paint_difference_l1045_104575

theorem paint_difference (R r : ℝ) (h : R > 0) (h' : r > 0) : 
  (4 / 3 * Real.pi * R^3 - 4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * r^3) = 14.625 →
  (4 * Real.pi * R^2 - 4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 0.84 :=
by sorry

end NUMINAMATH_CALUDE_paint_difference_l1045_104575


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formulas_l1045_104547

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formulas
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = 20)
  (h_diff : a 11 - a 8 = 18) :
  (∃ (an : ℕ → ℝ), ∀ n, an n = 6 * n - 14 ∧ a n = an n) ∧
  (∃ (bn : ℕ → ℝ), ∀ n, bn n = 2 * n - 10 ∧
    (∀ k, ∃ m, a m = bn (3*k - 2) ∧ a (m+1) = bn (3*k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formulas_l1045_104547


namespace NUMINAMATH_CALUDE_cos_90_degrees_equals_zero_l1045_104593

theorem cos_90_degrees_equals_zero : 
  let cos_def : ℝ → ℝ := λ θ => (Real.cos θ)
  let unit_circle_point : ℝ × ℝ := (0, 1)
  cos_def (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_equals_zero_l1045_104593


namespace NUMINAMATH_CALUDE_combined_average_age_l1045_104513

theorem combined_average_age (room_a_count : ℕ) (room_b_count : ℕ) 
  (room_a_avg : ℚ) (room_b_avg : ℚ) :
  room_a_count = 8 →
  room_b_count = 6 →
  room_a_avg = 35 →
  room_b_avg = 30 →
  let total_count := room_a_count + room_b_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg
  (total_age / total_count : ℚ) = 32.86 := by
  sorry

#eval (8 * 35 + 6 * 30) / (8 + 6)

end NUMINAMATH_CALUDE_combined_average_age_l1045_104513


namespace NUMINAMATH_CALUDE_max_value_x_1plusx_3minusx_l1045_104583

theorem max_value_x_1plusx_3minusx (x : ℝ) (h : x > 0) :
  x * (1 + x) * (3 - x) ≤ (70 + 26 * Real.sqrt 13) / 27 ∧
  ∃ y > 0, y * (1 + y) * (3 - y) = (70 + 26 * Real.sqrt 13) / 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_1plusx_3minusx_l1045_104583


namespace NUMINAMATH_CALUDE_namjoons_position_proof_l1045_104514

def namjoons_position (seokjins_position : ℕ) (people_between : ℕ) : ℕ :=
  seokjins_position + people_between

theorem namjoons_position_proof (seokjins_position : ℕ) (people_between : ℕ) :
  namjoons_position seokjins_position people_between = seokjins_position + people_between :=
by
  sorry

end NUMINAMATH_CALUDE_namjoons_position_proof_l1045_104514


namespace NUMINAMATH_CALUDE_positive_rational_cube_sum_representation_l1045_104506

theorem positive_rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_cube_sum_representation_l1045_104506


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l1045_104587

/-- For a parabola y = ax^2 with a > 0, if the length of its latus rectum is 4 units, then a = 1/4 -/
theorem parabola_latus_rectum (a : ℝ) (h1 : a > 0) :
  (1 / a = 4) → a = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l1045_104587


namespace NUMINAMATH_CALUDE_ali_circles_l1045_104557

theorem ali_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ernie_circles : ℕ) : 
  total_boxes = 80 → 
  ali_boxes_per_circle = 8 → 
  ernie_boxes_per_circle = 10 → 
  ernie_circles = 4 → 
  (total_boxes - ernie_boxes_per_circle * ernie_circles) / ali_boxes_per_circle = 5 := by
sorry

end NUMINAMATH_CALUDE_ali_circles_l1045_104557


namespace NUMINAMATH_CALUDE_angle_on_bisector_l1045_104579

-- Define the set of integers
variable (k : ℤ)

-- Define the angle in degrees
def angle (k : ℤ) : ℝ := k * 180 + 135

-- Define the property of being on the bisector of the second or fourth quadrant
def on_bisector_2nd_or_4th (θ : ℝ) : Prop :=
  ∃ n : ℤ, θ = 135 + n * 360 ∨ θ = 315 + n * 360

-- Theorem statement
theorem angle_on_bisector :
  ∀ θ : ℝ, on_bisector_2nd_or_4th θ ↔ ∃ k : ℤ, θ = angle k :=
sorry

end NUMINAMATH_CALUDE_angle_on_bisector_l1045_104579


namespace NUMINAMATH_CALUDE_quarters_collected_per_month_l1045_104539

/-- Represents the number of quarters Phil collected each month during the second year -/
def quarters_per_month : ℕ := sorry

/-- The initial number of quarters Phil had -/
def initial_quarters : ℕ := 50

/-- The number of quarters Phil had after doubling his initial collection -/
def after_doubling : ℕ := 2 * initial_quarters

/-- The number of quarters Phil collected in the third year -/
def third_year_quarters : ℕ := 4

/-- The number of quarters Phil had before losing some -/
def before_loss : ℕ := 140

/-- The number of quarters Phil had after losing some -/
def after_loss : ℕ := 105

/-- Theorem stating that the number of quarters collected each month in the second year is 3 -/
theorem quarters_collected_per_month : 
  quarters_per_month = 3 ∧
  after_doubling + 12 * quarters_per_month + third_year_quarters = before_loss ∧
  before_loss * 3 = after_loss * 4 := by sorry

end NUMINAMATH_CALUDE_quarters_collected_per_month_l1045_104539


namespace NUMINAMATH_CALUDE_halfway_fraction_l1045_104573

theorem halfway_fraction (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = (a + b) / 2 → c = 3/8 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1045_104573


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1045_104556

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1045_104556


namespace NUMINAMATH_CALUDE_max_product_sides_l1045_104570

/-- A convex quadrilateral with side lengths a, b, c, d and diagonal lengths e, f --/
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  convex : True  -- Assuming convexity without formal definition
  max_side : max a (max b (max c (max d (max e f)))) = 1

/-- The maximum product of side lengths in a convex quadrilateral with max side length 1 is 2 - √3 --/
theorem max_product_sides (q : ConvexQuadrilateral) : 
  ∃ (m : ℝ), m = q.a * q.b * q.c * q.d ∧ m ≤ 2 - Real.sqrt 3 := by
  sorry

#check max_product_sides

end NUMINAMATH_CALUDE_max_product_sides_l1045_104570


namespace NUMINAMATH_CALUDE_age_difference_l1045_104532

/-- Proves that A was half of B's age 10 years ago given the conditions -/
theorem age_difference (a b : ℕ) : 
  (a : ℚ) / b = 3 / 4 →  -- ratio of present ages is 3:4
  a + b = 35 →          -- sum of present ages is 35
  ∃ (y : ℕ), y = 10 ∧ (a - y : ℚ) = (1 / 2) * (b - y) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1045_104532


namespace NUMINAMATH_CALUDE_symmetric_point_l1045_104500

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line x + y = 0 -/
def symmetryLine (p : Point) : Prop :=
  p.x + p.y = 0

/-- Defines the property of two points being symmetric with respect to the line x + y = 0 -/
def isSymmetric (p1 p2 : Point) : Prop :=
  symmetryLine ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

/-- Theorem: The point symmetric to P(2, 5) with respect to the line x + y = 0 has coordinates (-5, -2) -/
theorem symmetric_point : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨-5, -2⟩
  isSymmetric p1 p2 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_l1045_104500


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1045_104563

def K : ℚ := (1 : ℚ) + (1/2 : ℚ) + (1/3 : ℚ) + (1/4 : ℚ)

def S (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * K

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_S :
  ∀ n : ℕ, (n > 0 ∧ is_integer (S n)) → n ≥ 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l1045_104563


namespace NUMINAMATH_CALUDE_expected_sedans_is_48_l1045_104520

/-- Represents the car dealership's sales plan -/
structure SalesPlan where
  sportsCarRatio : ℕ
  sedanRatio : ℕ
  totalTarget : ℕ
  plannedSportsCars : ℕ

/-- Calculates the number of sedans to be sold based on the sales plan -/
def expectedSedans (plan : SalesPlan) : ℕ :=
  plan.sedanRatio * plan.plannedSportsCars / plan.sportsCarRatio

/-- Theorem stating that the expected number of sedans is 48 given the specified conditions -/
theorem expected_sedans_is_48 (plan : SalesPlan)
  (h1 : plan.sportsCarRatio = 5)
  (h2 : plan.sedanRatio = 8)
  (h3 : plan.totalTarget = 78)
  (h4 : plan.plannedSportsCars = 30)
  (h5 : plan.plannedSportsCars + expectedSedans plan = plan.totalTarget) :
  expectedSedans plan = 48 := by
  sorry

#eval expectedSedans {sportsCarRatio := 5, sedanRatio := 8, totalTarget := 78, plannedSportsCars := 30}

end NUMINAMATH_CALUDE_expected_sedans_is_48_l1045_104520


namespace NUMINAMATH_CALUDE_equation_solutions_l1045_104530

theorem equation_solutions : 
  let f (r : ℝ) := (r^2 - 6*r + 9) / (r^2 - 9*r + 14)
  let g (r : ℝ) := (r^2 - 4*r - 21) / (r^2 - 2*r - 35)
  ∀ r : ℝ, f r = g r ↔ (r = 3 ∨ r = (-1 + Real.sqrt 69) / 2 ∨ r = (-1 - Real.sqrt 69) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1045_104530


namespace NUMINAMATH_CALUDE_scientific_notation_of_43000000_l1045_104564

theorem scientific_notation_of_43000000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43000000 = a * (10 : ℝ) ^ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_43000000_l1045_104564


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_midpoint_distances_l1045_104578

/-- The volume of a regular tetrahedron, given specific midpoint distances -/
theorem tetrahedron_volume_from_midpoint_distances :
  ∀ (midpoint_to_face midpoint_to_edge : ℝ),
    midpoint_to_face = 2 →
    midpoint_to_edge = Real.sqrt 5 →
    ∃ (volume : ℝ), volume = 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_midpoint_distances_l1045_104578


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1045_104543

/-- Coefficient of the r-th term in the binomial expansion of (x + 1/(2√x))^n -/
def coeff (n : ℕ) (r : ℕ) : ℚ :=
  (1 / 2^r) * (n.choose r)

/-- The expansion of (x + 1/(2√x))^n has coefficients forming an arithmetic sequence
    for the first three terms -/
def arithmetic_sequence (n : ℕ) : Prop :=
  coeff n 0 - coeff n 1 = coeff n 1 - coeff n 2

/-- The r-th term has the maximum coefficient in the expansion -/
def max_coeff (n : ℕ) (r : ℕ) : Prop :=
  ∀ k, k ≠ r → coeff n r ≥ coeff n k

theorem binomial_expansion_properties :
  ∃ n : ℕ,
    arithmetic_sequence n ∧
    max_coeff n 2 ∧
    max_coeff n 3 ∧
    ∀ r, r ≠ 2 ∧ r ≠ 3 → ¬(max_coeff n r) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1045_104543


namespace NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l1045_104510

def is_identity_function (f : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, f m = m

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ m : ℕ, f m = 1 ↔ m = 1)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n / f (Nat.gcd m n))
  (h3 : ∀ m : ℕ, (f^[2012]) m = m) :
  is_identity_function f :=
sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_conditions_l1045_104510


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l1045_104523

/-- The number of diagonals in a regular nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

theorem nonagon_diagonal_count :
  nonagon_diagonals = (nonagon_vertices.choose 2) - nonagon_sides := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l1045_104523


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l1045_104549

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24,
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768 -/
theorem equilateral_triangle_sum_product (a b c : ℂ) : 
  (∃ (ω : ℂ), ω ^ 3 = 1 ∧ ω ≠ 1 ∧ c - a = (b - a) * ω) →
  Complex.abs (b - a) = 24 →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l1045_104549


namespace NUMINAMATH_CALUDE_intern_teacher_arrangements_l1045_104540

def num_teachers : ℕ := 5
def num_classes : ℕ := 3

def arrangements (n m : ℕ) : ℕ := sorry

theorem intern_teacher_arrangements :
  let remaining_teachers := num_teachers - 1
  arrangements remaining_teachers num_classes = 50 :=
by sorry

end NUMINAMATH_CALUDE_intern_teacher_arrangements_l1045_104540


namespace NUMINAMATH_CALUDE_max_quarters_and_dimes_l1045_104542

theorem max_quarters_and_dimes (total : ℚ) (h_total : total = 425/100) :
  ∃ (quarters dimes pennies : ℕ),
    quarters = dimes ∧
    quarters * (25 : ℚ)/100 + dimes * (10 : ℚ)/100 + pennies * (1 : ℚ)/100 = total ∧
    ∀ q d p : ℕ, q = d →
      q * (25 : ℚ)/100 + d * (10 : ℚ)/100 + p * (1 : ℚ)/100 = total →
      q ≤ quarters :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_and_dimes_l1045_104542


namespace NUMINAMATH_CALUDE_research_budget_allocation_l1045_104599

theorem research_budget_allocation (microphotonics home_electronics gmo industrial_lubricants basic_astrophysics food_additives : ℝ) : 
  microphotonics = 10 →
  home_electronics = 24 →
  gmo = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics / 100 = 50.4 / 360 →
  microphotonics + home_electronics + gmo + industrial_lubricants + basic_astrophysics + food_additives = 100 →
  food_additives = 15 := by
  sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l1045_104599


namespace NUMINAMATH_CALUDE_odd_triangle_perimeter_l1045_104590

/-- A triangle with two sides of lengths 2 and 3, and the third side being an odd number -/
structure OddTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℕ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : Odd side3
  h4 : side3 > 0  -- Ensuring positive length
  h5 : side1 + side2 > side3  -- Triangle inequality
  h6 : side1 + side3 > side2
  h7 : side2 + side3 > side1

/-- The perimeter of an OddTriangle is 8 -/
theorem odd_triangle_perimeter (t : OddTriangle) : t.side1 + t.side2 + t.side3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_odd_triangle_perimeter_l1045_104590


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1045_104560

theorem integer_solutions_of_equation :
  let S : Set (ℤ × ℤ) := {(x, y) | x * y - 2 * x - 2 * y + 7 = 0}
  S = {(5, 1), (-1, 3), (3, -1), (1, 5)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1045_104560


namespace NUMINAMATH_CALUDE_janet_sculpture_weight_l1045_104505

/-- Calculates the weight of the second sculpture given Janet's work details --/
theorem janet_sculpture_weight
  (exterminator_rate : ℝ)
  (sculpture_rate : ℝ)
  (exterminator_hours : ℝ)
  (first_sculpture_weight : ℝ)
  (total_income : ℝ)
  (h1 : exterminator_rate = 70)
  (h2 : sculpture_rate = 20)
  (h3 : exterminator_hours = 20)
  (h4 : first_sculpture_weight = 5)
  (h5 : total_income = 1640)
  : ∃ (second_sculpture_weight : ℝ),
    second_sculpture_weight = 7 ∧
    total_income = exterminator_rate * exterminator_hours +
                   sculpture_rate * (first_sculpture_weight + second_sculpture_weight) :=
by
  sorry

end NUMINAMATH_CALUDE_janet_sculpture_weight_l1045_104505


namespace NUMINAMATH_CALUDE_polynomial_positive_reals_l1045_104588

def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_reals :
  (∀ x y : ℝ, P x y > 0) ∧
  (∀ c : ℝ, c > 0 → ∃ x y : ℝ, P x y = c) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_positive_reals_l1045_104588


namespace NUMINAMATH_CALUDE_jumping_contest_l1045_104533

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l1045_104533


namespace NUMINAMATH_CALUDE_parabola_chord_sum_constant_l1045_104569

/-- Theorem: For a parabola y = x^2, if there exists a constant d such that
    for all chords AB passing through D = (0,d), the sum s = 1/AD^2 + 1/BD^2 is constant,
    then d = 1/2 and s = 4. -/
theorem parabola_chord_sum_constant (d : ℝ) :
  (∃ (s : ℝ), ∀ (A B : ℝ × ℝ),
    A.2 = A.1^2 ∧ B.2 = B.1^2 ∧  -- A and B are on the parabola y = x^2
    (∃ (m : ℝ), A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →  -- AB passes through (0,d)
    1 / ((A.1^2 + (A.2 - d)^2) : ℝ) + 1 / ((B.1^2 + (B.2 - d)^2) : ℝ) = s) →
  d = 1/2 ∧ s = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_sum_constant_l1045_104569


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1045_104582

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_geometric_progression :
  ∀ (a r : ℝ),
  (geometric_progression a r 1 = 6^(1/2)) →
  (geometric_progression a r 2 = 6^(1/6)) →
  (geometric_progression a r 3 = 6^(1/12)) →
  (geometric_progression a r 4 = 6^0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_progression_l1045_104582


namespace NUMINAMATH_CALUDE_service_center_location_l1045_104585

/-- The location of the service center on a highway given the locations of two exits -/
theorem service_center_location 
  (fourth_exit_location : ℝ) 
  (twelfth_exit_location : ℝ) 
  (h1 : fourth_exit_location = 50)
  (h2 : twelfth_exit_location = 190)
  (service_center_location : ℝ) 
  (h3 : service_center_location = fourth_exit_location + (twelfth_exit_location - fourth_exit_location) / 2) :
  service_center_location = 120 := by
sorry

end NUMINAMATH_CALUDE_service_center_location_l1045_104585


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l1045_104568

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of dozens of cards each person has -/
def dozens_per_person : ℕ := 9

/-- The number of items in one dozen -/
def items_per_dozen : ℕ := 12

/-- Theorem: The total number of Pokemon cards owned by 4 people is 432,
    given that each person has 9 dozen cards and one dozen equals 12 items. -/
theorem total_pokemon_cards :
  num_people * dozens_per_person * items_per_dozen = 432 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l1045_104568


namespace NUMINAMATH_CALUDE_blocks_count_l1045_104580

/-- The number of blocks in Jacob's toy bin --/
def total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) : ℕ := red + yellow + blue

/-- Theorem: Given the conditions, the total number of blocks is 75 --/
theorem blocks_count :
  let red : ℕ := 18
  let yellow : ℕ := red + 7
  let blue : ℕ := red + 14
  total_blocks red yellow blue = 75 := by sorry

end NUMINAMATH_CALUDE_blocks_count_l1045_104580


namespace NUMINAMATH_CALUDE_intersection_condition_l1045_104511

/-- The function f(x) = (m-3)x^2 - 4x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 - 4*x + 2

/-- The graph of f intersects the x-axis at only one point -/
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! x, f m x = 0

/-- Theorem: The graph of f(x) = (m-3)x^2 - 4x + 2 intersects the x-axis at only one point
    if and only if m = 3 or m = 5 -/
theorem intersection_condition (m : ℝ) :
  intersects_at_one_point m ↔ m = 3 ∨ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1045_104511


namespace NUMINAMATH_CALUDE_triangle_area_l1045_104525

/-- The area of a triangle with vertices at (2, 2), (2, -3), and (7, 2) is 12.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (2, -3)
  let C : ℝ × ℝ := (7, 2)
  (1/2 : ℝ) * |A.1 - C.1| * |A.2 - B.2| = 12.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1045_104525


namespace NUMINAMATH_CALUDE_base_difference_calculation_l1045_104518

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem stating the result of the calculation -/
theorem base_difference_calculation :
  base6ToBase10 52143 - base7ToBase10 4310 = 5449 := by sorry

end NUMINAMATH_CALUDE_base_difference_calculation_l1045_104518


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1045_104561

def f (x : ℝ) := -x^2 + 4*x - 2

theorem min_value_on_interval :
  ∀ x ∈ Set.Icc 1 4, f x ≥ -2 ∧ ∃ y ∈ Set.Icc 1 4, f y = -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1045_104561


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l1045_104558

/-- Time taken for a train to pass a stationary point -/
theorem train_passing_time (platform_length : Real) (platform_passing_time : Real) (train_speed_kmh : Real) : Real :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let train_length := platform_passing_time * train_speed_ms - platform_length
  train_length / train_speed_ms

/-- Proof that a train passing a 30.0024-meter platform in 22 seconds at 54 km/hr takes approximately 20 seconds to pass a stationary point -/
theorem train_passing_man_time : 
  abs (train_passing_time 30.0024 22 54 - 20) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_man_time_l1045_104558


namespace NUMINAMATH_CALUDE_questions_left_blank_l1045_104507

/-- Given a math test with a total number of questions and the number of questions answered,
    prove that the number of questions left blank is the difference between the total and answered questions. -/
theorem questions_left_blank (total : ℕ) (answered : ℕ) (h : answered ≤ total) :
  total - answered = total - answered :=
by sorry

end NUMINAMATH_CALUDE_questions_left_blank_l1045_104507


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1045_104555

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    P = 21/4 ∧ Q = 15 ∧ R = -11/2 ∧
    ∀ (x : ℚ), x ≠ 2 → x ≠ 4 →
      5*x + 1 = (x - 4)*(x - 2)^2 * (P/(x - 4) + Q/(x - 2) + R/(x - 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1045_104555


namespace NUMINAMATH_CALUDE_expression_simplification_l1045_104577

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 - 2*a*b + b^2) / (2*a*b) - (2*a*b - b^2) / (3*a*b - 3*a^2) = (a - b)^2 / (2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1045_104577


namespace NUMINAMATH_CALUDE_train_speed_theorem_l1045_104503

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 70

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 80

/-- The time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total travel time of the first train in hours -/
def first_train_travel_time : ℝ := 8

/-- The total travel time of the second train in hours -/
def second_train_travel_time : ℝ := 7

theorem train_speed_theorem : 
  first_train_speed * first_train_travel_time = 
  second_train_speed * second_train_travel_time :=
by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l1045_104503


namespace NUMINAMATH_CALUDE_orchid_seed_weight_scientific_notation_l1045_104596

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation := sorry

theorem orchid_seed_weight_scientific_notation :
  toScientificNotation 0.0000005 = ScientificNotation.mk 5 (-7) (by norm_num) := by sorry

end NUMINAMATH_CALUDE_orchid_seed_weight_scientific_notation_l1045_104596


namespace NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l1045_104535

theorem max_side_length_of_special_triangle :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 30 →
  a < b + c ∧ b < a + c ∧ c < a + b →
  ∀ x : ℕ, x ≤ a ∧ x ≤ b ∧ x ≤ c →
  x ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l1045_104535


namespace NUMINAMATH_CALUDE_attendance_difference_l1045_104594

/-- Proves that the difference in student attendance between the second and first day is 40 --/
theorem attendance_difference (total_students : ℕ) (total_absent : ℕ) 
  (absent_day1 absent_day2 absent_day3 : ℕ) :
  total_students = 280 →
  total_absent = 240 →
  absent_day1 + absent_day2 + absent_day3 = total_absent →
  absent_day2 = 2 * absent_day3 →
  absent_day3 = total_students / 7 →
  absent_day2 < absent_day1 →
  (total_students - absent_day2) - (total_students - absent_day1) = 40 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l1045_104594


namespace NUMINAMATH_CALUDE_wood_carvings_per_shelf_example_l1045_104586

/-- Given a total number of wood carvings and a number of shelves,
    calculate the number of wood carvings per shelf. -/
def woodCarvingsPerShelf (totalCarvings : ℕ) (numShelves : ℕ) : ℕ :=
  totalCarvings / numShelves

/-- Theorem stating that with 56 total wood carvings and 7 shelves,
    each shelf contains 8 wood carvings. -/
theorem wood_carvings_per_shelf_example :
  woodCarvingsPerShelf 56 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_wood_carvings_per_shelf_example_l1045_104586


namespace NUMINAMATH_CALUDE_sum_of_squares_130_l1045_104567

theorem sum_of_squares_130 (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  a^2 + b^2 = 130 → 
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_130_l1045_104567


namespace NUMINAMATH_CALUDE_product_ab_equals_one_l1045_104508

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem product_ab_equals_one (h1 : a - b = 4) (h2 : a^2 + b^2 = 18) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_one_l1045_104508


namespace NUMINAMATH_CALUDE_rectangle_cut_squares_l1045_104528

/-- Given a rectangle with length 90 cm and width 42 cm, prove that when cut into the largest possible squares with integer side lengths, the minimum number of squares is 105 and their total perimeter is 2520 cm. -/
theorem rectangle_cut_squares (length width : ℕ) (h1 : length = 90) (h2 : width = 42) :
  let side_length := Nat.gcd length width
  let num_squares := (length / side_length) * (width / side_length)
  let total_perimeter := num_squares * (4 * side_length)
  num_squares = 105 ∧ total_perimeter = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_squares_l1045_104528


namespace NUMINAMATH_CALUDE_sadaf_height_l1045_104551

theorem sadaf_height (lily_height : ℝ) (anika_height : ℝ) (sadaf_height : ℝ) 
  (h1 : lily_height = 90)
  (h2 : anika_height = 4/3 * lily_height)
  (h3 : sadaf_height = 5/4 * anika_height) :
  sadaf_height = 150 := by
  sorry

end NUMINAMATH_CALUDE_sadaf_height_l1045_104551


namespace NUMINAMATH_CALUDE_ratio_of_segments_l1045_104548

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1045_104548


namespace NUMINAMATH_CALUDE_julia_tag_total_l1045_104512

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_total_l1045_104512


namespace NUMINAMATH_CALUDE_non_officers_count_l1045_104550

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Average salary of all employees in the office -/
def avg_salary_all : ℕ := 120

/-- Average salary of officers -/
def avg_salary_officers : ℕ := 430

/-- Average salary of non-officers -/
def avg_salary_non_officers : ℕ := 110

/-- Number of officers -/
def num_officers : ℕ := 15

/-- Theorem stating that the number of non-officers is 465 -/
theorem non_officers_count : num_non_officers = 465 := by
  sorry

end NUMINAMATH_CALUDE_non_officers_count_l1045_104550


namespace NUMINAMATH_CALUDE_parabola_equilateral_triangle_p_value_l1045_104524

/-- Parabola defined by x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle defined by center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: For a parabola C: x^2 = 2py (p > 0), if there exists a point A on C
    such that A is equidistant from O(0,0) and M(0,9), and triangle ABO is equilateral
    (where B is another point on the circle with center M and radius |OA|),
    then p = 3/4 -/
theorem parabola_equilateral_triangle_p_value
  (C : Parabola)
  (A : Point)
  (h_A_on_C : A.x^2 = 2 * C.p * A.y)
  (h_A_equidistant : A.x^2 + A.y^2 = A.x^2 + (A.y - 9)^2)
  (h_ABO_equilateral : ∃ B : Point, B.x^2 + (B.y - 9)^2 = A.x^2 + A.y^2 ∧
                       A.x^2 + A.y^2 = (A.x - B.x)^2 + (A.y - B.y)^2) :
  C.p = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equilateral_triangle_p_value_l1045_104524


namespace NUMINAMATH_CALUDE_fraction_inequality_l1045_104545

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  c / a > d / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1045_104545


namespace NUMINAMATH_CALUDE_min_max_f_l1045_104572

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max) ∧
    min = -3 * Real.pi / 2 ∧
    max = Real.pi / 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_max_f_l1045_104572


namespace NUMINAMATH_CALUDE_percentage_calculation_l1045_104517

theorem percentage_calculation (n : ℝ) : n = 4000 → (0.15 * (0.30 * (0.50 * n))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1045_104517


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l1045_104598

theorem pythagorean_triple_properties (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (Even a ∨ Even b) ∧
  (3 ∣ a ∨ 3 ∣ b) ∧
  (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l1045_104598


namespace NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_2_l1045_104519

theorem gcd_13m_plus_4_7m_plus_2_max_2 :
  (∀ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_2_l1045_104519


namespace NUMINAMATH_CALUDE_planes_distance_l1045_104591

/-- Represents a plane in 3D space defined by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The two planes in the problem -/
def plane1 : Plane := ⟨3, -1, 2, -4⟩
def plane2 : Plane := ⟨6, -2, 4, 3⟩

theorem planes_distance :
  distance_between_planes plane1 plane2 = 11 * Real.sqrt 14 / 28 := by
  sorry

end NUMINAMATH_CALUDE_planes_distance_l1045_104591


namespace NUMINAMATH_CALUDE_toms_ribbon_length_l1045_104504

theorem toms_ribbon_length 
  (num_gifts : ℕ) 
  (ribbon_per_gift : ℝ) 
  (remaining_ribbon : ℝ) 
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : remaining_ribbon = 3) :
  (num_gifts : ℝ) * ribbon_per_gift + remaining_ribbon = 15 := by
  sorry

end NUMINAMATH_CALUDE_toms_ribbon_length_l1045_104504


namespace NUMINAMATH_CALUDE_smallest_value_in_range_l1045_104559

theorem smallest_value_in_range (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y := by
  sorry

#check smallest_value_in_range

end NUMINAMATH_CALUDE_smallest_value_in_range_l1045_104559


namespace NUMINAMATH_CALUDE_zeros_of_odd_and_even_functions_l1045_104571

-- Define odd and even functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def EvenFunction (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the number of zeros for a function
def NumberOfZeros (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem zeros_of_odd_and_even_functions 
  (f g : ℝ → ℝ) 
  (hf : OddFunction f) 
  (hg : EvenFunction g) :
  (∃ k : ℕ, NumberOfZeros f = 2 * k + 1) ∧ 
  (∃ m : ℕ, NumberOfZeros g = m) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_odd_and_even_functions_l1045_104571


namespace NUMINAMATH_CALUDE_pictures_per_album_l1045_104565

/-- Given the number of pictures uploaded from phone and camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

#eval (35 + 5) / 5  -- Expected output: 8

end NUMINAMATH_CALUDE_pictures_per_album_l1045_104565


namespace NUMINAMATH_CALUDE_test_questions_count_l1045_104541

theorem test_questions_count (total_points : ℕ) (two_point_questions : ℕ) :
  total_points = 100 →
  two_point_questions = 30 →
  ∃ (four_point_questions : ℕ),
    total_points = 2 * two_point_questions + 4 * four_point_questions ∧
    two_point_questions + four_point_questions = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l1045_104541


namespace NUMINAMATH_CALUDE_floor_with_133_black_tiles_has_4489_total_tiles_l1045_104501

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side : ℕ
  black_tiles : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def diagonal_tiles (floor : TiledFloor) : ℕ :=
  2 * floor.side - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side ^ 2

/-- Theorem stating that a square floor with 133 black tiles on its diagonals has 4489 total tiles -/
theorem floor_with_133_black_tiles_has_4489_total_tiles (floor : TiledFloor) 
    (h : floor.black_tiles = 133) : total_tiles floor = 4489 := by
  sorry


end NUMINAMATH_CALUDE_floor_with_133_black_tiles_has_4489_total_tiles_l1045_104501


namespace NUMINAMATH_CALUDE_min_A_over_B_l1045_104592

theorem min_A_over_B (x A B : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B + 3) :
  A / B ≥ 6 + 2 * Real.sqrt 11 ∧
  (A / B = 6 + 2 * Real.sqrt 11 ↔ B = Real.sqrt 11) :=
sorry

end NUMINAMATH_CALUDE_min_A_over_B_l1045_104592


namespace NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l1045_104576

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2, 3) = (1, -6) := by sorry

-- Theorem for the pre-image of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l1045_104576


namespace NUMINAMATH_CALUDE_daejun_marbles_l1045_104566

/-- The number of bags Daejun has -/
def num_bags : ℕ := 20

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 156

/-- The total number of marbles Daejun has -/
def total_marbles : ℕ := num_bags * marbles_per_bag

theorem daejun_marbles : total_marbles = 3120 := by
  sorry

end NUMINAMATH_CALUDE_daejun_marbles_l1045_104566


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1045_104574

theorem polynomial_coefficient_sum : 
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6 - x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 28 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1045_104574


namespace NUMINAMATH_CALUDE_equation_solution_l1045_104597

theorem equation_solution : ∃! y : ℚ, y ≠ 2 ∧ (7 * y) / (y - 2) - 4 / (y - 2) = 3 / (y - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1045_104597


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1045_104502

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x - 3 = a * (x - h)^2 + k) →
  a + h + k = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1045_104502


namespace NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l1045_104531

/-- The minimum amount spent on boxes for packaging a collection -/
theorem minimum_amount_spent_on_boxes
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (box_cost : ℝ)
  (total_volume : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : box_cost = 0.5)
  (h5 : total_volume = 2160000) :
  ⌈total_volume / (box_length * box_width * box_height)⌉ * box_cost = 225 :=
sorry

end NUMINAMATH_CALUDE_minimum_amount_spent_on_boxes_l1045_104531


namespace NUMINAMATH_CALUDE_trailing_zeros_625_factorial_l1045_104595

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeros_625_factorial :
  trailingZeros (factorial 625) = 156 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_625_factorial_l1045_104595


namespace NUMINAMATH_CALUDE_probability_science_second_given_arts_first_l1045_104562

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of science questions
def science_questions : ℕ := 3

-- Define the number of arts questions
def arts_questions : ℕ := 2

-- Define the probability of drawing an arts question in the first draw
def prob_arts_first : ℚ := arts_questions / total_questions

-- Define the probability of drawing a science question in the second draw given an arts question was drawn first
def prob_science_second_given_arts_first : ℚ := science_questions / (total_questions - 1)

-- Theorem statement
theorem probability_science_second_given_arts_first :
  prob_science_second_given_arts_first = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_science_second_given_arts_first_l1045_104562


namespace NUMINAMATH_CALUDE_correct_remaining_insects_l1045_104516

/-- Calculates the number of remaining insects in the playground -/
def remaining_insects (spiders ants ladybugs flown_away : ℕ) : ℕ :=
  spiders + ants + ladybugs - flown_away

/-- Theorem stating that the number of remaining insects is correct -/
theorem correct_remaining_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_insects_l1045_104516


namespace NUMINAMATH_CALUDE_michael_saved_five_cookies_l1045_104552

/-- The number of cookies Michael saved to give Sarah -/
def michaels_cookies (sarahs_initial_cupcakes : ℕ) (sarahs_final_desserts : ℕ) : ℕ :=
  sarahs_final_desserts - (sarahs_initial_cupcakes - sarahs_initial_cupcakes / 3)

theorem michael_saved_five_cookies :
  michaels_cookies 9 11 = 5 :=
by sorry

end NUMINAMATH_CALUDE_michael_saved_five_cookies_l1045_104552


namespace NUMINAMATH_CALUDE_equation_characterizes_triangles_l1045_104529

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The equation given in the problem. -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.c^2 - t.a^2) / t.b + (t.b^2 - t.c^2) / t.a = t.b - t.a

/-- A right-angled triangle. -/
def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- An isosceles triangle. -/
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- The main theorem. -/
theorem equation_characterizes_triangles (t : Triangle) :
  satisfies_equation t ↔ is_right_angled t ∨ is_isosceles t := by
  sorry

end NUMINAMATH_CALUDE_equation_characterizes_triangles_l1045_104529


namespace NUMINAMATH_CALUDE_inequality_proof_l1045_104584

theorem inequality_proof (x : ℝ) : 
  -2 < (x^2 - 10*x + 9) / (x^2 - 4*x + 8) ∧ (x^2 - 10*x + 9) / (x^2 - 4*x + 8) < 2 ↔ 
  1/3 < x ∧ x < 14/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1045_104584


namespace NUMINAMATH_CALUDE_flat_percentage_calculation_l1045_104553

/-- The price of each flat -/
def flat_price : ℚ := 675958

/-- The overall gain from the transaction -/
def overall_gain : ℚ := 144 / 100

/-- The percentage of gain or loss on each flat -/
noncomputable def percentage : ℚ := overall_gain / (2 * flat_price) * 100

theorem flat_percentage_calculation :
  ∃ (ε : ℚ), abs (percentage - 1065 / 100000000) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_flat_percentage_calculation_l1045_104553


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1045_104589

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1045_104589


namespace NUMINAMATH_CALUDE_inequality_implies_b_leq_c_l1045_104544

theorem inequality_implies_b_leq_c
  (a b c x y : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (pos_x : 0 < x)
  (pos_y : 0 < y)
  (h : a * x + b * y ≤ b * x + c * y ∧ b * x + c * y ≤ c * x + a * y) :
  b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_b_leq_c_l1045_104544


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1045_104526

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  A = (h^2) / 4 →        -- area formula for isosceles right triangle
  A = 18 := by
    sorry

#check isosceles_right_triangle_area

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1045_104526


namespace NUMINAMATH_CALUDE_quadratic_solution_set_implies_linear_and_inverse_quadratic_l1045_104509

/-- Given a quadratic function f(x) = ax² + bx + c, where a, b, and c are real numbers and a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

theorem quadratic_solution_set_implies_linear_and_inverse_quadratic
  (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 3) →
  (∀ x, b * x - c > 0 ↔ x < 6) ∧
  (∀ x, c * x^2 - b * x + a ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_implies_linear_and_inverse_quadratic_l1045_104509


namespace NUMINAMATH_CALUDE_hyperbola_t_squared_l1045_104537

/-- A hyperbola is defined by its center, orientation, and three points it passes through. -/
structure Hyperbola where
  center : ℝ × ℝ
  horizontalOpening : Bool
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Given a hyperbola with specific properties, calculate t². -/
def calculateTSquared (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating that for a hyperbola with given properties, t² = 45/4. -/
theorem hyperbola_t_squared 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_opening : h.horizontalOpening = true)
  (h_point1 : h.point1 = (-3, 4))
  (h_point2 : h.point2 = (-3, 0))
  (h_point3 : ∃ t : ℝ, h.point3 = (t, 3)) :
  calculateTSquared h = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_squared_l1045_104537


namespace NUMINAMATH_CALUDE_mixture_replacement_l1045_104538

/-- Given a mixture of liquids A and B, this theorem proves that
    replacing a certain amount of the mixture with liquid B
    results in the specified final ratio. -/
theorem mixture_replacement (initial_a initial_b replacement : ℚ) :
  initial_a = 16 →
  initial_b = 4 →
  (initial_a - 4/5 * replacement) / (initial_b + 4/5 * replacement) = 2/3 →
  replacement = 10 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_l1045_104538


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_exists_l1045_104554

theorem min_value_of_sum (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∀ x y : ℝ, x > -1 → y > -2 → (x + 1) * (y + 2) = 16 → a + b ≤ x + y :=
sorry

theorem min_value_exists (a b : ℝ) (ha : a > -1) (hb : b > -2) (hab : (a + 1) * (b + 2) = 16) :
  ∃ x y : ℝ, x > -1 ∧ y > -2 ∧ (x + 1) * (y + 2) = 16 ∧ x + y = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_exists_l1045_104554


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1045_104534

theorem complex_number_in_first_quadrant :
  let z : ℂ := 1 / (1 - Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1045_104534


namespace NUMINAMATH_CALUDE_negation_equivalence_l1045_104522

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1045_104522
