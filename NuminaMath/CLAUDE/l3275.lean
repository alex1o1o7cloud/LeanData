import Mathlib

namespace NUMINAMATH_CALUDE_snack_slices_theorem_l3275_327560

/-- Represents the household bread consumption scenario -/
structure HouseholdBread where
  members : ℕ
  breakfast_slices_per_member : ℕ
  slices_per_loaf : ℕ
  loaves_consumed : ℕ
  days_lasted : ℕ

/-- Calculate the number of slices each member consumes for snacks daily -/
def snack_slices_per_member_per_day (hb : HouseholdBread) : ℕ :=
  let total_slices := hb.loaves_consumed * hb.slices_per_loaf
  let breakfast_slices := hb.members * hb.breakfast_slices_per_member * hb.days_lasted
  let snack_slices := total_slices - breakfast_slices
  snack_slices / (hb.members * hb.days_lasted)

/-- Theorem stating that each member consumes 2 slices of bread for snacks daily -/
theorem snack_slices_theorem (hb : HouseholdBread) 
  (h1 : hb.members = 4)
  (h2 : hb.breakfast_slices_per_member = 3)
  (h3 : hb.slices_per_loaf = 12)
  (h4 : hb.loaves_consumed = 5)
  (h5 : hb.days_lasted = 3) :
  snack_slices_per_member_per_day hb = 2 := by
  sorry


end NUMINAMATH_CALUDE_snack_slices_theorem_l3275_327560


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_405_l3275_327595

theorem tens_digit_of_3_to_405 : ∃ n : ℕ, 3^405 ≡ 40 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_405_l3275_327595


namespace NUMINAMATH_CALUDE_polynomial_equality_l3275_327599

theorem polynomial_equality (x : ℝ) (h : 2 * x^2 - x = 1) :
  4 * x^4 - 4 * x^3 + 3 * x^2 - x - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3275_327599


namespace NUMINAMATH_CALUDE_P_intersect_Q_l3275_327524

/-- The set P of vectors -/
def P : Set (ℝ × ℝ) := {a | ∃ m : ℝ, a = (1, 0) + m • (0, 1)}

/-- The set Q of vectors -/
def Q : Set (ℝ × ℝ) := {b | ∃ n : ℝ, b = (1, 1) + n • (-1, 1)}

/-- The theorem stating that the intersection of P and Q is the singleton set containing (1,1) -/
theorem P_intersect_Q : P ∩ Q = {(1, 1)} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l3275_327524


namespace NUMINAMATH_CALUDE_division_problem_l3275_327521

theorem division_problem (x : ℝ) : (x / 0.08 = 800) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3275_327521


namespace NUMINAMATH_CALUDE_probability_not_distinct_roots_greater_than_two_l3275_327527

def is_valid_pair (a c : ℤ) : Prop :=
  |a| ≤ 6 ∧ |c| ≤ 6

def has_distinct_roots_greater_than_two (a c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 2 ∧ x₂ > 2 ∧ a * x₁^2 - 3 * a * x₁ + c = 0 ∧ a * x₂^2 - 3 * a * x₂ + c = 0

def total_pairs : ℕ := 169

def valid_pairs : ℕ := 2

theorem probability_not_distinct_roots_greater_than_two :
  (total_pairs - valid_pairs) / total_pairs = 167 / 169 :=
sorry

end NUMINAMATH_CALUDE_probability_not_distinct_roots_greater_than_two_l3275_327527


namespace NUMINAMATH_CALUDE_raghu_investment_l3275_327547

/-- Proves that Raghu's investment is 2000 given the problem conditions --/
theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = raghu * 0.9 →
  vishal = trishul * 1.1 →
  raghu + trishul + vishal = 5780 →
  raghu = 2000 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l3275_327547


namespace NUMINAMATH_CALUDE_special_factors_count_l3275_327565

/-- A function that returns the number of positive factors of 60 that are multiples of 5 but not multiples of 3 -/
def count_special_factors : ℕ :=
  (Finset.filter (fun n => 60 % n = 0 ∧ n % 5 = 0 ∧ n % 3 ≠ 0) (Finset.range 61)).card

/-- Theorem stating that the number of positive factors of 60 that are multiples of 5 but not multiples of 3 is 2 -/
theorem special_factors_count : count_special_factors = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_factors_count_l3275_327565


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3275_327506

def U : Set ℕ := {2, 3, 5, 7, 8}
def A : Set ℕ := {2, 8}
def B : Set ℕ := {3, 5, 8}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3275_327506


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3275_327519

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) (hneq : x ≠ y) :
  1 / x + 1 / y > 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3275_327519


namespace NUMINAMATH_CALUDE_binomial_20_19_l3275_327537

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end NUMINAMATH_CALUDE_binomial_20_19_l3275_327537


namespace NUMINAMATH_CALUDE_scientific_notation_of_35_8_billion_l3275_327554

theorem scientific_notation_of_35_8_billion : 
  (35800000000 : ℝ) = 3.58 * (10 : ℝ)^10 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35_8_billion_l3275_327554


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l3275_327590

theorem geometric_sum_remainder (n : ℕ) :
  (7^(n+1) - 1) / 6 % 500 = 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_l3275_327590


namespace NUMINAMATH_CALUDE_magazines_per_box_l3275_327511

theorem magazines_per_box (total_magazines : ℕ) (num_boxes : ℕ) (h1 : total_magazines = 63) (h2 : num_boxes = 7) :
  total_magazines / num_boxes = 9 := by
  sorry

end NUMINAMATH_CALUDE_magazines_per_box_l3275_327511


namespace NUMINAMATH_CALUDE_ad_arrangement_count_l3275_327583

-- Define the number of original ads
def original_ads : Nat := 5

-- Define the number of ads to be kept
def kept_ads : Nat := 2

-- Define the number of new ads to be added
def new_ads : Nat := 1

-- Define the number of PSAs to be added
def psas : Nat := 2

-- Define the function to calculate the number of arrangements
def num_arrangements (n m : Nat) : Nat :=
  (n.choose m) * (m + 1) * 2

-- Theorem statement
theorem ad_arrangement_count :
  num_arrangements original_ads kept_ads = 120 :=
by sorry

end NUMINAMATH_CALUDE_ad_arrangement_count_l3275_327583


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l3275_327577

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) : 
  (2 : ℚ) / 3 * total = speedsters →
  (4 : ℚ) / 5 * speedsters = convertibles →
  total - speedsters = 40 →
  convertibles = 64 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l3275_327577


namespace NUMINAMATH_CALUDE_building_floors_l3275_327564

/-- The number of floors in a building given Earl's movements and position -/
theorem building_floors
  (P Q R S T X : ℕ)
  (h_x_lower : 1 < X)
  (h_x_upper : X < 50)
  : ∃ (F : ℕ), F = 1 + P - Q + R - S + T + X :=
by sorry

end NUMINAMATH_CALUDE_building_floors_l3275_327564


namespace NUMINAMATH_CALUDE_brick_breadth_is_10cm_l3275_327581

/-- Prove that the breadth of a brick is 10 cm given the specified conditions -/
theorem brick_breadth_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 20) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 16000) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width = (brick_length * brick_width) * total_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_breadth_is_10cm_l3275_327581


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3275_327523

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_shift_theorem :
  let initial_parabola : Parabola := { a := -1, h := 1, k := 2 }
  let shifted_parabola := shift_horizontal (shift_vertical initial_parabola 2) 1
  shifted_parabola = { a := -1, h := 0, k := 4 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3275_327523


namespace NUMINAMATH_CALUDE_seasonal_work_term_l3275_327546

/-- The established term of work for two seasonal workers -/
theorem seasonal_work_term (a r s : ℝ) (hr : r > 0) (hs : s > r) :
  ∃ x : ℝ, x > 0 ∧
  (x - a) * (s / (x + a)) = (x + a) * (r / (x - a)) ∧
  x = a * (s + r) / (s - r) := by
  sorry

end NUMINAMATH_CALUDE_seasonal_work_term_l3275_327546


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3275_327555

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C → 
  A > 0 → B > 0 → C > 0 → 
  A * B * C = 2310 → 
  A + B + C ≤ 52 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3275_327555


namespace NUMINAMATH_CALUDE_min_length_PQ_l3275_327539

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a b c : ℝ) : Prop := b = (a + c) / 2

-- Define the moving line l
def line_l (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define the line that Q lies on
def line_Q (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0

-- Define the minimum distance function
def min_distance (A P Q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem min_length_PQ (a b c : ℝ) :
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 →
  is_arithmetic_sequence a b c →
  ∃ (P Q : ℝ × ℝ),
    line_l a b c P.1 P.2 ∧
    line_Q Q.1 Q.2 ∧
    (∀ (P' Q' : ℝ × ℝ),
      line_l a b c P'.1 P'.2 →
      line_Q Q'.1 Q'.2 →
      min_distance point_A P Q ≤ min_distance point_A P' Q') →
    min_distance point_A P Q = 1 :=
sorry

end NUMINAMATH_CALUDE_min_length_PQ_l3275_327539


namespace NUMINAMATH_CALUDE_abs_negative_eight_l3275_327580

theorem abs_negative_eight : |(-8 : ℤ)| = 8 := by sorry

end NUMINAMATH_CALUDE_abs_negative_eight_l3275_327580


namespace NUMINAMATH_CALUDE_sequence_a_property_l3275_327534

def sequence_a (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def S (n : ℕ) : ℚ :=
  n^2 * sequence_a n

theorem sequence_a_property :
  ∀ n : ℕ, n ≥ 1 →
    (sequence_a 1 = 1) ∧
    (S n = n^2 * sequence_a n) ∧
    (sequence_a n = 1 / (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l3275_327534


namespace NUMINAMATH_CALUDE_problem_solution_l3275_327518

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem problem_solution (a : ℝ) :
  (f' a 3 = 0 → a = 3) ∧
  (∀ x < 0, Monotone (f a) ↔ a ≥ 0) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3275_327518


namespace NUMINAMATH_CALUDE_cos_pi_sixth_eq_sin_shifted_l3275_327544

theorem cos_pi_sixth_eq_sin_shifted (x : ℝ) : 
  Real.cos x + π/6 = Real.sin (x + 2*π/3) := by sorry

end NUMINAMATH_CALUDE_cos_pi_sixth_eq_sin_shifted_l3275_327544


namespace NUMINAMATH_CALUDE_factorize_x_squared_plus_2x_l3275_327536

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x_squared_plus_2x_l3275_327536


namespace NUMINAMATH_CALUDE_polygon_with_360_degree_sum_has_4_sides_l3275_327563

theorem polygon_with_360_degree_sum_has_4_sides :
  ∀ (n : ℕ), n ≥ 3 →
  (n - 2) * 180 = 360 →
  n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_360_degree_sum_has_4_sides_l3275_327563


namespace NUMINAMATH_CALUDE_monotone_increasing_inequalities_l3275_327551

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem monotone_increasing_inequalities 
  (h1 : ∀ x, f' x > 0) 
  (h2 : ∀ x, HasDerivAt f (f' x) x) 
  (x₁ x₂ : ℝ) 
  (h3 : x₁ ≠ x₂) : 
  (f x₁ - f x₂) * (x₁ - x₂) > 0 ∧ 
  (f x₁ - f x₂) * (x₂ - x₁) < 0 ∧ 
  (f x₂ - f x₁) * (x₂ - x₁) > 0 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_inequalities_l3275_327551


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3275_327508

/-- Given a square with perimeter 180 units divided into 3 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (num_rectangles : ℕ) : 
  square_perimeter = 180 →
  num_rectangles = 3 →
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / num_rectangles
  2 * (rect_length + rect_width) = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3275_327508


namespace NUMINAMATH_CALUDE_equation_unique_solution_l3275_327505

theorem equation_unique_solution :
  ∃! x : ℝ, (Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x) ∧ 
  (x = 400/49) := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l3275_327505


namespace NUMINAMATH_CALUDE_painted_cube_probability_l3275_327562

/-- Represents a 5x5x5 cube with three faces sharing a vertex painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  size_eq : size = 5
  faces_eq : painted_faces = 3

/-- The number of unit cubes with exactly three painted faces -/
def num_three_painted (cube : PaintedCube) : ℕ := 8

/-- The number of unit cubes with exactly one painted face -/
def num_one_painted (cube : PaintedCube) : ℕ := 3 * (cube.size - 2)^2

/-- The total number of unit cubes in the large cube -/
def total_cubes (cube : PaintedCube) : ℕ := cube.size^3

/-- The number of ways to choose two cubes from the total -/
def total_combinations (cube : PaintedCube) : ℕ := (total_cubes cube).choose 2

/-- The number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_outcomes (cube : PaintedCube) : ℕ := (num_three_painted cube) * (num_one_painted cube)

/-- The probability of selecting one cube with three painted faces and one with one painted face -/
def probability (cube : PaintedCube) : ℚ :=
  (favorable_outcomes cube : ℚ) / (total_combinations cube : ℚ)

theorem painted_cube_probability (cube : PaintedCube) :
  probability cube = 24 / 875 := by sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l3275_327562


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3275_327507

theorem sum_of_fractions (x y z : ℕ) : 
  (Nat.gcd x 9 = 1) → 
  (Nat.gcd y 15 = 1) → 
  (Nat.gcd z 14 = 1) → 
  (x * y * z : ℚ) / (9 * 15 * 14) = 1 / 6 → 
  x + y + z = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3275_327507


namespace NUMINAMATH_CALUDE_cubic_roots_l3275_327572

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 10

theorem cubic_roots :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 5) ∧
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_l3275_327572


namespace NUMINAMATH_CALUDE_max_sin_a_is_one_l3275_327526

theorem max_sin_a_is_one (a b : ℝ) (h : Real.sin (a + b) = Real.sin a + Real.sin b) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), Real.sin x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_sin_a_is_one_l3275_327526


namespace NUMINAMATH_CALUDE_infinite_triples_theorem_l3275_327513

def is_sum_of_two_squares (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem infinite_triples_theorem :
  (∃ f : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * 100^(f m)) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) - 1) ∧
    ¬is_sum_of_two_squares (2 * 100^(f m) + 1)) ∧
  (∃ g : ℕ → ℕ, ∀ m : ℕ,
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 1) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2) ∧
    is_sum_of_two_squares (2 * (g m^2 - g m)^2 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_theorem_l3275_327513


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3275_327522

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) :
  diagonal = 24 →
  offset1 = 9 →
  offset2 = 6 →
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3275_327522


namespace NUMINAMATH_CALUDE_diagonal_length_is_sqrt_457_l3275_327558

/-- An isosceles trapezoid with specific side lengths -/
structure IsoscelesTrapezoid :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 24)
  (bc_length : dist B C = 13)
  (cd_length : dist C D = 12)
  (da_length : dist D A = 13)
  (isosceles : dist B C = dist D A)

/-- The length of the diagonal AC in the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  dist t.A t.C

/-- Theorem stating that the diagonal length is √457 -/
theorem diagonal_length_is_sqrt_457 (t : IsoscelesTrapezoid) :
  diagonal_length t = Real.sqrt 457 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_length_is_sqrt_457_l3275_327558


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3275_327569

theorem rectangle_diagonal (side1 : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side1 = 6 → area = 48 → diagonal = 10 → 
  ∃ (side2 : ℝ), 
    side1 * side2 = area ∧ 
    diagonal^2 = side1^2 + side2^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3275_327569


namespace NUMINAMATH_CALUDE_theta_range_l3275_327543

theorem theta_range (θ : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l3275_327543


namespace NUMINAMATH_CALUDE_abs_inequality_range_l3275_327502

theorem abs_inequality_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 2| + |x + a|) →
  (∀ x : ℝ, f x ≥ 3) →
  (a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_range_l3275_327502


namespace NUMINAMATH_CALUDE_expression_equality_l3275_327548

theorem expression_equality : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 27 + (Real.pi + 1)^0 = 4 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3275_327548


namespace NUMINAMATH_CALUDE_line_inclination_45_deg_l3275_327510

/-- Given two points P(-2, m) and Q(m, 4) on a line with inclination angle 45°, prove m = 1 -/
theorem line_inclination_45_deg (m : ℝ) : 
  let P : ℝ × ℝ := (-2, m)
  let Q : ℝ × ℝ := (m, 4)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_45_deg_l3275_327510


namespace NUMINAMATH_CALUDE_unique_n_existence_l3275_327533

theorem unique_n_existence : ∃! n : ℤ,
  50 < n ∧ n < 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n % 4 = 1 ∧
  n = 105 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_existence_l3275_327533


namespace NUMINAMATH_CALUDE_frisbee_deck_difference_l3275_327500

/-- Represents the number of items Bella has -/
structure BellasItems where
  marbles : ℕ
  frisbees : ℕ
  deckCards : ℕ

/-- The conditions of the problem -/
def problemConditions (items : BellasItems) : Prop :=
  items.marbles = 2 * items.frisbees ∧
  items.marbles = 60 ∧
  (items.marbles + 2/5 * items.marbles + 
   items.frisbees + 2/5 * items.frisbees + 
   items.deckCards + 2/5 * items.deckCards) = 140

/-- The theorem to prove -/
theorem frisbee_deck_difference (items : BellasItems) 
  (h : problemConditions items) : 
  items.frisbees - items.deckCards = 20 := by
  sorry


end NUMINAMATH_CALUDE_frisbee_deck_difference_l3275_327500


namespace NUMINAMATH_CALUDE_pedestrian_speed_theorem_l3275_327529

/-- Given two pedestrians moving in the same direction, this theorem proves
    that the speed of the second pedestrian is either 6 m/s or 20/3 m/s,
    given the initial conditions. -/
theorem pedestrian_speed_theorem 
  (S₀ : ℝ) (v₁ : ℝ) (t : ℝ) (S : ℝ)
  (h₁ : S₀ = 200) 
  (h₂ : v₁ = 7)
  (h₃ : t = 5 * 60) -- 5 minutes in seconds
  (h₄ : S = 100) :
  ∃ v₂ : ℝ, (v₂ = 6 ∨ v₂ = 20/3) ∧ 
  (S₀ - S = (v₁ - v₂) * t) :=
by sorry

end NUMINAMATH_CALUDE_pedestrian_speed_theorem_l3275_327529


namespace NUMINAMATH_CALUDE_even_function_interval_sum_zero_l3275_327535

/-- A function f is even on an interval [a, b] if for all x in [a, b], f(x) = f(-x) -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x = f (-x)

/-- If f is an even function on the interval [a, b], then a + b = 0 -/
theorem even_function_interval_sum_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h : IsEvenOn f a b) : a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_interval_sum_zero_l3275_327535


namespace NUMINAMATH_CALUDE_existence_of_sequence_l3275_327509

theorem existence_of_sequence (p q : ℝ) (y : Fin 2017 → ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  ∃ x : Fin 2017 → ℝ, ∀ i : Fin 2017,
    p * max (x i) (x (i.succ)) + q * min (x i) (x (i.succ)) = y i :=
by sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l3275_327509


namespace NUMINAMATH_CALUDE_triangle_median_and_altitude_l3275_327570

structure Point where
  x : ℝ
  y : ℝ

def Triangle (A B C : Point) := True

def isMedian (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∃ D : Point, D.x = (B.x + C.x) / 2 ∧ D.y = (B.y + C.y) / 2 ∧ l D.x D.y ∧ l A.x A.y

def isAltitude (l : ℝ → ℝ → Prop) (A B C : Point) : Prop :=
  ∀ x y : ℝ, l x y → (x - A.x) * (C.x - A.x) + (y - A.y) * (C.y - A.y) = 0

theorem triangle_median_and_altitude 
  (A B C : Point)
  (h_triangle : Triangle A B C)
  (h_A : A.x = 1 ∧ A.y = 3)
  (h_B : B.x = 5 ∧ B.y = 1)
  (h_C : C.x = -1 ∧ C.y = -1) :
  (isMedian (fun x y => 3 * x + y - 6 = 0) A B C) ∧
  (isAltitude (fun x y => x + 2 * y - 7 = 0) B A C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_and_altitude_l3275_327570


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3275_327512

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  ∀ z w : ℝ, z > 0 ∧ w > 0 ∧ 2/z + 1/w = 1 → x + 2*y ≤ z + 2*w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2/a + 1/b = 1 ∧ a + 2*b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3275_327512


namespace NUMINAMATH_CALUDE_domain_of_f_x_squared_l3275_327592

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem domain_of_f_x_squared 
  (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ∈ Set.range f) : 
  (∀ x, f (x^2) ∈ Set.range f ↔ x ∈ Set.Icc (-2) 2) := by
  sorry

end NUMINAMATH_CALUDE_domain_of_f_x_squared_l3275_327592


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3275_327582

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex_x : q.f 2 = 5) 
  (point : q.f 1 = 6) : 
  q.a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3275_327582


namespace NUMINAMATH_CALUDE_section_B_avg_weight_l3275_327538

def section_A_students : ℕ := 50
def section_B_students : ℕ := 40
def total_students : ℕ := section_A_students + section_B_students
def section_A_avg_weight : ℝ := 50
def total_avg_weight : ℝ := 58.89

theorem section_B_avg_weight :
  let section_B_weight := total_students * total_avg_weight - section_A_students * section_A_avg_weight
  section_B_weight / section_B_students = 70.0025 := by
sorry

end NUMINAMATH_CALUDE_section_B_avg_weight_l3275_327538


namespace NUMINAMATH_CALUDE_triangle_area_specific_l3275_327514

/-- The area of a triangle with two sides of length 31 and one side of length 40 is 474 -/
theorem triangle_area_specific : ∃ (A : ℝ), 
  A = (Real.sqrt (51 * (51 - 31) * (51 - 31) * (51 - 40)) : ℝ) ∧ A = 474 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l3275_327514


namespace NUMINAMATH_CALUDE_total_pizzas_is_fifteen_l3275_327556

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

/-- Theorem: The total number of pizzas served today is 15 -/
theorem total_pizzas_is_fifteen : total_pizzas = 15 := by sorry

end NUMINAMATH_CALUDE_total_pizzas_is_fifteen_l3275_327556


namespace NUMINAMATH_CALUDE_jacket_pricing_l3275_327567

theorem jacket_pricing (x : ℝ) : 
  (0.8 * (1 + 0.5) * x = x + 28) ↔ 
  (∃ (markup : ℝ) (discount : ℝ) (profit : ℝ), 
    markup = 0.5 ∧ 
    discount = 0.2 ∧ 
    profit = 28 ∧ 
    (1 - discount) * (1 + markup) * x - x = profit) :=
by sorry

end NUMINAMATH_CALUDE_jacket_pricing_l3275_327567


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l3275_327591

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_about_x_axis (1, a) (b, 2) → a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l3275_327591


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l3275_327575

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_time > 0 →
  usual_rate > 0 →
  (7/6 * usual_rate) * (usual_time - 2) = usual_rate * usual_time →
  usual_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l3275_327575


namespace NUMINAMATH_CALUDE_multiple_properties_l3275_327549

theorem multiple_properties (a b c : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k)
  (hb : ∃ k : ℤ, b = 12 * k)
  (hc : ∃ k : ℤ, c = 9 * k) :
  (∃ k : ℤ, b = 3 * k) ∧
  (∃ k : ℤ, a - b = 3 * k) ∧
  (∃ k : ℤ, a - c = 3 * k) ∧
  (∃ k : ℤ, c - b = 3 * k) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l3275_327549


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3275_327566

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3275_327566


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3275_327579

theorem geometric_sequence_first_term :
  ∀ (a r : ℝ),
    a * r^2 = 720 →
    a * r^5 = 5040 →
    a = 720 / 7^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3275_327579


namespace NUMINAMATH_CALUDE_function_increment_proof_l3275_327501

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The initial x value -/
def x₀ : ℝ := 1

/-- The final x value -/
def x₁ : ℝ := 1.02

/-- The increment of x -/
def Δx : ℝ := x₁ - x₀

theorem function_increment_proof :
  f x₁ - f x₀ = 0.0808 :=
sorry

end NUMINAMATH_CALUDE_function_increment_proof_l3275_327501


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l3275_327520

/-- Represents the speed of a boat in km/hr -/
def boat_speed : ℝ := 6

/-- Represents the distance traveled against the stream in km -/
def distance_against : ℝ := 5

/-- Represents the time of travel in hours -/
def travel_time : ℝ := 1

/-- Calculates the speed of the stream based on the boat's speed and distance traveled against the stream -/
def stream_speed : ℝ := boat_speed - distance_against

/-- Calculates the effective speed of the boat along the stream -/
def effective_speed : ℝ := boat_speed + stream_speed

/-- Theorem: The boat travels 7 km along the stream in one hour -/
theorem boat_distance_along_stream :
  effective_speed * travel_time = 7 := by sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l3275_327520


namespace NUMINAMATH_CALUDE_john_jury_duty_days_l3275_327530

/-- The number of days John spends on jury duty -/
def jury_duty_days (jury_selection_days : ℕ) (trial_multiplier : ℕ) 
  (deliberation_days : ℕ) (deliberation_hours_per_day : ℕ) (hours_per_day : ℕ) : ℕ :=
  jury_selection_days + 
  (trial_multiplier * jury_selection_days) + 
  (deliberation_days * deliberation_hours_per_day) / hours_per_day

/-- Theorem stating that John spends 14 days on jury duty -/
theorem john_jury_duty_days : 
  jury_duty_days 2 4 6 16 24 = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_jury_duty_days_l3275_327530


namespace NUMINAMATH_CALUDE_phi_value_l3275_327516

/-- Given a function f and constants ω and φ, proves that φ = π/6 under certain conditions. -/
theorem phi_value (f g : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = 2 * Real.sin (ω * x + φ)) →
  ω > 0 →
  |φ| < π / 2 →
  (∀ x, f (x + π) = f x) →
  (∀ x, f (x - π / 6) = g x) →
  (∀ x, g (x + π / 3) = g (π / 3 - x)) →
  φ = π / 6 := by
sorry


end NUMINAMATH_CALUDE_phi_value_l3275_327516


namespace NUMINAMATH_CALUDE_barbara_colored_paper_bundles_l3275_327578

/-- Represents the number of sheets in different paper units -/
structure PaperUnits where
  sheets_per_bunch : ℕ
  sheets_per_bundle : ℕ
  sheets_per_heap : ℕ

/-- Represents the quantities of different types of paper -/
structure PaperQuantities where
  bunches_of_white : ℕ
  heaps_of_scrap : ℕ
  total_sheets_removed : ℕ

/-- Calculates the number of bundles of colored paper -/
def bundles_of_colored_paper (units : PaperUnits) (quantities : PaperQuantities) : ℕ :=
  let white_sheets := quantities.bunches_of_white * units.sheets_per_bunch
  let scrap_sheets := quantities.heaps_of_scrap * units.sheets_per_heap
  let colored_sheets := quantities.total_sheets_removed - (white_sheets + scrap_sheets)
  colored_sheets / units.sheets_per_bundle

/-- Theorem stating that Barbara found 3 bundles of colored paper -/
theorem barbara_colored_paper_bundles :
  let units := PaperUnits.mk 4 2 20
  let quantities := PaperQuantities.mk 2 5 114
  bundles_of_colored_paper units quantities = 3 := by
  sorry

end NUMINAMATH_CALUDE_barbara_colored_paper_bundles_l3275_327578


namespace NUMINAMATH_CALUDE_rows_containing_47_l3275_327532

-- Define Pascal's Triangle binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define a function to check if a number is in a row of Pascal's Triangle
def numberInRow (num row : ℕ) : Prop := sorry

-- Define primality
def isPrime (p : ℕ) : Prop := sorry

-- Theorem statement
theorem rows_containing_47 :
  isPrime 47 →
  (∃! row : ℕ, numberInRow 47 row) :=
by sorry

end NUMINAMATH_CALUDE_rows_containing_47_l3275_327532


namespace NUMINAMATH_CALUDE_defect_probability_is_22_900_l3275_327587

/-- Represents a machine in the production line -/
structure Machine where
  defectProb : ℝ
  productivityRatio : ℝ

/-- The production setup with three machines -/
def productionSetup : List Machine := [
  { defectProb := 0.02, productivityRatio := 3 },
  { defectProb := 0.03, productivityRatio := 1 },
  { defectProb := 0.04, productivityRatio := 0.5 }
]

/-- Calculates the probability of a randomly selected part being defective -/
def calculateDefectProbability (setup : List Machine) : ℝ :=
  sorry

/-- Theorem stating that the probability of a defective part is 22/900 -/
theorem defect_probability_is_22_900 :
  calculateDefectProbability productionSetup = 22 / 900 := by
  sorry

end NUMINAMATH_CALUDE_defect_probability_is_22_900_l3275_327587


namespace NUMINAMATH_CALUDE_expression_value_l3275_327584

theorem expression_value : 
  Real.sqrt ((16^12 + 8^15) / (16^5 + 8^16)) = (3 * Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3275_327584


namespace NUMINAMATH_CALUDE_coprime_polynomials_l3275_327531

theorem coprime_polynomials (n : ℕ) : 
  Nat.gcd (n^5 + 4*n^3 + 3*n) (n^4 + 3*n^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_polynomials_l3275_327531


namespace NUMINAMATH_CALUDE_average_first_six_l3275_327596

theorem average_first_six (total_count : Nat) (total_avg : ℝ) (last_six_avg : ℝ) (sixth_num : ℝ) :
  total_count = 11 →
  total_avg = 10.7 →
  last_six_avg = 11.4 →
  sixth_num = 13.700000000000017 →
  (6 * ((total_count : ℝ) * total_avg - 6 * last_six_avg + sixth_num)) / 6 = 10.5 := by
  sorry

#check average_first_six

end NUMINAMATH_CALUDE_average_first_six_l3275_327596


namespace NUMINAMATH_CALUDE_initial_interest_rate_l3275_327568

/-- Proves that the initial interest rate is 5% given the problem conditions --/
theorem initial_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_rate : ℝ) 
  (h1 : initial_investment = 8000)
  (h2 : additional_investment = 4000)
  (h3 : additional_rate = 8)
  (h4 : total_rate = 6)
  : (initial_investment * (100 * total_rate - additional_investment * additional_rate) / 
    (100 * (initial_investment + additional_investment))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_interest_rate_l3275_327568


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l3275_327545

theorem road_trip_time_calculation (dist_wa_id : ℝ) (dist_id_nv : ℝ) (speed_wa_id : ℝ) (speed_id_nv : ℝ)
  (h1 : dist_wa_id = 640)
  (h2 : dist_id_nv = 550)
  (h3 : speed_wa_id = 80)
  (h4 : speed_id_nv = 50)
  (h5 : speed_wa_id > 0)
  (h6 : speed_id_nv > 0) :
  dist_wa_id / speed_wa_id + dist_id_nv / speed_id_nv = 19 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_time_calculation_l3275_327545


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3275_327542

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x - k ≠ 0) → k < -9/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3275_327542


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3275_327553

theorem max_value_sqrt_sum (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_eq_two : x + y + z = 2) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ 
    Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) > Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)) 
  ∨ 
  Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3275_327553


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3275_327589

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 - i
  (z₁ / z₂).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l3275_327589


namespace NUMINAMATH_CALUDE_davids_math_marks_l3275_327588

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 75)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l3275_327588


namespace NUMINAMATH_CALUDE_apple_cost_proof_l3275_327594

/-- The cost of apples for the first 30 kgs (in rupees per kg) -/
def l : ℝ := sorry

/-- The cost of apples for each additional kg beyond 30 kgs (in rupees per kg) -/
def q : ℝ := sorry

/-- The total cost of 33 kgs of apples (in rupees) -/
def cost_33 : ℝ := 11.67

/-- The total cost of 36 kgs of apples (in rupees) -/
def cost_36 : ℝ := 12.48

/-- The cost of the first 10 kgs of apples (in rupees) -/
def cost_10 : ℝ := 10 * l

theorem apple_cost_proof :
  (30 * l + 3 * q = cost_33) ∧
  (30 * l + 6 * q = cost_36) →
  cost_10 = 3.62 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_proof_l3275_327594


namespace NUMINAMATH_CALUDE_unique_score_theorem_l3275_327586

/-- Represents a score in the mathematics competition. -/
structure Score where
  value : ℕ
  correct : ℕ
  wrong : ℕ
  total_questions : ℕ
  h1 : value = 5 * correct - 2 * wrong
  h2 : correct + wrong ≤ total_questions

/-- The unique score over 70 that allows determination of correct answers. -/
def unique_determinable_score : ℕ := 71

theorem unique_score_theorem (s : Score) (h_total : s.total_questions = 25) 
    (h_over_70 : s.value > 70) : 
  (∃! c w, s.correct = c ∧ s.wrong = w) ↔ s.value = unique_determinable_score :=
sorry

end NUMINAMATH_CALUDE_unique_score_theorem_l3275_327586


namespace NUMINAMATH_CALUDE_vector_operation_l3275_327598

/-- Given two vectors a and b in ℝ², prove that 3b - a equals the specified result. -/
theorem vector_operation (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
  (3 : ℝ) • b - a = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3275_327598


namespace NUMINAMATH_CALUDE_jack_euros_l3275_327541

/-- Calculates the number of euros Jack has given his dollar amount, 
    the exchange rate, and his total amount in dollars. -/
def calculate_euros (dollars : ℕ) (exchange_rate : ℕ) (total : ℕ) : ℕ :=
  (total - dollars) / exchange_rate

/-- Proves that Jack has 36 euros given the problem conditions. -/
theorem jack_euros : calculate_euros 45 2 117 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jack_euros_l3275_327541


namespace NUMINAMATH_CALUDE_complex_square_at_one_one_l3275_327573

theorem complex_square_at_one_one : 
  ∀ z : ℂ, (z.re = 1 ∧ z.im = 1) → z^2 = 2*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_square_at_one_one_l3275_327573


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3275_327552

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 8 < 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3275_327552


namespace NUMINAMATH_CALUDE_total_cans_collected_l3275_327515

/-- The number of cans collected by LaDonna -/
def ladonna_cans : ℕ := 25

/-- The number of cans collected by Prikya -/
def prikya_cans : ℕ := 2 * ladonna_cans

/-- The number of cans collected by Yoki -/
def yoki_cans : ℕ := 10

/-- The total number of cans collected -/
def total_cans : ℕ := ladonna_cans + prikya_cans + yoki_cans

theorem total_cans_collected : total_cans = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_collected_l3275_327515


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3275_327585

def point_A : ℝ × ℝ := (-1, -5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  let vector_AB : ℝ × ℝ := (3 * vector_a.1, 3 * vector_a.2)
  let point_B : ℝ × ℝ := (point_A.1 + vector_AB.1, point_A.2 + vector_AB.2)
  point_B = (5, 4) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l3275_327585


namespace NUMINAMATH_CALUDE_group_size_calculation_l3275_327503

theorem group_size_calculation (T : ℕ) (L : ℕ) : 
  T = L + 90 → -- Total is sum of young and old
  (L : ℚ) / T = 1/4 → -- Probability of selecting young person
  T = 120 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l3275_327503


namespace NUMINAMATH_CALUDE_average_sale_per_month_l3275_327550

def sales : List ℕ := [120, 80, 100, 140, 160]

theorem average_sale_per_month : 
  (List.sum sales) / (List.length sales) = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_per_month_l3275_327550


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l3275_327576

theorem y_in_terms_of_x (m : ℕ) (x y : ℝ) 
  (hx : x = 2^m + 1) 
  (hy : y = 3 + 2^(m+1)) : 
  y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l3275_327576


namespace NUMINAMATH_CALUDE_eggs_in_park_l3275_327561

theorem eggs_in_park (total_eggs club_house_eggs town_hall_eggs : ℕ) 
  (h1 : total_eggs = 20)
  (h2 : club_house_eggs = 12)
  (h3 : town_hall_eggs = 3) :
  total_eggs - club_house_eggs - town_hall_eggs = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_eggs_in_park_l3275_327561


namespace NUMINAMATH_CALUDE_triangle_properties_l3275_327504

-- Define a triangle with given properties
structure Triangle where
  a : ℝ  -- side BC
  m : ℝ  -- altitude from B to AC
  k : ℝ  -- median to side AC
  a_pos : 0 < a
  m_pos : 0 < m
  k_pos : 0 < k

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  let b := 2 * Real.sqrt (t.k^2 + t.a * (t.a - Real.sqrt (4 * t.k^2 - t.m^2)))
  let c := 2 * Real.sqrt (t.k^2 + (t.a/2) * ((t.a/2) - Real.sqrt (4 * t.k^2 - t.m^2)))
  (∃ (γ β : ℝ),
    b > 0 ∧
    c > 0 ∧
    Real.sin γ = t.m / b ∧
    Real.sin β = t.m / c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3275_327504


namespace NUMINAMATH_CALUDE_theater_seats_l3275_327525

theorem theater_seats (adult_price child_price total_income num_children : ℚ)
  (h1 : adult_price = 3)
  (h2 : child_price = 3/2)
  (h3 : total_income = 510)
  (h4 : num_children = 60)
  (h5 : ∃ num_adults : ℚ, num_adults * adult_price + num_children * child_price = total_income) :
  ∃ total_seats : ℚ, total_seats = num_children + (total_income - num_children * child_price) / adult_price ∧ total_seats = 200 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l3275_327525


namespace NUMINAMATH_CALUDE_shower_tiles_count_l3275_327571

/-- Calculates the total number of tiles in a 3-sided shower --/
def shower_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles_count : shower_tiles 3 8 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_count_l3275_327571


namespace NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l3275_327528

/-- Given a hyperbola (x²/16) - (y²/9) = 1 and a line y = kx - 1 parallel to one of its asymptotes,
    where k > 0, prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote
  (k : ℝ)
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16) - (y^2 / 9) = 1)
  (h3 : ∃ (m : ℝ), (∀ (x y : ℝ), y = m * x → (x^2 / 16) - (y^2 / 9) = 1) ∧
                   (∃ (b : ℝ), ∀ (x : ℝ), k * x - 1 = m * x + b)) :
  k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l3275_327528


namespace NUMINAMATH_CALUDE_max_value_of_function_l3275_327557

theorem max_value_of_function : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) - Real.cos (2 * x + π / 6)
  ∃ (M : ℝ), M = Real.sqrt 3 ∧ ∀ x, f x ≤ M ∧ ∃ x₀, f x₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3275_327557


namespace NUMINAMATH_CALUDE_exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l3275_327540

/-- Represents the outcome of tossing a coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The sample space of all possible outcomes when tossing two coins -/
def sampleSpace : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads),
   (CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads),
   (CoinOutcome.Tails, CoinOutcome.Tails)}

/-- The event "Exactly one head is up" -/
def exactlyOneHead : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Tails),
   (CoinOutcome.Tails, CoinOutcome.Heads)}

/-- The event "Exactly two heads are up" -/
def exactlyTwoHeads : Set TwoCoinsOutcome :=
  {(CoinOutcome.Heads, CoinOutcome.Heads)}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set TwoCoinsOutcome) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (A B : Set TwoCoinsOutcome) : Prop :=
  A ∪ B = sampleSpace

theorem exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary :
  mutuallyExclusive exactlyOneHead exactlyTwoHeads ∧
  ¬complementary exactlyOneHead exactlyTwoHeads :=
by sorry

end NUMINAMATH_CALUDE_exactlyOneHead_exactlyTwoHeads_mutuallyExclusive_not_complementary_l3275_327540


namespace NUMINAMATH_CALUDE_part_one_part_two_l3275_327593

-- Part 1
theorem part_one (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

-- Part 2
theorem part_two (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 3 / 16) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 3 / 64 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3275_327593


namespace NUMINAMATH_CALUDE_subset_condition_main_result_l3275_327574

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

def solution_set : Set ℝ := {0, 1/3, 1/5}

theorem main_result : {a : ℝ | B a ⊆ A} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_main_result_l3275_327574


namespace NUMINAMATH_CALUDE_fraction_pair_sum_equality_l3275_327559

theorem fraction_pair_sum_equality (n : ℕ) (h : n > 2009) :
  ∃ (a b c d : ℕ), a ≤ n ∧ b ≤ n ∧ c ≤ n ∧ d ≤ n ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (1 : ℚ) / (n + 1 - a) + (1 : ℚ) / (n + 1 - b) =
  (1 : ℚ) / (n + 1 - c) + (1 : ℚ) / (n + 1 - d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_pair_sum_equality_l3275_327559


namespace NUMINAMATH_CALUDE_riverton_soccer_team_l3275_327597

theorem riverton_soccer_team (total_players : ℕ) (math_players : ℕ) (both_players : ℕ) :
  total_players = 15 →
  math_players = 9 →
  both_players = 3 →
  math_players + (total_players - math_players) ≥ total_players →
  total_players - math_players + both_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_riverton_soccer_team_l3275_327597


namespace NUMINAMATH_CALUDE_chord_length_theorem_l3275_327517

/-- The chord length theorem -/
theorem chord_length_theorem (m : ℝ) : 
  m > 0 → 
  (∃ (x y : ℝ), x - y + m = 0 ∧ (x - 1)^2 + (y - 1)^2 = 3) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ - y₁ + m = 0 ∧ (x₁ - 1)^2 + (y₁ - 1)^2 = 3 ∧
    x₂ - y₂ + m = 0 ∧ (x₂ - 1)^2 + (y₂ - 1)^2 = 3 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = m^2) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l3275_327517
