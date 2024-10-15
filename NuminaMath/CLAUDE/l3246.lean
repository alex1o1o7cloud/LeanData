import Mathlib

namespace NUMINAMATH_CALUDE_number_of_divisor_pairs_l3246_324606

theorem number_of_divisor_pairs : ∃ (count : ℕ), count = 480 ∧
  count = (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 * p.2 ∣ (2008 * 2009 * 2010)) ∧ 
    p.1 > 0 ∧ p.2 > 0
  ) (Finset.product (Finset.range (2008 * 2009 * 2010 + 1)) (Finset.range (2008 * 2009 * 2010 + 1)))).card :=
by
  sorry

#check number_of_divisor_pairs

end NUMINAMATH_CALUDE_number_of_divisor_pairs_l3246_324606


namespace NUMINAMATH_CALUDE_additional_plates_count_l3246_324624

/-- The number of choices for the first letter in the original configuration -/
def original_first : Nat := 5

/-- The number of choices for the second letter in the original configuration -/
def original_second : Nat := 3

/-- The number of choices for the third letter in both original and new configurations -/
def third : Nat := 4

/-- The number of choices for the first letter in the new configuration -/
def new_first : Nat := 6

/-- The number of choices for the second letter in the new configuration -/
def new_second : Nat := 4

/-- The number of additional license plates that can be made -/
def additional_plates : Nat := new_first * new_second * third - original_first * original_second * third

theorem additional_plates_count : additional_plates = 36 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_count_l3246_324624


namespace NUMINAMATH_CALUDE_circle_equation_l3246_324686

theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-2, 3)
  let is_tangent_to_x_axis : Prop := ∃ (x_0 : ℝ), (x_0 + 2)^2 + 3^2 = (x + 2)^2 + (y - 3)^2
  is_tangent_to_x_axis →
  (x + 2)^2 + (y - 3)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3246_324686


namespace NUMINAMATH_CALUDE_stationery_sales_distribution_l3246_324690

theorem stationery_sales_distribution (pen_sales pencil_sales eraser_sales : ℝ) 
  (h_pen : pen_sales = 42)
  (h_pencil : pencil_sales = 25)
  (h_eraser : eraser_sales = 12)
  (h_total : pen_sales + pencil_sales + eraser_sales + (100 - pen_sales - pencil_sales - eraser_sales) = 100) :
  100 - pen_sales - pencil_sales - eraser_sales = 21 := by
sorry

end NUMINAMATH_CALUDE_stationery_sales_distribution_l3246_324690


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l3246_324685

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 100000 →
  (9 * (n - 3)^5 - 2 * n^3 + 17 * n - 33) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem ninety_nine_thousand_nine_hundred_ninety_nine_is_largest :
  (9 * (99999 - 3)^5 - 2 * 99999^3 + 17 * 99999 - 33) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 → (9 * (m - 3)^5 - 2 * m^3 + 17 * m - 33) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l3246_324685


namespace NUMINAMATH_CALUDE_lowest_possible_score_l3246_324673

-- Define the parameters of the problem
def mean : ℝ := 60
def std_dev : ℝ := 10
def z_score_95_percentile : ℝ := 1.645

-- Define the function to calculate the score from z-score
def score_from_z (z : ℝ) : ℝ := z * std_dev + mean

-- Define the conditions
def within_top_5_percent (score : ℝ) : Prop := score ≥ score_from_z z_score_95_percentile
def within_2_std_dev (score : ℝ) : Prop := score ≤ mean + 2 * std_dev

-- The theorem to prove
theorem lowest_possible_score :
  ∃ (score : ℕ), 
    (score : ℝ) = ⌈score_from_z z_score_95_percentile⌉ ∧
    within_top_5_percent score ∧
    within_2_std_dev score ∧
    ∀ (s : ℕ), s < score → ¬(within_top_5_percent s ∧ within_2_std_dev s) :=
by sorry

end NUMINAMATH_CALUDE_lowest_possible_score_l3246_324673


namespace NUMINAMATH_CALUDE_amber_work_hours_l3246_324697

theorem amber_work_hours :
  ∀ (amber armand ella : ℝ),
  armand = amber / 3 →
  ella = 2 * amber →
  amber + armand + ella = 40 →
  amber = 12 := by
sorry

end NUMINAMATH_CALUDE_amber_work_hours_l3246_324697


namespace NUMINAMATH_CALUDE_pentagon_area_l3246_324679

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a pentagon -/
structure Pentagon :=
  (F G H I J : Point)

/-- Calculates the area of a pentagon -/
def area (p : Pentagon) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Theorem: The area of the specific pentagon FGHIJ is 71√3/4 -/
theorem pentagon_area (p : Pentagon) :
  angle p.F p.G p.H = 120 * π / 180 →
  angle p.J p.F p.G = 120 * π / 180 →
  distance p.J p.F = 3 →
  distance p.F p.G = 3 →
  distance p.G p.H = 3 →
  distance p.H p.I = 5 →
  distance p.I p.J = 5 →
  area p = 71 * Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_l3246_324679


namespace NUMINAMATH_CALUDE_division_in_base5_l3246_324654

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Performs division in base 5 -/
def divBase5 (a b : ℕ) : ℕ := 
  base10ToBase5 (base5ToBase10 a / base5ToBase10 b)

theorem division_in_base5 : divBase5 1302 23 = 30 := by sorry

end NUMINAMATH_CALUDE_division_in_base5_l3246_324654


namespace NUMINAMATH_CALUDE_rays_remaining_nickels_l3246_324651

/-- Calculates the number of nickels Ray has left after giving money to Peter and Randi -/
theorem rays_remaining_nickels 
  (initial_cents : ℕ) 
  (cents_to_peter : ℕ) 
  (nickel_value : ℕ) 
  (h1 : initial_cents = 95)
  (h2 : cents_to_peter = 25)
  (h3 : nickel_value = 5) :
  (initial_cents - cents_to_peter - 2 * cents_to_peter) / nickel_value = 4 :=
by sorry

end NUMINAMATH_CALUDE_rays_remaining_nickels_l3246_324651


namespace NUMINAMATH_CALUDE_equation_solution_l3246_324677

theorem equation_solution (n : ℝ) : 
  let m := 5 * n + 5
  2 / (n + 2) + 3 / (n + 2) + m / (n + 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3246_324677


namespace NUMINAMATH_CALUDE_at_least_one_vertex_inside_or_on_boundary_l3246_324698

structure CentrallySymmetricPolygon where
  vertices : Set (ℝ × ℝ)
  is_centrally_symmetric : ∃ (center : ℝ × ℝ), ∀ v ∈ vertices, 
    ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

structure Polygon where
  vertices : Set (ℝ × ℝ)

def contained_in (T : Polygon) (M : CentrallySymmetricPolygon) : Prop :=
  ∀ v ∈ T.vertices, v ∈ M.vertices

def symmetric_image (T : Polygon) (P : ℝ × ℝ) : Polygon :=
  { vertices := {v' | ∃ v ∈ T.vertices, v' = (2 * P.1 - v.1, 2 * P.2 - v.2)} }

def vertex_inside_or_on_boundary (v : ℝ × ℝ) (M : CentrallySymmetricPolygon) : Prop :=
  v ∈ M.vertices

theorem at_least_one_vertex_inside_or_on_boundary 
  (M : CentrallySymmetricPolygon) (T : Polygon) (P : ℝ × ℝ) :
  contained_in T M →
  P ∈ {p | ∃ v ∈ T.vertices, p = v} →
  ∃ v ∈ (symmetric_image T P).vertices, vertex_inside_or_on_boundary v M :=
sorry

end NUMINAMATH_CALUDE_at_least_one_vertex_inside_or_on_boundary_l3246_324698


namespace NUMINAMATH_CALUDE_square_sum_equals_negative_45_l3246_324660

theorem square_sum_equals_negative_45 (x y : ℝ) 
  (h1 : x - 3 * y = 3) 
  (h2 : x * y = -9) : 
  x^2 + 9 * y^2 = -45 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_negative_45_l3246_324660


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_value_l3246_324622

/-- Given functions f and g, prove that if f(x) ≥ g(x) holds exactly for x ∈ [-1, 2], then m = 2 -/
theorem function_inequality_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + m ≥ 2*x^2 - 4*x) ↔ (-1 ≤ x ∧ x ≤ 2)) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_value_l3246_324622


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l3246_324691

theorem triangle_isosceles_or_right 
  (A B C : ℝ) 
  (triangle_angles : A + B + C = π) 
  (angle_condition : Real.sin (A + B - C) = Real.sin (A - B + C)) : 
  (B = C) ∨ (B + C = π / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l3246_324691


namespace NUMINAMATH_CALUDE_incenter_orthocenter_collinearity_l3246_324683

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the necessary concepts
def isIncenter (I : Point) (t : Triangle) : Prop := sorry
def isOrthocenter (H : Point) (t : Triangle) : Prop := sorry
def isMidpoint (M : Point) (A B : Point) : Prop := sorry
def liesOn (P : Point) (A B : Point) : Prop := sorry
def intersectsAt (A B C D : Point) (K : Point) : Prop := sorry
def isCircumcenter (O : Point) (t : Triangle) : Prop := sorry
def areCollinear (A B C : Point) : Prop := sorry
def areaTriangle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem incenter_orthocenter_collinearity 
  (t : Triangle) (I H B₁ C₁ B₂ C₂ K A₁ : Point) : 
  isIncenter I t → 
  isOrthocenter H t → 
  isMidpoint B₁ t.A t.C → 
  isMidpoint C₁ t.A t.B → 
  liesOn B₂ t.A t.B → 
  liesOn B₂ B₁ I → 
  B₂ ≠ t.B → 
  liesOn C₂ t.A C₁ → 
  liesOn C₂ C₁ I → 
  intersectsAt B₂ C₂ t.B t.C K → 
  isCircumcenter A₁ ⟨t.B, H, t.C⟩ → 
  (areCollinear t.A I A₁ ↔ areaTriangle t.B K B₂ = areaTriangle t.C K C₂) :=
sorry

end NUMINAMATH_CALUDE_incenter_orthocenter_collinearity_l3246_324683


namespace NUMINAMATH_CALUDE_inclination_angle_of_negative_unit_slope_l3246_324687

theorem inclination_angle_of_negative_unit_slope (α : Real) : 
  (Real.tan α = -1) → (0 ≤ α) → (α < Real.pi) → (α = 3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_inclination_angle_of_negative_unit_slope_l3246_324687


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3246_324607

theorem other_root_of_quadratic (a : ℝ) : 
  (3 : ℝ) ^ 2 - a * 3 - 2 * a = 0 → 
  ((-6 / 5 : ℝ) ^ 2 - a * (-6 / 5) - 2 * a = 0) ∧ 
  (3 + (-6 / 5) : ℝ) = a ∧ 
  (3 * (-6 / 5) : ℝ) = -2 * a := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3246_324607


namespace NUMINAMATH_CALUDE_angle_equality_in_triangle_l3246_324682

/-- Given an acute triangle ABC with its circumcircle, tangents at A and B intersecting at D,
    and M as the midpoint of AB, prove that ∠ACM = ∠BCD. -/
theorem angle_equality_in_triangle (A B C D M : ℂ) : 
  -- A, B, C are on the unit circle (representing the circumcircle)
  Complex.abs A = 1 ∧ Complex.abs B = 1 ∧ Complex.abs C = 1 →
  -- Triangle ABC is acute
  (0 < Real.cos (Complex.arg (B - A) - Complex.arg (C - A))) ∧
  (0 < Real.cos (Complex.arg (C - B) - Complex.arg (A - B))) ∧
  (0 < Real.cos (Complex.arg (A - C) - Complex.arg (B - C))) →
  -- D is the intersection of tangents at A and B
  D = (2 * A * B) / (A + B) →
  -- M is the midpoint of AB
  M = (A + B) / 2 →
  -- Conclusion: ∠ACM = ∠BCD
  Complex.arg ((M - C) / (A - C)) = Complex.arg ((B - C) / (D - C)) := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_in_triangle_l3246_324682


namespace NUMINAMATH_CALUDE_complex_expression_equals_zero_l3246_324652

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The theorem stating that the complex expression equals 0 -/
theorem complex_expression_equals_zero : (1 + i) / (1 - i) + i ^ 3 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_zero_l3246_324652


namespace NUMINAMATH_CALUDE_factorization_identities_l3246_324655

theorem factorization_identities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^4 - b^4 = (a - b) * (a + b) * (a^2 + b^2)) ∧
  (a + b - 2 * Real.sqrt (a * b) = (Real.sqrt a - Real.sqrt b)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l3246_324655


namespace NUMINAMATH_CALUDE_line_equation_proof_l3246_324629

theorem line_equation_proof (m b k : ℝ) : 
  (∃ k, (k^2 + 4*k + 3 - (m*k + b) = 3 ∨ k^2 + 4*k + 3 - (m*k + b) = -3) ∧ 
        (∀ k', k' ≠ k → ¬(k'^2 + 4*k' + 3 - (m*k' + b) = 3 ∨ k'^2 + 4*k' + 3 - (m*k' + b) = -3))) →
  (m * 2 + b = 5) →
  (b ≠ 0) →
  (m = 9/2 ∧ b = -4) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3246_324629


namespace NUMINAMATH_CALUDE_smallest_n_correct_l3246_324627

/-- The function f as defined in the problem -/
def f : ℕ → ℤ
| 0 => 0
| n + 1 => -f (n / 3) - 3 * ((n + 1) % 3)

/-- The smallest non-negative integer n such that f(n) = 2010 -/
def smallest_n : ℕ := 3 * (3^2010 - 1) / 4

/-- Theorem stating that f(smallest_n) = 2010 and smallest_n is indeed the smallest such n -/
theorem smallest_n_correct :
  f smallest_n = 2010 ∧ ∀ m : ℕ, m < smallest_n → f m ≠ 2010 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_correct_l3246_324627


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l3246_324661

theorem complex_magnitude_equation (t : ℝ) : 
  t > 0 ∧ Complex.abs (-7 + t * Complex.I) = 15 → t = 4 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l3246_324661


namespace NUMINAMATH_CALUDE_trick_or_treat_duration_l3246_324659

/-- The number of hours Tim and his children were out trick or treating -/
def trick_or_treat_hours (num_children : ℕ) (houses_per_hour : ℕ) (treats_per_child_per_house : ℕ) (total_treats : ℕ) : ℕ :=
  total_treats / (num_children * houses_per_hour * treats_per_child_per_house)

/-- Theorem stating that Tim and his children were out for 4 hours -/
theorem trick_or_treat_duration :
  trick_or_treat_hours 3 5 3 180 = 4 := by
  sorry

#eval trick_or_treat_hours 3 5 3 180

end NUMINAMATH_CALUDE_trick_or_treat_duration_l3246_324659


namespace NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3246_324626

theorem x_equation_implies_polynomial_value :
  ∀ x : ℝ, x + 1/x = 2 → x^9 - 5*x^5 + x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_implies_polynomial_value_l3246_324626


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3246_324680

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 2) :
  ∃ (min_value : ℝ), 
    (∀ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ → a₁ + 2 * a₃ ≥ min_value) ∧
    (∃ a₁ a₃, a 1 = a₁ ∧ a 3 = a₃ ∧ a₁ + 2 * a₃ = min_value) ∧
    min_value = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3246_324680


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l3246_324669

theorem power_tower_mod_500 : 7^(7^(7^7)) % 500 = 343 := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l3246_324669


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3246_324601

/-- Given a line L1 defined by 3x - 2y = 6, prove that a line L2 perpendicular to L1
    with y-intercept 2 has x-intercept 3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x - 2 * y = 6) →
  (∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ L2 ↔ y = m * x + 2) →
  (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L2 → 
    (x2 - x1) * (3 * (y2 - y1) + 2 * (x2 - x1)) = 0) →
  (3, 0) ∈ L2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3246_324601


namespace NUMINAMATH_CALUDE_travel_speed_problem_l3246_324670

/-- Proves that given the conditions of the problem, the speeds of person A and person B are 4.5 km/h and 6 km/h respectively. -/
theorem travel_speed_problem (distance_A distance_B : ℝ) (speed_ratio : ℚ) (time_difference : ℝ) :
  distance_A = 6 →
  distance_B = 10 →
  speed_ratio = 3/4 →
  time_difference = 1/3 →
  ∃ (speed_A speed_B : ℝ),
    speed_A = 4.5 ∧
    speed_B = 6 ∧
    speed_A / speed_B = speed_ratio ∧
    distance_B / speed_B - distance_A / speed_A = time_difference :=
by sorry

end NUMINAMATH_CALUDE_travel_speed_problem_l3246_324670


namespace NUMINAMATH_CALUDE_paul_collected_24_l3246_324656

/-- Represents the number of seashells collected by each person -/
structure Seashells where
  henry : ℕ
  paul : ℕ
  leo : ℕ

/-- The initial state of seashell collection -/
def initial_collection : Seashells → Prop
  | s => s.henry = 11 ∧ s.henry + s.paul + s.leo = 59

/-- The state after Leo gave away a quarter of his seashells -/
def after_leo_gives : Seashells → Prop
  | s => s.henry + s.paul + (s.leo - s.leo / 4) = 53

/-- Theorem stating that Paul collected 24 seashells -/
theorem paul_collected_24 (s : Seashells) :
  initial_collection s → after_leo_gives s → s.paul = 24 := by
  sorry

#check paul_collected_24

end NUMINAMATH_CALUDE_paul_collected_24_l3246_324656


namespace NUMINAMATH_CALUDE_stone_distance_l3246_324621

theorem stone_distance (n : ℕ) (total_distance : ℝ) : 
  n = 31 → 
  n % 2 = 1 → 
  total_distance = 4.8 → 
  (2 * (n / 2) * (n / 2 + 1) / 2) * (total_distance / (2 * (n / 2) * (n / 2 + 1) / 2)) = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_stone_distance_l3246_324621


namespace NUMINAMATH_CALUDE_correct_grade12_sample_l3246_324643

/-- Calculates the number of students to be drawn from grade 12 in a stratified sample -/
def students_from_grade12 (total_students : ℕ) (grade10_students : ℕ) (grade11_students : ℕ) (sample_size : ℕ) : ℕ :=
  let grade12_students := total_students - (grade10_students + grade11_students)
  (grade12_students * sample_size) / total_students

/-- Theorem stating the correct number of students to be drawn from grade 12 -/
theorem correct_grade12_sample : 
  students_from_grade12 2400 820 780 120 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_grade12_sample_l3246_324643


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l3246_324696

structure GeometricSpace where
  Line : Type
  Plane : Type
  subset : Line → Plane → Prop
  parallel : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop

variable (S : GeometricSpace)

theorem parallel_planes_condition
  (a b : S.Line) (α β : S.Plane)
  (h1 : S.subset a α)
  (h2 : S.subset b β) :
  (∃ (α' β' : S.Plane), S.plane_parallel α' β' →
    (S.parallel a β' ∧ S.parallel b α')) ∧
  ¬(∀ (α' β' : S.Plane), S.parallel a β' ∧ S.parallel b α' →
    S.plane_parallel α' β') := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l3246_324696


namespace NUMINAMATH_CALUDE_equation_solution_l3246_324637

theorem equation_solution :
  ∃ (x y z u : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧
    -1/x + 1/y + 1/z + 1/u = 2 ∧
    x = 1 ∧ y = 2 ∧ z = 3 ∧ u = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3246_324637


namespace NUMINAMATH_CALUDE_expenditure_increase_percentage_l3246_324650

theorem expenditure_increase_percentage
  (initial_expenditure : ℝ)
  (initial_savings : ℝ)
  (initial_income : ℝ)
  (h_ratio : initial_expenditure / initial_savings = 3 / 2)
  (h_income : initial_expenditure + initial_savings = initial_income)
  (h_new_income : ℝ)
  (h_income_increase : h_new_income = initial_income * 1.15)
  (h_new_savings : ℝ)
  (h_savings_increase : h_new_savings = initial_savings * 1.06)
  (h_new_expenditure : ℝ)
  (h_new_balance : h_new_expenditure + h_new_savings = h_new_income) :
  (h_new_expenditure - initial_expenditure) / initial_expenditure = 0.21 :=
sorry

end NUMINAMATH_CALUDE_expenditure_increase_percentage_l3246_324650


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3246_324653

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 2x² - 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

/-- Theorem: f(x) = 2x² - 2x + 1 is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3246_324653


namespace NUMINAMATH_CALUDE_ball_bounce_height_l3246_324623

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h_target : ℝ) (k : ℕ) 
  (h_initial : h₀ = 800)
  (h_rebound : r = 1 / 2)
  (h_target_def : h_target = 2) :
  (∀ n : ℕ, n < k → h₀ * r ^ n ≥ h_target) ∧
  (h₀ * r ^ k < h_target) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l3246_324623


namespace NUMINAMATH_CALUDE_smallest_n_for_2015_divisibility_l3246_324609

theorem smallest_n_for_2015_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (2^k - 1) % 2015 = 0 → k ≥ n) ∧ 
  (2^n - 1) % 2015 = 0 ∧ n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2015_divisibility_l3246_324609


namespace NUMINAMATH_CALUDE_vector_sum_proof_l3246_324649

def a : ℝ × ℝ := (2, 8)
def b : ℝ × ℝ := (-7, 2)

theorem vector_sum_proof : a + 2 • b = (-12, 12) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l3246_324649


namespace NUMINAMATH_CALUDE_train_passing_time_l3246_324631

/-- The time it takes for a train to pass a person moving in the opposite direction --/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 110 →
  train_speed = 84 * (5 / 18) →
  person_speed = 6 * (5 / 18) →
  (train_length / (train_speed + person_speed)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l3246_324631


namespace NUMINAMATH_CALUDE_solve_equation_l3246_324665

-- Define the operation * based on the given condition
def star (a b : ℝ) : ℝ := a * (a * b - 7)

-- Theorem statement
theorem solve_equation : 
  (∃ x : ℝ, (star 3 x) = (star 2 (-8))) ∧ 
  (∀ x : ℝ, (star 3 x) = (star 2 (-8)) → x = -25/9) := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3246_324665


namespace NUMINAMATH_CALUDE_remaining_card_is_seven_l3246_324612

def cards : List Nat := [2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_relatively_prime (a b : Nat) : Prop := Nat.gcd a b = 1

def is_consecutive (a b : Nat) : Prop := a.succ = b ∨ b.succ = a

def is_composite (n : Nat) : Prop := n > 3 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_multiple (a b : Nat) : Prop := ∃ k, k > 1 ∧ (a = k * b ∨ b = k * a)

theorem remaining_card_is_seven (A B C D : List Nat) : 
  A.length = 2 ∧ B.length = 2 ∧ C.length = 2 ∧ D.length = 2 →
  (∀ x ∈ A, x ∈ cards) ∧ (∀ x ∈ B, x ∈ cards) ∧ (∀ x ∈ C, x ∈ cards) ∧ (∀ x ∈ D, x ∈ cards) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (∀ a ∈ A, ∀ b ∈ A, a ≠ b → is_relatively_prime a b ∧ is_consecutive a b) →
  (∀ a ∈ B, ∀ b ∈ B, a ≠ b → ¬is_relatively_prime a b ∧ ¬is_multiple a b) →
  (∀ a ∈ C, ∀ b ∈ C, a ≠ b → is_composite a ∧ is_composite b ∧ is_relatively_prime a b) →
  (∀ a ∈ D, ∀ b ∈ D, a ≠ b → is_multiple a b ∧ ¬is_relatively_prime a b) →
  ∃! x, x ∈ cards ∧ x ∉ A ∧ x ∉ B ∧ x ∉ C ∧ x ∉ D ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_remaining_card_is_seven_l3246_324612


namespace NUMINAMATH_CALUDE_minus_one_circle_plus_four_equals_zero_l3246_324618

-- Define the new operation ⊕
def circle_plus (a b : ℝ) : ℝ := a * b + b

-- Theorem statement
theorem minus_one_circle_plus_four_equals_zero : 
  circle_plus (-1) 4 = 0 := by sorry

end NUMINAMATH_CALUDE_minus_one_circle_plus_four_equals_zero_l3246_324618


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3246_324668

theorem sin_alpha_value (α : Real) : 
  (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) ∈ {(x, y) | ∃ r > 0, x = r * Real.cos α ∧ y = r * Real.sin α} → 
  Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3246_324668


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3246_324633

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3246_324633


namespace NUMINAMATH_CALUDE_gcd_120_4_l3246_324657

/-- The greatest common divisor of 120 and 4 is 4, given they share exactly three positive divisors -/
theorem gcd_120_4 : 
  (∃ (S : Finset Nat), S = {d : Nat | d ∣ 120 ∧ d ∣ 4} ∧ Finset.card S = 3) →
  Nat.gcd 120 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_gcd_120_4_l3246_324657


namespace NUMINAMATH_CALUDE_min_area_is_zero_l3246_324605

/-- Represents a rectangle with one integer dimension and one half-integer dimension -/
structure Rectangle where
  x : ℕ  -- Integer dimension
  y : ℚ  -- Half-integer dimension
  y_half_int : ∃ (n : ℕ), y = n + 1/2
  perimeter_150 : 2 * (x + y) = 150

/-- The area of a rectangle -/
def area (r : Rectangle) : ℚ :=
  r.x * r.y

/-- Theorem stating that the minimum area of a rectangle with the given conditions is 0 -/
theorem min_area_is_zero :
  ∃ (r : Rectangle), ∀ (s : Rectangle), area r ≤ area s :=
sorry

end NUMINAMATH_CALUDE_min_area_is_zero_l3246_324605


namespace NUMINAMATH_CALUDE_initial_blocks_count_l3246_324671

/-- The initial number of blocks Adolfo had -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Adolfo added -/
def added_blocks : ℕ := 30

/-- The total number of blocks after adding -/
def total_blocks : ℕ := 65

/-- Theorem stating that the initial number of blocks is 35 -/
theorem initial_blocks_count : initial_blocks = 35 := by
  sorry

/-- Axiom representing the relationship between initial, added, and total blocks -/
axiom block_relationship : initial_blocks + added_blocks = total_blocks


end NUMINAMATH_CALUDE_initial_blocks_count_l3246_324671


namespace NUMINAMATH_CALUDE_wolf_chase_deer_l3246_324663

theorem wolf_chase_deer (t : ℕ) : t ≤ 28 ↔ ∀ (x y : ℝ), x > 0 → y > 0 → x * y > 0.78 * x * y * (1 + t / 100) := by
  sorry

end NUMINAMATH_CALUDE_wolf_chase_deer_l3246_324663


namespace NUMINAMATH_CALUDE_function_periodicity_l3246_324658

/-- A function f: ℝ → ℝ satisfying f(x-1) + f(x+1) = √2 f(x) for all x ∈ ℝ is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l3246_324658


namespace NUMINAMATH_CALUDE_coin_overlap_area_l3246_324644

theorem coin_overlap_area (square_side : ℝ) (triangle_leg : ℝ) (diamond_side : ℝ) (coin_diameter : ℝ) :
  square_side = 10 →
  triangle_leg = 3 →
  diamond_side = 3 * Real.sqrt 2 →
  coin_diameter = 2 →
  ∃ (overlap_area : ℝ),
    overlap_area = 52 ∧
    overlap_area = (36 + 16 * Real.sqrt 2 + 2 * Real.pi) / 
      ((square_side - coin_diameter) * (square_side - coin_diameter)) :=
by sorry

end NUMINAMATH_CALUDE_coin_overlap_area_l3246_324644


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3246_324632

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2 * x + a) / x + (2 * x - a) / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3246_324632


namespace NUMINAMATH_CALUDE_mary_walking_speed_l3246_324640

/-- The walking speeds of Mary and Sharon, and the time and distance between them -/
structure WalkingProblem where
  mary_speed : ℝ
  sharon_speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : WalkingProblem) : Prop :=
  p.sharon_speed = 6 ∧ p.time = 0.3 ∧ p.distance = 3 ∧
  p.distance = p.mary_speed * p.time + p.sharon_speed * p.time

/-- The theorem stating Mary's walking speed -/
theorem mary_walking_speed (p : WalkingProblem) 
  (h : problem_conditions p) : p.mary_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_walking_speed_l3246_324640


namespace NUMINAMATH_CALUDE_wolf_sheep_eating_time_l3246_324675

/-- If 7 wolves eat 7 sheep in 7 days, then 9 wolves will eat 9 sheep in 7 days. -/
theorem wolf_sheep_eating_time (initial_wolves initial_sheep initial_days : ℕ) 
  (new_wolves new_sheep : ℕ) : 
  initial_wolves = 7 → initial_sheep = 7 → initial_days = 7 →
  new_wolves = 9 → new_sheep = 9 →
  initial_wolves * initial_sheep * new_days = new_wolves * new_sheep * initial_days →
  new_days = 7 :=
by
  sorry

#check wolf_sheep_eating_time

end NUMINAMATH_CALUDE_wolf_sheep_eating_time_l3246_324675


namespace NUMINAMATH_CALUDE_fraction_addition_l3246_324681

theorem fraction_addition : (11 : ℚ) / 12 + 7 / 15 = 83 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3246_324681


namespace NUMINAMATH_CALUDE_candy_distribution_l3246_324694

theorem candy_distribution (total_candy : Nat) (num_friends : Nat) : 
  total_candy = 379 → num_friends = 6 → 
  ∃ (equal_distribution : Nat), 
    equal_distribution ≤ total_candy ∧ 
    equal_distribution.mod num_friends = 0 ∧
    ∀ n : Nat, n ≤ total_candy ∧ n.mod num_friends = 0 → n ≤ equal_distribution := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3246_324694


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l3246_324662

/-- The number of cards in a standard deck of playing cards -/
def standard_deck_size : ℕ := 52

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 319

/-- The number of additional cards the shopkeeper has -/
def additional_cards : ℕ := 7

/-- Theorem: The shopkeeper has 6 complete decks of playing cards -/
theorem shopkeeper_decks :
  (total_cards - additional_cards) / standard_deck_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l3246_324662


namespace NUMINAMATH_CALUDE_parallelogram_area_l3246_324613

theorem parallelogram_area (base height : ℝ) (h1 : base = 24) (h2 : height = 10) :
  base * height = 240 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3246_324613


namespace NUMINAMATH_CALUDE_sum_of_greatest_b_values_l3246_324648

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 →
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_greatest_b_values_l3246_324648


namespace NUMINAMATH_CALUDE_stream_speed_l3246_324610

/-- 
Proves that given a boat with a speed of 51 kmph in still water, 
if the time taken to row upstream is twice the time taken to row downstream, 
then the speed of the stream is 17 kmph.
-/
theorem stream_speed (D : ℝ) (v : ℝ) : 
  (D / (51 - v) = 2 * (D / (51 + v))) → v = 17 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3246_324610


namespace NUMINAMATH_CALUDE_circle_centers_distance_l3246_324615

/-- Given a right triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25,
    and two circles: one centered at O tangent to XZ at Z and passing through Y,
    and another centered at P tangent to XY at Y and passing through Z,
    prove that the length of OP is 25. -/
theorem circle_centers_distance (X Y Z O P : ℝ × ℝ) : 
  -- Right triangle XYZ with given side lengths
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 7^2 →
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 24^2 →
  (Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 = 25^2 →
  -- Circle O is tangent to XZ at Z and passes through Y
  ((O.1 - Z.1)^2 + (O.2 - Z.2)^2 = (O.1 - Y.1)^2 + (O.2 - Y.2)^2) →
  ((O.1 - Z.1) * (Z.1 - X.1) + (O.2 - Z.2) * (Z.2 - X.2) = 0) →
  -- Circle P is tangent to XY at Y and passes through Z
  ((P.1 - Y.1)^2 + (P.2 - Y.2)^2 = (P.1 - Z.1)^2 + (P.2 - Z.2)^2) →
  ((P.1 - Y.1) * (Y.1 - X.1) + (P.2 - Y.2) * (Y.2 - X.2) = 0) →
  -- The distance between O and P is 25
  (O.1 - P.1)^2 + (O.2 - P.2)^2 = 25^2 := by
sorry


end NUMINAMATH_CALUDE_circle_centers_distance_l3246_324615


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3246_324689

theorem largest_four_digit_congruent_to_17_mod_26 : ∃ (n : ℕ), 
  (n ≤ 9999) ∧ 
  (n ≥ 1000) ∧
  (n % 26 = 17) ∧
  (∀ m : ℕ, (m ≤ 9999) → (m ≥ 1000) → (m % 26 = 17) → m ≤ n) ∧
  (n = 9978) := by
sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_26_l3246_324689


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l3246_324638

theorem difference_of_squares_division : (204^2 - 196^2) / 16 = 200 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l3246_324638


namespace NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3246_324672

-- Define what it means for an angle to be in the first quadrant
def is_first_quadrant (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the condition we're interested in
def condition (α : Real) : Prop := Real.sin α * Real.cos α > 0

-- Theorem statement
theorem first_quadrant_sufficient_not_necessary :
  (∀ α : Real, is_first_quadrant α → condition α) ∧
  (∃ α : Real, condition α ∧ ¬is_first_quadrant α) := by sorry

end NUMINAMATH_CALUDE_first_quadrant_sufficient_not_necessary_l3246_324672


namespace NUMINAMATH_CALUDE_indefinite_integral_of_3x_squared_plus_1_l3246_324614

theorem indefinite_integral_of_3x_squared_plus_1 (x : ℝ) (C : ℝ) :
  deriv (fun x => x^3 + x + C) x = 3 * x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_of_3x_squared_plus_1_l3246_324614


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3246_324608

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3246_324608


namespace NUMINAMATH_CALUDE_age_difference_l3246_324664

theorem age_difference (x y z : ℕ) (h : z = x - 18) : 
  (x + y) - (y + z) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3246_324664


namespace NUMINAMATH_CALUDE_angle_conversion_l3246_324646

theorem angle_conversion (θ : Real) : 
  θ * (π / 180) = -10 * π + 7 * π / 4 → 
  ∃ (k : ℤ) (α : Real), 
    θ * (π / 180) = 2 * k * π + α ∧ 
    0 < α ∧ 
    α < 2 * π :=
by sorry

end NUMINAMATH_CALUDE_angle_conversion_l3246_324646


namespace NUMINAMATH_CALUDE_c_share_is_63_l3246_324674

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given renter -/
def calculateShare (renter : Renter) (totalRent : ℕ) (totalOxMonths : ℕ) : ℚ :=
  (renter.oxen * renter.months : ℚ) / totalOxMonths * totalRent

theorem c_share_is_63 (a b c : Renter) (totalRent : ℕ) :
  a.oxen = 10 →
  a.months = 7 →
  b.oxen = 12 →
  b.months = 5 →
  c.oxen = 15 →
  c.months = 3 →
  totalRent = 245 →
  calculateShare c totalRent (a.oxen * a.months + b.oxen * b.months + c.oxen * c.months) = 63 := by
  sorry

#eval calculateShare (Renter.mk 15 3) 245 175

end NUMINAMATH_CALUDE_c_share_is_63_l3246_324674


namespace NUMINAMATH_CALUDE_line_parameterization_l3246_324603

/-- Given a line y = (3/2)x - 25 parameterized by (x,y) = (f(t), 15t - 7),
    prove that f(t) = 10t + 12 is the correct parameterization for x. -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ t : ℝ, (3/2) * f t - 25 = 15 * t - 7) →
  f = λ t => 10 * t + 12 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3246_324603


namespace NUMINAMATH_CALUDE_recurrence_relation_l3246_324688

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 5
  | (n + 3) => (a (n + 2) * a (n + 1) - 2) / a n

def b (n : ℕ) : ℚ := a (2 * n)

theorem recurrence_relation (n : ℕ) :
  b (n + 2) - 4 * b (n + 1) + b n = 0 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_relation_l3246_324688


namespace NUMINAMATH_CALUDE_episodes_per_day_l3246_324636

/-- Given a TV series with 3 seasons of 20 episodes each, watched over 30 days,
    the number of episodes watched per day is 2. -/
theorem episodes_per_day (seasons : ℕ) (episodes_per_season : ℕ) (total_days : ℕ)
    (h1 : seasons = 3)
    (h2 : episodes_per_season = 20)
    (h3 : total_days = 30) :
    (seasons * episodes_per_season) / total_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_episodes_per_day_l3246_324636


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3246_324635

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3246_324635


namespace NUMINAMATH_CALUDE_first_term_range_l3246_324699

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 1 / (2 - a n)

/-- The property that each term is greater than the previous one -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The main theorem stating the range of the first term -/
theorem first_term_range
  (a : ℕ → ℝ)
  (h_recurrence : RecurrenceSequence a)
  (h_increasing : StrictlyIncreasing a) :
  a 1 < 1 :=
sorry

end NUMINAMATH_CALUDE_first_term_range_l3246_324699


namespace NUMINAMATH_CALUDE_population_growth_proof_l3246_324619

/-- Represents the annual growth rate of the population -/
def growth_rate : ℝ := 0.20

/-- Represents the population after one year of growth -/
def final_population : ℝ := 12000

/-- Represents the initial population before growth -/
def initial_population : ℝ := 10000

/-- Theorem stating that if a population grows by 20% in one year to reach 12,000,
    then the initial population was 10,000 -/
theorem population_growth_proof :
  final_population = initial_population * (1 + growth_rate) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_proof_l3246_324619


namespace NUMINAMATH_CALUDE_point_distance_from_two_l3246_324676

theorem point_distance_from_two : ∀ x : ℝ, |x - 2| = 3 → x = -1 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_two_l3246_324676


namespace NUMINAMATH_CALUDE_ticket_123123123_is_red_l3246_324642

/-- Represents the color of a lottery ticket -/
inductive TicketColor
| Red
| Blue
| Green

/-- Represents a 9-digit lottery ticket number -/
def TicketNumber := Fin 9 → Fin 3

/-- The coloring function for tickets -/
def ticketColor : TicketNumber → TicketColor := sorry

/-- Check if two tickets differ in all places -/
def differInAllPlaces (t1 t2 : TicketNumber) : Prop :=
  ∀ i : Fin 9, t1 i ≠ t2 i

/-- The main theorem to prove -/
theorem ticket_123123123_is_red :
  (∀ t1 t2 : TicketNumber, differInAllPlaces t1 t2 → ticketColor t1 ≠ ticketColor t2) →
  ticketColor (λ i => if i.val % 3 = 0 then 0 else if i.val % 3 = 1 then 1 else 2) = TicketColor.Red →
  ticketColor (λ _ => 1) = TicketColor.Green →
  ticketColor (λ i => i.val % 3) = TicketColor.Red :=
sorry

end NUMINAMATH_CALUDE_ticket_123123123_is_red_l3246_324642


namespace NUMINAMATH_CALUDE_balls_after_2010_actions_l3246_324616

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Adds a ball to the first available box and empties boxes to its left -/
def addBall (state : BoxState) : BoxState :=
  match state with
  | [] => [1]
  | (h::t) => if h < 6 then (h+1)::t else 0::addBall t

/-- Performs the ball-adding process n times -/
def performActions (n : Nat) : BoxState :=
  match n with
  | 0 => []
  | n+1 => addBall (performActions n)

/-- Calculates the sum of balls in all boxes -/
def totalBalls (state : BoxState) : Nat :=
  state.sum

/-- The main theorem to prove -/
theorem balls_after_2010_actions :
  totalBalls (performActions 2010) = 16 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_2010_actions_l3246_324616


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l3246_324692

theorem equidistant_complex_function (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ z : ℂ, Complex.abs ((a + Complex.I * b) * z - z) = Complex.abs ((a + Complex.I * b) * z)) →
  Complex.abs (a + Complex.I * b) = 8 →
  b^2 = 255/4 := by
sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l3246_324692


namespace NUMINAMATH_CALUDE_sum_first_three_eq_18_l3246_324693

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  fifth_term_eq_15 : a + 4 * d = 15
  d_eq_3 : d = 3

/-- The sum of the first three terms of the arithmetic sequence -/
def sum_first_three (seq : ArithmeticSequence) : ℕ :=
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d)

/-- Theorem stating that the sum of the first three terms is 18 -/
theorem sum_first_three_eq_18 (seq : ArithmeticSequence) :
  sum_first_three seq = 18 := by
  sorry

#eval sum_first_three ⟨3, 3, rfl, rfl⟩

end NUMINAMATH_CALUDE_sum_first_three_eq_18_l3246_324693


namespace NUMINAMATH_CALUDE_marble_distribution_l3246_324695

theorem marble_distribution (total_marbles : ℕ) (ratio_a ratio_b : ℕ) (given_marbles : ℕ) : 
  total_marbles = 36 →
  ratio_a = 4 →
  ratio_b = 5 →
  given_marbles = 2 →
  (ratio_b * (total_marbles / (ratio_a + ratio_b))) - given_marbles = 18 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l3246_324695


namespace NUMINAMATH_CALUDE_lucia_hip_hop_classes_l3246_324600

/-- Represents the number of hip-hop classes Lucia takes in a week -/
def hip_hop_classes : ℕ := sorry

/-- Represents the cost of one hip-hop class -/
def hip_hop_cost : ℕ := 10

/-- Represents the number of ballet classes Lucia takes in a week -/
def ballet_classes : ℕ := 2

/-- Represents the cost of one ballet class -/
def ballet_cost : ℕ := 12

/-- Represents the number of jazz classes Lucia takes in a week -/
def jazz_classes : ℕ := 1

/-- Represents the cost of one jazz class -/
def jazz_cost : ℕ := 8

/-- Represents the total cost of Lucia's dance classes in one week -/
def total_cost : ℕ := 52

/-- Theorem stating that Lucia takes 2 hip-hop classes in a week -/
theorem lucia_hip_hop_classes : 
  hip_hop_classes = 2 :=
by sorry

end NUMINAMATH_CALUDE_lucia_hip_hop_classes_l3246_324600


namespace NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3246_324628

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅

theorem smallest_odd_with_five_prime_factors :
  (is_odd 15015 ∧ has_five_different_prime_factors 15015) ∧
  ∀ m : ℕ, m < 15015 → ¬(is_odd m ∧ has_five_different_prime_factors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_with_five_prime_factors_l3246_324628


namespace NUMINAMATH_CALUDE_equation_three_solutions_l3246_324639

theorem equation_three_solutions (a : ℝ) :
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, x^2 * a - 2*x + 1 = 3 * |x| ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) ↔
  a = (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_three_solutions_l3246_324639


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3246_324604

/-- An isosceles triangle with sides 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = p) →  -- Definition of perimeter
    p = 22  -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3246_324604


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3246_324625

theorem trigonometric_identity (α : ℝ) :
  Real.cos (5 / 2 * Real.pi - 6 * α) * Real.sin (Real.pi - 2 * α)^3 -
  Real.cos (6 * α - Real.pi) * Real.sin (Real.pi / 2 - 2 * α)^3 =
  Real.cos (4 * α)^3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3246_324625


namespace NUMINAMATH_CALUDE_train_passengers_l3246_324645

theorem train_passengers (initial_passengers : ℕ) : 
  (initial_passengers - 263 + 419 = 725) → initial_passengers = 569 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l3246_324645


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3246_324667

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3246_324667


namespace NUMINAMATH_CALUDE_train_travel_time_equation_l3246_324678

/-- Proves that the equation for the difference in travel times between two trains is correct -/
theorem train_travel_time_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_equation_l3246_324678


namespace NUMINAMATH_CALUDE_representatives_selection_count_l3246_324634

def num_female : ℕ := 3
def num_male : ℕ := 4
def num_representatives : ℕ := 3

theorem representatives_selection_count :
  (Finset.sum (Finset.range (num_representatives - 1)) (λ k =>
    Nat.choose num_female (k + 1) * Nat.choose num_male (num_representatives - k - 1)))
  = 30 := by sorry

end NUMINAMATH_CALUDE_representatives_selection_count_l3246_324634


namespace NUMINAMATH_CALUDE_manuscript_revision_cost_l3246_324620

theorem manuscript_revision_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ)
  (initial_cost_per_page : ℚ) (total_cost : ℚ)
  (h1 : total_pages = 100)
  (h2 : revised_once = 30)
  (h3 : revised_twice = 20)
  (h4 : initial_cost_per_page = 5)
  (h5 : total_cost = 710)
  (h6 : total_pages = revised_once + revised_twice + (total_pages - revised_once - revised_twice)) :
  let revision_cost : ℚ := (total_cost - (initial_cost_per_page * total_pages)) / (revised_once + 2 * revised_twice)
  revision_cost = 3 := by sorry

end NUMINAMATH_CALUDE_manuscript_revision_cost_l3246_324620


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l3246_324602

/-- The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) -/
theorem probability_at_least_one_multiple_of_four : ℚ :=
  28051 / 50000

/-- The set of integers from 1 to 50 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

/-- The number of elements in set S -/
def S_size : ℕ := 50

/-- The set of multiples of 4 in S -/
def M : Set ℕ := {n ∈ S | n % 4 = 0}

/-- The number of elements in set M -/
def M_size : ℕ := 12

/-- The probability of selecting a number that is not a multiple of 4 -/
def p_not_multiple_of_four : ℚ := (S_size - M_size) / S_size

/-- The probability of selecting three numbers, none of which are multiples of 4 -/
def p_none_multiple_of_four : ℚ := p_not_multiple_of_four ^ 3

/-- Theorem: The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) is 28051/50000 -/
theorem prob_at_least_one_multiple_of_four :
  1 - p_none_multiple_of_four = probability_at_least_one_multiple_of_four :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l3246_324602


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3246_324647

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3246_324647


namespace NUMINAMATH_CALUDE_man_gained_three_toys_cost_l3246_324630

/-- The number of toys whose cost price the man gained -/
def toys_gained (num_sold : ℕ) (selling_price : ℕ) (cost_price : ℕ) : ℕ :=
  (selling_price - num_sold * cost_price) / cost_price

theorem man_gained_three_toys_cost :
  toys_gained 18 27300 1300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_man_gained_three_toys_cost_l3246_324630


namespace NUMINAMATH_CALUDE_problem_solution_l3246_324611

theorem problem_solution (x : ℝ) : (3 * x + 20 = (1 / 3) * (7 * x + 60)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3246_324611


namespace NUMINAMATH_CALUDE_lumberjack_trees_l3246_324666

theorem lumberjack_trees (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  total_firewood / (logs_per_tree * firewood_per_log) = 25 :=
by sorry

end NUMINAMATH_CALUDE_lumberjack_trees_l3246_324666


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3246_324641

theorem scientific_notation_equality (n : ℝ) (h : n = 58000) : n = 5.8 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3246_324641


namespace NUMINAMATH_CALUDE_inverse_99_mod_101_l3246_324617

theorem inverse_99_mod_101 : ∃ x : ℕ, x ∈ Finset.range 101 ∧ (99 * x) % 101 = 1 := by
  use 51
  sorry

end NUMINAMATH_CALUDE_inverse_99_mod_101_l3246_324617


namespace NUMINAMATH_CALUDE_natalia_clip_sales_l3246_324684

/-- Natalia's clip sales problem -/
theorem natalia_clip_sales 
  (x : ℝ) -- number of clips sold to each friend in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- total earnings in dollars
  (h1 : y = x / 2) -- y is half of x
  : (48 * x + y = 97 * x / 2) ∧ (z / (48 * x + y) = 2 * z / (97 * x)) := by
  sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l3246_324684
