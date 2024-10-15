import Mathlib

namespace NUMINAMATH_CALUDE_pentadecagon_diagonals_l4047_404737

/-- Number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

theorem pentadecagon_diagonals :
  num_diagonals pentadecagon_sides = 90 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_diagonals_l4047_404737


namespace NUMINAMATH_CALUDE_island_ocean_depth_l4047_404745

/-- Represents a cone-shaped island -/
structure ConeIsland where
  height : ℝ
  volumeAboveWater : ℝ

/-- Calculates the depth of the ocean at the base of a cone-shaped island -/
def oceanDepth (island : ConeIsland) : ℝ :=
  sorry

/-- Theorem stating the depth of the ocean for the given island -/
theorem island_ocean_depth :
  let island : ConeIsland := { height := 10000, volumeAboveWater := 1/10 }
  oceanDepth island = 350 :=
sorry

end NUMINAMATH_CALUDE_island_ocean_depth_l4047_404745


namespace NUMINAMATH_CALUDE_find_divisor_l4047_404782

theorem find_divisor (x : ℝ) (y : ℝ) 
  (h1 : (x - 5) / y = 7)
  (h2 : (x - 6) / 8 = 6) : 
  y = 7 := by sorry

end NUMINAMATH_CALUDE_find_divisor_l4047_404782


namespace NUMINAMATH_CALUDE_sequence_product_representation_l4047_404719

theorem sequence_product_representation (n a : ℕ) :
  ∃ u v : ℕ, (n : ℚ) / (n + a) = (u : ℚ) / (u + a) * (v : ℚ) / (v + a) := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_representation_l4047_404719


namespace NUMINAMATH_CALUDE_berry_picking_difference_l4047_404757

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  sergey_basket_ratio : ℚ
  dima_basket_ratio : ℚ
  sergey_speed_multiplier : ℕ

/-- The main theorem about the berry picking scenario -/
theorem berry_picking_difference (scenario : BerryPicking) 
  (h1 : scenario.total_berries = 900)
  (h2 : scenario.sergey_basket_ratio = 1/2)
  (h3 : scenario.dima_basket_ratio = 2/3)
  (h4 : scenario.sergey_speed_multiplier = 2) :
  ∃ (sergey_basket dima_basket : ℕ), 
    sergey_basket = 300 ∧ 
    dima_basket = 200 ∧ 
    sergey_basket - dima_basket = 100 := by
  sorry

#check berry_picking_difference

end NUMINAMATH_CALUDE_berry_picking_difference_l4047_404757


namespace NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l4047_404785

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_six_million_scientific_notation :
  toScientificNotation 2600000 = ScientificNotation.mk 2.6 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l4047_404785


namespace NUMINAMATH_CALUDE_matrix_equality_l4047_404771

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : A * B + !![2, 0; 0, 2] = A + B)
  (h2 : A * B = !![38/3, 4/3; -8/3, 4/3]) :
  B * A = !![44/3, 4/3; -8/3, 10/3] := by sorry

end NUMINAMATH_CALUDE_matrix_equality_l4047_404771


namespace NUMINAMATH_CALUDE_missing_number_is_five_l4047_404744

theorem missing_number_is_five : ∃ x : ℕ, 10111 - x * 2 * 5 = 10011 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_five_l4047_404744


namespace NUMINAMATH_CALUDE_five_polyhedra_types_l4047_404724

/-- A topologically correct (simply connected) polyhedron --/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler : vertices - edges + faces = 2

/-- The possible face types of a polyhedron --/
inductive FaceType
  | Triangle
  | Quadrilateral
  | Pentagon

/-- A function that checks if a polyhedron is valid given its face type --/
def is_valid_polyhedron (p : Polyhedron) (face_type : FaceType) : Prop :=
  match face_type with
  | FaceType.Triangle => p.edges = (3 * p.faces) / 2
  | FaceType.Quadrilateral => p.edges = 2 * p.faces
  | FaceType.Pentagon => p.edges = (5 * p.faces) / 2

/-- The theorem stating that there are exactly 5 types of topologically correct polyhedra --/
theorem five_polyhedra_types :
  ∃! (types : Finset Polyhedron),
    (∀ p ∈ types, ∃ face_type, is_valid_polyhedron p face_type) ∧
    types.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_polyhedra_types_l4047_404724


namespace NUMINAMATH_CALUDE_farmer_livestock_purchase_l4047_404796

theorem farmer_livestock_purchase :
  ∀ (num_cows num_sheep num_rabbits : ℕ)
    (cost_cow cost_sheep cost_rabbit : ℚ),
  num_cows + num_sheep + num_rabbits = 100 →
  cost_cow = 50 →
  cost_sheep = 10 →
  cost_rabbit = 1/2 →
  cost_cow * num_cows + cost_sheep * num_sheep + cost_rabbit * num_rabbits = 1000 →
  num_cows = 19 ∧ num_sheep = 1 ∧ num_rabbits = 80 ∧
  cost_cow * num_cows = 950 ∧ cost_sheep * num_sheep = 10 ∧ cost_rabbit * num_rabbits = 40 :=
by sorry

end NUMINAMATH_CALUDE_farmer_livestock_purchase_l4047_404796


namespace NUMINAMATH_CALUDE_perpendicular_slopes_not_always_negative_one_l4047_404763

/-- Two lines in a 2D plane --/
structure Line where
  slope : Option ℝ

/-- Perpendicularity relation between two lines --/
def perpendicular (l₁ l₂ : Line) : Prop :=
  match l₁.slope, l₂.slope with
  | some m₁, some m₂ => m₁ * m₂ = -1
  | none, some m => m = 0
  | some m, none => m = 0
  | none, none => False

/-- Theorem stating that there exist perpendicular lines whose slopes do not multiply to -1 --/
theorem perpendicular_slopes_not_always_negative_one :
  ∃ (l₁ l₂ : Line), perpendicular l₁ l₂ ∧
    (l₁.slope.isNone ∨ l₂.slope.isNone ∨
     ∃ (m₁ m₂ : ℝ), l₁.slope = some m₁ ∧ l₂.slope = some m₂ ∧ m₁ * m₂ ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slopes_not_always_negative_one_l4047_404763


namespace NUMINAMATH_CALUDE_least_n_for_inequality_l4047_404711

theorem least_n_for_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 15) := by
  sorry

end NUMINAMATH_CALUDE_least_n_for_inequality_l4047_404711


namespace NUMINAMATH_CALUDE_eighth_grade_higher_mean_l4047_404792

/-- Represents the score distribution for a grade --/
structure ScoreDistribution :=
  (score60to70 : Nat)
  (score70to80 : Nat)
  (score80to90 : Nat)
  (score90to100 : Nat)

/-- Represents the statistics for a grade --/
structure GradeStatistics :=
  (mean : Float)
  (median : Float)
  (mode : Nat)

/-- Theorem: 8th grade has a higher mean score than 7th grade --/
theorem eighth_grade_higher_mean
  (grade7_dist : ScoreDistribution)
  (grade8_dist : ScoreDistribution)
  (grade7_stats : GradeStatistics)
  (grade8_stats : GradeStatistics)
  (h1 : grade7_dist.score60to70 = 1)
  (h2 : grade7_dist.score70to80 = 4)
  (h3 : grade7_dist.score80to90 = 3)
  (h4 : grade7_dist.score90to100 = 2)
  (h5 : grade8_dist.score60to70 = 1)
  (h6 : grade8_dist.score70to80 = 2)
  (h7 : grade8_dist.score80to90 = 5)
  (h8 : grade8_dist.score90to100 = 2)
  (h9 : grade7_stats.mean = 84.6)
  (h10 : grade8_stats.mean = 86.3)
  : grade8_stats.mean > grade7_stats.mean := by
  sorry

#check eighth_grade_higher_mean

end NUMINAMATH_CALUDE_eighth_grade_higher_mean_l4047_404792


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l4047_404783

theorem complex_number_quadrant : ∃ (z : ℂ), 
  z = (1 - Complex.I * Real.sqrt 3) / (2 * Complex.I) ∧ 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l4047_404783


namespace NUMINAMATH_CALUDE_horner_v3_value_l4047_404779

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v3_value :
  let x := 2
  let v0 := 1
  let v1 := horner_step v0 x (-12)
  let v2 := horner_step v1 x 60
  let v3 := horner_step v2 x (-160)
  v3 = -80 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l4047_404779


namespace NUMINAMATH_CALUDE_sticker_problem_l4047_404768

/-- Given a number of stickers per page, a number of initial pages, and losing one page,
    calculate the remaining number of stickers. -/
def remaining_stickers (stickers_per_page : ℕ) (initial_pages : ℕ) : ℕ :=
  stickers_per_page * (initial_pages - 1)

/-- Theorem stating that with 20 stickers per page, 12 initial pages, and losing one page,
    the remaining number of stickers is 220. -/
theorem sticker_problem :
  remaining_stickers 20 12 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_problem_l4047_404768


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l4047_404704

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, x^2 + 1 < 2*x) ↔ (∀ x : ℝ, x^2 + 1 ≥ 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_l4047_404704


namespace NUMINAMATH_CALUDE_monkey_peach_division_l4047_404712

theorem monkey_peach_division (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, 100 = n * k + 10) →
  (∃ m : ℕ, 1000 = n * m + 10) :=
by sorry

end NUMINAMATH_CALUDE_monkey_peach_division_l4047_404712


namespace NUMINAMATH_CALUDE_number_and_percentage_l4047_404700

theorem number_and_percentage (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 35) : 
  (40/100) * N = 420 := by
sorry

end NUMINAMATH_CALUDE_number_and_percentage_l4047_404700


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l4047_404791

/-- A parallelogram with opposite vertices (2, -3) and (14, 9) has its diagonals intersecting at (8, 3) -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l4047_404791


namespace NUMINAMATH_CALUDE_dave_bought_26_tshirts_l4047_404761

/-- The number of T-shirts Dave bought -/
def total_tshirts (white_packs blue_packs : ℕ) (white_per_pack blue_per_pack : ℕ) : ℕ :=
  white_packs * white_per_pack + blue_packs * blue_per_pack

/-- Proof that Dave bought 26 T-shirts -/
theorem dave_bought_26_tshirts :
  total_tshirts 3 2 6 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_bought_26_tshirts_l4047_404761


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l4047_404778

theorem sum_of_four_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l4047_404778


namespace NUMINAMATH_CALUDE_angle_at_two_fifteen_l4047_404707

/-- Represents a clock with hour and minute hands -/
structure Clock where
  hourHand : ℝ  -- Position of hour hand (in hours)
  minuteHand : ℝ  -- Position of minute hand (in minutes)

/-- Calculates the angle between hour and minute hands at a given time -/
def angleBetweenHands (c : Clock) : ℝ :=
  let hourAngle := c.hourHand * 30 + c.minuteHand * 0.5
  let minuteAngle := c.minuteHand * 6
  abs (hourAngle - minuteAngle)

/-- Theorem stating that at 2:15, the angle between hour and minute hands is 22.5° -/
theorem angle_at_two_fifteen :
  let c : Clock := { hourHand := 2, minuteHand := 15 }
  angleBetweenHands c = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_angle_at_two_fifteen_l4047_404707


namespace NUMINAMATH_CALUDE_find_n_l4047_404715

theorem find_n : ∃ n : ℤ, 5^2 - 7 = 3^3 + n ∧ n = -9 := by sorry

end NUMINAMATH_CALUDE_find_n_l4047_404715


namespace NUMINAMATH_CALUDE_exists_divisible_power_minus_one_l4047_404793

theorem exists_divisible_power_minus_one (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k > 0 ∧ k < n ∧ (n ∣ 2^k - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_power_minus_one_l4047_404793


namespace NUMINAMATH_CALUDE_permutation_combination_inequality_l4047_404786

theorem permutation_combination_inequality (n : ℕ+) :
  (n.val.factorial / (n.val - 2).factorial)^2 > 6 * (n.val.choose 4) ↔ n.val ∈ ({2, 3, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_permutation_combination_inequality_l4047_404786


namespace NUMINAMATH_CALUDE_right_triangle_existence_l4047_404733

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + 2 + m

theorem right_triangle_existence (m : ℝ) :
  m > 0 →
  (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (f m a)^2 + (f m b)^2 = (f m c)^2 ∨
    (f m a)^2 + (f m c)^2 = (f m b)^2 ∨
    (f m b)^2 + (f m c)^2 = (f m a)^2) ↔
  0 < m ∧ m < 4 + 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l4047_404733


namespace NUMINAMATH_CALUDE_exist_unit_tetrahedron_with_interior_point_l4047_404760

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the volume of a tetrahedron given four points -/
def tetrahedronVolume (p1 p2 p3 p4 : Point3D) : ℝ := sorry

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if a point is inside a tetrahedron -/
def isInsideTetrahedron (p : Point3D) (t1 t2 t3 t4 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem exist_unit_tetrahedron_with_interior_point 
  (n : ℕ) 
  (points : Fin n → Point3D) 
  (h_not_coplanar : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → ¬areCoplanar (points i) (points j) (points k) (points l))
  (h_max_volume : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → tetrahedronVolume (points i) (points j) (points k) (points l) ≤ 0.037)
  : ∃ (t1 t2 t3 t4 : Point3D), 
    tetrahedronVolume t1 t2 t3 t4 = 1 ∧ 
    ∃ (i : Fin n), isInsideTetrahedron (points i) t1 t2 t3 t4 := by
  sorry

end NUMINAMATH_CALUDE_exist_unit_tetrahedron_with_interior_point_l4047_404760


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l4047_404706

def is_palindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, is_palindrome n → n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l4047_404706


namespace NUMINAMATH_CALUDE_triangle_side_length_l4047_404755

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4047_404755


namespace NUMINAMATH_CALUDE_probability_closer_to_five_than_one_l4047_404727

noncomputable def probability_closer_to_five (a b c : ℝ) : ℝ :=
  let midpoint := (a + c) / 2
  let favorable_length := b - midpoint
  let total_length := b - 0
  favorable_length / total_length

theorem probability_closer_to_five_than_one :
  probability_closer_to_five 1 6 5 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_five_than_one_l4047_404727


namespace NUMINAMATH_CALUDE_no_real_solutions_l4047_404740

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y + 2 * z^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4047_404740


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_l4047_404770

theorem mod_equivalence_unique : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -150 ≡ n [ZMOD 23] ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_l4047_404770


namespace NUMINAMATH_CALUDE_gpa_probability_l4047_404776

structure GradeSystem where
  a_points : ℕ := 4
  b_points : ℕ := 3
  c_points : ℕ := 2
  d_points : ℕ := 1

structure CourseGrades where
  math_grade : ℕ
  science_grade : ℕ
  english_grade : ℕ
  history_grade : ℕ

def calculate_gpa (gs : GradeSystem) (cg : CourseGrades) : ℚ :=
  (cg.math_grade + cg.science_grade + cg.english_grade + cg.history_grade : ℚ) / 4

def english_prob_a : ℚ := 1/3
def english_prob_b : ℚ := 1/4
def english_prob_c : ℚ := 1 - english_prob_a - english_prob_b

def history_prob_a : ℚ := 1/5
def history_prob_b : ℚ := 2/5
def history_prob_c : ℚ := 1 - history_prob_a - history_prob_b

def prob_gpa_at_least (target_gpa : ℚ) (gs : GradeSystem) : ℚ :=
  let prob_aa := english_prob_a * history_prob_a
  let prob_ab := english_prob_a * history_prob_b
  let prob_ba := english_prob_b * history_prob_a
  prob_aa + prob_ab + prob_ba

theorem gpa_probability (gs : GradeSystem) :
  prob_gpa_at_least (15/4) gs = 1/4 :=
sorry

end NUMINAMATH_CALUDE_gpa_probability_l4047_404776


namespace NUMINAMATH_CALUDE_student_council_committees_l4047_404734

theorem student_council_committees (n : ℕ) 
  (h : n * (n - 1) / 2 = 15) : 
  Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_council_committees_l4047_404734


namespace NUMINAMATH_CALUDE_angle_measure_of_extended_sides_l4047_404748

/-- A regular octagon is a polygon with 8 sides of equal length and 8 equal angles -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Given a regular octagon ABCDEFGH, extend sides AB and BC to meet at point P -/
def extend_sides (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- The measure of an angle in degrees -/
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_of_extended_sides (octagon : RegularOctagon) : 
  let p := extend_sides octagon
  angle_measure (octagon.vertices 0) p (octagon.vertices 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_of_extended_sides_l4047_404748


namespace NUMINAMATH_CALUDE_rabbit_hop_time_l4047_404773

/-- Proves that a rabbit hopping at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_hop_time :
  let speed : ℝ := 5  -- miles per hour
  let distance : ℝ := 2  -- miles
  let time_hours : ℝ := distance / speed
  let minutes_per_hour : ℝ := 60
  let time_minutes : ℝ := time_hours * minutes_per_hour
  time_minutes = 24 := by sorry

end NUMINAMATH_CALUDE_rabbit_hop_time_l4047_404773


namespace NUMINAMATH_CALUDE_bicyclists_meet_at_1030_l4047_404728

-- Define the problem parameters
def alicia_start_time : ℝ := 7.75  -- 7:45 AM in decimal hours
def david_start_time : ℝ := 8.25   -- 8:15 AM in decimal hours
def alicia_speed : ℝ := 15         -- miles per hour
def david_speed : ℝ := 18          -- miles per hour
def total_distance : ℝ := 84       -- miles

-- Define the meeting time in decimal hours (10:30 AM = 10.5)
def meeting_time : ℝ := 10.5

-- Theorem statement
theorem bicyclists_meet_at_1030 :
  let t := meeting_time - alicia_start_time
  alicia_speed * t + david_speed * (t - (david_start_time - alicia_start_time)) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bicyclists_meet_at_1030_l4047_404728


namespace NUMINAMATH_CALUDE_continued_fraction_simplification_l4047_404742

theorem continued_fraction_simplification :
  1 + (3 / (4 + (5 / 6))) = 47 / 29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_simplification_l4047_404742


namespace NUMINAMATH_CALUDE_fan_shaped_segment_edge_length_l4047_404797

theorem fan_shaped_segment_edge_length (r : ℝ) (angle : ℝ) :
  r = 2 →
  angle = 90 →
  let arc_length := (2 * π * r) * ((360 - angle) / 360)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 := by sorry

end NUMINAMATH_CALUDE_fan_shaped_segment_edge_length_l4047_404797


namespace NUMINAMATH_CALUDE_pies_sold_weekend_l4047_404717

/-- Represents the number of slices in each type of pie -/
structure PieSlices where
  apple : ℕ
  peach : ℕ
  cherry : ℕ

/-- Represents the number of customers who ordered each type of pie -/
structure PieOrders where
  apple : ℕ
  peach : ℕ
  cherry : ℕ

/-- Calculates the total number of pies sold given the number of slices per pie and the number of orders -/
def totalPiesSold (slices : PieSlices) (orders : PieOrders) : ℕ :=
  let applePies := (orders.apple + slices.apple - 1) / slices.apple
  let peachPies := (orders.peach + slices.peach - 1) / slices.peach
  let cherryPies := (orders.cherry + slices.cherry - 1) / slices.cherry
  applePies + peachPies + cherryPies

/-- Theorem stating that given the specific conditions, the total pies sold is 29 -/
theorem pies_sold_weekend (slices : PieSlices) (orders : PieOrders)
    (h1 : slices.apple = 8)
    (h2 : slices.peach = 6)
    (h3 : slices.cherry = 10)
    (h4 : orders.apple = 88)
    (h5 : orders.peach = 78)
    (h6 : orders.cherry = 45) :
    totalPiesSold slices orders = 29 := by
  sorry


end NUMINAMATH_CALUDE_pies_sold_weekend_l4047_404717


namespace NUMINAMATH_CALUDE_number_division_problem_l4047_404781

theorem number_division_problem : ∃ x : ℚ, (x / 5) - (x / 6) = 30 ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l4047_404781


namespace NUMINAMATH_CALUDE_other_group_cleaned_area_l4047_404703

theorem other_group_cleaned_area
  (total_area : ℕ)
  (lizzies_group_area : ℕ)
  (remaining_area : ℕ)
  (h1 : total_area = 900)
  (h2 : lizzies_group_area = 250)
  (h3 : remaining_area = 385) :
  total_area - remaining_area - lizzies_group_area = 265 :=
by sorry

end NUMINAMATH_CALUDE_other_group_cleaned_area_l4047_404703


namespace NUMINAMATH_CALUDE_candy_problem_l4047_404754

theorem candy_problem (C : ℕ) : 
  (C - (C / 3) - (C - (C / 3)) / 4 + C / 2 - 7 = 16) → C = 23 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l4047_404754


namespace NUMINAMATH_CALUDE_point_above_with_distance_l4047_404714

/-- Given two points P(3, a) and Q(3, 4) in a Cartesian coordinate system,
    if P is above Q and the distance between P and Q is 3,
    then the y-coordinate of P (which is a) equals 7. -/
theorem point_above_with_distance (a : ℝ) :
  a > 4 →  -- P is above Q
  (3 - 3)^2 + (a - 4)^2 = 3^2 →  -- Distance formula
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_point_above_with_distance_l4047_404714


namespace NUMINAMATH_CALUDE_parabola_chord_slope_l4047_404741

/-- Given a parabola y² = 2px, a point Q(q, 0) where q < 0, and a line x = s where p > 0 and s > 0,
    this theorem proves the slope of a chord through Q that intersects the parabola at two points
    equidistant from the line x = s. -/
theorem parabola_chord_slope (p s q : ℝ) (hp : p > 0) (hs : s > 0) (hq : q < 0) (h_feasible : s ≥ -q) :
  ∃ m : ℝ, m ^ 2 = p / (s - q) ∧
    ∀ x y : ℝ, y ^ 2 = 2 * p * x →
      y = m * (x - q) →
      ∃ x' y' : ℝ, y' ^ 2 = 2 * p * x' ∧
                  y' = m * (x' - q) ∧
                  x + x' = 2 * s :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_slope_l4047_404741


namespace NUMINAMATH_CALUDE_unbalanceable_pairs_l4047_404767

-- Define the set of weights
def weights : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2018}

-- Define a function to check if a pair can be balanced
def can_balance (a b : ℕ) : Prop :=
  ∃ (c d : ℕ), c ∈ weights ∧ d ∈ weights ∧ c ≠ a ∧ c ≠ b ∧ d ≠ a ∧ d ≠ b ∧ a + b = c + d

-- Main theorem
theorem unbalanceable_pairs :
  ∀ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a < b →
    (¬ can_balance a b ↔ (a = 1 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 2016 ∧ b = 2018) ∨ (a = 2017 ∧ b = 2018)) :=
by sorry

end NUMINAMATH_CALUDE_unbalanceable_pairs_l4047_404767


namespace NUMINAMATH_CALUDE_probability_at_least_one_juice_l4047_404725

def total_bottles : ℕ := 5
def juice_bottles : ℕ := 2
def selected_bottles : ℕ := 2

theorem probability_at_least_one_juice :
  let non_juice_bottles := total_bottles - juice_bottles
  let total_combinations := (total_bottles.choose selected_bottles : ℚ)
  let non_juice_combinations := (non_juice_bottles.choose selected_bottles : ℚ)
  (1 - non_juice_combinations / total_combinations) = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_juice_l4047_404725


namespace NUMINAMATH_CALUDE_company_fund_distribution_l4047_404756

/-- Represents the company fund distribution problem --/
theorem company_fund_distribution (n : ℕ) (initial_fund : ℕ) : 
  (75 * n = initial_fund + 15) →  -- Planned distribution
  (60 * n + 210 = initial_fund) →  -- Actual distribution
  initial_fund = 1110 := by
  sorry

end NUMINAMATH_CALUDE_company_fund_distribution_l4047_404756


namespace NUMINAMATH_CALUDE_proposition_evaluation_l4047_404752

theorem proposition_evaluation (a b : ℝ) : 
  (¬ (∀ a, a < 2 → a^2 < 4)) ∧ 
  (∀ a, a^2 < 4 → a < 2) ∧ 
  (∀ a, a ≥ 2 → a^2 ≥ 4) ∧ 
  (¬ (∀ a, a^2 ≥ 4 → a ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l4047_404752


namespace NUMINAMATH_CALUDE_price_increase_decrease_l4047_404780

theorem price_increase_decrease (x : ℝ) : 
  (100 + x) * (100 - 23.076923076923077) / 100 = 100 → x = 30 := by
sorry

end NUMINAMATH_CALUDE_price_increase_decrease_l4047_404780


namespace NUMINAMATH_CALUDE_correct_middle_managers_sample_l4047_404722

/-- Represents the composition of employees in a company -/
structure CompanyComposition where
  total_employees : ℕ
  middle_managers : ℕ
  sample_size : ℕ

/-- Calculates the number of middle managers to be selected in a stratified random sample -/
def middleManagersInSample (comp : CompanyComposition) : ℕ :=
  (comp.sample_size * comp.middle_managers) / comp.total_employees

/-- Theorem stating that for the given company composition, 
    the number of middle managers in the sample should be 30 -/
theorem correct_middle_managers_sample :
  let comp : CompanyComposition := {
    total_employees := 1000,
    middle_managers := 150,
    sample_size := 200
  }
  middleManagersInSample comp = 30 := by
  sorry


end NUMINAMATH_CALUDE_correct_middle_managers_sample_l4047_404722


namespace NUMINAMATH_CALUDE_system_solution_l4047_404751

theorem system_solution : 
  ∃ (x y z w : ℝ), 
    (x = 3 ∧ y = 1 ∧ z = 2 ∧ w = 2) ∧
    (x - y + z - w = 2) ∧
    (x^2 - y^2 + z^2 - w^2 = 6) ∧
    (x^3 - y^3 + z^3 - w^3 = 20) ∧
    (x^4 - y^4 + z^4 - w^4 = 66) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4047_404751


namespace NUMINAMATH_CALUDE_fold_equilateral_triangle_l4047_404702

-- Define an equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the folding operation
def FoldTriangle (A B C P Q : ℝ × ℝ) : Prop :=
  P.1 = B.1 + 7 * (A.1 - B.1) / 10 ∧
  P.2 = B.2 + 7 * (A.2 - B.2) / 10 ∧
  Q.1 = C.1 + 7 * (A.1 - C.1) / 10 ∧
  Q.2 = C.2 + 7 * (A.2 - C.2) / 10

theorem fold_equilateral_triangle :
  ∀ (A B C P Q : ℝ × ℝ),
  EquilateralTriangle A B C →
  dist A B = 10 →
  FoldTriangle A B C P Q →
  (dist P Q)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fold_equilateral_triangle_l4047_404702


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l4047_404764

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/5
def prob_traps : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands) *
  (prob_treasure ^ num_treasure_islands) *
  (prob_neither ^ (num_islands - num_treasure_islands)) =
  67/2500 := by sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l4047_404764


namespace NUMINAMATH_CALUDE_carol_owns_twice_as_many_as_cathy_l4047_404729

/-- Represents the number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  susan : ℕ
  carol : ℕ

/-- The conditions of the car ownership problem -/
def carProblemConditions (o : CarOwnership) : Prop :=
  o.lindsey = o.cathy + 4 ∧
  o.susan = o.carol - 2 ∧
  o.carol = 2 * o.cathy ∧
  o.cathy + o.lindsey + o.susan + o.carol = 32 ∧
  o.cathy = 5

theorem carol_owns_twice_as_many_as_cathy (o : CarOwnership) 
  (h : carProblemConditions o) : o.carol = 2 * o.cathy := by
  sorry

#check carol_owns_twice_as_many_as_cathy

end NUMINAMATH_CALUDE_carol_owns_twice_as_many_as_cathy_l4047_404729


namespace NUMINAMATH_CALUDE_consecutive_base_problem_l4047_404738

/-- Given two consecutive positive integers X and Y, 
    if 312 in base X minus 65 in base Y equals 97 in base (X+Y), 
    then X+Y equals 7 -/
theorem consecutive_base_problem (X Y : ℕ) : 
  X > 0 → Y > 0 → Y = X + 1 → 
  (3 * X^2 + X + 2) - (6 * Y + 5) = 9 * (X + Y) + 7 → 
  X + Y = 7 := by sorry

end NUMINAMATH_CALUDE_consecutive_base_problem_l4047_404738


namespace NUMINAMATH_CALUDE_students_not_in_biology_l4047_404794

theorem students_not_in_biology (total_students : ℕ) (enrolled_percentage : ℚ) : 
  total_students = 880 → 
  enrolled_percentage = 30 / 100 → 
  (total_students : ℚ) * (1 - enrolled_percentage) = 616 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l4047_404794


namespace NUMINAMATH_CALUDE_virginia_eggs_problem_l4047_404790

/-- Given that Virginia ends with 93 eggs after Amy takes 3 eggs, 
    prove that Virginia started with 96 eggs. -/
theorem virginia_eggs_problem (initial_eggs final_eggs eggs_taken : ℕ) :
  final_eggs = 93 → eggs_taken = 3 → initial_eggs = final_eggs + eggs_taken →
  initial_eggs = 96 := by
sorry

end NUMINAMATH_CALUDE_virginia_eggs_problem_l4047_404790


namespace NUMINAMATH_CALUDE_probability_white_ball_l4047_404746

def num_white_balls : ℕ := 8
def num_black_balls : ℕ := 7
def num_red_balls : ℕ := 5

def total_balls : ℕ := num_white_balls + num_black_balls + num_red_balls

theorem probability_white_ball :
  (num_white_balls : ℚ) / (total_balls : ℚ) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_white_ball_l4047_404746


namespace NUMINAMATH_CALUDE_train_length_l4047_404708

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 150 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 370 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l4047_404708


namespace NUMINAMATH_CALUDE_cosine_cutting_plane_angle_l4047_404769

/-- Regular hexagonal pyramid with specific cross-section --/
structure HexagonalPyramid where
  base_side : ℝ
  vertex_to_plane_dist : ℝ

/-- Theorem: Cosine of angle between cutting plane and base plane in specific hexagonal pyramid --/
theorem cosine_cutting_plane_angle (pyramid : HexagonalPyramid) 
  (h1 : pyramid.base_side = 8)
  (h2 : pyramid.vertex_to_plane_dist = 3 * Real.sqrt (13/7))
  : Real.sqrt 3 / 4 = 
    Real.sqrt (1 - (28 * pyramid.vertex_to_plane_dist^2) / (9 * pyramid.base_side^2)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_cutting_plane_angle_l4047_404769


namespace NUMINAMATH_CALUDE_units_digit_product_zero_exists_l4047_404747

theorem units_digit_product_zero_exists : ∃ (a b : ℕ), 
  (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) ∧ ((a * b) % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_zero_exists_l4047_404747


namespace NUMINAMATH_CALUDE_carrot_distribution_l4047_404739

theorem carrot_distribution (total_carrots : ℕ) (num_goats : ℕ) 
  (h1 : total_carrots = 47) (h2 : num_goats = 4) : 
  total_carrots % num_goats = 3 := by
  sorry

end NUMINAMATH_CALUDE_carrot_distribution_l4047_404739


namespace NUMINAMATH_CALUDE_min_g_14_l4047_404735

-- Define a tenuous function
def Tenuous (f : ℕ+ → ℕ) : Prop :=
  ∀ x y : ℕ+, f x + f y > y^2

-- Define the sum of g from 1 to 20
def SumG (g : ℕ+ → ℕ) : ℕ :=
  (Finset.range 20).sum (fun i => g ⟨i + 1, Nat.succ_pos i⟩)

-- Theorem statement
theorem min_g_14 (g : ℕ+ → ℕ) (h_tenuous : Tenuous g) (h_min : ∀ g' : ℕ+ → ℕ, Tenuous g' → SumG g ≤ SumG g') :
  g ⟨14, by norm_num⟩ ≥ 136 := by
  sorry

end NUMINAMATH_CALUDE_min_g_14_l4047_404735


namespace NUMINAMATH_CALUDE_quadratic_solution_l4047_404721

theorem quadratic_solution (x : ℝ) : x^2 - 4*x + 3 = 0 → x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l4047_404721


namespace NUMINAMATH_CALUDE_fraction_simplification_l4047_404774

theorem fraction_simplification :
  (3/7 + 5/8) / (5/12 + 1/3) = 59/42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4047_404774


namespace NUMINAMATH_CALUDE_circle_radii_l4047_404701

theorem circle_radii (r R : ℝ) (hr : r > 0) (hR : R > 0) : 
  ∃ (circumscribed_radius inscribed_radius : ℝ),
    circumscribed_radius = Real.sqrt (r * R) ∧
    inscribed_radius = (Real.sqrt (r * R) * (Real.sqrt R + Real.sqrt r - Real.sqrt (R + r))) / Real.sqrt (R + r) := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_l4047_404701


namespace NUMINAMATH_CALUDE_soccer_team_wins_l4047_404795

theorem soccer_team_wins (total_matches : ℕ) (total_points : ℕ) (lost_matches : ℕ) 
  (h1 : total_matches = 10)
  (h2 : total_points = 17)
  (h3 : lost_matches = 3) :
  ∃ (won_matches : ℕ) (drawn_matches : ℕ),
    won_matches = 5 ∧
    drawn_matches = total_matches - won_matches - lost_matches ∧
    3 * won_matches + drawn_matches = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l4047_404795


namespace NUMINAMATH_CALUDE_unique_cameras_l4047_404766

/-- The number of cameras in either Sarah's or Mike's collection, but not both,
    given their shared and individual camera counts. -/
theorem unique_cameras (shared cameras_sarah cameras_mike_not_sarah : ℕ)
  (h1 : shared = 12)
  (h2 : cameras_sarah = 24)
  (h3 : cameras_mike_not_sarah = 9) :
  cameras_sarah - shared + cameras_mike_not_sarah = 21 :=
by sorry

end NUMINAMATH_CALUDE_unique_cameras_l4047_404766


namespace NUMINAMATH_CALUDE_two_wheeled_bikes_count_l4047_404723

/-- Represents the number of wheels on a bike -/
inductive BikeType
| TwoWheeled
| FourWheeled

/-- Calculates the number of two-wheeled bikes in the shop -/
def count_two_wheeled_bikes (total_wheels : ℕ) (four_wheeled_count : ℕ) : ℕ :=
  let remaining_wheels := total_wheels - (4 * four_wheeled_count)
  remaining_wheels / 2

/-- Theorem stating the number of two-wheeled bikes in the shop -/
theorem two_wheeled_bikes_count :
  count_two_wheeled_bikes 48 9 = 6 := by
  sorry


end NUMINAMATH_CALUDE_two_wheeled_bikes_count_l4047_404723


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l4047_404705

/-- Calculates the remaining money after a purchase with sales tax -/
def remaining_money (initial_amount purchase_amount tax_rate : ℚ) : ℚ :=
  initial_amount - (purchase_amount * (1 + tax_rate))

/-- Proves that given the specific conditions, the remaining money is $86.96 -/
theorem olivia_remaining_money :
  remaining_money 128 38 (8/100) = 86.96 := by
  sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l4047_404705


namespace NUMINAMATH_CALUDE_bus_passengers_l4047_404775

theorem bus_passengers (total_seats : ℕ) (net_increase : ℕ) (empty_seats : ℕ) :
  total_seats = 92 →
  net_increase = 19 →
  empty_seats = 57 →
  total_seats - empty_seats - net_increase = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l4047_404775


namespace NUMINAMATH_CALUDE_tank_filling_time_l4047_404716

/-- The time it takes for A, B, and C to fill the tank together -/
def combined_time : ℝ := 17.14285714285714

/-- The time it takes for B to fill the tank alone -/
def b_time : ℝ := 20

/-- The time it takes for C to empty the tank -/
def c_time : ℝ := 40

/-- The time it takes for A to fill the tank alone -/
def a_time : ℝ := 30

theorem tank_filling_time :
  (1 / a_time + 1 / b_time - 1 / c_time) = (1 / combined_time) := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l4047_404716


namespace NUMINAMATH_CALUDE_min_beacons_for_unique_determination_l4047_404730

/-- Represents a room in the maze -/
structure Room where
  x : ℕ
  y : ℕ

/-- Represents the maze structure -/
structure Maze where
  rooms : List Room
  corridors : List (Room × Room)

/-- Represents a beacon in the maze -/
structure Beacon where
  location : Room

/-- Calculate the distance between two rooms -/
def distance (maze : Maze) (r1 r2 : Room) : ℕ := sorry

/-- Check if a room's location can be uniquely determined -/
def isUniquelyDetermined (maze : Maze) (beacons : List Beacon) (room : Room) : Bool := sorry

/-- The main theorem: At least 3 beacons are needed for unique determination -/
theorem min_beacons_for_unique_determination (maze : Maze) :
  ∀ (beacons : List Beacon),
    (∀ (room : Room), room ∈ maze.rooms → isUniquelyDetermined maze beacons room) →
    beacons.length ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_beacons_for_unique_determination_l4047_404730


namespace NUMINAMATH_CALUDE_triangle_area_l4047_404758

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) :
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4047_404758


namespace NUMINAMATH_CALUDE_six_students_three_groups_arrangements_l4047_404753

/-- The number of ways to divide n students into k equal groups -/
def divide_into_groups (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k topics -/
def assign_topics (k : ℕ) : ℕ := sorry

/-- The total number of arrangements for n students divided into k equal groups 
    and assigned to k different topics -/
def total_arrangements (n k : ℕ) : ℕ :=
  divide_into_groups n k * assign_topics k

theorem six_students_three_groups_arrangements :
  total_arrangements 6 3 = 540 := by sorry

end NUMINAMATH_CALUDE_six_students_three_groups_arrangements_l4047_404753


namespace NUMINAMATH_CALUDE_not_in_third_quadrant_l4047_404750

def linear_function (x : ℝ) : ℝ := -x + 2

theorem not_in_third_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_not_in_third_quadrant_l4047_404750


namespace NUMINAMATH_CALUDE_field_ratio_l4047_404772

/-- Given a rectangular field with perimeter 240 meters and width 50 meters,
    prove that the ratio of length to width is 7:5 -/
theorem field_ratio (perimeter width length : ℝ) : 
  perimeter = 240 ∧ width = 50 ∧ perimeter = 2 * (length + width) →
  length / width = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l4047_404772


namespace NUMINAMATH_CALUDE_power_comparison_specific_power_comparison_l4047_404799

theorem power_comparison (n : ℕ) (hn : n > 0) :
  (n ≤ 2 → n^(n+1) < (n+1)^n) ∧
  (n ≥ 3 → n^(n+1) > (n+1)^n) :=
sorry

theorem specific_power_comparison :
  2008^2009 > 2009^2008 :=
sorry

end NUMINAMATH_CALUDE_power_comparison_specific_power_comparison_l4047_404799


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l4047_404784

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l4047_404784


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l4047_404731

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l4047_404731


namespace NUMINAMATH_CALUDE_probability_geometric_progression_ratio_two_l4047_404787

/-- A fair die with 6 faces -/
def FairDie : Type := Fin 6

/-- The outcome of rolling four fair dice -/
def FourDiceRoll : Type := FairDie × FairDie × FairDie × FairDie

/-- The total number of possible outcomes when rolling four fair dice -/
def totalOutcomes : ℕ := 6^4

/-- Checks if a list of four numbers forms a geometric progression with a common ratio of two -/
def isGeometricProgressionWithRatioTwo (roll : List ℕ) : Prop :=
  roll.length = 4 ∧ ∃ a : ℕ, roll = [a, 2*a, 4*a, 8*a]

/-- The number of favorable outcomes (rolls that can be arranged to form a geometric progression with ratio two) -/
def favorableOutcomes : ℕ := 36

/-- The probability of rolling four dice such that the numbers can be arranged 
    to form a geometric progression with a common ratio of two -/
theorem probability_geometric_progression_ratio_two :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 36 := by
  sorry


end NUMINAMATH_CALUDE_probability_geometric_progression_ratio_two_l4047_404787


namespace NUMINAMATH_CALUDE_sufficiency_of_P_for_Q_l4047_404789

theorem sufficiency_of_P_for_Q :
  ∀ x : ℝ, x ≥ 0 → 2 * x + 1 / (2 * x + 1) ≥ 1 ∧
  ¬(∀ x : ℝ, 2 * x + 1 / (2 * x + 1) ≥ 1 → x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_of_P_for_Q_l4047_404789


namespace NUMINAMATH_CALUDE_function_extension_theorem_l4047_404788

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem function_extension_theorem (f : ℝ → ℝ) 
  (h1 : is_even_function (fun x ↦ f (x + 2)))
  (h2 : ∀ x : ℝ, x ≥ 2 → f x = x^2 - 6*x + 4) :
  ∀ x : ℝ, x < 2 → f x = x^2 - 2*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_function_extension_theorem_l4047_404788


namespace NUMINAMATH_CALUDE_good_number_properties_l4047_404710

def is_good (n : ℕ) : Prop := n % 6 = 3

theorem good_number_properties :
  (∀ n : ℕ, is_good n ↔ n % 6 = 3) ∧
  (is_good 2001 ∧ ¬is_good 3001) ∧
  (∀ a b : ℕ, is_good a → is_good b → is_good (a * b)) ∧
  (∀ a b : ℕ, is_good (a * b) → is_good a ∨ is_good b) :=
by sorry

end NUMINAMATH_CALUDE_good_number_properties_l4047_404710


namespace NUMINAMATH_CALUDE_polygon_number_formula_l4047_404762

/-- N(n, k) represents the n-th k-sided polygon number -/
def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
  | 4 => n^2
  | 5 => (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
  | 6 => 2 * n^2 - n
  | _ => 0  -- placeholder for other k values

/-- The general formula for N(n, k) -/
def N_general (n k : ℕ) : ℚ :=
  ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n

theorem polygon_number_formula (n k : ℕ) (h1 : k ≥ 3) (h2 : n ≥ 1) :
  N n k = N_general n k :=
by sorry

end NUMINAMATH_CALUDE_polygon_number_formula_l4047_404762


namespace NUMINAMATH_CALUDE_golf_problem_l4047_404720

/-- Calculates how far beyond the hole a golf ball lands given the total distance to the hole,
    the distance of the first hit, and that the second hit travels half as far as the first. -/
def beyond_hole (total_distance first_hit : ℕ) : ℕ :=
  let second_hit := first_hit / 2
  let distance_after_first := total_distance - first_hit
  second_hit - distance_after_first

/-- Theorem stating that under the given conditions, the ball lands 20 yards beyond the hole. -/
theorem golf_problem : beyond_hole 250 180 = 20 := by
  sorry

end NUMINAMATH_CALUDE_golf_problem_l4047_404720


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l4047_404798

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l4047_404798


namespace NUMINAMATH_CALUDE_value_of_expression_l4047_404709

theorem value_of_expression (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l4047_404709


namespace NUMINAMATH_CALUDE_derivative_sin_2x_at_pi_3_l4047_404732

theorem derivative_sin_2x_at_pi_3 :
  let f : ℝ → ℝ := fun x ↦ Real.sin (2 * x)
  (deriv f) (π / 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_at_pi_3_l4047_404732


namespace NUMINAMATH_CALUDE_journey_time_relation_l4047_404736

theorem journey_time_relation (x : ℝ) (h : x > 1.8) : 
  (202 / x) * 1.6 = 202 / (x - 1.8) :=
by
  sorry

#check journey_time_relation

end NUMINAMATH_CALUDE_journey_time_relation_l4047_404736


namespace NUMINAMATH_CALUDE_triangle_properties_l4047_404718

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a = b * (Real.cos C) + b →
  (Real.sin C = Real.tan B) ∧
  (a = 1 ∧ C < π/2 → 1/2 < c ∧ c < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l4047_404718


namespace NUMINAMATH_CALUDE_negative_three_squared_minus_negative_two_cubed_l4047_404765

theorem negative_three_squared_minus_negative_two_cubed : (-3)^2 - (-2)^3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_minus_negative_two_cubed_l4047_404765


namespace NUMINAMATH_CALUDE_ladybugs_per_leaf_l4047_404713

theorem ladybugs_per_leaf (total_leaves : ℕ) (total_ladybugs : ℕ) (ladybugs_per_leaf : ℕ) : 
  total_leaves = 84 → 
  total_ladybugs = 11676 → 
  total_ladybugs = total_leaves * ladybugs_per_leaf → 
  ladybugs_per_leaf = 139 := by
sorry

end NUMINAMATH_CALUDE_ladybugs_per_leaf_l4047_404713


namespace NUMINAMATH_CALUDE_intersection_of_M_and_P_l4047_404726

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def P : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_P : M ∩ P = {(3, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_P_l4047_404726


namespace NUMINAMATH_CALUDE_children_on_bus_l4047_404759

theorem children_on_bus (total : ℕ) (men : ℕ) (women : ℕ) 
  (h1 : total = 54)
  (h2 : men = 18)
  (h3 : women = 26) :
  total - men - women = 10 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_l4047_404759


namespace NUMINAMATH_CALUDE_digit_square_problem_l4047_404777

theorem digit_square_problem (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  a ≥ 1 → a ≤ 9 →
  c ≥ 1 → c ≤ 9 →
  (10 * a + b)^2 = 100 * c + 10 * c + b →
  100 * c + 10 * c + b > 300 →
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_digit_square_problem_l4047_404777


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l4047_404743

theorem divisibility_by_seven (n : ℕ) : 
  (3^(12*n^2 + 1) + 2^(6*n + 2)) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l4047_404743


namespace NUMINAMATH_CALUDE_conic_properties_l4047_404749

-- Define the conic section
def conic (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2/n = 1}

-- Define the foci for the conic
def foci (n : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the line with slope √3 passing through the left focus
def line_through_focus (n : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the intersection points A and B
def intersection_points (n : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the perimeter of triangle ABF₂
def perimeter_ABF2 (n : ℝ) : ℝ := sorry

-- Define the product PF₁ · PF₂
def focus_product (n : ℝ) (P : ℝ × ℝ) : ℝ := sorry

theorem conic_properties :
  (perimeter_ABF2 (-1) = 12) ∧
  (∀ P ∈ conic 4, focus_product 4 P ≤ 4) ∧
  (∃ P ∈ conic 4, focus_product 4 P = 4) ∧
  (∀ P ∈ conic 4, focus_product 4 P ≥ 1) ∧
  (∃ P ∈ conic 4, focus_product 4 P = 1) := by sorry

end NUMINAMATH_CALUDE_conic_properties_l4047_404749
