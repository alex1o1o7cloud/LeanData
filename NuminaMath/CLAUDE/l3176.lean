import Mathlib

namespace NUMINAMATH_CALUDE_triangle_top_angle_l3176_317639

theorem triangle_top_angle (total : ℝ) (right : ℝ) (left : ℝ) (top : ℝ) : 
  total = 250 →
  right = 60 →
  left = 2 * right →
  total = left + right + top →
  top = 70 := by
sorry

end NUMINAMATH_CALUDE_triangle_top_angle_l3176_317639


namespace NUMINAMATH_CALUDE_prob_random_twin_prob_twins_in_three_expected_twin_pairs_l3176_317678

/-- Represents the probability model for twins in Schwambrania -/
structure TwinProbability where
  /-- The probability of twins being born -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p_bounds : 0 ≤ p ∧ p ≤ 1
  /-- Assumption that triplets do not exist -/
  h_no_triplets : True

/-- Theorem for the probability of a random person being a twin -/
theorem prob_random_twin (model : TwinProbability) :
  (2 * model.p) / (model.p + 1) = Real.exp (Real.log (2 * model.p) - Real.log (model.p + 1)) :=
sorry

/-- Theorem for the probability of having at least one pair of twins in a family with three children -/
theorem prob_twins_in_three (model : TwinProbability) :
  (2 * model.p) / (2 * model.p + (1 - model.p)^2) =
  Real.exp (Real.log (2 * model.p) - Real.log (2 * model.p + (1 - model.p)^2)) :=
sorry

/-- Theorem for the expected number of twin pairs among N first-graders -/
theorem expected_twin_pairs (model : TwinProbability) (N : ℕ) :
  (N : ℝ) * model.p / (model.p + 1) =
  Real.exp (Real.log N + Real.log model.p - Real.log (model.p + 1)) :=
sorry

end NUMINAMATH_CALUDE_prob_random_twin_prob_twins_in_three_expected_twin_pairs_l3176_317678


namespace NUMINAMATH_CALUDE_distance_between_externally_tangent_circles_l3176_317623

/-- The distance between centers of two externally tangent circles is the sum of their radii -/
theorem distance_between_externally_tangent_circles 
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 8) 
  (h₃ : d = r₁ + r₂) : 
  d = 11 := by sorry

end NUMINAMATH_CALUDE_distance_between_externally_tangent_circles_l3176_317623


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l3176_317672

theorem dvd_book_capacity (total_capacity : ℕ) (current_count : ℕ) (h1 : total_capacity = 126) (h2 : current_count = 81) :
  total_capacity - current_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_capacity_l3176_317672


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l3176_317655

theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l3176_317655


namespace NUMINAMATH_CALUDE_skips_per_meter_correct_l3176_317677

/-- Represents the number of skips in one meter given the following conditions:
    * x hops equals y skips
    * z jumps equals w hops
    * u jumps equals v meters
-/
def skips_per_meter (x y z w u v : ℚ) : ℚ :=
  u * y * w / (v * x * z)

/-- Theorem stating that under the given conditions, 
    1 meter equals (uyw / (vxz)) skips -/
theorem skips_per_meter_correct
  (x y z w u v : ℚ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hu : u > 0) (hv : v > 0)
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : z * 1 = w)
  (jumps_to_meters : u * 1 = v) :
  skips_per_meter x y z w u v = u * y * w / (v * x * z) :=
by sorry

end NUMINAMATH_CALUDE_skips_per_meter_correct_l3176_317677


namespace NUMINAMATH_CALUDE_soccer_school_admission_probability_l3176_317663

/-- Represents the probability of being admitted to the soccer school -/
def admission_probability (p_assistant : ℝ) (p_head : ℝ) : ℝ :=
  p_assistant * p_assistant + 2 * p_assistant * (1 - p_assistant) * p_head

/-- The probability of the young soccer enthusiast being admitted to the well-known soccer school is 0.4 -/
theorem soccer_school_admission_probability : 
  admission_probability 0.5 0.3 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_soccer_school_admission_probability_l3176_317663


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3176_317668

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 - I) : 2 / z = 1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3176_317668


namespace NUMINAMATH_CALUDE_family_weight_gain_l3176_317695

/-- The weight gained by Orlando, in pounds -/
def orlando_weight : ℕ := 5

/-- The weight gained by Jose, in pounds -/
def jose_weight : ℕ := 2 * orlando_weight + 2

/-- The weight gained by Fernando, in pounds -/
def fernando_weight : ℕ := jose_weight / 2 - 3

/-- The total weight gained by the three family members, in pounds -/
def total_weight : ℕ := orlando_weight + jose_weight + fernando_weight

theorem family_weight_gain : total_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_family_weight_gain_l3176_317695


namespace NUMINAMATH_CALUDE_problem_statement_l3176_317600

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3176_317600


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3176_317621

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_of_M_in_U : 
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3176_317621


namespace NUMINAMATH_CALUDE_julia_played_with_34_kids_l3176_317609

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 17

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := monday_kids + tuesday_kids + wednesday_kids

theorem julia_played_with_34_kids : total_kids = 34 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_34_kids_l3176_317609


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l3176_317633

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l3176_317633


namespace NUMINAMATH_CALUDE_amy_work_schedule_l3176_317649

/-- Amy's work schedule and earnings problem -/
theorem amy_work_schedule (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
  (school_weeks : ℕ) (school_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 40 →
  summer_earnings = 4800 →
  school_weeks = 36 →
  school_earnings = 7200 →
  (school_earnings / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_weeks = 20 := by
  sorry

#check amy_work_schedule

end NUMINAMATH_CALUDE_amy_work_schedule_l3176_317649


namespace NUMINAMATH_CALUDE_sin_neg_ten_thirds_pi_l3176_317687

theorem sin_neg_ten_thirds_pi : Real.sin (-10/3 * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_ten_thirds_pi_l3176_317687


namespace NUMINAMATH_CALUDE_bread_loaves_from_flour_l3176_317616

/-- Given 5 cups of flour and requiring 2.5 cups per loaf, prove that 2 loaves can be baked. -/
theorem bread_loaves_from_flour (total_flour : ℝ) (flour_per_loaf : ℝ) (h1 : total_flour = 5) (h2 : flour_per_loaf = 2.5) :
  total_flour / flour_per_loaf = 2 := by
sorry

end NUMINAMATH_CALUDE_bread_loaves_from_flour_l3176_317616


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3176_317624

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (A = 1/3 ∧ B = 2/3 ∧ C = 1/3) ∧
    (∀ x : ℚ, x ≠ -2 ∧ x^2 + x + 1 ≠ 0 →
      (x + 1)^2 / ((x + 2) * (x^2 + x + 1)) =
      A / (x + 2) + (B * x + C) / (x^2 + x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3176_317624


namespace NUMINAMATH_CALUDE_pomelos_last_week_l3176_317679

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 10

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Theorem stating that the number of pomelos shipped last week is 240 -/
theorem pomelos_last_week :
  (total_dozens * dozen) / (boxes_last_week + boxes_this_week) * boxes_last_week = 240 := by
  sorry


end NUMINAMATH_CALUDE_pomelos_last_week_l3176_317679


namespace NUMINAMATH_CALUDE_quadratic_function_existence_l3176_317613

theorem quadratic_function_existence : ∃ (a b c : ℝ), 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) ∧
  |a * 2^2 + b * 2 + c| ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_existence_l3176_317613


namespace NUMINAMATH_CALUDE_boys_insects_count_l3176_317620

/-- The number of groups in the class -/
def num_groups : ℕ := 4

/-- The number of insects each group receives -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by the girls -/
def girls_insects : ℕ := 300

/-- The number of insects collected by the boys -/
def boys_insects : ℕ := num_groups * insects_per_group - girls_insects

theorem boys_insects_count :
  boys_insects = 200 := by sorry

end NUMINAMATH_CALUDE_boys_insects_count_l3176_317620


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3176_317627

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- 
  Theorem: The radius of a cylinder inscribed in a cone
  Given:
  - A right circular cone with diameter 8 and altitude 10
  - A right circular cylinder inscribed in the cone
  - The axes of the cylinder and cone coincide
  - The height of the cylinder is three times its radius
  Prove: The radius of the cylinder is 20/11
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  cyl.height = 3 * cyl.radius →
  cyl.radius = 20 / 11 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3176_317627


namespace NUMINAMATH_CALUDE_total_area_is_135_l3176_317619

/-- Represents the geometry of villages, roads, fields, and forest --/
structure VillageGeometry where
  /-- Side length of the square field --/
  r : ℝ
  /-- Side length of the rectangular field along the road --/
  p : ℝ
  /-- Side length of the rectangular forest along the road --/
  q : ℝ

/-- The total area of the forest and fields is 135 sq km --/
theorem total_area_is_135 (g : VillageGeometry) : 
  g.r^2 + 4 * g.p^2 + 12 * g.q = 135 :=
by
  sorry

/-- The forest area is 45 sq km more than the sum of field areas --/
axiom forest_area_relation (g : VillageGeometry) : 
  12 * g.q = g.r^2 + 4 * g.p^2 + 45

/-- The side of the rectangular field perpendicular to the road is 4 times longer --/
axiom rectangular_field_proportion (g : VillageGeometry) : 
  4 * g.p = g.q

/-- The side of the rectangular forest perpendicular to the road is 12 km --/
axiom forest_width (g : VillageGeometry) : g.q = 12

end NUMINAMATH_CALUDE_total_area_is_135_l3176_317619


namespace NUMINAMATH_CALUDE_complex_magnitude_l3176_317629

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3176_317629


namespace NUMINAMATH_CALUDE_solve_equation_l3176_317650

/-- Custom remainder operation Θ -/
def theta (m n : ℕ) : ℕ :=
  if m ≥ n then m % n else n % m

/-- Main theorem -/
theorem solve_equation :
  ∃ (A : ℕ), 0 < A ∧ A < 40 ∧ theta 20 (theta A 20) = 7 ∧ A = 33 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l3176_317650


namespace NUMINAMATH_CALUDE_sin_cos_identity_sin_tan_simplification_l3176_317669

-- Question 1
theorem sin_cos_identity :
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.sin (56 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by sorry

-- Question 2
theorem sin_tan_simplification :
  Real.sin (50 * π / 180) * (Real.sqrt 3 * Real.tan (10 * π / 180) + 1) = 
  Real.cos (20 * π / 180) / Real.cos (10 * π / 180) := by sorry

end NUMINAMATH_CALUDE_sin_cos_identity_sin_tan_simplification_l3176_317669


namespace NUMINAMATH_CALUDE_inverse_f_at_407_l3176_317618

noncomputable def f (x : ℝ) : ℝ := 5 * x^4 + 2

theorem inverse_f_at_407 : Function.invFun f 407 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_407_l3176_317618


namespace NUMINAMATH_CALUDE_binomial_coeff_n_n_l3176_317625

theorem binomial_coeff_n_n (n : ℕ) : (n.choose n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coeff_n_n_l3176_317625


namespace NUMINAMATH_CALUDE_jasons_games_this_month_l3176_317681

/-- 
Given that:
- Jason went to 17 games last month
- Jason plans to go to 16 games next month
- Jason will attend 44 games in all

Prove that Jason went to 11 games this month.
-/
theorem jasons_games_this_month 
  (games_last_month : ℕ) 
  (games_next_month : ℕ) 
  (total_games : ℕ) 
  (h1 : games_last_month = 17)
  (h2 : games_next_month = 16)
  (h3 : total_games = 44) :
  total_games - (games_last_month + games_next_month) = 11 := by
  sorry

end NUMINAMATH_CALUDE_jasons_games_this_month_l3176_317681


namespace NUMINAMATH_CALUDE_meet_once_l3176_317608

/-- Represents the movement of Hannah and the van --/
structure Movement where
  hannah_speed : ℝ
  van_speed : ℝ
  pail_distance : ℝ
  van_stop_time : ℝ

/-- Calculates the number of meetings between Hannah and the van --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- Theorem stating that Hannah and the van meet exactly once --/
theorem meet_once (m : Movement) 
  (h1 : m.hannah_speed = 6)
  (h2 : m.van_speed = 12)
  (h3 : m.pail_distance = 150)
  (h4 : m.van_stop_time = 45)
  : number_of_meetings m = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l3176_317608


namespace NUMINAMATH_CALUDE_part_one_part_two_l3176_317640

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : ¬¬p m) : m > 2 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ≥ 3 ∨ (1 < m ∧ m ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3176_317640


namespace NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l3176_317638

theorem triangle_area_with_given_base_and_height :
  ∀ (base height : ℝ), 
    base = 12 →
    height = 15 →
    (1 / 2 : ℝ) * base * height = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_base_and_height_l3176_317638


namespace NUMINAMATH_CALUDE_root_shift_cubic_l3176_317659

/-- Given a cubic polynomial with roots p, q, and r, 
    find the monic polynomial with roots p + 3, q + 3, and r + 3 -/
theorem root_shift_cubic (p q r : ℂ) : 
  (p^3 - 4*p^2 + 9*p - 7 = 0) ∧ 
  (q^3 - 4*q^2 + 9*q - 7 = 0) ∧ 
  (r^3 - 4*r^2 + 9*r - 7 = 0) → 
  ∃ (a b c : ℂ), 
    (∀ x : ℂ, x^3 - 13*x^2 + 60*x - 90 = (x - (p + 3)) * (x - (q + 3)) * (x - (r + 3))) :=
by sorry

end NUMINAMATH_CALUDE_root_shift_cubic_l3176_317659


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3176_317631

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : C * (60 / 360) = D * (40 / 360)) :
  (C^2 / D^2 : ℝ) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3176_317631


namespace NUMINAMATH_CALUDE_point_b_coordinates_l3176_317660

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetricPoint (p q : Point3D) : Point3D :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y, 2 * q.z - p.z⟩

def vector (p q : Point3D) : Point3D :=
  ⟨q.x - p.x, q.y - p.y, q.z - p.z⟩

theorem point_b_coordinates
  (A : Point3D)
  (P : Point3D)
  (A_prime : Point3D)
  (B_prime : Point3D)
  (h1 : A = ⟨-1, 3, -3⟩)
  (h2 : P = ⟨1, 2, 3⟩)
  (h3 : A_prime = symmetricPoint A P)
  (h4 : vector A_prime B_prime = ⟨3, 1, 5⟩)
  : ∃ B : Point3D, (B = ⟨-4, 2, -8⟩ ∧ symmetricPoint B P = B_prime) :=
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l3176_317660


namespace NUMINAMATH_CALUDE_negation_equivalence_l3176_317615

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3176_317615


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3176_317636

/-- A pyramid with a square base and equilateral triangular lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (edge_length : ℝ)
  (touches_base_center : True)
  (touches_apex : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.edge_length ^ 3

/-- The main theorem: volume of the inscribed cube in the given pyramid -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p)
  (h_base : p.base_side = 2) :
  cube_volume p c = 2 * Real.sqrt 6 / 9 := by
  sorry

#check inscribed_cube_volume

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3176_317636


namespace NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l3176_317676

/-- The polynomial P(x) = 8x³ - 4x² - 42x + 45 -/
def P (x : ℝ) : ℝ := 8 * x^3 - 4 * x^2 - 42 * x + 45

/-- The derivative of P(x) -/
def P' (x : ℝ) : ℝ := 24 * x^2 - 8 * x - 42

theorem polynomial_divisibility_theorem :
  ∃ (r : ℝ), (∀ (x : ℝ), (x - r)^2 ∣ P x) ∧ (abs (r - 1.52) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_theorem_l3176_317676


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficients_l3176_317635

theorem quadratic_rational_root_even_coefficients 
  (a b c : ℤ) (x : ℚ) : 
  (a * x^2 + b * x + c = 0) → (Even a ∧ Even b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficients_l3176_317635


namespace NUMINAMATH_CALUDE_range_of_m_l3176_317684

-- Define the conditions
def p (x : ℝ) : Prop := (x + 2) * (x - 10) ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  (0 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3176_317684


namespace NUMINAMATH_CALUDE_inverse_of_A_l3176_317666

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3176_317666


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3176_317682

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 148 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l3176_317682


namespace NUMINAMATH_CALUDE_greg_is_sixteen_l3176_317622

-- Define the ages of the siblings
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem to prove Greg's age
theorem greg_is_sixteen : greg_age = 16 := by
  sorry


end NUMINAMATH_CALUDE_greg_is_sixteen_l3176_317622


namespace NUMINAMATH_CALUDE_consecutive_even_integers_product_l3176_317630

theorem consecutive_even_integers_product (a b c d : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- positive integers
  Even a ∧ Even b ∧ Even c ∧ Even d →  -- even integers
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2 →  -- consecutive
  a * b * c * d = 5040 →  -- product is 5040
  d = 20 :=  -- largest integer is 20
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_product_l3176_317630


namespace NUMINAMATH_CALUDE_gcd_lcm_240_360_l3176_317652

theorem gcd_lcm_240_360 : 
  (Nat.gcd 240 360 = 120) ∧ (Nat.lcm 240 360 = 720) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_240_360_l3176_317652


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3176_317605

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x + 2| - |a * x|

-- Part 1
theorem solution_set_part1 : 
  {x : ℝ | f 2 x > 2} = {x : ℝ | x > -1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Ioo (-1) 1, f a x > x + 1) ↔ a ∈ Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3176_317605


namespace NUMINAMATH_CALUDE_washer_dryer_cost_washer_dryer_cost_proof_l3176_317657

/-- The total cost of a washer-dryer combination is 1200 dollars, given that the washer costs 710 dollars and is 220 dollars more expensive than the dryer. -/
theorem washer_dryer_cost : ℕ → ℕ → ℕ → Prop :=
  fun washer_cost dryer_cost total_cost =>
    washer_cost = 710 ∧
    washer_cost = dryer_cost + 220 ∧
    total_cost = washer_cost + dryer_cost →
    total_cost = 1200

/-- Proof of the washer-dryer cost theorem -/
theorem washer_dryer_cost_proof : washer_dryer_cost 710 490 1200 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_washer_dryer_cost_proof_l3176_317657


namespace NUMINAMATH_CALUDE_sum_reciprocal_f_equals_251_385_l3176_317644

/-- The function f(n) that returns the integer closest to the cube root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) from k=1 to 2023 -/
def sum_reciprocal_f : ℚ :=
  (Finset.range 2023).sum (λ k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of 1/f(k) from k=1 to 2023 is equal to 251.385 -/
theorem sum_reciprocal_f_equals_251_385 : sum_reciprocal_f = 251385 / 1000 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_f_equals_251_385_l3176_317644


namespace NUMINAMATH_CALUDE_zero_of_f_l3176_317673

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- Theorem: -1 is the zero of the function f(x) = (x+1)^2 -/
theorem zero_of_f : f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l3176_317673


namespace NUMINAMATH_CALUDE_five_people_arrangement_l3176_317697

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two specific people next to each other in a row of n people. -/
def adjacentPairArrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange 5 people in a row with two specific people next to each other. -/
theorem five_people_arrangement : 
  adjacentPairArrangements 5 * arrangements 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l3176_317697


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l3176_317685

theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 9) 
  (h2 : graded_worksheets = 5) 
  (h3 : problems_per_worksheet = 4) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l3176_317685


namespace NUMINAMATH_CALUDE_reflection_result_l3176_317692

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 2 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 2, p.1 - 2)

/-- The final position of point C after two reflections -/
def C_double_prime : ℝ × ℝ :=
  reflect_line (reflect_y_axis (5, 3))

theorem reflection_result :
  C_double_prime = (5, -7) :=
by sorry

end NUMINAMATH_CALUDE_reflection_result_l3176_317692


namespace NUMINAMATH_CALUDE_division_problem_l3176_317612

theorem division_problem (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 62976 → quotient = 123 → divisor = 512 → 
  dividend = divisor * quotient ∧ dividend = 62976 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3176_317612


namespace NUMINAMATH_CALUDE_exponent_division_l3176_317675

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3176_317675


namespace NUMINAMATH_CALUDE_additional_songs_count_l3176_317662

def original_songs : ℕ := 25
def song_duration : ℕ := 3
def total_duration : ℕ := 105

theorem additional_songs_count :
  (total_duration - original_songs * song_duration) / song_duration = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_songs_count_l3176_317662


namespace NUMINAMATH_CALUDE_deck_size_l3176_317665

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end NUMINAMATH_CALUDE_deck_size_l3176_317665


namespace NUMINAMATH_CALUDE_number_problem_l3176_317690

theorem number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 6) (h4 : x / y = 6) :
  x * y - (x - y) = 6 / 49 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3176_317690


namespace NUMINAMATH_CALUDE_specific_pyramid_perimeter_l3176_317634

/-- A square pyramid with specific dimensions and properties -/
structure SquarePyramid where
  base_edge : ℝ
  lateral_edge : ℝ
  front_view_isosceles : Prop
  side_view_isosceles : Prop
  views_congruent : Prop

/-- The perimeter of the front view of a square pyramid -/
def front_view_perimeter (p : SquarePyramid) : ℝ := sorry

/-- Theorem stating the perimeter of the front view for a specific square pyramid -/
theorem specific_pyramid_perimeter :
  ∀ (p : SquarePyramid),
    p.base_edge = 2 ∧
    p.lateral_edge = Real.sqrt 3 ∧
    p.front_view_isosceles ∧
    p.side_view_isosceles ∧
    p.views_congruent →
    front_view_perimeter p = 2 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_perimeter_l3176_317634


namespace NUMINAMATH_CALUDE_cauliflower_sales_value_l3176_317637

def farmers_market_sales (total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ) : Prop :=
  total_earnings = 500 ∧
  broccoli_sales = 57 ∧
  carrot_sales = 2 * broccoli_sales ∧
  spinach_sales = (carrot_sales / 2) + 16 ∧
  tomato_sales = broccoli_sales + spinach_sales ∧
  total_earnings = broccoli_sales + carrot_sales + spinach_sales + tomato_sales + cauliflower_sales

theorem cauliflower_sales_value :
  ∀ total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales : ℝ,
  farmers_market_sales total_earnings broccoli_sales carrot_sales spinach_sales tomato_sales cauliflower_sales →
  cauliflower_sales = 126 := by
sorry

end NUMINAMATH_CALUDE_cauliflower_sales_value_l3176_317637


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3176_317661

theorem complex_fraction_sum : (2 + 2 * Complex.I) / Complex.I + (1 + Complex.I) / (1 - Complex.I) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3176_317661


namespace NUMINAMATH_CALUDE_min_value_expression_l3176_317626

theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3176_317626


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l3176_317689

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l3176_317689


namespace NUMINAMATH_CALUDE_function_bound_l3176_317610

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, |x₁ - x₂| ≤ 1 → |f x₂ - f x₁| ≤ 1) ∧ f 0 = 1

/-- The main theorem -/
theorem function_bound (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x : ℝ, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l3176_317610


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3176_317642

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 →
    1 / (x^3 + 2*x^2 - 19*x - 30) = A / (x + 3) + B / (x - 2) + C / ((x - 2)^2)) →
  A = 1/25 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_constant_l3176_317642


namespace NUMINAMATH_CALUDE_sequence_properties_l3176_317656

/-- Definition of the sequence and its partial sum -/
def sequence_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (2 * S n) / n + n = 2 * a n + 1

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

/-- Definition of geometric sequence for three terms -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- Main theorem -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) :
  sequence_condition a S →
  (is_arithmetic_sequence a ∧
   (is_geometric_sequence (a 4) (a 7) (a 9) →
    ∃ n : ℕ, S n = -78 ∧ ∀ m : ℕ, S m ≥ -78)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3176_317656


namespace NUMINAMATH_CALUDE_continuous_functions_integral_bound_l3176_317658

open Set
open MeasureTheory
open Interval

theorem continuous_functions_integral_bound 
  (f g : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g)
  (hf_integral : ∫ x in (Icc 0 1), (f x)^2 = 1)
  (hg_integral : ∫ x in (Icc 0 1), (g x)^2 = 1) :
  ∃ c ∈ Icc 0 1, f c + g c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_continuous_functions_integral_bound_l3176_317658


namespace NUMINAMATH_CALUDE_simplify_expression_l3176_317699

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  ((x + y)^2 - (x + y)*(x - y)) / (2*y) = y + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3176_317699


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3176_317632

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3176_317632


namespace NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l3176_317674

theorem two_fifths_of_n_is_80 (n : ℚ) (h : n = 5 / 6 * 240) : 2 / 5 * n = 80 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_of_n_is_80_l3176_317674


namespace NUMINAMATH_CALUDE_simplify_fraction_l3176_317628

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3176_317628


namespace NUMINAMATH_CALUDE_inscribed_circle_arithmetic_progression_l3176_317667

theorem inscribed_circle_arithmetic_progression (a b c r : ℝ) :
  (0 < r) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  (∃ d : ℝ, d > 0 ∧ a = 2*r + d ∧ b = 2*r + 2*d ∧ c = 2*r + 3*d) →
  (∃ k : ℝ, k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_arithmetic_progression_l3176_317667


namespace NUMINAMATH_CALUDE_square_difference_sum_l3176_317694

theorem square_difference_sum : 
  20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l3176_317694


namespace NUMINAMATH_CALUDE_layla_point_difference_l3176_317671

theorem layla_point_difference (total_points layla_points : ℕ) 
  (h1 : total_points = 345) 
  (h2 : layla_points = 180) : 
  layla_points - (total_points - layla_points) = 15 := by
  sorry

end NUMINAMATH_CALUDE_layla_point_difference_l3176_317671


namespace NUMINAMATH_CALUDE_baseball_glove_discount_percentage_l3176_317680

def baseball_cards_price : ℝ := 25
def baseball_bat_price : ℝ := 10
def baseball_glove_original_price : ℝ := 30
def baseball_cleats_price : ℝ := 10
def baseball_cleats_pairs : ℕ := 2
def total_amount : ℝ := 79

theorem baseball_glove_discount_percentage :
  let other_items_total : ℝ := baseball_cards_price + baseball_bat_price + baseball_cleats_price * baseball_cleats_pairs
  let glove_sale_price : ℝ := total_amount - other_items_total
  let discount_percentage : ℝ := (baseball_glove_original_price - glove_sale_price) / baseball_glove_original_price * 100
  discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_baseball_glove_discount_percentage_l3176_317680


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l3176_317683

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x => 6 * x^2 - 31 * x + 35
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l3176_317683


namespace NUMINAMATH_CALUDE_power_quotient_nineteen_l3176_317645

theorem power_quotient_nineteen : 19^11 / 19^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_quotient_nineteen_l3176_317645


namespace NUMINAMATH_CALUDE_pipe_ratio_proof_l3176_317686

theorem pipe_ratio_proof (total_length longer_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : longer_length = 118)
  (h3 : ∃ k : ℕ, k * (total_length - longer_length) = longer_length) :
  longer_length / (total_length - longer_length) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pipe_ratio_proof_l3176_317686


namespace NUMINAMATH_CALUDE_emma_henry_weight_l3176_317643

theorem emma_henry_weight (e f g h : ℝ) 
  (ef_sum : e + f = 310)
  (fg_sum : f + g = 265)
  (gh_sum : g + h = 280) :
  e + h = 325 := by
sorry

end NUMINAMATH_CALUDE_emma_henry_weight_l3176_317643


namespace NUMINAMATH_CALUDE_top_price_calculation_l3176_317603

def shorts_price : ℝ := 7
def shoes_price : ℝ := 10
def hats_price : ℝ := 6
def socks_price : ℝ := 2

def shorts_quantity : ℕ := 5
def shoes_quantity : ℕ := 2
def hats_quantity : ℕ := 3
def socks_quantity : ℕ := 6
def tops_quantity : ℕ := 4

def total_spent : ℝ := 102

theorem top_price_calculation :
  let other_items_cost := shorts_price * shorts_quantity + shoes_price * shoes_quantity +
                          hats_price * hats_quantity + socks_price * socks_quantity
  let tops_total_cost := total_spent - other_items_cost
  tops_total_cost / tops_quantity = 4.25 := by sorry

end NUMINAMATH_CALUDE_top_price_calculation_l3176_317603


namespace NUMINAMATH_CALUDE_desk_height_calculation_l3176_317641

/-- Given three identical blocks of wood and a desk, if the height of the desk plus twice the length
    of a block equals 50 inches, and the height of the desk plus twice the width of a block equals
    40 inches, then the height of the desk is 30 inches. -/
theorem desk_height_calculation (h l w : ℝ) : 
  h + 2 * l = 50 → h + 2 * w = 40 → h = 30 :=
by sorry

end NUMINAMATH_CALUDE_desk_height_calculation_l3176_317641


namespace NUMINAMATH_CALUDE_tea_mixture_price_l3176_317664

/-- Given three types of tea mixed in a specific ratio, calculate the price of the mixture per kg -/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℕ) : 
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3 : ℚ) = 153 := by
sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l3176_317664


namespace NUMINAMATH_CALUDE_distance_between_points_l3176_317614

/-- The distance between points (3,4) and (8,17) is √194 -/
theorem distance_between_points : Real.sqrt ((8 - 3)^2 + (17 - 4)^2) = Real.sqrt 194 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3176_317614


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l3176_317604

theorem max_value_sine_cosine_sum :
  let f : ℝ → ℝ := λ x ↦ 6 * Real.sin x + 8 * Real.cos x
  ∃ M : ℝ, M = 10 ∧ ∀ x : ℝ, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_sum_l3176_317604


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3176_317691

theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Same perimeter condition
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3176_317691


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3176_317693

theorem solve_exponential_equation (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28 → n = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3176_317693


namespace NUMINAMATH_CALUDE_sphere_volume_surface_ratio_l3176_317653

/-- The ratio of volume to surface area of a sphere with an inscribed regular hexagon -/
theorem sphere_volume_surface_ratio (area_hexagon : ℝ) (distance : ℝ) :
  area_hexagon = 3 * Real.sqrt 3 / 2 →
  distance = 2 * Real.sqrt 2 →
  ∃ (V S : ℝ), V / S = 1 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_ratio_l3176_317653


namespace NUMINAMATH_CALUDE_paper_used_l3176_317654

theorem paper_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
  (h1 : initial = 900) 
  (h2 : remaining = 744) 
  (h3 : used = initial - remaining) : used = 156 := by
  sorry

end NUMINAMATH_CALUDE_paper_used_l3176_317654


namespace NUMINAMATH_CALUDE_chloe_winter_clothing_l3176_317698

/-- Calculates the total number of winter clothing items given the number of boxes and items per box. -/
def total_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : ℕ :=
  num_boxes * (scarves_per_box + mittens_per_box)

/-- Proves that Chloe has 32 pieces of winter clothing given the problem conditions. -/
theorem chloe_winter_clothing : 
  total_winter_clothing 4 2 6 = 32 := by
  sorry

#eval total_winter_clothing 4 2 6

end NUMINAMATH_CALUDE_chloe_winter_clothing_l3176_317698


namespace NUMINAMATH_CALUDE_max_sum_squares_triangle_sides_l3176_317647

theorem max_sum_squares_triangle_sides (a : ℝ) (α : ℝ) 
  (h_a_pos : a > 0) (h_α_acute : 0 < α ∧ α < π / 2) :
  ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ 
    b^2 + c^2 = a^2 / (1 - Real.cos α) ∧
    ∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
      b'^2 + a^2 = c'^2 + 2 * a * b' * Real.cos α →
      b'^2 + c'^2 ≤ a^2 / (1 - Real.cos α) := by
sorry


end NUMINAMATH_CALUDE_max_sum_squares_triangle_sides_l3176_317647


namespace NUMINAMATH_CALUDE_equation_solution_l3176_317696

theorem equation_solution : ∃ x : ℚ, (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 ∧ x = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3176_317696


namespace NUMINAMATH_CALUDE_cosine_sine_equality_l3176_317601

theorem cosine_sine_equality (α : ℝ) : 
  3.3998 * (Real.cos α)^4 - 4 * (Real.cos α)^3 - 8 * (Real.cos α)^2 + 3 * Real.cos α + 1 = 
  -2 * Real.sin (7 * α / 2) * Real.sin (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_equality_l3176_317601


namespace NUMINAMATH_CALUDE_b2_properties_b2_b4_equality_a_and_x_relation_l3176_317611

theorem b2_properties (B₂ : ℝ) (A : ℝ) (x : ℝ) : 
  B₂ = B₂^2 - 2 →
  (B₂ = -1 ∨ B₂ = 2) ∧
  (B₂ = -1 → (A = 1 ∨ A = -1) ∧ ¬(∃ x, x + 1/x = 1)) ∧
  (B₂ = 2 → (A = 2 ∨ A = -2) ∧ (x = 1 ∨ x = -1)) :=
by sorry

theorem b2_b4_equality (B₂ B₄ : ℝ) :
  B₂ = B₄ → B₂ = B₂^2 - 2 :=
by sorry

theorem a_and_x_relation (A x : ℝ) :
  A = x + 1/x →
  (A = 2 → x = 1) ∧
  (A = -2 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_b2_properties_b2_b4_equality_a_and_x_relation_l3176_317611


namespace NUMINAMATH_CALUDE_total_pupils_l3176_317646

theorem total_pupils (pizza : ℕ) (burgers : ℕ) (both : ℕ) 
  (h1 : pizza = 125) 
  (h2 : burgers = 115) 
  (h3 : both = 40) : 
  pizza + burgers - both = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l3176_317646


namespace NUMINAMATH_CALUDE_lecture_distribution_l3176_317606

def total_lecture_time : ℕ := 480
def max_disc_capacity : ℕ := 70

theorem lecture_distribution :
  ∃ (num_discs : ℕ) (minutes_per_disc : ℕ),
    num_discs > 0 ∧
    minutes_per_disc > 0 ∧
    minutes_per_disc ≤ max_disc_capacity ∧
    num_discs * minutes_per_disc = total_lecture_time ∧
    (∀ n : ℕ, n > 0 → n * max_disc_capacity < total_lecture_time → n < num_discs) ∧
    minutes_per_disc = 68 := by
  sorry

end NUMINAMATH_CALUDE_lecture_distribution_l3176_317606


namespace NUMINAMATH_CALUDE_alex_walk_distance_l3176_317651

def south_movement : ℝ := 50 + 15
def north_movement : ℝ := 30
def west_movement : ℝ := 80
def east_movement : ℝ := 40

def net_south : ℝ := south_movement - north_movement
def net_west : ℝ := west_movement - east_movement

theorem alex_walk_distance : 
  ∃ (length_AB : ℝ), length_AB = (net_south^2 + net_west^2).sqrt :=
sorry

end NUMINAMATH_CALUDE_alex_walk_distance_l3176_317651


namespace NUMINAMATH_CALUDE_boys_in_class_l3176_317648

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l3176_317648


namespace NUMINAMATH_CALUDE_jack_apple_distribution_l3176_317670

theorem jack_apple_distribution (total_apples : ℕ) (given_to_father : ℕ) (num_friends : ℕ) :
  total_apples = 55 →
  given_to_father = 10 →
  num_friends = 4 →
  (total_apples - given_to_father) % (num_friends + 1) = 0 →
  (total_apples - given_to_father) / (num_friends + 1) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_jack_apple_distribution_l3176_317670


namespace NUMINAMATH_CALUDE_simplify_expression_l3176_317617

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3176_317617


namespace NUMINAMATH_CALUDE_largest_non_odd_units_digit_proof_l3176_317688

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def largest_non_odd_units_digit : ℕ := 8

theorem largest_non_odd_units_digit_proof :
  ∀ d : ℕ, d ≤ 9 →
    (d > largest_non_odd_units_digit →
      ∃ n : ℕ, is_odd n ∧ units_digit n = d) ∧
    (d ≤ largest_non_odd_units_digit →
      d = largest_non_odd_units_digit ∨
      ∀ n : ℕ, is_odd n → units_digit n ≠ d) :=
sorry

end NUMINAMATH_CALUDE_largest_non_odd_units_digit_proof_l3176_317688


namespace NUMINAMATH_CALUDE_card_value_decrease_l3176_317607

theorem card_value_decrease (x : ℝ) : 
  (1 - x/100) * (1 - x/100) = 0.81 → x = 10 := by
sorry

end NUMINAMATH_CALUDE_card_value_decrease_l3176_317607


namespace NUMINAMATH_CALUDE_prime_divisor_ge_11_l3176_317602

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem prime_divisor_ge_11 (B : Nat) (h1 : B > 10) (h2 : all_digits_valid B) :
  ∃ p : Nat, p.Prime ∧ p ≥ 11 ∧ p ∣ B :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_ge_11_l3176_317602
