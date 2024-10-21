import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_stop_HH_fair_coin_l283_28354

/-- A fair coin has equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting a sequence of two coin flips -/
noncomputable def prob_sequence (p : ℝ) (seq : String) : ℝ :=
  if seq.length = 2 then p^(seq.count 'H') * (1-p)^(seq.count 'T') else 0

/-- The stopping sequences -/
def stopping_sequences : List String := ["HH", "TH"]

/-- The probability of stopping with a specific sequence -/
noncomputable def prob_stop_with (p : ℝ) (seq : String) : ℝ :=
  prob_sequence p seq / (stopping_sequences.map (prob_sequence p)).sum

/-- The main theorem: probability of stopping with HH is 1/2 for a fair coin -/
theorem prob_stop_HH_fair_coin (p : ℝ) (h : fair_coin p) :
  prob_stop_with p "HH" = 1/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_stop_HH_fair_coin_l283_28354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_length_l283_28304

/-- The side length of the large square in inches -/
noncomputable def large_square_side : ℝ := 120

/-- The area of the large square in square inches -/
noncomputable def large_square_area : ℝ := large_square_side ^ 2

/-- The number of L-shaped regions -/
def num_l_regions : ℕ := 4

/-- The fraction of the total area occupied by each L-shaped region -/
noncomputable def l_region_fraction : ℝ := 1 / 5

/-- The theorem stating that the side length of the center square is 60 inches -/
theorem center_square_side_length :
  let total_l_area : ℝ := num_l_regions * l_region_fraction * large_square_area
  let center_square_area : ℝ := large_square_area - total_l_area
  Real.sqrt center_square_area = 60 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_square_side_length_l283_28304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l283_28322

noncomputable def a (n : ℕ) : ℝ := 2 * n
noncomputable def S (n : ℕ) : ℝ := n^2 + (1/2) * a n

noncomputable def f (x : ℝ) (n : ℕ) : ℝ := x + a n / (2 * x)

noncomputable def b (n : ℕ) : ℝ :=
  if n % 4 = 1 then 2 * (4 * n - 2)
  else if n % 4 = 2 then 2 * (4 * n - 2) + 2 * (4 * n)
  else if n % 4 = 3 then 2 * (4 * n - 2) + 2 * (4 * n) + 2 * (4 * n + 2)
  else 2 * (4 * n - 2) + 2 * (4 * n) + 2 * (4 * n + 2) + 2 * (4 * n + 4)

noncomputable def g (n : ℕ) : ℝ := (1 + 1 / n)^n

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → S n / n = f n n) ∧
  (∀ n : ℕ, a n = 2 * n) ∧
  (b 5 + b 100 = 2010) ∧
  (∀ n : ℕ, n > 0 → 2 ≤ g n ∧ g n < 3) := by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l283_28322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_and_b_for_real_roots_l283_28327

theorem smallest_c_and_b_for_real_roots (c b : ℝ) : 
  (∀ x : ℝ, x^4 - c*x^3 + b*x^2 - c*x + 1 = 0 → x ∈ Set.univ) →
  c > 0 →
  (∀ c' : ℝ, c' > 0 → (∃ b' : ℝ, b' > 0 ∧ 
    (∀ x : ℝ, x^4 - c'*x^3 + b'*x^2 - c'*x + 1 = 0 → x ∈ Set.univ)) → 
    c ≤ c') →
  c = 4 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_c_and_b_for_real_roots_l283_28327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trade_balance_1994_1995_l283_28300

theorem trade_balance_1994_1995 (deficit_1994 deficit_1995 : ℝ) 
  (export_increase import_increase : ℝ) :
  deficit_1994 = 3.8 →
  deficit_1995 = 3 →
  export_increase = 0.11 →
  ∃ (export_1994 import_1994 : ℝ),
    (1 + import_increase) * import_1994 - (1 + export_increase) * export_1994 = deficit_1995 ∧
    ‖export_1994 - 11.425‖ < 0.001 ∧
    ‖import_1994 - 15.225‖ < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trade_balance_1994_1995_l283_28300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l283_28390

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  focal_length : ℝ
  major_minor_ratio : ℝ
  is_centered_at_origin : center = (0, 0)
  has_focal_length_2 : focal_length = 2
  has_specific_ratio : major_minor_ratio = Real.sqrt 2

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Represents a line passing through a point -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Predicate to check if a point is on an ellipse -/
def on_ellipse (E : Ellipse) (p : Point) : Prop :=
  (p.1 ^ 2) / 2 + p.2 ^ 2 = 1

/-- The theorem to be proved -/
theorem ellipse_dot_product_bound (E : Ellipse) (l : Line) (A B : Point) (P : Point) :
  let left_focus : Point := (-1, 0)
  l.point = left_focus ∧ 
  on_ellipse E A ∧ on_ellipse E B ∧
  P = (2, 0) →
  dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) ≤ 17/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l283_28390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l283_28359

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + Real.sin (ω * x - Real.pi / 2)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x / 2 + Real.pi / 8)

theorem problem_solution (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 3) (h3 : f ω (Real.pi / 6) = 0) :
  ω = 2 ∧ 
  ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4) ∧
    g ω x = -Real.sqrt 3 / 2 ∧
    ∀ y ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4), g ω y ≥ g ω x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l283_28359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_areas_l283_28361

/-- The sum of areas of two equilateral triangles formed from a wire of length 12 units -/
noncomputable def sum_of_areas (x : ℝ) : ℝ :=
  (Real.sqrt 3 / 36) * (2 * x^2 - 24 * x + 144)

/-- The minimum sum of areas of two equilateral triangles formed from a wire of length 12 units -/
theorem min_sum_of_areas :
  ∃ (x : ℝ), x > 0 ∧ x < 12 ∧
  (∀ (y : ℝ), y > 0 → y < 12 → sum_of_areas x ≤ sum_of_areas y) ∧
  sum_of_areas x = 2 * Real.sqrt 3 := by
  sorry

#check min_sum_of_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_areas_l283_28361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l283_28372

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Theorem statement
theorem hyperbola_properties :
  -- Vertex coordinates
  (∃ x : ℝ, x = 2 ∧ hyperbola x 0 ∧ hyperbola (-x) 0) ∧
  -- Length of real axis
  (∃ l : ℝ, l = 4 ∧ l = 2 * 2) ∧
  -- Asymptote equations
  (∀ x y : ℝ, (x + 2*y = 0 ∨ x - 2*y = 0) ↔ (∃ k : ℝ, k ≠ 0 ∧ hyperbola (k*x) (k*y))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l283_28372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_34_l283_28319

/-- Represents a student in the school system -/
structure Student where
  id : ℕ

/-- Represents a subject offered by the school -/
structure Subject where
  id : ℕ

/-- Represents the class type for a subject (ordinary or advanced) -/
inductive ClassType
  | Ordinary
  | Advanced

/-- The total number of subjects offered by the school -/
def totalSubjects : ℕ := 100

/-- The minimum number of subjects for which two students must be in different classes to be considered unfamiliar -/
def unfamiliarityThreshold : ℕ := 51

/-- Function to determine if two students are in different classes for a given subject -/
def differentClasses (s1 s2 : Student) (subj : Subject) : Prop := sorry

/-- Function to count the number of subjects for which two students are in different classes -/
def countDifferentClasses (s1 s2 : Student) : ℕ := sorry

/-- Proposition that all pairs of students are unfamiliar (in different classes for at least 51 subjects) -/
def allPairsUnfamiliar (students : List Student) : Prop :=
  ∀ s1 s2, s1 ∈ students → s2 ∈ students → s1 ≠ s2 → countDifferentClasses s1 s2 ≥ unfamiliarityThreshold

/-- Theorem stating that if all pairs of students are unfamiliar, then the maximum number of students is 34 -/
theorem max_students_34 (students : List Student) (h : allPairsUnfamiliar students) :
  students.length ≤ 34 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_34_l283_28319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sec_squared_alpha_minus_two_l283_28301

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_sec_squared_alpha_minus_two
  (f : ℝ → ℝ)
  (α : ℝ)
  (h_odd : is_odd_function f)
  (h_period : has_period f 5)
  (h_f_neg_three : f (-3) = 1)
  (h_tan_alpha : Real.tan α = 3) :
  f ((1 + Real.tan α ^ 2) - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sec_squared_alpha_minus_two_l283_28301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_l283_28314

-- Define the set T
def T : Set ℂ :=
  {z : ℂ | ∃ x y : ℝ, z = x + y * Complex.I ∧ Real.sqrt 2 / 2 ≤ x ∧ x ≤ 3 / 4}

-- Define the property P(m)
def P (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ T ∧ z ^ n = 1

-- State the theorem
theorem smallest_m : (∃ m : ℕ, P m) ∧ (∀ k : ℕ, k < 23 → ¬P k) ∧ P 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_l283_28314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l283_28373

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Conditions
  sine_law : c / a = Real.sin A / Real.sin B
  cosine_law : Real.sqrt 3 * b * c * Real.cos A = a * Real.sin B
  a_value : a = Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) : 
  t.A = π/3 ∧ t.a + t.b + t.c = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l283_28373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_tour_solution_l283_28367

/-- Represents the number of teachers -/
def x : ℕ := sorry

/-- Represents the number of students -/
def y : ℕ := sorry

/-- Represents the number of Type A buses -/
def m : ℕ := sorry

/-- The total number of buses -/
def total_buses : ℕ := 6

/-- The maximum allowed cost -/
def max_cost : ℕ := 2300

/-- The capacity of a Type A bus -/
def capacity_a : ℕ := 40

/-- The capacity of a Type B bus -/
def capacity_b : ℕ := 30

/-- The rental fee for a Type A bus -/
def fee_a : ℕ := 400

/-- The rental fee for a Type B bus -/
def fee_b : ℕ := 320

/-- The condition that 12 students per teacher leaves 8 students without a teacher -/
axiom condition1 : y = 12 * x + 8

/-- The condition that 13 students per teacher leaves 1 teacher short of 6 students -/
axiom condition2 : y = 13 * x - 6

/-- The total capacity of buses must accommodate all participants -/
def capacity_condition (m : ℕ) : Prop :=
  m * capacity_a + (total_buses - m) * capacity_b ≥ x + y

/-- The total rental cost must not exceed the maximum allowed cost -/
def cost_condition (m : ℕ) : Prop :=
  m * fee_a + (total_buses - m) * fee_b ≤ max_cost

/-- The main theorem stating the correct number of teachers and students, 
    the number of rental options, and the minimum rental cost -/
theorem study_tour_solution :
  x = 14 ∧ 
  y = 176 ∧ 
  (∃ (options : Finset ℕ), options.card = 4 ∧ 
    ∀ m, m ∈ options ↔ (capacity_condition m ∧ cost_condition m)) ∧
  (∃ min_cost : ℕ, min_cost = 2000 ∧
    ∀ m, capacity_condition m ∧ cost_condition m → 
      m * fee_a + (total_buses - m) * fee_b ≥ min_cost) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_tour_solution_l283_28367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_smallest_n_l283_28318

def sequence_a (n : ℕ) : ℚ := 2^n - 1

def S (n : ℕ) : ℚ := sequence_a n * n - n

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → S n + n = 2 * sequence_a n) →
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1)) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 2^n - 1) :=
sorry

def T (n : ℕ) : ℚ := (n - 1) * 2^(n + 1) + 2 - n * (n + 1) / 2

theorem smallest_n :
  (∀ n : ℕ, 0 < n ∧ n < 8 → T n + (n^2 + n) / 2 ≤ 2015) ∧
  (T 8 + (8^2 + 8) / 2 > 2015) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_smallest_n_l283_28318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_l283_28317

noncomputable def number_list : List ℚ := [4/3, 1, 314/100, 0, 1/10, -4, 100]

theorem count_positive_integers (list : List ℚ := number_list) : 
  (list.filter (λ x => x > 0 ∧ x.den == 1)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_l283_28317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l283_28310

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- The theorem statement -/
theorem hyperbola_focal_distance 
  (x y xF₁ yF₁ xF₂ yF₂ : ℝ) 
  (h_hyperbola : hyperbola x y)
  (h_F₁ : xF₁ < xF₂) -- F₁ is the left focus
  (h_distance : distance x y xF₁ yF₁ = 9) :
  distance x y xF₂ yF₂ = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l283_28310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_and_max_chord_l283_28325

-- Define the ellipse equation
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

-- Define the line equation
def is_on_line (x y m : ℝ) : Prop := y = (3/2) * x + m

-- Define the intersection condition
def intersects (m : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_ellipse x y ∧ is_on_line x y m

-- Define the chord length function
noncomputable def chord_length (m : ℝ) : ℝ :=
  Real.sqrt 13 / 3 * Real.sqrt (-m^2 + 18)

-- Theorem statement
theorem ellipse_line_intersection_and_max_chord :
  (∀ m : ℝ, intersects m ↔ m ∈ Set.Icc (-3 * Real.sqrt 2) (3 * Real.sqrt 2)) ∧
  (∃ m : ℝ, ∀ m' : ℝ, chord_length m ≥ chord_length m' ∧ chord_length m = Real.sqrt 26) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_and_max_chord_l283_28325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_trajectory_is_parabola_l283_28381

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The path along which point C moves -/
def CPath (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- Theorem: The trajectory of the centroid is a parabola -/
theorem centroid_trajectory_is_parabola
  (d : ℝ)
  (a b c : ℝ)
  (h_AB_fixed : ∀ x, x = -d ∨ x = d)
  (h_M_origin : ((-d + d) / 2 = 0))
  (h_C_path : ∀ x, ∃ y, y = CPath a b c x) :
  ∃ α β γ : ℝ, ∀ x : ℝ,
    let t : Triangle := { A := {x := -d, y := 0},
                          B := {x := d, y := 0},
                          C := {x := x, y := CPath a b c x} }
    (centroid t).y = α * (centroid t).x^2 + β * (centroid t).x + γ :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_trajectory_is_parabola_l283_28381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_leq_S_l283_28340

theorem T_leq_S (a b : ℝ) : (a + 2*b) ≤ (a + b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_leq_S_l283_28340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_triangles_distance_concentric_cubes_distance_l283_28369

/-- Two concentric equilateral triangles with side lengths 1 and 2 -/
structure ConcentricTriangles where
  T₁ : Set (Fin 2 → ℝ)
  T₂ : Set (Fin 2 → ℝ)
  center : Fin 2 → ℝ
  side_length_T₁ : ℝ
  side_length_T₂ : ℝ

/-- Two concentric cubes with edge lengths 1 and 2 -/
structure ConcentricCubes where
  K₁ : Set (Fin 3 → ℝ)
  K₂ : Set (Fin 3 → ℝ)
  center : Fin 3 → ℝ
  edge_length_K₁ : ℝ
  edge_length_K₂ : ℝ

/-- Distance between two sets -/
noncomputable def distance {n : ℕ} (A B : Set (Fin n → ℝ)) : ℝ := sorry

theorem concentric_triangles_distance 
  (ct : ConcentricTriangles) 
  (h1 : ct.side_length_T₁ = 1) 
  (h2 : ct.side_length_T₂ = 2) :
  distance ct.T₁ ct.T₂ = Real.sqrt 3 / 6 ∧ 
  distance ct.T₂ ct.T₁ = Real.sqrt 3 / 3 := by sorry

theorem concentric_cubes_distance 
  (cc : ConcentricCubes) 
  (h1 : cc.edge_length_K₁ = 1) 
  (h2 : cc.edge_length_K₂ = 2) :
  distance cc.K₁ cc.K₂ = 1 / 2 ∧ 
  distance cc.K₂ cc.K₁ = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_triangles_distance_concentric_cubes_distance_l283_28369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_wins_l283_28330

theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 130 → 
  win_percentage = 60 / 100 → 
  games_won = (win_percentage * total_games).floor → 
  games_won = 78 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_team_wins_l283_28330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l283_28348

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8_pos : a 8 > 0)
  (h_a8_a9_neg : a 8 + a 9 < 0) :
  (∀ n > 15, S a n ≤ 0) ∧
  (∀ n ≤ 15, S a n > 0) ∧
  (∀ n ∈ Finset.range 15, S a 8 / a 8 ≥ S a n / a n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l283_28348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_binomial_expansion_l283_28302

/-- The 4th term in the expansion of (x^3 + 2)^5 -/
def fourth_term (x : ℝ) : ℝ := 80 * x^6

/-- The binomial (x^3 + 2)^5 -/
def binomial (x : ℝ) : ℝ := (x^3 + 2)^5

/-- Coefficient of the kth term in the binomial expansion -/
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The kth term in the expansion of (x^3 + 2)^5 -/
def kth_term (x : ℝ) (k : ℕ) : ℝ :=
  (binomial_coeff 5 k : ℝ) * (x^3)^(5 - k) * 2^k

theorem fourth_term_of_binomial_expansion (x : ℝ) : 
  kth_term x 3 = fourth_term x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_binomial_expansion_l283_28302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_nuts_l283_28345

/-- Represents the number of nuts taken by each raccoon -/
structure RaccoonNuts where
  n1 : ℕ
  n2 : ℕ
  n3 : ℕ
  n4 : ℕ

/-- Calculates the final number of nuts for each raccoon -/
def finalNuts (r : RaccoonNuts) : Vector ℚ 4 :=
  ⟨[1/2 * r.n1 + 1/6 * r.n2 + 5/18 * r.n3 + 11/36 * r.n4,
    1/6 * r.n1 + 1/3 * r.n2 + 5/18 * r.n3 + 11/36 * r.n4,
    1/6 * r.n1 + 1/6 * r.n2 + 1/6 * r.n3 + 11/36 * r.n4,
    1/6 * r.n1 + 1/6 * r.n2 + 5/18 * r.n3 + 1/12 * r.n4],
   by simp⟩

/-- Checks if the final distribution satisfies the 4:3:2:1 ratio -/
def satisfiesRatio (r : RaccoonNuts) : Prop :=
  let v := finalNuts r
  v[0] = 4 * v[3] ∧ v[1] = 3 * v[3] ∧ v[2] = 2 * v[3]

/-- Checks if all divisions result in whole numbers -/
def wholeDivisions (r : RaccoonNuts) : Prop :=
  r.n1 % 2 = 0 ∧ r.n2 % 3 = 0 ∧ r.n3 % 6 = 0 ∧ r.n4 % 12 = 0

/-- The main theorem: the least possible total number of nuts is 6048 -/
theorem least_possible_nuts :
  ∃ (r : RaccoonNuts), 
    satisfiesRatio r ∧ 
    wholeDivisions r ∧
    (∀ (s : RaccoonNuts), satisfiesRatio s → wholeDivisions s → 
      r.n1 + r.n2 + r.n3 + r.n4 ≤ s.n1 + s.n2 + s.n3 + s.n4) ∧
    r.n1 + r.n2 + r.n3 + r.n4 = 6048 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_nuts_l283_28345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chrystal_mountain_speed_increase_l283_28356

/-- Calculates the speed during descent based on given parameters -/
noncomputable def descent_speed 
  (initial_speed : ℝ) 
  (ascent_distance : ℝ) 
  (descent_distance : ℝ) 
  (total_time : ℝ) : ℝ :=
  descent_distance / (total_time - ascent_distance / (initial_speed * 0.5))

/-- Proves that given the conditions of Chrystal's mountain trip, 
    the percentage increase in speed when descending the mountain is 20%. -/
theorem chrystal_mountain_speed_increase 
  (initial_speed : ℝ) 
  (ascent_distance : ℝ) 
  (descent_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : ascent_distance = 60)
  (h3 : descent_distance = 72)
  (h4 : total_time = 6)
  : (descent_speed initial_speed ascent_distance descent_distance total_time - initial_speed) / initial_speed * 100 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chrystal_mountain_speed_increase_l283_28356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l283_28395

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y = -1

/-- The area of the region -/
noncomputable def region_area : ℝ := 4 * Real.pi

/-- Theorem stating the existence of a circle that matches the region equation and its area -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Provide the center coordinates and radius
  let center_x := 2
  let center_y := -1
  let radius := 2
  
  -- Assert the existence of these values
  use center_x, center_y, radius
  
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l283_28395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_l283_28337

noncomputable section

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 + 1/x - a

-- Define the function g
def g (x : ℝ) (a b : ℝ) : ℝ := f x a + b

-- Define the function h (xf(x))
def h (x : ℝ) (a : ℝ) : ℝ := x * f x a

theorem tangent_line_and_range (a b : ℝ) :
  (∀ x, x ≠ 0 → (deriv (h · a)) x = 0 → x = 1) ∧  -- x = 1 is an extremum point of y = xf(x)
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ∧  -- f(x) has 2 zeros
  (∃ x₁ x₂ x₃ x₄ x₅ x₆, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ x₁ ≠ x₆ ∧
                         x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ x₂ ≠ x₆ ∧
                         x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₃ ≠ x₆ ∧
                         x₄ ≠ x₅ ∧ x₄ ≠ x₆ ∧
                         x₅ ≠ x₆ ∧
                         f (g x₁ a b) a = 0 ∧ f (g x₂ a b) a = 0 ∧ f (g x₃ a b) a = 0 ∧
                         f (g x₄ a b) a = 0 ∧ f (g x₅ a b) a = 0 ∧ f (g x₆ a b) a = 0) →  -- f(g(x)) has 6 zeros
  (∀ x, f x a = 7 * x - 14) ∧  -- Tangent line equation
  (a + b < 2)  -- Range of a + b
  := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_range_l283_28337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_f_l283_28362

noncomputable def f (x : ℝ) : ℝ := (4*(x+3)*(x-2)-24) / (x+4)

theorem y_intercept_of_f : f 0 = -12 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [mul_add, add_mul, mul_sub, sub_mul]
  -- Perform arithmetic
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_f_l283_28362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l283_28394

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define the line passing through (1,1)
def my_line (m : ℝ) (x y : ℝ) : Prop := y - 1 = m * (x - 1)

-- Define the condition for the shortest chord
def shortest_chord (m : ℝ) : Prop := m = -1

-- Theorem statement
theorem shortest_chord_line :
  ∃ (m : ℝ), my_line m 1 1 ∧ 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    my_circle x₁ y₁ ∧ my_circle x₂ y₂ ∧ 
    my_line m x₁ y₁ ∧ my_line m x₂ y₂ ∧
    shortest_chord m) →
  (∀ x y, my_line m x y ↔ x + y = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_line_l283_28394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpaste_lasts_two_days_l283_28334

structure ToothpasteUsage where
  toothpaste_amount : ℕ
  dad_usage : ℕ
  mom_usage : ℕ
  anne_usage : ℕ
  brother_usage : ℕ
  sister_usage : ℕ
  dad_brushings : ℕ
  mom_brushings : ℕ
  anne_brushings : ℕ
  brother_brushings : ℕ
  sister_brushings : ℕ

def anne_family_usage : ToothpasteUsage :=
  { toothpaste_amount := 90
  , dad_usage := 4
  , mom_usage := 3
  , anne_usage := 2
  , brother_usage := 1
  , sister_usage := 1
  , dad_brushings := 4
  , mom_brushings := 4
  , anne_brushings := 4
  , brother_brushings := 4
  , sister_brushings := 2 }

/-- Calculates the number of full days the toothpaste will last -/
def days_toothpaste_lasts (usage : ToothpasteUsage) : ℕ :=
  usage.toothpaste_amount / (
    usage.dad_usage * usage.dad_brushings +
    usage.mom_usage * usage.mom_brushings +
    usage.anne_usage * usage.anne_brushings +
    usage.brother_usage * usage.brother_brushings +
    usage.sister_usage * usage.sister_brushings
  )

theorem toothpaste_lasts_two_days :
  days_toothpaste_lasts anne_family_usage = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toothpaste_lasts_two_days_l283_28334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2009_eq_neg_one_l283_28352

/-- A function of the form m*sin(πx + α₁) + n*cos(πx + α₂) -/
noncomputable def f (m n α₁ α₂ : ℝ) (x : ℝ) : ℝ :=
  m * Real.sin (Real.pi * x + α₁) + n * Real.cos (Real.pi * x + α₂)

/-- Theorem stating that if f(2008) = 1, then f(2009) = -1 -/
theorem f_2009_eq_neg_one
  (m n α₁ α₂ : ℝ)
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (hα₁ : α₁ ≠ 0)
  (hα₂ : α₂ ≠ 0)
  (h2008 : f m n α₁ α₂ 2008 = 1) :
  f m n α₁ α₂ 2009 = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2009_eq_neg_one_l283_28352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_stones_needed_l283_28324

/-- The number of patio stones needed to cover a rectangular garden --/
def num_patio_stones (garden_length garden_width stone_length stone_width : ℚ) : ℕ :=
  (garden_length * garden_width / (stone_length * stone_width)).ceil.toNat

/-- Theorem stating that 120 patio stones of size 0.5 m by 0.5 m are needed to cover a 15 m by 2 m garden --/
theorem patio_stones_needed : 
  num_patio_stones 15 2 (1/2) (1/2) = 120 := by
  sorry

#eval num_patio_stones 15 2 (1/2) (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patio_stones_needed_l283_28324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_formula_l283_28320

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => deriv (f_n n)

theorem f_2017_formula : f_n 2017 = fun x => (x + 2017) * Real.exp x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2017_formula_l283_28320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_cos_x_div_2_solutions_l283_28366

theorem tan_2x_eq_cos_x_div_2_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧ 
  (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.cos (x / 2)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.cos (x / 2) → x ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_eq_cos_x_div_2_solutions_l283_28366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_24_l283_28379

theorem sum_of_factors_24 : Finset.sum (Finset.filter (λ d => 24 % d = 0) (Finset.range 25)) id = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_24_l283_28379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_is_six_l283_28316

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line -/
structure Line where
  -- We don't need to define the line explicitly for this problem

/-- Predicate to check if a point is a tangent point of a circle to a line -/
def is_tangent_point (circle : Circle) (line : Line) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if a point is between two other points on a line -/
def between (p1 p2 p3 : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate to check if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  sorry

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

theorem area_of_triangle_abc_is_six (A B C : Circle) (m : Line) :
  A.radius = 2 →
  B.radius = 3 →
  C.radius = 4 →
  (∃ A' B' C' : ℝ × ℝ, 
    is_tangent_point A m A' ∧ 
    is_tangent_point B m B' ∧ 
    is_tangent_point C m C' ∧
    between A' B' C') →
  externally_tangent B A →
  externally_tangent B C →
  triangle_area A.center B.center C.center = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_is_six_l283_28316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_ratio_l283_28399

def trains_per_year : ℕ := 3
def years : ℕ := 5
def total_trains : ℕ := 45

theorem trains_ratio : 
  (total_trains - trains_per_year * years : ℚ) / (trains_per_year * years) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_ratio_l283_28399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_speeds_doubled_l283_28377

-- Define the initial meeting time
def initial_meeting_time : ℕ := 720  -- 12:00 PM in minutes since midnight

-- Define the speeds of the cars
variable (speed_car1 speed_car2 : ℝ)

-- Define the distance between points A and B
def distance (speed_car1 speed_car2 : ℝ) : ℝ := 
  2 * (initial_meeting_time : ℝ) * (speed_car1 + speed_car2)

-- Define the new meeting times when speeds are doubled
def new_time_car1_doubled : ℝ := initial_meeting_time - 56
def new_time_car2_doubled : ℝ := initial_meeting_time - 65

-- Define the equations for the doubled speed scenarios
def equation_car1_doubled (speed_car1 speed_car2 : ℝ) : Prop := 
  distance speed_car1 speed_car2 = new_time_car1_doubled * (2 * speed_car1 + speed_car2)

def equation_car2_doubled (speed_car1 speed_car2 : ℝ) : Prop := 
  distance speed_car1 speed_car2 = new_time_car2_doubled * (speed_car1 + 2 * speed_car2)

-- Theorem to prove
theorem both_speeds_doubled 
  (h1 : equation_car1_doubled speed_car1 speed_car2)
  (h2 : equation_car2_doubled speed_car1 speed_car2) :
  ∃ (new_time : ℕ), new_time = 629 ∧ 
    distance speed_car1 speed_car2 = (new_time : ℝ) * (2 * speed_car1 + 2 * speed_car2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_speeds_doubled_l283_28377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_natural_solution_is_two_l283_28329

noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

def inequality1 (x : ℝ) : Prop := log_sqrt2 (x - 1) < 4

def inequality2 (x : ℝ) : Prop := x / (x - 3) + (x - 5) / x < 2 * x / (3 - x)

theorem only_natural_solution_is_two :
  ∃! (n : ℕ), inequality1 (n : ℝ) ∧ inequality2 (n : ℝ) ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_natural_solution_is_two_l283_28329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_triangle_area_l283_28387

-- Define the areas of the squares
def square1_area : ℝ := 4
def square2_area : ℝ := 9
def square3_area : ℝ := 36

-- Define the side lengths of the squares
noncomputable def square1_side : ℝ := Real.sqrt square1_area
noncomputable def square2_side : ℝ := Real.sqrt square2_area
noncomputable def square3_side : ℝ := Real.sqrt square3_area

-- Define the triangle
noncomputable def triangle_base : ℝ := square1_side + square2_side + square3_side
noncomputable def triangle_height : ℝ := square1_side + square2_side + square3_side

-- Theorem statement
theorem inscribed_squares_triangle_area :
  (1 / 2) * triangle_base * triangle_height = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_triangle_area_l283_28387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l283_28397

/-- Given a train passing a platform, calculate the platform length -/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 360)
  (h2 : train_speed_kmh = 45)
  (h3 : time_to_pass = 40) :
  (train_speed_kmh * 1000 / 3600 * time_to_pass - train_length) = 140 := by
  -- Convert speed from km/h to m/s
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  -- Calculate total distance
  let total_distance := train_speed_ms * time_to_pass
  -- Calculate platform length
  let platform_length := total_distance - train_length
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l283_28397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_circumscribed_sphere_volume_l283_28392

theorem cube_circumscribed_sphere_volume (surface_area : ℝ) (h : surface_area = 6) :
  (4 / 3) * Real.pi * ((Real.sqrt (surface_area / 6) * Real.sqrt 3 / 2) ^ 3) = (Real.sqrt 3 / 2) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_circumscribed_sphere_volume_l283_28392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_20min_speed_is_70_l283_28307

/-- Represents a driving journey with specific conditions -/
structure Journey where
  total_distance : ℚ
  total_time : ℚ
  first_hour_speed : ℚ
  last_40min_speed : ℚ

/-- Calculates the average speed during the middle 20 minutes of the journey -/
def middle_20min_speed (j : Journey) : ℚ :=
  let first_hour_distance := j.first_hour_speed * 1
  let last_40min_distance := j.last_40min_speed * (2/3)
  let middle_20min_distance := j.total_distance - first_hour_distance - last_40min_distance
  middle_20min_distance / (1/3)

/-- Theorem stating that for the given journey conditions, the middle 20 minutes speed is 70 mph -/
theorem middle_20min_speed_is_70 (j : Journey) 
  (h1 : j.total_distance = 120)
  (h2 : j.total_time = 2)
  (h3 : j.first_hour_speed = 50)
  (h4 : j.last_40min_speed = 70) : 
  middle_20min_speed j = 70 := by
  sorry

#eval middle_20min_speed { total_distance := 120, total_time := 2, first_hour_speed := 50, last_40min_speed := 70 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_20min_speed_is_70_l283_28307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l283_28393

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal distance of a hyperbola -/
noncomputable def focalDistance (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (c : ℝ) 
  (h_c : c = focalDistance h) 
  (h_distance : ∃ (B : ℝ × ℝ), 
    abs B.2 < 2 * (h.a + c) ∧ 
    B.1 = (h.b^4) / (h.a^2 * (h.a - c)) + c) :
  1 < eccentricity h ∧ eccentricity h < Real.sqrt 3 := by
  sorry

#check hyperbola_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l283_28393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telethon_total_money_l283_28378

/-- Calculates the total money raised in a telethon given the following conditions:
  * The telethon runs for a total of 26 hours
  * For the first 12 hours, money is generated at a rate of $5000 per hour
  * For the remaining 14 hours, money is generated at a rate 20% higher than the initial rate
-/
theorem telethon_total_money : ℝ := by
  let initial_rate : ℝ := 5000
  let initial_hours : ℕ := 12
  let remaining_hours : ℕ := 14
  let rate_increase : ℝ := 0.2
  
  let initial_money : ℝ := initial_rate * initial_hours
  let increased_rate : ℝ := initial_rate * (1 + rate_increase)
  let remaining_money : ℝ := increased_rate * remaining_hours
  
  have h : initial_money + remaining_money = 144000 := by
    -- Proof goes here
    sorry
  
  exact 144000


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telethon_total_money_l283_28378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_value_l283_28344

def a : ℕ → ℚ
  | 0 => 1/2  -- Adding the base case for 0
  | n + 1 => (1 + a n) / (1 - a n)

theorem a_2008_value : a 2008 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2008_value_l283_28344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_tournament_l283_28315

/-- Represents the state of a tournament after a round -/
structure TournamentState where
  no_losses : ℕ  -- Number of participants with no losses
  one_loss : ℕ   -- Number of participants with one loss

/-- Simulates a round of the tournament -/
def next_round (state : TournamentState) : TournamentState :=
  { no_losses := state.no_losses / 2,
    one_loss := state.no_losses / 2 + state.one_loss / 2 }

/-- Defines the tournament process -/
def tournament (initial_participants : ℕ) : ℕ :=
  let rec aux (state : TournamentState) (round : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then 0
    else if state.no_losses = 1 then
      state.no_losses + state.one_loss
    else
      aux (next_round state) (round + 1) (fuel - 1)
  aux { no_losses := initial_participants, one_loss := 0 } 0 (initial_participants + 1)

/-- The main theorem to be proved -/
theorem arm_wrestling_tournament :
  tournament 896 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arm_wrestling_tournament_l283_28315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_together_l283_28350

/-- The time it takes for two people to complete a task together, given their individual completion times -/
noncomputable def combined_time (time1 : ℝ) (time2 : ℝ) : ℝ :=
  1 / (1 / time1 + 1 / time2)

/-- Sally's time to paint a house -/
def sally_time : ℝ := 4

/-- John's time to paint a house -/
def john_time : ℝ := 6

theorem paint_house_together : combined_time sally_time john_time = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_house_together_l283_28350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_product_l283_28338

-- Define the circles and tangents
noncomputable def circle1_radius (a : ℝ) : ℝ := a
noncomputable def circle2_radius (b : ℝ) : ℝ := b
def tangent_AB_length : ℝ := 16
def tangent_PQ_length : ℝ := 14

-- Theorem statement
theorem circles_tangents_product (a b : ℝ) :
  circle1_radius a = a →
  circle2_radius b = b →
  tangent_AB_length = 16 →
  tangent_PQ_length = 14 →
  a * b = 15 := by
  intro h1 h2 h3 h4
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangents_product_l283_28338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l283_28376

/-- Calculates the amount of money in an account after a given time period with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (time_in_years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * time_in_years)

/-- Theorem: Given the specified conditions, the amount in the account after 6 months is $5,202 -/
theorem savings_account_balance : 
  let principal := (5000 : ℝ)
  let annual_rate := (0.08 : ℝ)
  let compounds_per_year := (4 : ℝ)
  let time_in_years := (0.5 : ℝ)
  abs (compound_interest principal annual_rate compounds_per_year time_in_years - 5202) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l283_28376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_increase_ratio_l283_28384

noncomputable def container_radius_narrow : ℝ := 3
noncomputable def container_radius_wide : ℝ := 6
noncomputable def sphere_radius : ℝ := 1

noncomputable def volume_sphere : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3

noncomputable def height_increase_narrow : ℝ := volume_sphere / (Real.pi * container_radius_narrow ^ 2)
noncomputable def height_increase_wide : ℝ := volume_sphere / (Real.pi * container_radius_wide ^ 2)

theorem height_increase_ratio :
  height_increase_narrow / height_increase_wide = 4 / 1 := by
  -- Expand definitions
  unfold height_increase_narrow height_increase_wide
  unfold volume_sphere
  unfold container_radius_narrow container_radius_wide sphere_radius
  
  -- Simplify the expression
  simp [Real.pi]
  
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_increase_ratio_l283_28384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l283_28368

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem f_and_g_properties :
  ∀ (A ω φ : ℝ),
  A > 0 → ω > 0 → -π < φ → φ ≤ π →
  (∀ x, f A ω φ x ≤ 2) →
  f A ω φ (π/6) = 2 →
  (∃ x₁ x₂, x₂ > x₁ ∧ x₂ - x₁ = π/2 ∧ f A ω φ x₁ = 0 ∧ f A ω φ x₂ = 0) →
  (∀ x, f A ω φ x = 2 * Real.sin (2*x + π/6)) ∧
  (∀ x, x ≠ π/2 + π * (↑n : ℝ) → 
    (6*(Real.cos x)^4 - (Real.sin x)^2 - 1) / ((f A ω φ (x/2 + π/6))^2 - 2) ∈ Set.Icc 1 (5/2) \ {7/4}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l283_28368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l283_28363

open Real

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := (1/2) * x^2

-- Define the line
noncomputable def line (x : ℝ) : ℝ := x - 2

-- Define the distance function from a point on the parabola to the line
noncomputable def distance (x : ℝ) : ℝ :=
  |x - parabola x - 2| / Real.sqrt 2

-- Theorem statement
theorem min_distance_parabola_to_line :
  ∃ (d : ℝ), d = (3 * Real.sqrt 2) / 4 ∧ ∀ (x : ℝ), distance x ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l283_28363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_5pi_6_l283_28358

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3)

theorem function_value_at_5pi_6 
  (ω : ℝ) 
  (h_ω : ω > 0)
  (α β : ℝ)
  (h_A : f ω α = 2)
  (h_B : f ω β = 0)
  (h_min_dist : (α - β)^2 + 4 = 4 + Real.pi^2 / 4) :
  f ω (5 * Real.pi / 6) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_5pi_6_l283_28358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_operations_proof_l283_28347

/-- An operation that erases two numbers whose sum is prime -/
def is_valid_operation (a b : ℕ) : Prop := Nat.Prime (a + b)

/-- The set of integers from 1 to 510 -/
def integer_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 510}

/-- The maximum number of operations possible -/
def max_operations : ℕ := 255

theorem max_operations_proof :
  ∀ (operations : List (ℕ × ℕ)),
    (∀ (pair : ℕ × ℕ), pair ∈ operations → 
      pair.fst ∈ integer_set ∧ pair.snd ∈ integer_set ∧ is_valid_operation pair.fst pair.snd) →
    (∀ n : ℕ, n ∈ integer_set → 
      (∃! pair, pair ∈ operations ∧ (pair.fst = n ∨ pair.snd = n))) →
    operations.length ≤ max_operations :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_operations_proof_l283_28347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l283_28328

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_to_focus (y : ℝ) (h : parabola 3 y) :
  distance (3, y) focus = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l283_28328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_half_l283_28396

theorem sin_2theta_plus_pi_half (θ : Real) :
  (3 : Real) = 5 * Real.cos θ → 
  (-4 : Real) = 5 * Real.sin θ → 
  Real.sin (2 * θ + Real.pi / 2) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_half_l283_28396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_distance_equality_l283_28357

-- Define the circle N
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

-- Define the point M
def point_M : ℝ × ℝ := (-1, 0)

-- Define the locus Ω
def locus_Ω (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a point on Ω in the first quadrant
def point_on_Ω_first_quadrant (p : ℝ × ℝ) : Prop :=
  locus_Ω p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≠ 1

-- Define the line x = 1
def line_x_eq_1 (x y : ℝ) : Prop := x = 1

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem locus_and_distance_equality
  (P : Set (ℝ × ℝ))  -- Set of all possible centers of circle P
  (A C : ℝ × ℝ)      -- Points A and C
  (B D : ℝ × ℝ)      -- Points B and D
  (E F : ℝ × ℝ)      -- Points E and F
  (N : ℝ × ℝ)        -- Center of circle N
  (h1 : ∀ p ∈ P, distance p point_M = distance p N - 4)  -- P passes through M and is tangent to N
  (h2 : point_on_Ω_first_quadrant A)
  (h3 : point_on_Ω_first_quadrant C)
  (h4 : locus_Ω B.1 B.2)
  (h5 : locus_Ω D.1 D.2)
  (h6 : line_x_eq_1 E.1 E.2)
  (h7 : line_x_eq_1 F.1 F.2)
  : (∀ p ∈ P, locus_Ω p.1 p.2) ∧ distance E N = distance F N :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_distance_equality_l283_28357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_leopard_arrangement_l283_28305

theorem snow_leopard_arrangement (n : ℕ) (h : n = 8) : 
  (2 : ℕ) * 3 * Nat.factorial (n - 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_leopard_arrangement_l283_28305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_minimum_value_l283_28342

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

-- Theorem for the minimum value
theorem minimum_value :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧
  m = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_minimum_value_l283_28342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l283_28380

theorem odd_power_sum_divisible (k : ℕ) (x y : ℤ) :
  (x + y) ∣ (x^(2*k + 1) + y^(2*k + 1)) →
  (x + y) ∣ (x^(2*k + 3) + y^(2*k + 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l283_28380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l283_28346

theorem divisors_count (n : ℕ) (h : n = 2^35 * 3^21) :
  (Finset.filter (fun d => d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 734 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_count_l283_28346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l283_28360

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

-- Main theorem
theorem f_properties (a : ℝ) (h_odd : ∀ x, f a x = -f a (-x)) :
  (a = 1/2) ∧ 
  (∀ y, y ∈ Set.range (f a) → -1/2 < y ∧ y < 1/2) ∧
  (∀ m n : ℝ, m + n ≠ 0 → (f a m + f a n) / (m^3 + n^3) > f a 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l283_28360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solute_volume_approx_l283_28303

/-- Represents a cylindrical container -/
structure Container where
  height : ℝ
  diameter : ℝ

/-- Represents the contents of the container -/
structure Contents where
  fillRatio : ℝ
  soluteToSolventRatio : ℝ

/-- Calculates the volume of solute in the container -/
noncomputable def soluteVolume (c : Container) (contents : Contents) : ℝ :=
  let radius := c.diameter / 2
  let solutionHeight := c.height * contents.fillRatio
  let solutionVolume := Real.pi * radius^2 * solutionHeight
  let soluteRatio := contents.soluteToSolventRatio / (1 + contents.soluteToSolventRatio)
  solutionVolume * soluteRatio

/-- The main theorem -/
theorem solute_volume_approx (c : Container) (contents : Contents) :
  c.height = 8 ∧ c.diameter = 3 ∧ contents.fillRatio = 1/4 ∧ contents.soluteToSolventRatio = 1/9 →
  abs (soluteVolume c contents - 1.41) < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solute_volume_approx_l283_28303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_overflow_water_in_sphere_l283_28385

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem no_overflow (cone_radius cone_height cylinder_radius cylinder_height : ℝ) 
  (h_cone_radius : cone_radius = 10)
  (h_cone_height : cone_height = 15)
  (h_cylinder_radius : cylinder_radius = 15)
  (h_cylinder_height : cylinder_height = 10) :
  cone_volume cone_radius cone_height ≤ cylinder_volume cylinder_radius cylinder_height :=
by sorry

theorem water_in_sphere 
  (cone_radius cone_height cylinder_radius cylinder_height sphere_volume : ℝ)
  (h_cone_radius : cone_radius = 10)
  (h_cone_height : cone_height = 15)
  (h_cylinder_radius : cylinder_radius = 15)
  (h_cylinder_height : cylinder_height = 10)
  (h_no_overflow : cone_volume cone_radius cone_height ≤ cylinder_volume cylinder_radius cylinder_height) :
  sphere_volume = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_overflow_water_in_sphere_l283_28385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l283_28375

open Real

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x - m / x

-- State the theorem
theorem local_minimum_condition (m : ℝ) :
  (∃ x > 0, IsLocalMin (f m) x ∧ f m x < 0) ↔ -1/exp 1 < m ∧ m < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_condition_l283_28375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l283_28353

/-- The equation of a hyperbola with parameter a ≠ 0 -/
def hyperbola (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => a ≠ 0 ∧ x^2 / (2*a) - y^2 / a = 1

/-- The equation of the asymptotes -/
def asymptote_equation : ℝ → ℝ → Prop :=
  fun x y => y = Real.sqrt 2 / 2 * x ∨ y = -Real.sqrt 2 / 2 * x

/-- Theorem stating that the given asymptote equation is correct for the hyperbola -/
theorem hyperbola_asymptotes (a : ℝ) :
  ∀ x y, hyperbola a x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l283_28353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l283_28386

/-- A cubic function with parameters c and d -/
noncomputable def f (c d x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + c * x + d

/-- The derivative of f with respect to x -/
noncomputable def f' (c x : ℝ) : ℝ := x^2 - x + c

/-- Theorem: If f has extreme values, then c < 1/4 -/
theorem extreme_value_condition (c d : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' c x₁ = 0 ∧ f' c x₂ = 0) → c < 1/4 := by
  sorry

#check extreme_value_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_l283_28386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l283_28365

/-- The line l: x - 2y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x - 2*y + 3 = 0

/-- Point M -/
def M : ℝ × ℝ := (2, 5)

/-- Point N -/
def N : ℝ × ℝ := (-2, 4)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The statement to be proved -/
theorem min_distance_theorem :
  (∃ (P : ℝ × ℝ), line_l P.1 P.2 ∧
    (∀ (Q : ℝ × ℝ), line_l Q.1 Q.2 →
      distance P M + distance P N ≤ distance Q M + distance Q N) ∧
    distance P M + distance P N = 3 * Real.sqrt 5) ∧
  (∃ (P : ℝ × ℝ), line_l P.1 P.2 ∧
    (∀ (Q : ℝ × ℝ), line_l Q.1 Q.2 →
      distance P M ^ 2 + distance P N ^ 2 ≤ distance Q M ^ 2 + distance Q N ^ 2) ∧
    distance P M ^ 2 + distance P N ^ 2 = 229 / 10) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l283_28365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_hypotenuse_one_unit_longer_l283_28364

theorem right_triangle_with_hypotenuse_one_unit_longer (k : ℕ) :
  let a : ℤ := (2 * k + 1 : ℕ)
  let b : ℤ := (2 * k * (k + 1) : ℕ)
  let c : ℤ := (2 * k^2 + 2 * k + 1 : ℕ)
  a^2 + b^2 = c^2 ∧ c = b + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_with_hypotenuse_one_unit_longer_l283_28364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_cycle_l283_28382

theorem no_integer_polynomial_cycle (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ¬ ∃ P : Polynomial ℤ, (P.eval a = b) ∧ (P.eval b = c) ∧ (P.eval c = a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_cycle_l283_28382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_a_l283_28335

noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

theorem work_time_a (time_b time_c : ℝ) : 
  time_b = 3 → 
  work_rate time_b + work_rate time_c = 1/2 → 
  work_rate 3 + work_rate time_c = 1/2 → 
  3 = 3 := by
  sorry

#check work_time_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_time_a_l283_28335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l283_28308

/-- A function f(x) with the given properties -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ :=
  4 * Real.cos (3 * x + φ)

/-- Theorem stating the properties and conclusion of the problem -/
theorem problem_statement (φ : ℝ) :
  (|φ| < π / 2) →
  (∀ x, f φ (x + 11 * π / 12) = f φ (11 * π / 12 - x)) →
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < 5 * π / 12 ∧
            0 < x₂ ∧ x₂ < 5 * π / 12 ∧
            x₁ ≠ x₂ →
            f φ x₁ = f φ x₂) →
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 5 * π / 12 ∧
           0 < x₂ ∧ x₂ < 5 * π / 12 ∧
           x₁ ≠ x₂ →
           f φ (x₁ + x₂) = 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l283_28308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_company_max_profits_l283_28370

/-- Represents the financial model of the fishing company -/
structure FishingCompany where
  initial_investment : ℚ
  first_year_expenses : ℚ
  annual_expense_increase : ℚ
  annual_income : ℚ

/-- Calculates the total profit after n years -/
noncomputable def total_profit (company : FishingCompany) (n : ℕ) : ℚ :=
  n * company.annual_income - 
  (company.initial_investment + company.first_year_expenses + 
   (n - 1) * company.annual_expense_increase * n / 2)

/-- Calculates the average profit after n years -/
noncomputable def average_profit (company : FishingCompany) (n : ℕ) : ℚ :=
  (total_profit company n) / n

/-- Theorem stating the existence of maximum total and average profits -/
theorem fishing_company_max_profits 
  (company : FishingCompany) 
  (h_initial : company.initial_investment = 98)
  (h_first_year : company.first_year_expenses = 12)
  (h_increase : company.annual_expense_increase = 4)
  (h_income : company.annual_income = 50) :
  ∃ (n1 n2 : ℕ) (max_total_profit max_average_profit : ℚ),
    (∀ (k : ℕ), total_profit company k ≤ total_profit company n1) ∧
    (total_profit company n1 = max_total_profit) ∧
    (∀ (k : ℕ), average_profit company k ≤ average_profit company n2) ∧
    (average_profit company n2 = max_average_profit) := by
  sorry

#check fishing_company_max_profits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fishing_company_max_profits_l283_28370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shares_owned_proof_l283_28374

/-- Calculates the number of shares owned based on dividend received and earnings information -/
def calculate_shares_owned (expected_earnings : ℚ) (actual_earnings : ℚ) (dividend_received : ℚ) : ℕ :=
  let expected_dividend := expected_earnings / 2
  let additional_earnings := actual_earnings - expected_earnings
  let additional_dividend := (additional_earnings / (1 / 10)) * (4 / 100)
  let total_dividend_per_share := expected_dividend + additional_dividend
  ((dividend_received / total_dividend_per_share).num).natAbs

theorem shares_owned_proof (expected_earnings actual_earnings dividend_received : ℚ) 
  (h1 : expected_earnings = 8 / 10)
  (h2 : actual_earnings = 11 / 10)
  (h3 : dividend_received = 208) :
  calculate_shares_owned expected_earnings actual_earnings dividend_received = 400 := by
  sorry

#eval calculate_shares_owned (8/10) (11/10) 208

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shares_owned_proof_l283_28374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l283_28343

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define membership for points in a circle
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define parallel lines
def Line.isParallel (l1 l2 : Line) : Prop :=
  let (x1, y1) := l1.point1
  let (x2, y2) := l1.point2
  let (x3, y3) := l2.point1
  let (x4, y4) := l2.point2
  (y2 - y1) * (x4 - x3) = (y4 - y3) * (x2 - x1)

-- Define the theorem
theorem circle_intersection_theorem 
  (A B : ℝ × ℝ) 
  (S : Circle) 
  (MN : Line) : 
  ∃ (X : ℝ × ℝ), 
    (S.contains X) ∧ 
    (∃ (C D : ℝ × ℝ), 
      S.contains C ∧ S.contains D ∧
      (Line.mk A X).point2 = C ∧
      (Line.mk B X).point2 = D ∧
      S.contains (Line.mk C D).point2 ∧
      S.contains (Line.mk C D).point1 ∧
      (Line.mk C D).isParallel MN) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l283_28343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l283_28336

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Theorem 1: f is always increasing
theorem f_increasing (a : ℝ) : 
  ∀ x y : ℝ, x < y → f a x < f a y := by
  sorry

-- Theorem 2: f(-x) + f(x) = 0 always holds if and only if a = 1
theorem f_odd_iff_a_eq_one (a : ℝ) : 
  (∀ x : ℝ, f a (-x) + f a x = 0) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_odd_iff_a_eq_one_l283_28336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_running_distance_is_1562_l283_28333

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the triangular race course -/
structure RaceCourse where
  A : Point
  B : Point
  wallStart : Point
  wallEnd : Point
  wallLength : ℝ
  touchPointDistance : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The minimum distance a runner must travel in the race course -/
noncomputable def minRunningDistance (course : RaceCourse) : ℝ :=
  Real.sqrt (course.wallLength^2 + (distance course.A course.wallStart + distance course.B course.wallStart)^2)

/-- The main theorem stating the minimum running distance -/
theorem min_running_distance_is_1562 (course : RaceCourse) :
  course.wallLength = 1000 ∧
  distance course.A course.wallStart = 400 ∧
  distance course.B course.wallStart = 800 ∧
  course.touchPointDistance = 200 →
  Int.floor (minRunningDistance course + 0.5) = 1562 := by
  sorry

#check min_running_distance_is_1562

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_running_distance_is_1562_l283_28333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_correct_l283_28351

/-- The overall population increase after three consecutive percentage increases -/
noncomputable def overall_increase (k l m : ℝ) : ℝ :=
  k + l + m + (k * l + k * m + l * m) / 100 + k * l * m / 10000

/-- Theorem stating that the overall population increase is correct -/
theorem population_increase_correct (k l m : ℝ) :
  let increase1 := 1 + k / 100
  let increase2 := 1 + l / 100
  let increase3 := 1 + m / 100
  (increase1 * increase2 * increase3 - 1) * 100 = overall_increase k l m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_increase_correct_l283_28351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l283_28313

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

-- State the theorem
theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) →  -- f is odd
  f 1 = 2 →                           -- f(1) = 2
  (∀ x : ℝ, x ≠ 0 → f x = x + 1/x) ∧  -- f(x) = x + 1/x for x ≠ 0
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x > f y)  -- f is monotonically decreasing on (0, 1)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l283_28313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_l283_28323

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x + 1

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 2*x + 1/x

-- Define the point of tangency
def point : ℝ × ℝ := (1, 2)

-- Define the slope at the point of tangency
noncomputable def tangent_slope : ℝ := f' point.1

-- Define the equation of the tangent line
noncomputable def tangent_line (x : ℝ) : ℝ := 3*x - 1

-- Theorem statement
theorem tangent_line_correct :
  (f point.1 = point.2) ∧ 
  (tangent_slope = 3) ∧
  (∀ x : ℝ, tangent_line x = tangent_slope * (x - point.1) + point.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_l283_28323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l283_28383

theorem count_integers_in_range : 
  ∃! n : ℕ, n = (Finset.filter (fun x : ℕ => 
    30 < x^2 + 5*x + 10 ∧ x^2 + 5*x + 10 < 60 ∧ x > 0) (Finset.range 6)).card ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_range_l283_28383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_tan_function_l283_28388

noncomputable def f (x φ : ℝ) : ℝ := Real.tan (3 * x + φ)

theorem symmetric_tan_function (φ : ℝ) 
  (h1 : |φ| ≤ π/4) 
  (h2 : ∀ x : ℝ, f x φ = f (-π/9 - (x + π/9)) φ) : 
  f (π/12) φ = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_tan_function_l283_28388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l283_28309

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

-- State the theorem
theorem f_inequality_range :
  {a : ℝ | f (2 - a^2) > f a} = Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l283_28309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_dune_probability_l283_28339

theorem sand_dune_probability : 
  (1 / 3 : ℚ) * (1 / 5 : ℚ) * (2 / 3 : ℚ) = 2 / 45 := by
  sorry

#eval (1 / 3 : ℚ) * (1 / 5 : ℚ) * (2 / 3 : ℚ) == 2 / 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_dune_probability_l283_28339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nonconvex_polyhedron_with_invisible_vertices_l283_28332

/-- A polyhedron in 3D space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  nonEmpty : vertices.Nonempty
  facesValid : ∀ f ∈ faces, f ⊆ vertices

/-- A point is visible from another point if the line segment between them doesn't intersect the polyhedron -/
def isVisible (p : Fin 3 → ℝ) (v : Fin 3 → ℝ) (poly : Polyhedron) : Prop :=
  ∀ x, x ∈ Set.Icc p v → x ∉ poly.vertices ∪ (⋃ f ∈ poly.faces, f)

/-- A polyhedron is convex if the line segment between any two points in the polyhedron is entirely contained within the polyhedron -/
def isConvex (poly : Polyhedron) : Prop :=
  ∀ p q, p ∈ poly.vertices ∪ (⋃ f ∈ poly.faces, f) →
         q ∈ poly.vertices ∪ (⋃ f ∈ poly.faces, f) →
         Set.Icc p q ⊆ poly.vertices ∪ (⋃ f ∈ poly.faces, f)

theorem existence_of_nonconvex_polyhedron_with_invisible_vertices :
  ∃ (poly : Polyhedron) (m : Fin 3 → ℝ), 
    ¬isConvex poly ∧ 
    m ∉ poly.vertices ∪ (⋃ f ∈ poly.faces, f) ∧
    ∀ v ∈ poly.vertices, ¬isVisible m v poly :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nonconvex_polyhedron_with_invisible_vertices_l283_28332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_radius_not_maximal_at_unit_length_l283_28349

noncomputable def triangle_inscribed_radius (base_length : ℝ) : ℝ :=
  let x := base_length / 2
  x * Real.sqrt (1 - x^2) / (1 + x)

noncomputable def tetrahedron_inscribed_radius (base_length : ℝ) : ℝ :=
  let x := base_length / 2
  x * Real.sqrt (1 - 2*x^2) / (Real.sqrt (1 - x^2) + x)

theorem inscribed_radius_not_maximal_at_unit_length :
  ∃ (t_base t_max : ℝ), 0 < t_base ∧ t_base < 1 ∧
    triangle_inscribed_radius t_base < triangle_inscribed_radius t_max ∧
  ∃ (p_base p_max : ℝ), 0 < p_base ∧ p_base < Real.sqrt 2 ∧
    tetrahedron_inscribed_radius p_base < tetrahedron_inscribed_radius p_max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_radius_not_maximal_at_unit_length_l283_28349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_drawings_theorem_l283_28398

/-- Given a number of markers, drawings per marker, and drawings already made,
    calculate the number of additional drawings that can be made. -/
def additional_drawings (markers : ℕ) (drawings_per_marker : ℚ) (drawings_made : ℕ) : ℕ :=
  (((markers : ℚ) * drawings_per_marker - drawings_made) : ℚ).floor.toNat

/-- Theorem stating that with 12 markers, 1.5 drawings per marker, and 8 drawings already made,
    10 additional drawings can be made. -/
theorem anne_drawings_theorem :
  additional_drawings 12 (3/2) 8 = 10 := by
  -- Unfold the definition of additional_drawings
  unfold additional_drawings
  -- Simplify the arithmetic expression
  simp [Nat.cast_ofNat, Int.floor_toNat]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_drawings_theorem_l283_28398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_is_approx_3_3_l283_28326

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a line in 2D space using two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculates the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- The main theorem stating that the length of AE is approximately 3.3 units -/
theorem length_of_AE_is_approx_3_3 : 
  let A : Point := ⟨0, 5⟩
  let B : Point := ⟨5, 0⟩
  let C : Point := ⟨3, 4⟩
  let D : Point := ⟨1, 0⟩
  let AB : Line := ⟨A, B⟩
  let CD : Line := ⟨C, D⟩
  let E : Point := intersectionPoint AB CD
  abs (distance A E - 3.3) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_is_approx_3_3_l283_28326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_line_l283_28341

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define the point that line l passes through
def point_on_l : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem minimal_chord_line :
  ∀ (A B : ℝ × ℝ),
  circle_eq A.1 A.2 →
  circle_eq B.1 B.2 →
  line_l A.1 A.2 →
  line_l B.1 B.2 →
  line_l point_on_l.1 point_on_l.2 →
  (∀ (C D : ℝ × ℝ),
    circle_eq C.1 C.2 →
    circle_eq D.1 D.2 →
    (∃ (m b : ℝ), ∀ (x y : ℝ), y = m*x + b → (x = C.1 ∧ y = C.2) ∨ (x = D.1 ∧ y = D.2)) →
    (C.1 - A.1)^2 + (C.2 - A.2)^2 + (D.1 - B.1)^2 + (D.2 - B.2)^2 ≤
    (C.1 - D.1)^2 + (C.2 - D.2)^2) →
  line_l = λ x y ↦ x - y + 5 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_chord_line_l283_28341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sqrt3_over_2_l283_28355

theorem function_value_at_sqrt3_over_2 :
  ∀ f : ℝ → ℝ, (∀ x, f (Real.sin x) = 1 - 2 * (Real.sin x)^2) →
  f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_sqrt3_over_2_l283_28355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l283_28331

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  apply Set.empty_subset


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l283_28331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l283_28391

theorem cube_root_fraction_equality : 
  (5 / 15.75) ^ (1/3 : ℝ) = (20 : ℝ) ^ (1/3) / (63 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l283_28391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nextNumberIsThree_l283_28389

def mySequence : List Nat := [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2]

def nextNumber (seq : List Nat) : Nat :=
  let maxNum := seq.foldl Nat.max 0
  match seq.getLast? with
  | none => 1
  | some n =>
    if n < maxNum then
      if n + 1 ≤ maxNum then n + 1 else 1
    else 1

theorem nextNumberIsThree :
  nextNumber mySequence = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nextNumberIsThree_l283_28389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_square_l283_28321

/-- Represents a triangle with sidelengths that are square roots of positive integers -/
structure SpecialTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  area_eq : (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2 : ℚ) = 4

/-- Represents a square with area 1/4 -/
def UnitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1/2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1/2}

/-- Represents a folding of a triangle -/
def Folding := Set (ℝ × ℝ) → Set (ℝ × ℝ)

/-- Theorem stating that the special triangle can be folded to cover the unit square -/
theorem triangle_covers_square (t : SpecialTriangle) :
  ∃ (f : Folding), 
    (∀ p ∈ UnitSquare, p ∈ f {x : ℝ × ℝ | x.fst^2 + x.snd^2 ≤ (t.a : ℝ) + (t.b : ℝ) + (t.c : ℝ)}) ∧
    (∀ p ∈ UnitSquare, ∃! q, q ∈ {x : ℝ × ℝ | x.fst^2 + x.snd^2 ≤ (t.a : ℝ) + (t.b : ℝ) + (t.c : ℝ)} ∧ 
      f {q} = {p}) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_square_l283_28321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_tau_inequality_a_is_greatest_l283_28371

/-- Sum of natural divisors of n -/
def sigma (n : ℕ) : ℕ := sorry

/-- Number of natural divisors of n -/
def tau (n : ℕ) : ℕ := sorry

/-- The greatest lower bound for σ(n)/(τ(n)√n) -/
noncomputable def a : ℝ := (3 * Real.sqrt 2) / 4

theorem sigma_tau_inequality (n : ℕ) (hn : n > 1) :
  (sigma n : ℝ) / (tau n : ℝ) ≥ a * Real.sqrt n := by sorry

theorem a_is_greatest : ∀ b : ℝ, (∀ n : ℕ, n > 1 → (sigma n : ℝ) / (tau n : ℝ) ≥ b * Real.sqrt n) → b ≤ a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_tau_inequality_a_is_greatest_l283_28371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_z_type_l283_28311

-- Define the Z-type function property
def is_z_type (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∃ (a b c : ℝ), a < b ∧ c > 0 ∧ Set.Ioo a b ⊆ I ∧
    (∀ x ∈ Set.Ioo a b, -c < f x ∧ f x < c) ∧
    (∀ x ∈ I \ Set.Ioo a b, |f x| = c)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2
  else if x < 3 then 4 - 2*x
  else -2

-- Theorem statement
theorem f_is_z_type : is_z_type f Set.univ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_z_type_l283_28311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l283_28306

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ (x : ℝ) in Set.Icc 0 1, (Real.sqrt (1 - x^2) + x) = π/4 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l283_28306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_set_size_l283_28312

/-- A finite multiset of real numbers -/
def NumSet := Multiset ℝ

/-- The median of a NumSet -/
noncomputable def median (s : NumSet) : ℝ := sorry

/-- The arithmetic mean of a NumSet -/
noncomputable def mean (s : NumSet) : ℝ := sorry

/-- The mode of a NumSet -/
def mode (s : NumSet) : Set ℝ := sorry

/-- A predicate that checks if a NumSet satisfies the given conditions -/
def satisfies_conditions (s : NumSet) : Prop :=
  median s = 3 ∧ mean s = 5 ∧ mode s = {6}

theorem minimum_set_size :
  ∀ s : NumSet, satisfies_conditions s → Multiset.card s ≥ 6 := by
  sorry

#check minimum_set_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_set_size_l283_28312
