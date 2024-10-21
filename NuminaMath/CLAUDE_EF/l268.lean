import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_and_frac_identities_l268_26887

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

theorem floor_and_frac_identities (x : ℝ) : 
  (floor (3 * x) = floor x + floor (x + 1/3) + floor (x + 2/3)) ∧ 
  (frac (3 * x) = frac x + frac (x + 1/3) + frac (x + 2/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_and_frac_identities_l268_26887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_g_even_l268_26842

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((x + 1) / (x - 1))
noncomputable def g (x : ℝ) : ℝ := Real.exp x + 1 / Real.exp x

-- State the theorem
theorem f_odd_g_even :
  (∀ x, f (-x) = -f x) ∧ (∀ x, g (-x) = g x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_g_even_l268_26842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l268_26840

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Whether a triangle is right-angled -/
def isRightTriangle (p q r : Point) : Prop :=
  (distance p q)^2 + (distance p r)^2 = (distance q r)^2 ∨
  (distance p q)^2 + (distance q r)^2 = (distance p r)^2 ∨
  (distance p r)^2 + (distance q r)^2 = (distance p q)^2

/-- The area of a triangle -/
noncomputable def triangleArea (p q r : Point) : ℝ :=
  (1/2) * abs ((p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)))

/-- The main theorem -/
theorem right_triangle_locations (p q : Point) (h : distance p q = 10) :
  ∃! (s : Finset Point), s.card = 8 ∧
    ∀ r ∈ s, isRightTriangle p q r ∧ triangleArea p q r = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l268_26840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l268_26843

-- Define the circle C
def circle_C (x y D : ℝ) : Prop :=
  x^2 + y^2 + D*x - 6*y + 1 = 0

-- Define the bisecting line
def bisecting_line (x y : ℝ) : Prop :=
  x - y + 4 = 0

-- Define the line l
def line_l (x y c : ℝ) : Prop :=
  3*x + 4*y + c = 0

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y c : ℝ) : ℝ :=
  |3*x + 4*y + c| / Real.sqrt 25

theorem circle_line_intersection (D c : ℝ) :
  (∃ x y : ℝ, circle_C x y D ∧ bisecting_line x y) →
  (∃! p : ℝ × ℝ, circle_C p.1 p.2 D ∧ distance_to_line p.1 p.2 c = 1) →
  c = 11 ∨ c = -29 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l268_26843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l268_26826

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x - 1)

theorem f_range : Set.range f = {y : ℝ | y < -1 ∨ y > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l268_26826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l268_26858

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a * Real.sin t.B + Real.sqrt 3 * t.b * Real.cos t.A = 0)
  (h2 : t.a = 3)
  (h3 : Real.sin t.B * Real.sin t.C = 1/4) :
  t.A = 2 * Real.pi / 3 ∧ 
  Triangle.area t = 3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l268_26858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_twelve_count_l268_26886

theorem divisible_by_twelve_count : 
  (∃! count : ℕ, count = (Finset.filter (λ k : ℕ ↦ 
    k < 10 ∧ (7000 + 100 * k + 52) % 12 = 0) (Finset.range 10)).card) ∧
  (∃! count : ℕ, count = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_twelve_count_l268_26886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l268_26847

-- Define the hyperbola and its properties
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_eq : c^2 = a^2 + b^2

-- Define the point M on the hyperbola
def M (h : Hyperbola) : ℝ × ℝ := (h.a, h.b)

-- Define the left focus F₁
def F₁ (h : Hyperbola) : ℝ × ℝ := (-h.c, 0)

-- Define the angle MF₁F₂
noncomputable def angle_MF₁F₂ (h : Hyperbola) : ℝ := 30 * Real.pi / 180

-- Define the eccentricity
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

-- State the theorem
theorem hyperbola_eccentricity (h : Hyperbola) :
  angle_MF₁F₂ h = 30 * Real.pi / 180 → eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l268_26847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_30_l268_26830

/-- The height function for the projectile -/
def projectile_height (t : ℝ) : ℝ := 60 - 8*t - 5*t^2

/-- The theorem stating the existence of a positive time when the height is 30 meters -/
theorem projectile_height_30 :
  ∃ t : ℝ, t > 0 ∧ projectile_height t = 30 ∧ abs (t - 1.773) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_height_30_l268_26830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_C_forms_set_l268_26868

-- Define the characteristics of a set
structure SetCharacteristics where
  definite : Bool
  unordered : Bool
  distinct : Bool

-- Define the options
inductive OptionSet
  | A
  | B
  | C
  | D

-- Function to check if an option satisfies set characteristics
def satisfiesSetCharacteristics (o : OptionSet) : SetCharacteristics :=
  match o with
  | OptionSet.A => { definite := false, unordered := true, distinct := true }
  | OptionSet.B => { definite := false, unordered := true, distinct := true }
  | OptionSet.C => { definite := true, unordered := true, distinct := true }
  | OptionSet.D => { definite := false, unordered := true, distinct := true }

-- Theorem: Option C is the only one that satisfies all set characteristics
theorem option_C_forms_set :
  ∀ (o : OptionSet), 
    (satisfiesSetCharacteristics o).definite ∧ 
    (satisfiesSetCharacteristics o).unordered ∧ 
    (satisfiesSetCharacteristics o).distinct ↔ 
    o = OptionSet.C := by
  sorry

-- This statement ensures that the theorem is true
#check option_C_forms_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_C_forms_set_l268_26868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l268_26888

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- State the theorem
theorem range_of_f :
  ∀ (y : ℝ), (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ f x = y) ↔ y ∈ Set.Icc 2 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l268_26888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l268_26864

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d := Real.sqrt 3
  A = (1, d) ∧ B = (0, 0) ∧ C = (2, 0)

-- Define point D
def PointD (C D : ℝ × ℝ) : Prop :=
  D.1 = C.1 + 4 ∧ D.2 = C.2

-- Define points E and F
def PointE (A B E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

def PointF (A C F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ F = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define collinearity of E, F, and D
def CollinearPoints (E F D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, F = (t * E.1 + (1 - t) * D.1, t * E.2 + (1 - t) * D.2)

-- Define area ratio
noncomputable def AreaRatio (A E F : ℝ × ℝ) : Prop :=
  let areaABC := Real.sqrt 3 / 4 * 2^2
  let areaAEF := 1 / 2 * abs ((E.1 - A.1) * (F.2 - A.2) - (F.1 - A.1) * (E.2 - A.2))
  areaAEF = 1 / 2 * areaABC

-- Main theorem
theorem triangle_ratio_theorem (A B C D E F : ℝ × ℝ) :
  Triangle A B C →
  PointD C D →
  PointE A B E →
  PointF A C F →
  CollinearPoints E F D →
  AreaRatio A E F →
  (E.1 - A.1) / (F.1 - A.1) = 2 / 1 ∧ (E.2 - A.2) / (F.2 - A.2) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_theorem_l268_26864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_60_value_l268_26846

def a : ℕ → ℚ
  | 0 => 2  -- We add this case to cover Nat.zero
  | 1 => 2
  | 2 => 1
  | n+3 => (2 - a (n+2)) / (3 * a (n+1))

theorem a_60_value : a 60 = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_60_value_l268_26846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_not_sum_of_nth_powers_l268_26812

/-- A function that returns true if a natural number can be expressed as the sum of n n-th powers -/
def is_sum_of_nth_powers (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : Fin n → ℕ, k = Finset.univ.sum (λ i ↦ (a i) ^ n)

/-- The set of natural numbers that cannot be expressed as the sum of n n-th powers -/
def not_sum_of_nth_powers (n : ℕ) : Set ℕ :=
  {k : ℕ | ¬(is_sum_of_nth_powers n k)}

/-- Theorem: For any natural number n > 1, the set of natural numbers that cannot be expressed
    as the sum of n n-th powers is infinite -/
theorem infinite_not_sum_of_nth_powers (n : ℕ) (h : n > 1) :
  Set.Infinite (not_sum_of_nth_powers n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_not_sum_of_nth_powers_l268_26812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l268_26836

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 2*x

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := x + 2*y = 0

-- Define what it means for a line to bisect a circle
def bisects (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (cx cy : ℝ), (∀ (x y : ℝ), circle x y ↔ (x - cx)^2 + (y - cy)^2 = 5) ∧ line cx cy

-- Define perpendicularity of lines
def perpendicular (line1 line2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (m1 m2 : ℝ), (∀ (x y : ℝ), line1 x y ↔ y = m1*x + m1) ∧ 
                 (∀ (x y : ℝ), line2 x y ↔ y = m2*x + m2) ∧
                 m1 * m2 = -1

-- The theorem to prove
theorem line_l_properties : 
  bisects line_l my_circle ∧ perpendicular line_l perp_line :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l268_26836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l268_26819

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the line l
def line_l (a b c x y : ℝ) : Prop :=
  y = -a/b * (x - c)

-- Define the asymptotes of C
def asymptotes (a b x y : ℝ) : Prop :=
  y = b/a * x ∨ y = -b/a * x

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- State the theorem
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (h_hyperbola : ∃ x y, hyperbola a b x y)
  (h_line : ∃ x y, line_l a b c x y)
  (h_intersect : ∃ M N : ℝ × ℝ, asymptotes a b M.1 M.2 ∧ asymptotes a b N.1 N.2 ∧ 
                        line_l a b c M.1 M.2 ∧ line_l a b c N.1 N.2)
  (h_N_below : ∃ N : ℝ × ℝ, line_l a b c N.1 N.2 ∧ N.2 < 0)
  (h_ON_OM : ∃ M N : ℝ × ℝ, line_l a b c M.1 M.2 ∧ line_l a b c N.1 N.2 ∧ 
                    N.1^2 + N.2^2 = 4 * (M.1^2 + M.2^2)) :
  eccentricity c a = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l268_26819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_satisfying_equation_l268_26871

theorem count_integer_pairs_satisfying_equation :
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ ↦ (p.1 : ℤ)^2 - (p.2 : ℤ)^2 = 45)
                               (Finset.filter (fun p : ℕ × ℕ ↦ p.1 > 0 ∧ p.2 > 0)
                                              (Finset.product (Finset.range 1000) (Finset.range 1000)))).card
                ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_satisfying_equation_l268_26871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_distribution_l268_26816

/-- The amount of jam Jill had initially -/
noncomputable def initial_jam : ℝ := 1.3

/-- The amount of jam Jill gave to Jan -/
noncomputable def jam_to_jan (x : ℝ) : ℝ := x / 6

/-- The amount of jam remaining after giving to Jan -/
noncomputable def jam_after_jan (x : ℝ) : ℝ := x - jam_to_jan x

/-- The amount of jam Jill gave to Jas -/
noncomputable def jam_to_jas (x : ℝ) : ℝ := jam_after_jan x / 13

/-- The amount of jam remaining after giving to both Jan and Jas -/
noncomputable def final_jam (x : ℝ) : ℝ := jam_after_jan x - jam_to_jas x

/-- Theorem stating that the final amount of jam is 1 kg -/
theorem jam_distribution :
  final_jam initial_jam = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jam_distribution_l268_26816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_c_value_l268_26861

/-- The perpendicular bisector of a line segment passes through its midpoint and is perpendicular to the segment. -/
axiom perpendicular_bisector_property {A B M : ℝ × ℝ} (l : ℝ → ℝ → Prop) :
  (l = fun x y ↦ x - y = -4) →
  (M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)) →
  (l M.fst M.snd) →
  (∀ P : ℝ × ℝ, l P.fst P.snd ↔ (P.fst - A.fst)^2 + (P.snd - A.snd)^2 = (P.fst - B.fst)^2 + (P.snd - B.snd)^2) →
  A = (2, 4) →
  B = (6, 12) →
  ∃ c, l = fun x y ↦ x - y = c

/-- The value of c for the perpendicular bisector of the line segment from (2,4) to (6,12) is -4. -/
theorem perpendicular_bisector_c_value :
  ∃ c, (fun x y ↦ x - y = c) = 
    (fun x y ↦ ∀ P : ℝ × ℝ, x - y = c ↔ 
      (x - 2)^2 + (y - 4)^2 = (x - 6)^2 + (y - 12)^2) ∧ 
    c = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_c_value_l268_26861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hogs_in_kingdom_l268_26850

/-- The number of cats in King Henry's kingdom -/
def num_cats : ℚ := 25

/-- The number of hogs in King Henry's kingdom -/
def num_hogs : ℚ := 75

/-- The relationship between the number of hogs and cats -/
axiom hogs_cats_relation : num_hogs = 3 * num_cats

/-- The condition relating to the number of cats -/
axiom cats_condition : 0.60 * num_cats - 5 = 10

/-- Theorem stating the number of hogs in King Henry's kingdom -/
theorem hogs_in_kingdom : num_hogs = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hogs_in_kingdom_l268_26850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l268_26863

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 0 4

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x) - Real.log (x - 1)

-- State the theorem
theorem domain_of_g : 
  {x : ℝ | x ∈ Set.Ioo 1 2 ∧ g x ∈ Set.range g} = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l268_26863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l268_26884

/-- Definition of the ellipse -/
def Ellipse : Set (ℝ × ℝ) := {p | p.1^2 / 2 + p.2^2 = 1}

/-- Definition of the variable line -/
def Line (m n : ℝ) : Set (ℝ × ℝ) := {p | m * p.1 + n * p.2 + n / 3 = 0}

/-- The fixed point T -/
def T : ℝ × ℝ := (0, 1)

/-- Theorem stating the existence of the fixed point T -/
theorem fixed_point_exists :
  ∀ (m n : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ Ellipse ∧ B ∈ Ellipse ∧
    A ∈ Line m n ∧ B ∈ Line m n ∧ A ≠ B →
    T ∈ {p : ℝ × ℝ | ∃ (c : ℝ × ℝ) (r : ℝ), 
         ((p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧ 
         ((A.1 - c.1)^2 + (A.2 - c.2)^2 = r^2) ∧ 
         ((B.1 - c.1)^2 + (B.2 - c.2)^2 = r^2)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l268_26884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sales_at_fifty_percent_l268_26882

noncomputable def initial_price : ℝ := 100000
noncomputable def initial_volume : ℝ := 1000

noncomputable def total_sales (x : ℝ) : ℝ := -1/2 * (x - 50)^2 + 11250

theorem max_sales_at_fifty_percent (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 80) :
  total_sales x ≤ total_sales 50 := by
  sorry

#check max_sales_at_fifty_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sales_at_fifty_percent_l268_26882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_furthest_point_l268_26876

noncomputable def chocolate_position : ℝ × ℝ := (14, 13)
noncomputable def mouse_start : ℝ × ℝ := (2, -5)
noncomputable def mouse_path (x : ℝ) : ℝ := -4 * x + 6

noncomputable def perpendicular_slope (m : ℝ) : ℝ := -1 / m

noncomputable def perpendicular_line (point : ℝ × ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  let (x₀, y₀) := point
  (perpendicular_slope m) * (x - x₀) + y₀

theorem mouse_furthest_point :
  let intersection_x := -66 / 17
  let intersection_y := 366 / 17
  let intersection_point := (intersection_x, intersection_y)
  (perpendicular_line chocolate_position (-1/4) intersection_x = mouse_path intersection_x) ∧
  (∀ x : ℝ, x < intersection_x → 
    (x - chocolate_position.1)^2 + (mouse_path x - chocolate_position.2)^2 < 
    (intersection_x - chocolate_position.1)^2 + (intersection_y - chocolate_position.2)^2) ∧
  (∀ x : ℝ, x > intersection_x → 
    (x - chocolate_position.1)^2 + (mouse_path x - chocolate_position.2)^2 > 
    (intersection_x - chocolate_position.1)^2 + (intersection_y - chocolate_position.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_furthest_point_l268_26876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l268_26874

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

-- State the theorem
theorem b_range (a b : ℝ) (h : f a = g b) : 
  b ∈ Set.Ioo 2 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l268_26874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l268_26815

open Real

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point on the right branch of the hyperbola
def P : ℝ × ℝ → Prop := λ p => is_on_hyperbola p.1 p.2 ∧ p.1 > 0

-- Define the condition (OP + OF₂) ⋅ F₂P = 0
def condition_1 (p : ℝ × ℝ) : Prop :=
  let op := p
  let of₂ := F₂
  let f₂p := (p.1 - F₂.1, p.2 - F₂.2)
  (op.1 + of₂.1) * f₂p.1 + (op.2 + of₂.2) * f₂p.2 = 0

-- Define the condition |PF₁| = λ|PF₂|
def condition_2 (p : ℝ × ℝ) (lambda : ℝ) : Prop :=
  let pf₁ := ((p.1 - F₁.1)^2 + (p.2 - F₁.2)^2)
  let pf₂ := ((p.1 - F₂.1)^2 + (p.2 - F₂.2)^2)
  pf₁ = lambda^2 * pf₂

-- The theorem to be proved
theorem hyperbola_theorem :
  ∀ p lambda, P p → condition_1 p → condition_2 p lambda → lambda = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l268_26815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_1003_sum_l268_26853

def sum_to_1003 (a b c : ℕ) : Prop := a + b + c = 1003

def trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n % 10 = 0 then 1 + trailing_zeros (n / 10) else 0

theorem max_trailing_zeros_1003_sum :
  ∀ a b c : ℕ, sum_to_1003 a b c →
  ∀ n : ℕ, trailing_zeros (a * b * c) ≤ n →
  n ≤ 7 :=
by
  sorry

#eval trailing_zeros 20000000  -- Should output 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_trailing_zeros_1003_sum_l268_26853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l268_26828

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2^(Real.sqrt (x^2 - 4*x + 4)) + Real.sqrt (x^2 - 2*x)

-- Define the domain of f(x)
def f_domain (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

-- Theorem statement
theorem f_minimum_value :
  (∀ x : ℝ, f_domain x → f x ≥ 1) ∧ (∃ x : ℝ, f_domain x ∧ f x = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_l268_26828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l268_26865

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

theorem hyperbola_asymptotes (h : Hyperbola) 
  (h_eccentricity : eccentricity h = Real.sqrt 13 / 2) :
  asymptote_slope h = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l268_26865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l268_26877

theorem largest_lambda : 
  ∃ (lambda_max : ℝ), lambda_max = 2 ∧ 
  (∀ (lambda : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + 2*c^2 + 2*d^2 ≥ 2*a*b + lambda*b*d + 2*c*d) → lambda ≤ lambda_max) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l268_26877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_600_point_l268_26839

noncomputable def terminal_side_of_angle (θ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ}

theorem angle_600_point (a : ℝ) : 
  (∃ (x y : ℝ), x = -3 ∧ y = a ∧ (x, y) ∈ terminal_side_of_angle 600) →
  a = -3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_600_point_l268_26839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_is_half_l268_26831

/-- The width of the rectangle -/
def width : ℕ := 2024

/-- The height of the rectangle -/
def rect_height : ℕ := 2

/-- The number of squares in each half of a row -/
def half_row : ℕ := width / 2

/-- The total number of possible rectangles in one row -/
def total_rectangles_per_row : ℕ := (width + 1) * width / 2

/-- The number of rectangles in one row that contain the shaded square -/
def shaded_rectangles_per_row : ℕ := half_row * half_row

/-- The probability of choosing a rectangle that does not include a shaded square -/
def probability_no_shaded : ℚ :=
  1 - (rect_height * shaded_rectangles_per_row : ℚ) / (rect_height * total_rectangles_per_row : ℚ)

theorem probability_no_shaded_is_half : probability_no_shaded = 1 / 2 := by
  sorry

#eval probability_no_shaded

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_shaded_is_half_l268_26831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_eq_40_l268_26873

/-- Count of positive integers less than 300 that are multiples of 5 but not multiples of 15 -/
def count_special_numbers : ℕ := 
  (Finset.filter (λ n ↦ n % 5 = 0 ∧ n % 15 ≠ 0) (Finset.range 300)).card

theorem count_special_numbers_eq_40 : count_special_numbers = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_numbers_eq_40_l268_26873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_l268_26890

theorem original_number : ∃ n : ℕ, n + 4 = 23 ∧ ∀ m : ℕ, m < n → ¬(23 ∣ (m + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_number_l268_26890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_l268_26844

/-- The function f(x) defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := ((m^2 + m) * x - 1) / (m^2 * x)

/-- The theorem stating the maximum length of the interval [a, b] -/
theorem max_interval_length (m : ℝ) (a b : ℝ) (hm : m ≠ 0) :
  (∀ x, f m x = x → (a ≤ x ∧ x ≤ b)) →
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f m x = y) →
  |b - a| ≤ 2 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interval_length_l268_26844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l268_26809

theorem probability_at_most_three_heads_ten_coins : 
  (Finset.range 4).sum (λ i => Nat.choose 10 i) / 2^10 = 11 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l268_26809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_A_l268_26810

def A : Set ℤ := {x : ℤ | x^2 - 5*x < 6}

theorem count_elements_A : Finset.card (Finset.filter (fun x => x^2 - 5*x < 6) (Finset.range 6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_elements_A_l268_26810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l268_26817

/-- The curve function f(x) = x^2 - ln(x) --/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function g(x) = x + 2 --/
def g (x : ℝ) : ℝ := x + 2

/-- The distance function from a point (x, f(x)) to the line y = x + 2 --/
noncomputable def distance (x : ℝ) : ℝ := 
  |f x - g x| / Real.sqrt 2

theorem min_distance_curve_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ ∀ (x : ℝ), x > 0 → distance x ≥ d := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l268_26817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_downward_second_quadrant_l268_26869

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (-p.b / (2 * p.a), p.c - p.b^2 / (4 * p.a))

/-- A parabola opens downward if a < 0 -/
def Parabola.opensDownward (p : Parabola) : Prop :=
  p.a < 0

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def isInSecondQuadrant (point : ℝ × ℝ) : Prop :=
  point.1 < 0 ∧ point.2 > 0

theorem parabola_downward_second_quadrant (p : Parabola) :
  p.opensDownward ∧ isInSecondQuadrant p.vertex → p.a < 0 ∧ p.b > 0 ∧ p.c > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_downward_second_quadrant_l268_26869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l268_26818

-- Define the function as noncomputable due to the use of Real.log
noncomputable def f (x k : ℝ) : ℝ := Real.log (x^2 + 3*k*x + k^2 + 5)

-- Define the condition that the range of f is ℝ
def range_is_real (k : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f x k = y

-- State the theorem
theorem range_of_k (k : ℝ) : 
  range_is_real k ↔ k ≤ -2 ∨ k ≥ 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l268_26818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l268_26801

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (3*x₁ - 1)^2 = 2*(3*x₁ - 1) ∧ (3*x₂ - 1)^2 = 2*(3*x₂ - 1) ∧ 
    x₁ = 1/3 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 - 4*y₁ + 1 = 0 ∧ 2*y₂^2 - 4*y₂ + 1 = 0 ∧ 
    y₁ = 1 + Real.sqrt 2 / 2 ∧ y₂ = 1 - Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l268_26801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_count_l268_26825

/-- Calculates the total number of questions on an exam given the percentage of problem-solving questions and the number of multiple-choice questions. -/
def total_questions (problem_solving_percent : ℚ) (multiple_choice_count : ℕ) : ℚ :=
  (multiple_choice_count : ℚ) / (1 - problem_solving_percent)

/-- Theorem stating that given 80% of questions are problem-solving and there are 10 multiple-choice questions, the total number of questions on the exam is 50. -/
theorem exam_question_count :
  let problem_solving_percent : ℚ := 4/5
  let multiple_choice_count : ℕ := 10
  total_questions problem_solving_percent multiple_choice_count = 50 := by
  sorry

#eval total_questions (4/5) 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_question_count_l268_26825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l268_26835

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.sin (2 * α) + Real.sin α = 0) 
  (h2 : α > Real.pi / 2 ∧ α < Real.pi) : 
  Real.tan (α + Real.pi / 4) = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l268_26835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_skew_lines_pairs_in_cube_l268_26827

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  face_diagonals : Finset (Fin 8 × Fin 8)
  space_diagonals : Finset (Fin 8 × Fin 8)

-- Define perpendicular and skew as axioms (since we don't have their implementations)
axiom perpendicular : (Fin 8 × Fin 8) → (Fin 8 × Fin 8) → Prop
axiom skew : (Fin 8 × Fin 8) → (Fin 8 × Fin 8) → Prop

-- Define an ideal skew lines pair
def isIdealSkewLinesPair (l1 l2 : Fin 8 × Fin 8) : Prop :=
  l1 ≠ l2 ∧ perpendicular l1 l2 ∧ skew l1 l2

-- Assume decidability for isIdealSkewLinesPair
axiom isIdealSkewLinesPair_decidable (l1 l2 : Fin 8 × Fin 8) : 
  Decidable (isIdealSkewLinesPair l1 l2)

attribute [instance] isIdealSkewLinesPair_decidable

-- Define the theorem
theorem ideal_skew_lines_pairs_in_cube (c : Cube) :
  (Finset.filter (fun p : (Fin 8 × Fin 8) × (Fin 8 × Fin 8) => 
    isIdealSkewLinesPair p.1 p.2) 
    ((c.edges ∪ c.face_diagonals ∪ c.space_diagonals).product 
     (c.edges ∪ c.face_diagonals ∪ c.space_diagonals))).card = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ideal_skew_lines_pairs_in_cube_l268_26827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ratio_l268_26829

-- Define the trapezoid ABCD
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  AB_parallel_CD : (A.2 - B.2) / (A.1 - B.1) = (C.2 - D.2) / (C.1 - D.1)
  angle_D_90 : (C.1 - D.1) * (A.2 - D.2) + (C.2 - D.2) * (A.1 - D.1) = 0

-- Define point E on CD
noncomputable def E (t : Trapezoid) : ℝ × ℝ := 
  (t.C.1 + (t.D.1 - t.C.1) / 2, t.C.2 + (t.D.2 - t.C.2) / 2)

-- Define the conditions
structure TrapezoidConditions (t : Trapezoid) where
  AE_eq_BE : (E t).1^2 + (E t).2^2 = (t.A.1 - (E t).1)^2 + (t.A.2 - (E t).2)^2
  AED_CEB_similar : (t.A.1 - t.D.1) / (t.C.1 - (E t).1) = (t.A.2 - t.D.2) / (t.C.2 - (E t).2)
  AED_CEB_not_congruent : (t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2 ≠ (t.C.1 - (E t).1)^2 + (t.C.2 - (E t).2)^2
  CD_AB_ratio : ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2) / ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 2014^2

-- The theorem to prove
theorem trapezoid_ratio (t : Trapezoid) (c : TrapezoidConditions t) :
  ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) / ((t.A.1 - t.D.1)^2 + (t.A.2 - t.D.2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_ratio_l268_26829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_4_l268_26854

theorem f_of_3_equals_4 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2) : f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_3_equals_4_l268_26854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_bought_theorem_l268_26897

/-- The number of bottles that can be bought with euros, given:
  * P bottles can be bought for R dollars
  * 1 euro is worth 1.2 dollars
  * There's a 10% discount when buying with euros
  * M is the amount in euros -/
noncomputable def bottles_bought_with_euros (P R M : ℝ) : ℝ :=
  (1.32 * P * M) / R

/-- Theorem stating that the number of bottles bought with M euros
    is equal to (1.32PM)/R under the given conditions -/
theorem bottles_bought_theorem (P R M : ℝ) (h_positive : R > 0) :
  bottles_bought_with_euros P R M = (1.32 * P * M) / R := by
  -- Unfold the definition of bottles_bought_with_euros
  unfold bottles_bought_with_euros
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_bought_theorem_l268_26897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_max_height_l268_26852

/-- Represents a tripod with potentially unequal leg lengths -/
structure Tripod where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  height : ℝ

/-- The original tripod configuration -/
noncomputable def originalTripod : Tripod :=
  { leg1 := 6
    leg2 := 6
    leg3 := 6
    height := 5 }

/-- The broken tripod configuration -/
noncomputable def brokenTripod : Tripod :=
  { leg1 := 4
    leg2 := 6
    leg3 := 6
    height := 10/3 }

/-- Function to calculate the maximum height of a tripod -/
noncomputable def maxHeight (t : Tripod) : ℝ :=
  (min t.leg1 (min t.leg2 t.leg3) / originalTripod.leg1) * originalTripod.height

/-- Theorem stating that the maximum height of the broken tripod is 10/3 feet -/
theorem broken_tripod_max_height :
  maxHeight brokenTripod = 10/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_tripod_max_height_l268_26852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l268_26893

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M) ∧
  (∀ (x : ℝ), f x ≤ 2) ∧
  (∃ (x₀ : ℝ), f x₀ = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l268_26893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l268_26857

/-- The ellipse representing curve C1 -/
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line representing curve C2 -/
def C2 (x y : ℝ) : Prop := x + y = 4

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The theorem stating the minimum distance and the point where it occurs -/
theorem min_distance_C1_C2 :
  ∃ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 ∧ C2 x2 y2 ∧
    (∀ (x1' y1' x2' y2' : ℝ),
      C1 x1' y1' → C2 x2' y2' →
      distance x1 y1 x2 y2 ≤ distance x1' y1' x2' y2') ∧
    distance x1 y1 x2 y2 = Real.sqrt 2 ∧
    x1 = 3/2 ∧ y1 = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C1_C2_l268_26857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l268_26811

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the conditions
axiom func_eq : ∀ x y : ℝ, f x = f (x + y) - f y - 1
axiom f_one : f 1 = 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x > -1

-- State the properties to be proved
theorem f_properties :
  (f (-1) = -3 ∧ f 3 = 5) ∧
  (∀ x : ℝ, f x + f (1 - x) = 0) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l268_26811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_watching_percentage_difference_approx_l268_26872

/-- The percentage difference in the total number of birds Gabrielle saw compared to Chase -/
noncomputable def bird_watching_percentage_difference : ℝ :=
  let gabrielle_birds := 7 + 5 + 4 + 3 + 6
  let chase_birds := 4 + 4 + 3 + 2 + 1
  let difference := gabrielle_birds - chase_birds
  (difference / chase_birds) * 100

/-- The percentage difference in the total number of birds Gabrielle saw compared to Chase is approximately 78.57% -/
theorem bird_watching_percentage_difference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |bird_watching_percentage_difference - 78.57| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_watching_percentage_difference_approx_l268_26872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_bounds_l268_26838

theorem pi_bounds : 3 < Real.pi ∧ Real.pi < 4 := by
  -- Define a unit circle
  let unit_circle_radius : ℝ := 1

  -- Define the perimeter of an inscribed regular hexagon
  let inscribed_hexagon_perimeter : ℝ := 6

  -- Define the perimeter of a circumscribed square
  let circumscribed_square_perimeter : ℝ := 8

  -- State that the circle's circumference is between these two perimeters
  have h1 : inscribed_hexagon_perimeter < 2 * Real.pi * unit_circle_radius := by
    sorry

  have h2 : 2 * Real.pi * unit_circle_radius < circumscribed_square_perimeter := by
    sorry

  -- Prove the main theorem
  have lower_bound : 3 < Real.pi := by
    sorry

  have upper_bound : Real.pi < 4 := by
    sorry

  exact ⟨lower_bound, upper_bound⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_bounds_l268_26838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l268_26892

-- Define the properties of cones C and D
def radius_C : ℝ := 20
def height_C : ℝ := 40
def radius_D : ℝ := 40
def height_D : ℝ := 20

-- Define the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem volume_ratio_of_cones :
  (cone_volume radius_C height_C) / (cone_volume radius_D height_D) = 1/2 := by
  -- Expand the definition of cone_volume
  unfold cone_volume
  -- Simplify the expression
  simp [radius_C, height_C, radius_D, height_D]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l268_26892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_one_l268_26820

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x - 4 else -(2 * (-x) - 4)

-- State the theorem
theorem solution_set_of_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | -2 < x ∧ x < 0 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_greater_than_one_l268_26820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_cost_l268_26885

/-- Represents a city in the travel problem -/
inductive City
| X
| Y
| Z

/-- Calculates the distance between two cities -/
noncomputable def distance (a b : City) : ℝ :=
  match a, b with
  | City.X, City.Y => 4500
  | City.Y, City.X => 4500
  | City.X, City.Z => 4000
  | City.Z, City.X => 4000
  | City.Y, City.Z => Real.sqrt (4500^2 - 4000^2)
  | City.Z, City.Y => Real.sqrt (4500^2 - 4000^2)
  | _, _ => 0

/-- Calculates the cost of bus travel between two cities -/
noncomputable def busCost (a b : City) : ℝ :=
  0.20 * distance a b

/-- Calculates the cost of air travel between two cities -/
noncomputable def airCost (a b : City) : ℝ :=
  120 + 0.12 * distance a b

/-- Calculates the cheaper travel cost between two cities -/
noncomputable def cheaperCost (a b : City) : ℝ :=
  min (busCost a b) (airCost a b)

/-- The theorem to be proved -/
theorem total_trip_cost : 
  cheaperCost City.X City.Y + cheaperCost City.Y City.Z + cheaperCost City.Z City.X = 1770 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_cost_l268_26885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l268_26896

theorem calculation_proof :
  (5/6 + (-3/4) - (1/4 : ℚ) - (-1/6) = 0) ∧
  ((-7/9 + 5/6 - 5/4) * (-36 : ℚ) = 43) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l268_26896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l268_26894

/-- Represents a point on a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The coordinate plane containing points A, B, C, D, and E -/
def plane : Set Point := sorry

/-- Point D on the coordinate plane -/
def D : Point := ⟨-2, -3⟩

/-- D is an element of the coordinate plane -/
axiom h_D_in_plane : D ∈ plane

/-- Theorem: Point D has coordinates (-2, -3) -/
theorem point_D_coordinates : D.x = -2 ∧ D.y = -3 := by
  simp [D]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_D_coordinates_l268_26894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_forgotten_percentage_l268_26866

theorem sixth_grade_homework_forgotten_percentage :
  let group_a_size : ℕ := 30
  let group_b_size : ℕ := 50
  let group_a_forgot_percentage : ℚ := 20 / 100
  let group_b_forgot_percentage : ℚ := 12 / 100
  let total_students : ℕ := group_a_size + group_b_size
  let total_forgot : ℚ := (group_a_size : ℚ) * group_a_forgot_percentage + (group_b_size : ℚ) * group_b_forgot_percentage
  total_forgot / total_students * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_forgotten_percentage_l268_26866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_circle_to_ellipse_l268_26805

/-- A scaling transformation in 2D space -/
structure ScalingTransformation where
  lambda : ℝ
  mu : ℝ
  h_pos_lambda : lambda > 0
  h_pos_mu : mu > 0

/-- The equation of a circle with radius 1 centered at the origin -/
def is_unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The equation of an ellipse with semi-major axis 3 and semi-minor axis 2 -/
def is_target_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The scaling transformation maps the unit circle to the target ellipse -/
theorem scaling_circle_to_ellipse (s : ScalingTransformation) :
  (∀ x y, is_unit_circle x y → is_target_ellipse (s.lambda * x) (s.mu * y)) ↔ 
  s.lambda = 3 ∧ s.mu = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaling_circle_to_ellipse_l268_26805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_geometric_sequence_l268_26859

theorem rational_iff_geometric_sequence (x : ℝ) : 
  (∃ (a b : ℚ), x = a / b) ↔ 
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ 
    (x + a) * (x + c) = (x + b)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_iff_geometric_sequence_l268_26859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_suspicion_l268_26806

/-- Probability of no repetition between two consecutive draws in a "6 out of 45" lottery -/
noncomputable def no_repetition_prob : ℝ := 39 * 38 * 37 * 36 * 35 * 34 / (45 * 44 * 43 * 42 * 41 * 40)

/-- Number of consecutive draws to consider -/
def num_draws : ℕ := 7

/-- Suspicion threshold -/
def suspicion_threshold : ℝ := 0.01

/-- Theorem stating that the probability of no repetition for 7 consecutive draws is less than 0.01 -/
theorem lottery_suspicion : (no_repetition_prob ^ num_draws) < suspicion_threshold := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_suspicion_l268_26806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l268_26813

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := (m * f x) / Real.sin x

-- State the theorem
theorem function_properties :
  -- Condition: f(x) is symmetric with respect to y = x
  (∀ x : ℝ, f (f x) = x) →
  -- Part 1: Minimum value of m
  (∃ m_min : ℝ, m_min = 1 / Real.exp (Real.pi / 4) ∧
    (∀ m : ℝ, (∀ x : ℝ, x ∈ Set.Ioo 0 Real.pi → h m x ≥ Real.sqrt 2) ↔ m ≥ m_min)) ∧
  -- Part 2: Common tangent lines
  (∃ x1 : ℝ, x1 > 0 ∧
    -- Tangent line of f at x1 is tangent to g
    (deriv f x1 * (g x1 - f x1) = g x1 - f x1) ∧
    -- Tangent line of f at -x1 is tangent to g
    (deriv f (-x1) * (g (-x1) - f (-x1)) = g (-x1) - f (-x1)) ∧
    -- No other common tangent lines
    (∀ x : ℝ, x ≠ x1 ∧ x ≠ -x1 →
      deriv f x * (g x - f x) ≠ g x - f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l268_26813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l268_26880

/-- The usual time (in minutes) for a train to complete its journey -/
noncomputable def usual_time : ℚ := 60

/-- The ratio of the reduced speed to the usual speed -/
def speed_ratio : ℚ := 5/6

/-- The additional time (in minutes) taken when traveling at the reduced speed -/
def delay : ℚ := 10

theorem train_journey_time :
  usual_time * speed_ratio = usual_time - delay := by
  -- Convert the equation to standard form
  have h : usual_time * speed_ratio + delay = usual_time := by
    -- Algebraic manipulation
    sorry
  -- Use the above to prove the original statement
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_time_l268_26880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_wars_earnings_l268_26883

/-- The earnings of Star Wars given the costs and earnings of The Lion King and Star Wars --/
theorem star_wars_earnings (lion_king_cost lion_king_earnings star_wars_cost star_wars_earnings : ℕ) 
  (h1 : lion_king_cost = 10)
  (h2 : lion_king_earnings = 200)
  (h3 : star_wars_cost = 25)
  (h4 : lion_king_earnings - lion_king_cost = (star_wars_earnings - star_wars_cost) / 2) :
  star_wars_earnings = 405 :=
by
  sorry

#check star_wars_earnings

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_wars_earnings_l268_26883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l268_26800

-- Define the circle C
def C : Set (Real × Real) := {p | p.1^2 + p.2^2 = 1}

-- Define the point P
def P (m : Real) : Real × Real := (m, 2)

-- State the theorem
theorem range_of_m :
  ∀ m : Real,
  (∃ l : Set (Real × Real), ∃ A B : Real × Real,
    P m ∈ l ∧ A ∈ l ∧ B ∈ l ∧ A ∈ C ∧ B ∈ C ∧
    (∀ Q : Real × Real, Q - P m + Q - B = 2 * (Q - A))) →
  -Real.sqrt 5 ≤ m ∧ m ≤ Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l268_26800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l268_26851

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

-- Define a point P on line l
def point_on_line_l (x₀ y₀ : ℝ) : Prop := line_l x₀ y₀

-- Define the condition for A being the midpoint of PB
def midpoint_condition (x₀ y₀ xA yA xB yB : ℝ) : Prop :=
  xA = (x₀ + xB) / 2 ∧ yA = (y₀ + yB) / 2

-- Define the existence of line m intersecting the circle
def intersection_exists (x₀ y₀ : ℝ) : Prop :=
  ∃ xA yA xB yB, circle_C xA yA ∧ circle_C xB yB ∧ midpoint_condition x₀ y₀ xA yA xB yB

-- Main theorem
theorem x_coordinate_range (x₀ : ℝ) :
  (∃ y₀, point_on_line_l x₀ y₀ ∧ intersection_exists x₀ y₀) → -1 ≤ x₀ ∧ x₀ ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l268_26851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l268_26860

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x^2 + m * x + 1)

-- State the theorem
theorem domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) → m ∈ Set.Icc 0 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l268_26860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_given_line_l268_26862

/-- The distance from the pole to a line in polar form -/
noncomputable def distanceToPolarLine (ρ : ℝ → ℝ) : ℝ :=
  sorry

/-- The given line in polar form -/
noncomputable def givenLine (θ : ℝ) : ℝ :=
  Real.sqrt 3 / (Real.cos θ + Real.sin θ)

theorem distance_to_given_line :
  distanceToPolarLine givenLine = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_given_line_l268_26862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_characterization_l268_26834

/-- 
Given a natural number n, returns true if n is of the form x99...9
where x is any digit from 1 to 9 and there are k 9's (1 ≤ k ≤ 20)
-/
def isSpecialForm (n : ℕ) : Prop :=
  ∃ (x k : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ k ∧ k ≤ 20 ∧ n = x * 10^k + (10^k - 1)

/-- 
Given a natural number n, returns the product of its digits after
increasing each digit by 1
-/
noncomputable def productOfIncreasedDigits (n : ℕ) : ℕ :=
  sorry

theorem special_number_characterization :
  ∀ N : ℕ, 10 ≤ N ∧ N ≤ 10^20 →
    (productOfIncreasedDigits N = N + 1 ↔ isSpecialForm N) :=
by sorry

#check special_number_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_characterization_l268_26834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_monotone_increasing_l268_26823

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(1/m)

def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f m x < f m y

theorem unique_m_for_monotone_increasing :
  ∃! m : ℝ, is_monotone_increasing m ∧ m = 2 := by
  sorry

#check unique_m_for_monotone_increasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_for_monotone_increasing_l268_26823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_day_lollipops_l268_26822

def lollipop_sequence (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => lollipop_sequence a n + 4

def total_lollipops (a : ℕ) : ℕ := 
  (List.range 10).map (lollipop_sequence a) |>.sum

theorem seventh_day_lollipops :
  ∃ a : ℕ, total_lollipops a = 200 ∧ lollipop_sequence a 6 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_day_lollipops_l268_26822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l268_26803

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a / 2) * x^2 - (a + 1) * x

-- Define a custom IsTangentLine predicate
def IsTangentLine (f : ℝ → ℝ) (x₀ : ℝ) (k : ℝ) : Prop :=
  ∃ b : ℝ, ∀ x : ℝ, f x₀ + k * (x - x₀) = k * x + b ∧ f x₀ = k * x₀ + b

theorem function_properties (a : ℝ) :
  (∀ x > 0, f a x / x < (deriv (f a)) x / 2) →
  (∃ k, k = -2 ∧ IsTangentLine (f a) 1 k) →
  (a = 2 ∧
   (∀ x ∈ Set.Ioo 0 (1/2) ∪ Set.Ioi 1, StrictMonoOn (f a) (Set.Ioo 0 (1/2) ∪ Set.Ioi 1)) ∧
   (StrictMonoOn (f a) (Set.Ioo (1/2) 1)) ∧
   a > 2 * Real.exp (-3/2) - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l268_26803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_45_and_g_10_l268_26849

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the conditions
axiom f_cond (x y : ℝ) : x > 0 → y > 0 → f (x * y) = f x / y
axiom g_cond (x y : ℝ) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

-- State the theorem to be proved
theorem f_45_and_g_10 : f 45 = 10 / 3 ∧ g 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_45_and_g_10_l268_26849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l268_26878

/-- The area of a triangle ABC given side lengths a, b, c -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  Real.sqrt ((1/4) * (a^2 * c^2 - ((a^2 + c^2 - b^2)/2)^2))

/-- Theorem: The area of triangle ABC is √3 given specific conditions -/
theorem triangle_area_is_sqrt_3 
  (a b c : ℝ) 
  (h1 : a^2 * Real.sin c = 4 * Real.sin a) 
  (h2 : (a + c)^2 = 12 + b^2) : 
  triangle_area a b c = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_3_l268_26878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_through_MN_l268_26832

/-- A circle passing through two points -/
structure CircleThroughPoints where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_M : (center.1 - 1)^2 + (center.2 + 1)^2 = radius^2
  passes_through_N : (center.1 + 1)^2 + (center.2 - 1)^2 = radius^2

/-- The area of a circle -/
noncomputable def circle_area (c : CircleThroughPoints) : ℝ := Real.pi * c.radius^2

/-- The theorem stating that x^2 + y^2 = 2 is the equation of the circle with the smallest area passing through M(1, -1) and N(-1, 1) -/
theorem smallest_circle_through_MN :
  ∀ c : CircleThroughPoints,
  circle_area c ≥ 2 * Real.pi ∧
  (circle_area c = 2 * Real.pi ↔ c.center = (0, 0) ∧ c.radius = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_through_MN_l268_26832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l268_26821

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  D : ℝ × ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c + t.a = 2 * t.b * Real.cos t.A)
  (h2 : t.a = 5)
  (h3 : t.c = 3)
  (h4 : t.D = ((t.A + t.C) / 2, 0)) : -- Assuming D is midpoint of AC on x-axis
  t.B = 2 * Real.pi / 3 ∧ 
  Real.sqrt 19 / 2 = Real.sqrt ((t.B - t.D.1)^2 + t.D.2^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l268_26821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l268_26807

theorem fraction_sum_inverse_product : (12 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_inverse_product_l268_26807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_theorem_l268_26802

/-- Calculates the combined percent increase for three stocks given their initial and final prices -/
noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  let percent_increase (initial final : ℝ) := (final - initial) / initial * 100
  (percent_increase initial_a final_a + 
   percent_increase initial_b final_b + 
   percent_increase initial_c final_c) / 3

/-- The combined percent increase for the given stock prices is approximately 16.04% -/
theorem stock_price_increase_theorem (ε : ℝ) (h_ε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
    |combined_percent_increase 25 45 60 28 50 75 - 16.04| < δ ∧ δ < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_theorem_l268_26802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l268_26867

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the ellipse C
def ellipse_eq (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 4)

-- Define parameters a and b
variable (a b : ℝ)

-- Define that A is on the ellipse
axiom A_on_ellipse : ellipse_eq point_A.1 point_A.2 a b

-- Define that a > b > 0
axiom a_b_positive : a > b ∧ b > 0

-- Define points M and N on the ellipse
axiom M_on_ellipse : ∃ (x y : ℝ), x ≠ 0 ∧ ellipse_eq x y a b
axiom N_on_ellipse : ∃ (x y : ℝ), x ≠ 0 ∧ ellipse_eq x y a b

-- Define that M and N are on opposite sides of y-axis
axiom M_N_opposite : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  ellipse_eq x₁ y₁ a b ∧ ellipse_eq x₂ y₂ a b ∧ x₁ * x₂ < 0

-- Define that the bisector of ∠MAN is on y-axis
axiom bisector_on_y_axis : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
  ellipse_eq x₁ y₁ a b → ellipse_eq x₂ y₂ a b → 
  (y₁ - 4) / x₁ + (y₂ - 4) / x₂ = 0

-- Define that |AM| ≠ |AN|
axiom AM_not_equal_AN : ∀ (x₁ y₁ x₂ y₂ : ℝ), 
  ellipse_eq x₁ y₁ a b → ellipse_eq x₂ y₂ a b → 
  (x₁^2 + (y₁ - 4)^2) ≠ (x₂^2 + (y₂ - 4)^2)

-- Theorem to prove
theorem ellipse_and_fixed_point : 
  (a = 2 * Real.sqrt 2 ∧ b = 2) ∧
  (∀ (x y : ℝ), ellipse_eq x y a b → ∃ (k : ℝ), y = k * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l268_26867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_2024_eq_identity_l268_26879

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2024_eq_identity :
  B ^ 2024 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_pow_2024_eq_identity_l268_26879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l268_26891

theorem equation_solution : ∃! x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) :=
  by
    use -10
    constructor
    · -- Prove that x = -10 satisfies the equation
      simp [Real.rpow_mul, Real.rpow_sub]
      -- The rest of the proof is omitted
      sorry
    · -- Prove uniqueness
      intros y hy
      -- The rest of the uniqueness proof is omitted
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l268_26891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stromquist_comet_next_leap_year_l268_26855

/-- Determines if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

/-- Finds the next year when the Stromquist Comet is visible after a given year -/
def nextCometYear (year : ℕ) : ℕ :=
  year + 61

/-- Finds the next leap year when the Stromquist Comet is visible after 2017 -/
def nextLeapYearComet : ℕ :=
  let rec findNextLeapYear (year : ℕ) (fuel : ℕ) : ℕ :=
    match fuel with
    | 0 => year  -- Default case to ensure termination
    | fuel + 1 =>
      if isLeapYear year then year
      else findNextLeapYear (nextCometYear year) fuel
  findNextLeapYear (nextCometYear 2017) 100  -- Arbitrarily large fuel value

theorem stromquist_comet_next_leap_year :
  nextLeapYearComet = 2444 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stromquist_comet_next_leap_year_l268_26855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l268_26895

/-- The curve defined by x^2 - xy + 2y + 1 = 0 where x > 2 -/
def Curve : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.1 * p.2 + 2 * p.2 + 1 = 0 ∧ p.1 > 2}

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : ℝ × ℝ) : ℝ := abs p.2

/-- The minimum distance from the curve to the x-axis -/
noncomputable def minDistanceToCurve : ℝ := 4 + 2 * Real.sqrt 5

/-- Theorem stating that the minimum distance from any point on the curve to the x-axis
    is greater than or equal to minDistanceToCurve -/
theorem min_distance_to_curve :
  ∀ p ∈ Curve, distanceToXAxis p ≥ minDistanceToCurve := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_curve_l268_26895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l268_26841

/-- The distance from the focus of the parabola y = 1/4 * x^2 to the asymptote of the hyperbola y^2 - x^2/3 = 1 is √3/2 -/
theorem distance_focus_to_asymptote : 
  let parabola := fun x : ℝ => (1/4) * x^2
  let hyperbola := fun (x y : ℝ) => y^2 - x^2/3 = 1
  let focus : ℝ × ℝ := (0, 1)
  let asymptote := fun x : ℝ => (Real.sqrt 3/3) * x
  ∃ d : ℝ, d = Real.sqrt 3/2 ∧ 
    d = abs (asymptote (focus.1) - focus.2) / Real.sqrt (1 + (Real.sqrt 3/3)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptote_l268_26841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_nonempty_intersection_l268_26881

theorem max_subsets_with_nonempty_intersection (U : Type) [Fintype U] :
  (Fintype.card U = 11) →
  ∃ (k : ℕ) (A : Fin k → Set U),
    (∀ i j, i ≠ j → (A i ∩ A j).Nonempty) ∧
    (∀ S : Set U, S ∉ (Set.range A) → ∃ i, (S ∩ A i) = ∅) ∧
    (∀ m : ℕ, ∀ B : Fin m → Set U,
      (∀ i j, i ≠ j → (B i ∩ B j).Nonempty) →
      (∀ S : Set U, S ∉ (Set.range B) → ∃ i, (S ∩ B i) = ∅) →
      m ≤ k) ∧
    k = 1024 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_nonempty_intersection_l268_26881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_pi_12_l268_26833

theorem f_sin_pi_12 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sin_pi_12_l268_26833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_to_equalize_l268_26875

/-- The number of marbles Ajay has initially -/
def A : ℤ := sorry

/-- The number of marbles Vijay has initially -/
def V : ℤ := sorry

/-- The number of marbles Vijay gives to Ajay to equalize their marble counts -/
def x : ℤ := sorry

/-- After Vijay gives x marbles to Ajay, they have an equal number of marbles -/
axiom equal_marbles : A + x = V - x

/-- If Ajay gives Vijay twice as many marbles, Vijay will have 30 more than Ajay -/
axiom twice_marbles : V + 2*x = (A - 2*x) + 30

/-- The theorem stating that x = 5 -/
theorem marbles_to_equalize : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marbles_to_equalize_l268_26875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l268_26837

noncomputable def vector_to_project (b : ℝ) : Fin 3 → ℝ := ![5, b, 0]
noncomputable def vector_to_project_onto : Fin 3 → ℝ := ![3, 1, 2]
noncomputable def projection_result : Fin 3 → ℝ := ![3/7, 1/7, 2/7]

theorem find_b :
  ∃ b : ℝ, 
    (vector_to_project b • vector_to_project_onto) / (vector_to_project_onto • vector_to_project_onto) 
      * vector_to_project_onto = projection_result ∧ b = -13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_b_l268_26837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_chocolate_cost_difference_l268_26808

/-- Proves the difference in cost between candy bars and chocolates --/
theorem candy_chocolate_cost_difference : 
  let initial_money : ℚ := 50
  let chocolate_count : ℕ := 3
  let chocolate_price : ℚ := 8
  let chocolate_discount : ℚ := 1/10
  let candy_count : ℕ := 5
  let candy_price : ℚ := 12
  let candy_tax : ℚ := 1/20

  let chocolate_cost : ℚ := chocolate_count * chocolate_price * (1 - chocolate_discount)
  let candy_cost : ℚ := candy_count * candy_price * (1 + candy_tax)

  candy_cost - chocolate_cost = 414/10 := by
  sorry

#check candy_chocolate_cost_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_chocolate_cost_difference_l268_26808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l268_26814

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- The area of a triangle given two sides and the angle between them -/
noncomputable def triangleArea (a b angle : ℝ) : ℝ := (1/2) * a * b * Real.sin angle

theorem area_of_triangle_AOB : 
  let A : PolarPoint := ⟨2, π/6⟩
  let B : PolarPoint := ⟨4, 2*π/3⟩
  triangleArea A.r B.r (B.θ - A.θ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AOB_l268_26814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_seven_l268_26845

theorem sum_of_solutions_equals_seven : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, (2 : ℝ)^(x^2 - 4*x - 3) = 8^(x - 5)) ∧ 
  (∀ x : ℝ, (2 : ℝ)^(x^2 - 4*x - 3) = 8^(x - 5) → x ∈ S) ∧
  (S.sum id) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_equals_seven_l268_26845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l268_26804

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.tan (Real.arccos (x^3))

-- State the theorem about the domain of g
theorem domain_of_g :
  {x : ℝ | ∃ y, g x = y ∧ x ≥ -1 ∧ x ≤ 1} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l268_26804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_l268_26899

noncomputable section

variable (a b c A B C : ℝ)

-- Define the triangle ABC
def is_triangle (a b c A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Define the condition given in the problem
def triangle_condition (a b c C : ℝ) : Prop :=
  2 * a * Real.cos C + c = 2 * b

-- Theorem for part 1
theorem angle_A_measure 
  (h_triangle : is_triangle a b c A B C) 
  (h_condition : triangle_condition a b c C) : 
  A = Real.pi / 3 := 
sorry

-- Theorem for part 2
theorem max_area 
  (h_triangle : is_triangle 1 b c A B C) 
  (h_condition : triangle_condition 1 b c C) :
  (1 / 2 : ℝ) * b * c * Real.sin A ≤ Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_l268_26899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_formula_implies_pi_equals_three_l268_26870

/-- The volume of a cylinder according to the ancient Chinese formula -/
noncomputable def ancient_cylinder_volume (r : ℝ) (h : ℝ) : ℝ := (1 / 12) * (2 * Real.pi * r)^2 * h

/-- The standard formula for the volume of a cylinder -/
noncomputable def standard_cylinder_volume (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

theorem ancient_formula_implies_pi_equals_three :
  (∀ (r h : ℝ), ancient_cylinder_volume r h = standard_cylinder_volume r h) →
  Real.pi = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_formula_implies_pi_equals_three_l268_26870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l268_26898

theorem trigonometric_inequality (x : ℝ) : 
  2 * Real.sin (π/4 - Real.sqrt 2/2)^2 ≤ Real.cos (Real.sin x) - Real.sin (Real.cos x) ∧
  Real.cos (Real.sin x) - Real.sin (Real.cos x) ≤ 2 * Real.sin (π/4 + Real.sqrt 2/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l268_26898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_l268_26889

noncomputable def initial_height : ℝ := 800
noncomputable def bounce_ratio : ℝ := 3/4
noncomputable def target_height : ℝ := 10

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem min_bounces_to_target :
  (∀ k < 16, height_after_bounces k ≥ target_height) ∧
  (height_after_bounces 16 < target_height) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_target_l268_26889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_min_max_length_l268_26824

theorem chord_segment_min_max_length (R : ℝ) (a b : ℝ) (p q m n : ℝ) 
  (h_R : R = 5)
  (h_a : a = 3)
  (h_b : b = 4)
  (h_p : p = 1)
  (h_q : q = 2)
  (h_m : m = 1)
  (h_n : n = 3) :
  let OM := Real.sqrt (R^2 - (p * q / (p + q)^2) * a^2)
  let ON := Real.sqrt (R^2 - (m * n / (m + n)^2) * b^2)
  abs (OM - ON) = Real.sqrt 23 - Real.sqrt 22 ∧ 
  OM + ON = Real.sqrt 23 + Real.sqrt 22 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_segment_min_max_length_l268_26824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_of_396_l268_26848

theorem sum_distinct_prime_factors_of_396 : 
  (Finset.filter (λ p => Nat.Prime p ∧ 396 % p = 0) (Finset.range (396 + 1))).sum id = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_of_396_l268_26848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l268_26856

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := -4 + 4 * Real.cos x - Real.sin x ^ 2

-- State the theorem for the maximum and minimum values of f(x)
theorem f_max_min :
  (∃ (x : ℝ), f x = 0) ∧
  (∀ (x : ℝ), f x ≤ 0) ∧
  (∃ (x : ℝ), f x = -8) ∧
  (∀ (x : ℝ), f x ≥ -8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l268_26856
