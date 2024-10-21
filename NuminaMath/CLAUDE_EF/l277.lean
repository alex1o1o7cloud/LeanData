import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l277_27779

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin x * Real.sin (x + Real.pi / 3) - 1

theorem f_properties :
  (f (5 * Real.pi / 6) = -2) ∧
  (∀ A : ℝ, 0 < A ∧ A ≤ Real.pi / 3 → f A = 8 / 5 → f (A + Real.pi / 4) = 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l277_27779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l277_27746

/-- Hyperbola struct representing x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through a point -/
structure Line where
  point : Point
  direction : ℝ × ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def perpendicular (l1 l2 : Line) : Prop :=
  l1.direction.1 * l2.direction.1 + l1.direction.2 * l2.direction.2 = 0

noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity
  (h : Hyperbola)
  (f1 f2 p q : Point)
  (l : Line)
  (h_f2_on_line : l.point = f2)
  (h_p_on_hyperbola : p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1)
  (h_q_on_hyperbola : q.x^2 / h.a^2 - q.y^2 / h.b^2 = 1)
  (h_p_on_line : ∃ t : ℝ, p.x = f2.x + t * l.direction.1 ∧ p.y = f2.y + t * l.direction.2)
  (h_q_on_line : ∃ t : ℝ, q.x = f2.x + t * l.direction.1 ∧ q.y = f2.y + t * l.direction.2)
  (h_pq_perp_pf1 : perpendicular (Line.mk p (q.x - p.x, q.y - p.y)) (Line.mk p (f1.x - p.x, f1.y - p.y)))
  (h_pq_ratio : distance p q = 5/12 * distance p f1) :
  eccentricity h = Real.sqrt 37 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l277_27746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_l277_27755

/-- Number of derangements for n items -/
def derangement (n : ℕ) : ℕ := sorry

/-- The cube vertices -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Adjacent vertices for each vertex of the cube -/
def adjacentVertices : Vertex → List Vertex
| Vertex.A => [Vertex.B, Vertex.D, Vertex.E]
| Vertex.B => [Vertex.A, Vertex.C, Vertex.F]
| Vertex.C => [Vertex.B, Vertex.D, Vertex.G]
| Vertex.D => [Vertex.A, Vertex.C, Vertex.H]
| Vertex.E => [Vertex.A, Vertex.F, Vertex.H]
| Vertex.F => [Vertex.B, Vertex.E, Vertex.G]
| Vertex.G => [Vertex.C, Vertex.F, Vertex.H]
| Vertex.H => [Vertex.D, Vertex.E, Vertex.G]

theorem ant_movement_probability :
  let n : ℕ := 8
  let totalOutcomes : ℕ := 3^n
  let favorableOutcomes : ℕ := derangement n
  (favorableOutcomes : ℚ) / totalOutcomes = (derangement n : ℚ) / 3^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_movement_probability_l277_27755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_oxygen_approx_l277_27708

/-- The mass percentage of oxygen in 2-(Chloromethyl)oxirane (C3H5ClO) -/
noncomputable def mass_percentage_oxygen : ℝ :=
  let atomic_mass_C : ℝ := 12.01
  let atomic_mass_H : ℝ := 1.01
  let atomic_mass_Cl : ℝ := 35.45
  let atomic_mass_O : ℝ := 16.00
  let molar_mass : ℝ := 3 * atomic_mass_C + 5 * atomic_mass_H + atomic_mass_Cl + atomic_mass_O
  (atomic_mass_O / molar_mass) * 100

/-- Theorem: The mass percentage of oxygen in 2-(Chloromethyl)oxirane (C3H5ClO) is approximately 17.29% -/
theorem mass_percentage_oxygen_approx :
  |mass_percentage_oxygen - 17.29| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_oxygen_approx_l277_27708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_with_area_l277_27720

def P : Fin 3 → ℝ := ![2, -3, 1]
def Q : Fin 3 → ℝ := ![4, -7, 4]
def R : Fin 3 → ℝ := ![3, -2, -1]
def S : Fin 3 → ℝ := ![5, -6, 2]

def vector_sub (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => a i - b i

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (v 0^2 + v 1^2 + v 2^2)

theorem quadrilateral_is_parallelogram_with_area (P Q R S : Fin 3 → ℝ) :
  vector_sub Q P = vector_sub S R →
  magnitude (cross_product (vector_sub Q P) (vector_sub R P)) = Real.sqrt 110 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_parallelogram_with_area_l277_27720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_jog_day_l277_27786

/-- Represents a date in January --/
structure JanuaryDate :=
  (day : Nat)
  (is_monday : Bool)
  (is_jog_day : Bool)

/-- The starting date (January 5) --/
def start_date : JanuaryDate :=
  { day := 5, is_monday := true, is_jog_day := true }

/-- Calculates the next jog date --/
def next_jog_date (d : JanuaryDate) : JanuaryDate :=
  { day := d.day + 3,
    is_monday := (d.day + 3 - 5) % 7 == 0,
    is_jog_day := true }

/-- Calculates the next Monday --/
def next_monday (d : JanuaryDate) : JanuaryDate :=
  { day := d.day + (7 - (d.day - 5) % 7) % 7,
    is_monday := true,
    is_jog_day := (d.day + (7 - (d.day - 5) % 7) % 7 - 5) % 3 == 0 }

/-- The theorem to be proved --/
theorem next_monday_jog_day : 
  ∃ (n : Nat), (Nat.iterate next_jog_date n start_date).day = 26 ∧ 
               (Nat.iterate next_jog_date n start_date).is_monday = true := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_monday_jog_day_l277_27786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_when_c_is_22_l277_27772

/-- The sequence a_n defined by the integral of x^2 * (1-x)^n from 0 to 1 -/
noncomputable def a (n : ℕ) : ℝ := ∫ x in (0:ℝ)..1, x^2 * (1-x)^n

/-- The theorem stating that the value of c satisfying the infinite sum equation is 22 -/
theorem sum_equals_two_when_c_is_22 :
  ∃ c : ℝ, (∑' n : ℕ, ((n : ℝ) + c) * (a n - a (n + 1))) = 2 ∧ c = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_when_c_is_22_l277_27772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l277_27734

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ
  B : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 10 ∧ t.A = Real.pi / 6

-- Part 1: Prove B = π/2 when a = 5
theorem part_one (t : Triangle) : 
  triangle_conditions t → t.a = 5 → t.B = Real.pi / 2 := by
  sorry

-- Part 2: Prove B has two solutions when 5 < a < 10
theorem part_two (t : Triangle) : 
  triangle_conditions t → (5 < t.a ∧ t.a < 10) → 
  ∃ B₁ B₂, B₁ ≠ B₂ ∧ Real.sin B₁ = Real.sin B₂ ∧ Real.sin B₁ = (t.b * Real.sin t.A) / t.a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l277_27734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l277_27704

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc 0 1

-- Define the domain of f(2^x-2)
def domain_f_2_pow_x_minus_2 : Set ℝ := Set.Icc (Real.log 3 / Real.log 2) 2

-- Theorem statement
theorem domain_equivalence (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ≠ 0) :
  ∀ x, x ∈ domain_f_2_pow_x_minus_2 ↔ f (2^x - 2) ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l277_27704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_agreement_l277_27723

def Movie := Fin 5

def movie_A : Movie := ⟨0, by norm_num⟩
def movie_B : Movie := ⟨1, by norm_num⟩
def movie_C : Movie := ⟨2, by norm_num⟩
def movie_D : Movie := ⟨3, by norm_num⟩
def movie_E : Movie := ⟨4, by norm_num⟩

def xiao_zhao_set : Set Movie := {m | m ≠ movie_B}
def xiao_zhang_set : Set Movie := {movie_B, movie_C, movie_D, movie_E}
def xiao_li_set : Set Movie := {m | m ≠ movie_C}
def xiao_liu_set : Set Movie := {m | m ≠ movie_E}

theorem movie_agreement :
  xiao_zhao_set ∩ xiao_zhang_set ∩ xiao_li_set ∩ xiao_liu_set = {movie_D} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_agreement_l277_27723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_july_mixture_cost_l277_27748

/-- Represents the cost of the tea and coffee mixture in July -/
noncomputable def mixture_cost (june_cost green_tea_july coffee_july mixture_weight : ℝ) : ℝ :=
  (green_tea_july * mixture_weight / 2) + (coffee_july * mixture_weight / 2)

/-- Theorem stating the cost of the mixture in July -/
theorem july_mixture_cost :
  ∀ (june_cost : ℝ),
  june_cost > 0 →
  let green_tea_july := june_cost * 0.1
  let coffee_july := june_cost * 2
  green_tea_july = 0.1 →
  mixture_cost june_cost green_tea_july coffee_july 3 = 3.15 := by
  intro june_cost h_positive
  have green_tea_july := june_cost * 0.1
  have coffee_july := june_cost * 2
  intro h_green_tea
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check july_mixture_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_july_mixture_cost_l277_27748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l277_27761

open Real
open Set

noncomputable def f (x : ℝ) : ℝ := log (2 * sin x + 1) + sqrt (2 * cos x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 6 < x ∧ x ≤ 2 * k * π + π / 3} =
  {x : ℝ | 2 * sin x + 1 > 0 ∧ 2 * cos x - 1 ≥ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l277_27761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_perimeter_range_l277_27753

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  hABC : A + B + C = π

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

-- Theorem 1: Measure of angle A
theorem angle_A_measure (t : Triangle) (h : condition t) : t.A = 2 * π / 3 := by
  sorry

-- Theorem 2: Range of perimeter when a = √3
theorem perimeter_range (t : Triangle) (h : condition t) (ha : t.a = Real.sqrt 3) :
  2 * Real.sqrt 3 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_perimeter_range_l277_27753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l277_27797

/-- The time difference in minutes between Abel and Alice's arrival at their destination --/
noncomputable def timeDifference (totalDistance : ℝ) (abelSpeed : ℝ) (aliceSpeed : ℝ) (headStart : ℝ) : ℝ :=
  let abelTime := totalDistance / abelSpeed
  let aliceTime := totalDistance / aliceSpeed + headStart
  (aliceTime - abelTime) * 60

/-- Theorem stating that Abel arrives 360 minutes earlier than Alice --/
theorem abel_arrives_earlier :
  timeDifference 1000 50 40 1 = 360 := by
  -- Unfold the definition of timeDifference
  unfold timeDifference
  -- Simplify the arithmetic expressions
  simp [div_eq_mul_inv]
  -- Perform the numerical calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abel_arrives_earlier_l277_27797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l277_27777

/-- A function from real numbers to real numbers -/
def f : ℝ → ℝ := sorry

/-- The transformation applied to f -/
def g (x : ℝ) : ℝ := f (x + 1) - 2

/-- Theorem stating the translation of the graph -/
theorem graph_translation (x y : ℝ) :
  (y = g x) ↔ (y + 2 = f (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l277_27777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_resistor_value_l277_27728

/-- Calculates the combined resistance of two resistors in parallel -/
noncomputable def parallel_resistance (r1 r2 : ℝ) : ℝ :=
  1 / (1 / r1 + 1 / r2)

/-- Theorem: Given two resistors in parallel with R1 = 8 ohms and a combined resistance
    R_total = 4.235294117647059 ohms, the resistance of the second resistor R2
    is approximately 9 ohms. -/
theorem second_resistor_value (r_total : ℝ) (h1 : r_total = 4.235294117647059) :
  ∃ (r2 : ℝ), parallel_resistance 8 r2 = r_total ∧ abs (r2 - 9) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_resistor_value_l277_27728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resolve_ambiguous_grouping_l277_27789

/-- Represents a data point in a dataset -/
structure DataPoint where
  value : Float

/-- Represents a dividing point in a frequency distribution histogram -/
structure DividingPoint where
  value : Float

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  data : List DataPoint
  dividing_points : List DividingPoint

/-- Function to check if a data point exactly matches a dividing point -/
def exactMatch (d : DataPoint) (p : DividingPoint) : Prop :=
  d.value = p.value

/-- Function to check if a dividing point has one more decimal place than a data point -/
def hasOneMoreDecimalPlace (d : DataPoint) (p : DividingPoint) : Prop :=
  (p.value * 10).floor ≠ (p.value * 10)

/-- Relation to represent that a data point belongs to a group -/
def belongs_to_group (d : DataPoint) (g : ℕ) : Prop :=
  sorry -- Definition of group assignment

/-- Theorem stating that using dividing points with one more decimal place
    resolves ambiguous group assignment in frequency distribution histograms -/
theorem resolve_ambiguous_grouping (h : FrequencyHistogram) :
  (∀ d ∈ h.data, ∀ p ∈ h.dividing_points,
    hasOneMoreDecimalPlace d p → ¬exactMatch d p) →
  (∀ d ∈ h.data, ∃! g : ℕ, belongs_to_group d g) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resolve_ambiguous_grouping_l277_27789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_when_a_neg_one_g_is_min_of_f_l277_27784

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the domain of x
def domain : Set ℝ := Set.Icc (-2) 2

-- Define the minimum value function g
noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 7 + 4*a
  else if a ≤ 2 then 3 - a^2
  else 7 - 4*a

-- Theorem for part (1)
theorem f_min_max_when_a_neg_one :
  (∀ x ∈ domain, f (-1) x ≥ 2) ∧
  (∃ x ∈ domain, f (-1) x = 2) ∧
  (∀ x ∈ domain, f (-1) x ≤ 11) ∧
  (∃ x ∈ domain, f (-1) x = 11) := by
  sorry

-- Theorem for part (2)
theorem g_is_min_of_f :
  ∀ a : ℝ, ∀ x ∈ domain, f a x ≥ g a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_when_a_neg_one_g_is_min_of_f_l277_27784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l277_27785

theorem negation_of_sin_leq_one :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l277_27785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_l277_27717

noncomputable section

theorem right_triangle_30_60_90 (a b c : ℝ) (h : a > 0) :
  a^2 + b^2 = c^2 →
  Real.cos (π/6) = b / c →
  c = 20 →
  a = 10 * Real.sqrt 3 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_30_60_90_l277_27717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l277_27775

/-- An isosceles triangle with two sides of length 13 and one side of length 10 has an area of 60 -/
theorem isosceles_triangle_area :
  ∀ (P Q R : ℝ × ℝ),
  let d (a b : ℝ × ℝ) := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  d P Q = 13 ∧ d P R = 13 ∧ d Q R = 10 →
  (1/2) * 10 * Real.sqrt (13^2 - 5^2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l277_27775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_of_right_triangle_l277_27733

/-- The set of points forming a triangle in 2D space -/
def Triangle : Type := Set ℝ × Set ℝ

/-- Predicate to check if a triangle is a right triangle -/
def IsRightTriangle : Triangle → Prop := sorry

/-- Calculate the area of a triangle -/
def Area : Triangle → ℝ := sorry

/-- Calculate the hypotenuse of a triangle -/
def Hypotenuse : Triangle → ℝ := sorry

/-- Calculate the radius of the inscribed circle of a triangle -/
def InscribedCircleRadius : Triangle → ℝ := sorry

/-- Proves that for a right triangle with an area of 24 cm² and a hypotenuse of 10 cm, 
    the radius of its inscribed circle is 2 cm. -/
theorem inscribed_circle_radius_of_right_triangle 
  (triangle : Triangle) 
  (is_right_triangle : IsRightTriangle triangle) 
  (area : Area triangle = 24) 
  (hypotenuse : Hypotenuse triangle = 10) : 
  InscribedCircleRadius triangle = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_of_right_triangle_l277_27733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_value_l277_27771

-- Define vectors a and b as functions of t
def a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
def b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)

-- Define the distance between a and b
noncomputable def distance (t : ℝ) : ℝ := Real.sqrt ((1 - t - 2)^2 + (1 - t - t)^2 + (t - t)^2)

-- State the theorem
theorem min_distance_value :
  ∃ (t : ℝ), ∀ (s : ℝ), distance t ≤ distance s ∧ distance t = 3 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_value_l277_27771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l277_27732

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b

/-- A point on the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_property (h : Hyperbola) (p : HyperbolaPoint h)
    (h_dist : ∃ (f₁ f₂ : ℝ × ℝ), 
      (f₁.1 < 0 ∧ f₂.1 > 0) ∧  -- f₁ is left focus, f₂ is right focus
      (Real.sqrt ((p.x - f₁.1)^2 + (p.y - f₁.2)^2) = 
       2 * Real.sqrt ((p.x - f₂.1)^2 + (p.y - f₂.2)^2)))
    (h_angle : ∃ (f₁ f₂ : ℝ × ℝ),
      (f₁.1 < 0 ∧ f₂.1 > 0) ∧  -- f₁ is left focus, f₂ is right focus
      Real.sin (Real.arccos ((p.x - f₁.1) * (p.x - f₂.1) + (p.y - f₁.2) * (p.y - f₂.2)) /
        (Real.sqrt ((p.x - f₁.1)^2 + (p.y - f₁.2)^2) * 
         Real.sqrt ((p.x - f₂.1)^2 + (p.y - f₂.2)^2))) = Real.sqrt 15 / 4) :
  (eccentricity h = 2 ∧ h.b = Real.sqrt 3 * h.a) ∨
  (eccentricity h = Real.sqrt 6 ∧ h.b = Real.sqrt 5 * h.a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l277_27732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_l277_27700

noncomputable def g (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

theorem g_composition (x : ℝ) (h : -1 < x ∧ x < 1) : 
  g ((4 * x - x^4) / (1 + 4 * x^3)) = 4 * g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_l277_27700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AHF_l277_27725

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus point
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus
noncomputable def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x + 1

-- Define point A as the intersection of the line and the right side of the parabola
noncomputable def point_A : ℝ × ℝ := (2 * Real.sqrt 3, 3)

-- Define point H as the projection of A onto the y-axis
def point_H : ℝ × ℝ := (0, 3)

-- Theorem statement
theorem area_of_triangle_AHF :
  let A := point_A
  let H := point_H
  let F := focus
  (parabola A.1 A.2) ∧
  (line A.1 A.2) ∧
  (A.1 > 0) →
  (1/2 * |A.1 - H.1| * |A.2 - F.2|) = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_AHF_l277_27725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l277_27780

open Real

theorem trigonometric_equality : 
  let d : ℝ := π / 7
  (Real.sin (2 * d) * Real.cos (3 * d) * Real.sin (4 * d)) / (Real.sin d * Real.sin (2 * d) * Real.sin (3 * d) * Real.cos (4 * d)) = 
  (Real.sin (2 * π / 7) * Real.sin (4 * π / 7)) / (Real.sin (π / 7) * Real.sin (3 * π / 7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equality_l277_27780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_congruence_l277_27712

theorem smallest_n_congruence : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, 0 < m ∧ m < n → (3^m : ℤ) % 4 ≠ (m^3 : ℤ) % 4) ∧ 
  (3^n : ℤ) % 4 = (n^3 : ℤ) % 4 := by
  sorry

#check smallest_n_congruence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_congruence_l277_27712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_is_three_l277_27740

/-- Represents the investment partnership of A, B, and C -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  annual_gain : ℚ
  a_share : ℚ

/-- Calculates the ratio of C's investment to A's investment -/
def investment_ratio (p : Partnership) : ℚ :=
  p.c_investment / p.a_investment

/-- Theorem stating the ratio of C's investment to A's investment is 3 -/
theorem investment_ratio_is_three (p : Partnership) 
  (h1 : p.b_investment = 2 * p.a_investment)
  (h2 : p.annual_gain = 12000)
  (h3 : p.a_share = 4000)
  (h4 : p.annual_gain / p.a_share = 
        (p.a_investment * 12 + p.b_investment * 6 + p.c_investment * 4) / (p.a_investment * 12)) :
  investment_ratio p = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_ratio_is_three_l277_27740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_cut_percentage_l277_27763

/-- Proves that the percentage of marble cut away in the second week is 15% --/
theorem second_week_cut_percentage 
  (initial_weight : ℝ) 
  (first_week_cut : ℝ) 
  (third_week_cut : ℝ) 
  (final_weight : ℝ) : ℝ :=
by
  have h1 : initial_weight = 190 := by sorry
  have h2 : first_week_cut = 25 := by sorry
  have h3 : third_week_cut = 10 := by sorry
  have h4 : final_weight = 109.0125 := by sorry

  let weight_after_first_week := initial_weight * (1 - first_week_cut / 100)
  let second_week_cut : ℝ := 15
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut / 100)
  let weight_after_third_week := weight_after_second_week * (1 - third_week_cut / 100)

  have h5 : weight_after_third_week = final_weight := by sorry

  exact second_week_cut


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_week_cut_percentage_l277_27763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l277_27783

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x - x^2) / (abs x - x)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-1) 0 :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l277_27783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_theorem_l277_27727

theorem ceiling_sum_theorem :
  ⌈Real.sqrt (25 / 9 : ℝ)⌉ + ⌈(25 / 9 : ℝ)⌉ + ⌈((25 / 9 : ℝ)^2)⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_theorem_l277_27727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_l277_27729

open Real

/-- The function g(x) = x ln x - k(x-1) -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := x * log x - k * (x - 1)

/-- The theorem stating the maximum value of g(x) on [1, e] -/
theorem max_value_g (k : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), g k x ≤ exp 1 - k * exp 1 + k) ∧
  (k < exp 1 / (exp 1 - 1) →
    ∃ x ∈ Set.Icc 1 (exp 1), g k x = exp 1 - k * exp 1 + k) ∧
  (k ≥ exp 1 / (exp 1 - 1) →
    (∀ x ∈ Set.Icc 1 (exp 1), g k x ≤ 0) ∧
    (∃ x ∈ Set.Icc 1 (exp 1), g k x = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_l277_27729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_omega_l277_27769

/-- If ω is positive and the period of tan(ωx) is 2π, then ω equals 1/2. -/
theorem tangent_period_omega (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, Real.tan (ω * (x + 2 * π)) = Real.tan (ω * x)) :
  ω = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_period_omega_l277_27769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l277_27756

def valid_numbers : Finset ℕ := {1, 2, 4, 5, 6, 8}

structure Puzzle where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  h1 : {A, B, C, D, E, F} = valid_numbers
  h2 : A - 10 = 8
  h3 : C - 6 = 2
  h4 : E - D = 7
  h5 : F - E = 3

theorem puzzle_solution (p : Puzzle) : p.A + p.C = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puzzle_solution_l277_27756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_excircle_inradius_inequality_l277_27737

theorem triangle_excircle_inradius_inequality (a b c r ra rb rc : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ ra > 0 ∧ rb > 0 ∧ rc > 0)
  (h_inradius : r = (a * b * c) / (4 * (a + b + c) * (a * b + b * c + c * a - a * b * c)))
  (h_exradius_a : ra = (a * b * c) / ((b + c - a) * (a + b + c)))
  (h_exradius_b : rb = (a * b * c) / ((a + c - b) * (a + b + c)))
  (h_exradius_c : rc = (a * b * c) / ((a + b - c) * (a + b + c))) :
  ra + rb + rc ≥ 4 * r :=
by
  sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_excircle_inradius_inequality_l277_27737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_hand_walking_speed_is_two_l277_27744

/-- Pete's hand-walking speed in miles per hour -/
noncomputable def petes_hand_walking_speed (petes_backward_speed susan_forward_speed tracy_cartwheel_speed : ℝ) : ℝ :=
  tracy_cartwheel_speed / 4

theorem petes_hand_walking_speed_is_two 
  (h1 : petes_backward_speed = 3 * susan_forward_speed)
  (h2 : tracy_cartwheel_speed = 2 * susan_forward_speed)
  (h3 : petes_backward_speed = 12) :
  petes_hand_walking_speed petes_backward_speed susan_forward_speed tracy_cartwheel_speed = 2 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petes_hand_walking_speed_is_two_l277_27744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_fourth_vertex_l277_27790

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by a point and a direction vector -/
structure Line where
  point : Point
  direction : Point

/-- Definition of a rectangle given four points -/
structure Rectangle where
  O : Point
  A : Point
  B : Point
  C : Point

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.direction.x * l2.direction.x + l1.direction.y * l2.direction.y = 0

/-- Check if a point is on a line -/
def onLine (p : Point) (l : Line) : Prop :=
  ∃ t : ℝ, p.x = l.point.x + t * l.direction.x ∧ p.y = l.point.y + t * l.direction.y

/-- The theorem to be proved -/
theorem rectangle_fourth_vertex 
  (O A B : Point) 
  (l1 l2 : Line)
  (h1 : perpendicular l1 l2)
  (h2 : onLine A l1)
  (h3 : onLine B l2)
  (h4 : O = ⟨0, 0⟩) :
  ∃ C : Point, 
    onLine C (Line.mk O ⟨-A.y, A.x⟩) ∧ 
    ∃ (rect : Rectangle), rect.O = O ∧ rect.A = A ∧ rect.B = B ∧ rect.C = C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_fourth_vertex_l277_27790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_l277_27745

noncomputable def i : ℂ := Complex.I

theorem complex_power_difference (h : i^2 = -1) :
  ∃ (k : ℝ), (2 + i)^24 - (2 - i)^24 = -5^12 * k * i ∧ abs (k - 0.544) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_l277_27745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_parallel_planes_l277_27718

-- Define the types for lines and planes
variable (L : Type) (P : Type)

-- Define the parallel and perpendicular relations as functions
variable (parallel : P → P → Prop)
variable (perpendicular : L → P → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (l : L) (α β : P) 
  (h1 : perpendicular l β) 
  (h2 : parallel α β) : 
  perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perp_parallel_planes_l277_27718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_four_eight_l277_27742

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := a / b + b / a

-- Theorem statement
theorem otimes_four_eight : otimes 4 8 = 5/2 := by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp [div_add_div]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_four_eight_l277_27742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_amount_l277_27754

/-- Represents the weights of items in a suitcase -/
structure SuitcaseWeights where
  books : ℚ
  clothes : ℚ
  electronics : ℚ

/-- Calculates the ratio of two numbers -/
def ratio (a b : ℚ) : ℚ := a / b

/-- Represents the initial state of the suitcase -/
noncomputable def initial_state : SuitcaseWeights :=
  { books := 7 * (12 / 3)
  , clothes := 4 * (12 / 3)
  , electronics := 12 }

/-- Represents the final state of the suitcase after removing clothes -/
noncomputable def final_state : SuitcaseWeights :=
  { books := initial_state.books
  , clothes := initial_state.clothes - 8
  , electronics := initial_state.electronics }

theorem clothing_removed_amount :
  (ratio initial_state.books initial_state.clothes = 7 / 4) →
  (ratio initial_state.clothes initial_state.electronics = 4 / 3) →
  (initial_state.electronics = 12) →
  (ratio final_state.books final_state.clothes = 2 * ratio initial_state.books initial_state.clothes) →
  (initial_state.clothes - final_state.clothes = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_removed_amount_l277_27754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_pipe_times_correct_l277_27781

/-- Represents the time to empty a pool through different pipe combinations -/
structure PoolEmptyingTime where
  a : ℝ  -- Time to empty through first and second pipes
  b : ℝ  -- Time to empty through first and third pipes
  c : ℝ  -- Time to empty through second and third pipes

/-- Calculates the time to empty the pool through each pipe individually -/
noncomputable def individual_pipe_times (t : PoolEmptyingTime) : ℝ × ℝ × ℝ :=
  let x := 2 * t.a * t.b * t.c / (t.a * t.c + t.b * t.c - t.a * t.b)
  let y := 2 * t.a * t.b * t.c / (t.a * t.b + t.b * t.c - t.a * t.c)
  let z := 2 * t.a * t.b * t.c / (t.a * t.b + t.a * t.c - t.b * t.c)
  (x, y, z)

theorem individual_pipe_times_correct (t : PoolEmptyingTime) :
  let (x, y, z) := individual_pipe_times t
  1/x + 1/y = 1/t.a ∧ 1/x + 1/z = 1/t.b ∧ 1/y + 1/z = 1/t.c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_pipe_times_correct_l277_27781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_shirts_for_condition_l277_27759

/-- Represents the color of a shirt -/
inductive ShirtColor
  | Red
  | Blue
  | Green
  deriving BEq, Repr

/-- Represents a collection of shirts -/
def ShirtDrawer := List ShirtColor

/-- Checks if a list of shirts contains 3 of the same color -/
def hasThreeSameColor (shirts : ShirtDrawer) : Bool :=
  (shirts.count ShirtColor.Red ≥ 3) ||
  (shirts.count ShirtColor.Blue ≥ 3) ||
  (shirts.count ShirtColor.Green ≥ 3)

/-- Checks if a list of shirts contains at least one of each color -/
def hasThreeDifferentColors (shirts : ShirtDrawer) : Bool :=
  (shirts.count ShirtColor.Red ≥ 1) &&
  (shirts.count ShirtColor.Blue ≥ 1) &&
  (shirts.count ShirtColor.Green ≥ 1)

/-- The main theorem -/
theorem minimum_shirts_for_condition (drawer : ShirtDrawer) 
  (h1 : drawer.count ShirtColor.Red = 3)
  (h2 : drawer.count ShirtColor.Blue = 3)
  (h3 : drawer.count ShirtColor.Green = 3) :
  ∃ (n : Nat), n = 5 ∧ 
  (∀ (subset : ShirtDrawer), subset.length = n → 
    (hasThreeSameColor subset || hasThreeDifferentColors subset)) ∧
  (∀ (m : Nat), m < n → 
    ∃ (subset : ShirtDrawer), subset.length = m ∧
    ¬(hasThreeSameColor subset || hasThreeDifferentColors subset)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_shirts_for_condition_l277_27759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l277_27752

def is_valid_n (n : ℕ) : Prop :=
  ∃ (divisors : List ℕ),
    divisors.Sorted (· ≤ ·) ∧
    divisors ≠ [] ∧
    (∀ d, d ∈ divisors ↔ n % d = 0) ∧
    divisors.length ≥ 3 ∧
    divisors.get? (divisors.length - 3) = some (21 * divisors.get! 1)

theorem max_n_value :
  ∀ n : ℕ, is_valid_n n → n ≤ 441 :=
by sorry

#check max_n_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_value_l277_27752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_range_of_f_l277_27766

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (4^x + 1)

-- Theorem 1: Solve f(x) < 1/3
theorem solve_inequality :
  ∀ x : ℝ, f x < 1/3 ↔ x < 1/2 := by sorry

-- Theorem 2: Find the range of f(x)
theorem range_of_f :
  Set.range f = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_inequality_range_of_f_l277_27766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_state_after_three_turns_l277_27782

/-- Represents the state of the game as a list of three integers -/
def GameState := List Nat

/-- The initial state of the game -/
def initialState : GameState := [1, 1, 1]

/-- Represents one turn of the game -/
noncomputable def gameTurn (state : GameState) : GameState :=
  sorry

/-- The probability of transitioning from one state to another in one turn -/
noncomputable def transitionProbability (fromState toState : GameState) : ℝ :=
  sorry

/-- The probability of being in the state [1, 1, 1] after n turns -/
noncomputable def probEqualStateAfterNTurns (n : Nat) : ℝ :=
  sorry

theorem prob_equal_state_after_three_turns :
  probEqualStateAfterNTurns 3 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equal_state_after_three_turns_l277_27782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l277_27706

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) (m : ℝ) : ℝ :=
  |x - y + m| / Real.sqrt 2

-- Theorem statement
theorem distance_center_to_line (m : ℝ) :
  distance_point_to_line circle_center.1 circle_center.2 m = Real.sqrt 2 ↔ m = -1 ∨ m = -5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_line_l277_27706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l277_27703

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l277_27703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l277_27792

theorem determinant_solution (b : ℝ) (hb : b ≠ 0) :
  ∃ y : ℝ, ((y + 2*b) * ((y + 2*b)^2 - y^2) - y * (y*(y + 2*b) - y^2) + y * (y^2 - y*(y + 2*b)) = 0) ↔ (y = -b ∨ y = 2*b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l277_27792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electronic_item_price_l277_27793

/-- Calculates the final price of an item after discounts and tax --/
def finalPrice (originalPrice : ℝ) (firstDiscount secondDiscount tax : ℝ) : ℝ :=
  let priceAfterFirstDiscount := originalPrice * (1 - firstDiscount)
  let priceAfterSecondDiscount := priceAfterFirstDiscount * (1 - secondDiscount)
  priceAfterSecondDiscount * (1 + tax)

/-- Theorem stating the final price of the electronic item --/
theorem electronic_item_price :
  abs (finalPrice 240 0.3 0.15 0.08 - 154.22) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electronic_item_price_l277_27793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_symmetric_absolute_value_l277_27701

-- Define the function f(x) = |x + a|
def f (a : ℝ) (x : ℝ) : ℝ := |x + a|

-- Define symmetry about y-axis
def symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define decreasing interval
def decreasing_interval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x > f y

-- Theorem statement
theorem decreasing_interval_of_symmetric_absolute_value (a : ℝ) :
  symmetric_about_y_axis (f a) →
  decreasing_interval (f a) (Set.Iio 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_symmetric_absolute_value_l277_27701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l277_27773

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

theorem perpendicular_vector (l : ℝ) : 
  (a.1 - l * b.1) * a.1 + (a.2 - l * b.2) * a.2 = 0 ↔ l = -2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_l277_27773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_floor_l277_27719

/-- The minimum number of rectangular tiles needed to cover a rectangular region -/
def min_tiles_needed (tile_length tile_width region_length region_width : ℚ) : ℕ :=
  (((region_length / tile_length) * (region_width / tile_width)).ceil).toNat

/-- Theorem: 270 tiles of size 2 inches by 5 inches are needed to cover a 3 feet by 6 feet region -/
theorem tiles_needed_for_floor : 
  min_tiles_needed (2 : ℚ) (5 : ℚ) (3 * 12 : ℚ) (6 * 12 : ℚ) = 270 := by
  sorry

#eval min_tiles_needed (2 : ℚ) (5 : ℚ) (3 * 12 : ℚ) (6 * 12 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_needed_for_floor_l277_27719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_green_marbles_probability_l277_27714

def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def total_marbles : ℕ := green_marbles + purple_marbles
def drawn_marbles : ℕ := 6
def target_green : ℕ := 3

def probability_three_green : ℚ := 392 / 1001

theorem three_green_marbles_probability :
  (Nat.choose green_marbles target_green * Nat.choose purple_marbles (drawn_marbles - target_green)) /
  (Nat.choose total_marbles drawn_marbles : ℚ) = probability_three_green := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_green_marbles_probability_l277_27714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_iff_n_ge_4_l277_27713

/-- A function that checks if a set of positive integers satisfies the given conditions -/
def satisfies_conditions (S : Finset ℕ) : Prop :=
  let n := S.card
  (∀ x ∈ S, x < 2^(n-1)) ∧
  (∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ ∅ → B ≠ ∅ → A ≠ B →
    A.sum (λ x ↦ x) ≠ B.sum (λ x ↦ x))

/-- The main theorem stating that a set S satisfying the conditions exists if and only if n ≥ 4 -/
theorem exists_set_iff_n_ge_4 :
  ∀ n : ℕ, n > 0 → (∃ S : Finset ℕ, S.card = n ∧ satisfies_conditions S) ↔ n ≥ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_set_iff_n_ge_4_l277_27713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_eccentricity_l277_27739

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem: Two ellipses with equations (x²/a² + y²/b² = 1) and (x²/a² + y²/b² = k) 
    where k > 0, have the same eccentricity -/
theorem same_eccentricity (a b k : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : k > 0) :
  eccentricity a b = eccentricity (a * Real.sqrt k) (b * Real.sqrt k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_eccentricity_l277_27739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_50_51_l277_27774

def f (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_50_51_l277_27774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_given_amount_l277_27711

/-- Compound interest formula -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- The interest rate that results in the given final amount -/
theorem interest_rate_for_given_amount :
  ∃ (r : ℝ), 
    r > 0 ∧ 
    r < 1 ∧ 
    compound_interest 100 r 2 1 = 121.00000000000001 ∧ 
    r = 0.2 := by
  sorry

#check interest_rate_for_given_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_for_given_amount_l277_27711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_in_specific_rhombus_l277_27767

/-- Represents a rhombus ABCD -/
structure Rhombus where
  sideLength : ℝ
  angleB : ℝ

/-- Represents the region R inside the rhombus -/
def RegionR (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
def areaR (r : Rhombus) : ℝ :=
  sorry

theorem area_of_region_R_in_specific_rhombus :
  ∃ (r : Rhombus), r.sideLength = 3 ∧ r.angleB = 150 ∧ 
  (areaR r ≥ 0.19 ∧ areaR r ≤ 0.21) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_R_in_specific_rhombus_l277_27767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l277_27721

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)
def g (a x : ℝ) : ℝ := a * x^2 + x

-- Theorem for the minimum value of f
theorem min_value_f : ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = -1 / Real.exp 1 := by
  sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x ≥ 0 → f x ≤ g a x) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l277_27721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_over_3n_value_l277_27778

theorem cos_pi_over_3n_value (n : ℝ) (h : (3 : ℝ)^n = 3) : 
  Real.cos (π / (3 * n)) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_over_3n_value_l277_27778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l277_27747

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem equation_solution (x : ℝ) :
  (2 : ℝ) ^ (floor (Real.sin x)) = (3 : ℝ) ^ (1 - Real.cos x) ↔ ∃ n : ℤ, x = 2 * Real.pi * ↑n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l277_27747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l277_27794

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O1 A.1 A.2 ∧ circle_O1 B.1 B.2 ∧
  circle_O2 A.1 A.2 ∧ circle_O2 B.1 B.2 ∧
  A ≠ B

-- Define the line through two points
def line_through (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)

-- Define the perpendicular bisector of two points
def perpendicular_bisector (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  2 * (x - (A.1 + B.1) / 2) * (B.1 - A.1) + 2 * (y - (A.2 + B.2) / 2) * (B.2 - A.2) = 0

-- Theorem statement
theorem circles_properties
  (A B : ℝ × ℝ)
  (h_intersection : intersection_points A B) :
  (∀ x y, x - y = 0 ↔ line_through A B x y) ∧
  (∀ x y, x + y - 1 = 0 ↔ perpendicular_bisector A B x y) ∧
  Real.sqrt ((1 - (-2))^2 + (0 - 3)^2) = 3 * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_properties_l277_27794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l277_27765

noncomputable section

-- Define f as a function from reals to reals
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Ioo 0 2

-- Define the domain of f(-2x)
def domain_f_neg2x : Set ℝ := Set.Ioo (-1) 0

-- Theorem statement
theorem domain_transformation :
  (∀ x ∈ domain_f, f x ≠ 0) →
  (∀ x ∈ domain_f_neg2x, f (-2 * x) ≠ 0) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_transformation_l277_27765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l277_27724

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := |Real.sin (ω * x - Real.pi / 3)|

theorem f_properties (ω : ℝ) :
  -- 1. The range of f is [0, 1]
  (∀ x, 0 ≤ f ω x ∧ f ω x ≤ 1) ∧
  -- 2. The smallest positive period is π iff ω = ±1
  ((∃ T : ℝ, T > 0 ∧ 
    (∀ x, f ω (x + T) = f ω x) ∧ 
    (∀ S, 0 < S → S < T → ∃ y, f ω (y + S) ≠ f ω y)) ↔ (ω = 1 ∨ ω = -1)) ∧
  -- 3. When ω = 2, the increasing intervals are [kπ/2 + π/6, kπ/2 + 5π/12] for all integers k
  (ω = 2 →
    ∀ k : ℤ, ∀ x y, 
      k * Real.pi / 2 + Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ k * Real.pi / 2 + 5 * Real.pi / 12 →
      f ω x < f ω y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l277_27724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_budget_calculation_l277_27710

/-- Represents the remaining budget from last year -/
def remaining_budget_last_year : ℕ := 0  -- Initialize with 0, will be proven later

/-- Cost of the first school supply item -/
def supply_cost_1 : ℕ := 13

/-- Cost of the second school supply item -/
def supply_cost_2 : ℕ := 24

/-- Budget given for this year -/
def this_year_budget : ℕ := 50

/-- Amount remaining after purchases -/
def remaining_after_purchases : ℕ := 19

theorem teacher_budget_calculation :
  remaining_budget_last_year = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_budget_calculation_l277_27710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertex_count_l277_27702

/-- A complex number z satisfies the square vertex property if 0, z, and z^4 form three of the four distinct vertices of a square in the complex plane. -/
def square_vertex_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  (∃ w : ℂ, w ≠ 0 ∧ w ≠ z ∧ w ≠ z^4 ∧
    (Complex.abs (z - 0) = Complex.abs (z^4 - z) ∧ 
     Complex.abs (z - 0) = Complex.abs (w - z) ∧ 
     Complex.abs (z - 0) = Complex.abs (w - z^4)) ∧
    ((z - 0).re * (z^4 - z).re + (z - 0).im * (z^4 - z).im = 0 ∨
     (z^4 - z).re * (w - z^4).re + (z^4 - z).im * (w - z^4).im = 0 ∨
     (w - z^4).re * (z - 0).re + (w - z^4).im * (z - 0).im = 0))

/-- There are exactly 4 complex numbers satisfying the square vertex property. -/
theorem square_vertex_count : 
  ∃! (s : Finset ℂ), s.card = 4 ∧ ∀ z ∈ s, square_vertex_property z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertex_count_l277_27702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_specific_l277_27762

/-- Given an angle θ whose initial side coincides with the non-negative half-axis of the x-axis
    and whose terminal side passes through the point (-3, 4), prove that cos(2θ) = -7/25. -/
theorem cos_double_angle_specific (θ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) →
  Real.cos (2 * θ) = -7/25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_specific_l277_27762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l277_27798

/-- Represents a natural number with exactly 2015 digits -/
def Number2015 : Type := { n : ℕ // n ≥ 10^2014 ∧ n < 10^2015 }

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem digit_sum_property {n : ℕ} (hn : Number2015) (h : n % 9 = 0) :
  let a := sumOfDigits n
  let b := sumOfDigits a
  let c := sumOfDigits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_property_l277_27798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cut_l277_27770

-- Define a square sheet of paper
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define a folded square (after folding twice)
structure FoldedSquare where
  original : Square
  fold_lines : Fin 2 → ℝ  -- Two fold lines

-- Define a cut on the folded square
structure Cut where
  folded : FoldedSquare
  intersection : Set (ℝ × ℝ)  -- The shape of the cut

-- Define the property of intersecting both fold lines without including the center
def intersects_both_fold_lines_without_center (c : Cut) : Prop :=
  (∃ p q : ℝ × ℝ, p ∈ c.intersection ∧ q ∈ c.intersection ∧ 
    p.1 = c.folded.fold_lines 0 ∧ q.2 = c.folded.fold_lines 1) ∧
  (0, 0) ∉ c.intersection

-- The main theorem
theorem impossible_cut (s : Square) (f : FoldedSquare) (c : Cut) 
  (h_f : f.original = s) (h_c : c.folded = f) :
  ¬(intersects_both_fold_lines_without_center c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_cut_l277_27770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_second_quadrant_sine_range_l277_27764

theorem angle_second_quadrant_sine_range (α : ℝ) (m : ℝ) :
  (π / 2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α = 4 - 3 * m) →  -- given condition
  (1 < m ∧ m < 4 / 3) :=  -- range of m to be proved
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_second_quadrant_sine_range_l277_27764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l277_27796

/-- The function for which we're finding the vertical asymptote -/
noncomputable def f (x : ℝ) : ℝ := (x + 3) / (6 * x - 8)

/-- The x-value of the vertical asymptote -/
noncomputable def asymptote_x : ℝ := 4 / 3

/-- Theorem: The vertical asymptote of f occurs at x = 4/3 -/
theorem vertical_asymptote_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - asymptote_x| ∧ |x - asymptote_x| < δ → |f x| > 1/ε := by
  sorry

#check vertical_asymptote_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l277_27796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l277_27758

/-- A circle with center C passing through two points and its chord properties -/
theorem circle_and_chord_properties 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C.1 - C.2 + 1 = 0) -- C is on the line x-y+1=0
  (h2 : (C.1 - 1)^2 + (C.2 - 1)^2 = (C.1 - 2)^2 + (C.2 + 2)^2) -- Circle passes through (1,1) and (2,-2)
  : 
  -- The equation of the circle
  ((∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = (C.1 - 1)^2 + (C.2 - 1)^2) 
  ∧ 
  -- The trajectory of midpoint M
  (∀ E F : ℝ × ℝ, 
    (E.1 - C.1)^2 + (E.2 - C.2)^2 = (C.1 - 1)^2 + (C.2 - 1)^2 → 
    (F.1 - C.1)^2 + (F.2 - C.2)^2 = (C.1 - 1)^2 + (C.2 - 1)^2 → 
    (E.1 - 1) * (F.2 - 0) = (F.1 - 1) * (E.2 - 0) → 
    let M := ((E.1 + F.1) / 2, (E.2 + F.2) / 2)
    (M.1 + 1)^2 + (M.2 + 1)^2 = 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_chord_properties_l277_27758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_is_correct_l277_27750

/-- The function f(x, a) represents the left side of the inequality -/
def f (x a : ℝ) : ℝ := 2 * x^2 - (x - a) * |x - a| - 2

/-- The property that f(x, a) is non-negative for all real x -/
def always_non_negative (a : ℝ) : Prop := ∀ x, f x a ≥ 0

/-- The minimum value of 'a' that satisfies the always_non_negative property -/
noncomputable def min_a : ℝ := Real.sqrt 3

theorem min_a_is_correct :
  (∀ a, always_non_negative a → a ≥ min_a) ∧
  always_non_negative min_a :=
by sorry

#check min_a_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_is_correct_l277_27750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l277_27795

/-- Given two parallel lines l₁ and l₂, prove that the distance between them is 2 -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y ↦ 3 * x + 4 * y - 3 / 4
  let l₂ : ℝ → ℝ → ℝ := λ x y ↦ 12 * x + 16 * y + 37
  ∀ x y, l₁ x y = 0 → l₂ x y = 0 →
  let A : ℝ := 3
  let B : ℝ := 4
  let C₁ : ℝ := -3 / 4
  let C₂ : ℝ := 37 / 4
  (|C₁ - C₂| / (A^2 + B^2).sqrt : ℝ) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_lines_l277_27795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l277_27705

def product_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_1024 (n : ℕ) : product_of_divisors n = 1024 → n = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l277_27705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bryan_walking_time_l277_27722

/-- Represents the time in minutes Bryan spends walking one segment of his journey -/
def walking_time : ℕ := sorry

/-- Represents the total time in minutes Bryan spends traveling per day -/
def total_daily_travel_time : ℕ := 2 * walking_time + 40

/-- Represents the total time in hours Bryan spends traveling per year -/
def total_yearly_travel_time : ℕ := 365

theorem bryan_walking_time :
  walking_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bryan_walking_time_l277_27722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_result_l277_27768

noncomputable section

/-- A vector on the line y = 3x + 1 -/
def VectorOnLine (v : ℝ × ℝ) : Prop :=
  v.2 = 3 * v.1 + 1

/-- The projection of vector v onto vector w -/
def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot := v.1 * w.1 + v.2 * w.2
  let norm_sq := w.1 * w.1 + w.2 * w.2
  (dot / norm_sq * w.1, dot / norm_sq * w.2)

/-- The theorem stating the constant projection result -/
theorem constant_projection_result (w : ℝ × ℝ) : 
  (∃ p : ℝ × ℝ, ∀ v : ℝ × ℝ, VectorOnLine v → proj v w = p) →
  (∃ p : ℝ × ℝ, p = (-3/10, 1/10) ∧ ∀ v : ℝ × ℝ, VectorOnLine v → proj v w = p) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_result_l277_27768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l277_27715

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 
  (1 / Real.sqrt 2) * arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l277_27715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l277_27799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (3 * x^2 - a * x + 5) / Real.log (1/2)

theorem a_range (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x ∧ x < y → f a y < f a x) →
  -8 < a ∧ a ≤ -6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l277_27799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l277_27757

theorem range_of_m : ∀ m : ℝ, 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) ↔ m ∈ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l277_27757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_collinearity_l277_27709

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The right focus of the ellipse -/
noncomputable def Ellipse.rightFocus (e : Ellipse) : Point :=
  ⟨Real.sqrt (e.a^2 - e.b^2), 0⟩

/-- The left focus of the ellipse -/
noncomputable def Ellipse.leftFocus (e : Ellipse) : Point :=
  ⟨-Real.sqrt (e.a^2 - e.b^2), 0⟩

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Main theorem -/
theorem ellipse_collinearity (e : Ellipse) 
  (h_eq : e.a^2 = 5 ∧ e.b^2 = 4)
  (P : Point) 
  (h_P : P.x = 3)
  (T : Point)
  (h_T : e.equation T)
  (h_intersect : ∃! T, e.equation T ∧ 
    (T.y - (P.y + e.rightFocus.y) / 2) * (T.x - (P.x + e.rightFocus.x) / 2) = 
    -(P.x - e.rightFocus.x) * (P.y - e.rightFocus.y)) :
  collinear e.leftFocus T P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_collinearity_l277_27709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_financials_theorem_l277_27716

/-- Elaine's financial situation over two years -/
structure ElaineFincancials where
  last_year_earnings : ℝ
  last_year_rent_percent : ℝ
  last_year_other_expenses_percent : ℝ
  this_year_earnings_increase_percent : ℝ
  this_year_rent_percent : ℝ
  this_year_other_expenses_percent : ℝ

/-- Calculations based on Elaine's financials -/
noncomputable def financial_ratios (e : ElaineFincancials) : 
  (ℝ × ℝ × ℝ × ℝ) :=
  let this_year_earnings := e.last_year_earnings * (1 + e.this_year_earnings_increase_percent)
  let last_year_rent := e.last_year_earnings * e.last_year_rent_percent
  let this_year_rent := this_year_earnings * e.this_year_rent_percent
  let last_year_other_expenses := e.last_year_earnings * e.last_year_other_expenses_percent
  let this_year_other_expenses := this_year_earnings * e.this_year_other_expenses_percent
  (
    this_year_rent / last_year_rent,
    this_year_other_expenses / last_year_other_expenses,
    (last_year_rent + last_year_other_expenses) / e.last_year_earnings,
    (this_year_rent + this_year_other_expenses) / this_year_earnings
  )

theorem elaine_financials_theorem (e : ElaineFincancials) 
  (h1 : e.last_year_rent_percent = 0.10)
  (h2 : e.last_year_other_expenses_percent = 0.20)
  (h3 : e.this_year_earnings_increase_percent = 0.15)
  (h4 : e.this_year_rent_percent = 0.30)
  (h5 : e.this_year_other_expenses_percent = 0.25) :
  let (rent_ratio, other_expenses_ratio, last_year_total_ratio, this_year_total_ratio) 
    := financial_ratios e
  rent_ratio = 3.45 ∧ 
  other_expenses_ratio = 1.4375 ∧
  last_year_total_ratio = 0.30 ∧
  this_year_total_ratio = 0.55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elaine_financials_theorem_l277_27716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_square_l277_27760

theorem tens_digit_square (n : ℕ) : 
  (((n^2 / 10) % 10 = 7) → 
  n - 100 * (n / 100) ∈ ({24, 26, 74, 76} : Set ℕ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_square_l277_27760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_t_1_range_of_t_below_line_l277_27735

-- Define the curve C
noncomputable def curve_C (t : ℝ) (α : ℝ) : ℝ × ℝ := (t * Real.cos α, Real.sin α)

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := Real.sqrt 2 * ρ * Real.sin (θ + Real.pi/4) = 3

-- Define the distance function from a point to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + y - 3| / Real.sqrt 2

-- Theorem 1: Maximum distance when t=1
theorem max_distance_t_1 : 
  ∃ (α : ℝ), ∀ (β : ℝ), 
    distance_to_line (curve_C 1 α).1 (curve_C 1 α).2 ≥ 
    distance_to_line (curve_C 1 β).1 (curve_C 1 β).2 ∧
    distance_to_line (curve_C 1 α).1 (curve_C 1 α).2 = (2 + 3 * Real.sqrt 2) / 2 :=
by sorry

-- Theorem 2: Range of t for all points on C to be below l
theorem range_of_t_below_line : 
  ∀ (t : ℝ), (∀ (α : ℝ), (curve_C t α).1 + (curve_C t α).2 < 3) ↔ 
    (0 < t ∧ t < 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_t_1_range_of_t_below_line_l277_27735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_population_growth_l277_27731

/-- Represents the population growth model of a city --/
structure CityPopulationModel where
  initial_population : ℝ
  annual_relocation : ℝ
  natural_growth_rate : ℝ

/-- Calculates the population after a given number of years --/
noncomputable def population_after_years (model : CityPopulationModel) (years : ℝ) : ℝ :=
  let q := 1 + model.natural_growth_rate / 100
  model.initial_population * q^years + 
    model.annual_relocation * (q^years - 1) / (q - 1)

/-- Calculates the annual growth percentage --/
noncomputable def annual_growth_percentage (model : CityPopulationModel) : ℝ :=
  let population_10_years := population_after_years model 10
  (population_10_years / model.initial_population)^(1/10) - 1

/-- Calculates the number of years to reach a target population --/
noncomputable def years_to_reach_population (model : CityPopulationModel) (target : ℝ) : ℝ :=
  let q := 1 + model.natural_growth_rate / 100
  (Real.log (target / model.initial_population)) / (Real.log q)

theorem city_population_growth 
  (model : CityPopulationModel)
  (h_initial : model.initial_population = 117751)
  (h_relocation : model.annual_relocation = 640)
  (h_growth_rate : model.natural_growth_rate = 0.87) :
  (abs (population_after_years model 10 - 135140) < 1) ∧
  (abs (annual_growth_percentage model - 0.014) < 0.0001) ∧
  (abs (years_to_reach_population model 200000 - 43) < 1) := by
  sorry

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_population_growth_l277_27731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_tangents_l277_27730

/-- Given two tangent lines to an ellipse and the condition that its axes are aligned with the coordinate system axes, prove the equation of the ellipse. -/
theorem ellipse_equation_from_tangents (x y : ℝ) :
  (∃ (t₁ t₂ : ℝ → ℝ → Prop),
    (∀ x y, t₁ x y ↔ 4*x + 5*y = 25) ∧
    (∀ x y, t₂ x y ↔ 9*x + 20*y = 75) ∧
    (∃ (e : ℝ → ℝ → Prop),
      (∀ x y, e x y ↔ x^2/25 + y^2/9 = 1) ∧
      (∀ x y, t₁ x y → (∃ x₀ y₀, e x₀ y₀ ∧ (x - x₀) * ((2*x₀)/25) + (y - y₀) * ((2*y₀)/9) = 0)) ∧
      (∀ x y, t₂ x y → (∃ x₀ y₀, e x₀ y₀ ∧ (x - x₀) * ((2*x₀)/25) + (y - y₀) * ((2*y₀)/9) = 0)) ∧
      (∀ x y, e x y → x^2/25 + y^2/9 = 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_tangents_l277_27730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_segment_with_specific_colors_l277_27736

/-- Given a set of 6n points on a line with 4n blue points and 2n green points,
    there exists a line segment containing 3n points, of which 2n are blue and n are green. -/
theorem exists_segment_with_specific_colors (n : ℕ) (S : Finset ℝ) (blue green : Finset ℝ) :
  (S.card = 6 * n) →
  (blue.card = 4 * n) →
  (green.card = 2 * n) →
  (S = blue ∪ green) →
  (blue ∩ green = ∅) →
  ∃ (a b : ℝ), a < b ∧ 
    ((S.filter (fun x => a ≤ x ∧ x ≤ b)).card = 3 * n) ∧
    ((blue.filter (fun x => a ≤ x ∧ x ≤ b)).card = 2 * n) ∧
    ((green.filter (fun x => a ≤ x ∧ x ≤ b)).card = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_segment_with_specific_colors_l277_27736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_cosine_l277_27707

/-- Given a line l with inclination angle α parallel to x - 2y + 2 = 0,
    prove that cos(α) = 2√5/5 -/
theorem parallel_line_cosine (α : Real) : 
  0 < α → α < π → Real.tan α = (1 : Real) / 2 → Real.cos α = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_cosine_l277_27707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_numbers_with_zero_l277_27788

def six_digit_numbers : ℕ := 9 * 10^5

def six_digit_numbers_without_zero : ℕ := 9^6

theorem six_digit_numbers_with_zero
  (h1 : six_digit_numbers = 9 * 10^5)
  (h2 : six_digit_numbers_without_zero = 9^6)
  : six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_numbers_with_zero_l277_27788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_path_correctness_optimal_path_minimizes_time_l277_27751

/-- Represents the optimal path for a swimmer to cross a square pool -/
inductive OptimalPath
  | RunAround
  | SwimAcross
  | AnyDiagonal

/-- The side length of the square pool -/
def poolSide : ℝ := 1

/-- The optimal path for the swimmer based on the speed ratio -/
noncomputable def optimalPath (k : ℝ) : OptimalPath :=
  if k > Real.sqrt 2 then OptimalPath.RunAround
  else if k < Real.sqrt 2 then OptimalPath.SwimAcross
  else OptimalPath.AnyDiagonal

/-- Theorem stating the optimal path for the swimmer -/
theorem optimal_path_correctness (k : ℝ) :
  optimalPath k = 
    if k > Real.sqrt 2 then OptimalPath.RunAround
    else if k < Real.sqrt 2 then OptimalPath.SwimAcross
    else OptimalPath.AnyDiagonal := by sorry

/-- The time taken for the optimal path -/
noncomputable def optimalTime (k : ℝ) : ℝ :=
  match optimalPath k with
  | OptimalPath.RunAround => 2 / k
  | OptimalPath.SwimAcross => Real.sqrt 2
  | OptimalPath.AnyDiagonal => 1 / Real.sqrt 2 + Real.sqrt 2 + 1 / Real.sqrt 2

/-- Theorem stating that the optimal path gives the shortest time -/
theorem optimal_path_minimizes_time (k : ℝ) (t : ℝ) :
  t ≥ optimalTime k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_path_correctness_optimal_path_minimizes_time_l277_27751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l277_27743

theorem remainder_problem (x : ℕ) (h : (4 * x) % 9 = 2) : x % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l277_27743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l277_27741

-- Define the ellipse equation
noncomputable def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 6 = 1

-- Define the eccentricity
noncomputable def eccentricity : ℝ := Real.sqrt 6 / 6

-- Theorem statement
theorem ellipse_a_values (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, ellipse_equation x y a → 
    Real.sqrt (1 - (6 / a^2)) = eccentricity ∨ 
    Real.sqrt (1 - (a^2 / 6)) = eccentricity) :
  a = (6 * Real.sqrt 5) / 5 ∨ a = Real.sqrt 5 := by
  sorry

#check ellipse_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_a_values_l277_27741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_l277_27787

theorem tan_pi_plus_alpha (α : ℝ) (h1 : Real.sin α = -2/3) (h2 : π < α ∧ α < 3*π/2) :
  Real.tan (π + α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_l277_27787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_calculation_l277_27726

noncomputable def distance_AB : ℝ := 271 / 3

noncomputable def passenger_speed : ℝ := 30

noncomputable def fast_speed : ℝ := 60

noncomputable def slowdown_fraction : ℝ := 2 / 3

noncomputable def speed_reduction_factor : ℝ := 1 / 2

noncomputable def overtake_distance : ℝ := 271 / 9

theorem distance_AB_calculation :
  ∃ (d : ℝ), d = distance_AB ∧
  d > 0 ∧
  (d - overtake_distance) / fast_speed =
  (slowdown_fraction * d / passenger_speed) +
  ((1 - slowdown_fraction) * d / (speed_reduction_factor * passenger_speed)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_calculation_l277_27726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l277_27738

/-- Given a geometric sequence of 7 terms with common ratio -2,
    the ratio of the sum of even terms to the sum of odd terms is -42/85 -/
theorem geometric_sequence_ratio (a : ℝ) : 
  let seq := [a, -2*a, 4*a, -8*a, 16*a, -32*a, 64*a]
  (seq[1]! + seq[3]! + seq[5]!) / (seq[0]! + seq[2]! + seq[4]! + seq[6]!) = -42 / 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l277_27738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_l277_27776

theorem matrix_multiplication (M : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : M.mulVec (![3, 1]) = ![2, 4])
  (h2 : M.mulVec (![1, 4]) = ![1, 2]) :
  M.mulVec (![7, 5]) = ![6, 12] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_multiplication_l277_27776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_and_triangle_area_l277_27791

-- Define the polar coordinate type
structure PolarCoord where
  r : ℝ
  θ : ℝ

-- Define the given points
noncomputable def A : PolarCoord := ⟨1, Real.pi/3⟩
noncomputable def B : PolarCoord := ⟨9, Real.pi/3⟩

-- Define the perpendicular bisector line l
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/3) = 5

-- Define point C
noncomputable def C : PolarCoord := ⟨10, 0⟩

-- State the theorem
theorem perpendicular_bisector_and_triangle_area :
  (∀ ρ θ, line_l ρ θ ↔ ρ * Real.cos (θ - Real.pi/3) = 5) ∧
  (1/2 * (B.r - A.r) * C.r * Real.sin (Real.pi/3) = 20 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_and_triangle_area_l277_27791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diminished_value_proof_l277_27749

theorem diminished_value_proof (n : ℕ) : 
  (∀ d ∈ ({15, 30, 45, 60} : Set ℕ), (200 - 160) % d = 0) ∧
  (∀ m : ℕ, m < 160 → ∃ d ∈ ({15, 30, 45, 60} : Set ℕ), (200 - m) % d ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diminished_value_proof_l277_27749
