import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_covering_5x7_l961_96157

/-- Represents an L-shaped piece -/
structure LPiece where
  cells : Fin 3 → Nat × Nat

/-- Represents a rectangle -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a covering of a rectangle with L-shaped pieces -/
structure Covering where
  rectangle : Rectangle
  pieces : List LPiece
  layers : Nat

/-- Checks if a covering is valid according to the problem constraints -/
def is_valid_covering (c : Covering) : Prop :=
  ∀ (x y : Nat), x < c.rectangle.width → y < c.rectangle.height →
    (c.pieces.filter (λ p ↦ ∃ i, p.cells i = (x, y))).length = c.layers

/-- The main theorem stating the impossibility of the covering -/
theorem no_valid_covering_5x7 :
  ¬ ∃ (c : Covering), c.rectangle.width = 5 ∧ c.rectangle.height = 7 ∧ is_valid_covering c := by
  sorry

#check no_valid_covering_5x7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_covering_5x7_l961_96157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_houses_l961_96106

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Maximum possible distance between two houses -/
theorem max_distance_between_houses
  (school jiajia qiqi : Point)
  (h1 : distance school jiajia = 2)
  (h2 : distance school qiqi = 3)
  (h3 : ¬ (∃ t : ℝ, qiqi = Point.mk (t * jiajia.x + (1 - t) * school.x) (t * jiajia.y + (1 - t) * school.y)))
  : ∃ d : ℝ, d ≤ 5 ∧ d = 4 ∧ distance jiajia qiqi ≤ d ∧
    ∀ d' : ℝ, d' ≤ 5 → distance jiajia qiqi ≤ d' → d' ≤ d := by
  sorry

#check max_distance_between_houses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_houses_l961_96106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_less_than_zero_l961_96177

-- Define the function f(x) = |2^x - 1|
noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

-- State the theorem
theorem sum_less_than_zero (a b : ℝ) (h1 : a < b) (h2 : f a = f b) : a + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_less_than_zero_l961_96177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_distinct_numbers_eight_sided_dice_l961_96169

theorem probability_six_distinct_numbers_eight_sided_dice : 
  (8 * 7 * 6 * 5 * 4 * 3 : ℚ) / (8^6) = 315 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_six_distinct_numbers_eight_sided_dice_l961_96169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_l961_96141

noncomputable def g (n : ℕ) : ℝ := ∑' k : ℕ, (1 : ℝ) / ((k + 2 : ℝ) ^ n)

theorem sum_of_g : ∑' n : ℕ, g (n + 2) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_l961_96141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l961_96126

theorem problem_statement (a b : ℕ) (h1 : a > b) (h2 : b > 0) 
  (h3 : Nat.Coprime a b) (h4 : (a^3 - b^3) / (a - b)^3 = 73/3) : a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l961_96126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l961_96138

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4*x

-- Theorem statement
theorem tangent_line_at_negative_one :
  let p : ℝ × ℝ := (-1, f (-1))
  let m : ℝ := f' (-1)
  (λ (x y : ℝ) => x + y = 0) = (λ (x y : ℝ) => y - p.2 = m * (x - p.1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_l961_96138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l961_96191

theorem coefficient_x6_in_expansion : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ), 
    (fun x : ℝ ↦ (2 * x^2 + 1)^5) = 
    (fun x : ℝ ↦ a₀ + a₁*x^2 + a₂*x^4 + a₃*x^6 + a₄*x^8 + a₅*x^10) ∧ 
    a₃ = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x6_in_expansion_l961_96191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l961_96188

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x*y > a*b) → a*b = 1/8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → (2:ℝ)^x + (4:ℝ)^y ≥ (2:ℝ)^a + (4:ℝ)^b) → 
    (2:ℝ)^a + (4:ℝ)^b = 2 * Real.sqrt 2 ∧
  0 < b ∧ b < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l961_96188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_periodic_l961_96194

noncomputable def b : ℕ → ℝ
  | 0 => Real.sqrt 2 - 1  -- Add case for 0
  | 1 => Real.sqrt 2 - 1
  | n + 1 => if 0 < b n ∧ b n ≤ 1 then 1 / b n else b n - 1

theorem b_periodic : ∀ n : ℕ, b (n + 5) = b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_periodic_l961_96194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_work_days_l961_96193

-- Define the work rates
noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

-- Define the total work as 1 (representing 100% of the work)
def total_work : ℝ := 1

-- Matt and Peter's combined work rate
noncomputable def combined_rate : ℝ := work_rate 20

-- Peter's individual work rate
noncomputable def peter_rate : ℝ := work_rate 20

-- Theorem statement
theorem peter_work_days :
  -- Given conditions
  (combined_rate * 10 = total_work / 2) →  -- Half work done in 10 days together
  (peter_rate * 10 = total_work / 2) →     -- Peter completes remaining half in 10 days
  -- Conclusion
  (total_work / peter_rate = 20) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_work_days_l961_96193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_left_percentage_l961_96190

def initial_cards : ℕ := 16
def fraction_given : ℚ := 3/8
def additional_cards : ℕ := 2

theorem cards_left_percentage (initial_cards : ℕ) (fraction_given : ℚ) (additional_cards : ℕ) :
  initial_cards > 0 →
  fraction_given ≥ 0 →
  fraction_given < 1 →
  additional_cards ≥ 0 →
  (initial_cards - (fraction_given * ↑initial_cards).floor - additional_cards : ℚ) / initial_cards * 100 = 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_left_percentage_l961_96190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l961_96142

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 4)

-- Define the line equation for the center C
def center_line (x y : ℝ) : Prop := x + 3 * y - 15 = 0

-- Define the circle passing through A and B with center C on the center_line
def circle_eq (C : ℝ × ℝ) : Prop :=
  center_line C.1 C.2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ :=
  abs ((P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1)) / 2

-- State the theorem
theorem max_triangle_area :
  ∃ (C : ℝ × ℝ), circle_eq C →
  (∀ (P : ℝ × ℝ), (P.1 - C.1)^2 + (P.2 - C.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 →
    area_triangle P A B ≤ 16 + 8 * Real.sqrt 5) ∧
  (∃ (P : ℝ × ℝ), (P.1 - C.1)^2 + (P.2 - C.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    area_triangle P A B = 16 + 8 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l961_96142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_volume_l961_96146

theorem pyramid_section_volume (V V₁ A : ℝ) (hV : V > 0) (hA : A > 0) (hV₁ : 0 < V₁ ∧ V₁ < V) :
  ∃ x : ℝ, x > 0 ∧ x < (3 * V / A) ∧
    x = (3 / A) * (V - (V^2 * (V - V₁))^(1/3)) := by
  sorry

#check pyramid_section_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_section_volume_l961_96146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l961_96107

/-- Compound interest calculation -/
theorem compound_interest_calculation 
  (P : ℝ) -- Principal (initial investment)
  (r : ℝ) -- Annual interest rate (as a decimal)
  (n : ℕ) -- Number of years
  (h1 : P = 2500)
  (h2 : r = 0.04)
  (h3 : n = 21) :
  abs (P * (1 + r) ^ n - 6101.50) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_calculation_l961_96107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_survey_size_l961_96104

-- Define the percentage of people who liked product A
def X : ℚ := 60

-- Define the survey results
def liked_A : ℚ := X
def liked_B : ℚ := X - 20
def liked_both : ℚ := 23
def liked_neither : ℚ := 23

-- Define the theorem
theorem minimum_survey_size :
  ∃ (n : ℕ), n > 0 ∧
  (n * liked_A / 100).num % (n * liked_A / 100).den = 0 ∧
  (n * liked_B / 100).num % (n * liked_B / 100).den = 0 ∧
  (n * liked_both / 100).num % (n * liked_both / 100).den = 0 ∧
  (n * liked_neither / 100).num % (n * liked_neither / 100).den = 0 ∧
  ∀ (m : ℕ), m < n →
    (m * liked_A / 100).num % (m * liked_A / 100).den ≠ 0 ∨
    (m * liked_B / 100).num % (m * liked_B / 100).den ≠ 0 ∨
    (m * liked_both / 100).num % (m * liked_both / 100).den ≠ 0 ∨
    (m * liked_neither / 100).num % (m * liked_neither / 100).den ≠ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_survey_size_l961_96104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l961_96198

-- Define a circular sector
structure CircularSector where
  perimeter : ℝ
  area : ℝ

-- Helper definition
def IsTheCentralAngle (s : CircularSector) (θ : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ 
    s.perimeter = 2 * r + r * θ ∧
    s.area = (1/2) * r^2 * θ

-- Theorem statement
theorem central_angle_of_sector (s : CircularSector) 
  (h_perimeter : s.perimeter = 8)
  (h_area : s.area = 4) :
  ∃ θ : ℝ, θ > 0 ∧ θ = 2 ∧ IsTheCentralAngle s θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_of_sector_l961_96198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rhombus_MN_length_l961_96174

/-- A rhombus with a specific configuration of points -/
structure SpecialRhombus where
  -- The side length of the rhombus
  a : ℝ
  -- Angle BAD is 60 degrees
  angle_BAD : ℝ
  -- Point M on side AD
  M : EuclideanSpace ℝ (Fin 2)
  -- Point N on side BC
  N : EuclideanSpace ℝ (Fin 2)
  -- DM : AM = 2 : 1
  DM_AM_ratio : ℝ
  -- BN : NC = 2 : 1
  BN_NC_ratio : ℝ
  -- Conditions
  angle_condition : angle_BAD = 60
  DM_AM_condition : DM_AM_ratio = 2 / 1
  BN_NC_condition : BN_NC_ratio = 2 / 1

/-- The length of MN in the special rhombus configuration -/
noncomputable def MN_length (r : SpecialRhombus) : ℝ := r.a * Real.sqrt 13 / 3

/-- Theorem stating that MN_length is correct for any SpecialRhombus -/
theorem special_rhombus_MN_length (r : SpecialRhombus) : 
  ‖r.M - r.N‖ = MN_length r := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_rhombus_MN_length_l961_96174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_OZ_l961_96148

noncomputable def complex_i : ℂ := Complex.I

noncomputable def z : ℂ := 1 - complex_i

noncomputable def vector_OZ : ℂ := 2 / z + z^2

theorem magnitude_of_vector_OZ : Complex.abs vector_OZ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_OZ_l961_96148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l961_96125

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x^3 + 9*x^4) / (10 - x^4)

theorem f_nonnegative_iff (x : ℝ) : 
  f x ≥ 0 ↔ x ∈ Set.Icc (-Real.rpow 10 (1/4 : ℝ)) 0 ∪ 
             Set.Icc 0 (1/3) ∪ 
             Set.Icc (1/3) (Real.rpow 10 (1/4 : ℝ)) :=
by
  sorry

#check f_nonnegative_iff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l961_96125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l961_96145

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = x^3 + sin(x) - 2x -/
noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2*x

theorem zero_in_interval :
  IsOdd f ∧ ∃ z ∈ Set.Ioo 1 (Real.pi/2), f z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l961_96145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_coordinates_l961_96182

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: Given A(1,2,1) and B(2,2,2), point P on x-axis with |PA| = |PB| has coordinates (3,0,0) -/
theorem point_p_coordinates : 
  let A : Point3D := ⟨1, 2, 1⟩
  let B : Point3D := ⟨2, 2, 2⟩
  ∀ P : Point3D, 
    P.y = 0 ∧ P.z = 0 → -- P is on x-axis
    distance P A = distance P B → -- |PA| = |PB|
    P = ⟨3, 0, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_p_coordinates_l961_96182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_count_l961_96175

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (h_AB : dist A B = AB)
  (h_BC : dist B C = BC)
  (h_CD : dist C D = CD)
  (h_DA : dist D A = DA)

def AC_length (q : Quadrilateral) : ℝ := dist q.A q.C

theorem diagonal_length_count (q : Quadrilateral) 
  (h_AB : q.AB = 7)
  (h_BC : q.BC = 13)
  (h_CD : q.CD = 15)
  (h_DA : q.DA = 10) :
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ q' : Quadrilateral, AC_length q' = n ∧ 
                                                 q'.AB = q.AB ∧ 
                                                 q'.BC = q.BC ∧ 
                                                 q'.CD = q.CD ∧ 
                                                 q'.DA = q.DA) ∧ 
                 Finset.card S = 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_count_l961_96175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_m_equals_10_l961_96155

/-- An exponential sequence with common ratio r -/
def exponential_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
| 0 => a₁
| n + 1 => exponential_sequence a₁ r n * r

theorem exponential_sequence_m_equals_10 
  (a₁ : ℝ) (r : ℝ) (m : ℕ) 
  (hr : r ≠ 1) 
  (ha₁ : a₁ ≠ 0)
  (h1 : exponential_sequence a₁ r 5 * exponential_sequence a₁ r 6 + 
        exponential_sequence a₁ r 4 * exponential_sequence a₁ r 7 = 18)
  (h2 : a₁ * exponential_sequence a₁ r m = 9) : 
  m = 10 := by
  sorry

#check exponential_sequence_m_equals_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_sequence_m_equals_10_l961_96155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l961_96186

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The asymptote equation -/
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 / 3 * x ∨ y = -Real.sqrt 3 / 3 * x

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The parabola equation -/
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

theorem parabola_equation (A : ℝ × ℝ) (p : ℝ) :
  hyperbola A.1 A.2 →
  asymptote A.1 A.2 →
  distance A right_focus = 2 →
  p > 0 →
  parabola p A.1 A.2 →
  p = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l961_96186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EQ_length_l961_96101

/-- Represents a trapezoid with a circle inscribed in it -/
structure InscribedTrapezoid where
  EF : ℚ
  FG : ℚ
  GH : ℚ
  HE : ℚ
  EQ : ℚ

/-- The properties of the specific trapezoid in the problem -/
def problemTrapezoid : InscribedTrapezoid where
  EF := 110
  FG := 60
  GH := 23
  HE := 75
  EQ := 250 / 3

/-- Theorem stating that the EQ length in the problem trapezoid is 250/3 -/
theorem trapezoid_EQ_length (t : InscribedTrapezoid) (h1 : t = problemTrapezoid) :
    t.EQ = 250 / 3 := by
  rw [h1]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EQ_length_l961_96101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_C₂_to_C₁_l961_96134

/-- Polar equation of curve C₁ -/
def C₁ (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - Real.pi/3) = -1

/-- Polar equation of curve C₂ -/
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.cos (θ - Real.pi/4)

/-- Maximum distance from a point on C₂ to C₁ -/
noncomputable def max_distance : ℝ := (3 + Real.sqrt 3 + 2 * Real.sqrt 2) / 2

theorem max_distance_from_C₂_to_C₁ :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (∀ (a b : ℝ), (∃ (ρ' θ' : ℝ), C₁ ρ' θ' ∧ a = ρ' * Real.cos θ' ∧ b = ρ' * Real.sin θ') →
    Real.sqrt ((x - a)^2 + (y - b)^2) ≤ max_distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_from_C₂_to_C₁_l961_96134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l961_96113

theorem largest_power_of_18_dividing_30_factorial : 
  ∃ k : ℕ, k = 7 ∧ 
  (∀ m : ℕ, 18^m ∣ Nat.factorial 30 → m ≤ k) ∧ 
  (18^k ∣ Nat.factorial 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_18_dividing_30_factorial_l961_96113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_y_range_l961_96163

def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

def foci (F₁ F₂ : ℝ × ℝ) : Prop := F₁ = (Real.sqrt 3, 0) ∧ F₂ = (-Real.sqrt 3, 0)

theorem hyperbola_y_range (x₀ y₀ : ℝ) (F₁ F₂ : ℝ × ℝ) :
  hyperbola x₀ y₀ →
  foci F₁ F₂ →
  (F₁.1 - x₀) * (F₂.1 - x₀) + (F₁.2 - y₀) * (F₂.2 - y₀) < 0 →
  -Real.sqrt 3 / 3 < y₀ ∧ y₀ < Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_y_range_l961_96163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l961_96120

/-- The function g -/
noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The theorem statement -/
theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 20 = 20)
  (h2 : g a b c d 99 = 99)
  (h3 : ∀ x, x ≠ -d/c → g a b c d (g a b c d x) = x) :
  ∃! y, (∀ x, g a b c d x ≠ y) ∧ y = 59.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l961_96120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_and_minimum_l961_96152

noncomputable section

-- Define the quadratic function
def q (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set A
def A (b : ℝ) : Set ℝ := {x | 1 < x ∧ x < b}

-- Define the function f
def f (a b x : ℝ) : ℝ := (2 * a + b) * x - 1 / ((a - b) * (x - 1))

theorem quadratic_solution_and_minimum (a b : ℝ) :
  (∀ x, x ∈ A b ↔ q a x < 0) →
  a = 1 ∧ b = 2 ∧ ∀ x ∈ A b, f a b x ≥ 8 ∧ ∃ x ∈ A b, f a b x = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_and_minimum_l961_96152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_500_and_1500_l961_96133

theorem perfect_cubes_between_500_and_1500 :
  (Finset.filter (fun n : ℕ => 500 ≤ n^3 ∧ n^3 ≤ 1500) (Finset.range 12)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cubes_between_500_and_1500_l961_96133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l961_96149

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : Prop := x^2 - a*x - 2 = 0

-- Define the inequality condition
def inequality_condition (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  ∀ a : ℝ, a ∈ Set.Icc (-1) 1 → |m^2 - 5*m - 3| ≥ |x₁ - x₂|

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (4*x^2 + (m-2)*x + 1)

-- Define the range condition for f
def f_range_condition (m : ℝ) : Prop :=
  Set.range (f m) = Set.univ

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ,
  (∃ x₁ x₂ : ℝ, quadratic_eq 0 x₁ ∧ quadratic_eq 0 x₂) →
  (∀ a : ℝ, ∃ x₁ x₂ : ℝ, quadratic_eq a x₁ ∧ quadratic_eq a x₂ ∧ inequality_condition m x₁ x₂) →
  f_range_condition m →
  m ≥ 6 ∨ m ≤ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l961_96149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l961_96122

/-- The solution to the Cauchy problem -/
noncomputable def y (x : ℝ) : ℝ := 2 * Real.exp ((8 + 2 * Real.sqrt 11) * x) + Real.exp ((8 - 2 * Real.sqrt 11) * x)

/-- The first derivative of y -/
noncomputable def y' (x : ℝ) : ℝ := (8 + 2 * Real.sqrt 11) * 2 * Real.exp ((8 + 2 * Real.sqrt 11) * x) + 
                      (8 - 2 * Real.sqrt 11) * Real.exp ((8 - 2 * Real.sqrt 11) * x)

/-- The second derivative of y -/
noncomputable def y'' (x : ℝ) : ℝ := (8 + 2 * Real.sqrt 11)^2 * 2 * Real.exp ((8 + 2 * Real.sqrt 11) * x) + 
                       (8 - 2 * Real.sqrt 11)^2 * Real.exp ((8 - 2 * Real.sqrt 11) * x)

theorem cauchy_problem_solution :
  (∀ x, y'' x - 16 * y' x + 20 * y x = 0) ∧
  y 0 = 3 ∧
  y' 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l961_96122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_below_surface_l961_96162

def goldfish_problem (surface_percentage : ℚ) (surface_count : ℕ) : ℕ :=
  let total := (surface_count : ℚ) / surface_percentage
  (total - surface_count).floor.toNat

theorem goldfish_below_surface :
  goldfish_problem (1/4) 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldfish_below_surface_l961_96162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l961_96168

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

-- Theorem statement
theorem f_properties :
  -- 1. The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- 2. The minimum value on [0, π/2] is 0
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 0 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≥ 0) ∧
  -- 3. The maximum value on [0, π/2] is 1 + √2
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1 + Real.sqrt 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ 1 + Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l961_96168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l961_96139

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- The axis of symmetry is x = π/3 + kπ/2, k ∈ ℤ
  (∀ (x : ℝ), ∃ (k : ℤ), f (2 * (Real.pi / 3 + k * Real.pi / 2) - x) = f x) ∧
  -- The function is monotonically increasing in the intervals [−π/6 + kπ, π/3 + kπ], k ∈ ℤ
  (∀ (k : ℤ) (x y : ℝ), -Real.pi/6 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi/3 + k*Real.pi → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l961_96139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l961_96179

theorem trigonometric_sum (t : ℝ) (p q r : ℕ) : 
  (1 + Real.sin t)^2 - (1 + Real.cos t)^2 = 5/8 →
  (1 - Real.sin t)^2 + (1 - Real.cos t)^2 = (p : ℝ)/q + Real.sqrt r →
  Nat.Coprime p q →
  r + p + q = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l961_96179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l961_96199

open Set

theorem complement_intersection_theorem :
  let A : Set ℝ := {x | x + 1 < 0}
  let B : Set ℝ := {x | x - 3 < 0}
  (Aᶜ ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l961_96199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l961_96197

-- Define the function f(x) = ln x + 2x - 6
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 2 3, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l961_96197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l961_96170

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (2 - x) + Real.sqrt (x - 1)

theorem f_domain : Set ℝ = { x | 1 ≤ x ∧ x < 2 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l961_96170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_consumption_l961_96110

/-- Represents the daily lollipop consumption over 7 days -/
def LollipopSequence : Type := Fin 7 → ℚ

/-- The lollipop sequence satisfies the given conditions -/
def is_valid_sequence (seq : LollipopSequence) : Prop :=
  (∀ i : Fin 6, seq (Fin.succ i) = seq i + 3) ∧
  (Finset.sum Finset.univ seq) = 150

theorem fourth_day_consumption (seq : LollipopSequence) 
  (h : is_valid_sequence seq) : seq 3 = 150 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_day_consumption_l961_96110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_only_for_4_or_6_l961_96195

/-- Represents the relationship between two knights -/
inductive Relationship
| Friend
| Enemy
deriving Repr, DecidableEq

/-- A knight arrangement is valid if it satisfies all the given conditions -/
def ValidArrangement (n : ℕ) (relationship : Fin n → Fin n → Relationship) : Prop :=
  -- Every pair of knights has a defined relationship
  (∀ i j : Fin n, i ≠ j → relationship i j = relationship j i) ∧
  -- Each knight has exactly three enemies
  (∀ i : Fin n, (Finset.univ.filter (λ j : Fin n => relationship i j = Relationship.Enemy)).card = 3) ∧
  -- The enemies of a knight's friends are also that knight's enemies
  (∀ i j k : Fin n, relationship i j = Relationship.Friend → 
    relationship j k = Relationship.Enemy → relationship i k = Relationship.Enemy)

/-- The main theorem stating that a valid arrangement is only possible for 4 or 6 knights -/
theorem valid_arrangement_only_for_4_or_6 :
  ∀ n : ℕ, (∃ relationship : Fin n → Fin n → Relationship, ValidArrangement n relationship) 
  ↔ n = 4 ∨ n = 6 := by
  sorry

#check valid_arrangement_only_for_4_or_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_only_for_4_or_6_l961_96195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_polynomials_l961_96109

-- Define the set K
def K : Set ℕ := {n : ℕ | ∀ d, d ∈ n.digits 10 → d ≠ 7}

-- Define the set of valid polynomials
def ValidPoly : Set (ℕ → ℕ) :=
  {f | (∃ k ∈ K, f = λ n ↦ k) ∨
       (∃ a b m : ℕ, a = 10^m ∧ b ∈ K ∧ b < a ∧ f = λ n ↦ a * n + b)}

-- Statement of the theorem
theorem characterization_of_valid_polynomials :
  ∀ f : ℕ → ℕ,
  (∀ n ∈ K, f n ∈ K) ↔ f ∈ ValidPoly :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_valid_polynomials_l961_96109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_percentage_l961_96154

theorem marked_price_percentage (x : ℝ) : 
  (∀ cost_price : ℝ, cost_price > 0 →
    let marked_price := cost_price * (1 + x / 100)
    let selling_price := marked_price * (1 - 20 / 100)
    selling_price = cost_price * (1 + 8 / 100)) →
  x = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_price_percentage_l961_96154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l961_96181

/-- A polar curve defined by r = 3 * sin(θ) -/
noncomputable def polar_curve (θ : ℝ) : ℝ := 3 * Real.sin θ

/-- The Cartesian coordinates of the polar curve -/
noncomputable def cartesian_coords (θ : ℝ) : ℝ × ℝ :=
  (polar_curve θ * Real.cos θ, polar_curve θ * Real.sin θ)

theorem polar_curve_is_circle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ θ, (cartesian_coords θ).1 ^ 2 + (cartesian_coords θ).2 ^ 2 = radius ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l961_96181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l961_96172

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  (∀ x : ℕ, x < a → (∃ y : ℕ, y * x % 39 = 1 ∨ ∃ y : ℕ, y * x % 70 = 1)) ∧
  (∀ y : ℕ, y * a % 39 ≠ 1) ∧
  (∀ y : ℕ, y * a % 70 ≠ 1) →
  a = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l961_96172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l961_96114

/-- Given the total sale price, sales tax rate, and profit rate, 
    calculate the cost price of an article -/
noncomputable def cost_price (total_sale_price : ℝ) (sales_tax_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  total_sale_price / ((1 + sales_tax_rate) * (1 + profit_rate))

/-- Theorem stating that under the given conditions, 
    the cost price is approximately 482.76 -/
theorem cost_price_calculation :
  let total_sale_price : ℝ := 616
  let sales_tax_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.16
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |cost_price total_sale_price sales_tax_rate profit_rate - 482.76| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l961_96114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l961_96159

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x^2 - x) / Real.log a

-- State the theorem
theorem f_monotone_increasing (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Ioo (1/2 : ℝ) 1, f a x > 0) →
  StrictMonoOn (f a) (Set.Iio 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l961_96159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AMN_l961_96144

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line with inclination angle π/4
def line (x y b : ℝ) : Prop := y = x + b

-- Define the point A
def point_A : ℝ × ℝ := (5, 0)

-- Define the function to calculate the area of triangle AMN
noncomputable def area_AMN (b : ℝ) : ℝ := 2 * |5 + b| * (4 * Real.sqrt (1 - b))

-- State the theorem
theorem max_area_triangle_AMN :
  ∃ (b : ℝ), b > -5 ∧ b < 0 ∧
  (∀ (b' : ℝ), b' > -5 → b' < 0 → area_AMN b ≥ area_AMN b') ∧
  area_AMN b = 8 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_AMN_l961_96144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_sequence_result_l961_96143

def derek_sequence : ℕ → ℕ
  | 0 => 100000000
  | (m + 1) => if m % 2 = 0 then (derek_sequence m) / 5 else (derek_sequence m) * 3

theorem derek_sequence_result : derek_sequence 16 = 6^8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derek_sequence_result_l961_96143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_pie_cost_per_serving_l961_96128

/-- Represents the cost of ingredients and number of servings for an apple pie --/
structure ApplePie where
  apple_pounds : ℚ
  apple_price_per_pound : ℚ
  crust_price : ℚ
  lemon_price : ℚ
  butter_price : ℚ
  num_servings : ℕ

/-- Calculates the cost per serving of an apple pie --/
def cost_per_serving (pie : ApplePie) : ℚ :=
  (pie.apple_pounds * pie.apple_price_per_pound + pie.crust_price + pie.lemon_price + pie.butter_price) / pie.num_servings

/-- Theorem: The cost per serving of Hannah's apple pie is $1.00 --/
theorem hannah_pie_cost_per_serving :
  let hannah_pie : ApplePie := {
    apple_pounds := 2,
    apple_price_per_pound := 2,
    crust_price := 2,
    lemon_price := 1/2,
    butter_price := 3/2,
    num_servings := 8
  }
  cost_per_serving hannah_pie = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hannah_pie_cost_per_serving_l961_96128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l961_96118

/-- The length of the longest diagonal of a rhombus -/
noncomputable def longest_diagonal_of_rhombus (area : ℝ) (ratio_long_to_short : ℝ) : ℝ :=
  2 * (ratio_long_to_short / (ratio_long_to_short + 1)) * Real.sqrt (2 * area * (ratio_long_to_short + 1) / ratio_long_to_short)

/-- Theorem: The longest diagonal of a rhombus with area 150 and diagonal ratio 5:3 is 10√5 -/
theorem rhombus_longest_diagonal : longest_diagonal_of_rhombus 150 (5/3) = 10 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longest_diagonal_l961_96118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l961_96165

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

noncomputable def sum_geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_problem :
  ∀ a₁ q : ℝ,
  (a₁ * (a₁ * q) * (a₁ * q^2) = 27) →
  (a₁ * q + a₁ * q^3 = 30) →
  ((a₁ = 1 ∧ q = 3 ∧ sum_geometric_sequence a₁ q 6 = 364) ∨
   (a₁ = -1 ∧ q = -3 ∧ sum_geometric_sequence a₁ q 6 = -182)) := by
  sorry

#check geometric_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l961_96165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l961_96185

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ c : ℝ, c = (9 : ℝ) / 4 ∧ 
   c = (9 : ℕ).choose 8 * a * (-(1 : ℝ) / Real.sqrt 2) ^ 8) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l961_96185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_stationary_points_and_asymptotic_behavior_l961_96102

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := 1 - 2 / ((x + 1)^2)

-- Theorem statement
theorem f_stationary_points_and_asymptotic_behavior :
  (∃ x : ℝ, f' x = 0 ∧ (x = Real.sqrt 2 - 1 ∨ x = -Real.sqrt 2 - 1)) ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ x > M, |f x - (x - 1)| < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_stationary_points_and_asymptotic_behavior_l961_96102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l961_96129

noncomputable def f (lambda : ℝ) (x : ℝ) : ℝ := x + 2 * Real.cos x + lambda

theorem triangle_existence (lambda : ℝ) : 
  (∀ x₁ x₂ x₃ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ Real.pi/2 ∧ 
                   0 ≤ x₂ ∧ x₂ ≤ Real.pi/2 ∧ 
                   0 ≤ x₃ ∧ x₃ ≤ Real.pi/2 → 
    f lambda x₁ + f lambda x₂ > f lambda x₃ ∧ 
    f lambda x₂ + f lambda x₃ > f lambda x₁ ∧ 
    f lambda x₃ + f lambda x₁ > f lambda x₂) ↔ 
  lambda > Real.sqrt 3 - 5 * Real.pi / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l961_96129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rectangle_in_circle_l961_96173

/-- The area of the largest rectangle that can be inscribed in a circle --/
def maxInscribedRectangleArea (r : ℝ) : ℝ :=
  2 * r^2

theorem largest_rectangle_in_circle (r : ℝ) (h : r = 6) : 
  ∃ (a : ℝ), a = 72 ∧ a = maxInscribedRectangleArea r :=
by
  use 72
  constructor
  · rfl
  · rw [maxInscribedRectangleArea, h]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rectangle_in_circle_l961_96173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_4_l961_96140

/-- The set of numbers on the balls -/
def ball_numbers : Finset ℕ := Finset.range 12 ∪ {12}

/-- The condition for the sum of two numbers being divisible by 4 -/
def sum_divisible_by_4 (a b : ℕ) : Prop := (a + b) % 4 = 0

/-- The number of ways to choose two balls -/
def total_outcomes : ℕ := 12 * 11

/-- The number of ways to choose two balls where their sum is divisible by 4 -/
def favorable_outcomes : ℕ := 24

theorem probability_sum_divisible_by_4 :
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_4_l961_96140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l961_96147

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^(-2 : ℤ) * y^(-1 : ℤ)) / (x^(-4 : ℤ) - y^(-2 : ℤ)) = (x^2 * y) / (y^2 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_fraction_l961_96147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_g_geq_h_implies_m_bound_l961_96127

-- Define the functions g and h
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 5 * Real.log x

def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- Part I: g is increasing implies a > 5/2
theorem g_increasing_implies_a_bound (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x < y → g a x < g a y) → a > 5/2 := by sorry

-- Part II: Existence of x₁ such that g(x₁) ≥ h(x₂) for all x₂ ∈ [1, 2] implies m ≥ 8 - 5ln(2)
theorem g_geq_h_implies_m_bound (m : ℝ) :
  (∃ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Icc 1 2, g 2 x₁ ≥ h m x₂) →
  m ≥ 8 - 5 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_implies_a_bound_g_geq_h_implies_m_bound_l961_96127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l961_96153

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Define the solution set M
def M : Set ℝ := {x | -2/3 ≤ x ∧ x ≤ 6}

-- Define the range of a
def A : Set ℝ := Set.Iic (-2) ∪ Set.Ici 4

theorem problem_solution :
  (∀ x, f x ≥ -2 ↔ x ∈ M) ∧
  (∀ a, (∀ x, x ≥ a → f x ≤ x - a) ↔ a ∈ A) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l961_96153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l961_96150

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 2

-- Define the line
def lineLB (x y b : ℝ) : Prop := y = 3 * x + b

-- Define the area S (enclosed by circle C and y-axis in second quadrant)
noncomputable def area_S : ℝ := sorry

-- Define the area enclosed by the line and circle
noncomputable def area_enclosed_by_line_and_circle (b : ℝ) : ℝ := sorry

-- Statement of the theorem
theorem circle_line_intersection (b : ℝ) : 
  (∃ (S' : ℝ), S' = area_S ∧ 
    (area_enclosed_by_line_and_circle b = S')) → 
  b = -1 - Real.sqrt 10 ∨ b = -1 + Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l961_96150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_problem_l961_96137

/-- Calculates the amount of milk remaining after a certain number of dilutions -/
noncomputable def milkRemaining (initialVolume : ℝ) (removedVolume : ℝ) (numDilutions : ℕ) : ℝ :=
  let f (x : ℝ) := x * (1 - removedVolume / initialVolume)
  (f^[numDilutions]) initialVolume

/-- The problem statement -/
theorem milk_dilution_problem :
  let initialVolume : ℝ := 100
  let removedVolume : ℝ := 15
  let numDilutions : ℕ := 4
  abs (milkRemaining initialVolume removedVolume numDilutions - 52.2) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_problem_l961_96137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l961_96166

-- Define the circles
def circle_E (x y : ℝ) : Prop := x^2 + y^2 = 1/4
def circle_F (x y : ℝ) : Prop := (x-3)^2 + (y+1)^2 = 9/4

-- Define the point P
def P (t : ℝ) : ℝ × ℝ := (t, t-1)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem max_distance_difference :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (t ex ey fx fy : ℝ),
    circle_E ex ey →
    circle_F fx fy →
    distance (P t).1 (P t).2 fx fy - distance (P t).1 (P t).2 ex ey ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l961_96166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l961_96184

/-- The cost of a single bottle cap in currency A -/
def single_cap_cost : ℚ := 2

/-- The number of bottle caps that qualify for a discount -/
def discount_threshold : ℕ := 4

/-- The discount rate applied when buying the discount threshold number of caps -/
def discount_rate : ℚ := 1/10

/-- The total number of bottle caps to be purchased -/
def total_caps : ℕ := 6

/-- The conversion rate from currency A to currency B -/
def conversion_rate : ℚ := 3

/-- The function to calculate the total cost in currency B for purchasing 'total_caps' bottle caps -/
def total_cost_currency_B : ℚ :=
  let discounted_caps := min discount_threshold total_caps
  let full_price_caps := total_caps - discounted_caps
  let discounted_cost := (single_cap_cost * discounted_caps) * (1 - discount_rate)
  let full_price_cost := single_cap_cost * full_price_caps
  (discounted_cost + full_price_cost) * conversion_rate

theorem total_cost_is_correct : total_cost_currency_B = 336/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_correct_l961_96184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_l961_96121

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 1 + a n)

theorem max_sum_at_6 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a)
  (h_sum : a 6 + a 7 < 0) (h_s11 : S a 11 > 0) :
  ∀ n : ℕ, S a n ≤ S a 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_l961_96121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l961_96161

def A : Matrix (Fin 2) (Fin 3) ℝ := !![1, 2, -1; 2, 2, -3]

def B (a : ℝ) : Matrix (Fin 3) (Fin 1) ℝ := !![a; -2*a; 3*a]

theorem matrix_equation_solution (a : ℝ) :
  A * (B a) = !![12; 22] → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_solution_l961_96161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l961_96123

theorem polynomial_remainder : 
  ∃ q : Polynomial ℝ, X^100 = q * ((X^2 + 1) * (X - 1)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l961_96123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_polynomial_trig_identity_l961_96178

/-- A non-constant homogeneous polynomial with real coefficients -/
def HomogeneousPolynomial (P : ℝ → ℝ → ℝ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ ∀ (r x y : ℝ), P (r * x) (r * y) = r^n * P x y

/-- The property that P(sin t, cos t) = 1 for all real t -/
def TrigIdentity (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P (Real.sin t) (Real.cos t) = 1

theorem homogeneous_polynomial_trig_identity 
  (P : ℝ → ℝ → ℝ) 
  (h_homog : HomogeneousPolynomial P) 
  (h_trig : TrigIdentity P) : 
  ∃ (k : ℕ), k > 0 ∧ ∀ (x y : ℝ), P x y = (x^2 + y^2)^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_polynomial_trig_identity_l961_96178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_customers_l961_96103

/-- Rachel's earnings as a waitress --/
structure WaitressEarnings where
  hourly_wage : ℚ
  tip_per_customer : ℚ
  total_earnings : ℚ

/-- Calculate the number of customers served --/
def customers_served (earnings : WaitressEarnings) : ℚ :=
  (earnings.total_earnings - earnings.hourly_wage) / earnings.tip_per_customer

/-- Theorem: Rachel served 20 customers --/
theorem rachel_customers (earnings : WaitressEarnings)
  (h1 : earnings.hourly_wage = 12)
  (h2 : earnings.tip_per_customer = 5/4)
  (h3 : earnings.total_earnings = 37) :
  customers_served earnings = 20 := by
  sorry

#eval customers_served { hourly_wage := 12, tip_per_customer := 5/4, total_earnings := 37 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_customers_l961_96103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_12_percent_l961_96115

-- Define the given values
noncomputable def total_value : ℚ := 1720
noncomputable def non_taxable_amount : ℚ := 600
noncomputable def tax_paid : ℚ := 134.4

-- Define the tax rate calculation
noncomputable def calculate_tax_rate (total : ℚ) (non_taxable : ℚ) (tax : ℚ) : ℚ :=
  (tax / (total - non_taxable)) * 100

-- Theorem to prove
theorem tax_rate_is_12_percent :
  calculate_tax_rate total_value non_taxable_amount tax_paid = 12 := by
  -- Unfold the definition of calculate_tax_rate
  unfold calculate_tax_rate
  -- Simplify the expression
  simp [total_value, non_taxable_amount, tax_paid]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_is_12_percent_l961_96115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_nine_example_achieves_smallest_difference_l961_96124

/-- A type representing 9-digit integers containing all digits from 1 to 9 exactly once -/
def NineDigitPermutation : Type := 
  { n : ℕ // n ≥ 100000000 ∧ n < 1000000000 ∧ 
    (∀ d : Fin 9, ∃! i : Fin 9, (n / (10 ^ i.val)) % 10 = d.val + 1) }

/-- The smallest difference between two different NineDigitPermutations -/
def smallestDifference : ℕ := 9

/-- Theorem stating that smallestDifference is indeed the smallest possible difference -/
theorem smallest_difference_is_nine :
  ∀ (x y : NineDigitPermutation), x ≠ y → x.val - y.val ≥ smallestDifference := by
  sorry

/-- Example of two NineDigitPermutations with the smallest difference -/
def example_x : NineDigitPermutation := 
  ⟨987654321, by sorry⟩

def example_y : NineDigitPermutation := 
  ⟨987654312, by sorry⟩

/-- Theorem verifying that the example achieves the smallest difference -/
theorem example_achieves_smallest_difference :
  example_x.val - example_y.val = smallestDifference := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_is_nine_example_achieves_smallest_difference_l961_96124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_135_degrees_l961_96151

theorem cos_sin_135_degrees :
  let angle : Real := 135 * π / 180
  Real.cos angle = -Real.sqrt 2 / 2 ∧ Real.sin angle = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_135_degrees_l961_96151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wrapping_paper_size_correct_l961_96130

/-- The minimum size of a square sheet of wrapping paper for a box -/
noncomputable def min_wrapping_paper_size (l w h : ℝ) : ℝ :=
  Real.sqrt ((l^2 / 2) + (w^2 / 2) + 2 * h^2)

/-- Theorem stating the minimum size of wrapping paper for a box -/
theorem min_wrapping_paper_size_correct (l w h : ℝ) (l_pos : 0 < l) (w_pos : 0 < w) (h_pos : 0 < h) :
  ∀ s : ℝ, s ≥ min_wrapping_paper_size l w h ↔ 
    (∃ (paper : ℝ × ℝ → ℝ), 
      (∀ x y, 0 ≤ x ∧ x ≤ s ∧ 0 ≤ y ∧ y ≤ s → 0 ≤ paper (x, y)) ∧
      (paper (0, 0) = 0) ∧
      (paper (l/2, w/2) = h) ∧
      (paper (s, s) = 0) ∧
      (∀ x y, 0 ≤ x ∧ x ≤ s ∧ 0 ≤ y ∧ y ≤ s → 
        paper (x, y) + paper (s-x, y) + paper (x, s-y) + paper (s-x, s-y) = h)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_wrapping_paper_size_correct_l961_96130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l961_96136

/-- A hyperbola centered at the origin passing through specific points -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  s : ℝ
  -- The hyperbola passes through (2, 5)
  point1 : 2^2 / a^2 - 5^2 / b^2 = 1
  -- The hyperbola passes through (3, 0)
  point2 : 3^2 / a^2 - 0^2 / b^2 = 1
  -- The hyperbola passes through (s, -3)
  point3 : s^2 / a^2 - (-3)^2 / b^2 = 1
  -- a and b are positive real numbers
  a_pos : a > 0
  b_pos : b > 0

/-- The theorem stating that s² = 288/25 for the given hyperbola -/
theorem hyperbola_s_squared (h : Hyperbola) : h.s^2 = 288/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_s_squared_l961_96136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prosecutor_and_guilt_conclusion_l961_96183

-- Define the possible types of prosecutors
inductive Prosecutor
| Knight
| Liar

-- Define the guilt status of a resident
inductive GuiltStatus
| Guilty
| NotGuilty

-- Define the function that determines if a statement is true based on the prosecutor type
def statementTruth (p : Prosecutor) (s : Prop) : Prop :=
  match p with
  | Prosecutor.Knight => s
  | Prosecutor.Liar => ¬s

-- Define the function that represents the prosecutor's first statement
def statement1 (p : Prosecutor) (x : GuiltStatus) : Prop :=
  statementTruth p (x = GuiltStatus.Guilty)

-- Define the function that represents the prosecutor's second statement
def statement2 (p : Prosecutor) (x y : GuiltStatus) : Prop :=
  statementTruth p (¬(x = GuiltStatus.Guilty ∧ y = GuiltStatus.Guilty))

-- Theorem to prove
theorem prosecutor_and_guilt_conclusion :
  ∃ (p : Prosecutor) (x y : GuiltStatus),
    statement1 p x ∧
    statement2 p x y ∧
    p = Prosecutor.Knight ∧
    x = GuiltStatus.Guilty ∧
    y = GuiltStatus.NotGuilty := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prosecutor_and_guilt_conclusion_l961_96183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l961_96111

/-- The function f to be minimized -/
noncomputable def f (a b c : ℝ) : ℝ := (a+b)^4 + (b+c)^4 + (c+a)^4 - (4/7)*(a^4+b^4+c^4)

/-- Theorem stating that 26/45 is the minimal value of f -/
theorem f_min_value :
  ∀ a b c : ℝ, f a b c ≥ 26/45 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l961_96111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l961_96156

/-- Given a hyperbola with focal length 10 and imaginary axis length 8,
    its standard equation is either (x²/9) - (y²/16) = 1 or (y²/9) - (x²/16) = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (x^2 / 9 - y^2 / 16 = 1) ∨ (y^2 / 9 - x^2 / 16 = 1) :=
by
  -- Define constants
  let focal_length : ℝ := 10
  let imaginary_axis_length : ℝ := 8
  let a : ℝ := Real.sqrt (25 - 16)
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l961_96156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l961_96135

theorem positive_difference_theorem : |((8^2 + 8^2) / 8 : ℝ) - ((8^2 * 8^2) / 8 : ℝ)| = 496 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_theorem_l961_96135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l961_96131

/-- Vector a in ℝ³ -/
def a : Fin 3 → ℝ := ![4, -2, -4]

/-- Vector b in ℝ³ -/
def b : Fin 3 → ℝ := ![6, -3, 2]

theorem vector_operations :
  (a + b = ![10, -5, -2]) ∧ (Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2) = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l961_96131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_intersection_l961_96100

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define the slope of a line through the origin and a point
noncomputable def line_slope (p : ℝ × ℝ) : ℝ :=
  p.2 / p.1

-- Theorem statement
theorem parabola_fixed_point_intersection
  (A B : ℝ × ℝ)
  (hA : point_on_parabola A)
  (hB : point_on_parabola B)
  (hO : A ≠ (0, 0) ∧ B ≠ (0, 0) ∧ A ≠ B)
  (hslope : line_slope A * line_slope B = -1/3) :
  ∃ (t : ℝ), t * A.1 + (1 - t) * B.1 = 12 ∧
             t * A.2 + (1 - t) * B.2 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fixed_point_intersection_l961_96100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_savings_percentage_l961_96105

noncomputable def net_salary : ℝ := 3500

noncomputable def discretionary_income (salary : ℝ) : ℝ := salary / 5

def vacation_fund_percentage : ℝ := 30
def eating_out_percentage : ℝ := 35
def gifts_amount : ℝ := 105

noncomputable def savings_percentage (salary : ℝ) : ℝ :=
  let d := discretionary_income salary
  let remaining := d - (vacation_fund_percentage / 100 * d) - (eating_out_percentage / 100 * d) - gifts_amount
  (remaining / d) * 100

theorem jills_savings_percentage :
  savings_percentage net_salary = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_savings_percentage_l961_96105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_at_5_l961_96167

/-- A line in 3D space parameterized by t -/
def line (t : ℝ) : ℝ × ℝ × ℝ := sorry

/-- The vector on the line at t = 1 -/
axiom vector_at_1 : line 1 = (2, -1, 3)

/-- The vector on the line at t = 4 -/
axiom vector_at_4 : line 4 = (8, -5, 11)

/-- Theorem: The vector on the line at t = 5 -/
theorem vector_at_5 : line 5 = (10, -19/3, 41/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_at_5_l961_96167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l961_96192

-- Define the function v(x)
noncomputable def v (x : ℝ) : ℝ := Real.sqrt (x - 2) + (x - 3) ^ (1/3) + 1 / (x - 5)

-- Define the domain of v(x)
def domain_v : Set ℝ := {x | x > 2 ∧ x ≠ 5}

-- Theorem statement
theorem domain_of_v :
  ∀ x : ℝ, x ∈ domain_v ↔ (∃ y : ℝ, v x = y) :=
by
  sorry

#check domain_of_v

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_v_l961_96192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_circle_l961_96112

-- Define the ∘ operation
def myCircle (x y : ℝ) : ℝ := 4 * x - 2 * y + 2 * x * y

-- State the theorem
theorem unique_solution_circle :
  ∃! y : ℝ, myCircle 5 y = 20 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_circle_l961_96112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_only_students_l961_96176

theorem art_only_students (total science art : ℕ) 
  (h1 : total = 120)
  (h2 : science = 85)
  (h3 : art = 65)
  (h4 : ∀ s, s < total → (s < science ∨ s < art))
  : art - (science + art - total) = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_only_students_l961_96176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_l961_96158

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (4 - x) * Real.exp (x - 2)

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (3 - x) * Real.exp (x - 2)

-- Theorem statement
theorem no_tangent_line :
  ¬ ∃ (m : ℝ), ∃ (x : ℝ), 
    (f x = (3 * x + m) / 2) ∧ 
    (f_derivative x = 3 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_tangent_line_l961_96158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_exceeds_2018_l961_96171

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then sequence_a n + 1 else 2 * sequence_a n + 1

def sum_sequence (n : ℕ) : ℕ :=
  (List.range n).map sequence_a |> List.sum

theorem min_sum_exceeds_2018 : 
  (∀ k < 17, sum_sequence k ≤ 2018) ∧ sum_sequence 17 > 2018 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_exceeds_2018_l961_96171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_stripes_area_is_36_l961_96119

/-- Represents an isosceles right triangle with legs of length 12,
    partitioned into 36 congruent smaller triangles. -/
structure PartitionedTriangle where
  leg_length : ℝ
  num_partitions : ℕ
  leg_length_eq : leg_length = 12
  num_partitions_eq : num_partitions = 36

/-- Calculates the area of alternating vertical stripes in the partitioned triangle,
    starting from the second stripe and ending at the second-to-last stripe. -/
def alternating_stripes_area (t : PartitionedTriangle) : ℝ :=
  -- Implementation to be filled later
  36 -- Placeholder return value

theorem alternating_stripes_area_is_36 (t : PartitionedTriangle) :
  alternating_stripes_area t = 36 := by
  -- Proof to be implemented later
  sorry

#check alternating_stripes_area_is_36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_stripes_area_is_36_l961_96119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_rotated_semicircle_l961_96117

/-- The area of a shaded figure formed by rotating a semicircle -/
theorem shaded_area_rotated_semicircle (R : ℝ) (h : R > 0) : 
  let α : ℝ := 45 * π / 180  -- Convert 45° to radians
  (π * R^2) / 2 = (π * R^2) / 2 :=
by
  -- Introduce the let binding
  intro α
  -- The left-hand side and right-hand side are identical, so we can use rfl
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_rotated_semicircle_l961_96117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l961_96116

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m - 2) * x^m

-- State the theorem
theorem decreasing_power_function (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂) → m = -3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l961_96116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kramer_needs_403959_votes_l961_96132

def election_problem (kramer_votes : ℕ) (kramer_percentage : ℚ) (total_candidates : ℕ) : ℕ :=
  let total_votes : ℕ := (kramer_votes : ℚ) / kramer_percentage |>.ceil.toNat
  let majority_votes : ℕ := (total_votes + 1) / 2
  majority_votes - kramer_votes

theorem kramer_needs_403959_votes :
  election_problem 942568 (35/100) 3 = 403959 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kramer_needs_403959_votes_l961_96132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l961_96187

theorem root_of_equation : ∃ (x : ℝ), x = (2 : ℝ)^(1/3) + (3 : ℝ)^(1/3) ∧ x^9 - 15*x^6 - 87*x^3 - 125 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_equation_l961_96187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l961_96189

-- Define the hyperbola and its properties
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

-- Define the points and their properties
def Points (O F B A : ℝ × ℝ) (a b : ℝ) : Prop :=
  let c := Real.sqrt (a^2 + b^2)
  F = (c, 0) ∧
  (B.1^2 / a^2) - (B.2^2 / b^2) = 1 ∧
  B.1 < 0 ∧
  (∃ t : ℝ, A = (t * a / b, t))

-- Define the vector properties
def VectorProperties (O F B A : ℝ × ℝ) : Prop :=
  ((F.1 - B.1) * A.1 + (F.2 - B.2) * A.2 = 0) ∧
  (2 * A.1 = B.1 + F.1 ∧ 2 * A.2 = B.2 + F.2)

-- Main theorem
theorem hyperbola_eccentricity
  (a b : ℝ)
  (O F B A : ℝ × ℝ)
  (h1 : Hyperbola a b)
  (h2 : Points O F B A a b)
  (h3 : VectorProperties O F B A) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l961_96189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_in_interval_l961_96160

/-- The function f(x) = 6x / (1 + x^2) -/
noncomputable def f (x : ℝ) : ℝ := 6 * x / (1 + x^2)

/-- The maximum value of f(x) in [0, 3] is 3 -/
theorem f_max_value_in_interval : 
  (∀ x ∈ Set.Icc 0 3, f x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 3, f x = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_in_interval_l961_96160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40_l961_96180

noncomputable section

/-- The area of a square with side length s -/
def square_area (s : ℝ) : ℝ := s * s

/-- The area of a right-angled triangle with base b and height h -/
def triangle_area (b h : ℝ) : ℝ := (1 / 2) * b * h

/-- The sum of areas of eight right-angled triangles with base 2 and heights 4, 6, 8, 10 in pairs -/
def eight_triangles_area : ℝ := 2 * (triangle_area 2 4 + triangle_area 2 6 + triangle_area 2 8 + triangle_area 2 10)

/-- The area of the shaded region -/
def shaded_area : ℝ := square_area 10 - square_area 2 - eight_triangles_area

theorem shaded_area_is_40 : shaded_area = 40 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_40_l961_96180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_50_factorial_l961_96108

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i ↦ i + 1)

def count_multiples (n : ℕ) (d : ℕ) : ℕ := n / d

theorem exponent_of_five_in_50_factorial :
  ∃ (k : ℕ), factorial 50 = k * (5^12) ∧ k % 5 ≠ 0 := by
  sorry

#eval factorial 50
#eval count_multiples 50 5
#eval count_multiples 50 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_five_in_50_factorial_l961_96108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_l961_96164

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- Given point P -/
def P : ℝ × ℝ := (1, -1)

/-- Point M on the ellipse -/
noncomputable def M : ℝ × ℝ := (2 * Real.sqrt 6 / 3, -1)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Expression to be minimized -/
noncomputable def expr (x : ℝ × ℝ) : ℝ := distance x P + 2 * distance x F

theorem ellipse_minimum :
  ellipse M.1 M.2 ∧
  ∀ x y : ℝ, ellipse x y → expr (x, y) ≥ expr M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_l961_96164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l961_96196

/-- The coefficient of the term containing x in the expansion of (√x + 1/⁴√x)^8 is 70 -/
theorem binomial_expansion_coefficient (x : ℝ) (x_pos : x > 0) : 
  (Finset.range 9).sum (fun k => (Nat.choose 8 k : ℝ) * (x^(1/2 : ℝ))^(8-k) * (x^(-(1/4 : ℝ)))^k) = 
  70 * x + (Finset.range 9).sum (fun k => 
    if k ≠ 4 then (Nat.choose 8 k : ℝ) * (x^(1/2 : ℝ))^(8-k) * (x^(-(1/4 : ℝ)))^k else 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l961_96196
