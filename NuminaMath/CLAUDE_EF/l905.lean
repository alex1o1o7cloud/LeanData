import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_papaya_height_after_five_years_l905_90516

/-- Represents the height growth of a papaya tree over 5 years -/
def papaya_growth : Fin 5 → ℝ
| 0 => 2  -- First year growth
| 1 => 2 * 1.5  -- Second year growth
| 2 => 2 * 1.5 * 1.5  -- Third year growth
| 3 => 2 * 1.5 * 1.5 * 2  -- Fourth year growth
| 4 => 2 * 1.5 * 1.5 * 2 * 0.5  -- Fifth year growth

/-- The total height of the papaya tree after 5 years -/
def total_height : ℝ := (Finset.range 5).sum (fun i => papaya_growth i)

/-- Theorem stating that the total height of the papaya tree after 5 years is 23 feet -/
theorem papaya_height_after_five_years : total_height = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_papaya_height_after_five_years_l905_90516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_matrix_det_is_one_l905_90521

theorem binomial_matrix_det_is_one (n : ℕ) : 
  let D : Matrix (Fin (n + 1)) (Fin (n + 1)) ℤ := 
    λ i j => (Nat.cast i + Nat.cast j - 2).choose (Nat.cast j - 1)
  Matrix.det D = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_matrix_det_is_one_l905_90521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_properties_l905_90574

def sequence_b : ℕ → ℝ
  | 0 => 2  -- We define b₀ as 2 to match b₁ in the problem
  | 1 => 3  -- This matches b₂ in the problem
  | (n + 2) => sequence_b n * sequence_b (n + 1) + 1

def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem sequence_b_properties :
  ¬ is_geometric_progression sequence_b ∧ sequence_b 2 = 7 := by
  constructor
  · sorry  -- Proof that the sequence is not a geometric progression
  · sorry  -- Proof that b₃ (which is sequence_b 2 in our indexing) equals 7

#eval sequence_b 2  -- This should output 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_properties_l905_90574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l905_90584

-- Define the variables
noncomputable def a : Real := Real.log 0.32
noncomputable def b : Real := Real.log 0.33
def c : Real := 20.3
def d : Real := 0.32

-- State the theorem
theorem order_relation : b < a ∧ a < d ∧ d < c := by
  -- Split the conjunctions
  apply And.intro
  · -- Prove b < a
    sorry
  apply And.intro
  · -- Prove a < d
    sorry
  · -- Prove d < c
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l905_90584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_person_who_announced_six_l905_90563

/-- Represents the circle of 10 people and their announced averages. -/
def Circle := Fin 10 → ℝ

/-- The given announced averages. -/
def announced_averages : Circle := fun i =>
  match i with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6
  | 6 => 7
  | 7 => 8
  | 8 => 9
  | 9 => 10

/-- The original numbers picked by each person. -/
def original_numbers : Circle → Fin 10 → ℝ := sorry

/-- Theorem stating that the number picked by the person who announced 6 is 1. -/
theorem number_of_person_who_announced_six (c : Circle) 
  (h : c = announced_averages) :
  ∃ (nums : Fin 10 → ℝ), 
    (∀ i, c i = (nums (i - 1) + nums (i + 1)) / 2) ∧ 
    nums 5 = 1 := by
  sorry

#check number_of_person_who_announced_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_person_who_announced_six_l905_90563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_prime_digits_l905_90597

-- Define the set of prime digits
def prime_digits : Finset Nat := {2, 3, 5, 7}

-- Define a function to check if a number is a three-digit integer with all prime digits
def is_valid_number (n : Nat) : Bool :=
  100 ≤ n && n < 1000 &&
  (n / 100) ∈ prime_digits &&
  ((n / 10) % 10) ∈ prime_digits &&
  (n % 10) ∈ prime_digits

-- State the theorem
theorem count_three_digit_prime_digits :
  (Finset.filter (fun n => is_valid_number n) (Finset.range 1000)).card = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_prime_digits_l905_90597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l905_90598

/-- Represents the total number of students in the class. -/
def N : ℕ := 23

/-- Represents the number of honor students in the class. -/
def honorStudents : ℕ := N - 8

/-- The statement made by each student is true for honor students and false for bullies. -/
def statementConsistency (n : ℕ) : Prop :=
  (8 : ℚ) / (n - 1 : ℚ) ≥ 1 / 3 ∧ (7 : ℚ) / (n - 1 : ℚ) < 1 / 3

/-- The theorem stating that N can only be 23, 24, or 25. -/
theorem class_size_theorem : N = 23 ∨ N = 24 ∨ N = 25 := by
  sorry

#check class_size_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_theorem_l905_90598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tg_arccos_le_cos_arctg_l905_90569

theorem tg_arccos_le_cos_arctg (x : ℝ) : 
  Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x) ↔ 
    x ∈ Set.Icc (-1) 0 ∪ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tg_arccos_le_cos_arctg_l905_90569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l905_90589

/-- The eccentricity of an ellipse with given conditions -/
theorem ellipse_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (E : ℝ × ℝ → Prop)
  (hE : E = λ (x, y) => x^2 / a^2 + y^2 / b^2 = 1)
  (M N : ℝ × ℝ)
  (hMN : E M ∧ E N)
  (hMidpoint : (M.1 + N.1) / 2 = -1)
  (hPerpBisector : ∃ (k : ℝ), k * (-3/4 - (-1)) = (M.2 + N.2) / 2) :
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l905_90589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l905_90582

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the focal length of the ellipse
noncomputable def focal_length_ellipse : ℝ := 4

-- Define the sum of eccentricities
noncomputable def sum_eccentricities : ℝ := 14 / 5

-- Define the standard form of a hyperbola
noncomputable def hyperbola (a b x y : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), 
    (∀ x y, hyperbola a b x y ↔ y^2 / 4 - x^2 / 12 = 1) ∧
    (∃ c, c = focal_length_ellipse ∧ c^2 = a^2 + b^2) ∧
    (focal_length_ellipse / a + (focal_length_ellipse / 5) = sum_eccentricities) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l905_90582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_scale_3_l905_90501

/-- Represents the weight of a triangle -/
def triangle_weight : ℚ := sorry

/-- Represents the weight of a square -/
def square_weight : ℚ := sorry

/-- Represents the weight of a circle -/
def circle_weight : ℚ := sorry

/-- Scale (1): 3 triangles + 1 circle = 6 squares -/
axiom scale_1 : 3 * triangle_weight + circle_weight = 6 * square_weight

/-- Scale (2): 2 triangles + 4 circles = 8 squares -/
axiom scale_2 : 2 * triangle_weight + 4 * circle_weight = 8 * square_weight

/-- The number of squares needed to balance scale (3) -/
def squares_needed : ℚ := 10

/-- Theorem: The number of squares needed to balance scale (3) is 10 -/
theorem balance_scale_3 : 4 * triangle_weight + 3 * circle_weight = squares_needed * square_weight := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_scale_3_l905_90501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_current_age_l905_90510

/-- The current age of person A -/
def a : ℕ := sorry

/-- The current age of person B -/
def b : ℕ := sorry

/-- In 10 years, A's age will be twice B's age from 10 years ago -/
axiom future_age_relation : a + 10 = 2 * (b - 10)

/-- A is currently 5 years older than B -/
axiom current_age_difference : a = b + 5

/-- Theorem: B's current age is 35 years -/
theorem b_current_age : b = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_current_age_l905_90510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l905_90593

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : sum_arithmetic a 10 = 60) 
  (h_seventh : a 7 = 7) : 
  a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_fourth_term_l905_90593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tesla_dealership_theorem_l905_90539

/-- Represents the financial details of a Tesla Model S car dealership --/
structure TeslaDealership where
  purchasePrice : ℝ  -- in million rubles
  customsFees : ℝ    -- in million rubles
  monthlyRent : ℝ    -- in million rubles
  monthlySalary : ℝ  -- in million rubles
  monthlyExpenses : ℝ -- in million rubles
  orderSize : ℕ
  competitorPrice : ℝ -- in million rubles
  chargingStationPrice : ℝ -- in million rubles

/-- Calculates the minimum possible price per car --/
noncomputable def minPricePerCar (d : TeslaDealership) : ℝ :=
  ((d.purchasePrice + d.customsFees) * d.orderSize + d.monthlyRent + d.monthlySalary + d.monthlyExpenses) / d.orderSize

/-- Calculates the minimum order volume to compete --/
noncomputable def minOrderVolume (d : TeslaDealership) : ℕ :=
  ⌈(d.monthlyRent + d.monthlySalary + d.monthlyExpenses) / (d.competitorPrice - d.chargingStationPrice - d.purchasePrice - d.customsFees)⌉.toNat

/-- Theorem stating the minimum price per car and minimum order volume to compete --/
theorem tesla_dealership_theorem (d : TeslaDealership) 
  (h1 : d.purchasePrice = 2.5)
  (h2 : d.customsFees = 2)
  (h3 : d.monthlyRent = 0.05)
  (h4 : d.monthlySalary = 0.37)
  (h5 : d.monthlyExpenses = 0.18)
  (h6 : d.orderSize = 30)
  (h7 : d.competitorPrice = 5.3)
  (h8 : d.chargingStationPrice = 0.4) :
  minPricePerCar d = 4.52 ∧ minOrderVolume d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tesla_dealership_theorem_l905_90539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_eccentricity_l905_90571

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with equation y² = 8x -/
structure Parabola where

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_parabola_eccentricity 
  (h : Hyperbola) 
  (p : Parabola) 
  (F : Point) -- Common focus
  (M : Point) -- Intersection point
  (h_focus : F.x = 2 ∧ F.y = 0) -- Focus of parabola y² = 8x
  (h_intersect : M.y^2 = 8 * M.x ∧ M.x^2 / h.a^2 - M.y^2 / h.b^2 = 1) -- M is on both curves
  (h_distance : distance M F = 5) -- |MF| = 5
  : eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_eccentricity_l905_90571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l905_90522

-- Define the sequence aₙ
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a n

-- Define the sum function Sₙ
def S (n : ℕ) : ℕ := (Finset.range n).sum (λ i => a i)

-- Theorem statement
theorem sum_of_first_five_terms : S 5 = 31 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_five_terms_l905_90522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rectangles_and_circles_are_axisymmetric_and_centrosymmetric_l905_90576

-- Define the set of shapes we're considering
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | IsoscelesTrapezoid
  | Circle

-- Define properties
axiom isAxisymmetric : Shape → Prop
axiom isCentrosymmetric : Shape → Prop

-- Define the property of being both axisymmetric and centrosymmetric
def isAxisymmetricAndCentrosymmetric (s : Shape) : Prop :=
  isAxisymmetric s ∧ isCentrosymmetric s

-- Theorem statement
theorem only_rectangles_and_circles_are_axisymmetric_and_centrosymmetric :
  ∀ s : Shape, isAxisymmetricAndCentrosymmetric s ↔ (s = Shape.Rectangle ∨ s = Shape.Circle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_rectangles_and_circles_are_axisymmetric_and_centrosymmetric_l905_90576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_earth_study_method_l905_90564

/-- Represents the possible answers to the question about Earth's shape study -/
inductive EarthStudyMethod
  | RemoteSensingAndGIS
  | GPSAndGIS
  | RemoteSensingAndGPS
  | DigitalEarth

/-- The correct method for precise study of Earth's shape -/
def correctMethod : EarthStudyMethod := EarthStudyMethod.RemoteSensingAndGIS

/-- Theorem stating that the correct method is Remote Sensing and GIS -/
theorem correct_earth_study_method :
  correctMethod = EarthStudyMethod.RemoteSensingAndGIS := by
  -- The proof is trivial as it's defined to be true
  rfl

#check correct_earth_study_method

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_earth_study_method_l905_90564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l905_90567

-- Define the triangle and its properties
def Triangle (A B C M : ℝ × ℝ) : Prop :=
  ∃ (E D : ℝ × ℝ), 
    -- AM is a median
    M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2 ∧
    -- A'B'C' is the reflection of ABC over AM
    ∃ (A' B' C' : ℝ × ℝ), 
      (A' = A) ∧
      (B'.1 - A.1 = A.1 - B.1) ∧ (B'.2 - A.2 = A.2 - B.2) ∧
      (C'.1 - A.1 = A.1 - C.1) ∧ (C'.2 - A.2 = A.2 - C.2) ∧
    -- Given lengths
    dist A E = 8 ∧
    dist E C = 16 ∧
    dist B D = 14

-- The theorem to prove
theorem triangle_side_length (A B C M : ℝ × ℝ) :
  Triangle A B C M → dist A B = 2 * Real.sqrt 94 := by
  sorry

-- Helper function for Euclidean distance
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l905_90567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_negative_exponent_l905_90551

theorem division_negative_exponent (x : ℝ) (h : x ≠ 0) : 1 / (x^2)⁻¹ = x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_negative_exponent_l905_90551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_equals_expected_l905_90595

-- Define the number of cubes and spheres
def num_cubes : ℕ := 3
def num_spheres : ℕ := 4

-- Define the side length of cubes and radius of spheres
def cube_side : ℝ := 3
def sphere_radius : ℝ := 2

-- Define the volume of a cube
def cube_volume (side : ℝ) : ℝ := side ^ 3

-- Define the volume of a sphere
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * (radius ^ 3)

-- Theorem statement
theorem total_volume_equals_expected : 
  (num_cubes : ℝ) * cube_volume cube_side + (num_spheres : ℝ) * sphere_volume sphere_radius = 81 + (128 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_equals_expected_l905_90595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l905_90561

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

theorem symmetry_center_of_tan :
  ∃ (k : ℤ), ∀ (x : ℝ), f (-Real.pi/8 + x) = -f (-Real.pi/8 - x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l905_90561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l905_90537

theorem log_inequality (a b c : ℝ) (m n r : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (hm : m = Real.log c / Real.log a)
  (hn : n = Real.log c / Real.log b)
  (hr : r = a^c) : 
  n < m ∧ m < r := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l905_90537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_and_line_l905_90556

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 7 = 0

-- Define a circle with center (a, b) and radius r
def circleWithCenter (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Define the locus of centers of circles tangent to both given circles
def locus (a b : ℝ) : Prop :=
  ∃ (r : ℝ), (∀ (x y : ℝ), circleWithCenter a b r x y → (circle1 x y ∨ circle2 x y))
           ∧ (∃ (x y : ℝ), circleWithCenter a b r x y ∧ circle1 x y)
           ∧ (∃ (x y : ℝ), circleWithCenter a b r x y ∧ circle2 x y)

-- Define predicates for hyperbola and line (these are placeholders and should be properly defined)
def IsHyperbola (s : Set (ℝ × ℝ)) : Prop := sorry
def IsLine (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem stating that the locus is a hyperbola and a line
theorem locus_is_hyperbola_and_line :
  ∃ (h : Set (ℝ × ℝ)) (l : Set (ℝ × ℝ)),
    IsHyperbola h ∧ IsLine l ∧ 
    (∀ a b, locus a b ↔ (a, b) ∈ h ∨ (a, b) ∈ l) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_and_line_l905_90556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l905_90577

/-- Calculates the sale price after applying successive discounts -/
noncomputable def salePriceAfterDiscounts (originalPrice : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount / 100)) originalPrice

/-- Theorem: The sale price of a saree listed for Rs. 600 after successive discounts of 22%, 35%, 15%, and 7% is approximately Rs. 240.47 -/
theorem saree_sale_price :
  let originalPrice : ℝ := 600
  let discounts : List ℝ := [22, 35, 15, 7]
  abs (salePriceAfterDiscounts originalPrice discounts - 240.47) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_saree_sale_price_l905_90577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameter_b_range_l905_90562

/-- The set of real numbers b for which, given any a in [-2, 1], there exists at least one x 
    such that a^2 + b^2 - sin^2(2x) - 2(a + b)cos(2x) - 2 ≤ 0 is equal to [-1.5, √3 - 1] -/
theorem parameter_b_range :
  ∀ b : ℝ, (∀ a ∈ Set.Icc (-2 : ℝ) 1, ∃ x : ℝ, 
    a^2 + b^2 - Real.sin (2*x)^2 - 2*(a + b)*Real.cos (2*x) - 2 ≤ 0) ↔ 
  b ∈ Set.Icc (-1.5 : ℝ) (Real.sqrt 3 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameter_b_range_l905_90562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_discriminant_root_cycle_l905_90502

theorem no_discriminant_root_cycle (d₁ d₂ d₃ : ℝ) : 
  0 ≤ d₁ → 0 ≤ d₂ → 0 ≤ d₃ →
  d₁ < d₂ → d₂ < d₃ →
  Real.sqrt d₂ / 2 = d₃ - d₁ →
  Real.sqrt d₃ / 2 = d₂ - d₁ →
  False :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_discriminant_root_cycle_l905_90502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_totals_l905_90543

/-- Represents the assignment of values to vertices of a cube -/
def VertexAssignment := Fin 8 → Int

/-- Calculates the product of four integers -/
def product_four (a b c d : Int) : Int := a * b * c * d

/-- Represents a face of a cube as a 4-tuple of vertex indices -/
def Face := (Fin 4 → Fin 8)

/-- Calculates the value of a face given a vertex assignment -/
def face_value (va : VertexAssignment) (f : Face) : Int :=
  product_four (va (f 0)) (va (f 1)) (va (f 2)) (va (f 3))

/-- Calculates the sum of all vertex and face values -/
def total_sum (va : VertexAssignment) (faces : Fin 6 → Face) : Int :=
  (Finset.sum (Finset.univ : Finset (Fin 8)) fun i => va i) + 
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i => face_value va (faces i))

/-- The main theorem stating the possible totals -/
theorem cube_totals (va : VertexAssignment) (faces : Fin 6 → Face) 
  (h1 : ∀ i, va i = 1 ∨ va i = -1) :
  ∃ n : Int, total_sum va faces = n ∧ n ∈ ({14, 6, 2, -2, -6, -10} : Finset Int) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_totals_l905_90543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l905_90535

/-- Calculates the speed in miles per hour given distance in feet and time in seconds -/
noncomputable def speed_mph (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_feet / 5280) / (time_seconds / 3600)

/-- The approximation of the object's speed -/
def approximate_speed : ℝ := 40.909

/-- Theorem stating that the calculated speed is close to the approximate speed -/
theorem object_speed_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |speed_mph 300 5 - approximate_speed| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l905_90535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_bound_l905_90565

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

-- State the theorem
theorem f_difference_bound (a m : ℝ) : 
  (a < -2) → 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → 
    |f a x₁ - f a x₂| < (m + Real.log 3) * a - 2 * Real.log 3) → 
  m ≤ -13/3 := by
  sorry

-- Note: Set.Icc 1 3 represents the closed interval [1, 3]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_bound_l905_90565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l905_90549

open Real

-- Define the interval (0, +∞)
def I : Set ℝ := {t : ℝ | 0 < t}

-- Define the functions x₁ and x₂
noncomputable def x₁ (t : ℝ) : ℝ := -1 / t^2
noncomputable def x₂ (t : ℝ) : ℝ := -t * log t

-- State the theorem
theorem solution_verification (t : ℝ) (h : t ∈ I) :
  (deriv x₁ t = 2 * t * (x₁ t)^2) ∧
  (deriv x₂ t = x₂ t / t - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verification_l905_90549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approximation_l905_90541

/-- Calculate the original price given the selling price and profit percentage -/
noncomputable def originalPrice (sellingPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  sellingPrice / (1 + profitPercentage)

/-- The total original cost of three items -/
noncomputable def totalOriginalCost : ℝ :=
  originalPrice 550 0.35 + originalPrice 1500 0.50 + originalPrice 1000 0.25

/-- Theorem stating that the total original cost is approximately $2207.41 -/
theorem total_cost_approximation :
  abs (totalOriginalCost - 2207.41) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approximation_l905_90541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_section_l905_90512

-- Define the geometric sequence condition
def is_geometric_sequence (a : ℝ) : Prop := a^2 = 9

-- Define the conic section
def conic_section (a x y : ℝ) : Prop := x^2 / a + y^2 / 2 = 1

-- Define the eccentricity of an ellipse
noncomputable def ellipse_eccentricity : ℝ := (3 : ℝ).sqrt / 3

-- Define the eccentricity of a hyperbola
noncomputable def hyperbola_eccentricity : ℝ := (10 : ℝ).sqrt / 2

-- Theorem statement
theorem eccentricity_of_conic_section (a : ℝ) :
  is_geometric_sequence a →
  ∃ (e : ℝ), (e = ellipse_eccentricity ∨ e = hyperbola_eccentricity) ∧
    ∀ (x y : ℝ), conic_section a x y → 
      (x^2 / (1 - e^2) + y^2 / (1 - e^2 * (a / 2)) = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_conic_section_l905_90512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l905_90550

/-- Given an ellipse ax^2 + by^2 = 1 intersecting the line x + y - 1 = 0,
    if the distance between intersection points is 2√2 and the slope of the line
    from the origin to the midpoint of the intersection points is √2/2,
    then the equation of the ellipse is x^2/3 + (√2 y^2)/3 = 1 -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (h_intersect : ∃ (A B : ℝ × ℝ), 
    a * A.1^2 + b * A.2^2 = 1 ∧ A.1 + A.2 = 1 ∧
    a * B.1^2 + b * B.2^2 = 1 ∧ B.1 + B.2 = 1)
  (h_distance : ∃ (A B : ℝ × ℝ), 
    a * A.1^2 + b * A.2^2 = 1 ∧ A.1 + A.2 = 1 ∧
    a * B.1^2 + b * B.2^2 = 1 ∧ B.1 + B.2 = 1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)
  (h_slope : ∃ (A B : ℝ × ℝ), 
    a * A.1^2 + b * A.2^2 = 1 ∧ A.1 + A.2 = 1 ∧
    a * B.1^2 + b * B.2^2 = 1 ∧ B.1 + B.2 = 1 ∧
    ((A.2 + B.2) / 2) / ((A.1 + B.1) / 2) = Real.sqrt 2 / 2) :
  a = 1/3 ∧ b = Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l905_90550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approximation_l905_90518

/-- The number of revolutions the wheel makes -/
def revolutions : ℝ := 19.017288444040037

/-- The total distance covered by the wheel in centimeters -/
def total_distance : ℝ := 1672

/-- The diameter of the wheel in centimeters -/
noncomputable def diameter : ℝ := total_distance / (Real.pi * revolutions)

theorem wheel_diameter_approximation :
  ∃ ε > 0, |diameter - 28| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approximation_l905_90518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l905_90572

def parabola_Pi1 (x y : ℝ) : Prop := sorry
def parabola_Pi2 (x y : ℝ) : Prop := sorry

theorem parabola_intersection :
  (∀ x y, parabola_Pi1 x y → y ≥ 0) →
  parabola_Pi1 10 0 →
  parabola_Pi1 13 0 →
  (∀ x y, parabola_Pi2 x y → y ≥ 0) →
  parabola_Pi2 13 0 →
  (∃ x₁ y₁, parabola_Pi1 x₁ y₁ ∧ 
    x₁ = (0 + x₂) / 2 ∧
    y₁ = (0 + y₂) / 2 ∧
    parabola_Pi2 x₂ y₂) →
  (∃ x, parabola_Pi2 x 0 ∧ x ≠ 13 ∧ x = 33) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l905_90572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_P_l905_90532

-- Define a polynomial type
def MyPolynomial (R : Type) [Ring R] := R → R

-- Define the degree of a polynomial
noncomputable def degree (p : MyPolynomial ℝ) : ℕ :=
  sorry

-- Define the polynomial P(x) = -x^3+2x^2+16
def P : MyPolynomial ℝ := λ x => -x^3 + 2*x^2 + 16

-- Theorem: The degree of P is 3
theorem degree_of_P : degree P = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_P_l905_90532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hyperbola_asymptotes_l905_90513

/-- The distance from a point to the asymptotes of a hyperbola -/
theorem distance_to_hyperbola_asymptotes (x₀ y₀ a b : ℝ) :
  let P : ℝ × ℝ := (x₀, y₀)
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x y : ℝ) ↦ b * x - a * y = 0
  let distance := fun (A B C : ℝ) ↦ |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)
  x₀ = 2 ∧ y₀ = 0 ∧ a = 3 ∧ b = 4 →
  distance b (-a) 0 = 8/5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hyperbola_asymptotes_l905_90513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_value_l905_90529

/-- A straight n-sided prism with 15 edges -/
structure StraightPrism where
  n : ℕ
  edges : ℕ
  faces : ℕ
  vertices : ℕ
  h_edges : edges = 15
  h_faces : faces = n + 2
  h_vertices : vertices = 2 * n

/-- The value of n - ab for a straight n-sided prism with 15 edges is -65 -/
theorem prism_value (p : StraightPrism) : (p.n : ℤ) - (p.faces * p.vertices : ℤ) = -65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_value_l905_90529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l905_90586

open Real

noncomputable def f (x : ℝ) := sin ((1/5) * x + (13*π)/6)

noncomputable def g (x : ℝ) := f (x - (10*π)/3)

theorem g_increasing_on_interval : 
  StrictMonoOn g (Set.Icc π (2*π)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l905_90586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dimension_theorem_l905_90519

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Represents a hexagon -/
structure Hexagon where
  y : ℝ

noncomputable def rectangle_to_square (rect : Rectangle) (hex1 hex2 : Hexagon) : Square :=
  { side := Real.sqrt (rect.width * rect.length) }

theorem hexagon_dimension_theorem (rect : Rectangle) (hex1 hex2 : Hexagon) 
    (h1 : rect.width = 8)
    (h2 : rect.length = 18)
    (h3 : hex1 = hex2)
    (h4 : (rectangle_to_square rect hex1 hex2).side = Real.sqrt (rect.width * rect.length))
    (h5 : hex1.y * 2 = (rectangle_to_square rect hex1 hex2).side) :
  hex1.y = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_dimension_theorem_l905_90519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_holder_inequality_l905_90557

open MeasureTheory

theorem generalized_holder_inequality 
  {X : Type*} [MeasurableSpace X] {μ : Measure X} 
  {f g h : X → ℝ} (hf : Measurable f) (hg : Measurable g) (hh : Measurable h)
  {p q r : ℝ} (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) 
  (hpqr : p⁻¹ + q⁻¹ + r⁻¹ = 1) :
  ∫ x, |f x * g x * h x| ∂μ ≤ 
    (∫ x, |f x|^p ∂μ)^(1/p) * (∫ x, |g x|^q ∂μ)^(1/q) * (∫ x, |h x|^r ∂μ)^(1/r) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generalized_holder_inequality_l905_90557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_cos_bounds_l905_90515

theorem cos_sin_cos_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_cos_bounds_l905_90515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_order_l905_90531

-- Define the constants
noncomputable def a : ℝ := Real.log (2/3)
noncomputable def b : ℝ := -(Real.log (3/2) / Real.log 3)
noncomputable def c : ℝ := (2/3) ^ (1/3 : ℝ)

-- State the theorem
theorem magnitude_order : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_order_l905_90531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_cube_root_seven_l905_90581

theorem cube_root_sum_equals_cube_root_seven :
  (7 + 2 * Real.sqrt 21) ^ (1/3 : ℝ) + (7 - 2 * Real.sqrt 21) ^ (1/3 : ℝ) = 7 ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_cube_root_seven_l905_90581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_divides_power_of_two_minus_one_l905_90552

theorem odd_number_divides_power_of_two_minus_one (a : ℕ) (h : Odd a) :
  ∃ b : ℕ, a ∣ (2^b - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_number_divides_power_of_two_minus_one_l905_90552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complexPercentageExpression_value_l905_90568

/-- Calculates the result of a complex percentage expression -/
noncomputable def complexPercentageExpression : ℝ :=
  let a := (63 + 4/5) / 100 * 3150
  let b := (48 + 3/7) / 100 * 2800
  let c := (17 + 2/3) / 100 * 945
  a - b + c / 2

/-- The complex percentage expression equals 737.175 -/
theorem complexPercentageExpression_value : 
  complexPercentageExpression = 737.175 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complexPercentageExpression_value_l905_90568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_cost_proof_l905_90588

/-- The cost of a water bottle in dollars -/
noncomputable def water_bottle_cost : ℚ := 2

/-- The amount Jack started with in dollars -/
def initial_amount : ℚ := 100

/-- The number of water bottles Jack initially bought -/
def initial_bottles : ℕ := 4

/-- The cost of cheese per pound in dollars -/
def cheese_cost_per_pound : ℚ := 10

/-- The amount of cheese Jack bought in pounds -/
def cheese_amount : ℚ := 1/2

/-- The amount Jack has remaining in dollars -/
def remaining_amount : ℚ := 71

theorem water_bottle_cost_proof :
  water_bottle_cost * (initial_bottles + 2 * initial_bottles) + cheese_cost_per_pound * cheese_amount
  = initial_amount - remaining_amount := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_bottle_cost_proof_l905_90588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l905_90559

-- Define the power function f
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log (1/3) / Real.log 9)

-- State the theorem
theorem power_function_value : f 25 = 1/5 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l905_90559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_properties_l905_90570

def z₁ : ℂ := Complex.mk 1 2
def z₂ : ℂ := Complex.mk 3 (-4)

theorem complex_product_properties : 
  (z₁ * z₂).im = 2 ∧ 
  z₁.re > 0 ∧ z₁.im > 0 ∧ (z₁ * z₂).re > 0 ∧ (z₁ * z₂).im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_properties_l905_90570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_is_2_pow_n_l905_90591

/-- Represents a coloring of a 2^n × 2^n table -/
def TableColoring (n : ℕ) := Fin (2^n) → Fin (2^n) → ℕ

/-- Predicate that checks if a coloring satisfies the given condition -/
def ValidColoring (n : ℕ) (coloring : TableColoring n) : Prop :=
  ∀ i j : Fin (2^n), coloring i j = coloring j ((i + j) % (2^n))

/-- The maximum number of colors in a valid coloring -/
def MaxColors (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the maximum number of colors in a valid coloring is 2^n -/
theorem max_colors_is_2_pow_n (n : ℕ) :
  (∃ (coloring : TableColoring n), ValidColoring n coloring ∧
    (∀ (other_coloring : TableColoring n),
      ValidColoring n other_coloring →
      (Finset.card (Finset.image (λ (pair : Fin (2^n) × Fin (2^n)) ↦ coloring pair.1 pair.2) (Finset.univ.product Finset.univ)) ≥
       Finset.card (Finset.image (λ (pair : Fin (2^n) × Fin (2^n)) ↦ other_coloring pair.1 pair.2) (Finset.univ.product Finset.univ))))) ∧
  (∀ (coloring : TableColoring n),
    ValidColoring n coloring →
    Finset.card (Finset.image (λ (pair : Fin (2^n) × Fin (2^n)) ↦ coloring pair.1 pair.2) (Finset.univ.product Finset.univ)) ≤ MaxColors n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colors_is_2_pow_n_l905_90591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l905_90590

theorem sin_cos_sixth_power_min (x : ℝ) : (Real.sin x) ^ 6 + (Real.cos x) ^ 6 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sixth_power_min_l905_90590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_8_095_l905_90575

/-- Rounds a number to the nearest multiple of 0.01 -/
noncomputable def roundToTwoDecimalPlaces (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- Rounds a number to the nearest integer -/
noncomputable def roundToNearestWholeNumber (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem rounding_8_095 :
  let x : ℝ := 8.095
  roundToTwoDecimalPlaces x = 8.10 ∧ roundToNearestWholeNumber x = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_8_095_l905_90575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cleans_half_l905_90553

/-- The time (in hours) it takes John to clean the entire house alone. -/
noncomputable def john_time : ℝ := 6

/-- The time (in hours) it takes John and Nick together to clean the entire house. -/
noncomputable def john_nick_time : ℝ := 3.6

/-- The portion of the house John cleans in one-third of Nick's time. -/
noncomputable def john_portion : ℝ := 1/2

/-- Theorem stating that John cleans half the house in one-third of Nick's time. -/
theorem john_cleans_half :
  john_portion = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_cleans_half_l905_90553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l905_90554

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

theorem f_properties :
  ∀ k : ℤ,
  (∀ x ∈ Set.Icc (-Real.pi/6 + k*Real.pi) (Real.pi/3 + k*Real.pi), 
    StrictMono (f ∘ (fun y => y + (-Real.pi/6 + k*Real.pi))) ∧ Continuous f) ∧
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≤ (Real.sqrt 3 - 1) / 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x ≥ -3/2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x = (Real.sqrt 3 - 1) / 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/4) (Real.pi/4), f x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l905_90554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_root_difference_l905_90503

theorem cubic_equation_root_difference (opponent_choice : ℚ → ℚ) : 
  ∃ (a b c : ℚ), 
    let f := fun x : ℝ => x^3 + a*x^2 + b*x + c
    (∃ (x y : ℝ), f x = 0 ∧ f y = 0 ∧ |x - y| = 2014) ∧
    (a = opponent_choice b ∨ b = opponent_choice a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_root_difference_l905_90503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l905_90538

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 9 = 0
def circle2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 + 12*y + 36 = 0

-- Define the shortest distance between the circles
noncomputable def shortest_distance : ℝ := Real.sqrt 164 - 5

-- Theorem statement
theorem shortest_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), 
  circle1 x1 y1 → circle2 x2 y2 →
  ∃ (d : ℝ), d ≥ shortest_distance ∧ 
  (x2 - x1)^2 + (y2 - y1)^2 = d^2 :=
by
  sorry

#check shortest_distance_between_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_l905_90538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l905_90545

/-- Given a hyperbola and a parabola with specific properties, prove that the parameter p of the parabola equals 2 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) : 
  a > 0 → 
  b > 0 → 
  p > 0 → 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → ∃ k : ℝ, y = k*x ∧ (k = b/a ∨ k = -b/a)) →  -- Hyperbola equation and its asymptotes
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Parabola equation
  (a^2 + b^2)/(a^2) = 4 →  -- Eccentricity of hyperbola is 2
  (Real.sqrt 3 * p^2) / 2 = Real.sqrt 3 →  -- Area of triangle AOB is √3
  p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l905_90545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l905_90544

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_ratio
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8_2a3 : a 8 = 2 * a 3)
  (h_a8_2 : a 8 = 2) :
  S a 15 / S a 5 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l905_90544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_four_to_ninth_power_l905_90528

theorem sixth_root_of_four_to_ninth_power : (4 : ℝ) ^ ((1/6 : ℝ) * 9) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_four_to_ninth_power_l905_90528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l905_90579

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + 4*y^2 - 8*y + 9 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := 2 * Real.pi

/-- Theorem stating that the area of the ellipse defined by the given equation is 2π -/
theorem area_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, ellipse_equation x y ↔ (x + 3)^2 / (2*a)^2 + (y - 1)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l905_90579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_is_true_l905_90536

theorem proposition_p_and_q_is_true : (∃ x : ℝ, Real.sin x < 1) ∧ (∀ x : ℝ, Real.exp (|x|) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_and_q_is_true_l905_90536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_l905_90526

/-- A polyhedron formed by folding a specific figure -/
structure Polyhedron where
  /-- The number of isosceles right triangles in the figure -/
  num_isosceles_triangles : ℕ
  /-- The number of squares in the figure -/
  num_squares : ℕ
  /-- The side length of the squares -/
  square_side_length : ℝ
  /-- The side length of the equilateral triangle -/
  equilateral_side_length : ℝ
  /-- Condition: The number of isosceles right triangles is 3 -/
  h_num_isosceles : num_isosceles_triangles = 3
  /-- Condition: The number of squares is 3 -/
  h_num_squares : num_squares = 3
  /-- Condition: The square side length is 2 -/
  h_square_side : square_side_length = 2
  /-- Condition: The equilateral triangle side length is 2√2 -/
  h_equilateral_side : equilateral_side_length = 2 * Real.sqrt 2

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron) : ℝ := 22 * Real.sqrt 2 / 3

/-- Theorem stating that the volume of the polyhedron is 22√2/3 -/
theorem polyhedron_volume (p : Polyhedron) : volume p = 22 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_volume_l905_90526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohit_initial_south_distance_l905_90599

/-- Represents Rohit's position in a 2D coordinate system -/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents the distance and direction of Rohit's movements -/
inductive Movement
  | South : ℝ → Movement
  | East : ℝ → Movement
  | North : ℝ → Movement

/-- Calculates the new position after a movement -/
def move (pos : Position) (m : Movement) : Position :=
  match m with
  | Movement.South d => ⟨pos.x, pos.y - d⟩
  | Movement.East d => ⟨pos.x + d, pos.y⟩
  | Movement.North d => ⟨pos.x, pos.y + d⟩

/-- Calculates the final position after a sequence of movements -/
def finalPosition (movements : List Movement) : Position :=
  movements.foldl move ⟨0, 0⟩

/-- Calculates the distance between two positions -/
noncomputable def distance (p1 p2 : Position) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem rohit_initial_south_distance (d : ℝ) :
  let movements := [Movement.South d, Movement.East 20, Movement.North d, Movement.East 15]
  let finalPos := finalPosition movements
  distance ⟨0, 0⟩ finalPos = 35 → d = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohit_initial_south_distance_l905_90599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l905_90592

noncomputable def f (x : ℝ) := (x^2 - 2*x + 2) / (2*x - 2)

theorem f_max_value :
  ∀ x ∈ Set.Ioo (-4 : ℝ) 1, f x ≤ -1 ∧ ∃ y ∈ Set.Ioo (-4 : ℝ) 1, f y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l905_90592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_fraction_is_three_eighths_l905_90514

/-- Represents the tiling of a plane with squares and hexagons -/
structure PlaneTiling where
  b : ℝ  -- Side length of smaller squares
  large_square_area : ℝ := 16 * b^2
  small_square_area : ℝ := b^2
  hexagon_area : ℝ := 3 * b^2
  hexagons_per_large_square : ℕ := 2

/-- The fraction of the plane enclosed by hexagons -/
noncomputable def hexagon_fraction (t : PlaneTiling) : ℝ :=
  (t.hexagons_per_large_square : ℝ) * t.hexagon_area / t.large_square_area

/-- Theorem stating that the fraction of the plane enclosed by hexagons is 3/8 -/
theorem hexagon_fraction_is_three_eighths (t : PlaneTiling) :
  hexagon_fraction t = 3/8 := by
  sorry

#check hexagon_fraction_is_three_eighths

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_fraction_is_three_eighths_l905_90514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l905_90587

/-- Represents a truncated cone with given proportions and slant height. -/
structure TruncatedCone where
  r : ℝ  -- radius of top base
  R : ℝ  -- radius of bottom base
  h : ℝ  -- height
  l : ℝ  -- slant height
  prop_r_R : R = 4 * r  -- proportion of radii
  prop_h : h = 4 * r    -- proportion of height to smaller radius
  slant : l = 10        -- slant height in cm
  pythagoras : l^2 = h^2 + (R - r)^2  -- Pythagorean theorem for slant height

/-- Volume of a truncated cone. -/
noncomputable def volume (tc : TruncatedCone) : ℝ :=
  (1/3) * Real.pi * tc.h * (tc.r^2 + tc.R^2 + tc.r * tc.R)

/-- Theorem stating the volume of the specific truncated cone. -/
theorem truncated_cone_volume (tc : TruncatedCone) : volume tc = 224 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_l905_90587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_S_in_open_interval_l905_90548

/-- The sum of the areas of the cylinder's two bases and lateral surface area -/
noncomputable def S (r : ℝ) : ℝ := 2 * Real.pi * (10 * r - r^2)

/-- Theorem stating that there is no maximum value of S(r) for r in (0, 5) -/
theorem no_max_S_in_open_interval : 
  ¬ ∃ (r : ℝ), 0 < r ∧ r < 5 ∧ ∀ (x : ℝ), 0 < x → x < 5 → S x ≤ S r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_max_S_in_open_interval_l905_90548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l905_90540

noncomputable section

-- Define the fixed point F
def F : ℝ × ℝ := (0, 1)

-- Define the fixed line l
def l : Set (ℝ × ℝ) := {p | p.2 = -1}

-- Define the trajectory C
def C : Set (ℝ × ℝ) := {p | p.1^2 = 4 * p.2}

-- Define a predicate for a circle passing through F and tangent to l
def is_valid_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - F.1)^2 + (center.2 - F.2)^2 = radius^2 ∧
  |center.2 + 1| = radius

-- Define a function to calculate the area of a circle
def circle_area (radius : ℝ) : ℝ := Real.pi * radius^2

-- Main theorem
theorem trajectory_and_min_area :
  (∀ c r, is_valid_circle c r → c ∈ C) ∧
  (∃ min_area : ℝ, 
    min_area = 4 * Real.pi ∧
    ∀ A B P : ℝ × ℝ,
      A ∈ C → B ∈ C → P ∈ C →
      (∃ k : ℝ, A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1) →
      (∃ t₁ t₂ : ℝ, 
        (P.1 - A.1) * t₁ = A.2 - P.2 ∧
        (P.1 - B.1) * t₂ = B.2 - P.2) →
      circle_area (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2) ≥ min_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l905_90540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_radius_is_sqrt_59_l905_90517

/-- A sphere intersecting two planes --/
structure IntersectingSphere where
  xy_center : ℝ × ℝ × ℝ
  xy_radius : ℝ
  yz_center : ℝ × ℝ × ℝ

/-- The radius of the circle formed by the intersection of the sphere and the yz-plane --/
noncomputable def yz_radius (s : IntersectingSphere) : ℝ :=
  Real.sqrt 59

/-- Theorem: The radius of the circle formed by the intersection of the sphere and the yz-plane is √59 --/
theorem yz_radius_is_sqrt_59 (s : IntersectingSphere) 
    (h1 : s.xy_center = (3, 5, 0)) 
    (h2 : s.xy_radius = 2) 
    (h3 : s.yz_center = (0, 5, -8)) : 
  yz_radius s = Real.sqrt 59 := by
  rfl

#check yz_radius_is_sqrt_59

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_radius_is_sqrt_59_l905_90517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l905_90573

theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ P : ℝ × ℝ, P = (x, 3) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) →
  Real.cos θ = -4/5 →
  x = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_l905_90573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_equation_solutions_l905_90596

theorem integer_equation_solutions : 
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ 1 + ⌊(98 * n : ℚ) / 99⌋ = ⌈(97 * n : ℚ) / 98⌉) ∧ Finset.card S = 9602 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_equation_solutions_l905_90596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l905_90542

-- Define the triangle ABC
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C : V)

-- Define the properties of the triangle
def is_triangle_ABC {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : Triangle V) (M : V) : Prop :=
  ∃ (angleBAC angleACB : ℝ),
    angleBAC = Real.pi/4 ∧ 
    angleACB = Real.pi/6 ∧ 
    M = (1/2 : ℝ) • (t.B + t.C)

-- State the theorem
theorem triangle_properties {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (t : Triangle V) (M : V) (h : is_triangle_ABC t M) :
  ∃ (angleAMB : ℝ),
    angleAMB = Real.pi/4 ∧ 
    ‖t.B - t.C‖ * ‖t.A - t.C‖ = 2 * ‖t.A - M‖ * ‖t.A - t.B‖ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l905_90542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersects_first_and_second_quadrants_l905_90504

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the circle -/
def circleEquation (p : Point) : Prop :=
  (p.x - 1)^2 + (p.y - 3)^2 = 4

/-- Definition of the first quadrant -/
def first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that the circle intersects both the first and second quadrants -/
theorem circle_intersects_first_and_second_quadrants :
  (∃ p : Point, circleEquation p ∧ first_quadrant p) ∧
  (∃ p : Point, circleEquation p ∧ second_quadrant p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersects_first_and_second_quadrants_l905_90504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_car_packages_l905_90509

theorem tom_car_packages : ℕ := 
  let cars_per_package : ℕ := 5
  let fraction_given : ℚ := 1 / 5
  let num_nephews : ℕ := 2
  let cars_left : ℕ := 30
  let num_packages : ℕ := 10

  have h1 : (cars_per_package * num_packages) - 
             (↑num_nephews * fraction_given * ↑(cars_per_package * num_packages)) = cars_left := by
    sorry
  have h2 : num_packages > 0 := by
    sorry

  num_packages

#check tom_car_packages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_car_packages_l905_90509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l905_90546

open Real

-- Define the function f on the open interval (-1, 1)
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Define the domain of f
def D : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem f_properties : 
  (∀ x, x ∈ D → f (-x) = -f x) ∧  -- f is odd
  (f (-1/2) = -2/5) ∧         -- f(-1/2) = -2/5
  (∀ x y, x ∈ D → y ∈ D → x < y → f x < f y) -- f is increasing on (-1, 1)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l905_90546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flooring_cost_l905_90520

/-- Represents the dimensions of a rectangular area -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Represents the cost per square meter of a flooring material -/
abbrev MaterialCost := ℝ

/-- Calculates the cost of flooring a rectangular area with a given material -/
def flooringCost (r : Rectangle) (m : MaterialCost) : ℝ :=
  area r * m

theorem total_flooring_cost (largeRect : Rectangle) (smallRect : Rectangle) 
    (materialA : MaterialCost) (materialB : MaterialCost) :
  largeRect.length = 5.5 ∧ 
  largeRect.width = 3.75 ∧
  smallRect.length = 2.5 ∧ 
  smallRect.width = 1.5 ∧
  materialA = 600 ∧
  materialB = 450 →
  flooringCost largeRect materialA + flooringCost smallRect materialB = 14062.5 := by
  sorry

#eval flooringCost ⟨5.5, 3.75⟩ 600 + flooringCost ⟨2.5, 1.5⟩ 450

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_flooring_cost_l905_90520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l905_90555

/-- Represents a player in the game -/
inductive Player
| Petya
| Vasya

/-- Represents a single cell in the strip -/
structure Cell where
  digit : Nat
  filled : Bool

/-- Represents the game state -/
structure GameState where
  strip : Array Cell
  currentPlayer : Player

/-- A strategy for Petya is a function that takes the current game state and returns the next move -/
def PetyaStrategy := GameState → Nat

/-- A move by Vasya is represented by a natural number (the digit) and its position -/
structure VasyaMove where
  digit : Nat
  position : Nat

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Function to compute the resulting number after all moves -/
def resultingNumber (strategy : PetyaStrategy) (vasyaMove : VasyaMove) : Nat :=
  sorry -- Implementation details omitted for brevity

/-- The main theorem stating that Petya has a winning strategy -/
theorem petya_has_winning_strategy :
  ∃ (strategy : PetyaStrategy),
    ∀ (vasyaMove : VasyaMove),
      isPerfectSquare (resultingNumber strategy vasyaMove) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_has_winning_strategy_l905_90555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l905_90525

-- Define the triangle OCD
def O : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (7, 0)

-- Define the angle measures
def angle_CDO : ℝ := 90
def angle_COD : ℝ := 45

-- Define the rotation angle
def rotation_angle : ℝ := 120

-- Function to rotate a point counterclockwise about the origin
noncomputable def rotate (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  let x := p.1
  let y := p.2
  let θ := angle * Real.pi / 180
  (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ)

-- Theorem statement
theorem triangle_rotation :
  ∃ C : ℝ × ℝ,
    C.1 > 0 ∧ C.2 > 0 ∧  -- C is in the first quadrant
    (C.2 - D.2) / (C.1 - D.1) = 1 ∧  -- Slope of CD is 1 (45° angle)
    (C.1 - O.1) * (D.1 - O.1) + (C.2 - O.2) * (D.2 - O.2) = 0 ∧  -- CD ⊥ OD
    rotate C rotation_angle = (-7 * (1 + Real.sqrt 3) / 2, 7 * (Real.sqrt 3 - 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rotation_l905_90525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_minimum_value_l905_90560

theorem complex_minimum_value (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (1 + z + 3 * z^2 + z^3 + z^4) ≥ 
  Complex.abs (1 + (-1/4 + (Real.sqrt 15)/4 * I) + 
               3 * (-1/4 + (Real.sqrt 15)/4 * I)^2 + 
               (-1/4 + (Real.sqrt 15)/4 * I)^3 + 
               (-1/4 + (Real.sqrt 15)/4 * I)^4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_minimum_value_l905_90560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l905_90580

noncomputable def f (x : ℝ) := 3 * Real.cos (2 * x - Real.pi / 3)

theorem f_symmetry (x : ℝ) : f (2 * Real.pi / 3 + x) = f (2 * Real.pi / 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l905_90580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_distinct_integers_satisfying_equation_l905_90508

theorem no_distinct_integers_satisfying_equation :
  ∀ x y : ℕ, x > 0 → y > 0 → x ≠ y → x^2007 + Nat.factorial y ≠ y^2007 + Nat.factorial x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_distinct_integers_satisfying_equation_l905_90508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_weight_l905_90511

/-- Calculates the weight of raisins after transformation from grapes -/
theorem raisin_weight (grape_weight : ℝ) (grape_water_percent : ℝ) (raisin_water_percent : ℝ) 
  (h1 : grape_weight = 50)
  (h2 : grape_water_percent = 0.92)
  (h3 : raisin_water_percent = 0.20) : 
  (grape_weight * (1 - grape_water_percent)) / (1 - raisin_water_percent) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_raisin_weight_l905_90511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_in_tank_l905_90566

/-- Calculates the unoccupied volume in a rectangular prism tank -/
theorem unoccupied_volume_in_tank (tank_length tank_width tank_height : ℚ)
  (water_fill_ratio : ℚ) (num_ice_cubes : ℕ) (ice_cube_side : ℚ) :
  tank_length = 8 →
  tank_width = 10 →
  tank_height = 12 →
  water_fill_ratio = 1/3 →
  num_ice_cubes = 8 →
  ice_cube_side = 2 →
  let tank_volume := tank_length * tank_width * tank_height
  let water_volume := water_fill_ratio * tank_volume
  let ice_cube_volume := ice_cube_side ^ (3 : ℕ)
  let total_ice_volume := (num_ice_cubes : ℚ) * ice_cube_volume
  let occupied_volume := water_volume + total_ice_volume
  tank_volume - occupied_volume = 576 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_in_tank_l905_90566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_is_one_l905_90507

/-- Represents the rate of rainfall in inches per hour during a specific time period -/
structure RainfallRate where
  value : ℝ

/-- Represents the height of collected rainwater in inches -/
structure RainwaterHeight where
  value : ℝ

/-- The problem setup -/
def rainfall_problem (initial_rainfall : RainwaterHeight) 
                     (unknown_rate : RainfallRate) 
                     (final_rate : RainfallRate) 
                     (tank_height : RainwaterHeight) : Prop :=
  let initial_duration : ℝ := 1
  let unknown_duration : ℝ := 4
  let final_duration : ℝ := 4
  initial_rainfall.value + 
  unknown_rate.value * unknown_duration + 
  final_rate.value * final_duration = 
  tank_height.value

/-- The theorem to prove -/
theorem rainfall_rate_is_one :
  ∀ (initial_rainfall : RainwaterHeight) 
    (unknown_rate : RainfallRate) 
    (final_rate : RainfallRate) 
    (tank_height : RainwaterHeight),
  initial_rainfall.value = 2 →
  final_rate.value = 3 →
  tank_height.value = 18 →
  rainfall_problem initial_rainfall unknown_rate final_rate tank_height →
  unknown_rate.value = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_is_one_l905_90507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l905_90506

theorem inequality_range (x k : ℝ) : 
  (∀ k ∈ Set.Ioi (-4 : ℝ), (1 + k) * x ≤ k^2 + k + 4) → 
  x ∈ Set.Icc (-5 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_range_l905_90506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l905_90594

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - a*x + 3*a)

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x ≥ 2, ∀ Δx > 0, f a (x + Δx) > f a x) ↔ a ∈ Set.Icc (-4) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l905_90594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_number_is_9357_l905_90534

-- Define the set of digits
def digits : Finset Nat := {3, 5, 7, 9}

-- Define a function to generate all 4-digit numbers using the given digits
def generate_numbers (digits : Finset Nat) : List Nat :=
  sorry

-- Define a function to sort the generated numbers
def sort_numbers (numbers : List Nat) : List Nat :=
  sorry

-- Theorem statement
theorem nineteenth_number_is_9357 (numbers : List Nat) :
  numbers = sort_numbers (generate_numbers digits) →
  numbers.get? 18 = some 9357 := by
  sorry

#check nineteenth_number_is_9357

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_number_is_9357_l905_90534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_closed_existence_of_real_not_in_union_l905_90524

def ClosedSet {α : Type*} [Add α] [Sub α] (A : Set α) : Prop :=
  ∀ a b : α, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def MultiplesOfThree : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multiples_of_three_closed : ClosedSet MultiplesOfThree := by
  sorry

theorem existence_of_real_not_in_union :
  ∃ A₁ A₂ : Set ℝ, ClosedSet A₁ ∧ ClosedSet A₂ ∧ ∃ c : ℝ, c ∉ (A₁ ∪ A₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_three_closed_existence_of_real_not_in_union_l905_90524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_m_pow_n_l905_90523

/-- If the sum of two monomials is still a monomial, then m^n = 16 -/
theorem monomial_sum_implies_m_pow_n (m n : ℕ) : 
  (∃ (a b c : ℝ) (x y : ℝ) (k l : ℕ), a * x^(m+1) * y^3 + b * x^3 * y^(n-1) = c * x^k * y^l 
    ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  m^n = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_implies_m_pow_n_l905_90523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_not_in_throw_l905_90547

def cube_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_valid_throw (throw : Finset ℕ) : Prop :=
  throw.card = 5 ∧ throw ⊆ cube_faces

noncomputable def mean (s : Finset ℕ) : ℚ :=
  (s.sum (fun x => (x : ℚ))) / s.card

noncomputable def variance (s : Finset ℕ) : ℚ :=
  (s.sum (fun x => ((x : ℚ) - mean s) ^ 2)) / s.card

theorem six_not_in_throw (throw : Finset ℕ) 
  (h_valid : is_valid_throw throw) 
  (h_mean : mean throw = 3) 
  (h_var : variance throw = 2) : 
  6 ∉ throw := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_not_in_throw_l905_90547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_prime_b_prime_distance_l905_90558

noncomputable def A : ℝ × ℝ := (0, 4)
noncomputable def B : ℝ × ℝ := (0, 7)
noncomputable def C : ℝ × ℝ := (3, 6)

def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

def on_line (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem a_prime_b_prime_distance :
  ∃ (A' B' : ℝ × ℝ),
    line_y_eq_x A' ∧
    line_y_eq_x B' ∧
    on_line A C A' ∧
    on_line B C B' ∧
    distance A' B' = 6.75 * Real.sqrt 2 := by
  sorry

#check a_prime_b_prime_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_prime_b_prime_distance_l905_90558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l905_90500

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + x + 3

-- Define the set of x that satisfies f(x) ≤ 5
def S : Set ℝ := {x | f x ≤ 5}

-- State the theorem
theorem f_range : S = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l905_90500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l905_90585

theorem expression_value :
  (Real.tan (12 * π / 180) - Real.sqrt 3) / ((4 * (Real.cos (12 * π / 180))^2 - 2) * Real.sin (12 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l905_90585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_is_52_l905_90527

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : ℕ
  halfway_lead : ℕ
  chelsea_min_score : ℕ
  possible_scores : List ℕ

/-- The specific archery competition in the problem -/
def competition : ArcheryCompetition :=
  { total_shots := 120
  , halfway_lead := 60
  , chelsea_min_score := 3
  , possible_scores := [10, 7, 3, 0]
  }

/-- Calculate the minimum number of 10-point shots Chelsea needs to secure victory -/
def min_bullseyes_to_win (comp : ArcheryCompetition) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of 10-point shots Chelsea needs is 52 -/
theorem min_bullseyes_is_52 :
  min_bullseyes_to_win competition = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_is_52_l905_90527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l905_90530

theorem product_of_four_integers (p q r s : ℕ+) 
  (eq1 : p * q + p + q = 1074)
  (eq2 : q * r + q + r = 506)
  (eq3 : r * s + r + s = 208)
  (prod : p * q * r * s = Nat.factorial 12) :
  p - s = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_four_integers_l905_90530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_is_191_l905_90533

def mySequence : List ℕ := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551, 671]

theorem sixth_number_is_191 : mySequence[5] = 191 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_number_is_191_l905_90533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l905_90578

theorem count_positive_integers_satisfying_inequality :
  (Finset.filter (fun n : ℕ => n > 0 ∧ (n + 6) * (n - 5) * (n - 10) < 0) (Finset.range 11)).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l905_90578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l905_90583

/-- Represents the compound interest scenario -/
structure CompoundInterestScenario where
  principal : ℝ
  rate : ℝ
  time : ℝ
  compoundFrequency : ℝ

/-- Calculates the compound interest -/
noncomputable def compoundInterest (scenario : CompoundInterestScenario) : ℝ :=
  scenario.principal * (1 + scenario.rate / scenario.compoundFrequency) ^ (scenario.compoundFrequency * scenario.time) - scenario.principal

/-- Calculates the total amount after compound interest -/
noncomputable def totalAmount (scenario : CompoundInterestScenario) : ℝ :=
  scenario.principal + compoundInterest scenario

/-- Theorem: If the compound interest earned is 370.80 with a 6% annual rate for 2 years,
    then the total amount received is 3370.80 -/
theorem compound_interest_problem (P : ℝ) (h : P > 0) :
  let scenario := CompoundInterestScenario.mk P 0.06 2 1
  compoundInterest scenario = 370.80 → totalAmount scenario = 3370.80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l905_90583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_12m_squared_l905_90505

theorem divisors_of_12m_squared (m : ℕ) (h_odd : Odd m) (h_divisors : (Nat.divisors m).card = 13) :
  (Nat.divisors (12 * m^2)).card = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_12m_squared_l905_90505
