import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_l835_83540

theorem complex_purely_imaginary (m : ℝ) : 
  (Complex.I * (m + 1) = Complex.mk (m^2 - 1) (m + 1)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_purely_imaginary_l835_83540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_flippant_numbers_l835_83594

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := 
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

/-- Checks if a number is flippant -/
def isFlippant (n : ℕ) : Bool :=
  10 ≤ n && n < 1000 &&
  n % 10 ≠ 0 &&
  n % 7 = 0 &&
  (reverseDigits n) % 7 = 0

/-- The count of flippant numbers between 10 and 1000 -/
def flippantCount : ℕ := (List.range 991).map (·+10) |>.filter isFlippant |>.length

theorem count_flippant_numbers : flippantCount = 17 := by
  sorry

#eval flippantCount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_flippant_numbers_l835_83594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_ratio_l835_83508

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the ratio of line segments -/
noncomputable def ratio (P Q R : Point3D) : ℝ := 
  sorry

/-- Checks if a point is inside a tetrahedron -/
def O_inside_tetrahedron (ABCD : Tetrahedron) (O : Point3D) : Prop := 
  sorry

/-- Checks if a point is on a face of a tetrahedron -/
def on_face (A B C P : Point3D) : Prop := 
  sorry

/-- Checks if three points are collinear -/
def collinear (P Q R : Point3D) : Prop := 
  sorry

/-- Theorem: The common ratio in a tetrahedron with specific intersections is 3 -/
theorem tetrahedron_intersection_ratio 
  (ABCD : Tetrahedron) 
  (O : Point3D) 
  (A₁ B₁ C₁ D₁ : Point3D) 
  (h_inside : O_inside_tetrahedron ABCD O)
  (h_A₁ : on_face ABCD.B ABCD.C ABCD.D A₁)
  (h_B₁ : on_face ABCD.A ABCD.C ABCD.D B₁)
  (h_C₁ : on_face ABCD.A ABCD.B ABCD.D C₁)
  (h_D₁ : on_face ABCD.A ABCD.B ABCD.C D₁)
  (h_AO : collinear ABCD.A O A₁)
  (h_BO : collinear ABCD.B O B₁)
  (h_CO : collinear ABCD.C O C₁)
  (h_DO : collinear ABCD.D O D₁)
  (h_equal_ratios : 
    ratio ABCD.A O A₁ = ratio ABCD.B O B₁ ∧ 
    ratio ABCD.B O B₁ = ratio ABCD.C O C₁ ∧ 
    ratio ABCD.C O C₁ = ratio ABCD.D O D₁) :
  ratio ABCD.A O A₁ = 3 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_intersection_ratio_l835_83508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l835_83545

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def areaOfTriangle (t : Triangle) : ℝ := 
  (1/2) * t.a * t.c * Real.sin t.B

theorem triangle_area_is_two (t : Triangle) 
  (h1 : t.c^2 * Real.sin t.A = 5 * Real.sin t.C)
  (h2 : (t.a + t.c)^2 = 16 + t.b^2) :
  areaOfTriangle t = 2 := by
  sorry

#check triangle_area_is_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_two_l835_83545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_optimal_thickness_l835_83512

/-- Annual energy consumption cost function -/
noncomputable def C (k : ℝ) (x : ℝ) : ℝ := k / (3 * x + 5)

/-- Total cost function over 20 years -/
noncomputable def f (x : ℝ) : ℝ := 20 * C 40 x + 6 * x

/-- Theorem stating the optimal thickness and minimum cost -/
theorem insulation_optimal_thickness :
  ∃ (x_opt : ℝ), x_opt = 5 ∧ 
  (∀ x : ℝ, 0 ≤ x → x ≤ 10 → f x ≥ f x_opt) ∧
  f x_opt = 70 := by
  sorry

/-- Lemma proving the value of k -/
lemma k_value : C 40 0 = 8 := by
  sorry

/-- Lemma proving the derivative of f -/
lemma f_derivative (x : ℝ) : 
  deriv f x = 6 - 2400 / (3 * x + 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_optimal_thickness_l835_83512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_coordinate_l835_83556

theorem circle_point_x_coordinate : 
  ∀ (x : ℝ),
  let center : ℝ × ℝ := ((27 + (-3)) / 2, 0)
  let radius : ℝ := (27 - (-3)) / 2
  let on_circle := (x - center.1)^2 + (10 - center.2)^2 = radius^2
  on_circle → (x = 12 + 5 * Real.sqrt 5 ∨ x = 12 - 5 * Real.sqrt 5) :=
by
  intro x
  intro h
  sorry

#check circle_point_x_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_coordinate_l835_83556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_is_one_l835_83509

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ :=
  5 * (x - x^4) - 4 * (2*x^2 - x^4 + x^6) + 3 * (3*x^2 - x^10)

-- Theorem statement
theorem coefficient_of_x_squared_is_one :
  ∃ (a b c : ℝ), ∀ x, polynomial x = a * x^2 + b * x + c + x^3 * (polynomial x - (a * x^2 + b * x + c)) ∧ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_is_one_l835_83509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_M_and_M_l835_83546

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop := x + 4*y + 3*z + 5 = 0

/-- The point M -/
def M : ℝ × ℝ × ℝ := (1, 1, 1)

/-- The point M' -/
def M' : ℝ × ℝ × ℝ := (0, -3, -2)

/-- Definition of symmetry with respect to a plane -/
def symmetric_wrt_plane (A B : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (C : ℝ × ℝ × ℝ), 
    plane C.1 C.2.1 C.2.2 ∧ 
    (A.1 + B.1) / 2 = C.1 ∧ 
    (A.2.1 + B.2.1) / 2 = C.2.1 ∧ 
    (A.2.2 + B.2.2) / 2 = C.2.2

theorem symmetry_of_M_and_M' : 
  symmetric_wrt_plane M M' plane_equation := by
  sorry

#check symmetry_of_M_and_M'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_M_and_M_l835_83546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l835_83576

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- A point in the 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The line l₁: y = 2x -/
def l₁ : Line := { slope := 2, intercept := 0 }

/-- The line l: y = 3x + 3 -/
def l : Line := { slope := 3, intercept := 3 }

/-- The line l₂: 11x - 2y + 21 = 0, rewritten as y = (11/2)x + (21/2) -/
def l₂ : Line := { slope := 11/2, intercept := 21/2 }

/-- Two points are symmetric with respect to a line if the line is the perpendicular bisector of the segment connecting the two points -/
def symmetric_points (p₁ p₂ : Point) (l : Line) : Prop :=
  let midpoint : Point := { x := (p₁.x + p₂.x) / 2, y := (p₁.y + p₂.y) / 2 }
  midpoint.onLine l ∧ (p₂.y - p₁.y) * l.slope = -(p₂.x - p₁.x)

/-- Two lines are symmetric with respect to a third line if every point on one line has a symmetric point on the other line with respect to the third line -/
def symmetric_lines (l₁ l₂ l : Line) : Prop :=
  ∀ p₁ : Point, p₁.onLine l₁ → ∃ p₂ : Point, p₂.onLine l₂ ∧ symmetric_points p₁ p₂ l

theorem line_symmetry : symmetric_lines l₁ l₂ l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_symmetry_l835_83576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scroll_age_ratio_l835_83589

/-- The ratio of age difference between consecutive scrolls to the age of the last scroll -/
noncomputable def age_ratio (first_scroll_age last_scroll_age : ℝ) : ℝ :=
  (last_scroll_age - first_scroll_age) / (4 * last_scroll_age)

theorem scroll_age_ratio :
  let first_scroll_age : ℝ := 4080
  let last_scroll_age : ℝ := 20655
  let n : ℕ := 5  -- number of scrolls
  ∃ ε > 0, |age_ratio first_scroll_age last_scroll_age - 0.2006| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scroll_age_ratio_l835_83589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l835_83535

open Real

theorem equation_solution (x : ℝ) (k : ℤ) : 
  (sin x ≠ 0) → 
  (cos x ≠ 0) → 
  (cos (2 * x) ≠ 0) → 
  (cos (4 * x) ≠ 0) → 
  ((1 / tan x) - tan x - 2 * tan (2 * x) - 4 * tan (4 * x) + 8 = 0) ↔ 
  (x = π / 32 * (4 * ↑k + 3)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l835_83535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_discount_is_50_percent_l835_83573

/-- Calculates the discount percentage on an aquarium given the original price,
    sales tax rate, and final price after discount and tax. -/
noncomputable def calculate_discount (original_price : ℝ) (tax_rate : ℝ) (final_price : ℝ) : ℝ :=
  let discounted_price := final_price / (1 + tax_rate)
  (original_price - discounted_price) / original_price * 100

theorem aquarium_discount_is_50_percent :
  calculate_discount 120 0.05 63 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aquarium_discount_is_50_percent_l835_83573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_submarine_rise_l835_83537

/-- The change in depth when a submarine moves from 27 meters below sea level
    to 18 meters below sea level is 9 meters upward. -/
theorem submarine_rise 
  (initial_depth : Int)
  (final_depth : Int)
  (h1 : initial_depth = -27)
  (h2 : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_submarine_rise_l835_83537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_palindromic_prime_l835_83596

def is_palindrome (n : Nat) : Prop :=
  (Nat.digits 10 n).reverse = Nat.digits 10 n

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_palindromic_prime :
  ∃ (n : Nat), is_four_digit n ∧ is_palindrome n ∧ Nat.Prime n ∧
  (∀ m : Nat, is_four_digit m → is_palindrome m → Nat.Prime m → n ≤ m) ∧
  n = 1441 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_palindromic_prime_l835_83596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_theorem_l835_83527

/-- A coloring function that assigns a color (represented by a natural number) to each element of a set -/
def Coloring (α : Type*) := α → ℕ

/-- The property that every color is used at least once in a coloring -/
def every_color_used (S : Finset ℕ) (c : Coloring ℕ) (k : ℕ) : Prop :=
  ∀ i, i < k → ∃ s, s ∈ S ∧ c s = i

/-- The property that any three elements whose product is a perfect square have exactly two different colors -/
def square_product_two_colors (S : Finset ℕ) (c : Coloring ℕ) : Prop :=
  ∀ a b d, a ∈ S → b ∈ S → d ∈ S → 
    (∃ n : ℕ, a * b * d = n^2) → 
    (c a = c b ∧ c b ≠ c d) ∨ (c b = c d ∧ c d ≠ c a) ∨ (c d = c a ∧ c a ≠ c b)

/-- The main theorem stating that for a set S of 2^n - 1 elements, 
    if we can color S with k colors satisfying the given conditions, then k must equal n -/
theorem coloring_theorem (n : ℕ) (h : n > 0) :
  ∀ (S : Finset ℕ) (c : Coloring ℕ) (k : ℕ),
    Finset.card S = 2^n - 1 →
    every_color_used S c k →
    square_product_two_colors S c →
    k = n := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_theorem_l835_83527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_sumReciprocals_relation_l835_83510

/-- Represents a geometric progression with n terms -/
structure GeometricProgression where
  n : ℕ
  a : ℝ
  r : ℝ

/-- The product of terms in a geometric progression -/
noncomputable def product (gp : GeometricProgression) : ℝ := 
  gp.a^gp.n * gp.r^((gp.n * (gp.n - 1)) / 2)

/-- The sum of terms in a geometric progression -/
noncomputable def sum (gp : GeometricProgression) : ℝ := 
  gp.a * (1 - gp.r^gp.n) / (1 - gp.r)

/-- The sum of reciprocals of terms in a geometric progression -/
noncomputable def sumReciprocals (gp : GeometricProgression) : ℝ := 
  (gp.r^gp.n - 1) / (gp.a * (gp.r - 1))

/-- Theorem stating the relationship between product, sum, and sum of reciprocals -/
theorem product_sum_sumReciprocals_relation (gp : GeometricProgression) :
  product gp = (sum gp / sumReciprocals gp) ^ (gp.n / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sum_sumReciprocals_relation_l835_83510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_odd_sum_arrangement_l835_83536

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def is_even (n : ℕ) : Prop := ∃ k, n = 2*k

def circular_arrangement (arr : List ℕ) : Prop :=
  arr.length = 2018 ∧ (∀ n, n ∈ arr ↔ 1 ≤ n ∧ n ≤ 2018)

theorem impossible_odd_sum_arrangement :
  ¬ ∃ arr : List ℕ, 
    circular_arrangement arr ∧ 
    (∀ i, i < arr.length → is_odd (arr[i]! + arr[(i+1) % arr.length]! + arr[(i+2) % arr.length]! + arr[(i+3) % arr.length]!)) :=
by
  sorry

#check impossible_odd_sum_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_odd_sum_arrangement_l835_83536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l835_83523

-- Define the function h as noncomputable
noncomputable def h (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 1)

-- State the theorem
theorem range_of_h :
  ∀ y : ℝ, (∃ t : ℝ, h t = y) ↔ ((4 - Real.sqrt 41) / 8 ≤ y ∧ y ≤ (4 + Real.sqrt 41) / 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l835_83523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l835_83549

def Nstar : Type := {n : ℕ | n > 0}

theorem exists_special_function : ∃ f : Nstar → Nstar, 
  (f ⟨1, by norm_num⟩ = ⟨2, by norm_num⟩) ∧ 
  (∀ n : Nstar, f (f n) = ⟨(f n).val + n.val, by sorry⟩) ∧ 
  (∀ n : Nstar, (f n).val < (f ⟨n.val + 1, by sorry⟩).val) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l835_83549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l835_83547

/-- Circle C with equation (x+1)^2 + (y-1)^2 = 1 -/
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Point M is on the line y = -x -/
def point_M_on_line (x y : ℝ) : Prop := y = -x

/-- Point M is on the circle -/
def point_M_on_circle (x y : ℝ) : Prop := circle_C x y ∧ point_M_on_line x y

/-- Equation of the tangent line at point (a, b) -/
def tangent_line (a b x y : ℝ) : Prop := y - b = (x - a)

theorem tangent_line_at_M :
  ∃ x y : ℝ, point_M_on_circle x y ∧
  tangent_line x y x y ∧
  ∀ x' y' : ℝ, tangent_line x y x' y' ↔ y' = x' + 2 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_M_l835_83547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellis_family_water_bottles_l835_83517

/-- Calculates the number of water bottles needed for a family road trip. -/
def water_bottles_needed (family_size : ℕ) (trip_duration : ℕ) (water_per_person_per_hour : ℚ) : ℕ :=
  (((family_size : ℚ) * trip_duration * water_per_person_per_hour).ceil).toNat

/-- Proves that Ellis' family needs 32 water bottles for their road trip. -/
theorem ellis_family_water_bottles :
  water_bottles_needed 4 16 (1/2) = 32 := by
  -- Unfold the definition of water_bottles_needed
  unfold water_bottles_needed
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl

#eval water_bottles_needed 4 16 (1/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellis_family_water_bottles_l835_83517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l835_83500

/-- 
Given a bus that stops for 30 minutes per hour and has an average speed of 30 km/hr including stoppages,
prove that its average speed excluding stoppages is 60 km/hr.
-/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 30) 
  (h2 : avg_speed_with_stops = 30) : 
  (avg_speed_with_stops * 60) / (60 - stop_time) = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stoppages_l835_83500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83590

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

theorem f_properties :
  ∃ (x₀ : ℝ),
    (∀ x ∈ Set.Icc 0 Real.pi, f x ≤ f x₀) ∧
    (∀ x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8), 
     ∀ y ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8), 
     x < y → f y < f x) ∧
    f x₀ + f (2 * x₀) + f (3 * x₀) = 2 - Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gini_coefficient_change_example_country_gini_change_l835_83501

-- Define the regions
structure Region where
  population : ℕ
  ppc : ℝ → ℝ
  maximizeIncome : ℝ

-- Define the country
structure Country where
  north : Region
  south : Region

-- Define the Gini coefficient calculation
noncomputable def giniCoefficient (poorerPopulation totalPopulation poorerIncome totalIncome : ℝ) : ℝ :=
  poorerPopulation / totalPopulation - poorerIncome / totalIncome

-- Define the problem
theorem gini_coefficient_change (country : Country) :
  let initialGini := giniCoefficient 24 30 36000 60000
  let finalGini := giniCoefficient 24 30 24339 61000
  initialGini = 0.2 ∧ initialGini - finalGini = 0.001 := by
  sorry

-- Define the country with given conditions
def exampleCountry : Country :=
  { north := { population := 24,
               ppc := λ x => 13.5 - 9 * x,
               maximizeIncome := 36000 },
    south := { population := 6,
               ppc := λ x => 1.5 * x^2 - 24,
               maximizeIncome := 24000 } }

-- Prove the theorem for the example country
theorem example_country_gini_change :
  let country := exampleCountry
  let initialGini := giniCoefficient 24 30 36000 60000
  let finalGini := giniCoefficient 24 30 24339 61000
  initialGini = 0.2 ∧ initialGini - finalGini = 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gini_coefficient_change_example_country_gini_change_l835_83501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l835_83571

/-- Given vectors a and b with an acute angle between them, 
    prove the range of lambda in a = (lambda,2) and b = (3,4) -/
theorem lambda_range (a b : ℝ × ℝ) (lambda : ℝ) 
  (h1 : a = (lambda, 2))
  (h2 : b = (3, 4))
  (h3 : 0 < a.1 * b.1 + a.2 * b.2) :
  lambda > -8/3 ∧ lambda ≠ -3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l835_83571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_area_l835_83579

noncomputable def f (x : ℝ) : ℝ := -x^3
noncomputable def g (x : ℝ) : ℝ := 2 / (|x^3| + x^3)

noncomputable def tangent_intersection_y (x₀ : ℝ) : ℝ × ℝ :=
  ((256/27) * x₀^3, (27/4) * x₀^3)

noncomputable def total_area (x₀ : ℝ) : ℝ :=
  (1/2) * ((256/27) * x₀^3 + (27/4) * x₀^3)

theorem min_total_area :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  (∀ (y : ℝ), y > 0 → total_area x₀ ≤ total_area y) ∧
  total_area x₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_area_l835_83579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l835_83555

/-- Regular tetrahedron with height 2 and base side length √2 -/
structure RegularTetrahedron where
  height : ℝ
  base_side : ℝ
  height_eq : height = 2
  base_side_eq : base_side = Real.sqrt 2

/-- The shortest distance between skew lines BD and SC in the tetrahedron -/
noncomputable def shortest_distance (t : RegularTetrahedron) : ℝ := 2 * Real.sqrt 5 / 5

/-- Theorem stating that the shortest distance between any point on BD and any point on SC
    in a regular tetrahedron with height 2 and base side length √2 is (2√5)/5 -/
theorem shortest_distance_theorem (t : RegularTetrahedron) :
  shortest_distance t = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_theorem_l835_83555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_distribution_l835_83560

theorem pirate_treasure_distribution
  (n b c : ℕ+) 
  (bags : Fin (n * b) → ℕ)
  (h_total_coins : (Finset.univ.sum bags) = n * c)
  (h_empty_bags : (Finset.univ.filter (λ i => bags i = 0)).card ≥ n - 1) :
  ∃ (moves : ℕ) (final_bags : Fin (n * b) → ℕ),
    moves ≤ n - 1 ∧
    (Finset.univ.sum final_bags) = n * c ∧
    ∀ pirate : Fin n, 
      ∃ pirate_bags : Finset (Fin (n * b)),
        pirate_bags.card = b ∧
        (pirate_bags.sum final_bags) = c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_treasure_distribution_l835_83560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_clinic_cost_l835_83569

/-- Calculates the total cost for Mr. Sean's veterinary clinic given the number of animals and pricing rules -/
def total_cost (dogs cats parrots rabbits : ℕ) : ℚ :=
  let dog_price := 60
  let cat_price := 40
  let parrot_price := 70
  let rabbit_price := 50
  let dog_discount := if dogs > 20 then 1/10 else 0
  let cat_discount := if cats > 30 then 3/20 else 0
  let parrot_discount := if parrots > 10 then 1/20 else 0
  let dog_cost := (1 - dog_discount) * (dogs * dog_price)
  let cat_cost := (1 - cat_discount) * (cats * cat_price)
  let parrot_cost := (1 - parrot_discount) * (parrots * parrot_price)
  let rabbit_cost := rabbits * rabbit_price
  ↑dog_cost + ↑cat_cost + ↑parrot_cost + ↑rabbit_cost

theorem sean_clinic_cost :
  total_cost 25 35 12 10 = 3838 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sean_clinic_cost_l835_83569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_equation_l835_83530

noncomputable def solutions : Set ℂ := {
  (8 : ℂ) ^ (1/6 : ℂ) * Complex.exp (Real.pi * Complex.I / 6),
  (8 : ℂ) ^ (1/6 : ℂ) * Complex.exp (5 * Real.pi * Complex.I / 6),
  (8 : ℂ) ^ (1/6 : ℂ) * Complex.exp (7 * Real.pi * Complex.I / 6),
  (8 : ℂ) ^ (1/6 : ℂ) * Complex.exp (11 * Real.pi * Complex.I / 6),
  Complex.I * (2 : ℂ) ^ (1/3 : ℂ),
  -Complex.I * (2 : ℂ) ^ (1/3 : ℂ)
}

theorem solutions_of_equation : {z : ℂ | z^6 = -8} = solutions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_equation_l835_83530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l835_83526

/-- Given two planar vectors OA and OB of unit length with an angle of 120° between them,
    and a point C on the circular arc AB with O as the center,
    prove that the maximum value of x + y is 2,
    where OC = x*OA + y*OB and x, y are real numbers. -/
theorem max_sum_of_coefficients (OA OB OC : ℝ × ℝ) (x y : ℝ) :
  (norm OA = 1) →
  (norm OB = 1) →
  (OA.1 * OB.1 + OA.2 * OB.2 = -1/2) →  -- cos 120° = -1/2
  (OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2)) →
  (norm OC = 1) →
  (x + y ≤ 2) ∧ (∃ x' y' : ℝ, x' + y' = 2 ∧ 
    ((x' * OA.1 + y' * OB.1)^2 + (x' * OA.2 + y' * OB.2)^2 = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_coefficients_l835_83526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83554

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x - a) * Real.exp x

-- State the theorem
theorem f_properties :
  ∀ a, a ∈ Set.Ioo 0 2 →
  (∀ x : ℝ, x < -2 → StrictMonoOn (f 1) (Set.Iic x)) ∧
  (∀ x : ℝ, x > 1 → StrictMonoOn (f 1) (Set.Ici x)) ∧
  (StrictAntiOn (f 1) (Set.Ioo (-2) 1)) ∧
  (∀ m : ℝ, (∀ x₁ x₂, x₁ ∈ Set.Icc (-4) 0 → x₂ ∈ Set.Icc (-4) 0 → 
    |f a x₁ - f a x₂| < (6 * Real.exp (-2) + 2) * m) ↔ m ≥ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_riders_l835_83519

/-- The number of seats on the ferris wheel -/
def num_seats : ℝ := 6.0

/-- The number of times the ferris wheel needs to run -/
def num_runs : ℝ := 2.333333333

/-- The number of people who want to ride the ferris wheel -/
def num_people : ℕ := 14

theorem ferris_wheel_riders :
  Int.floor (num_seats * num_runs) = num_people := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_riders_l835_83519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l835_83558

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define midpoints P and Q
noncomputable def P : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
noncomputable def Q : ℝ × ℝ := ((B.1 + D.1) / 2, (B.2 + D.2) / 2)

-- Define the intersection point O
noncomputable def O : ℝ × ℝ := sorry

-- Define midpoints X, Y, Z, T of the sides of ABCD
noncomputable def X : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def Y : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
noncomputable def Z : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
noncomputable def T : ℝ × ℝ := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

-- Define the areas of the four quadrilaterals
noncomputable def area_OXBY : ℝ := sorry
noncomputable def area_OYCZ : ℝ := sorry
noncomputable def area_OZDT : ℝ := sorry
noncomputable def area_OTAX : ℝ := sorry

-- Theorem statement
theorem equal_areas :
  area_OXBY = area_OYCZ ∧ area_OYCZ = area_OZDT ∧ area_OZDT = area_OTAX :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_l835_83558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_properties_l835_83565

/-- Represents a sphere inscribed in a cube -/
structure InscribedSphere where
  cubeEdge : ℝ
  sphereRadius : ℝ
  sphereRadius_eq : sphereRadius = cubeEdge / 2

/-- Calculate the volume of a sphere -/
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Calculate the surface area of a sphere -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem inscribed_sphere_properties (s : InscribedSphere) (h : s.cubeEdge = 10) :
  sphereVolume s.sphereRadius = (500 / 3) * Real.pi ∧
  sphereSurfaceArea s.sphereRadius = 100 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_properties_l835_83565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l835_83543

/-- Represents the election results for 5 candidates -/
structure ElectionResult where
  a : ℕ  -- votes for candidate A
  b : ℕ  -- votes for candidate B
  c : ℕ  -- votes for candidate C
  d : ℕ  -- votes for candidate D
  e : ℕ  -- votes for candidate E

/-- The total number of votes in the election -/
def totalVotes : ℕ := 50000

/-- The conditions of the election result -/
def validElectionResult (result : ElectionResult) : Prop :=
  result.a + result.b + result.c + result.d + result.e = totalVotes ∧
  result.b = (30 * totalVotes) / 100 ∧
  result.c = result.b - (20 * result.b) / 100 ∧
  result.a = result.b + (12 * result.b) / 100

/-- The theorem stating that the winning candidate received 33.6% of the total votes -/
theorem winning_candidate_percentage (result : ElectionResult) 
  (h : validElectionResult result) : 
  (result.a : ℚ) / (totalVotes : ℚ) = 336 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_candidate_percentage_l835_83543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_1996_is_perfect_square_l835_83528

/-- The number of ways to write n as a sum of 1's, 3's, and 4's (order matters) -/
def a : ℕ → ℕ := sorry

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ := sorry

/-- For all natural numbers n, a_(2n) is a perfect square -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by
  sorry

/-- a_1996 is a perfect square -/
theorem a_1996_is_perfect_square : ∃ k : ℕ, a 1996 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2n_is_perfect_square_a_1996_is_perfect_square_l835_83528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_points_l835_83515

theorem basketball_points (x : ℚ) (y : ℕ) : 
  (1/3 : ℚ) * x + (3/8 : ℚ) * x + 18 + y = x ∧ 
  y ≤ 24 ∧ 
  (∀ (i : ℕ), i ≤ 8 → 
    (∃ (p : ℕ), p ≤ 3 ∧ y = (Finset.sum (Finset.range i) (λ _ => p)) + (8 - i) * p)) →
  y = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_points_l835_83515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_belly_percentage_l835_83533

/-- Represents the number of minnows in a pond with different belly colors -/
structure MinnowCount where
  total : ℕ
  red : ℕ
  white : ℕ
  green : ℕ

/-- Calculates the percentage of minnows with a specific belly color -/
def percentageOfColor (count : MinnowCount) (color : ℕ) : ℚ :=
  (color : ℚ) / (count.total : ℚ) * 100

/-- Theorem stating that 30% of minnows have green bellies given the conditions -/
theorem green_belly_percentage (count : MinnowCount) :
  count.red = 20 →
  count.white = 15 →
  percentageOfColor count count.red = 40 →
  count.total = count.red + count.white + count.green →
  percentageOfColor count count.green = 30 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_belly_percentage_l835_83533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_lights_l835_83521

/-- Represents the number of lights on each level of a pagoda -/
def pagoda_lights : ℕ → ℕ := sorry

/-- The total number of stories in the pagoda -/
def num_stories : ℕ := 7

/-- The sum of lights on all stories of the pagoda -/
def total_lights : ℕ := 381

/-- Each lower level has twice the lights of the level above it -/
axiom light_progression (n : ℕ) (h : n < num_stories - 1) : 
  pagoda_lights n = 2 * pagoda_lights (n + 1)

/-- The sum of lights on all stories equals the total lights -/
axiom sum_equals_total : 
  (Finset.range num_stories).sum pagoda_lights = total_lights

/-- The number of lights at the top of the pagoda is 3 -/
theorem top_lights : pagoda_lights (num_stories - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_lights_l835_83521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l835_83557

def is_simplest_square_root (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x^2 = y → ¬∃ (a b : ℝ), a ≠ 1 ∧ b ≠ x ∧ x = a * Real.sqrt b

theorem simplest_square_root :
  is_simplest_square_root (Real.sqrt 6) ∧
  ¬is_simplest_square_root (Real.sqrt 8) ∧
  ¬is_simplest_square_root (Real.sqrt (2/3)) ∧
  ¬is_simplest_square_root (Real.sqrt 0.5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_square_root_l835_83557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l835_83582

-- Define the quadratic function f
noncomputable def f (x : ℝ) : ℝ := 1/2 * (x - 2)^2 + 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- minimum value is 1
  (f 0 = 3) ∧ (f 4 = 3) ∧  -- f(0) = f(4) = 3
  (∀ x, f x = 1/2 * (x - 2)^2 + 1) ∧  -- explicit expression
  (∀ a, (∀ x ∈ Set.Icc (2*a) (3*a + 1), StrictMono f ∨ StrictAnti f) ↔ 
    (a ≤ 1/3 ∨ a ≥ 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l835_83582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_problem_l835_83548

-- Define the sliding distance function for the first skier
noncomputable def s (t : ℝ) : ℝ := t^2 + t

-- Define the sliding distance function for the second skier
noncomputable def s2 (t : ℝ) : ℝ := (5/2) * t^2 + 2 * t

-- Define the data points
def data_points : List (ℝ × ℝ) := [(0, 0), (1, 2), (2, 6), (3, 12), (4, 20)]

-- Theorem statement
theorem skier_problem :
  (∀ (p : ℝ × ℝ), p ∈ data_points → s p.1 = p.2) ∧
  (∃ (t2 : ℝ), t2 > 0 ∧ s2 t2 = s 4 ∧ t2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skier_problem_l835_83548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_path_theorem_l835_83553

/-- Represents the number of paths from point 1 to point 5 in 2n steps in a regular octagon -/
noncomputable def num_paths (n : ℕ) : ℝ :=
  (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))

/-- The conditions of the octagon path problem -/
structure OctagonPathProblem where
  /-- The octagon has 8 vertices -/
  num_vertices : ℕ
  num_vertices_eq : num_vertices = 8
  /-- The drunkard starts from point 1 -/
  start_point : ℕ
  start_point_eq : start_point = 1
  /-- The drunkard stops at point 5 -/
  end_point : ℕ
  end_point_eq : end_point = 5
  /-- At each vertex except 5, the drunkard has two choices -/
  choices_at_vertex : ℕ
  choices_at_vertex_eq : choices_at_vertex = 2
  /-- Each move counts as one step -/
  steps_per_move : ℕ
  steps_per_move_eq : steps_per_move = 1

/-- Represents the number of paths from 1 to 5 in 2n steps for a given problem -/
def number_of_paths_from_1_to_5_in_2n_steps (problem : OctagonPathProblem) (steps : ℕ) : ℕ :=
  sorry -- This function is not implemented, but we define it to use in the theorem

/-- Theorem stating that num_paths gives the correct number of paths for the octagon problem -/
theorem octagon_path_theorem (problem : OctagonPathProblem) (n : ℕ) :
  ∃ (paths : ℕ), paths = ⌊num_paths n⌋ ∧ 
  paths = number_of_paths_from_1_to_5_in_2n_steps problem (2 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_path_theorem_l835_83553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l835_83583

/-- The measure of an exterior angle of a regular octagon is 45 degrees. -/
theorem exterior_angle_regular_octagon : ℝ := by
  let n : ℕ := 8  -- number of sides in an octagon
  let exterior_angle : ℝ := 360 / n
  have h : exterior_angle = 45 := by
    -- Proof goes here
    sorry
  exact exterior_angle


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_regular_octagon_l835_83583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_80_l835_83597

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_sqrt_80_l835_83597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l835_83532

/-- Given three real numbers 2, m, 8 forming a geometric sequence,
    the eccentricity of the conic section x^2/m + y^2/2 = 1 is either √2/2 or √3 -/
theorem conic_section_eccentricity (m : ℝ) :
  (2 * m = m * 8) →
  let e := if m > 0 then Real.sqrt (1 - 2 / m) else Real.sqrt (1 + m / 2)
  (e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l835_83532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_with_347_l835_83591

def is_valid_fraction (m n : ℕ) : Prop :=
  m < n ∧ Nat.Coprime m n ∧ ∃ k : ℕ, (1000 * m - 347 * n) * 10^k ≥ n ∧ (1000 * m - 347 * n) * 10^k < 2 * n

theorem smallest_fraction_with_347 :
  ∀ m n : ℕ, is_valid_fraction m n →
    (∀ m' n' : ℕ, is_valid_fraction m' n' → n' ≥ n) →
    m = 6 ∧ n = 17 := by
  sorry

#check smallest_fraction_with_347

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_fraction_with_347_l835_83591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_inscribed_triangle_l835_83550

/-- Given a right triangle inscribed in a semicircle with an area of 5 square meters,
    prove that the area of the full circle is 78.5 square meters, assuming π = 3.14. -/
theorem circle_area_from_inscribed_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let triangle_area := (1/2) * a * b
  let diameter := Real.sqrt (a^2 + b^2)
  let circle_area := Real.pi * (diameter/2)^2
  triangle_area = 5 → circle_area = 78.5 :=
by
  sorry

/-- Assuming π = 3.14 for this problem -/
axiom pi_approx : Real.pi = 3.14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_inscribed_triangle_l835_83550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l835_83541

theorem problem_statement : (-1)^49 + 3^(Nat.factorial (2^3 + 5^2 - 4^2)) = -1 + 3^355687428096000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l835_83541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_inclination_ellipse_l835_83563

open Real

/-- The angle parameter for the ellipse -/
noncomputable def φ : ℝ := π / 6

/-- The x-coordinate of point P on the ellipse -/
noncomputable def x : ℝ := 3 * cos φ

/-- The y-coordinate of point P on the ellipse -/
noncomputable def y : ℝ := 2 * sin φ

/-- The tangent of the angle of inclination of line OP -/
noncomputable def tan_angle_inclination : ℝ := y / x

theorem tangent_angle_inclination_ellipse :
  tan_angle_inclination = 2 * sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_inclination_ellipse_l835_83563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_at_1200_l835_83507

/-- Represents the sequence of positive integers starting with 2 -/
def sequence_starting_with_2 : ℕ → ℕ := sorry

/-- Counts the number of digits in a positive integer -/
def digit_count (n : ℕ) : ℕ := sorry

/-- Returns the nth digit in the sequence of digits formed by concatenating 
    the numbers in sequence_starting_with_2 -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem three_digit_number_at_1200 :
  (toString (nth_digit 1198)) ++ (toString (nth_digit 1199)) ++ (toString (nth_digit 1200)) = "220" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_number_at_1200_l835_83507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_85_cents_combination_l835_83567

/-- Represents the types of coins available. -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents. -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Represents a selection of five coins. -/
def CoinSelection := Fin 5 → Coin

/-- Calculates the total value of a coin selection in cents. -/
def totalValue (selection : CoinSelection) : ℕ :=
  (Finset.sum Finset.univ fun i => coinValue (selection i))

/-- Theorem stating that no selection of five coins can sum to 85 cents. -/
theorem no_85_cents_combination : ¬ ∃ (selection : CoinSelection), totalValue selection = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_85_cents_combination_l835_83567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actress_plays_l835_83513

/-- Represents the total number of plays an actress participated in -/
def total_plays : ℕ → ℕ := sorry

/-- Represents the number of plays where an actress was not the lead -/
def not_lead_plays : ℕ → ℕ := sorry

/-- Represents the percentage of plays where an actress was the lead -/
def lead_percentage : ℕ → ℚ := sorry

theorem actress_plays (n : ℕ) (h1 : lead_percentage n = 4/5) 
  (h2 : not_lead_plays n = 20) : 
  total_plays n = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_actress_plays_l835_83513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_path_bound_l835_83566

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- A point on the surface of a regular tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) where
  point : ℝ × ℝ × ℝ
  on_surface : Unit  -- Placeholder for the condition

/-- A piecewise linear path on the surface of a regular tetrahedron -/
structure SurfacePath (t : RegularTetrahedron) where
  start : SurfacePoint t
  finish : SurfacePoint t
  path : List (ℝ × ℝ × ℝ)
  on_surface : Unit  -- Placeholder for the condition
  is_piecewise_linear : Unit  -- Placeholder for the condition

/-- The length of a piecewise linear path -/
noncomputable def path_length (t : RegularTetrahedron) (p : SurfacePath t) : ℝ := sorry

/-- The main theorem -/
theorem tetrahedron_surface_path_bound (t : RegularTetrahedron) 
  (p1 p2 : SurfacePoint t) : 
  ∃ (path : SurfacePath t), path.start = p1 ∧ path.finish = p2 ∧ 
    path_length t path ≤ 2 / Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_surface_path_bound_l835_83566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_probability_theorem_l835_83539

/-- Represents a stack of notebooks -/
structure NotebookStack :=
  (color : String)
  (count : Nat)

/-- Represents the outcome of drawing notebooks -/
structure DrawOutcome :=
  (blue : Nat)
  (red : Nat)

/-- The probability of an event -/
noncomputable def probability (favorableOutcomes totalOutcomes : Nat) : ℝ :=
  (favorableOutcomes : ℝ) / (totalOutcomes : ℝ)

/-- The setup of the notebook problem -/
def notebookProblem :=
  let blueStack : NotebookStack := ⟨"blue", 5⟩
  let redStack : NotebookStack := ⟨"red", 5⟩
  (blueStack, redStack)

theorem notebook_probability_theorem 
  (problem : NotebookStack × NotebookStack := notebookProblem) :
  -- Condition a
  probability 1 5 = (1 : ℝ) / 5 ∧
  -- Condition b
  probability 1 9 = (1 : ℝ) / 9 ∧
  -- Condition c
  probability 1 5 = (1 : ℝ) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_probability_theorem_l835_83539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_temperature_l835_83502

def monday_temp : ℝ := 50
def tuesday_temp : ℝ := 65
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℝ := 7

theorem sunday_temperature :
  ∃ (sunday_temp : ℝ),
    (sunday_temp + monday_temp + tuesday_temp + wednesday_temp +
     thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp ∧
    sunday_temp = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunday_temperature_l835_83502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_sufficient_not_necessary_correlation_coefficient_incorrect_conjunction_implication_l835_83581

-- Statement A
theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ (∃ x_0 : ℝ, x_0^3 - x_0^2 - 1 > 0) :=
sorry

-- Statement B
theorem sufficient_not_necessary (a b : ℝ) :
  (∃ m : ℝ, a * m^2 < b * m^2) → (a < b) ∧
  ¬ ((a < b) → (∀ m : ℝ, a * m^2 < b * m^2)) :=
sorry

-- Statement C
-- We'll define a simple correlation coefficient function for this example
def simple_correlation_coefficient (n : ℕ) (x y : Fin n → ℝ) : ℝ :=
  sorry -- Placeholder for the actual implementation

theorem correlation_coefficient (n : ℕ) (x y : Fin n → ℝ) :
  (∀ i : Fin n, y i = -2 * x i + 1) →
  simple_correlation_coefficient n x y = -1 :=
sorry

-- Statement D (incorrect)
theorem incorrect_conjunction_implication (p q : Prop) :
  ¬ ((p ∧ q → False) → (p → False) ∧ (q → False)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_sufficient_not_necessary_correlation_coefficient_incorrect_conjunction_implication_l835_83581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_5_and_13_l835_83511

theorem three_digit_divisible_by_5_and_13 : 
  Finset.card (Finset.filter (fun n => 100 ≤ n ∧ n ≤ 999 ∧ n % 5 = 0 ∧ n % 13 = 0) (Finset.range 1000)) = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_5_and_13_l835_83511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_l835_83585

/-- An equilateral triangle -/
structure EquilateralTriangle where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- A point on a median of the triangle -/
structure MedianPoint (T : EquilateralTriangle) where
  /-- The distance from the vertex to the point -/
  dist_from_vertex : ℝ
  /-- The distance from the point to the midpoint of the opposite side -/
  dist_to_midpoint : ℝ
  /-- The point divides the median in ratio 3:1 from the vertex -/
  divides_3_1 : 3 * dist_to_midpoint = dist_from_vertex
  /-- The sum of distances equals the length of the median -/
  on_median : dist_from_vertex + dist_to_midpoint = T.side * Real.sqrt 3 / 2

/-- Calculate the area of a triangle given three points -/
noncomputable def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of an equilateral triangle -/
noncomputable def area_of_equilateral_triangle (T : EquilateralTriangle) : ℝ := 
  (T.side^2 * Real.sqrt 3) / 4

/-- The theorem to be proved -/
theorem area_ratio_equilateral_triangle (T : EquilateralTriangle) 
  (p1 p2 p3 : MedianPoint T) : 
  ∃ (A : ℝ), A > 0 ∧ A = (area_of_triangle (0, 0) (1, 0) (1/2, Real.sqrt 3 / 2)) * 64 ∧ 
             A = area_of_equilateral_triangle T := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_equilateral_triangle_l835_83585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_20_l835_83580

-- Define the function f as noncomputable due to the use of Real.pi
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin ((x + 1) * Real.pi) + b * (x - 1) ^ (1/3) + 2

-- State the theorem using Real.log instead of Real.log 2
theorem f_log_20 (a b : ℝ) :
  (f a b (Real.log 5 / Real.log 2) = 5) →
  (f a b (Real.log 20 / Real.log 2) = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_log_20_l835_83580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_decreasing_interval_l835_83564

/-- A function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 3 * (m - 1) * x^2 - m^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3 * m * x^2 + 6 * (m - 1) * x

theorem function_monotonically_decreasing_interval (m : ℝ) (h1 : m > 0) :
  (∀ x ∈ Set.Ioo 0 4, StrictAntiOn (f m) (Set.Ioo 0 4)) → m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_decreasing_interval_l835_83564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_3_necessary_not_sufficient_condition_l835_83570

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | |x + a| < 1}

def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem union_when_a_is_3 : A ∪ B 3 = Set.Ioo (-4) 1 := by sorry

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬q a x) ↔ 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_when_a_is_3_necessary_not_sufficient_condition_l835_83570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_scoring_l835_83538

/-- Coin flip scoring problem from 32nd Annual Putnam Mathematical Competition, 1971 -/
theorem coin_flip_scoring (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (n : ℕ), ∀ (m : ℕ), m ≥ n → ∃ (x y : ℕ), m = a * x + b * y) ∧
  (∃! (s : Finset ℕ), s.card = 35 ∧ ∀ k ∈ s, ∀ (x y : ℕ), k ≠ a * x + b * y) →
  a = 11 ∧ b = 8 :=
by
  sorry

#check coin_flip_scoring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_scoring_l835_83538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l835_83551

-- Define the square side length and circle radius
def square_side : ℝ := 8
def circle_radius : ℝ := 3

-- Define the area of the square
def square_area : ℝ := square_side ^ 2

-- Define the area of one circle
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

-- Define the area of one 90-degree sector of a circle
noncomputable def sector_area : ℝ := circle_area / 4

-- Define the total area covered by the four sectors
noncomputable def total_sectors_area : ℝ := 4 * sector_area

-- Theorem: The area of the shaded region is equal to 64 - 9π
theorem shaded_area_theorem :
  square_area - total_sectors_area = 64 - 9 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l835_83551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l835_83586

theorem least_subtraction (n : ℕ) : 
  (∀ d : ℕ, d ∈ ({9, 11, 13} : Set ℕ) → (2590 - n) % d = 6) → n ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l835_83586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_range_of_x_when_f_positive_l835_83520

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

-- Theorem for the domain of f(x)
theorem domain_of_f : Set.Ioo (-1 : ℝ) 1 = {x : ℝ | ∃ y, f x = y} := by sorry

-- Theorem for f(x) being an odd function
theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

-- Theorem for the range of x when f(x) > 0
theorem range_of_x_when_f_positive : Set.Ioo (0 : ℝ) 1 = {x : ℝ | f x > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_is_odd_range_of_x_when_f_positive_l835_83520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83504

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 4)^2 + Real.log (x^(1/2)) / Real.log (1/2) - 3

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 8}

theorem f_properties :
  let range := {y | ∃ x ∈ domain, f x = y}
  (∀ y ∈ range, -13/4 ≤ y ∧ y ≤ -9/4) ∧
  (∃ x ∈ domain, f x = -13/4 ∧ x = 2) ∧
  (∃ x ∈ domain, f x = -9/4 ∧ x = 8) ∧
  (∀ x ∈ domain, f x ≥ -13/4) ∧
  (∀ x ∈ domain, f x ≤ -9/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l835_83572

-- Define the lines
noncomputable def line1 (x y : ℝ) : Prop := y = x
noncomputable def line2 (x : ℝ) : Prop := x = -8
noncomputable def line3 (x y : ℝ) : Prop := y = -2 * x + 4

-- Define the intersection points
noncomputable def intersection1 : ℝ × ℝ := (-8, -8)
noncomputable def intersection2 : ℝ × ℝ := (4/3, 4/3)
noncomputable def intersection3 : ℝ × ℝ := (-8, 20)

-- Define the trapezoid
noncomputable def trapezoid_vertices : List (ℝ × ℝ) := [(-8, 0), (-8, -8), (4/3, 0), (4/3, 4/3)]

-- Theorem stating the area of the figure
theorem area_of_figure : 
  let vertices := trapezoid_vertices
  let base1 := |vertices[0].1 - vertices[2].1|
  let base2 := |vertices[1].1 - vertices[3].1|
  let height := vertices[3].2
  (1/2) * (base1 + base2) * height = 112/9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_figure_l835_83572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_traversal_impossible_l835_83592

theorem maze_traversal_impossible (n : Nat) (h : n = 6) : 
  ¬ ∃ (path : List (Nat × Nat)),
    (path.length = n * n) ∧
    (path.Nodup) ∧
    (∀ (i : Nat), i < path.length - 1 → 
      ((path.get! i).1 + (path.get! i).2) % 2 ≠ ((path.get! (i+1)).1 + (path.get! (i+1)).2) % 2) ∧
    ((path.head!.1 + path.head!.2) % 2 = (path.getLast!.1 + path.getLast!.2) % 2) :=
by
  sorry

#check maze_traversal_impossible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maze_traversal_impossible_l835_83592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l835_83534

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line with slope 1 passing through a point -/
structure Line where
  p : Point

/-- Definition of the ellipse equation -/
def on_ellipse (E : Ellipse) (p : Point) : Prop :=
  p.x^2 / E.a^2 + p.y^2 / E.b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (E : Ellipse) : ℝ :=
  Real.sqrt (1 - E.b^2 / E.a^2)

/-- Definition of arithmetic sequence -/
def is_arithmetic_seq (a b c : ℝ) : Prop :=
  2 * b = a + c

/-- Definition of point on circle -/
def on_circle (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

/-- Distance between two points -/
noncomputable def dist (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Main theorem -/
theorem ellipse_properties (E : Ellipse) (F₁ F₂ A B : Point) (l : Line) :
  on_ellipse E F₁ →
  on_ellipse E F₂ →
  on_ellipse E A →
  on_ellipse E B →
  F₁.x < F₂.x →
  l.p = F₁ →
  is_arithmetic_seq (dist A F₂) (dist A B) (dist B F₂) →
  ∃ (r : ℝ), on_circle ⟨-2, 0⟩ r A ∧ on_circle ⟨-2, 0⟩ r B →
  eccentricity E = Real.sqrt 2 / 2 ∧
  E.a = 6 * Real.sqrt 2 ∧
  E.b = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l835_83534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_equals_one_l835_83559

theorem log_difference_equals_one : Real.log 20 - Real.log 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_difference_equals_one_l835_83559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l835_83503

/-- Parabola type -/
structure Parabola where
  a : ℚ
  vertex : ℚ × ℚ

/-- Line type -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Point type -/
def Point := ℚ × ℚ

/-- Given a parabola y^2 = 4x with focus F(1,0), and a line y = (2/3)x + 4/3
    intersecting the parabola at points M and N, the dot product of FM and FN equals 8. -/
theorem parabola_line_intersection_dot_product
  (C : Parabola)
  (l : Line)
  (F M N : Point) :
  C.a = 4 ∧ C.vertex = (0, 0) ∧
  F = (1, 0) ∧
  l.slope = 2/3 ∧ l.intercept = 4/3 ∧
  (M.2^2 = 4 * M.1) ∧ (N.2^2 = 4 * N.1) ∧
  (M.2 = l.slope * M.1 + l.intercept) ∧
  (N.2 = l.slope * N.1 + l.intercept) →
  ((M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_dot_product_l835_83503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_coords_l835_83529

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of AB
noncomputable def mag_AB : ℝ := Real.sqrt (AB.1^2 + AB.2^2)

-- Define the unit vector opposite to AB
noncomputable def opposite_unit_vector : ℝ × ℝ := (-AB.1 / mag_AB, -AB.2 / mag_AB)

-- Theorem statement
theorem opposite_unit_vector_coords :
  opposite_unit_vector = (-3/5, 4/5) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_unit_vector_coords_l835_83529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83531

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi/6) * Real.cos x + 1

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), (0 < T' ∧ T' < T) → ∃ (x : ℝ), f (x + T') ≠ f x) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (Real.pi/12) (Real.pi/2) → f x ≤ 5/4) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (Real.pi/12) (Real.pi/2) ∧ f x = 5/4) ∧
    (∀ (x : ℝ), x ∈ Set.Icc (Real.pi/12) (Real.pi/2) → f x ≥ 3/4) ∧
    (∃ (x : ℝ), x ∈ Set.Icc (Real.pi/12) (Real.pi/2) ∧ f x = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_page_difference_is_34_l835_83584

-- Define the number of pages for each book
def poetry_pages : ℤ := sorry
def documents_pages : ℤ := sorry
def rites_pages : ℤ := sorry
def changes_pages : ℤ := sorry
def spring_autumn_pages : ℤ := sorry

-- Define the conditions
axiom diff_poetry_documents : |poetry_pages - documents_pages| = 24
axiom diff_documents_rites : |documents_pages - rites_pages| = 17
axiom diff_rites_changes : |rites_pages - changes_pages| = 27
axiom diff_changes_spring_autumn : |changes_pages - spring_autumn_pages| = 19
axiom diff_spring_autumn_poetry : |spring_autumn_pages - poetry_pages| = 15

-- Define a function to get the maximum of five integers
def max_of_five (a b c d e : ℤ) : ℤ := max a (max b (max c (max d e)))

-- Define a function to get the minimum of five integers
def min_of_five (a b c d e : ℤ) : ℤ := min a (min b (min c (min d e)))

-- Theorem to prove
theorem page_difference_is_34 : 
  max_of_five poetry_pages documents_pages rites_pages changes_pages spring_autumn_pages - 
  min_of_five poetry_pages documents_pages rites_pages changes_pages spring_autumn_pages = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_page_difference_is_34_l835_83584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_difference_l835_83506

def digits : List ℕ := [2, 4, 5, 6, 9]

def to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧ b % 2 = 1 ∧
  (∀ d, d ∈ digits → (d ∈ (to_digits a) ∨ d ∈ (to_digits b))) ∧
  (∀ d, (d ∈ (to_digits a) ∨ d ∈ (to_digits b)) → d ∈ digits)

def smallest_difference : ℕ := 176

theorem smallest_possible_difference :
  ∀ a b : ℕ, is_valid_pair a b → a - b ≥ smallest_difference :=
by
  sorry

#eval smallest_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_difference_l835_83506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_greater_than_two_l835_83525

theorem subset_implies_greater_than_two (a : ℝ) :
  let A : Set ℝ := {x | 1 < x ∧ x ≤ 2}
  let B : Set ℝ := {x | x < a}
  A ⊂ B → a > 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_implies_greater_than_two_l835_83525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_count_l835_83514

def sequenceNum (n : ℕ) : ℕ := 47 * (10^(2*n) - 1) / 99

theorem sequence_prime_count : 
  (∃! k : ℕ, Nat.Prime (sequenceNum k)) ∧ sequenceNum 0 = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_prime_count_l835_83514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l835_83518

theorem exponential_equation_solution (k : ℝ) (x y : ℝ) : 
  9 * (3 : ℝ) ^ (k * x) = (7 : ℝ) ^ (y + 12) → y = -12 → x = -2 / k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l835_83518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_grill_l835_83593

/-- Represents a metal bar of length 2 that can be folded or cut -/
structure MetalBar where
  length : ℝ := 2
  can_fold : Bool
  can_cut : Bool

/-- Represents an n × n unit net grill -/
structure Grill where
  n : ℕ

/-- Function to calculate the minimum number of pieces required for a grill -/
def min_pieces (g : Grill) : ℕ :=
  g.n * (g.n + 1)

/-- Predicate to check if two metal bars overlap -/
def overlaps (piece1 piece2 : MetalBar) : Prop :=
  sorry -- Definition of overlap would go here

/-- Theorem stating the minimum number of pieces required for an n × n grill -/
theorem min_pieces_for_grill (g : Grill) (bar : MetalBar) :
  (bar.length = 2) →
  bar.can_fold →
  bar.can_cut →
  (∀ (piece1 piece2 : MetalBar), ¬(overlaps piece1 piece2)) →
  (min_pieces g = g.n * (g.n + 1)) :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_grill_l835_83593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83574

-- Define a as a real number
noncomputable def a : ℝ := 0

-- Define g as a function ℝ → ℝ
noncomputable def g : ℝ → ℝ := λ x => -x^2 - 2*x + 5

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 2*x - 5
  else if x = 0 then a
  else g x

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

theorem f_properties : a = 0 ∧ f (g (-1)) = 3 := by
  have h1 : a = 0 := by
    -- Proof for a = 0
    sorry
  have h2 : f (g (-1)) = 3 := by
    -- Proof for f(g(-1)) = 3
    sorry
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l835_83574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminated_portion_correct_l835_83561

/-- Represents a right circular cone -/
structure Cone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  (r_pos : r > 0)
  (h_pos : h > 0)

/-- Represents the light source position -/
structure LightSource where
  H : ℝ  -- distance from the plane
  l : ℝ  -- distance from the height of the cone

/-- Calculates the illuminated portion of a circle -/
noncomputable def illuminatedPortion (cone : Cone) (light : LightSource) (R : ℝ) : ℝ :=
  let s := |cone.h / (light.H - cone.h)|
  let α := Real.arccos (cone.r / R)
  let β := Real.arccos (cone.r / s)
  if light.H > cone.h then
    2 * Real.pi - (β - α)
  else if light.H = cone.h then
    2 * Real.pi - (Real.pi / 2 - α)
  else
    2 * Real.pi - (Real.pi - (α + β))

theorem illuminated_portion_correct (cone : Cone) (light : LightSource) (R : ℝ) 
  (h_R_pos : R > 0) :
  illuminatedPortion cone light R = 
    if light.H > cone.h then
      2 * Real.pi - (Real.arccos (cone.r / |cone.h / (light.H - cone.h)|) - Real.arccos (cone.r / R))
    else if light.H = cone.h then
      2 * Real.pi - (Real.pi / 2 - Real.arccos (cone.r / R))
    else
      2 * Real.pi - (Real.pi - (Real.arccos (cone.r / R) + Real.arccos (cone.r / |cone.h / (light.H - cone.h)|))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_illuminated_portion_correct_l835_83561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_meets_danny_at_225_meters_l835_83587

-- Define the setup
noncomputable def distance_CD : ℝ := 150
noncomputable def clara_speed : ℝ := 9
noncomputable def danny_speed : ℝ := 10
noncomputable def clara_angle : ℝ := 45 * Real.pi / 180  -- 45 degrees in radians

-- Define the function to calculate the meeting time
noncomputable def meeting_time : ℝ :=
  let a := danny_speed^2 - clara_speed^2
  let b := -2 * clara_speed * distance_CD * Real.cos clara_angle
  let c := distance_CD^2
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

-- Define Clara's distance
noncomputable def clara_distance : ℝ := clara_speed * meeting_time

-- Theorem statement
theorem clara_meets_danny_at_225_meters :
  ∃ ε > 0, |clara_distance - 225| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clara_meets_danny_at_225_meters_l835_83587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_two_minus_cos_range_sin_plus_cos_range_when_cubic_sum_negative_l835_83505

theorem sin_over_two_minus_cos_range : 
  ∀ x : ℝ, -Real.sqrt 3 / 3 ≤ Real.sin x / (2 - Real.cos x) ∧ 
           Real.sin x / (2 - Real.cos x) ≤ Real.sqrt 3 / 3 := by
  sorry

theorem sin_plus_cos_range_when_cubic_sum_negative :
  ∀ θ : ℝ, Real.sin θ ^ 3 + Real.cos θ ^ 3 < 0 →
           -Real.sqrt 2 ≤ Real.sin θ + Real.cos θ ∧ Real.sin θ + Real.cos θ < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_over_two_minus_cos_range_sin_plus_cos_range_when_cubic_sum_negative_l835_83505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_x_gt_1_l835_83595

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative_exists : Differentiable ℝ f

axiom f_plus_f'_gt_2 : ∀ x, f x + deriv f x > 2

axiom f_at_1 : f 1 = 2 + 4 / Real.exp 1

theorem f_inequality_iff_x_gt_1 :
  ∀ x, Real.exp x * f x > 4 + 2 * Real.exp x ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_x_gt_1_l835_83595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_inequality_l835_83524

-- Define the plane as a real inner product space
variable (Plane : Type) [NormedAddCommGroup Plane] [InnerProductSpace ℝ Plane]

-- Define a distance function
def dist (x y : Plane) : ℝ := ‖x - y‖

-- Define points A, B, C, D, and P
variable (A B C D P : Plane)

-- State that B and C lie within segment AD
variable (h1 : ∃ t₁ t₂ : ℝ, 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧ 
  B = (1 - t₁) • A + t₁ • D ∧
  C = (1 - t₂) • A + t₂ • D)

-- State that AB = CD
variable (h2 : dist A B = dist C D)

-- Theorem statement
theorem segment_inequality :
  dist P A + dist P D ≥ dist P B + dist P C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_inequality_l835_83524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_U_ratio_l835_83552

def G (n : ℕ+) : ℕ :=
  (Finset.filter (fun (x : Fin n → Bool × Bool) => 
    (Finset.sum Finset.univ (fun i => (if (x i).1 ∧ (x i).2 then 1 else 0 : ℕ))) % 2 = 0) 
    (Finset.univ : Finset (Fin n → Bool × Bool))).card

def U (n : ℕ+) : ℕ :=
  (Finset.filter (fun (x : Fin n → Bool × Bool) => 
    (Finset.sum Finset.univ (fun i => (if (x i).1 ∧ (x i).2 then 1 else 0 : ℕ))) % 2 = 1) 
    (Finset.univ : Finset (Fin n → Bool × Bool))).card

theorem G_U_ratio (n : ℕ+) : 
  (G n : ℚ) / (U n : ℚ) = (2^(n : ℕ) + 1 : ℚ) / (2^(n : ℕ) - 1 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_U_ratio_l835_83552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_l835_83575

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x - (1 - x) * Real.log (1 - x)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x^(1 - x) + (1 - x)^x

theorem max_value_and_inequality :
  (∀ x ∈ Set.Ioo 0 (1/2), f x ≤ 0) ∧
  (f (1/2) = 0) ∧
  (∀ x ∈ Set.Ioo 0 1, g x ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_inequality_l835_83575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_from_square_and_semicircle_l835_83542

/-- The breadth of a rectangle with length 18 cm, given that its perimeter equals that of a square whose side length is the diameter of a semicircle with circumference 25 cm. -/
theorem rectangle_breadth_from_square_and_semicircle : 
  ∃ (s b : ℝ),
    -- Semicircle circumference condition
    π * (s / 2) + s = 25 ∧ 
    -- Perimeter equality condition
    4 * s = 2 * (18 + b) ∧ 
    -- Breadth calculation
    abs (b - 1.454) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_from_square_and_semicircle_l835_83542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S₆_eq_15_l835_83599

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℚ := sorry

/-- The first term of the arithmetic sequence -/
def a₁ : ℚ := sorry

/-- The common difference of the arithmetic sequence -/
def d : ℚ := sorry

/-- The sum of an arithmetic sequence is given by this formula -/
axiom sum_formula (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Given conditions -/
axiom S₃_eq_6 : S 3 = 6
axiom S₉_eq_27 : S 9 = 27

/-- The theorem to prove -/
theorem S₆_eq_15 : S 6 = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S₆_eq_15_l835_83599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l835_83588

open Real

-- Define the arctan function with its properties
noncomputable def my_arctan : ℝ → ℝ := arctan

-- Define the properties of arctan
axiom arctan_range : ∀ x, -π/2 < my_arctan x ∧ my_arctan x < π/2
axiom arctan_increasing : StrictMono my_arctan

-- Define the equation
def equation (x : ℝ) : Prop := my_arctan x = x^2 - 1.6

-- Theorem statement
theorem equation_has_two_solutions : 
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, equation x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_two_solutions_l835_83588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_ac_bd_range_l835_83516

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x + 3| - |x - 1| + 5)

-- State the theorem
theorem f_range_and_ac_bd_range :
  (∀ x, 1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ a b c d : ℝ, a^2 + b^2 = 1 → c^2 + d^2 = 3 →
    -Real.sqrt 3 ≤ a*c + b*d ∧ a*c + b*d ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_and_ac_bd_range_l835_83516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_over_x_l835_83578

theorem derivative_of_sin_over_x (x : ℝ) (h : x ≠ 0) :
  deriv (fun x => Real.sin x / x) x = (x * Real.cos x - Real.sin x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sin_over_x_l835_83578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_root_l835_83562

/-- A polynomial with integer coefficients -/
def polynomial (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

/-- The property that xp(x) = yp(y) for infinitely many pairs of unequal integers -/
def infinitelyManySolutions (p : ℤ → ℤ) : Prop :=
  ∃ S : Set (ℤ × ℤ), Set.Infinite S ∧ (∀ (x y : ℤ), (x, y) ∈ S → x ≠ y ∧ x * p x = y * p y)

theorem polynomial_integer_root (a b c d : ℤ) (ha : a ≠ 0) :
  infinitelyManySolutions (polynomial a b c d) →
  ∃ k : ℤ, polynomial a b c d k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_root_l835_83562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_marked_cells_is_optimal_l835_83544

/-- The smallest number of initially marked cells required to mark all cells on an infinite checkered plane. -/
def smallest_initial_marked_cells (k : ℕ) : ℕ :=
  (Finset.range k).sum (fun i => Int.toNat (Int.ceil ((k - i : ℚ) / 2)))

/-- A turn is valid if the cross of the cell being marked contains at least k marked cells. -/
def valid_turn (k : ℕ) (marked : Set (ℤ × ℤ)) (cell : ℤ × ℤ) : Prop :=
  let cross := {p : ℤ × ℤ | p.1 = cell.1 ∨ p.2 = cell.2}
  (marked ∩ cross).ncard ≥ k

/-- All cells can be marked starting from the initial configuration. -/
def all_cells_markable (k : ℕ) (initial_marked : Set (ℤ × ℤ)) : Prop :=
  ∀ cell : ℤ × ℤ, ∃ sequence : List (ℤ × ℤ),
    (∀ i, i < sequence.length → valid_turn k (initial_marked ∪ (sequence.take i).toFinset) (sequence.get ⟨i, by sorry⟩)) ∧
    cell ∈ initial_marked ∪ sequence.toFinset

/-- The main theorem stating that smallest_initial_marked_cells gives the smallest number of initially marked cells required. -/
theorem smallest_initial_marked_cells_is_optimal (k : ℕ) :
  ∀ N : ℕ, (∃ initial_marked : Set (ℤ × ℤ), initial_marked.ncard = N ∧ all_cells_markable k initial_marked) →
    N ≥ smallest_initial_marked_cells k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_marked_cells_is_optimal_l835_83544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_l835_83598

theorem population_change (initial_population : ℝ) (h : initial_population > 0) :
  let year1 := initial_population * (1 + 0.2)
  let year2 := year1 * (1 - 0.1)
  let year3 := year2 * (1 + 0.3)
  let year4 := year3 * (1 - 0.2)
  let net_change := (year4 - initial_population) / initial_population * 100
  ∃ ε > 0, |net_change - 12| < ε := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_change_l835_83598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l835_83522

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def L (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem circle_line_intersection :
  -- The line l always passes through (1,1)
  (∀ m : ℝ, ∃ x y : ℝ, L m x y ∧ x = 1 ∧ y = 1) ∧
  -- If l intersects C at A and B with |AB| = √17, then l has one of two specific equations
  (∀ m x₁ y₁ x₂ y₂ : ℝ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    L m x₁ y₁ ∧ L m x₂ y₂ ∧ 
    distance x₁ y₁ x₂ y₂ = Real.sqrt 17 →
    (m = Real.sqrt 3 ∨ m = -Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l835_83522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_parts_l835_83577

/-- A triangle is scalene if all its sides have different lengths. -/
def IsScalene (triangle : Set ℝ × Set ℝ) : Prop :=
  ∀ a b c, a ∈ triangle.1 ∧ b ∈ triangle.1 ∧ c ∈ triangle.1 → a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- A line divides a triangle into two parts. -/
def DividesTriangle (line : Set ℝ × Set ℝ) (triangle : Set ℝ × Set ℝ) : Prop :=
  ∃ part1 part2, part1 ∪ part2 = triangle.1 ∧ part1 ∩ part2 ⊆ line.1

/-- Two geometric figures are equal if they have the same area. -/
noncomputable def EqualFigures (fig1 : Set ℝ × Set ℝ) (fig2 : Set ℝ × Set ℝ) : Prop :=
  MeasureTheory.volume fig1.1 = MeasureTheory.volume fig2.1

/-- Theorem: A scalene triangle cannot be divided into two equal parts by any line. -/
theorem scalene_triangle_no_equal_parts (triangle : Set ℝ × Set ℝ) (line : Set ℝ × Set ℝ) :
  IsScalene triangle →
  DividesTriangle line triangle →
  ∃ part1 part2, part1 ∪ part2 = triangle.1 ∧ part1 ∩ part2 ⊆ line.1 →
  ¬(EqualFigures (part1, triangle.2) (part2, triangle.2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_parts_l835_83577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_bound_l835_83568

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The lengths of the sides of the triangle -/
  a : ℝ
  b : ℝ
  c : ℝ
  /-- The triangle inequality holds -/
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  /-- All sides are positive -/
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- The hexagon formed by tangents to the inscribed circle -/
noncomputable def hexagon_perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  let p := (t.a + t.b + t.c) / 2
  2 * (t.a * (p - t.a) / p + t.b * (p - t.b) / p + t.c * (p - t.c) / p)

/-- The perimeter of the original triangle -/
def triangle_perimeter (t : TriangleWithInscribedCircle) : ℝ :=
  t.a + t.b + t.c

/-- The theorem stating that the hexagon perimeter is at most 2/3 of the triangle perimeter -/
theorem hexagon_perimeter_bound (t : TriangleWithInscribedCircle) :
  hexagon_perimeter t ≤ (2/3) * triangle_perimeter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_bound_l835_83568
