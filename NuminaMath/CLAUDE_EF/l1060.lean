import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_second_quadrant_line_equation_given_area_l1060_106056

/-- Represents a line in the form kx - 2y - 3 + k = 0 --/
def Line (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y => k * x - 2 * y - 3 + k = 0

/-- Checks if a point (x, y) is in the second quadrant --/
def InSecondQuadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The range of k for which the line does not pass through the second quadrant --/
theorem line_not_in_second_quadrant (k : ℝ) :
  (∀ x y, Line k x y → ¬InSecondQuadrant x y) ↔ 0 ≤ k ∧ k ≤ 3 :=
sorry

/-- The intersection point of the line with the negative x-axis --/
noncomputable def PointA (k : ℝ) : ℝ × ℝ := ((3 - k) / k, 0)

/-- The intersection point of the line with the negative y-axis --/
noncomputable def PointB (k : ℝ) : ℝ × ℝ := (0, (3 - k) / 2)

/-- The area of the triangle formed by PointA, PointB, and the origin --/
noncomputable def TriangleArea (k : ℝ) : ℝ :=
  abs ((3 - k) / k * (3 - k) / 2) / 2

/-- The equation of the line given the area condition --/
theorem line_equation_given_area :
  ∃ k, k ≠ 0 ∧ Line k (PointA k).1 (PointA k).2 ∧
              Line k (PointB k).1 (PointB k).2 ∧
              TriangleArea k = 4 ∧
              (∀ x y, Line k x y ↔ x + 2 * y + 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_not_in_second_quadrant_line_equation_given_area_l1060_106056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_l1060_106089

-- Define the lines and circle
def line1 (a x y : ℝ) : Prop := x + 2*y = a + 2
def line2 (a x y : ℝ) : Prop := 2*x - y = 2*a - 1
def circle_eq (a x y : ℝ) : Prop := (x - a)^2 + (y - 1)^2 = 16

-- Define the quadrilateral ABCD
def quadrilateral_ABCD (a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (line1 a x y ∨ line2 a x y) ∧ circle_eq a x y ∧ p = (x, y)}

-- Define what it means for a circle to be inscribed in a quadrilateral
def is_inscribed_circle (quad : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ quad → dist p center ≥ radius

-- Theorem statement
theorem inscribed_circle_area (a : ℝ) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = 8*π ∧ 
  ∃ (c : ℝ × ℝ), is_inscribed_circle (quadrilateral_ABCD a) c r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_l1060_106089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1060_106007

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_value (ω : ℝ) (φ : ℝ) :
  ω > 0 ∧
  |φ| ≤ π/2 ∧
  f ω φ (-π/4) = 0 ∧
  (∀ x : ℝ, f ω φ (π/4 - x) = f ω φ (π/4 + x)) ∧
  (∀ x y : ℝ, π/18 < x ∧ x < y ∧ y < 5*π/36 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  ω ≤ 9 ∧ ∃ (ω' : ℝ), ω' = 9 ∧
    ω' > 0 ∧
    ∃ (φ' : ℝ), |φ'| ≤ π/2 ∧
    f ω' φ' (-π/4) = 0 ∧
    (∀ x : ℝ, f ω' φ' (π/4 - x) = f ω' φ' (π/4 + x)) ∧
    (∀ x y : ℝ, π/18 < x ∧ x < y ∧ y < 5*π/36 → f ω' φ' x < f ω' φ' y ∨ f ω' φ' x > f ω' φ' y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l1060_106007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l1060_106030

theorem pure_imaginary_product (b : ℝ) : 
  Complex.I.im = 1 →
  Complex.I * Complex.I = -1 →
  (b : ℂ).im = 0 →
  Complex.I.re = 0 →
  ((1 : ℂ) + b * Complex.I) * ((2 : ℂ) + Complex.I) = Complex.I * ((1 + 2*b) : ℝ) →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_product_l1060_106030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_f_equals_min_f_l1060_106093

/-- The function f(x) = x² + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

/-- The minimum value of f(x) -/
noncomputable def min_f (a : ℝ) : ℝ := -(a^2 / 4)

/-- Theorem stating the equivalence between the minimum value condition and the range of a -/
theorem min_f_f_equals_min_f (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ min_f a) ∧ (∃ y : ℝ, f a (f a y) = min_f a) ↔ a ≥ 2 ∨ a ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_f_f_equals_min_f_l1060_106093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l1060_106079

noncomputable def original_function (x : ℝ) : ℝ := Real.cos (x + 4 * Real.pi / 3)

noncomputable def translated_function (x θ : ℝ) : ℝ := Real.cos (x - θ + 4 * Real.pi / 3)

theorem smallest_translation_for_symmetry :
  ∃ (θ : ℝ), θ > 0 ∧
  (∀ (x : ℝ), translated_function x θ = translated_function (-x) θ) ∧
  (∀ (θ' : ℝ), θ' > 0 →
    (∀ (x : ℝ), translated_function x θ' = translated_function (-x) θ') →
    θ ≤ θ') ∧
  θ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l1060_106079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_odd_implies_symmetric_f_is_symmetric_l1060_106080

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

-- Define the property of being symmetric about the origin
def symmetric_about_origin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem f_is_odd : odd_function f := by
  -- The proof goes here
  sorry

-- Theorem stating that if a function is odd, it is symmetric about the origin
theorem odd_implies_symmetric {g : ℝ → ℝ} (h : odd_function g) : symmetric_about_origin g := by
  -- The proof goes here
  sorry

-- Theorem stating that f is symmetric about the origin
theorem f_is_symmetric : symmetric_about_origin f := by
  -- Apply the previous theorems
  apply odd_implies_symmetric
  exact f_is_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_odd_implies_symmetric_f_is_symmetric_l1060_106080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_exists_l1060_106008

noncomputable def average (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem unique_number_exists : ∃! x : ℝ, average 10 60 x = average 20 40 60 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_exists_l1060_106008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1060_106064

theorem equation_solution : ∃! x : ℝ, 64 = 2 * (16 : ℝ) ^ (x - 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1060_106064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_fraction_of_AD_l1060_106043

-- Define the line segment AD and points B, C, E on it
variable (AD : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (E : ℝ)

-- Define the conditions
axiom B_on_AD : 0 ≤ B ∧ B ≤ AD
axiom C_on_AD : 0 ≤ C ∧ C ≤ AD
axiom E_on_AD : 0 ≤ E ∧ E ≤ AD

-- Length of AB is twice the length of BD
axiom AB_twice_BD : B = 2 * (AD - B)

-- Length of AC is 5 times the length of CD
axiom AC_five_CD : C = 5 * (AD - C)

-- Length of BE is one-third the length of EC
axiom BE_third_EC : E - B = (1/3) * (C - E)

-- Theorem to prove
theorem AE_fraction_of_AD : E / AD = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_fraction_of_AD_l1060_106043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1060_106000

-- Define the curves
noncomputable def C₁ (φ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos φ, 2 * Real.sin φ)

noncomputable def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

def C₃ (α : ℝ) : ℝ → ℝ := fun _ ↦ α

-- Define the theorem
theorem intersection_angle (α : ℝ) : 
  (0 < α) → 
  (α < Real.pi) → 
  (∃ (ρA ρB : ℝ), 
    (C₁ α).1^2 + (C₁ α).2^2 = ρA^2 ∧ 
    C₂ α = ρB ∧ 
    ρA ≠ 0 ∧ 
    ρB ≠ 0 ∧ 
    (ρA - ρB)^2 = 32) → 
  α = 3 * Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angle_l1060_106000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_determines_a_l1060_106038

-- Define the sets P and S
def P : Set ℝ := {1, 5, 10}
def S (a : ℝ) : Set ℝ := {1, 3, a^2 + 1}

-- Define the theorem
theorem union_determines_a (a : ℝ) : 
  (S a ∪ P) = {1, 3, 5, 10} → (a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_determines_a_l1060_106038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1060_106021

-- Define the curve C
noncomputable def curve_C (β : ℝ) : ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β + 1)

-- Define the line l
def line_l : ℝ × ℝ → Prop :=
  fun (x, y) ↦ x + y = 2

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := ((1 + Real.sqrt 7) / 2, (3 - Real.sqrt 7) / 2)
noncomputable def N : ℝ × ℝ := ((1 - Real.sqrt 7) / 2, (3 + Real.sqrt 7) / 2)

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem intersection_distance_product :
  distance P M * distance P N = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l1060_106021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_l1060_106036

/-- Given a wire cut into two pieces of lengths a and b, where a forms an equilateral triangle
    and b forms a square, and the areas of the triangle and square are equal,
    prove that a/b = √(3√3) / 2 -/
theorem wire_cutting (a b : ℝ) (h_positive : a > 0 ∧ b > 0) 
  (h_equal_area : (Real.sqrt 3 / 4) * (a / 3)^2 = (b / 4)^2) : 
  a / b = Real.sqrt (3 * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_l1060_106036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1060_106041

/-- The speed of a train in km/h, given its length and time to cross a point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 120) 
  (h2 : time = 16) : 
  train_speed length time = 27 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1060_106041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1060_106057

noncomputable section

variable (x : ℝ)

def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def b : ℝ × ℝ := (Real.sqrt 3, -1)

def f (x : ℝ) : ℝ := (a x).1 * ((a x).1 + b.1) + (a x).2 * ((a x).2 + b.2)

theorem vector_problem (x₀ : ℝ) 
  (h : ∀ x, f x = f (2 * x₀ - x)) : 
  (∃ (y : ℝ), ‖2 • (a y) - b‖ = 4) ∧ 
  (∃ (z : ℝ), ‖2 • (a z) - b‖ = 0) ∧ 
  (f x₀ = 3 ∨ f x₀ = -1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1060_106057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_4x_mod_9_l1060_106084

theorem remainder_4x_mod_9 (x : ℕ) (h : x % 9 = 5) : (4 * x) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_4x_mod_9_l1060_106084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_puzzle_solution_l1060_106009

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert a digit to its natural number representation -/
def digitToNat (d : Digit) : ℕ := d.val

/-- Convert a two-digit number represented by two digits to a natural number -/
def twoDigitToNat (d1 d2 : Digit) : ℕ := 10 * (digitToNat d1) + (digitToNat d2)

theorem digit_puzzle_solution (E F G H : Digit) 
  (h1 : twoDigitToNat E F + twoDigitToNat G E = twoDigitToNat H E)
  (h2 : twoDigitToNat E F - twoDigitToNat G E = digitToNat F) :
  H = ⟨0, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_puzzle_solution_l1060_106009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comic_book_arrangement_count_l1060_106083

/-- The number of ways to stack comic books of different types, keeping each type grouped together. -/
def comic_book_arrangements (batman : Nat) (xmen : Nat) (calvin_hobbes : Nat) : Nat :=
  (batman.factorial * xmen.factorial * calvin_hobbes.factorial) * Nat.factorial 3

/-- Theorem stating the number of arrangements for the given comic book collection. -/
theorem comic_book_arrangement_count :
  comic_book_arrangements 7 3 5 = 21772800 := by
  sorry

#eval comic_book_arrangements 7 3 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comic_book_arrangement_count_l1060_106083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1060_106022

noncomputable def g (x : ℝ) : ℝ := (2 * Real.cos x^4 + Real.sin x^2) / (2 * Real.sin x^4 + 3 * Real.cos x^2)

theorem g_properties :
  (∀ x : ℝ, g x = 1/2 ↔ (∃ k : ℤ, x = π/4 + k*π/2 ∨ x = π/2 + k*π)) ∧
  (∀ x : ℝ, g x ≥ 3/7) ∧
  (∀ x : ℝ, g x ≤ 2/3) ∧
  (∃ x : ℝ, g x = 3/7) ∧
  (∃ x : ℝ, g x = 2/3) := by
  sorry

#check g_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1060_106022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_equals_target_l1060_106024

/-- Triangle formed by the line 8x + 3y = 48 and the coordinate axes -/
structure Triangle where
  -- Line equation coefficients
  a : ℝ := 8
  b : ℝ := 3
  c : ℝ := 48

  -- Ensure the line forms a triangle with coordinate axes
  h1 : a > 0
  h2 : b > 0
  h3 : c > 0

/-- Sum of altitudes of the triangle -/
noncomputable def sumOfAltitudes (t : Triangle) : ℝ :=
  t.c / t.a + t.c / t.b + t.c / Real.sqrt (t.a^2 + t.b^2)

/-- Theorem stating the sum of altitudes equals 22 + 48/√73 -/
theorem sum_of_altitudes_equals_target (t : Triangle) :
  sumOfAltitudes t = 22 + 48 / Real.sqrt 73 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_equals_target_l1060_106024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_theorem_l1060_106006

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := |x^2 - x*y|

/-- The maximum value of f(x,y) for a fixed y and x in [0,1] -/
noncomputable def g (y : ℝ) : ℝ := ⨆ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1), f x y

/-- The minimum value of g(y) over all real y -/
noncomputable def min_max_value : ℝ := ⨅ (y : ℝ), g y

theorem min_max_theorem : min_max_value = 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_theorem_l1060_106006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eleven_l1060_106090

/-- A five-digit palindrome -/
def FiveDigitPalindrome : Type := { n : ℕ // 10000 ≤ n ∧ n < 100000 ∧ ∃ a b c : ℕ, n = 10001*a + 1010*b + 100*c ∧ a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 }

/-- The set of all five-digit palindromes -/
def AllFiveDigitPalindromes : Finset FiveDigitPalindrome := sorry

/-- The set of five-digit palindromes divisible by 11 -/
def DivisibleByEleven : Finset FiveDigitPalindrome := sorry

/-- The probability of a randomly chosen five-digit palindrome being divisible by 11 -/
noncomputable def ProbabilityDivisibleByEleven : ℚ := (DivisibleByEleven.card : ℚ) / (AllFiveDigitPalindromes.card : ℚ)

theorem probability_divisible_by_eleven : ProbabilityDivisibleByEleven = 1 / 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_eleven_l1060_106090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1060_106020

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- Prove that the distance from the point (1, 2) to the line x/3 + y/4 = 1 is 2/5 -/
theorem distance_point_to_line : 
  let x₀ : ℝ := 1
  let y₀ : ℝ := 2
  let a : ℝ := 4  -- Coefficient of x in standard form
  let b : ℝ := 3  -- Coefficient of y in standard form
  let c : ℝ := -12 -- Constant term in standard form
  distancePointToLine x₀ y₀ a b c = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1060_106020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_proof_l1060_106047

/-- The distance traveled by a truck in yards -/
noncomputable def truck_distance (b t : ℝ) : ℝ :=
  (10 * b / t) + (10 * b / (t + 30))

/-- Theorem stating the correct distance traveled by the truck -/
theorem truck_distance_proof (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  let d₁ := (b / 4) * (120 / t)  -- Distance in first 2 minutes (in feet)
  let d₂ := (b / 4) * (120 / (t + 30))  -- Distance in last 2 minutes (in feet)
  let total_feet := d₁ + d₂
  let yards := total_feet / 3
  yards = truck_distance b t :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_distance_proof_l1060_106047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_shape_l1060_106005

/-- Given a frame with four rods of length l, negligible weight, hinged corners, and a liquid film
    with surface tension σ, when suspended by one corner with weight G attached to the opposite corner,
    the shape of the frame is characterized by the angle α. -/
theorem frame_shape (l σ G : ℝ) (hl : l > 0) (hσ : σ > 0) (hG : G > 0) :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧
    Real.cos α = (G / (8 * l * σ)) + Real.sqrt ((G / (8 * l * σ))^2 + 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_shape_l1060_106005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_p_value_l1060_106004

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem parabola_p_value (p : ℝ) (x y : ℝ) :
  p > 0 →
  parabola p x y →
  distance x y (focus p).1 (focus p).2 = 12 →
  distance x y 0 0 = 9 →
  p = 6 := by
  sorry

#check parabola_p_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_p_value_l1060_106004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diego_payment_calculation_l1060_106013

/-- Represents the payment received by Diego for painting murals. -/
def diego_payment : ℝ := sorry

/-- Represents the payment received by Celina for painting murals. -/
def celina_payment : ℝ := sorry

/-- The total number of murals painted by both artists. -/
def total_murals : ℕ := 50

/-- The total amount paid to both artists. -/
def total_payment : ℝ := 50000

theorem diego_payment_calculation :
  celina_payment = 1000 + 4 * diego_payment ∧
  celina_payment + diego_payment = total_payment →
  diego_payment = 9800 := by
  sorry

#check diego_payment_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diego_payment_calculation_l1060_106013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1060_106015

theorem triangle_cosine_problem (X Y Z : ℝ) :
  -- Conditions
  0 < X ∧ 0 < Y ∧ 0 < Z ∧  -- Angles are positive
  X + Y + Z = Real.pi ∧  -- Sum of angles in a triangle
  Real.sin X = 4/5 ∧
  Real.cos Y = 3/5 →
  -- Conclusion
  Real.cos Z = 7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l1060_106015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l1060_106027

theorem rectangle_length_fraction_of_circle_radius 
  (square_area : ℝ) 
  (rectangle_area : ℝ) 
  (rectangle_breadth : ℝ) 
  (h1 : square_area = 2500) 
  (h2 : rectangle_area = 200) 
  (h3 : rectangle_breadth = 10) : 
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

#check rectangle_length_fraction_of_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_fraction_of_circle_radius_l1060_106027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l1060_106073

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem translation_symmetry (φ : ℝ) (h1 : 0 < φ) (h2 : φ ≤ Real.pi / 2) :
  (∀ x, f (x + φ) = f (-x + φ)) → φ = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_symmetry_l1060_106073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_series_coefficients_l1060_106002

/-- Taylor series expansion of e^(bx) cos(cx) -/
noncomputable def taylor_series (b c : ℝ) (x : ℝ) : ℝ := Real.exp (b * x) * Real.cos (c * x)

/-- Coefficients of the Taylor series expansion -/
noncomputable def a (n : ℕ) (b c : ℝ) : ℝ :=
  (1 / n.factorial : ℝ) * Real.cos (n * Real.arctan (c / b)) * (Real.sqrt (b^2 + c^2))^n

/-- Statement: Either all coefficients are non-zero or infinitely many are zero -/
theorem taylor_series_coefficients (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (∀ n, a n b c ≠ 0) ∨ (∀ m : ℕ, ∃ n ≥ m, a n b c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taylor_series_coefficients_l1060_106002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1060_106069

theorem binomial_expansion_coefficient (a : ℝ) : 
  (∃ c : ℝ, c = 84 ∧ 
    c = (Nat.choose 7 2) * 2^2 * a^5 ∧ 
    c * (1 / x^3) = Finset.sum (Finset.range 8) (λ k ↦ 
      (Nat.choose 7 k) * (2^k) * ((-a)^(7-k)) * (1/x^(7-k)))) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1060_106069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_function_decomposition_l1060_106062

-- Define the properties of even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem even_odd_function_decomposition 
  (f g : ℝ → ℝ) 
  (hf : IsEven f) 
  (hg : IsOdd g) 
  (h : ∀ x, x ≠ 1 → f x + g x = 1 / (x - 1)) : 
  (∀ x, x ≠ 1 ∧ x ≠ -1 → f x = 1 / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 ∧ x ≠ -1 → g x = x / (x^2 - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_function_decomposition_l1060_106062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_10_l1060_106086

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- State the theorem
theorem f_cos_10 :
  (∀ x ∈ Set.Icc (-π/2) (π/2), f (Real.sin x) = -2*x + 1) →
  f (Real.cos 10) = 7*π - 19 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_cos_10_l1060_106086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_swap_tokens_l1060_106050

structure Token where
  can_move_adjacent_free_cell : Bool
  can_move_vertical_or_horizontal : Bool

structure State where
  white_tokens_on_bottom_row : Bool
  black_tokens_on_top_row : Bool
  white_tokens_on_top_row : Bool
  black_tokens_on_bottom_row : Bool

def min_moves_to_swap (n : Nat) : Nat :=
  n * (2 * n - 1)

theorem min_moves_to_swap_tokens (n : Nat) (h : n = 7 ∨ n = 8) :
  let chessboard := n
  let white_tokens := n
  let black_tokens := n
  (∀ token : Token, token.can_move_adjacent_free_cell = true) →
  (∀ token : Token, token.can_move_vertical_or_horizontal = true) →
  (∃ initial_state : State,
    initial_state.white_tokens_on_bottom_row = true ∧
    initial_state.black_tokens_on_top_row = true) →
  (∃ final_state : State,
    final_state.white_tokens_on_top_row = true ∧
    final_state.black_tokens_on_bottom_row = true) →
  min_moves_to_swap n = n * (2 * n - 1) :=
by
  intro chessboard white_tokens black_tokens
  intro h_adjacent h_vertical_horizontal h_initial h_final
  cases h with
  | inl h_seven =>
    rw [h_seven]
    rfl
  | inr h_eight =>
    rw [h_eight]
    rfl

#check min_moves_to_swap_tokens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_to_swap_tokens_l1060_106050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1060_106095

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * (x - 1) - y - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 9

-- Define the center of the circle
def center : ℝ × ℝ := (-1, -2)

-- Define the intersection points M and N as variables
variable (M N : ℝ × ℝ)

-- Define the dot product of CM and CN
def dot_product (p q : ℝ × ℝ) : ℝ :=
  (p.1 - center.1) * (q.1 - center.1) + (p.2 - center.2) * (q.2 - center.2)

theorem min_dot_product :
  ∃ (m : ℝ),
    (∀ x y, line_l m x y → circle_C x y → (x, y) = M ∨ (x, y) = N) →
    (∀ x y, circle_C x y → dot_product (x, y) (x, y) = 9) →
    ∀ p q, circle_C p.1 p.2 → circle_C q.1 q.2 → dot_product p q ≥ -9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1060_106095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_fuel_calculation_l1060_106044

/-- Represents the total amount of fuel for a trip -/
def total_fuel : ℝ → ℝ := sorry

/-- Represents the amount of fuel used in each third of the trip -/
def fuel_per_third : ℝ → ℝ × ℝ × ℝ := sorry

theorem road_trip_fuel_calculation (initial_third : ℝ) :
  initial_third = 30 →
  (let (first, second, third) := fuel_per_third (total_fuel initial_third)
   first = initial_third ∧
   second = (1 / 3) * total_fuel initial_third ∧
   third = (1 / 2) * second ∧
   first + second + third = total_fuel initial_third) →
  total_fuel initial_third = 75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_fuel_calculation_l1060_106044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merge_lightest_minimizes_dangerous_level_l1060_106026

/-- A merge sequence for a list of weights -/
def MergeSequence (weights : List ℝ) := List (Nat × Nat)

/-- Check if a merge is dangerous (one weight > 2.020 times the other) -/
noncomputable def isDangerousMerge (w1 w2 : ℝ) : Bool :=
  w1 > 2.020 * w2 || w2 > 2.020 * w1

/-- Count the number of dangerous merges in a sequence -/
def dangerousLevel (weights : List ℝ) (seq : MergeSequence weights) : Nat :=
  sorry

/-- Merge the two lightest weights at each step -/
def mergeLightest (weights : List ℝ) : MergeSequence weights :=
  sorry

/-- The main theorem: merging lightest weights minimizes dangerous level -/
theorem merge_lightest_minimizes_dangerous_level (weights : List ℝ) :
  ∀ (seq : MergeSequence weights),
    dangerousLevel weights (mergeLightest weights) ≤ dangerousLevel weights seq := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_merge_lightest_minimizes_dangerous_level_l1060_106026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_cost_example_l1060_106075

/-- The total cost of covering a rectangular floor with square carpet tiles. -/
noncomputable def total_carpet_cost (floor_length floor_width carpet_side cost_per_tile : ℝ) : ℝ :=
  let floor_area := floor_length * floor_width
  let tile_area := carpet_side * carpet_side
  let num_tiles := floor_area / tile_area
  num_tiles * cost_per_tile

/-- Theorem stating that the total cost for covering a 6m by 10m floor with 2m by 2m carpet squares at $15 each is $225. -/
theorem carpet_cost_example : total_carpet_cost 6 10 2 15 = 225 := by
  -- Unfold the definition of total_carpet_cost
  unfold total_carpet_cost
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_cost_example_l1060_106075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_remaining_money_l1060_106010

-- Define the initial amount Jesse received
def initial_amount : ℚ := 500

-- Define the cost of one novel
def novel_cost : ℚ := 13

-- Define the number of novels bought
def novel_count : ℕ := 10

-- Define the discount rate for novels
def novel_discount_rate : ℚ := 20 / 100

-- Define the tax rate for lunch
def lunch_tax_rate : ℚ := 12 / 100

-- Define the jacket discount rate
def jacket_discount_rate : ℚ := 30 / 100

-- Define the function to calculate the total cost of novels after discount
def novels_total_cost (cost : ℚ) (count : ℕ) (discount : ℚ) : ℚ :=
  cost * (count : ℚ) * (1 - discount)

-- Define the function to calculate the lunch cost including tax
def lunch_cost (novel_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  novel_cost * 3 * (1 + tax_rate)

-- Define the function to calculate the jacket cost after discount
def jacket_cost (bookstore_spend : ℚ) (discount : ℚ) : ℚ :=
  bookstore_spend * 2 * (1 - discount)

-- Theorem to prove
theorem jesse_remaining_money :
  initial_amount - 
  (novels_total_cost novel_cost novel_count novel_discount_rate +
   lunch_cost novel_cost lunch_tax_rate +
   jacket_cost (novel_cost * (novel_count : ℚ)) jacket_discount_rate) = 170.32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jesse_remaining_money_l1060_106010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_range_l1060_106081

/-- Represents a hyperbola with parameter m -/
structure Hyperbola (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 - y^2/m^2 = 1)
  (asymptotes : ∀ x : ℝ, {y : ℝ | y = m*x ∨ y = -m*x})
  (m_pos : m > 0)

/-- Represents a circle with center (0, -2) and radius 1 -/
structure Circle :=
  (eq : ∀ x y : ℝ, x^2 + (y+2)^2 = 1)
  (center : ℝ × ℝ := (0, -2))
  (radius : ℝ := 1)

/-- Focal length of a hyperbola -/
noncomputable def focal_length (m : ℝ) : ℝ := 2 * Real.sqrt (1 + m^2)

/-- Asymptotes do not intersect the circle -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, (y = m*x ∨ y = -m*x) → x^2 + (y+2)^2 ≠ 1

/-- Main theorem -/
theorem hyperbola_focal_length_range (m : ℝ) (h : Hyperbola m) 
  (ni : no_intersection m) : 
  2 < focal_length m ∧ focal_length m < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_range_l1060_106081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_special_triangle_l1060_106032

/-- The radius of the circumscribed circle of a triangle -/
noncomputable def circumradius (a b c : ℝ) (A : ℝ) : ℝ :=
  b / (2 * Real.sin A)

/-- Theorem: The radius of the circumscribed circle of a triangle with sides 4 and 2, and the angle between them 60°, is 2 -/
theorem circumradius_special_triangle : 
  circumradius 4 2 (2 * Real.sqrt 3) (π / 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_special_triangle_l1060_106032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_of_triangle_l1060_106037

-- Define the triangle ABC
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (0, 2)

-- Define the condition AB = AC
def AB_eq_AC : dist A B = dist A C := sorry

-- Define the Euler line
def euler_line (p : ℝ × ℝ) : Prop := 2 * p.1 + 4 * p.2 - 3 = 0

-- Theorem statement
theorem euler_line_of_triangle :
  euler_line A ∧ euler_line B ∧ euler_line C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_line_of_triangle_l1060_106037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1060_106085

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

theorem geometric_series_sum :
  let a : ℝ := 2
  let r : ℝ := 3
  let n : ℕ := 6
  geometric_sum a r n = 728 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l1060_106085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1060_106099

/-- Two functions f and g from ℤ to [0,∞) that are zero except for finitely many integers -/
def FinitelyNonzero (f g : ℤ → ℝ) : Prop :=
  ∃ S : Finset ℤ, ∀ n : ℤ, n ∉ S → f n = 0 ∧ g n = 0

/-- The function h defined as the maximum product of f and g -/
noncomputable def h (f g : ℤ → ℝ) : ℤ → ℝ :=
  λ n => ⨆ k : ℤ, f (n - k) * g k

/-- The statement of the inequality to be proved -/
theorem inequality_theorem (f g : ℤ → ℝ) (hf : ∀ n, f n ≥ 0) (hg : ∀ n, g n ≥ 0)
    (h_finite : FinitelyNonzero f g) (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : 1 / p + 1 / q = 1) :
    ∑' n, h f g n ≥ (∑' n, (f n) ^ p) ^ (1 / p) * (∑' n, (g n) ^ q) ^ (1 / q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_theorem_l1060_106099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_25_rubles_l1060_106017

theorem exchange_25_rubles :
  ∃ (solutions : List (Nat × Nat × Nat)),
    (∀ (x y z : Nat), (x, y, z) ∈ solutions →
      x + y + z = 11 ∧ x + 3*y + 5*z = 25) ∧
    solutions.length = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_25_rubles_l1060_106017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_l1060_106051

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1}

def f (x : ℤ) : ℤ := x^2

theorem f_is_function : 
  (∀ x, x ∈ A → f x ∈ B) ∧ 
  (∀ x y, x ∈ A → y ∈ A → x = y → f x = f y) :=
by
  sorry

#check f_is_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_function_l1060_106051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_game_probability_l1060_106096

/-- Represents the probability of winning for the first player in a coin tossing game -/
noncomputable def probability_first_player_wins (p1 p2 : ℝ) : ℝ :=
  p1 / (1 - (1 - p1) * (1 - p2))

/-- Theorem stating that in a specific coin tossing game, the probability of the first player winning is 1/2 -/
theorem coin_toss_game_probability :
  probability_first_player_wins (1/4 : ℝ) (1/3 : ℝ) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_game_probability_l1060_106096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_order_circle_X_radius_circle_Y_circumference_circle_Z_area_l1060_106071

-- Define the circles
noncomputable def circle_X : ℝ := Real.pi
noncomputable def circle_Y : ℝ := 4
noncomputable def circle_Z : ℝ := 3

-- Define the theorem
theorem circle_radius_order :
  circle_Z < circle_X ∧ circle_X < circle_Y := by
  sorry

-- Define the conditions
theorem circle_X_radius : circle_X = Real.pi := by
  rfl

theorem circle_Y_circumference : 2 * Real.pi * circle_Y = 8 * Real.pi := by
  sorry

theorem circle_Z_area : Real.pi * circle_Z^2 = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_order_circle_X_radius_circle_Y_circumference_circle_Z_area_l1060_106071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l1060_106003

/-- Calculates the amount owed after compound interest --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (compounds_per_year : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate / compounds_per_year) ^ (compounds_per_year * years)

/-- The difference in amount owed between monthly and semi-annual compounding --/
noncomputable def interest_difference : ℝ :=
  compound_interest 8000 0.12 12 3 - compound_interest 8000 0.12 2 3

theorem interest_difference_approx :
  abs (interest_difference - 58.08) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_approx_l1060_106003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_theorem_l1060_106055

noncomputable def initial_milk_volume : ℝ := 60

def replacement_volumes : List ℝ := [5, 7, 9, 4]

noncomputable def milk_after_replacement (current_milk : ℝ) (replacement_volume : ℝ) : ℝ :=
  current_milk * (1 - replacement_volume / initial_milk_volume)

noncomputable def final_milk_volume : ℝ :=
  replacement_volumes.foldl milk_after_replacement initial_milk_volume

theorem milk_dilution_theorem :
  |final_milk_volume - 38.525| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_dilution_theorem_l1060_106055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_nine_l1060_106049

theorem probability_factor_less_than_nine (n : ℕ) (h : n = 120) :
  (Finset.filter (λ x : ℕ ↦ x < 9) (Nat.divisors n)).card /
  (Nat.divisors n).card = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_nine_l1060_106049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1060_106028

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the perpendicular relation between a line and a plane
def perpendicular_line_plane (l : Line) (α : Plane) : Prop :=
  sorry

-- Define the perpendicular relation between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry

-- Define the subset relation between a line and a plane
def line_subset_plane (a : Line) (α : Plane) : Prop :=
  sorry

theorem perpendicular_necessary_not_sufficient 
  (a l : Line) (α : Plane) (h : line_subset_plane a α) :
  (perpendicular_line_plane l α → perpendicular_lines l a) ∧
  ¬(perpendicular_lines l a → perpendicular_line_plane l α) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_necessary_not_sufficient_l1060_106028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_acute_angles_in_polygon_l1060_106023

/-- A convex n-gon in a 2D plane -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : Convex ℝ (Set.range vertices)

/-- Angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- An angle is non-acute if it's greater than or equal to π/2 -/
def isNonAcute (θ : ℝ) : Prop := θ ≥ Real.pi / 2

/-- Main theorem -/
theorem non_acute_angles_in_polygon (n : ℕ) (poly : ConvexPolygon n) (O : ℝ × ℝ) 
  (h_inside : O ∈ interior (Set.range poly.vertices)) :
  ∃ (S : Finset (Fin n × Fin n)),
    S.card ≥ n - 1 ∧ 
    ∀ (i j : Fin n), (i, j) ∈ S → isNonAcute (angle (poly.vertices i) O (poly.vertices j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_acute_angles_in_polygon_l1060_106023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_averages_l1060_106082

/-- Represents a stock with its dividend yield and quote percentage -/
structure Stock where
  dividendYield : ℝ
  quotePercentage : ℝ

/-- Calculates the average of a list of real numbers -/
noncomputable def average (list : List ℝ) : ℝ :=
  list.sum / list.length

/-- Theorem stating the combined average dividend yield and quote percentage for given stocks -/
theorem stock_averages (stockA stockB stockC : Stock)
  (hA : stockA = { dividendYield := 0.15, quotePercentage := 0.12 })
  (hB : stockB = { dividendYield := 0.10, quotePercentage := -0.05 })
  (hC : stockC = { dividendYield := 0.08, quotePercentage := 0.18 }) :
  let stocks := [stockA, stockB, stockC]
  let avgDividendYield := average (stocks.map Stock.dividendYield)
  let avgQuotePercentage := average (stocks.map Stock.quotePercentage)
  avgDividendYield = 0.11 ∧ |avgQuotePercentage - 0.0833| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_averages_l1060_106082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l1060_106034

-- Define a monic polynomial of degree 4
def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + p 0

-- Define the polynomial p
noncomputable def p : ℝ → ℝ := sorry

-- State the theorem
theorem polynomial_sum_theorem (h_monic : is_monic_degree_4 p)
  (h1 : p 1 = 21) (h2 : p 2 = 42) (h3 : p 3 = 63) :
  p 0 + p 4 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_theorem_l1060_106034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_shape_example_l1060_106040

/-- A function representing a quadratic equation -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The coefficient of the quadratic term in a quadratic function -/
def QuadraticCoefficient (f : ℝ → ℝ) : ℝ := sorry

/-- Two quadratic functions have the same shape if and only if
    the absolute values of their quadratic coefficients are equal -/
def SameShape (f g : ℝ → ℝ) : Prop :=
  |QuadraticCoefficient f| = |QuadraticCoefficient g|

theorem same_shape_example :
  let f := QuadraticFunction (-5) 0 2
  let g := QuadraticFunction 5 0 0
  SameShape f g := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_shape_example_l1060_106040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_cube_root_five_hundred_l1060_106078

-- Define the given approximations
def cube_root_half : ℝ := 0.7937
def cube_root_five : ℝ := 1.7100

-- Define the approximation we want to prove
def cube_root_five_hundred : ℝ := 7.937

-- Define a tolerance for approximations
def tolerance : ℝ := 0.0001

-- Theorem statement
theorem approx_cube_root_five_hundred :
  (|cube_root_half^3 - 0.5| < tolerance) ∧
  (|cube_root_five^3 - 5| < tolerance) →
  |cube_root_five_hundred^3 - 500| < tolerance := by
  sorry

#check approx_cube_root_five_hundred

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approx_cube_root_five_hundred_l1060_106078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_fourth_powers_l1060_106054

theorem prime_sum_of_fourth_powers (X Y : ℤ) :
  Nat.Prime (Int.natAbs (X^4 + 4*Y^4)) ↔ (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_of_fourth_powers_l1060_106054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shaded_region_l1060_106019

-- Define the rectangle MNPQ
def Rectangle (M N P Q : ℝ × ℝ) : Prop :=
  M.2 = N.2 ∧ P.2 = Q.2 ∧ M.1 = P.1 ∧ N.1 = Q.1

-- Define the condition that MN is twice MP
def MNTwiceMP (M N P : ℝ × ℝ) : Prop :=
  N.1 - M.1 = 2 * (P.2 - M.2)

-- Define the length of MP
def MPLength (M P : ℝ × ℝ) : Prop :=
  Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2) = 2

-- Define the extensions MS and RQ
def Extensions (M Q S R : ℝ × ℝ) : Prop :=
  Real.sqrt ((M.1 - S.1)^2 + (M.2 - S.2)^2) = 2 ∧
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 2

-- Define the area of the trapezoid
noncomputable def TrapezoidArea (A B C D : ℝ × ℝ) : ℝ :=
  1/2 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2) -
             (A.2 * B.1 + B.2 * C.1 + C.2 * D.1 + D.2 * A.1))

theorem area_of_shaded_region
  (M N P Q S R : ℝ × ℝ)
  (h1 : Rectangle M N P Q)
  (h2 : MNTwiceMP M N P)
  (h3 : MPLength M P)
  (h4 : Extensions M Q S R) :
  ∃ A B C D : ℝ × ℝ, TrapezoidArea A B C D = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_shaded_region_l1060_106019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1060_106016

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 10)^2 / 15^2 = 1

-- Define the focus coordinates
noncomputable def focus : ℝ × ℝ := (5 + Real.sqrt 274, 10)

-- Theorem statement
theorem hyperbola_focus :
  ∀ x y : ℝ, hyperbola x y → 
  ∀ f : ℝ × ℝ, (hyperbola f.1 f.2 ∧ f.1 ≠ focus.1 ∧ f.2 ≠ focus.2) → 
  f.1 < focus.1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_l1060_106016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_location_l1060_106091

/-- Given the asymptotes and a point on a hyperbola, prove the location of its foci -/
theorem hyperbola_foci_location 
  (a b : ℝ) 
  (x₁ y₁ : ℝ) 
  (asymptotes : Set (ℝ × ℝ))
  (hyperbola : Set (ℝ × ℝ))
  (tangent_line : Set (ℝ × ℝ))
  (foci : Set (ℝ × ℝ))
  (h_asymptotes : ∀ (x y : ℝ), (y = (b/a) * x ∨ y = -(b/a) * x) → (x, y) ∈ asymptotes)
  (h_point : (x₁, y₁) ∈ hyperbola)
  (h_center : (0, 0) ∈ hyperbola)
  (h_tangent : ∀ (x y : ℝ), (x * x₁ / a^2 - y * y₁ / b^2 = 1) → (x, y) ∈ tangent_line)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0) :
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 
  ((c, 0) ∈ foci ∧ (-c, 0) ∈ foci) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_location_l1060_106091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1060_106046

-- Define the functions f and g
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

noncomputable def g (x : ℝ) : ℝ := 1 - Real.rpow 2 x

-- State the theorem
theorem problem_solution :
  ∀ (a b : ℝ), a ≠ 0 →
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x, f a b x + 2 * x = f a b (-x) + 2 * (-x)) →
  (∃! x, x ∈ Set.Icc 0 1 ∧ (x - 1)^2 + g x = 0) ∧
  (∀ m n : ℝ, (m - 1)^2 = g n → n ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1060_106046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_transformed_curve_l1060_106011

/-- Given a function g: ℝ → ℝ, this function returns the area between y = g(x) and the x-axis -/
noncomputable def areaUnderCurve (g : ℝ → ℝ) : ℝ := sorry

theorem area_under_transformed_curve 
  (g : ℝ → ℝ) 
  (h : areaUnderCurve g = 12) : 
  areaUnderCurve (fun x ↦ 4 * g (2 * x + 3)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_transformed_curve_l1060_106011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_bound_l1060_106077

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- State the theorem
theorem ellipse_foci_product_bound 
  (F1x F1y F2x F2y : ℝ) -- Coordinates of foci
  (Mx My : ℝ) -- Coordinates of point M on the ellipse
  (h1 : Ellipse Mx My) -- M is on the ellipse
  (h2 : ∃ c, ∀ x y, Ellipse x y → distance x y F1x F1y + distance x y F2x F2y = c) -- F1 and F2 are foci
  : distance Mx My F1x F1y * distance Mx My F2x F2y ≤ 9 := by
  sorry

#check ellipse_foci_product_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_product_bound_l1060_106077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_polygon_circles_l1060_106059

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define 90° rotation invariance
def Rotation90Invariant (M : Set (ℝ × ℝ)) (center : ℝ × ℝ) : Prop := sorry

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) := sorry

-- Define circumscribed circle
def CircumscribedCircle (M : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop := sorry

-- Define inscribed circle
def InscribedCircle (M : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop := sorry

theorem rotation_invariant_polygon_circles 
  (M : Set (ℝ × ℝ)) 
  (center : ℝ × ℝ) 
  (h1 : ConvexPolygon M) 
  (h2 : Rotation90Invariant M center) :
  ∃ (R r : ℝ), 
    R / r = Real.sqrt 2 ∧
    CircumscribedCircle M (Circle center R) ∧
    InscribedCircle M (Circle center r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_invariant_polygon_circles_l1060_106059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_l1060_106035

/-- The number of pages in a novel given the fraction read and total pages read --/
theorem novel_pages (fraction_yesterday fraction_today : ℚ) (pages_read : ℕ) : 
  fraction_yesterday = 3/10 →
  fraction_today = 4/10 →
  (fraction_yesterday + fraction_today) * pages_read = 140 →
  pages_read = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_pages_l1060_106035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_gcd_condition_l1060_106001

/-- Sequence a_n defined recursively --/
def a (p q : ℕ) : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => p * a p q (n+1) + q * a p q n

theorem sequence_gcd_condition (p q : ℕ) (hp : p > 0) (hq : q > 0) :
  (∀ m n : ℕ, m > 0 → n > 0 → Nat.gcd (a p q m) (a p q n) = a p q (Nat.gcd m n)) ↔ p = 1 := by
  sorry

#check sequence_gcd_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_gcd_condition_l1060_106001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_rate_is_twenty_percent_l1060_106061

/-- Represents the total number of products --/
def total_products : ℕ := 10

/-- Represents the number of items selected for inspection --/
def items_inspected : ℕ := 2

/-- Represents the probability of selecting exactly one defective item --/
def prob_one_defective : ℚ := 16/45

/-- Represents the maximum allowed defective rate --/
def max_defective_rate : ℚ := 2/5

/-- Represents the actual number of defective items --/
def num_defective : ℕ := 2

theorem defective_rate_is_twenty_percent :
  (num_defective : ℚ) / total_products = 1/5 ∧
  (num_defective : ℕ) ≤ (max_defective_rate * total_products).floor ∧
  (num_defective.choose 1 * (total_products - num_defective).choose 1 : ℚ) /
    total_products.choose items_inspected = prob_one_defective :=
by
  sorry

#eval (num_defective : ℚ) / total_products
#eval (max_defective_rate * total_products).floor
#eval (num_defective.choose 1 * (total_products - num_defective).choose 1 : ℚ) /
  total_products.choose items_inspected

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_rate_is_twenty_percent_l1060_106061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_sequence_property_l1060_106092

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

def appears_in_sequence (k : ℕ) (f : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, ⌊f n⌋ = k

def appears_twice_in_sequence (k : ℕ) (f : ℕ → ℝ) : Prop :=
  ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ ⌊f n₁⌋ = k ∧ ⌊f n₂⌋ = k ∧
  ∀ n : ℕ, n ≠ n₁ ∧ n ≠ n₂ → ⌊f n⌋ ≠ k

theorem golden_ratio_sequence_property :
  ∀ k : ℕ, k > 0 →
  (appears_in_sequence k (λ n => n * φ) ↔ appears_twice_in_sequence k (λ n => n / φ)) := by
  sorry

#check golden_ratio_sequence_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_sequence_property_l1060_106092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_input_is_input_statement_l1060_106088

-- Define the possible statement types
inductive StatementType
| Input
| Output
| Other

-- Define the programming languages
inductive ProgrammingLanguage
| PRINT
| INPUT
| IF
| LET

-- Define a function that maps languages to their statement types
def statementType : ProgrammingLanguage → StatementType
| ProgrammingLanguage.PRINT => StatementType.Output
| ProgrammingLanguage.INPUT => StatementType.Input
| _ => StatementType.Other

-- Theorem to prove
theorem input_is_input_statement :
  statementType ProgrammingLanguage.INPUT = StatementType.Input :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_input_is_input_statement_l1060_106088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_problem_l1060_106012

noncomputable def average_of_multiples (x : ℝ) : ℝ := (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7

def median_of_multiples (n : ℕ) : ℝ := 2 * n

theorem multiples_problem (x : ℝ) (h1 : x > 0) :
  let a := average_of_multiples x
  let b := median_of_multiples 12
  a^2 - b^2 = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_problem_l1060_106012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_theorem_l1060_106018

noncomputable def stock_price_increase (opening_price closing_price : ℝ) : ℝ :=
  (closing_price - opening_price) / opening_price * 100

noncomputable def average_percent_increase (increases : List ℝ) : ℝ :=
  (increases.sum) / (increases.length : ℝ)

theorem stock_price_increase_theorem (ε : ℝ) (ε_pos : ε > 0) : 
  ∀ (opening_prices closing_prices : List ℝ),
    opening_prices = [20, 30, 40] →
    closing_prices = [24, 36, 44] →
    |average_percent_increase (List.zipWith stock_price_increase opening_prices closing_prices) - 50/3| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_price_increase_theorem_l1060_106018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_values_l1060_106063

theorem ordering_of_special_values :
  let a : ℝ := (1.01 : ℝ) ^ (-100 : ℤ)
  let b : ℝ := Real.sin (π / 10)
  let c : ℝ := 1 / π
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_values_l1060_106063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_score_l1060_106014

-- Define the probability of scoring a basket
noncomputable def prob_score : ℝ := 3/4

-- Define the probability of missing a basket
noncomputable def prob_miss : ℝ := 1/4

-- State the theorem
theorem prob_one_score : 
  prob_score = 3 * prob_miss → -- Given condition
  prob_score + prob_miss = 1 → -- Probability axiom
  prob_score * (1 - prob_score) = 3/16 := by -- P(X=1) = p(1-p)
  intro h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_one_score_l1060_106014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1060_106039

-- Define the function f(x) = √x + 1
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x + 1

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = (1 : ℝ) / 2 := by
  -- Calculate the derivative
  have h1 : deriv f = fun x => 1 / (2 * Real.sqrt x) := by
    sorry -- Proof of the derivative calculation
  
  -- Evaluate the derivative at x = 1
  have h2 : deriv f 1 = 1 / (2 * Real.sqrt 1) := by
    sorry -- Proof of the evaluation at x = 1
  
  -- Simplify the result
  calc
    deriv f 1 = 1 / (2 * Real.sqrt 1) := h2
    _         = 1 / (2 * 1)           := by sorry -- Proof that √1 = 1
    _         = 1 / 2                 := by sorry -- Simplification

  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1060_106039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l1060_106087

noncomputable section

-- Define the curve C
def curve_C (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sin α)

-- Define the line l
def line_l (θ : ℝ) : ℝ := Real.sqrt 2 / Real.sin (θ - Real.pi/4)

-- Define point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), 
    A = curve_C θ₁ ∧ 
    B = curve_C θ₂ ∧ 
    (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2 = line_l θ₁^2 ∧
    (B.1 - point_P.1)^2 + (B.2 - point_P.2)^2 = line_l θ₂^2

-- Theorem statement
theorem distance_sum_theorem (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) + 
  Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) = 
  18 * Real.sqrt 2 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_theorem_l1060_106087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_millisecond_is_0_023_l1060_106058

/-- Represents the cost structure and run details of a computer program -/
structure ProgramCost where
  overhead : ℚ  -- Operating-system overhead cost
  tapeMounting : ℚ  -- Cost for mounting a data tape
  runTime : ℚ  -- Run time in seconds
  totalCost : ℚ  -- Total cost for the run

/-- Calculates the cost per millisecond given the program cost structure -/
def costPerMillisecond (p : ProgramCost) : ℚ :=
  (p.totalCost - p.overhead - p.tapeMounting) / (p.runTime * 1000)

/-- Theorem stating that the cost per millisecond is 0.023 for the given program -/
theorem cost_per_millisecond_is_0_023 :
  let p : ProgramCost := {
    overhead := 107/100,
    tapeMounting := 535/100,
    runTime := 3/2,
    totalCost := 4092/100
  }
  costPerMillisecond p = 23/1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_millisecond_is_0_023_l1060_106058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fifth_25_l1060_106098

theorem log_one_fifth_25 : Real.log 25 / Real.log (1/5) = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_one_fifth_25_l1060_106098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l1060_106053

/-- Represents a cube with six faces --/
structure Cube where
  faces : Fin 6 → ℕ
  valid_faces : ∀ i, faces i ∈ ({1, 3, 9, 27, 81, 243} : Set ℕ)

/-- Represents a stack of four cubes --/
def CubeStack := Fin 4 → Cube

/-- Calculates the sum of visible faces in a stack of cubes --/
def visible_sum (stack : CubeStack) : ℕ :=
  sorry

/-- Theorem stating that the maximum visible sum is 1444 --/
theorem max_visible_sum :
  ∃ (stack : CubeStack), 
    (∀ other_stack : CubeStack, visible_sum other_stack ≤ visible_sum stack) ∧
    visible_sum stack = 1444 := by
  sorry

#check max_visible_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_visible_sum_l1060_106053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_f_lower_bound_l1060_106052

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (1/2) * (x - a) * Real.log x - Real.log (Real.log x)

-- Theorem for the tangent line equation
theorem tangent_line_at_e (a : ℝ) :
  a = Real.exp 1 →
  ∃ m b : ℝ, m = 1/2 - 1/(Real.exp 1) ∧ b = 1 - (Real.exp 1)/2 ∧
  ∀ x : ℝ, (f a x - f a (Real.exp 1)) = m * (x - Real.exp 1) := by
  sorry

-- Theorem for the range of a
theorem f_lower_bound (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x ≥ 1 - Real.log 2) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_f_lower_bound_l1060_106052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_not_binomial_l1060_106065

/-- A probability distribution representing the number of trials until first success -/
def GeometricDistribution (p : ℝ) (k : ℕ) : ℝ := (1 - p) ^ (k - 1) * p

/-- The binomial distribution -/
def BinomialDistribution (n : ℕ) (p : ℝ) (k : ℕ) : ℝ := 
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- Theorem stating that the geometric distribution is not a binomial distribution -/
theorem geometric_not_binomial (p : ℝ) (k : ℕ) : 
  ∃ n, ∀ q, GeometricDistribution p k ≠ BinomialDistribution n q k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_not_binomial_l1060_106065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_composition_l1060_106045

theorem continuous_function_composition (a : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = (x - a)^2) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuous_function_composition_l1060_106045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_circle_line_intersection_range_reverse_l1060_106033

-- Define the circle
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + a - 5 = 0

-- Define the line
def line_eq (x y : ℝ) : Prop := 3*x - 4*y - 15 = 0

-- Define the distance from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  abs (3*x - 4*y - 15) / Real.sqrt (3^2 + 4^2)

-- Define the condition of exactly two points on the circle at distance 1 from the line
def two_points_at_distance_one (a : ℝ) : Prop :=
  ∃! (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    circle_eq p.1 p.2 a ∧ 
    circle_eq q.1 q.2 a ∧ 
    distance_to_line p.1 p.2 = 1 ∧ 
    distance_to_line q.1 q.2 = 1

-- Theorem statement
theorem circle_line_intersection_range (a : ℝ) :
  two_points_at_distance_one a → -15 < a ∧ a < 1 := by
  sorry

-- Additional lemma to show the reverse implication
theorem circle_line_intersection_range_reverse (a : ℝ) :
  -15 < a ∧ a < 1 → two_points_at_distance_one a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_range_circle_line_intersection_range_reverse_l1060_106033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l1060_106025

/-- The set of prime digits used to form the integers -/
def prime_digits : Finset Nat := {2, 3, 5, 7}

/-- A function that checks if a natural number is a valid three-digit integer
    formed using the given prime digits without repetition -/
def is_valid_integer (n : Nat) : Bool :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) ∈ prime_digits ∧
  ((n / 10) % 10) ∈ prime_digits ∧
  (n % 10) ∈ prime_digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

/-- The theorem stating that the number of valid integers is 24 -/
theorem count_valid_integers :
  (Finset.filter (fun n => is_valid_integer n) (Finset.range 1000)).card = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l1060_106025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_l1060_106072

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def coplanar (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : Prop :=
  let v1 := (p2.1 - p1.1, p2.2.1 - p1.2.1, p2.2.2 - p1.2.2)
  let v2 := (p3.1 - p1.1, p3.2.1 - p1.2.1, p3.2.2 - p1.2.2)
  let v3 := (p4.1 - p1.1, p4.2.1 - p1.2.1, p4.2.2 - p1.2.2)
  v1.1 * (v2.2.1 * v3.2.2 - v2.2.2 * v3.2.1) -
  v1.2.1 * (v2.1 * v3.2.2 - v2.2.2 * v3.1) +
  v1.2.2 * (v2.1 * v3.2.1 - v2.2.1 * v3.1) = 0

theorem coplanar_points (b : ℝ) :
  coplanar (point 0 0 0) (point 1 b 0) (point 0 1 b) (point b 0 1) ↔ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_l1060_106072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_still_water_time_for_given_times_l1060_106060

/-- Represents the time taken for a ship to travel between two ports -/
structure TravelTime where
  downstream : ℚ  -- Time taken downstream
  upstream : ℚ    -- Time taken upstream

/-- Calculates the time taken to travel in still water given downstream and upstream times -/
def stillWaterTime (t : TravelTime) : ℚ :=
  (2 * t.downstream * t.upstream) / (t.downstream + t.upstream)

theorem still_water_time_for_given_times :
  let t : TravelTime := { downstream := 6, upstream := 8 }
  stillWaterTime t = 48 / 7 := by
  -- Unfold the definition of stillWaterTime
  unfold stillWaterTime
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval stillWaterTime { downstream := 6, upstream := 8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_still_water_time_for_given_times_l1060_106060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_perimeter_regular_octagon_perimeter_proof_l1060_106068

/-- The perimeter of a regular octagon with side length 12 meters is 96 meters. -/
theorem regular_octagon_perimeter : ℝ → Prop :=
  fun perimeter =>
    let side_length : ℝ := 12
    let num_sides : ℝ := 8
    perimeter = side_length * num_sides

#check regular_octagon_perimeter 96

/-- Proof of the regular octagon perimeter theorem -/
theorem regular_octagon_perimeter_proof : regular_octagon_perimeter 96 := by
  unfold regular_octagon_perimeter
  simp
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_perimeter_regular_octagon_perimeter_proof_l1060_106068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1060_106076

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  (c * Real.cos A) / (a * Real.cos C) - c / (2 * b - c) = 0 →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  A = π / 3 ∧
  (a = 2 → ∃ h : ℝ, h ≤ Real.sqrt 3 ∧
    ∀ h' : ℝ, (∃ b' c' : ℝ, b' > 0 ∧ c' > 0 ∧ b'^2 + c'^2 - b' * c' = 4 ∧
      h' = 2 * Real.sin (π / 6)) → h' ≤ h) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1060_106076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_invariant_sum_l1060_106067

/-- Sum of digits function -/
def sigma (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → sigma n = sigma (k * n)

/-- Theorem statement -/
theorem smallest_three_digit_invariant_sum :
  ∃ n : ℕ, n = 999 ∧ n ≥ 100 ∧ n < 1000 ∧ satisfiesCondition n ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ satisfiesCondition m → n ≤ m := by
  sorry

#check smallest_three_digit_invariant_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_three_digit_invariant_sum_l1060_106067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_factors_of_28_l1060_106070

-- Define what a perfect number is
def isPerfectNumber (n : Nat) : Prop :=
  n > 0 ∧ (Finset.sum (Nat.divisors n) id) = 2 * n

-- Define the sum of reciprocals of factors
noncomputable def sumOfReciprocalsOfFactors (n : Nat) : ℚ :=
  Finset.sum (Nat.divisors n) (fun d => (1 : ℚ) / d)

-- Theorem statement
theorem sum_of_reciprocals_of_factors_of_28 :
  isPerfectNumber 28 → sumOfReciprocalsOfFactors 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_factors_of_28_l1060_106070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_income_l1060_106094

-- Define the rental company's parameters
def total_cars : ℕ := 100
def base_rent : ℚ := 3000
def rent_increment : ℚ := 50
def rented_maintenance : ℚ := 150
def non_rented_maintenance : ℚ := 50

-- Define the number of rented cars as a function of rent
noncomputable def rented_cars (rent : ℚ) : ℚ :=
  ↑total_cars - (rent - base_rent) / rent_increment

-- Define the income function
noncomputable def income (rent : ℚ) : ℚ :=
  let rented := rented_cars rent
  rent * rented - rented_maintenance * rented - non_rented_maintenance * (↑total_cars - rented)

-- Theorem: The maximum income is achieved at rent = 4050 and equals 307050
theorem max_income :
  ∃ (max_rent : ℚ), max_rent = 4050 ∧
  ∀ (rent : ℚ), rent ≥ base_rent → income rent ≤ income max_rent ∧
  income max_rent = 307050 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_income_l1060_106094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_l1060_106029

noncomputable def a : ℕ → ℂ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | 2 => 1
  | 3 => 1
  | 4 => -1
  | 5 => 0
  | (n + 6) => 3 * a (n + 5) - 4 * a (n + 4) + 4 * a (n + 3) - 3 * a (n + 2) + a (n + 1)

theorem a_general_formula (n : ℕ) :
  a n = ((-3 + 7*Complex.I)/8) * Complex.I^n + ((-3 - 7*Complex.I)/8) * (-Complex.I)^n + (3*n^2 - 19*n + 27) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_l1060_106029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_clear_table_l1060_106042

/-- Represents a pile of pebbles -/
structure Pile :=
  (count : Nat)

/-- Represents the state of all piles on the table -/
def Table := List Pile

/-- A step in the pebble removal process -/
structure Step :=
  (pebblesToRemove : Nat)
  (pilesToReduce : List Nat)

/-- Applies a step to the table, reducing the specified piles -/
def applyStep (table : Table) (step : Step) : Table :=
  sorry

/-- Checks if all piles are empty -/
def allPilesEmpty (table : Table) : Prop :=
  table.all (λ pile => pile.count = 0)

/-- Creates the initial table with piles from 1 to 100 pebbles -/
def initialTable : Table :=
  (List.range 100).map (λ i => Pile.mk (i + 1))

/-- The main theorem stating that 7 steps are necessary and sufficient -/
theorem min_steps_to_clear_table :
  ∃ (steps : List Step),
    steps.length = 7 ∧
    allPilesEmpty (steps.foldl applyStep initialTable) ∧
    ∀ (steps' : List Step),
      steps'.length < 7 →
      ¬allPilesEmpty (steps'.foldl applyStep initialTable) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_clear_table_l1060_106042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l1060_106048

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1) * log x + a * x^2 + 1

-- State the theorem
theorem function_inequality_implies_a_range (a : ℝ) :
  (a < -1) →
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → |f a x₁ - f a x₂| ≥ 4 * |x₁ - x₂|) →
  a ≤ -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_a_range_l1060_106048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1060_106074

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

-- Theorem statement
theorem f_composition_value : f (f (1/9)) = 1/4 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1060_106074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_br2_consumed_equals_hbr_produced_l1060_106066

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  ch4_consumed : Moles
  br2_consumed : Moles
  hbr_produced : Moles
  ch3br_produced : Moles

/-- The balanced equation for CH4 + Br2 → CH3Br + HBr -/
def balanced_equation (r : Reaction) : Prop :=
  r.ch4_consumed = r.br2_consumed ∧
  r.ch4_consumed = r.hbr_produced ∧
  r.ch4_consumed = r.ch3br_produced

theorem br2_consumed_equals_hbr_produced (r : Reaction) 
  (h1 : balanced_equation r)
  (h2 : r.ch4_consumed = (1 : ℝ))
  (h3 : r.hbr_produced = (1 : ℝ)) :
  r.br2_consumed = (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_br2_consumed_equals_hbr_produced_l1060_106066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_service_charge_is_two_percent_l1060_106097

/-- Represents a bank account with transactions and service charges. -/
structure BankAccount where
  initial_balance : ℚ
  transfer1 : ℚ
  transfer2 : ℚ
  final_balance : ℚ
  service_charge_percent : ℚ

/-- Calculates the final balance after transactions and service charges. -/
def calculate_final_balance (account : BankAccount) : ℚ :=
  account.initial_balance - 
  (account.transfer1 + account.transfer1 * account.service_charge_percent / 100) - 
  (account.transfer2 * account.service_charge_percent / 100)

/-- Theorem stating that the service charge percentage is 2% given the conditions. -/
theorem service_charge_is_two_percent (account : BankAccount) : 
  account.initial_balance = 400 ∧ 
  account.transfer1 = 90 ∧ 
  account.transfer2 = 60 ∧ 
  account.final_balance = 307 ∧
  calculate_final_balance account = account.final_balance →
  account.service_charge_percent = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_service_charge_is_two_percent_l1060_106097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_problem_l1060_106031

/-- The length of a string wrapped around a cylindrical post -/
noncomputable def string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  loops * Real.sqrt (((height / loops) ^ 2) + (circumference ^ 2))

/-- Theorem: The length of the string is 15√5 feet -/
theorem string_length_problem : 
  string_length 6 15 5 = 15 * Real.sqrt 5 := by
  -- Unfold the definition of string_length
  unfold string_length
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_problem_l1060_106031
