import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l144_14403

theorem perpendicular_lines_a_values (a : ℝ) :
  let l1 : ℝ × ℝ → Prop := λ p => a * p.1 + 2 * p.2 = 0
  let l2 : ℝ × ℝ → Prop := λ p => (a - 1) * p.1 - p.2 = 0
  (∀ p q : ℝ × ℝ, l1 p ∧ l2 q → (p.1 - q.1) * ((a - 1) * (p.2 - q.2) + (p.1 - q.1)) = 0) →
  a = 2 ∨ a = -1 := by
  sorry

#check perpendicular_lines_a_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_a_values_l144_14403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algae_free_day_l144_14414

/-- Represents the coverage of algae in the pond on a given day -/
def algae_coverage (day : ℕ) : ℚ :=
  3 ^ (day - 28)  / 9

theorem algae_free_day : ∃ d : ℕ, d ≤ 30 ∧ abs (algae_coverage d - 1/10) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algae_free_day_l144_14414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_is_one_l144_14491

noncomputable section

-- Define the complex number i
def i : ℂ := Complex.I

-- Define z as given in the problem
noncomputable def z : ℂ := (1 - i) / (1 + i) + 2 * i

-- Theorem statement
theorem magnitude_of_z_is_one : Complex.abs z = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_is_one_l144_14491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_of_specific_parabola_l144_14421

/-- The focus of a parabola y = ax^2 + k is at (0, 1/(4a) + k) --/
theorem parabola_focus (a k : ℝ) (h : a ≠ 0) :
  (0, 1/(4*a) + k).2 = (4*a*k + 1)/(4*a) :=
by sorry

/-- The focus of the parabola y = 9x^2 + 6 is at (0, 217/36) --/
theorem focus_of_specific_parabola :
  (0, 217/36) = (0, 1/(4*9) + 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_focus_of_specific_parabola_l144_14421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_5_7_l144_14498

def sequence_a : ℕ → ℚ
  | 0 => 6/7  -- Add a case for 0
  | 1 => 6/7
  | n + 1 => 
    let a_n := sequence_a n
    if 0 ≤ a_n ∧ a_n < 1/2 then 2 * a_n
    else if 1/2 ≤ a_n ∧ a_n < 1 then 2 * a_n - 1
    else 0  -- This case should never occur given the initial condition

theorem a_20_equals_5_7 : sequence_a 20 = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_5_7_l144_14498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l144_14425

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 3 * y - 7 = 0

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (0, 4)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := y + 3 * x - 4 = 0

-- Theorem statement
theorem perpendicular_line_equation : 
  ∀ (x y : ℝ), 
    (given_line x y → 
      ∃ (m : ℝ), perpendicular_line x y ∧ given_line x y) ∧
    perpendicular_line point.1 point.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l144_14425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_473_decimal_l144_14446

def is_valid_fraction (m n : ℕ) : Prop :=
  m < n ∧ Nat.Coprime m n ∧ ∃ k, 473 * n + k = 1000 * m ∧ 0 ≤ k ∧ k < n

theorem smallest_n_for_473_decimal (m n : ℕ) :
  is_valid_fraction m n → n ≥ 477 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_473_decimal_l144_14446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_consumption_l144_14430

/-- The number of pickle slices Sammy can eat -/
noncomputable def sammy : ℝ := 25

/-- The number of pickle slices Tammy can eat -/
noncomputable def tammy : ℝ := 3.5 * sammy

/-- The number of pickle slices Ron can eat -/
noncomputable def ron : ℝ := tammy * (3/4)

/-- The number of pickle slices Amy can eat -/
noncomputable def amy : ℝ := sammy * 1.408

/-- The number of pickle slices Tim can eat -/
noncomputable def tim : ℝ := (ron + amy) * (15/16)

/-- Sammy's increased consumption -/
noncomputable def sammy_increased : ℝ := sammy * (1 + 2/5)

theorem tim_consumption : tim = 94.515625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_consumption_l144_14430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmony_sum_l144_14424

def alphabet_value (n : ℕ) : ℤ :=
  match n % 7 with
  | 0 => 3
  | 1 => 2
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -2
  | _ => -3

def letter_to_num (c : Char) : ℕ :=
  c.toNat - 'a'.toNat + 1

def word_sum (w : String) : ℤ :=
  w.toList.map (fun c => alphabet_value (letter_to_num c)) |>.sum

theorem harmony_sum : word_sum "harmony" = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmony_sum_l144_14424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l144_14428

noncomputable def A : ℝ × ℝ := (Real.sqrt 3, 1)
def E : ℝ × ℝ := (0, -1)

noncomputable def ellipse (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, Real.sin θ)

theorem dot_product_range :
  ∀ θ : ℝ, 
  let F := ellipse θ
  let AE := (E.1 - A.1, E.2 - A.2)
  let AF := (F.1 - A.1, F.2 - A.2)
  let dot_product := AE.1 * AF.1 + AE.2 * AF.2
  5 - Real.sqrt 13 ≤ dot_product ∧ dot_product ≤ 5 + Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_range_l144_14428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l144_14482

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]
variable (M : Matrix (Fin 3) (Fin 3) ℝ)
variable (v w : Fin 3 → ℝ)

theorem matrix_vector_computation 
  (h1 : M.mulVec v = ![1, 4, 2])
  (h2 : M.mulVec w = ![0, -1, 3]) :
  M.mulVec (2 • v - 4 • w) = ![2, 4, -8] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l144_14482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l144_14489

noncomputable def f (x : ℝ) : ℝ := 6 / (2^x + 3^x)

theorem min_value_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≥ 6/5 ∧ ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, f x₀ = 6/5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l144_14489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_of_3_to_20_l144_14472

theorem third_of_3_to_20 (x : ℝ) :
  (1/3) * (3:ℝ)^20 = 3^x → x = 19 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_of_3_to_20_l144_14472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l144_14481

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define the line l
def l (x y : ℝ) : Prop := 3*x + 4*y - 5 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State that A and B are on both C and l
axiom A_on_C : C A.1 A.2
axiom A_on_l : l A.1 A.2
axiom B_on_C : C B.1 B.2
axiom B_on_l : l B.1 B.2

-- Define the center of the circle
def center : ℝ × ℝ := (3, 4)

-- State the theorem
theorem area_of_triangle_ABC : ∃ ABC_area : ℝ, ABC_area = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l144_14481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l144_14438

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the midpoint of the chord
def chord_midpoint : ℝ × ℝ := (-1, 1)

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 3*x - 4*y + 7 = 0

-- Theorem statement
theorem chord_line_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
  ((x₁ + x₂) / 2 = chord_midpoint.1 ∧ (y₁ + y₂) / 2 = chord_midpoint.2) →
  line_equation x₁ y₁ ∧ line_equation x₂ y₂ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_line_equation_l144_14438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_with_110_divisors_sum_to_7billion_l144_14470

theorem two_numbers_with_110_divisors_sum_to_7billion : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  (∃ (k : ℕ), a = k * 10^9) ∧ 
  (∃ (m : ℕ), b = m * 10^9) ∧ 
  (Finset.card (Nat.divisors a) = 110) ∧ 
  (Finset.card (Nat.divisors b) = 110) ∧ 
  (a + b = 7000000000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_with_110_divisors_sum_to_7billion_l144_14470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_necessary_not_sufficient_for_Q_l144_14435

-- Define the propositions
def Q : Prop := sorry
def D : Prop := sorry

-- State the theorem
theorem D_necessary_not_sufficient_for_Q : 
  (Q → D) ∧ ¬(D → Q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_necessary_not_sufficient_for_Q_l144_14435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l144_14443

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8*x

noncomputable def line (x y : ℝ) : Prop := 4*x + 3*y + 8 = 0

noncomputable def distance_to_line (x y : ℝ) : ℝ := |4*x + 3*y + 8| / Real.sqrt (4^2 + 3^2)

theorem min_distance_sum : 
  ∃ (d1_min d2_min : ℝ),
    (∀ (x y : ℝ), parabola x y → 
      ∃ (d1 d2 : ℝ), 
        d1 + d2 ≥ d1_min + d2_min ∧ 
        d1_min + d2_min = distance_to_line 2 0) ∧
    d1_min + d2_min = 16/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l144_14443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14431

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) - Real.sqrt 3 * Real.cos (2 * x) - 1 / 2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ k : ℤ, ∃ c : ℝ × ℝ, c = (k * Real.pi / 2 + Real.pi / 6, -1 / 2) ∧
    ∀ x : ℝ, f (2 * c.fst - x) = f x) ∧
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3) →
    f x₀ = Real.sqrt 3 / 3 - 1 / 2 →
    Real.cos (2 * x₀) = -(3 + Real.sqrt 6) / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_l144_14455

/-- The sequence defined by the recurrence relation -/
noncomputable def sequenceF (α β : ℝ) : ℕ → ℝ → ℝ
  | 0, a => a
  | n + 1, a => (sequenceF α β n a + α) / (β * sequenceF α β n a + 1)

/-- The set of fixed points for the recurrence relation -/
def fixedPoints (α β : ℝ) : Set ℝ :=
  {a | a = (a + α) / (β * a + 1)}

/-- The theorem stating the set of values for which there is no infinite sequence -/
theorem no_infinite_sequence (α β : ℝ) (h : α * β > 0) :
  {a : ℝ | ¬∃ (x : ℕ → ℝ), ∀ n, x n = sequenceF α β n a} = {-Real.sqrt (α / β), Real.sqrt (α / β)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_l144_14455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_distance_ratio_l144_14411

/-- Parabola with parameter p > 0 -/
structure Parabola (p : ℝ) where
  (p_pos : p > 0)

/-- Point on the parabola -/
structure ParabolaPoint (p : ℝ) extends Parabola p where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * p * x

/-- Distance ratio for a point on the parabola -/
noncomputable def distance_ratio (p : ℝ) (pt : ParabolaPoint p) : ℝ :=
  Real.sqrt (pt.x^2 + pt.y^2) / Real.sqrt ((pt.x - p/2)^2 + pt.y^2)

/-- Theorem: The maximum distance ratio is 2/√3, achieved when x = p -/
theorem parabola_max_distance_ratio (p : ℝ) (hp : p > 0) :
  ∀ (pt : ParabolaPoint p), distance_ratio p pt ≤ 2 / Real.sqrt 3 ∧
  (distance_ratio p pt = 2 / Real.sqrt 3 ↔ pt.x = p) := by
  sorry

#check parabola_max_distance_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_max_distance_ratio_l144_14411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_y_to_cd_l144_14417

/-- Given a square ABCD with side length s, Y is the intersection point of two quarter-circle arcs:
    one with radius s centered at A, and another with radius s/2 centered at D.
    This theorem proves that the distance from Y to side CD is s/8. -/
theorem distance_y_to_cd (s : ℝ) (A B C D Y : ℝ × ℝ) : 
  let square_side := λ (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let on_circle := λ (center point : ℝ × ℝ) (radius : ℝ) => 
    (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2
  s > 0 ∧
  square_side A B = s ∧ square_side B C = s ∧ square_side C D = s ∧ square_side D A = s ∧
  on_circle A Y s ∧
  on_circle D Y (s/2) →
  Y.2 - D.2 = s/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_y_to_cd_l144_14417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_implications_l144_14497

theorem count_implications {p q r : Prop} [Decidable p] [Decidable q] [Decidable r] : 
  ∃! (n : Nat), n = 2 ∧ 
    n = (if ((¬p ∧ q ∧ ¬r) → ((¬p → q) → r)) then 1 else 0) +
        (if ((p ∧ q ∧ ¬r) → ((¬p → q) → r)) then 1 else 0) +
        (if ((p ∧ ¬q ∧ r) → ((¬p → q) → r)) then 1 else 0) +
        (if ((¬p ∧ ¬q ∧ r) → ((¬p → q) → r)) then 1 else 0) :=
by
  -- The proof goes here
  sorry

#check count_implications

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_implications_l144_14497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l144_14445

variable (a b c : ℝ)
variable (A B C : ℝ)

/-- Triangle ABC with sides a, b, c and angles A, B, C is right-angled if a^2 + b^2 + c^2 = 2bc*cos(A) + 2ac*cos(B) -/
theorem triangle_right_angled (h : a^2 + b^2 + c^2 = 2*b*c*Real.cos A + 2*a*c*Real.cos B) :
  A = π/2 ∨ B = π/2 ∨ C = π/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l144_14445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_correct_l144_14434

/-- Calculates the percentage savings for a hat purchase under specific conditions -/
noncomputable def calculateSavings (regularPrice : ℝ) (secondHatDiscount : ℝ) (thirdHatDiscount : ℝ) (extraDiscount : ℝ) (numHats : ℕ) : ℝ :=
  let totalRegularPrice := regularPrice * (numHats : ℝ)
  let discountedPrice := 
    regularPrice + 
    (1 - secondHatDiscount) * regularPrice + 
    (1 - thirdHatDiscount) * regularPrice + 
    regularPrice
  let finalPrice := (1 - extraDiscount) * discountedPrice
  let savings := totalRegularPrice - finalPrice
  (savings / totalRegularPrice) * 100

/-- Theorem stating that the savings percentage for the given conditions is approximately 24.625% -/
theorem savings_percentage_is_correct : 
  let regularPrice : ℝ := 60
  let secondHatDiscount : ℝ := 0.3
  let thirdHatDiscount : ℝ := 0.35
  let extraDiscount : ℝ := 0.1
  let numHats : ℕ := 4
  abs (calculateSavings regularPrice secondHatDiscount thirdHatDiscount extraDiscount numHats - 24.625) < 0.001 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_is_correct_l144_14434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l144_14456

open Real

-- Define the equation
noncomputable def f (θ : ℝ) : ℝ := 2 + 4 * sin (2 * θ) - 3 * cos (4 * θ) + 2 * tan θ

-- Theorem statement
theorem solution_count :
  ∃ (S : Finset ℝ), S.card = 16 ∧
  (∀ θ ∈ S, 0 < θ ∧ θ ≤ 4 * π ∧ f θ = 0) ∧
  (∀ θ, 0 < θ ∧ θ ≤ 4 * π ∧ f θ = 0 → θ ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l144_14456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l144_14478

open Real

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define vectors m and n -/
noncomputable def m (t : Triangle) : ℝ × ℝ := (1 - cos (t.A + t.B), cos ((t.A - t.B) / 2))
noncomputable def n (t : Triangle) : ℝ × ℝ := (5/8, cos ((t.A - t.B) / 2))

/-- Define dot product of vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h : dot_product (m t) (n t) = 9/8) : 
  (tan t.A * tan t.B = 1/9) ∧ 
  (∀ t' : Triangle, (t'.a * t'.b * sin t'.C) / (t'.a^2 + t'.b^2 - t'.c^2) ≤ -3/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l144_14478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_equals_15_l144_14419

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := ((1/2) * x - Real.sqrt 2) ^ 6

-- Define the coefficient of x^2 in the expansion of f(x)
noncomputable def coefficient_x_squared : ℝ := 
  (Nat.choose 6 4) * ((1/2) ^ 2) * ((-Real.sqrt 2) ^ 4)

-- Theorem statement
theorem coefficient_equals_15 : coefficient_x_squared = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_equals_15_l144_14419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l144_14499

/-- If z = cos θ - 5/13 + (12/13 - sin θ)i is purely imaginary, then tan θ = -12/5 -/
theorem tan_theta_value (θ : Real) (z : ℂ) 
  (h : z = Complex.mk (Real.cos θ - 5/13) (12/13 - Real.sin θ))
  (h_imag : z.re = 0) : Real.tan θ = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l144_14499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_midline_l144_14471

/-- An isosceles trapezoid with a circle on its smaller leg -/
structure IsoscelesTrapezoidWithCircle where
  /-- The length of the longer leg -/
  a : ℝ
  /-- The circle is constructed on the smaller leg as its diameter -/
  circle_on_smaller_leg : Prop
  /-- The circle touches the longer leg -/
  circle_touches_longer_leg : Prop

/-- The midline of an isosceles trapezoid with a circle on its smaller leg -/
noncomputable def midline (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  t.a / 2

/-- Theorem: The midline of the described isosceles trapezoid is half the length of its longer leg -/
theorem isosceles_trapezoid_midline (t : IsoscelesTrapezoidWithCircle) :
  midline t = t.a / 2 := by
  -- Unfold the definition of midline
  unfold midline
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_midline_l144_14471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_chord_length_l144_14457

/-- Definition of an ellipse with given parameters -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Definition of the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt ((e.a^2 - e.b^2) / e.a^2)

/-- Definition of the distance from the end of minor axis to focus -/
noncomputable def minorAxisToFocus (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- Theorem about the equation of the ellipse and chord length -/
theorem ellipse_equation_and_chord_length (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 6 / 3)
  (h_dist : minorAxisToFocus e = Real.sqrt 3) :
  (∃ (x y : ℝ), x^2 / 3 + y^2 = 1) ∧ 
  (∃ (A B : ℝ × ℝ), A ∈ {(x, y) | x^2 / 3 + y^2 = 1 ∧ y = x + 1} ∧ 
                     B ∈ {(x, y) | x^2 / 3 + y^2 = 1 ∧ y = x + 1} ∧
                     Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_chord_length_l144_14457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l144_14459

def sequence_a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => 5 * sequence_a (n + 2) - 6 * sequence_a (n + 1)

theorem sequence_a_general_term (n : ℕ) : 
  sequence_a n = 2^(n - 1) := by
  induction n with
  | zero => simp [sequence_a]
  | succ n ih => 
    cases n with
    | zero => simp [sequence_a]
    | succ n => 
      cases n with
      | zero => simp [sequence_a]
      | succ n => 
        simp [sequence_a]
        sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l144_14459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_3_equals_10_l144_14464

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

-- State the theorem
theorem f_of_g_of_3_equals_10 : f (g 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_3_equals_10_l144_14464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_journey_distance_l144_14409

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem carlos_journey_distance :
  let start := (3, -7)
  let mid := (1, 1)
  let finish := (-6, 3)
  distance start.1 start.2 mid.1 mid.2 + distance mid.1 mid.2 finish.1 finish.2 = 2 * Real.sqrt 17 + Real.sqrt 53 := by
  sorry

#check carlos_journey_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlos_journey_distance_l144_14409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_achieve_12_percent_return_l144_14401

/-- Represents a stock with its properties -/
structure Stock where
  price : ℝ
  dividend_rate : ℝ
  transaction_fee_rate : ℝ

/-- Calculates the net return rate for a stock -/
noncomputable def net_return_rate (s : Stock) (tax_rate : ℝ) : ℝ :=
  let net_cost := s.price * (1 + s.transaction_fee_rate)
  let gross_dividend := s.price * s.dividend_rate
  let net_dividend := gross_dividend * (1 - tax_rate)
  net_dividend / net_cost

/-- Theorem stating that it's impossible to achieve a 12% effective interest rate -/
theorem impossible_to_achieve_12_percent_return 
  (stock_a stock_b stock_c : Stock)
  (tax_rate target_rate : ℝ)
  (h_tax : tax_rate = 0.1)
  (h_target : target_rate = 0.12)
  (h_a : stock_a = { price := 52, dividend_rate := 0.09, transaction_fee_rate := 0.02 })
  (h_b : stock_b = { price := 80, dividend_rate := 0.07, transaction_fee_rate := 0.015 })
  (h_c : stock_c = { price := 40, dividend_rate := 0.10, transaction_fee_rate := 0.01 }) :
  ∀ w_a w_b w_c : ℝ, 
    w_a ≥ 0 ∧ w_b ≥ 0 ∧ w_c ≥ 0 ∧ w_a + w_b + w_c = 1 →
    w_a * net_return_rate stock_a tax_rate + 
    w_b * net_return_rate stock_b tax_rate + 
    w_c * net_return_rate stock_c tax_rate < target_rate :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_achieve_12_percent_return_l144_14401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_l144_14468

/-- Given points A and B in 3D space, prove that the vector from A to B is as calculated. -/
theorem vector_AB (A B : ℝ × ℝ × ℝ) :
  A = (3, 1, 2) →
  B = (4, -2, -2) →
  (B.fst - A.fst, B.snd - A.snd, (B.2).2 - (A.2).2) = (1, -3, -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_AB_l144_14468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l144_14494

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 9*x + 20⌋

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≤ 4 ∨ x ≥ 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l144_14494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l144_14454

-- Define the circle
def circle_equation (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + a*x + 2 = 0

-- Define the line
def line_equation (k : ℝ) (x y : ℝ) : Prop := y - 1 = k*(x - 3)

-- Define the tangent point
def tangent_point (x y : ℝ) : Prop := x = 3 ∧ y = 1

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ k : ℝ) : ℝ :=
  abs (1 - k) / Real.sqrt (1 + k^2)

theorem circle_tangent_line :
  ∃ a k : ℝ,
    (∀ x y : ℝ, tangent_point x y → circle_equation a x y) ∧
    (∀ x y : ℝ, tangent_point x y → line_equation k x y) ∧
    (distance_point_to_line 2 0 k = Real.sqrt 2) ∧
    (∀ x y : ℝ, line_equation k x y ↔ x + y = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_l144_14454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_seven_proof_l144_14458

/-- The number of three-digit positive integers with digit sum 7 -/
def count_three_digit_sum_seven : ℕ := 28

/-- A three-digit positive integer -/
structure ThreeDigitInt where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Finite type instance for ThreeDigitInt -/
instance : Fintype ThreeDigitInt := sorry

theorem count_three_digit_sum_seven_proof :
  (Finset.filter (λ n : ThreeDigitInt => digit_sum n.value = 7) Finset.univ).card = count_three_digit_sum_seven := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_seven_proof_l144_14458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_difference_l144_14413

theorem repeating_decimal_difference : 
  (∃ (x : ℚ), x = 8/11 ∧ x - 72/100 = 2/275) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_difference_l144_14413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l144_14487

theorem factorial_ratio (N : ℕ) : 
  (Nat.factorial (N + 2)) / (Nat.factorial (N + 4)) = 1 / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_l144_14487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l144_14486

theorem power_equation_solution (x : ℝ) :
  (2 : ℝ)^x - (2 : ℝ)^(x-2) = 3 * (2 : ℝ)^12 → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l144_14486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l144_14410

theorem cos_pi_4_minus_alpha (α : ℝ) 
  (h1 : Real.sin (2 * α) = 24 / 25) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi / 2) : 
  Real.sqrt 2 * Real.cos (Real.pi / 4 - α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_4_minus_alpha_l144_14410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_rhombus_existence_l144_14485

/-- The ellipse C with equation x²/2 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The focus F of the ellipse -/
def focus_F : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through the focus F -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

/-- The perpendicular bisector of a line segment -/
def perp_bisector (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (x - (x1 + x2)/2) * ((x2 - x1)/2) + (y - (y1 + y2)/2) * ((y2 - y1)/2) = 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Main theorem -/
theorem ellipse_rhombus_existence :
  ∃ (k : ℝ) (xA yA xB yB xD xE yE : ℝ),
    k ≠ 0 ∧
    ellipse_C xA yA ∧
    ellipse_C xB yB ∧
    ellipse_C xE yE ∧
    line_l k xA yA ∧
    line_l k xB yB ∧
    perp_bisector xA yA xB yB xD 0 ∧
    distance xA yA xD 0 = distance xD 0 xB yB ∧
    distance xA yA xE yE = distance xD 0 xB yB ∧
    distance xE yE xD 0 = distance xA yA xB yB ∧
    xE = (12 - 3 * Real.sqrt 2) / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_rhombus_existence_l144_14485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l144_14442

/-- A parabola with equation y^2 = 8x -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ y^2 = 8*x

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.eq x y

/-- The focus of the parabola y^2 = 8x -/
def focus (p : Parabola) : ℝ × ℝ := (2, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem statement -/
theorem distance_to_focus (p : Parabola) (P : PointOnParabola p) 
  (h : distance (P.x, P.y) (0, P.y) = 4) :
  distance (P.x, P.y) (focus p) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l144_14442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_and_monotonicity_l144_14405

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

-- State the theorem
theorem cosine_symmetry_and_monotonicity (ω : ℝ) :
  (∀ x : ℝ, f ω x = f ω (3 * Real.pi / 2 - x)) →  -- Symmetry about (3π/4, 0)
  (ω > 0) →  -- ω is positive
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * Real.pi / 3 → 
    (f ω x₁ > f ω x₂ ∨ f ω x₁ < f ω x₂)) →  -- Monotonicity in (0, 2π/3)
  ω = 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_and_monotonicity_l144_14405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l144_14473

/-- Parabola type -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle type -/
structure Circle where
  center : Point
  radius : ℝ

/-- Directrix type -/
structure Directrix where
  x : ℝ

noncomputable def focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

def line_passes_through_point (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.b

def point_on_parabola (c : Parabola) (p : Point) : Prop :=
  p.y^2 = 2 * c.p * p.x

def circle_tangent_to_directrix (circ : Circle) (d : Directrix) : Prop :=
  |circ.center.x - d.x| = circ.radius

theorem parabola_line_intersection 
  (c : Parabola) 
  (l : Line) 
  (d : Directrix) 
  (hf : line_passes_through_point l (focus c))
  (hl : l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3) :
  ∃ (M N : Point),
    point_on_parabola c M ∧
    point_on_parabola c N ∧
    line_passes_through_point l M ∧
    line_passes_through_point l N ∧
    c.p = 2 ∧
    circle_tangent_to_directrix 
      { center := { x := (M.x + N.x) / 2, y := (M.y + N.y) / 2 },
        radius := |M.x - N.x| / 2 } 
      { x := -c.p / 2 } :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l144_14473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_semicircle_l144_14467

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A right triangle with legs 1 and √3 -/
def RightTriangle : Set Point :=
  {p : Point | 0 ≤ p.x ∧ 0 ≤ p.y ∧ p.x + p.y / Real.sqrt 3 ≤ 1}

/-- A semicircle with diameter 1/√3 -/
def Semicircle (center : Point) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 ≤ (1 / (2 * Real.sqrt 3))^2 ∧ p.y ≥ center.y}

/-- The main theorem -/
theorem three_points_in_semicircle 
  (points : Finset Point) 
  (h1 : points.card = 20) 
  (h2 : ∀ p ∈ points, p ∈ RightTriangle) : 
  ∃ (center : Point) (s : Finset Point), 
    s ⊆ points ∧ s.card = 3 ∧ (∀ p ∈ s, p ∈ Semicircle center) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_semicircle_l144_14467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_94_75_main_theorem_l144_14429

/-- The sum of odd-indexed terms in the series -/
noncomputable def S : ℚ := ∑' n, (2*n - 1) / (2^(2*n - 1 : ℕ))

/-- The sum of even-indexed terms in the series -/
noncomputable def T : ℚ := ∑' n, (2*n) / (4^(2*n : ℕ))

/-- The total sum of the series -/
noncomputable def seriesSum : ℚ := S + T

theorem series_sum_equals_94_75 : seriesSum = 94 / 75 := by sorry

theorem main_theorem (p q : ℕ) (h1 : Nat.Coprime p q) (h2 : (p : ℚ) / q = seriesSum) : 
  p + q = 169 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_94_75_main_theorem_l144_14429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_18_l144_14490

def b : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | k+2 => (1/2) * b (k+1) + (1/3) * b k

noncomputable def sequence_sum : ℚ := ∑' n, b n

theorem sum_of_sequence_is_18 : sequence_sum = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_18_l144_14490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_number_with_properties_properties_of_1236_l144_14453

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < digits.length → digits.get ⟨i, by sorry⟩ ≠ digits.get ⟨j, by sorry⟩

def has_prime_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ is_prime d

def divisible_by_all_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem least_four_digit_number_with_properties :
  ∀ n : ℕ,
    1000 ≤ n ∧ n < 10000 ∧
    all_digits_different n ∧
    has_prime_digit n ∧
    divisible_by_all_digits n →
    n ≥ 1236 :=
by sorry

theorem properties_of_1236 :
  1000 ≤ 1236 ∧ 1236 < 10000 ∧
  all_digits_different 1236 ∧
  has_prime_digit 1236 ∧
  divisible_by_all_digits 1236 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_four_digit_number_with_properties_properties_of_1236_l144_14453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_26_l144_14412

-- Define the vertices of the square
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (-4, 3)
def R : ℝ × ℝ := (-3, 8)
def S : ℝ × ℝ := (2, 7)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the function to calculate the area of a square given its side length
def squareArea (sideLength : ℝ) : ℝ := sideLength^2

-- Theorem statement
theorem square_area_is_26 :
  squareArea (distance P Q) = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_is_26_l144_14412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_problem_l144_14480

theorem sugar_solution_problem (initial_sugar_percent : ℝ) (final_sugar_percent : ℝ) 
  (replaced_fraction : ℝ) (second_solution_sugar_percent : ℝ) :
  initial_sugar_percent = 22 →
  final_sugar_percent = 35 →
  replaced_fraction = 1/4 →
  (1 - replaced_fraction) * initial_sugar_percent + 
    replaced_fraction * second_solution_sugar_percent = final_sugar_percent →
  second_solution_sugar_percent = 74 := by
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

#check sugar_solution_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_problem_l144_14480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l144_14488

theorem exam_average (group1_count group2_count : ℕ) (group1_avg group2_avg : ℚ) :
  group1_count = 15 →
  group1_avg = 70/100 →
  group2_count = 10 →
  group2_avg = 95/100 →
  let total_count : ℕ := group1_count + group2_count
  let total_avg : ℚ := (group1_count * group1_avg + group2_count * group2_avg) / total_count
  total_avg = 80/100 := by
    intro h1 h2 h3 h4
    -- The proof goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l144_14488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_passage_time_l144_14432

/-- Calculates the time (in minutes) for a train to pass through a tunnel -/
noncomputable def train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length : ℝ) : ℝ :=
  let train_speed_mpm := train_speed_kmh * 1000 / 60
  let total_distance := tunnel_length * 1000 + train_length
  total_distance / train_speed_mpm

/-- Theorem stating that a train of length 100 meters traveling at 72 km/hr through a tunnel of length 3.5 km takes 3 minutes to pass through -/
theorem train_tunnel_passage_time :
  train_tunnel_time 100 72 3.5 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_passage_time_l144_14432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l144_14475

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x ≥ 0 then -2^x + x + m else -((-2^(-x) + (-x) + m))

-- State the theorem
theorem odd_function_value (m : ℝ) :
  (∀ x, f x m = -f (-x) m) →  -- f is an odd function
  f (-2) m = 1 :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l144_14475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersections_l144_14422

/-- A line in a 2D plane defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The x-coordinate of the intersection point of a line with the x-axis -/
noncomputable def xAxisIntersection (l : Line) : ℝ := l.yIntercept / (-l.slope)

/-- The y-coordinate of the intersection point of a line with the y-axis -/
def yAxisIntersection (l : Line) : ℝ := l.yIntercept

/-- Theorem: The line y = -2x + 4 intersects the x-axis at (2, 0) and the y-axis at (0, 4) -/
theorem line_intersections :
  let l : Line := { slope := -2, yIntercept := 4 }
  xAxisIntersection l = 2 ∧ yAxisIntersection l = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersections_l144_14422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_positive_period_of_f_l144_14426

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (1 + Real.cos (4 * x))

theorem minimal_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_positive_period_of_f_l144_14426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_monotonic_increasing_interval_l144_14466

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) - Real.sin (2 * x)

theorem y_monotonic_increasing_interval :
  ∀ x : ℝ, (Real.pi / 12 ≤ x ∧ x ≤ 7 * Real.pi / 12) ↔ 
  (∀ x₁ x₂ : ℝ, Real.pi / 12 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 7 * Real.pi / 12 → y x₁ < y x₂) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_monotonic_increasing_interval_l144_14466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l144_14469

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^2 + b^2 - a*b = c^2) : 
  Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l144_14469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sincos_product_l144_14463

open Real

theorem max_value_sincos_product :
  (∃ (t_max : ℝ), t_max ∈ Set.Icc 0 (2 * π) ∧
    ∀ (s : ℝ), s ∈ Set.Icc 0 (2 * π) →
      (1 + sin s) * (1 + cos s) ≤ (1 + sin t_max) * (1 + cos t_max)) ∧
  (∃ (t_max : ℝ), t_max ∈ Set.Icc 0 (2 * π) ∧
    (1 + sin t_max) * (1 + cos t_max) = (3 + 2 * sqrt 2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sincos_product_l144_14463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_squares_implies_equilateral_l144_14448

/-- Triangle ABC with three inscribed squares -/
structure TriangleWithSquares where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  square_AB : Set (ℝ × ℝ)
  square_BC : Set (ℝ × ℝ)
  square_CA : Set (ℝ × ℝ)

/-- The three inscribed squares are equal -/
def equal_squares (t : TriangleWithSquares) : Prop :=
  t.square_AB = t.square_BC ∧ t.square_BC = t.square_CA

/-- Triangle ABC is equilateral -/
def is_equilateral (t : TriangleWithSquares) : Prop :=
  let d₁ := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let d₂ := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let d₃ := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  d₁ = d₂ ∧ d₂ = d₃

/-- Main theorem: If the three inscribed squares are equal, then the triangle is equilateral -/
theorem equal_squares_implies_equilateral (t : TriangleWithSquares) :
  equal_squares t → is_equilateral t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_squares_implies_equilateral_l144_14448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_kx_range_l144_14407

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin x else -x^2 - 1

-- Theorem statement
theorem f_leq_kx_range :
  ∀ k : ℝ, (∀ x : ℝ, f x ≤ k * x) ↔ (1 ≤ k ∧ k ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_kx_range_l144_14407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_zero_l144_14495

theorem sin_beta_zero (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : Real.sin α = 3/5) (h5 : Real.sin (α + β) = 3/5) : 
  Real.sin β = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_zero_l144_14495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equality_l144_14449

theorem sin_sum_equality (α β : ℝ) :
  Real.sin α + Real.sin β = Real.sin (α + β) ↔
  (∃ n : ℤ, α + β = 2 * Real.pi * n) ∨
  (∃ n : ℤ, α = 2 * Real.pi * n) ∨
  (∃ n : ℤ, β = 2 * Real.pi * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equality_l144_14449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_l144_14444

-- Define a right triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0

-- Define the length of a side
noncomputable def side_length (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the angle between two sides
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ :=
  Real.arccos (((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) /
    (side_length P Q * side_length P R))

-- Theorem statement
theorem isosceles_right_triangle_side_length
  (abc : RightTriangle)
  (angle_equality : angle abc.B abc.A abc.C = angle abc.C abc.A abc.B)
  (hypotenuse_length : side_length abc.A abc.C = 8 * Real.sqrt 2) :
  side_length abc.A abc.B = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_side_length_l144_14444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l144_14452

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x > 0 then -x^2 + 2*x
  else if x < 0 then x^2 - 2*x
  else 0

-- State the theorem
theorem max_a_value (a : ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, x > 0 → f x = -x^2 + 2*x) →  -- f(x) = -x^2 + 2x for x > 0
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ a - 2 → f x < f y) →  -- f is monotonically increasing on [-1, a-2]
  a ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l144_14452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_basket_weight_l144_14439

/-- Represents the capacity of a basket in terms of number of apples -/
abbrev BasketCapacity := Nat

/-- Represents the weight in grams -/
abbrev Weight := Nat

structure Basket where
  capacity : BasketCapacity
  emptyWeight : Weight

def appleWeight : Weight := 150

def jacksBasket : Basket := ⟨12, 200⟩

def jillsBasket : Basket := ⟨2 * jacksBasket.capacity, 300⟩

def spaceInJacksBasket : BasketCapacity := 4

theorem jills_basket_weight : 
  jillsBasket.capacity * appleWeight = 3600 := by
  -- Unfold the definitions
  unfold jillsBasket
  unfold jacksBasket
  -- Simplify the arithmetic
  simp [Nat.mul_assoc, Nat.mul_comm]
  -- The proof is complete
  rfl

#eval jillsBasket.capacity * appleWeight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_basket_weight_l144_14439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l144_14462

/-- Calculates the simple interest given principal, rate, and time --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_problem (loan_B loan_C time_B time_C total_interest : ℝ) :
  loan_B = 5000 →
  loan_C = 3000 →
  time_B = 2 →
  time_C = 4 →
  total_interest = 2200 →
  ∃ (rate : ℝ), 
    simple_interest loan_B rate time_B + simple_interest loan_C rate time_C = total_interest ∧
    rate = 10 := by
  intro hB hC htB htC hI
  use 10
  constructor
  · rw [hB, hC, htB, htC, hI]
    simp [simple_interest]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_problem_l144_14462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_l144_14474

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.B = 60 * Real.pi / 180 ∧ t.b = 4

-- Statement 1
theorem statement_1 (t : Triangle) (h : triangle_conditions t) :
  t.c = Real.sqrt 3 → ∃! x, x = t.C := by sorry

-- Statement 2
theorem statement_2 (t : Triangle) (h : triangle_conditions t) :
  t.c * t.a * Real.cos t.B = 12 → 
  (2 * (t.a * t.c * Real.sin t.B) / t.b) = 3 * Real.sqrt 3 := by sorry

-- Statement 3
theorem statement_3 (t : Triangle) (h : triangle_conditions t) :
  t.a + t.c < 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_statement_2_statement_3_l144_14474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l144_14476

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
  h_sum_angles : A + B + C = π
  h_a_opposite_A : a = b * Real.sin C / Real.sin B
  h_b_opposite_B : b = a * Real.sin C / Real.sin A
  h_c_opposite_C : c = a * Real.sin B / Real.sin A

-- State the Cosine Theorem
theorem cosine_theorem (t : Triangle) : t.a^2 = t.b^2 + t.c^2 - 2*t.b*t.c*Real.cos t.A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_theorem_l144_14476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_tangent_half_angle_l144_14404

/-- Predicate to represent a cyclic quadrilateral -/
def IsCyclicQuadrilateral (a b c d : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a ≤ 2 * r ∧ b ≤ 2 * r ∧ c ≤ 2 * r ∧ d ≤ 2 * r

/-- Function to represent the angle between two sides of a quadrilateral -/
noncomputable def AngleBetween (side1 side2 : ℝ) : ℝ :=
  Real.arccos ((side1^2 + side2^2 - (side1 + side2)^2) / (2 * side1 * side2))

/-- Given a cyclic quadrilateral with sides a, b, c, d, perimeter 2s, and angle α between sides a and b,
    prove that tg²(α/2) = ((s-a)(s-b))/((s-c)(s-d)) -/
theorem cyclic_quadrilateral_tangent_half_angle 
  (a b c d s : ℝ) (α : ℝ) 
  (h_cyclic : IsCyclicQuadrilateral a b c d)
  (h_perimeter : a + b + c + d = 2 * s)
  (h_angle : AngleBetween a b = α) :
  Real.tan (α / 2) ^ 2 = ((s - a) * (s - b)) / ((s - c) * (s - d)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_tangent_half_angle_l144_14404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_intersection_is_zero_l144_14440

/-- Polynomial P(x) = x^4 - 6x^3 + 11x^2 - 6x + a -/
def P (a : ℝ) (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 - 6*x + a

/-- Line L(x) = x + b -/
def L (b : ℝ) (x : ℝ) : ℝ := x + b

/-- The theorem stating that the smallest intersection point is 0 -/
theorem smallest_intersection_is_zero (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    P a x₁ = L b x₁ ∧ P a x₂ = L b x₂ ∧ P a x₃ = L b x₃ ∧ P a x₄ = L b x₄) →
  (∃ x : ℝ, P a x = L b x ∧ ∀ y : ℝ, P a y = L b y → x ≤ y) →
  ∃ x : ℝ, x = 0 ∧ P a x = L b x ∧ ∀ y : ℝ, P a y = L b y → x ≤ y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_intersection_is_zero_l144_14440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABG_measure_l144_14492

/-- The measure of an angle in a regular octagon, in degrees -/
noncomputable def regular_octagon_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Angle ABG in a regular octagon ABCDEFGH -/
noncomputable def angle_ABG (regular_octagon_angle : ℝ) : ℝ := (180 - regular_octagon_angle) / 2

theorem angle_ABG_measure :
  angle_ABG regular_octagon_angle = 22.5 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABG_measure_l144_14492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_result_l144_14436

theorem betting_result (initial_amount : ℝ) (num_bets num_wins num_losses : ℕ) :
  initial_amount = 64 ∧
  num_bets = 6 ∧
  num_wins = 3 ∧
  num_losses = 3 ∧
  num_wins + num_losses = num_bets →
  let final_amount := initial_amount * (3/2)^num_wins * (1/2)^num_losses
  final_amount = 27 ∧ initial_amount - final_amount = 37 := by
  intro h
  -- The proof steps would go here
  sorry

#check betting_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_result_l144_14436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_of_ellipse_l144_14450

/-- Given an ellipse and a chord, prove the equation of the chord --/
theorem chord_equation_of_ellipse :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let midpoint := (1, 1)
  let chord_eq := (λ p : ℝ × ℝ => p.1 + 2*p.2 - 3 = 0)
  (∀ p ∈ ellipse, 
    ∃ q ∈ ellipse, 
      ((p.1 + q.1) / 2, (p.2 + q.2) / 2) = midpoint ∧
      chord_eq p ∧ chord_eq q) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_of_ellipse_l144_14450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mason_test_additional_needed_l144_14479

/-- Represents a math test with different types of questions -/
structure MathTest where
  total : ℕ
  arithmetic : ℕ
  algebra : ℕ
  geometry : ℕ
  correctArithmetic : ℚ
  correctAlgebra : ℚ
  correctGeometry : ℚ
  passingGrade : ℚ

/-- Calculates the number of additional correct answers needed to pass the test -/
def additionalCorrectNeeded (test : MathTest) : ℚ :=
  let totalCorrect := test.correctArithmetic * test.arithmetic +
                      test.correctAlgebra * test.algebra +
                      test.correctGeometry * test.geometry
  let passingScore := test.passingGrade * test.total
  passingScore - totalCorrect

/-- Theorem stating that for Mason's test, 2.5 more correct answers were needed to pass -/
theorem mason_test_additional_needed (test : MathTest)
  (h1 : test.total = 80)
  (h2 : test.arithmetic = 15)
  (h3 : test.algebra = 25)
  (h4 : test.geometry = 40)
  (h5 : test.correctArithmetic = 3/5)
  (h6 : test.correctAlgebra = 1/2)
  (h7 : test.correctGeometry = 4/5)
  (h8 : test.passingGrade = 7/10) :
  additionalCorrectNeeded test = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mason_test_additional_needed_l144_14479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_equality_l144_14441

/-- A sequence of radii satisfying a second-order linear recurrence relation -/
def RadiiSequence (k : ℝ) : ℕ → ℝ := sorry

/-- The recurrence relation for the radii sequence -/
axiom radii_recurrence (k : ℝ) (i : ℕ) :
  RadiiSequence k (i + 2) - k * RadiiSequence k (i + 1) + RadiiSequence k i = 0

/-- The main theorem to be proved -/
theorem radii_equality (k : ℝ) (n N : ℕ) (h : 3 * n - 2 > N) :
  (RadiiSequence k (2 * n - 1)) * (RadiiSequence k 1 + RadiiSequence k (2 * n - 1)) =
  (RadiiSequence k n) * (RadiiSequence k n + RadiiSequence k (3 * n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radii_equality_l144_14441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_packets_weight_l144_14400

/-- Represents a balance with unequal arms -/
structure UnequalBalance where
  leftArm : ℝ
  rightArm : ℝ

/-- The weight of an object on the unequal balance -/
noncomputable def apparentWeight (b : UnequalBalance) (trueWeight : ℝ) (onLeftArm : Bool) : ℝ :=
  if onLeftArm then trueWeight * b.rightArm / b.leftArm else trueWeight * b.leftArm / b.rightArm

theorem eight_packets_weight {x : ℝ} (b : UnequalBalance) 
  (h1 : apparentWeight b (3 * 1) true = apparentWeight b (8 * x) false)
  (h2 : apparentWeight b x true = apparentWeight b (6 * 1) false)
  : 8 * x = 12 := by
  sorry

#check eight_packets_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_packets_weight_l144_14400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l144_14427

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The theorem stating the relationship between S_4, S_8, and S_12 -/
theorem geometric_sequence_sum_ratio
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l144_14427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l144_14415

noncomputable def data_set : List ℝ := [0.7, 1, 0.8, 0.9, 1.1]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ) ^ 2)).sum / (xs.length - 1)

theorem variance_of_data_set :
  variance data_set = 0.02 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l144_14415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_values_l144_14483

theorem sum_of_possible_b_values : 
  ∃ (S : Finset ℤ),
    (∀ b ∈ S, ∃ a c : ℤ, ∀ x : ℝ, (x - ↑a) * (x - 6) + 3 = (x + ↑b) * (x + ↑c)) ∧
    (∀ b : ℤ, (∃ a c : ℤ, ∀ x : ℝ, (x - ↑a) * (x - 6) + 3 = (x + ↑b) * (x + ↑c)) → b ∈ S) ∧
    (S.sum id = -24) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_possible_b_values_l144_14483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adrien_winning_strategy_l144_14460

def is_losing_position (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k + 2^(k+1) - 1

def game_rules (initial_tokens : ℕ) (current_player : ℕ) (tokens_removed : ℕ) (remaining_tokens : ℕ) : Prop :=
  initial_tokens > 1 ∧
  current_player ∈ ({0, 1} : Set ℕ) ∧
  tokens_removed ≥ 1 ∧
  tokens_removed ≤ initial_tokens / 2 ∧
  remaining_tokens = initial_tokens - tokens_removed

theorem adrien_winning_strategy (initial_tokens : ℕ) :
  initial_tokens = 2023 →
  ∃ (strategy : ℕ → ℕ), ∀ (game : ℕ → ℕ → ℕ → ℕ → Prop),
    (∀ t p r s, game t p r s → game_rules t p r s) →
    (∀ t, game t 0 (strategy t) (t - strategy t)) →
    ¬(is_losing_position initial_tokens) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adrien_winning_strategy_l144_14460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_equality_l144_14447

/-- Represents the annual income in dollars -/
def annual_income : ℝ := 56000

/-- Represents the tax rate percentage for the first tax bracket -/
def q : ℝ := 10 -- Assuming q = 10 for simplicity

/-- Calculates the tax amount based on the given income and tax rates -/
noncomputable def calculate_tax (income : ℝ) : ℝ :=
  if income ≤ 30000 then
    0.01 * q * income
  else if income ≤ 50000 then
    0.01 * q * 30000 + 0.01 * (q + 3) * (income - 30000)
  else
    0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (income - 50000)

/-- States that the calculated tax is equal to (q + 0.5)% of the annual income -/
theorem tax_equality : calculate_tax annual_income = 0.01 * (q + 0.5) * annual_income := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_equality_l144_14447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_18_l144_14465

-- Define a second-degree polynomial
def second_degree_polynomial (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

-- State the theorem
theorem sum_of_roots_is_18 :
  ∀ a b c : ℝ,
    let f := second_degree_polynomial a b c
    f 2 = 1 ∧ f 4 = 2 ∧ f 8 = 3 →
    -(b / a) = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_18_l144_14465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l144_14423

-- Part 1
theorem problem_part1 : 
  Real.sqrt (25 / 9) - (8 / 27) ^ (1 / 3 : ℝ) - (Real.pi + Real.exp 1) ^ (0 : ℝ) + (1 / 4) ^ (-(1 / 2 : ℝ)) = 2 := by
  sorry

-- Part 2
theorem problem_part2 : 
  2 * Real.log 5 + Real.log 4 + Real.log (Real.sqrt (Real.exp 1)) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l144_14423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_range_l144_14493

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.sin x) ^ 2

-- Define the angle α
noncomputable def α : ℝ := Real.arcsin (Real.sqrt 3 / 2)

-- Theorem 1
theorem f_at_alpha : f α = -3 := by sorry

-- Theorem 2
theorem f_range (x : ℝ) (h : x ∈ Set.Icc (-π/6) (π/3)) : 
  f x ∈ Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_range_l144_14493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14406

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m - 1)*x - m * Real.log x

-- State the theorem
theorem f_properties :
  -- Part 1: Minimum value when m = 1
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧ 
  (∃ (x : ℝ), x > 0 ∧ f 1 x = 3/4 - Real.log 2) ∧
  -- Part 2: Condition for t
  (∀ (m t : ℝ), m > 2 ∧ m < 3 → 
    ((∀ (x : ℝ), x ≥ 1 ∧ x ≤ 3 → m*t - f m x < 1) → t ≤ 7/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l144_14433

open Real

theorem slope_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ < x₂) :
  let f := λ x => log x
  let k := (f x₂ - f x₁) / (x₂ - x₁)
  1 / x₂ < k ∧ k < 1 / x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_inequality_l144_14433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_oranges_l144_14461

theorem claire_oranges (liam_oranges : ℕ) (liam_price : ℚ) (claire_price : ℚ) (total_saved : ℚ) (claire_oranges : ℕ) :
  liam_oranges = 40 →
  liam_price = 5/4 →
  claire_price = 6/5 →
  total_saved = 86 →
  (liam_oranges : ℚ) * liam_price + (claire_oranges : ℚ) * claire_price = total_saved →
  claire_oranges = 30 :=
by
  sorry

#check claire_oranges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_oranges_l144_14461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorine_atomic_weight_l144_14418

/-- The atomic weight of chlorine in a compound with one aluminum atom and three chlorine atoms -/
noncomputable def atomic_weight_chlorine (molecular_weight : ℝ) (atomic_weight_aluminum : ℝ) : ℝ :=
  (molecular_weight - atomic_weight_aluminum) / 3

/-- Theorem stating that the atomic weight of chlorine is (132 - 26.98) / 3 given the conditions -/
theorem chlorine_atomic_weight :
  let molecular_weight : ℝ := 132
  let atomic_weight_aluminum : ℝ := 26.98
  atomic_weight_chlorine molecular_weight atomic_weight_aluminum = (132 - 26.98) / 3 := by
  sorry

-- Use #eval only for computable functions
def approx_atomic_weight_chlorine : Float :=
  (132 - 26.98) / 3

#eval approx_atomic_weight_chlorine

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chlorine_atomic_weight_l144_14418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_for_circleominus_l144_14420

-- Define the ⊖ operation
def circleominus : ℝ → ℝ → ℝ := sorry

-- State the property of ⊖
axiom circleominus_property : ∀ x y z : ℝ, 
  (circleominus x y) + (circleominus y z) + (circleominus z x) = 0

-- Theorem statement
theorem exists_function_for_circleominus : 
  ∃ f : ℝ → ℝ, ∀ x y : ℝ, circleominus x y = f x - f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_for_circleominus_l144_14420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadrilateral_area_l144_14437

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the domain of g
variable (x₁ x₂ x₃ x₄ : ℝ)

-- Define the area of the original quadrilateral
variable (A : ℝ)

-- Define a function to calculate the area of a quadrilateral
noncomputable def area_quadrilateral (f : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem transformed_quadrilateral_area
  (h1 : Set.range g = {g x₁, g x₂, g x₃, g x₄})
  (h2 : A = 50)
  (h3 : Set.range (fun x ↦ 3 * g (3 * x)) = {3 * g x₁, 3 * g x₂, 3 * g x₃, 3 * g x₄}) :
  ∃ (A' : ℝ), A' = A ∧ A' = area_quadrilateral (fun x ↦ 3 * g (3 * x)) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadrilateral_area_l144_14437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l144_14477

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The theorem stating the ratio of areas of two specific triangles -/
theorem triangle_area_ratio : 
  let triangle_p_area := triangle_area 18 18 12
  let triangle_q_area := triangle_area 18 18 30
  triangle_p_area / triangle_q_area = 102 / 149.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l144_14477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l144_14408

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ my_circle p.1 p.2}

-- Theorem statement
theorem intersection_distance :
  ∃ (C D : ℝ × ℝ), C ∈ intersection_points ∧ D ∈ intersection_points ∧
    C ≠ D ∧ Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l144_14408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l144_14496

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity_relation (a : ℝ) :
  a > 1 →
  let e₁ := eccentricity a 1
  let e₂ := eccentricity 2 1
  e₂ = Real.sqrt 3 * e₁ →
  a = 2 * Real.sqrt 3 / 3 := by
  sorry

#check ellipse_eccentricity_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l144_14496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_racer_can_complete_loop_l144_14402

/-- Represents a car on the circular highway -/
structure Car where
  position : ℝ  -- Position on the highway (in km)
  fuel : ℝ      -- Amount of fuel (in km that can be covered)

/-- Represents the circular highway scenario -/
structure CircularHighway where
  length : ℝ
  cars : List Car
  totalFuel : ℝ

/-- Helper function to check if the racer has enough fuel at a given distance -/
def racer_has_enough_fuel (highway : CircularHighway) (start : ℕ) (distance : ℝ) : Prop :=
  sorry

/-- Theorem: A racer can complete a loop on the circular highway -/
theorem racer_can_complete_loop (highway : CircularHighway)
  (h_length : highway.length = 300)
  (h_fuel : highway.totalFuel ≥ 301)
  (h_cars : highway.cars.length > 0) :
  ∃ (start : ℕ), ∀ (distance : ℝ),
    0 ≤ distance ∧ distance ≤ highway.length →
    racer_has_enough_fuel highway start distance :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_racer_can_complete_loop_l144_14402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l144_14451

/-- The function f(x) = 1 + ln(x) - ax^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x - a * x^2

/-- Theorem: For all x > 0 and any real a, xf(x) < 2/e^2 * e^x + x - ax^3 -/
theorem f_inequality (a : ℝ) (x : ℝ) (h : x > 0) :
  x * f a x < 2 / Real.exp 2 * Real.exp x + x - a * x^3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l144_14451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14484

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4)

theorem f_properties :
  -- Smallest positive period is π
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧
    (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  -- Intervals of monotonic increase
  (∀ k : ℤ, ∀ x y : ℝ, -3*Real.pi/8 + k*Real.pi ≤ x ∧ x < y ∧ y ≤ Real.pi/8 + k*Real.pi → f x < f y) ∧
  -- Maximum and minimum values in [-π/8, π/2]
  (∀ x : ℝ, -Real.pi/8 ≤ x ∧ x ≤ Real.pi/2 → f x ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, -Real.pi/8 ≤ x ∧ x ≤ Real.pi/2 → f x ≥ -1) ∧
  f (Real.pi/8) = Real.sqrt 2 ∧
  f (Real.pi/2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l144_14484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l144_14416

/-- A set of points on a plane where each point is the midpoint of two other points in the set -/
def MidpointSet (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p, p ∈ M → ∃ q r, q ∈ M ∧ r ∈ M ∧ p = (q + r) / 2

/-- Theorem: If M is a midpoint set, then M is infinite -/
theorem midpoint_set_infinite (M : Set (ℝ × ℝ)) (h : MidpointSet M) : Set.Infinite M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_set_infinite_l144_14416
