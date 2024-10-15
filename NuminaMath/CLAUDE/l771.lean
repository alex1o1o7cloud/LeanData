import Mathlib

namespace NUMINAMATH_CALUDE_annes_height_l771_77140

/-- Proves that Anne's height is 80 cm given the relationships between heights of Anne, her sister, and Bella -/
theorem annes_height (sister_height : ℝ) (anne_height : ℝ) (bella_height : ℝ) : 
  anne_height = 2 * sister_height →
  bella_height = 3 * anne_height →
  bella_height - sister_height = 200 →
  anne_height = 80 := by
sorry

end NUMINAMATH_CALUDE_annes_height_l771_77140


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_from_equation_l771_77107

/-- Given a triangle ABC with side lengths a, b, and c satisfying the equation
    √(c² - a² - b²) + |a - b| = 0, prove that ABC is an isosceles right triangle. -/
theorem isosceles_right_triangle_from_equation 
  (a b c : ℝ) 
  (triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h : Real.sqrt (c^2 - a^2 - b^2) + |a - b| = 0) : 
  a = b ∧ c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_from_equation_l771_77107


namespace NUMINAMATH_CALUDE_joe_first_lift_weight_l771_77102

theorem joe_first_lift_weight (first_lift second_lift : ℝ) 
  (total_weight : first_lift + second_lift = 800)
  (lift_relation : 3 * first_lift = 2 * second_lift + 450) :
  first_lift = 410 := by
sorry

end NUMINAMATH_CALUDE_joe_first_lift_weight_l771_77102


namespace NUMINAMATH_CALUDE_divided_hexagon_areas_l771_77177

/-- Represents a regular hexagon divided by four diagonals -/
structure DividedHexagon where
  /-- The area of the central quadrilateral -/
  quadrilateralArea : ℝ
  /-- The areas of the six triangles -/
  triangleAreas : Fin 6 → ℝ

/-- Theorem about the areas of triangles in a divided regular hexagon -/
theorem divided_hexagon_areas (h : DividedHexagon) 
  (hq : h.quadrilateralArea = 1.8) : 
  (h.triangleAreas 0 = 1.2 ∧ 
   h.triangleAreas 1 = 1.2 ∧ 
   h.triangleAreas 2 = 0.6 ∧ 
   h.triangleAreas 3 = 0.6 ∧ 
   h.triangleAreas 4 = 1.2 ∧ 
   h.triangleAreas 5 = 0.6) := by
  sorry

end NUMINAMATH_CALUDE_divided_hexagon_areas_l771_77177


namespace NUMINAMATH_CALUDE_parabola_minimum_distance_l771_77149

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with slope k passing through (2, 0) -/
def line (k x y : ℝ) : Prop := y = k*(x - 2)

/-- Distance between two x-coordinates on the parabola -/
def distance_on_parabola (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem parabola_minimum_distance (k1 k2 : ℝ) :
  k1 * k2 = -2 →
  ∃ (xA xC xB xD : ℝ),
    parabola xA (k1*(xA - 2)) ∧
    parabola xC (k1*(xC - 2)) ∧
    parabola xB (k2*(xB - 2)) ∧
    parabola xD (k2*(xD - 2)) ∧
    (∀ x1A x1C x2B x2D : ℝ,
      parabola x1A (k1*(x1A - 2)) →
      parabola x1C (k1*(x1C - 2)) →
      parabola x2B (k2*(x2B - 2)) →
      parabola x2D (k2*(x2D - 2)) →
      distance_on_parabola xA xC + distance_on_parabola xB xD ≤
      distance_on_parabola x1A x1C + distance_on_parabola x2B x2D) ∧
    distance_on_parabola xA xC + distance_on_parabola xB xD = 24 :=
by sorry


end NUMINAMATH_CALUDE_parabola_minimum_distance_l771_77149


namespace NUMINAMATH_CALUDE_base_6_addition_l771_77192

/-- Addition of two numbers in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 6 -/
def to_base_6 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 6 to base 10 -/
def from_base_6 (n : ℕ) : ℕ :=
  sorry

theorem base_6_addition :
  add_base_6 (from_base_6 52301) (from_base_6 34122) = from_base_6 105032 :=
sorry

end NUMINAMATH_CALUDE_base_6_addition_l771_77192


namespace NUMINAMATH_CALUDE_average_first_5_subjects_l771_77197

-- Define the given conditions
def total_subjects : ℕ := 6
def average_6_subjects : ℚ := 77
def marks_6th_subject : ℕ := 92

-- Define the theorem to prove
theorem average_first_5_subjects :
  let total_marks := average_6_subjects * total_subjects
  let marks_5_subjects := total_marks - marks_6th_subject
  (marks_5_subjects / (total_subjects - 1) : ℚ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_average_first_5_subjects_l771_77197


namespace NUMINAMATH_CALUDE_det_submatrix_l771_77164

theorem det_submatrix (a b c d : ℝ) :
  Matrix.det !![1, a, b; 2, c, d; 3, 0, 0] = 6 →
  Matrix.det !![a, b; c, d] = 2 := by
sorry

end NUMINAMATH_CALUDE_det_submatrix_l771_77164


namespace NUMINAMATH_CALUDE_indefinite_integral_arctg_sqrt_2x_minus_1_l771_77169

theorem indefinite_integral_arctg_sqrt_2x_minus_1 (x : ℝ) :
  HasDerivAt (fun x => x * Real.arctan (Real.sqrt (2 * x - 1)) - (1/2) * Real.sqrt (2 * x - 1))
             (Real.arctan (Real.sqrt (2 * x - 1)))
             x :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_arctg_sqrt_2x_minus_1_l771_77169


namespace NUMINAMATH_CALUDE_lost_ship_depth_l771_77101

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def depth_of_lost_ship (descent_rate : ℝ) (time_taken : ℝ) : ℝ :=
  descent_rate * time_taken

/-- Theorem: The depth of the lost ship is 3600 feet. -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 60
  let time_taken : ℝ := 60
  depth_of_lost_ship descent_rate time_taken = 3600 := by
sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l771_77101


namespace NUMINAMATH_CALUDE_functional_equation_solution_l771_77153

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that the only functions satisfying the equation are x - 1 or -x - 1 -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x - 1 ∨ f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l771_77153


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l771_77119

theorem quadratic_inequality_no_solution : ∀ x : ℝ, x^2 - 2*x + 3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l771_77119


namespace NUMINAMATH_CALUDE_no_natural_solution_for_equation_l771_77126

theorem no_natural_solution_for_equation : ¬∃ (a b : ℕ), a^2 - 3*b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_equation_l771_77126


namespace NUMINAMATH_CALUDE_g_is_max_g_symmetric_points_l771_77185

noncomputable def f (a x : ℝ) : ℝ := a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)

noncomputable def g (a : ℝ) : ℝ := 
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1/(2*a)
  else Real.sqrt 2

theorem g_is_max (a : ℝ) : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ g a := by sorry

theorem g_symmetric_points (a : ℝ) : 
  ((-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1) ↔ g a = g (1/a) := by sorry

end NUMINAMATH_CALUDE_g_is_max_g_symmetric_points_l771_77185


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l771_77144

theorem triangle_angle_proof (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- angles are positive
  A + B + C = π ∧ -- sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- sides are positive
  a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B → -- given condition
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l771_77144


namespace NUMINAMATH_CALUDE_jane_change_l771_77157

def skirt_price : ℕ := 13
def skirt_quantity : ℕ := 2
def blouse_price : ℕ := 6
def blouse_quantity : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_quantity + blouse_price * blouse_quantity

theorem jane_change : amount_paid - total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l771_77157


namespace NUMINAMATH_CALUDE_apollonius_circle_exists_l771_77155

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a tangency relation between two circles
def is_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

-- Theorem statement
theorem apollonius_circle_exists (S1 S2 S3 : Circle) :
  ∃ S : Circle, is_tangent S S1 ∧ is_tangent S S2 ∧ is_tangent S S3 :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_exists_l771_77155


namespace NUMINAMATH_CALUDE_rationalize_denominator_l771_77191

theorem rationalize_denominator : 7 / Real.sqrt 175 = Real.sqrt 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l771_77191


namespace NUMINAMATH_CALUDE_negation_of_proposition_l771_77183

theorem negation_of_proposition (P : ℕ → Prop) :
  (∀ m : ℕ, 4^m ≥ 4*m) ↔ ¬(∃ m : ℕ, 4^m < 4*m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l771_77183


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_five_l771_77160

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_five_l771_77160


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_2_3_5_7_l771_77100

theorem least_four_digit_divisible_by_2_3_5_7 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1050 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n)) ∧
  (1050 ≥ 1000) ∧
  (2 ∣ 1050) ∧ (3 ∣ 1050) ∧ (5 ∣ 1050) ∧ (7 ∣ 1050) :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_2_3_5_7_l771_77100


namespace NUMINAMATH_CALUDE_prop_p_false_prop_q_true_l771_77141

-- Define the curve C
def curve_C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (25 - k) + p.2^2 / (k - 9) = 1}

-- Define what it means for a curve to be an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

-- Define what it means for a curve to be a hyperbola with foci on the x-axis
def is_hyperbola_x_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ S = {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

-- Theorem 1: Proposition p is false
theorem prop_p_false : ¬(∀ k : ℝ, 9 < k ∧ k < 25 → is_ellipse (curve_C k)) :=
  sorry

-- Theorem 2: Proposition q is true
theorem prop_q_true : ∀ k : ℝ, is_hyperbola_x_axis (curve_C k) → k < 9 :=
  sorry

end NUMINAMATH_CALUDE_prop_p_false_prop_q_true_l771_77141


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l771_77174

def is_in_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, 180 * (2 * n + 1) < α ∧ α < 180 * (2 * n + 1) + 90

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60) : 
  is_in_third_quadrant α :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l771_77174


namespace NUMINAMATH_CALUDE_polyhedron_volume_from_parallelepiped_l771_77111

/-- Given a parallelepiped with volume V, the volume of the polyhedron formed by
    connecting the centers of its faces is 1/6 * V -/
theorem polyhedron_volume_from_parallelepiped (V : ℝ) (V_pos : V > 0) :
  ∃ (polyhedron_volume : ℝ),
    polyhedron_volume = (1 / 6 : ℝ) * V ∧
    polyhedron_volume > 0 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_from_parallelepiped_l771_77111


namespace NUMINAMATH_CALUDE_circle_radius_implies_c_l771_77195

/-- Given a circle with equation x^2 + 6x + y^2 - 4y + c = 0 and radius 6, prove that c = -23 -/
theorem circle_radius_implies_c (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 - 4*y + c = 0 → (x+3)^2 + (y-2)^2 = 36) → 
  c = -23 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_implies_c_l771_77195


namespace NUMINAMATH_CALUDE_next_four_valid_numbers_l771_77143

/-- Represents a bag of milk with a unique number -/
structure BagOfMilk where
  number : Nat
  h_number : number ≤ 850

/-- Checks if a number is valid for bag selection -/
def isValidNumber (n : Nat) : Bool :=
  n ≥ 1 ∧ n ≤ 850

/-- Selects the next valid numbers from a given sequence -/
def selectNextValidNumbers (sequence : List Nat) (count : Nat) : List Nat :=
  sequence.filter isValidNumber |>.take count

theorem next_four_valid_numbers 
  (sequence : List Nat)
  (h_sequence : sequence = [614, 593, 379, 242, 203, 722, 104, 887, 088]) :
  selectNextValidNumbers (sequence.drop 4) 4 = [203, 722, 104, 088] := by
  sorry

#eval selectNextValidNumbers [614, 593, 379, 242, 203, 722, 104, 887, 088] 4

end NUMINAMATH_CALUDE_next_four_valid_numbers_l771_77143


namespace NUMINAMATH_CALUDE_ellipse_properties_l771_77123

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line l with slope m passing through the right focus
def line_l (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the area of a triangle given three points
def triangle_area (A B P : ℝ × ℝ) : ℝ := sorry

-- Define the area of the incircle of a triangle
def incircle_area (A B F : ℝ × ℝ) : ℝ := sorry

theorem ellipse_properties :
  ∃ (P₁ P₂ : ℝ × ℝ),
    ellipse P₁.1 P₁.2 ∧ 
    ellipse P₂.1 P₂.2 ∧
    P₁ ≠ P₂ ∧
    (∀ (A B : ℝ × ℝ), 
      ellipse A.1 A.2 ∧ 
      ellipse B.1 B.2 ∧ 
      line_l 1 A.1 A.2 ∧ 
      line_l 1 B.1 B.2 →
      triangle_area A B P₁ = (2 * Real.sqrt 5 - 2) / 3 ∧
      triangle_area A B P₂ = (2 * Real.sqrt 5 - 2) / 3) ∧
    (∀ (P : ℝ × ℝ),
      ellipse P.1 P.2 ∧ 
      P ≠ P₁ ∧ 
      P ≠ P₂ →
      triangle_area A B P ≠ (2 * Real.sqrt 5 - 2) / 3) ∧
    (∃ (A B : ℝ × ℝ) (m : ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      line_l m A.1 A.2 ∧
      line_l m B.1 B.2 ∧
      incircle_area A B (-1, 0) = π / 8 ∧
      (∀ (C D : ℝ × ℝ) (n : ℝ),
        ellipse C.1 C.2 ∧
        ellipse D.1 D.2 ∧
        line_l n C.1 C.2 ∧
        line_l n D.1 D.2 →
        incircle_area C D (-1, 0) ≤ π / 8)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l771_77123


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l771_77156

/-- Definition of Line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of Line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (m : ℝ), ∀ (k x y : ℝ), l₁ k x y ↔ l₂ k (m * x) y

/-- Theorem stating that for l₁ and l₂ to be parallel, k must be 2, 3, or 6 -/
theorem parallel_lines_k_values :
  parallel l₁ l₂ ↔ (∃ k : ℝ, k = 2 ∨ k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l771_77156


namespace NUMINAMATH_CALUDE_pet_store_birds_l771_77131

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 7

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 4

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 3

/-- The number of cockatiels in each cage -/
def cockatiels_per_cage : ℕ := 2

/-- The number of canaries in each cage -/
def canaries_per_cage : ℕ := 1

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage)

theorem pet_store_birds : total_birds = 70 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l771_77131


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l771_77113

/-- Proves that the incorrect height of a student is 151 cm given the conditions of the problem -/
theorem incorrect_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧ 
  initial_avg = 175 ∧ 
  actual_height = 136 ∧ 
  actual_avg = 174.5 → 
  ∃ (incorrect_height : ℝ), 
    incorrect_height = 151 ∧ 
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l771_77113


namespace NUMINAMATH_CALUDE_percentage_of_720_is_356_4_l771_77112

theorem percentage_of_720_is_356_4 : 
  let whole : ℝ := 720
  let part : ℝ := 356.4
  let percentage : ℝ := (part / whole) * 100
  percentage = 49.5 := by sorry

end NUMINAMATH_CALUDE_percentage_of_720_is_356_4_l771_77112


namespace NUMINAMATH_CALUDE_tournament_ordered_victories_l771_77161

/-- A round-robin tournament with 2^n players -/
def Tournament (n : ℕ) := Fin (2^n)

/-- The result of a match between two players -/
def Defeats (t : Tournament n) : Tournament n → Tournament n → Prop := sorry

/-- The property that player i defeats player j if and only if i < j -/
def OrderedVictories (t : Tournament n) (s : Fin (n+1) → Tournament n) : Prop :=
  ∀ i j, i < j → Defeats t (s i) (s j)

/-- The main theorem: In any tournament of 2^n players, there exists an ordered sequence of n+1 players -/
theorem tournament_ordered_victories (n : ℕ) :
  ∀ t : Tournament n, ∃ s : Fin (n+1) → Tournament n, OrderedVictories t s := by
  sorry

end NUMINAMATH_CALUDE_tournament_ordered_victories_l771_77161


namespace NUMINAMATH_CALUDE_min_value_of_exp_minus_x_l771_77115

theorem min_value_of_exp_minus_x :
  ∃ (x : ℝ), ∀ (y : ℝ), Real.exp y - y ≥ Real.exp x - x ∧ Real.exp x - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exp_minus_x_l771_77115


namespace NUMINAMATH_CALUDE_calculator_display_l771_77189

/-- The special key function -/
def f (x : ℚ) : ℚ := 1 / (1 - x)

/-- Applies the function n times to the initial value -/
def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem calculator_display : iterate_f 120 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculator_display_l771_77189


namespace NUMINAMATH_CALUDE_expression_evaluation_l771_77199

theorem expression_evaluation : -(-2) + 2 * Real.cos (60 * π / 180) + (-1/8)⁻¹ + (Real.pi - 3.14)^0 = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l771_77199


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l771_77190

/-- Given that the solution set of ax^2 - 5x + b > 0 is {x | -3 < x < 2},
    prove that the solution set of bx^2 - 5x + a > 0 is {x | x < -1/3 or x > 1/2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ -3 < x ∧ x < 2) :
  ∀ x : ℝ, b*x^2 - 5*x + a > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l771_77190


namespace NUMINAMATH_CALUDE_fraction_equality_l771_77132

theorem fraction_equality : (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l771_77132


namespace NUMINAMATH_CALUDE_watch_cost_price_l771_77171

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp * 0.79 = cp * 1.04 - 140) ∧ 
  cp = 560 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l771_77171


namespace NUMINAMATH_CALUDE_derivative_symmetry_l771_77137

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  (4 * a + 2 * b = 2) → 
  (fun (x : ℝ) => 4 * a * x^3 + 2 * b * x) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l771_77137


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l771_77128

/-- A parabola defined by y = 2(x+1)² + c -/
structure Parabola where
  c : ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * (x + 1)^2 + p.c

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem y_coordinate_relationship (p : Parabola) 
  (A : PointOnParabola p) (B : PointOnParabola p) (C : PointOnParabola p)
  (hA : A.x = -2) (hB : B.x = 1) (hC : C.x = 2) :
  C.y > B.y ∧ B.y > A.y := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l771_77128


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l771_77188

/-- Given two congruent cylinders with radius 10 inches and height 4 inches,
    where the radius of one cylinder and the height of the other are increased by x inches,
    prove that the only nonzero solution for equal volumes is x = 5. -/
theorem cylinder_volume_equality (x : ℝ) (hx : x ≠ 0) :
  π * (10 + x)^2 * 4 = π * 100 * (4 + x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l771_77188


namespace NUMINAMATH_CALUDE_club_membership_increase_l771_77147

theorem club_membership_increase (current_members : ℕ) (h : current_members = 10) : 
  (2 * current_members + 5) - current_members = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_increase_l771_77147


namespace NUMINAMATH_CALUDE_boyGirlRatio_in_example_college_l771_77122

/-- Represents the number of students in a college -/
structure College where
  total : ℕ
  girls : ℕ
  boys : ℕ
  total_eq : total = girls + boys

/-- The ratio of boys to girls in a college -/
def boyGirlRatio (c : College) : ℚ :=
  c.boys / c.girls

theorem boyGirlRatio_in_example_college :
  ∃ c : College, c.total = 600 ∧ c.girls = 200 ∧ boyGirlRatio c = 2 := by
  sorry

end NUMINAMATH_CALUDE_boyGirlRatio_in_example_college_l771_77122


namespace NUMINAMATH_CALUDE_megan_problem_solving_rate_l771_77172

theorem megan_problem_solving_rate 
  (math_problems : ℕ) 
  (spelling_problems : ℕ) 
  (total_hours : ℕ) 
  (h1 : math_problems = 36)
  (h2 : spelling_problems = 28)
  (h3 : total_hours = 8) :
  (math_problems + spelling_problems) / total_hours = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_megan_problem_solving_rate_l771_77172


namespace NUMINAMATH_CALUDE_positive_integer_from_operations_l771_77105

def integers : Set ℚ := {0, -3, 5, -100, 2008, -1}
def fractions : Set ℚ := {1/2, -1/3, 1/5, -3/2, -1/100}

theorem positive_integer_from_operations : ∃ (a b : ℚ) (c d : ℚ) (op1 op2 : ℚ → ℚ → ℚ),
  a ∈ integers ∧ b ∈ integers ∧ c ∈ fractions ∧ d ∈ fractions ∧
  (op1 = (· + ·) ∨ op1 = (· - ·) ∨ op1 = (· * ·) ∨ op1 = (· / ·)) ∧
  (op2 = (· + ·) ∨ op2 = (· - ·) ∨ op2 = (· * ·) ∨ op2 = (· / ·)) ∧
  ∃ (n : ℕ), (op2 (op1 a b) (op1 c d) : ℚ) = n := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_from_operations_l771_77105


namespace NUMINAMATH_CALUDE_no_rational_solution_l771_77162

theorem no_rational_solution : ¬∃ (x y z : ℚ), 
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^5 + 2*y^5 + 5*z^5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l771_77162


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l771_77125

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = -1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (∃ ε > 0, ∀ x' y' : ℝ, hyperbola x' y' ∧ x'^2 + y'^2 > 1/ε^2 →
    |y' - (Real.sqrt 2 * x')| < ε ∨ |y' - (-Real.sqrt 2 * x')| < ε) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l771_77125


namespace NUMINAMATH_CALUDE_equation_proof_l771_77118

theorem equation_proof : 441 + 2 * 21 * 7 + 49 = 784 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l771_77118


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_seven_l771_77158

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Theorem for part (1)
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem for part (2)
theorem a_greater_than_seven (h : A ⊆ C a) : a > 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_a_greater_than_seven_l771_77158


namespace NUMINAMATH_CALUDE_product_expansion_l771_77103

theorem product_expansion (x : ℝ) : (3 * x + 4) * (2 * x^2 + 3 * x + 6) = 6 * x^3 + 17 * x^2 + 30 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l771_77103


namespace NUMINAMATH_CALUDE_min_distance_squared_l771_77193

theorem min_distance_squared (a b c d : ℝ) : 
  b = a - 2 * Real.exp a → 
  c + d = 4 → 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (a - x)^2 + (b - y)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l771_77193


namespace NUMINAMATH_CALUDE_triangle_abc_solutions_l771_77129

theorem triangle_abc_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    Real.sin A / a = Real.sin B / b ∧
    Real.sin C / c = Real.sin B / b :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_solutions_l771_77129


namespace NUMINAMATH_CALUDE_equation_solutions_l771_77134

theorem equation_solutions : 
  let f (x : ℝ) := (x - 2) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 5) * (x - 2)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  let h (x : ℝ) := f x / g x
  ∀ x : ℝ, h x = 1 ↔ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l771_77134


namespace NUMINAMATH_CALUDE_twoDigitNumberRepresentation_l771_77178

/-- Represents a two-digit number with x in the tens place and 5 in the ones place -/
def twoDigitNumber (x : ℕ) : ℕ := 10 * x + 5

/-- Proves that a two-digit number with x in the tens place and 5 in the ones place
    can be represented as 10x + 5 -/
theorem twoDigitNumberRepresentation (x : ℕ) (h : x < 10) :
  twoDigitNumber x = 10 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_twoDigitNumberRepresentation_l771_77178


namespace NUMINAMATH_CALUDE_charlie_seashells_l771_77138

theorem charlie_seashells (c e : ℕ) : 
  c = e + 10 →  -- Charlie collected 10 more seashells than Emily
  e = c / 3 →   -- Emily collected one-third the number of seashells Charlie collected
  c = 15 :=     -- Charlie collected 15 seashells
by sorry

end NUMINAMATH_CALUDE_charlie_seashells_l771_77138


namespace NUMINAMATH_CALUDE_two_integers_sum_l771_77168

theorem two_integers_sum (a b : ℕ+) : a - b = 4 → a * b = 96 → a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l771_77168


namespace NUMINAMATH_CALUDE_min_value_product_l771_77166

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2) :
  (x + y) * (y + 3 * z) * (2 * x * z + 1) ≥ 16 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l771_77166


namespace NUMINAMATH_CALUDE_equal_prod_of_divisors_implies_equal_numbers_l771_77196

/-- The sum of positive divisors of a natural number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The product of positive divisors of a natural number -/
def prod_of_divisors (n : ℕ) : ℕ := n ^ ((sum_of_divisors n).div 2)

/-- The number of positive divisors of a natural number -/
def num_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all positive divisors of two natural numbers are equal, 
    then the two numbers are equal -/
theorem equal_prod_of_divisors_implies_equal_numbers (n m : ℕ) : 
  prod_of_divisors n = prod_of_divisors m → n = m := by sorry

end NUMINAMATH_CALUDE_equal_prod_of_divisors_implies_equal_numbers_l771_77196


namespace NUMINAMATH_CALUDE_complex_division_result_l771_77104

theorem complex_division_result : 
  let i : ℂ := Complex.I
  (1 + i) / (-2 * i) = -1/2 + 1/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l771_77104


namespace NUMINAMATH_CALUDE_coin_jar_theorem_l771_77151

/-- Represents the number of coins added or removed in each hour. 
    Positive numbers represent additions, negative numbers represent removals. -/
def coin_changes : List Int := [20, 30, 30, 40, -20, 50, 60, -15, 70, -25]

/-- The total number of hours -/
def total_hours : Nat := 10

/-- Calculates the final number of coins in the jar -/
def final_coin_count (changes : List Int) : Int :=
  changes.sum

/-- Theorem stating that the final number of coins in the jar is 240 -/
theorem coin_jar_theorem : 
  final_coin_count coin_changes = 240 := by
  sorry

end NUMINAMATH_CALUDE_coin_jar_theorem_l771_77151


namespace NUMINAMATH_CALUDE_pink_balls_count_l771_77109

/-- The number of pink balls initially in the bag -/
def initial_pink_balls : ℕ := 23

/-- The number of green balls initially in the bag -/
def initial_green_balls : ℕ := 9

/-- The number of green balls added -/
def added_green_balls : ℕ := 14

theorem pink_balls_count :
  (initial_green_balls + added_green_balls = initial_pink_balls) ∧
  (initial_green_balls + added_green_balls : ℚ) / initial_pink_balls = 1 := by
  sorry

end NUMINAMATH_CALUDE_pink_balls_count_l771_77109


namespace NUMINAMATH_CALUDE_distance_between_centers_l771_77135

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the centers of the circles
def center_M : ℝ × ℝ := (0, 0)
def center_N : ℝ × ℝ := (0, 2)

-- State the theorem
theorem distance_between_centers :
  let (x₁, y₁) := center_M
  let (x₂, y₂) := center_N
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l771_77135


namespace NUMINAMATH_CALUDE_matrix_addition_problem_l771_77194

theorem matrix_addition_problem : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 2; 7, -10]
  A + B = !![-2, -1; 7, -5] := by
sorry

end NUMINAMATH_CALUDE_matrix_addition_problem_l771_77194


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l771_77127

theorem cos_30_minus_cos_60 : Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l771_77127


namespace NUMINAMATH_CALUDE_sweater_cost_l771_77173

def original_savings : ℚ := 80

def makeup_fraction : ℚ := 3/4

theorem sweater_cost :
  let makeup_cost : ℚ := makeup_fraction * original_savings
  let sweater_cost : ℚ := original_savings - makeup_cost
  sweater_cost = 20 := by sorry

end NUMINAMATH_CALUDE_sweater_cost_l771_77173


namespace NUMINAMATH_CALUDE_short_story_pages_approx_l771_77145

/-- Calculates the number of pages in each short story --/
def pages_per_short_story (stories_per_week : ℕ) (weeks : ℕ) (reams : ℕ) 
  (sheets_per_ream : ℕ) (pages_per_sheet : ℕ) : ℚ :=
  let total_sheets := reams * sheets_per_ream
  let total_pages := total_sheets * pages_per_sheet
  let total_stories := stories_per_week * weeks
  (total_pages : ℚ) / total_stories

theorem short_story_pages_approx : 
  let result := pages_per_short_story 3 12 3 500 2
  ∃ ε > 0, |result - 83.33| < ε := by
  sorry

end NUMINAMATH_CALUDE_short_story_pages_approx_l771_77145


namespace NUMINAMATH_CALUDE_parallel_EX_AP_l771_77159

noncomputable section

-- Define the points on a complex plane
variable (a b c p h e q r x : ℂ)

-- Define the triangle ABC on the unit circle
def on_unit_circle (z : ℂ) : Prop := Complex.abs z = 1

-- Define the orthocenter condition
def is_orthocenter (a b c h : ℂ) : Prop := a + b + c = h

-- Define the circumcircle condition
def on_circumcircle (a b c p : ℂ) : Prop := on_unit_circle p

-- Define the foot of altitude condition
def is_foot_of_altitude (a b c e : ℂ) : Prop :=
  e = (1 / 2) * (a + b + c - (a * c) / b)

-- Define parallelogram conditions
def is_parallelogram_PAQB (a b p q : ℂ) : Prop := q = a + b - p
def is_parallelogram_PARC (a c p r : ℂ) : Prop := r = a + c - p

-- Define the intersection point condition
def is_intersection (a q h r x : ℂ) : Prop :=
  ∃ t₁ t₂ : ℝ, x = a + t₁ * (q - a) ∧ x = h + t₂ * (r - h)

-- Main theorem
theorem parallel_EX_AP (a b c p h e q r x : ℂ) 
  (h_circle : on_unit_circle a ∧ on_unit_circle b ∧ on_unit_circle c)
  (h_orthocenter : is_orthocenter a b c h)
  (h_circumcircle : on_circumcircle a b c p)
  (h_foot : is_foot_of_altitude a b c e)
  (h_para1 : is_parallelogram_PAQB a b p q)
  (h_para2 : is_parallelogram_PARC a c p r)
  (h_intersect : is_intersection a q h r x) :
  ∃ k : ℂ, e - x = k * (a - p) :=
sorry

end NUMINAMATH_CALUDE_parallel_EX_AP_l771_77159


namespace NUMINAMATH_CALUDE_xyz_equals_seven_l771_77179

theorem xyz_equals_seven (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
  x * y * z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_seven_l771_77179


namespace NUMINAMATH_CALUDE_geometric_sequence_max_value_l771_77187

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) 
  (h_geo : is_positive_geometric_sequence a)
  (h_condition : a 3 * a 6 + a 2 * a 7 = 2 * Real.exp 4) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y ≤ 4) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y = 4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_value_l771_77187


namespace NUMINAMATH_CALUDE_calculate_expression_l771_77110

theorem calculate_expression : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 5^2 - 7^2) = 1099511627775 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l771_77110


namespace NUMINAMATH_CALUDE_exam_mean_score_l771_77167

/-- Given an exam where a score of 58 is 2 standard deviations below the mean
    and a score of 98 is 3 standard deviations above the mean,
    prove that the mean score is 74. -/
theorem exam_mean_score (mean std_dev : ℝ) 
    (h1 : 58 = mean - 2 * std_dev)
    (h2 : 98 = mean + 3 * std_dev) : 
  mean = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l771_77167


namespace NUMINAMATH_CALUDE_inequality_proof_l771_77170

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l771_77170


namespace NUMINAMATH_CALUDE_sqrt_expression_equivalence_l771_77139

theorem sqrt_expression_equivalence (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - x * (1 - 1 / (x + 1)))) = abs x :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equivalence_l771_77139


namespace NUMINAMATH_CALUDE_students_with_b_in_dawsons_class_l771_77154

theorem students_with_b_in_dawsons_class 
  (charles_total : ℕ) 
  (charles_b : ℕ) 
  (dawson_total : ℕ) 
  (h1 : charles_total = 20)
  (h2 : charles_b = 12)
  (h3 : dawson_total = 30)
  (h4 : charles_b * dawson_total = charles_total * dawson_b) :
  dawson_b = 18 := by
    sorry

#check students_with_b_in_dawsons_class

end NUMINAMATH_CALUDE_students_with_b_in_dawsons_class_l771_77154


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l771_77150

theorem sufficient_not_necessary_implies_necessary_not_sufficient 
  (A B : Prop) (h : (A → B) ∧ ¬(B → A)) : 
  ((¬B → ¬A) ∧ ¬(¬A → ¬B)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l771_77150


namespace NUMINAMATH_CALUDE_sin_75_cos_75_eq_half_l771_77163

theorem sin_75_cos_75_eq_half : 2 * Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_eq_half_l771_77163


namespace NUMINAMATH_CALUDE_mistaken_division_correction_l771_77180

theorem mistaken_division_correction (N : ℕ) : 
  N % 23 = 17 ∧ N / 23 = 3 → (N / 32) + (N % 32) = 24 := by
sorry

end NUMINAMATH_CALUDE_mistaken_division_correction_l771_77180


namespace NUMINAMATH_CALUDE_marble_difference_prove_marble_difference_l771_77175

/-- The difference in marbles between Ed and Doug after a series of events -/
theorem marble_difference : ℤ → Prop :=
  fun initial_difference =>
    ∀ (doug_initial : ℤ) (doug_lost : ℤ) (susan_found : ℤ),
      initial_difference = 22 →
      doug_lost = 8 →
      susan_found = 5 →
      (doug_initial + initial_difference + susan_found) - (doug_initial - doug_lost) = 35

/-- Proof of the marble difference theorem -/
theorem prove_marble_difference : marble_difference 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_prove_marble_difference_l771_77175


namespace NUMINAMATH_CALUDE_kim_total_points_l771_77148

/-- Calculates the total points in a math contest with three rounds -/
def totalPoints (easyPoints averagePoints hardPoints : ℕ) 
                (easyCorrect averageCorrect hardCorrect : ℕ) : ℕ :=
  easyPoints * easyCorrect + averagePoints * averageCorrect + hardPoints * hardCorrect

/-- Theorem: Kim's total points in the contest -/
theorem kim_total_points :
  totalPoints 2 3 5 6 2 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_kim_total_points_l771_77148


namespace NUMINAMATH_CALUDE_children_with_vip_seats_l771_77165

/-- Proves the number of children with VIP seats in a concert hall -/
theorem children_with_vip_seats
  (total_attendees : ℕ)
  (children_percentage : ℚ)
  (vip_children_percentage : ℚ)
  (h1 : total_attendees = 400)
  (h2 : children_percentage = 75 / 100)
  (h3 : vip_children_percentage = 20 / 100) :
  ⌊(total_attendees : ℚ) * children_percentage * vip_children_percentage⌋ = 60 := by
  sorry

#check children_with_vip_seats

end NUMINAMATH_CALUDE_children_with_vip_seats_l771_77165


namespace NUMINAMATH_CALUDE_expression_evaluation_l771_77116

theorem expression_evaluation (a b : ℚ) (ha : a = 7) (hb : b = 5) :
  3 * (a^3 + b^3) / (a^2 - a*b + b^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l771_77116


namespace NUMINAMATH_CALUDE_bob_questions_proof_l771_77184

def question_rate (hour : Nat) : Nat :=
  match hour with
  | 0 => 13
  | n + 1 => 2 * question_rate n

def total_questions (hours : Nat) : Nat :=
  match hours with
  | 0 => 0
  | n + 1 => question_rate n + total_questions n

theorem bob_questions_proof :
  total_questions 3 = 91 :=
by sorry

end NUMINAMATH_CALUDE_bob_questions_proof_l771_77184


namespace NUMINAMATH_CALUDE_repeating_decimal_incorrect_expression_l771_77114

/-- Represents a repeating decimal -/
structure RepeatingDecimal where
  P : ℕ  -- non-repeating part
  Q : ℕ  -- repeating part
  r : ℕ  -- number of digits in P
  s : ℕ  -- number of digits in Q

/-- The theorem stating that the given expression is not always true for repeating decimals -/
theorem repeating_decimal_incorrect_expression (D : RepeatingDecimal) :
  ¬ (∀ (D : RepeatingDecimal), 10^D.r * (10^D.s - 1) * (D.P / 10^D.r + D.Q / (10^D.r * (10^D.s - 1))) = D.Q * (D.P - 1)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_incorrect_expression_l771_77114


namespace NUMINAMATH_CALUDE_debby_jogged_nine_km_on_wednesday_l771_77117

/-- The distance Debby jogged on Monday in kilometers -/
def monday_distance : ℕ := 2

/-- The distance Debby jogged on Tuesday in kilometers -/
def tuesday_distance : ℕ := 5

/-- The total distance Debby jogged over three days in kilometers -/
def total_distance : ℕ := 16

/-- The distance Debby jogged on Wednesday in kilometers -/
def wednesday_distance : ℕ := total_distance - (monday_distance + tuesday_distance)

theorem debby_jogged_nine_km_on_wednesday :
  wednesday_distance = 9 := by sorry

end NUMINAMATH_CALUDE_debby_jogged_nine_km_on_wednesday_l771_77117


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l771_77181

theorem largest_solution_of_equation (x : ℝ) :
  (x^2 - x - 72) / (x - 9) = 5 / (x + 4) →
  x ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l771_77181


namespace NUMINAMATH_CALUDE_parallel_planes_lines_relationship_l771_77182

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the positional relationships between lines
variable (is_parallel : Line → Line → Prop)
variable (is_skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_lines_relationship 
  (α β : Plane) (a b : Line) 
  (h1 : parallel α β) 
  (h2 : contained_in a α) 
  (h3 : contained_in b β) : 
  is_parallel a b ∨ is_skew a b :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_relationship_l771_77182


namespace NUMINAMATH_CALUDE_dividend_divisor_change_l771_77186

theorem dividend_divisor_change (a b : ℝ) (h : b ≠ 0) :
  (11 * a) / (10 * b) ≠ a / b :=
sorry

end NUMINAMATH_CALUDE_dividend_divisor_change_l771_77186


namespace NUMINAMATH_CALUDE_four_digit_count_l771_77130

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from 1000 to 9999 (inclusive) is equal to 9000 -/
theorem four_digit_count :
  count_four_digit_numbers = 9000 := by sorry

end NUMINAMATH_CALUDE_four_digit_count_l771_77130


namespace NUMINAMATH_CALUDE_vector_magnitude_l771_77152

/-- Given two vectors a and b in ℝ², prove that if a = (1, -1) and a + b = (3, 1), 
    then the magnitude of b is 2√2. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (1, -1) → a + b = (3, 1) → ‖b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l771_77152


namespace NUMINAMATH_CALUDE_second_frog_hops_l771_77146

theorem second_frog_hops :
  ∀ (h1 h2 h3 : ℕ),
  h1 = 4 * h2 →
  h2 = 2 * h3 →
  h1 + h2 + h3 = 99 →
  h2 = 18 := by
sorry

end NUMINAMATH_CALUDE_second_frog_hops_l771_77146


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l771_77124

/-- The equation 9x^2 - 16y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 9 * x^2 - 16 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l771_77124


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l771_77198

theorem complex_fraction_equals_i : (3 + 2*I) / (2 - 3*I) = I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l771_77198


namespace NUMINAMATH_CALUDE_equation_solutions_l771_77120

theorem equation_solutions :
  (∃ x : ℚ, (5 / (x + 1) = 1 / (x - 3)) ∧ x = 4) ∧
  (∃ x : ℚ, ((2 - x) / (x - 3) + 2 = 1 / (3 - x)) ∧ x = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l771_77120


namespace NUMINAMATH_CALUDE_lily_pad_growth_rate_l771_77176

/-- Represents the coverage of the lake by lily pads -/
def LakeCoverage := ℝ

/-- The time it takes for the lily pads to cover the entire lake -/
def fullCoverageTime : ℕ := 50

/-- The time it takes for the lily pads to cover half the lake -/
def halfCoverageTime : ℕ := 49

/-- The growth rate of the lily pad patch -/
def growthRate : ℝ → Prop := λ r => 
  ∀ t : ℝ, (1 : ℝ) = (1/2 : ℝ) * (1 + r) ^ (t + 1) → t = (fullCoverageTime - halfCoverageTime : ℝ)

theorem lily_pad_growth_rate : 
  growthRate 1 := by sorry

end NUMINAMATH_CALUDE_lily_pad_growth_rate_l771_77176


namespace NUMINAMATH_CALUDE_rectangle_split_area_l771_77121

theorem rectangle_split_area (c : ℝ) : 
  let total_area : ℝ := 8
  let smaller_area : ℝ := total_area / 3
  let larger_area : ℝ := 2 * smaller_area
  let triangle_area : ℝ := 2 * (4 - c)
  (4 + total_area - triangle_area = larger_area) → c = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_split_area_l771_77121


namespace NUMINAMATH_CALUDE_disjunction_true_given_p_l771_77106

theorem disjunction_true_given_p (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by sorry

end NUMINAMATH_CALUDE_disjunction_true_given_p_l771_77106


namespace NUMINAMATH_CALUDE_cards_per_set_is_13_l771_77142

/-- The number of trading cards in one set -/
def cards_per_set (initial_cards : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) (total_cards_given : ℕ) : ℕ :=
  total_cards_given / (sets_to_brother + sets_to_sister + sets_to_friend)

/-- Proof that the number of trading cards in one set is 13 -/
theorem cards_per_set_is_13 :
  cards_per_set 365 8 5 2 195 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_set_is_13_l771_77142


namespace NUMINAMATH_CALUDE_decagon_flip_impossible_l771_77136

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  deriving Repr

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : List Point
  deriving Repr

/-- Represents an operation to flip signs -/
inductive FlipOperation
  | Side
  | Diagonal

/-- Applies a flip operation to the configuration -/
def applyFlip (config : DecagonConfig) (op : FlipOperation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are negative -/
def allNegative (config : DecagonConfig) : Bool :=
  sorry

/-- Theorem: It's impossible to make all points negative in a decagon configuration -/
theorem decagon_flip_impossible (initial : DecagonConfig) :
  ∀ (ops : List FlipOperation), ¬(allNegative (ops.foldl applyFlip initial)) :=
sorry

end NUMINAMATH_CALUDE_decagon_flip_impossible_l771_77136


namespace NUMINAMATH_CALUDE_common_point_for_gp_lines_l771_77108

/-- A line in the form ax + by = c where a, b, c form a geometric progression -/
structure GPLine where
  a : ℝ
  r : ℝ
  h_r_nonzero : r ≠ 0

/-- The equation of a GPLine -/
def GPLine.equation (l : GPLine) (x y : ℝ) : Prop :=
  l.a * x + (l.a * l.r) * y = l.a * l.r^2

theorem common_point_for_gp_lines :
  ∀ (l : GPLine), l.equation 1 0 :=
sorry

end NUMINAMATH_CALUDE_common_point_for_gp_lines_l771_77108


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l771_77133

theorem algebraic_expression_value : 
  let a : ℝ := Real.sqrt 5 + 1
  (a^2 - 2*a + 7) = 11 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l771_77133
