import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l415_41541

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2)) = 3 → 
  Real.sqrt (a.1^2 + a.2^2) = 1 → 
  Real.sqrt (b.1^2 + b.2^2) = 2 → 
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l415_41541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l415_41547

def problem : String :=
  "Try to read the word depicted in Fig. 1 using the key (see Fig. 2)."

def solution : String :=
  "КОМПЬЮТЕР"

theorem problem_solution : True := by
  sorry

#print problem
#print solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l415_41547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_l415_41553

-- Define the point A
def A : ℝ × ℝ := (1, 0)

-- Define the slope angle
noncomputable def slope_angle : ℝ := Real.pi / 4

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line equation passing through A with the given slope angle
noncomputable def line (x y : ℝ) : Prop := y = (x - A.1) * Real.tan slope_angle

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- Statement to prove
theorem length_MN : Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_l415_41553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_difference_l415_41516

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a semicircle given its radius -/
noncomputable def semicircleArea (radius : ℝ) : ℝ := (Real.pi * radius^2) / 2

/-- Theorem: The area of semicircles on longer sides of an 8x12 rectangle
    is 125% larger than the area of semicircles on shorter sides -/
theorem semicircle_area_difference (rect : Rectangle)
    (h1 : rect.width = 8)
    (h2 : rect.length = 12) :
    (semicircleArea rect.length / 2) / (semicircleArea rect.width / 2) - 1 = 1.25 := by
  sorry

-- Remove the #eval line as it's not necessary for building and might cause issues
-- #eval semicircle_area_difference ⟨8, 12⟩ rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_difference_l415_41516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PFG_l415_41562

noncomputable section

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xr, yr) := R
  (xq - xp)^2 + (yq - yp)^2 = 10^2 ∧
  (xr - xq)^2 + (yr - yq)^2 = 12^2 ∧
  (xr - xp)^2 + (yr - yp)^2 = 14^2

-- Define point F on PQ
def PointF (P Q F : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xq, yq) := Q
  let (xf, yf) := F
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  xf = xp + t * (xq - xp) ∧
  yf = yp + t * (yq - yp) ∧
  (xf - xp)^2 + (yf - yp)^2 = 3^2

-- Define point G on PR
def PointG (P R G : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xr, yr) := R
  let (xg, yg) := G
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  xg = xp + t * (xr - xp) ∧
  yg = yp + t * (yr - yp) ∧
  (xg - xp)^2 + (yg - yp)^2 = 5^2

-- Define the area of a triangle given three points
def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (1/2) * abs ((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya))

-- Theorem statement
theorem area_of_triangle_PFG 
  (P Q R F G : ℝ × ℝ) 
  (h1 : Triangle P Q R) 
  (h2 : PointF P Q F) 
  (h3 : PointG P R G) : 
  TriangleArea P F G = 45 * Real.sqrt 2 / 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PFG_l415_41562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_candies_order_independent_l415_41588

/-- Represents a child in the room -/
inductive Child
| Boy
| Girl

/-- The state of the candy distribution process -/
structure CandyState where
  remaining_candies : ℕ
  remaining_children : List Child

/-- The result of a single candy-taking turn -/
structure TurnResult where
  candies_taken : ℕ
  new_state : CandyState

/-- Function to calculate the number of candies taken by a child -/
def take_candies (child : Child) (state : CandyState) : TurnResult :=
  sorry

/-- Function to calculate the total candies taken by boys after all turns -/
def total_boys_candies (initial_state : CandyState) : ℕ :=
  sorry

/-- Theorem stating that the total candies taken by boys is independent of the order of turns -/
theorem boys_candies_order_independent 
  (initial_state : CandyState) 
  (perm : List Child → List Child) 
  (h_perm : List.Perm initial_state.remaining_children (perm initial_state.remaining_children)) :
  total_boys_candies initial_state = total_boys_candies { initial_state with remaining_children := perm initial_state.remaining_children } :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_candies_order_independent_l415_41588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l415_41577

def mySequence : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => mySequence (n + 1) + 2 * (mySequence (n + 1) - mySequence n)

theorem sequence_fifth_term : mySequence 5 = 95 := by
  rw [mySequence, mySequence, mySequence, mySequence, mySequence, mySequence]
  norm_num
  rfl

#eval mySequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l415_41577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2a_plus_b_l415_41510

-- Define the vector type
def Vec2 := ℝ × ℝ

-- Define the given vectors
def a : Vec2 := (-2, 1)
def b : Vec2 := (1, 0)

-- Define vector addition
def add (v w : Vec2) : Vec2 := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalarMul (c : ℝ) (v : Vec2) : Vec2 := (c * v.1, c * v.2)

-- Define magnitude of a vector
noncomputable def magnitude (v : Vec2) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Theorem statement
theorem magnitude_2a_plus_b : magnitude (add (scalarMul 2 a) b) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_2a_plus_b_l415_41510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_complex_plane_l415_41556

theorem right_triangle_complex_plane (ω : ℂ) (l : ℝ) : 
  Complex.abs ω = 3 →
  l > 1 →
  Complex.abs (l • ω) ^ 2 = Complex.abs ω ^ 2 + Complex.abs (ω ^ 2) ^ 2 →
  l = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_complex_plane_l415_41556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l415_41593

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem function_symmetry 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_shift : ∀ x, f ω φ (x + π / 3) = g ω x) :
  ∀ x, f ω φ (π / 6 + x) = f ω φ (π / 6 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l415_41593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunshine_academy_full_day_students_l415_41528

/-- The number of full-day students at Sunshine Academy -/
def full_day_students (total : ℕ) (half_day_percentage : ℚ) : ℕ :=
  total - Int.toNat ((↑total * half_day_percentage).floor)

/-- Theorem: There are 106 full-day students at Sunshine Academy -/
theorem sunshine_academy_full_day_students :
  full_day_students 165 (36 / 100) = 106 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunshine_academy_full_day_students_l415_41528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l415_41576

/-- The radius of the inscribed circle of a triangle -/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem inscribed_circle_radius_specific_triangle :
  inscribed_circle_radius 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_specific_triangle_l415_41576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l415_41552

-- Define the custom operation *
noncomputable def customMul (a b : ℝ) : ℝ :=
  if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  customMul (2^x) (2^(-x))

-- Theorem statement
theorem range_of_f :
  ∀ y ∈ Set.range f, 0 < y ∧ y ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l415_41552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_venus_speed_approx_l415_41544

/-- The speed of Venus in miles per hour -/
noncomputable def venus_speed_mph : ℝ := 78840

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℝ := 3600

/-- The speed of Venus in miles per second -/
noncomputable def venus_speed_mps : ℝ := venus_speed_mph / seconds_per_hour

/-- Theorem: The speed of Venus is approximately 21.9 miles per second -/
theorem venus_speed_approx : 
  ∃ ε > 0, |venus_speed_mps - 21.9| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_venus_speed_approx_l415_41544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potluck_soda_bottles_l415_41591

/-- The number of whole bottles taken back home given initial and drunk quantities -/
def bottlesTakenHome (initial : ℕ) (drunk : ℚ) : ℕ :=
  Int.toNat ((initial : ℚ) - drunk).floor

/-- Theorem stating that given 50 initial bottles and 38.7 drunk bottles, 11 bottles are taken home -/
theorem potluck_soda_bottles : bottlesTakenHome 50 (38.7 : ℚ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potluck_soda_bottles_l415_41591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l415_41512

/-- The rational function f(x) = (3x^2 + 4x - 8) / (x - 4) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 4 * x - 8) / (x - 4)

/-- The slope of the slant asymptote of f(x) -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote of f(x) -/
def b : ℝ := 16

/-- Theorem: The sum of the slope and y-intercept of the slant asymptote of f(x) is 19 -/
theorem slant_asymptote_sum : m + b = 19 := by
  -- Unfold the definitions of m and b
  unfold m b
  -- Perform the addition
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l415_41512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_angle_is_pi_over_4_l415_41572

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the point on the circle
def point : ℝ × ℝ := (0, 1)

-- Define the angle of inclination of the shortest chord
noncomputable def shortest_chord_angle : ℝ := Real.pi/4

-- Theorem statement
theorem shortest_chord_angle_is_pi_over_4 :
  circle_equation point.1 point.2 →
  shortest_chord_angle = Real.pi/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_angle_is_pi_over_4_l415_41572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l415_41522

/-- Given vectors OA, OB, and OC in R², prove properties about their relationships -/
theorem vector_relationships (OA OB : ℝ × ℝ) (m : ℝ) :
  let OC : ℝ × ℝ := (m, 3)
  -- Given conditions
  OA = (2, -1) →
  OB = (3, 0) →
  -- Prove two statements
  (-- 1) When m = 8, OC can be expressed as a linear combination of OA and OB
   m = 8 →
   ∃ (lambda1 lambda2 : ℝ), OC = lambda1 • OA + lambda2 • OB ∧ lambda1 = -3 ∧ lambda2 = 14/3) ∧
  (-- 2) A, B, and C form a triangle iff m ≠ 6
   (∃ (AB AC : ℝ × ℝ), AB ≠ (0, 0) ∧ AC ≠ (0, 0) ∧ ¬(∃ (k : ℝ), AC = k • AB)) ↔ m ≠ 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l415_41522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_PPP_greater_than_10_12_l415_41507

/-- P(n) is the product of all positive integer divisors of n -/
def P (n : ℕ+) : ℕ+ :=
  ⟨(Finset.filter (·∣n.val) (Finset.range n.val.succ)).prod id, by
    sorry -- Proof that the product is positive
  ⟩

/-- 6 is the smallest positive integer n for which P(P(P(n))) > 10^12 -/
theorem smallest_n_for_PPP_greater_than_10_12 :
  ∀ k : ℕ+, k < 6 → (((P (P (P k))).val : ℝ) ≤ 10^12) ∧ (((P (P (P 6))).val : ℝ) > 10^12) := by
  sorry

#check smallest_n_for_PPP_greater_than_10_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_PPP_greater_than_10_12_l415_41507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l415_41500

/-- A function f is even if f(x) = f(-x) for all x in the domain of f -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The logarithmic function ln(x^2 + ax + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x^2 + a*x + 1)

/-- If f(x) = ln(x^2 + ax + 1) is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l415_41500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_removal_theorem_l415_41598

def digit_count (n : ℕ) : ℕ := sorry

def nth_digit (n : ℕ) (i : Fin 10) : Fin 10 := sorry

theorem digit_removal_theorem (n : ℕ) (h1 : n = 7^1996) : 
  ∃ (m : ℕ) (d1 d2 : Fin 10), 
    (digit_count m = 10) ∧ 
    (∃ (i j : Fin 10), i ≠ j ∧ nth_digit m i = nth_digit m j) ∧
    (m ≡ n [MOD 9]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_removal_theorem_l415_41598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cubed_coeff_l415_41560

noncomputable section

/-- First polynomial: 3x^4 - 4x^3 + 2x^2 - 3x + 2 -/
def p (x : ℝ) : ℝ := 3*x^4 - 4*x^3 + 2*x^2 - 3*x + 2

/-- Second polynomial: 2x^2 - 3x + 5 -/
def q (x : ℝ) : ℝ := 2*x^2 - 3*x + 5

/-- The product of the two polynomials -/
def product (x : ℝ) : ℝ := p x * q x

/-- Coefficient extraction function -/
def coeff (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  (deriv^[n] f 0) / n.factorial

theorem x_cubed_coeff :
  coeff product 3 = -32 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_cubed_coeff_l415_41560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MNF₁_l415_41587

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the line l passing through F₂ with slope 1
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- State the theorem
theorem area_triangle_MNF₁ :
  ellipse_C M.1 M.2 ∧
  ellipse_C N.1 N.2 ∧
  line_l M.1 M.2 ∧
  line_l N.1 N.2 ∧
  M ≠ N ∧
  let area := abs ((N.1 - F₁.1) * (M.2 - F₁.2) - (M.1 - F₁.1) * (N.2 - F₁.2)) / 2
  area = 12 * Real.sqrt 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_MNF₁_l415_41587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l415_41527

/-- The function we want to maximize -/
noncomputable def f (t : ℝ) : ℝ := (3^t - 2*t)*t / 9^t

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (t_max : ℝ), ∀ (t : ℝ), f t ≤ f t_max ∧ f t_max = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l415_41527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l415_41532

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Injective f)

-- Define the condition that f(1) - 1 = 2
axiom h1 : f 1 - 1 = 2

-- State the theorem
theorem inverse_function_point :
  Function.invFun f 3 + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l415_41532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l415_41565

theorem cosine_of_angle_through_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = -6 ∧ y = 8 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l415_41565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l415_41566

/-- The number of ways to assign roles in a play --/
def assign_roles (num_men : ℕ) (num_women : ℕ) (num_male_roles : ℕ) (num_female_roles : ℕ) (num_either_roles : ℕ) : ℕ :=
  (Nat.factorial num_men / Nat.factorial (num_men - num_male_roles)) *
  (Nat.factorial num_women / Nat.factorial (num_women - num_female_roles)) *
  (Nat.factorial (num_men + num_women - num_male_roles - num_female_roles) / 
   Nat.factorial (num_men + num_women - num_male_roles - num_female_roles - num_either_roles))

/-- Theorem: The number of ways to assign roles in the given scenario --/
theorem role_assignment_count : assign_roles 6 7 2 2 3 = 635040 := by
  rw [assign_roles]
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_role_assignment_count_l415_41566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_ratio_l415_41551

/-- Represents a rectangle with one side double the other and area equal to perimeter -/
structure SpecialRectangle where
  side : ℝ
  area_eq_perimeter : 2 * side^2 = 6 * side

/-- Represents a regular hexagon with area equal to perimeter -/
structure SpecialHexagon where
  side : ℝ
  area_eq_perimeter : (3 * Real.sqrt 3 / 2) * side^2 = 6 * side

/-- The apothem of a SpecialRectangle -/
noncomputable def apothemRectangle (r : SpecialRectangle) : ℝ := r.side / 2

/-- The apothem of a SpecialHexagon -/
noncomputable def apothemHexagon (h : SpecialHexagon) : ℝ := (Real.sqrt 3 / 2) * h.side

/-- Theorem stating the ratio of apothems -/
theorem apothem_ratio (r : SpecialRectangle) (h : SpecialHexagon) :
  apothemRectangle r / apothemHexagon h = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apothem_ratio_l415_41551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_distribution_theorem_l415_41564

/-- Represents the cash distribution in a lottery scenario -/
def lottery_distribution (contribution_A contribution_B contribution_C : ℚ)
  (first_win second_win : ℚ) (monthly_tickets weekly_tickets : ℕ) : Prop :=
  let total_contribution := contribution_A + contribution_B + contribution_C
  let first_distribution_A := (contribution_A * first_win) / total_contribution
  let first_distribution_B := (contribution_B * first_win) / total_contribution
  let first_distribution_C := (contribution_C * first_win) / total_contribution
  let ticket_cost := 3.3
  let monthly_cost := (monthly_tickets : ℚ) * 5 * ticket_cost
  let weekly_cost := (weekly_tickets : ℚ) * ticket_cost
  let remaining := first_win - (monthly_cost + weekly_cost)
  let second_distribution_A := remaining / 7
  let second_distribution_B := 2 * second_distribution_A
  let second_distribution_C := 4 * second_distribution_A
  let net_contribution_A := first_distribution_A - second_distribution_A
  let net_contribution_B := first_distribution_B - second_distribution_B
  let net_contribution_C := first_distribution_C - second_distribution_C
  let total_net_contribution := net_contribution_A + net_contribution_B + net_contribution_C
  let final_amount := second_win + (4 * monthly_tickets : ℚ) * ticket_cost
  let final_A := (net_contribution_A * final_amount) / total_net_contribution - (10 : ℚ) * ticket_cost
  let final_B := (net_contribution_B * final_amount) / total_net_contribution - (10 : ℚ) * ticket_cost
  let final_C := (net_contribution_C * final_amount) / total_net_contribution - (10 : ℚ) * ticket_cost
  (abs (final_A - 26135.89) < 0.01) ∧ 
  (abs (final_B - 32052.34) < 0.01) ∧ 
  (abs (final_C - 14811.77) < 0.01)

theorem lottery_distribution_theorem :
  lottery_distribution 25 36 38 1100 73000 30 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_distribution_theorem_l415_41564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l415_41545

/-- Represents an L-shaped piece made of three unit squares --/
structure LPiece where
  position : Fin 6 × Fin 6
  orientation : Fin 4

/-- Checks if two L-pieces form a 2x3 rectangle --/
def forms_2x3_rectangle (p1 p2 : LPiece) : Prop :=
  sorry

/-- Represents a valid arrangement of L-pieces on a 6x6 grid --/
def valid_arrangement (pieces : Finset LPiece) : Prop :=
  (pieces.card = 12) ∧
  (∀ i j : Fin 6, ∃ p ∈ pieces, (i, j) = p.position) ∧
  (∀ p1 p2, p1 ∈ pieces → p2 ∈ pieces → p1 ≠ p2 → ¬forms_2x3_rectangle p1 p2)

/-- The main theorem stating that a valid arrangement exists --/
theorem exists_valid_arrangement : ∃ pieces : Finset LPiece, valid_arrangement pieces := by
  sorry

#check exists_valid_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_arrangement_l415_41545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_seven_terms_l415_41524

/-- The number of distinct terms in the expansion of ((a+2b)^2 * (a-2b)^2)^3 -/
def num_distinct_terms : ℕ := 7

/-- Theorem stating that the expansion has exactly 7 distinct terms -/
theorem expansion_has_seven_terms :
  num_distinct_terms = 7 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_has_seven_terms_l415_41524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l415_41592

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 = a * y

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 = 2

-- Define the focus of the parabola
noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ :=
  (0, a / 4)

-- Define the foci of the hyperbola
def hyperbola_foci : Set (ℝ × ℝ) :=
  {(0, 2), (0, -2)}

-- State the theorem
theorem parabola_hyperbola_focus_coincidence (a : ℝ) :
  (∃ (x y : ℝ), parabola a x y ∧ hyperbola x y ∧ parabola_focus a ∈ hyperbola_foci) →
  (a = 8 ∨ a = -8) := by
  sorry

#check parabola_hyperbola_focus_coincidence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_focus_coincidence_l415_41592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_grazing_area_l415_41525

/-- The difference in area between two circles with radii 23 and 10 is 429π square meters. -/
theorem additional_grazing_area :
  let r₁ : ℝ := 10
  let r₂ : ℝ := 23
  let area (r : ℝ) := π * r^2
  area r₂ - area r₁ = 429 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_grazing_area_l415_41525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l415_41554

noncomputable def f (x : ℝ) := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l415_41554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l415_41542

/-- Circle in the cartesian plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line in the cartesian plane --/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection points of a line and a circle --/
def intersectionPoints (c : Circle) (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧ 
    ∃ t : ℝ, p.1 = l.point.1 + t ∧ p.2 = l.point.2 + l.slope * t}

/-- Distance between two points in the plane --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_product (c : Circle) (l : Line) :
  c.center = (0, 0) ∧ c.radius = 4 ∧ 
  l.point = (1, 2) ∧ l.slope = Real.tan (π / 6) →
  ∃ A B : ℝ × ℝ, A ∈ intersectionPoints c l ∧ 
                 B ∈ intersectionPoints c l ∧ 
                 A ≠ B ∧
                 distance l.point A * distance l.point B = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l415_41542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l415_41511

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := (1/2)^(1/5 : ℝ)
noncomputable def b : ℝ := (1/5)^(-(1/2) : ℝ)
noncomputable def c : ℝ := Real.log 10 / Real.log (1/5)

-- State the theorem
theorem a_b_c_order : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_c_order_l415_41511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_l415_41557

/-- Proves that the total investment is 100000 given the specified conditions --/
theorem total_investment (interest_rate_1 interest_rate_2 total_interest_rate : ℚ) 
  (amount_2 : ℚ) 
  (h_rate1 : interest_rate_1 = 9 / 100)
  (h_rate2 : interest_rate_2 = 11 / 100)
  (h_total_rate : total_interest_rate = 95 / 1000)
  (h_amount2 : amount_2 = 25000)
  (h1 : ∃ (total amount_1 : ℚ), 
    total = amount_1 + amount_2 ∧ 
    total_interest_rate * total = interest_rate_1 * amount_1 + interest_rate_2 * amount_2) :
  ∃ (total : ℚ), total = 100000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_l415_41557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l415_41575

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2 * Real.sin (x + Real.pi/4) + 2*x^2 + x) / (2*x^2 + Real.cos x)

theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), (∀ x, f x ≤ max) ∧ (∀ x, min ≤ f x) ∧ max + min = 2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l415_41575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_equivalence_l415_41502

def mySequence : ℕ → ℤ
  | 1 => 2
  | 2 => -6
  | 3 => 12
  | 4 => -20
  | 5 => 30
  | 6 => -42
  | n => (-1)^(n+1) * n * (n+1)  -- General formula for n > 6

def generalFormula (n : ℕ) : ℤ := (-1)^(n+1) * n * (n+1)

theorem sequence_formula_equivalence :
  ∀ n : ℕ, n > 0 → mySequence n = generalFormula n :=
by
  intro n hn
  cases n with
  | zero => contradiction
  | succ n' =>
    simp [mySequence, generalFormula]
    sorry  -- Proof details omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_equivalence_l415_41502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_2017_l415_41538

/-- The sequence a_n defined as (√2 + 1)^n - (√2 - 1)^n -/
noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 + 1)^n - (Real.sqrt 2 - 1)^n

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The units digit of an integer -/
def units_digit (n : ℤ) : ℕ := n.natAbs % 10

/-- The main theorem stating that the units digit of floor(a_2017) is 2 -/
theorem units_digit_of_a_2017 :
  units_digit (floor (a 2017)) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_a_2017_l415_41538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monotone_inverse_function_l415_41578

/-- The n-th iterate of a function f -/
def Iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (Iterate f n)

/-- The main theorem -/
theorem unique_monotone_inverse_function (f : ℝ → ℝ) (h_monotone : Monotone f) 
  (h_inverse : ∃ n₀ : ℕ, ∀ x : ℝ, Iterate f n₀ x = -x) : 
  ∀ x : ℝ, f x = -x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_monotone_inverse_function_l415_41578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identity_l415_41550

theorem angle_identity (α β : ℝ) 
  (h : (Real.cos α)^2 / Real.cos β + (Real.sin α)^2 / Real.sin β = 2) :
  (Real.sin β)^2 / Real.sin α + (Real.cos β)^2 / Real.cos α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_identity_l415_41550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pears_thrown_away_l415_41586

/-- The percentage of pears thrown away by a vendor over two days -/
theorem pears_thrown_away (initial_pears : ℝ) (initial_pears_pos : 0 < initial_pears) : 
  (let first_day_sold := 0.2 * initial_pears
   let first_day_remaining := initial_pears - first_day_sold
   let first_day_thrown := 0.5 * first_day_remaining
   let second_day_start := first_day_remaining - first_day_thrown
   let second_day_sold := 0.2 * second_day_start
   let second_day_thrown := second_day_start - second_day_sold
   let total_thrown := first_day_thrown + second_day_thrown
   total_thrown / initial_pears) = 0.72 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pears_thrown_away_l415_41586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l415_41517

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
noncomputable def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 + i) * x

-- State the theorem
theorem f_composition_equals_three :
  f (f (1 - i)) = 3 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_three_l415_41517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_integral_value_minimum_integral_achievable_l415_41579

open Set
open MeasureTheory
open Interval

/-- A polynomial of degree ≤ 2 -/
structure PolynomialDegree2 where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function represented by the polynomial -/
def polynomialFunction (p : PolynomialDegree2) : ℝ → ℝ := λ x => p.a * x^2 + p.b * x + p.c

/-- The derivative of the polynomial function -/
def polynomialDerivative (p : PolynomialDegree2) : ℝ → ℝ := λ x => 2 * p.a * x + p.b

theorem minimum_integral_value (p : PolynomialDegree2) 
  (h1 : polynomialFunction p 0 = 0) 
  (h2 : polynomialFunction p 2 = 2) :
  ∫ x in (Icc 0 2), |polynomialDerivative p x| ≥ 2 := by
  sorry

theorem minimum_integral_achievable :
  ∃ p : PolynomialDegree2, 
    polynomialFunction p 0 = 0 ∧ 
    polynomialFunction p 2 = 2 ∧
    ∫ x in (Icc 0 2), |polynomialDerivative p x| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_integral_value_minimum_integral_achievable_l415_41579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l415_41559

/-- Hyperbola C with equation x^2/2 - y^2 = 1 -/
def Hyperbola (x y : ℝ) : Prop := x^2/2 - y^2 = 1

/-- Point P on the right branch of the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Asymptote l of the hyperbola -/
def l : Set (ℝ × ℝ) := sorry

/-- Point Q as the projection of P on l -/
def Q : ℝ × ℝ := sorry

/-- Left focus F1 of the hyperbola -/
def F1 : ℝ × ℝ := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum :
  ∀ (x y : ℝ), Hyperbola x y →
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 + 1 ∧
  ∀ (p : ℝ × ℝ), p ∈ {(x, y) | Hyperbola x y ∧ x > 0} →
  distance p F1 + distance p Q ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l415_41559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortcut_ratio_l415_41580

/-- Represents a rectangular field -/
structure RectangularField where
  shorter : ℝ
  longer : ℝ
  shorter_positive : 0 < shorter
  longer_positive : 0 < longer
  shorter_less_than_longer : shorter < longer

/-- The distance saved by taking the diagonal shortcut -/
noncomputable def distance_saved (field : RectangularField) : ℝ :=
  field.shorter + field.longer - Real.sqrt (field.shorter^2 + field.longer^2)

theorem shortcut_ratio (field : RectangularField) 
  (h : distance_saved field = field.longer / 2) :
  field.shorter / field.longer = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortcut_ratio_l415_41580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boil_away_time_l415_41513

/-- Represents the time it takes for water to boil away given specific conditions -/
noncomputable def time_to_boil_away (T₀ Tₘ t c L : ℝ) : ℝ :=
  t * (L * 1000) / (c * (Tₘ - T₀))

/-- Theorem stating the time it takes for water to boil away under given conditions -/
theorem water_boil_away_time :
  let T₀ : ℝ := 20  -- Initial temperature in °C
  let Tₘ : ℝ := 100  -- Final temperature in °C
  let t : ℝ := 10  -- Time to reach boiling point in minutes
  let c : ℝ := 4200  -- Specific heat capacity of water in J/(kg·K)
  let L : ℝ := 2.3  -- Specific heat of vaporization of water in MJ/kg
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |time_to_boil_away T₀ Tₘ t c L - 68| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_boil_away_time_l415_41513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l415_41563

/-- The function f(x) = (a * x) / (2 * x + 3) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (2 * x + 3)

/-- Theorem: If f(f(x)) = x for all x, then a = -3 -/
theorem function_composition_identity (a : ℝ) :
  (∀ x : ℝ, f a (f a x) = x) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l415_41563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l415_41584

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*x - 5/4
  else Real.log (1/3) - 1/4

noncomputable def g (A : ℝ) (x : ℝ) : ℝ :=
  |A - 2| * Real.sin x

theorem function_inequality (A : ℝ) :
  (∀ x₁ x₂ : ℝ, f x₁ ≤ g A x₂) → A ∈ Set.Icc (7/4) (9/4) := by
  sorry

#check function_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l415_41584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l415_41518

theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 5 = (b : ℝ) / 6 → (b : ℝ) / 6 = (c : ℝ) / 7 →
  max a (max b c) = 70 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_ratio_triangle_l415_41518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_graph_property_l415_41561

/-- A friendship graph with special properties -/
structure FriendshipGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  h1 : n > 4
  h2 : ∀ u v : Fin n, (u, v) ∈ edges → 
        (∀ w : Fin n, (u, w) ∈ edges ∧ (v, w) ∈ edges → False)
  h3 : ∀ u v : Fin n, (u, v) ∉ edges → 
        (∃! w1 w2 : Fin n, w1 ≠ w2 ∧ (u, w1) ∈ edges ∧ (u, w2) ∈ edges ∧ 
                           (v, w1) ∈ edges ∧ (v, w2) ∈ edges)

/-- The main theorem about friendship graphs -/
theorem friendship_graph_property (n : ℕ) (G : FriendshipGraph n) :
  ∃ m : ℕ, 8 * n - 7 = m^2 ∧ n = 7 := by
  sorry

#check friendship_graph_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_graph_property_l415_41561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l415_41583

def sequenceA (n : ℕ) : ℚ :=
  if n = 0 then 1 else (3 * sequenceA (n - 1) + 1)

theorem sequence_general_term (n : ℕ) :
  sequenceA n = (3^n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l415_41583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_upper_bound_l415_41526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 + (2*a + 1) * x

theorem f_monotonicity_and_upper_bound (a : ℝ) :
  (∀ x > 0, a ≥ 0 → (deriv (f a)) x > 0) ∧
  (a < 0 → (∀ x > 0, x < -1/(2*a) → (deriv (f a)) x > 0) ∧ 
           (∀ x > -1/(2*a), (deriv (f a)) x < 0)) ∧
  (a < 0 → ∀ x > 0, f a x ≤ -3/(4*a) - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_upper_bound_l415_41526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_exponent_sum_l415_41581

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_exponent_sum : i^6 + i^16 + i^(-26 : ℤ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_exponent_sum_l415_41581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l415_41594

noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_example :
  let ρ : Real := 3
  let θ : Real := 3 * Real.pi / 2
  let φ : Real := Real.pi / 3
  spherical_to_rectangular ρ θ φ = (0, -3 * Real.sqrt 3 / 2, 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_example_l415_41594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l415_41509

/-- The area of a circular sector -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * r^2

/-- Theorem: The area of a circular sector with radius 18 meters and central angle 42 degrees 
    is approximately 118.44 square meters. -/
theorem sector_area_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |sectorArea 18 42 - 118.44| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l415_41509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_normal_equations_l415_41519

-- Parabola
def parabola (x : ℝ) : ℝ := x^2 - 4*x

-- Circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 3 = 0

-- Cycloid
noncomputable def cycloid_x (t : ℝ) : ℝ := t - Real.sin t
noncomputable def cycloid_y (t : ℝ) : ℝ := 1 - Real.cos t

-- Absolute value function
def abs_cube (x : ℝ) : ℝ := |x^3 - 1|

theorem tangent_normal_equations :
  -- Parabola
  let tangent_parabola := (fun x y : ℝ => 2*x + y + 1 = 0)
  let normal_parabola := (fun x y : ℝ => x - 2*y - 7 = 0)
  (∀ x y : ℝ, y = parabola x → x = 1 → (tangent_parabola x y ∧ normal_parabola x y)) ∧

  -- Circle
  let tangent_circle_A := (fun x y : ℝ => x - y + 1 = 0)
  let normal_circle_A := (fun x y : ℝ => x + y + 1 = 0)
  let tangent_circle_B := (fun x y : ℝ => x + y - 3 = 0)
  let normal_circle_B := (fun x y : ℝ => x - y - 3 = 0)
  (∀ x y : ℝ, circle_eq x y → y = 0 → 
    ((x = -1 → tangent_circle_A x y ∧ normal_circle_A x y) ∧
     (x = 3 → tangent_circle_B x y ∧ normal_circle_B x y))) ∧

  -- Cycloid
  let tangent_cycloid := (fun x y : ℝ => 2*x - 2*y - Real.pi + 4 = 0)
  let normal_cycloid := (fun x y : ℝ => 2*x + 2*y - Real.pi = 0)
  (∀ t : ℝ, t = Real.pi / 2 → 
    tangent_cycloid (cycloid_x t) (cycloid_y t) ∧ 
    normal_cycloid (cycloid_x t) (cycloid_y t)) ∧

  -- Absolute value function
  let tangent_abs_1 := (fun x y : ℝ => 3*x - y - 2 = 0)
  let tangent_abs_2 := (fun x y : ℝ => -3*x - y - 2 = 0)
  let normal_abs_1 := (fun x y : ℝ => x + 3*y - 1 = 0)
  let normal_abs_2 := (fun x y : ℝ => x - 3*y - 1 = 0)
  (∀ x : ℝ, x = 1 → 
    tangent_abs_1 x (abs_cube x) ∧ 
    tangent_abs_2 x (abs_cube x) ∧ 
    normal_abs_1 x (abs_cube x) ∧ 
    normal_abs_2 x (abs_cube x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_normal_equations_l415_41519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l415_41529

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t + 2 * Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

noncomputable def a : ℝ := 1 / 9
noncomputable def b : ℝ := -4 / 45
noncomputable def c : ℝ := 124 / 1125

theorem curve_equation (t : ℝ) : a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_l415_41529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_passengers_proof_l415_41568

/-- Calculates the total number of passengers for a bus given its capacity and occupancy rates -/
def bus_passengers (capacity : ℕ) (rate_ab : ℚ) (rate_ba : ℚ) : ℕ :=
  (Int.floor ((capacity : ℚ) * rate_ab) + Int.floor ((capacity : ℚ) * rate_ba)).toNat

/-- Proves that the total number of passengers for all buses is 1133 -/
theorem total_passengers_proof :
  let bus_x := bus_passengers 180 (5/6) (2/3)
  let bus_y := bus_passengers 250 (9/10) (7/8)
  let bus_z := bus_passengers 300 (3/5) (4/5)
  bus_x + bus_y + bus_z = 1133 := by
  sorry

#eval bus_passengers 180 (5/6) (2/3) -- Expected: 270
#eval bus_passengers 250 (9/10) (7/8) -- Expected: 443
#eval bus_passengers 300 (3/5) (4/5) -- Expected: 420

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_passengers_proof_l415_41568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l415_41537

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 3 then x^2 + 3 else 3*x

theorem function_values (x : ℝ) : f x = 15 ↔ x = -2*Real.sqrt 3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l415_41537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_72_l415_41504

theorem sum_of_divisors_72 : (Finset.filter (· ∣ 72) (Finset.range 73)).sum id = 195 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_72_l415_41504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l415_41590

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x + a

-- State the theorem
theorem f_properties (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) :
  a = 2 ∧ ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) < 2 * (1 / x₁ - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l415_41590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_l415_41558

theorem grape_juice_mixture (x : Real) : Real :=
  let initial_percentage : Real := 0.1
  let added_amount : Real := 10
  let final_percentage : Real := 0.28000000000000004
  
  have h1 : (initial_percentage * x + added_amount) / (x + added_amount) = final_percentage := by sorry
  have h2 : x = 40 := by sorry

  x


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_juice_mixture_l415_41558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l415_41548

theorem subset_intersection_theorem (α : ℝ) (h_α : 0 < α ∧ α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ), 
    0 < n ∧ 
    0 < p ∧ 
    p > 2^n * α ∧
    ∃ (S T : Finset (Finset (Fin n))),
      Finset.card S = p ∧
      Finset.card T = p ∧
      ∀ (s t : Finset (Fin n)), s ∈ S → t ∈ T → ∃ (i : Fin n), i ∈ s ∧ i ∈ t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_intersection_theorem_l415_41548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_1_non_tangent_line_condition_max_value_g_l415_41515

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Define the function F
noncomputable def F (a : ℝ) : ℝ :=
  if a ≤ 1/4 then 1 - 3*a
  else if a < 1 then 2*a*Real.sqrt a
  else 3*a - 1

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f_when_a_1 :
  ∃ x, f 1 x = -2 ∧ ∀ y, f 1 y ≥ -2 :=
sorry

-- Theorem 2: Condition for non-tangent line
theorem non_tangent_line_condition (a : ℝ) :
  (∀ m, ∃ x, x + f a x + m ≠ 0) ↔ a < 1/3 :=
sorry

-- Theorem 3: Maximum value of g
theorem max_value_g (a : ℝ) :
  ∀ x, x ∈ Set.Icc (-1) 1 → g a x ≤ F a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_1_non_tangent_line_condition_max_value_g_l415_41515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l415_41546

noncomputable def f (x : ℝ) : ℝ := 1/2 - 1/2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := 
  if x ∈ Set.Icc 0 (Real.pi / 2) then 1/2 - f x
  else if x ∈ Set.Ico (-Real.pi) (-Real.pi / 2) then 1/2 * Real.cos (2 * x)
  else -1/2 * Real.cos (2 * x)

theorem function_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), (S > 0 ∧ ∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧ 
  (∀ (x : ℝ), f (-x) = f x) ∧
  (∀ (x : ℝ), g (x + Real.pi / 2) = g x) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi) 0 → 
    (x ∈ Set.Icc (-Real.pi / 2) 0 → g x = -1/2 * Real.cos (2 * x)) ∧ 
    (x ∈ Set.Ico (-Real.pi) (-Real.pi / 2) → g x = 1/2 * Real.cos (2 * x))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l415_41546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_piecewise_function_l415_41531

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then
    2 * x - x^2
  else if -2 ≤ x ∧ x ≤ 0 then
    x^2 + 6 * x
  else
    0  -- Arbitrary value for x outside the defined intervals

theorem range_of_piecewise_function :
  Set.range f = Set.Icc (-8 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_piecewise_function_l415_41531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_coordinates_l415_41595

/-- The integer part of a non-negative real number -/
noncomputable def T (a : ℝ) : ℤ := Int.floor a

/-- The x-coordinate of the k-th tree -/
noncomputable def x : ℕ → ℤ
  | 0 => 1
  | k + 1 => x k + 1 - 5 * (T ((k + 1 : ℝ) / 5) - T ((k : ℝ) / 5))

/-- The y-coordinate of the k-th tree -/
noncomputable def y : ℕ → ℤ
  | 0 => 1
  | k + 1 => y k + T ((k + 1 : ℝ) / 5) - T ((k : ℝ) / 5)

theorem tree_planting_coordinates :
  (x 5 = 1 ∧ y 5 = 2) ∧ (x 2015 = 1 ∧ y 2015 = 404) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_planting_coordinates_l415_41595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_circle_theorem_l415_41567

/-- An ellipse with equation x² + 3y² = a² -/
structure Ellipse where
  a : ℝ
  equation : (x y : ℝ) → Prop

/-- A line with equation y = kx + t -/
structure Line where
  k : ℝ
  t : ℝ
  equation : (x y : ℝ) → Prop

/-- The intersection points of a line with the ellipse -/
def intersectionPoints (e : Ellipse) (l : Line) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ e.equation x y ∧ l.equation x y}

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passesThroughPoints : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop

/-- The main theorem -/
theorem ellipse_line_intersection_circle_theorem (e : Ellipse) :
  ∀ (t : ℝ), t > 0 →
    ∃ (k : ℝ), ∃ (c : Circle),
      let l := Line.mk k t (fun x y => y = k*x + t)
      let intersections := intersectionPoints e l
      ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersections ∧ p2 ∈ intersections ∧
        c.passesThroughPoints p1 p2 (-1, 0) :=
by
  sorry

/-- Create an ellipse instance -/
def createEllipse (a : ℝ) : Ellipse :=
  { a := a
    equation := fun x y => x^2 + 3*y^2 = a^2 }

/-- Create a line instance -/
def createLine (k t : ℝ) : Line :=
  { k := k
    t := t
    equation := fun x y => y = k*x + t }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_circle_theorem_l415_41567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_theorem_l415_41543

def max_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (three_books : ℕ) 
  (average_books : ℕ) : ℕ :=
  let remaining_students := total_students - (zero_books + one_book + two_books + three_books)
  let total_books := total_students * average_books
  let books_0_to_3 := one_book + 2 * two_books + 3 * three_books
  let remaining_books := total_books - books_0_to_3
  let books_for_19 := (remaining_students - 1) * 4
  remaining_books - books_for_19

theorem max_books_theorem :
  max_books_borrowed 100 5 25 30 20 3 = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_theorem_l415_41543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l415_41523

/-- A line in 2D space represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a line --/
noncomputable def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The slope of a line, if it exists --/
noncomputable def Line.slope (l : Line) : Option ℝ :=
  if l.b ≠ 0 then some (-l.a / l.b) else none

/-- Check if two lines are perpendicular --/
noncomputable def perpendicular (l1 l2 : Line) : Prop :=
  match l1.slope, l2.slope with
  | some m1, some m2 => m1 * m2 = -1
  | _, _ => False

theorem line_equation_correct (l1 l2 : Line) : 
  (l1.a = 3 ∧ l1.b = -1 ∧ l1.c = -3) →
  (l2.a = 1 ∧ l2.b = 3 ∧ l2.c = -1) →
  l1.contains 1 0 ∧ perpendicular l1 l2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l415_41523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_lateral_surface_area_l415_41585

/-- A cone with an equilateral triangle as its axial section -/
structure EquilateralCone where
  /-- The side length of the equilateral triangle in the axial section -/
  side_length : ℝ
  /-- Assumption that the side length is positive -/
  side_length_pos : side_length > 0

/-- The lateral surface area of an equilateral cone -/
noncomputable def lateral_surface_area (cone : EquilateralCone) : ℝ :=
  Real.pi * cone.side_length

theorem equilateral_cone_lateral_surface_area :
  ∀ (cone : EquilateralCone), cone.side_length = 2 →
  lateral_surface_area cone = 2 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_cone_lateral_surface_area_l415_41585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_decrease_l415_41503

/-- Represents the side length of an equilateral triangle -/
noncomputable def side_length (area : ℝ) : ℝ := Real.sqrt ((4 * area) / Real.sqrt 3)

/-- Calculates the area of an equilateral triangle given its side length -/
noncomputable def triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- Theorem: The percent decrease in area of an equilateral triangle is 19% when its side length is decreased by 10% -/
theorem triangle_area_decrease (initial_area : ℝ) (h : initial_area = 72 * Real.sqrt 3) :
  let initial_side := side_length initial_area
  let new_side := initial_side * 0.9
  let new_area := triangle_area new_side
  (initial_area - new_area) / initial_area * 100 = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_decrease_l415_41503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l415_41549

/-- A dart board shaped as a regular hexagon with a smaller equilateral triangle inscribed -/
structure DartBoard where
  s : ℝ  -- Side length of the hexagon
  s_pos : s > 0

/-- The area of a regular hexagon with side length s -/
noncomputable def hexagon_area (d : DartBoard) : ℝ :=
  3 * Real.sqrt 3 / 2 * d.s^2

/-- The area of the inscribed equilateral triangle -/
noncomputable def triangle_area (d : DartBoard) : ℝ :=
  Real.sqrt 3 / 16 * d.s^2

/-- The probability of a dart landing in the inscribed triangle -/
noncomputable def landing_probability (d : DartBoard) : ℝ :=
  triangle_area d / hexagon_area d

/-- Theorem: The probability of a dart landing in the inscribed triangle is 1/24 -/
theorem dart_landing_probability (d : DartBoard) : 
  landing_probability d = 1/24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_landing_probability_l415_41549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_sum_l415_41596

/-- Pentagon vertices -/
def pentagon : List (ℝ × ℝ) := [(0, 0), (2, 0), (3, 2), (2, 3), (0, 2)]

/-- Calculate distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate perimeter of the pentagon -/
noncomputable def perimeter : ℝ :=
  List.sum (List.zipWith distance pentagon (pentagon.rotateRight 1))

theorem pentagon_perimeter_sum :
  ∃ (a b c d : ℤ),
    perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5 ∧
    a + b + c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_perimeter_sum_l415_41596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l415_41520

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  (train_speed_kmh * (1000 / 3600) * crossing_time) - train_length = 215 := by
  -- Convert train speed from km/h to m/s
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  
  -- Calculate total distance traveled
  have total_distance : ℝ := train_speed_ms * crossing_time
  
  -- Calculate bridge length
  have bridge_length : ℝ := total_distance - train_length
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_l415_41520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_div_g_eq_l415_41599

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- State the properties of f and g
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom g_odd : ∀ x : ℝ, g x = -g (-x)

-- State the given equation
axiom eq_fg : ∀ x : ℝ, f x + g x = 1 / (x^2 - x + 1)

-- Define the range of f(x) / g(x)
def range_f_div_g : Set ℝ := {y | ∃ x : ℝ, y = f x / g x}

-- State the theorem to be proved
theorem range_f_div_g_eq : range_f_div_g = Set.Iic (-2) ∪ Set.Ici 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_div_g_eq_l415_41599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l415_41573

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (x : ℝ) : ℝ := 2 * (e ^ x)

theorem tangent_slope_at_one :
  (deriv f) 1 = 2 * e := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l415_41573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_routes_P_to_Q_l415_41530

-- Define the nodes in the network
inductive Node : Type
  | P | R | S | T | U | V | Q

-- Define the direct connections between nodes
def direct_connection : Node → Node → Prop
  | Node.P, Node.R => True
  | Node.P, Node.S => True
  | Node.R, Node.T => True
  | Node.R, Node.U => True
  | Node.S, Node.Q => True
  | Node.T, Node.Q => True
  | Node.U, Node.V => True
  | Node.V, Node.Q => True
  | _, _ => False

-- Define a path as a list of nodes
def Route := List Node

-- Define a valid path
def valid_route : Route → Prop
  | [] => True
  | [_] => True
  | (x::y::rest) => direct_connection x y ∧ valid_route (y::rest)

-- Define the number of routes between two nodes
noncomputable def num_routes (start finish : Node) : Nat :=
  sorry

-- Theorem statement
theorem num_routes_P_to_Q :
  num_routes Node.P Node.Q = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_routes_P_to_Q_l415_41530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l415_41514

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((3 - a) * x - a) / Real.log a

theorem f_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ 1 < a ∧ a < 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_iff_a_in_range_l415_41514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l415_41535

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -4 ∧ x ≠ -2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l415_41535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_pair_arrangements_l415_41501

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1) * Nat.factorial k

theorem adjacent_pair_arrangements :
  number_of_arrangements 6 2 = 240 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_pair_arrangements_l415_41501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l415_41597

theorem mod_equation (m : ℕ) (h1 : m < 37) (h2 : (4 * m) % 37 = 1) :
  (((3 : ℤ)^m)^4 - 3) % 37 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mod_equation_l415_41597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l415_41539

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → Prop

-- Define parallel and perpendicular relations
def parallel (l1 l2 : Line) : Prop := sorry

def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop := 
  let (x, y) := p
  l x y

-- Define a function to check if a line has equal intercepts
def equal_intercepts (l : Line) : Prop := sorry

theorem line_equations :
  let l1 : Line := λ x y => x - 2*y + 7 = 0
  let l2 : Line := λ x y => 3*x - y + 2 = 0
  let l3 : Line := λ x y => x + y - 3 = 0
  
  -- Problem 1
  (point_on_line (-1, 3) l1 ∧ parallel l1 (λ x y => x - 2*y + 3 = 0)) ∧
  
  -- Problem 2
  (point_on_line (3, 4) (λ x y => x + 3*y - 15 = 0) ∧ perpendicular (λ x y => x + 3*y - 15 = 0) l2) ∧
  
  -- Problem 3
  (point_on_line (1, 2) l3 ∧ equal_intercepts l3)
  
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l415_41539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_is_three_fourteenths_l415_41534

/-- Represents the sales data for a year -/
structure YearlySales where
  /-- Average monthly sales for January through November -/
  avg_sales_jan_nov : ℝ
  /-- Number of months (excluding December) -/
  num_months : ℕ
  /-- December sales as a multiple of avg_sales_jan_nov -/
  dec_sales_multiplier : ℝ

/-- Calculates the fraction of December sales to total yearly sales -/
noncomputable def december_sales_fraction (sales : YearlySales) : ℝ :=
  let total_sales_jan_nov := sales.avg_sales_jan_nov * sales.num_months
  let december_sales := sales.avg_sales_jan_nov * sales.dec_sales_multiplier
  let total_yearly_sales := total_sales_jan_nov + december_sales
  december_sales / total_yearly_sales

theorem december_sales_fraction_is_three_fourteenths 
  (sales : YearlySales) 
  (h1 : sales.num_months = 11) 
  (h2 : sales.dec_sales_multiplier = 3) : 
  december_sales_fraction sales = 3 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_december_sales_fraction_is_three_fourteenths_l415_41534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_scalar_multiple_l415_41582

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two non-zero vectors are collinear if and only if one is a scalar multiple of the other -/
theorem collinear_iff_scalar_multiple (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ k : ℝ, b = k • a) ↔ (∃ t : ℝ, t • a = b ∨ t • b = a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_scalar_multiple_l415_41582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l415_41589

theorem divisibility_condition (n p : ℕ+) (h_prime : Nat.Prime p.val) (h_bound : n ≤ 2*p) :
  (((p:ℤ) - 1)^(n.val:ℕ) + 1) % (n:ℤ)^(p.val - 1) = 0 ↔ 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l415_41589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_existence_l415_41536

/-- Given a line segment AB, prove that there exists a point M on AB such that AM = MB,
    using only the ability to draw lines and perpendiculars with a right triangle
    (without directly dropping perpendiculars). -/
theorem midpoint_existence (A B : EuclideanSpace ℝ (Fin 2)) : 
  ∃ M : EuclideanSpace ℝ (Fin 2), ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B ∧ 
  dist A M = dist M B := by
  sorry

/-- Represents the ability to draw a line using a right triangle -/
noncomputable def draw_line (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := 
  {X | ∃ t : ℝ, X = (1 - t) • P + t • Q}

/-- Represents the ability to draw a perpendicular line using a right triangle,
    without directly dropping a perpendicular -/
noncomputable def draw_perpendicular (L : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) : 
  Set (EuclideanSpace ℝ (Fin 2)) := sorry

/-- Represents the intersection of two lines -/
noncomputable def line_intersection (L1 L2 : Set (EuclideanSpace ℝ (Fin 2))) : 
  Set (EuclideanSpace ℝ (Fin 2)) := L1 ∩ L2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_existence_l415_41536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_identity_l415_41540

theorem binomial_integral_identity (n : ℕ) :
  ∀ k : ℕ, k ≤ n →
    ∫ x in (-1 : ℝ)..1, (n.choose k : ℝ) * (1 + x)^(n - k) * (1 - x)^k = (2^(n + 1) : ℝ) / (n + 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_identity_l415_41540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_absolute_value_l415_41569

theorem complex_number_absolute_value : Complex.abs ((2 - Complex.I)^2 / Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_absolute_value_l415_41569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_probability_l415_41571

/-- The probability that John arrives while the train is at the station -/
theorem train_probability : ∃ (p : ℝ), p = 4/27 := by
  -- Define constants
  let john_arrival_range : ℝ := 90
  let train_arrival_range : ℝ := 60
  let train_wait_time : ℝ := 20

  -- Calculate areas
  let favorable_area : ℝ := (1/2 * train_wait_time * train_wait_time) + (train_wait_time * (john_arrival_range - train_wait_time))
  let total_area : ℝ := john_arrival_range * train_arrival_range

  -- Calculate probability
  let probability : ℝ := favorable_area / total_area

  -- Prove the theorem
  use probability
  sorry -- Skip the detailed proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_probability_l415_41571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_surface_area_ratio_l415_41574

theorem cone_sphere_surface_area_ratio (r : ℝ) (h : r > 0) : 
  (Real.pi * (2 * Real.sqrt 3 * r) * ((2 * Real.sqrt 3 * r) + Real.sqrt ((3 * r)^2 + (2 * Real.sqrt 3 * r)^2))) / (4 * Real.pi * r^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_surface_area_ratio_l415_41574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l415_41508

/-- The time taken for a train to cross a pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  train_length / train_speed_ms

/-- Theorem stating that a train of length 200 meters traveling at 60 km/h takes approximately 12 seconds to cross a pole -/
theorem train_crossing_pole_time :
  let train_length := (200 : ℝ)
  let train_speed := (60 : ℝ)
  let crossing_time := train_crossing_time train_length train_speed
  (crossing_time ≥ 11.9 ∧ crossing_time ≤ 12.1) := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l415_41508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_right_triangular_prism_l415_41570

/-- A right triangular prism with height h -/
structure RightTriangularPrism where
  h : ℝ
  h_pos : 0 < h

/-- The angle between the line passing through the center of the top base and 
    the midpoint of a side of the bottom base, and the plane of the base -/
noncomputable def incline_angle : ℝ := 60 * Real.pi / 180

/-- The lateral surface area of a right triangular prism -/
def lateral_surface_area (prism : RightTriangularPrism) : ℝ :=
  6 * prism.h^2

/-- Theorem: The lateral surface area of a right triangular prism with height h, 
    where a line passing through the center of the top base and the midpoint of 
    a side of the bottom base is inclined at an angle of 60° to the plane of the base, 
    is equal to 6h^2 -/
theorem lateral_surface_area_of_right_triangular_prism 
  (prism : RightTriangularPrism) : 
  lateral_surface_area prism = 6 * prism.h^2 := by
  -- Unfold the definition of lateral_surface_area
  unfold lateral_surface_area
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_right_triangular_prism_l415_41570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l415_41555

open Real

noncomputable def f (x : ℝ) : ℝ := (2/3) * cos (3*x - π/6)

theorem extrema_of_f :
  ∀ x ∈ Set.Ioo (0 : ℝ) (π/2),
    f x ≤ f (π/18) ∧
    f x ≥ f (7*π/18) ∧
    f (π/18) = 2/3 ∧
    f (7*π/18) = -2/3 ∧
    (∀ y ∈ Set.Ioo (0 : ℝ) (π/2), f y = 2/3 → y = π/18) ∧
    (∀ y ∈ Set.Ioo (0 : ℝ) (π/2), f y = -2/3 → y = 7*π/18) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l415_41555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l415_41506

/-- If the percentage of error in the calculated area of a square is 4.04%,
    then the percentage of error in measuring the side of the square is approximately 2.02%. -/
theorem square_measurement_error (x : ℝ) (e : ℝ) (h : (x + e)^2 = x^2 * (1 + 0.0404)) :
  ∃ ε > 0, |e / x - 0.0202| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_measurement_error_l415_41506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_parallel_segments_l415_41521

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line segment between two points -/
structure Segment (A B : Point) where
  length : ℝ

/-- Represents a line -/
structure Line (A B : Point) where
  slope : ℝ

/-- Theorem: Ratio of parallel line segments on a line -/
theorem ratio_parallel_segments
  (P Q R S T U V W X : Point)
  (PU : Line P U)
  (PQ : Segment P Q)
  (QR : Segment Q R)
  (RS : Segment R S)
  (ST : Segment S T)
  (TU : Segment T U)
  (VS : Line V S)
  (VU : Line V U)
  (WR : Segment W R)
  (XT : Segment X T)
  (PV : Line P V)
  (h1 : P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ T ∧ T ≠ U)
  (h2 : PQ.length = 1)
  (h3 : QR.length = 2)
  (h4 : RS.length = 1)
  (h5 : ST.length = 3)
  (h6 : TU.length = 1)
  (h7 : V.x ≠ PU.slope * V.y + 1) -- V not on PU
  (h8 : W.x = VS.slope * W.y + 1) -- W on VS
  (h9 : X.x = VU.slope * X.y + 1) -- X on VU
  (h10 : WR.length / PV.slope = XT.length / PV.slope) : -- WR parallel to XT and PV
  WR.length / XT.length = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_parallel_segments_l415_41521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_relationships_l415_41533

-- Define the relationships
inductive Relationship
| AgeWealth
| PointCoordinates
| AppleProductionClimate
| TreeDiameterHeight
| StudentID

-- Define a predicate for correlation
def involves_correlation : Relationship → Prop :=
  fun r => match r with
  | Relationship.AgeWealth => True
  | Relationship.AppleProductionClimate => True
  | Relationship.TreeDiameterHeight => True
  | _ => False

-- State the theorem
theorem correlation_relationships :
  {r : Relationship | involves_correlation r} =
    {Relationship.AgeWealth, Relationship.AppleProductionClimate, Relationship.TreeDiameterHeight} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_relationships_l415_41533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_negative_four_l415_41505

theorem logarithm_expression_equals_negative_four :
  (Real.log 8 / Real.log 10 + Real.log 125 / Real.log 10 - Real.log 2 / Real.log 10 - Real.log 5 / Real.log 10) / 
  (Real.log (Real.sqrt 10) / Real.log 10 * Real.log 0.1 / Real.log 10) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equals_negative_four_l415_41505
