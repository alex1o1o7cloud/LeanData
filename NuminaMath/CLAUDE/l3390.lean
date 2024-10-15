import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3390_339070

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3390_339070


namespace NUMINAMATH_CALUDE_final_total_cost_is_correct_l3390_339081

def spiral_notebook_price : ℝ := 15
def personal_planner_price : ℝ := 10
def spiral_notebook_discount_threshold : ℕ := 5
def personal_planner_discount_threshold : ℕ := 10
def spiral_notebook_discount_rate : ℝ := 0.2
def personal_planner_discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.07
def spiral_notebooks_bought : ℕ := 6
def personal_planners_bought : ℕ := 12

def calculate_discounted_price (price : ℝ) (quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  price * quantity * (1 - discount_rate)

def calculate_total_cost : ℝ :=
  let spiral_notebook_cost := 
    calculate_discounted_price spiral_notebook_price spiral_notebooks_bought spiral_notebook_discount_rate
  let personal_planner_cost := 
    calculate_discounted_price personal_planner_price personal_planners_bought personal_planner_discount_rate
  let subtotal := spiral_notebook_cost + personal_planner_cost
  subtotal * (1 + sales_tax_rate)

theorem final_total_cost_is_correct : calculate_total_cost = 186.18 := by sorry

end NUMINAMATH_CALUDE_final_total_cost_is_correct_l3390_339081


namespace NUMINAMATH_CALUDE_fundraising_goal_exceeded_l3390_339095

theorem fundraising_goal_exceeded (goal ken_amount : ℕ) 
  (h1 : ken_amount = 600)
  (h2 : goal = 4000) : 
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  ken_amount + mary_amount + scott_amount - goal = 600 := by
sorry

end NUMINAMATH_CALUDE_fundraising_goal_exceeded_l3390_339095


namespace NUMINAMATH_CALUDE_parabola_shift_l3390_339000

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- The original parabola y = -3x^2 -/
def original_parabola : Parabola :=
  { f := fun x => -3 * x^2 }

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The final parabola after shifting -/
def final_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 5) 2

theorem parabola_shift :
  final_parabola.f = fun x => -3 * (x - 5)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l3390_339000


namespace NUMINAMATH_CALUDE_inequality_proof_l3390_339030

theorem inequality_proof (x : ℝ) (h : 1 ≤ x ∧ x ≤ 5) : 2*x + 1/x + 1/(x+1) < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3390_339030


namespace NUMINAMATH_CALUDE_shaded_area_of_circles_l3390_339076

theorem shaded_area_of_circles (R : ℝ) (h : R = 8) : 
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let shaded_area := large_circle_area - 2 * small_circle_area
  shaded_area = 32 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_circles_l3390_339076


namespace NUMINAMATH_CALUDE_planes_through_three_points_l3390_339065

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), (p3.x - p1.x) = t * (p2.x - p1.x) ∧
             (p3.y - p1.y) = t * (p2.y - p1.y) ∧
             (p3.z - p1.z) = t * (p2.z - p1.z)

-- Define a function to count the number of planes through three points
def count_planes (p1 p2 p3 : Point3D) : Nat ⊕ Nat → Prop
  | Sum.inl 1 => ¬collinear p1 p2 p3
  | Sum.inr 0 => collinear p1 p2 p3
  | _ => False

-- Theorem statement
theorem planes_through_three_points (p1 p2 p3 : Point3D) :
  (count_planes p1 p2 p3 (Sum.inl 1)) ∨ (count_planes p1 p2 p3 (Sum.inr 0)) :=
sorry

end NUMINAMATH_CALUDE_planes_through_three_points_l3390_339065


namespace NUMINAMATH_CALUDE_fence_birds_count_l3390_339053

/-- The number of birds on a fence after new birds land -/
def total_birds (initial_pairs : ℕ) (birds_per_pair : ℕ) (new_birds : ℕ) : ℕ :=
  initial_pairs * birds_per_pair + new_birds

/-- Theorem stating the total number of birds on the fence -/
theorem fence_birds_count :
  let initial_pairs : ℕ := 12
  let birds_per_pair : ℕ := 2
  let new_birds : ℕ := 8
  total_birds initial_pairs birds_per_pair new_birds = 32 := by
  sorry

end NUMINAMATH_CALUDE_fence_birds_count_l3390_339053


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l3390_339048

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (∀ (k : ℕ), k > N → ¬(1743 % k = 2019 % k ∧ 2019 % k = 3008 % k)) ∧
  (1743 % N = 2019 % N ∧ 2019 % N = 3008 % N) :=
by sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l3390_339048


namespace NUMINAMATH_CALUDE_union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l3390_339059

-- Define the sets A and B
def A : Set ℝ := {x | (1 - 2*x) / (x - 3) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a ≤ 0}

-- Theorem 1: Union of A and B when a = -4
theorem union_A_B_when_a_neg_four :
  A ∪ B (-4) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Condition for (CᵣA) ∩ B = B
theorem complement_A_intersect_B_eq_B (a : ℝ) :
  (Aᶜ ∩ B a = B a) ↔ a > -1/4 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_neg_four_complement_A_intersect_B_eq_B_l3390_339059


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3390_339003

theorem boys_to_girls_ratio (B G : ℕ) (h_positive : B > 0 ∧ G > 0) : 
  (1/3 : ℚ) * B + (2/3 : ℚ) * G = (192/360 : ℚ) * (B + G) → 
  (B : ℚ) / G = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3390_339003


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3390_339071

-- Define the functions
def f (a b x : ℝ) : ℝ := -|x - a| + b
def g (c d x : ℝ) : ℝ := |x - c| + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) :
  (f a b 3 = 6 ∧ f a b 9 = 2) ∧
  (g c d 3 = 6 ∧ g c d 9 = 2) →
  a + c = 12 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3390_339071


namespace NUMINAMATH_CALUDE_g_divisibility_l3390_339008

def g : ℕ → ℕ
  | 0 => 1
  | n + 1 => g n ^ 2 + g n + 1

theorem g_divisibility (n : ℕ) : 
  (g n ^ 2 + 1) ∣ (g (n + 1) ^ 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_g_divisibility_l3390_339008


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3390_339019

/-- An arithmetic sequence with first term 18 and non-zero common difference d -/
def arithmeticSequence (d : ℝ) (n : ℕ) : ℝ := 18 + (n - 1 : ℝ) * d

theorem arithmetic_geometric_sequence (d : ℝ) (h1 : d ≠ 0) :
  (arithmeticSequence d 4) ^ 2 = (arithmeticSequence d 1) * (arithmeticSequence d 8) →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3390_339019


namespace NUMINAMATH_CALUDE_complex_number_simplification_l3390_339039

/-- Given a complex number z = (-1-2i) / (1+i)^2, prove that z = -1 + (1/2)i -/
theorem complex_number_simplification :
  let z : ℂ := (-1 - 2*I) / (1 + I)^2
  z = -1 + (1/2)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l3390_339039


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3390_339096

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x + 7)^2 + (y + 4)^2 = 64) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3390_339096


namespace NUMINAMATH_CALUDE_x_value_in_terms_of_acd_l3390_339049

theorem x_value_in_terms_of_acd (x y z a b c d : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : y * z / (y - z) = d) :
  x = 2 * a * c / (a - c - d) := by
sorry

end NUMINAMATH_CALUDE_x_value_in_terms_of_acd_l3390_339049


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l3390_339038

theorem unique_solution_of_equation :
  ∃! (x y z : ℝ), x^2 + 5*y^2 + 5*z^2 - 4*x*z - 2*y - 4*y*z + 1 = 0 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l3390_339038


namespace NUMINAMATH_CALUDE_complex_power_eight_l3390_339017

theorem complex_power_eight : (3 * Complex.cos (π / 4) - 3 * Complex.I * Complex.sin (π / 4)) ^ 8 = 6552 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l3390_339017


namespace NUMINAMATH_CALUDE_percentage_problem_l3390_339016

theorem percentage_problem (p : ℝ) : p = 80 :=
  by
  -- Define the number as 15
  let number : ℝ := 15
  
  -- Define the condition: 40% of 15 is greater than p% of 5 by 2
  have h : 0.4 * number = p / 100 * 5 + 2 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3390_339016


namespace NUMINAMATH_CALUDE_divisibility_property_l3390_339007

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3390_339007


namespace NUMINAMATH_CALUDE_problem_statement_l3390_339012

theorem problem_statement (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |x| = 2) : 
  x^4 + c*d*x^2 - a - b = 20 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3390_339012


namespace NUMINAMATH_CALUDE_folded_paper_properties_l3390_339025

/-- Represents a folded rectangular paper with specific properties -/
structure FoldedPaper where
  short_edge : ℝ
  long_edge : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- Theorem stating the properties of the folded paper -/
theorem folded_paper_properties (paper : FoldedPaper) 
  (h1 : paper.short_edge = 12)
  (h2 : paper.long_edge = 18)
  (h3 : paper.congruent_triangles)
  : paper.fold_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_properties_l3390_339025


namespace NUMINAMATH_CALUDE_opposite_leg_length_l3390_339064

/-- Represents a right triangle with a 30° angle -/
structure RightTriangle30 where
  /-- Length of the hypotenuse -/
  hypotenuse : ℝ
  /-- Length of the leg opposite to the 30° angle -/
  opposite_leg : ℝ
  /-- Constraint that the hypotenuse is twice the opposite leg -/
  hyp_constraint : hypotenuse = 2 * opposite_leg

/-- 
Theorem: In a right triangle with a 30° angle and hypotenuse of 18 inches, 
the leg opposite to the 30° angle is 9 inches long.
-/
theorem opposite_leg_length (triangle : RightTriangle30) 
  (h : triangle.hypotenuse = 18) : triangle.opposite_leg = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_leg_length_l3390_339064


namespace NUMINAMATH_CALUDE_square_root_equality_l3390_339032

theorem square_root_equality (x a : ℝ) (hx : x > 0) : 
  Real.sqrt x = 2 * a - 3 ∧ Real.sqrt x = 5 - a → a = 8/3 ∧ x = 49/9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l3390_339032


namespace NUMINAMATH_CALUDE_picnic_adult_child_difference_l3390_339018

/-- A picnic scenario with men, women, adults, and children -/
structure Picnic where
  total : ℕ
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Conditions for the picnic scenario -/
def PicnicConditions (p : Picnic) : Prop :=
  p.total = 240 ∧
  p.men = p.women + 40 ∧
  p.men = 90 ∧
  p.adults > p.children ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

/-- Theorem stating the difference between adults and children -/
theorem picnic_adult_child_difference (p : Picnic) 
  (h : PicnicConditions p) : p.adults - p.children = 40 := by
  sorry

#check picnic_adult_child_difference

end NUMINAMATH_CALUDE_picnic_adult_child_difference_l3390_339018


namespace NUMINAMATH_CALUDE_problem_solution_l3390_339042

theorem problem_solution :
  (∀ x y : ℝ, 28 * x^4 * y^2 / (7 * x^3 * y) = 4 * x * y) ∧
  ((2 * (1/3 : ℝ) + 3 * (1/2 : ℝ))^2 - (2 * (1/3 : ℝ) + (1/2 : ℝ)) * (2 * (1/3 : ℝ) - (1/2 : ℝ)) = 4.5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3390_339042


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l3390_339027

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.25 * A
  let r' := Real.sqrt (A' / π)
  r' / r = 0.5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l3390_339027


namespace NUMINAMATH_CALUDE_orchestra_admission_l3390_339060

theorem orchestra_admission (initial_ratio_violinists : ℝ) (initial_ratio_cellists : ℝ) (initial_ratio_trumpeters : ℝ)
  (violinist_increase : ℝ) (cellist_decrease : ℝ) (total_admitted : ℕ) :
  initial_ratio_violinists = 1.6 →
  initial_ratio_cellists = 1 →
  initial_ratio_trumpeters = 0.4 →
  violinist_increase = 0.25 →
  cellist_decrease = 0.2 →
  total_admitted = 32 →
  ∃ (violinists cellists trumpeters : ℕ),
    violinists = 20 ∧
    cellists = 8 ∧
    trumpeters = 4 ∧
    violinists + cellists + trumpeters = total_admitted :=
by sorry

end NUMINAMATH_CALUDE_orchestra_admission_l3390_339060


namespace NUMINAMATH_CALUDE_curve_symmetry_l3390_339028

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Theorem: The curve is symmetrical about the line x + y = 0
theorem curve_symmetry :
  ∀ (x y : ℝ), curve x y → 
  ∃ (x' y' : ℝ), curve x' y' ∧ symmetry_line ((x + x')/2) ((y + y')/2) :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l3390_339028


namespace NUMINAMATH_CALUDE_sine_function_omega_l3390_339024

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, 
    if f(π/6) = f(π/3) and f(x) has a maximum value but no minimum value 
    in the interval (π/6, π/3), then ω = 2/3 -/
theorem sine_function_omega (ω : ℝ) (h_pos : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 3)
  (f (π / 6) = f (π / 3)) → 
  (∃ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) ∧ 
    (∀ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) → f y ≤ f x)) →
  (∀ (x : ℝ), x ∈ Set.Ioo (π / 6) (π / 3) → 
    ∃ (y : ℝ), y ∈ Set.Ioo (π / 6) (π / 3) ∧ f y < f x) →
  ω = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_omega_l3390_339024


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l3390_339063

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 2311 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), n % d = 1) ∧ 
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), m % d = 1) → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l3390_339063


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3390_339052

theorem tan_theta_minus_pi_fourth (θ : Real) :
  (-π/2 < θ) → (θ < 0) → -- θ is in the fourth quadrant
  (Real.sin (θ + π/4) = 3/5) →
  (Real.tan (θ - π/4) = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l3390_339052


namespace NUMINAMATH_CALUDE_range_of_f_l3390_339078

-- Define the function f
def f (x : ℝ) : ℝ := (x - 2)^2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3,
  ∃ y ∈ Set.Ico 0 9,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Ico 0 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3390_339078


namespace NUMINAMATH_CALUDE_point_in_bottom_right_region_of_line_l3390_339068

/-- A point (x, y) is in the bottom-right region of the line ax + by + c = 0 (including the boundary) if ax + by + c ≥ 0 -/
def in_bottom_right_region (a b c x y : ℝ) : Prop := a * x + b * y + c ≥ 0

theorem point_in_bottom_right_region_of_line (t : ℝ) :
  in_bottom_right_region 1 (-2) 4 2 t → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_bottom_right_region_of_line_l3390_339068


namespace NUMINAMATH_CALUDE_mary_eggs_count_l3390_339029

/-- Given that Mary starts with 27 eggs and finds 4 more eggs, prove that she ends up with 31 eggs in total. -/
theorem mary_eggs_count (initial_eggs found_eggs : ℕ) : 
  initial_eggs = 27 → found_eggs = 4 → initial_eggs + found_eggs = 31 := by
  sorry

end NUMINAMATH_CALUDE_mary_eggs_count_l3390_339029


namespace NUMINAMATH_CALUDE_minimum_m_value_l3390_339066

theorem minimum_m_value (a b c m : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : m > 0) 
  (h4 : (1 / (a - b)) + (m / (b - c)) ≥ (9 / (a - c))) : 
  m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_minimum_m_value_l3390_339066


namespace NUMINAMATH_CALUDE_black_chess_pieces_count_l3390_339089

theorem black_chess_pieces_count 
  (white_pieces : ℕ) 
  (white_probability : ℚ) 
  (h1 : white_pieces = 9)
  (h2 : white_probability = 3/10) : 
  ∃ (black_pieces : ℕ), 
    (white_pieces : ℚ) / (white_pieces + black_pieces) = white_probability ∧ 
    black_pieces = 21 := by
  sorry

end NUMINAMATH_CALUDE_black_chess_pieces_count_l3390_339089


namespace NUMINAMATH_CALUDE_horner_V₃_eq_71_l3390_339085

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- Coefficients of the polynomial f(x) = 2x⁶ + 5x⁵ + 6x⁴ + 23x³ - 8x² + 10x - 3 -/
def f_coeffs : List ℚ := [2, 5, 6, 23, -8, 10, -3]

/-- The value of x -/
def x : ℚ := 2

/-- V₃ in Horner's method -/
def V₃ : ℚ := 
  let v₀ : ℚ := f_coeffs[0]!
  let v₁ : ℚ := v₀ * x + f_coeffs[1]!
  let v₂ : ℚ := v₁ * x + f_coeffs[2]!
  v₂ * x + f_coeffs[3]!

theorem horner_V₃_eq_71 : V₃ = 71 := by
  sorry

#eval V₃

end NUMINAMATH_CALUDE_horner_V₃_eq_71_l3390_339085


namespace NUMINAMATH_CALUDE_present_expenditure_l3390_339062

theorem present_expenditure (P : ℝ) : 
  P * (1 + 0.1)^2 = 24200.000000000004 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_present_expenditure_l3390_339062


namespace NUMINAMATH_CALUDE_range_of_a_l3390_339037

theorem range_of_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - a) ≥ 5) → a ≥ (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3390_339037


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l3390_339021

/-- Proves that given the conditions, the ratio of father's age to son's age is 19:7 -/
theorem father_son_age_ratio :
  ∀ (son_age father_age : ℕ),
    (father_age - 6 = 3 * (son_age - 6)) →
    (son_age + father_age = 156) →
    (father_age : ℚ) / son_age = 19 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l3390_339021


namespace NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l3390_339051

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 4) / (4 * x^2 + 7 * x + 3)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N : ℝ, ∀ x > N, |f x - 7/4| < ε :=
by sorry

end NUMINAMATH_CALUDE_horizontal_asymptote_of_f_l3390_339051


namespace NUMINAMATH_CALUDE_chair_price_l3390_339005

theorem chair_price (num_tables : ℕ) (num_chairs : ℕ) (total_cost : ℕ) 
  (h1 : num_tables = 2)
  (h2 : num_chairs = 3)
  (h3 : total_cost = 110)
  (h4 : ∀ (chair_price : ℕ), num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost) :
  ∃ (chair_price : ℕ), chair_price = 10 ∧ 
    num_tables * (4 * chair_price) + num_chairs * chair_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chair_price_l3390_339005


namespace NUMINAMATH_CALUDE_harry_pumpkin_packets_l3390_339084

/-- The number of pumpkin seed packets Harry bought -/
def pumpkin_packets : ℕ := 3

/-- The cost of one packet of pumpkin seeds in dollars -/
def pumpkin_cost : ℚ := 2.5

/-- The cost of one packet of tomato seeds in dollars -/
def tomato_cost : ℚ := 1.5

/-- The cost of one packet of chili pepper seeds in dollars -/
def chili_cost : ℚ := 0.9

/-- The number of tomato seed packets Harry bought -/
def tomato_packets : ℕ := 4

/-- The number of chili pepper seed packets Harry bought -/
def chili_packets : ℕ := 5

/-- The total amount Harry spent in dollars -/
def total_spent : ℚ := 18

theorem harry_pumpkin_packets :
  pumpkin_packets * pumpkin_cost + 
  tomato_packets * tomato_cost + 
  chili_packets * chili_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_harry_pumpkin_packets_l3390_339084


namespace NUMINAMATH_CALUDE_initial_oranges_l3390_339015

/-- Theorem: Initial number of oranges in the bin -/
theorem initial_oranges (thrown_away removed : ℕ) (added new_count : ℕ) :
  removed = 25 →
  added = 21 →
  new_count = 36 →
  ∃ initial : ℕ, initial - removed + added = new_count ∧ initial = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_l3390_339015


namespace NUMINAMATH_CALUDE_bobs_family_adults_l3390_339031

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : apples_per_child = 10)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 40 := by
  sorry

end NUMINAMATH_CALUDE_bobs_family_adults_l3390_339031


namespace NUMINAMATH_CALUDE_grapes_and_watermelon_cost_l3390_339022

/-- The cost of a pack of peanuts -/
def peanuts_cost : ℝ := sorry

/-- The cost of a cluster of grapes -/
def grapes_cost : ℝ := sorry

/-- The cost of a watermelon -/
def watermelon_cost : ℝ := sorry

/-- The cost of a box of figs -/
def figs_cost : ℝ := sorry

/-- The total cost of all items -/
def total_cost : ℝ := 30

/-- The statement of the problem -/
theorem grapes_and_watermelon_cost :
  (peanuts_cost + grapes_cost + watermelon_cost + figs_cost = total_cost) →
  (figs_cost = 2 * peanuts_cost) →
  (watermelon_cost = peanuts_cost - grapes_cost) →
  (grapes_cost + watermelon_cost = 7.5) :=
by sorry

end NUMINAMATH_CALUDE_grapes_and_watermelon_cost_l3390_339022


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_02008_l3390_339014

def original_number : ℝ := 0.02008

def scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_0_02008 :
  scientific_notation original_number 3 = (2.01, -2) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_02008_l3390_339014


namespace NUMINAMATH_CALUDE_a_is_arithmetic_sequence_b_max_min_l3390_339044

-- Define the sequence a_n implicitly through S_n
def S (n : ℕ+) : ℚ := (1 / 2) * n^2 - 2 * n

-- Define a_n as the difference of consecutive S_n terms
def a (n : ℕ+) : ℚ := S n - S (n - 1)

-- Define b_n
def b (n : ℕ+) : ℚ := (a n + 1) / (a n)

-- Theorem 1: a_n is an arithmetic sequence with common difference 1
theorem a_is_arithmetic_sequence : ∀ n : ℕ+, n > 1 → a (n + 1) - a n = 1 :=
sorry

-- Theorem 2: Maximum and minimum values of b_n
theorem b_max_min :
  (∀ n : ℕ+, b n ≤ b 3) ∧
  (∀ n : ℕ+, b n ≥ b 2) ∧
  (b 3 = 3) ∧
  (b 2 = -1) :=
sorry

end NUMINAMATH_CALUDE_a_is_arithmetic_sequence_b_max_min_l3390_339044


namespace NUMINAMATH_CALUDE_expand_expression_l3390_339090

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3*x - 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3390_339090


namespace NUMINAMATH_CALUDE_cookies_difference_l3390_339082

def cookies_bought : ℝ := 125.75
def cookies_eaten : ℝ := 8.5

theorem cookies_difference : cookies_bought - cookies_eaten = 117.25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l3390_339082


namespace NUMINAMATH_CALUDE_number_solution_l3390_339083

theorem number_solution (x : ℝ) : 0.6 * x = 0.3 * 10 + 27 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3390_339083


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3390_339073

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > (1 / 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3390_339073


namespace NUMINAMATH_CALUDE_fraction_order_l3390_339035

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l3390_339035


namespace NUMINAMATH_CALUDE_equal_percentage_price_l3390_339079

/-- Represents the cost and various selling prices of an article -/
structure Article where
  cp : ℝ  -- Cost price
  sp_profit : ℝ  -- Selling price with 25% profit
  sp_loss : ℝ  -- Selling price with loss
  sp_equal : ℝ  -- Selling price where % profit = % loss

/-- Conditions for the article pricing problem -/
def article_conditions (a : Article) : Prop :=
  a.sp_profit = a.cp * 1.25 ∧  -- 25% profit condition
  a.sp_profit = 1625 ∧
  a.sp_loss = 1280 ∧
  a.sp_loss < a.cp  -- Ensures sp_loss results in a loss

/-- Theorem stating the selling price where percentage profit equals percentage loss -/
theorem equal_percentage_price (a : Article) 
  (h : article_conditions a) : a.sp_equal = 1320 := by
  sorry

#check equal_percentage_price

end NUMINAMATH_CALUDE_equal_percentage_price_l3390_339079


namespace NUMINAMATH_CALUDE_factorization_proof_l3390_339045

theorem factorization_proof (a : ℝ) : 180 * a^2 + 45 * a = 45 * a * (4 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3390_339045


namespace NUMINAMATH_CALUDE_max_halls_visited_l3390_339086

structure Museum :=
  (total_halls : ℕ)
  (painting_halls : ℕ)
  (sculpture_halls : ℕ)
  (is_even : total_halls % 2 = 0)
  (half_paintings : painting_halls = total_halls / 2)
  (half_sculptures : sculpture_halls = total_halls / 2)

def alternating_tour (m : Museum) (start_painting : Bool) (end_painting : Bool) : ℕ → Prop
  | 0 => start_painting
  | 1 => ¬start_painting
  | (n+2) => alternating_tour m start_painting end_painting n

theorem max_halls_visited 
  (m : Museum) 
  (h : m.total_halls = 16) 
  (start_painting : Bool) 
  (end_painting : Bool) 
  (h_start_end : start_painting = end_painting) :
  ∃ (n : ℕ), n ≤ m.total_halls - 1 ∧ 
    alternating_tour m start_painting end_painting n ∧ 
    ∀ (k : ℕ), alternating_tour m start_painting end_painting k → k ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_halls_visited_l3390_339086


namespace NUMINAMATH_CALUDE_aluminum_carbonate_molecular_weight_l3390_339057

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of the given moles of Aluminum carbonate in grams -/
def given_molecular_weight : ℝ := 1170

/-- The molecular formula of Aluminum carbonate: Al₂(CO₃)₃ -/
structure AluminumCarbonate where
  Al : Nat
  C : Nat
  O : Nat

/-- The correct molecular formula of Aluminum carbonate -/
def Al2CO3_3 : AluminumCarbonate := ⟨2, 3, 9⟩

/-- Calculate the molecular weight of Aluminum carbonate -/
def molecular_weight (formula : AluminumCarbonate) : ℝ :=
  formula.Al * atomic_weight_Al + formula.C * atomic_weight_C + formula.O * atomic_weight_O

/-- Theorem: The molecular weight of Aluminum carbonate is approximately 234.99 g/mol -/
theorem aluminum_carbonate_molecular_weight :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |molecular_weight Al2CO3_3 - 234.99| < ε :=
sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_molecular_weight_l3390_339057


namespace NUMINAMATH_CALUDE_lola_cupcakes_count_l3390_339058

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := sorry

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

theorem lola_cupcakes_count : lola_cupcakes = 13 := by
  sorry

end NUMINAMATH_CALUDE_lola_cupcakes_count_l3390_339058


namespace NUMINAMATH_CALUDE_negative_reciprocal_of_negative_three_l3390_339026

theorem negative_reciprocal_of_negative_three :
  -(1 / -3) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_negative_reciprocal_of_negative_three_l3390_339026


namespace NUMINAMATH_CALUDE_odd_number_power_divisibility_l3390_339041

theorem odd_number_power_divisibility (a : ℕ) (h_odd : Odd a) :
  (∀ m : ℕ, ∃ (k : ℕ → ℕ), Function.Injective k ∧ ∀ n : ℕ, (a ^ (k n) - 1) % (2 ^ m) = 0) ∧
  (∃ (S : Finset ℕ), ∀ m : ℕ, (a ^ m - 1) % (2 ^ m) = 0 → m ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_odd_number_power_divisibility_l3390_339041


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l3390_339013

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem base_subtraction_proof :
  let base7_num := base_to_decimal [5, 4, 3, 2, 1, 0] 7
  let base8_num := base_to_decimal [4, 5, 3, 2, 1] 8
  base7_num - base8_num = 75620 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l3390_339013


namespace NUMINAMATH_CALUDE_subtract_three_numbers_l3390_339091

theorem subtract_three_numbers : 15 - 3 - 15 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_three_numbers_l3390_339091


namespace NUMINAMATH_CALUDE_tangent_circle_slope_l3390_339033

/-- Circle represented by its equation in the form x² + y² + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line y = mx contains the center of a circle tangent to two given circles -/
def has_tangent_circle (w₁ w₂ : Circle) (m : ℝ) : Prop :=
  ∃ (x y r : ℝ),
    y = m * x ∧
    (x - 4)^2 + (y - 10)^2 = (r + 3)^2 ∧
    (x + 4)^2 + (y - 10)^2 = (11 - r)^2

/-- The main theorem -/
theorem tangent_circle_slope (w₁ w₂ : Circle) :
  w₁.a = 8 ∧ w₁.b = -20 ∧ w₁.c = -75 ∧
  w₂.a = -8 ∧ w₂.b = -20 ∧ w₂.c = 125 →
  ∃ (m : ℝ),
    m > 0 ∧
    has_tangent_circle w₁ w₂ m ∧
    (∀ m' : ℝ, 0 < m' ∧ m' < m → ¬ has_tangent_circle w₁ w₂ m') ∧
    m^2 = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_slope_l3390_339033


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3390_339046

theorem smallest_number_with_remainders (n : ℕ) :
  (∃ m : ℕ, n = 5 * m + 4) ∧
  (∃ m : ℕ, n = 6 * m + 5) ∧
  (((∃ m : ℕ, n = 7 * m + 6) → n ≥ 209) ∧
   ((∃ m : ℕ, n = 8 * m + 7) → n ≥ 119)) ∧
  (n = 209 ∨ n = 119) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3390_339046


namespace NUMINAMATH_CALUDE_forty_percent_value_l3390_339094

theorem forty_percent_value (x : ℝ) (h : 0.5 * x = 200) : 0.4 * x = 160 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l3390_339094


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3390_339099

/-- Given two quadratic functions f and g, if the sum of roots of f equals the product of roots of g,
    and the product of roots of f equals the sum of roots of g, then f attains its minimum at x = 3 -/
theorem quadratic_minimum (r s : ℝ) : 
  let f (x : ℝ) := x^2 + r*x + s
  let g (x : ℝ) := x^2 - 9*x + 6
  let sum_roots_f := -r
  let prod_roots_f := s
  let sum_roots_g := 9
  let prod_roots_g := 6
  (sum_roots_f = prod_roots_g) → (prod_roots_f = sum_roots_g) →
  ∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), f x ≥ f a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3390_339099


namespace NUMINAMATH_CALUDE_rush_order_cost_rush_order_cost_is_five_l3390_339074

/-- Calculate the extra amount paid for a rush order given the following conditions:
  * There are 4 people ordering dinner
  * Each main meal costs $12.0
  * 2 appetizers are ordered at $6.00 each
  * A 20% tip is included
  * The total amount spent is $77
-/
theorem rush_order_cost (num_people : ℕ) (main_meal_cost : ℚ) (num_appetizers : ℕ) 
  (appetizer_cost : ℚ) (tip_percentage : ℚ) (total_spent : ℚ) : ℚ :=
  let subtotal := num_people * main_meal_cost + num_appetizers * appetizer_cost
  let tip := subtotal * tip_percentage
  let total_before_rush := subtotal + tip
  total_spent - total_before_rush

/-- The extra amount paid for the rush order is $5.0 -/
theorem rush_order_cost_is_five : 
  rush_order_cost 4 12 2 6 (1/5) 77 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rush_order_cost_rush_order_cost_is_five_l3390_339074


namespace NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l3390_339009

theorem sqrt_inequality_equivalence : 
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔ 
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 3 + Real.sqrt 6)^2) := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l3390_339009


namespace NUMINAMATH_CALUDE_sunset_colors_l3390_339056

/-- The duration of the sunset in hours -/
def sunset_duration : ℕ := 2

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The interval between color changes in minutes -/
def color_change_interval : ℕ := 10

/-- The number of colors the sky turns during the sunset -/
def number_of_colors : ℕ := sunset_duration * minutes_per_hour / color_change_interval

theorem sunset_colors :
  number_of_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_sunset_colors_l3390_339056


namespace NUMINAMATH_CALUDE_even_plus_abs_odd_is_even_l3390_339010

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem even_plus_abs_odd_is_even
  (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsEven (fun x ↦ f x + |g x|) := by
  sorry

end NUMINAMATH_CALUDE_even_plus_abs_odd_is_even_l3390_339010


namespace NUMINAMATH_CALUDE_sin_x_equals_x_unique_root_l3390_339088

theorem sin_x_equals_x_unique_root :
  ∃! x : ℝ, x ∈ Set.Icc (-π) π ∧ x = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_sin_x_equals_x_unique_root_l3390_339088


namespace NUMINAMATH_CALUDE_domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l3390_339002

-- Define a function type
def Function (α β : Type) := α → β

-- Statement 1
theorem domain_and_rule_determine_function (α β : Type) :
  ∀ (D : Set α) (f : Function α β), ∃! (F : Function α β), ∀ x ∈ D, F x = f x :=
sorry

-- Statement 2
theorem exists_non_increasing_power_function :
  ∃ (n : ℝ), ¬ (∀ x y : ℝ, 0 < x ∧ x < y → x^n < y^n) :=
sorry

-- Statement 3
theorem exists_function_without_zero :
  ∃ (f : ℝ → ℝ) (a b : ℝ), a ≠ b ∧ f a > 0 ∧ f b < 0 ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0) :=
sorry

end NUMINAMATH_CALUDE_domain_and_rule_determine_function_exists_non_increasing_power_function_exists_function_without_zero_l3390_339002


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3390_339050

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 2 * x + y = 7) 
  (eq2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3390_339050


namespace NUMINAMATH_CALUDE_range_of_t_l3390_339077

def A (t : ℝ) : Set ℝ := {1, t}

theorem range_of_t (t : ℝ) : t ∈ {x : ℝ | x ≠ 1} ↔ t ∈ A t := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l3390_339077


namespace NUMINAMATH_CALUDE_magnitude_of_AB_l3390_339001

def vector_AB : ℝ × ℝ := (1, 1)

theorem magnitude_of_AB : Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_AB_l3390_339001


namespace NUMINAMATH_CALUDE_polygon_sides_l3390_339067

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360) →
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3390_339067


namespace NUMINAMATH_CALUDE_quadratic_equations_solution_l3390_339011

theorem quadratic_equations_solution (m n k : ℝ) : 
  (∃ x : ℝ, m * x^2 + n = 0) ∧
  (∃ x : ℝ, n * x^2 + k = 0) ∧
  (∃ x : ℝ, k * x^2 + m = 0) →
  m = 0 ∧ n = 0 ∧ k = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_solution_l3390_339011


namespace NUMINAMATH_CALUDE_first_degree_function_composition_l3390_339034

theorem first_degree_function_composition (f : ℝ → ℝ) :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_first_degree_function_composition_l3390_339034


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_one_l3390_339087

theorem x_squared_plus_y_squared_equals_one
  (x y : ℝ)
  (h1 : (x^2 + y^2 + 1) * (x^2 + y^2 + 3) = 8)
  (h2 : x^2 + y^2 ≥ 0) :
  x^2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_one_l3390_339087


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3390_339098

theorem profit_percentage_calculation (selling_price profit : ℝ) 
  (h1 : selling_price = 900)
  (h2 : profit = 150) :
  (profit / (selling_price - profit)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3390_339098


namespace NUMINAMATH_CALUDE_total_liquid_drunk_l3390_339040

/-- Converts pints to cups -/
def pints_to_cups (pints : ℝ) : ℝ := 2 * pints

/-- The amount of coffee Elijah drank in pints -/
def elijah_coffee : ℝ := 8.5

/-- The amount of water Emilio drank in pints -/
def emilio_water : ℝ := 9.5

/-- Theorem: The total amount of liquid drunk by Elijah and Emilio is 36 cups -/
theorem total_liquid_drunk : 
  pints_to_cups elijah_coffee + pints_to_cups emilio_water = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_liquid_drunk_l3390_339040


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l3390_339092

-- Problem 1
theorem sqrt_three_minus_sqrt_two_plus_sqrt_six : 
  Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 2) + Real.sqrt 6 = 3 := by sorry

-- Problem 2
theorem simplify_sqrt_expression (a : ℝ) (ha : a > 0) : 
  2 * Real.sqrt (12 * a) + Real.sqrt (6 * a^2) + Real.sqrt (2 * a) = 
  4 * Real.sqrt (3 * a) + Real.sqrt 6 * a + Real.sqrt (2 * a) := by sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l3390_339092


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3390_339093

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 3 * f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (-1) = 7 → f (-1001) = -3493 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3390_339093


namespace NUMINAMATH_CALUDE_square_of_T_number_is_T_number_l3390_339004

/-- Definition of a T number -/
def is_T_number (x : ℤ) : Prop := ∃ (a b : ℤ), x = a^2 + a*b + b^2

/-- Theorem: The square of a T number is still a T number -/
theorem square_of_T_number_is_T_number (x : ℤ) (h : is_T_number x) : is_T_number (x^2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_T_number_is_T_number_l3390_339004


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3390_339047

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3390_339047


namespace NUMINAMATH_CALUDE_existence_of_irreducible_fractions_l3390_339075

theorem existence_of_irreducible_fractions : ∃ (a b : ℕ), 
  (Nat.gcd a b = 1) ∧ 
  (Nat.gcd (a + 1) b = 1) ∧ 
  (Nat.gcd (a + 1) (b + 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irreducible_fractions_l3390_339075


namespace NUMINAMATH_CALUDE_identity_function_proof_l3390_339036

theorem identity_function_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l3390_339036


namespace NUMINAMATH_CALUDE_square_diagonal_point_l3390_339072

-- Define the square EFGH
def Square (E F G H : ℝ × ℝ) : Prop :=
  let side := dist E F
  dist E F = side ∧ dist F G = side ∧ dist G H = side ∧ dist H E = side ∧
  (E.1 - G.1) * (F.1 - H.1) + (E.2 - G.2) * (F.2 - H.2) = 0

-- Define point Q on diagonal AC
def OnDiagonal (Q E G : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t * E.1 + (1 - t) * G.1, t * E.2 + (1 - t) * G.2)

-- Define circumcenter
def Circumcenter (R E F Q : ℝ × ℝ) : Prop :=
  dist R E = dist R F ∧ dist R F = dist R Q

-- Main theorem
theorem square_diagonal_point (E F G H Q R₁ R₂ : ℝ × ℝ) :
  Square E F G H →
  dist E F = 8 →
  OnDiagonal Q E G →
  dist E Q > dist G Q →
  Circumcenter R₁ E F Q →
  Circumcenter R₂ G H Q →
  (R₁.1 - Q.1) * (R₂.1 - Q.1) + (R₁.2 - Q.2) * (R₂.2 - Q.2) = 0 →
  dist E Q = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_point_l3390_339072


namespace NUMINAMATH_CALUDE_nancy_count_l3390_339043

theorem nancy_count (a b c d e f : ℕ) (h_mean : (a + b + c + d + e + f) / 6 = 7)
  (h_a : a = 6) (h_b : b = 12) (h_c : c = 1) (h_d : d = 12) (h_f : f = 8) :
  e = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_count_l3390_339043


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l3390_339061

-- First equation
theorem solution_equation_one : 
  ∃ x : ℝ, (2 - x) / (x - 3) = 3 / (3 - x) ↔ x = 5 := by sorry

-- Second equation
theorem solution_equation_two : 
  ∃ x : ℝ, 4 / (x^2 - 1) + 1 = (x - 1) / (x + 1) ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l3390_339061


namespace NUMINAMATH_CALUDE_vasya_always_wins_l3390_339069

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player

/-- Represents a move in the game -/
inductive Move : Type
| Positive : Move
| Negative : Move

/-- Represents the game state -/
structure GameState :=
(moves : List Move)
(current_player : Player)

/-- The number of divisions on each side of the triangle -/
def n : Nat := 2008

/-- The total number of cells in the triangle -/
def total_cells : Nat := n * n

/-- Determines the winner based on the final game state -/
def winner (final_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Vasya always wins -/
theorem vasya_always_wins :
  ∀ (game : GameState),
  game.moves.length = total_cells →
  game.current_player = Player.Vasya →
  winner game = Player.Vasya :=
sorry

end NUMINAMATH_CALUDE_vasya_always_wins_l3390_339069


namespace NUMINAMATH_CALUDE_opposite_numbers_not_just_opposite_signs_l3390_339054

theorem opposite_numbers_not_just_opposite_signs : ¬ (∀ a b : ℝ, (a > 0 ∧ b < 0) → (a = -b)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_not_just_opposite_signs_l3390_339054


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3390_339097

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3390_339097


namespace NUMINAMATH_CALUDE_apollo_chariot_wheels_cost_l3390_339055

/-- Represents the cost in golden apples for chariot wheels over a year -/
structure ChariotWheelsCost where
  total : ℕ  -- Total cost for the year
  second_half_multiplier : ℕ  -- Multiplier for the second half of the year

/-- 
Calculates the cost for the first half of the year given the total cost
and the multiplier for the second half of the year.
-/
def first_half_cost (c : ChariotWheelsCost) : ℕ :=
  c.total / (1 + c.second_half_multiplier)

/-- 
Theorem: If the total cost for a year is 54 golden apples, and the cost for the 
second half of the year is double the cost for the first half, then the cost 
for the first half of the year is 18 golden apples.
-/
theorem apollo_chariot_wheels_cost : 
  let c := ChariotWheelsCost.mk 54 2
  first_half_cost c = 18 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_wheels_cost_l3390_339055


namespace NUMINAMATH_CALUDE_remainder_of_1731_base12_div_9_l3390_339080

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

/-- The base-12 representation of the number --/
def base12Number : List Nat := [1, 7, 3, 1]

theorem remainder_of_1731_base12_div_9 :
  (base12ToDecimal base12Number) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1731_base12_div_9_l3390_339080


namespace NUMINAMATH_CALUDE_water_container_problem_l3390_339023

theorem water_container_problem :
  ∀ (x : ℝ),
    x > 0 →
    x / 2 + (2 * x) / 3 + (4 * x) / 4 = 26 →
    x + 2 * x + 4 * x + 26 = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_water_container_problem_l3390_339023


namespace NUMINAMATH_CALUDE_student_group_allocation_schemes_l3390_339006

theorem student_group_allocation_schemes (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n = 12) 
  (h2 : k = 4) 
  (h3 : m = 3) 
  (h4 : n = k * m) : 
  (Nat.choose n m * Nat.choose (n - m) m * Nat.choose (n - 2*m) m * m^k : ℕ) = 
  (Nat.choose 12 3 * Nat.choose 9 3 * Nat.choose 6 3 * 3^4 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_student_group_allocation_schemes_l3390_339006


namespace NUMINAMATH_CALUDE_calculation_proof_l3390_339020

theorem calculation_proof : (4 + 6 + 10) / 3 - 2 / 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3390_339020
