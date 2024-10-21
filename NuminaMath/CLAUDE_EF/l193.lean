import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_parabola_l193_19397

/-- The maximum slope of OM given a parabola y²=2px -/
theorem max_slope_parabola (p : ℝ) (h : p > 0) :
  ∃ (max_slope : ℝ), 
    (∀ (y₀ : ℝ), y₀ > 0 → 
      (y₀^2 / 3) / (y₀^2 / (6*p) + p/3) ≤ max_slope) ∧
    (∃ (y₀ : ℝ), y₀ > 0 ∧
      (y₀^2 / 3) / (y₀^2 / (6*p) + p/3) = max_slope) ∧
    max_slope = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_parabola_l193_19397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l193_19318

/-- Ellipse parameters -/
structure EllipseParams where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- Ellipse equation -/
def ellipseEq (p : EllipseParams) (x y : ℝ) : Prop :=
  x^2 / p.a^2 + y^2 / p.b^2 = 1

/-- Focal distance -/
noncomputable def focalDistance (p : EllipseParams) : ℝ := 2 * Real.sqrt 6

/-- Sum of distances from any point on ellipse to foci -/
def sumDistancesToFoci (p : EllipseParams) : ℝ := 6

/-- Line equation -/
def lineEq (x y : ℝ) : Prop := y = x + 1

theorem ellipse_chord_theorem (p : EllipseParams) 
  (h1 : focalDistance p = 2 * Real.sqrt 6)
  (h2 : sumDistancesToFoci p = 6) :
  (∀ x y, ellipseEq p x y ↔ x^2 / 9 + y^2 / 3 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    ellipseEq p A.1 A.2 ∧ 
    ellipseEq p B.1 B.2 ∧
    lineEq A.1 A.2 ∧
    lineEq B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 66 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l193_19318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_logarithm_expression_l193_19311

open Real

theorem min_max_logarithm_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (M : ℝ), M = max (log (x / y + z)) (max (log (y * z + 1 / x)) (log (y / (x * z) + y))) ∧
  M ≥ log 2 ∧
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 →
    max (log (x' / y' + z')) (max (log (y' * z' + 1 / x')) (log (y' / (x' * z') + y'))) ≥ log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_logarithm_expression_l193_19311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_can_intersect_both_skew_lines_infinite_skew_lines_exist_l193_19393

-- Define the basic structures
structure Plane where
  dummy : Unit

structure Line where
  dummy : Unit

-- Define the concept of skew lines
def are_skew (l1 l2 : Line) : Prop :=
  sorry

-- Define the concept of a line lying on a plane
def lies_on (l : Line) (p : Plane) : Prop :=
  sorry

-- Define the concept of intersection between a line and a plane
def intersects (l : Line) (p : Plane) : Prop :=
  sorry

-- Define the intersection line of two planes
def intersection_line (p1 p2 : Plane) : Line :=
  sorry

-- Statement 1
theorem intersection_line_can_intersect_both_skew_lines 
  (a b : Line) (α β : Plane) (h1 : are_skew a b) (h2 : lies_on a α) (h3 : lies_on b β) :
  ∃ (c : Line), c = intersection_line α β ∧ intersects c α ∧ intersects c β := by
  sorry

-- Statement 2
theorem infinite_skew_lines_exist :
  ∃ (S : Set Line), Set.Infinite S ∧ ∀ (l1 l2 : Line), l1 ∈ S → l2 ∈ S → l1 ≠ l2 → are_skew l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_can_intersect_both_skew_lines_infinite_skew_lines_exist_l193_19393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l193_19358

-- Define the statements p and q
def p : Prop := ∃ x : ℝ, Real.sin x = Real.pi / 2
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem problem_statement :
  (¬p) ∧ 
  q ∧ 
  ¬(p ∧ q) ∧ 
  ¬(p ∧ ¬q) ∧ 
  (¬p ∧ q) ∧ 
  (¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l193_19358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l193_19372

theorem game_probability (n : ℕ) (p_alex p_jamie p_casey : ℚ) : 
  n = 8 →
  p_alex = 1/3 →
  p_jamie = 1/2 →
  p_casey = 1/6 →
  p_alex + p_jamie + p_casey = 1 →
  (Nat.choose n 4 * Nat.choose 4 3 : ℚ) * p_alex^4 * p_jamie^3 * p_casey = 35/486 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l193_19372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l193_19348

-- Define the equation
def equation (x k : ℝ) : Prop :=
  ((1/2)^(abs x) - 1/2)^2 - abs ((1/2)^(abs x) - 1/2) - k = 0

-- Theorem statement
theorem equation_solutions (k : ℝ) :
  (∀ x, ¬ equation x k) ∨ 
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ equation x₁ k ∧ equation x₂ k) ∨ 
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    equation x₁ k ∧ equation x₂ k ∧ equation x₃ k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l193_19348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_interval_l193_19331

/-- Given 5 children born at equal intervals, with the youngest being 5 years old
    and the sum of all ages being 55 years, the interval between births is 3.5 years. -/
theorem birth_interval (num_children : ℕ) (youngest_age : ℝ) (total_age : ℝ) :
  num_children = 5 →
  youngest_age = 5 →
  total_age = 55 →
  (total_age - num_children * youngest_age) / (num_children * (num_children - 1) / 2) = 3.5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_interval_l193_19331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l193_19394

/-- Given a hyperbola with center (3, -1), one focus at (3, 7), and one vertex at (3, 2),
    prove that h + k + a + b = 5 + √55, where (y - k)²/a² - (x - h)²/b² = 1 is its equation. -/
theorem hyperbola_sum (h k a b : ℝ) : 
  (h, k) = (3, -1) → -- center
  (h, k + 8) = (3, 7) → -- focus
  (h, k + a) = (3, 2) → -- vertex
  b^2 = 64 - a^2 → -- from c² = a² + b²
  h + k + a + b = 5 + Real.sqrt 55 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_l193_19394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_times_one_minus_sin_max_value_l193_19389

theorem cos_half_times_one_minus_sin_max_value (θ : ℝ) (h : 0 < θ ∧ θ < π) :
  Real.cos (θ / 2) * (1 - Real.sin θ) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_times_one_minus_sin_max_value_l193_19389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_fish_count_l193_19317

def fish_population (initial : ℕ) (days : ℕ) : ℕ :=
  2^days * initial

def remove_fraction (n : ℕ) (d : ℕ) (total : ℕ) : ℕ :=
  total - (total / d)

theorem final_fish_count : 
  let initial := 6
  let day3 := remove_fraction 1 3 (fish_population initial 2)
  let day5 := remove_fraction 1 4 (fish_population day3 2)
  let day7 := fish_population day5 2 + 15
  day7 = 207 := by
    -- Proof steps would go here
    sorry

#eval 
  let initial := 6
  let day3 := remove_fraction 1 3 (fish_population initial 2)
  let day5 := remove_fraction 1 4 (fish_population day3 2)
  let day7 := fish_population day5 2 + 15
  day7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_fish_count_l193_19317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l193_19330

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + 5 * Real.pi / 12) + Real.cos (ω * x - Real.pi / 12)

theorem omega_values (ω : ℝ) :
  ω > 0 ∧
  f ω Real.pi = 3 ∧
  (∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi / 12 ∧
    f ω x₁ = 3 * Real.sqrt 3 / 2 ∧
    f ω x₂ = 3 * Real.sqrt 3 / 2 ∧
    ∀ (x : ℝ), 0 < x ∧ x < Real.pi / 12 ∧ f ω x = 3 * Real.sqrt 3 / 2 → x = x₁ ∨ x = x₂) →
  ω = 24 + 1 / 12 ∨ ω = 26 + 1 / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l193_19330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l193_19370

/-- Represents the loss percentage when selling a watch -/
noncomputable def loss_percentage (cost_price selling_price : ℝ) : ℝ :=
  (cost_price - selling_price) / cost_price * 100

/-- Represents the selling price of the watch -/
noncomputable def selling_price (cost_price loss_percentage : ℝ) : ℝ :=
  cost_price * (1 - loss_percentage / 100)

theorem watch_loss_percentage :
  let cost_price : ℝ := 2000
  let actual_selling_price := selling_price cost_price (loss_percentage cost_price actual_selling_price)
  let new_selling_price := actual_selling_price + 520
  new_selling_price = cost_price * 1.06 →
  loss_percentage cost_price actual_selling_price = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_loss_percentage_l193_19370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_magnitude_l193_19355

/-- Prove that the absolute value of (1+i)^19 - (1-i)^19 is equal to 512√2, where i is the imaginary unit. -/
theorem complex_power_difference_magnitude : 
  Complex.abs ((1 + Complex.I)^19 - (1 - Complex.I)^19) = 512 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_difference_magnitude_l193_19355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l193_19305

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y^2 = 8*x

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector type -/
structure Vec where
  x : ℝ
  y : ℝ

def focus : Point := ⟨2, 0⟩

def directrix : ℝ := -2

def on_directrix (p : Point) : Prop := p.x = directrix

def on_parabola (p : Point) (c : Parabola) : Prop := p.y^2 = 8*p.x

def vector_mul (v : Vec) (k : ℝ) : Vec := ⟨k * v.x, k * v.y⟩

def vector_eq (v1 v2 : Vec) : Prop := v1.x = v2.x ∧ v1.y = v2.y

noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_segment_length 
  (c : Parabola) (p m n : Point) (pf mf : Vec) :
  on_directrix p →
  on_parabola m c →
  on_parabola n c →
  vector_eq pf (vector_mul mf 3) →
  distance m n = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l193_19305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_combined_equals_nacl_formed_l193_19371

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the reaction between HCl and NaHCO3 -/
structure Reaction where
  HCl : Moles
  NaHCO3 : Moles
  NaCl : Moles
  molar_ratio : HCl = NaHCO3 ∧ NaHCO3 = NaCl

theorem hcl_combined_equals_nacl_formed (r : Reaction) (h : r.NaHCO3 = (3 : ℝ)) :
  r.HCl = r.NaCl := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hcl_combined_equals_nacl_formed_l193_19371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_theorem_l193_19319

/-- A participant in the math competition -/
structure Participant where
  problems_in_part1 : ℕ

/-- A math competition with two parts -/
structure MathCompetition where
  total_problems : ℕ
  part1_problems : ℕ
  part2_problems : ℕ
  participants : Finset Participant
  problems_per_participant : ℕ
  pairs_per_problem : ℕ

/-- The theorem statement -/
theorem math_competition_theorem (c : MathCompetition)
  (h1 : c.total_problems = 28)
  (h2 : c.total_problems = c.part1_problems + c.part2_problems)
  (h3 : c.problems_per_participant = 7)
  (h4 : c.pairs_per_problem = 2)
  : ∃ p ∈ c.participants, (p.problems_in_part1 = 0 ∨ p.problems_in_part1 ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_theorem_l193_19319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_bulb_std_dev_approx_l193_19307

noncomputable def light_bulb_lifespans : List ℝ := [1050, 1100, 1120, 1280, 1250, 1040, 1030, 1110, 1240, 1300]

noncomputable def sample_mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def sample_variance (xs : List ℝ) : ℝ :=
  let mean := sample_mean xs
  (xs.map (fun x => (x - mean) ^ 2)).sum / (xs.length - 1)

noncomputable def sample_std_dev (xs : List ℝ) : ℝ :=
  Real.sqrt (sample_variance xs)

theorem light_bulb_std_dev_approx :
  ∃ ε > 0, ε < 0.1 ∧ |sample_std_dev light_bulb_lifespans - 104.4| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_bulb_std_dev_approx_l193_19307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_last_standing_l193_19357

/-- Represents a person in the circle --/
inductive Person : Type
| Laura : Person
| Mike : Person
| Nicky : Person
| Olivia : Person

/-- Checks if a number should cause elimination --/
def isEliminationNumber (n : Nat) : Bool :=
  n % 6 = 0 || n % 7 = 0 || n.repr.any (fun c => c = '6' || c = '7')

/-- Simulates the elimination process and returns the last person standing --/
def lastPersonStanding (people : List Person) : Person :=
  match people with
  | [] => Person.Laura  -- Default to Laura if the list is empty (shouldn't happen)
  | [p] => p  -- If there's only one person left, return that person
  | _ => sorry  -- The actual implementation would go here

theorem nicky_last_standing :
  lastPersonStanding [Person.Laura, Person.Mike, Person.Nicky, Person.Olivia] = Person.Nicky := by
  sorry  -- The proof would go here

#eval isEliminationNumber 6  -- Should return true
#eval isEliminationNumber 7  -- Should return true
#eval isEliminationNumber 16  -- Should return true
#eval isEliminationNumber 5  -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_last_standing_l193_19357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l193_19325

noncomputable section

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x > 0 then x^2 + 1/x else -(-x^2 - 1/(-x))

theorem f_neg_one_eq_neg_two
  (h_odd : IsOdd f)
  (h_def : ∀ x > 0, f x = x^2 + 1/x) :
  f (-1) = -2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_eq_neg_two_l193_19325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_overlap_ratio_l193_19383

/-- Given a rectangle ABCD and a square EFGH that overlap, if the rectangle shares 25% of its area
    with the square, and the square shares 10% of its area with the rectangle, then the ratio of
    the rectangle's length to its width is 6. -/
theorem rectangle_square_overlap_ratio (A B C D E F G H : EuclideanSpace ℝ (Fin 2)) :
  ∀ (rect_area sq_area overlap_area : ℝ),
    overlap_area = 0.25 * rect_area →
    overlap_area = 0.1 * sq_area →
    ∃ (length width : ℝ),
      length * width = rect_area ∧
      length / width = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_square_overlap_ratio_l193_19383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_18_l193_19340

def rectangle_factors : Set (Nat × Nat) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18}

theorem rectangle_area_18 :
  rectangle_factors = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_18_l193_19340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_represents_six_l193_19337

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : Nat // n ≥ 10 ∧ n < 100 }

theorem square_represents_six 
  (triangle : Digit) 
  (circle : Digit) 
  (square : Digit) 
  (two_digit : TwoDigitNumber) :
  triangle.val = 1 →
  circle.val = 9 →
  two_digit.val + triangle.val + circle.val + square.val + square.val = 111 →
  square.val = 6 := by
  intro h_triangle h_circle h_sum
  sorry

#check square_represents_six

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_represents_six_l193_19337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balloons_special_sale_l193_19344

/-- Represents the sale price of a pair of balloons -/
def sale_price (regular_price : ℚ) : ℚ := regular_price + (2/3 * regular_price)

/-- Calculates the maximum number of balloons that can be bought given the total money and regular price -/
def max_balloons (total_money : ℚ) (regular_price : ℚ) : ℕ :=
  (2 * (total_money / sale_price regular_price).floor).toNat

theorem max_balloons_special_sale (regular_price : ℚ) :
  max_balloons (30 * regular_price) regular_price = 36 := by
  sorry

#eval max_balloons 90 3  -- Should output 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balloons_special_sale_l193_19344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_weight_sum_l193_19363

theorem pumpkin_weight_sum 
  (pumpkin1 : Real)
  (pumpkin2 : Real)
  (h1 : pumpkin1 = 12.6)
  (h2 : pumpkin2 = 23.4) :
  pumpkin1 + pumpkin2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pumpkin_weight_sum_l193_19363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangent_l193_19346

theorem triangle_angle_tangent (A : ℝ) 
  (eq : (Real.sqrt 3 * Real.cos A + Real.sin A) / (Real.sqrt 3 * Real.sin A - Real.cos A) = Real.tan (-7/12 * Real.pi)) : 
  Real.tan A = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_tangent_l193_19346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l193_19385

theorem triangle_properties (a c : ℝ) (h_positive_a : a > 0) (h_positive_c : c > 0) :
  let A : ℝ := 60 * Real.pi / 180
  let h_c : c = 3/7 * a := by sorry
  let sin_C : ℝ := c / a * Real.sin A
  let area : ℝ := 1/2 * a * c * Real.sin (Real.pi - A - Real.arcsin sin_C)
  sin_C = 3 * Real.sqrt 3 / 14 ∧ (a = 7 → area = 6 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l193_19385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l193_19322

-- Define the points as elements of EuclideanPlane
variable (A B C D P T : EuclideanPlane)

-- Define the convexity of the quadrilateral
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the angle measure function
noncomputable def angle_measure (A B C : EuclideanPlane) : ℝ := sorry

-- Define the property of P being on both AD and BC
def P_on_AD_and_BC (A B C D P : EuclideanPlane) : Prop := sorry

-- Define the property of T being on BD and PT parallel to AB
def T_on_BD_and_PT_parallel_AB (A B C D P T : EuclideanPlane) : Prop := sorry

-- State the theorem
theorem angle_equality 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_equal_angles : angle_measure A B C = angle_measure B C D)
  (h_P_intersection : P_on_AD_and_BC A B C D P)
  (h_T_parallel : T_on_BD_and_PT_parallel_AB A B C D P T) :
  angle_measure A C B = angle_measure P C T :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l193_19322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_between_five_and_six_count_l193_19391

theorem sqrt_between_five_and_six_count : 
  (Finset.filter (fun x => 25 < x ∧ x < 36) (Finset.range 36)).card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_between_five_and_six_count_l193_19391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_abc_triplet_l193_19333

/-- The radical of a natural number -/
def rad (n : Nat) : Nat := sorry

/-- Theorem stating the existence of a triplet satisfying the given conditions -/
theorem exists_abc_triplet : ∃ (A B C : Nat), 
  (Nat.Coprime A B ∧ Nat.Coprime B C ∧ Nat.Coprime A C) ∧ 
  A + B = C ∧ 
  C > 1000 * rad (A * B * C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_abc_triplet_l193_19333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bugs_eaten_l193_19301

/-- The number of bugs eaten by various animals in a garden. -/
def garden_bugs (gecko lizard frog tortoise toad crocodile turtle : ℝ) : ℝ :=
  gecko + lizard + frog + tortoise + toad + crocodile + turtle

/-- Theorem stating the total number of bugs eaten in the garden. -/
theorem total_bugs_eaten :
  ∀ (gecko : ℝ),
  gecko = 18 →
  ∀ (lizard : ℝ),
  lizard = gecko / 2 →
  ∀ (frog : ℝ),
  frog = 3 * lizard →
  ∀ (tortoise : ℝ),
  tortoise = gecko * 0.75 →
  ∀ (toad : ℝ),
  toad = frog * 1.5 →
  ∀ (crocodile : ℝ),
  crocodile = gecko + toad →
  ∀ (turtle : ℝ),
  turtle = crocodile / 3 →
  garden_bugs gecko lizard frog tortoise toad crocodile turtle = 186 :=
by
  intros gecko h_gecko lizard h_lizard frog h_frog tortoise h_tortoise toad h_toad crocodile h_crocodile turtle h_turtle
  simp [garden_bugs]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bugs_eaten_l193_19301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l193_19380

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x + (2^x - 1)/(2^x + 1) + 5

-- State the theorem
theorem f_symmetry (h : f (-7) = -7) : f 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l193_19380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_X_l193_19384

theorem max_elements_in_X : ∃ (a b : Fin 20 → ℕ), 
  (∀ i, a i ∈ Finset.range 5) ∧ 
  (∀ i, b i ∈ Finset.range 10) ∧ 
  (Finset.card (Finset.filter (λ p : Fin 20 × Fin 20 => p.1 < p.2 ∧ (a p.1 - a p.2) * (b p.1 - b p.2) < 0) (Finset.univ.product Finset.univ)) = 160) ∧
  (∀ a' b' : Fin 20 → ℕ, 
    (∀ i, a' i ∈ Finset.range 5) → 
    (∀ i, b' i ∈ Finset.range 10) → 
    Finset.card (Finset.filter (λ p : Fin 20 × Fin 20 => p.1 < p.2 ∧ (a' p.1 - a' p.2) * (b' p.1 - b' p.2) < 0) (Finset.univ.product Finset.univ)) ≤ 160) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_X_l193_19384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_revenue_l193_19351

theorem tax_consumption_revenue (original_tax : ℝ) (original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.84 * original_tax
  let new_revenue := 0.966 * (original_tax * original_consumption)
  ∃ new_consumption : ℝ, 
    new_consumption = 1.15 * original_consumption ∧
    new_tax * new_consumption = new_revenue :=
by
  intro new_tax new_revenue
  use 1.15 * original_consumption
  constructor
  · rfl
  · calc
      new_tax * (1.15 * original_consumption)
        = 0.84 * original_tax * (1.15 * original_consumption) := by rfl
      _ = 0.84 * 1.15 * (original_tax * original_consumption) := by ring
      _ = 0.966 * (original_tax * original_consumption) := by ring
      _ = new_revenue := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_consumption_revenue_l193_19351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l193_19309

noncomputable def total_cost : ℝ := 340
noncomputable def pants_price : ℝ := 120
noncomputable def shirt_price : ℝ := (3/4) * pants_price

theorem price_difference : 
  total_cost - shirt_price - pants_price - pants_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_l193_19309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_or_six_count_divisible_by_four_or_six_l193_19362

theorem divisible_by_four_or_six (n : ℕ) : 
  (Finset.filter (λ x : ℕ => x % 4 = 0 ∨ x % 6 = 0) (Finset.range n)).card = 
  (n / 4) + (n / 6) - (n / 12) :=
sorry

theorem count_divisible_by_four_or_six : 
  (Finset.filter (λ x : ℕ => x % 4 = 0 ∨ x % 6 = 0) (Finset.range 61)).card = 20 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_four_or_six_count_divisible_by_four_or_six_l193_19362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wrench_force_calculation_l193_19341

/-- Represents the force required to loosen a bolt -/
structure Force where
  value : ℝ

/-- Represents the length of a wrench handle -/
structure Length where
  value : ℝ

instance : HMul Force Length ℝ where
  hMul f l := f.value * l.value

instance : OfNat Force n where
  ofNat := ⟨n⟩

instance : OfNat Length n where
  ofNat := ⟨n⟩

/-- The relation between force and length for loosening a bolt -/
def force_length_relation (f : Force) (l : Length) : Prop :=
  ∃ k : ℝ, f * l = k

theorem wrench_force_calculation 
  (force_inv_length : ∀ (f : Force) (l : Length), force_length_relation f l)
  (initial_force : Force) (initial_length : Length)
  (new_length : Length)
  (h_initial : initial_force = 300 ∧ initial_length = 12)
  (h_new_length : new_length = 18) :
  ∃ (new_force : Force), new_force = 200 ∧ force_length_relation new_force new_length :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wrench_force_calculation_l193_19341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_run_home_time_l193_19377

/-- Represents Justin's running speed in blocks per minute -/
noncomputable def justin_speed : ℚ := 2 / (3/2)

/-- The distance from Justin to his home in blocks -/
def distance_to_home : ℕ := 8

/-- The time it takes Justin to run home in minutes -/
noncomputable def time_to_run_home : ℚ := distance_to_home / justin_speed

theorem justin_run_home_time :
  time_to_run_home = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_justin_run_home_time_l193_19377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l193_19310

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_5 (a : ℝ) :
  let q := (2 : ℝ)
  let S := geometric_sum a q
  S 2 = 3 → S 3 = 7 → S 5 = 31 := by
  intro h1 h2
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_5_l193_19310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inv_z_forms_ellipse_l193_19390

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z| = 3
def z_on_circle (z : ℂ) : Prop := Complex.abs z = 3

-- Define the function f(z) = z + 1/z
noncomputable def f (z : ℂ) : ℂ := z + 1 / z

-- Define what it means for a set of complex numbers to form an ellipse
def is_ellipse (S : Set ℂ) : Prop := 
  ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b), 
    S = {w : ℂ | (Complex.re w)^2 / a^2 + (Complex.im w)^2 / b^2 = 1}

-- State the theorem
theorem z_plus_inv_z_forms_ellipse :
  ∀ z : ℂ, z_on_circle z → is_ellipse {w : ℂ | ∃ z, z_on_circle z ∧ w = f z} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_plus_inv_z_forms_ellipse_l193_19390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_l193_19374

/-- The sticker price of a laptop. -/
def stickerPrice : ℝ := 950

/-- The final price at Shop A after discount and rebate. -/
def shopAPrice : ℝ := 0.80 * stickerPrice - 120

/-- The final price at Shop B after discount. -/
def shopBPrice : ℝ := 0.70 * stickerPrice

/-- The savings when purchasing from Shop A compared to Shop B. -/
def savings : ℝ := 25

theorem laptop_price :
  shopBPrice = shopAPrice + savings → stickerPrice = 950 := by
  intro h
  -- Expand the definitions
  simp [shopBPrice, shopAPrice, stickerPrice] at h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_l193_19374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_2a_plus_b_cos_angle_b_minus_3a_and_a_l193_19367

noncomputable section

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (1, 1)

-- Define the function to calculate the unit vector
noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let magnitude := Real.sqrt (v.1^2 + v.2^2)
  (v.1 / magnitude, v.2 / magnitude)

-- Define the function to calculate the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Define the function to calculate the cosine of the angle between two vectors
noncomputable def cos_angle (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

-- Theorem I
theorem unit_vector_2a_plus_b :
  unit_vector (2 * a.1 + b.1, 2 * a.2 + b.2) = (3 / Real.sqrt 10, 1 / Real.sqrt 10) :=
by sorry

-- Theorem II
theorem cos_angle_b_minus_3a_and_a :
  cos_angle (b.1 - 3 * a.1, b.2 - 3 * a.2) a = - 2 * Real.sqrt 5 / 5 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_2a_plus_b_cos_angle_b_minus_3a_and_a_l193_19367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_internal_angle_is_36_l193_19334

/-- The number of sides in a regular pentagon -/
def n : ℕ := 5

/-- The internal angle of a regular polygon with n sides -/
noncomputable def internal_angle (n : ℕ) : ℝ := ((n - 2) * 180) / n

/-- The external angle of a regular polygon -/
noncomputable def external_angle (n : ℕ) : ℝ := 180 - internal_angle n

/-- The internal angle at each vertex of the star-like figure formed from a regular pentagon -/
noncomputable def star_internal_angle (n : ℕ) : ℝ := 180 - 2 * external_angle n

/-- Theorem: The internal angle at each vertex of the star-like figure formed from a regular pentagon is 36° -/
theorem star_internal_angle_is_36 : star_internal_angle n = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_internal_angle_is_36_l193_19334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_right_triangle_l193_19313

theorem unique_right_triangle :
  ∃! (a b : ℕ), 
    a^2 + b^2 = (b + 3)^2 ∧
    b < 50 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_right_triangle_l193_19313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_fraction_l193_19314

theorem perfect_square_fraction (n m : ℕ) (h_parity : n % 2 ≠ m % 2) (h_gt : n > m) :
  ∀ x : ℤ, (∃ k : ℤ, (x^(2^n) - 1) / (x^(2^m) - 1) = k^2) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_fraction_l193_19314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_is_300_l193_19303

/-- Represents a test with a passing threshold, student's score, and failure margin. -/
structure Test where
  passingThreshold : Rat
  studentScore : Nat
  failureMargin : Nat

/-- Calculates the maximum marks for a test given the test parameters. -/
def calculateMaxMarks (test : Test) : Rat :=
  (test.studentScore + test.failureMargin : Rat) / test.passingThreshold

/-- Theorem stating that for the given test parameters, the maximum marks is 300. -/
theorem max_marks_is_300 (test : Test)
  (h1 : test.passingThreshold = 3/5)
  (h2 : test.studentScore = 80)
  (h3 : test.failureMargin = 100) :
  calculateMaxMarks test = 300 := by
  sorry

#eval calculateMaxMarks { passingThreshold := 3/5, studentScore := 80, failureMargin := 100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_is_300_l193_19303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_bananas_l193_19398

/-- Represents the distribution of bananas among three monkeys -/
structure BananaDistribution where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the distribution satisfies the problem conditions -/
def is_valid_distribution (d : BananaDistribution) : Prop :=
  -- Total is the sum of individual parts
  d.total = d.first + d.second + d.third
  -- The ratio of final bananas is 3:2:1
  ∧ 2 * d.first = 3 * d.second
  ∧ 3 * d.third = d.second
  -- First monkey's distribution
  ∧ ∃ (a : ℕ), 3 * a = 4 * d.first ∧ a ≤ d.total
  -- Second monkey's distribution
  ∧ ∃ (b : ℕ), b = 4 * d.second ∧ b ≤ d.total - a
  -- Third monkey's distribution
  ∧ ∃ (c : ℕ), 11 * c = 12 * d.third ∧ c = d.total - a - b

/-- The theorem stating the least possible total number of bananas -/
theorem least_possible_bananas :
  ∀ d : BananaDistribution, is_valid_distribution d → d.total ≥ 51 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_bananas_l193_19398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_cost_example_l193_19387

/-- The total cost of carpeting a rectangular room -/
noncomputable def carpet_cost (room_length room_breadth carpet_width carpet_cost_per_meter : ℝ) : ℝ :=
  let room_area := room_length * room_breadth
  let carpet_width_m := carpet_width / 100
  let num_strips := room_breadth / carpet_width_m
  let total_carpet_length := num_strips * room_length
  total_carpet_length * carpet_cost_per_meter

/-- Theorem stating that the carpet cost for the given room and carpet specifications is 810 -/
theorem carpet_cost_example : 
  carpet_cost 18 7.5 75 4.5 = 810 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_cost_example_l193_19387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l193_19366

theorem cube_root_equation (x : ℝ) : (Real.rpow (5 + Real.sqrt x) (1/3 : ℝ) = 4) → x = 3481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l193_19366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l193_19350

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

-- Define the theorem
theorem triangle_area (A B C a b c : ℝ) : 
  -- Given conditions
  (f (A/2 - Real.pi/6) = Real.sqrt 3) →
  (a = 7) →
  (Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14) →
  -- Conclusion
  (1/2 * b * c * Real.sin A = 10 * Real.sqrt 3) :=
by
  -- Placeholder for the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l193_19350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grandfathers_age_l193_19329

theorem grandfathers_age (grandfather_age : ℕ) (xiaoming_age : ℕ) : 
  grandfather_age > 7 * xiaoming_age →
  (∃ (y z w : ℕ), y > 0 ∧ z > y ∧ w > z ∧
  grandfather_age - y = 6 * (xiaoming_age + y) ∧
  grandfather_age - z = 5 * (xiaoming_age + z) ∧
  grandfather_age - w = 4 * (xiaoming_age + w)) →
  grandfather_age = 69 := by
  sorry

#check grandfathers_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grandfathers_age_l193_19329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_five_two_l193_19328

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := a^2 + a^2 / b

-- Theorem statement
theorem diamond_five_two : diamond 5 2 = 37.5 := by
  -- Unfold the definition of diamond
  unfold diamond
  -- Simplify the expression
  simp [pow_two]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_five_two_l193_19328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l193_19316

/-- Represents the properties of a train --/
structure Train where
  initial_speed : ℝ
  travel_time : ℝ
  stop_time : ℝ
  final_speed : ℝ

/-- Calculates the time for two trains to meet after their stops --/
noncomputable def time_to_meet (train_a train_b : Train) (initial_distance : ℝ) : ℝ :=
  let distance_a := train_a.initial_speed * train_a.travel_time
  let distance_b := train_b.initial_speed * train_b.travel_time
  let remaining_distance := initial_distance - (distance_a + distance_b)
  let relative_speed := train_a.final_speed + train_b.final_speed
  remaining_distance / relative_speed

theorem trains_meet_time (train_a train_b : Train) (initial_distance : ℝ) :
  train_a.initial_speed = 60 →
  train_b.initial_speed = 50 →
  train_a.travel_time = 2 →
  train_b.travel_time = 2 →
  train_a.stop_time = 0.5 →
  train_b.stop_time = 0.25 →
  train_a.final_speed = 50 →
  train_b.final_speed = 40 →
  initial_distance = 270 →
  time_to_meet train_a train_b initial_distance = 50 / 90 :=
by
  sorry

#check trains_meet_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l193_19316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_a_in_range_l193_19386

def line_equation (k : ℝ) (x y : ℝ) : Prop := k * x - y - k + 2 = 0

def circle_equation (a : ℝ) (x y : ℝ) : Prop := x^2 + 2*a*x + y^2 - a + 2 = 0

theorem no_common_points_iff_a_in_range :
  ∀ a : ℝ, (∃ k : ℝ, ∀ x y : ℝ, ¬(line_equation k x y ∧ circle_equation a x y)) ↔ 
  (a > -7 ∧ a < -2) ∨ (a > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_common_points_iff_a_in_range_l193_19386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l193_19332

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  equation : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1
  a_gt_b : a > b

/-- The angle between the asymptotes of a hyperbola in radians -/
noncomputable def asymptote_angle (h : Hyperbola a b) : ℝ := 2 * Real.arctan (b / a)

/-- Theorem: For a hyperbola with asymptote angle of π/3 (60°), a/b = √3 -/
theorem hyperbola_asymptote_ratio
  (a b : ℝ)
  (h : Hyperbola a b)
  (angle_60 : asymptote_angle h = π / 3) :
  a / b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l193_19332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_transfer_l193_19302

noncomputable def tank_radius_1 : ℝ := 4
noncomputable def tank_height_1 : ℝ := 10
noncomputable def tank_radius_2 : ℝ := 6
noncomputable def tank_height_2 : ℝ := 8

noncomputable def initial_volume : ℝ := Real.pi * tank_radius_1^2 * tank_height_1

theorem water_depth_after_transfer : 
  ∃ (h : ℝ), 
    h > 0 ∧ 
    h ≤ tank_height_1 ∧ 
    h ≤ tank_height_2 ∧
    Real.pi * tank_radius_1^2 * h + Real.pi * tank_radius_2^2 * h = initial_volume ∧
    h = 40 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_transfer_l193_19302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equilateral_triangle_area_l193_19320

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- Point A is the left vertex of the hyperbola -/
def A : ℝ × ℝ := (-1, 0)

/-- Points B and C are on the right branch of the hyperbola -/
def on_right_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 > 0

/-- Triangle ABC is equilateral -/
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2

/-- The area of triangle ABC -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

/-- The main theorem -/
theorem hyperbola_equilateral_triangle_area :
  ∀ B C : ℝ × ℝ,
  on_right_branch B ∧ on_right_branch C ∧
  is_equilateral A B C →
  triangle_area A B C = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equilateral_triangle_area_l193_19320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l193_19345

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 18 ≥ 0}
def B : Set ℝ := {x | (x + 5) / (x - 14) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}

-- Define the complement of B
def complementB : Set ℝ := {x | x ∉ B}

-- Statement for part 1
theorem part1 : complementB ∩ A = Set.Iic (-5) ∪ Set.Ici 14 := by sorry

-- Statement for part 2
theorem part2 : ∀ a : ℝ, (B ∩ C a = C a) ↔ a ≥ -5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l193_19345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l193_19353

/-- Represents a class of students -/
structure StudentClass where
  total : Nat
  range_start : Nat
  range_end : Nat

/-- Represents a systematic sample of students -/
structure Sample where
  size : Nat
  numbers : List Nat

/-- Checks if a sample is valid for a given class using systematic sampling -/
def is_valid_systematic_sample (c : StudentClass) (s : Sample) : Prop :=
  s.size = 5 ∧
  c.total = 50 ∧
  c.range_start = 0 ∧
  c.range_end = 50 ∧
  s.numbers = [4, 13, 22, 31, 40]

/-- Theorem stating that the given sample is the correct systematic sample for the class -/
theorem correct_systematic_sample (c : StudentClass) (s : Sample) :
  is_valid_systematic_sample c s → 
  s.numbers = [4, 13, 22, 31, 40] := by
  intro h
  exact h.right.right.right.right


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_systematic_sample_l193_19353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_drummers_l193_19395

/-- Represents a 50x50 grid of drummers --/
def DrummerGrid := Fin 50 → Fin 50 → Bool

/-- Checks if a position is valid in the 50x50 grid --/
def is_valid_pos (i j : Nat) : Prop := i < 50 ∧ j < 50

/-- Checks if two positions are adjacent (including diagonally) --/
def are_adjacent (i1 j1 i2 j2 : Nat) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2)) ∨
  (i1 + 1 = i2 ∧ j1 + 1 = j2) ∨
  (i1 + 1 = i2 ∧ j1 = j2 + 1) ∨
  (i1 = i2 + 1 ∧ j1 + 1 = j2) ∨
  (i1 = i2 + 1 ∧ j1 = j2 + 1)

/-- A valid drummer configuration ensures no two blue drummers are adjacent --/
def is_valid_configuration (grid : DrummerGrid) : Prop :=
  ∀ i1 j1 i2 j2, is_valid_pos i1 j1 → is_valid_pos i2 j2 →
    are_adjacent i1 j1 i2 j2 →
    ¬(grid ⟨i1, by sorry⟩ ⟨j1, by sorry⟩ ∧ grid ⟨i2, by sorry⟩ ⟨j2, by sorry⟩)

/-- Counts the number of blue drummers in the grid --/
def count_blue_drummers (grid : DrummerGrid) : Nat :=
  (Finset.sum (Finset.univ : Finset (Fin 50)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 50)) (λ j =>
      if grid i j then 1 else 0)))

/-- The main theorem stating the maximum number of blue drummers --/
theorem max_blue_drummers :
  ∃ (grid : DrummerGrid), is_valid_configuration grid ∧
    ∀ (other_grid : DrummerGrid), is_valid_configuration other_grid →
      count_blue_drummers other_grid ≤ count_blue_drummers grid ∧
      count_blue_drummers grid = 625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_drummers_l193_19395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_sum_reciprocals_l193_19312

theorem power_equality_sum_reciprocals (a b : ℝ) (h1 : (2 : ℝ)^a = 100) (h2 : (5 : ℝ)^b = 100) :
  1/a + 1/b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_sum_reciprocals_l193_19312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l193_19388

-- Define the game board (round table)
structure RoundTable where
  radius : ℝ
  radius_pos : radius > 0

-- Define a coin
structure Coin where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define the game state
structure GameState where
  table : RoundTable
  coins : List Coin
  current_player : Bool  -- True for first player, False for second player

-- Define a valid move
def is_valid_move (state : GameState) (new_coin : Coin) : Prop :=
  -- The coin is within the table
  (new_coin.center.1^2 + new_coin.center.2^2).sqrt + new_coin.radius ≤ state.table.radius
  -- The coin doesn't overlap with any existing coin
  ∧ ∀ c ∈ state.coins, 
    (c.center.1 - new_coin.center.1)^2 + (c.center.2 - new_coin.center.2)^2 
    > (c.radius + new_coin.radius)^2

-- Define the winning strategy for the first player
def first_player_strategy (state : GameState) : Option Coin := sorry

-- Helper function to update the game state
def update_state (state : GameState) (new_coin : Coin) : GameState :=
  { state with 
    coins := new_coin :: state.coins,
    current_player := ¬state.current_player }

-- Theorem stating that the first player always wins
theorem first_player_always_wins :
  ∀ (initial_state : GameState),
  initial_state.current_player = true →
  ∃ (strategy : GameState → Option Coin),
  (∀ (state : GameState), 
    (strategy state).map (is_valid_move state) = some true) ∧
  (∀ (state : GameState) (opponent_move : Coin),
   is_valid_move state opponent_move →
   ∃ (next_move : Coin), 
     is_valid_move (update_state state opponent_move) next_move) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l193_19388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_innings_problem_l193_19321

/-- Represents the number of innings a cricket player has played --/
def innings : ℕ → ℕ := sorry

/-- Represents the total runs scored by a cricket player --/
def total_runs : ℕ → ℕ := sorry

/-- Calculates the average runs per innings --/
def average (i : ℕ) : ℚ := (total_runs i : ℚ) / (innings i : ℚ)

theorem cricket_innings_problem (i : ℕ) : 
  average i = 37 →
  average (i + 1) = 41 →
  total_runs (i + 1) = total_runs i + 81 →
  innings i = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_innings_problem_l193_19321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_a_range_l193_19338

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (q : ℝ) : ℕ → ℝ
| 0 => 0  -- Define a_0 as 0 for completeness
| n + 1 => q^n * 4  -- a_n = 2^(n+1) = 4 * 2^n = 4 * q^n, where q = 2

-- Define the conditions
axiom q_gt_one : 1 < (2 : ℝ)
axiom sum_first_third : arithmetic_sequence 2 1 + arithmetic_sequence 2 3 = 20
axiom second_term : arithmetic_sequence 2 2 = 8

-- Define b_n
noncomputable def b (n : ℕ) : ℝ := n / arithmetic_sequence 2 n

-- Define S_n (sum of first n terms of b_n)
noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i => b (i + 1))

-- Define the inequality condition
def inequality_condition (n : ℕ) (a : ℝ) : Prop := 
  S n + n / 2^(n+1) > (-1)^n * a

-- Theorem statements
theorem general_term_formula (n : ℕ) : 
  arithmetic_sequence 2 n = 2^(n+1) := sorry

theorem a_range (a : ℝ) 
  (h : ∀ n : ℕ, inequality_condition n a) : 
  -1/2 < a ∧ a < 3/4 := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_formula_a_range_l193_19338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l193_19306

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line with slope k and y-intercept m -/
structure Line where
  k : ℝ
  m : ℝ

/-- Check if a point (x, y) is on the ellipse -/
def on_ellipse (C : Ellipse) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem statement -/
theorem ellipse_intersection_range (C : Ellipse) (l : Line) :
  C.a^2 = C.b^2 + (2 * Real.sqrt 3)^2 →  -- Focal distance condition
  (Real.sqrt 3 / 2)^2 * C.a^2 = (2 * Real.sqrt 3)^2 →  -- Eccentricity condition
  (∃ A B : ℝ × ℝ, 
    on_ellipse C A.1 A.2 ∧ on_ellipse C B.1 B.2 ∧  -- A and B are on the ellipse
    A.2 = l.k * A.1 + l.m ∧ B.2 = l.k * B.1 + l.m ∧  -- A and B are on the line
    area_triangle A B (0, 0) = 4) →  -- Area of triangle AOB is 4
  l.m ∈ Set.Icc (-Real.sqrt 2) (-Real.sqrt 2) ∪ Set.Icc (Real.sqrt 2) (Real.sqrt 2) :=  -- Range of m
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l193_19306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_cost_l193_19347

/-- The cost of one bag of potatoes from the farmer -/
def x : ℝ := 250

/-- The number of bags each trader bought -/
def bags : ℕ := 60

/-- Andrey's total earnings -/
def andrey_earnings : ℝ := bags * (2 * x)

/-- Boris's earnings from first 15 bags -/
def boris_first_earnings : ℝ := 15 * (1.6 * x)

/-- Boris's earnings from remaining 45 bags -/
def boris_second_earnings : ℝ := 45 * (2.24 * x)

/-- Boris's total earnings -/
def boris_total_earnings : ℝ := boris_first_earnings + boris_second_earnings

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_cost : x = 250 :=
  by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_cost_l193_19347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_volume_cylinder_l193_19339

-- Define the volume of a cylinder
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Theorem statement
theorem triple_volume_cylinder (r₁ h₁ r₂ h₂ : ℝ) 
  (hr₁ : r₁ = 8) (hh₁ : h₁ = 7) (hr₂ : r₂ = 8) (hh₂ : h₂ = 21) :
  cylinderVolume r₂ h₂ = 3 * cylinderVolume r₁ h₁ := by
  sorry

#check triple_volume_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_volume_cylinder_l193_19339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_remains_triangle_minimal_area_remains_acute_l193_19359

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate for a triangle remaining a triangle after increasing any side by 1 -/
def remains_triangle (t : Triangle) : Prop :=
  (t.a + 1 + t.b > t.c) ∧ (t.a + t.b + 1 > t.c) ∧
  (t.b + 1 + t.c > t.a) ∧ (t.b + t.c + 1 > t.a) ∧
  (t.c + 1 + t.a > t.b) ∧ (t.c + t.a + 1 > t.b)

/-- Predicate for a triangle remaining acute after increasing any side by 1 -/
def remains_acute (t : Triangle) : Prop :=
  (t.a + 1)^2 < t.b^2 + t.c^2 ∧
  (t.b + 1)^2 < t.a^2 + t.c^2 ∧
  (t.c + 1)^2 < t.a^2 + t.b^2

/-- Area of a triangle given its side lengths -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

theorem minimal_area_remains_triangle :
  ∀ t : Triangle, remains_triangle t → triangle_area t ≥ Real.sqrt 3 / 4 := by sorry

theorem minimal_area_remains_acute :
  ∀ t : Triangle, remains_acute t → triangle_area t ≥ (Real.sqrt 3 / 4) * (3 + 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_area_remains_triangle_minimal_area_remains_acute_l193_19359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l193_19323

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 1 / (2 * x^2)

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < Real.sqrt 2 / 2 → f x1 > f x2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l193_19323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_equivalence_l193_19376

theorem exponential_inequality_equivalence (x y : ℝ) : x > y ↔ (2:ℝ)^x > (2:ℝ)^y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_equivalence_l193_19376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_function_simplification_l193_19336

theorem logical_function_simplification 
  (A B C : ℝ) : 
  A * (1 - B) + B * (1 - C) + B * C + A * B = A + B :=
by
  ring  -- This tactic should handle the algebraic simplification
  -- If 'ring' doesn't work, you can use 'sorry' to skip the proof
  -- sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_function_simplification_l193_19336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perimeter_constant_l193_19315

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def circle_eq (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

theorem ellipse_perimeter_constant 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_focus : ellipse a b 1 0)
  (h_point : ellipse a b 2 (2 * Real.sqrt 10 / 3))
  (M : ℝ × ℝ)
  (h_M : circle_eq b M.1 M.2 ∧ M.1 ≥ 0 ∧ M.2 ≥ 0)
  (P Q : ℝ × ℝ)
  (h_PQ : ellipse a b P.1 P.2 ∧ ellipse a b Q.1 Q.2)
  (h_tangent : ∃ (l : ℝ × ℝ → Prop), l M ∧ l P ∧ l Q ∧ 
    (∀ (x y : ℝ), circle_eq b x y → l (x, y) → (x, y) = M)) :
  ∃ (perimeter : ℝ), perimeter = 6 ∧
    ∀ (P' Q' : ℝ × ℝ), ellipse a b P'.1 P'.2 → ellipse a b Q'.1 Q'.2 →
      (∃ (l' : ℝ × ℝ → Prop), l' M ∧ l' P' ∧ l' Q' ∧ 
        (∀ (x y : ℝ), circle_eq b x y → l' (x, y) → (x, y) = M)) →
      dist P' (1, 0) + dist Q' (1, 0) + dist P' Q' = perimeter :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perimeter_constant_l193_19315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_bound_and_parallel_lines_l193_19349

/-- The function f(x) = x³ - ax² -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_line_slope_bound_and_parallel_lines (a : ℝ) :
  (∀ x, f' a x ≥ -a^2/3) ∧
  (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, f' a x = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_bound_and_parallel_lines_l193_19349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_defined_for_all_reals_domain_of_h_is_real_l193_19378

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := (x^4 - 16*x + 3) / (|x - 4| + |x + 2| + x - 3)

-- Theorem stating that h(x) is defined for all real numbers
theorem h_defined_for_all_reals :
  ∀ x : ℝ, |x - 4| + |x + 2| + x - 3 ≠ 0 := by
  sorry

-- The domain of h is all real numbers
theorem domain_of_h_is_real :
  Set.range h = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_defined_for_all_reals_domain_of_h_is_real_l193_19378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_speed_calculation_l193_19361

/-- The speed of a person traveling on a skateboard -/
def peter_speed : ℝ := sorry

/-- The speed of a person traveling on a bike -/
def juan_speed : ℝ := peter_speed + 3

/-- The time they travel -/
def travel_time : ℝ := 1.5

/-- The total distance between them after traveling -/
def total_distance : ℝ := 19.5

theorem peter_speed_calculation : 
  peter_speed * travel_time + juan_speed * travel_time = total_distance → 
  peter_speed = 5 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_peter_speed_calculation_l193_19361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_guesses_l193_19368

theorem binary_sequence_guesses (n k : ℕ) (h : n > k) :
  let guess_count := if k = n / 2 then 2 else 1
  ∀ (leader_seq : Fin n → Bool),
    ∃ (guesses : Fin guess_count → (Fin n → Bool)),
      ∃ (correct_guess : Fin guess_count),
        guesses correct_guess = leader_seq ∧
        (∀ (seq : Fin n → Bool),
          (Finset.card (Finset.filter (λ i => seq i ≠ leader_seq i) (Finset.univ : Finset (Fin n))) = k) →
          ∃ (i : Fin guess_count), guesses i = seq) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_sequence_guesses_l193_19368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_to_second_largest_volume_ratio_l193_19304

/-- Represents a right circular cone sliced into five pieces by planes parallel to its base -/
structure SlicedCone where
  height : ℝ
  baseRadius : ℝ
  sliceHeights : Fin 5 → ℝ

/-- The ratio of slice heights is 1:1:1:1:2 -/
def validSliceHeights (c : SlicedCone) : Prop :=
  c.sliceHeights 0 = c.sliceHeights 1 ∧
  c.sliceHeights 1 = c.sliceHeights 2 ∧
  c.sliceHeights 2 = c.sliceHeights 3 ∧
  c.sliceHeights 3 = c.sliceHeights 4 / 2 ∧
  c.height = c.sliceHeights 0 + c.sliceHeights 1 + c.sliceHeights 2 + c.sliceHeights 3 + c.sliceHeights 4

/-- Volume of a cone slice -/
noncomputable def sliceVolume (c : SlicedCone) (i : Fin 5) : ℝ :=
  (1/3) * Real.pi * (c.baseRadius * (i.val + 1 : ℝ) / 5) ^ 2 * c.sliceHeights i

/-- The theorem to be proved -/
theorem largest_to_second_largest_volume_ratio 
  (c : SlicedCone) 
  (h : validSliceHeights c) : 
  sliceVolume c 4 / sliceVolume c 3 = 25/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_to_second_largest_volume_ratio_l193_19304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l193_19300

open Real

theorem extreme_points_inequality (a : ℝ) :
  let f := fun x : ℝ => x^2 - 2*a*x
  let g := fun x : ℝ => Real.log x
  let h := fun x : ℝ => f x + g x
  ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo 0 (1/2) →
  (∀ x : ℝ, x ≠ x₁ → x ≠ x₂ → deriv h x ≠ 0) →
  deriv h x₁ = 0 → deriv h x₂ = 0 →
  h x₁ - h x₂ > 3/4 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l193_19300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employed_females_percentage_l193_19308

/-- The percentage of employed people in the population -/
noncomputable def employed_percentage : ℝ := 96

/-- The percentage of employed males in the population -/
noncomputable def employed_males_percentage : ℝ := 24

/-- The percentage of employed females out of all employed people -/
noncomputable def employed_females_ratio : ℝ := (employed_percentage - employed_males_percentage) / employed_percentage * 100

theorem employed_females_percentage :
  employed_females_ratio = 75 := by
  -- Unfold the definition of employed_females_ratio
  unfold employed_females_ratio
  -- Unfold the definitions of employed_percentage and employed_males_percentage
  unfold employed_percentage employed_males_percentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employed_females_percentage_l193_19308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l193_19335

theorem quadratic_roots_theorem (d : ℝ) : 
  (∀ x : ℝ, x^2 - 7*x + d = 0 ↔ x = (7 + Real.sqrt (1 + d))/2 ∨ x = (7 - Real.sqrt (1 + d))/2) 
  → d = 48/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_theorem_l193_19335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_tangent_line_l193_19375

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

-- Define the function g
noncomputable def g (a b x : ℝ) : ℝ := f a b x + Real.exp (2*x - 1)

-- State the theorem
theorem extremum_and_tangent_line 
  (a b : ℝ) 
  (h1 : ∃ (y : ℝ), y = f a b (-1) ∧ y = 1)
  (h2 : ∀ x : ℝ, f a b x ≤ f a b (-1)) :
  (a = 1 ∧ b = -1) ∧
  (∃ (m c : ℝ), m = 2 * Real.exp 1 ∧ c = -Real.exp 1 - 3 ∧
    ∀ x y : ℝ, y = g 1 (-1) x ↔ y = m * (x - 1) + g 1 (-1) 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_tangent_line_l193_19375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l193_19364

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition
  Real.sin A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 2 + Real.sin B * Real.sin C →
  -- a = 3
  a = 3 →
  -- Conclusions
  A = 2 * Real.pi / 3 ∧
  ∃ (D : ℝ), D ≤ Real.sqrt 3 / 2 ∧
    ∀ (E : ℝ), (B * D / (B + C) = C * E / (B + C) → E ≤ D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l193_19364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_values_l193_19392

/-- A structure representing the figure with given properties -/
structure Figure where
  -- Three right angles are given
  right_angle_1 : Real
  right_angle_2 : Real
  right_angle_3 : Real
  -- Three segment lengths are marked
  segment_1 : ℝ
  segment_2 : ℝ
  segment_3 : ℝ
  -- Conditions for right angles and segment lengths
  h1 : right_angle_1 = 90
  h2 : right_angle_2 = 90
  h3 : right_angle_3 = 90
  h4 : segment_1 = 8
  h5 : segment_2 = 9
  h6 : segment_3 = 20

/-- Theorem stating the values of v, w, x, y, and z in the figure -/
theorem figure_values (fig : Figure) :
  ∃ (v w x y z : ℝ),
    v = 6 ∧ w = 10 ∧ x = 12 ∧ y = 15 ∧ z = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_values_l193_19392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l193_19373

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the theorem
theorem parabola_segment_length 
  (M N : PointOnParabola) 
  (P : ℝ × ℝ) 
  (h_P : directrix P.1)
  (h_collinear : ∃ (t : ℝ), (P.1 - focus.1, P.2 - focus.2) = t • (M.x - focus.1, M.y - focus.2))
  (h_ratio : (P.1 - focus.1, P.2 - focus.2) = 3 • (M.x - focus.1, M.y - focus.2)) :
  Real.sqrt ((N.x - M.x)^2 + (N.y - M.y)^2) = 32/3 := 
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_segment_length_l193_19373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_l193_19369

/-- Represents a cone with side surface that can be expanded into a sector -/
structure Cone where
  l : ℝ  -- radius of the sector
  θ : ℝ  -- central angle of the sector in radians

/-- The side area of the cone -/
noncomputable def sideArea (c : Cone) : ℝ := (1/2) * c.l^2 * c.θ

/-- The base radius of the cone -/
noncomputable def baseRadius (c : Cone) : ℝ := c.l * c.θ / (2 * Real.pi)

/-- The base area of the cone -/
noncomputable def baseArea (c : Cone) : ℝ := Real.pi * (baseRadius c)^2

/-- The total surface area of the cone -/
noncomputable def surfaceArea (c : Cone) : ℝ := sideArea c + baseArea c

/-- Theorem: For a cone with side surface that can be expanded into a sector
    with central angle 120° (2π/3 radians), the ratio of surface area to side area is 4:3 -/
theorem cone_area_ratio (c : Cone) (h : c.θ = 2 * Real.pi / 3) :
  surfaceArea c / sideArea c = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_area_ratio_l193_19369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l193_19326

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_sq := u.1 * u.1 + u.2 * u.2
  (dot / norm_sq * u.1, dot / norm_sq * u.2)

theorem vector_satisfies_projections :
  let v : ℝ × ℝ := (7, 5)
  let u₁ : ℝ × ℝ := (2, 1)
  let u₂ : ℝ × ℝ := (2, 3)
  proj u₁ v = (38/5, 19/5) ∧ proj u₂ v = (58/13, 87/13) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_satisfies_projections_l193_19326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l193_19342

open Real

/-- The function f(x) = sin(ωx) - √3 cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) := sin (ω * x) - Real.sqrt 3 * cos (ω * x)

/-- The theorem stating the range of ω given the conditions -/
theorem omega_range (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (a b : ℝ), 0 < a ∧ a < b ∧ b < π ∧ 
    f ω a = 0 ∧ f ω b = 0 ∧
    ∀ x, 0 < x ∧ x < π ∧ f ω x = 0 → (x = a ∨ x = b)) :
  4/3 < ω ∧ ω ≤ 7/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l193_19342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l193_19343

-- Define a circle with center O and radius 5
def myCircle (O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) ≤ 5}

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem statement
theorem point_outside_circle (O P : ℝ × ℝ) :
  P ∉ myCircle O → distance O P ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l193_19343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l193_19352

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 1) => 2 * a n + 2^n

def b (n : ℕ) : ℚ := if n = 0 then 1 else (a n : ℚ) / 2^(n - 1)

def S (n : ℕ) : ℕ := (n - 1) * 2^n + 1

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) - b n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → S n = (n - 1) * 2^n + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l193_19352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l193_19327

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race configuration -/
structure RaceConfig where
  anton : Runner
  seryozha : Runner
  tolya : Runner
  race_distance : ℝ

/-- The main theorem about the race -/
theorem race_result (config : RaceConfig) 
  (h1 : config.race_distance = 100)
  (h2 : config.race_distance - config.seryozha.speed * (config.race_distance / config.anton.speed) = 10)
  (h3 : config.race_distance - config.tolya.speed * (config.race_distance / config.seryozha.speed) = 10)
  (h4 : config.anton.speed > 0)
  (h5 : config.seryozha.speed > 0)
  (h6 : config.tolya.speed > 0) :
  config.race_distance - config.tolya.speed * (config.race_distance / config.anton.speed) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l193_19327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_over_2_l193_19379

-- Define the function f(x) = x * cos(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

-- State the theorem
theorem f_derivative_at_pi_over_2 :
  deriv f (π / 2) = -π / 2 := by
  -- Calculate the derivative
  have h1 : deriv f = fun x ↦ Real.cos x - x * Real.sin x := by
    sorry -- Proof of the derivative calculation
  
  -- Evaluate at π/2
  have h2 : (fun x ↦ Real.cos x - x * Real.sin x) (π / 2) = -π / 2 := by
    sorry -- Proof of the evaluation at π/2
  
  -- Combine the results
  rw [h1]
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_pi_over_2_l193_19379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_pi_fourth_l193_19396

theorem cos_x_plus_pi_fourth (x : ℝ) 
  (h1 : π/4 < x ∧ x < π/2) 
  (h2 : Real.sqrt 2 * Real.cos x + Real.sqrt 2 * Real.sin x = 8/5) : 
  Real.cos (x + π/4) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_x_plus_pi_fourth_l193_19396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millys_math_homework_time_l193_19354

/-- Represents the time in minutes Milly spends on math homework -/
def math_time : ℝ → Prop := λ _ => True

/-- Represents the time in minutes Milly spends on geography homework -/
noncomputable def geography_time (m : ℝ) : ℝ := m / 2

/-- Represents the time in minutes Milly spends on science homework -/
noncomputable def science_time (m : ℝ) : ℝ := (m + geography_time m) / 2

/-- The total time Milly spends studying -/
def total_study_time : ℝ := 135

theorem millys_math_homework_time :
  ∃ m : ℝ, math_time m ∧
    m + geography_time m + science_time m = total_study_time ∧
    m = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_millys_math_homework_time_l193_19354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l193_19365

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sin (Real.pi / 2 - x) * Real.cos (Real.pi / 2 + x)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f x = f (x + p)) ∧
  (∀ (x : ℝ), f (Real.pi / 8 + x) = f (Real.pi / 8 - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l193_19365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l193_19399

-- Define vectors in R²
def a : Fin 2 → ℝ := ![1, 2]

noncomputable def b (α : ℝ) : Fin 2 → ℝ := ![Real.cos α, Real.sin α]

noncomputable def c (α t : ℝ) : Fin 2 → ℝ := a - t • (b α)

-- Part 1
theorem part_one (α : ℝ) : 
  (∃ k : ℝ, c α 1 = k • (b α)) → 2 * (Real.cos α)^2 - Real.sin (2 * α) = -2/5 := by sorry

-- Part 2
theorem part_two :
  let c_length (t : ℝ) := Real.sqrt ((c (π/4) t 0)^2 + (c (π/4) t 1)^2)
  let min_c_length := Real.sqrt 2 / 2
  let t_min := 3 * Real.sqrt 2 / 2
  min_c_length = Real.sqrt 2 / 2 ∧
  (a 0 * c (π/4) t_min 0 + a 1 * c (π/4) t_min 1) / 
  Real.sqrt ((c (π/4) t_min 0)^2 + (c (π/4) t_min 1)^2) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l193_19399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_loaned_books_l193_19360

/-- Represents a book collection in the library --/
structure Collection where
  initial : ℕ
  returned_percent : ℚ
  remaining : ℕ

/-- Calculates the number of books loaned out from a collection --/
def books_loaned_out (c : Collection) : ℕ :=
  (((c.initial - c.remaining : ℚ) / c.returned_percent).floor).toNat

theorem library_loaned_books :
  let collection_a : Collection := ⟨300, 3/5, 244⟩
  let collection_b : Collection := ⟨450, 7/10, 386⟩
  let collection_c : Collection := ⟨350, 4/5, 290⟩
  let collection_d : Collection := ⟨100, 1/2, 76⟩
  books_loaned_out collection_a = 93 ∧
  books_loaned_out collection_b = 91 ∧
  books_loaned_out collection_c = 75 ∧
  books_loaned_out collection_d = 48 :=
by
  sorry

#eval books_loaned_out ⟨300, 3/5, 244⟩
#eval books_loaned_out ⟨450, 7/10, 386⟩
#eval books_loaned_out ⟨350, 4/5, 290⟩
#eval books_loaned_out ⟨100, 1/2, 76⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_loaned_books_l193_19360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l193_19381

-- Define the triangle ABC
def Triangle (A B C : Real) (a b c : Real) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * Real.sin B = b * Real.cos A

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x * Real.cos x

theorem triangle_theorem (A B C a b c : Real) 
  (h : Triangle A B C a b c) 
  (hB : 0 < B ∧ B < 3 * Real.pi / 4) : 
  A = Real.pi / 4 ∧ 
  ∀ y ∈ Set.Icc (-(1 + Real.sqrt 3) / 2) (1 / 2), ∃ x, 0 < x ∧ x < 3 * Real.pi / 4 ∧ f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l193_19381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l193_19382

noncomputable def floor (z : ℝ) : ℤ := ⌊z⌋

theorem problem_statement (x y : ℝ) : 
  (y = 4 * (floor x) + 5) →
  (y = 5 * (floor (x - 3)) + 2 * x + 7) →
  (∀ n : ℤ, x ≠ ↑n) →
  (x > 3) →
  (x + y = 32.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l193_19382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_power_minus_n_is_even_l193_19324

-- Define z as noncomputable
noncomputable def z : ℝ := (3 + Real.sqrt 17) / 2

-- Theorem statement
theorem floor_power_minus_n_is_even (n : ℕ+) :
  ∃ (k : ℤ), ⌊z^(n : ℝ)⌋ - (n : ℤ) = 2 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_power_minus_n_is_even_l193_19324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l193_19356

def S (n : ℕ) : ℕ := n^2 + n

noncomputable def b (x : ℝ) (n : ℕ) : ℝ := x^(n-1)

noncomputable def c (x : ℝ) (n : ℕ) : ℝ := 2*n * (b x n)

noncomputable def T (x : ℝ) (n : ℕ) : ℝ := 
  if x = 1 then n^2 + n
  else (2 - 2*(n+1)*x^n + 2*n*x^(n+1)) / (1-x)^2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → S n - S (n-1) = 2*n) ∧
  (∀ x : ℝ, x ≠ 1 → ∀ n : ℕ, T x n = (2 - 2*(n+1)*x^n + 2*n*x^(n+1)) / (1-x)^2) ∧
  (∀ n : ℕ, T 1 n = n^2 + n) ∧
  (∀ n : ℕ, n > 0 → (n * T 2 (n+1) - 2*n) / (T 2 (n+2) - 2) ≥ 1/4) ∧
  (∃ n : ℕ, n > 0 ∧ (n * T 2 (n+1) - 2*n) / (T 2 (n+2) - 2) = 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l193_19356
