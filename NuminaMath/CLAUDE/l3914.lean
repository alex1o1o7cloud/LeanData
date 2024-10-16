import Mathlib

namespace NUMINAMATH_CALUDE_log_product_simplification_l3914_391413

theorem log_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x / Real.log (y^6)) * (Real.log (y^2) / Real.log (x^5)) *
  (Real.log (x^3) / Real.log (y^4)) * (Real.log (y^4) / Real.log (x^3)) *
  (Real.log (x^5) / Real.log (y^2)) = (1/6) * (Real.log x / Real.log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_simplification_l3914_391413


namespace NUMINAMATH_CALUDE_projection_of_b_onto_a_l3914_391459

def a : Fin 3 → ℝ := ![2, -1, 2]
def b : Fin 3 → ℝ := ![1, -2, 1]

theorem projection_of_b_onto_a :
  let proj := (a • b) / (a • a) • a
  proj 0 = 4/3 ∧ proj 1 = -2/3 ∧ proj 2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_b_onto_a_l3914_391459


namespace NUMINAMATH_CALUDE_generate_numbers_l3914_391431

/-- A type representing arithmetic expressions using five 3's -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | pow : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.pow e1 e2 => (eval e1) ^ (eval e2).num

/-- Count the number of 3's used in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2
  | Expr.pow e1 e2 => count_threes e1 + count_threes e2

/-- The main theorem stating that all integers from 1 to 39 can be generated -/
theorem generate_numbers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 39 →
  ∃ e : Expr, count_threes e = 5 ∧ eval e = n := by sorry

end NUMINAMATH_CALUDE_generate_numbers_l3914_391431


namespace NUMINAMATH_CALUDE_peach_boxes_theorem_l3914_391409

/-- Given the initial number of peaches per basket, the number of baskets,
    the number of peaches eaten, and the number of peaches per smaller box,
    calculate the number of smaller boxes of peaches. -/
def number_of_boxes (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_small_box : ℕ) : ℕ :=
  ((peaches_per_basket * num_baskets) - peaches_eaten) / peaches_per_small_box

/-- Prove that under the given conditions, the number of smaller boxes of peaches is 8. -/
theorem peach_boxes_theorem :
  number_of_boxes 25 5 5 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_boxes_theorem_l3914_391409


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3914_391414

theorem max_candy_leftover (x : ℕ) (h : x > 0) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3914_391414


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3914_391439

theorem basketball_score_proof (total_points : ℕ) : 
  (∃ (linda_points maria_points other_points : ℕ),
    linda_points = total_points / 5 ∧ 
    maria_points = total_points * 3 / 8 ∧
    other_points ≤ 16 ∧
    linda_points + maria_points + 18 + other_points = total_points ∧
    other_points ≤ 8 * 2) →
  (∃ (other_points : ℕ), 
    other_points = 16 ∧
    other_points ≤ 8 * 2 ∧
    ∃ (linda_points maria_points : ℕ),
      linda_points = total_points / 5 ∧ 
      maria_points = total_points * 3 / 8 ∧
      linda_points + maria_points + 18 + other_points = total_points) :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3914_391439


namespace NUMINAMATH_CALUDE_probability_of_red_is_one_fifth_l3914_391406

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball -/
def probabilityOfRed (bag : BagContents) : ℚ :=
  bag.red / (bag.red + bag.white + bag.black)

/-- Theorem stating that the probability of drawing a red ball is 1/5 -/
theorem probability_of_red_is_one_fifth (bag : BagContents) 
  (h1 : bag.red = 2) 
  (h2 : bag.white = 3) 
  (h3 : bag.black = 5) : 
  probabilityOfRed bag = 1/5 := by
  sorry

#check probability_of_red_is_one_fifth

end NUMINAMATH_CALUDE_probability_of_red_is_one_fifth_l3914_391406


namespace NUMINAMATH_CALUDE_students_without_A_l3914_391419

theorem students_without_A (total : ℕ) (english_A : ℕ) (math_A : ℕ) (both_A : ℕ) : 
  total = 40 →
  english_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - (english_A + math_A - both_A) = 18 := by
sorry

end NUMINAMATH_CALUDE_students_without_A_l3914_391419


namespace NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l3914_391456

/-- The star operation defined as a ☆ b = ab^2 - ab - 1 -/
def star (a b : ℝ) : ℝ := a * b^2 - a * b - 1

/-- Theorem stating that the equation 1 ☆ x = 0 has two distinct real roots -/
theorem star_equation_has_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ star 1 x = 0 ∧ star 1 y = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l3914_391456


namespace NUMINAMATH_CALUDE_factorial_sum_quotient_l3914_391488

theorem factorial_sum_quotient : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_quotient_l3914_391488


namespace NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l3914_391447

-- Define the structure for a tile
structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

-- Define the tiles
def tileI : Tile := ⟨1, 2, 5, 6⟩
def tileII : Tile := ⟨6, 3, 1, 5⟩
def tileIII : Tile := ⟨5, 7, 2, 3⟩
def tileIV : Tile := ⟨3, 5, 7, 2⟩

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) (side : String) : Prop :=
  match side with
  | "right" => t1.right = t2.left
  | "left" => t1.left = t2.right
  | "top" => t1.top = t2.bottom
  | "bottom" => t1.bottom = t2.top
  | _ => False

-- Theorem stating that Tile IV is the only tile that can be placed in Rectangle C
theorem tileIV_in_rectangle_C :
  (canBeAdjacent tileIV tileIII "left") ∧
  (¬ canBeAdjacent tileI tileIII "left") ∧
  (¬ canBeAdjacent tileII tileIII "left") ∧
  (∃ (t : Tile), t = tileIV ∧ canBeAdjacent t tileIII "left") :=
sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l3914_391447


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l3914_391448

theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →              -- b is the next odd number after a
  c = a + 4 →              -- c is the third odd number
  d = a + 6 →              -- d is the fourth odd number
  e = a + 8 →              -- e is the fifth odd number
  a + c = 146 →            -- sum of a and c is 146
  e = 79 := by             -- prove that e equals 79
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l3914_391448


namespace NUMINAMATH_CALUDE_solve_system_l3914_391442

theorem solve_system (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3914_391442


namespace NUMINAMATH_CALUDE_triangle_geometric_sequence_l3914_391450

/-- In a triangle ABC, if sides a, b, c form a geometric sequence and angle A is 60°,
    then (b * sin B) / c = √3/2 -/
theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b / c = a / b) →  -- Geometric sequence condition
  A = π / 3 →        -- 60° in radians
  A + B + C = π →    -- Sum of angles in a triangle
  a = b * Real.sin A / Real.sin B →  -- Sine rule
  b = c * Real.sin B / Real.sin C →  -- Sine rule
  c = a * Real.sin C / Real.sin A →  -- Sine rule
  (b * Real.sin B) / c = Real.sqrt 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_geometric_sequence_l3914_391450


namespace NUMINAMATH_CALUDE_concert_ticket_price_l3914_391408

theorem concert_ticket_price :
  ∃ (P : ℝ) (x : ℕ),
    x + 2 + 1 = 5 ∧
    x * P + (2 * 2.4 * P - 10) + 0.6 * P = 360 →
    P = 50 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l3914_391408


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3914_391443

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem quadratic_function_range :
  ∃ (a b : ℝ), a = -4 ∧ b = 5 ∧
  (∀ x, x ∈ Set.Icc 0 5 → f x ∈ Set.Icc a b) ∧
  (∀ y, y ∈ Set.Icc a b → ∃ x, x ∈ Set.Icc 0 5 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3914_391443


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3914_391402

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + 5 * b < 100) :
  ab * (100 - 4 * a - 5 * b) ≤ 50000 / 27 :=
by sorry

theorem max_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 4 * a + 5 * b < 100 ∧
  ab * (100 - 4 * a - 5 * b) > 50000 / 27 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l3914_391402


namespace NUMINAMATH_CALUDE_ellipse_equation_l3914_391454

/-- Given an ellipse with focal distance 4 passing through (√2, √3), prove its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) → 
  (2 / a^2 + 3 / b^2 = 1) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3914_391454


namespace NUMINAMATH_CALUDE_reflection_of_point_p_l3914_391490

/-- The coordinates of a point with respect to the center of the coordinate origin -/
def reflection_through_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- Theorem: The coordinates of P(-3,1) with respect to the center of the coordinate origin are (3,-1) -/
theorem reflection_of_point_p : reflection_through_origin (-3) 1 = (3, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_p_l3914_391490


namespace NUMINAMATH_CALUDE_number_of_officers_prove_number_of_officers_l3914_391445

theorem number_of_officers (num_jawans : ℕ) (total_ways : ℕ) : ℕ :=
  let officers := total_ways / (num_jawans.choose 5)
  officers

#check number_of_officers 8 224 = 4

theorem prove_number_of_officers :
  number_of_officers 8 224 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_officers_prove_number_of_officers_l3914_391445


namespace NUMINAMATH_CALUDE_triple_angle_square_equal_to_circle_l3914_391440

-- Tripling an angle
theorem triple_angle (α : Real) : ∃ β, β = 3 * α := by sorry

-- Constructing a square equal in area to a given circle
theorem square_equal_to_circle (r : Real) : 
  ∃ s, s^2 = π * r^2 := by sorry

end NUMINAMATH_CALUDE_triple_angle_square_equal_to_circle_l3914_391440


namespace NUMINAMATH_CALUDE_cubic_roots_theorem_l3914_391420

theorem cubic_roots_theorem (a b c r s t : ℝ) : 
  (∀ x, x^3 + 3*x^2 + 4*x - 11 = (x - a) * (x - b) * (x - c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) →
  t = 23 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_theorem_l3914_391420


namespace NUMINAMATH_CALUDE_correct_calculation_l3914_391421

theorem correct_calculation (a b : ℝ) : 3 * a * b^2 - 5 * b^2 * a = -2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3914_391421


namespace NUMINAMATH_CALUDE_injured_cats_count_l3914_391484

/-- The number of injured cats Jeff found on Tuesday -/
def injured_cats : ℕ :=
  let initial_cats : ℕ := 20
  let kittens_found : ℕ := 2
  let cats_adopted : ℕ := 3 * 2
  let final_cats : ℕ := 17
  final_cats - (initial_cats + kittens_found - cats_adopted)

theorem injured_cats_count : injured_cats = 1 := by
  sorry

end NUMINAMATH_CALUDE_injured_cats_count_l3914_391484


namespace NUMINAMATH_CALUDE_common_point_on_intersection_circle_l3914_391410

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  p : ℝ
  q : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : x₁ ≠ 0
  h₂ : x₂ ≠ 0
  h₃ : q ≠ 0
  h₄ : x₁ ≠ x₂
  h₅ : x₁^2 + p*x₁ + q = 0  -- x₁ is a root
  h₆ : x₂^2 + p*x₂ + q = 0  -- x₂ is a root

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersectionCircle (para : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  {pt : ℝ × ℝ | ∃ (r : ℝ), (pt.1 - 0)^2 + (pt.2 - 0)^2 = r^2 ∧
                           (pt.1 - para.x₁)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - para.x₂)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - 0)^2 + (pt.2 - para.q)^2 = r^2}

/-- The theorem stating that R(0, 1) lies on the intersection circle for all valid parabolas -/
theorem common_point_on_intersection_circle (para : TripleIntersectingParabola) :
  (0, 1) ∈ intersectionCircle para := by
  sorry

end NUMINAMATH_CALUDE_common_point_on_intersection_circle_l3914_391410


namespace NUMINAMATH_CALUDE_symmetry_axis_implies_a_equals_one_l3914_391482

/-- The line equation -/
def line_equation (x y a : ℝ) : Prop := x - 2*a*y - 3 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 3 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The theorem stating that if the line is a symmetry axis of the circle, then a = 1 -/
theorem symmetry_axis_implies_a_equals_one (a : ℝ) :
  (∀ x y : ℝ, line_equation x y a → circle_equation x y) →
  (line_equation (circle_center.1) (circle_center.2) a) →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_axis_implies_a_equals_one_l3914_391482


namespace NUMINAMATH_CALUDE_eggs_and_cakes_l3914_391416

def dozen : ℕ := 12

def initial_eggs : ℕ := 7 * dozen
def used_eggs : ℕ := 5 * dozen
def eggs_per_cake : ℕ := (3 * dozen) / 2

theorem eggs_and_cakes :
  let remaining_eggs := initial_eggs - used_eggs
  let possible_cakes := remaining_eggs / eggs_per_cake
  remaining_eggs = 24 ∧ possible_cakes = 1 := by sorry

end NUMINAMATH_CALUDE_eggs_and_cakes_l3914_391416


namespace NUMINAMATH_CALUDE_f_e_plus_f_prime_e_l3914_391403

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem f_e_plus_f_prime_e : f (Real.exp 1) + (deriv f) (Real.exp 1) = 2 * Real.exp (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_f_e_plus_f_prime_e_l3914_391403


namespace NUMINAMATH_CALUDE_inequality_solution_l3914_391476

theorem inequality_solution (x : ℝ) : 
  x ≠ -2 ∧ x ≠ 2 →
  ((2 * x + 1) / (x + 2) - (x - 3) / (3 * x - 6) ≤ 0 ↔ 
   (x > -2 ∧ x < 0) ∨ (x > 2 ∧ x ≤ 14/5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3914_391476


namespace NUMINAMATH_CALUDE_count_distinct_z_values_l3914_391474

def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℤ) : ℤ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  100 * ones + 10 * tens + hundreds

def z_value (x : ℤ) : ℤ := |x - reverse_digits x|

def satisfies_conditions (x : ℤ) : Prop :=
  is_three_digit x ∧ 
  is_three_digit (reverse_digits x) ∧
  (z_value x) % 33 = 0

theorem count_distinct_z_values :
  ∃ (S : Finset ℤ), 
    (∀ x, satisfies_conditions x → z_value x ∈ S) ∧ 
    (∀ z ∈ S, ∃ x, satisfies_conditions x ∧ z_value x = z) ∧
    Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_z_values_l3914_391474


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l3914_391405

-- Proposition 2
theorem prop_2 (p q : Prop) : ¬(p ∨ q) → (¬p ∧ ¬q) := by sorry

-- Proposition 4
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem prop_4 (a : ℝ) : (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l3914_391405


namespace NUMINAMATH_CALUDE_complex_number_with_given_real_part_and_magnitude_l3914_391411

theorem complex_number_with_given_real_part_and_magnitude (z : ℂ) : 
  (z.re = 5) → (Complex.abs z = Complex.abs (4 - 3*I)) → (z.im = 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_with_given_real_part_and_magnitude_l3914_391411


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3914_391467

theorem arithmetic_sqrt_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l3914_391467


namespace NUMINAMATH_CALUDE_smallest_box_for_vase_l3914_391487

/-- Represents a cylindrical vase -/
structure Vase where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube-shaped box -/
structure CubeBox where
  sideLength : ℝ

/-- The volume of a cube-shaped box -/
def boxVolume (box : CubeBox) : ℝ := box.sideLength ^ 3

/-- Predicate to check if a vase fits upright in a box -/
def fitsUpright (v : Vase) (b : CubeBox) : Prop :=
  v.height ≤ b.sideLength ∧ v.baseDiameter ≤ b.sideLength

theorem smallest_box_for_vase (v : Vase) (h1 : v.height = 15) (h2 : v.baseDiameter = 8) :
  ∃ (b : CubeBox), fitsUpright v b ∧
    (∀ (b' : CubeBox), fitsUpright v b' → boxVolume b ≤ boxVolume b') ∧
    boxVolume b = 3375 := by
  sorry

end NUMINAMATH_CALUDE_smallest_box_for_vase_l3914_391487


namespace NUMINAMATH_CALUDE_legos_lost_l3914_391415

def initial_legos : ℕ := 2080
def current_legos : ℕ := 2063

theorem legos_lost : initial_legos - current_legos = 17 := by
  sorry

end NUMINAMATH_CALUDE_legos_lost_l3914_391415


namespace NUMINAMATH_CALUDE_product_mod_25_l3914_391434

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (105 * 77 * 132) % 25 = m ∧ m = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l3914_391434


namespace NUMINAMATH_CALUDE_problem_statement_l3914_391429

-- Define the basic geometric shapes
def Quadrilateral : Type := Unit
def Square : Type := Unit
def Trapezoid : Type := Unit
def Parallelogram : Type := Unit

-- Define the properties
def has_equal_sides (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define the propositions
def proposition_1 : Prop :=
  ∀ q : Quadrilateral, ¬(has_equal_sides q → is_square q)

def proposition_2 : Prop :=
  ∀ q : Quadrilateral, is_parallelogram q → ¬is_trapezoid q

def proposition_3 (a b c : ℝ) : Prop :=
  a > b → a * c^2 > b * c^2

theorem problem_statement :
  proposition_1 ∧
  proposition_2 ∧
  ¬(∀ a b c : ℝ, proposition_3 a b c) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3914_391429


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l3914_391479

theorem trishul_investment_percentage (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →
  trishul + vishal + raghu = 5780 →
  raghu = 2000 →
  (raghu - trishul) / raghu = 0.1 := by
sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l3914_391479


namespace NUMINAMATH_CALUDE_printer_equation_l3914_391471

theorem printer_equation (y : ℝ) : y > 0 →
  (300 : ℝ) / 6 + 300 / y = 300 / 3 ↔ 1 / 6 + 1 / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_printer_equation_l3914_391471


namespace NUMINAMATH_CALUDE_fundraising_contribution_l3914_391457

theorem fundraising_contribution (total_goal : ℕ) (already_raised : ℕ) (num_people : ℕ) :
  total_goal = 2400 →
  already_raised = 600 →
  num_people = 8 →
  (total_goal - already_raised) / num_people = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_fundraising_contribution_l3914_391457


namespace NUMINAMATH_CALUDE_painting_price_increase_l3914_391489

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 20 / 100) = 104 / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l3914_391489


namespace NUMINAMATH_CALUDE_value_after_percentage_increase_l3914_391438

theorem value_after_percentage_increase 
  (x : ℝ) (p : ℝ) (y : ℝ) 
  (h1 : x = 400) 
  (h2 : p = 20) :
  y = x * (1 + p / 100) → y = 480 := by
  sorry

end NUMINAMATH_CALUDE_value_after_percentage_increase_l3914_391438


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l3914_391494

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.0000084

/-- The expected scientific notation representation -/
def expected_representation : ScientificNotation := {
  coefficient := 8.4
  exponent := -6
  is_normalized := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : 
  given_number = expected_representation.coefficient * (10 : ℝ) ^ expected_representation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l3914_391494


namespace NUMINAMATH_CALUDE_triangle_theorem_l3914_391446

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin t.C - Real.sqrt 3 * t.c * Real.cos t.A = 0)
  (h2 : t.a = 2)
  (h3 : (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3) : 
  t.A = π/3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3914_391446


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3914_391401

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + (2*m - 5) * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 + (2*m - 5) * y + 12 = 0 → y = x) ↔ 
  m = 8.5 ∨ m = -3.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3914_391401


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3914_391483

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The statement to be proved -/
theorem floor_equation_solution :
  let x : ℚ := 22 / 7
  x * (floor (x * (floor (x * (floor x))))) = 88 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3914_391483


namespace NUMINAMATH_CALUDE_quadratic_roots_l3914_391496

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 3 ∧ x₂ = -1 ∧ 
  x₁^2 - 2*x₁ - 3 = 0 ∧ x₂^2 - 2*x₂ - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3914_391496


namespace NUMINAMATH_CALUDE_no_zero_points_implies_a_leq_two_l3914_391449

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) - 2 * Real.log x

theorem no_zero_points_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0) →
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_no_zero_points_implies_a_leq_two_l3914_391449


namespace NUMINAMATH_CALUDE_games_for_champion_l3914_391451

/-- Represents a single-elimination tournament -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- The number of games required to determine the champion in a single-elimination tournament -/
def games_required (t : Tournament) : ℕ :=
  t.num_players - 1

theorem games_for_champion (t : Tournament) (h1 : t.single_elimination = true) (h2 : t.num_players = 512) :
  games_required t = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_for_champion_l3914_391451


namespace NUMINAMATH_CALUDE_partition_inequality_l3914_391436

def f (n : ℕ) : ℕ := sorry

theorem partition_inequality (n : ℕ) (h : n ≥ 1) :
  f (n + 1) ≤ (f n + f (n + 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_partition_inequality_l3914_391436


namespace NUMINAMATH_CALUDE_barbara_scrap_paper_heaps_l3914_391495

/-- The number of bundles of colored paper Barbara found -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper Barbara found -/
def white_bunches : ℕ := 2

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The total number of sheets Barbara removed -/
def total_sheets_removed : ℕ := 114

/-- Theorem stating that Barbara found 5 heaps of scrap paper -/
theorem barbara_scrap_paper_heaps : 
  (total_sheets_removed - (colored_bundles * sheets_per_bundle + white_bunches * sheets_per_bunch)) / sheets_per_heap = 5 := by
  sorry

end NUMINAMATH_CALUDE_barbara_scrap_paper_heaps_l3914_391495


namespace NUMINAMATH_CALUDE_sine_is_periodic_l3914_391404

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the sine function
def sin : ℝ → ℝ := sorry

-- Theorem statement
theorem sine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric sin →
  IsPeriodic sin := by sorry

end NUMINAMATH_CALUDE_sine_is_periodic_l3914_391404


namespace NUMINAMATH_CALUDE_morgans_blue_pens_l3914_391437

theorem morgans_blue_pens (red_pens black_pens total_pens : ℕ) 
  (h1 : red_pens = 65)
  (h2 : black_pens = 58)
  (h3 : total_pens = 168)
  : total_pens - (red_pens + black_pens) = 45 := by
  sorry

end NUMINAMATH_CALUDE_morgans_blue_pens_l3914_391437


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_g_l3914_391466

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8|

-- Define the interval [3, 10]
def I : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ I, g x ≤ max_g) ∧
    (∃ x ∈ I, g x = max_g) ∧
    (∀ x ∈ I, min_g ≤ g x) ∧
    (∃ x ∈ I, g x = min_g) ∧
    max_g + min_g = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_g_l3914_391466


namespace NUMINAMATH_CALUDE_dragon_head_configuration_l3914_391497

-- Define the type for dragons
inductive Dragon
| Truthful
| Deceitful

-- Define the type for heads
structure Head where
  id : Nat
  statement : Prop
  dragon : Dragon

-- Define the statements made by each head
def statement1 : Prop := true  -- "I am a truthful head"
def statement2 (h3 : Head) : Prop := h3.dragon = Dragon.Truthful  -- "The third head is my original head"
def statement3 (h2 : Head) : Prop := h2.dragon ≠ Dragon.Truthful  -- "The second head is not my original head"
def statement4 (h3 : Head) : Prop := h3.dragon = Dragon.Deceitful  -- "The third head is a liar"

-- Define the theorem
theorem dragon_head_configuration 
  (h1 h2 h3 h4 : Head)
  (h1_def : h1 = { id := 1, statement := statement1, dragon := Dragon.Truthful })
  (h2_def : h2 = { id := 2, statement := statement2 h3, dragon := Dragon.Deceitful })
  (h3_def : h3 = { id := 3, statement := statement3 h2, dragon := Dragon.Truthful })
  (h4_def : h4 = { id := 4, statement := statement4 h3, dragon := Dragon.Deceitful })
  (truthful_condition : ∀ h : Head, h.dragon = Dragon.Truthful → h.statement)
  (deceitful_condition : ∀ h : Head, h.dragon = Dragon.Deceitful → ¬h.statement)
  : (h1.dragon = h3.dragon ∧ h1.dragon = Dragon.Truthful) ∧ 
    (h2.dragon = h4.dragon ∧ h2.dragon = Dragon.Deceitful) :=
sorry

end NUMINAMATH_CALUDE_dragon_head_configuration_l3914_391497


namespace NUMINAMATH_CALUDE_bank_account_growth_l3914_391452

/-- Calculates the final amount after compound interest is applied -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that $100 invested at 10% annual interest for 2 years results in $121 -/
theorem bank_account_growth : compound_interest 100 0.1 2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_growth_l3914_391452


namespace NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3914_391444

/-- 
Given two congruent right circular cones with base radius 5 and height 12,
whose axes of symmetry intersect at right angles at a point 4 units from
the base of each cone, prove that the maximum possible value of r^2 for a
sphere lying within both cones is 625/169.
-/
theorem max_sphere_in_intersecting_cones (r : ℝ) : 
  let base_radius : ℝ := 5
  let cone_height : ℝ := 12
  let intersection_distance : ℝ := 4
  let slant_height : ℝ := Real.sqrt (cone_height^2 + base_radius^2)
  let max_r_squared : ℝ := (base_radius * (slant_height - intersection_distance) / slant_height)^2
  max_r_squared = 625 / 169 :=
by sorry

end NUMINAMATH_CALUDE_max_sphere_in_intersecting_cones_l3914_391444


namespace NUMINAMATH_CALUDE_rectangle_area_l3914_391418

/-- Theorem: Area of a rectangle with specific properties -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 12 →
  width * 1.2 = length →
  length * width = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3914_391418


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3914_391441

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3914_391441


namespace NUMINAMATH_CALUDE_largest_triple_product_digit_sum_l3914_391407

def is_single_digit_prime (p : Nat) : Prop :=
  p ≥ 2 ∧ p < 10 ∧ Nat.Prime p

def is_valid_triple (d e : Nat) : Prop :=
  is_single_digit_prime d ∧ 
  is_single_digit_prime e ∧ 
  Nat.Prime (d + 10 * e)

def product_of_triple (d e : Nat) : Nat :=
  d * e * (d + 10 * e)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_triple_product_digit_sum :
  ∃ (d e : Nat),
    is_valid_triple d e ∧
    (∀ (d' e' : Nat), is_valid_triple d' e' → product_of_triple d' e' ≤ product_of_triple d e) ∧
    sum_of_digits (product_of_triple d e) = 21 :=
by sorry

end NUMINAMATH_CALUDE_largest_triple_product_digit_sum_l3914_391407


namespace NUMINAMATH_CALUDE_figures_per_shelf_l3914_391412

theorem figures_per_shelf (total_figures : ℕ) (num_shelves : ℕ) 
  (h1 : total_figures = 64) (h2 : num_shelves = 8) :
  total_figures / num_shelves = 8 := by
  sorry

end NUMINAMATH_CALUDE_figures_per_shelf_l3914_391412


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l3914_391468

theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 10 → 
  (math + chem) / 2 = 35 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l3914_391468


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3914_391458

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24300)
  (b_divisible : 100 ∣ b)
  (b_div_100 : b / 100 = a) :
  b - a = 23760 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3914_391458


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3914_391473

theorem lcm_of_ratio_and_hcf (a b c : ℕ+) : 
  (∃ (k : ℕ+), a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) → 
  Nat.gcd a (Nat.gcd b c) = 6 →
  Nat.lcm a (Nat.lcm b c) = 180 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3914_391473


namespace NUMINAMATH_CALUDE_angle_expression_value_l3914_391492

theorem angle_expression_value (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ - π) = -1/2) :
  Real.sqrt ((1 + Real.cos θ) / (1 - Real.sin (π/2 - θ))) - 
  Real.sqrt ((1 - Real.cos θ) / (1 + Real.sin (θ - 3*π/2))) = -4 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l3914_391492


namespace NUMINAMATH_CALUDE_smallest_fraction_l3914_391424

theorem smallest_fraction (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) : a / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l3914_391424


namespace NUMINAMATH_CALUDE_seungjus_class_size_l3914_391417

theorem seungjus_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 :=
sorry

end NUMINAMATH_CALUDE_seungjus_class_size_l3914_391417


namespace NUMINAMATH_CALUDE_tims_weekly_water_consumption_l3914_391472

/-- Calculates Tim's weekly water consumption in ounces -/
theorem tims_weekly_water_consumption :
  let quart_to_oz : ℚ → ℚ := (· * 32)
  let daily_bottle_oz := 2 * quart_to_oz 1.5
  let daily_total_oz := daily_bottle_oz + 20
  let weekly_oz := 7 * daily_total_oz
  weekly_oz = 812 := by sorry

end NUMINAMATH_CALUDE_tims_weekly_water_consumption_l3914_391472


namespace NUMINAMATH_CALUDE_classroom_setup_l3914_391428

/-- Represents the number of desks in a classroom setup for an exam. -/
def num_desks : ℕ := 33

/-- Represents the number of chairs per desk. -/
def chairs_per_desk : ℕ := 4

/-- Represents the number of legs per chair. -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per desk. -/
def legs_per_desk : ℕ := 6

/-- Represents the total number of legs from all desks and chairs. -/
def total_legs : ℕ := 728

theorem classroom_setup :
  num_desks * chairs_per_desk * legs_per_chair + num_desks * legs_per_desk = total_legs :=
by sorry

end NUMINAMATH_CALUDE_classroom_setup_l3914_391428


namespace NUMINAMATH_CALUDE_x_value_l3914_391430

theorem x_value : ∃ x : ℚ, (10 * x = x + 20) ∧ (x = 20 / 9) := by sorry

end NUMINAMATH_CALUDE_x_value_l3914_391430


namespace NUMINAMATH_CALUDE_expression_value_l3914_391463

theorem expression_value : 
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  x^2 * y * z - x * y * z^2 = 48 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3914_391463


namespace NUMINAMATH_CALUDE_simplify_expression_l3914_391461

theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^(2/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3914_391461


namespace NUMINAMATH_CALUDE_nails_on_square_plate_l3914_391453

/-- Represents a square plate with nails along its edges -/
structure SquarePlate where
  side_length : ℕ
  total_nails : ℕ
  nails_per_side : ℕ
  h1 : total_nails > 0
  h2 : nails_per_side > 0
  h3 : total_nails = 4 * nails_per_side

/-- Theorem: For a square plate with 96 nails evenly distributed along its edges,
    there are 24 nails on each side -/
theorem nails_on_square_plate :
  ∀ (plate : SquarePlate), plate.total_nails = 96 → plate.nails_per_side = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_nails_on_square_plate_l3914_391453


namespace NUMINAMATH_CALUDE_calculate_train_speed_goods_train_speed_l3914_391469

/-- Calculates the speed of a train given the speed of another train traveling in the opposite direction, the length of the train, and the time it takes to pass. -/
theorem calculate_train_speed (speed_a : ℝ) (length_b : ℝ) (pass_time : ℝ) : ℝ :=
  let speed_a_ms := speed_a * 1000 / 3600
  let relative_speed := length_b / pass_time
  let speed_b_ms := relative_speed - speed_a_ms
  let speed_b_kmh := speed_b_ms * 3600 / 1000
  speed_b_kmh

/-- Proves that given a train A traveling at 50 km/h and a goods train B of length 280 m passing train A in the opposite direction in 9 seconds, the speed of train B is approximately 62 km/h. -/
theorem goods_train_speed : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |calculate_train_speed 50 280 9 - 62| < ε :=
sorry

end NUMINAMATH_CALUDE_calculate_train_speed_goods_train_speed_l3914_391469


namespace NUMINAMATH_CALUDE_count_solutions_equation_l3914_391480

theorem count_solutions_equation : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 500) / 50 = ⌊Real.sqrt (2 * n)⌋) ∧ 
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_equation_l3914_391480


namespace NUMINAMATH_CALUDE_water_depth_conversion_l3914_391460

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : Real) : Real :=
  sorry

/-- Calculates the depth of water when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) (horizontalDepth : Real) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_conversion (tank : WaterTank) (horizontalDepth : Real) :
  tank.height = 10 ∧ tank.baseDiameter = 6 ∧ horizontalDepth = 4 →
  verticalWaterDepth tank horizontalDepth = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_conversion_l3914_391460


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l3914_391425

/-- The perimeter of a triangle with average side length 12 is 36 -/
theorem triangle_perimeter_from_average_side_length :
  ∀ (a b c : ℝ), 
  (a + b + c) / 3 = 12 →
  a + b + c = 36 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_average_side_length_l3914_391425


namespace NUMINAMATH_CALUDE_three_tangent_planes_l3914_391423

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents an equilateral triangle in 3D space -/
structure EquilateralTriangle where
  vertices : List (ℝ × ℝ × ℝ)
  side_length : ℝ

/-- Configuration of three spheres whose centers form an equilateral triangle -/
structure SphereConfiguration where
  spheres : List Sphere
  triangle : EquilateralTriangle

/-- Returns the number of planes tangent to all spheres in the configuration -/
def count_tangent_planes (config : SphereConfiguration) : ℕ :=
  sorry

theorem three_tangent_planes (config : SphereConfiguration) :
  (config.spheres.length = 3) →
  (config.triangle.side_length = 11) →
  (config.spheres.map Sphere.radius = [3, 4, 6]) →
  (count_tangent_planes config = 3) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_planes_l3914_391423


namespace NUMINAMATH_CALUDE_train_crossing_time_l3914_391435

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 500 ∧ 
  train_speed = 63 * 1000 / 3600 ∧ 
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3914_391435


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l3914_391486

theorem divisibility_of_expression : ∃ k : ℤ, 27195^8 - 10887^8 + 10152^8 = 26460 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l3914_391486


namespace NUMINAMATH_CALUDE_only_math_scores_need_census_l3914_391498

-- Define the survey types
inductive SurveyType
  | Sampling
  | Census

-- Define the survey options
inductive SurveyOption
  | WeeklyAllowance
  | MathTestScores
  | TVWatchTime
  | ExtracurricularReading

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.MathTestScores => SurveyType.Census
  | _ => SurveyType.Sampling

-- Theorem stating that only the MathTestScores option requires a census
theorem only_math_scores_need_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.MathTestScores :=
by sorry


end NUMINAMATH_CALUDE_only_math_scores_need_census_l3914_391498


namespace NUMINAMATH_CALUDE_solution_pairs_l3914_391433

theorem solution_pairs (x y : ℝ) : 
  (|x + y| = 3 ∧ x * y = -10) → 
  ((x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3914_391433


namespace NUMINAMATH_CALUDE_hayden_ironing_weeks_l3914_391478

/-- Calculates the number of weeks Hayden spends ironing given his daily routine and total ironing time. -/
def ironingWeeks (shirtTime minutesPerDay weekDays totalMinutes : ℕ) : ℕ :=
  totalMinutes / (shirtTime + minutesPerDay) / weekDays

/-- Proves that Hayden spends 4 weeks ironing given his routine and total ironing time. -/
theorem hayden_ironing_weeks :
  ironingWeeks 5 3 5 160 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_weeks_l3914_391478


namespace NUMINAMATH_CALUDE_no_solutions_iff_b_geq_neg_four_thirds_l3914_391462

/-- The equation has no solutions for a > 1 iff b ≥ -4/3 -/
theorem no_solutions_iff_b_geq_neg_four_thirds (b : ℝ) :
  (∀ a x : ℝ, a > 1 → a^(2 - 2*x^2) + (b + 4)*a^(1 - x^2) + 3*b + 4 ≠ 0) ↔
  b ≥ -4/3 :=
sorry

end NUMINAMATH_CALUDE_no_solutions_iff_b_geq_neg_four_thirds_l3914_391462


namespace NUMINAMATH_CALUDE_roots_of_equation_l3914_391465

theorem roots_of_equation : 
  ∀ x : ℝ, (x^3 - 2*x^2 - x + 2)*(x - 5) = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3914_391465


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3914_391477

def euler_family_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem euler_family_mean_age :
  mean euler_family_ages = 13.71 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3914_391477


namespace NUMINAMATH_CALUDE_physics_score_l3914_391432

/-- Represents the scores in physics, chemistry, and mathematics --/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The average score of all three subjects is 65 --/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 65

/-- The average score of physics and mathematics is 90 --/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 --/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Given the conditions, prove that the score in physics is 125 --/
theorem physics_score (s : Scores) 
  (h1 : average_all s) 
  (h2 : average_physics_math s) 
  (h3 : average_physics_chem s) : 
  s.physics = 125 := by
  sorry

end NUMINAMATH_CALUDE_physics_score_l3914_391432


namespace NUMINAMATH_CALUDE_work_completion_time_l3914_391464

/-- Given workers a and b, where:
    - a can complete the work in 20 days
    - a and b together can complete the work in 15 days when b works half-time
    Prove that a and b together can complete the work in 12 days when b works full-time -/
theorem work_completion_time (a b : ℝ) 
  (ha : a = 1 / 20)  -- a's work rate per day
  (hab_half : a + b / 2 = 1 / 15)  -- combined work rate when b works half-time
  : a + b = 1 / 12 := by  -- combined work rate when b works full-time
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3914_391464


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3914_391400

theorem rhombus_side_length 
  (diag1 diag2 : ℝ) 
  (m : ℝ) 
  (h1 : diag1^2 - 10*diag1 + m = 0)
  (h2 : diag2^2 - 10*diag2 + m = 0)
  (h3 : diag1 * diag2 / 2 = 11) :
  ∃ (side : ℝ), side^2 = 14 ∧ 
    side = Real.sqrt ((diag1/2)^2 + (diag2/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3914_391400


namespace NUMINAMATH_CALUDE_eighty_one_to_negative_two_to_negative_two_equals_three_l3914_391426

theorem eighty_one_to_negative_two_to_negative_two_equals_three :
  (81 : ℝ) ^ (-(2 : ℝ)^(-(2 : ℝ))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_eighty_one_to_negative_two_to_negative_two_equals_three_l3914_391426


namespace NUMINAMATH_CALUDE_karls_clothing_store_l3914_391427

/-- Karl's clothing store problem -/
theorem karls_clothing_store (tshirt_price : ℝ) (pants_price : ℝ) (skirt_price : ℝ) :
  tshirt_price = 5 →
  pants_price = 4 →
  (2 * tshirt_price + pants_price + 4 * skirt_price + 6 * (tshirt_price / 2) = 53) →
  skirt_price = 6 := by
sorry

end NUMINAMATH_CALUDE_karls_clothing_store_l3914_391427


namespace NUMINAMATH_CALUDE_distance_P_to_xaxis_l3914_391470

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- The point P -/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 -/
theorem distance_P_to_xaxis : distanceToXAxis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_xaxis_l3914_391470


namespace NUMINAMATH_CALUDE_option_A_is_incorrect_l3914_391493

-- Define the set of angles whose terminal sides lie on y=x
def AnglesOnYEqualsX : Set ℝ := {β | ∃ n : ℤ, β = 45 + n * 180}

-- Define the set given in option A
def OptionASet : Set ℝ := {β | ∃ k : ℤ, β = 45 + k * 360 ∨ β = -45 + k * 360}

-- Theorem statement
theorem option_A_is_incorrect : OptionASet ≠ AnglesOnYEqualsX := by
  sorry

end NUMINAMATH_CALUDE_option_A_is_incorrect_l3914_391493


namespace NUMINAMATH_CALUDE_min_y_max_x_l3914_391499

theorem min_y_max_x (x y : ℝ) (h : x^2 + y^2 = 18*x + 40*y) : 
  (∀ y' : ℝ, x^2 + y'^2 = 18*x + 40*y' → y ≤ y') ∧ 
  (∀ x' : ℝ, x'^2 + y^2 = 18*x' + 40*y → x' ≤ x) → 
  y = 20 - Real.sqrt 481 ∧ x = 9 + Real.sqrt 481 :=
by sorry

end NUMINAMATH_CALUDE_min_y_max_x_l3914_391499


namespace NUMINAMATH_CALUDE_sum_of_absolute_values_l3914_391455

def S (n : ℕ+) : ℤ := n^2 + 6*n + 1

def a (n : ℕ+) : ℤ := S n - S (n-1)

theorem sum_of_absolute_values : |a 1| + |a 2| + |a 3| + |a 4| = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_values_l3914_391455


namespace NUMINAMATH_CALUDE_brooke_math_problems_l3914_391475

/-- The number of math problems Brooke has -/
def num_math_problems : ℕ := sorry

/-- The number of social studies problems Brooke has -/
def num_social_studies_problems : ℕ := 6

/-- The number of science problems Brooke has -/
def num_science_problems : ℕ := 10

/-- The time (in minutes) it takes to solve one math problem -/
def time_per_math_problem : ℚ := 2

/-- The time (in minutes) it takes to solve one social studies problem -/
def time_per_social_studies_problem : ℚ := 1/2

/-- The time (in minutes) it takes to solve one science problem -/
def time_per_science_problem : ℚ := 3/2

/-- The total time (in minutes) it takes Brooke to complete all homework -/
def total_homework_time : ℚ := 48

theorem brooke_math_problems : 
  num_math_problems = 15 ∧
  (num_math_problems : ℚ) * time_per_math_problem + 
  (num_social_studies_problems : ℚ) * time_per_social_studies_problem +
  (num_science_problems : ℚ) * time_per_science_problem = total_homework_time :=
sorry

end NUMINAMATH_CALUDE_brooke_math_problems_l3914_391475


namespace NUMINAMATH_CALUDE_not_always_separate_triangles_l3914_391422

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Check if a point is inside or on the boundary of a triangle -/
def Point2D.inTriangle (p : Point2D) (t : Triangle) : Prop :=
  sorry

/-- Check if two triangles have a common point -/
def Triangle.haveCommonPoint (t1 t2 : Triangle) : Prop :=
  ∃ p : Point2D, p.inTriangle t1 ∧ p.inTriangle t2

/-- A configuration of six points -/
structure SixPointConfig where
  points : Fin 6 → Point2D

/-- A division of six points into two triangles -/
structure TriangleDivision where
  config : SixPointConfig
  t1 : Triangle
  t2 : Triangle
  valid : 
    (∃ i j k l m n : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ l ≠ m ∧ m ≠ n ∧ l ≠ n ∧
      i ≠ l ∧ i ≠ m ∧ i ≠ n ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧ k ≠ l ∧ k ≠ m ∧ k ≠ n ∧
      t1 = ⟨config.points i, config.points j, config.points k⟩ ∧
      t2 = ⟨config.points l, config.points m, config.points n⟩)

theorem not_always_separate_triangles : 
  ∃ c : SixPointConfig, ∀ d : TriangleDivision, d.config = c → d.t1.haveCommonPoint d.t2 :=
sorry

end NUMINAMATH_CALUDE_not_always_separate_triangles_l3914_391422


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l3914_391491

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧ 
  1000000 ≤ b ∧ b < 10000000 ∧ 
  10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000 →
  Nat.gcd a b < 10000 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l3914_391491


namespace NUMINAMATH_CALUDE_max_M_value_l3914_391481

def J (k : ℕ) : ℕ := 10^(k+2) + 100

def M (k : ℕ) : ℕ := (J k).factorization 2

theorem max_M_value :
  ∃ (k : ℕ), k > 0 ∧ M k = 4 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l3914_391481


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l3914_391485

-- Define the function f(x) = x^2 - ax + 4
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Part 1: Range of f(x) on [1, 3] when a = 3
theorem range_of_f_on_interval (x : ℝ) (h : x ∈ Set.Icc 1 3) :
  ∃ y ∈ Set.Icc (7/4) 4, y = f 3 x :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Icc 0 2, f a x ≤ 4) :
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l3914_391485
