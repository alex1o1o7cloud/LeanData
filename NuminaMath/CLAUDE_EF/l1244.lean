import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_theory_theorem_l1244_124424

theorem number_theory_theorem (n : ℕ) (S : Finset ℕ) 
  (h1 : S ⊆ Finset.range (2 * n + 1)) 
  (h2 : S.card = n + 1) :
  (∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ Nat.gcd x y = 1) ∧
  (∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_theory_theorem_l1244_124424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_for_geometric_sequence_coefficients_l1244_124456

/-- a, b, and c form a geometric sequence -/
def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

/-- Quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

/-- Number of real roots of a quadratic function -/
noncomputable def num_real_roots (a b c : ℝ) : ℕ :=
  if (b^2 - 4*a*c < 0) then 0
  else if (b^2 - 4*a*c = 0) then 1
  else 2

theorem no_real_roots_for_geometric_sequence_coefficients 
  (a b c : ℝ) (ha : a ≠ 0) :
  is_geometric_sequence a b c → num_real_roots a b c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_for_geometric_sequence_coefficients_l1244_124456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carlson_jam_consumption_l1244_124441

/-- Represents the eating speed ratio between Carlson and Junior -/
noncomputable def speed_ratio : ℝ := 3

/-- Represents the total amount of jam -/
noncomputable def total_jam : ℝ := 1

/-- Represents the amount of cookies each person ate -/
noncomputable def cookies_per_person : ℝ := 1/2

/-- Carlson's jam consumption -/
noncomputable def carlson_jam : ℝ := 9/10

theorem carlson_jam_consumption :
  speed_ratio = 3 →
  cookies_per_person * 2 = 1 →
  carlson_jam = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carlson_jam_consumption_l1244_124441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1244_124412

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp x

theorem problem_solution :
  ∀ a b c : ℝ,
  (f a b c 0 = 1) →
  (f a b c 1 = 0) →
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x ≤ y → f a b c x ≥ f a b c y) →
  (a ∈ Set.Icc 0 1) ∧
  (a = 0 →
    ∃ m : ℝ, m = 4 ∧
    ∀ x : ℝ, 2 * (f a b c x) + 4 * x * Real.exp x ≥ m * x + 1 ∧
              m * x + 1 ≥ -x^2 + 4 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1244_124412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_explicit_l1244_124437

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

-- Define the centers and radii
def center1 : ℝ × ℝ := (-1, -4)
def center2 : ℝ × ℝ := (2, 2)
def radius1 : ℝ := 5
def radius2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 45

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > abs (radius1 - radius2) ∧
  distance_between_centers < radius1 + radius2 := by
  sorry

-- Additional theorem to show the circles are intersecting
theorem circles_intersect_explicit :
  3 * Real.sqrt 5 > 2 ∧ 3 * Real.sqrt 5 < 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_intersect_explicit_l1244_124437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_l1244_124439

-- Define the cans and their properties
structure Can where
  radius : ℝ
  height : ℝ

-- Define the problem parameters
def problem_setup (can_b can_c : Can) (fill_cost_half_b fill_cost_full_c : ℝ) : Prop :=
  can_c.height = 1/2 * can_b.height ∧
  fill_cost_half_b = 4 ∧
  fill_cost_full_c = 16

-- Define the volume of a cylinder
noncomputable def cylinder_volume (can : Can) : ℝ :=
  Real.pi * can.radius^2 * can.height

-- Theorem stating the ratio of radii
theorem radius_ratio (can_b can_c : Can) (fill_cost_half_b fill_cost_full_c : ℝ) 
  (h : problem_setup can_b can_c fill_cost_half_b fill_cost_full_c) : 
  can_c.radius / can_b.radius = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_ratio_l1244_124439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_subsegment_length_l1244_124427

-- Define the triangle DEF
def Triangle (D E F : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), 
    dist D F = 3 * x ∧ 
    dist E F = 4 * x ∧ 
    dist D E = 5 * x ∧ 
    dist D E = 15

-- Define the angle bisector
def AngleBisector (D E F G : ℝ × ℝ) : Prop :=
  dist D G / dist G F = dist D E / dist E F

-- Main theorem
theorem longest_subsegment_length 
  (D E F G : ℝ × ℝ) 
  (h_triangle : Triangle D E F) 
  (h_bisector : AngleBisector D E F G) :
  max (dist D G) (dist G E) = 60 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_subsegment_length_l1244_124427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_ranges_l1244_124457

open Real

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the condition that the triangle is acute
def AcuteTriangle (A B C : ℝ) : Prop :=
  Triangle A B C ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- Define the condition that all angles are unequal
def UnequalAngles (A B C : ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C

-- Define the relationship between sides and angles
def SideLengthRelation (a b : ℝ) (A : ℝ) : Prop :=
  b = 2 * a * Real.cos A

-- Define the function f(A)
noncomputable def f (A : ℝ) : ℝ :=
  2 * Real.cos A * (Real.sqrt 3 * Real.cos A + Real.sin A)

-- Theorem statement
theorem triangle_angle_and_function_ranges
  (A B C a b : ℝ)
  (h_acute : AcuteTriangle A B C)
  (h_unequal : UnequalAngles A B C)
  (h_relation : SideLengthRelation a b A) :
  (∃ x, 0 < x ∧ x < Real.pi/4 ∧ A = x ∨ Real.pi/4 < x ∧ x < Real.pi/2 ∧ A = x) ∧
  (∃ y, 0 < y ∧ y < 1 + Real.sqrt 3 ∧ f A = y ∨
        1 + Real.sqrt 3 < y ∧ y ≤ 2 + Real.sqrt 3 ∧ f A = y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_ranges_l1244_124457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1244_124499

/-- Represents a side of a triangle -/
inductive Side
  | AB
  | BC
  | CA

/-- Represents a vertex of a triangle -/
inductive Vertex
  | A
  | B
  | C

/-- Represents a triangular arrangement as described in the problem -/
structure TriangularArrangement where
  /-- The large equilateral triangle -/
  large_triangle : Unit
  /-- Each side of the large triangle is subdivided into 3 equal segments -/
  side_subdivision : Side → Nat
  /-- Each vertex is connected to the midpoint of the opposite side -/
  vertex_to_midpoint_lines : Vertex → Unit

/-- Counts the number of triangles in the triangular arrangement -/
def count_triangles (arrangement : TriangularArrangement) : Nat :=
  sorry

/-- The main theorem stating that the number of triangles in the arrangement is 23 -/
theorem triangle_count (arrangement : TriangularArrangement) : 
  count_triangles arrangement = 23 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_l1244_124499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_gt_gf_iff_neg_l1244_124449

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x else 0

noncomputable def g (x : ℝ) : ℝ := if x < 2 then 2 - x else 0

theorem fg_gt_gf_iff_neg (x : ℝ) : f (g x) > g (f x) ↔ x < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fg_gt_gf_iff_neg_l1244_124449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_equality_l1244_124486

/-- A random variable following a normal distribution with mean 4 and variance 5 -/
def ξ : Real → Real := sorry

/-- The probability density function of the normal distribution N(4,5) -/
noncomputable def normal_pdf (x : Real) : Real :=
  (1 / (Real.sqrt (2 * Real.pi * 5))) * Real.exp (-((x - 4)^2) / (2 * 5))

/-- The cumulative distribution function of the normal distribution N(4,5) -/
noncomputable def normal_cdf (x : Real) : Real :=
  ∫ y in Set.Iio x, normal_pdf y

/-- The statement that P(ξ < 2a-3) = P(ξ > a + 2) for the given normal distribution -/
theorem normal_distribution_equality (a : Real) : 
  normal_cdf (2*a - 3) = 1 - normal_cdf (a + 2) → a = 3 := by
  sorry

#check normal_distribution_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_equality_l1244_124486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_one_half_l1244_124448

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a) ^ 2)

/-- Theorem: For an ellipse with equation x²/(m+1) + y²/m = 1 and eccentricity 1/2, m = 3 -/
theorem ellipse_eccentricity_one_half (m : ℝ) 
  (h_pos_m : 0 < m) 
  (h_pos_m_plus_one : 0 < m + 1) 
  (e : Ellipse) 
  (h_eq : e = { a := Real.sqrt (m + 1), b := Real.sqrt m, h_pos_a := sorry, h_pos_b := sorry }) 
  (h_ecc : eccentricity e = 1/2) : 
  m = 3 := by
  sorry

#check ellipse_eccentricity_one_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_one_half_l1244_124448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l1244_124418

/-- The distance from a point to a plane --/
noncomputable def distance_point_to_plane (normal : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : ℝ :=
  let (nx, ny, nz) := normal
  let (px, py, pz) := point
  abs (nx * px + ny * py + nz * pz) / Real.sqrt (nx^2 + ny^2 + nz^2)

/-- The problem statement --/
theorem distance_to_plane :
  let normal : ℝ × ℝ × ℝ := (2, -2, 1)
  let point : ℝ × ℝ × ℝ := (-1, 3, 2)
  distance_point_to_plane normal point = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l1244_124418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1244_124485

def repeating_decimal (whole : ℚ) (repeating : ℚ) : ℚ :=
  whole + repeating / (10^(Nat.log 10 (Nat.floor repeating + 1)) - 1)

theorem sum_of_repeating_decimals :
  repeating_decimal 0 (6/10) + repeating_decimal 0 (2/10) - 
  repeating_decimal 0 (4/10) + repeating_decimal 0 (9/10) = 13 / 9 := by
  sorry

#eval repeating_decimal 0 (6/10) + repeating_decimal 0 (2/10) - 
      repeating_decimal 0 (4/10) + repeating_decimal 0 (9/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1244_124485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_can_achieve_goal_l1244_124466

/-- Represents the state of a player's coins -/
structure CoinState where
  gold : ℕ
  silver : ℕ

/-- Represents a bet placed by the player -/
structure Bet where
  red_gold : ℕ
  red_silver : ℕ
  black_gold : ℕ
  black_silver : ℕ

/-- Represents the possible outcomes of a round -/
inductive Outcome
| Red
| Black

/-- Determines if a CoinState satisfies the player's goal -/
def goal_achieved (state : CoinState) : Prop :=
  state.gold = 3 * state.silver ∨ state.silver = 3 * state.gold

/-- Applies the result of a round to a CoinState given a Bet and an Outcome -/
def apply_round (state : CoinState) (bet : Bet) (outcome : Outcome) : CoinState :=
  match outcome with
  | Outcome.Red => { 
      gold := state.gold + bet.red_gold - bet.black_gold,
      silver := state.silver + bet.red_silver - bet.black_silver 
    }
  | Outcome.Black => { 
      gold := state.gold - bet.red_gold + bet.black_gold,
      silver := state.silver - bet.red_silver + bet.black_silver 
    }

/-- The main theorem to be proved -/
theorem player_can_achieve_goal (m n : ℕ) (h : (m + n) % 4 = 0) :
  ∃ (sequence : List Bet), ∃ (outcomes : List Outcome),
    goal_achieved (List.foldl (λ state (bet, outcome) => apply_round state bet outcome) 
                         { gold := m, silver := n } 
                         (List.zip sequence outcomes)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_can_achieve_goal_l1244_124466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l1244_124432

theorem m_greater_than_n (x y : ℝ) : 
  (x^2 + y^2 + 1) > (2*x + 2*y - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l1244_124432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_cost_is_two_l1244_124492

/-- Represents the sale of brownies at a bake sale -/
structure BrownieSale where
  revenue : ℚ
  num_pans : ℕ
  pieces_per_pan : ℕ
  all_sold : Bool

/-- Calculates the cost per brownie piece -/
def cost_per_piece (sale : BrownieSale) : ℚ :=
  sale.revenue / (sale.num_pans * sale.pieces_per_pan)

/-- Theorem stating that for the given conditions, each brownie piece costs $2 -/
theorem brownie_cost_is_two :
  ∀ (sale : BrownieSale),
    sale.revenue = 32 ∧
    sale.num_pans = 2 ∧
    sale.pieces_per_pan = 8 ∧
    sale.all_sold = true →
    cost_per_piece sale = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brownie_cost_is_two_l1244_124492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_proof_l1244_124484

/-- Proves that the breadth of a rectangle is approximately 1.22 cm given specific conditions -/
theorem rectangle_breadth_proof (s : ℝ) (b : ℝ) : 
  (4 * s = 2 * (22 + b)) →  -- Perimeter of square equals perimeter of rectangle
  (Real.pi * s / 2 + s = 29.85) →  -- Circumference of semicircle
  (abs (b - 1.22) < 0.01) :=  -- Breadth is approximately 1.22 cm (within 0.01)
by
  sorry

#check rectangle_breadth_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_breadth_proof_l1244_124484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_alpha_l1244_124404

theorem sin_2alpha_plus_cos_alpha (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_cos_alpha_l1244_124404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_915_l1244_124401

noncomputable section

/-- The angle of the minute hand from 12 o'clock at 15 minutes past the hour -/
def minute_hand_angle : ℝ := 15 / 60 * 360

/-- The angle of the hour hand from 12 o'clock at 9:00 -/
def hour_hand_angle_9 : ℝ := 9 * 30

/-- The additional angle the hour hand moves in 15 minutes -/
def hour_hand_additional_angle : ℝ := 15 / 60 * 30

/-- The total angle of the hour hand from 12 o'clock at 9:15 -/
def hour_hand_angle_915 : ℝ := hour_hand_angle_9 + hour_hand_additional_angle

/-- The absolute difference between the hour and minute hand angles -/
def angle_difference : ℝ := |hour_hand_angle_915 - minute_hand_angle|

/-- The smaller angle between the hour and minute hands at 9:15 p.m. -/
def smaller_angle : ℝ := min angle_difference (360 - angle_difference)

theorem clock_angle_915 : smaller_angle = 187.5 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_915_l1244_124401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1244_124473

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.sqrt 0.4) 
  (hb : b = (2 : ℝ)^(0.4 : ℝ)) 
  (hc : c = (0.4 : ℝ)^(0.2 : ℝ)) : 
  b > c ∧ c > a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1244_124473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l1244_124417

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- Calculate the first trial point x₁ --/
noncomputable def x₁ (a b : ℝ) : ℝ := a + φ * (b - a)

/-- Calculate the second trial point x₂ --/
noncomputable def x₂ (a b : ℝ) : ℝ := a + (b - x₁ a b)

/-- Calculate the third trial point x₃ --/
noncomputable def x₃ (a b : ℝ) : ℝ := b - φ * (b - x₂ a b)

/-- Round a real number to three decimal places --/
noncomputable def round_to_3dp (x : ℝ) : ℝ := (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

theorem golden_section_search_third_point :
  let a := (2 : ℝ)
  let b := (4 : ℝ)
  round_to_3dp (x₃ a b) = 3.236 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_search_third_point_l1244_124417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l1244_124479

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  AC = 15 ∧ BC = 36 ∧ AB^2 = AC^2 + BC^2

-- Define the altitude CD
def altitude_CD (C D : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (D.2 - A.2) * (B.1 - A.1) = (B.2 - A.2) * (D.1 - A.1)

-- Define the circle ω
def circle_ω (ω : Set (ℝ × ℝ)) (C D : ℝ × ℝ) : Prop :=
  ∀ P : ℝ × ℝ, P ∈ ω ↔ (P.1 - C.1)^2 + (P.2 - C.2)^2 = ((D.1 - C.1)^2 + (D.2 - C.2)^2) / 4

-- Define the point I and tangent lines
def point_I_and_tangents (I : ℝ × ℝ) (A B : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  ∀ P : ℝ × ℝ, P ∈ ω → 
    ((P.1 - A.1) * (I.1 - A.1) + (P.2 - A.2) * (I.2 - A.2))^2 = 
      ((I.1 - A.1)^2 + (I.2 - A.2)^2) * ((P.1 - A.1)^2 + (P.2 - A.2)^2) ∧
    ((P.1 - B.1) * (I.1 - B.1) + (P.2 - B.2) * (I.2 - B.2))^2 = 
      ((I.1 - B.1)^2 + (I.2 - B.2)^2) * ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Main theorem
theorem perimeter_ratio 
  (A B C D I : ℝ × ℝ) 
  (ω : Set (ℝ × ℝ)) 
  (h1 : triangle_ABC A B C)
  (h2 : altitude_CD C D A B)
  (h3 : circle_ω ω C D)
  (h4 : point_I_and_tangents I A B ω) :
  let perimeter_ABI := Real.sqrt ((A.1 - I.1)^2 + (A.2 - I.2)^2) + 
                       Real.sqrt ((B.1 - I.1)^2 + (B.2 - I.2)^2) + 
                       Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  perimeter_ABI / AB = 23 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ratio_l1244_124479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_and_efficiency_l1244_124419

-- Define the processing cost function
noncomputable def y (x : ℝ) : ℝ :=
  if 120 ≤ x ∧ x < 144 then (1/3) * x^3 - 80 * x^2 + 5040 * x
  else if 144 ≤ x ∧ x < 500 then (1/2) * x^2 - 200 * x + 80000
  else 0

-- Define the profit function
noncomputable def S (x : ℝ) : ℝ := 200 * x - y x

-- Define the average processing cost function
noncomputable def avg_cost (x : ℝ) : ℝ := y x / x

theorem project_profitability_and_efficiency :
  (∀ x ∈ Set.Icc 200 300, S x < 0) ∧
  (∃ x ∈ Set.Icc 200 300, ∀ y ∈ Set.Icc 200 300, S x ≤ S y) ∧
  (S 300 = -5000) ∧
  (∀ x ∈ Set.Ioo 120 500, avg_cost 400 ≤ avg_cost x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_profitability_and_efficiency_l1244_124419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l1244_124471

theorem complex_simplification :
  (4 + 2*Complex.I) / (1 - Complex.I) = 1 + 3*Complex.I := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_simplification_l1244_124471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_interval_calculation_l1244_124452

/-- Given birth and death rates per interval and daily net population increase,
    calculate the time interval in days. -/
theorem time_interval_calculation (birth_rate death_rate net_increase : ℚ) :
  birth_rate = 7 →
  death_rate = 3 →
  net_increase = 172800 →
  (birth_rate - death_rate) * (1 / (birth_rate - death_rate) * net_increase) = net_increase →
  1 / ((birth_rate - death_rate) * net_increase) = 1 / 43200 := by
  sorry

#eval (1 : ℚ) / 43200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_interval_calculation_l1244_124452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1244_124445

/-- Calculates the annual interest rate given the principal, time period, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time_months : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * (time_months / 12))

theorem interest_rate_calculation (principal time_months simple_interest : ℝ) 
  (h1 : principal = 68800)
  (h2 : time_months = 9)
  (h3 : simple_interest = 8625) :
  ⌊calculate_interest_rate principal time_months simple_interest⌋ = 16 ∧ 
  |calculate_interest_rate principal time_months simple_interest - 16.71| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1244_124445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_isosceles_triangle_l1244_124403

-- Define the three lines
noncomputable def line1 (x : ℝ) : ℝ := 4 * x + 3
noncomputable def line2 (x : ℝ) : ℝ := -4 * x + 3
noncomputable def line3 : ℝ := -3

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (0, 3)
noncomputable def point2 : ℝ × ℝ := (-3/2, -3)
noncomputable def point3 : ℝ × ℝ := (3/2, -3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_forms_isosceles_triangle :
  let d12 := distance point1 point2
  let d13 := distance point1 point3
  let d23 := distance point2 point3
  d12 = d13 ∧ d12 ≠ d23 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_isosceles_triangle_l1244_124403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1244_124495

/-- The time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_speed : ℝ) (train_length : ℝ) (platform_length : ℝ) : ℝ :=
  (train_length + platform_length) / (train_speed * (5/18))

/-- Theorem: A train with speed 72 km/hr and length 370 m takes 26 seconds to cross a 150 m long platform -/
theorem train_crossing_theorem :
  train_crossing_time 72 370 150 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1244_124495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_nine_small_to_one_large_triangle_is_one_l1244_124416

/-- The ratio of areas between nine small equilateral triangles and one large equilateral triangle formed with the same perimeter --/
def area_ratio_nine_small_to_one_large_triangle (small_side : ℝ) (small_perimeter : ℝ) : Prop :=
  small_side > 0 ∧
  small_perimeter = 3 * small_side ∧
  let large_perimeter : ℝ := 9 * small_perimeter
  let large_side : ℝ := large_perimeter / 3
  let small_area : ℝ := (Real.sqrt 3 / 4) * small_side ^ 2
  let large_area : ℝ := (Real.sqrt 3 / 4) * large_side ^ 2
  9 * small_area = large_area

/-- Proof of the area ratio theorem --/
theorem area_ratio_nine_small_to_one_large_triangle_is_one :
  ∀ (small_side : ℝ) (small_perimeter : ℝ),
    area_ratio_nine_small_to_one_large_triangle small_side small_perimeter :=
by
  sorry

#check area_ratio_nine_small_to_one_large_triangle
#check area_ratio_nine_small_to_one_large_triangle_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_nine_small_to_one_large_triangle_is_one_l1244_124416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l1244_124435

/-- Calculates the percentage increase between two amounts -/
noncomputable def percentageIncrease (originalAmount newAmount : ℝ) : ℝ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem salary_increase_percentage :
  let originalSalary : ℝ := 60
  let newSalary : ℝ := 90
  percentageIncrease originalSalary newSalary = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l1244_124435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1244_124475

/-- A rectangle with sides divided into segments -/
structure DividedRectangle where
  length : ℝ
  width : ℝ
  n : ℕ+
  m : ℕ+

/-- Triangle formed by connecting center to segment endpoints -/
inductive Triangle
  | A : Triangle
  | B : Triangle

/-- Area of a triangle in the divided rectangle -/
noncomputable def triangleArea (r : DividedRectangle) (t : Triangle) : ℝ :=
  match t with
  | Triangle.A => (r.length * r.width) / (4 * r.n)
  | Triangle.B => (r.length * r.width) / (4 * r.m)

/-- Theorem stating the ratio of areas of triangles A and B -/
theorem triangle_area_ratio (r : DividedRectangle) :
  triangleArea r Triangle.A / triangleArea r Triangle.B = r.m / r.n := by
  sorry

#check triangle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l1244_124475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l1244_124429

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 3 * Real.sin x

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → f x ≤ max) ∧
    (∃ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ f x = max) ∧
    (∀ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) → min ≤ f x) ∧
    (∃ x, x ∈ Set.Icc (-Real.pi/2) (Real.pi/2) ∧ f x = min) ∧
    max = 25/8 ∧ min = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l1244_124429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_plastering_cost_per_sq_m_l1244_124477

/-- Calculates the cost per square meter for plastering a tank -/
theorem tank_plastering_cost_per_sq_m 
  (length width depth : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 25)
  (h2 : width = 12)
  (h3 : depth = 6)
  (h4 : total_cost = 558) :
  total_cost / (2 * (length * depth + width * depth) + length * width) = 0.75 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_plastering_cost_per_sq_m_l1244_124477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l1244_124474

/-- Represents the train's journey with an accident -/
structure TrainJourney where
  speed : ℝ  -- Original speed of the train in miles per hour
  distance : ℝ  -- Total distance of the trip in miles

/-- The actual journey time given the accident -/
noncomputable def actual_time (j : TrainJourney) : ℝ :=
  2 + 1 + (3 * (j.distance - 2 * j.speed)) / (2 * j.speed)

/-- The journey time if the accident occurred 60 miles further -/
noncomputable def alternative_time (j : TrainJourney) : ℝ :=
  (2 * j.speed + 60) / j.speed + 1 + (3 * (j.distance - 2 * j.speed - 60)) / (2 * j.speed)

/-- The theorem stating the conditions and the conclusion -/
theorem train_journey_distance : 
  ∃ (j : TrainJourney), 
    j.speed > 0 ∧ 
    actual_time j = alternative_time j + 0.5 ∧ 
    j.distance = 720 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_journey_distance_l1244_124474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_l₀_l1244_124408

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y + 1 = 0
def l₂ (x y : ℝ) : Prop := x - 2 * y - 5 = 0
def l₀ (x y : ℝ) : Prop := y = -3/4 * x - 5/2

-- Define point A as the intersection of l₁ and l₂
def A : ℝ × ℝ := (1, -2)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * p.1 + b * p.2 + c)) / (Real.sqrt (a^2 + b^2))

theorem distance_A_to_l₀ :
  distance_point_to_line A (3/4) 1 (5/2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_l₀_l1244_124408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1244_124450

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the right branch of a hyperbola -/
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x > 0 ∧ p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a point on the right branch of the given hyperbola,
    if its distance to the right focus is 4, then its distance to the left focus is 10 -/
theorem hyperbola_focus_distance 
  (h : Hyperbola) 
  (p rightFocus leftFocus : Point) 
  (h_eq : h.a = 3 ∧ h.b = 4) 
  (h_right : isOnRightBranch h p) 
  (h_dist : distance p rightFocus = 4) : 
  distance p leftFocus = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l1244_124450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strategy_always_works_l1244_124400

/-- Represents a point on a circle --/
structure CirclePoint where
  angle : ℝ
  is_valid : 0 ≤ angle ∧ angle < 2 * π

/-- Represents a set of points on a circle --/
def CirclePoints := Finset CirclePoint

/-- Represents a semicircle on the circle --/
structure Semicircle where
  start_angle : ℝ
  is_valid : 0 ≤ start_angle ∧ start_angle < 2 * π

/-- The strategy function that determines which point to remove --/
noncomputable def remove_strategy (points : CirclePoints) : CirclePoint :=
  sorry

/-- The strategy function that determines which semicircle to choose --/
noncomputable def semicircle_strategy (points : CirclePoints) : Semicircle :=
  sorry

/-- The main theorem stating that the strategy always works --/
theorem strategy_always_works (points : CirclePoints) 
  (h : points.card = 2007) :
  ∀ (removed : CirclePoint), removed ∈ points.toSet →
  ∃ (semi : Semicircle), 
    removed.angle ≥ semi.start_angle ∧ 
    removed.angle < semi.start_angle + π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strategy_always_works_l1244_124400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_triangle_l1244_124490

noncomputable def triangle_area (s : ℝ) : ℝ := s^2 * Real.sqrt 3 / 4

noncomputable def hexagon_area (s : ℝ) : ℝ := 3 * s^2 * Real.sqrt 3 / 2

def triangle_perimeter (s : ℝ) : ℝ := 3 * s

def hexagon_perimeter (s : ℝ) : ℝ := 6 * s

theorem hexagon_area_given_triangle (s t : ℝ) 
  (h1 : triangle_area s = 9)
  (h2 : hexagon_perimeter t = 2 * triangle_perimeter s) :
  hexagon_area t = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_given_triangle_l1244_124490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l1244_124405

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem largest_distance_between_spheres :
  let sphere1 : Sphere := { center := { x := 3, y := -5, z := 7 }, radius := 15 }
  let sphere2 : Sphere := { center := { x := -10, y := 20, z := -25 }, radius := 95 }
  ∀ (p1 p2 : Point3D),
    (distance p1 sphere1.center ≤ sphere1.radius) →
    (distance p2 sphere2.center ≤ sphere2.radius) →
    distance p1 p2 ≤ 110 + Real.sqrt 1818 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_distance_between_spheres_l1244_124405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1244_124476

/-- Parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2)^2 = 4 * p.1}

/-- Line with equation 2x + y - 4 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- Point A on both the parabola and the line -/
def A : ℝ × ℝ := (1, 2)

/-- Point B is the other intersection of the parabola and the line -/
def B : ℝ × ℝ := (4, -4)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_line_intersection :
  A ∈ Parabola ∧ A ∈ Line ∧ B ∈ Parabola ∧ B ∈ Line ∧
  distance F A + distance F B = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l1244_124476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_scenario_properties_l1244_124498

/-- Represents the walking speeds and meeting time of two people -/
structure WalkingScenario where
  speed_a : ℝ  -- Speed of person A in meters per minute
  speed_b : ℝ  -- Speed of person B in meters per minute
  meeting_time : ℝ  -- Time taken for the two people to meet in minutes

/-- Calculates the total distance between the starting points of two people -/
noncomputable def total_distance (scenario : WalkingScenario) : ℝ :=
  (scenario.speed_a + scenario.speed_b) * scenario.meeting_time

/-- Calculates the distance from the meeting point to the midpoint of the starting positions -/
noncomputable def distance_to_midpoint (scenario : WalkingScenario) : ℝ :=
  abs (scenario.speed_a * scenario.meeting_time - (total_distance scenario) / 2)

/-- Theorem stating the properties of the specific walking scenario -/
theorem walking_scenario_properties :
  let scenario : WalkingScenario := { speed_a := 65, speed_b := 55, meeting_time := 10 }
  total_distance scenario = 1200 ∧ distance_to_midpoint scenario = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_scenario_properties_l1244_124498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_formula_l1244_124465

theorem sin_sum_formula (α β : ℝ) : 
  Real.sin α + Real.sin β = 2 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_formula_l1244_124465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_segment_l1244_124453

/-- The midpoint of a line segment in polar coordinates -/
noncomputable def midpoint_polar (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ × ℝ :=
  (r1 / 2 + r2 / 2, (θ1 + θ2) / 2)

/-- Theorem: The midpoint of the line segment between (10, π/3) and (10, -π/6) is (10, π/4) -/
theorem midpoint_specific_segment :
  let A : ℝ × ℝ := (10, Real.pi/3)
  let B : ℝ × ℝ := (10, -Real.pi/6)
  let M : ℝ × ℝ := midpoint_polar A.1 A.2 B.1 B.2
  M = (10, Real.pi/4) ∧ M.1 > 0 ∧ 0 ≤ M.2 ∧ M.2 < 2*Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_specific_segment_l1244_124453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_exact_l1244_124470

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 7*x + 10) / (x + 1)

-- State the theorem
theorem f_min_value (x : ℝ) (h : x > -1) : f x ≥ 9 := by
  sorry

-- State that 9 is the minimum value
theorem f_min_value_exact : ∃ (x : ℝ), x > -1 ∧ f x = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_min_value_exact_l1244_124470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l1244_124454

-- Define the piecewise function h
noncomputable def h (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 4 else 3 * x - 18

-- Theorem statement
theorem h_solutions :
  ∀ x : ℝ, h x = 0 ↔ x = -1 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l1244_124454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_zero_l1244_124428

-- Define the quadratic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a - 2

-- Define the solution set condition
def solution_set (a : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ < 0 ∧ 0 < x₂ ∧ 
  (∀ x, f a x > 0 ↔ (x < x₁ ∨ x > x₂))

-- Define the expression to maximize
noncomputable def expr_to_maximize (x₁ x₂ : ℝ) : ℝ := 
  x₁ + x₂ + 2/x₁ + 2/x₂

-- State the theorem
theorem max_value_is_zero (a : ℝ) (x₁ x₂ : ℝ) 
  (h : solution_set a x₁ x₂) :
  ∀ y₁ y₂, solution_set a y₁ y₂ → 
    expr_to_maximize y₁ y₂ ≤ 0 := by
  sorry

#check max_value_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_zero_l1244_124428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_naturals_in_sequence_l1244_124455

noncomputable def sequenceX (x₀ x₁ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | 1 => x₁
  | n + 2 => (sequenceX x₀ x₁ n * sequenceX x₀ x₁ (n + 1)) / 
             (3 * sequenceX x₀ x₁ n - 2 * sequenceX x₀ x₁ (n + 1))

def containsInfinitelyManyNaturals (s : ℕ → ℝ) : Prop :=
  ∀ N : ℕ, ∃ n ≥ N, ∃ k : ℕ, s n = k

theorem no_infinite_naturals_in_sequence :
  ¬∃ x₀ x₁ : ℝ, containsInfinitelyManyNaturals (sequenceX x₀ x₁) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_naturals_in_sequence_l1244_124455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_equals_10_l1244_124482

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 / (2 - x)

-- Define the function g in terms of f's inverse
noncomputable def g (x : ℝ) : ℝ := 1 / (Function.invFun f x) + 9

-- Theorem statement
theorem g_of_3_equals_10 : g 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_equals_10_l1244_124482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l1244_124489

theorem square_side_length (total_width total_height : ℕ) 
  (hw : total_width = 3500) (hh : total_height = 2350) : ∃ s : ℕ, s = 575 := by
  -- r represents the shorter side of rectangles R1 and R2
  -- s represents the side length of square S2
  -- S1 and S3 have side lengths (r + s)
  -- Total height: r + s + r = total_height
  -- Total width: (r + s) + s + (r + s) = total_width
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_length_l1244_124489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1244_124463

/-- An ellipse with equation x²/8 + y²/2 = 1 -/
structure Ellipse where
  x : ℝ
  y : ℝ
  eq : x^2/8 + y^2/2 = 1

/-- The foci of the ellipse -/
structure Foci where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
def Point (e : Ellipse) : ℝ × ℝ := (e.x, e.y)

/-- The angle between two vectors -/
noncomputable def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area (e : Ellipse) (f : Foci) :
  angle (f.F₁ - Point e) (f.F₂ - Point e) = 2 * Real.pi / 3 →
  triangleArea f.F₁ (Point e) f.F₂ = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1244_124463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1244_124411

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (5 - Real.sqrt (4 - Real.sqrt (x + 1)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 15} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1244_124411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l1244_124420

/-- A circle with center on y = (1/3)x, tangent to positive y-axis, and x-axis chord of 4√2 -/
def SpecialCircle : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 
    -- Center is on y = (1/3)x
    p.2 = (1/3) * p.1 ∧
    -- Circle is tangent to positive y-axis
    p.1 ≥ 0 ∧
    -- Chord on x-axis has length 4√2
    (p.1 + 2*Real.sqrt 2) * (p.1 - 2*Real.sqrt 2) = 0 ∧
    -- Distance from center to any point on the circle is constant (radius)
    (p.1 - 3*t)^2 + (p.2 - t)^2 = (3*t)^2}

/-- The standard equation of the special circle -/
theorem special_circle_equation : 
  SpecialCircle = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_circle_equation_l1244_124420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1244_124480

/-- The radius of the larger circle Ω -/
def R : ℝ := 123

/-- The radius of the smaller circle ω -/
def r : ℝ := 61

/-- Represents a chord of the larger circle Ω -/
structure Chord where
  length : ℝ
  is_not_diameter : length < 2 * R

/-- Represents the three segments of the chord cut by ω -/
structure ChordSegments (c : Chord) where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  sum_eq_chord : s1 + s2 + s3 = c.length
  ratio : s1 = (1 : ℝ) ∧ s2 = (2 : ℝ) ∧ s3 = (3 : ℝ)

/-- The theorem stating the length of the chord -/
theorem chord_length (c : Chord) (s : ChordSegments c) :
  c.length = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1244_124480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_signal_pole_l1244_124414

/-- The time it takes for a train to cross a signal pole -/
noncomputable def train_crossing_time (train_length platform_length platform_crossing_time : ℝ) : ℝ :=
  train_length / ((train_length + platform_length) / platform_crossing_time)

/-- Theorem: A 300 m long train that crosses a 300 m platform in 36 seconds will take 18 seconds to cross a signal pole -/
theorem train_crossing_signal_pole :
  train_crossing_time 300 300 36 = 18 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_signal_pole_l1244_124414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_to_divide_l1244_124458

theorem total_amount_to_divide (
  intended_ratio : Fin 4 → ℚ)
  (actual_ratio : Fin 4 → ℚ)
  (extra_amount : ℚ)
  (x : ℚ) :
  intended_ratio = ![3, 4, 5, 6] →
  actual_ratio = ![3, 4, 6, 7] →
  (actual_ratio 2 + actual_ratio 3) * (Finset.sum Finset.univ intended_ratio)⁻¹ * x -
  (intended_ratio 2 + intended_ratio 3) * (Finset.sum Finset.univ actual_ratio)⁻¹ * x = extra_amount →
  extra_amount = 1400 →
  x = 36000 := by
  sorry

#check total_amount_to_divide

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_amount_to_divide_l1244_124458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_voucher_placement_theorem_l1244_124415

/-- The number of boxes a customer buys -/
def num_boxes : ℕ := 10

/-- The probability of finding a voucher in a single box -/
noncomputable def p (n : ℝ) : ℝ := 1 / n

/-- The probability of not finding a voucher in a single box -/
noncomputable def q (n : ℝ) : ℝ := 1 - p n

/-- The probability of finding at least one voucher in num_boxes boxes -/
noncomputable def prob_at_least_one (n : ℝ) : ℝ :=
  1 - (q n) ^ num_boxes

/-- The smallest number of boxes among which one voucher should be placed -/
def min_n : ℝ := 14.93

theorem voucher_placement_theorem :
  ∃ (n : ℕ), n > ⌈min_n⌉ ∧ prob_at_least_one (↑n) ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_voucher_placement_theorem_l1244_124415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sum_reciprocal_binomial_l1244_124487

/-- The binomial coefficient -/
def binomial (n i : ℕ) : ℕ := Nat.choose n i

/-- The sum of reciprocals of binomial coefficients -/
def sum_reciprocal_binomial (n : ℕ) : ℚ :=
  Finset.sum (Finset.range (n + 1)) (fun i => 1 / (binomial n i : ℚ))

/-- The limit of the sum of reciprocals of binomial coefficients as n approaches infinity is 2 -/
theorem limit_sum_reciprocal_binomial :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |sum_reciprocal_binomial n - 2| < ε := by
  sorry

#check limit_sum_reciprocal_binomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sum_reciprocal_binomial_l1244_124487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l1244_124421

def spinner1 : Finset ℕ := {3, 4, 5}
def spinner2 : Finset ℕ := {5, 6, 7, 8}

def is_even_product (a b : ℕ) : Bool := Even (a * b)

theorem probability_even_product :
  let total_outcomes := (spinner1.card : ℚ) * spinner2.card
  let even_outcomes := (spinner1.card * spinner2.card : ℚ) - (spinner1.filter (λ a => a % 2 = 1)).card * (spinner2.filter (λ b => b % 2 = 1)).card
  even_outcomes / total_outcomes = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_even_product_l1244_124421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l1244_124402

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with sum S_n, if S_6/S_3 = 3, then S_9/S_6 = 7/3 -/
theorem geometric_sum_ratio (a₁ : ℝ) (q : ℝ) :
  (geometric_sum a₁ q 6) / (geometric_sum a₁ q 3) = 3 →
  (geometric_sum a₁ q 9) / (geometric_sum a₁ q 6) = 7/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_ratio_l1244_124402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1244_124460

theorem triangle_problem (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  0 < a ∧ 0 < b ∧ 0 < c →  -- Side lengths are positive
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →  -- Sine law
  a^2 + b^2 + c^2 = 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a^2 + b^2 + c^2) →  -- Cosine law
  a / b + b / a = 4 * Real.cos C →  -- Given condition
  1 / Real.tan B = 1 / Real.tan A + 1 / Real.tan C →  -- Given condition
  (a^2 + b^2) / c^2 = 2 ∧ Real.cos A = Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1244_124460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l1244_124451

/-- Given a line l passing through point A(0,1) with slope k, and a circle C: (x-2)^2 + (y-3)^2 = 1.
    The line l intersects with the circle C at points M and N. -/
def line_intersects_circle (k : ℝ) (M N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), y = k * x + 1 ∧ (x - 2)^2 + (y - 3)^2 = 1 ∧
  (M.1 = x ∧ M.2 = y) ∨ (N.1 = x ∧ N.2 = y)

/-- The dot product of OM and ON is 12, where O is the origin -/
def dot_product_condition (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 12

/-- The distance between M and N -/
noncomputable def distance (M N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)

/-- Main theorem -/
theorem line_circle_intersection_theorem (k : ℝ) (M N : ℝ × ℝ) :
  line_intersects_circle k M N → dot_product_condition M N → distance M N = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l1244_124451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_leq_one_l1244_124446

-- Define the function f(x) = e^(|x-a|)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (|x - a|)

-- State the theorem
theorem increasing_f_implies_a_leq_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) →
  a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_leq_one_l1244_124446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_squared_equals_n_l1244_124459

def d (n : ℕ) : ℕ := Finset.card (Nat.divisors n)

theorem divisor_squared_equals_n (n : ℕ) : n = (d n)^2 ↔ n = 1 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_squared_equals_n_l1244_124459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_location_l1244_124468

theorem terminal_side_location (a : ℝ) (h : -1 < Real.sin a ∧ Real.sin a < 0) :
  (a % (2 * Real.pi) ∈ Set.Icc Real.pi (3 * Real.pi / 2)) ∨
  (a % (2 * Real.pi) ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_location_l1244_124468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hours_is_two_l1244_124431

/-- Represents the parking cost structure -/
structure ParkingCost where
  initialHours : ℝ
  initialCost : ℝ
  excessHourRate : ℝ

/-- Calculates the total cost for a given number of hours -/
noncomputable def totalCost (p : ParkingCost) (hours : ℝ) : ℝ :=
  p.initialCost + max 0 (hours - p.initialHours) * p.excessHourRate

/-- Calculates the average cost per hour for a given number of hours -/
noncomputable def averageCostPerHour (p : ParkingCost) (hours : ℝ) : ℝ :=
  totalCost p hours / hours

/-- Theorem stating the initial hours for the given parking cost structure -/
theorem initial_hours_is_two :
  ∃ p : ParkingCost,
    p.initialCost = 20 ∧
    p.excessHourRate = 1.75 ∧
    averageCostPerHour p 9 = 3.5833333333333335 ∧
    p.initialHours = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_hours_is_two_l1244_124431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_pi_minus_alpha_l1244_124462

theorem cosine_of_pi_minus_alpha (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2))
  (h2 : Real.tan α = -12 / 5) : 
  Real.cos (π - α) = 5 / 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_pi_minus_alpha_l1244_124462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_octagon_cover_ground_l1244_124443

-- Define the interior angle of a regular polygon
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

-- Theorem statement
theorem square_octagon_cover_ground :
  interior_angle 4 + 2 * interior_angle 8 = 360 := by
  -- Expand the definition of interior_angle
  unfold interior_angle
  -- Simplify the arithmetic
  simp [mul_add, add_mul]
  -- The proof is completed by normalization
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_octagon_cover_ground_l1244_124443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l1244_124413

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y = f (x^2 + y^2)

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ _ ↦ 0

/-- The constant function 1 -/
def OneFunction : ℝ → ℝ := λ _ ↦ 1

/-- The function that is 1 at 0 and 0 elsewhere -/
noncomputable def ZeroOneFunction : ℝ → ℝ := λ x ↦ if x = 0 then 1 else 0

/-- The main theorem stating that only three functions satisfy the equation -/
theorem functional_equation_solutions (f : ℝ → ℝ) :
  SatisfiesFunctionalEquation f ↔ 
  (f = ZeroFunction ∨ f = OneFunction ∨ f = ZeroOneFunction) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solutions_l1244_124413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_0_8_approx_l1244_124447

-- Define the logarithm base 4 of 0.8
noncomputable def log4_0_8 : ℝ := Real.log 0.8 / Real.log 4

-- State the theorem
theorem log4_0_8_approx : 
  ∃ ε > 0, abs (log4_0_8 + 0.1608) < ε ∧ ε < 0.0001 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log4_0_8_approx_l1244_124447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1244_124488

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 3*x + 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 1 ∨ (1 < x ∧ x < 2) ∨ 2 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1244_124488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l1244_124483

theorem largest_prime_factor_of_factorial_sum : 
  (Nat.factorial 7 + Nat.factorial 8).factors.maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_factorial_sum_l1244_124483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_rectangle_l1244_124442

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a function that represents the division of a triangle into three parts
def divide_triangle (t : Triangle) : List (ℝ × ℝ × ℝ × ℝ) :=
  sorry

-- Define a function that checks if a list of quadrilaterals can form a rectangle
def can_form_rectangle (parts : List (ℝ × ℝ × ℝ × ℝ)) : Bool :=
  sorry

-- The main theorem
theorem triangle_to_rectangle (t : Triangle) : 
  ∃ (parts : List (ℝ × ℝ × ℝ × ℝ)), 
    divide_triangle t = parts ∧ can_form_rectangle parts = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_to_rectangle_l1244_124442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1244_124467

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line type
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def fixed_point_theorem (given_circle : Circle) (l : Line) (N M A C D P Q : ℝ × ℝ) : Prop :=
  -- l is tangent to given_circle at N
  (∃ h1 : True, sorry) ∧
  -- NM is a diameter of given_circle
  (∃ h2 : True, sorry) ∧
  -- A is on line NM
  (∃ h3 : True, sorry) ∧
  -- C and D are on l
  (∃ h4 : True, sorry) ∧
  (∃ h5 : True, sorry) ∧
  -- P is intersection of MC and given_circle
  (∃ h6 : True, sorry) ∧
  -- Q is intersection of MD and given_circle
  (∃ h7 : True, sorry) ∧
  -- Arbitrary circle passing through A with center on l
  (∃ (arbitrary_circle : Circle),
    -- Center of arbitrary_circle is on l
    (∃ h8 : True, sorry) ∧
    -- arbitrary_circle passes through A, C, and D
    (∃ h9 h10 h11 : True, sorry)) →
  -- Theorem statement
  ∃ (K : ℝ × ℝ),
    -- K is on line MN
    (∃ h12 : True, sorry) ∧
    -- PQ passes through K
    (∃ h13 : True, sorry) ∧
    -- MK/KN = MN²/NA²
    (∃ h14 : True, sorry)

-- The theorem (no proof required)
theorem fixed_point_exists (given_circle : Circle) (l : Line) (N M A C D P Q : ℝ × ℝ) :
  fixed_point_theorem given_circle l N M A C D P Q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_exists_l1244_124467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1244_124481

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The triangle satisfies the given condition -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a + 2 * t.a * Real.cos t.B = t.c

/-- The triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi / 2 ∧
  0 < t.B ∧ t.B < Real.pi / 2 ∧
  0 < t.C ∧ t.C < Real.pi / 2

theorem triangle_theorem (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : isAcute t) 
  (h3 : t.c = 2) : 
  t.B = 2 * t.A ∧ 1 < t.a ∧ t.a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1244_124481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoints_distance_l1244_124494

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The ellipse equation -/
def ellipseEquation (e : Ellipse) (p : Point) : Prop :=
  25 * (p.x + 2)^2 + 4 * (p.y - 3)^2 = 100

/-- Check if a point is on the major axis of the ellipse -/
def isOnMajorAxis (e : Ellipse) (p : Point) : Prop :=
  p.x = e.center.x

/-- Check if a point is on the minor axis of the ellipse -/
def isOnMinorAxis (e : Ellipse) (p : Point) : Prop :=
  p.y = e.center.y

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem ellipse_axis_endpoints_distance :
  ∀ (e : Ellipse) (A B : Point),
  ellipseEquation e A ∧ 
  ellipseEquation e B ∧
  isOnMajorAxis e A ∧
  isOnMinorAxis e B →
  distance A B = Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axis_endpoints_distance_l1244_124494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_1234_l1244_124464

/-- The final position on a 10-point circle after n moves, where the kth move is k^k steps clockwise -/
def final_position (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc k => (acc + (k + 1)^(k + 1)) % 10) 0

/-- The theorem stating that after 1234 moves, the final position is 7 -/
theorem final_position_1234 : final_position 1234 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_position_1234_l1244_124464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1244_124478

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1 ∧ f x ≤ 2) ∧
  (∀ x₀ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 
    f x₀ = 6 / 5 → Real.cos (2 * x₀) = (3 - 4 * Real.sqrt 3) / 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1244_124478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1244_124430

noncomputable section

-- Define the original function
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

-- Define the transformation functions
def shift_left (g : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => g (x + a)
def double_abscissa (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x / 2)

-- State the theorem
theorem transform_sin_function :
  (double_abscissa (shift_left f (Real.pi / 6))) = Real.sin := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_function_l1244_124430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_coverage_l1244_124434

/-- The side length of the square -/
noncomputable def squareSide : ℝ := 20

/-- The area of the square -/
noncomputable def squareArea : ℝ := squareSide ^ 2

/-- The ratio of the area covered by circles around lattice points -/
noncomputable def coverageRatio : ℝ := 3 / 4

/-- The radius we're trying to find -/
noncomputable def d : ℝ := Real.sqrt (coverageRatio / Real.pi)

/-- Theorem stating that d is approximately equal to 0.5 -/
theorem lattice_point_coverage :
  ‖d - 0.5‖ < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_coverage_l1244_124434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1244_124410

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

-- State the theorem
theorem function_properties :
  (∀ x > 3, f x ≥ 9) ∧
  (∃ x > 3, f x = 9) ∧
  (∀ t : ℝ, (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ t ∈ Set.Iic (-2) ∪ Set.Ioi (-1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1244_124410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_4sin_10deg_approx_l1244_124406

theorem tan_plus_4sin_10deg_approx :
  ∃ ε > 0, |Real.tan (10 * π / 180) + 4 * Real.sin (10 * π / 180) - 1.355| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_4sin_10deg_approx_l1244_124406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_mapping_l1244_124440

-- Define the sets A and B
def A : Set ℝ := Set.univ
def B : Set ℝ := Set.univ

-- Define the function f
def f (x : ℝ) : ℝ := -x + 1

-- Theorem stating that f is a mapping from A to B
theorem f_is_mapping : 
  (∀ x, x ∈ A → ∃! y, y ∈ B ∧ f x = y) ∧ 
  (∀ x, x ∈ A → f x ∈ B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_mapping_l1244_124440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_angles_l1244_124491

def is_valid_z (z : ℂ) : Prop :=
  z^24 - z^6 - 1 = 0 ∧ Complex.abs z = 1

def angle_set : Set ℝ :=
  {θ | ∃ z, is_valid_z z ∧ z = Complex.exp (θ * Complex.I)}

theorem sum_of_even_angles :
  ∃ (angles : List ℝ), 
    angles.length > 0 ∧
    List.Sorted (· < ·) angles ∧
    (∀ θ ∈ angles, θ ∈ angle_set) ∧
    (∀ θ ∈ angle_set, θ ∈ angles) ∧
    (∀ θ ∈ angles, 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
    (List.sum (List.map (fun i => angles[2*i+1]!) (List.range (angles.length / 2))) = 20 * Real.pi / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_even_angles_l1244_124491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_acute_angles_not_imply_congruence_l1244_124409

-- Define a right-angled triangle
structure RightAngledTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2
  positive_sides : leg1 > 0 ∧ leg2 > 0 ∧ hypotenuse > 0

-- Define congruence for right-angled triangles
def congruent (t1 t2 : RightAngledTriangle) : Prop :=
  t1.leg1 = t2.leg1 ∧ t1.leg2 = t2.leg2 ∧ t1.hypotenuse = t2.hypotenuse

-- Define acute angles of a right-angled triangle
noncomputable def acute_angle1 (t : RightAngledTriangle) : ℝ := Real.arcsin (t.leg1 / t.hypotenuse)
noncomputable def acute_angle2 (t : RightAngledTriangle) : ℝ := Real.arcsin (t.leg2 / t.hypotenuse)

-- Theorem stating that congruent acute angles do not imply congruence
theorem congruent_acute_angles_not_imply_congruence :
  ∃ (t1 t2 : RightAngledTriangle),
    acute_angle1 t1 = acute_angle1 t2 ∧
    acute_angle2 t1 = acute_angle2 t2 ∧
    ¬(congruent t1 t2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_acute_angles_not_imply_congruence_l1244_124409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_braking_distance_l1244_124493

/-- The distance-time equation for a braking train -/
noncomputable def s (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The velocity of the train at time t -/
noncomputable def v (t : ℝ) : ℝ := 27 - 0.9 * t

/-- The time at which the train comes to a complete stop -/
noncomputable def stop_time : ℝ := 27 / 0.9

theorem train_braking_distance :
  s stop_time = 405 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_braking_distance_l1244_124493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_l1244_124438

/-- The amount lent by A to B and by B to C -/
def P : ℝ := 2000

/-- The interest rate at which A lends to B (as a decimal) -/
def rate_A_to_B : ℝ := 0.10

/-- The interest rate at which B lends to C (as a decimal) -/
def rate_B_to_C : ℝ := 0.115

/-- The time period in years -/
def time : ℝ := 3

/-- The gain of B over the time period -/
def gain : ℝ := 90

/-- Theorem stating that given the conditions, the amount lent (P) is 2000 -/
theorem amount_lent : gain = (rate_B_to_C - rate_A_to_B) * P * time := by
  -- Substitute the values
  simp [gain, rate_B_to_C, rate_A_to_B, P, time]
  -- Perform the calculation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_lent_l1244_124438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l1244_124407

/-- The slope angle of the tangent line to y = (1/2)x^2 - 2x at (1, -3/2) is 135° -/
theorem tangent_slope_angle :
  let f : ℝ → ℝ := λ x ↦ (1/2) * x^2 - 2*x
  let x₀ : ℝ := 1
  let y₀ : ℝ := -3/2
  let slope : ℝ := deriv f x₀
  let angle : ℝ := Real.arctan slope
  f x₀ = y₀ → angle = 135 * (π / 180) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_l1244_124407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l1244_124461

/-- Predicate to check if three complex numbers form an equilateral triangle --/
def IsEquilateralTriangle (a b c : ℂ) : Prop :=
  Complex.abs (a - b) = Complex.abs (b - c) ∧ 
  Complex.abs (b - c) = Complex.abs (c - a)

theorem equilateral_triangle_side_length 
  (d e f : ℂ) 
  (s t u : ℂ) 
  (h1 : d^3 + s*d^2 + t*d + u = 0)
  (h2 : e^3 + s*e^2 + t*e + u = 0)
  (h3 : f^3 + s*f^2 + t*f + u = 0)
  (h4 : Complex.abs d^2 + Complex.abs e^2 + Complex.abs f^2 = 300)
  (h5 : IsEquilateralTriangle d e f) : 
  Complex.abs (d - e)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l1244_124461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1244_124472

theorem two_solutions_exist (a : ℝ) : 
  ((a ≥ 3.25 ∧ a < 4) ∨ (a ≥ -0.5 ∧ a < 1)) →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁ ∈ Set.Icc (-2 * Real.pi / 3) Real.pi ∧ 
  x₂ ∈ Set.Icc (-2 * Real.pi / 3) Real.pi ∧
  (Real.cos (2 * x₁) + 14 * Real.cos x₁ - 14 * a)^7 - 
  (6 * a * Real.cos x₁ - 4 * a^2 - 1)^7 = 
  (6 * a - 14) * Real.cos x₁ + 2 * Real.sin x₁^2 - 4 * a^2 + 14 * a - 2 ∧
  (Real.cos (2 * x₂) + 14 * Real.cos x₂ - 14 * a)^7 - 
  (6 * a * Real.cos x₂ - 4 * a^2 - 1)^7 = 
  (6 * a - 14) * Real.cos x₂ + 2 * Real.sin x₂^2 - 4 * a^2 + 14 * a - 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_exist_l1244_124472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_not_sold_percentage_l1244_124426

def initial_stock : ℕ := 900

def books_sold : List ℕ := [75, 50, 64, 78, 135]
def books_returned : List ℕ := [5, 10, 3, 7, 15]

def total_books_sold : ℕ := books_sold.sum
def total_books_returned : ℕ := books_returned.sum

def net_books_sold : ℕ := total_books_sold - total_books_returned
def books_remaining : ℕ := initial_stock - net_books_sold

noncomputable def percentage_not_sold : ℝ := (books_remaining : ℝ) / (initial_stock : ℝ) * 100

theorem books_not_sold_percentage :
  |percentage_not_sold - 59.78| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_not_sold_percentage_l1244_124426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profile_line_theorem_l1244_124425

/-- Represents a line in general position -/
structure GeneralLine where
  g' : ℝ
  g'' : ℝ

/-- Represents an axis -/
structure Axis where
  angle : ℝ

/-- Determines if a line is a profile line in a given system -/
def is_profile_line (g : GeneralLine) (system : Axis) : Prop := sorry

/-- Constructs the g^IV line from a general position line -/
def construct_g_IV (g : GeneralLine) : GeneralLine := sorry

/-- Checks if an axis is perpendicular to a line -/
def is_perpendicular_to_line (a : Axis) (g : GeneralLine) : Prop := sorry

theorem profile_line_theorem 
  (g : GeneralLine) 
  (x_14 : Axis) 
  (x_15 : Axis) :
  is_perpendicular_to_line x_15 (construct_g_IV g) → 
  is_profile_line g (Axis.mk (x_15.angle - x_14.angle)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profile_line_theorem_l1244_124425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_domain_sum_l1244_124497

-- Define the function f
noncomputable def f (x c : ℝ) : ℝ := x * Real.cos x + c

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_domain_sum (a b c : ℝ) :
  (∀ x ∈ Set.Icc a b, f x c = f x c) →  -- f is defined on [a, b]
  is_odd (fun x ↦ f x c) →              -- f is an odd function
  a + b + c = 0 := by
  sorry

#check odd_function_domain_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_domain_sum_l1244_124497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_constraint_l1244_124422

theorem max_value_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + 5 * y < 90) :
  x * y * (90 - 4 * x - 5 * y) ≤ 1350 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_xy_constraint_l1244_124422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1244_124444

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
  h_sum : A + B + C = π
  h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the area of a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h_relation : t.b / t.c = Real.sqrt 3 * Real.sin t.A + Real.cos t.A) :
  t.C = π / 6 ∧ 
  (t.c = 2 → 
    (∀ t' : Triangle, t'.c = 2 → t'.A = t.A → t'.C = t.C → 
      2 * t'.area ≤ 4 + 2 * Real.sqrt 3)) ∧
  (∃ t' : Triangle, t'.c = 2 ∧ t'.A = t.A ∧ t'.C = t.C ∧ 
    2 * t'.area = 4 + 2 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1244_124444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_circle_tangent_to_asymptotes_l1244_124423

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0), 
    if a circle with radius a centered at the right focus of the hyperbola 
    touches both asymptotes of the hyperbola, 
    then the eccentricity of the hyperbola is √2. -/
theorem hyperbola_eccentricity_with_circle_tangent_to_asymptotes 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let right_focus := (Real.sqrt (a^2 + b^2), 0)
  let circle := fun (x y : ℝ) ↦ (x - right_focus.1)^2 + y^2 = a^2
  let asymptote1 := fun (x y : ℝ) ↦ y = (b / a) * x
  let asymptote2 := fun (x y : ℝ) ↦ y = -(b / a) * x
  let eccentricity := Real.sqrt (a^2 + b^2) / a
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle x1 y1 ∧ asymptote1 x1 y1 ∧
    circle x2 y2 ∧ asymptote2 x2 y2) →
  eccentricity = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_circle_tangent_to_asymptotes_l1244_124423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_Q_l1244_124469

-- Define the points P and Q
noncomputable def P : ℝ × ℝ := (2, 0)
noncomputable def Q : ℝ × ℝ := (-2, 4 * Real.sqrt 3 / 3)

-- Define the distance from Q to the line
noncomputable def distance_to_line : ℝ := 4

-- Define the two possible lines
def vertical_line (x : ℝ) : Set (ℝ × ℝ) := {p | p.1 = x}
def sloped_line (m k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * (p.1 - k)}

-- Define the distance function from a point to a line
noncomputable def point_to_line_distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem line_through_P_distance_Q :
  (∃ (x : ℝ), vertical_line x = {p | p.1 = 2} ∧ 
    point_to_line_distance Q (vertical_line x) = distance_to_line) ∨
  (∃ (m : ℝ), m = Real.sqrt 3 / 3 ∧ 
    point_to_line_distance Q (sloped_line m P.1) = distance_to_line) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_Q_l1244_124469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shortest_path_perp_distance_l1244_124433

/-- Represents a cone with vertex I, base radius r, and slant height s -/
structure Cone where
  r : ℝ  -- base radius
  s : ℝ  -- slant height

/-- Represents a point on the surface of the cone -/
structure ConePoint (c : Cone) where
  height : ℝ  -- height from the base
  angle : ℝ    -- angle around the cone (0 ≤ angle < 2π)

/-- The shortest path on the surface of the cone between two points -/
noncomputable def shortestPath (c : Cone) (p1 p2 : ConePoint c) : ℝ := sorry

/-- The perpendicular distance from the vertex to a point on the surface -/
noncomputable def perpDistance (c : Cone) (p : ConePoint c) : ℝ := sorry

/-- Find the point on the shortest path that is closest to the vertex -/
noncomputable def closestPoint (c : Cone) (p1 p2 : ConePoint c) : ConePoint c := sorry

theorem cone_shortest_path_perp_distance
  (c : Cone)
  (h_r : c.r = 1)
  (h_s : c.s = 4)
  (R : ConePoint c)
  (h_R : R.height = 3)
  (A : ConePoint c)
  (h_A : A.height = 0)
  : perpDistance c (closestPoint c R A) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shortest_path_perp_distance_l1244_124433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1244_124436

theorem min_abs_difference (a b : ℕ) (h : a * b - 3 * a + 4 * b = 149) :
  ∃ (a' b' : ℕ), a' * b' - 3 * a' + 4 * b' = 149 ∧
  ∀ (x y : ℕ), x * y - 3 * x + 4 * y = 149 →
  |Int.ofNat a' - Int.ofNat b'| ≤ |Int.ofNat x - Int.ofNat y| ∧ |Int.ofNat a' - Int.ofNat b'| = 33 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l1244_124436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1244_124496

theorem trig_inequality (α : ℝ) : (Real.cos α) ^ 6 + (Real.sin α) ^ 6 ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_inequality_l1244_124496
