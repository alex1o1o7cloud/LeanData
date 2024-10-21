import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution1_satisfies_equation_and_condition_solution2_satisfies_equation_and_condition_l416_41657

-- Define the differential equation
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x) * Real.sin x = y x * Real.log (y x)

-- Define the solutions
noncomputable def solution1 (x : ℝ) : ℝ := Real.exp (abs (Real.tan (x / 2)))
def solution2 : ℝ → ℝ := λ _ => 1

-- Theorem for the first solution
theorem solution1_satisfies_equation_and_condition :
  differential_equation solution1 ∧ solution1 (Real.pi / 2) = Real.exp 1 := by sorry

-- Theorem for the second solution
theorem solution2_satisfies_equation_and_condition :
  differential_equation solution2 ∧ solution2 (Real.pi / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution1_satisfies_equation_and_condition_solution2_satisfies_equation_and_condition_l416_41657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_mapping_with_five_queries_l416_41631

-- Define the type for letters
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J

-- Define the type for digits
def Digit := Fin 10

-- Define the type for a mapping from letters to digits
def Mapping := Letter → Digit

-- Define a function to represent a query
def Query := List Letter → Nat

-- Define the property of a valid mapping
def ValidMapping (m : Mapping) : Prop :=
  ∀ d : Digit, ∃ l : Letter, m l = d

-- Define the property of a correct query result
def CorrectQueryResult (m : Mapping) (q : Query) : Prop :=
  ∀ ls : List Letter, q ls = (ls.map (fun l => (m l).val)).sum

-- Theorem statement
theorem determine_mapping_with_five_queries :
  ∃ (queries : List Query),
    queries.length ≤ 5 ∧
    ∀ (m : Mapping),
      ValidMapping m →
      ∀ (q : Query),
        CorrectQueryResult m q →
        ∃ (m' : Mapping),
          ValidMapping m' ∧
          (∀ l : Letter, m l = m' l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_mapping_with_five_queries_l416_41631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_circles_existence_l416_41651

-- Define the basic types
variable (Point Line Circle : Type)

-- Define the necessary operations and relations
variable (Center : Circle → Point)
variable (Radius : Circle → ℝ)
variable (Intersect : Circle → Circle → Prop)
variable (InternalTangent ExternalTangent : Line → Circle → Circle → Prop)
variable (Tangent : Line → Circle → Point → Prop)

-- Define the points and lines
variable (O₁ O₂ O : Point) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : Line)

-- Define the circles
variable (c₁ c₂ : Circle)

-- Define the conditions
axiom non_intersecting : ¬ Intersect c₁ c₂
axiom center_c₁ : Center c₁ = O₁
axiom center_c₂ : Center c₂ = O₂

axiom internal_tangent_1 : InternalTangent a₁ c₁ c₂
axiom internal_tangent_2 : InternalTangent a₂ c₁ c₂
axiom external_tangent_1 : ExternalTangent a₃ c₁ c₂
axiom external_tangent_2 : ExternalTangent a₄ c₁ c₂

axiom tangent_O₂_to_c₁_1 : Tangent a₅ c₁ O₂
axiom tangent_O₂_to_c₁_2 : Tangent a₆ c₁ O₂
axiom tangent_O₁_to_c₂_1 : Tangent a₇ c₂ O₁
axiom tangent_O₁_to_c₂_2 : Tangent a₈ c₂ O₁

variable (LineContains : Line → Point → Prop)
axiom intersection_point : LineContains a₁ O ∧ LineContains a₂ O

-- Theorem to prove
theorem two_circles_existence :
  ∃ (c₃ c₄ : Circle),
    Center c₃ = O ∧
    Center c₄ = O ∧
    Tangent a₃ c₃ O ∧
    Tangent a₄ c₃ O ∧
    Tangent a₅ c₄ O ∧
    Tangent a₆ c₄ O ∧
    Tangent a₇ c₄ O ∧
    Tangent a₈ c₄ O ∧
    Radius c₄ = (1/2) * Radius c₃ :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_circles_existence_l416_41651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l416_41676

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  r1 : ℝ  -- radius of upper base
  r2 : ℝ  -- radius of lower base
  h : ℝ   -- height of frustum

/-- Calculates the lateral surface area of a frustum -/
noncomputable def lateral_surface_area (f : Frustum) : ℝ :=
  Real.pi * (f.r1 + f.r2) * Real.sqrt (f.h^2 + (f.r2 - f.r1)^2)

/-- Theorem stating the lateral surface area of the specific frustum -/
theorem frustum_lateral_surface_area :
  let f : Frustum := { r1 := 2, r2 := 8, h := 9 }
  lateral_surface_area f = 10 * Real.pi * Real.sqrt 117 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l416_41676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l416_41696

def A (x : ℝ) : Set ℝ := {3, x, x^2 - 2*x}

theorem set_A_properties (x : ℝ) :
  (x ∉ ({0, -1, 3} : Set ℝ)) ∧
  (-2 ∈ A x → x = -2) :=
by
  constructor
  · intro h
    sorry -- Proof that x is not in {0, -1, 3}
  · intro h
    sorry -- Proof that if -2 is in A(x), then x = -2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_properties_l416_41696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l416_41660

def P : Set ℕ := {0, 1, 2}
def Q : Set ℕ := {y | ∃ x : ℕ, y = 3^x}

theorem intersection_P_Q : P ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_P_Q_l416_41660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_existence_l416_41683

/-- Triangle DEF with side lengths -/
structure Triangle :=
  (DE : ℝ) (EF : ℝ) (DF : ℝ)

/-- Point N on side DF -/
structure PointN :=
  (DN : ℝ) (NF : ℝ)

/-- Incircle of a triangle -/
structure Incircle :=
  (radius : ℝ)

/-- Helper function to construct the incircle of triangle DEN -/
noncomputable def triangle_DEN_incircle (t : Triangle) (N : PointN) : Incircle :=
  ⟨0⟩ -- Placeholder value, replace with actual calculation if needed

/-- Helper function to construct the incircle of triangle EFN -/
noncomputable def triangle_EFN_incircle (t : Triangle) (N : PointN) : Incircle :=
  ⟨0⟩ -- Placeholder value, replace with actual calculation if needed

/-- The main theorem -/
theorem equal_incircle_radii_existence (t : Triangle) 
  (h_DE : t.DE = 14) (h_EF : t.EF = 15) (h_DF : t.DF = 21) :
  ∃ (N : PointN), 
    N.DN + N.NF = t.DF ∧
    (∃ (r : ℝ), 
      (triangle_DEN_incircle t N).radius = r ∧
      (triangle_EFN_incircle t N).radius = r) ∧
    (∃ (p q : ℕ), 
      N.DN / N.NF = p / q ∧ 
      Nat.Coprime p q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_incircle_radii_existence_l416_41683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_wu_position_total_distance_walked_l416_41665

-- Define the walking distances
def walk_distances : List Int := [620, -580, 450, 650, -520, -480, -660, 550]

-- Theorem for Teacher Wu's final position
theorem teacher_wu_position (distances : List Int := walk_distances) :
  distances.sum = 30 := by sorry

-- Theorem for total distance walked
theorem total_distance_walked (distances : List Int := walk_distances) :
  (distances.map Int.natAbs).sum + 30 = 4540 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_wu_position_total_distance_walked_l416_41665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l416_41636

theorem problem_solution :
  ∀ (a b : ℝ),
    (∃ (x : ℝ), x > 0 ∧ Real.sqrt x = 2*a - 7 ∧ Real.sqrt x = a + 4) →
    ((b - 12) ^ (1/3 : ℝ) = -2) →
    (a = 1 ∧ b = 4 ∧ Real.sqrt (5*a + b) = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l416_41636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_cut_max_loss_l416_41680

-- Define the value function for a diamond
noncomputable def diamond_value (k : ℝ) (w : ℝ) : ℝ := k * w^2

-- Define the percentage loss function
noncomputable def percentage_loss (m n : ℝ) : ℝ :=
  1 - ((m / (m + n))^2 + (n / (m + n))^2)

theorem diamond_cut_max_loss (k m n : ℝ) (hk : k > 0) (hm : m > 0) (hn : n > 0) :
  percentage_loss m n ≤ 1/2 ∧
  (percentage_loss m n = 1/2 ↔ m = n) :=
by
  sorry

#check diamond_cut_max_loss

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_cut_max_loss_l416_41680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l416_41638

/-- Represents a tetrahedron A-BCD with specific properties -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BAC : ℝ
  BAD : ℝ
  CAD : ℝ
  ab_eq_ac : AB = AC
  ab_eq_2 : AB = 2
  ad_eq_3 : AD = 3
  bac_eq_60 : BAC = 60 * π / 180
  bad_eq_60 : BAD = 60 * π / 180
  cad_eq_60 : CAD = 60 * π / 180

/-- The circumradius of the tetrahedron -/
def circumradius (t : Tetrahedron) : ℝ := 
  sorry

/-- The surface area of the circumscribed sphere of a tetrahedron with given properties is 19π/2 -/
theorem circumscribed_sphere_surface_area (t : Tetrahedron) : 
  ∃ (S : ℝ), S = 19 * π / 2 ∧ S = 4 * π * (circumradius t)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l416_41638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l416_41610

theorem divisibility_problem (a b : ℕ) 
  (h1 : (b + 1) ∣ a) 
  (h2 : (a + b) ∣ 67) : 
  (∃ x ∈ ({34, 51, 64, 66} : Set ℕ), x = a) ∧ 
  b ∈ ({1, 3, 16, 33} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_problem_l416_41610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l416_41653

-- Define the function f
noncomputable def f (x a : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x + (5/8) * a - 3/2

-- State the theorem
theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 2) →
  a = 3.6 := by
  sorry

#check min_value_implies_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l416_41653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l416_41695

noncomputable def vector1 : Fin 2 → ℝ := ![4, 5]
noncomputable def vector2 : Fin 2 → ℝ := ![2, -1]

def dot_product (v1 v2 : Fin 2 → ℝ) : ℝ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

theorem angle_between_vectors :
  (dot_product vector1 vector2) / (magnitude vector1 * magnitude vector2) = 3 / Real.sqrt 205 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l416_41695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l416_41666

/-- Represents the different animal masks --/
inductive Mask
  | elephant
  | mouse
  | pig
  | panda

/-- Assigns a digit to each mask --/
def mask_assignment : Mask → Nat := sorry

/-- The product of the mouse digit with itself ends with the elephant digit --/
def mouse_elephant_relation (assignment : Mask → Nat) : Prop :=
  (assignment Mask.mouse * assignment Mask.mouse) % 10 = assignment Mask.elephant

/-- All assigned digits are unique --/
def unique_digits (assignment : Mask → Nat) : Prop :=
  ∀ m₁ m₂ : Mask, m₁ ≠ m₂ → assignment m₁ ≠ assignment m₂

/-- All assigned digits are single-digit numbers (0-9) --/
def valid_digits (assignment : Mask → Nat) : Prop :=
  ∀ m : Mask, assignment m < 10

/-- The main theorem stating the unique valid assignment --/
theorem unique_valid_assignment :
  ∃! assignment : Mask → Nat,
    mouse_elephant_relation assignment ∧
    unique_digits assignment ∧
    valid_digits assignment ∧
    assignment Mask.elephant = 6 ∧
    assignment Mask.mouse = 4 ∧
    assignment Mask.pig = 8 ∧
    assignment Mask.panda = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l416_41666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_proof_l416_41675

def recurring_decimal : ℚ := 356 / 999

theorem recurring_decimal_proof :
  recurring_decimal = 356 / 999 ∧
  (Nat.gcd 356 999 = 1) ∧
  (356 + 999 = 1355) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_proof_l416_41675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l416_41672

/-- An arithmetic sequence with specific properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 4 = 9 ∧ a 3 + a 7 = 22

/-- Sum of the first n terms of the arithmetic sequence -/
noncomputable def S (n : ℕ) : ℝ := n^2 + 2*n

/-- The b_n sequence defined in terms of a_n -/
noncomputable def b (a : ℕ → ℝ) (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

/-- Sum of the first n terms of the b sequence -/
noncomputable def T (n : ℕ) : ℝ := n / (3 * (2*n + 3))

/-- Main theorem encapsulating all parts of the problem -/
theorem arithmetic_sequence_properties (a : ℕ → ℝ) (ha : arithmetic_sequence a) :
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, S n = n^2 + 2*n) ∧
  (∀ n, T n = n / (3 * (2*n + 3))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l416_41672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_sum_2345678_l416_41664

def digit_transform (d : ℕ) : ℕ :=
  if d > 2 ∧ d ≤ 9 then d - 2
  else if d < 8 then d + 2
  else d

def is_valid_transformation (A B : ℕ) : Prop :=
  ∀ (i : ℕ), i < (Nat.digits 10 A).length →
    (Nat.digits 10 B).get! i = digit_transform ((Nat.digits 10 A).get! i)

theorem no_valid_sum_2345678 :
  ¬ ∃ (A B : ℕ), is_valid_transformation A B ∧ A + B = 2345678 := by
  sorry

#check no_valid_sum_2345678

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_sum_2345678_l416_41664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l416_41684

/-- Represents a grid on which the cutting game is played -/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a player in the game -/
inductive Player where
  | First
  | Second

/-- Represents the state of the game -/
structure GameState where
  grid : Grid
  currentPlayer : Player
  cutsMade : ℕ

/-- Represents a strategy for playing the game -/
def Strategy := GameState → ℕ × ℕ  -- (row, col) of the next cut

/-- Helper function to apply strategies and advance the game state -/
def applyStrategies (g : Grid) (s1 s2 : Strategy) (n : ℕ) : GameState :=
  sorry

/-- Predicate to check if the sheet is separated into two pieces -/
def sheetSeparated (gs : GameState) : Prop :=
  sorry

/-- Determines if a given strategy is winning for the first player -/
def isWinningStrategy (g : Grid) (s : Strategy) : Prop :=
  ∀ (opponent : Strategy), 
    ∃ (n : ℕ), 
      let finalState := applyStrategies g s opponent n
      finalState.cutsMade % 2 = 1 ∧ sheetSeparated finalState

/-- The main theorem: there exists a winning strategy for the first player -/
theorem first_player_wins (g : Grid) (h1 : g.rows = 30) (h2 : g.cols = 45) :
  ∃ (s : Strategy), isWinningStrategy g s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l416_41684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l416_41644

/-- The parabola y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The line 4x - 3y + 6 = 0 -/
def Line1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 + 6 = 0}

/-- The line x = -1 -/
def Line2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- The distance from a point to a line -/
noncomputable def distToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- The sum of distances from a point to two lines -/
noncomputable def sumOfDistances (p : ℝ × ℝ) : ℝ :=
  distToLine p Line1 + distToLine p Line2

theorem min_sum_of_distances :
  ∃ (d : ℝ), d = 2 ∧ ∀ (p : ℝ × ℝ), p ∈ Parabola → sumOfDistances p ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l416_41644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_2250_l416_41674

/-- The length of a bridge in meters -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time * 1000 / 60

/-- Theorem: The length of the bridge is 2250 meters -/
theorem bridge_length_is_2250 :
  bridge_length 9 15 = 2250 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_2250_l416_41674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l416_41681

theorem equation_solutions (x : ℝ) : 
  (16:ℝ)^x + (81:ℝ)^x = (9/8) * ((24:ℝ)^x + (36:ℝ)^x) ↔ x = 0 ∨ x = Real.log (9/8) / Real.log (2/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l416_41681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l416_41634

/-- T(r) is the sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... -/
noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

/-- Theorem stating the sum of T(b) and T(-b) given the conditions -/
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 2700) :
  T b + T (-b) = 360 := by
  sorry

#check sum_of_T

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l416_41634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_division_count_l416_41627

-- Define the number of friends and teams
def n : ℕ := 8
def k : ℕ := 4

-- Define the function to calculate the number of ways to divide friends
def divide_friends (n k : ℕ) : ℕ := 
  -- Placeholder implementation
  0

-- State the theorem
theorem friends_division_count :
  divide_friends n k = 39824 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_division_count_l416_41627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41645

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_properties :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f (-x) = -f x) ∧ 
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f x < f y) ∧
  (Set.Ioo 0 (1/2 : ℝ) = {x | x ∈ Set.Icc (-1 : ℝ) 1 ∧ f x < f (1 - x)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41622

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 - Real.cos x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties : 
  (f (2 * Real.pi / 3) = 2) ∧ 
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
    (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → S ≥ T) ∧ T = Real.pi) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (-5 * Real.pi / 6 + k * Real.pi) (-Real.pi / 3 + k * Real.pi))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_first_15_rows_l416_41630

/-- Pascal's Triangle coefficient -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Check if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Count even numbers in a row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k ↦ isEven (pascal row k)) |>.length

/-- Count even numbers in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ :=
  (List.range n).map countEvenInRow |>.sum

/-- The count of even integers in the first 15 rows of Pascal's Triangle is 61 -/
theorem even_count_first_15_rows : countEvenInTriangle 15 = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_first_15_rows_l416_41630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l416_41698

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors :
  ∃ (p q r : ℕ) (a b c : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    sum_of_divisors 450 = p^a * q^b * r^c ∧
    ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 450 → (s = p ∨ s = q ∨ s = r) :=
by sorry

#check sum_of_divisors_450_prime_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_450_prime_factors_l416_41698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_max_profit_value_l416_41692

-- Define the profit function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    -(1/3) * x^2 + 40 * x - 250
  else if x ≥ 80 then
    1200 - (x + 10000 / x)
  else
    0  -- undefined for x ≤ 0

-- State the theorem
theorem max_profit_at_100 :
  ∀ x : ℝ, 0 < x → L x ≤ L 100 :=
by
  sorry

-- Optionally, we can add a corollary to state the exact maximum profit
theorem max_profit_value :
  L 100 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_100_max_profit_value_l416_41692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l416_41641

theorem exam_candidates_count : ∃ (total_candidates : ℕ),
  let girls : ℕ := 900
  let boys : ℕ := total_candidates - girls
  let boys_pass_rate : ℚ := 38 / 100
  let girls_pass_rate : ℚ := 32 / 100
  let total_fail_rate : ℚ := 647 / 1000
  (boys_pass_rate * (boys : ℚ) + girls_pass_rate * (girls : ℚ)) / (total_candidates : ℚ) = 1 - total_fail_rate ∧
  total_candidates = 2000 := by
  use 2000
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_candidates_count_l416_41641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_sequence_non_periodic_l416_41637

/-- The sequence of zeros and ones based on the sum of digits of natural numbers -/
def digit_sum_sequence (k : ℕ) : Bool :=
  (Nat.digits 10 k).sum % 2 ≠ 0

/-- Proposition: The sequence is non-periodic -/
theorem digit_sum_sequence_non_periodic :
  ∀ d m : ℕ, d > 0 → ∃ n : ℕ, n ≥ m ∧ digit_sum_sequence n ≠ digit_sum_sequence (n + d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_sequence_non_periodic_l416_41637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_and_unique_l416_41673

/-- Definition of the left-hand side of the equation -/
noncomputable def f (x : ℝ) : ℝ :=
  (x + (x + (x + x^(1/4))^(1/4))^(1/4))^(1/4)

/-- Definition of the right-hand side of the equation -/
noncomputable def g (x : ℝ) : ℝ :=
  (x * (x * (x * x^(1/4))^(1/4))^(1/4))^(1/4)

/-- Theorem stating the existence and uniqueness of the solution -/
theorem equation_solution_exists_and_unique :
  ∃! x : ℝ, x > 0 ∧ f x = g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_exists_and_unique_l416_41673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l416_41694

noncomputable def original_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

def translate_right (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x - h)

noncomputable def translated_function : ℝ → ℝ := translate_right original_function (Real.pi / 6)

noncomputable def expected_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem translation_equivalence : translated_function = expected_function := by
  funext x
  simp [translated_function, translate_right, original_function, expected_function]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_equivalence_l416_41694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_and_B_intersection_l416_41690

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | |x - a| > 3}

-- Theorem statement
theorem complement_A_and_B_intersection (a : ℝ) : 
  (Set.compl A = Set.Icc (-1) 3) ∧ 
  ((Set.compl A ∩ B a = ∅) ↔ (0 ≤ a ∧ a ≤ 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_and_B_intersection_l416_41690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l416_41640

/-- The distance rowed upstream and downstream -/
def distance : ℝ := 18

/-- Time difference between upstream and downstream when rowing at full speed -/
def time_diff_full : ℝ := 3

/-- Time difference between upstream and downstream when rowing at half speed -/
def time_diff_half : ℝ := 2

/-- Equation for full speed rowing -/
def full_speed_equation (r w : ℝ) : Prop :=
  distance / (r - w) - distance / (r + w) = time_diff_full

/-- Equation for half speed rowing -/
def half_speed_equation (r w : ℝ) : Prop :=
  distance / ((r/2) - w) - distance / ((r/2) + w) = time_diff_half

/-- The speed of the river's current is 2 miles per hour -/
theorem river_current_speed (r w : ℝ) 
  (h1 : full_speed_equation r w) 
  (h2 : half_speed_equation r w) : w = 2 := by
  sorry

#check river_current_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l416_41640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_f_deriv_plus_three_halves_l416_41679

open Real

noncomputable def f (x : ℝ) : ℝ := x - log x + (2*x - 1) / x^2

noncomputable def f_deriv (x : ℝ) : ℝ := 1 - 1/x + (2*x^2 - (2*x - 1)*2*x) / x^4

theorem f_greater_than_f_deriv_plus_three_halves :
  ∀ x ∈ Set.Icc 1 2, f x > f_deriv x + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_f_deriv_plus_three_halves_l416_41679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_intersection_l416_41609

def S : Finset ℕ := {0, 1, 2}
def T : Finset ℕ := {0, 3}
def P : Finset ℕ := S ∩ T

theorem number_of_proper_subsets_of_intersection : 
  Finset.card (Finset.powerset P \ {∅, P}) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_proper_subsets_of_intersection_l416_41609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_volume_approximation_l416_41667

/-- Represents a cylindrical vase -/
structure Vase where
  height : ℝ
  diameter : ℝ

/-- Represents a solution of vinegar and water -/
structure Solution where
  vinegarRatio : ℝ
  waterRatio : ℝ

/-- Calculates the volume of vinegar in a partially filled vase -/
noncomputable def vinegarVolume (v : Vase) (s : Solution) (fillRatio : ℝ) : ℝ :=
  let radius := v.diameter / 2
  let filledHeight := v.height * fillRatio
  let totalVolume := Real.pi * radius^2 * filledHeight
  let vinegarProportion := s.vinegarRatio / (s.vinegarRatio + s.waterRatio)
  totalVolume * vinegarProportion

/-- Theorem stating that the volume of vinegar is approximately 7.07 cubic inches -/
theorem vinegar_volume_approximation (v : Vase) (s : Solution) :
  v.height = 12 →
  v.diameter = 3 →
  s.vinegarRatio = 1 →
  s.waterRatio = 3 →
  ∃ ε > 0, |vinegarVolume v s (1/3) - 7.07| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vinegar_volume_approximation_l416_41667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_1000th_1003rd_term_l416_41607

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem difference_1000th_1003rd_term :
  let a₁ := (3 : ℤ)
  let d := (7 : ℤ)
  |arithmeticSequence a₁ d 1003 - arithmeticSequence a₁ d 1000| = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_1000th_1003rd_term_l416_41607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_condition_equivalence_l416_41600

-- Define the floor function as noncomputable
noncomputable def floor (a : ℝ) : ℤ := 
  Int.floor a

-- Theorem statement
theorem floor_condition_equivalence (x : ℤ) : 
  floor ((4 * x - 5 : ℝ) / 5) = -5 ↔ x = -5 ∨ x = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_condition_equivalence_l416_41600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_geometric_implies_arithmetic_convex_l416_41691

/-- A function is quadratic convex on an interval if for all x₁, x₂ in the interval:
    f(√((x₁² + x₂²)/2)) ≤ √((f(x₁)² + f(x₂)²)/2) --/
def QuadraticConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f (Real.sqrt ((x₁^2 + x₂^2)/2)) ≤ Real.sqrt ((f x₁^2 + f x₂^2)/2)

/-- A function is geometric convex on an interval if for all x₁, x₂ in the interval:
    f(√(x₁ * x₂)) ≤ √(f(x₁) * f(x₂)) --/
def GeometricConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f (Real.sqrt (x₁ * x₂)) ≤ Real.sqrt (f x₁ * f x₂)

/-- A function is arithmetic convex on an interval if for all x₁, x₂ in the interval:
    f((x₁ + x₂)/2) ≤ (f(x₁) + f(x₂))/2 --/
def ArithmeticConvex (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f ((x₁ + x₂)/2) ≤ (f x₁ + f x₂)/2

/-- Theorem: If a function is both quadratic convex and geometric convex on an interval,
    then it is also arithmetic convex on that interval. --/
theorem quadratic_geometric_implies_arithmetic_convex
  (f : ℝ → ℝ) (I : Set ℝ)
  (hq : QuadraticConvex f I) (hg : GeometricConvex f I) :
  ArithmeticConvex f I :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_geometric_implies_arithmetic_convex_l416_41691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_3_seconds_l416_41682

noncomputable section

/-- The velocity of a particle at a given time -/
def velocity (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  deriv s t

/-- The equation of motion for the particle -/
def motion (t : ℝ) : ℝ :=
  1 / t^4

theorem velocity_at_3_seconds :
  velocity motion 3 = -4 / 243 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_3_seconds_l416_41682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l416_41616

theorem power_equality (x : ℝ) (h : (128 : ℝ)^3 = (16 : ℝ)^x) : 
  (2 : ℝ)^(-x) = 1 / (2 : ℝ)^(21/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l416_41616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_order_l416_41611

/-- Probability of drawing a trapezoid that is both isosceles and right --/
noncomputable def P₁ : ℝ := 0

/-- Probability of winning 10 major prizes in a row in sports lottery --/
noncomputable def P₃ : ℝ := 1 / 1000000

/-- Probability of getting TTHH when tossing a coin 4 times --/
noncomputable def P₂ : ℝ := 1 / 8

/-- Probability of getting 4 national emblems when tossing a 1 yuan coin 4 times --/
noncomputable def P₄ : ℝ := 1 / 8

/-- Probability that a random 3-digit positive integer n satisfies n³ - n is a multiple of 6 --/
noncomputable def P₅ : ℝ := 1

theorem probability_order : P₁ < P₃ ∧ P₃ < P₂ ∧ P₂ = P₄ ∧ P₄ < P₅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_order_l416_41611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_circles_radius_l416_41603

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the side length a and inscribed circle radius r
variable (a r : ℝ)

-- Define the function for the radius of the equal circles
noncomputable def circle_radius (a r : ℝ) : ℝ := (a * r) / (a + 2 * r)

-- State the theorem
theorem equal_circles_radius (h1 : a > 0) (h2 : r > 0) :
  ∃ x : ℝ, x = circle_radius a r ∧ 
  x > 0 ∧
  -- One circle touches BC and BA
  -- The other circle touches BC and CA
  -- The two circles touch each other
  True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_circles_radius_l416_41603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l416_41689

noncomputable def f (x : ℝ) := Real.log (x - 2)

theorem domain_of_f :
  Set.Ioi 2 = {x : ℝ | ∃ y, f x = y} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l416_41689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_implies_a_value_l416_41648

/-- Given a function f(x) = (a-x)/(x-a-1) with center of symmetry (3, -1), prove that a = 2 -/
theorem symmetry_center_implies_a_value (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = (a - x) / (x - a - 1)) →
  (∃ y : ℝ, ∀ x : ℝ, f (6 - x) = 2 * y - f x) →
  (f 3 = -1) →
  a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_implies_a_value_l416_41648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l416_41612

noncomputable section

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := Real.sqrt 3 * x + y = 0

-- Define the symmetry condition for points A and F
def symmetric_points (xA yA xF yF : ℝ) : Prop :=
  (yA - yF) / (xA - xF) = Real.sqrt 3 / 3 ∧
  ((Real.sqrt 3 * xA + yA) - (Real.sqrt 3 * xF + yF))^2 = 3 * (xA - xF)^2 + (yA - yF)^2

-- State the theorem
theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hF : c > 0)  -- Assume c > 0 for the left focus
  (hA : ∃ xA yA, hyperbola a b xA yA ∧ symmetric_points xA yA (-c) 0) :
  c / a = Real.sqrt 3 + 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l416_41612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_roots_probability_l416_41649

-- Define the random variable ξ following N(2,σ²)
noncomputable def ξ : Real → Real := sorry

-- Define the probability measure
noncomputable def P : Set Real → Real := sorry

-- Define the quadratic function
def f (x ξ : Real) : Real := 2 * x^2 - 4 * x + ξ

-- State the theorem
theorem quadratic_no_roots_probability :
  P {ξ | ∀ x, f x ξ ≠ 0} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_no_roots_probability_l416_41649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_oh_squared_l416_41686

-- Define a triangle with its properties
structure Triangle where
  X : ℝ × ℝ × ℝ
  Y : ℝ × ℝ × ℝ
  Z : ℝ × ℝ × ℝ
  O : ℝ × ℝ × ℝ
  H : ℝ × ℝ × ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  R : ℝ

-- Define functions for CircumcenterOf and OrthocenterOf
def CircumcenterOf (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

def OrthocenterOf (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- Define a function to calculate the squared distance between two points
def distanceSquared (p q : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := p
  let (x₂, y₂, z₂) := q
  (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2

-- State the theorem
theorem triangle_oh_squared (t : Triangle) 
  (h_circumcenter : t.O = CircumcenterOf t.X t.Y t.Z)
  (h_orthocenter : t.H = OrthocenterOf t.X t.Y t.Z)
  (h_side_lengths : t.x^2 + t.y^2 + t.z^2 = 75)
  (h_circumradius : t.R = 5) :
  distanceSquared t.O t.H = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_oh_squared_l416_41686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l416_41626

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line x + (m+1)y = 2-m -/
noncomputable def slope1 (m : ℝ) : ℝ := -(1 / (m + 1))

/-- The slope of the second line mx + 2y = -8 -/
noncomputable def slope2 (m : ℝ) : ℝ := -(m / 2)

theorem perpendicular_lines (m : ℝ) : 
  perpendicular (slope1 m) (slope2 m) → m = -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l416_41626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_solution_l416_41613

def is_valid_permutation (a : Fin 9 → Nat) : Prop :=
  (∀ i j : Fin 9, i ≠ j → a i ≠ a j) ∧
  (∀ n : Nat, n ∈ Finset.range 9 → ∃ i : Fin 9, a i = n + 1)

def satisfies_conditions (a : Fin 9 → Nat) : Prop :=
  (a 0 + a 1 + a 2 + a 3 = a 3 + a 4 + a 5 + a 6) ∧
  (a 3 + a 4 + a 5 + a 6 = a 6 + a 7 + a 8 + a 0) ∧
  (a 0^2 + a 1^2 + a 2^2 + a 3^2 = a 3^2 + a 4^2 + a 5^2 + a 6^2) ∧
  (a 3^2 + a 4^2 + a 5^2 + a 6^2 = a 6^2 + a 7^2 + a 8^2 + a 0^2)

def base_permutation : Fin 9 → Nat
  | 0 => 2
  | 1 => 9
  | 2 => 4
  | 3 => 5
  | 4 => 1
  | 5 => 6
  | 6 => 8
  | 7 => 3
  | 8 => 7

theorem permutation_solution :
  is_valid_permutation base_permutation ∧
  satisfies_conditions base_permutation ∧
  ∀ a : Fin 9 → Nat, is_valid_permutation a → satisfies_conditions a →
    ∃ p : Equiv.Perm (Fin 9), a = base_permutation ∘ p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_solution_l416_41613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_constant_l416_41605

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define a line not passing through origin and not parallel to axes
noncomputable def Line (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the midpoint of two points
noncomputable def Midpoint (x₁ y₁ x₂ y₂ : ℝ) : (ℝ × ℝ) := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

-- Main theorem
theorem slope_product_constant (k b : ℝ) (h₁ : k ≠ 0) (h₂ : b ≠ 0) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧
    y₁ = Line k b x₁ ∧ y₂ = Line k b x₂ ∧
    let (xₘ, yₘ) := Midpoint x₁ y₁ x₂ y₂
    (yₘ / xₘ) * k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_product_constant_l416_41605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l416_41677

/-- Given an ellipse with the following properties:
  * Standard form equation: x²/a² + y²/b² = 1, where a > b > 0
  * F₁ and F₂ are the left and right foci
  * B₁ and B₂ are the upper and lower vertices
  * C is the midpoint of B₁F₂
  * The dot product of vectors B₁F₁ and B₁F₂ equals 2
  * CF₁ is perpendicular to B₁F₂
Prove that the equation of the ellipse is x²/4 + y²/3 = 1 -/
theorem ellipse_equation (a b : ℝ) (F₁ F₂ B₁ B₂ C : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (a * Real.cos t, b * Real.sin t))) ∧
  F₁.1 < 0 ∧ F₁.2 = 0 ∧ F₂.1 > 0 ∧ F₂.2 = 0 ∧
  B₁ = (0, b) ∧ B₂ = (0, -b) ∧
  C = ((F₂.1 + B₁.1) / 2, (F₂.2 + B₁.2) / 2) ∧
  (B₁.1 - F₁.1) * (B₁.1 - F₂.1) + (B₁.2 - F₁.2) * (B₁.2 - F₂.2) = 2 ∧
  (C.1 - F₁.1) * (B₁.1 - F₂.1) + (C.2 - F₁.2) * (B₁.2 - F₂.2) = 0 →
  ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ↔ (x, y) ∈ Set.range (λ t : ℝ ↦ (2 * Real.cos t, Real.sqrt 3 * Real.sin t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l416_41677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_area_is_60_l416_41625

/-- Represents a triangle divided into 9 equal-width stripes parallel to its base -/
structure StripedTriangle where
  /-- The area of the smallest (topmost) stripe -/
  base_area : ℝ
  /-- The total number of stripes -/
  num_stripes : ℕ
  /-- Assertion that the number of stripes is 9 -/
  stripes_count : num_stripes = 9

/-- Calculates the area of a specific stripe given its position from the top -/
def stripe_area (t : StripedTriangle) (position : ℕ) : ℝ :=
  t.base_area * (2 * position - 1)

/-- Calculates the total area of the darkened stripes -/
def darkened_area (t : StripedTriangle) : ℝ :=
  stripe_area t 1 + stripe_area t 3 + stripe_area t 5 + stripe_area t 7 + stripe_area t 9

/-- Calculates the total area of the light-colored stripes -/
def light_area (t : StripedTriangle) : ℝ :=
  stripe_area t 2 + stripe_area t 4 + stripe_area t 6 + stripe_area t 8

/-- Theorem stating that if the darkened area is 135, then the light area is 60 -/
theorem light_area_is_60 (t : StripedTriangle) (h : darkened_area t = 135) :
  light_area t = 60 := by
  sorry

-- Remove the #eval line as it's not necessary and was causing an error

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_area_is_60_l416_41625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l416_41655

/-- Two lines are parallel if their direction vectors are linearly dependent -/
def LineParallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ (k * x, k * y) ∈ l₂

theorem parallel_lines_a_value (a : ℝ) : 
  let l₁ := {(x, y) : ℝ × ℝ | a * x - y + a = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (2 * a - 3) * x + a * y - a = 0}
  (LineParallel l₁ l₂ ∧ l₁ ≠ l₂) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l416_41655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_problem_l416_41633

def quarter_value : ℚ := 1/4

def oliver_quarters_given (initial_cash : ℚ) (initial_quarters : ℕ) 
  (cash_given : ℚ) (final_total : ℚ) : ℕ :=
  let initial_total := initial_cash + (initial_quarters : ℚ) * quarter_value
  let remaining_after_cash := initial_total - cash_given
  let quarters_value_given := remaining_after_cash - final_total
  (quarters_value_given / quarter_value).floor.toNat

theorem oliver_problem :
  oliver_quarters_given 40 200 5 55 = 120 := by
  sorry

#eval oliver_quarters_given 40 200 5 55

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oliver_problem_l416_41633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_cost_l416_41693

/-- The optimal distance that minimizes total cost -/
noncomputable def optimal_distance : ℝ := 2.5

/-- Monthly rent cost as a function of distance -/
noncomputable def rent_cost (k₁ : ℝ) (x : ℝ) : ℝ := k₁ / x

/-- Monthly transportation cost as a function of distance -/
def transport_cost (k₂ : ℝ) (x : ℝ) : ℝ := k₂ * x

/-- Total monthly cost as a function of distance -/
noncomputable def total_cost (k₁ k₂ : ℝ) (x : ℝ) : ℝ := rent_cost k₁ x + transport_cost k₂ x

theorem optimal_distance_minimizes_cost (k₁ k₂ : ℝ) :
  k₁ > 0 → k₂ > 0 →
  rent_cost k₁ 5 = 1 →
  transport_cost k₂ 5 = 4 →
  ∀ x > 0, total_cost k₁ k₂ optimal_distance ≤ total_cost k₁ k₂ x := by
  sorry

#check optimal_distance_minimizes_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_cost_l416_41693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centaur_chess_second_player_wins_l416_41601

/-- Represents a position on the chess board -/
structure Position where
  x : Nat
  y : Nat
deriving Repr

/-- Represents the game state -/
structure GameState where
  board_size : Nat
  removed_rectangles : List (Position × Position)
  centaur_position : Position
  current_player : Nat
deriving Repr

/-- Checks if a position is within the board and not in a removed rectangle -/
def is_valid_position (state : GameState) (pos : Position) : Prop :=
  pos.x < state.board_size ∧ pos.y < state.board_size ∧
  ∀ (rect : Position × Position), rect ∈ state.removed_rectangles →
    ¬(rect.1.x ≤ pos.x ∧ pos.x < rect.2.x ∧ rect.1.y ≤ pos.y ∧ pos.y < rect.2.y)

/-- Checks if a move is valid for the centaur -/
def is_valid_move (state : GameState) (new_pos : Position) : Prop :=
  is_valid_position state new_pos ∧
  ((new_pos.x = state.centaur_position.x ∧ new_pos.y = state.centaur_position.y + 1) ∨
   (new_pos.x = state.centaur_position.x - 1 ∧ new_pos.y = state.centaur_position.y) ∨
   (new_pos.x = state.centaur_position.x + 1 ∧ new_pos.y = state.centaur_position.y + 1))

/-- Defines the winning strategy for the second player -/
def second_player_wins (initial_state : GameState) : Prop :=
  ∃ (strategy : GameState → Position),
    ∀ (game : Nat → GameState),
      game 0 = initial_state →
      (∀ n, (game (n + 1)).centaur_position = 
        if (game n).current_player = 1 
        then strategy (game n) 
        else (game n).centaur_position) →
      ∃ k, ¬∃ pos, is_valid_move (game k) pos ∧ (game k).current_player = 1

theorem centaur_chess_second_player_wins :
  let initial_state : GameState := {
    board_size := 1000,
    removed_rectangles := [
      ({x := 0, y := 0}, {x := 2, y := 994}),
      ({x := 998, y := 0}, {x := 1000, y := 994}),
      ({x := 0, y := 996}, {x := 2, y := 1000}),
      ({x := 998, y := 996}, {x := 1000, y := 1000})
    ],
    centaur_position := {x := 500, y := 496},
    current_player := 1
  }
  second_player_wins initial_state := by
  sorry

#eval Position.mk 1 2
#eval GameState.mk 1000 [] (Position.mk 500 496) 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centaur_chess_second_player_wins_l416_41601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_correct_l416_41620

/-- The sum of the coefficients of terms containing y in (2x + 3y + 4)(5x + 6y + 7) -/
def sum_of_y_coefficients : ℕ := 90

/-- The expression to be expanded -/
def expression (x y : ℝ) : ℝ := (2*x + 3*y + 4) * (5*x + 6*y + 7)

/-- Theorem stating that the sum of coefficients of terms containing y in the expansion of the expression is equal to sum_of_y_coefficients -/
theorem sum_of_y_coefficients_correct :
  ∃ (a b c : ℝ), (∀ x y, expression x y = 10*x^2 + a*x*y + 34*x + b*y^2 + c*y + 28) ∧ a + b + c = sum_of_y_coefficients :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_correct_l416_41620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l416_41619

/-- Represents a trapezoid with given base and side lengths -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  side1 : ℝ
  side2 : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  ((t.base1 + t.base2) / 2) * Real.sqrt (((t.side1^2 - (t.base2 - t.base1)^2 / 4) + (t.side2^2 - (t.base2 - t.base1)^2 / 4)) / 2)

/-- Theorem: The area of a trapezoid with bases 4 and 7, and side lengths 4 and 5, is 22 -/
theorem trapezoid_area_theorem : 
  let t : Trapezoid := { base1 := 4, base2 := 7, side1 := 4, side2 := 5 }
  trapezoidArea t = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l416_41619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_points_l416_41628

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin (2 * x) + 3 * Real.cos x ^ 2

theorem f_minimum_value_and_points :
  (∃ (min : ℝ), ∀ x, f x ≥ min ∧ min = 2 - Real.sqrt 2) ∧
  (∀ x, f x = 2 - Real.sqrt 2 ↔ ∃ k : ℤ, x = k * Real.pi - 3 * Real.pi / 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_and_points_l416_41628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l416_41647

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 2 else 2^(-x) - 2

theorem solution_set_of_inequality (h1 : ∀ x, f (-x) = f x) 
  (h2 : ∀ x ≥ 0, f x = 2^x - 2) :
  {x : ℝ | f (x - 1) ≤ 2} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l416_41647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_arrangement_count_l416_41662

inductive RibbonColor
  | Red
  | Blue
  | Yellow

def arrange_ribbons (n : ℕ) (colors : Fin n → RibbonColor) : ℕ :=
  sorry

theorem ribbon_arrangement_count : 
  arrange_ribbons 4 (λ i => match i with
    | 0 => RibbonColor.Red
    | 1 => RibbonColor.Blue
    | 2 => RibbonColor.Yellow
    | 3 => RibbonColor.Yellow
    | _ => RibbonColor.Red  -- Unreachable, but needed for exhaustiveness
  ) = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_arrangement_count_l416_41662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_triangles_with_perimeter_12_l416_41687

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of an integer triangle -/
def perimeter (t : IntTriangle) : ℕ :=
  t.a + t.b + t.c

/-- Predicate for triangles with perimeter 12 -/
def has_perimeter_12 (t : IntTriangle) : Prop :=
  perimeter t = 12

/-- The set of all triangles with integer sides and perimeter 12 -/
def triangles_with_perimeter_12 : Set IntTriangle :=
  {t : IntTriangle | has_perimeter_12 t}

/-- The theorem stating that there are exactly 3 distinct triangles with integer sides and perimeter 12 -/
theorem exactly_three_triangles_with_perimeter_12 :
  ∃ (s : Finset IntTriangle), s.card = 3 ∧ ∀ t : IntTriangle, t ∈ triangles_with_perimeter_12 ↔ t ∈ s :=
sorry

#check exactly_three_triangles_with_perimeter_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_triangles_with_perimeter_12_l416_41687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_twelfth_term_is_one_l416_41650

def a : ℕ → ℚ
  | 0 => 1  -- Adding the case for 0
  | 1 => 1
  | (n + 1) => if a n + 2 = 0 then 2 * a n else
                 if (2 * a n) * (a n + 2) = 1 then 1 / (a n + 2)
                 else 2 * a n

theorem sequence_property : ∀ n : ℕ, n ≥ 1 → (a (n + 1) - 2 * a n) * (a (n + 1) - 1 / (a n + 2)) = 0 := by
  sorry

theorem twelfth_term_is_one : a 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_twelfth_term_is_one_l416_41650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l416_41697

-- Define the curves
noncomputable def C1 (t : ℝ) : ℝ × ℝ := (-1 + Real.sqrt 2 * t, 1 + Real.sqrt 2 * t)

def C2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

def C3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}

-- Define the intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_sum : 
  A ∈ C2 ∧ A ∈ C3 ∧ B ∈ C2 ∧ B ∈ C3 → 
  1 / Real.sqrt (A.1^2 + A.2^2) + 1 / Real.sqrt (B.1^2 + B.2^2) = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l416_41697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_volume_from_concentration_change_l416_41624

/-- Represents the volume of a bottle in liters -/
def bottle_volume : ℝ → Prop := sorry

/-- Represents the process of pouring out 1 liter and refilling with water -/
noncomputable def pour_and_refill (initial_concentration : ℝ) (volume : ℝ) : ℝ :=
  (volume - 1) * initial_concentration / volume

/-- The theorem stating the bottle volume given the concentration change -/
theorem bottle_volume_from_concentration_change :
  ∀ x : ℝ,
  x > 0 →
  pour_and_refill (pour_and_refill 0.36 x) x = 0.01 →
  bottle_volume x →
  x = 1.2 := by
    sorry

#check bottle_volume_from_concentration_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottle_volume_from_concentration_change_l416_41624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_complementary_l416_41654

/-- Represents the group of students -/
structure StudentGroup where
  boys : ℕ
  girls : ℕ

/-- Represents an event in the sample space -/
structure Event where
  boys_chosen : ℕ
  girls_chosen : ℕ

/-- The sample space of all possible outcomes -/
def sample_space (g : StudentGroup) : Set Event :=
  {e : Event | e.boys_chosen + e.girls_chosen = 2 ∧ 
               e.boys_chosen ≤ g.boys ∧ 
               e.girls_chosen ≤ g.girls}

/-- The event "at least 1 boy is chosen" -/
def at_least_one_boy (g : StudentGroup) : Set Event :=
  {e ∈ sample_space g | e.boys_chosen ≥ 1}

/-- The event "all girls are chosen" -/
def all_girls (g : StudentGroup) : Set Event :=
  {e ∈ sample_space g | e.girls_chosen = g.girls}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive (A B : Set Event) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary if they are mutually exclusive and their union is the entire sample space -/
def complementary (g : StudentGroup) (A B : Set Event) : Prop :=
  mutually_exclusive A B ∧ A ∪ B = sample_space g

theorem events_mutually_exclusive_but_not_complementary (g : StudentGroup) 
  (h1 : g.boys = 3) 
  (h2 : g.girls = 2) : 
  mutually_exclusive (at_least_one_boy g) (all_girls g) ∧
  ¬ complementary g (at_least_one_boy g) (all_girls g) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_but_not_complementary_l416_41654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_adding_one_l416_41669

noncomputable section

variable (x₁ x₂ x₃ : ℝ)

noncomputable def variance (x₁ x₂ x₃ : ℝ) : ℝ := (1/3) * (x₁^2 + x₂^2 + x₃^2 - 12)

noncomputable def average (x₁ x₂ x₃ : ℝ) : ℝ := (x₁ + x₂ + x₃) / 3

theorem average_after_adding_one (h : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_variance : variance x₁ x₂ x₃ = (1/3) * (x₁^2 + x₂^2 + x₃^2 - 12)) :
  average (x₁ + 1) (x₂ + 1) (x₃ + 1) = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_adding_one_l416_41669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_f_l416_41668

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - 2 * a * x + a

-- Theorem for the tangent line equation when a = 2
theorem tangent_line_at_one (x y : ℝ) :
  f 2 1 = -2 ∧ (deriv (f 2)) 1 = -2 → 2 * x + y = 0 := by sorry

-- Theorem for the monotonicity of f
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, (deriv (f a)) x > 0) ∧
  (a > 0 → (∀ x ∈ Set.Ioo 0 (1/a), (deriv (f a)) x > 0) ∧
           (∀ x ∈ Set.Ioi (1/a), (deriv (f a)) x < 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_monotonicity_of_f_l416_41668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l416_41606

-- Define the quadrilateral ABCD and points K, L, M
variable (A B C D K L M : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def divides_side_in_ratio (P Q R : ℝ × ℝ) (ratio : ℝ) : Prop := sorry

noncomputable def circumradius (P Q R : ℝ × ℝ) : ℝ := sorry

def distance (P Q : ℝ × ℝ) : ℝ := sorry

noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_quadrilateral 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_ratio_K : divides_side_in_ratio A K B (1/3))
  (h_ratio_L : divides_side_in_ratio B L C (1/3))
  (h_ratio_M : divides_side_in_ratio C M D (1/3))
  (h_radius : circumradius K L M = 5/2)
  (h_KL : distance K L = 4)
  (h_LM : distance L M = 3)
  (h_KM_less : distance K M < distance K L) :
  area_quadrilateral A B C D = 189/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l416_41606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_cos_and_point_l416_41646

theorem tan_value_from_cos_and_point (α : ℝ) (m : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = m ∧ P.2 = 1) →
  Real.cos α = -1/3 →
  Real.tan α = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_cos_and_point_l416_41646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l416_41663

noncomputable section

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def angle (a b : V) : ℝ := Real.arccos ((inner a b) / (norm a * norm b))

theorem vector_angle_problem (a b : V) 
  (ha : norm a = Real.sqrt 2)
  (hb : norm b = 1) :
  (norm (a - b) = 2 → Real.cos (angle a b) = -Real.sqrt 2 / 4) ∧
  ((∀ x : ℝ, norm (a + x • b) ≥ norm (a + b)) → angle a b = 3 * Real.pi / 4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l416_41663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeA_rate_correct_l416_41699

/-- The rate at which Pipe A fills the tank -/
noncomputable def pipeA_rate : ℚ := 200

/-- The rate at which Pipe B fills the tank -/
def pipeB_rate : ℚ := 50

/-- The rate at which Pipe C drains the tank -/
def pipeC_rate : ℚ := 25

/-- The capacity of the tank in liters -/
def tank_capacity : ℚ := 5000

/-- The total time taken to fill the tank in minutes -/
def total_time : ℚ := 100

/-- The duration of one cycle in minutes -/
def cycle_duration : ℚ := 5

/-- The number of complete cycles -/
noncomputable def num_cycles : ℚ := total_time / cycle_duration

/-- Theorem stating that the given rate of Pipe A satisfies the problem conditions -/
theorem pipeA_rate_correct : 
  num_cycles * (pipeA_rate * 1 + pipeB_rate * 2 - pipeC_rate * 2) = tank_capacity :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipeA_rate_correct_l416_41699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l416_41659

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Theorem statement -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A : Point) :
  (∃ x y : ℝ, x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ A.x = x ∧ A.y = y) →  -- A is on the hyperbola
  (A.y - F₂.y) / (A.x - F₂.x) = 1 →  -- Line F₂A has inclination angle π/4
  (A.x - F₁.x)^2 + (A.y - F₁.y)^2 = (A.x - F₂.x)^2 + (A.y - F₂.y)^2 →  -- F₁A = F₂A
  (F₂.x - F₁.x)^2 + (F₂.y - F₁.y)^2 = 2 * ((A.x - F₁.x)^2 + (A.y - F₁.y)^2) →  -- F₁F₂ = √2 * F₁A
  eccentricity h = Real.sqrt 2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l416_41659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l416_41618

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := (a - 3) * Real.sin x + b
noncomputable def g (a b x : ℝ) : ℝ := a + b * Real.cos x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x, g a b x ≥ -1) →  -- minimum value of g is -1
  (g a b Real.pi = -1) →  -- g achieves its minimum at π
  (Real.sin b > 0) →  -- sin b > 0
  (a = 3 ∧ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l416_41618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_30_students_max_9_l416_41656

/-- Given a total number of students and a maximum group size, 
    calculates the minimum number of groups needed, ensuring each group has an odd number of students. -/
def min_groups (total_students : ℕ) (max_group_size : ℕ) : ℕ :=
  let odd_divisors := (Finset.range (max_group_size + 1)).filter (λ x => x % 2 = 1 ∧ x > 0 ∧ total_students % x = 0)
  if h : odd_divisors.Nonempty then
    (total_students + odd_divisors.max' h - 1) / odd_divisors.max' h
  else
    0

/-- The theorem stating that the minimum number of groups for 30 students with a maximum group size of 9 is 4. -/
theorem min_groups_for_30_students_max_9 :
  min_groups 30 9 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_groups_for_30_students_max_9_l416_41656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_y_l416_41608

/-- The function y in terms of x -/
noncomputable def y (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

/-- The theorem stating that y = -1/3 is unattainable -/
theorem unattainable_y (x : ℝ) (h : x ≠ -4/3) : 
  y x ≠ -1/3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_y_l416_41608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_covered_circle_radius_theorem_l416_41661

/-- The radius of the largest circle that can be covered by three circles with radii R₁, R₂, R₃ -/
noncomputable def largest_covered_circle_radius (R₁ R₂ R₃ : ℝ) : ℝ :=
  max (max (2 * R₁ / Real.sqrt 3) (2 * R₂ / Real.sqrt 3)) 
      (max (2 * R₃ / Real.sqrt 3) (max R₁ (max R₂ R₃)))

/-- Definition of a circle in 2D real space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is inscribed in a set of circles -/
def Circle.isInscribedIn (c : Circle) (cs : Set Circle) : Prop :=
  ∀ c' ∈ cs, (c.center.1 - c'.center.1)^2 + (c.center.2 - c'.center.2)^2 ≤ (c'.radius - c.radius)^2

/-- Theorem: The radius of the largest circle that can be covered by three circles
    with radii R₁, R₂, R₃ is given by largest_covered_circle_radius R₁ R₂ R₃ -/
theorem largest_covered_circle_radius_theorem (R₁ R₂ R₃ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : R₃ > 0) :
  ∃ (r : ℝ), r = largest_covered_circle_radius R₁ R₂ R₃ ∧ 
  (∀ (s : ℝ), (∃ (c₁ c₂ c₃ : Circle), 
    c₁.radius = R₁ ∧ c₂.radius = R₂ ∧ c₃.radius = R₃ ∧
    (∃ (c : Circle), c.radius = s ∧ c.isInscribedIn {c₁, c₂, c₃})) → 
  s ≤ r) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_covered_circle_radius_theorem_l416_41661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_with_specific_distances_l416_41652

/-- A line in the coordinate plane. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point (x, y) to a line ax + by + c = 0. -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  (abs (l.a * x + l.b * y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- The statement of the problem. -/
theorem two_lines_with_specific_distances : 
  ∃! (s : Finset Line), 
    s.card = 2 ∧ 
    (∀ l ∈ s, distancePointToLine 1 2 l = 1 ∧ distancePointToLine 3 1 l = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_with_specific_distances_l416_41652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_f_max_value_implies_m_equals_one_smallest_positive_period_monotonic_decreasing_interval_l416_41678

/-- Vector a defined as (2 * cos^2(x), 1) -/
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x)^2, 1)

/-- Vector b defined as (1, 2 * √3 * sin(x) * cos(x) + m) -/
noncomputable def b (x m : ℝ) : ℝ × ℝ := (1, 2 * Real.sqrt 3 * Real.sin x * Real.cos x + m)

/-- Function f(x) representing the dot product of vectors a and b -/
noncomputable def f (x m : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + m + 1

/-- Theorem stating the equivalence of the dot product and the function f(x) -/
theorem dot_product_equals_f (x m : ℝ) : 
  (a x).1 * (b x m).1 + (a x).2 * (b x m).2 = f x m := by sorry

/-- Theorem stating that if the maximum value of f(x) is 4 for x in [0, π/2], then m = 1 -/
theorem max_value_implies_m_equals_one (m : ℝ) 
  (h : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≤ 4) 
  (h_max : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x m = 4) : 
  m = 1 := by sorry

/-- Theorem stating that the smallest positive period of f(x) is π -/
theorem smallest_positive_period (x m : ℝ) : 
  ∀ t > 0, (∀ x, f (x + t) m = f x m) → t ≥ Real.pi := by sorry

/-- Theorem stating the monotonically decreasing interval of f(x) -/
theorem monotonic_decreasing_interval (x m k : ℝ) : 
  StrictMonoOn (fun x => f x m) (Set.Icc (Real.pi / 6 + k * Real.pi) (2 * Real.pi / 3 + k * Real.pi)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_f_max_value_implies_m_equals_one_smallest_positive_period_monotonic_decreasing_interval_l416_41678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l416_41614

-- Define an odd function on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x ≥ 0
noncomputable def f_nonneg (b : ℝ) (x : ℝ) : ℝ :=
  (2 : ℝ)^x + 2*x + b

-- Main theorem
theorem odd_function_value_at_negative_one 
  (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = f_nonneg b x) :
  f (-1) = -3 := by
  sorry

#check odd_function_value_at_negative_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_at_negative_one_l416_41614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_prob_15_yuan_combined_fees_30_yuan_prob_l416_41629

-- Define the parking fee structure
def parkingFee (hours : ℕ) : ℕ :=
  if hours ≤ 1 then 5 else 5 + 10 * (hours - 1)

-- Define the probabilities for Person A
def probUpTo1Hour : ℚ := 1/2
def probMoreThan15Yuan : ℚ := 1/6

-- Define the maximum parking duration
def maxParkingHours : ℕ := 4

-- Theorem for Person A's probability of paying exactly 15 yuan
theorem person_a_prob_15_yuan :
  1 - (probUpTo1Hour + probMoreThan15Yuan) = 1/3 := by sorry

-- Define the possible parking fees
def possibleFees : List ℕ := [5, 15, 25, 35]

-- Theorem for the probability of combined parking fees totaling 30 yuan
theorem combined_fees_30_yuan_prob :
  (((List.countP (fun pair => pair.1 + pair.2 = 30) 
    (List.product possibleFees possibleFees)) : ℚ) / 
    ((List.length (List.product possibleFees possibleFees)) : ℚ)) = 3/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_a_prob_15_yuan_combined_fees_30_yuan_prob_l416_41629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_half_hour_l416_41632

/-- Represents the journey of Little Red Riding Hood --/
structure Journey where
  distance_one_way : ℝ
  speed_flat : ℝ
  speed_uphill : ℝ
  speed_downhill : ℝ

/-- Calculates the total time for the journey --/
noncomputable def total_time (j : Journey) : ℝ :=
  2 * j.distance_one_way / j.speed_flat

/-- Theorem stating that the total time for the journey is 0.5 hours --/
theorem journey_time_is_half_hour (j : Journey) 
  (h1 : j.distance_one_way = 1)
  (h2 : j.speed_flat = 4)
  (h3 : j.speed_uphill = 3)
  (h4 : j.speed_downhill = 6) : 
  total_time j = 0.5 := by
  sorry

#check journey_time_is_half_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_half_hour_l416_41632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_l416_41658

/-- The modulus used in the problem -/
def m : ℕ := 17

/-- The x-coefficient in the congruence -/
def a : ℕ := 7

/-- The y-coefficient in the congruence -/
def b : ℕ := 3

/-- The constant term in the congruence -/
def c : ℕ := 2

/-- The x-intercept of the congruence -/
def x₀ : ℕ := 10

/-- The y-intercept of the congruence -/
def y₀ : ℕ := 7

/-- Theorem stating that the sum of x-intercept and y-intercept is 17 -/
theorem intercept_sum :
  (a * x₀ ≡ c [MOD m]) ∧
  (b * y₀ ≡ m - c [MOD m]) ∧
  x₀ < m ∧ y₀ < m →
  x₀ + y₀ = m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_sum_l416_41658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l416_41688

noncomputable def f (x : ℝ) := Real.cos x ^ 4 - Real.sin x ^ 4

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l416_41688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l416_41685

theorem cube_root_simplification :
  (27 - 8 : ℝ) ^ (1/3) * (9 - 27 ^ (1/3) : ℝ) ^ (1/3) = 114 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l416_41685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41643

noncomputable def f (x : ℝ) := Real.cos x ^ 2

theorem f_properties :
  (∀ x, f (x - π) = f x) ∧
  (∀ x, f (π/2 + x) = f (π/2 - x)) ∧
  (f (π/2 + π/4) = 0) ∧
  (∀ x ∈ Set.Ioo (π/2) π, ∀ y ∈ Set.Ioo (π/2) π, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l416_41643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pawn_movement_impossible_l416_41635

/-- A position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  h_x : x ≤ 8
  h_y : y ≤ 8

/-- The set of initial pawn positions -/
def initial_positions : Set Position := {
  ⟨1, 1, by norm_num, by norm_num⟩, ⟨1, 2, by norm_num, by norm_num⟩, ⟨1, 3, by norm_num, by norm_num⟩,
  ⟨2, 1, by norm_num, by norm_num⟩, ⟨2, 2, by norm_num, by norm_num⟩, ⟨2, 3, by norm_num, by norm_num⟩,
  ⟨3, 1, by norm_num, by norm_num⟩, ⟨3, 2, by norm_num, by norm_num⟩, ⟨3, 3, by norm_num, by norm_num⟩
}

/-- The set of target pawn positions -/
def target_positions : Set Position := {
  ⟨6, 6, by norm_num, by norm_num⟩, ⟨6, 7, by norm_num, by norm_num⟩, ⟨6, 8, by norm_num, by norm_num⟩,
  ⟨7, 6, by norm_num, by norm_num⟩, ⟨7, 7, by norm_num, by norm_num⟩, ⟨7, 8, by norm_num, by norm_num⟩,
  ⟨8, 6, by norm_num, by norm_num⟩, ⟨8, 7, by norm_num, by norm_num⟩, ⟨8, 8, by norm_num, by norm_num⟩
}

/-- A symmetric jump preserves the parity of coordinates -/
def symmetric_jump (p q : Position) : Prop :=
  (p.x + q.x) % 2 = 0 ∧ (p.y + q.y) % 2 = 0

/-- The main theorem: it's impossible to transform initial_positions to target_positions using only symmetric jumps -/
theorem pawn_movement_impossible : ¬∃ (f : Position → Position),
  (∀ p ∈ initial_positions, f p ∈ target_positions) ∧
  (∀ p q : Position, symmetric_jump p (f q)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pawn_movement_impossible_l416_41635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_condition_l416_41623

/-- If the equation √(x + a√x + b) = c has infinitely many solutions, 
    then a = -2c, b = c², and c > 0 -/
theorem infinite_solutions_condition (a b c : ℝ) : 
  (∀ x : ℝ, Real.sqrt (x + a * Real.sqrt x + b) = c) → 
  (a = -2 * c ∧ b = c^2 ∧ c > 0) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_condition_l416_41623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_perpendicular_lines_l416_41642

/-- The slope of the first line -/
noncomputable def m₁ : ℝ := 3

/-- The y-intercept of the first line -/
noncomputable def b₁ : ℝ := -4

/-- The x-coordinate of the given point -/
noncomputable def x₀ : ℝ := 3

/-- The y-coordinate of the given point -/
noncomputable def y₀ : ℝ := 2

/-- The slope of the perpendicular line -/
noncomputable def m₂ : ℝ := -1 / m₁

/-- The x-coordinate of the intersection point -/
noncomputable def x_intersect : ℝ := 27 / 10

/-- The y-coordinate of the intersection point -/
noncomputable def y_intersect : ℝ := 41 / 10

theorem intersection_point_of_perpendicular_lines :
  ∃ (x y : ℝ),
    (y = m₁ * x + b₁) ∧
    (y - y₀ = m₂ * (x - x₀)) ∧
    x = x_intersect ∧
    y = y_intersect :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_perpendicular_lines_l416_41642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l416_41615

/-- A predicate stating that three real numbers form a geometric sequence -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem geometric_sequence_middle_term (a : ℝ) :
  is_geometric_sequence 1 a 2 → (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l416_41615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_of_powers_min_value_is_four_min_value_achieved_l416_41670

theorem min_value_of_sum_of_powers (a b : ℝ) (h : a + b = 2) :
  ∀ x y : ℝ, x + y = 2 → (2 : ℝ)^x + (2 : ℝ)^y ≥ (2 : ℝ)^a + (2 : ℝ)^b :=
by
  sorry

theorem min_value_is_four (a b : ℝ) (h : a + b = 2) :
  (2 : ℝ)^a + (2 : ℝ)^b ≥ 4 :=
by
  sorry

theorem min_value_achieved (a b : ℝ) (h : a + b = 2) :
  ∃ x y : ℝ, x + y = 2 ∧ (2 : ℝ)^x + (2 : ℝ)^y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_of_powers_min_value_is_four_min_value_achieved_l416_41670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l416_41639

/-- Calculates the annual interest rate given initial deposit, final value, compounding frequency, and time period. -/
noncomputable def calculate_interest_rate (initial_deposit : ℝ) (final_value : ℝ) (compounding_frequency : ℕ) (time_period : ℝ) : ℝ :=
  compounding_frequency * ((final_value / initial_deposit) ^ (1 / (compounding_frequency * time_period)) - 1)

/-- Theorem stating that the calculated interest rate for the given problem is approximately 5.84% -/
theorem interest_rate_calculation :
  let initial_deposit : ℝ := 650
  let final_value : ℝ := 914.6152747265625
  let compounding_frequency : ℕ := 12
  let time_period : ℝ := 7
  let calculated_rate := calculate_interest_rate initial_deposit final_value compounding_frequency time_period
  abs (calculated_rate - 0.0584) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l416_41639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_and_reciprocal_l416_41621

theorem cube_root_sum_and_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  (x ^ (1/3 : ℝ)) + 1 / (x ^ (1/3 : ℝ)) = 53 ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_and_reciprocal_l416_41621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l416_41604

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^(2*x + 6) = (27 : ℝ)^(x + 1) ∧ x = 3 := by
  use 3
  constructor
  · -- Prove the equation holds for x = 3
    simp [Real.rpow_add, Real.rpow_mul]
    norm_num
  · -- Prove x = 3
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l416_41604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_ride_time_l416_41602

/-- Represents the escalator problem with Clea's walking times -/
structure EscalatorProblem where
  /-- Time taken to walk down when escalator is not moving (in seconds) -/
  stationary_time : ℚ
  /-- Time taken to walk down when escalator is moving (in seconds) -/
  moving_time : ℚ

/-- Calculates the time taken to ride the escalator without walking -/
def ride_time (p : EscalatorProblem) : ℚ :=
  p.stationary_time * p.moving_time / (p.stationary_time - p.moving_time)

/-- Theorem stating that given the problem conditions, the ride time is 50 seconds -/
theorem clea_escalator_ride_time :
  ∀ (p : EscalatorProblem),
  p.stationary_time = 75 ∧ p.moving_time = 30 →
  ride_time p = 50 := by
  intro p ⟨h1, h2⟩
  simp [ride_time, h1, h2]
  -- The proof is completed by normalization
  sorry

#eval ride_time { stationary_time := 75, moving_time := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clea_escalator_ride_time_l416_41602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_cosine_function_l416_41671

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.cos (Real.pi * x / 3 + φ)

theorem phase_shift_of_cosine_function 
  (φ : ℝ) 
  (h1 : ∃ (x : ℝ), f x φ = f (4 - x) φ ∧ f x φ = 0) 
  (h2 : f 1 φ > f 3 φ) : 
  φ = 2 * Real.pi / 3 :=
sorry

#check phase_shift_of_cosine_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_cosine_function_l416_41671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_rhombuses_l416_41617

/-- Represents a rhombus with given diagonal lengths -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Calculates the area of the common part of two rhombuses -/
noncomputable def commonArea (r : Rhombus) : ℝ :=
  let sideLength := Real.sqrt ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2)
  let rhombusArea := r.diagonal1 * r.diagonal2 / 2
  let triangleArea := r.diagonal1 * r.diagonal2 / 8
  let bisectorLength := 6 * Real.sqrt 2 / 5
  let smallTriangleArea := (r.diagonal2 - r.diagonal1) / 2 * (bisectorLength / Real.sqrt 2) / 2
  4 * (triangleArea - smallTriangleArea)

/-- Theorem stating that the area of the common part of two rhombuses is 9.6 cm² -/
theorem common_area_of_rhombuses (r : Rhombus) 
    (h1 : r.diagonal1 = 4)
    (h2 : r.diagonal2 = 6) : 
  commonArea r = 9.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_rhombuses_l416_41617
