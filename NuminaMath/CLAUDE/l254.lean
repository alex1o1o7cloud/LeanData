import Mathlib

namespace NUMINAMATH_CALUDE_factorization_equality_l254_25499

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l254_25499


namespace NUMINAMATH_CALUDE_last_row_value_l254_25460

/-- Represents a triangular table with the given properties -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowProperty (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (from the second row onwards) is the sum of the two elements directly above it -/
def SumProperty (t : TriangularTable 100) : Prop :=
  ∀ i j : Fin 100, i > 0 → j < i → t i j = t (i-1) j + t (i-1) (j+1)

/-- The last row contains only one element -/
def LastRowProperty (t : TriangularTable 100) : Prop :=
  t 99 0 = t 99 0  -- This is always true, but it ensures the element exists

/-- The main theorem: the value in the last row is 101 × 2^98 -/
theorem last_row_value (t : TriangularTable 100) 
  (h1 : FirstRowProperty t) 
  (h2 : SumProperty t) 
  (h3 : LastRowProperty t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_last_row_value_l254_25460


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l254_25450

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
theorem parabola_kite_sum (a b : ℝ) : 
  (∀ x y : ℝ, (y = a * x^2 + 4 ∨ y = 6 - b * x^2) → 
    (x = 0 ∨ y = 0)) →  -- intersect coordinate axes
  (∃! p q r s : ℝ × ℝ, 
    (p.2 = a * p.1^2 + 4 ∨ p.2 = 6 - b * p.1^2) ∧
    (q.2 = a * q.1^2 + 4 ∨ q.2 = 6 - b * q.1^2) ∧
    (r.2 = a * r.1^2 + 4 ∨ r.2 = 6 - b * r.1^2) ∧
    (s.2 = a * s.1^2 + 4 ∨ s.2 = 6 - b * s.1^2) ∧
    (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧
    (r.1 = 0 ∨ r.2 = 0) ∧ (s.1 = 0 ∨ s.2 = 0)) →  -- exactly four intersection points
  (∃ d₁ d₂ : ℝ, d₁ * d₂ / 2 = 18) →  -- kite area is 18
  a + b = 2/81 :=
by sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l254_25450


namespace NUMINAMATH_CALUDE_nuts_per_bag_l254_25451

theorem nuts_per_bag (bags : ℕ) (students : ℕ) (nuts_per_student : ℕ) 
  (h1 : bags = 65)
  (h2 : students = 13)
  (h3 : nuts_per_student = 75) :
  (students * nuts_per_student) / bags = 15 := by
sorry

end NUMINAMATH_CALUDE_nuts_per_bag_l254_25451


namespace NUMINAMATH_CALUDE_six_disks_common_point_implies_center_inside_l254_25435

-- Define a disk in 2D space
structure Disk :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define what it means for a point to be inside a disk
def isInside (p : ℝ × ℝ) (d : Disk) : Prop :=
  let (x, y) := p
  let (cx, cy) := d.center
  (x - cx)^2 + (y - cy)^2 < d.radius^2

-- Define a set of six disks
def SixDisks := Fin 6 → Disk

-- The theorem statement
theorem six_disks_common_point_implies_center_inside
  (disks : SixDisks)
  (common_point : ℝ × ℝ)
  (h : ∀ i : Fin 6, isInside common_point (disks i)) :
  ∃ i j : Fin 6, i ≠ j ∧ isInside (disks j).center (disks i) :=
sorry

end NUMINAMATH_CALUDE_six_disks_common_point_implies_center_inside_l254_25435


namespace NUMINAMATH_CALUDE_terms_difference_l254_25484

theorem terms_difference (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end NUMINAMATH_CALUDE_terms_difference_l254_25484


namespace NUMINAMATH_CALUDE_prisoners_puzzle_solution_l254_25480

-- Define the hair colors
inductive HairColor
| Blonde
| Red
| Brunette

-- Define the prisoners
inductive Prisoner
| P1
| P2
| P3
| P4
| P5

-- Define the ladies
structure Lady where
  name : String
  hairColor : HairColor

-- Define the statement of a prisoner
structure Statement where
  prisoner : Prisoner
  ownLady : Lady
  neighborLadies : List HairColor

-- Define the truthfulness of a prisoner
inductive Truthfulness
| AlwaysTruth
| AlwaysLie
| Variable

-- Define the problem setup
def prisonerSetup : List (Prisoner × Truthfulness) := 
  [(Prisoner.P1, Truthfulness.AlwaysTruth),
   (Prisoner.P2, Truthfulness.AlwaysLie),
   (Prisoner.P3, Truthfulness.AlwaysTruth),
   (Prisoner.P4, Truthfulness.AlwaysLie),
   (Prisoner.P5, Truthfulness.Variable)]

-- Define the statements of the prisoners
def prisonerStatements : List Statement := 
  [{ prisoner := Prisoner.P1, 
     ownLady := { name := "Anna", hairColor := HairColor.Blonde },
     neighborLadies := [HairColor.Blonde] },
   { prisoner := Prisoner.P2,
     ownLady := { name := "Brynhild", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P3,
     ownLady := { name := "Clotilde", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Red, HairColor.Red] },
   { prisoner := Prisoner.P4,
     ownLady := { name := "Gudrun", hairColor := HairColor.Red },
     neighborLadies := [HairColor.Brunette, HairColor.Brunette] },
   { prisoner := Prisoner.P5,
     ownLady := { name := "Johanna", hairColor := HairColor.Brunette },
     neighborLadies := [HairColor.Brunette, HairColor.Blonde] }]

-- Define the correct solution
def correctSolution : List Lady := 
  [{ name := "Anna", hairColor := HairColor.Blonde },
   { name := "Brynhild", hairColor := HairColor.Red },
   { name := "Clotilde", hairColor := HairColor.Red },
   { name := "Gudrun", hairColor := HairColor.Red },
   { name := "Johanna", hairColor := HairColor.Brunette }]

-- Theorem statement
theorem prisoners_puzzle_solution :
  ∀ (solution : List Lady),
  (∀ p ∈ prisonerSetup, 
   ∀ s ∈ prisonerStatements,
   p.1 = s.prisoner →
   (p.2 = Truthfulness.AlwaysTruth → 
    (s.ownLady ∈ solution ∧ 
     ∀ c ∈ s.neighborLadies, ∃ l ∈ solution, l.hairColor = c)) ∧
   (p.2 = Truthfulness.AlwaysLie → 
    (s.ownLady ∉ solution ∨ 
     ∃ c ∈ s.neighborLadies, ∀ l ∈ solution, l.hairColor ≠ c))) →
  solution = correctSolution :=
sorry

end NUMINAMATH_CALUDE_prisoners_puzzle_solution_l254_25480


namespace NUMINAMATH_CALUDE_wedge_volume_l254_25428

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) (V : ℝ) : 
  d = 10 → -- diameter of the log
  α = 60 → -- angle between the two cuts in degrees
  V = (125/18) * Real.pi → -- volume of the wedge
  ∃ (r h : ℝ),
    r = d/2 ∧ -- radius of the log
    h = r ∧ -- height of the cone (equal to radius due to 60° angle)
    V = (1/6) * ((1/3) * Real.pi * r^2 * h) -- volume formula
  :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l254_25428


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l254_25414

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (4, 2)
  let n : ℝ × ℝ := (x, -3)
  parallel m n → x = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l254_25414


namespace NUMINAMATH_CALUDE_cheryl_basil_harvest_l254_25403

-- Define the variables and constants
def basil_per_pesto : ℝ := 4
def harvest_weeks : ℕ := 8
def total_pesto : ℝ := 32

-- Define the theorem
theorem cheryl_basil_harvest :
  (basil_per_pesto * total_pesto) / harvest_weeks = 16 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_basil_harvest_l254_25403


namespace NUMINAMATH_CALUDE_digit_sum_problem_l254_25459

theorem digit_sum_problem (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  (10 * E + A) + (10 * E + C) = 10 * D + A →
  (10 * E + A) - (10 * E + C) = A →
  D = 8 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l254_25459


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l254_25497

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) := True

-- Define that ABC is a right-angled triangle
def IsRightAngled (A B C : ℝ × ℝ) := True

-- Define AD as an angle bisector
def IsAngleBisector (A B C D : ℝ × ℝ) := True

-- Define the lengths of the sides
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_theorem (A B C D : ℝ × ℝ) (x : ℝ) :
  Triangle A B C →
  IsRightAngled A B C →
  IsAngleBisector A B C D →
  SideLength A B = 100 →
  SideLength B C = x →
  SideLength A C = x + 10 →
  Int.floor (TriangleArea A D C + 0.5) = 20907 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l254_25497


namespace NUMINAMATH_CALUDE_no_f_iteration_to_one_l254_25436

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^2 + 1 else n / 2 + 3

def iterateF (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => f (iterateF n k)

theorem no_f_iteration_to_one :
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 100 → ∀ k : ℕ, iterateF n k ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_no_f_iteration_to_one_l254_25436


namespace NUMINAMATH_CALUDE_contrapositive_at_least_one_even_l254_25432

theorem contrapositive_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_at_least_one_even_l254_25432


namespace NUMINAMATH_CALUDE_like_terms_exponents_l254_25467

theorem like_terms_exponents (a b : ℝ) (x y : ℝ) : 
  (∃ k : ℝ, 2 * a^(2*x) * b^(3*y) = k * (-3 * a^2 * b^(2-x))) → 
  x = 1 ∧ y = 1/3 :=
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l254_25467


namespace NUMINAMATH_CALUDE_billy_sodas_l254_25454

/-- The number of sodas in Billy's pack -/
def sodas_in_pack (sisters : ℕ) (brothers : ℕ) (sodas_per_sibling : ℕ) : ℕ :=
  (sisters + brothers) * sodas_per_sibling

/-- Theorem: The number of sodas in Billy's pack is 12 -/
theorem billy_sodas :
  ∀ (sisters brothers sodas_per_sibling : ℕ),
    brothers = 2 * sisters →
    sisters = 2 →
    sodas_per_sibling = 2 →
    sodas_in_pack sisters brothers sodas_per_sibling = 12 := by
  sorry

end NUMINAMATH_CALUDE_billy_sodas_l254_25454


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l254_25427

/-- The distance between the foci of an ellipse with semi-major axis 9 and semi-minor axis 3 -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 9) (hb : b = 3) :
  2 * Real.sqrt (a^2 - b^2) = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l254_25427


namespace NUMINAMATH_CALUDE_percent_defective_units_l254_25490

/-- Given that 4% of defective units are shipped for sale and 0.32% of all units
    produced are defective units that are shipped for sale, prove that 8% of
    all units produced are defective. -/
theorem percent_defective_units (shipped_defective_ratio : Real)
                                 (total_shipped_defective_ratio : Real)
                                 (h1 : shipped_defective_ratio = 0.04)
                                 (h2 : total_shipped_defective_ratio = 0.0032) :
  shipped_defective_ratio * (total_shipped_defective_ratio / shipped_defective_ratio) = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_percent_defective_units_l254_25490


namespace NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l254_25442

theorem lcm_of_8_24_36_54 : Nat.lcm 8 (Nat.lcm 24 (Nat.lcm 36 54)) = 216 := by sorry

end NUMINAMATH_CALUDE_lcm_of_8_24_36_54_l254_25442


namespace NUMINAMATH_CALUDE_isosceles_triangle_removal_l254_25456

/-- Given a square with isosceles right triangles removed from each corner to form a rectangle,
    if the diagonal of the resulting rectangle is 15 units,
    then the combined area of the four removed triangles is 112.5 square units. -/
theorem isosceles_triangle_removal (r s : ℝ) : 
  r > 0 → s > 0 →  -- r and s are positive real numbers
  (r + s)^2 + (r - s)^2 = 15^2 →  -- diagonal of resulting rectangle is 15
  2 * r * s = 112.5  -- combined area of four removed triangles
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_removal_l254_25456


namespace NUMINAMATH_CALUDE_kira_away_time_l254_25407

/-- Represents the eating rate of the cat in hours per pound of kibble -/
def eating_rate : ℝ := 4

/-- Represents the initial amount of kibble in pounds -/
def initial_kibble : ℝ := 3

/-- Represents the remaining amount of kibble in pounds -/
def remaining_kibble : ℝ := 1

/-- Calculates the time Kira was away based on the given conditions -/
def time_away : ℝ := (initial_kibble - remaining_kibble) * eating_rate

/-- Proves that the time Kira was away from home is 8 hours -/
theorem kira_away_time : time_away = 8 := by
  sorry

end NUMINAMATH_CALUDE_kira_away_time_l254_25407


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l254_25498

theorem cubic_roots_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 4 * r^2 + 1500 * r + 3000 = 0) →
  (6 * s^3 + 4 * s^2 + 1500 * s + 3000 = 0) →
  (6 * t^3 + 4 * t^2 + 1500 * t + 3000 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992/27 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l254_25498


namespace NUMINAMATH_CALUDE_vector_magnitude_l254_25438

def a : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (2, -4)

theorem vector_magnitude (x : ℝ) (b : ℝ × ℝ) 
  (h1 : b = (-1, x))
  (h2 : ∃ k : ℝ, b.1 = k * c.1 ∧ b.2 = k * c.2) :
  ‖a + b‖ = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l254_25438


namespace NUMINAMATH_CALUDE_trajectory_equation_l254_25429

theorem trajectory_equation (x y : ℝ) (h1 : x > 0) :
  (((x - 1/2)^2 + y^2)^(1/2) = x + 1/2) → y^2 = 2*x := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l254_25429


namespace NUMINAMATH_CALUDE_club_contribution_proof_l254_25408

/-- Proves that the initial contribution per member is $300 --/
theorem club_contribution_proof (n : ℕ) (x : ℝ) : 
  n = 10 → -- Initial number of members
  (n + 5) * (x - 100) = n * x → -- Total amount remains constant with 5 more members
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_club_contribution_proof_l254_25408


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l254_25482

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  1259 = 23 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧
  ∀ (q' r' : ℕ), (1259 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0) → q' - r' ≤ q - r ∧ 
  q - r = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l254_25482


namespace NUMINAMATH_CALUDE_remaining_customers_l254_25401

theorem remaining_customers (initial : ℕ) (left : ℕ) (remaining : ℕ) : 
  initial = 14 → left = 11 → remaining = initial - left → remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_customers_l254_25401


namespace NUMINAMATH_CALUDE_local_min_implies_b_in_open_unit_interval_l254_25463

/-- If f(x) = x^3 - 3bx + b has a local minimum in (0, 1), then b ∈ (0, 1) -/
theorem local_min_implies_b_in_open_unit_interval (b : ℝ) : 
  (∃ c ∈ Set.Ioo 0 1, IsLocalMin (fun x => x^3 - 3*b*x + b) c) → 
  b ∈ Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_local_min_implies_b_in_open_unit_interval_l254_25463


namespace NUMINAMATH_CALUDE_min_participants_in_tournament_l254_25422

theorem min_participants_in_tournament : ∃ (n : ℕ) (k : ℕ),
  n = 11 ∧
  k < n / 2 ∧
  k > (45 * n) / 100 ∧
  ∀ (m : ℕ) (j : ℕ), m < n →
    (j < m / 2 ∧ j > (45 * m) / 100) → False :=
by sorry

end NUMINAMATH_CALUDE_min_participants_in_tournament_l254_25422


namespace NUMINAMATH_CALUDE_min_boxes_for_load_l254_25441

theorem min_boxes_for_load (total_load : ℝ) (max_box_weight : ℝ) : 
  total_load = 13.5 * 1000 → 
  max_box_weight = 350 → 
  ⌈total_load / max_box_weight⌉ ≥ 39 := by
sorry

end NUMINAMATH_CALUDE_min_boxes_for_load_l254_25441


namespace NUMINAMATH_CALUDE_lawn_care_supplies_cost_l254_25445

/-- The total cost of supplies for a lawn care company -/
theorem lawn_care_supplies_cost 
  (num_blades : ℕ) 
  (blade_cost : ℕ) 
  (string_cost : ℕ) :
  num_blades = 4 →
  blade_cost = 8 →
  string_cost = 7 →
  num_blades * blade_cost + string_cost = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_lawn_care_supplies_cost_l254_25445


namespace NUMINAMATH_CALUDE_self_employed_tax_calculation_l254_25472

/-- Calculates the tax amount for a self-employed citizen --/
def calculate_tax_amount (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * tax_rate

/-- Theorem: The tax amount for a self-employed citizen with a gross income of 350,000.00 rubles and a tax rate of 6% is 21,000.00 rubles --/
theorem self_employed_tax_calculation :
  let gross_income : ℝ := 350000.00
  let tax_rate : ℝ := 0.06
  calculate_tax_amount gross_income tax_rate = 21000.00 := by
  sorry

#eval calculate_tax_amount 350000.00 0.06

end NUMINAMATH_CALUDE_self_employed_tax_calculation_l254_25472


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_increase_decrease_intervals_l254_25434

noncomputable section

-- Define the function f(x) = ln x - ax
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem for the tangent line equation when a = -2
theorem tangent_line_at_x_1 (a : ℝ) (h : a = -2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 → m * x + b = y) ∧ 
  (m * x - y + b = 0 ↔ 3 * x - y - 1 = 0) :=
sorry

-- Theorem for the intervals of increase and decrease
theorem increase_decrease_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a > 0 → (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂ : ℝ, 1/a < x₁ → x₁ < x₂ → f a x₂ < f a x₁)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_increase_decrease_intervals_l254_25434


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_seven_l254_25466

-- Define the triangle sides
variable (a b c : ℝ)

-- Define the condition equation
def condition (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 - 4*a - 4*b - 6*c + 17 = 0

-- Define what it means for a, b, c to form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- State the theorem
theorem triangle_perimeter_is_seven 
  (h1 : is_triangle a b c) 
  (h2 : condition a b c) : 
  perimeter a b c = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_seven_l254_25466


namespace NUMINAMATH_CALUDE_ellipse_foci_l254_25440

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(Real.sqrt 7, 0), (-Real.sqrt 7, 0)}

-- Theorem statement
theorem ellipse_foci :
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (f : ℝ × ℝ), f ∈ foci ∧
  (x - f.1)^2 + y^2 = (4 + Real.sqrt 7)^2 ∨
  (x - f.1)^2 + y^2 = (4 - Real.sqrt 7)^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_l254_25440


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l254_25446

theorem arithmetic_calculation : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l254_25446


namespace NUMINAMATH_CALUDE_total_participants_is_260_l254_25485

/-- Represents the voting scenario for a school disco date --/
structure VotingScenario where
  initial_oct22_percent : ℝ
  initial_oct29_percent : ℝ
  additional_oct22_votes : ℕ
  final_oct29_percent : ℝ

/-- Calculates the total number of participants in the voting --/
def total_participants (scenario : VotingScenario) : ℕ :=
  sorry

/-- Theorem stating that the total number of participants is 260 --/
theorem total_participants_is_260 (scenario : VotingScenario) 
  (h1 : scenario.initial_oct22_percent = 0.35)
  (h2 : scenario.initial_oct29_percent = 0.65)
  (h3 : scenario.additional_oct22_votes = 80)
  (h4 : scenario.final_oct29_percent = 0.45) :
  total_participants scenario = 260 := by
  sorry

end NUMINAMATH_CALUDE_total_participants_is_260_l254_25485


namespace NUMINAMATH_CALUDE_sector_properties_l254_25437

/-- Represents a circular sector --/
structure Sector where
  α : Real  -- Central angle in radians
  r : Real  -- Radius
  h_r_pos : r > 0

/-- Calculates the arc length of a sector --/
def arcLength (s : Sector) : Real :=
  s.α * s.r

/-- Calculates the perimeter of a sector --/
def perimeter (s : Sector) : Real :=
  s.r * (s.α + 2)

/-- Calculates the area of a sector --/
def area (s : Sector) : Real :=
  0.5 * s.α * s.r^2

theorem sector_properties :
  ∃ (s1 s2 : Sector),
    s1.α = 2 * Real.pi / 3 ∧
    s1.r = 6 ∧
    arcLength s1 = 4 * Real.pi ∧
    perimeter s2 = 24 ∧
    s2.α = 2 ∧
    area s2 = 36 ∧
    ∀ (s : Sector), perimeter s = 24 → area s ≤ area s2 := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l254_25437


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l254_25413

def total_arrangements : ℕ := Nat.choose 6 2

def non_adjacent_arrangements : ℕ := Nat.choose 5 2

theorem zeros_not_adjacent_probability :
  (non_adjacent_arrangements : ℚ) / total_arrangements = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l254_25413


namespace NUMINAMATH_CALUDE_yulia_number_l254_25417

theorem yulia_number (x : ℝ) : x + 13 = 4 * (x + 1) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_yulia_number_l254_25417


namespace NUMINAMATH_CALUDE_max_envelopes_proof_l254_25458

def number_of_bus_tickets : ℕ := 18
def number_of_subway_tickets : ℕ := 12

def max_envelopes : ℕ := Nat.gcd number_of_bus_tickets number_of_subway_tickets

theorem max_envelopes_proof :
  (∀ k : ℕ, k ∣ number_of_bus_tickets ∧ k ∣ number_of_subway_tickets → k ≤ max_envelopes) ∧
  (max_envelopes ∣ number_of_bus_tickets) ∧
  (max_envelopes ∣ number_of_subway_tickets) :=
sorry

end NUMINAMATH_CALUDE_max_envelopes_proof_l254_25458


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l254_25431

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l254_25431


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l254_25447

theorem quadratic_integer_roots_count :
  ∃! (S : Finset ℝ), 
    (∀ a ∈ S, ∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) ∧
    (∀ a : ℝ, (∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) → a ∈ S) ∧
    S.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l254_25447


namespace NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l254_25496

/-- The number of distinct elements in the set -/
def n : ℕ := 5

/-- The number of times each element appears -/
def k : ℕ := 6

/-- The length of the sequences to be formed -/
def seq_length : ℕ := 6

/-- The number of possible sequences -/
def num_sequences : ℕ := n ^ seq_length

theorem acme_vowel_soup_sequences :
  num_sequences = 15625 :=
sorry

end NUMINAMATH_CALUDE_acme_vowel_soup_sequences_l254_25496


namespace NUMINAMATH_CALUDE_hex_A08_equals_2568_l254_25453

/-- Converts a single hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if 'A' ≤ c ∧ c ≤ 'F' then c.toNat - 'A'.toNat + 10
  else 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => hex_to_dec c + 16 * acc) 0

/-- The hexadecimal representation of the number -/
def hex_number : String := "A08"

/-- Theorem stating that the hexadecimal number A08 is equal to 2568 in decimal -/
theorem hex_A08_equals_2568 : hex_string_to_dec hex_number = 2568 := by
  sorry


end NUMINAMATH_CALUDE_hex_A08_equals_2568_l254_25453


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l254_25433

theorem polygon_interior_angles (n : ℕ) (h : n = 14) : 
  (n - 2) * 180 - 180 = 2000 :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l254_25433


namespace NUMINAMATH_CALUDE_cricket_match_average_l254_25424

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- The conditions of the cricket match -/
def cricket_match_conditions (scores : BatsmanScores) : Prop :=
  scores.d = scores.e + 5 ∧
  scores.e = scores.a - 8 ∧
  scores.b = scores.d + scores.e ∧
  scores.b + scores.c = 107 ∧
  scores.e = 20

/-- The theorem stating that the average score is 36 -/
theorem cricket_match_average (scores : BatsmanScores) 
  (h : cricket_match_conditions scores) : 
  (scores.a + scores.b + scores.c + scores.d + scores.e) / 5 = 36 := by
  sorry

#check cricket_match_average

end NUMINAMATH_CALUDE_cricket_match_average_l254_25424


namespace NUMINAMATH_CALUDE_cube_with_holes_properties_l254_25470

/-- Represents a cube with square holes on each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ
  hole_depth : ℝ

/-- Calculate the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (c : CubeWithHoles) : ℝ :=
  6 * c.edge_length^2 + 6 * (c.hole_side_length^2 + 4 * c.hole_side_length * c.hole_depth)

/-- Calculate the total volume of material removed from a cube due to holes -/
def total_volume_removed (c : CubeWithHoles) : ℝ :=
  6 * c.hole_side_length^2 * c.hole_depth

/-- The main theorem stating the properties of the specific cube with holes -/
theorem cube_with_holes_properties :
  let c := CubeWithHoles.mk 4 2 1
  total_surface_area c = 144 ∧ total_volume_removed c = 24 := by
  sorry


end NUMINAMATH_CALUDE_cube_with_holes_properties_l254_25470


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l254_25416

theorem polynomial_constant_term (a b c d e : ℝ) :
  (2^7 * a + 2^5 * b + 2^3 * c + 2 * d + e = 23) →
  ((-2)^7 * a + (-2)^5 * b + (-2)^3 * c + (-2) * d + e = -35) →
  e = -6 := by sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l254_25416


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l254_25469

theorem fraction_sum_simplification : (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l254_25469


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l254_25406

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C < π) :
  (1 / 3) * ((Real.tan A)^2 / (Real.tan B * Real.tan C) +
             (Real.tan B)^2 / (Real.tan C * Real.tan A) +
             (Real.tan C)^2 / (Real.tan A * Real.tan B)) +
  3 * (1 / (Real.tan A + Real.tan B + Real.tan C))^(2/3) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l254_25406


namespace NUMINAMATH_CALUDE_sequence_sum_l254_25493

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

def geometric_sequence (b₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₁ * r^(n - 1)

theorem sequence_sum (a₁ : ℕ) :
  let a := arithmetic_sequence a₁ 2
  let b := geometric_sequence 1 2
  a (b 2) + a (b 3) + a (b 4) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l254_25493


namespace NUMINAMATH_CALUDE_unique_circle_construction_l254_25411

/-- A line in a plane -/
structure Line : Type :=
  (l : Set (Real × Real))

/-- A point in a plane -/
structure Point : Type :=
  (x : Real) (y : Real)

/-- A circle in a plane -/
structure Circle : Type :=
  (center : Point) (radius : Real)

/-- Predicate to check if a point belongs to a line -/
def PointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Predicate to check if a circle passes through a point -/
def CirclePassesThrough (c : Circle) (p : Point) : Prop := sorry

/-- Predicate to check if a circle is tangent to a line at a point -/
def CircleTangentToLineAt (c : Circle) (l : Line) (p : Point) : Prop := sorry

/-- Main theorem: Existence and uniqueness of a circle passing through B and tangent to l at A -/
theorem unique_circle_construction (l : Line) (A B : Point) 
  (h1 : PointOnLine A l) 
  (h2 : ¬PointOnLine B l) : 
  ∃! k : Circle, CirclePassesThrough k B ∧ CircleTangentToLineAt k l A := by
  sorry

end NUMINAMATH_CALUDE_unique_circle_construction_l254_25411


namespace NUMINAMATH_CALUDE_total_checks_purchased_l254_25479

/-- Represents the number of travelers checks purchased -/
structure TravelersChecks where
  fifty : ℕ    -- number of $50 checks
  hundred : ℕ  -- number of $100 checks

/-- The total value of all travelers checks -/
def total_value (tc : TravelersChecks) : ℕ :=
  50 * tc.fifty + 100 * tc.hundred

/-- The average value of remaining checks after spending 6 $50 checks -/
def average_remaining (tc : TravelersChecks) : ℚ :=
  (total_value tc - 300) / (tc.fifty + tc.hundred - 6 : ℚ)

/-- Theorem stating the total number of travelers checks purchased -/
theorem total_checks_purchased :
  ∃ (tc : TravelersChecks),
    total_value tc = 1800 ∧
    average_remaining tc = 62.5 ∧
    tc.fifty + tc.hundred = 33 :=
  sorry

end NUMINAMATH_CALUDE_total_checks_purchased_l254_25479


namespace NUMINAMATH_CALUDE_symmetry_shift_l254_25464

noncomputable def smallest_shift_for_symmetry : ℝ := 7 * Real.pi / 6

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem symmetry_shift :
  let f (x m : ℝ) := Real.cos (x + m) - Real.sqrt 3 * Real.sin (x + m)
  ∀ m : ℝ, m > 0 → (
    (is_symmetric_about_y_axis (f · m)) ↔ 
    m ≥ smallest_shift_for_symmetry
  ) :=
sorry

end NUMINAMATH_CALUDE_symmetry_shift_l254_25464


namespace NUMINAMATH_CALUDE_room_height_proof_l254_25402

/-- Proves that the height of a room with given dimensions and openings is 6 feet -/
theorem room_height_proof (width length : ℝ) (doorway1_width doorway1_height : ℝ)
  (window_width window_height : ℝ) (doorway2_width doorway2_height : ℝ)
  (total_paint_area : ℝ) (h : ℝ) :
  width = 20 ∧ length = 20 ∧
  doorway1_width = 3 ∧ doorway1_height = 7 ∧
  window_width = 6 ∧ window_height = 4 ∧
  doorway2_width = 5 ∧ doorway2_height = 7 ∧
  total_paint_area = 560 ∧
  total_paint_area = 4 * width * h - (doorway1_width * doorway1_height + window_width * window_height + doorway2_width * doorway2_height) →
  h = 6 := by
  sorry

#check room_height_proof

end NUMINAMATH_CALUDE_room_height_proof_l254_25402


namespace NUMINAMATH_CALUDE_unique_triple_solution_l254_25421

theorem unique_triple_solution :
  ∃! (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 + z^2 = 3 ∧
  (x + y + z) * (x^2 + y^2 + z^2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l254_25421


namespace NUMINAMATH_CALUDE_rectangle_area_l254_25471

/-- The area of a rectangle with given vertices in a rectangular coordinate system -/
theorem rectangle_area (v1 v2 v3 v4 : ℝ × ℝ) : 
  v1 = (-8, 1) → v2 = (1, 1) → v3 = (1, -7) → v4 = (-8, -7) →
  let width := |v2.1 - v1.1|
  let height := |v2.2 - v3.2|
  width * height = 72 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l254_25471


namespace NUMINAMATH_CALUDE_product_segment_doubles_when_unit_halved_l254_25468

/-- Theorem: Product segment length doubles when unit segment is halved -/
theorem product_segment_doubles_when_unit_halved 
  (a b e d : ℝ) 
  (h1 : d = a * b / e) 
  (e' : ℝ) 
  (h2 : e' = e / 2) 
  (d' : ℝ) 
  (h3 : d' = a * b / e') : 
  d' = 2 * d := by
sorry

end NUMINAMATH_CALUDE_product_segment_doubles_when_unit_halved_l254_25468


namespace NUMINAMATH_CALUDE_number_problem_l254_25483

theorem number_problem : 
  let x : ℝ := 25
  80 / 100 * 60 - (4 / 5 * x) = 28 := by sorry

end NUMINAMATH_CALUDE_number_problem_l254_25483


namespace NUMINAMATH_CALUDE_valid_arrangement_has_four_rows_of_seven_l254_25404

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  rows_of_seven : ℕ
  rows_of_six : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.rows_of_seven * 7 + s.rows_of_six * 6 = 52

/-- Theorem stating that the valid arrangement has 4 rows of 7 people -/
theorem valid_arrangement_has_four_rows_of_seven :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_of_seven = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_has_four_rows_of_seven_l254_25404


namespace NUMINAMATH_CALUDE_range_of_m_l254_25430

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem range_of_m :
  (∀ x m : ℝ, (¬(p x) → ¬(q x m)) ∧ ∃ x, ¬(p x) ∧ (q x m)) →
  ∀ m : ℝ, -3 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l254_25430


namespace NUMINAMATH_CALUDE_meeting_speed_l254_25495

theorem meeting_speed
  (total_distance : ℝ)
  (time : ℝ)
  (speed_diff : ℝ)
  (h1 : total_distance = 45)
  (h2 : time = 5)
  (h3 : speed_diff = 1)
  (h4 : ∀ (v_a v_b : ℝ), v_a = v_b + speed_diff → v_a * time + v_b * time = total_distance)
  : ∃ (v_a : ℝ), v_a = 5 ∧ ∃ (v_b : ℝ), v_a = v_b + speed_diff ∧ v_a * time + v_b * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_meeting_speed_l254_25495


namespace NUMINAMATH_CALUDE_annabelle_savings_l254_25455

/-- Calculates the amount saved from a weekly allowance after spending on junk food and sweets -/
def calculate_savings (weekly_allowance : ℚ) (junk_food_fraction : ℚ) (sweets_cost : ℚ) : ℚ :=
  weekly_allowance - (weekly_allowance * junk_food_fraction + sweets_cost)

/-- Proves that given a weekly allowance of $30, spending 1/3 of it on junk food and an additional $8 on sweets, the remaining amount saved is $12 -/
theorem annabelle_savings :
  calculate_savings 30 (1/3) 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_annabelle_savings_l254_25455


namespace NUMINAMATH_CALUDE_mitch_weekend_to_weekday_ratio_l254_25473

/-- Represents Mitch's work schedule and earnings --/
structure MitchSchedule where
  weekdayHours : ℕ  -- Hours worked per weekday
  weekendHours : ℕ  -- Hours worked per weekend day
  weekdayRate : ℚ   -- Hourly rate for weekdays
  totalEarnings : ℚ -- Total weekly earnings

/-- Calculates the ratio of weekend rate to weekday rate --/
def weekendToWeekdayRatio (schedule : MitchSchedule) : ℚ :=
  let totalWeekdayHours := schedule.weekdayHours * 5
  let totalWeekendHours := schedule.weekendHours * 2
  let weekdayEarnings := schedule.weekdayRate * totalWeekdayHours
  let weekendEarnings := schedule.totalEarnings - weekdayEarnings
  let weekendRate := weekendEarnings / totalWeekendHours
  weekendRate / schedule.weekdayRate

/-- Theorem stating that Mitch's weekend to weekday rate ratio is 2:1 --/
theorem mitch_weekend_to_weekday_ratio :
  let schedule : MitchSchedule := {
    weekdayHours := 5,
    weekendHours := 3,
    weekdayRate := 3,
    totalEarnings := 111
  }
  weekendToWeekdayRatio schedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_mitch_weekend_to_weekday_ratio_l254_25473


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l254_25412

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l254_25412


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l254_25481

theorem chickens_and_rabbits (x y : ℕ) : 
  (x + y = 35 ∧ 2*x + 4*y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_l254_25481


namespace NUMINAMATH_CALUDE_quarter_percent_of_120_l254_25475

theorem quarter_percent_of_120 : (1 / 4 : ℚ) / 100 * 120 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_quarter_percent_of_120_l254_25475


namespace NUMINAMATH_CALUDE_volume_eq_cross_section_area_l254_25415

/-- A right prism with an equilateral triangular base -/
structure EquilateralPrism where
  /-- Side length of the equilateral triangular base -/
  a : ℝ
  /-- Angle between the cross-section plane and the base -/
  φ : ℝ
  /-- Area of the cross-section -/
  Q : ℝ
  /-- Side length is positive -/
  h_a_pos : 0 < a
  /-- Angle is between 0 and π/2 -/
  h_φ_range : 0 < φ ∧ φ < Real.pi / 2
  /-- Area is positive -/
  h_Q_pos : 0 < Q

/-- The volume of the equilateral prism -/
def volume (p : EquilateralPrism) : ℝ := p.Q

theorem volume_eq_cross_section_area (p : EquilateralPrism) :
  volume p = p.Q := by sorry

end NUMINAMATH_CALUDE_volume_eq_cross_section_area_l254_25415


namespace NUMINAMATH_CALUDE_article_original_price_l254_25443

/-- Calculates the original price of an article given its selling price and loss percentage. -/
def originalPrice (sellingPrice : ℚ) (lossPercent : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercent / 100)

/-- Theorem stating that an article sold for 450 with a 25% loss had an original price of 600. -/
theorem article_original_price :
  originalPrice 450 25 = 600 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l254_25443


namespace NUMINAMATH_CALUDE_torn_page_numbers_l254_25439

theorem torn_page_numbers (n : ℕ) (k : ℕ) : 
  n > 0 ∧ k > 1 ∧ k < n ∧ (n * (n + 1)) / 2 - (2 * k - 1) = 15000 → k = 113 := by
  sorry

end NUMINAMATH_CALUDE_torn_page_numbers_l254_25439


namespace NUMINAMATH_CALUDE_parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l254_25486

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

-- Theorem 1
theorem parallel_implies_magnitude (k : ℝ) :
  (∃ (t : ℝ), a = t • (b k)) → ‖b k‖ = 3 * Real.sqrt 5 := by
  sorry

-- Theorem 2
theorem perpendicular_implies_k :
  (a • (a + 2 • (b (1/4))) = 0) → (1/4 : ℝ) = 1/4 := by
  sorry

-- Theorem 3
theorem obtuse_angle_implies_k_range (k : ℝ) :
  (a • (b k) < 0) → k < 3/2 ∧ k ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_implies_magnitude_perpendicular_implies_k_obtuse_angle_implies_k_range_l254_25486


namespace NUMINAMATH_CALUDE_no_double_application_function_l254_25465

theorem no_double_application_function :
  ¬∃ (f : ℕ → ℕ), ∀ (x : ℕ), f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_function_l254_25465


namespace NUMINAMATH_CALUDE_compound_interest_principal_l254_25426

/-- Given a principal amount and an annual compound interest rate,
    prove that the principal amount is approximately 5967.79 if it grows
    to 8000 after 2 years and 9261 after 3 years under compound interest. -/
theorem compound_interest_principal (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 8000)
  (h2 : P * (1 + r)^3 = 9261) :
  ∃ ε > 0, |P - 5967.79| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_l254_25426


namespace NUMINAMATH_CALUDE_star_three_four_l254_25409

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

/-- Theorem stating that 3 ⋆ 4 = 0 -/
theorem star_three_four : star 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_three_four_l254_25409


namespace NUMINAMATH_CALUDE_olivia_correct_answers_l254_25491

theorem olivia_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_problems = 15)
  (h2 : correct_points = 4)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 25) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 10 ∧ 
    correct_answers ≤ total_problems ∧
    (correct_points * correct_answers + incorrect_points * (total_problems - correct_answers) = total_score) := by
  sorry

end NUMINAMATH_CALUDE_olivia_correct_answers_l254_25491


namespace NUMINAMATH_CALUDE_lease_problem_l254_25462

theorem lease_problem (elapsed_time : ℝ) : 
  elapsed_time > 0 ∧ 
  elapsed_time < 99 ∧
  (2 / 3) * elapsed_time = (4 / 5) * (99 - elapsed_time) →
  elapsed_time = 54 := by
sorry

end NUMINAMATH_CALUDE_lease_problem_l254_25462


namespace NUMINAMATH_CALUDE_vector_relations_l254_25461

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallel vectors -/
def parallel (v w : Vec2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Perpendicular vectors -/
def perpendicular (v w : Vec2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

theorem vector_relations (m : ℝ) :
  let a : Vec2D := ⟨1, 2⟩
  let b : Vec2D := ⟨-2, m⟩
  (parallel a b → m = -4) ∧
  (perpendicular a b → m = 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l254_25461


namespace NUMINAMATH_CALUDE_oscar_swag_bag_scarves_l254_25478

/-- Represents the contents and value of an Oscar swag bag -/
structure SwagBag where
  totalValue : ℕ
  earringCost : ℕ
  iphoneCost : ℕ
  scarfCost : ℕ
  numScarves : ℕ

/-- Theorem stating that given the specific costs and total value, 
    the number of scarves in the swag bag is 4 -/
theorem oscar_swag_bag_scarves (bag : SwagBag) 
    (h1 : bag.totalValue = 20000)
    (h2 : bag.earringCost = 6000)
    (h3 : bag.iphoneCost = 2000)
    (h4 : bag.scarfCost = 1500)
    (h5 : bag.totalValue = 2 * bag.earringCost + bag.iphoneCost + bag.numScarves * bag.scarfCost) :
  bag.numScarves = 4 := by
  sorry

#check oscar_swag_bag_scarves

end NUMINAMATH_CALUDE_oscar_swag_bag_scarves_l254_25478


namespace NUMINAMATH_CALUDE_second_share_interest_rate_l254_25418

theorem second_share_interest_rate 
  (total_investment : ℝ)
  (first_share_yield : ℝ)
  (total_interest_rate : ℝ)
  (second_share_investment : ℝ)
  (h1 : total_investment = 100000)
  (h2 : first_share_yield = 0.09)
  (h3 : total_interest_rate = 0.0925)
  (h4 : second_share_investment = 12500) :
  ∃ (second_share_yield : ℝ),
    second_share_yield = 0.11 ∧
    total_investment * total_interest_rate = 
      (total_investment - second_share_investment) * first_share_yield +
      second_share_investment * second_share_yield :=
by
  sorry

end NUMINAMATH_CALUDE_second_share_interest_rate_l254_25418


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l254_25488

/-- Represents the result of a weighing -/
inductive WeighingResult
| Equal : WeighingResult
| LeftHeavier : WeighingResult
| RightHeavier : WeighingResult

/-- Represents a coin -/
inductive Coin
| Real : Coin
| Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := CoinSet → CoinSet → WeighingResult

/-- The maximum number of weighings allowed -/
def MaxWeighings : Nat := 4

/-- The number of unknown coins -/
def UnknownCoins : Nat := 12

/-- The number of known real coins -/
def KnownRealCoins : Nat := 5

/-- The number of known counterfeit coins -/
def KnownCounterfeitCoins : Nat := 5

/-- A strategy is a function that takes the current state and returns the next weighing to perform -/
def Strategy := List WeighingResult → Weighing

/-- Determines if a strategy is successful in identifying the number of counterfeit coins -/
def IsSuccessfulStrategy (s : Strategy) : Prop := sorry

/-- The main theorem: There exists a successful strategy to determine the number of counterfeit coins -/
theorem counterfeit_coin_identification :
  ∃ (s : Strategy), IsSuccessfulStrategy s := by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l254_25488


namespace NUMINAMATH_CALUDE_correct_calculation_l254_25444

theorem correct_calculation : (-36 : ℚ) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l254_25444


namespace NUMINAMATH_CALUDE_lcm_and_sum_of_numbers_l254_25449

def numbers : List Nat := [14, 21, 35]

theorem lcm_and_sum_of_numbers :
  (Nat.lcm (Nat.lcm 14 21) 35 = 210) ∧ (numbers.sum = 70) := by
  sorry

end NUMINAMATH_CALUDE_lcm_and_sum_of_numbers_l254_25449


namespace NUMINAMATH_CALUDE_track_completion_time_l254_25405

/-- Represents a runner on the circular track -/
structure Runner :=
  (position : ℝ)
  (speed : ℝ)

/-- Represents the circular track -/
structure Track :=
  (circumference : ℝ)
  (runners : List Runner)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℝ)

/-- The main theorem to be proved -/
theorem track_completion_time 
  (track : Track) 
  (meeting1 : Meeting) 
  (meeting2 : Meeting) 
  (meeting3 : Meeting) :
  meeting1.runner1 = meeting2.runner1 ∧ 
  meeting1.runner2 = meeting2.runner2 ∧
  meeting2.runner2 = meeting3.runner1 ∧
  meeting2.runner1 = meeting3.runner2 ∧
  meeting2.time - meeting1.time = 15 ∧
  meeting3.time - meeting2.time = 25 →
  track.circumference = 80 := by
  sorry

end NUMINAMATH_CALUDE_track_completion_time_l254_25405


namespace NUMINAMATH_CALUDE_volleyball_scoring_l254_25420

/-- Volleyball team scoring problem -/
theorem volleyball_scoring
  (lizzie_score : ℕ)
  (nathalie_score : ℕ)
  (aimee_score : ℕ)
  (teammates_score : ℕ)
  (total_score : ℕ)
  (h1 : lizzie_score = 4)
  (h2 : nathalie_score > lizzie_score)
  (h3 : aimee_score = 2 * (lizzie_score + nathalie_score))
  (h4 : total_score = 50)
  (h5 : teammates_score = 17)
  (h6 : lizzie_score + nathalie_score + aimee_score + teammates_score = total_score) :
  nathalie_score = lizzie_score + 3 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_scoring_l254_25420


namespace NUMINAMATH_CALUDE_remaining_debt_percentage_l254_25477

def original_debt : ℝ := 500
def initial_payment : ℝ := 125

theorem remaining_debt_percentage :
  (original_debt - initial_payment) / original_debt * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_remaining_debt_percentage_l254_25477


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l254_25494

/-- An isosceles triangle with side lengths 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      (a = b ∨ b = c ∨ c = a) →  -- Isosceles condition
      perimeter = a + b + c →  -- Perimeter definition
      perimeter = 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l254_25494


namespace NUMINAMATH_CALUDE_student_congress_size_l254_25423

/-- The number of classes in the school -/
def num_classes : ℕ := 40

/-- The number of representatives sent from each class -/
def representatives_per_class : ℕ := 3

/-- The sample size (number of students in the "Student Congress") -/
def sample_size : ℕ := num_classes * representatives_per_class

theorem student_congress_size :
  sample_size = 120 :=
by sorry

end NUMINAMATH_CALUDE_student_congress_size_l254_25423


namespace NUMINAMATH_CALUDE_dogs_not_eating_l254_25410

theorem dogs_not_eating (total : ℕ) (like_apples : ℕ) (like_chicken : ℕ) (like_both : ℕ) :
  total = 75 →
  like_apples = 18 →
  like_chicken = 55 →
  like_both = 10 →
  total - (like_apples + like_chicken - like_both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_eating_l254_25410


namespace NUMINAMATH_CALUDE_student_cannot_enter_finals_l254_25457

/-- Represents the competition structure and student's performance -/
structure Competition where
  total_rounds : ℕ
  required_specified : ℕ
  required_creative : ℕ
  min_selected_for_award : ℕ
  rounds_for_finals : ℕ
  specified_selected : ℕ
  total_specified : ℕ
  creative_selected : ℕ
  total_creative : ℕ
  prob_increase : ℚ

/-- Calculates the probability of winning the "Skillful Hands Award" in one round -/
def prob_win_award (c : Competition) : ℚ :=
  sorry

/-- Calculates the expected number of times winning the award in all rounds after intensive training -/
def expected_wins_after_training (c : Competition) : ℚ :=
  sorry

/-- Main theorem: The student cannot enter the finals -/
theorem student_cannot_enter_finals (c : Competition) 
  (h1 : c.total_rounds = 5)
  (h2 : c.required_specified = 2)
  (h3 : c.required_creative = 2)
  (h4 : c.min_selected_for_award = 3)
  (h5 : c.rounds_for_finals = 4)
  (h6 : c.specified_selected = 4)
  (h7 : c.total_specified = 5)
  (h8 : c.creative_selected = 3)
  (h9 : c.total_creative = 5)
  (h10 : c.prob_increase = 1/10) :
  prob_win_award c = 33/50 ∧ expected_wins_after_training c < 4 :=
sorry

end NUMINAMATH_CALUDE_student_cannot_enter_finals_l254_25457


namespace NUMINAMATH_CALUDE_function_property_l254_25452

theorem function_property (f : ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x + y) = f x + f y + 2 * x * y + 1) 
  (h2 : f (-2) = 1) :
  ∀ n : ℕ+, f (2 * n) = 4 * n^2 + 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_property_l254_25452


namespace NUMINAMATH_CALUDE_defective_units_shipped_l254_25425

theorem defective_units_shipped (total_units : ℝ) (defective_rate : ℝ) (shipped_rate : ℝ) :
  defective_rate = 0.04 →
  shipped_rate = 0.04 →
  (defective_rate * shipped_rate * 100) = 0.16 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l254_25425


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l254_25474

/-- Proves that the profit percent is 26% when selling an article at a certain price,
    given that selling it at 2/3 of that price results in a 16% loss. -/
theorem profit_percent_calculation (P C : ℝ) 
  (h : (2/3) * P = 0.84 * C) : 
  (P - C) / C * 100 = 26 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l254_25474


namespace NUMINAMATH_CALUDE_work_completion_men_count_l254_25492

theorem work_completion_men_count :
  ∀ (M : ℕ),
  (∃ (W : ℕ), W = M * 9) →  -- Original work amount
  (∃ (W : ℕ), W = (M + 10) * 6) →  -- Same work amount after adding 10 men
  M = 20 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l254_25492


namespace NUMINAMATH_CALUDE_digits_of_3_15_times_5_10_l254_25448

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in 3^15 * 5^10 is 14 -/
theorem digits_of_3_15_times_5_10 : num_digits (3^15 * 5^10) = 14 := by sorry

end NUMINAMATH_CALUDE_digits_of_3_15_times_5_10_l254_25448


namespace NUMINAMATH_CALUDE_tshirt_price_proof_l254_25476

/-- The regular price of a T-shirt -/
def regular_price : ℝ := 14.50

/-- The cost of a discounted T-shirt -/
def discount_price : ℝ := 1

/-- The total number of T-shirts bought -/
def total_shirts : ℕ := 12

/-- The total cost of all T-shirts -/
def total_cost : ℝ := 120

/-- The number of T-shirts in a "lot" (2 regular + 1 discounted) -/
def lot_size : ℕ := 3

theorem tshirt_price_proof :
  regular_price * (2 * (total_shirts / lot_size)) + 
  discount_price * (total_shirts / lot_size) = total_cost :=
sorry

end NUMINAMATH_CALUDE_tshirt_price_proof_l254_25476


namespace NUMINAMATH_CALUDE_special_function_property_l254_25487

/-- A real-valued function on rational numbers satisfying specific properties -/
def special_function (f : ℚ → ℝ) : Prop :=
  (f 0 = 0) ∧
  (∀ α, α ≠ 0 → f α > 0) ∧
  (∀ α β, f (α * β) = f α * f β) ∧
  (∀ α β, f (α + β) ≤ f α + f β) ∧
  (∀ m : ℤ, f m ≤ 1989)

/-- Theorem stating that f(α + β) = max{f(α), f(β)} when f(α) ≠ f(β) -/
theorem special_function_property (f : ℚ → ℝ) (h : special_function f) :
  ∀ α β : ℚ, f α ≠ f β → f (α + β) = max (f α) (f β) :=
sorry

end NUMINAMATH_CALUDE_special_function_property_l254_25487


namespace NUMINAMATH_CALUDE_bipin_twice_chandan_age_l254_25419

/-- Proves that Bipin's age will be twice Chandan's age after 10 years -/
theorem bipin_twice_chandan_age (alok_age bipin_age chandan_age : ℕ) : 
  alok_age = 5 →
  bipin_age = 6 * alok_age →
  chandan_age = 10 →
  ∃ (years : ℕ), years = 10 ∧ bipin_age + years = 2 * (chandan_age + years) :=
by
  sorry

end NUMINAMATH_CALUDE_bipin_twice_chandan_age_l254_25419


namespace NUMINAMATH_CALUDE_sin_theta_value_l254_25400

/-- Definition of determinant for 2x2 matrix -/
def det (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: If the determinant of the given matrix is 1/2, then sin θ = ±√3/2 -/
theorem sin_theta_value (θ : ℝ) (h : det (Real.sin (θ/2)) (Real.cos (θ/2)) (Real.cos (3*θ/2)) (Real.sin (3*θ/2)) = 1/2) :
  Real.sin θ = Real.sqrt 3 / 2 ∨ Real.sin θ = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l254_25400


namespace NUMINAMATH_CALUDE_circle_radius_l254_25489

theorem circle_radius (A : ℝ) (h : A = 196 * Real.pi) : 
  ∃ r : ℝ, r > 0 ∧ A = Real.pi * r^2 ∧ r = 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l254_25489
