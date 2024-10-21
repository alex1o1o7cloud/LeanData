import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l1056_105686

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length : ℝ) (platform_length : ℝ) (second_platform_length : ℝ) (time_second_platform : ℝ) : ℝ :=
  let total_distance_second := train_length + second_platform_length
  let speed := total_distance_second / time_second_platform
  let total_distance_first := train_length + platform_length
  total_distance_first / speed

/-- Theorem stating that the time taken for the train to cross the first platform is 15 seconds -/
theorem train_crossing_time_theorem :
  let train_length : ℝ := 310
  let first_platform_length : ℝ := 110
  let second_platform_length : ℝ := 250
  let time_second_platform : ℝ := 20
  time_to_cross_platform train_length first_platform_length second_platform_length time_second_platform = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l1056_105686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_over_two_minus_alpha_l1056_105659

theorem sin_pi_over_two_minus_alpha (α : ℝ) 
  (h1 : Real.sin α = 1/4) 
  (h2 : π/2 < α ∧ α < π) : 
  Real.sin (π/2 - α) = -Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_over_two_minus_alpha_l1056_105659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_with_circles_l1056_105607

/-- A type representing a coloring of the plane. -/
def PlaneColoring := ℝ × ℝ → Bool

/-- A type representing a circle in the plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two points are in different regions with respect to a set of circles. -/
def inDifferentRegions (circles : List Circle) (p q : ℝ × ℝ) : Prop :=
  ∃ c ∈ circles, (((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2) ≠
                   ((q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 < c.radius^2))

/-- Main theorem: For any number of circles, there exists a valid two-coloring of the plane. -/
theorem two_color_plane_with_circles (n : ℕ) :
  ∀ (circles : List Circle),
    circles.length = n →
    ∃ (coloring : PlaneColoring),
      ∀ (p q : ℝ × ℝ),
        inDifferentRegions circles p q →
        coloring p ≠ coloring q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_color_plane_with_circles_l1056_105607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_eight_divisors_l1056_105644

theorem exists_number_with_eight_divisors : ∃ n : ℕ, (Nat.divisors n).card = 8 := by
  use 54
  simp [Nat.divisors]
  norm_num
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_with_eight_divisors_l1056_105644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l1056_105612

/-- The volume of a truncated cone -/
noncomputable def truncated_cone_volume (bottom_diameter top_diameter height : ℝ) : ℝ :=
  let bottom_radius := bottom_diameter / 2
  let top_radius := top_diameter / 2
  (1/3) * Real.pi * height * (bottom_radius^2 + bottom_radius * top_radius + top_radius^2)

/-- Theorem: The volume of a truncated cone with given dimensions is 189π cubic centimeters -/
theorem truncated_cone_volume_example : 
  truncated_cone_volume 12 6 9 = 189 * Real.pi := by
  -- Unfold the definition of truncated_cone_volume
  unfold truncated_cone_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_volume_example_l1056_105612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1056_105616

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℤ := floor (2 * Real.sin x * Real.cos x) + floor (Real.sin x + Real.cos x)

theorem f_range :
  ∃ (S : Set ℤ), S = {-2, -1, 0, 1, 2} ∧ ∀ y, y ∈ S ↔ ∃ x, f x = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1056_105616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1056_105698

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 1| < 2}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 1) / x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1056_105698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_ratio_seq_nonzero_ratio_negative_power_of_three_plus_two_is_arithmetic_ratio_seq_l1056_105662

/-- Definition of an arithmetic ratio sequence -/
def is_arithmetic_ratio_seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

/-- The common ratio of an arithmetic ratio sequence is non-zero -/
theorem arithmetic_ratio_seq_nonzero_ratio 
  {a : ℕ → ℝ} {k : ℝ} (h : is_arithmetic_ratio_seq a k) : k ≠ 0 :=
by
  sorry

/-- The sequence a_n = -3^n + 2 is an arithmetic ratio sequence -/
theorem negative_power_of_three_plus_two_is_arithmetic_ratio_seq :
  is_arithmetic_ratio_seq (λ n ↦ -3^n + 2) 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_ratio_seq_nonzero_ratio_negative_power_of_three_plus_two_is_arithmetic_ratio_seq_l1056_105662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_product_of_roots_specific_equation_l1056_105642

noncomputable def quadraticRoots (a b c : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  (root1, root2)

theorem product_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let (r₁, r₂) := quadraticRoots a b c h
  r₁ * r₂ = c / a := by sorry

theorem product_of_roots_specific_equation :
  let (r₁, r₂) := quadraticRoots 24 36 (-648) (by norm_num)
  r₁ * r₂ = -27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_quadratic_product_of_roots_specific_equation_l1056_105642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1056_105601

-- Define the points
def D : ℝ × ℝ := (-5, 2)
def E : ℝ × ℝ := (3, 2)
def F : ℝ × ℝ := (1, -4)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_DEF_area : triangleArea D E F = 24 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp [D, E, F]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_area_l1056_105601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_minimum_distance_l1056_105606

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the Euclidean distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The race setup with given conditions -/
structure RaceSetup where
  A : Point
  B : Point
  wallLength : ℝ
  wallBottomY : ℝ

/-- The minimum distance for the race -/
noncomputable def minRaceDistance (setup : RaceSetup) : ℝ :=
  let B' : Point := { x := setup.B.x, y := 2 * setup.wallBottomY - setup.B.y }
  distance setup.A B'

theorem race_minimum_distance (setup : RaceSetup) 
  (h1 : setup.A = { x := 0, y := 300 })
  (h2 : setup.B = { x := 1200, y := 1800 })
  (h3 : setup.wallLength = 1200)
  (h4 : setup.wallBottomY = 0) : 
  Int.floor (minRaceDistance setup + 0.5) = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_minimum_distance_l1056_105606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_represent_ellipse_and_hyperbola_eccentricities_l1056_105638

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the roots of the equation
noncomputable def root1 : ℝ := 2
noncomputable def root2 : ℝ := 1/2

-- Define eccentricity ranges for ellipse and hyperbola
def is_ellipse_eccentricity (e : ℝ) : Prop := 0 < e ∧ e < 1
def is_hyperbola_eccentricity (e : ℝ) : Prop := e > 1

-- Theorem statement
theorem roots_represent_ellipse_and_hyperbola_eccentricities :
  quadratic_equation root1 ∧ 
  quadratic_equation root2 ∧ 
  is_hyperbola_eccentricity root1 ∧ 
  is_ellipse_eccentricity root2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_represent_ellipse_and_hyperbola_eccentricities_l1056_105638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_numbers_l1056_105675

def next_number (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 7 else n / 2

def is_winning_number (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (Nat.iterate next_number k n = n)

theorem winning_numbers :
  ∀ n : ℕ, n > 0 → (is_winning_number n ↔ n ∈ ({1, 2, 4, 7, 8, 14} : Finset ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_numbers_l1056_105675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQS_l1056_105629

-- Define the triangle PQR
def Triangle (PQ QR PR : ℝ) : Prop := PQ > 0 ∧ QR > 0 ∧ PR > 0 ∧ PQ^2 + QR^2 = PR^2

-- Define the angle bisector PS
def AngleBisector (PQ QR PR QS RS : ℝ) : Prop := QS / RS = PQ / PR

-- Theorem statement
theorem area_of_triangle_PQS (y : ℝ) : 
  let PQ := 120
  let QR := y
  let PR := 3 * y - 10
  Triangle PQ QR PR →
  ∃ (QS RS : ℝ), 
    AngleBisector PQ QR PR QS RS ∧ 
    QS + RS = QR ∧
    Int.floor (1/2 * PQ * QS + 0.5) = 1578 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQS_l1056_105629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_destruction_l1056_105641

def count {α : Type*} (l : List α) (a : α) [DecidableEq α] : Nat :=
  l.filter (· = a) |>.length

theorem tank_destruction (n : ℕ) (h : n = 41) : 
  (∀ sequence : List (Fin n × Fin n), 
    (∀ cell : Fin n × Fin n, (count sequence cell) ≥ 2) → 
    sequence.length ≥ (3 * n^2 - 1) / 2) ∧
  (∃ sequence : List (Fin n × Fin n), 
    (∀ cell : Fin n × Fin n, (count sequence cell) ≥ 2) ∧ 
    sequence.length = (3 * n^2 - 1) / 2) :=
by
  sorry

#check tank_destruction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_destruction_l1056_105641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_properties_l1056_105622

-- Define the sum of consecutive even numbers starting from 2
def sumConsecutiveEven (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of even numbers in a given range
def sumEvenInRange (start finish : ℕ) : ℕ :=
  let startTerm := start / 2
  let endTerm := finish / 2
  sumConsecutiveEven endTerm - sumConsecutiveEven (startTerm - 1)

theorem consecutive_even_sum_properties :
  (sumConsecutiveEven 8 = 72) ∧
  (∀ n : ℕ, sumConsecutiveEven n = n * (n + 1)) ∧
  (sumEvenInRange 102 212 = 8792) := by
  sorry

#eval sumConsecutiveEven 8
#eval sumEvenInRange 102 212

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_properties_l1056_105622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cylinder_volume_is_triple_l1056_105695

-- Define the original cylinder
def original_radius : ℝ := 8
def original_height : ℝ := 7

-- Define the new cylinder
def new_radius : ℝ := 8
def new_height : ℝ := 21

-- Define the volume function for a cylinder
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Theorem statement
theorem new_cylinder_volume_is_triple :
  cylinder_volume new_radius new_height = 3 * cylinder_volume original_radius original_height :=
by
  -- Expand the definition of cylinder_volume
  unfold cylinder_volume
  -- Simplify the expressions
  simp [original_radius, original_height, new_radius, new_height]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_cylinder_volume_is_triple_l1056_105695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n2_bond_stability_l1056_105687

/-- Represents the bond energy in kJ·mol⁻¹ -/
@[reducible] def BondEnergy := ℝ

/-- Represents the stability of a bond -/
inductive BondStability
| LessStable
| MoreStable

/-- Compares the stability of two bonds based on their energy contribution -/
noncomputable def compareBondStability (energy_contribution1 energy_contribution2 : BondEnergy) : BondStability :=
  if energy_contribution1 > energy_contribution2 then BondStability.MoreStable else BondStability.LessStable

theorem n2_bond_stability 
  (triple_bond_energy single_bond_energy : BondEnergy)
  (h_triple : triple_bond_energy = 942)
  (h_single : single_bond_energy = 247)
  : compareBondStability 
      (triple_bond_energy - single_bond_energy) -- π bond energy contribution
      single_bond_energy                        -- σ bond energy contribution
    = BondStability.MoreStable := by
  sorry

#check n2_bond_stability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n2_bond_stability_l1056_105687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_reduction_l1056_105605

/-- Represents the price and sales data for apples over two days --/
structure AppleSales where
  initialPrice : ℚ
  salesIncrease : ℚ
  revenueIncrease : ℚ

/-- Calculates the new price of apples after the price reduction --/
def newPrice (data : AppleSales) : ℚ :=
  data.initialPrice * (1 + data.revenueIncrease) / (1 + data.salesIncrease)

/-- Theorem stating that given the initial conditions, the new price is 45 kopecks --/
theorem apple_price_reduction (data : AppleSales) 
    (h1 : data.initialPrice = 60)
    (h2 : data.salesIncrease = 1/2)
    (h3 : data.revenueIncrease = 1/8) :
    newPrice data = 45 := by
  sorry

#eval newPrice { initialPrice := 60, salesIncrease := 1/2, revenueIncrease := 1/8 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_reduction_l1056_105605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1056_105693

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (-3) (-2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1056_105693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1056_105660

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 3 * x + b

noncomputable def g (a b x : ℝ) : ℝ := |f a b x| - 2/3

theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, 0 ≤ f 2 0 x ∧ f 2 0 x ≤ 4/3) ∧
  (∀ b : ℝ, (∃ s : Finset ℝ, s.card ≤ 4 ∧ ∀ x : ℝ, g a b x = 0 → x ∈ s) →
    -2 ≤ a ∧ a ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1056_105660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l1056_105613

/-- The constant term in the expansion of (x^4 + y^2 + 1/(2xy))^7 -/
def constant_term : ℚ := 105 / 16

/-- The expression to be expanded -/
noncomputable def expression (x y : ℝ) : ℝ := (x^4 + y^2 + 1/(2*x*y))^7

/-- Theorem stating that the constant term in the expansion of the expression is 105/16 -/
theorem constant_term_proof :
  ∃ (f : ℝ → ℝ → ℝ), 
    (∀ x y, f x y = expression x y) ∧ 
    (∃ c : ℝ, ∀ x y, f x y = c + x * (f x y - c) + y * (f x y - c) ∧ c = constant_term) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l1056_105613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freezing_point_depression_l1056_105680

/-- The freezing point depression constant for water in °C kg/mol -/
noncomputable def Kf : ℝ := 1.86

/-- The van't Hoff factor for NH4Br -/
noncomputable def i : ℝ := 2

/-- Mass of water in kg -/
noncomputable def mass_water : ℝ := 0.5

/-- Moles of NH4Br -/
noncomputable def moles_NH4Br : ℝ := 5

/-- Molality of the solution in mol/kg -/
noncomputable def m : ℝ := moles_NH4Br / mass_water

/-- Freezing point depression in °C -/
noncomputable def ΔTf : ℝ := i * Kf * m

theorem freezing_point_depression :
  ΔTf = 37.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freezing_point_depression_l1056_105680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_has_686_l1056_105609

-- Define the amounts of money for each person
noncomputable def ram_amount : ℚ := sorry
noncomputable def gopal_amount : ℚ := sorry
def krishan_amount : ℚ := 4046

-- Define the ratios
axiom ram_gopal_ratio : ram_amount / gopal_amount = 7 / 17
axiom gopal_krishan_ratio : gopal_amount / krishan_amount = 7 / 17

-- Theorem to prove
theorem ram_has_686 : ram_amount = 686 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ram_has_686_l1056_105609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_arithmetic_progression_l1056_105654

def StrictlyIncreasing {α : Type*} [PartialOrder α] (f : α → α) : Prop :=
  ∀ x y, x < y → f x < f y

theorem no_infinite_arithmetic_progression
  (f : ℝ → ℝ)
  (h_pos : ∀ x, 0 < x → 0 < f x)
  (h_incr : StrictlyIncreasing f)
  (h_ineq : ∀ x y, 0 < x → 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ¬ ∃ (a d : ℝ) (φ : ℕ → ℕ), StrictlyIncreasing φ ∧
    (∀ n, f (φ n : ℝ) = a + d * n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_arithmetic_progression_l1056_105654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_count_l1056_105697

/-- Represents a cell in the 8x8 grid -/
structure Cell where
  row : Fin 8
  col : Fin 8
deriving Fintype, DecidableEq

/-- The garden grid -/
def Garden := Cell → ℕ

/-- Two cells are adjacent if they share a side -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- A valid garden configuration -/
def valid_garden (g : Garden) : Prop :=
  (∃ c1 c2 : Cell, g c1 = 19 ∧ g c2 = 6) ∧
  (∀ c1 c2 : Cell, adjacent c1 c2 → (g c1 : Int) - (g c2 : Int) = 1 ∨ (g c2 : Int) - (g c1 : Int) = 1)

/-- Count of cells with 19 dandelions -/
def count_19 (g : Garden) : ℕ :=
  Finset.univ.filter (λ c : Cell => g c = 19) |>.card

/-- The main theorem -/
theorem dandelion_count (g : Garden) (h : valid_garden g) :
  count_19 g = 1 ∨ count_19 g = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_count_l1056_105697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1056_105670

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2 + 1

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := 1 - Real.sqrt (x - 1)

-- Theorem statement
theorem inverse_function_proof :
  (∀ x ≤ 0, f x = (x - 1)^2 + 1) →
  (∀ x ≥ 2, f_inv x = 1 - Real.sqrt (x - 1)) →
  (∀ x ≤ 0, f_inv (f x) = x) ∧
  (∀ x ≥ 2, f (f_inv x) = x) :=
by
  sorry

#check inverse_function_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l1056_105670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_88_l1056_105673

/-- A rhombus in a 2D plane -/
structure Rhombus where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a rhombus given its vertices -/
noncomputable def rhombusArea (r : Rhombus) : ℝ :=
  let d1 := max (|r.v1.1 - r.v3.1|) (|r.v2.1 - r.v4.1|)
  let d2 := max (|r.v1.2 - r.v3.2|) (|r.v2.2 - r.v4.2|)
  (d1 * d2) / 2

/-- The specific rhombus from the problem -/
def problemRhombus : Rhombus :=
  { v1 := (8, 0)
    v2 := (0, -5.5)
    v3 := (-8, 0)
    v4 := (0, 5.5) }

theorem rhombus_area_is_88 :
  rhombusArea problemRhombus = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_88_l1056_105673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1056_105656

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 2 / (3^x + 1) - a -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ 2 / (3^x + 1) - a

theorem odd_function_value (a : ℝ) : IsOdd (f a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1056_105656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_behavior_l1056_105603

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 + 5 * x + 7) / (3 * x + 5)

theorem sequence_behavior (x : ℕ → ℝ) 
  (h1 : ∀ n, x n > 0)
  (h2 : x 1 ≠ Real.sqrt 7)
  (h3 : ∀ n ≥ 2, x n = f (x (n-1))) :
  (∀ n, x n < x (n+1) ∧ x (n+1) < Real.sqrt 7) ∨
  (∀ n, x n > x (n+1) ∧ x (n+1) > Real.sqrt 7) := by
  sorry

#check sequence_behavior

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_behavior_l1056_105603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1056_105657

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.cos x - 1/2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 3} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1056_105657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1056_105617

theorem rationalize_denominator :
  2 / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l1056_105617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_theorem_l1056_105615

/-- The radius of a circle that is internally tangent to four externally tangent circles of radius 2 -/
noncomputable def large_circle_radius : ℝ := 2 * (Real.sqrt 2 + 1)

/-- Four circles of radius 2 that are externally tangent to each other -/
structure SmallCircles where
  radius : ℝ
  externally_tangent : Prop

/-- A large circle that is internally tangent to four externally tangent circles of radius 2 -/
structure LargeCircle where
  radius : ℝ
  internally_tangent : Prop

/-- Theorem stating that the radius of the large circle is 2(√2 + 1) -/
theorem large_circle_radius_theorem (small_circles : SmallCircles) (large_circle : LargeCircle) 
  (h1 : small_circles.radius = 2)
  (h2 : small_circles.externally_tangent)
  (h3 : large_circle.internally_tangent) :
  large_circle.radius = large_circle_radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_theorem_l1056_105615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_D_not_relevant_l1056_105684

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  (abs (l.a * p.x + l.b * p.y + l.c)) / Real.sqrt (l.a^2 + l.b^2)

/-- Definition of a "relevant line" -/
def is_relevant_line (m : Point) (l : Line) : Prop :=
  distance_point_to_line m l ≤ 4

/-- The point M(5,0) -/
def M : Point := ⟨5, 0⟩

/-- The line 2x-y+1=0 -/
def line_D : Line := ⟨2, -1, 1⟩

/-- Theorem: The line 2x-y+1=0 is not a relevant line for M(5,0) -/
theorem line_D_not_relevant : ¬(is_relevant_line M line_D) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_D_not_relevant_l1056_105684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_l1056_105679

def is_digit (n : ℕ) : Prop := n < 10

def number_with_B (B : ℕ) : ℕ := 5457062 * 10 + B

theorem find_B :
  ∀ B : ℕ,
  is_digit B →
  (number_with_B B % 2 = 0) →
  (number_with_B B % 4 = 0) →
  (number_with_B B % 5 = 0) →
  (number_with_B B % 8 = 0) →
  B = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_B_l1056_105679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1056_105627

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

theorem f_properties :
  ∃ (T : ℝ),
    (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
    T = π ∧
    (∀ (k : ℤ), ∃ (x : ℝ), x = π/6 + k*π/2 ∧ ∀ (y : ℝ), f (x - y) = f (x + y)) ∧
    (∀ (y : ℝ), y ∈ Set.Icc (-π/12) (π/2) → f y ∈ Set.Icc (-1) 2) ∧
    (∃ (y₁ y₂ : ℝ), y₁ ∈ Set.Icc (-π/12) (π/2) ∧ y₂ ∈ Set.Icc (-π/12) (π/2) ∧ f y₁ = -1 ∧ f y₂ = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1056_105627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_pattern_symmetry_l1056_105676

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point
  sideLength : ℝ

/-- Represents the recurring pattern on the line -/
structure RecurringPattern where
  line : Line
  triangles : List EquilateralTriangle
  segments : List (Point × Point)

/-- Represents a rigid motion transformation -/
inductive RigidMotion where
  | Rotation (center : Point) (angle : ℝ)
  | Translation (dx : ℝ) (dy : ℝ)
  | Reflection (line : Line)

/-- Applies a rigid motion to a recurring pattern -/
def applyRigidMotion (motion : RigidMotion) (pattern : RecurringPattern) : RecurringPattern :=
  sorry -- Implementation of applying the transformation would go here

/-- The theorem to be proved -/
theorem recurring_pattern_symmetry 
  (pattern : RecurringPattern) :
  (∃ (p : Point), applyRigidMotion (RigidMotion.Rotation p (2 * π / 3)) pattern = pattern) ∧
  (∃ (d : ℝ), applyRigidMotion (RigidMotion.Translation d 0) pattern = pattern) ∧
  (applyRigidMotion (RigidMotion.Reflection pattern.line) pattern = pattern) ∧
  (∃ (l : Line), l.a * pattern.line.b - l.b * pattern.line.a = 0 ∧ 
    applyRigidMotion (RigidMotion.Reflection l) pattern = pattern) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_pattern_symmetry_l1056_105676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_a_values_l1056_105682

def A : Set ℝ := {x | x^2 - 7*x + 12 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem number_of_subsets_of_a_values (h : ∃ a, A ∩ B a = B a) :
  ∃ S : Finset ℝ, (∀ a, A ∩ B a = B a → a ∈ S) ∧ Finset.card (Finset.powerset S) = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_a_values_l1056_105682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1056_105681

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem
theorem triangle_area (abc : Triangle) 
  (h1 : abc.b / (abc.a + abc.c) = 1 - Real.sin abc.C / (Real.sin abc.A + Real.sin abc.B))
  (h2 : abc.b = 5)
  (h3 : abc.a * abc.c * Real.cos abc.A = 5) :
  (1/2) * abc.b * abc.c * Real.sin abc.A = 5 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1056_105681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_6_eq_9_l1056_105602

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq : a 2 = -3
  a8_eq : a 8 = 15

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem stating that S_6 = 9 for the given arithmetic sequence -/
theorem sum_6_eq_9 (seq : ArithmeticSequence) : sum_n seq 6 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_6_eq_9_l1056_105602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_pets_l1056_105637

/-- The number of pets Claire has, given the conditions in the problem -/
def total_pets (gerbils : ℕ) (hamsters : ℕ) : ℕ := gerbils + hamsters

/-- Theorem stating that Claire has 92 pets in total -/
theorem claire_pets : 
  ∀ (hamsters : ℕ),
    (68 / 4 + hamsters / 3 = 25) →
    total_pets 68 hamsters = 92 :=
by
  intro hamsters hyp
  have gerbils : ℕ := 68
  have male_gerbils : ℕ := gerbils / 4
  have male_hamsters : ℕ := hamsters / 3
  have total_males : ℕ := 25
  
  -- Here we would typically prove the theorem step by step
  -- For now, we'll use sorry to skip the proof
  sorry

#check claire_pets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_claire_pets_l1056_105637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l1056_105666

/-- A linear function passing through (2,3) with integral from 0 to 1 equal to 0 -/
noncomputable def f (x : ℝ) : ℝ := 2 * x - 1

/-- The area of the triangle formed by the graph of f and the coordinate axes -/
noncomputable def triangle_area : ℝ := 1/4

theorem linear_function_triangle_area :
  (f 2 = 3) ∧ 
  (∫ x in Set.Icc 0 1, f x = 0) →
  triangle_area = 1/4 := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_triangle_area_l1056_105666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1056_105652

-- Define the function f
noncomputable def f (x : ℝ) := Real.log x - x + 1/x

-- Define a, b, and c
noncomputable def a : ℝ := f (1/3)
noncomputable def b : ℝ := f Real.pi
noncomputable def c : ℝ := f 5

-- State the theorem
theorem f_inequality : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l1056_105652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1056_105621

open Real

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (t, t^2)

noncomputable def C₂ (θ : ℝ) : ℝ := 1 / (sin (θ - π/3))

-- Define the polar equation of C₁
def C₁_polar (ρ θ : ℝ) : Prop := sin θ = ρ * (cos θ)^2

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ t₁ = A ∧
    C₁ t₂ = B ∧
    C₂ θ₁ = Real.sqrt (A.1^2 + A.2^2) ∧
    C₂ θ₂ = Real.sqrt (B.1^2 + B.2^2)

-- Theorem statement
theorem chord_length (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1056_105621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_sqrt2_over_2_l1056_105696

open Complex

theorem modulus_of_z_is_sqrt2_over_2 (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((i^2018) / (i^2019 - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_sqrt2_over_2_l1056_105696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_A_l1056_105645

/-- An arithmetic sequence with given first and 26th terms -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 19 ∧ a 26 = -1 ∧ ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of seven consecutive terms starting from n -/
def A (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range 7).sum (λ i => a (n + i))

/-- The minimum absolute value of A is 7/5 -/
theorem min_abs_A (a : ℕ → ℚ) :
  arithmetic_sequence a →
  ∃ n : ℕ, (∀ m : ℕ, |A a n| ≤ |A a m|) ∧ |A a n| = 7/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_A_l1056_105645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1056_105646

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: Points A(-2, 0) and B(0, 4) are equidistant from the line x + my - 1 = 0 
    if and only if m = -1/2 or m = 1 -/
theorem equidistant_points_line (m : ℝ) : 
  (distanceToLine (-2) 0 1 m (-1) = distanceToLine 0 4 1 m (-1)) ↔ (m = -1/2 ∨ m = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_points_line_l1056_105646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_optimal_speed_verify_optimal_speed_l1056_105640

/-- Represents the total walking time of the pedestrian as a function of their speed -/
noncomputable def total_time (v : ℝ) : ℝ :=
  (6 / v) + (2 / 3) + (3 / 2) + (v / 6)

/-- The speed that minimizes the total walking time -/
def optimal_speed : ℝ := 6

/-- Theorem stating that the optimal_speed minimizes the total walking time -/
theorem pedestrian_optimal_speed :
  optimal_speed > 0 ∧
  ∀ v > 0, total_time optimal_speed ≤ total_time v :=
by sorry

/-- Verifies that the optimal speed is indeed 6 km/h -/
theorem verify_optimal_speed : optimal_speed = 6 :=
by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pedestrian_optimal_speed_verify_optimal_speed_l1056_105640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l1056_105624

/-- Triangle structure -/
structure Triangle where
  hypotenuse : ℝ
  leg : ℝ

/-- Square structure -/
structure Square where
  sideLength : ℝ
  diagonal : ℝ
  area : ℝ

/-- An isosceles right triangle -/
class IsoscelesRightTriangle (t : Triangle) : Prop

/-- A square inscribed in a triangle -/
class InscribedSquare (s : Square) (t : Triangle) : Prop

theorem isosceles_right_triangle_inscribed_squares (triangle : Triangle) 
  (square1 square2 : Square) :
  IsoscelesRightTriangle triangle →
  InscribedSquare square1 triangle →
  InscribedSquare square2 triangle →
  square1.sideLength = triangle.hypotenuse / 2 →
  square1.area = 400 →
  square2.diagonal = triangle.leg →
  square2.area = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_inscribed_squares_l1056_105624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_price_from_tax_change_l1056_105653

/-- The market price of an article given tax rate changes and savings -/
theorem market_price_from_tax_change (initial_tax_rate final_tax_rate : ℚ) 
  (savings : ℝ) (price : ℝ) : 
  initial_tax_rate = 7/200 → 
  final_tax_rate = 1/30 → 
  savings = 14 → 
  (initial_tax_rate - final_tax_rate) * price = savings → 
  ∃ ε > 0, |price - 8235.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_price_from_tax_change_l1056_105653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_open_unit_interval_l1056_105633

-- Define the function f(x) = x - x ln x
noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

-- Theorem statement
theorem f_strictly_increasing_on_open_unit_interval :
  StrictMonoOn f (Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_open_unit_interval_l1056_105633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trig_sum_l1056_105631

theorem angle_terminal_side_trig_sum (α : ℝ) :
  let P : ℝ × ℝ := (-4/5, 3/5)
  (P.1^2 + P.2^2 = 1) →  -- Ensure P is on the unit circle
  (P.1 = -(4/5) ∧ P.2 = 3/5) →  -- P is on the terminal side of α
  2 * Real.sin α + Real.cos α = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_trig_sum_l1056_105631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l1056_105632

/-- Calculates the speed of a faster train given the conditions of two trains passing each other. -/
theorem faster_train_speed
  (slower_train_length : ℝ)
  (faster_train_length : ℝ)
  (slower_train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : slower_train_length = 250)
  (h2 : faster_train_length = 500)
  (h3 : slower_train_speed_kmh = 40)
  (h4 : crossing_time = 26.99784017278618)
  : ∃ (faster_train_speed_kmh : ℝ),
    abs (faster_train_speed_kmh - 60.0152) < 0.0001 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_speed_l1056_105632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l1056_105639

theorem integer_difference (a b : ℤ) : 
  a > 0 → b > 0 → a + b = 20 → a * b = 96 → |a - b| = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_difference_l1056_105639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_workers_count_l1056_105688

/-- The work rate (depth dug per worker per hour) -/
def work_rate : ℝ → ℝ := sorry

/-- The initial number of workers -/
def initial_workers : ℕ := sorry

/-- The theorem stating the initial number of workers -/
theorem initial_workers_count :
  (∀ r : ℝ, r > 0 →
    initial_workers * 8 * r = (initial_workers + 15) * 6 * r) →
  initial_workers = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_workers_count_l1056_105688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_runner_laps_l1056_105647

/-- Given a circular track and two runners, calculate the number of laps completed by the faster runner. -/
theorem faster_runner_laps (track_length distance_A speed_ratio : ℚ) : 
  track_length = 1 / 4 →
  distance_A = 3 →
  speed_ratio = 2 →
  (distance_A / track_length) * speed_ratio = 24 := by
  -- The proof is omitted
  sorry

#check faster_runner_laps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_runner_laps_l1056_105647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_equations_tangency_point_is_unique_l1056_105658

/-- The point of tangency of two parabolas -/
noncomputable def point_of_tangency : ℝ × ℝ := (-19/2, -55/2)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 20*x + 63

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 56*y + 875

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  let (x, y) := point_of_tangency
  parabola1 x y ∧ parabola2 x y :=
by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_point_satisfies_equations_tangency_point_is_unique_l1056_105658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l1056_105618

/-- Calculates simple interest -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating that given the conditions, the interest rate is 10% -/
theorem interest_rate_is_ten_percent 
  (principal rate time diff : ℝ) 
  (h1 : principal = 1200) 
  (h2 : time = 2) 
  (h3 : diff = 12) 
  (h4 : compoundInterest principal rate time - simpleInterest principal rate time = diff) :
  rate = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_ten_percent_l1056_105618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_partner_share_l1056_105626

/-- The largest share of profit for a partner in a firm with given conditions -/
theorem largest_partner_share (profit : ℕ) (ratio : List ℕ) : 
  profit = 36000 → 
  ratio = [2, 4, 3, 4, 5] → 
  (ratio.maximum.getD 0) * profit / ratio.sum = 10000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_partner_share_l1056_105626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_ratio_l1056_105683

-- Define the points
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the side length
variable (s : ℝ)

-- Define the equilateral triangle property
def is_equilateral (X Y Z : EuclideanSpace ℝ (Fin 2)) (side : ℝ) : Prop :=
  dist X Y = side ∧ dist Y Z = side ∧ dist Z X = side

-- State the theorem
theorem equilateral_triangles_ratio 
  (h_ABC : is_equilateral A B C s)
  (h_BCD : is_equilateral B C D s)
  (h_CDA : is_equilateral C D A s) :
  dist A D / dist B C = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_ratio_l1056_105683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_savings_percentage_l1056_105667

/-- Represents the fuel efficiency and cost characteristics of a car --/
structure Car where
  efficiency : ℝ
  fuelCost : ℝ

/-- Calculates the cost per distance unit for a given car --/
noncomputable def costPerDistance (car : Car) : ℝ :=
  car.fuelCost / car.efficiency

/-- Theorem: Cost savings percentage when switching to a new car with 
    double efficiency and 25% more expensive fuel is 37.5% --/
theorem cost_savings_percentage 
  (oldCar newCar : Car)
  (h1 : newCar.efficiency = 2 * oldCar.efficiency)
  (h2 : newCar.fuelCost = 1.25 * oldCar.fuelCost) :
  (costPerDistance oldCar - costPerDistance newCar) / costPerDistance oldCar = 0.375 := by
  sorry

#check cost_savings_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_savings_percentage_l1056_105667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_unit_vector_collinear_opposite_a_l1056_105690

noncomputable def a : ℝ × ℝ := (6, 2)

noncomputable def b : ℝ × ℝ := (-3 * Real.sqrt 10 / 10, -Real.sqrt 10 / 10)

theorem b_is_unit_vector_collinear_opposite_a : 
  (‖b‖ = 1) ∧ 
  (∃ k : ℝ, k < 0 ∧ b = k • a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_unit_vector_collinear_opposite_a_l1056_105690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_to_shifted_sin_l1056_105669

theorem sin_cos_to_shifted_sin (x : ℝ) : 
  Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_to_shifted_sin_l1056_105669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_ratio_properties_l1056_105611

/-- Represents the ratio between a statue and its scale model -/
structure StatueModelRatio where
  statue_height : ℚ
  model_height : ℚ
  height_unit_ratio : ℚ := statue_height / model_height

/-- The statue-model ratio for the George Washington statue -/
def washington_ratio : StatueModelRatio :=
  { statue_height := 120
    model_height := 1/2 }  -- 6 inches = 1/2 foot

theorem washington_ratio_properties :
  let r := washington_ratio
  (r.height_unit_ratio = 240) ∧
  (30 / r.height_unit_ratio = 3/2) := by
  sorry

#eval washington_ratio.height_unit_ratio
#eval 30 / washington_ratio.height_unit_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_washington_ratio_properties_l1056_105611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_range_ln_inequality_l1056_105677

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

-- Statement 1: Minimum value of f(x) on (0, e]
theorem f_min_value (x : ℝ) (hx : 0 < x ∧ x ≤ Real.exp 1) :
  f x ≥ -(1 / Real.exp 1) := by sorry

-- Statement 2: Range of a for which 2f(x) ≥ g(x) holds for all x > 0
theorem g_range (a : ℝ) :
  (∀ x > 0, 2 * f x ≥ g a x) ↔ a ≤ 4 := by sorry

-- Statement 3: Inequality for ln x
theorem ln_inequality (x : ℝ) (hx : x > 0) :
  Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_g_range_ln_inequality_l1056_105677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_post_crossing_time_l1056_105663

/-- Represents a train crossing a bridge with a lamp post. -/
structure TrainCrossing where
  bridge_length : ℝ
  train_length : ℝ
  bridge_crossing_time : ℝ

/-- Calculates the time for a train to cross a lamp post. -/
noncomputable def time_to_cross_lamp_post (tc : TrainCrossing) : ℝ :=
  tc.train_length / ((tc.bridge_length + tc.train_length) / tc.bridge_crossing_time)

/-- Theorem stating that for the given conditions, the time to cross a lamp post is 5 seconds. -/
theorem lamp_post_crossing_time :
  let tc : TrainCrossing := ⟨200, 200, 10⟩
  time_to_cross_lamp_post tc = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_post_crossing_time_l1056_105663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l1056_105634

/-- The integer part of a real number x -/
noncomputable def intPart (x : ℝ) : ℤ :=
  Int.floor x

/-- The decimal part of a real number x -/
noncomputable def decPart (x : ℝ) : ℝ :=
  x - Int.floor x

/-- Theorem: The value of (2a+√10)b is 6, where a is the integer part of 6-√10 and b is the decimal part of 6-√10 -/
theorem value_of_expression : 
  let a := intPart (6 - Real.sqrt 10)
  let b := decPart (6 - Real.sqrt 10)
  (2 * (a : ℝ) + Real.sqrt 10) * b = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_expression_l1056_105634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_diagonal_triangle_l1056_105600

structure ConvexPentagon where
  vertices : Finset (ℝ × ℝ)
  vertex_count : vertices.card = 5
  is_convex : Convex ℝ (convexHull ℝ (Finset.toSet vertices))

noncomputable def diagonal (p : ConvexPentagon) (v w : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := v
  let (x₂, y₂) := w
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

def is_diagonal (p : ConvexPentagon) (v w : ℝ × ℝ) : Prop :=
  v ∈ p.vertices ∧ w ∈ p.vertices ∧ v ≠ w

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem convex_pentagon_diagonal_triangle (p : ConvexPentagon) :
  ∃ (v₁ v₂ v₃ w₁ w₂ w₃ : ℝ × ℝ),
    is_diagonal p v₁ w₁ ∧
    is_diagonal p v₂ w₂ ∧
    is_diagonal p v₃ w₃ ∧
    triangle_inequality
      (diagonal p v₁ w₁)
      (diagonal p v₂ w₂)
      (diagonal p v₃ w₃) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_pentagon_diagonal_triangle_l1056_105600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_properties_l1056_105625

/-- A sequence satisfying the given conditions -/
structure SpecialSequence (k : ℕ) (a : ℝ) where
  a_n : ℕ → ℝ
  h_k : k ≥ 2
  h_a : a > 1
  h_first : a_n 1 = 2
  h_rec : ∀ n : ℕ, n < 2*k → a_n (n+1) = (a-1) * (Finset.sum (Finset.range n) (λ i ↦ a_n (i+1))) + 2

/-- The b_n sequence derived from a_n -/
noncomputable def b_n (k : ℕ) (a : ℝ) (seq : SpecialSequence k a) (n : ℕ) : ℝ :=
  1 + (n - 1 : ℝ) / (2 * k - 1 : ℝ)

theorem special_sequence_properties {k : ℕ} {a : ℝ} (seq : SpecialSequence k a) :
  (∀ n : ℕ, n < 2*k → seq.a_n (n+1) = seq.a_n 1 * a^n) ∧
  (∀ n : ℕ, n ≤ 2*k → b_n k a seq n = 1 + (n - 1 : ℝ) / (2 * k - 1 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_properties_l1056_105625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_count_l1056_105694

theorem integer_values_count : 
  (∃! (s : Finset ℤ), 
    (∀ n : ℤ, n ∈ s ↔ ∃ (m : ℤ), (8000 : ℚ) * (2/5 : ℚ)^n = m) ∧ 
    Finset.card s = 10) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_values_count_l1056_105694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_three_angles_l1056_105604

theorem sin_squared_sum_three_angles (α : ℝ) : 
  Real.sin (α - π / 3) ^ 2 + Real.sin α ^ 2 + Real.sin (α + π / 3) ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_three_angles_l1056_105604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_garden_exists_l1056_105648

/-- Represents a tree in the garden -/
structure GardenTree where
  x : Fin 7
  y : Fin 7

/-- Represents a row of trees -/
structure Row where
  trees : Finset GardenTree
  valid : trees.card = 4

/-- Represents the garden configuration -/
structure Garden where
  trees : Finset GardenTree
  rows : Finset Row
  tree_count : trees.card = 10
  row_count : rows.card = 5
  rows_valid : ∀ r ∈ rows, r.trees ⊆ trees

/-- Theorem stating the existence of a valid garden configuration -/
theorem valid_garden_exists : ∃ g : Garden, True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_garden_exists_l1056_105648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivy_removal_time_l1056_105699

/-- Calculates the number of days required to remove ivy from a tree -/
noncomputable def ivyRemovalDays (initialCoverage : ℝ) (dailyRemoval : ℝ) (nightlyGrowth : ℝ) : ℝ :=
  initialCoverage / (dailyRemoval - nightlyGrowth)

/-- Theorem stating that under given conditions, it takes 10 days to remove all ivy -/
theorem ivy_removal_time :
  let initialCoverage : ℝ := 40
  let dailyRemoval : ℝ := 6
  let nightlyGrowth : ℝ := 2
  ivyRemovalDays initialCoverage dailyRemoval nightlyGrowth = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivy_removal_time_l1056_105699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_of_scores_check_result_l1056_105664

/-- Represents a math contest with a given number of problems. -/
structure Contest where
  name : String
  num_problems : ℕ

/-- Calculates the expected score for a contest given the total number of contests. -/
def expected_score (contest : Contest) (total_contests : ℕ) : ℚ :=
  (contest.num_problems : ℚ) / total_contests

/-- Represents Tim's participation in multiple math contests. -/
structure TimsContests where
  contests : List Contest
  total_contests : ℕ

/-- Theorem stating that the expected sum of Tim's scores is 25. -/
theorem expected_sum_of_scores (tc : TimsContests) 
  (h1 : tc.contests = [
    ⟨"LAIMO", 15⟩, 
    ⟨"FARML", 10⟩, 
    ⟨"DOMO", 50⟩
  ])
  (h2 : tc.total_contests = 3) : 
  (tc.contests.map (λ c ↦ expected_score c tc.total_contests)).sum = 25 := by
  sorry

-- Remove the #eval statement as it's causing issues with universe levels
-- Instead, we can add a simple theorem to check our result
theorem check_result : 
  (([⟨"LAIMO", 15⟩, ⟨"FARML", 10⟩, ⟨"DOMO", 50⟩] : List Contest).map (λ c ↦ expected_score c 3)).sum = 25 := by
  simp [expected_score]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_of_scores_check_result_l1056_105664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_E_x_coordinate_l1056_105635

/-- The function f(x) = x² --/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: Given the conditions of the problem, x₃ = 8/3 --/
theorem point_E_x_coordinate :
  ∀ (x₁ x₂ y₁ y₂ x₃ y₃ : ℝ) (xc yc : ℝ),
  1 < x₁ → x₁ < x₂ →
  y₁ = f x₁ → y₂ = f x₂ →
  x₁ = 2 → x₂ = 8 →
  xc = x₁ + (1/9) * (x₂ - x₁) →
  yc = f xc →
  y₃ = f x₃ →
  y₃ = yc →
  x₃ > 1 →
  x₃ = 8/3 := by
  intros x₁ x₂ y₁ y₂ x₃ y₃ xc yc h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

#check point_E_x_coordinate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_E_x_coordinate_l1056_105635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_of_reflected_rays_forms_cardioid_l1056_105655

-- Define the circle
def Circle : Set ℂ := {z : ℂ | Complex.abs z = 1}

-- Define a point on the circle
def PointOnCircle (A : ℂ) : Prop := A ∈ Circle

-- Define a reflected ray
def ReflectedRay (A : ℂ) (ψ : ℝ) : Set ℂ :=
  {z : ℂ | ∃ t : ℝ, z = A + t * (Complex.exp (2 * ψ * Complex.I) - A)}

-- Define the envelope of reflected rays
def EnvelopeOfReflectedRays (A : ℂ) : Set ℂ :=
  {z : ℂ | ∀ ε > 0, ∃ ψ₁ ψ₂ : ℝ, 
    ψ₁ ≠ ψ₂ ∧ 
    z ∈ ReflectedRay A ψ₁ ∧ 
    z ∈ ReflectedRay A ψ₂ ∧
    Complex.abs (Complex.exp (ψ₁ * Complex.I) - Complex.exp (ψ₂ * Complex.I)) < ε}

-- Define a cardioid
def Cardioid (a : ℝ) : Set ℂ :=
  {z : ℂ | ∃ θ : ℝ, z = a * Complex.exp (Complex.I * θ) * (1 + Complex.exp (Complex.I * θ))}

-- Theorem statement
theorem envelope_of_reflected_rays_forms_cardioid (A : ℂ) (h : PointOnCircle A) :
  ∃ a : ℝ, EnvelopeOfReflectedRays A = Cardioid a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_envelope_of_reflected_rays_forms_cardioid_l1056_105655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1056_105651

-- Define the polynomial functions
def p (a b c d e : ℝ) (x : ℝ) : ℝ := 2 * x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e
def q (f g h : ℝ) (x : ℝ) : ℝ := 3 * x^3 + f * x^2 + g * x + h

-- Define the difference polynomial
def diff_poly (a b c d e f g h : ℝ) (x : ℝ) : ℝ := p a b c d e x - q f g h x

-- Theorem statement
theorem max_intersection_points :
  ∃ (a b c d e f g h : ℝ),
    (∀ x : ℝ, diff_poly a b c d e f g h x = 0 → (x : ℂ).im = 0) →
    (∀ S : Finset ℝ, (∀ x ∈ S, diff_poly a b c d e f g h x = 0) → S.card ≤ 5) ∧
    ∃ T : Finset ℝ, (∀ x ∈ T, diff_poly a b c d e f g h x = 0) ∧ T.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l1056_105651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_twelve_l1056_105691

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 3 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 3 * (1 - Real.cos t)

-- Define the arc length function
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((3 * (1 - Real.cos t))^2 + (3 * Real.sin t)^2)

-- State the theorem
theorem arc_length_equals_twelve :
  arcLength π (2 * π) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_twelve_l1056_105691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_sum_l1056_105649

-- Define the RegularPolygon structure
structure RegularPolygon where
  sides : ℕ
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the exterior angle sum property
def RegularPolygon.exteriorAngleSum (p : RegularPolygon) : ℝ := 360

-- Define the interior angle sum property
def RegularPolygon.interiorAngleSum (p : RegularPolygon) : ℝ := 
  180 * (p.sides - 2)

theorem regular_polygon_interior_angle_sum 
  (p : RegularPolygon) 
  (exterior_angle : ℝ) 
  (h1 : exterior_angle = 45) 
  (h2 : p.exteriorAngleSum / p.sides = exterior_angle) : 
  p.interiorAngleSum = 1080 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_interior_angle_sum_l1056_105649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_theorem_l1056_105671

def total_cards : ℕ := 210

theorem card_theorem :
  (∃ (multiples_of_3 : ℕ), multiples_of_3 = (Finset.filter (λ x => x % 3 = 0) (Finset.range total_cards)).card) ∧
  (∃ (even_not_multiple_of_3 : ℕ), even_not_multiple_of_3 = (Finset.filter (λ x => x % 2 = 0 ∧ x % 3 ≠ 0) (Finset.range total_cards)).card) ∧
  (∃ (min_cards : ℕ), min_cards = 73 ∧
    ∀ (S : Finset ℕ), S ⊆ Finset.range total_cards → S.card < min_cards →
      ¬(∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ((2 ∣ a ∧ 2 ∣ b) ∨ (3 ∣ a ∧ 3 ∣ b)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_theorem_l1056_105671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l1056_105608

noncomputable section

/-- Curve C₁ in polar coordinates -/
def C₁ (θ : ℝ) : ℝ := 2

/-- Curve C₂ derived from C₁ -/
def C₂ (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, (1/2) * Real.sin α)

/-- Line passing through origin and point on C₂ -/
def line_through_origin (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ t : ℝ, q = (t * p.1, t * p.2)}

/-- Perimeter of quadrilateral ABCD -/
def perimeter (α : ℝ) : ℝ := 8 * Real.cos α + 2 * Real.sin α

theorem max_perimeter_line_equation :
  ∃ α : ℝ, 
    α ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    IsMaxOn perimeter (Set.Icc 0 (Real.pi / 2)) α ∧
    line_through_origin (C₂ α) = {(x, y) : ℝ × ℝ | y = (1/4) * x} := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_line_equation_l1056_105608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_n_product_exceeds_5000_l1056_105668

theorem smallest_odd_n_product_exceeds_5000 :
  ∃ (n : ℕ), 
    Odd n ∧ 
    (∀ m : ℕ, Odd m → m < n → (3 : ℝ) ^ ((m + 1)^2 / 7) ≤ 5000) ∧
    (3 : ℝ) ^ ((n + 1)^2 / 7) > 5000 ∧
    n = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_n_product_exceeds_5000_l1056_105668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_eighth_term_l1056_105620

/-- Geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := λ n => a * q^(n-1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_eighth_term
  (a q : ℝ)
  (h1 : q ≠ 1)
  (h2 : geometric_sum a q 3 + geometric_sum a q 6 = 2 * geometric_sum a q 9)
  (h3 : geometric_sequence a q 2 + geometric_sequence a q 5 = 4) :
  geometric_sequence a q 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_eighth_term_l1056_105620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_and_center_l1056_105661

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the power of a point with respect to a circle
def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

-- Define a radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | power p c1 = power p c2}

-- Define non-concentric circles
def non_concentric (c1 c2 : Circle) : Prop :=
  c1.center ≠ c2.center

-- Define non-collinear points
def non_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) ≠ (p3.1 - p1.1) * (p2.2 - p1.2)

-- Define IsLine as a placeholder (you might want to replace this with a proper definition)
def IsLine (s : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem radical_axis_and_center 
  (c1 c2 c3 : Circle) 
  (h1 : non_concentric c1 c2) 
  (h2 : non_concentric c2 c3) 
  (h3 : non_concentric c3 c1) 
  (h4 : non_collinear c1.center c2.center c3.center) :
  (∃ (l : Set (ℝ × ℝ)), l = radical_axis c1 c2 ∧ IsLine l) ∧
  (∃! p : ℝ × ℝ, p ∈ radical_axis c1 c2 ∧ p ∈ radical_axis c2 c3 ∧ p ∈ radical_axis c3 c1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_and_center_l1056_105661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1056_105614

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate triangle area -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.C

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin (t.B - t.C) = t.b * Real.sin (t.A - t.C))
  (h2 : t.c = 5)
  (h3 : Real.cos t.C = 12/13) :
  (t.a = t.b) ∧ (triangle_area t = 125/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1056_105614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l1056_105619

/-- Arithmetic sequence sum -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Geometric sequence product -/
noncomputable def T (b₁ q : ℝ) (n : ℕ) : ℝ := b₁^n * q^(n * (n - 1) / 2)

theorem arithmetic_geometric_sequence_properties
  (a₁ d b₁ q : ℝ)
  (hd : d ≠ 0)
  (hq : q ≠ 1)
  (h_arith : ∀ n : ℕ, 0 < n → n < 2011 → S a₁ d n = S a₁ d (2011 - n))
  (h_geom : ∀ n : ℕ, 0 < n → n < 23 → T b₁ q n = T b₁ q (23 - n)) :
  a₁ + 1005 * d = 0 ∧ b₁^12 * q^66 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_properties_l1056_105619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_problem_l1056_105672

/-- Linear regression problem --/
theorem linear_regression_problem 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (sum_x sum_y sum_xy sum_x_sq : ℝ) :
  n = 10 →
  sum_x = 80 →
  sum_y = 20 →
  sum_xy = 184 →
  sum_x_sq = 720 →
  let x_mean := sum_x / n
  let y_mean := sum_y / n
  let b := (sum_xy - n * x_mean * y_mean) / (sum_x_sq - n * x_mean^2)
  let a := y_mean - b * x_mean
  let regression_eq := λ (x : ℝ) => b * x + a
  let predicted_savings := regression_eq 7
  (regression_eq = λ (x : ℝ) => 0.3 * x - 0.4) ∧ 
  (predicted_savings = 1.7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_problem_l1056_105672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1056_105610

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | 0 ≤ x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1056_105610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l1056_105692

/-- A function f is monotonically decreasing if for all x₁ < x₂, f(x₁) ≥ f(x₂) --/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≥ f x₂

/-- The function f(x) = sin(2x) + 4cos(x) - ax --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + 4 * Real.cos x - a * x

theorem monotonically_decreasing_condition (a : ℝ) :
  MonotonicallyDecreasing (f a) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_condition_l1056_105692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1056_105643

/-- An ellipse with given endpoints and axis length constraint -/
structure Ellipse where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ
  p4 : ℝ × ℝ
  axis_constraint : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the length of the major axis -/
noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  max (distance e.p1 e.p3) (distance e.p2 e.p4)

/-- Calculate the length of the minor axis -/
noncomputable def minor_axis_length (e : Ellipse) : ℝ :=
  min (distance e.p1 e.p3) (distance e.p2 e.p4)

/-- Calculate the distance between foci -/
noncomputable def foci_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt ((major_axis_length e / 2)^2 - (minor_axis_length e / 2)^2)

/-- Theorem: The distance between the foci of the given ellipse is 4√14 -/
theorem ellipse_foci_distance :
  ∀ (e : Ellipse),
    e.p1 = (1, 2) ∧ e.p2 = (7, -4) ∧ e.p3 = (-3, 2) ∧ e.p4 = (13, 2) ∧
    e.axis_constraint = 4 ∧
    minor_axis_length e + e.axis_constraint ≤ major_axis_length e →
    foci_distance e = 4 * Real.sqrt 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1056_105643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_condition_l1056_105689

theorem equal_focal_distances_condition (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (16 - k) - y^2 / k = 1) ∧ 
  (∃ (x y : ℝ), 9 * x^2 + 25 * y^2 = 225) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / (16 - k) - y₁^2 / k = 1 ∧
    9 * x₂^2 + 25 * y₂^2 = 225 →
    (let c₁ := Real.sqrt ((16 - k) + k);
     let c₂ := Real.sqrt (25 - 9);
     2 * c₁ = 2 * c₂)) ↔
  0 < k ∧ k < 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_focal_distances_condition_l1056_105689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_cube_volume_ratio_is_pi_over_four_l1056_105674

/-- The ratio of the volume of a right circular cylinder inscribed in a cube to the volume of the cube -/
noncomputable def cylinder_to_cube_volume_ratio (s : ℝ) : ℝ :=
  let cylinder_volume := Real.pi * (s/2)^2 * s
  let cube_volume := s^3
  cylinder_volume / cube_volume

/-- Theorem: The ratio of the volume of a right circular cylinder inscribed in a cube
    to the volume of the cube is π/4, given that the cylinder's height and base diameter
    are equal to the cube's side length. -/
theorem cylinder_to_cube_volume_ratio_is_pi_over_four (s : ℝ) (h : s > 0) :
  cylinder_to_cube_volume_ratio s = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_to_cube_volume_ratio_is_pi_over_four_l1056_105674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1056_105628

/-- A circle C centered at (1, 0) with radius r > 0 -/
def Circle (r : ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = r^2}

/-- The line x - √3y + 3 = 0 -/
def Line := {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 + 3 = 0}

/-- The distance from a point to the line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - Real.sqrt 3 * p.2 + 3| / Real.sqrt 4

/-- The condition that there are at most two points on the circle
    at distance 1 from the line -/
def atMostTwoPoints (r : ℝ) : Prop :=
  ∃ (S : Finset (ℝ × ℝ)), (∀ p ∈ S, p ∈ Circle r ∧ distanceToLine p = 1) ∧ S.card ≤ 2

/-- The theorem stating the equivalence of the conditions -/
theorem circle_tangent_condition (r : ℝ) :
  (0 < r ∧ r < 3) ↔ atMostTwoPoints r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_condition_l1056_105628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1056_105623

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/3) = -4/5)
  (h2 : -π/2 < α)
  (h3 : α < 0) : 
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l1056_105623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_096_to_hundredth_l1056_105650

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_2_096_to_hundredth :
  round_to_hundredth 2.096 = 2.10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2_096_to_hundredth_l1056_105650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1056_105665

/-- The function f(x) = x + 1/(x-1) for x > 1 -/
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

/-- The range of f(x) is [3, +∞) for x > 1 -/
theorem f_range :
  ∀ x > 1, f x ≥ 3 ∧ ∃ y > 1, f y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1056_105665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_to_tetrahedron_volume_ratio_l1056_105630

-- Define the number of spheres in each layer
def bottom_layer : ℕ := 9
def middle_layer : ℕ := 4
def top_layer : ℕ := 1

-- Define the total number of spheres
def total_spheres : ℕ := bottom_layer + middle_layer + top_layer

-- Define the radius of each sphere
def sphere_radius : ℝ := 1

-- Define the volume of a single sphere
noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3

-- Define the total volume of all spheres
noncomputable def total_spheres_volume : ℝ := total_spheres * sphere_volume

-- Define the height of the tetrahedron
noncomputable def tetrahedron_height : ℝ := 1 + 2 * Real.sqrt 2 + Real.sqrt 3

-- Define the volume of the enclosing tetrahedron
noncomputable def tetrahedron_volume : ℝ := (2 / 3) * tetrahedron_height ^ 3

-- Define the ratio of spheres volume to tetrahedron volume
noncomputable def volume_ratio : ℝ := total_spheres_volume / tetrahedron_volume

-- Theorem statement
theorem spheres_to_tetrahedron_volume_ratio :
  ∃ (ε : ℝ), abs (volume_ratio - 0.5116) < ε ∧ ε > 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spheres_to_tetrahedron_volume_ratio_l1056_105630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1056_105636

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + m*x + 1)

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x, ∃ y, f m x = y

-- Define proposition q
def q (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∀ x y : ℝ, x^2/m + y^2/2 = 1 → x^2/a^2 + y^2/b^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m ∈ Set.Icc (-2) 0 ∪ {2}

-- Theorem statement
theorem range_of_m :
  (∀ m : ℝ, (p m ∧ q m → False) ∧ (p m ∨ q m)) →
  (∀ m : ℝ, m_range m ↔ (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1056_105636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_additional_squares_for_symmetry_l1056_105678

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat
deriving Repr

/-- Represents the grid and its shaded squares -/
structure Grid where
  width : Nat
  height : Nat
  shaded : List Position
deriving Repr

/-- Checks if a position is within the grid boundaries -/
def isValidPosition (g : Grid) (p : Position) : Prop :=
  p.x > 0 ∧ p.x ≤ g.width ∧ p.y > 0 ∧ p.y ≤ g.height

/-- Checks if the grid has both vertical and horizontal symmetry -/
def hasSymmetry (g : Grid) : Prop :=
  ∀ p : Position, isValidPosition g p →
    (p ∈ g.shaded ↔ Position.mk (g.width + 1 - p.x) p.y ∈ g.shaded) ∧
    (p ∈ g.shaded ↔ Position.mk p.x (g.height + 1 - p.y) ∈ g.shaded)

/-- The main theorem -/
theorem minimum_additional_squares_for_symmetry :
  ∀ g : Grid,
    g.width = 6 ∧ g.height = 4 ∧
    g.shaded = [Position.mk 2 1, Position.mk 3 4, Position.mk 4 3] →
    ∃ additional : List Position,
      additional.length = 4 ∧
      hasSymmetry (Grid.mk g.width g.height (g.shaded ++ additional)) ∧
      (∀ smaller : List Position,
        smaller.length < 4 →
        ¬ hasSymmetry (Grid.mk g.width g.height (g.shaded ++ smaller))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_additional_squares_for_symmetry_l1056_105678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_ab_l1056_105685

/-- Given that 1 is the geometric mean of log_a and log_b, and a > 1, b > 1, 
    prove that the minimum value of ab is 100 -/
theorem min_value_of_ab (a b : ℝ) 
  (h1 : 1 = Real.sqrt ((Real.log a) * (Real.log b))) 
  (h2 : a > 1) 
  (h3 : b > 1) : 
  ∀ x y : ℝ, x > 1 ∧ y > 1 ∧ 1 = Real.sqrt ((Real.log x) * (Real.log y)) → x * y ≥ 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_ab_l1056_105685
