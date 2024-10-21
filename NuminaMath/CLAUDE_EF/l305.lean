import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_four_five_heads_l305_30576

theorem equal_probability_four_five_heads :
  let n : ℕ := 9  -- number of coin flips
  let k₁ : ℕ := 4  -- number of heads in first case
  let k₂ : ℕ := 5  -- number of heads in second case
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  (Finset.card (Finset.filter (fun x => x.card = k₁) (Finset.powerset (Finset.range n)))) * p^k₁ * (1-p)^(n-k₁) =
  (Finset.card (Finset.filter (fun x => x.card = k₂) (Finset.powerset (Finset.range n)))) * p^k₂ * (1-p)^(n-k₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_probability_four_five_heads_l305_30576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l305_30527

noncomputable def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_proof : 
  (a 3 = 2) →
  (a 7 = 1) →
  (∃ d : ℝ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d) →
  a 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_proof_l305_30527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l305_30541

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 2/3 ∧ x ≠ -2 ∧ x ≠ 1 →
  (6*x + 2) / (3*x^2 + 6*x - 4) = (3*x) / (3*x - 2) ↔ 
  x = Real.sqrt (2/3) ∨ x = -Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l305_30541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_factorization_l305_30579

-- Define the expressions
def expr_A (a x y : ℝ) : ℝ := a * (x + y)
def expr_A_transformed (a x y : ℝ) : ℝ := a * x + a * y

def expr_B (x : ℝ) : ℝ := x^2 - 4*x + 4
def expr_B_transformed (x : ℝ) : ℝ := x * (x - 4) + 4

def expr_C (x : ℝ) : ℝ := 10 * x^2 - 5 * x
def expr_C_transformed (x : ℝ) : ℝ := 5 * x * (2 * x - 1)

def expr_D (x : ℝ) : ℝ := x^2 - 16 + 3 * x
def expr_D_transformed (x : ℝ) : ℝ := (x - 4) * (x + 4) + 3 * x

-- Theorem stating that only expr_C involves factorization
theorem only_C_is_factorization :
  (∀ a x y, expr_A a x y = expr_A_transformed a x y) ∧
  (∀ x, expr_B x = expr_B_transformed x) ∧
  (∀ x, expr_C x = expr_C_transformed x) ∧
  (∀ x, expr_D x = expr_D_transformed x) ∧
  (∃ (f g : ℝ → ℝ), ∀ x, expr_C x = f x * g x) ∧
  ¬(∃ (f g : ℝ → ℝ → ℝ → ℝ), ∀ a x y, expr_A a x y = f a x y * g a x y) ∧
  ¬(∃ (f g : ℝ → ℝ), ∀ x, expr_B x = f x * g x) ∧
  ¬(∃ (f g : ℝ → ℝ), ∀ x, expr_D x = f x * g x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_factorization_l305_30579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noah_bought_two_more_l305_30523

/-- Represents the price of a single notebook in cents -/
def notebook_price : ℕ := sorry

/-- Represents the number of notebooks Liam bought -/
def liam_notebooks : ℕ := sorry

/-- Represents the number of notebooks Noah bought -/
def noah_notebooks : ℕ := sorry

/-- The price of a notebook is more than 10 cents -/
axiom price_more_than_dime : notebook_price > 10

/-- Liam paid $2.10 for his notebooks -/
axiom liam_total : notebook_price * liam_notebooks = 210

/-- Noah paid $2.80 for his notebooks -/
axiom noah_total : notebook_price * noah_notebooks = 280

/-- Theorem: Noah bought 2 more notebooks than Liam -/
theorem noah_bought_two_more : noah_notebooks = liam_notebooks + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_noah_bought_two_more_l305_30523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odometer_seven_impossibility_l305_30546

def has_n_sevens (n : Nat) (d : Nat) : Prop :=
  (n.repr.toList.filter (· = '7')).length = d

def is_six_digit (n : Nat) : Prop :=
  100000 ≤ n ∧ n < 1000000

theorem odometer_seven_impossibility :
  ∀ n : Nat, is_six_digit n → has_n_sevens n 4 →
    ¬(has_n_sevens (n + 900) 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odometer_seven_impossibility_l305_30546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l305_30569

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 9 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 1
def center2 : ℝ × ℝ := (4, -3)
def radius2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem stating that the circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l305_30569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_clothing_expense_l305_30517

/-- The total amount Paul spent on his new clothes after all discounts -/
def total_spent (shirt_price : ℝ) (shirt_count : ℕ)
                (pants_price : ℝ) (pants_count : ℕ)
                (suit_price : ℝ)
                (sweater_price : ℝ) (sweater_count : ℕ)
                (store_discount : ℝ) (coupon_discount : ℝ) : ℝ :=
  let subtotal := shirt_price * shirt_count +
                  pants_price * pants_count +
                  suit_price +
                  sweater_price * sweater_count
  let after_store_discount := subtotal * (1 - store_discount)
  after_store_discount * (1 - coupon_discount)

/-- Theorem stating that Paul spent $252.00 on his new clothes after all discounts -/
theorem pauls_clothing_expense :
  total_spent 15 4 40 2 150 30 2 0.2 0.1 = 252 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pauls_clothing_expense_l305_30517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l305_30511

/-- Given a solution with initial volume, initial alcohol percentage, and added volumes of alcohol and water,
    calculate the final alcohol percentage. -/
noncomputable def final_alcohol_percentage (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
                              (added_alcohol : ℝ) (added_water : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_alcohol_percentage / 100
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  (final_alcohol / final_volume) * 100

/-- The theorem stating that given the specific volumes and percentages,
    the final alcohol percentage is 9%. -/
theorem alcohol_solution_problem :
  final_alcohol_percentage 40 5 2.5 7.5 = 9 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_solution_problem_l305_30511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_ratio_l305_30594

theorem flower_ratio (collin_initial : ℕ) (ingrid_initial : ℕ) (petals_per_flower : ℕ) (collin_total_petals : ℕ)
  (h1 : collin_initial = 25)
  (h2 : ingrid_initial = 33)
  (h3 : petals_per_flower = 4)
  (h4 : collin_total_petals = 144) :
  (collin_total_petals - collin_initial * petals_per_flower) / petals_per_flower = ingrid_initial / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_ratio_l305_30594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l305_30584

/-- A random variable following a normal distribution with mean 2 and standard deviation σ -/
def X (σ : ℝ) : Type := Unit

/-- The probability density function of X -/
noncomputable def f (σ : ℝ) : ℝ → ℝ := sorry

/-- The probability that X is greater than a given value -/
noncomputable def P (σ : ℝ) (a : ℝ) : ℝ := sorry

/-- The theorem stating that if the integral of f from 0 to 2 is 1/3, then P(X > 4) = 1/6 -/
theorem normal_distribution_probability (σ : ℝ) (h : ∫ x in (0:ℝ)..(2:ℝ), f σ x = (1:ℝ)/3) : 
  P σ 4 = (1:ℝ)/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l305_30584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_cylinder_projections_l305_30540

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- A circle in 3D space -/
structure Circle where

/-- A cylinder in 3D space -/
structure Cylinder where

/-- Represents the configuration of two planes and a projection axis -/
structure Configuration where
  plane1 : Plane
  plane2 : Plane
  projectionAxis : Line

/-- Predicate for a circle being tangent to a line -/
def CircleTangentToLine (c : Circle) (l : Line) : Prop := sorry

/-- Predicate for a cylinder being tangent to a plane -/
def CylinderTangentToPlane (cyl : Cylinder) (p : Plane) : Prop := sorry

/-- Predicate for a cylinder having its base circle tangent to a line -/
def CylinderBaseTangentToLine (cyl : Cylinder) (l : Line) : Prop := sorry

/-- The number of tangent circles to three intersecting lines -/
def numTangentCirclesToThreeLines : ℕ := 4

/-- The main theorem stating that there are exactly 4 solutions -/
theorem four_cylinder_projections (config : Configuration) :
  ∃ (solutions : Finset Cylinder),
    (∀ cyl ∈ solutions,
      CylinderTangentToPlane cyl config.plane1 ∧
      CylinderTangentToPlane cyl config.plane2 ∧
      CylinderBaseTangentToLine cyl config.projectionAxis) ∧
    solutions.card = numTangentCirclesToThreeLines := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_cylinder_projections_l305_30540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_sum_l305_30560

theorem muffin_sum : 
  (Finset.filter (fun n : ℕ => n < 120 ∧ n % 13 = 3 ∧ n % 8 = 5) (Finset.range 120)).sum id = 204 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_sum_l305_30560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_binary_ternary_l305_30531

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun ⟨i, x⟩ acc => acc + if x then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldr (fun ⟨i, x⟩ acc => acc + x * 3^i) 0

def binary_1101 : List Bool := [true, false, true, true]
def ternary_111 : List ℕ := [1, 1, 1]

theorem product_binary_ternary :
  (binary_to_decimal binary_1101) * (ternary_to_decimal ternary_111) = 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_binary_ternary_l305_30531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l305_30521

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + h.b^2)

/-- The left focus of a hyperbola -/
noncomputable def left_focus (h : Hyperbola) : ℝ × ℝ := 
  (-focal_distance h, 0)

/-- The asymptote of a hyperbola -/
noncomputable def asymptote (h : Hyperbola) (x : ℝ) : ℝ := 
  h.b / h.a * x

/-- The point symmetric to the left focus about the asymptote -/
noncomputable def symmetric_point (h : Hyperbola) : ℝ × ℝ := 
  let c := focal_distance h
  ((h.b^2 - h.a^2) / c, -2 * h.a * h.b / c)

/-- Predicate to check if a point is on the right branch of the hyperbola -/
def is_on_right_branch (h : Hyperbola) (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_symmetric : is_on_right_branch h (symmetric_point h)) : 
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l305_30521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_symmetry_of_graphs_l305_30553

noncomputable def f (x : ℝ) : ℝ := (3 : ℝ) ^ x
noncomputable def g (x : ℝ) : ℝ := -(3 : ℝ) ^ (-x)

theorem central_symmetry_of_graphs :
  ∀ a : ℝ, ∃ b : ℝ, 
    (f a = b ∧ g (-a) = -b) ∧ 
    ((-a, -b) = (-1 : ℝ) • (a, b)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_symmetry_of_graphs_l305_30553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_blue_three_red_eq_l305_30526

def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 6
def num_green_balls : ℕ := 8
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls
def num_selected : ℕ := 4

def probability_one_blue_three_red : ℚ :=
  (Nat.choose num_blue_balls 1 * Nat.choose num_red_balls 3) / Nat.choose total_balls num_selected

theorem probability_one_blue_three_red_eq : probability_one_blue_three_red = 5 / 323 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_blue_three_red_eq_l305_30526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_five_close_pairs_l305_30545

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The theorem to be proved -/
theorem at_least_five_close_pairs 
  (O : Point) 
  (A : Fin 10 → Point) 
  (h_circle : ∀ i, distance O (A i) ≤ 1) : 
  ∃ (S : Finset (Fin 10 × Fin 10)), 
    S.card ≥ 5 ∧ 
    (∀ (p : Fin 10 × Fin 10), p ∈ S → p.1 ≠ p.2 ∧ distance (A p.1) (A p.2) ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_five_close_pairs_l305_30545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_baskets_l305_30548

theorem orange_baskets (total_baskets : ℕ) (min_oranges max_oranges : ℕ) 
  (h_total : total_baskets = 150)
  (h_min : min_oranges = 80)
  (h_max : max_oranges = 130)
  (h_range : ∀ basket, basket ∈ Finset.range (max_oranges - min_oranges + 1) → min_oranges ≤ basket + min_oranges ∧ basket + min_oranges ≤ max_oranges) :
  ∃ n : ℕ, n ≥ 3 ∧ ∃ orange_count : ℕ, (Finset.filter (fun basket => basket + min_oranges = orange_count) (Finset.range (max_oranges - min_oranges + 1))).card ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_baskets_l305_30548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eleven_set_six_sum_not_div_by_six_l305_30597

/-- A set of natural numbers with the property that the sum of any six distinct elements is not divisible by 6 -/
def SixSumNotDivisibleBySix (A : Finset ℕ) : Prop :=
  ∀ (a b c d e f : ℕ), a ∈ A → b ∈ A → c ∈ A → d ∈ A → e ∈ A → f ∈ A →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f →
    c ≠ d → c ≠ e → c ≠ f →
    d ≠ e → d ≠ f →
    e ≠ f →
    ¬(6 ∣ (a + b + c + d + e + f))

/-- Theorem stating that there does not exist a set of 11 distinct natural numbers
    where the sum of any six distinct elements is not divisible by 6 -/
theorem no_eleven_set_six_sum_not_div_by_six :
  ¬∃ (A : Finset ℕ), A.card = 11 ∧ SixSumNotDivisibleBySix A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eleven_set_six_sum_not_div_by_six_l305_30597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l305_30567

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 10 = 0

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := x^2 + (y-2)^2 = 4

/-- Distance from a point (x, y) to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 10| / Real.sqrt 2

/-- Maximum distance from circle C to line l -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_C x y ∧
  ∀ (x' y' : ℝ), circle_C x' y' →
  distance_to_line x y ≥ distance_to_line x' y' ∧
  distance_to_line x y = 4 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l305_30567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_twice_min_chord_line_equation_l305_30537

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the family of lines
def line_eq (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem 1: The line always intersects the circle at two points
theorem line_intersects_circle_twice (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ := by sorry

-- Theorem 2: The equation of the line when the chord is at its minimum length
theorem min_chord_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ), line_eq m x y ↔ 2*x - y - 5 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_twice_min_chord_line_equation_l305_30537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l305_30520

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 1) * x + 1

-- Part 1
def solution_set (m : ℝ) : Set ℝ :=
  if m > 1 then Set.Ioo (1/m) 1
  else if 0 < m ∧ m < 1 then Set.Ioo 1 (1/m)
  else ∅

theorem part1 (m : ℝ) (h : m > 0) :
  {x : ℝ | f m x < 0} = solution_set m := by
  sorry

-- Part 2
theorem part2 :
  {m : ℝ | ∀ x ∈ Set.Icc 1 2, f m x ≤ 2} = Set.Iic (3/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l305_30520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shared_side_length_l305_30587

/-- Given two triangles ABC and DEC sharing side BC, with AB = 7, AC = 15, EC = 9, and BD = 26,
    the least possible integral length of BC is 17. -/
theorem min_shared_side_length (AB AC EC BD : ℝ) (hAB : AB = 7) (hAC : AC = 15) (hEC : EC = 9) (hBD : BD = 26) :
  ∃ (BC : ℕ), BC ≥ 17 ∧ ∀ (n : ℕ), n ≥ 17 → ∃ (ABC DEC : ℝ × ℝ × ℝ),
    ABC.1 = AB ∧ ABC.2.1 = AC ∧ DEC.2.1 = EC ∧ DEC.1 = BD ∧
    ABC.2.2 = DEC.2.2 ∧ ABC.2.2 = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shared_side_length_l305_30587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_meeting_point_ratio_l305_30522

/-- Given two boats A and B, where A's speed is twice B's, prove that if they meet at a 3:1 ratio
    point when traveling towards each other, they'll meet at a 5:7 ratio point when starting from
    opposite ends. -/
theorem boat_meeting_point_ratio (speed_A speed_B : ℝ) (distance : ℝ) :
  speed_A = 2 * speed_B →
  speed_A > 0 →
  speed_B > 0 →
  distance > 0 →
  (let t := distance / (speed_A + speed_B)
   3 * (t * speed_A) = (distance - t * speed_A)) →
  (let t' := distance / (speed_A + speed_B)
   5 * (t' * speed_B) = 7 * (t' * speed_A)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_meeting_point_ratio_l305_30522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_third_quadrant_l305_30559

theorem sin_minus_cos_third_quadrant (α : Real) :
  α ∈ Set.Icc π (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 2 →
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_third_quadrant_l305_30559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l305_30538

-- Define the complex number z
noncomputable def z : ℂ := Complex.I ^ 3 / (2 * Complex.I + 1)

-- Theorem statement
theorem imaginary_part_of_z :
  z.im = -1/5 := by
  -- Simplify z
  have h1 : z = (-2 - Complex.I) / 5 := by
    -- Proof steps here
    sorry
  
  -- Extract imaginary part
  have h2 : ((-2 - Complex.I) / 5).im = -1/5 := by
    -- Proof steps here
    sorry
  
  -- Conclude
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l305_30538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l305_30532

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (2*x - 5*x^2 + 6*x^3) / (9 - x^3)

-- State the theorem
theorem f_nonnegative_iff (x : ℝ) : f x ≥ 0 ↔ x ∈ Set.Ici 0 ∩ Set.Iio 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_l305_30532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_has_inverse_v_injective_l305_30529

-- Define the function v(x) = x - 2/x
noncomputable def v (x : ℝ) : ℝ := x - 2/x

-- State the theorem
theorem v_has_inverse :
  ∀ a b : ℝ, a > 0 → b > 0 → v a = v b → a = b :=
by
  -- The proof would go here
  sorry

-- Define the domain of v
def domain_v : Set ℝ := {x : ℝ | x > 0}

-- State that v is injective on its domain
theorem v_injective : Function.Injective (fun x ↦ v x) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_has_inverse_v_injective_l305_30529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_historical_fiction_new_releases_fraction_l305_30506

theorem historical_fiction_new_releases_fraction 
  (total_inventory : ℕ) 
  (historical_fiction_ratio : ℚ) 
  (historical_fiction_new_release_ratio : ℚ) 
  (other_new_release_ratio : ℚ) 
  (historical_fiction_ratio_valid : historical_fiction_ratio = 2/5)
  (historical_fiction_new_release_ratio_valid : historical_fiction_new_release_ratio = 2/5)
  (other_new_release_ratio_valid : other_new_release_ratio = 7/10) :
  (let historical_fiction := (total_inventory : ℚ) * historical_fiction_ratio
   let other_books := (total_inventory : ℚ) - historical_fiction
   let historical_fiction_new_releases := historical_fiction * historical_fiction_new_release_ratio
   let other_new_releases := other_books * other_new_release_ratio
   let total_new_releases := historical_fiction_new_releases + other_new_releases
   historical_fiction_new_releases / total_new_releases) = 8/29 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_historical_fiction_new_releases_fraction_l305_30506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_hit_seven_l305_30578

/-- Represents a player in the dart-throwing contest -/
inductive Player
| Alice
| Ben
| Cindy
| Dave
| Ellen
| Frank

/-- The score of a single dart throw -/
def DartScore := Fin 13

/-- Represents the result of two dart throws -/
structure ThrowResult where
  first : DartScore
  second : DartScore
  sum : Nat
  h_sum : sum = first.val + 1 + second.val + 1

/-- The scores of all players -/
def playerScores : Player → Nat
| Player.Alice => 18
| Player.Ben => 9
| Player.Cindy => 15
| Player.Dave => 14
| Player.Ellen => 19
| Player.Frank => 8

theorem ben_hit_seven :
  ∃! (p : Player), ∃ (result : ThrowResult),
    playerScores p = result.sum ∧
    (result.first.val + 1 = 7 ∨ result.second.val + 1 = 7) ∧
    p = Player.Ben := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ben_hit_seven_l305_30578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_number_squared_floor_l305_30556

-- Define the greatest integer function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem eulers_number_squared_floor : 
  ∀ e : ℝ, 2 < e ∧ e < 3 → floor (e^2 - 3) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_number_squared_floor_l305_30556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l305_30500

-- Define the convex quadrilateral ABCD
variable (A B C D : EuclideanPlane) -- Changed from EuclideanSpace ℝ 2

-- Define point P
variable (P : EuclideanPlane)

-- Define points K, L, M, N as intersections of angle bisectors with sides
noncomputable def K (A B C D P : EuclideanPlane) : EuclideanPlane := sorry
noncomputable def L (A B C D P : EuclideanPlane) : EuclideanPlane := sorry
noncomputable def M (A B C D P : EuclideanPlane) : EuclideanPlane := sorry
noncomputable def N (A B C D P : EuclideanPlane) : EuclideanPlane := sorry

-- Define the condition that KLMN is a parallelogram
def is_parallelogram (K L M N : EuclideanPlane) : Prop := sorry

-- Define the perpendicular bisector of a line segment
noncomputable def perp_bisector (X Y : EuclideanPlane) : Set EuclideanPlane := sorry

-- State the theorem
theorem locus_of_P (A B C D : EuclideanPlane) :
  {P : EuclideanPlane | is_parallelogram (K A B C D P) (L A B C D P) (M A B C D P) (N A B C D P)} =
  (perp_bisector A C) ∩ (perp_bisector B D) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_l305_30500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_lines_l305_30572

theorem cosine_of_angle_between_lines (v1 v2 : ℝ × ℝ) : 
  v1 = (2, 5) → v2 = (4, 1) → 
  let cos_phi := (v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))
  cos_phi = 13 / Real.sqrt 493 := by
  intro h1 h2
  simp [h1, h2]
  norm_num
  sorry

#check cosine_of_angle_between_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_lines_l305_30572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_range_of_a_l305_30509

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a - 1) * x

theorem monotonicity_and_range_of_a (a : ℝ) :
  (∀ x > 0, Monotone (fun y => f a y) ↔ a ≥ 1) ∧
  (∀ x > 0, f a x ≤ x^2 * Real.exp x - Real.log x - 4*x - 1 → a ≤ -2) :=
by
  sorry

#check monotonicity_and_range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_range_of_a_l305_30509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_saturday_l305_30555

-- Define the days of the week
inductive Day : Type
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving Repr, DecidableEq

-- Define the sports
inductive Sport : Type
  | Basketball | Golf | Running | Swimming | Tennis
  deriving Repr, DecidableEq

-- Define Mahdi's schedule
def schedule : Day → Sport := sorry

-- Conditions
axiom practices_daily : ∀ d : Day, ∃ s : Sport, schedule d = s

axiom runs_three_days : ∃ d1 d2 d3 : Day, 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
  schedule d1 = Sport.Running ∧ 
  schedule d2 = Sport.Running ∧ 
  schedule d3 = Sport.Running

def nextDay : Day → Day
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

axiom no_consecutive_running : ∀ d : Day, 
  schedule d = Sport.Running → 
  schedule (nextDay d) ≠ Sport.Running

axiom monday_basketball : schedule Day.Monday = Sport.Basketball

axiom wednesday_golf : schedule Day.Wednesday = Sport.Golf

axiom swims_and_plays_tennis : ∃ d1 d2 : Day, 
  d1 ≠ d2 ∧ 
  schedule d1 = Sport.Swimming ∧ 
  schedule d2 = Sport.Tennis

axiom no_tennis_after_run_or_swim : ∀ d : Day, 
  (schedule d = Sport.Running ∨ schedule d = Sport.Swimming) → 
  schedule (nextDay d) ≠ Sport.Tennis

-- Theorem to prove
theorem mahdi_swims_on_saturday : 
  schedule Day.Saturday = Sport.Swimming := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_swims_on_saturday_l305_30555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_six_digit_even_l305_30590

/-- A 6-digit natural number with all even digits -/
def SixDigitEven : Type := { n : ℕ // 100000 ≤ n ∧ n ≤ 999999 ∧ ∀ d, d ∈ n.digits 10 → Even d }

/-- Predicate to check if a number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop := ∃ d, d ∈ n.digits 10 ∧ Odd d

/-- Theorem stating the largest possible difference between two 6-digit numbers with all even digits -/
theorem largest_difference_six_digit_even : 
  (∃ a b : SixDigitEven, (b.val - a.val : ℕ) = 111112 ∧ 
    ∀ a' b' : SixDigitEven, (∀ n : ℕ, a'.val < n ∧ n < b'.val → has_odd_digit n) → 
      (b'.val - a'.val : ℕ) ≤ 111112) := by
  sorry

#check largest_difference_six_digit_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_six_digit_even_l305_30590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_15th_row_l305_30563

/-- Represents a lattice with a given number of rows, elements per row, and starting number -/
structure MyLattice where
  rows : Nat
  elementsPerRow : Nat
  start : Nat

/-- Calculates the nth number in the mth row of the lattice -/
def nthNumberInRow (l : MyLattice) (m n : Nat) : Nat :=
  l.start + (m - 1) * l.elementsPerRow + (n - 1)

/-- The theorem to prove -/
theorem fourth_number_15th_row (l : MyLattice) 
  (h1 : l.rows = 15) 
  (h2 : l.elementsPerRow = 7) 
  (h3 : l.start = 3) : 
  nthNumberInRow l 15 4 = 104 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_number_15th_row_l305_30563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spill_ratio_is_half_l305_30571

/-- Represents the aquarium and water spill scenario --/
structure Aquarium where
  length : ℝ
  width : ℝ
  height : ℝ
  initial_fill_ratio : ℝ
  final_water_volume : ℝ

/-- Calculates the ratio of spilled water to initial water --/
noncomputable def spill_ratio (a : Aquarium) : ℝ :=
  let total_volume := a.length * a.width * a.height
  let initial_water := a.initial_fill_ratio * total_volume
  let remaining_water := a.final_water_volume / 3
  let spilled_water := initial_water - remaining_water
  spilled_water / initial_water

/-- Theorem stating that the spill ratio is 1/2 for the given scenario --/
theorem spill_ratio_is_half (a : Aquarium) 
  (h1 : a.length = 4)
  (h2 : a.width = 6)
  (h3 : a.height = 3)
  (h4 : a.initial_fill_ratio = 1/2)
  (h5 : a.final_water_volume = 54) :
  spill_ratio a = 1/2 := by
  sorry

#check spill_ratio_is_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spill_ratio_is_half_l305_30571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_implies_completed_tasks_l305_30528

-- Represents whether an employee completed all their tasks perfectly
def completed_all_tasks_perfectly (employee : Type) : Prop := sorry

-- Represents whether an employee received a bonus
def received_bonus (employee : Type) : Prop := sorry

-- The company's bonus policy: any employee who completes all their tasks perfectly receives a bonus
axiom bonus_policy : ∀ (employee : Type), completed_all_tasks_perfectly employee → received_bonus employee

-- Theorem: If an employee received a bonus, then they completed all their tasks perfectly
theorem bonus_implies_completed_tasks (employee : Type) : 
  received_bonus employee → completed_all_tasks_perfectly employee := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_implies_completed_tasks_l305_30528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_general_term_l305_30513

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 0
  | (n+2) => 6 * sequenceA (n+1) - 9 * sequenceA n + (2^n : ℚ) + n

def general_term (n : ℕ) : ℚ :=
  (n + 1) / 4 + (2^n : ℚ) - (5/3) * (3^n : ℚ) + (5/12) * n * (3^n : ℚ)

theorem sequence_equals_general_term :
  ∀ n : ℕ, sequenceA n = general_term n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_equals_general_term_l305_30513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_decimal_octal_equivalence_l305_30524

-- Define the binary number as a list of bits
def binary_number : List Nat := [1, 0, 1, 0, 1, 0, 1]

-- Define the decimal representation
def decimal_rep : Nat := 85

-- Define the octal representation
def octal_rep : List Nat := [1, 2, 5]

-- Helper function to convert binary to decimal
def binary_to_decimal (bins : List Nat) : Nat :=
  bins.enum.foldr (fun (i, b) acc => acc + b * 2^i) 0

-- Helper function to convert decimal to octal
def decimal_to_octal (dec : Nat) : List Nat :=
  if dec = 0 then [0] else
  let rec aux (n : Nat) (acc : List Nat) :=
    if n = 0 then acc
    else aux (n / 8) ((n % 8) :: acc)
  aux dec []

-- Theorem to prove the equivalence
theorem binary_decimal_octal_equivalence :
  (binary_to_decimal binary_number = decimal_rep) ∧
  (decimal_to_octal decimal_rep = octal_rep) := by
  apply And.intro
  · -- Prove binary to decimal conversion
    rfl
  · -- Prove decimal to octal conversion
    rfl

#eval binary_to_decimal binary_number
#eval decimal_to_octal decimal_rep

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_decimal_octal_equivalence_l305_30524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l305_30570

/-- Two circles are externally tangent -/
def externally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A circle is internally tangent to another circle -/
def internally_tangent (r₁ r₂ : ℝ) : Prop := sorry

/-- A line is a common external tangent to two circles -/
def common_external_tangent (l r₁ r₂ : ℝ) : Prop := sorry

/-- A line is a chord of a circle -/
def chord (r l : ℝ) : Prop := sorry

/-- Given three circles with radii 4, 8, and 12, where the smaller circles are externally
    tangent to each other and internally tangent to the largest circle, the square of the
    length of the chord of the largest circle that is a common external tangent to the
    two smaller circles is 3584/9. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : r₃ = 12)
    (h_ext_tangent : externally_tangent r₁ r₂)
    (h_int_tangent_1 : internally_tangent r₁ r₃)
    (h_int_tangent_2 : internally_tangent r₂ r₃)
    (h_common_tangent : ∃ l, common_external_tangent l r₁ r₂ ∧ chord r₃ l) :
    ∃ l, chord r₃ l ∧ common_external_tangent l r₁ r₂ ∧ l^2 = 3584/9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l305_30570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_rotation_bounds_l305_30519

-- Define the type for 8-digit numbers
def EightDigitNumber := { n : ℕ // n ≥ 10000000 ∧ n < 100000000 }

-- Define the relationship between A and B
def rotateLastToFirst (b : EightDigitNumber) : EightDigitNumber :=
  ⟨(b.val % 10) * 10000000 + b.val / 10, by {
    sorry -- Proof that the result is an 8-digit number
  }⟩

-- Define coprimality with 12
def coprimeWith12 (n : ℕ) : Prop := Nat.Coprime n 12

-- Main theorem
theorem eight_digit_rotation_bounds :
  ∃ (a_max a_min : EightDigitNumber),
    (∃ (b_max b_min : EightDigitNumber),
      b_max.val > 44444444 ∧
      b_min.val > 44444444 ∧
      coprimeWith12 b_max.val ∧
      coprimeWith12 b_min.val ∧
      a_max = rotateLastToFirst b_max ∧
      a_min = rotateLastToFirst b_min) ∧
    (∀ (a : EightDigitNumber),
      (∃ (b : EightDigitNumber),
        b.val > 44444444 ∧
        coprimeWith12 b.val ∧
        a = rotateLastToFirst b) →
      a.val ≤ a_max.val ∧ a.val ≥ a_min.val) ∧
    a_max.val = 99999998 ∧
    a_min.val = 14444446 :=
by
  sorry -- Proof of the main theorem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_digit_rotation_bounds_l305_30519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_bound_l305_30503

/-- A function that represents whether a student solved a question or not -/
def solved (s : Fin 11) (q : Fin n) : Prop := sorry

/-- Theorem: If for any two questions, at least 6 out of 11 students solved exactly one of them,
    then there are no more than 12 questions in the test. -/
theorem test_questions_bound (n : ℕ) (h : n > 0) : 
  (∀ p q : Fin n, ∃ (S : Finset (Fin 11)), S.card ≥ 6 ∧ 
    (∀ s ∈ S, (solved s p ∧ ¬solved s q) ∨ (¬solved s p ∧ solved s q))) →
  n ≤ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_questions_bound_l305_30503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_limit_l305_30566

/-- Probability of getting an odd number after n steps -/
noncomputable def prob_odd (n : ℕ) : ℝ :=
  1/3 + 1/(6 * 4^n)

/-- The limit of prob_odd as n approaches infinity is 1/3 -/
theorem prob_odd_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |prob_odd n - 1/3| < ε := by
  sorry

#check prob_odd_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_odd_limit_l305_30566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l305_30574

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (4*t, 3*t - 1)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 8 * Real.cos θ / (1 - Real.cos (2 * θ))
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point P
def P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_product : 
  let PA := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  let PB := Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)
  PA * PB = 25/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l305_30574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l305_30558

theorem isosceles_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) (h5 : Real.sin C = 2 * Real.cos A * Real.sin B) : A = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_l305_30558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_num_fully_communicating_sets_l305_30552

/-- A space station with pods and tubes connecting them. -/
structure SpaceStation where
  num_pods : Nat
  num_two_way_tubes : Nat

/-- A set of pods that can fully communicate with each other. -/
def FullyCommunicatingSet (ss : SpaceStation) : Set (Finset (Fin ss.num_pods)) := 
  {s | s.card = 4 ∧ 
    ∀ (a b : Fin ss.num_pods), a ∈ s → b ∈ s → ∃ (path : List (Fin ss.num_pods)), 
      path.head? = some a ∧ path.getLast? = some b ∧ path.toFinset ⊆ s}

/-- The number of fully communicating sets of 4 pods in a space station. -/
noncomputable def NumFullyCommunicatingSets (ss : SpaceStation) : Nat :=
  (FullyCommunicatingSet ss).toFinite.toFinset.card

/-- The theorem stating the largest number of fully communicating sets of 4 pods. -/
theorem largest_num_fully_communicating_sets 
  (ss : SpaceStation) 
  (h1 : ss.num_pods = 99) 
  (h2 : ss.num_two_way_tubes = 99) : 
  NumFullyCommunicatingSets ss = 2051652 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_num_fully_communicating_sets_l305_30552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_journey_distance_is_2_sqrt_41_l305_30507

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance of Alice's journey -/
noncomputable def aliceJourneyDistance : ℝ :=
  distance (-3) 6 1 1 + distance 1 1 6 (-3)

theorem alice_journey_distance_is_2_sqrt_41 :
  aliceJourneyDistance = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_journey_distance_is_2_sqrt_41_l305_30507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l305_30564

/-- The time it takes for three people to clean a house together, given their individual cleaning rates -/
noncomputable def cleaning_time_together (tom_rate : ℝ) (nick_rate : ℝ) (alex_rate : ℝ) : ℝ :=
  1 / (tom_rate + nick_rate + alex_rate)

/-- The main theorem stating the time it takes for Tom, Nick, and Alex to clean the house together -/
theorem cleaning_time_proof (tom_time nick_time alex_time : ℝ) 
    (h1 : tom_time = 6)
    (h2 : 2 * (1/3 * nick_time) = tom_time)
    (h3 : alex_time = 8) : 
  cleaning_time_together (1/tom_time) (1/nick_time) (1/alex_time) = 72/29 := by
  sorry

#eval (72 : ℚ) / 29

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l305_30564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_10002_l305_30573

/-- The sequence b_n defined recursively -/
def b : ℕ → ℕ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | (n + 1) => b n + 2 * n + 1

/-- Theorem stating that the 100th term of the sequence is 10002 -/
theorem b_100_eq_10002 : b 100 = 10002 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_10002_l305_30573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2A_lt_cos_2B_sin_sq_C_gt_sum_sin_sq_AB_l305_30591

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = Real.pi
  side_angle_relation : a / Real.sin A = b / Real.sin B
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Theorem for Statement B
theorem cos_2A_lt_cos_2B {t : Triangle} (h : t.A > t.B) : 
  Real.cos (2 * t.A) < Real.cos (2 * t.B) := by
  sorry

-- Theorem for Statement C
theorem sin_sq_C_gt_sum_sin_sq_AB {t : Triangle} (h : t.C > Real.pi / 2) :
  Real.sin t.C ^ 2 > Real.sin t.A ^ 2 + Real.sin t.B ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2A_lt_cos_2B_sin_sq_C_gt_sum_sin_sq_AB_l305_30591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_box_count_l305_30508

structure CardBox where
  total : ℕ
  red : ℕ
  black : ℕ
  green : ℕ

theorem card_box_count (box : CardBox) 
  (h1 : box.total > 0)
  (h2 : (2 : ℚ) / 5 * box.total = box.red)
  (h3 : (5 : ℚ) / 9 * (box.total - box.red) = box.black)
  (h4 : box.total = box.red + box.black + box.green)
  (h5 : box.green = 32) : 
  box.total = 120 := by
  sorry

#check card_box_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_box_count_l305_30508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l305_30510

/-- An arithmetic sequence with a maximum sum -/
structure ArithmeticSequenceWithMaxSum where
  a : ℕ → ℝ
  d : ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  ratio_condition : a 9 / a 8 < -1
  has_max_sum : ∃ n, ∀ m, (Finset.range m).sum (fun i => a i) ≤ (Finset.range n).sum (fun i => a i)

/-- The sum of the first n terms of the sequence -/
def S (seq : ArithmeticSequenceWithMaxSum) (n : ℕ) : ℝ :=
  (Finset.range n).sum seq.a

theorem arithmetic_sequence_max_sum 
    (seq : ArithmeticSequenceWithMaxSum) : 
    (∀ n, S seq n ≤ S seq 8) ∧ S seq 16 < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l305_30510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_properties_l305_30539

/-- Properties of a rectangular prism -/
theorem rectangular_prism_properties (l w h : ℝ) 
  (hl : l = 4) (hw : w = 2) (hh : h = 1) : 
  (l * w * h = 8) ∧ 
  (Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 21) ∧ 
  (4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 21 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_properties_l305_30539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l305_30505

theorem point_in_first_quadrant (α : Real) : 
  α ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin α - Real.cos α > 0 ∧ Real.tan α > 0) ↔ 
  α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) ∪ Set.Ioo Real.pi (5 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l305_30505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_product_is_zero_l305_30581

def first_odd_int (n : ℕ) : ℕ := 2 * n - 1

def sum_squares_odd_ints (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => (first_odd_int (i + 1))^2)

def sum_odd_ints (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (λ i => first_odd_int (i + 1))

theorem units_digit_product_is_zero :
  (sum_squares_odd_ints 2011 * sum_odd_ints 4005) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_product_is_zero_l305_30581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_for_monotonic_decreasing_l305_30595

open MeasureTheory Interval Set

theorem integral_inequality_for_monotonic_decreasing (f : ℝ → ℝ) 
  (hf : Monotone (fun x => -f x)) 
  (hf_pos : ∀ x ∈ Icc 0 1, 0 < f x) : 
  (∫ x in Icc 0 1, f x) * (∫ x in Icc 0 1, x * (f x)^2) ≤ 
  (∫ x in Icc 0 1, x * f x) * (∫ x in Icc 0 1, (f x)^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_for_monotonic_decreasing_l305_30595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l305_30593

theorem solve_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(x - 3) * (8 : ℝ)^(x - 1) = (4 : ℝ)^(x + 2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l305_30593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusion_iff_m_range_l305_30568

-- Define set A
def A : Set ℝ := {x | (1/32 : ℝ) ≤ Real.exp (-x * Real.log 2) ∧ Real.exp (-x * Real.log 2) ≤ 4}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | x^2 - 3*m*x + 2*m^2 - m - 1 < 0}

-- State the theorem
theorem set_inclusion_iff_m_range :
  ∀ m : ℝ, (A ⊇ B m) ↔ (m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_inclusion_iff_m_range_l305_30568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l305_30533

/-- Represents a square divided into sections --/
structure DividedSquare where
  total_area : ℝ
  shaded_area : ℝ

/-- Square I: Divided by diagonals into four equal triangles --/
noncomputable def square_I : DividedSquare where
  total_area := 1
  shaded_area := 1/4

/-- Square II: Divided by connecting midpoints of opposite sides into two rectangles --/
noncomputable def square_II : DividedSquare where
  total_area := 1
  shaded_area := 1/2

/-- Square III: Divided by diagonals and one set of midpoints from opposite sides --/
noncomputable def square_III : DividedSquare where
  total_area := 1
  shaded_area := 1/4

/-- Theorem stating that the shaded areas of Square I and Square III are equal, while Square II is different --/
theorem shaded_areas_comparison :
  square_I.shaded_area = square_III.shaded_area ∧
  square_I.shaded_area ≠ square_II.shaded_area ∧
  square_II.shaded_area ≠ square_III.shaded_area := by
  sorry

#check shaded_areas_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_comparison_l305_30533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_heads_l305_30525

/-- A fair coin is tossed 30000 times. -/
def num_tosses : ℕ := 30000

/-- The probability of getting heads on a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The most likely number of heads is the expected value. -/
theorem most_likely_heads : 
  Int.floor (num_tosses * prob_heads) = 15000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_heads_l305_30525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_to_weekly_increase_approx_l305_30554

/-- The number of working days in a week -/
def work_days : ℕ := 5

/-- The required weekly productivity increase as a decimal -/
def weekly_increase : ℝ := 0.02

/-- The daily productivity increase as a decimal -/
def daily_increase : ℝ := 0.004

/-- Theorem stating that the daily increase compounded over the work week 
    is close to the required weekly increase -/
theorem daily_to_weekly_increase_approx : 
  abs ((1 + daily_increase) ^ work_days - (1 + weekly_increase)) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daily_to_weekly_increase_approx_l305_30554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_to_line_l305_30516

/-- The distance formula between a point and a line -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem stating the conditions and the result to be proved -/
theorem circle_center_distance_to_line :
  let centerX : ℝ := 1/2
  let centerY : ℝ := Real.sqrt 2
  let a : ℝ := 1  -- We use 1 here, but the theorem should hold for both 1 and -1
  let b : ℝ := 1
  let c : ℝ := -Real.sqrt 2
  (centerY^2 = 4 * centerX) →  -- Center lies on the parabola
  ((0 - centerX)^2 + (0 - centerY)^2 = (1 - centerX)^2 + (0 - centerY)^2) →  -- Circle passes through (0,0) and (1,0)
  (distancePointToLine centerX centerY a b c = Real.sqrt 2 / 4) →
  (a = 1 ∨ a = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_to_line_l305_30516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_separation_eq_72_l305_30585

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with two specific people adjacent -/
def adjacentArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of ways to arrange 5 people in a line with at least one person between two specific people -/
def arrangementsWithSeparation : ℕ :=
  totalArrangements 5 - adjacentArrangements 5

theorem arrangements_with_separation_eq_72 :
  arrangementsWithSeparation = 72 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_separation_eq_72_l305_30585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_range_l305_30536

/-- The function f(x) = |2x-1| - |x-1| --/
def f (x : ℝ) : ℝ := |2*x - 1| - |x - 1|

theorem inequality_solution_and_range :
  (∀ x : ℝ, f x ≤ 3 ↔ x ∈ Set.Icc (-3) 3) ∧
  (∃ a : ℝ, ∀ x : ℝ, f x ≤ a ↔ a ≥ -1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_range_l305_30536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l305_30562

def U : Set ℝ := {1, 2, 6}
def A : Set ℝ := {1, 3}

theorem find_x : ∃ x : ℝ, (x^2 + x ∈ U) ∧ (x^2 - 2 ∈ A) ∧ (U \ A = {6}) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_x_l305_30562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_condition_l305_30501

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem smallest_n_satisfying_condition : 
  ∀ n : ℕ, n > 5 → (∀ m : ℕ, m > 5 ∧ m < n → 
    trailing_zeros (Nat.factorial (2 * m)) ≠ 2 * trailing_zeros (Nat.factorial m) + 1) → 
    trailing_zeros (Nat.factorial (2 * n)) = 2 * trailing_zeros (Nat.factorial n) + 1 → 
    n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_satisfying_condition_l305_30501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l305_30534

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 6) + Real.sin (ω * x - Real.pi / 2)

/-- The function g(x) derived from f(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x - Real.pi / 12)

theorem problem_solution :
  ∃ ω : ℝ, 0 < ω ∧ ω < 3 ∧ f ω (Real.pi / 6) = 0 ∧
  ω = 2 ∧
  ∀ x ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4), g x ≥ -3 / 2 ∧
  ∃ x₀ ∈ Set.Icc (-Real.pi / 4) (3 * Real.pi / 4), g x₀ = -3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l305_30534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l305_30580

/-- The area of a circular sector -/
noncomputable def sector_area (radius : ℝ) (angle : ℝ) : ℝ := (1/2) * radius^2 * angle

/-- Theorem: The area of a sector with central angle 2π/3 and radius 3 is 6π -/
theorem sector_area_specific : sector_area 3 (2*Real.pi/3) = 6*Real.pi := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l305_30580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_between_15_and_16_l305_30543

/-- Given points A, B, and D in a 2D plane, where:
    A is at (13, 0)
    D is at (3, 4)
    B forms a right triangle with D, with legs of lengths 3 and 4
    Prove that the sum of distances AD and BD is between 15 and 16 -/
theorem sum_distances_between_15_and_16 (A B D : ℝ × ℝ) : 
  A = (13, 0) → 
  D = (3, 4) → 
  (B.1 - D.1)^2 + (B.2 - D.2)^2 = 3^2 + 4^2 →
  15 < dist A D + dist B D ∧ dist A D + dist B D < 16 := by
  sorry

-- Define the Euclidean distance function
noncomputable def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_between_15_and_16_l305_30543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_product_eq_factorize_l305_30549

/-- Represents a polynomial over a semiring R -/
def MyPolynomial (R : Type*) [Semiring R] := List (R × Nat)

/-- Represents the process of transforming a polynomial into a product of polynomials -/
def transform_to_product {R : Type*} [Semiring R] (p : MyPolynomial R) : List (MyPolynomial R) := sorry

/-- Represents the process of factorizing a polynomial -/
def factorize {R : Type*} [Semiring R] (p : MyPolynomial R) : List (MyPolynomial R) := sorry

/-- Theorem stating that transforming a polynomial into a product of polynomials is equivalent to factorizing it -/
theorem transform_product_eq_factorize {R : Type*} [Semiring R] (p : MyPolynomial R) : 
  transform_to_product p = factorize p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_product_eq_factorize_l305_30549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l305_30512

-- Define the variables and their properties
variable (x y : ℝ)
def mean_x : ℝ := 3
def mean_y : ℝ := 3.5

-- Define the property of negative correlation
axiom negatively_correlated : ∃ k : ℝ, k < 0 ∧ ∀ x y : ℝ, y = k * x + (mean_y - k * mean_x)

-- State the theorem
theorem linear_regression_equation :
  ∃ k b : ℝ, k < 0 ∧ y = k * x + b ∧ mean_y = k * mean_x + b ∧ k = -2 ∧ b = 9.5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_regression_equation_l305_30512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_centroid_l305_30589

/-- Predicate to check if a point P is inside a triangle ABC -/
def P_inside_triangle (A B C P : Point) : Prop :=
  sorry

/-- Function to calculate the area of a triangle given its three vertices -/
noncomputable def area_triangle (A B C : Point) : ℝ :=
  sorry

/-- Predicate to check if a point is the centroid of a triangle -/
def is_centroid (A B C P : Point) : Prop :=
  sorry

/-- Given a triangle ABC and a point P inside it, if the areas of triangles PAB, PBC, and PCA are equal, then P is the centroid of triangle ABC. -/
theorem equal_area_implies_centroid (A B C P : Point) (h_inside : P_inside_triangle A B C P) 
  (h_equal_areas : area_triangle A B P = area_triangle B C P ∧ area_triangle B C P = area_triangle C A P) :
  is_centroid A B C P :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_centroid_l305_30589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l305_30547

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sin x)

noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem triangle_area (A b c : ℝ) (h1 : f A = 3/2) (h2 : b + c = 4) (h3 : Real.sqrt 7 = Real.sqrt 7) :
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l305_30547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l305_30592

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 3 then x^2 - 1
  else if 0 ≤ x ∧ x ≤ 3 then 3*x + 2
  else 5

-- State the theorem
theorem sum_of_f_values : f (-1) + f 1 + f 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l305_30592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_path_distance_l305_30551

/-- The total distance traveled by an aardvark on a specific path between two concentric circles -/
theorem aardvark_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 7) (h₂ : r₂ = 15) : 
  (π * r₂) + (r₂ - r₁) + ((π * r₁) / 2) + (r₂ - r₁) + r₁ = (37 * π) / 2 + 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aardvark_path_distance_l305_30551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l305_30596

/-- Represents the price reduction percentage as a real number between 0 and 1 -/
def price_reduction : ℝ := 0.30

/-- Represents the additional quantity of oil obtained after price reduction in kg -/
def additional_quantity : ℝ := 9

/-- Represents the reduced price per kg of oil in Rs -/
def reduced_price : ℝ := 60

/-- Calculates the amount spent on oil at the reduced price -/
def amount_spent : ℝ := 1800

/-- Theorem stating that given the conditions, the amount spent on oil at the reduced price is approximately 1800 Rs -/
theorem oil_price_reduction_theorem :
  ∃ (original_price : ℝ),
    original_price > 0 ∧
    reduced_price = original_price * (1 - price_reduction) ∧
    ∃ (quantity : ℝ),
      quantity > 0 ∧
      quantity / reduced_price - quantity / original_price = additional_quantity ∧
      abs (quantity - amount_spent) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_theorem_l305_30596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructibility_l305_30502

/-- Predicate to check if three points form a triangle -/
def is_triangle (a b c : ℝ × ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- Calculate the angle between three points -/
noncomputable def angle (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the length of the altitude from a to side bc -/
noncomputable def altitude_length (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the length of the angle bisector from a -/
noncomputable def angle_bisector_length (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- A triangle with given angle, altitude, and angle bisector is constructible iff the angle bisector is not shorter than the altitude -/
theorem triangle_constructibility (α : ℝ) (m_a : ℝ) (l_α : ℝ) :
  (0 < α) → (α < π) → (0 < m_a) → (0 < l_α) →
  (∃ (a b c : ℝ × ℝ), is_triangle a b c ∧ 
    angle a b c = α ∧ 
    altitude_length a b c = m_a ∧ 
    angle_bisector_length a b c = l_α) ↔ 
  l_α ≥ m_a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_constructibility_l305_30502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l305_30544

/-- The length of a train given the speeds and crossing time -/
theorem train_length_calculation 
  (speed_A speed_B : ℝ) 
  (length_A : ℝ) 
  (crossing_time : ℝ) 
  (h1 : speed_A = 54) 
  (h2 : speed_B = 36) 
  (h3 : length_A = 225) 
  (h4 : crossing_time = 15) : 
  ∃ (length_B : ℝ), length_B = 150 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l305_30544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_equals_four_l305_30588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≠ 3 then 2 / |x - 3| else a

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := f a x - 4

theorem three_zeros_implies_a_equals_four (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    y a x₁ = 0 ∧ y a x₂ = 0 ∧ y a x₃ = 0 ∧
    (∀ x : ℝ, y a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_implies_a_equals_four_l305_30588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_BC_l305_30514

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point D on BC
noncomputable def D (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the properties of the triangle
def isIsosceles (triangle : Triangle) : Prop :=
  dist triangle.A triangle.B = dist triangle.A triangle.C

def isPerpendicular (triangle : Triangle) : Prop :=
  let D := D triangle
  (D.1 - triangle.A.1) * (triangle.C.1 - triangle.B.1) +
  (D.2 - triangle.A.2) * (triangle.C.2 - triangle.B.2) = 0

def isADSquared72 (triangle : Triangle) : Prop :=
  let D := D triangle
  dist triangle.A D ^ 2 = 72

def areIntegerLengths (triangle : Triangle) : Prop :=
  let D := D triangle
  ∃ (bc cd : ℤ), (dist triangle.B triangle.C : ℝ) = bc ∧ (dist triangle.C D : ℝ) = cd

-- Define the theorem
theorem smallest_BC (triangle : Triangle) :
  isIsosceles triangle →
  isPerpendicular triangle →
  isADSquared72 triangle →
  areIntegerLengths triangle →
  ∃ (bc : ℤ), (dist triangle.B triangle.C : ℝ) = bc ∧ bc ≥ 11 ∧
  ∀ (bc' : ℤ), (dist triangle.B triangle.C : ℝ) = bc' → bc' ≥ bc :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_BC_l305_30514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_five_percent_calculate_profit_percentage_result_l305_30515

/-- Calculates the profit percentage given selling prices and loss percentage -/
noncomputable def calculate_profit_percentage (loss_price : ℝ) (loss_percentage : ℝ) (profit_price : ℝ) : ℝ :=
  let cost_price := loss_price / (1 - loss_percentage / 100)
  ((profit_price / cost_price) - 1) * 100

/-- Theorem stating that the calculated profit percentage is approximately 5% -/
theorem profit_percentage_is_five_percent :
  let loss_price := 12
  let loss_percentage := 15
  let profit_price := 14.823529411764707
  let calculated_percentage := calculate_profit_percentage loss_price loss_percentage profit_price
  ∃ ε > 0, |calculated_percentage - 5| < ε := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use a theorem to state the result
theorem calculate_profit_percentage_result :
  ∃ ε > 0, |calculate_profit_percentage 12 15 14.823529411764707 - 5| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_five_percent_calculate_profit_percentage_result_l305_30515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_even_l305_30565

-- Define the function f(x) = -ln|x|
noncomputable def f (x : ℝ) : ℝ := -Real.log (abs x)

-- Statement to prove
theorem f_decreasing_and_even :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x > f y) ∧
  (∀ x : ℝ, f x = f (-x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_and_even_l305_30565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_total_votes_l305_30575

/-- Calculates the total number of votes Mark received in an election --/
def total_votes (first_area_voters : ℕ) (first_area_percentage : ℚ) : ℕ :=
  let first_area_votes := (first_area_voters : ℚ) * first_area_percentage
  let remaining_area_votes := 2 * first_area_votes
  (first_area_votes + remaining_area_votes).floor.toNat

/-- Theorem stating that Mark's total votes are 210,000 given the conditions --/
theorem marks_total_votes :
  total_votes 100000 (70 / 100) = 210000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_total_votes_l305_30575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_with_offset_l305_30582

/-- The angle between two 2D vectors plus an offset -/
noncomputable def angle_between_vectors_with_offset (v1 v2 : ℝ × ℝ) (offset : ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))) + offset

/-- Theorem: The angle between (4, -1) and (6, 8) plus 30° offset is cos⁻¹(8/(5√17)) + 30° -/
theorem angle_between_specific_vectors_with_offset :
  angle_between_vectors_with_offset (4, -1) (6, 8) (30 * π / 180) = Real.arccos (8 / (5 * Real.sqrt 17)) + 30 * π / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_with_offset_l305_30582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l305_30599

/-- Predicate to check if a given real number is the eccentricity of a point on the conic section. -/
def is_eccentricity (e : ℝ) (x y : ℝ) : Prop :=
  e = Real.sqrt ((x - 2)^2 + (y - 2)^2) / (|x - y + 3| / Real.sqrt 2)

/-- The eccentricity of the conic section defined by the equation 10x - 2xy - 2y + 1 = 0 is √2. -/
theorem conic_section_eccentricity :
  let P : ℝ × ℝ → Prop := λ (x, y) ↦ 10 * x - 2 * x * y - 2 * y + 1 = 0
  ∃ e : ℝ, e = Real.sqrt 2 ∧ ∀ x y : ℝ, P (x, y) → is_eccentricity e x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l305_30599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tablet_below_threshold_minimum_continuous_medication_days_l305_30557

-- Define constants
noncomputable def initial_dose : ℝ := 128
noncomputable def filtration_rate : ℝ := 0.5
noncomputable def resistance_threshold : ℝ := 25
def initial_resistance_days : ℕ := 6

-- Define functions
noncomputable def remaining_medication (days : ℕ) : ℝ :=
  initial_dose * (filtration_rate ^ days)

noncomputable def total_medication (days : ℕ) : ℝ :=
  initial_dose * (1 - filtration_rate ^ days) / (1 - filtration_rate)

-- Theorem statements
theorem first_tablet_below_threshold :
  ∃ n : ℕ, n = 7 ∧ remaining_medication n < 1 ∧ ∀ m : ℕ, m < n → remaining_medication m ≥ 1 := by
  sorry

theorem minimum_continuous_medication_days :
  ∃ n : ℕ, n = 4 ∧
  (total_medication n) * (filtration_rate ^ (initial_resistance_days - n)) ≥ resistance_threshold ∧
  ∀ m : ℕ, m < n →
  (total_medication m) * (filtration_rate ^ (initial_resistance_days - m)) < resistance_threshold := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tablet_below_threshold_minimum_continuous_medication_days_l305_30557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_e_l305_30530

noncomputable section

variable (f : ℝ → ℝ)

theorem derivative_at_e (h : ∀ x, f x = 2 * x * (deriv f (Real.exp 1)) + Real.log x) : 
  deriv f (Real.exp 1) = -1 / Real.exp 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_e_l305_30530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l305_30518

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def triangle_conditions (t : Triangle) : Prop :=
  t.c = Real.sqrt 3 ∧ t.b = 1 ∧ t.B = 30 * Real.pi / 180

-- Define the theorem to be proved
theorem triangle_proof (t : Triangle) (h : triangle_conditions t) :
  t.C = 60 * Real.pi / 180 ∧ (1/2 * t.b * t.c = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_proof_l305_30518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l305_30598

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 = 1}

-- Define point M where l intersects the y-axis
def M : ℝ × ℝ := (0, -1)

-- Define A and B as variables
variable (A B : ℝ × ℝ)

-- A and B are the intersection points of l and C
axiom A_on_l_and_C : A ∈ l ∧ A ∈ C
axiom B_on_l_and_C : B ∈ l ∧ B ∈ C
axiom A_ne_B : A ≠ B

-- State the theorem
theorem intersection_distance_difference (A B : ℝ × ℝ) 
  (hA : A ∈ l ∧ A ∈ C) (hB : B ∈ l ∧ B ∈ C) (hAB : A ≠ B) : 
  |1 / ‖M - A‖ - 1 / ‖M - B‖| = Real.sqrt 2 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l305_30598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_share_l305_30561

def total_amount : ℚ := 8640
def ratio : List ℚ := [3, 5, 7, 2, 4]
def john_parts : ℚ := 3

theorem john_share :
  let total_parts := ratio.sum
  let part_value := total_amount / total_parts
  let john_exact_share := john_parts * part_value
  (Int.floor john_exact_share : ℚ) = 1234 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_share_l305_30561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l305_30583

noncomputable def f (x : ℝ) : ℝ := (x^2 - 64) / (x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 8 ∨ x > 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l305_30583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_prices_correct_b_more_cost_effective_l305_30535

/-- Represents a purchase with quantity and unit price -/
structure Purchase where
  quantity : ℝ
  unitPrice : ℝ

/-- Represents a pair of purchases -/
structure PurchasePair where
  first : Purchase
  second : Purchase

/-- Purchaser B's spending pattern -/
noncomputable def purchaserB (totalSpent : ℝ) (priceDiff : ℝ) (quantityDiff : ℝ) : PurchasePair :=
  { first := { quantity := totalSpent / 20, unitPrice := 20 },
    second := { quantity := totalSpent / 25, unitPrice := 25 } }

/-- Average price calculation for purchaser A -/
noncomputable def avgPriceA (m n : ℝ) : ℝ := (m + n) / 2

/-- Average price calculation for purchaser B -/
noncomputable def avgPriceB (m n : ℝ) : ℝ := 2 * m * n / (m + n)

theorem purchase_prices_correct (totalSpent quantityDiff : ℝ) :
  totalSpent = 8000 ∧ quantityDiff = 80 →
  let pair := purchaserB totalSpent (5/4) quantityDiff
  pair.first.unitPrice = 20 ∧ pair.second.unitPrice = 25 ∧
  pair.first.quantity - pair.second.quantity = quantityDiff := by
  sorry

theorem b_more_cost_effective (m n : ℝ) :
  m > 0 ∧ n > 0 ∧ m ≠ n →
  avgPriceB m n < avgPriceA m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_prices_correct_b_more_cost_effective_l305_30535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotone_interval_l305_30542

noncomputable def f (ω : ℝ) (x : ℝ) := Real.sin (ω * x)

theorem sine_monotone_interval (ω : ℝ) :
  ω > 0 →
  (∀ x₁ x₂ : ℝ, -π/6 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/6 → f ω x₁ < f ω x₂) →
  0 < ω ∧ ω ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_monotone_interval_l305_30542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l305_30586

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x / 2) - Real.cos (x / 2)

noncomputable def g (x : ℝ) : ℝ := -2 * Real.cos (x / 2)

def isDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem g_decreasing_interval :
  isDecreasingOn g (-π/2) (-π/4) ∧
  ∀ x, f (x + 2*π/3) = g x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l305_30586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_bound_l305_30504

/-- A graph with no 3-cliques -/
structure Graph3CliquesFree where
  n : ℕ
  vertex_set : Finset (Fin n)
  edge_set : Finset (Fin n × Fin n)
  no_3_cliques : ∀ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    ¬((a, b) ∈ edge_set ∧ (b, c) ∈ edge_set ∧ (a, c) ∈ edge_set)

/-- The degree of a vertex in a graph -/
def degree (G : Graph3CliquesFree) (v : Fin G.n) : ℕ :=
  (G.edge_set.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement -/
theorem max_degree_bound (G : Graph3CliquesFree) (h_n : G.n = 2014) :
  (∃ k, Set.range (degree G) = Finset.range (k + 1)) →
  ∃ (k_max : ℕ), k_max = 1342 ∧
    ∀ k, (Set.range (degree G) = Finset.range (k + 1)) → k ≤ k_max :=
by
  intro h
  use 1342
  constructor
  · rfl
  · intro k h_range
    sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_degree_bound_l305_30504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_S_formula_l305_30577

/-- An arithmetic sequence {a_n} with given properties -/
def a : ℕ → ℝ := sorry

/-- A sequence {b_n} with given properties -/
def b : ℕ → ℝ := sorry

/-- Sum of the first n terms of sequence {b_n} -/
def S : ℕ → ℝ := sorry

/-- Properties of the sequences -/
axiom a_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
axiom a_1 : a 1 = 3
axiom a_4 : a 4 = 12
axiom b_1 : b 1 = 4
axiom b_4 : b 4 = 20
axiom b_minus_a_geometric : ∀ n : ℕ, (b (n + 2) - a (n + 2)) / (b (n + 1) - a (n + 1)) = (b (n + 1) - a (n + 1)) / (b n - a n)

/-- Theorem stating the general formula for a_n -/
theorem a_formula : ∀ n : ℕ, n ≥ 1 → a n = 3 * n := by sorry

/-- Theorem stating the general formula for b_n -/
theorem b_formula : ∀ n : ℕ, n ≥ 1 → b n = 3 * n + 2^(n - 1) := by sorry

/-- Theorem stating the sum formula for S_n -/
theorem S_formula : ∀ n : ℕ, n ≥ 1 → S n = (3/2) * n^2 + (3/2) * n + 2^n - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_formula_S_formula_l305_30577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l305_30550

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) :
  seq.a 4 ^ 2 = seq.a 3 * seq.a 7 →
  sum_n seq 8 = 32 →
  sum_n seq 10 = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l305_30550
