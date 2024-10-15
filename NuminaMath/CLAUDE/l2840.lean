import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_inequality_l2840_284005

theorem polynomial_inequality (x : ℝ) : (x + 2) * (x - 8) * (x - 3) > 0 ↔ x ∈ Set.Ioo (-2 : ℝ) 3 ∪ Set.Ioi 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l2840_284005


namespace NUMINAMATH_CALUDE_fraction_simplification_l2840_284091

theorem fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 13 + (1 : ℚ) / 13) = 65 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2840_284091


namespace NUMINAMATH_CALUDE_volume_ADBFE_l2840_284027

-- Define the pyramid ABCD
def Pyramid (A B C D : Point) : Set Point := sorry

-- Define the volume of a set of points
def volume : Set Point → ℝ := sorry

-- Define a median of a triangle
def isMedian (E : Point) (triangle : Set Point) : Prop := sorry

-- Define a midpoint of a line segment
def isMidpoint (F : Point) (segment : Set Point) : Prop := sorry

theorem volume_ADBFE (A B C D E F : Point) 
  (hpyramid : Pyramid A B C D)
  (hmedian : isMedian E {A, B, C})
  (hmidpoint : isMidpoint F {D, C})
  (hvolume : volume (Pyramid A B C D) = 40) :
  volume {A, D, B, F, E} = (3/4) * volume (Pyramid A B C D) := by
  sorry

end NUMINAMATH_CALUDE_volume_ADBFE_l2840_284027


namespace NUMINAMATH_CALUDE_cosine_sum_upper_bound_l2840_284090

theorem cosine_sum_upper_bound (α β γ : Real) 
  (h : Real.sin α + Real.sin β + Real.sin γ ≥ 2) : 
  Real.cos α + Real.cos β + Real.cos γ ≤ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_upper_bound_l2840_284090


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l2840_284069

theorem tennis_tournament_matches (total_players : Nat) (seeded_players : Nat) : 
  total_players = 128 → seeded_players = 32 → total_players - 1 = 127 :=
by
  sorry

#check tennis_tournament_matches

end NUMINAMATH_CALUDE_tennis_tournament_matches_l2840_284069


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l2840_284075

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b ∧ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l2840_284075


namespace NUMINAMATH_CALUDE_keith_pears_l2840_284085

theorem keith_pears (jason_pears mike_ate remaining_pears : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : mike_ate = 12)
  (h3 : remaining_pears = 81) :
  ∃ keith_pears : ℕ, jason_pears + keith_pears - mike_ate = remaining_pears ∧ keith_pears = 47 :=
by sorry

end NUMINAMATH_CALUDE_keith_pears_l2840_284085


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_always_positive_l2840_284032

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem quadratic_always_positive : 
  (¬ ∃ x : ℝ, x^2 + x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_always_positive_l2840_284032


namespace NUMINAMATH_CALUDE_system_solution_l2840_284056

theorem system_solution : 
  ∀ x : ℝ, (3 * x^2 = Real.sqrt (36 * x^2) ∧ 3 * x^2 + 21 = 24 * x) ↔ (x = 7 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2840_284056


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l2840_284088

/-- A function f: ℝ → ℝ has no fixed points if for all x: ℝ, f x ≠ x -/
def has_no_fixed_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- The quadratic function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + 2*a*x + 1

/-- Theorem stating that f(x) = x^2 + 2ax + 1 has no fixed points iff a ∈ (-1/2, 3/2) -/
theorem no_fixed_points_iff_a_in_range (a : ℝ) :
  has_no_fixed_points (f a) ↔ -1/2 < a ∧ a < 3/2 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_a_in_range_l2840_284088


namespace NUMINAMATH_CALUDE_range_of_c_l2840_284034

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b = a * b) (h2 : a + b + c = a * b * c) :
  1 < c ∧ c ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_c_l2840_284034


namespace NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2840_284008

/-- Given a cone with slant height 10 cm and curved surface area 157.07963267948966 cm²,
    the radius of the base is 5 cm. -/
theorem cone_radius_from_slant_height_and_surface_area :
  let slant_height : ℝ := 10
  let curved_surface_area : ℝ := 157.07963267948966
  let radius : ℝ := curved_surface_area / (Real.pi * slant_height)
  radius = 5 := by sorry

end NUMINAMATH_CALUDE_cone_radius_from_slant_height_and_surface_area_l2840_284008


namespace NUMINAMATH_CALUDE_exponential_base_theorem_l2840_284086

theorem exponential_base_theorem (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^x ≤ max a a⁻¹) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, min a a⁻¹ ≤ a^x) ∧
  (max a a⁻¹ - min a a⁻¹ = 1) →
  a = (Real.sqrt 5 + 1) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_exponential_base_theorem_l2840_284086


namespace NUMINAMATH_CALUDE_remainder_seven_n_l2840_284033

theorem remainder_seven_n (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 7*n ≡ 1 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_l2840_284033


namespace NUMINAMATH_CALUDE_permutation_reachable_l2840_284074

/-- A transformation step on a tuple of natural numbers -/
def transform (a : Fin 2015 → ℕ) (k l : Fin 2015) (h : Even (a k)) : Fin 2015 → ℕ :=
  fun i => if i = k then a k / 2
           else if i = l then a l + a k / 2
           else a i

/-- The set of all permutations of (1, 2, ..., 2015) -/
def permutations : Set (Fin 2015 → ℕ) :=
  {p | ∃ σ : Equiv.Perm (Fin 2015), ∀ i, p i = σ i + 1}

/-- The initial tuple (1, 2, ..., 2015) -/
def initial : Fin 2015 → ℕ := fun i => i + 1

/-- The set of all tuples reachable from the initial tuple -/
inductive reachable : (Fin 2015 → ℕ) → Prop
  | init : reachable initial
  | step {a b} (k l : Fin 2015) (h : Even (a k)) :
      reachable a → b = transform a k l h → reachable b

theorem permutation_reachable :
  ∀ p ∈ permutations, reachable p :=
sorry

end NUMINAMATH_CALUDE_permutation_reachable_l2840_284074


namespace NUMINAMATH_CALUDE_range_of_a_l2840_284084

-- Define the propositions p and q
def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4
def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, p a x → q x) →
  (a ≤ -2 ∨ a ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2840_284084


namespace NUMINAMATH_CALUDE_cable_lengths_theorem_l2840_284078

/-- Given two pieces of cable with specific mass and length relationships,
    prove that their lengths are either (5, 8) meters or (19.5, 22.5) meters. -/
theorem cable_lengths_theorem (mass1 mass2 : ℝ) (length_diff mass_per_meter_diff : ℝ) :
  mass1 = 65 →
  mass2 = 120 →
  length_diff = 3 →
  mass_per_meter_diff = 2 →
  ∃ (l1 l2 : ℝ),
    ((l1 = 5 ∧ l2 = 8) ∨ (l1 = 19.5 ∧ l2 = 22.5)) ∧
    (mass1 / l1 + mass_per_meter_diff) * (l1 + length_diff) = mass2 :=
by sorry

end NUMINAMATH_CALUDE_cable_lengths_theorem_l2840_284078


namespace NUMINAMATH_CALUDE_radical_simplification_l2840_284077

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (98 * x) = 210 * x * Real.sqrt (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2840_284077


namespace NUMINAMATH_CALUDE_tangent_product_l2840_284089

theorem tangent_product (A B : ℝ) (h1 : A + B = 5 * Real.pi / 4) 
  (h2 : ∀ k : ℤ, A + B ≠ k * Real.pi + Real.pi / 2) : 
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l2840_284089


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_cube_sum_l2840_284072

/-- Checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 3

/-- Checks if three natural numbers are distinct and nonzero -/
def areDistinctNonzero (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- The main theorem statement -/
theorem smallest_n_for_perfect_cube_sum : 
  (∃ a b c : ℕ, areDistinctNonzero a b c ∧ 
    10 = a + b + c ∧ 
    isPerfectCube ((a + b) * (b + c) * (c + a))) ∧ 
  (∀ n : ℕ, n < 10 → 
    ¬(∃ a b c : ℕ, areDistinctNonzero a b c ∧ 
      n = a + b + c ∧ 
      isPerfectCube ((a + b) * (b + c) * (c + a)))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_cube_sum_l2840_284072


namespace NUMINAMATH_CALUDE_pyramid_sphere_radius_l2840_284099

-- Define the pyramid
structure RegularQuadrilateralPyramid where
  base_side : ℝ
  lateral_edge : ℝ

-- Define the spheres
structure Sphere where
  radius : ℝ

-- Define the problem
def pyramid_problem (p : RegularQuadrilateralPyramid) (q1 q2 : Sphere) : Prop :=
  p.base_side = 12 ∧
  p.lateral_edge = 10 ∧
  -- Q1 is inscribed in the pyramid (this is implied, not explicitly stated in Lean)
  -- Q2 touches Q1 and all lateral faces (this is implied, not explicitly stated in Lean)
  q2.radius = 6 * Real.sqrt 7 / 49

-- Theorem statement
theorem pyramid_sphere_radius 
  (p : RegularQuadrilateralPyramid) 
  (q1 q2 : Sphere) 
  (h : pyramid_problem p q1 q2) : 
  q2.radius = 6 * Real.sqrt 7 / 49 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_sphere_radius_l2840_284099


namespace NUMINAMATH_CALUDE_approximation_place_l2840_284007

/-- A function that returns the number of decimal places in a given number -/
def decimal_places (x : ℚ) : ℕ := sorry

/-- A function that returns the name of the decimal place given its position -/
def place_name (n : ℕ) : String := sorry

theorem approximation_place (x : ℚ) (h : decimal_places x = 2) :
  place_name (decimal_places x) = "hundredths" := by sorry

end NUMINAMATH_CALUDE_approximation_place_l2840_284007


namespace NUMINAMATH_CALUDE_savings_calculation_l2840_284030

theorem savings_calculation (initial_savings : ℝ) : 
  let february_spend := 0.20 * initial_savings
  let march_spend := 0.40 * initial_savings
  let april_spend := 1500
  let remaining := 2900
  february_spend + march_spend + april_spend + remaining = initial_savings →
  initial_savings = 11000 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l2840_284030


namespace NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l2840_284092

/-- A pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- The volume of the pyramid -/
  volume : ℝ
  /-- The radius of the inscribed sphere -/
  radius : ℝ
  /-- The total surface area of the pyramid -/
  surface_area : ℝ
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The surface area is positive -/
  surface_area_pos : surface_area > 0

/-- 
Theorem: The volume of a pyramid with an inscribed sphere is equal to 
one-third of the product of the radius of the sphere and the total surface area of the pyramid.
-/
theorem inscribed_sphere_pyramid_volume 
  (p : InscribedSpherePyramid) : p.volume = (1 / 3) * p.surface_area * p.radius := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l2840_284092


namespace NUMINAMATH_CALUDE_third_angle_measure_l2840_284028

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧
  t.angle1 + t.angle2 + t.angle3 = 180

-- Theorem statement
theorem third_angle_measure (t : Triangle) :
  is_valid_triangle t → t.angle1 = 25 → t.angle2 = 70 → t.angle3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_measure_l2840_284028


namespace NUMINAMATH_CALUDE_xyz_bound_l2840_284096

theorem xyz_bound (x y z : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x ≤ 2) (hy_bound : y ≤ 3) (hsum : x + y + z = 11) :
  x * y * z ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_xyz_bound_l2840_284096


namespace NUMINAMATH_CALUDE_grid_difference_theorem_l2840_284044

def Grid := Fin 8 → Fin 8 → Fin 64

def adjacent (x₁ y₁ x₂ y₂ : Fin 8) : Prop :=
  (x₁ = x₂ ∧ (y₁.val + 1 = y₂.val ∨ y₂.val + 1 = y₁.val)) ∨
  (y₁ = y₂ ∧ (x₁.val + 1 = x₂.val ∨ x₂.val + 1 = x₁.val))

theorem grid_difference_theorem (g : Grid) (h : Function.Injective g) :
    ∃ (x₁ y₁ x₂ y₂ : Fin 8), adjacent x₁ y₁ x₂ y₂ ∧ 
    (g x₁ y₁).val.succ.succ.succ.succ ≤ (g x₂ y₂).val ∨ 
    (g x₂ y₂).val.succ.succ.succ.succ ≤ (g x₁ y₁).val := by
  sorry

end NUMINAMATH_CALUDE_grid_difference_theorem_l2840_284044


namespace NUMINAMATH_CALUDE_solutions_of_x_fourth_minus_16_l2840_284024

theorem solutions_of_x_fourth_minus_16 :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end NUMINAMATH_CALUDE_solutions_of_x_fourth_minus_16_l2840_284024


namespace NUMINAMATH_CALUDE_clarence_oranges_l2840_284052

-- Define the initial number of oranges
def initial_oranges : ℝ := 5.0

-- Define the number of oranges given away
def oranges_given : ℝ := 3.0

-- Define the number of Skittles bought (not used in the calculation, but mentioned in the problem)
def skittles_bought : ℝ := 9.0

-- Define the function to calculate the remaining oranges
def remaining_oranges : ℝ := initial_oranges - oranges_given

-- Theorem to prove
theorem clarence_oranges : remaining_oranges = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_clarence_oranges_l2840_284052


namespace NUMINAMATH_CALUDE_quadratic_at_most_one_solution_l2840_284040

theorem quadratic_at_most_one_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 3 * x + 1 = 0) ∨ (∀ x y : ℝ, a * x^2 + 3 * x + 1 = 0 → a * y^2 + 3 * y + 1 = 0 → x = y) ↔
  a = 0 ∨ a ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_at_most_one_solution_l2840_284040


namespace NUMINAMATH_CALUDE_subset_union_subset_l2840_284093

theorem subset_union_subset (U M N : Set α) : M ⊆ U → N ⊆ U → (M ∪ N) ⊆ U := by sorry

end NUMINAMATH_CALUDE_subset_union_subset_l2840_284093


namespace NUMINAMATH_CALUDE_existence_of_n_l2840_284016

theorem existence_of_n : ∃ n : ℕ, n > 0 ∧ (1.001 : ℝ)^n > 10 ∧ (0.999 : ℝ)^n < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l2840_284016


namespace NUMINAMATH_CALUDE_mary_cut_ten_roses_l2840_284038

/-- The number of roses Mary cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Proof that Mary cut 10 roses from her flower garden -/
theorem mary_cut_ten_roses (initial_roses final_roses : ℕ) 
  (h1 : initial_roses = 6) 
  (h2 : final_roses = 16) : 
  roses_cut initial_roses final_roses = 10 := by
  sorry

#check mary_cut_ten_roses

end NUMINAMATH_CALUDE_mary_cut_ten_roses_l2840_284038


namespace NUMINAMATH_CALUDE_solve_for_s_l2840_284087

theorem solve_for_s (t : ℚ) (h1 : 7 * ((t / 2) + 3) + 6 * t = 156) : (t / 2) + 3 = 192 / 19 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_s_l2840_284087


namespace NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2840_284036

theorem sqrt_27_div_sqrt_3_eq_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_sqrt_3_eq_3_l2840_284036


namespace NUMINAMATH_CALUDE_antonia_pills_left_l2840_284037

/-- Calculates the number of pills left after taking supplements for two weeks -/
def pills_left (bottles_120 : Nat) (bottles_30 : Nat) (supplements : Nat) (weeks : Nat) : Nat :=
  let total_pills := bottles_120 * 120 + bottles_30 * 30
  let days := weeks * 7
  let pills_used := days * supplements
  total_pills - pills_used

/-- Theorem stating that given the specific conditions, the number of pills left is 350 -/
theorem antonia_pills_left :
  pills_left 3 2 5 2 = 350 := by
  sorry

end NUMINAMATH_CALUDE_antonia_pills_left_l2840_284037


namespace NUMINAMATH_CALUDE_total_assembly_time_l2840_284062

def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

def chair_time : ℕ := 4
def table_time : ℕ := 8
def bookshelf_time : ℕ := 12
def lamp_time : ℕ := 2

theorem total_assembly_time :
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time + lamps * lamp_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_assembly_time_l2840_284062


namespace NUMINAMATH_CALUDE_circles_tangent_to_ellipse_l2840_284012

theorem circles_tangent_to_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x-r)^2 + y^2 = r^2) ∧ 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x+r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_circles_tangent_to_ellipse_l2840_284012


namespace NUMINAMATH_CALUDE_three_tangents_implies_a_greater_than_three_l2840_284067

/-- A curve of the form y = x³ + ax² + bx -/
structure Curve where
  a : ℝ
  b : ℝ

/-- The number of tangent lines to the curve that pass through (0,-1) -/
noncomputable def numTangentLines (c : Curve) : ℕ := sorry

/-- Theorem stating that if there are exactly three tangent lines passing through (0,-1), then a > 3 -/
theorem three_tangents_implies_a_greater_than_three (c : Curve) :
  numTangentLines c = 3 → c.a > 3 := by sorry

end NUMINAMATH_CALUDE_three_tangents_implies_a_greater_than_three_l2840_284067


namespace NUMINAMATH_CALUDE_range_of_m_l2840_284045

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x y : ℝ, x^2/2 + y^2/(m-1) = 1 → (m - 1 > 2)

-- Define proposition q
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + 4*m ≠ 0

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (¬p m) ∧ (p m ∨ q m) → m ∈ Set.Ioo (1/4 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2840_284045


namespace NUMINAMATH_CALUDE_animal_shelter_cats_l2840_284080

theorem animal_shelter_cats (total : ℕ) (cats dogs : ℕ) : 
  total = 60 →
  cats = dogs + 20 →
  cats + dogs = total →
  cats = 40 := by
sorry

end NUMINAMATH_CALUDE_animal_shelter_cats_l2840_284080


namespace NUMINAMATH_CALUDE_electronic_shop_purchase_cost_l2840_284021

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars -/
def pc_price_difference : ℕ := 500

/-- The price of a personal computer in dollars -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars -/
def tablet_price : ℕ := smartphone_price + pc_price

/-- The total cost of buying one of each product in dollars -/
def total_cost : ℕ := smartphone_price + pc_price + tablet_price

theorem electronic_shop_purchase_cost : total_cost = 2200 := by
  sorry

end NUMINAMATH_CALUDE_electronic_shop_purchase_cost_l2840_284021


namespace NUMINAMATH_CALUDE_original_number_is_seven_l2840_284082

theorem original_number_is_seven : ∃ x : ℝ, 3 * x - 5 = 16 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_seven_l2840_284082


namespace NUMINAMATH_CALUDE_cube_sphere_volume_l2840_284064

theorem cube_sphere_volume (cube_volume : Real) (h : cube_volume = 8) :
  ∃ (sphere_volume : Real), sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_l2840_284064


namespace NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l2840_284057

-- Define the functions
def f (x : ℝ) : ℝ := 6 * x
def g (x : ℝ) : ℝ := x * |x|

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) := by sorry

end NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l2840_284057


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l2840_284083

theorem four_digit_number_problem (a b c d : Nat) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9) →  -- Ensuring each digit is between 0 and 9
  (a ≠ 0) →  -- Ensuring 'a' is not 0 (as it's a four-digit number)
  (1000 * a + 100 * b + 10 * c + d) - (100 * a + 10 * b + c) - (10 * a + b) - a = 1787 →
  (1000 * a + 100 * b + 10 * c + d = 2009 ∨ 1000 * a + 100 * b + 10 * c + d = 2010) :=
by sorry


end NUMINAMATH_CALUDE_four_digit_number_problem_l2840_284083


namespace NUMINAMATH_CALUDE_parabola_points_order_l2840_284001

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 9

-- Define the points on the parabola
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_points_order : y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_order_l2840_284001


namespace NUMINAMATH_CALUDE_physics_value_l2840_284098

def letterValue (n : Nat) : Int :=
  match n % 9 with
  | 0 => 0
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | _ => -1

def wordValue (word : List Nat) : Int :=
  List.sum (List.map letterValue word)

theorem physics_value :
  wordValue [16, 8, 25, 19, 9, 3, 19] = 1 := by
  sorry

end NUMINAMATH_CALUDE_physics_value_l2840_284098


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l2840_284076

theorem quadratic_equation_proof (a : ℝ) :
  (a^2 - 4*a + 5 ≠ 0) ∧
  (∀ x : ℝ, (2^2 - 4*2 + 5)*x^2 + 2*2*x + 4 = 0 ↔ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l2840_284076


namespace NUMINAMATH_CALUDE_circle_roll_position_l2840_284050

theorem circle_roll_position (d : ℝ) (start : ℝ) (h_d : d = 1) (h_start : start = 3) : 
  let circumference := π * d
  let end_position := start - circumference
  end_position = 3 - π := by
sorry

end NUMINAMATH_CALUDE_circle_roll_position_l2840_284050


namespace NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_2x_l2840_284081

open MeasureTheory Interval Real

theorem integral_sqrt_4_minus_x_squared_plus_2x : 
  ∫ x in (-2)..2, (Real.sqrt (4 - x^2) + 2*x) = 2*π := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_4_minus_x_squared_plus_2x_l2840_284081


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2840_284000

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2840_284000


namespace NUMINAMATH_CALUDE_sum_of_cubes_squares_and_product_l2840_284060

theorem sum_of_cubes_squares_and_product : (3 + 7)^3 + (3^2 + 7^2) + 3 * 7 = 1079 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_squares_and_product_l2840_284060


namespace NUMINAMATH_CALUDE_perpendicular_sum_maximized_l2840_284029

theorem perpendicular_sum_maximized (r : ℝ) (α : ℝ) :
  let s := r * (Real.sin α + Real.cos α)
  ∀ β, 0 ≤ β ∧ β ≤ 2 * Real.pi → s ≤ r * (Real.sin (Real.pi / 4) + Real.cos (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_sum_maximized_l2840_284029


namespace NUMINAMATH_CALUDE_square_sum_of_two_integers_l2840_284022

theorem square_sum_of_two_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 72) : 
  x^2 + y^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_two_integers_l2840_284022


namespace NUMINAMATH_CALUDE_chef_used_41_apples_l2840_284018

/-- The number of apples the chef used to make pies -/
def apples_used (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: Given the initial number of apples and the remaining number of apples,
    prove that the number of apples used is 41. -/
theorem chef_used_41_apples (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 43)
  (h2 : remaining = 2) :
  apples_used initial remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_chef_used_41_apples_l2840_284018


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2840_284043

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 6 = 0) →     -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  (((n % 100) / 10) * (n % 10) = 56 ∨ ((n % 100) / 10) * (n % 10) = 54) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2840_284043


namespace NUMINAMATH_CALUDE_vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l2840_284049

def e₁ : Fin 2 → ℝ := ![(-1 : ℝ), 2]
def e₂ : Fin 2 → ℝ := ![(5 : ℝ), -1]

theorem vectors_form_basis (v : Fin 2 → ℝ) : 
  ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i :=
sorry

theorem vectors_not_collinear : 
  e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0 :=
sorry

theorem basis_iff_not_collinear :
  (∀ (v : Fin 2 → ℝ), ∃ (a b : ℝ), v = fun i => a * e₁ i + b * e₂ i) ↔
  (e₁ 0 * e₂ 1 ≠ e₁ 1 * e₂ 0) :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_vectors_not_collinear_basis_iff_not_collinear_l2840_284049


namespace NUMINAMATH_CALUDE_parabola_point_l2840_284070

/-- Given a point P (x₀, y₀) on the parabola y = 3x² with derivative 6 at x₀, prove P = (1, 3) -/
theorem parabola_point (x₀ y₀ : ℝ) : 
  y₀ = 3 * x₀^2 →                   -- Point P lies on the parabola
  (6 : ℝ) = 6 * x₀ →                -- Derivative at x₀ is 6
  (x₀, y₀) = (1, 3) :=              -- Conclusion: P = (1, 3)
by sorry

end NUMINAMATH_CALUDE_parabola_point_l2840_284070


namespace NUMINAMATH_CALUDE_corn_planting_ratio_l2840_284026

/-- Represents the problem of calculating the ratio of dinner cost to total earnings for corn planting kids. -/
theorem corn_planting_ratio :
  -- Define constants based on the problem conditions
  let ears_per_row : ℕ := 70
  let seeds_per_bag : ℕ := 48
  let seeds_per_ear : ℕ := 2
  let pay_per_row : ℚ := 3/2  -- $1.5 expressed as a rational number
  let dinner_cost : ℚ := 36
  let bags_used : ℕ := 140

  -- Calculate total ears planted
  let total_ears : ℕ := (bags_used * seeds_per_bag) / seeds_per_ear

  -- Calculate rows planted
  let rows_planted : ℕ := total_ears / ears_per_row

  -- Calculate total earnings
  let total_earned : ℚ := pay_per_row * rows_planted

  -- The ratio of dinner cost to total earnings is 1/2
  dinner_cost / total_earned = 1/2 :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_corn_planting_ratio_l2840_284026


namespace NUMINAMATH_CALUDE_intersection_M_N_l2840_284014

def M : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2840_284014


namespace NUMINAMATH_CALUDE_rectangle_area_42_implies_y_7_l2840_284094

/-- Rectangle PQRS with vertices P(0, 0), Q(0, 6), R(y, 6), and S(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of a rectangle is the product of its length and width -/
def area (rect : Rectangle) : ℝ := 6 * rect.y

theorem rectangle_area_42_implies_y_7 (rect : Rectangle) (h_area : area rect = 42) : rect.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_42_implies_y_7_l2840_284094


namespace NUMINAMATH_CALUDE_locus_is_circle_l2840_284025

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  s : ℝ  -- side length
  A : ℝ × ℝ  -- coordinates of vertex A
  B : ℝ × ℝ  -- coordinates of vertex B
  C : ℝ × ℝ  -- coordinates of vertex C
  is_equilateral : 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

/-- The locus of points with constant sum of squared distances to triangle vertices -/
def ConstantSumLocus (tri : EquilateralTriangle) (a : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | 
    (P.1 - tri.A.1)^2 + (P.2 - tri.A.2)^2 + 
    (P.1 - tri.B.1)^2 + (P.2 - tri.B.2)^2 + 
    (P.1 - tri.C.1)^2 + (P.2 - tri.C.2)^2 = a}

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

theorem locus_is_circle (tri : EquilateralTriangle) (a : ℝ) (h : a > tri.s^2) :
  ∃ (c : Circle), ConstantSumLocus tri a = {P : ℝ × ℝ | (P.1 - c.center.1)^2 + (P.2 - c.center.2)^2 = c.radius^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l2840_284025


namespace NUMINAMATH_CALUDE_class_2_score_l2840_284010

/-- Calculates the comprehensive score for a class based on weighted scores -/
def comprehensive_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the comprehensive score for the given class is 82.5 -/
theorem class_2_score : comprehensive_score 80 90 84 70 = 82.5 := by
  sorry

end NUMINAMATH_CALUDE_class_2_score_l2840_284010


namespace NUMINAMATH_CALUDE_fence_cost_l2840_284017

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 144 → price_per_foot = 58 → cost = 2784 → 
  cost = 4 * Real.sqrt area * price_per_foot := by
  sorry

#check fence_cost

end NUMINAMATH_CALUDE_fence_cost_l2840_284017


namespace NUMINAMATH_CALUDE_typing_time_proof_l2840_284071

def original_speed : ℕ := 212
def speed_reduction : ℕ := 40
def document_length : ℕ := 3440

theorem typing_time_proof :
  (document_length : ℚ) / (original_speed - speed_reduction) = 20 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l2840_284071


namespace NUMINAMATH_CALUDE_line_mb_product_l2840_284015

/-- A line passing through two points (0, -2) and (2, 4) has mb = -6 --/
theorem line_mb_product (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (-2 : ℝ) = b →                -- Line passes through (0, -2)
  4 = m * 2 + b →               -- Line passes through (2, 4)
  m * b = -6 := by sorry

end NUMINAMATH_CALUDE_line_mb_product_l2840_284015


namespace NUMINAMATH_CALUDE_master_wang_parts_per_day_l2840_284041

/-- The number of parts Master Wang processed per day -/
def parts_per_day (a : ℕ) : ℚ :=
  (a + 3 : ℚ) / 8

/-- Theorem stating that the number of parts processed per day is (a + 3) / 8 -/
theorem master_wang_parts_per_day (a : ℕ) :
  parts_per_day a = (a + 3 : ℚ) / 8 := by
  sorry

#check master_wang_parts_per_day

end NUMINAMATH_CALUDE_master_wang_parts_per_day_l2840_284041


namespace NUMINAMATH_CALUDE_uninterrupted_viewing_time_movie_problem_solution_l2840_284031

/-- Calculates the uninterrupted viewing time at the end of a movie given the total viewing time,
    initial viewing periods, and rewind times. -/
theorem uninterrupted_viewing_time 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 
  total_time - ((first_viewing + second_viewing) + (first_rewind + second_rewind)) :=
by sorry

/-- Proves that the uninterrupted viewing time at the end of the movie is 20 minutes
    given the specific conditions from the problem. -/
theorem movie_problem_solution 
  (total_time : ℕ) 
  (first_viewing : ℕ) 
  (first_rewind : ℕ) 
  (second_viewing : ℕ) 
  (second_rewind : ℕ) 
  (h1 : total_time = 120) 
  (h2 : first_viewing = 35) 
  (h3 : first_rewind = 5) 
  (h4 : second_viewing = 45) 
  (h5 : second_rewind = 15) : 
  total_time - (first_viewing + second_viewing + first_rewind + second_rewind) = 20 :=
by sorry

end NUMINAMATH_CALUDE_uninterrupted_viewing_time_movie_problem_solution_l2840_284031


namespace NUMINAMATH_CALUDE_eleven_twelfths_squared_between_half_and_one_l2840_284058

theorem eleven_twelfths_squared_between_half_and_one :
  (11 / 12 : ℚ)^2 > 1/2 ∧ (11 / 12 : ℚ)^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_eleven_twelfths_squared_between_half_and_one_l2840_284058


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2840_284095

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 7 ≤ 2) ↔ (x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2840_284095


namespace NUMINAMATH_CALUDE_area_difference_l2840_284035

/-- Given a square BDEF with side length 4 + 2√2, surrounded by a rectangle with sides 2s and s
    (where s is the side length of BDEF), and with a regular octagon inscribed in BDEF,
    the total area of the shape composed of the rectangle and square minus the area of
    the inscribed regular octagon is 56 + 24√2 square units. -/
theorem area_difference (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) : 
  2 * s^2 + s^2 - (2 * (1 + Real.sqrt 2) * (2 * Real.sqrt 2)^2) = 56 + 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_l2840_284035


namespace NUMINAMATH_CALUDE_dads_dimes_proof_l2840_284002

/-- The number of dimes Melanie's dad gave her -/
def dads_dimes : ℕ := 83 - (19 + 25)

/-- Melanie's initial number of dimes -/
def initial_dimes : ℕ := 19

/-- Number of dimes Melanie's mother gave her -/
def mothers_dimes : ℕ := 25

/-- Melanie's total number of dimes after receiving from both parents -/
def total_dimes : ℕ := 83

theorem dads_dimes_proof : 
  dads_dimes = total_dimes - (initial_dimes + mothers_dimes) := by
  sorry

end NUMINAMATH_CALUDE_dads_dimes_proof_l2840_284002


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2840_284042

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 = 1 →
  a 8 = a 6 + 2 * a 4 →
  a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2840_284042


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2840_284061

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2840_284061


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_t_eq_zero_l2840_284063

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) (t : ℝ) : ℝ := n^2 + 5*n + t

/-- The nth term of the sequence -/
def a (n : ℕ) (t : ℝ) : ℝ :=
  if n = 1 then S 1 t
  else S n t - S (n-1) t

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n+1) - f n = d

theorem arithmetic_sequence_iff_t_eq_zero (t : ℝ) :
  is_arithmetic_sequence (λ n => a n t) ↔ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_t_eq_zero_l2840_284063


namespace NUMINAMATH_CALUDE_multiply_powers_of_x_l2840_284054

theorem multiply_powers_of_x (x : ℝ) : 2 * (x^3) * (x^3) = 2 * (x^6) := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_x_l2840_284054


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2840_284006

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (Real.sin (3 * x) * Real.cos (3 * x) + Real.cos (3 * x) * Real.sin (3 * x) = 3 / 8) ↔
  (∃ k : ℤ, x = (7.5 * π / 180) + k * (π / 2) ∨ x = (37.5 * π / 180) + k * (π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2840_284006


namespace NUMINAMATH_CALUDE_locus_of_centers_l2840_284053

/-- The locus of centers of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)) →
  8 * a^2 + 9 * b^2 - 16 * a - 64 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2840_284053


namespace NUMINAMATH_CALUDE_gcd_16_12_l2840_284066

def operation_process : List (Nat × Nat) := [(16, 12), (4, 12), (4, 8), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_l2840_284066


namespace NUMINAMATH_CALUDE_f_symmetry_l2840_284039

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem f_symmetry (a : ℝ) : f a - f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2840_284039


namespace NUMINAMATH_CALUDE_min_value_expression_l2840_284013

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) : 
  (6 * z) / (x + 2 * y + k) + (6 * x) / (2 * z + y + k) + (3 * y) / (x + z + k) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2840_284013


namespace NUMINAMATH_CALUDE_smallest_value_l2840_284048

def x : ℝ := 4
def y : ℝ := 2

theorem smallest_value : 
  min (x + y) (min (x * y) (min (x - y) (min (x / y) (y / x)))) = y / x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l2840_284048


namespace NUMINAMATH_CALUDE_zeoland_speeding_fine_l2840_284059

/-- The speeding fine structure in Zeoland -/
structure SpeedingFine where
  totalFine : ℕ      -- Total fine amount
  speedLimit : ℕ     -- Posted speed limit
  actualSpeed : ℕ    -- Actual speed of the driver
  finePerMph : ℕ     -- Fine per mile per hour over the limit

/-- Theorem: Given Jed's speeding fine details, prove the fine per mph over the limit -/
theorem zeoland_speeding_fine (fine : SpeedingFine) 
  (h1 : fine.totalFine = 256)
  (h2 : fine.speedLimit = 50)
  (h3 : fine.actualSpeed = 66) :
  fine.finePerMph = 16 := by
  sorry


end NUMINAMATH_CALUDE_zeoland_speeding_fine_l2840_284059


namespace NUMINAMATH_CALUDE_unfair_coin_prob_theorem_l2840_284004

/-- An unfair coin with probabilities of heads and tails -/
structure UnfairCoin where
  pH : ℝ  -- Probability of heads
  pT : ℝ  -- Probability of tails
  sum_one : pH + pT = 1
  unfair : pH ≠ pT

/-- The probability of getting one head and one tail in two tosses -/
def prob_one_head_one_tail (c : UnfairCoin) : ℝ :=
  2 * c.pH * c.pT

/-- The probability of getting two heads and two tails in four tosses -/
def prob_two_heads_two_tails (c : UnfairCoin) : ℝ :=
  6 * c.pH * c.pH * c.pT * c.pT

/-- Theorem: For an unfair coin where the probability of getting one head and one tail
    in two tosses is 1/2, the probability of getting two heads and two tails in four tosses is 3/8 -/
theorem unfair_coin_prob_theorem (c : UnfairCoin) 
    (h : prob_one_head_one_tail c = 1/2) : 
    prob_two_heads_two_tails c = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_prob_theorem_l2840_284004


namespace NUMINAMATH_CALUDE_sqrt_2a_plus_b_equals_3_l2840_284097

theorem sqrt_2a_plus_b_equals_3 (a b : ℝ) 
  (h1 : (2*a - 1) = 9)
  (h2 : a - 2*b + 1 = 8) :
  Real.sqrt (2*a + b) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2a_plus_b_equals_3_l2840_284097


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2840_284065

theorem polynomial_division_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, x^5 + x^2 + 3 = (x - 3)^2 * q x + 219 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2840_284065


namespace NUMINAMATH_CALUDE_eric_chicken_farm_eggs_l2840_284009

/-- Calculates the number of eggs collected given the number of chickens, eggs per chicken per day, and number of days. -/
def eggs_collected (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_chickens * eggs_per_chicken_per_day * num_days

/-- Proves that 4 chickens laying 3 eggs per day will produce 36 eggs in 3 days. -/
theorem eric_chicken_farm_eggs : eggs_collected 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_eric_chicken_farm_eggs_l2840_284009


namespace NUMINAMATH_CALUDE_right_triangle_sine_inequality_l2840_284003

/-- Given a right-angled triangle with hypotenuse parallel to plane α, and angles θ₁ and θ₂
    between the lines containing the two legs of the triangle and plane α,
    prove that sin²θ₁ + sin²θ₂ ≤ 1 -/
theorem right_triangle_sine_inequality (θ₁ θ₂ : Real) 
    (h₁ : 0 ≤ θ₁ ∧ θ₁ ≤ π / 2) 
    (h₂ : 0 ≤ θ₂ ∧ θ₂ ≤ π / 2) 
    (h_right_angle : θ₁ + θ₂ ≤ π / 2) : 
    Real.sin θ₁ ^ 2 + Real.sin θ₂ ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_inequality_l2840_284003


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l2840_284047

/-- Custom multiplication operation for integers -/
def customMult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y10 = 90 under the custom multiplication, then y = 11 -/
theorem custom_mult_theorem (y : ℤ) (h : customMult y 10 = 90) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_theorem_l2840_284047


namespace NUMINAMATH_CALUDE_min_difference_l2840_284051

open Real

noncomputable def f (x : ℝ) : ℝ := exp (x - 1)

noncomputable def g (x : ℝ) : ℝ := 1/2 + log (x/2)

theorem min_difference (a b : ℝ) (h : f a = g b) :
  ∃ (min : ℝ), min = 1 + log 2 ∧ ∀ (a' b' : ℝ), f a' = g b' → b' - a' ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_difference_l2840_284051


namespace NUMINAMATH_CALUDE_football_games_per_month_l2840_284011

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end NUMINAMATH_CALUDE_football_games_per_month_l2840_284011


namespace NUMINAMATH_CALUDE_remainder_problem_l2840_284019

theorem remainder_problem (N : ℤ) : 
  N % 19 = 7 → N % 20 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2840_284019


namespace NUMINAMATH_CALUDE_triangle_inequality_l2840_284020

theorem triangle_inequality (a b c p : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_half_perimeter : p = (a + b + c) / 2) :
  a^2 * (p - a) * (p - b) + b^2 * (p - b) * (p - c) + c^2 * (p - c) * (p - a) ≤ (4 / 27) * p^4 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2840_284020


namespace NUMINAMATH_CALUDE_triangle_angle_c_l2840_284073

theorem triangle_angle_c (A B C : Real) (a b c : Real) :
  -- Conditions
  a + b + c = Real.sqrt 2 + 1 →  -- Perimeter condition
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →  -- Area condition
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →  -- Sine relation
  -- Conclusion
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l2840_284073


namespace NUMINAMATH_CALUDE_tv_production_reduction_l2840_284079

/-- Given a factory that produces televisions, calculate the percentage reduction in production from the first year to the second year. -/
theorem tv_production_reduction (daily_rate : ℕ) (second_year_total : ℕ) : 
  daily_rate = 10 →
  second_year_total = 3285 →
  (1 - (second_year_total : ℝ) / (daily_rate * 365 : ℝ)) * 100 = 10 := by
  sorry

#check tv_production_reduction

end NUMINAMATH_CALUDE_tv_production_reduction_l2840_284079


namespace NUMINAMATH_CALUDE_inequality_solution_length_l2840_284046

theorem inequality_solution_length (k : ℝ) : 
  (∃ a b : ℝ, a < b ∧ 
    (∀ x : ℝ, a ≤ x ∧ x ≤ b ↔ 1 ≤ x^2 - 3*x + k ∧ x^2 - 3*x + k ≤ 5) ∧
    b - a = 8) →
  k = 9/4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_length_l2840_284046


namespace NUMINAMATH_CALUDE_sum_f_1_to_10_l2840_284023

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_f_1_to_10 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : is_periodic_3 f) 
  (h_f_neg_1 : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_f_1_to_10_l2840_284023


namespace NUMINAMATH_CALUDE_optimal_numbering_scheme_l2840_284055

/-- Represents a numbering scheme for a population --/
structure NumberingScheme where
  start : Nat
  digits : Nat

/-- Checks if a numbering scheme is valid for a given population size --/
def isValidScheme (populationSize : Nat) (scheme : NumberingScheme) : Prop :=
  scheme.start = 0 ∧
  scheme.digits = 3 ∧
  10 ^ scheme.digits > populationSize

/-- Theorem stating the optimal numbering scheme for the given conditions --/
theorem optimal_numbering_scheme
  (populationSize : Nat)
  (sampleSize : Nat)
  (h1 : populationSize = 106)
  (h2 : sampleSize = 10)
  (h3 : sampleSize < populationSize) :
  ∃ (scheme : NumberingScheme),
    isValidScheme populationSize scheme ∧
    scheme.start = 0 ∧
    scheme.digits = 3 :=
  sorry

end NUMINAMATH_CALUDE_optimal_numbering_scheme_l2840_284055


namespace NUMINAMATH_CALUDE_otimes_h_otimes_h_l2840_284068

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^2 - y

-- Theorem statement
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_otimes_h_l2840_284068
