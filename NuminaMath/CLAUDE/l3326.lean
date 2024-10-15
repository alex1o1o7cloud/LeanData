import Mathlib

namespace NUMINAMATH_CALUDE_charles_skittles_l3326_332604

/-- The number of Skittles Charles has left after Diana takes some away. -/
def skittles_left (initial : ℕ) (taken : ℕ) : ℕ := initial - taken

/-- Theorem: If Charles has 25 Skittles initially and Diana takes 7 Skittles away,
    then Charles will have 18 Skittles left. -/
theorem charles_skittles : skittles_left 25 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_charles_skittles_l3326_332604


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_value_l3326_332651

/-- Regular triangular pyramid with inscribed sphere -/
structure RegularTriangularPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere radius is one-fourth of the base side length -/
  radius_relation : sphere_radius = base_side / 4

/-- Dihedral angle at the apex of a regular triangular pyramid -/
def dihedral_angle_cosine (pyramid : RegularTriangularPyramid) : ℝ :=
  -- Definition of the dihedral angle cosine
  sorry

/-- Theorem: The cosine of the dihedral angle at the apex is 23/26 -/
theorem dihedral_angle_cosine_value (pyramid : RegularTriangularPyramid) :
  dihedral_angle_cosine pyramid = 23 / 26 :=
by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_value_l3326_332651


namespace NUMINAMATH_CALUDE_library_reorganization_l3326_332634

theorem library_reorganization (total_books : ℕ) (books_per_section : ℕ) (remainder : ℕ) : 
  total_books = 1521 * 41 →
  books_per_section = 45 →
  remainder = total_books % books_per_section →
  remainder = 36 := by
sorry

end NUMINAMATH_CALUDE_library_reorganization_l3326_332634


namespace NUMINAMATH_CALUDE_simplified_tax_system_is_most_suitable_l3326_332638

-- Define the business characteristics
structure BusinessCharacteristics where
  isFlowerSelling : Bool
  hasNoExperience : Bool
  hasSingleOutlet : Bool
  isSelfOperated : Bool

-- Define the tax regimes
inductive TaxRegime
  | UnifiedAgricultural
  | Simplified
  | General
  | Patent

-- Define a function to determine the most suitable tax regime
def mostSuitableTaxRegime (business : BusinessCharacteristics) : TaxRegime :=
  sorry

-- Theorem statement
theorem simplified_tax_system_is_most_suitable 
  (leonidBusiness : BusinessCharacteristics)
  (h1 : leonidBusiness.isFlowerSelling = true)
  (h2 : leonidBusiness.hasNoExperience = true)
  (h3 : leonidBusiness.hasSingleOutlet = true)
  (h4 : leonidBusiness.isSelfOperated = true) :
  mostSuitableTaxRegime leonidBusiness = TaxRegime.Simplified :=
sorry

end NUMINAMATH_CALUDE_simplified_tax_system_is_most_suitable_l3326_332638


namespace NUMINAMATH_CALUDE_prime_power_equation_l3326_332631

theorem prime_power_equation (p q s : Nat) (y : Nat) (hp : Prime p) (hq : Prime q) (hs : Prime s) (hy : y > 1) 
  (h : 2^s * q = p^y - 1) : p = 3 ∨ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_equation_l3326_332631


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3326_332671

theorem scientific_notation_proof : ∃ (a : ℝ) (n : ℤ), 
  0.00076 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7.6 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3326_332671


namespace NUMINAMATH_CALUDE_complex_power_simplification_l3326_332653

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_power_simplification :
  ((1 + i) / (1 - i)) ^ 1002 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l3326_332653


namespace NUMINAMATH_CALUDE_axis_of_symmetry_shifted_cosine_l3326_332611

open Real

theorem axis_of_symmetry_shifted_cosine :
  let f : ℝ → ℝ := λ x ↦ Real.sin (π / 2 - x)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x + π / 6)
  ∀ x : ℝ, g (5 * π / 6 + x) = g (5 * π / 6 - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_shifted_cosine_l3326_332611


namespace NUMINAMATH_CALUDE_claire_male_pets_l3326_332612

theorem claire_male_pets (total_pets : ℕ) (gerbils : ℕ) (hamsters : ℕ)
  (h_total : total_pets = 92)
  (h_only_gerbils_hamsters : total_pets = gerbils + hamsters)
  (h_gerbils : gerbils = 68)
  (h_male_gerbils : ℕ → ℕ := λ x => x / 4)
  (h_male_hamsters : ℕ → ℕ := λ x => x / 3)
  : h_male_gerbils gerbils + h_male_hamsters hamsters = 25 := by
  sorry

end NUMINAMATH_CALUDE_claire_male_pets_l3326_332612


namespace NUMINAMATH_CALUDE_sports_equipment_choices_l3326_332608

theorem sports_equipment_choices (basketballs volleyballs : ℕ) 
  (hb : basketballs = 5) (hv : volleyballs = 4) : 
  basketballs * volleyballs = 20 := by
  sorry

end NUMINAMATH_CALUDE_sports_equipment_choices_l3326_332608


namespace NUMINAMATH_CALUDE_polynomial_sum_coefficients_l3326_332680

theorem polynomial_sum_coefficients : 
  ∀ A B C D E : ℚ, 
  (∀ x : ℚ, (x + 2) * (x + 3) * (3*x^2 - x + 5) = A*x^4 + B*x^3 + C*x^2 + D*x + E) →
  A + B + C + D + E = 84 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_coefficients_l3326_332680


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3326_332646

theorem cistern_filling_time (p q : ℝ) (h1 : q = 15) (h2 : 2/p + 2/q + 10.5/q = 1) : p = 12 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3326_332646


namespace NUMINAMATH_CALUDE_literature_tech_cost_difference_l3326_332652

theorem literature_tech_cost_difference :
  let num_books : ℕ := 45
  let lit_cost : ℕ := 7
  let tech_cost : ℕ := 5
  (num_books * lit_cost) - (num_books * tech_cost) = 90 := by
sorry

end NUMINAMATH_CALUDE_literature_tech_cost_difference_l3326_332652


namespace NUMINAMATH_CALUDE_calculate_expression_l3326_332620

theorem calculate_expression : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (60 * π / 180) = -1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3326_332620


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3326_332682

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3326_332682


namespace NUMINAMATH_CALUDE_trig_identities_l3326_332659

/-- Given an angle θ with vertex at the origin, initial side on positive x-axis,
    and terminal side on y = 1/2x (x ≤ 0), prove trigonometric identities. -/
theorem trig_identities (θ α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = (1/2) * x ∧ 
   Real.cos θ = x / Real.sqrt (x^2 + y^2) ∧ 
   Real.sin θ = y / Real.sqrt (x^2 + y^2)) →
  (Real.cos (π/2 + θ) = Real.sqrt 5 / 5) ∧
  (Real.cos (α + π/4) = Real.sin θ → 
   (Real.sin (2*α + π/4) = 7 * Real.sqrt 2 / 10 ∨ 
    Real.sin (2*α + π/4) = - Real.sqrt 2 / 10)) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3326_332659


namespace NUMINAMATH_CALUDE_blue_paint_cans_l3326_332663

/-- Given a paint mixture with a blue to green ratio of 4:3 and a total of 35 cans,
    prove that 20 cans of blue paint are needed. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : 
  total_cans = 35 → 
  blue_ratio = 4 → 
  green_ratio = 3 → 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l3326_332663


namespace NUMINAMATH_CALUDE_points_on_line_relationship_l3326_332629

/-- Given two points A(-2, y₁) and B(1, y₂) on the line y = -2x + 3, prove that y₁ > y₂ -/
theorem points_on_line_relationship (y₁ y₂ : ℝ) : 
  ((-2 : ℝ), y₁) ∈ {(x, y) | y = -2*x + 3} → 
  ((1 : ℝ), y₂) ∈ {(x, y) | y = -2*x + 3} → 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_relationship_l3326_332629


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3326_332639

theorem geometric_sequence_fourth_term 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h1 : a₁ = 2^(1/4)) 
  (h2 : a₂ = 2^(1/5)) 
  (h3 : a₃ = 2^(1/10)) 
  (h_geometric : ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r) : 
  a₄ = 2^(1/10) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3326_332639


namespace NUMINAMATH_CALUDE_vectors_collinear_necessary_not_sufficient_l3326_332603

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a vector in 3D space
def Vector3D (A B : Point3D) : Point3D :=
  ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

-- Define collinearity for vectors
def vectorsCollinear (v1 v2 : Point3D) : Prop :=
  ∃ k : ℝ, v1 = ⟨k * v2.x, k * v2.y, k * v2.z⟩

-- Define collinearity for points
def pointsCollinear (A B C D : Point3D) : Prop :=
  ∃ t u v : ℝ, Vector3D A B = ⟨t * (C.x - A.x), t * (C.y - A.y), t * (C.z - A.z)⟩ ∧
               Vector3D A C = ⟨u * (D.x - A.x), u * (D.y - A.y), u * (D.z - A.z)⟩ ∧
               Vector3D A D = ⟨v * (B.x - A.x), v * (B.y - A.y), v * (B.z - A.z)⟩

theorem vectors_collinear_necessary_not_sufficient (A B C D : Point3D) :
  (pointsCollinear A B C D → vectorsCollinear (Vector3D A B) (Vector3D C D)) ∧
  ¬(vectorsCollinear (Vector3D A B) (Vector3D C D) → pointsCollinear A B C D) :=
by sorry

end NUMINAMATH_CALUDE_vectors_collinear_necessary_not_sufficient_l3326_332603


namespace NUMINAMATH_CALUDE_curve_and_tangent_properties_l3326_332625

/-- The function f(x) -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + n

/-- The tangent line of f at x = 0 -/
def tangent_line (x : ℝ) : ℝ := 4*x + 3

/-- The inequality condition for x > 1 -/
def inequality_condition (m n : ℝ) (k : ℤ) (x : ℝ) : Prop :=
  x > 1 → f m n (x + x * Real.log x) > f m n (↑k * (x - 1))

theorem curve_and_tangent_properties :
  ∃ (m n : ℝ) (k : ℤ),
    (∀ x, f m n x = tangent_line x) ∧
    (∀ x, inequality_condition m n k x) ∧
    m = 4 ∧ n = 3 ∧ k = 3 ∧
    (∀ k' : ℤ, (∀ x, inequality_condition m n k' x) → k' ≤ k) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_tangent_properties_l3326_332625


namespace NUMINAMATH_CALUDE_wheel_diameter_l3326_332673

theorem wheel_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end NUMINAMATH_CALUDE_wheel_diameter_l3326_332673


namespace NUMINAMATH_CALUDE_relationship_abc_l3326_332686

-- Define a, b, and c
def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

-- Theorem stating the relationship between a, b, and c
theorem relationship_abc : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3326_332686


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_12_l3326_332678

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def lastTwoDigits (n : ℕ) : ℕ := n % 100

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |> List.sum

theorem last_two_digits_sum_factorials_12 :
  lastTwoDigits (sumFactorials 12) = 13 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_12_l3326_332678


namespace NUMINAMATH_CALUDE_carbon_mass_percentage_l3326_332607

/-- The mass percentage of an element in a compound -/
def mass_percentage (element : String) (compound : String) : ℝ := sorry

/-- The given mass percentage of C in the compound -/
def given_percentage : ℝ := 54.55

theorem carbon_mass_percentage (compound : String) :
  mass_percentage "C" compound = given_percentage := by sorry

end NUMINAMATH_CALUDE_carbon_mass_percentage_l3326_332607


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3326_332685

theorem inequality_solution_set (a : ℝ) : 
  ((3 - a) / 2 - 2 = 2) → 
  {x : ℝ | (2 - a / 5) < (1 / 3) * x} = {x : ℝ | x > 9} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3326_332685


namespace NUMINAMATH_CALUDE_intersection_of_parallel_planes_l3326_332670

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLines : Line → Line → Prop)

-- State the theorem
theorem intersection_of_parallel_planes 
  (α β γ : Plane) (m n : Line) :
  α ≠ β → α ≠ γ → β ≠ γ →
  m = intersect α γ →
  n = intersect β γ →
  parallelPlanes α β →
  parallelLines m n :=
sorry

end NUMINAMATH_CALUDE_intersection_of_parallel_planes_l3326_332670


namespace NUMINAMATH_CALUDE_function_intersection_theorem_l3326_332689

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x + b) / x

noncomputable def g (a x : ℝ) : ℝ := a + 2 - x - 2 / x

def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

def exactly_one_intersection (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x ≤ b ∧ f x = g x

theorem function_intersection_theorem (a b : ℝ) :
  a ≤ 2 →
  a ≠ 0 →
  has_extremum_at (f a b) (1 / Real.exp 1) →
  (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2)) ↔
  exactly_one_intersection (f a b) (g a) 0 2 :=
sorry

end NUMINAMATH_CALUDE_function_intersection_theorem_l3326_332689


namespace NUMINAMATH_CALUDE_pauls_crayons_and_erasers_l3326_332666

theorem pauls_crayons_and_erasers 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 531)
  (h2 : initial_erasers = 38)
  (h3 : final_crayons = 391)
  (h4 : initial_erasers = final_erasers) :
  initial_crayons - final_crayons - initial_erasers = 102 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_and_erasers_l3326_332666


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3326_332626

/-- Given a rectangle with length thrice its breadth and area 507 m², 
    prove that its perimeter is 104 m. -/
theorem rectangle_perimeter (breadth length : ℝ) : 
  length = 3 * breadth → 
  breadth * length = 507 → 
  2 * (length + breadth) = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3326_332626


namespace NUMINAMATH_CALUDE_stone_145_is_5_l3326_332633

def stone_number (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

theorem stone_145_is_5 : stone_number 145 = stone_number 5 := by
  sorry

end NUMINAMATH_CALUDE_stone_145_is_5_l3326_332633


namespace NUMINAMATH_CALUDE_transform_F_coordinates_l3326_332600

/-- Reflects a point over the x-axis -/
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Rotates a point 90 degrees counterclockwise around the origin -/
def rotate_90_ccw (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

/-- The initial coordinates of point F -/
def F : ℝ × ℝ := (3, -1)

theorem transform_F_coordinates :
  (rotate_90_ccw (reflect_over_x F)) = (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_transform_F_coordinates_l3326_332600


namespace NUMINAMATH_CALUDE_chloe_score_l3326_332649

/-- Calculates the total score in Chloe's video game. -/
def total_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Proves that Chloe's total score is 81 points given the specified conditions. -/
theorem chloe_score :
  total_score 9 6 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_chloe_score_l3326_332649


namespace NUMINAMATH_CALUDE_lollipop_distribution_theorem_l3326_332647

/-- Given a number of lollipops and kids, calculate the minimum number of additional
    lollipops needed for equal distribution -/
def min_additional_lollipops (total_lollipops : ℕ) (num_kids : ℕ) : ℕ :=
  let lollipops_per_kid := (total_lollipops + num_kids - 1) / num_kids
  lollipops_per_kid * num_kids - total_lollipops

/-- Theorem stating that for 650 lollipops and 42 kids, 
    the minimum number of additional lollipops needed is 22 -/
theorem lollipop_distribution_theorem :
  min_additional_lollipops 650 42 = 22 := by
  sorry


end NUMINAMATH_CALUDE_lollipop_distribution_theorem_l3326_332647


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l3326_332635

open Real

theorem proposition_p_and_not_q : 
  (∃ x : ℝ, x^2 + 2*x + 5 ≤ 4) ∧ 
  (∀ x ∈ Set.Ioo 0 (π/2), sin x + 4/(sin x) > 4) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l3326_332635


namespace NUMINAMATH_CALUDE_juvy_garden_rosemary_rows_l3326_332622

/-- Represents a garden with rows of plants -/
structure Garden where
  total_rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  chive_plants : ℕ

/-- Calculates the number of rows planted with rosemary -/
def rosemary_rows (g : Garden) : ℕ :=
  g.total_rows - g.parsley_rows - (g.chive_plants / g.plants_per_row)

/-- Theorem stating that Juvy's garden has 2 rows of rosemary -/
theorem juvy_garden_rosemary_rows :
  let g : Garden := {
    total_rows := 20,
    plants_per_row := 10,
    parsley_rows := 3,
    chive_plants := 150
  }
  rosemary_rows g = 2 := by sorry

end NUMINAMATH_CALUDE_juvy_garden_rosemary_rows_l3326_332622


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l3326_332628

def product_of_evens (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    Finset.prod (Finset.range ((x / 2) + 1)) (fun i => 2 * i)
  else
    Finset.prod (Finset.range (x / 2)) (fun i => 2 * i)

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem greatest_prime_factor_of_sum : 
  greatest_prime_factor (product_of_evens 26 + product_of_evens 24) = 23 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l3326_332628


namespace NUMINAMATH_CALUDE_b_range_l3326_332648

noncomputable section

def y (a x : ℝ) : ℝ := a^x

def f (a b x : ℝ) : ℝ := a^x + (Real.log x) / (Real.log a) + b

theorem b_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, y a x ≤ 6 - y a x) →
  (∃ x ∈ Set.Ioo 1 2, f a b x = 0) →
  -5 < b ∧ b < -2 :=
sorry

end NUMINAMATH_CALUDE_b_range_l3326_332648


namespace NUMINAMATH_CALUDE_power_equation_solution_l3326_332660

theorem power_equation_solution : 5^3 - 7 = 6^2 + 82 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3326_332660


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3326_332687

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point P with coordinates (m+3, m+1) -/
def P (m : ℝ) : Point :=
  { x := m + 3, y := m + 1 }

/-- Theorem: If P(m+3, m+1) lies on the x-axis, then its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (P m).y = 0 → P m = { x := 2, y := 0 } := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3326_332687


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l3326_332645

theorem soccer_ball_cost (ball_cost shirt_cost : ℝ) : 
  ball_cost + shirt_cost = 100 →
  2 * ball_cost + 3 * shirt_cost = 262 →
  ball_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l3326_332645


namespace NUMINAMATH_CALUDE_expression_equivalence_l3326_332665

theorem expression_equivalence : 
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * 
  (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3326_332665


namespace NUMINAMATH_CALUDE_cricket_score_problem_l3326_332697

theorem cricket_score_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 36 ∧  -- average score
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, a = 4 * k₁ ∧ b = 4 * k₂ ∧ c = 4 * k₃ ∧ d = 4 * k₄ ∧ e = 4 * k₅) ∧  -- scores are multiples of 4
  d = e + 12 ∧  -- D scored 12 more than E
  e = a - 8 ∧  -- E scored 8 fewer than A
  b = d + e ∧  -- B scored as many as D and E combined
  b + c = 107 ∧  -- B and C scored 107 between them
  a > b ∧ a > c ∧ a > d ∧ a > e  -- A scored the maximum runs
  →
  e = 20 := by
sorry

end NUMINAMATH_CALUDE_cricket_score_problem_l3326_332697


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_l3326_332606

/-- The greatest common factor of 15, 20, and 30 -/
def A : ℕ := Nat.gcd 15 (Nat.gcd 20 30)

/-- The least common multiple of 15, 20, and 30 -/
def B : ℕ := Nat.lcm 15 (Nat.lcm 20 30)

/-- The sum of the greatest common factor and least common multiple of 15, 20, and 30 is 65 -/
theorem sum_gcf_lcm : A + B = 65 := by sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_l3326_332606


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l3326_332683

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l3326_332683


namespace NUMINAMATH_CALUDE_incorrect_calculation_correction_l3326_332662

theorem incorrect_calculation_correction (x : ℝ) (h : x * 7 = 115.15) : 
  115.15 / 49 = 2.35 := by
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_correction_l3326_332662


namespace NUMINAMATH_CALUDE_range_of_b_monotonicity_condition_comparison_inequality_l3326_332621

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - x * Real.log x

-- Statement 1
theorem range_of_b (a : ℝ) (b : ℝ) (h1 : a > 0) (h2 : f a 1 = 2) 
  (h3 : ∀ x > 0, f a x ≥ b * x^2 + 2 * x) : b ≤ 0 := sorry

-- Statement 2
theorem monotonicity_condition (a : ℝ) (h : a > 0) : 
  (∀ x > 0, Monotone (f a)) ↔ a ≥ 1 / (2 * Real.exp 1) := sorry

-- Statement 3
theorem comparison_inequality (x y : ℝ) (h1 : 1 / Real.exp 1 < x) (h2 : x < y) (h3 : y < 1) :
  y / x < (1 + Real.log y) / (1 + Real.log x) := sorry

end NUMINAMATH_CALUDE_range_of_b_monotonicity_condition_comparison_inequality_l3326_332621


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_and_product_l3326_332693

theorem quadratic_roots_sum_squares_and_product (u v : ℝ) : 
  u^2 - 5*u + 3 = 0 → v^2 - 5*v + 3 = 0 → u^2 + v^2 + u*v = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_and_product_l3326_332693


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3326_332688

theorem unique_integer_solution : 
  ∃! (n : ℤ), (n^2 + 3*n + 5) / (n + 2 : ℚ) = 1 + Real.sqrt (6 - 2*n) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3326_332688


namespace NUMINAMATH_CALUDE_meet_once_l3326_332684

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def meetingCount (m : Movement) : ℕ :=
  sorry

/-- The theorem to be proved --/
theorem meet_once (m : Movement) 
  (h1 : m.michaelSpeed = 6)
  (h2 : m.truckSpeed = 10)
  (h3 : m.pailDistance = 200)
  (h4 : m.truckStopTime = 40)
  (h5 : m.pailDistance = m.truckSpeed * (m.pailDistance / m.michaelSpeed - m.truckStopTime)) :
  meetingCount m = 1 := by
  sorry

#check meet_once

end NUMINAMATH_CALUDE_meet_once_l3326_332684


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3326_332636

theorem absolute_value_equation_solution :
  ∃ x : ℝ, (|x - 25| + |x - 21| = |3*x - 75|) ∧ (x = 71/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3326_332636


namespace NUMINAMATH_CALUDE_highway_project_deadline_l3326_332695

/-- Represents the initial deadline for completing the highway project --/
def initial_deadline : ℝ := 37.5

/-- The number of initial workers --/
def initial_workers : ℕ := 100

/-- The number of additional workers hired --/
def additional_workers : ℕ := 60

/-- The initial daily work hours --/
def initial_hours : ℕ := 8

/-- The new daily work hours after hiring additional workers --/
def new_hours : ℕ := 10

/-- The number of days worked before hiring additional workers --/
def days_worked : ℕ := 25

/-- The fraction of work completed before hiring additional workers --/
def work_completed : ℚ := 1/3

/-- Theorem stating that the initial deadline is correct given the conditions --/
theorem highway_project_deadline :
  ∃ (total_work : ℝ),
    total_work = initial_workers * days_worked * initial_hours ∧
    (2/3 : ℝ) * total_work = (initial_workers + additional_workers) * (initial_deadline - days_worked) * new_hours :=
by sorry

end NUMINAMATH_CALUDE_highway_project_deadline_l3326_332695


namespace NUMINAMATH_CALUDE_morning_sodas_count_l3326_332601

theorem morning_sodas_count (afternoon_sodas : ℕ) (total_sodas : ℕ) 
  (h1 : afternoon_sodas = 19)
  (h2 : total_sodas = 96) :
  total_sodas - afternoon_sodas = 77 := by
  sorry

end NUMINAMATH_CALUDE_morning_sodas_count_l3326_332601


namespace NUMINAMATH_CALUDE_jackfruit_division_l3326_332667

/-- Represents the fair division of jackfruits between Renato and Leandro -/
def fair_division (renato_watermelons leandro_watermelons marcelo_jackfruits : ℕ) 
  (renato_jackfruits leandro_jackfruits : ℕ) : Prop :=
  renato_watermelons = 30 ∧
  leandro_watermelons = 18 ∧
  marcelo_jackfruits = 24 ∧
  renato_jackfruits + leandro_jackfruits = marcelo_jackfruits ∧
  (renato_watermelons + leandro_watermelons) / 3 = 16 ∧
  renato_jackfruits * 2 = renato_watermelons ∧
  leandro_jackfruits * 2 = leandro_watermelons

theorem jackfruit_division :
  ∃ (renato_jackfruits leandro_jackfruits : ℕ),
    fair_division 30 18 24 renato_jackfruits leandro_jackfruits ∧
    renato_jackfruits = 15 ∧
    leandro_jackfruits = 9 := by
  sorry

end NUMINAMATH_CALUDE_jackfruit_division_l3326_332667


namespace NUMINAMATH_CALUDE_attendees_with_all_items_l3326_332630

def venue_capacity : ℕ := 5400
def tshirt_interval : ℕ := 90
def cap_interval : ℕ := 45
def wristband_interval : ℕ := 60

theorem attendees_with_all_items :
  (venue_capacity / (Nat.lcm tshirt_interval (Nat.lcm cap_interval wristband_interval))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_attendees_with_all_items_l3326_332630


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l3326_332656

/-- A four-digit palindrome is a number of the form abba where a and b are digits and a ≠ 0 -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b : ℕ), 0 < a ∧ a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, is_four_digit_palindrome n → n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l3326_332656


namespace NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l3326_332668

-- Define the polynomial function
def f (x : ℝ) : ℝ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

-- Define Horner's method for this specific polynomial
def horner_method (x : ℝ) : ℝ :=
  ((((((8 * x + 5) * x + 0) * x + 3) * x + 0) * x + 0) * x + 2) * x + 1

-- Theorem statement
theorem horner_method_equals_f_at_2 : 
  horner_method 2 = f 2 ∧ horner_method 2 = 1397 := by sorry

end NUMINAMATH_CALUDE_horner_method_equals_f_at_2_l3326_332668


namespace NUMINAMATH_CALUDE_min_value_of_f_l3326_332609

theorem min_value_of_f (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, 2 * (Real.cos y)^2 + Real.sin y ≥ m) ∧ 
  (∃ z : ℝ, 2 * (Real.cos z)^2 + Real.sin z = m) ∧ 
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3326_332609


namespace NUMINAMATH_CALUDE_transformation_is_left_shift_l3326_332618

/-- A function representing a horizontal shift transformation -/
def horizontalShift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => f (x + shift)

/-- The original function composition -/
def originalFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x - 1)

/-- The transformed function composition -/
def transformedFunc (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x => f (2*x + 1)

theorem transformation_is_left_shift (f : ℝ → ℝ) :
  transformedFunc f = horizontalShift (originalFunc f) 1 := by
  sorry

end NUMINAMATH_CALUDE_transformation_is_left_shift_l3326_332618


namespace NUMINAMATH_CALUDE_count_divisors_of_twenty_divisible_by_five_l3326_332624

theorem count_divisors_of_twenty_divisible_by_five : 
  let a : ℕ → Prop := λ n => 
    n > 0 ∧ 5 ∣ n ∧ n ∣ 20
  (Finset.filter a (Finset.range 21)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_of_twenty_divisible_by_five_l3326_332624


namespace NUMINAMATH_CALUDE_triangle_conversion_cost_l3326_332614

theorem triangle_conversion_cost 
  (side1 : ℝ) (side2 : ℝ) (angle : ℝ) (cost_per_sqm : ℝ) :
  side1 = 32 →
  side2 = 68 →
  angle = 30 * π / 180 →
  cost_per_sqm = 50 →
  (1/2 * side1 * side2 * Real.sin angle) * cost_per_sqm = 54400 :=
by sorry

end NUMINAMATH_CALUDE_triangle_conversion_cost_l3326_332614


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l3326_332658

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) + 1
  f (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l3326_332658


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_specific_l3326_332681

theorem gcd_lcm_sum_specific : Nat.gcd 45 4410 + Nat.lcm 45 4410 = 4455 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_specific_l3326_332681


namespace NUMINAMATH_CALUDE_common_chord_length_l3326_332617

theorem common_chord_length (r : ℝ) (h : r = 15) :
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3326_332617


namespace NUMINAMATH_CALUDE_students_not_picked_l3326_332650

theorem students_not_picked (total : ℕ) (groups : ℕ) (per_group : ℕ) (h1 : total = 64) (h2 : groups = 4) (h3 : per_group = 7) : 
  total - (groups * per_group) = 36 := by
sorry

end NUMINAMATH_CALUDE_students_not_picked_l3326_332650


namespace NUMINAMATH_CALUDE_f_properties_l3326_332674

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x * Real.exp x else x^2 - 2*x + 1/2

theorem f_properties :
  (∀ x, x ≤ 0 → (deriv f) x = 2 * (1 + x) * Real.exp x) ∧
  (∀ x, x > 0 → (deriv f) x = 2*x - 2) ∧
  ((deriv f) (-2) = -2 / Real.exp 2) ∧
  (∀ x, f x ≥ -2 / Real.exp 1) ∧
  (∃ x, f x = -2 / Real.exp 1) ∧
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3326_332674


namespace NUMINAMATH_CALUDE_train_speed_l3326_332679

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 600) (h2 : time = 25) :
  length / time = 24 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3326_332679


namespace NUMINAMATH_CALUDE_pet_store_cages_l3326_332616

/-- The number of bird cages in a pet store --/
def num_cages (num_parrots : Float) (num_parakeets : Float) (avg_birds_per_cage : Float) : Float :=
  (num_parrots + num_parakeets) / avg_birds_per_cage

/-- Theorem: The pet store has approximately 6 bird cages --/
theorem pet_store_cages :
  let num_parrots : Float := 6.0
  let num_parakeets : Float := 2.0
  let avg_birds_per_cage : Float := 1.333333333
  (num_cages num_parrots num_parakeets avg_birds_per_cage).round = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3326_332616


namespace NUMINAMATH_CALUDE_earnings_per_dog_l3326_332655

def dogs_monday_wednesday_friday : ℕ := 7
def dogs_tuesday : ℕ := 12
def dogs_thursday : ℕ := 9
def weekly_earnings : ℕ := 210

def total_dogs : ℕ := dogs_monday_wednesday_friday * 3 + dogs_tuesday + dogs_thursday

theorem earnings_per_dog :
  weekly_earnings / total_dogs = 5 := by sorry

end NUMINAMATH_CALUDE_earnings_per_dog_l3326_332655


namespace NUMINAMATH_CALUDE_trig_identity_l3326_332632

theorem trig_identity (α : Real) (h : π < α ∧ α < 3*π/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3326_332632


namespace NUMINAMATH_CALUDE_sea_glass_collection_l3326_332637

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_red rose_blue : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_red = 9)
  (h4 : rose_blue = 11)
  (dorothy_red : ℕ)
  (h5 : dorothy_red = 2 * (blanche_red + rose_red))
  (dorothy_blue : ℕ)
  (h6 : dorothy_blue = 3 * rose_blue) :
  dorothy_red + dorothy_blue = 57 := by
sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l3326_332637


namespace NUMINAMATH_CALUDE_f_range_l3326_332610

/-- The function f(x) = |x+10| - |3x-1| -/
def f (x : ℝ) : ℝ := |x + 10| - |3*x - 1|

/-- The range of f is (-∞, 31] -/
theorem f_range :
  Set.range f = Set.Iic 31 := by sorry

end NUMINAMATH_CALUDE_f_range_l3326_332610


namespace NUMINAMATH_CALUDE_value_of_b_l3326_332623

theorem value_of_b : ∀ b : ℕ, (5 ^ 5 * b = 3 * 15 ^ 5) ∧ (b = 9 ^ 3) → b = 729 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l3326_332623


namespace NUMINAMATH_CALUDE_program_output_l3326_332615

theorem program_output : 
  let a₀ := 10
  let b := a₀ - 8
  let a₁ := a₀ - b
  a₁ = 8 := by sorry

end NUMINAMATH_CALUDE_program_output_l3326_332615


namespace NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l3326_332692

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l3326_332692


namespace NUMINAMATH_CALUDE_triangle_inequality_third_side_bounds_l3326_332640

theorem triangle_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, 0 < c ∧ a - b < c ∧ c < a + b :=
by
  sorry

theorem third_side_bounds (side1 side2 : ℝ) 
  (h1 : side1 = 6) (h2 : side2 = 10) :
  ∃ side3 : ℝ, 4 < side3 ∧ side3 < 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_third_side_bounds_l3326_332640


namespace NUMINAMATH_CALUDE_N_is_composite_l3326_332627

def N (y : ℕ) : ℚ := (y^125 - 1) / (3^22 - 1)

theorem N_is_composite : ∃ (y : ℕ) (a b : ℕ), a > 1 ∧ b > 1 ∧ N y = (a * b : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_N_is_composite_l3326_332627


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l3326_332642

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics (P C M : ℕ) : 
  P + C + M = P + 180 → (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l3326_332642


namespace NUMINAMATH_CALUDE_inequality_proof_l3326_332657

theorem inequality_proof (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) : 
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + x) * (1 + z))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3326_332657


namespace NUMINAMATH_CALUDE_crate_dimensions_for_largest_tank_l3326_332676

/-- Represents a rectangular crate with length, width, and height -/
structure RectangularCrate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank with radius and height -/
structure CylindricalTank where
  radius : ℝ
  height : ℝ

/-- The tank fits in the crate when standing upright -/
def tankFitsInCrate (tank : CylindricalTank) (crate : RectangularCrate) : Prop :=
  2 * tank.radius ≤ min crate.length crate.width ∧ tank.height ≤ crate.height

theorem crate_dimensions_for_largest_tank (crate : RectangularCrate) 
    (h : ∃ tank : CylindricalTank, tank.radius = 10 ∧ tankFitsInCrate tank crate) :
    crate.length ≥ 20 ∧ crate.width ≥ 20 := by
  sorry

end NUMINAMATH_CALUDE_crate_dimensions_for_largest_tank_l3326_332676


namespace NUMINAMATH_CALUDE_area_FDBG_is_155_l3326_332698

/-- Triangle ABC with given properties -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (AB : Real)
  (AC : Real)
  (area : Real)
  (h_AB : AB = 60)
  (h_AC : AC = 15)
  (h_area : area = 180)

/-- Point D on AB -/
def D (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point E on AC -/
def E (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point F on DE and angle bisector of BAC -/
def F (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Point G on BC and angle bisector of BAC -/
def G (t : Triangle) : ℝ × ℝ :=
  sorry

/-- Length of AD -/
def AD (t : Triangle) : Real :=
  20

/-- Length of DB -/
def DB (t : Triangle) : Real :=
  40

/-- Length of AE -/
def AE (t : Triangle) : Real :=
  5

/-- Length of EC -/
def EC (t : Triangle) : Real :=
  10

/-- Area of quadrilateral FDBG -/
def area_FDBG (t : Triangle) : Real :=
  sorry

/-- Main theorem: Area of FDBG is 155 -/
theorem area_FDBG_is_155 (t : Triangle) :
  area_FDBG t = 155 := by
  sorry

end NUMINAMATH_CALUDE_area_FDBG_is_155_l3326_332698


namespace NUMINAMATH_CALUDE_g_at_7_equals_neg_20_l3326_332644

/-- A polynomial function g(x) of degree 7 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

/-- Theorem stating that g(7) = -20 given g(-7) = 12 -/
theorem g_at_7_equals_neg_20 (a b c : ℝ) : g a b c (-7) = 12 → g a b c 7 = -20 := by
  sorry

end NUMINAMATH_CALUDE_g_at_7_equals_neg_20_l3326_332644


namespace NUMINAMATH_CALUDE_smallest_denominators_sum_l3326_332661

theorem smallest_denominators_sum (p q : ℕ) (h : q > 0) :
  (∃ k : ℕ, Nat.div p q = k ∧ 
   ∃ r : ℕ, r < q ∧ 
   ∃ n : ℕ, Nat.div (r * 1000 + 171) q = n ∧
   ∃ s : ℕ, s < q ∧ Nat.div (s * 1000 + 171) q = n) →
  (∃ q1 q2 : ℕ, q1 < q2 ∧ 
   (∀ q' : ℕ, q' ≠ q1 ∧ q' ≠ q2 → q' > q2) ∧
   q1 + q2 = 99) :=
sorry

end NUMINAMATH_CALUDE_smallest_denominators_sum_l3326_332661


namespace NUMINAMATH_CALUDE_sphere_radius_from_shadows_l3326_332699

/-- Given a sphere and a stick under parallel sun rays, prove the radius of the sphere -/
theorem sphere_radius_from_shadows
  (shadow_sphere : ℝ)  -- Length of the sphere's shadow
  (height_stick : ℝ)   -- Height of the stick
  (shadow_stick : ℝ)   -- Length of the stick's shadow
  (h_shadow_sphere : shadow_sphere = 20)
  (h_height_stick : height_stick = 1)
  (h_shadow_stick : shadow_stick = 4)
  : ∃ (radius : ℝ), radius = 5 ∧ (radius / shadow_sphere = height_stick / shadow_stick) :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_shadows_l3326_332699


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3326_332664

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * Real.sqrt 2 * x

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 1

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∃ (x y : ℝ), asymptote x y ∧ (∃ (k : ℝ), y = (b/a) * x + k)) 
  (h4 : ∃ (x y : ℝ), hyperbola a b x y ∧ directrix x ∧ parabola x y) :
  a^2 = 2 ∧ b^2 = 6 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3326_332664


namespace NUMINAMATH_CALUDE_commute_speed_theorem_l3326_332613

theorem commute_speed_theorem (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/15)) (h2 : d = 70 * (t - 1/15)) :
  d / t = 58 := by sorry

end NUMINAMATH_CALUDE_commute_speed_theorem_l3326_332613


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3326_332643

theorem polynomial_evaluation :
  let f (x : ℝ) := 2 * x^4 + 3 * x^3 + 5 * x^2 + x + 4
  f (-2) = 30 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3326_332643


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l3326_332696

theorem sum_of_squares_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + 2*a = 0 ∧ y^2 + a*y + 2*a = 0 ∧ x^2 + y^2 = 21) ↔ a = -3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l3326_332696


namespace NUMINAMATH_CALUDE_p_false_and_q_true_l3326_332641

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 - 2 * x + 1 ≤ 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, Real.sin x + Real.cos x = Real.sqrt 2

-- Theorem stating that p is false and q is true
theorem p_false_and_q_true : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_p_false_and_q_true_l3326_332641


namespace NUMINAMATH_CALUDE_ellipse_equation_max_distance_max_distance_point_l3326_332672

/-- Definition of an ellipse with eccentricity 1/2 passing through (0, √3) -/
def Ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 - b^2) / a^2 = 1/4 ∧ b^2 = 3

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  Ellipse x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

/-- The maximum distance from a point on the ellipse to (0, √3) is 2√3 -/
theorem max_distance :
  ∃ (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  Real.sqrt (x₀^2 + (y₀ - Real.sqrt 3)^2) = 2 * Real.sqrt 3 :=
sorry

/-- The point that maximizes the distance has coordinates (-√3, 0) -/
theorem max_distance_point :
  ∃! (x₀ y₀ : ℝ), Ellipse x₀ y₀ ∧
  ∀ (x y : ℝ), Ellipse x y →
  (x₀^2 + (y₀ - Real.sqrt 3)^2) ≥ (x^2 + (y - Real.sqrt 3)^2) ∧
  x₀ = -Real.sqrt 3 ∧ y₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_max_distance_max_distance_point_l3326_332672


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3326_332675

/-- The perimeter of a rectangle formed when a smaller square is cut from the corner of a larger square -/
theorem rectangle_perimeter (t s : ℝ) (h : t > s) : 2 * s + 2 * (t - s) = 2 * t := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3326_332675


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l3326_332694

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_to_decimal :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l3326_332694


namespace NUMINAMATH_CALUDE_science_club_election_l3326_332619

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

theorem science_club_election (total : ℕ) (former : ℕ) (board_size : ℕ)
  (h1 : total = 18)
  (h2 : former = 8)
  (h3 : board_size = 6)
  (h4 : former ≤ total)
  (h5 : board_size ≤ total) :
  binomial total board_size - binomial (total - former) board_size = 18354 := by
  sorry

end NUMINAMATH_CALUDE_science_club_election_l3326_332619


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3326_332602

theorem max_books_borrowed (total_students : ℕ) (zero_book_students : ℕ) (one_book_students : ℕ) (two_book_students : ℕ) (avg_books : ℕ) :
  total_students = 40 →
  zero_book_students = 2 →
  one_book_students = 12 →
  two_book_students = 12 →
  avg_books = 2 →
  ∃ (max_books : ℕ),
    max_books = 5 ∧
    ∀ (student_books : ℕ),
      student_books ≤ max_books ∧
      (total_students * avg_books =
        0 * zero_book_students +
        1 * one_book_students +
        2 * two_book_students +
        (total_students - zero_book_students - one_book_students - two_book_students) * 3 +
        (max_books - 3)) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3326_332602


namespace NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l3326_332654

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, (a n : ℕ) * (a (n + 1) : ℕ) = (a (n + 2) : ℕ) * (a (n + 3) : ℕ)

/-- Definition of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ n ≥ N, a n = a (n + k)

/-- Theorem stating that a special sequence is eventually periodic -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ+) 
  (h : SpecialSequence a) : EventuallyPeriodic a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l3326_332654


namespace NUMINAMATH_CALUDE_marathon_debate_duration_in_minutes_l3326_332605

/-- Converts hours, minutes, and seconds to total minutes and rounds to the nearest whole number -/
def totalMinutesRounded (hours minutes seconds : ℕ) : ℕ :=
  let totalMinutes : ℚ := hours * 60 + minutes + seconds / 60
  (totalMinutes + 1/2).floor.toNat

/-- The marathon debate duration -/
def marathonDebateDuration : ℕ × ℕ × ℕ := (12, 15, 30)

theorem marathon_debate_duration_in_minutes :
  totalMinutesRounded marathonDebateDuration.1 marathonDebateDuration.2.1 marathonDebateDuration.2.2 = 736 := by
  sorry

end NUMINAMATH_CALUDE_marathon_debate_duration_in_minutes_l3326_332605


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3326_332690

/-- The number of people that can ride the Ferris wheel at the same time -/
def total_riders : ℕ := 4

/-- The number of people each seat can hold -/
def people_per_seat : ℕ := 2

/-- The number of seats on the Ferris wheel -/
def num_seats : ℕ := total_riders / people_per_seat

theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3326_332690


namespace NUMINAMATH_CALUDE_rectangle_area_l3326_332669

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 → 
  ratio = 3 → 
  (2 * r) * (ratio * 2 * r) = 432 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3326_332669


namespace NUMINAMATH_CALUDE_nancy_money_l3326_332691

def five_dollar_bills : ℕ := 9
def ten_dollar_bills : ℕ := 4
def one_dollar_bills : ℕ := 7

def total_money : ℕ := five_dollar_bills * 5 + ten_dollar_bills * 10 + one_dollar_bills * 1

theorem nancy_money : total_money = 92 := by
  sorry

end NUMINAMATH_CALUDE_nancy_money_l3326_332691


namespace NUMINAMATH_CALUDE_quadrilateral_division_theorem_l3326_332677

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  /-- The sum of internal angles is 360 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 360

/-- A diagonal of a quadrilateral -/
structure Diagonal (Q : ConvexQuadrilateral) where
  /-- The diagonal divides the quadrilateral into two triangles -/
  divides_into_triangles : Prop

/-- A triangle formed by a diagonal in a quadrilateral -/
structure Triangle (Q : ConvexQuadrilateral) (D : Diagonal Q) where
  /-- The sum of angles in the triangle is 180 degrees -/
  angle_sum : ℝ
  angle_sum_eq : angle_sum = 180

/-- Theorem: In a convex quadrilateral, it's impossible to divide it by a diagonal into two acute triangles, 
    while it's possible to divide it into two right triangles or two obtuse triangles -/
theorem quadrilateral_division_theorem (Q : ConvexQuadrilateral) :
  (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum < 180 ∧ T2.angle_sum < 180) → False
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum = 180 ∧ T2.angle_sum = 180)
  ∧ (∃ D : Diagonal Q, ∃ T1 T2 : Triangle Q D, 
    T1.angle_sum > 180 ∧ T2.angle_sum > 180) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_division_theorem_l3326_332677
