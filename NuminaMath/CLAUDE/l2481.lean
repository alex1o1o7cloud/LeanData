import Mathlib

namespace NUMINAMATH_CALUDE_twenty_people_handshakes_l2481_248199

/-- The number of handshakes when n people shake hands with each other exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 20 people, the total number of handshakes is 190. -/
theorem twenty_people_handshakes :
  handshakes 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_twenty_people_handshakes_l2481_248199


namespace NUMINAMATH_CALUDE_count_complementary_sets_l2481_248186

/-- Represents a card with four attributes -/
structure Card :=
  (shape : Fin 3)
  (color : Fin 3)
  (shade : Fin 3)
  (size : Fin 3)

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet := Finset Card

/-- Predicate for a complementary set -/
def is_complementary (s : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementary_sets : Finset ThreeCardSet := sorry

theorem count_complementary_sets :
  Finset.card complementary_sets = 6483 := by sorry

end NUMINAMATH_CALUDE_count_complementary_sets_l2481_248186


namespace NUMINAMATH_CALUDE_minimal_sequence_is_first_l2481_248104

/-- A sequence of n natural numbers -/
def Sequence (n : ℕ) := Fin n → ℕ

/-- Property: strictly decreasing -/
def IsStrictlyDecreasing (s : Sequence n) : Prop :=
  ∀ i j, i < j → s i > s j

/-- Property: no term divides any other term -/
def NoDivisibility (s : Sequence n) : Prop :=
  ∀ i j, i ≠ j → ¬(s i ∣ s j)

/-- Ordering relation between sequences -/
def Precedes (a b : Sequence n) : Prop :=
  ∃ k, (∀ i < k, a i = b i) ∧ a k < b k

/-- The proposed minimal sequence -/
def MinimalSequence (n : ℕ) : Sequence n :=
  λ i => 2 * n - 1 - 2 * i.val

theorem minimal_sequence_is_first (n : ℕ) :
  IsStrictlyDecreasing (MinimalSequence n) ∧
  NoDivisibility (MinimalSequence n) ∧
  (∀ s : Sequence n, IsStrictlyDecreasing s → NoDivisibility s →
    s = MinimalSequence n ∨ Precedes (MinimalSequence n) s) :=
sorry

end NUMINAMATH_CALUDE_minimal_sequence_is_first_l2481_248104


namespace NUMINAMATH_CALUDE_negative_three_to_fourth_power_l2481_248149

theorem negative_three_to_fourth_power :
  -3^4 = -(3 * 3 * 3 * 3) := by sorry

end NUMINAMATH_CALUDE_negative_three_to_fourth_power_l2481_248149


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l2481_248121

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l2481_248121


namespace NUMINAMATH_CALUDE_cube_split_theorem_l2481_248127

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) : 
  m^2 - m + 1 = 73 → m = 9 :=
by
  sorry

#check cube_split_theorem

end NUMINAMATH_CALUDE_cube_split_theorem_l2481_248127


namespace NUMINAMATH_CALUDE_square_root_positive_l2481_248192

theorem square_root_positive (x : ℝ) (h : x > 0) : Real.sqrt x > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_root_positive_l2481_248192


namespace NUMINAMATH_CALUDE_CD_distance_l2481_248115

-- Define the points on a line
variable (A B C D : ℝ)

-- Define the order of points on the line
axiom order : A ≤ B ∧ B ≤ C ∧ C ≤ D

-- Define the given distances
axiom AB_dist : B - A = 2
axiom AC_dist : C - A = 5
axiom BD_dist : D - B = 6

-- Theorem to prove
theorem CD_distance : D - C = 3 := by
  sorry

end NUMINAMATH_CALUDE_CD_distance_l2481_248115


namespace NUMINAMATH_CALUDE_min_value_of_expression_range_of_a_l2481_248102

theorem min_value_of_expression (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

theorem range_of_a (a : ℝ) : (∃ x > 1, a ≤ x + 1 / (x - 1)) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_range_of_a_l2481_248102


namespace NUMINAMATH_CALUDE_building_entry_exit_ways_l2481_248194

/-- The number of ways to enter and exit a building with 4 doors, entering and exiting through different doors -/
def number_of_ways (num_doors : ℕ) : ℕ :=
  num_doors * (num_doors - 1)

/-- Theorem stating that for a building with 4 doors, there are 12 ways to enter and exit through different doors -/
theorem building_entry_exit_ways :
  number_of_ways 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_building_entry_exit_ways_l2481_248194


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l2481_248136

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (h : edge_length = Real.sqrt 3) : 
  let cube_diagonal := Real.sqrt (3 * edge_length ^ 2)
  let sphere_radius := cube_diagonal / 2
  4 * Real.pi * sphere_radius ^ 2 = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l2481_248136


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2481_248110

/-- The remaining volume of a cube after removing two perpendicular cylindrical sections. -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_cylinder_radius : cylinder_radius = 1) :
  cube_side ^ 3 - 2 * π * cylinder_radius ^ 2 * cube_side = 216 - 12 * π := by
  sorry

#check remaining_cube_volume

end NUMINAMATH_CALUDE_remaining_cube_volume_l2481_248110


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l2481_248103

theorem sum_c_plus_d (a b c d : ℤ) 
  (h1 : a + b = 16) 
  (h2 : b + c = 9) 
  (h3 : a + d = 10) : 
  c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l2481_248103


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l2481_248159

/-- Calculates the amount of money John spent out of pocket to buy a computer and accessories after selling his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value discount_rate : ℝ) 
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (h4 : discount_rate = 0.2) : 
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
  sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l2481_248159


namespace NUMINAMATH_CALUDE_ellipse_properties_l2481_248116

/-- Represents an ellipse defined by the equation (x^2 / 36) + (y^2 / 9) = 4 -/
def Ellipse := {(x, y) : ℝ × ℝ | (x^2 / 36) + (y^2 / 9) = 4}

/-- The distance between the foci of the ellipse -/
def focalDistance (e : Set (ℝ × ℝ)) : ℝ := 
  5.196

/-- The eccentricity of the ellipse -/
def eccentricity (e : Set (ℝ × ℝ)) : ℝ := 
  0.866

theorem ellipse_properties : 
  focalDistance Ellipse = 5.196 ∧ eccentricity Ellipse = 0.866 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2481_248116


namespace NUMINAMATH_CALUDE_lcm_of_231_and_300_l2481_248134

theorem lcm_of_231_and_300 (lcm hcf : ℕ) (a b : ℕ) : 
  hcf = 30 → a = 231 → b = 300 → lcm * hcf = a * b → lcm = 2310 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_231_and_300_l2481_248134


namespace NUMINAMATH_CALUDE_roots_opposite_signs_l2481_248132

theorem roots_opposite_signs (n : ℝ) : 
  n^2 + n - 1 = 0 → 
  ∃ (x : ℝ), x ≠ 0 ∧ 
    (x^2 + (n-2)*x) / (2*n*x - 4) = (n+1) / (n-1) ∧
    (-x^2 + (n-2)*(-x)) / (2*n*(-x) - 4) = (n+1) / (n-1) := by
  sorry

end NUMINAMATH_CALUDE_roots_opposite_signs_l2481_248132


namespace NUMINAMATH_CALUDE_arrange_4_into_8_eq_1680_l2481_248141

/-- The number of ways to arrange 4 distinct objects into 8 distinct positions -/
def arrange_4_into_8 : ℕ := 8 * 7 * 6 * 5

/-- Theorem: The number of ways to arrange 4 distinct objects into 8 distinct positions is 1680 -/
theorem arrange_4_into_8_eq_1680 : arrange_4_into_8 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_arrange_4_into_8_eq_1680_l2481_248141


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l2481_248178

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l2481_248178


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2481_248162

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 2) = (a^2 - 3*a + 2) + Complex.I * (a - 2)) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2481_248162


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l2481_248131

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- Define the property of having exactly two roots
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂

-- Theorem statement
theorem two_roots_implies_c_values (c : ℝ) :
  has_exactly_two_roots (f c) → c = -2 ∨ c = 2 :=
sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l2481_248131


namespace NUMINAMATH_CALUDE_find_b_l2481_248128

theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 3 * x - 7)
  (hq : ∀ x, q x = 3 * x - b)
  (h : p (q 5) = 11) :
  b = 9 := by sorry

end NUMINAMATH_CALUDE_find_b_l2481_248128


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l2481_248195

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = (2*x - 1) / x :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((x + 1) / (x - 2) + 1) / ((x^2 - 2*x) / (x^2 - 4*x + 4)) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l2481_248195


namespace NUMINAMATH_CALUDE_investment_time_period_l2481_248166

theorem investment_time_period (P : ℝ) (rate_diff : ℝ) (interest_diff : ℝ) :
  P = 8400 →
  rate_diff = 0.05 →
  interest_diff = 840 →
  (P * rate_diff * 2 = interest_diff) := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l2481_248166


namespace NUMINAMATH_CALUDE_range_of_sin_cos_function_l2481_248147

theorem range_of_sin_cos_function : 
  ∀ x : ℝ, 3/4 ≤ Real.sin x ^ 4 + Real.cos x ^ 2 ∧ 
  Real.sin x ^ 4 + Real.cos x ^ 2 ≤ 1 ∧
  ∃ y z : ℝ, Real.sin y ^ 4 + Real.cos y ^ 2 = 3/4 ∧
            Real.sin z ^ 4 + Real.cos z ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sin_cos_function_l2481_248147


namespace NUMINAMATH_CALUDE_matthias_basketballs_l2481_248193

/-- Given information about Matthias' balls, prove the total number of basketballs --/
theorem matthias_basketballs 
  (total_soccer : ℕ)
  (soccer_with_holes : ℕ)
  (basketball_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : soccer_with_holes = 30)
  (h3 : basketball_with_holes = 7)
  (h4 : total_without_holes = 18)
  (h5 : total_without_holes = total_soccer - soccer_with_holes + (total_basketballs - basketball_with_holes)) :
  total_basketballs = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_matthias_basketballs_l2481_248193


namespace NUMINAMATH_CALUDE_inscribed_prism_lateral_area_l2481_248181

theorem inscribed_prism_lateral_area (sphere_surface_area : ℝ) (prism_height : ℝ) :
  sphere_surface_area = 24 * Real.pi →
  prism_height = 4 →
  ∃ (prism_lateral_area : ℝ),
    prism_lateral_area = 32 ∧
    prism_lateral_area = 4 * prism_height * (Real.sqrt ((4 * sphere_surface_area / Real.pi) / 4 - prism_height^2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_prism_lateral_area_l2481_248181


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2481_248197

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) ∧
  (∀ (m : ℝ), (∀ (a b : ℝ), a + b = 5 →
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 = m) →
  m = 441/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2481_248197


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l2481_248143

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total - (total * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total / 3) / 6 = (total / 3) - ((total / 3) * 5 / 6))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) - ((total * 2 / 3) * 2 / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : (total * 2 / 3) * 2 / 3 = 400)  -- 400 sculptures are not on display
  : total = 900 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l2481_248143


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l2481_248198

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes for assigning 3 people to 7 communities with at most 2 people per community. -/
def totalAllocationSchemes : ℕ := A 7 3 + C 3 2 * C 1 1 * A 7 2

theorem allocation_schemes_count :
  totalAllocationSchemes = 336 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l2481_248198


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2481_248184

theorem quadratic_equation_roots (m : ℤ) :
  (∃ a b : ℕ+, a ≠ b ∧ 
    (a : ℝ)^2 + m * (a : ℝ) - m + 1 = 0 ∧
    (b : ℝ)^2 + m * (b : ℝ) - m + 1 = 0) →
  m = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2481_248184


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_and_minimum_value_l2481_248118

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) + (x^2 + a*x - 1)) * Real.exp (x - 1)

theorem extremum_point_implies_a_and_minimum_value 
  (a : ℝ) 
  (h1 : f_derivative a (-2) = 0) :
  a = -1 ∧ ∀ x, f (-1) x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_and_minimum_value_l2481_248118


namespace NUMINAMATH_CALUDE_pizza_both_toppings_l2481_248111

/-- Represents a pizza with cheese and olive toppings -/
structure Pizza where
  total_slices : ℕ
  cheese_slices : ℕ
  olive_slices : ℕ
  both_toppings : ℕ

/-- Theorem: Given the conditions, prove that 7 slices have both cheese and olives -/
theorem pizza_both_toppings (p : Pizza) 
  (h1 : p.total_slices = 24)
  (h2 : p.cheese_slices = 15)
  (h3 : p.olive_slices = 16)
  (h4 : p.total_slices = p.both_toppings + (p.cheese_slices - p.both_toppings) + (p.olive_slices - p.both_toppings)) :
  p.both_toppings = 7 := by
  sorry

#check pizza_both_toppings

end NUMINAMATH_CALUDE_pizza_both_toppings_l2481_248111


namespace NUMINAMATH_CALUDE_min_days_for_eleven_groups_l2481_248140

/-- A Festival is a collection of daily performances where groups either perform or watch --/
structure Festival (n : ℕ) where
  days : ℕ
  schedule : Fin days → Finset (Fin n)
  watched : Fin n → Fin n → Prop

/-- A Festival is valid if every group watches every other group at least once --/
def ValidFestival (f : Festival n) : Prop :=
  ∀ i j, i ≠ j → f.watched i j

/-- The minimum number of days required for a valid festival with 11 groups is 6 --/
theorem min_days_for_eleven_groups :
  (∃ f : Festival 11, ValidFestival f ∧ f.days = 6) ∧
  (∀ f : Festival 11, ValidFestival f → f.days ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_min_days_for_eleven_groups_l2481_248140


namespace NUMINAMATH_CALUDE_intersection_points_l2481_248165

/-- The set of possible values for k such that the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point -/
def possible_k_values : Set ℝ :=
  {k : ℝ | k = 1.5 ∨ k = 6}

/-- The condition that |z - 3| = 3|z + 3| -/
def condition (z : ℂ) : Prop :=
  Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

/-- The theorem stating that the only values of k for which the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point are 1.5 and 6 -/
theorem intersection_points (k : ℝ) :
  (∃! z : ℂ, condition z ∧ Complex.abs z = k) ↔ k ∈ possible_k_values := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_l2481_248165


namespace NUMINAMATH_CALUDE_mod_inverse_sum_five_l2481_248117

theorem mod_inverse_sum_five : ∃ x y : ℤ,
  (x * 5) % 31 = 1 ∧
  (y * 25) % 31 = 1 ∧
  (x + y) % 31 = 26 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_five_l2481_248117


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_min_value_f_range_of_a_l2481_248100

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 2|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≥ M := by sorry

-- Theorem 3: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → 2^x + a ≥ 2) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_min_value_f_range_of_a_l2481_248100


namespace NUMINAMATH_CALUDE_system_solution_l2481_248172

theorem system_solution (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  (∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ ∧ x = 8 ∧ y = 5) →
  (∃ x y : ℝ, 4 * a₁ * x - 5 * b₁ * y = 3 * c₁ ∧ 4 * a₂ * x - 5 * b₂ * y = 3 * c₂ ∧ x = 6 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2481_248172


namespace NUMINAMATH_CALUDE_opposite_of_negative_mixed_number_l2481_248146

theorem opposite_of_negative_mixed_number : 
  -(-(7/4)) = 7/4 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_mixed_number_l2481_248146


namespace NUMINAMATH_CALUDE_red_bellies_percentage_l2481_248183

/-- Represents the total number of minnows in the pond -/
def total_minnows : ℕ := 50

/-- Represents the number of minnows with red bellies -/
def red_bellies : ℕ := 20

/-- Represents the number of minnows with white bellies -/
def white_bellies : ℕ := 15

/-- Represents the percentage of minnows with green bellies -/
def green_bellies_percent : ℚ := 30 / 100

/-- Theorem stating that the percentage of minnows with red bellies is 40% -/
theorem red_bellies_percentage :
  (red_bellies : ℚ) / total_minnows * 100 = 40 := by
  sorry

/-- Lemma verifying that the total number of minnows is correct -/
lemma total_minnows_check :
  total_minnows = red_bellies + white_bellies + (green_bellies_percent * total_minnows) := by
  sorry

end NUMINAMATH_CALUDE_red_bellies_percentage_l2481_248183


namespace NUMINAMATH_CALUDE_median_and_midpoint_lengths_l2481_248169

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  -- The length of side XY
  xy : ℝ
  -- The length of side YZ
  yz : ℝ
  -- The point W on side YZ
  w : ℝ
  -- Condition: XY = 6
  xy_eq : xy = 6
  -- Condition: YZ = 8
  yz_eq : yz = 8
  -- Condition: W is the midpoint of YZ
  w_midpoint : w = yz / 2

/-- The length of XW is 5 and the length of WZ is 4 in the given right triangle -/
theorem median_and_midpoint_lengths (t : RightTriangleWithMedian) : 
  Real.sqrt (t.xy^2 + (t.yz - t.w)^2) = 5 ∧ t.w = 4 := by
  sorry


end NUMINAMATH_CALUDE_median_and_midpoint_lengths_l2481_248169


namespace NUMINAMATH_CALUDE_refrigerator_cash_savings_l2481_248182

/-- Calculates the savings when buying a refrigerator with cash instead of installments. -/
theorem refrigerator_cash_savings 
  (cash_price : ℕ) 
  (deposit : ℕ) 
  (num_installments : ℕ) 
  (installment_amount : ℕ) 
  (h1 : cash_price = 8000)
  (h2 : deposit = 3000)
  (h3 : num_installments = 30)
  (h4 : installment_amount = 300) :
  deposit + num_installments * installment_amount - cash_price = 4000 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_cash_savings_l2481_248182


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_l2481_248148

/-- The quadratic equation z^2 + (12 + ci)z + (45 + di) = 0 has complex conjugate roots if and only if c = 0 and d = 0 -/
theorem complex_conjugate_roots (c d : ℝ) : 
  (∀ z : ℂ, z^2 + (12 + c * Complex.I) * z + (45 + d * Complex.I) = 0 → 
    ∃ x y : ℝ, z = x + y * Complex.I ∧ x - y * Complex.I ∈ {w : ℂ | w^2 + (12 + c * Complex.I) * w + (45 + d * Complex.I) = 0}) ↔ 
  c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_l2481_248148


namespace NUMINAMATH_CALUDE_chord_existence_l2481_248168

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16
def line (x y : ℝ) : Prop := y = x + 1

-- Theorem statement
theorem chord_existence :
  ∃ (length : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ ∧ line x₂ y₂ ∧
    length = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_existence_l2481_248168


namespace NUMINAMATH_CALUDE_avg_speed_BC_l2481_248156

/-- Represents the journey of a motorcyclist --/
structure Journey where
  distanceAB : ℝ
  distanceBC : ℝ
  timeAB : ℝ
  timeBC : ℝ
  avgSpeedTotal : ℝ

/-- Theorem stating the average speed from B to C given the journey conditions --/
theorem avg_speed_BC (j : Journey)
  (h1 : j.distanceAB = 120)
  (h2 : j.distanceBC = j.distanceAB / 2)
  (h3 : j.timeAB = 3 * j.timeBC)
  (h4 : j.avgSpeedTotal = 20)
  (h5 : j.avgSpeedTotal = (j.distanceAB + j.distanceBC) / (j.timeAB + j.timeBC)) :
  j.distanceBC / j.timeBC = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_avg_speed_BC_l2481_248156


namespace NUMINAMATH_CALUDE_winter_wheat_harvest_scientific_notation_l2481_248135

theorem winter_wheat_harvest_scientific_notation :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_winter_wheat_harvest_scientific_notation_l2481_248135


namespace NUMINAMATH_CALUDE_slipper_cost_l2481_248106

/-- Calculates the total cost of a pair of embroidered slippers with shipping --/
theorem slipper_cost (original_price discount_percentage embroidery_cost_per_shoe shipping_cost : ℚ) :
  original_price = 50 →
  discount_percentage = 10 →
  embroidery_cost_per_shoe = (11/2) →
  shipping_cost = 10 →
  original_price * (1 - discount_percentage / 100) + 2 * embroidery_cost_per_shoe + shipping_cost = 66 := by
sorry


end NUMINAMATH_CALUDE_slipper_cost_l2481_248106


namespace NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_prob_l2481_248133

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The probability of no two adjacent people standing up in a circular arrangement of n people, each flipping a fair coin. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n : ℚ)

theorem ten_people_no_adjacent_standing_prob :
  noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_no_adjacent_standing_prob_l2481_248133


namespace NUMINAMATH_CALUDE_units_digit_37_power_37_l2481_248174

theorem units_digit_37_power_37 : 37^37 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_37_power_37_l2481_248174


namespace NUMINAMATH_CALUDE_circle_equation_l2481_248154

/-- Given a circle with center at (a,1) tangent to two lines, prove its standard equation -/
theorem circle_equation (a : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x y : ℝ), (2*x - y + 4 = 0 ∨ 2*x - y - 6 = 0) → 
      ((x - a)^2 + (y - 1)^2 = r^2))) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2481_248154


namespace NUMINAMATH_CALUDE_runner_speeds_l2481_248123

/-- The speed of runner A in meters per second -/
def speed_A : ℝ := 9

/-- The speed of runner B in meters per second -/
def speed_B : ℝ := 7

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The time in seconds it takes for A and B to meet when running in opposite directions -/
def opposite_meeting_time : ℝ := 25

/-- The time in seconds it takes for A to catch up with B when running in the same direction -/
def same_direction_catchup_time : ℝ := 200

theorem runner_speeds :
  speed_A * opposite_meeting_time + speed_B * opposite_meeting_time = track_length ∧
  speed_A * same_direction_catchup_time - speed_B * same_direction_catchup_time = track_length :=
by sorry

end NUMINAMATH_CALUDE_runner_speeds_l2481_248123


namespace NUMINAMATH_CALUDE_grade10_sample_size_l2481_248151

/-- Calculates the number of students to be selected from a specific grade in a stratified random sample. -/
def stratifiedSampleSize (gradeSize : ℕ) (totalSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeSize * sampleSize) / totalSize

/-- The number of students to be selected from grade 10 in a stratified random sample is 40. -/
theorem grade10_sample_size :
  stratifiedSampleSize 1200 3000 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_grade10_sample_size_l2481_248151


namespace NUMINAMATH_CALUDE_f_range_l2481_248112

noncomputable def f (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 6) / (Real.sin x - 1)

theorem f_range : 
  ∀ (y : ℝ), (∃ (x : ℝ), Real.sin x ≠ 1 ∧ f x = y) ↔ (0 ≤ y ∧ y < 8) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2481_248112


namespace NUMINAMATH_CALUDE_solve_candy_problem_l2481_248160

def candy_problem (packs : ℕ) (paid : ℕ) (change : ℕ) : Prop :=
  packs = 3 ∧ paid = 20 ∧ change = 11 →
  (paid - change) / packs = 3

theorem solve_candy_problem : candy_problem 3 20 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l2481_248160


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2481_248107

/-- Given that P(5, 9) is the midpoint of segment CD and C has coordinates (11, 5),
    prove that the sum of the coordinates of D is 12. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) :
  C = (11, 5) →
  (5, 9) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2481_248107


namespace NUMINAMATH_CALUDE_line_through_points_l2481_248161

theorem line_through_points (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, (b / a : ℝ) = n) (h4 : ∃ m : ℝ, ∀ x y : ℝ, y = k * x + m → (x = a ∧ y = a) ∨ (x = b ∧ y = 8 * b)) :
  k = 9 ∨ k = 15 :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l2481_248161


namespace NUMINAMATH_CALUDE_three_numbers_average_l2481_248122

theorem three_numbers_average (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  y = 90 →
  (x + y + z) / 3 = 165 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_average_l2481_248122


namespace NUMINAMATH_CALUDE_stationery_cost_l2481_248170

theorem stationery_cost (x y : ℚ) 
  (h1 : 2 * x + 3 * y = 18) 
  (h2 : 3 * x + 2 * y = 22) : 
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l2481_248170


namespace NUMINAMATH_CALUDE_smallest_number_inequality_l2481_248144

theorem smallest_number_inequality (x y z m : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : 
  m ≤ x * y^2 * z^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_inequality_l2481_248144


namespace NUMINAMATH_CALUDE_complex_roots_l2481_248176

theorem complex_roots (a' b' c' d' k' : ℂ) 
  (h1 : a' * k' ^ 2 + b' * k' + c' = 0)
  (h2 : b' * k' ^ 2 + c' * k' + d' = 0)
  (h3 : d' = a')
  (h4 : k' ≠ 0) :
  k' = 1 ∨ k' = (-1 + Complex.I * Real.sqrt 3) / 2 ∨ k' = (-1 - Complex.I * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_l2481_248176


namespace NUMINAMATH_CALUDE_zhang_bing_age_problem_l2481_248177

def current_year : ℕ := 2023  -- Assuming current year is 2023

def birth_year : ℕ := 1953

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem zhang_bing_age_problem :
  ∃! x : ℕ, 
    birth_year < x ∧ 
    x ≤ current_year ∧
    (x - birth_year) % 9 = 0 ∧
    x - birth_year = sum_of_digits x ∧
    x - birth_year = 18 := by
  sorry

end NUMINAMATH_CALUDE_zhang_bing_age_problem_l2481_248177


namespace NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l2481_248189

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Converts meters to centimeters -/
def metersToCentimeters (m : ℕ) : ℕ := m * 100

/-- The dimensions of the large wooden box in meters -/
def largeBoxDimMeters : BoxDimensions := {
  length := 4,
  width := 2,
  height := 4
}

/-- The dimensions of the large wooden box in centimeters -/
def largeBoxDimCm : BoxDimensions := {
  length := metersToCentimeters largeBoxDimMeters.length,
  width := metersToCentimeters largeBoxDimMeters.width,
  height := metersToCentimeters largeBoxDimMeters.height
}

/-- The dimensions of the small rectangular box in centimeters -/
def smallBoxDimCm : BoxDimensions := {
  length := 4,
  width := 2,
  height := 2
}

theorem max_small_boxes_in_large_box :
  (boxVolume largeBoxDimCm) / (boxVolume smallBoxDimCm) = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_max_small_boxes_in_large_box_l2481_248189


namespace NUMINAMATH_CALUDE_smallest_root_quadratic_l2481_248142

theorem smallest_root_quadratic (x : ℝ) :
  (9 * x^2 - 45 * x + 50 = 0) →
  (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) →
  x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_quadratic_l2481_248142


namespace NUMINAMATH_CALUDE_beautiful_point_coordinates_l2481_248153

/-- A point (x,y) is called a "beautiful point" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x,y) from the y-axis is the absolute value of x -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
sorry

end NUMINAMATH_CALUDE_beautiful_point_coordinates_l2481_248153


namespace NUMINAMATH_CALUDE_original_loaf_size_l2481_248125

def slices_per_sandwich : ℕ := 2
def days_with_one_sandwich : ℕ := 5
def sandwiches_on_saturday : ℕ := 2
def slices_left : ℕ := 6

theorem original_loaf_size :
  slices_per_sandwich * days_with_one_sandwich +
  slices_per_sandwich * sandwiches_on_saturday +
  slices_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_loaf_size_l2481_248125


namespace NUMINAMATH_CALUDE_remainder_790123_div_15_l2481_248120

theorem remainder_790123_div_15 : 790123 % 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_790123_div_15_l2481_248120


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2481_248109

theorem trigonometric_identity (α : Real) 
  (h : (Real.sin (11 * Real.pi - α) - Real.cos (-α)) / Real.cos ((7 * Real.pi / 2) + α) = 3) : 
  (Real.tan α = -1/2) ∧ (Real.sin (2*α) + Real.cos (2*α) = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2481_248109


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l2481_248158

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l2481_248158


namespace NUMINAMATH_CALUDE_tickets_sold_and_given_away_l2481_248126

theorem tickets_sold_and_given_away (initial_tickets : ℕ) (h : initial_tickets = 5760) :
  let sold_tickets := initial_tickets / 2
  let remaining_tickets := initial_tickets - sold_tickets
  let given_away_tickets := remaining_tickets / 4
  sold_tickets + given_away_tickets = 3600 :=
by sorry

end NUMINAMATH_CALUDE_tickets_sold_and_given_away_l2481_248126


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2481_248173

theorem points_three_units_from_negative_two :
  ∃! (S : Set ℝ), (∀ x ∈ S, |x - (-2)| = 3) ∧ S = {-5, 1} := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2481_248173


namespace NUMINAMATH_CALUDE_exists_complementary_not_acute_not_obtuse_l2481_248130

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 180

-- Define acute angle
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

-- Define obtuse angle
def obtuse (a : ℝ) : Prop := 90 < a ∧ a < 180

-- Theorem statement
theorem exists_complementary_not_acute_not_obtuse :
  ∃ (a b : ℝ), complementary a b ∧ ¬(acute a ∨ obtuse a) ∧ ¬(acute b ∨ obtuse b) :=
sorry

end NUMINAMATH_CALUDE_exists_complementary_not_acute_not_obtuse_l2481_248130


namespace NUMINAMATH_CALUDE_intersection_implies_value_l2481_248105

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, a^2+1, 2*a-1}
  (A ∩ B = {-3}) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_value_l2481_248105


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2481_248191

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 6) → (abs b = 4) → (a * b < 0) → (a + b = 2 ∨ a + b = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l2481_248191


namespace NUMINAMATH_CALUDE_curve_is_parabola_l2481_248163

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) = |3*x - 4*y + 2| / 5

-- Define the fixed point F
def F : ℝ × ℝ := (2, 0)

-- Define the line
def line (x y : ℝ) : Prop :=
  3*x - 4*y + 2 = 0

-- Theorem statement
theorem curve_is_parabola :
  ∃ (f : ℝ × ℝ) (l : ℝ → ℝ → Prop),
    (∀ x y, curve x y ↔ 
      (Real.sqrt ((x - f.1)^2 + (y - f.2)^2) = 
       Real.sqrt ((3*x - 4*y + 2)^2) / 5)) ∧
    (f = F) ∧
    (∀ x y, l x y ↔ line x y) ∧
    (¬ l F.1 F.2) :=
  sorry

end NUMINAMATH_CALUDE_curve_is_parabola_l2481_248163


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2481_248145

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a + b + c = 21) : 
  a*b + b*c + a*c = 100 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2481_248145


namespace NUMINAMATH_CALUDE_polynomial_degree_is_12_l2481_248185

/-- The degree of a polynomial (x^5 + ax^8 + bx^2 + c)(y^3 + dy^2 + e)(z + f) -/
def polynomial_degree (a b c d e f : ℝ) : ℕ :=
  let p1 := fun (x : ℝ) => x^5 + a*x^8 + b*x^2 + c
  let p2 := fun (y : ℝ) => y^3 + d*y^2 + e
  let p3 := fun (z : ℝ) => z + f
  let product := fun (x y z : ℝ) => p1 x * p2 y * p3 z
  12

theorem polynomial_degree_is_12 (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
    polynomial_degree a b c d e f = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_is_12_l2481_248185


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_l2481_248187

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 2

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 9*t + 5

/-- The curve crosses itself if there exist two distinct real numbers that yield the same point -/
def curve_crosses_itself : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (7, 5)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  curve_crosses_itself ∧ ∃ t : ℝ, (x t, y t) = crossing_point :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_l2481_248187


namespace NUMINAMATH_CALUDE_root_equation_n_value_l2481_248167

theorem root_equation_n_value : 
  ∀ n : ℝ, (1 : ℝ)^2 + 3*(1 : ℝ) + n = 0 → n = -4 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_n_value_l2481_248167


namespace NUMINAMATH_CALUDE_bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l2481_248171

/-- Bernoulli random variable -/
structure BernoulliRV where
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Joint distribution of two Bernoulli random variables -/
structure JointDistribution (X Y : BernoulliRV) where
  pxy : ℝ × ℝ → ℝ
  sum_to_one : (pxy (0, 0)) + (pxy (0, 1)) + (pxy (1, 0)) + (pxy (1, 1)) = 1

/-- Covariance of two Bernoulli random variables -/
def cov (X Y : BernoulliRV) : ℝ := sorry

/-- Main theorem -/
theorem bernoulli_joint_distribution_theorem (X Y : BernoulliRV) :
  ∃! (j : JointDistribution X Y),
    (j.pxy (1, 1) = cov X Y + X.p * Y.p) ∧
    (j.pxy (0, 1) = Y.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (1, 0) = X.p - (cov X Y + X.p * Y.p)) ∧
    (j.pxy (0, 0) = 1 - X.p - Y.p + (cov X Y + X.p * Y.p)) :=
  sorry

/-- Independence theorem -/
theorem bernoulli_independence_theorem (X Y : BernoulliRV) (j : JointDistribution X Y) :
  (∀ x y, j.pxy (x, y) = (if x = 1 then X.p else 1 - X.p) * (if y = 1 then Y.p else 1 - Y.p)) ↔
  cov X Y = 0 :=
  sorry

end NUMINAMATH_CALUDE_bernoulli_joint_distribution_theorem_bernoulli_independence_theorem_l2481_248171


namespace NUMINAMATH_CALUDE_adam_bought_26_books_l2481_248180

/-- Represents Adam's bookcase and book shopping scenario -/
structure Bookcase where
  shelves : Nat
  booksPerShelf : Nat
  initialBooks : Nat
  leftoverBooks : Nat

/-- Calculates the number of books Adam bought -/
def booksBought (b : Bookcase) : Nat :=
  b.shelves * b.booksPerShelf + b.leftoverBooks - b.initialBooks

/-- Theorem stating that Adam bought 26 books -/
theorem adam_bought_26_books (b : Bookcase) 
    (h1 : b.shelves = 4)
    (h2 : b.booksPerShelf = 20)
    (h3 : b.initialBooks = 56)
    (h4 : b.leftoverBooks = 2) : 
  booksBought b = 26 := by
  sorry

end NUMINAMATH_CALUDE_adam_bought_26_books_l2481_248180


namespace NUMINAMATH_CALUDE_grid_paths_6x5_10_l2481_248108

/-- The number of different paths on a grid --/
def grid_paths (width height path_length : ℕ) : ℕ :=
  Nat.choose path_length height

/-- Theorem: The number of different paths on a 6x5 grid with path length 10 is 210 --/
theorem grid_paths_6x5_10 :
  grid_paths 6 5 10 = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6x5_10_l2481_248108


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l2481_248139

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l2481_248139


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l2481_248114

theorem product_of_four_consecutive_integers (n : ℤ) :
  ∃ M : ℤ, 
    Even M ∧ 
    (n - 1) * n * (n + 1) * (n + 2) = (M - 2) * M := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l2481_248114


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l2481_248155

/-- Given two terms 3x^m*y and -5x^2*y^n that are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^m * y = -5 * x^2 * y^n) → m + n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l2481_248155


namespace NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l2481_248175

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 8th term of a geometric sequence given the 4th and 6th terms -/
theorem eighth_term_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_4 : a 4 = 7)
  (h_6 : a 6 = 21) : 
  a 8 = 63 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l2481_248175


namespace NUMINAMATH_CALUDE_tom_initial_balloons_l2481_248164

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- The initial number of balloons Tom had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem tom_initial_balloons : initial_balloons = 30 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_balloons_l2481_248164


namespace NUMINAMATH_CALUDE_log_equation_solution_l2481_248196

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ log y 81 = 4/2 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2481_248196


namespace NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l2481_248138

theorem inscribed_circles_area_ratio (s : ℝ) (h : s > 0) : 
  let square_side := s
  let large_circle_radius := s / 2
  let triangle_side := s * (Real.sqrt 3) / 2
  let small_circle_radius := s * (Real.sqrt 3) / 12
  (π * (small_circle_radius ^ 2)) / (square_side ^ 2) = π / 48 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_ratio_l2481_248138


namespace NUMINAMATH_CALUDE_average_age_of_nine_students_l2481_248129

theorem average_age_of_nine_students 
  (total_students : ℕ)
  (total_average : ℚ)
  (five_students : ℕ)
  (five_average : ℚ)
  (fifteenth_student_age : ℕ)
  (h1 : total_students = 15)
  (h2 : total_average = 15)
  (h3 : five_students = 5)
  (h4 : five_average = 13)
  (h5 : fifteenth_student_age = 16) :
  (total_students * total_average - five_students * five_average - fifteenth_student_age) / (total_students - five_students - 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_nine_students_l2481_248129


namespace NUMINAMATH_CALUDE_binary_101101_conversion_l2481_248157

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_101101_conversion :
  let binary := [true, false, true, true, false, true]
  (binary_to_decimal binary = 45) ∧
  (decimal_to_base7 (binary_to_decimal binary) = [6, 3]) := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_conversion_l2481_248157


namespace NUMINAMATH_CALUDE_complex_square_sum_l2481_248150

theorem complex_square_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a + b * i) ^ 2 = 3 + 4 * i → a ^ 2 + b ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l2481_248150


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2481_248152

theorem geometric_sequence_sum (a r : ℝ) : 
  (a + a * r = 15) →
  (a * (1 - r^6) / (1 - r) = 195) →
  (a * (1 - r^4) / (1 - r) = 82) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2481_248152


namespace NUMINAMATH_CALUDE_gcd_f_x_eq_one_l2481_248190

def f (x : ℤ) : ℤ := (3*x+4)*(8*x+5)*(15*x+11)*(x+14)

theorem gcd_f_x_eq_one (x : ℤ) (h : ∃ k : ℤ, x = 54321 * k) :
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_f_x_eq_one_l2481_248190


namespace NUMINAMATH_CALUDE_convex_figure_integer_points_l2481_248119

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't need to define the structure fully, just declare it
  dummy : Unit

/-- The area of a convex figure -/
noncomputable def area (φ : ConvexFigure) : ℝ := sorry

/-- The semiperimeter of a convex figure -/
noncomputable def semiperimeter (φ : ConvexFigure) : ℝ := sorry

/-- The number of integer points contained in a convex figure -/
noncomputable def integerPoints (φ : ConvexFigure) : ℕ := sorry

/-- 
If the area of a convex figure is greater than n times its semiperimeter,
then it contains at least n integer points.
-/
theorem convex_figure_integer_points (φ : ConvexFigure) (n : ℕ) :
  area φ > n • (semiperimeter φ) → integerPoints φ ≥ n := by sorry

end NUMINAMATH_CALUDE_convex_figure_integer_points_l2481_248119


namespace NUMINAMATH_CALUDE_sequence_properties_l2481_248137

/-- Geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- Arithmetic sequence with positive common difference -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ ∀ n, b (n + 1) = b n + d

theorem sequence_properties (a b : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_arith : arithmetic_sequence b)
  (h_eq3 : a 3 = b 3)
  (h_eq7 : a 7 = b 7) :
  a 5 < b 5 ∧ a 1 > b 1 ∧ a 9 > b 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2481_248137


namespace NUMINAMATH_CALUDE_number_calculation_l2481_248179

theorem number_calculation (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2481_248179


namespace NUMINAMATH_CALUDE_average_b_c_l2481_248188

theorem average_b_c (a b c : ℝ) 
  (h1 : (a + b) / 2 = 80) 
  (h2 : a - c = 200) : 
  (b + c) / 2 = -20 := by
sorry

end NUMINAMATH_CALUDE_average_b_c_l2481_248188


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2481_248113

theorem simplify_and_evaluate : 
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2481_248113


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2481_248101

theorem min_sum_of_product (a b : ℤ) : 
  a ≤ 0 → b ≤ 0 → a * b = 144 → (∀ x y : ℤ, x ≤ 0 → y ≤ 0 → x * y = 144 → a + b ≤ x + y) → a + b = -30 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2481_248101


namespace NUMINAMATH_CALUDE_some_number_in_formula_l2481_248124

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (x : ℕ) (n : ℚ) : ℚ := 2.5 + 0.5 * (x - n)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℚ := 4

theorem some_number_in_formula : 
  ∃ n : ℚ, toll_formula axles_18_wheel_truck n = toll_18_wheel_truck ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_some_number_in_formula_l2481_248124
