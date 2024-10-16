import Mathlib

namespace NUMINAMATH_CALUDE_milk_problem_l840_84012

theorem milk_problem (initial_milk : ℚ) (given_milk : ℚ) (result : ℚ) : 
  initial_milk = 4 →
  given_milk = 16/3 →
  result = initial_milk - given_milk →
  result = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_milk_problem_l840_84012


namespace NUMINAMATH_CALUDE_lisa_eggs_for_husband_l840_84009

/-- The number of eggs Lisa makes for her husband in a year -/
def eggs_for_husband (
  days_per_week : ℕ)
  (weeks_per_year : ℕ)
  (children : ℕ)
  (eggs_per_child : ℕ)
  (eggs_for_self : ℕ)
  (total_eggs_per_year : ℕ) : ℕ :=
  total_eggs_per_year - (days_per_week * weeks_per_year * (children * eggs_per_child + eggs_for_self))

/-- Theorem stating that Lisa makes 780 eggs for her husband in a year -/
theorem lisa_eggs_for_husband :
  eggs_for_husband 5 52 4 2 2 3380 = 780 := by
  sorry

end NUMINAMATH_CALUDE_lisa_eggs_for_husband_l840_84009


namespace NUMINAMATH_CALUDE_geometric_sum_specific_l840_84002

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_specific :
  geometric_sum (3/4) (3/4) 12 = 48758625/16777216 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_specific_l840_84002


namespace NUMINAMATH_CALUDE_chocos_remainder_l840_84077

theorem chocos_remainder (n : ℕ) (h : n % 11 = 5) : (4 * n) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chocos_remainder_l840_84077


namespace NUMINAMATH_CALUDE_exponent_operations_l840_84085

theorem exponent_operations (x : ℝ) (x_nonzero : x ≠ 0) :
  (x^2 * x^3 = x^5) ∧
  (x^2 + x^3 ≠ x^5) ∧
  ((x^3)^2 ≠ x^5) ∧
  (x^15 / x^3 ≠ x^5) :=
by sorry

end NUMINAMATH_CALUDE_exponent_operations_l840_84085


namespace NUMINAMATH_CALUDE_kim_shirts_l840_84014

theorem kim_shirts (D : ℕ) : 
  (2 / 3 : ℚ) * (12 * D) = 32 → D = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_l840_84014


namespace NUMINAMATH_CALUDE_book_sale_revenue_l840_84057

theorem book_sale_revenue (total_books : ℕ) (sold_books : ℕ) (price_per_book : ℕ) 
  (h1 : sold_books = (2 * total_books) / 3)
  (h2 : total_books - sold_books = 36)
  (h3 : price_per_book = 4) :
  sold_books * price_per_book = 288 := by
sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l840_84057


namespace NUMINAMATH_CALUDE_fraction_simplification_l840_84055

theorem fraction_simplification (x : ℝ) : 
  (2*x - 3)/4 + (3*x + 5)/5 - (x - 1)/2 = (12*x + 15)/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l840_84055


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l840_84082

/-- The probability of a specific pairing in a classroom with random pairings -/
theorem specific_pairing_probability 
  (n : ℕ) -- Total number of students
  (h : n = 32) -- Given number of students in the classroom
  : (1 : ℚ) / (n - 1 : ℚ) = 1 / 31 := by
  sorry

#check specific_pairing_probability

end NUMINAMATH_CALUDE_specific_pairing_probability_l840_84082


namespace NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l840_84061

/-- Represents a convex polyhedron with specific face types -/
structure SpecialPolyhedron where
  square_faces : ℕ
  hexagon_faces : ℕ
  octagon_faces : ℕ
  vertex_configuration : Bool  -- True if each vertex meets one square, one hexagon, and one octagon

/-- Calculates the number of interior segments in the special polyhedron -/
def interior_segments (p : SpecialPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of interior segments in the specific polyhedron -/
theorem special_polyhedron_interior_segments :
  let p : SpecialPolyhedron := {
    square_faces := 12,
    hexagon_faces := 8,
    octagon_faces := 6,
    vertex_configuration := true
  }
  interior_segments p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_interior_segments_l840_84061


namespace NUMINAMATH_CALUDE_happy_children_count_l840_84096

theorem happy_children_count (total_children : ℕ) 
                              (sad_children : ℕ) 
                              (neutral_children : ℕ) 
                              (total_boys : ℕ) 
                              (total_girls : ℕ) 
                              (happy_boys : ℕ) 
                              (sad_girls : ℕ) :
  total_children = 60 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 18 →
  total_girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  ∃ (happy_children : ℕ), 
    happy_children = 30 ∧
    happy_children + sad_children + neutral_children = total_children ∧
    happy_boys + (sad_children - sad_girls) + (neutral_children - (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls)))) = total_boys ∧
    (happy_children - happy_boys) + sad_girls + (neutral_children - (total_boys - happy_boys - (sad_children - sad_girls))) = total_girls :=
by
  sorry


end NUMINAMATH_CALUDE_happy_children_count_l840_84096


namespace NUMINAMATH_CALUDE_rogers_remaining_years_l840_84098

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions of the coworkers' experience -/
def valid_experience (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- Roger's retirement years -/
def retirement_years : ℕ := 50

/-- Theorem stating that Roger needs to work 8 more years before retirement -/
theorem rogers_remaining_years (e : Experience) (h : valid_experience e) :
  retirement_years - e.roger = 8 := by
  sorry


end NUMINAMATH_CALUDE_rogers_remaining_years_l840_84098


namespace NUMINAMATH_CALUDE_color_copies_proof_l840_84072

/-- The price per color copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per color copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The difference in total charge between print shop Y and X -/
def charge_difference : ℚ := 120

/-- The number of color copies -/
def num_copies : ℚ := 80

theorem color_copies_proof :
  price_y * num_copies = price_x * num_copies + charge_difference :=
by sorry

end NUMINAMATH_CALUDE_color_copies_proof_l840_84072


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l840_84099

-- Define the triangle XYZ
def triangle_XYZ (X Y Z : ℝ × ℝ) : Prop :=
  let d := λ a b : ℝ × ℝ => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  d X Z = 13 ∧ d X Y = 12 ∧ (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0

-- Define the theorem
theorem sum_of_square_areas (X Y Z : ℝ × ℝ) (h : triangle_XYZ X Y Z) :
  (13 : ℝ)^2 + (Real.sqrt ((13 : ℝ)^2 - 12^2))^2 = 194 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_square_areas_l840_84099


namespace NUMINAMATH_CALUDE_river_improvement_equation_l840_84032

theorem river_improvement_equation (x : ℝ) (h : x > 0) : 
  (4800 / x) - (4800 / (x + 200)) = 4 ↔ 
  (∃ (planned_days actual_days : ℝ),
    planned_days = 4800 / x ∧
    actual_days = 4800 / (x + 200) ∧
    planned_days - actual_days = 4) :=
by sorry

end NUMINAMATH_CALUDE_river_improvement_equation_l840_84032


namespace NUMINAMATH_CALUDE_A_in_B_l840_84062

-- Define the set A
def A : Set ℕ := {0, 1}

-- Define the set B
def B : Set (Set ℕ) := {x | x ⊆ A}

-- Theorem statement
theorem A_in_B : A ∈ B := by sorry

end NUMINAMATH_CALUDE_A_in_B_l840_84062


namespace NUMINAMATH_CALUDE_ratio_arithmetic_property_l840_84094

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Our specific sequence -/
def our_sequence (a : ℕ → ℚ) : Prop :=
  is_ratio_arithmetic a 2 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3

theorem ratio_arithmetic_property (a : ℕ → ℚ) (h : our_sequence a) :
  a 2019 / a 2017 = 4 * 2017^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_property_l840_84094


namespace NUMINAMATH_CALUDE_trigonometric_identity_l840_84063

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2016) : 
  1 / Real.cos (2 * α) + Real.tan (2 * α) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l840_84063


namespace NUMINAMATH_CALUDE_inequality_problem_l840_84025

theorem inequality_problem (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l840_84025


namespace NUMINAMATH_CALUDE_square_root_product_plus_one_l840_84039

theorem square_root_product_plus_one : 
  Real.sqrt ((34 : ℝ) * 33 * 32 * 31 + 1) = 1055 := by sorry

end NUMINAMATH_CALUDE_square_root_product_plus_one_l840_84039


namespace NUMINAMATH_CALUDE_count_divisible_sum_l840_84054

theorem count_divisible_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0) ∧
  (∀ n : ℕ, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧
  Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l840_84054


namespace NUMINAMATH_CALUDE_congruence_problem_l840_84066

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^2 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 1 := by sorry

end NUMINAMATH_CALUDE_congruence_problem_l840_84066


namespace NUMINAMATH_CALUDE_sin_2alpha_plus_5pi_18_l840_84021

theorem sin_2alpha_plus_5pi_18 (α : ℝ) (h : Real.sin (π / 9 - α) = 1 / 3) : 
  Real.sin (2 * α + 5 * π / 18) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_plus_5pi_18_l840_84021


namespace NUMINAMATH_CALUDE_four_numbers_proof_l840_84038

theorem four_numbers_proof :
  ∀ (a b c d : ℝ),
    (a / b = 1/5 / (1/3)) →
    (a / c = 1/5 / (1/20)) →
    (b / c = 1/3 / (1/20)) →
    (d = 0.15 * b) →
    (b = a + c + d + 8) →
    (a = 48 ∧ b = 80 ∧ c = 12 ∧ d = 12) := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_proof_l840_84038


namespace NUMINAMATH_CALUDE_equation_equivalence_l840_84030

theorem equation_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 * x = 7 * y) : 
  x / 7 = y / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l840_84030


namespace NUMINAMATH_CALUDE_four_spheres_existence_l840_84001

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray starting from a point
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a point is inside a sphere
def isInside (p : Point3D) (s : Sphere) : Prop := sorry

-- Function to check if two spheres intersect
def intersect (s1 s2 : Sphere) : Prop := sorry

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem four_spheres_existence (A : Point3D) : 
  ∃ (s1 s2 s3 s4 : Sphere),
    (¬ isInside A s1) ∧ (¬ isInside A s2) ∧ (¬ isInside A s3) ∧ (¬ isInside A s4) ∧
    (¬ intersect s1 s2) ∧ (¬ intersect s1 s3) ∧ (¬ intersect s1 s4) ∧
    (¬ intersect s2 s3) ∧ (¬ intersect s2 s4) ∧ (¬ intersect s3 s4) ∧
    (∀ (r : Ray), r.origin = A → 
      rayIntersectsSphere r s1 ∨ rayIntersectsSphere r s2 ∨ 
      rayIntersectsSphere r s3 ∨ rayIntersectsSphere r s4) :=
by
  sorry

end NUMINAMATH_CALUDE_four_spheres_existence_l840_84001


namespace NUMINAMATH_CALUDE_num_lines_formula_l840_84047

/-- The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear, is n(n-1)/2 -/
theorem num_lines_formula (n : ℕ) (h : n ≥ 3) :
  num_lines n = n * (n - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_num_lines_formula_l840_84047


namespace NUMINAMATH_CALUDE_preimage_of_two_neg_one_l840_84034

/-- A mapping f from ℝ² to ℝ² defined by f(a,b) = (a+b, a-b) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (a, b) ↦ (a + b, a - b)

/-- The theorem stating that the preimage of (2, -1) under f is (1/2, 3/2) -/
theorem preimage_of_two_neg_one : 
  f (1/2, 3/2) = (2, -1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_two_neg_one_l840_84034


namespace NUMINAMATH_CALUDE_monomial_coefficient_l840_84042

/-- The coefficient of a monomial is the numerical factor that multiplies the variables. -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The monomial -9/4 * x^2 * y -/
def monomial (x y : ℚ) : ℚ := -9/4 * x^2 * y

theorem monomial_coefficient :
  coefficient (-9/4) x y = -9/4 :=
by sorry

end NUMINAMATH_CALUDE_monomial_coefficient_l840_84042


namespace NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_l840_84078

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (2 * x) + 4 / x ≥ 6 ∧
  (2 * Real.sqrt (2 * x) + 4 / x = 6 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_l840_84078


namespace NUMINAMATH_CALUDE_matrix_problem_l840_84051

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -1; -4, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2; -4, 6]

theorem matrix_problem :
  (∃ X : Matrix (Fin 2) (Fin 2) ℚ, A * X = B) ∧
  A⁻¹ = !![3/2, 1/2; 2, 1] ∧
  A * !![1, 0; 0, 2] = B := by sorry

end NUMINAMATH_CALUDE_matrix_problem_l840_84051


namespace NUMINAMATH_CALUDE_remainder_of_1234567_div_112_l840_84045

theorem remainder_of_1234567_div_112 : Int.mod 1234567 112 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1234567_div_112_l840_84045


namespace NUMINAMATH_CALUDE_corridor_width_l840_84065

theorem corridor_width 
  (w : ℝ) -- width of corridor
  (a : ℝ) -- length of ladder
  (h k : ℝ) -- heights on walls
  (h_pos : h > 0)
  (k_pos : k > 0)
  (a_pos : a > 0)
  (w_pos : w > 0)
  (angle_h : Real.cos (70 * π / 180) = h / a)
  (angle_k : Real.cos (60 * π / 180) = k / a)
  : w = h * Real.tan ((π / 2) - (70 * π / 180)) + k * Real.tan ((π / 2) - (60 * π / 180)) :=
by
  sorry


end NUMINAMATH_CALUDE_corridor_width_l840_84065


namespace NUMINAMATH_CALUDE_function_properties_l840_84022

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties (m : ℝ) :
  f m (π / 2) = 1 →
  (∃ T : ℝ, ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) →
  (∃ M : ℝ, ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) →
  m = 1 ∧
  (∀ x : ℝ, f m x = Real.sqrt 2 * Real.sin (x + π / 4)) ∧
  (∃ T : ℝ, T = 2 * π ∧ ∀ x : ℝ, f m x = f m (x + T) ∧ T > 0 ∧ ∀ S : ℝ, (∀ x : ℝ, f m x = f m (x + S) ∧ S > 0) → T ≤ S) ∧
  (∃ M : ℝ, M = Real.sqrt 2 ∧ ∀ x : ℝ, f m x ≤ M ∧ ∃ y : ℝ, f m y = M) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l840_84022


namespace NUMINAMATH_CALUDE_polygon_E_largest_area_l840_84091

-- Define the polygons and their areas
def polygon_A_area : ℝ := 4
def polygon_B_area : ℝ := 4.5
def polygon_C_area : ℝ := 4.5
def polygon_D_area : ℝ := 5
def polygon_E_area : ℝ := 5.5

-- Define a function to compare areas
def has_largest_area (x y z w v : ℝ) : Prop :=
  v ≥ x ∧ v ≥ y ∧ v ≥ z ∧ v ≥ w

-- Theorem statement
theorem polygon_E_largest_area :
  has_largest_area polygon_A_area polygon_B_area polygon_C_area polygon_D_area polygon_E_area :=
sorry

end NUMINAMATH_CALUDE_polygon_E_largest_area_l840_84091


namespace NUMINAMATH_CALUDE_math_club_team_selection_l840_84003

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 12) 
  (h_team_size : team_size = 8) : 
  (Nat.choose (boys + girls) team_size) - 
  (Nat.choose girls team_size) - 
  (Nat.choose boys team_size) = 319230 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l840_84003


namespace NUMINAMATH_CALUDE_range_of_a_l840_84024

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) = False → a < -2 ∨ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l840_84024


namespace NUMINAMATH_CALUDE_cans_collected_difference_l840_84074

theorem cans_collected_difference :
  let sarah_yesterday : ℕ := 50
  let lara_yesterday : ℕ := sarah_yesterday + 30
  let sarah_today : ℕ := 40
  let lara_today : ℕ := 70
  let total_yesterday : ℕ := sarah_yesterday + lara_yesterday
  let total_today : ℕ := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cans_collected_difference_l840_84074


namespace NUMINAMATH_CALUDE_apple_cost_price_l840_84081

-- Define the selling price
def selling_price : ℚ := 19

-- Define the ratio of selling price to cost price
def selling_to_cost_ratio : ℚ := 5/6

-- Theorem statement
theorem apple_cost_price :
  ∃ (cost_price : ℚ), 
    cost_price = selling_price / selling_to_cost_ratio ∧ 
    cost_price = 114/5 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_price_l840_84081


namespace NUMINAMATH_CALUDE_P_subset_Q_l840_84041

def P : Set ℝ := {x : ℝ | |x| < 2}
def Q : Set ℝ := {x : ℝ | x < 2}

theorem P_subset_Q : P ⊆ Q := by sorry

end NUMINAMATH_CALUDE_P_subset_Q_l840_84041


namespace NUMINAMATH_CALUDE_book_arrangement_count_l840_84023

/-- Represents the number of ways to arrange books on a shelf. -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  let group_arrangements := 6
  let math_arrangements := Nat.factorial math_books
  let english_arrangements := Nat.factorial english_books
  let science_arrangements := Nat.factorial science_books
  group_arrangements * math_arrangements * english_arrangements * science_arrangements

/-- Theorem stating the number of ways to arrange the books on the shelf. -/
theorem book_arrangement_count :
  arrange_books 4 6 2 = 207360 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l840_84023


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l840_84090

/-- The number of bottle caps Evelyn starts with -/
def initial_caps : ℕ := 18

/-- The number of bottle caps Evelyn finds -/
def found_caps : ℕ := 63

/-- The total number of bottle caps Evelyn ends up with -/
def total_caps : ℕ := initial_caps + found_caps

theorem evelyn_bottle_caps : total_caps = 81 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l840_84090


namespace NUMINAMATH_CALUDE_digit_equation_proof_l840_84020

theorem digit_equation_proof :
  ∀ (A B D : ℕ),
    A ≤ 9 → B ≤ 9 → D ≤ 9 →
    A ≥ B →
    (100 * A + 10 * B + D) * (A + B + D) = 1323 →
    D = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_proof_l840_84020


namespace NUMINAMATH_CALUDE_problem_statement_l840_84084

theorem problem_statement (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 5) 
  (h2 : 2 * x + 5 * y = 8) : 
  9 * x^2 + 38 * x * y + 41 * y^2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l840_84084


namespace NUMINAMATH_CALUDE_clouddale_rainfall_2005_l840_84087

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  initialYear : Nat
  initialAvgMonthlyRainfall : Real
  yearlyIncrease : Real

/-- Calculates the average monthly rainfall for a given year -/
def avgMonthlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  data.initialAvgMonthlyRainfall + (year - data.initialYear : Real) * data.yearlyIncrease

/-- Calculates the total yearly rainfall for a given year -/
def totalYearlyRainfall (data : ClouddaleRainfall) (year : Nat) : Real :=
  (avgMonthlyRainfall data year) * 12

/-- Theorem: The total rainfall in Clouddale in 2005 was 522 mm -/
theorem clouddale_rainfall_2005 (data : ClouddaleRainfall) 
    (h1 : data.initialYear = 2003)
    (h2 : data.initialAvgMonthlyRainfall = 37.5)
    (h3 : data.yearlyIncrease = 3) : 
    totalYearlyRainfall data 2005 = 522 := by
  sorry


end NUMINAMATH_CALUDE_clouddale_rainfall_2005_l840_84087


namespace NUMINAMATH_CALUDE_tangent_line_smallest_slope_l840_84049

/-- Given a cubic curve, find the equation of the tangent line with the smallest slope -/
theorem tangent_line_smallest_slope :
  let f (x : ℝ) := x^3 + 3*x^2 + 6*x + 4
  let f' (x : ℝ) := 3*x^2 + 6*x + 6
  let min_slope_point := (-1 : ℝ)
  ∃ (a b c : ℝ), 
    (∀ x, f' x ≥ f' min_slope_point) ∧
    (a * min_slope_point + b * f min_slope_point + c = 0) ∧
    (∀ x y, y = f x → a * x + b * y + c = 0 → y - f min_slope_point = f' min_slope_point * (x - min_slope_point)) ∧
    a = 3 ∧ b = -1 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_smallest_slope_l840_84049


namespace NUMINAMATH_CALUDE_max_total_length_tetrahedron_l840_84010

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f --/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f

/-- The condition that at most one edge is longer than 1 --/
def atMostOneLongerThanOne (t : Tetrahedron) : Prop :=
  (t.a ≤ 1 ∨ (t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.b ≤ 1 ∨ (t.a ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.c ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.d ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.e ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.f ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1))

/-- The total length of all edges in a tetrahedron --/
def totalLength (t : Tetrahedron) : ℝ := t.a + t.b + t.c + t.d + t.e + t.f

/-- The theorem stating the maximum total length of edges in a tetrahedron --/
theorem max_total_length_tetrahedron :
  ∀ t : Tetrahedron, atMostOneLongerThanOne t → totalLength t ≤ 5 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_total_length_tetrahedron_l840_84010


namespace NUMINAMATH_CALUDE_unique_quadratic_function_max_min_values_l840_84073

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem unique_quadratic_function 
  (a b : ℝ) 
  (h1 : a ≠ 0)
  (h2 : f a b 2 = 0)
  (h3 : ∃! x, f a b x = x) :
  ∀ x, f a b x = -1/2 * x^2 + x := by
sorry

-- For the second part of the question
theorem max_min_values
  (h : ∀ x, f 1 (-2) x = x^2 - 2*x) :
  (∀ x ∈ [-1, 2], f 1 (-2) x ≤ 3) ∧
  (∀ x ∈ [-1, 2], f 1 (-2) x ≥ -1) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = 3) ∧
  (∃ x ∈ [-1, 2], f 1 (-2) x = -1) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_max_min_values_l840_84073


namespace NUMINAMATH_CALUDE_problem_solution_l840_84036

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * x^2 * (1/x) = 100/81) : x = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l840_84036


namespace NUMINAMATH_CALUDE_cosine_identity_l840_84060

theorem cosine_identity (θ : ℝ) 
  (h : Real.cos (π / 4 - θ) = Real.sqrt 3 / 3) : 
  Real.cos (3 * π / 4 + θ) - Real.sin (θ - π / 4) ^ 2 = -(2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l840_84060


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l840_84053

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a and b, if they are parallel, then t = 2 or t = -2 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : ℝ × ℝ := (1, t)
  let b : ℝ × ℝ := (t, 4)
  parallel a b → t = 2 ∨ t = -2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_t_value_l840_84053


namespace NUMINAMATH_CALUDE_distributeBalls_eq_180_l840_84095

/-- The number of ways to choose k items from n distinct items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n distinct items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 4 balls out of 5 distinct colored balls into 3 different non-empty boxes -/
def distributeBalls : ℕ :=
  choose 5 4 * choose 4 2 * arrange 3 3

theorem distributeBalls_eq_180 : distributeBalls = 180 := by sorry

end NUMINAMATH_CALUDE_distributeBalls_eq_180_l840_84095


namespace NUMINAMATH_CALUDE_chris_bill_calculation_l840_84052

/-- Calculates the total internet bill based on base charge, overage rate, and data usage over the limit. -/
def total_bill (base_charge : ℝ) (overage_rate : ℝ) (data_over_limit : ℝ) : ℝ :=
  base_charge + overage_rate * data_over_limit

/-- Theorem stating that Chris's total bill is equal to the sum of the base charge and overage charge. -/
theorem chris_bill_calculation (base_charge overage_rate data_over_limit : ℝ) :
  total_bill base_charge overage_rate data_over_limit = base_charge + overage_rate * data_over_limit :=
by sorry

end NUMINAMATH_CALUDE_chris_bill_calculation_l840_84052


namespace NUMINAMATH_CALUDE_book_price_increase_l840_84048

/-- Given a book with an original price of $300 and a price increase of 60%,
    prove that the new price is $480. -/
theorem book_price_increase (original_price : ℝ) (increase_percentage : ℝ) : 
  original_price = 300 → 
  increase_percentage = 60 → 
  original_price * (1 + increase_percentage / 100) = 480 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l840_84048


namespace NUMINAMATH_CALUDE_zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l840_84043

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + b * x + (b - 1)

-- Part 1
theorem zeros_of_specific_quadratic :
  let f₁ := f 1 (-2)
  (f₁ 3 = 0) ∧ (f₁ (-1) = 0) ∧ (∀ x, f₁ x = 0 → x = 3 ∨ x = -1) := by sorry

-- Part 2
theorem range_of_a_for_distinct_zeros (a : ℝ) :
  (a ≠ 0) →
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (0 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_zeros_of_specific_quadratic_range_of_a_for_distinct_zeros_l840_84043


namespace NUMINAMATH_CALUDE_cos_180_degrees_l840_84089

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l840_84089


namespace NUMINAMATH_CALUDE_correct_balloons_popped_l840_84069

/-- The number of blue balloons Sally popped -/
def balloons_popped (joan_initial : ℕ) (jessica : ℕ) (total_now : ℕ) : ℕ :=
  joan_initial - total_now

theorem correct_balloons_popped (joan_initial jessica total_now : ℕ) 
  (h1 : joan_initial = 9)
  (h2 : jessica = 2)
  (h3 : total_now = 6) :
  balloons_popped joan_initial jessica total_now = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_balloons_popped_l840_84069


namespace NUMINAMATH_CALUDE_divisor_count_l840_84044

theorem divisor_count (m n : ℕ+) (h_coprime : Nat.Coprime m n) 
  (h_divisors : (Nat.divisors (m^3 * n^5)).card = 209) : 
  (Nat.divisors (m^5 * n^3)).card = 217 := by
  sorry

end NUMINAMATH_CALUDE_divisor_count_l840_84044


namespace NUMINAMATH_CALUDE_square_25_solutions_l840_84056

theorem square_25_solutions (x : ℝ) (h : x^2 = 25) :
  (∃ y : ℝ, y^2 = 25 ∧ y ≠ x) ∧
  (∀ z : ℝ, z^2 = 25 → z = x ∨ z = -x) ∧
  x + (-x) = 0 ∧
  x * (-x) = -25 := by
sorry

end NUMINAMATH_CALUDE_square_25_solutions_l840_84056


namespace NUMINAMATH_CALUDE_mollys_age_l840_84018

/-- Given that the ratio of Sandy's age to Molly's age is 4:3, and Sandy will be 66 years old in 6 years, prove that Molly's current age is 45 years. -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 66 →
  molly_age = 45 := by
sorry


end NUMINAMATH_CALUDE_mollys_age_l840_84018


namespace NUMINAMATH_CALUDE_parabola_chord_length_l840_84059

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 2py -/
def Parabola (p : ℝ) : Type :=
  {point : ParabolaPoint // point.x^2 = 2 * p * point.y}

theorem parabola_chord_length (p : ℝ) (h_p : p > 0) :
  ∀ (A B : Parabola p),
    (A.val.y + B.val.y) / 2 = 3 →
    (∀ C D : Parabola p, |C.val.y - D.val.y + p| ≤ 8) →
    (∃ E F : Parabola p, |E.val.y - F.val.y + p| = 8) →
    p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l840_84059


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l840_84035

theorem similar_triangles_leg_sum (a₁ a₂ : ℝ) (s : ℝ) :
  a₁ > 0 → a₂ > 0 → s > 0 →
  a₁ = 8 → a₂ = 200 → -- areas of the triangles
  s = 2 → -- shorter leg of smaller triangle
  ∃ (l₁ l₂ : ℝ), 
    l₁ > 0 ∧ l₂ > 0 ∧
    a₁ = (1/2) * s * l₁ ∧ -- area of smaller triangle
    a₂ = (1/2) * (5*s) * (5*l₁) ∧ -- area of larger triangle
    l₁ + l₂ = 50 -- sum of legs of larger triangle
  := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l840_84035


namespace NUMINAMATH_CALUDE_night_flying_hours_l840_84016

theorem night_flying_hours (total_required : ℕ) (day_flying : ℕ) (cross_country : ℕ) (monthly_hours : ℕ) (months : ℕ) : 
  total_required = 1500 →
  day_flying = 50 →
  cross_country = 121 →
  monthly_hours = 220 →
  months = 6 →
  total_required - (day_flying + cross_country) - (monthly_hours * months) = 9 := by
  sorry

end NUMINAMATH_CALUDE_night_flying_hours_l840_84016


namespace NUMINAMATH_CALUDE_readers_of_both_l840_84067

def total_readers : ℕ := 400
def science_fiction_readers : ℕ := 250
def literary_works_readers : ℕ := 230

theorem readers_of_both : ℕ := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_l840_84067


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l840_84088

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^45 + i^345 = 2 * i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l840_84088


namespace NUMINAMATH_CALUDE_four_valid_numbers_l840_84013

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  (∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
    ((a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 9 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 8 ∧ d = 9) ∨
     (a = 9 ∧ b = 1 ∧ c = 8 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 1 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 1 ∧ c = 9 ∧ d = 8) ∨
     (a = 8 ∧ b = 1 ∧ c = 8 ∧ d = 9) ∨
     (a = 8 ∧ b = 9 ∧ c = 1 ∧ d = 8) ∨
     (a = 8 ∧ b = 9 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 8 ∧ c = 1 ∧ d = 9) ∨
     (a = 8 ∧ b = 8 ∧ c = 9 ∧ d = 1))) ∧
  n % 11 = 8

theorem four_valid_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_four_valid_numbers_l840_84013


namespace NUMINAMATH_CALUDE_farmer_brown_additional_cost_l840_84017

/-- The additional cost for Farmer Brown's new hay requirements -/
theorem farmer_brown_additional_cost :
  let original_bales : ℕ := 10
  let original_cost_per_bale : ℕ := 15
  let new_bales : ℕ := 2 * original_bales
  let new_cost_per_bale : ℕ := 18
  (new_bales * new_cost_per_bale) - (original_bales * original_cost_per_bale) = 210 :=
by sorry

end NUMINAMATH_CALUDE_farmer_brown_additional_cost_l840_84017


namespace NUMINAMATH_CALUDE_root_sum_product_l840_84008

theorem root_sum_product (a b : ℝ) : 
  (a^2 + a - 3 = 0) → (b^2 + b - 3 = 0) → (ab - 2023*a - 2023*b = 2020) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_product_l840_84008


namespace NUMINAMATH_CALUDE_parabola_point_distance_l840_84075

/-- A point on the parabola y² = 6x whose distance to the focus is twice its distance to the y-axis has x-coordinate 3/2 -/
theorem parabola_point_distance (x y : ℝ) : 
  y^2 = 6*x →  -- Point (x,y) is on the parabola
  ((x - 3/2)^2 + y^2) = 4*x^2 →  -- Distance to focus is twice distance to y-axis
  x = 3/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l840_84075


namespace NUMINAMATH_CALUDE_factor_expression_l840_84026

theorem factor_expression (x : ℝ) : 63 * x^19 + 147 * x^38 = 21 * x^19 * (3 + 7 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l840_84026


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l840_84033

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l840_84033


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l840_84046

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l840_84046


namespace NUMINAMATH_CALUDE_g_16_value_l840_84068

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 - x + 1

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^4 + b*x^3 + c*x^2 + d*x + (-1)) ∧
  (g 0 = -1) ∧
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g (r^2) = 0)

-- Theorem statement
theorem g_16_value (g : ℝ → ℝ) (h : is_valid_g g) : g 16 = -69905 := by
  sorry

end NUMINAMATH_CALUDE_g_16_value_l840_84068


namespace NUMINAMATH_CALUDE_route_length_difference_l840_84028

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the trip details of Jerry and Beth -/
structure TripDetails where
  jerry_speed : ℝ
  jerry_time : ℝ
  beth_speed : ℝ
  beth_extra_time : ℝ

/-- Theorem stating the difference in route lengths -/
theorem route_length_difference (trip : TripDetails) : 
  trip.jerry_speed = 40 →
  trip.jerry_time = 0.5 →
  trip.beth_speed = 30 →
  trip.beth_extra_time = 1/3 →
  distance trip.beth_speed (trip.jerry_time + trip.beth_extra_time) - 
  distance trip.jerry_speed trip.jerry_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_route_length_difference_l840_84028


namespace NUMINAMATH_CALUDE_loot_box_loss_l840_84004

/-- Calculates the average amount lost when buying loot boxes --/
theorem loot_box_loss (loot_box_cost : ℝ) (average_item_value : ℝ) (total_spent : ℝ) :
  loot_box_cost = 5 →
  average_item_value = 3.5 →
  total_spent = 40 →
  total_spent - (total_spent / loot_box_cost * average_item_value) = 12 := by
  sorry

#check loot_box_loss

end NUMINAMATH_CALUDE_loot_box_loss_l840_84004


namespace NUMINAMATH_CALUDE_sales_difference_is_48_l840_84005

/-- Represents the baker's sales data --/
structure BakerSales where
  usualPastries : ℕ
  usualBread : ℕ
  todayPastries : ℕ
  todayBread : ℕ
  pastryPrice : ℕ
  breadPrice : ℕ

/-- Calculates the difference between today's sales and the daily average sales --/
def salesDifference (sales : BakerSales) : ℕ :=
  let usualTotal := sales.usualPastries * sales.pastryPrice + sales.usualBread * sales.breadPrice
  let todayTotal := sales.todayPastries * sales.pastryPrice + sales.todayBread * sales.breadPrice
  todayTotal - usualTotal

/-- Theorem stating the difference in sales --/
theorem sales_difference_is_48 :
  ∃ (sales : BakerSales),
    sales.usualPastries = 20 ∧
    sales.usualBread = 10 ∧
    sales.todayPastries = 14 ∧
    sales.todayBread = 25 ∧
    sales.pastryPrice = 2 ∧
    sales.breadPrice = 4 ∧
    salesDifference sales = 48 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_is_48_l840_84005


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l840_84064

theorem quadratic_equation_integer_roots (a : ℚ) :
  (∃ x y : ℤ, a * x^2 + (a + 1) * x + (a - 1) = 0 ∧
               a * y^2 + (a + 1) * y + (a - 1) = 0 ∧
               x ≠ y) →
  (a = 0 ∨ a = -1/7 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l840_84064


namespace NUMINAMATH_CALUDE_spot_difference_l840_84037

theorem spot_difference (granger cisco rover : ℕ) : 
  granger = 5 * cisco →
  granger + cisco = 108 →
  rover = 46 →
  cisco < rover / 2 →
  rover / 2 - cisco = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_spot_difference_l840_84037


namespace NUMINAMATH_CALUDE_two_solutions_l840_84092

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line 2x - 3y + 5 = 0 -/
def onLine (p : Point) : Prop :=
  2 * p.x - 3 * p.y + 5 = 0

/-- The distance between two points is √13 -/
def hasDistance13 (p : Point) : Prop :=
  (p.x - 2)^2 + (p.y - 3)^2 = 13

/-- The two solutions -/
def solution1 : Point := ⟨-1, 1⟩
def solution2 : Point := ⟨5, 5⟩

theorem two_solutions :
  ∀ p : Point, (onLine p ∧ hasDistance13 p) ↔ (p = solution1 ∨ p = solution2) := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_l840_84092


namespace NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l840_84079

def fibonacci_factorial_series : List Nat := [1, 1, 2, 3, 5, 8, 13, 21]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum % 100

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 5 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_factorial_last_two_digits_sum_l840_84079


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l840_84027

theorem complex_root_quadratic_equation (b c : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 2 : ℂ) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l840_84027


namespace NUMINAMATH_CALUDE_birds_to_asia_count_l840_84058

/-- The number of bird families that flew away to Asia -/
def birds_to_asia : ℕ := sorry

/-- The number of bird families living near the mountain -/
def birds_near_mountain : ℕ := 38

/-- The number of bird families that flew away to Africa -/
def birds_to_africa : ℕ := 47

theorem birds_to_asia_count : birds_to_asia = 94 := by
  sorry

end NUMINAMATH_CALUDE_birds_to_asia_count_l840_84058


namespace NUMINAMATH_CALUDE_fraction_of_odd_products_in_table_l840_84086

-- Define the size of the multiplication table
def table_size : Nat := 16

-- Define a function to check if a number is odd
def is_odd (n : Nat) : Bool := n % 2 = 1

-- Define a function to count odd numbers in a range
def count_odd (n : Nat) : Nat :=
  (List.range n).filter is_odd |>.length

-- Statement of the theorem
theorem fraction_of_odd_products_in_table :
  (count_odd table_size ^ 2 : Rat) / (table_size ^ 2) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_odd_products_in_table_l840_84086


namespace NUMINAMATH_CALUDE_simplify_fraction_l840_84007

-- Define the expression
def f : ℚ := 1 / (1 / (1/2)^1 + 1 / (1/2)^3 + 1 / (1/2)^4)

-- Theorem statement
theorem simplify_fraction : f = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l840_84007


namespace NUMINAMATH_CALUDE_price_reduction_for_1750_profit_max_profit_1800_at_20_l840_84029

-- Define the initial conditions
def initial_sales : ℕ := 40
def initial_profit_per_shirt : ℕ := 40
def sales_increase_rate : ℚ := 2  -- 1 shirt per 0.5 yuan decrease

-- Define the profit function
def profit_function (price_reduction : ℚ) : ℚ :=
  (initial_profit_per_shirt - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem 1: The price reduction for 1750 yuan profit is 15 yuan
theorem price_reduction_for_1750_profit :
  ∃ (x : ℚ), profit_function x = 1750 ∧ x = 15 := by sorry

-- Theorem 2: The maximum profit is 1800 yuan at 20 yuan price reduction
theorem max_profit_1800_at_20 :
  ∃ (max_profit : ℚ) (optimal_reduction : ℚ),
    max_profit = 1800 ∧
    optimal_reduction = 20 ∧
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function optimal_reduction = max_profit := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_1750_profit_max_profit_1800_at_20_l840_84029


namespace NUMINAMATH_CALUDE_inequality_proof_l840_84076

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l840_84076


namespace NUMINAMATH_CALUDE_burger_expenditure_l840_84070

theorem burger_expenditure (total : ℚ) (movies music ice_cream : ℚ) 
  (h1 : total = 30)
  (h2 : movies = 1/3 * total)
  (h3 : music = 3/10 * total)
  (h4 : ice_cream = 1/5 * total) :
  total - (movies + music + ice_cream) = 5 := by
  sorry

end NUMINAMATH_CALUDE_burger_expenditure_l840_84070


namespace NUMINAMATH_CALUDE_negation_of_all_rectangles_equal_diagonals_l840_84083

-- Define a type for rectangles
variable (Rectangle : Type)

-- Define a predicate for equal diagonals
variable (has_equal_diagonals : Rectangle → Prop)

-- Statement to prove
theorem negation_of_all_rectangles_equal_diagonals :
  (¬ ∀ r : Rectangle, has_equal_diagonals r) ↔ (∃ r : Rectangle, ¬ has_equal_diagonals r) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_rectangles_equal_diagonals_l840_84083


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l840_84093

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  Nat.choose n k = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l840_84093


namespace NUMINAMATH_CALUDE_mixture_problem_l840_84019

/-- Given a mixture of milk and water with an initial ratio of 3:2, 
    if adding 10 liters of water changes the ratio to 2:3, 
    then the initial total quantity was 20 liters. -/
theorem mixture_problem (milk water : ℝ) : 
  milk / water = 3 / 2 →
  milk / (water + 10) = 2 / 3 →
  milk + water = 20 := by
sorry

end NUMINAMATH_CALUDE_mixture_problem_l840_84019


namespace NUMINAMATH_CALUDE_sum_of_roots_l840_84031

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 6) = 7 → b * (b - 6) = 7 → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l840_84031


namespace NUMINAMATH_CALUDE_division_multiplication_relation_l840_84071

theorem division_multiplication_relation (a b c : ℕ) (h : a / b = c) : 
  c * b = a ∧ a / c = b :=
by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_relation_l840_84071


namespace NUMINAMATH_CALUDE_parking_lot_capacity_l840_84000

/-- Represents a multi-story parking lot -/
structure ParkingLot where
  levels : Nat
  capacity_per_level : Nat

/-- Calculates the total capacity of a parking lot -/
def total_capacity (p : ParkingLot) : Nat :=
  p.levels * p.capacity_per_level

/-- Theorem stating the total capacity of the specific parking lot -/
theorem parking_lot_capacity :
  ∃ (p : ParkingLot), p.levels = 5 ∧ p.capacity_per_level = 23 + 62 ∧ total_capacity p = 425 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_capacity_l840_84000


namespace NUMINAMATH_CALUDE_expression_value_l840_84011

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y - 6 * x * z = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l840_84011


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l840_84006

theorem regular_polygon_sides (d : ℕ) : d = 14 → ∃ n : ℕ, n > 2 ∧ d = n * (n - 3) / 2 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l840_84006


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l840_84040

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = 180 →
  A / 3 = B / 4 →
  B / 4 = C / 5 →
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l840_84040


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_l840_84015

theorem x_in_terms_of_y (x y : ℝ) :
  (x + 1) / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y - 5) →
  x = (y^2 + 3*y - 1) / 7 :=
by sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_l840_84015


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l840_84097

theorem subtracted_value_proof (x : ℕ) (h : x = 124) :
  ∃! y : ℕ, 2 * x - y = 110 :=
sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l840_84097


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l840_84050

def n : ℕ := 120000

-- Define the prime factorization of n
axiom prime_factorization : n = 2^8 * 3^2 * 5^5

-- Define a function to count perfect square factors
def count_perfect_square_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem perfect_square_factors_count :
  count_perfect_square_factors n = 30 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l840_84050


namespace NUMINAMATH_CALUDE_max_value_function_l840_84080

theorem max_value_function (a b : ℝ) (h1 : a > b) (h2 : b ≥ 0) :
  ∃ M : ℝ, M = Real.sqrt ((a - b)^2 + a^2) ∧
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 →
    (a - b) * Real.sqrt (1 - x^2) + a * x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_function_l840_84080
