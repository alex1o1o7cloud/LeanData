import Mathlib

namespace NUMINAMATH_CALUDE_tan_difference_inequality_l2430_243014

theorem tan_difference_inequality (x y n : ℝ) (hn : n > 0) (h : Real.tan x = n * Real.tan y) :
  Real.tan (x - y) ^ 2 ≤ (n - 1) ^ 2 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_inequality_l2430_243014


namespace NUMINAMATH_CALUDE_add_like_terms_l2430_243086

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l2430_243086


namespace NUMINAMATH_CALUDE_housewife_spending_fraction_l2430_243056

theorem housewife_spending_fraction (initial_amount : ℝ) (remaining_amount : ℝ)
  (h1 : initial_amount = 150)
  (h2 : remaining_amount = 50) :
  (initial_amount - remaining_amount) / initial_amount = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_housewife_spending_fraction_l2430_243056


namespace NUMINAMATH_CALUDE_exists_a_for_min_g_zero_l2430_243065

-- Define the function f
def f (x : ℝ) : ℝ := x^(3/2)

-- Define the function g
def g (a x : ℝ) : ℝ := x + a * (f x)^(1/3)

-- State the theorem
theorem exists_a_for_min_g_zero :
  (∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂) →  -- f is increasing
  ∃ a : ℝ, (∀ x ∈ Set.Icc 1 9, g a x ≥ 0) ∧ 
           (∃ x ∈ Set.Icc 1 9, g a x = 0) ∧
           a = -1 :=
by sorry

end NUMINAMATH_CALUDE_exists_a_for_min_g_zero_l2430_243065


namespace NUMINAMATH_CALUDE_roots_transformation_l2430_243025

theorem roots_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 5*s₁ - 1 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 5*s₂ - 1 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 5*s₃ - 1 = 0) →
  ((3*s₁)^3 - 12*(3*s₁)^2 + 135*(3*s₁) - 27 = 0) ∧
  ((3*s₂)^3 - 12*(3*s₂)^2 + 135*(3*s₂) - 27 = 0) ∧
  ((3*s₃)^3 - 12*(3*s₃)^2 + 135*(3*s₃) - 27 = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l2430_243025


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_twelve_l2430_243020

theorem sum_of_roots_eq_twelve : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 6)^2 = 16 ∧ 
  (x₂ - 6)^2 = 16 ∧ 
  x₁ + x₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_twelve_l2430_243020


namespace NUMINAMATH_CALUDE_greatest_of_four_consecutive_integers_l2430_243087

theorem greatest_of_four_consecutive_integers (a b c d : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (a + b + c + d = 102) → d = 27 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_four_consecutive_integers_l2430_243087


namespace NUMINAMATH_CALUDE_diamond_value_in_treasure_l2430_243028

/-- Represents the treasure of precious stones -/
structure Treasure where
  diamond_masses : List ℝ
  crystal_mass : ℝ
  total_value : ℝ
  martin_value : ℝ

/-- Calculates the value of diamonds given their masses -/
def diamond_value (masses : List ℝ) : ℝ :=
  100 * (masses.map (λ m => m^2)).sum

/-- Calculates the value of crystals given their mass -/
def crystal_value (mass : ℝ) : ℝ :=
  3 * mass

/-- The main theorem about the value of diamonds in the treasure -/
theorem diamond_value_in_treasure (t : Treasure) : 
  t.total_value = 5000000 ∧ 
  t.martin_value = 2000000 ∧ 
  t.total_value = diamond_value t.diamond_masses + crystal_value t.crystal_mass ∧
  t.martin_value = diamond_value (t.diamond_masses.map (λ m => m/2)) + crystal_value (t.crystal_mass/2) →
  diamond_value t.diamond_masses = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_in_treasure_l2430_243028


namespace NUMINAMATH_CALUDE_balloons_lost_l2430_243078

theorem balloons_lost (initial : ℕ) (current : ℕ) (lost : ℕ) : 
  initial = 9 → current = 7 → lost = initial - current → lost = 2 := by sorry

end NUMINAMATH_CALUDE_balloons_lost_l2430_243078


namespace NUMINAMATH_CALUDE_sector_cone_theorem_l2430_243099

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  sector_angle : ℝ  -- Central angle of the sector in degrees
  sector_radius : ℝ  -- Radius of the sector
  base_radius : ℝ    -- Radius of the cone's base
  slant_height : ℝ   -- Slant height of the cone

/-- Checks if the cone's dimensions are consistent with the sector -/
def is_valid_sector_cone (cone : SectorCone) : Prop :=
  cone.sector_angle = 270 ∧
  cone.sector_radius = 12 ∧
  cone.base_radius = 9 ∧
  cone.slant_height = 12 ∧
  cone.sector_angle / 360 * (2 * Real.pi * cone.sector_radius) = 2 * Real.pi * cone.base_radius ∧
  cone.slant_height = cone.sector_radius

theorem sector_cone_theorem (cone : SectorCone) :
  is_valid_sector_cone cone :=
sorry

end NUMINAMATH_CALUDE_sector_cone_theorem_l2430_243099


namespace NUMINAMATH_CALUDE_factorial_ratio_l2430_243021

theorem factorial_ratio : Nat.factorial 13 / Nat.factorial 12 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2430_243021


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l2430_243018

theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l2430_243018


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l2430_243002

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC is (4,3,2) -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (4, 3, 2) :=
by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l2430_243002


namespace NUMINAMATH_CALUDE_max_rectangles_correct_max_rectangles_optimal_l2430_243085

def max_rectangles (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 5
  | 3 => 8
  | _ => 4 * n - 4

theorem max_rectangles_correct (n : ℕ) :
  max_rectangles n = 
    if n = 1 then 2
    else if n = 2 then 5
    else if n = 3 then 8
    else 4 * n - 4 :=
by sorry

theorem max_rectangles_optimal (n : ℕ) :
  max_rectangles n ≤ (2 * n * 2 * n) / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_rectangles_correct_max_rectangles_optimal_l2430_243085


namespace NUMINAMATH_CALUDE_one_common_sale_day_in_july_l2430_243057

def is_bookstore_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ day % 5 = 0

def is_shoe_store_sale_day (day : Nat) : Prop :=
  day ≤ 31 ∧ ∃ k : Nat, day = 3 + 7 * k

def both_stores_sale_day (day : Nat) : Prop :=
  is_bookstore_sale_day day ∧ is_shoe_store_sale_day day

theorem one_common_sale_day_in_july :
  ∃! day : Nat, both_stores_sale_day day :=
sorry

end NUMINAMATH_CALUDE_one_common_sale_day_in_july_l2430_243057


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2430_243027

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x^2 > 1 → 1/x < 1) ∧
  (∃ x, 1/x < 1 ∧ x^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2430_243027


namespace NUMINAMATH_CALUDE_max_product_843_l2430_243073

def digits : List Nat := [1, 3, 4, 5, 7, 8]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def three_digit_number (a b c : Nat) : Nat := 100 * a + 10 * b + c
def two_digit_number (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  (three_digit_number a b c) * (two_digit_number d e)

theorem max_product_843 :
  ∀ a b c d e,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 4 3 7 5 :=
sorry

end NUMINAMATH_CALUDE_max_product_843_l2430_243073


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l2430_243058

/-- The length of the path traversed by vertex C of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length :
  ∀ (triangle_side square_side : ℝ),
  triangle_side = 3 →
  square_side = 6 →
  ∃ (path_length : ℝ),
  path_length = 18 * Real.pi ∧
  path_length = 12 * (triangle_side * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l2430_243058


namespace NUMINAMATH_CALUDE_expression_value_l2430_243012

theorem expression_value (a b c : ℝ) (h1 : a * b * c > 0) (h2 : a * b < 0) :
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = -5/4 ∨
  (|a| / a + 2 * b / |b| - b * c / |4 * b * c|) = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2430_243012


namespace NUMINAMATH_CALUDE_bottom_right_value_mod_2011_l2430_243003

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- The value of a cell on the board -/
def cellValue (board : Board) (cell : Cell) : ℕ :=
  sorry

/-- Theorem stating that the bottom-right corner value is congruent to 2 modulo 2011 -/
theorem bottom_right_value_mod_2011 (board : Board) 
  (h1 : board.size = 2012)
  (h2 : ∀ c ∈ board.markedCells, c.row + c.col = 2011 ∧ c.row ≠ 1 ∧ c.col ≠ 1)
  (h3 : ∀ c, c.row = 1 ∨ c.col = 1 → cellValue board c = 1)
  (h4 : ∀ c ∈ board.markedCells, cellValue board c = 0)
  (h5 : ∀ c, c.row > 1 ∧ c.col > 1 ∧ c ∉ board.markedCells → 
    cellValue board c = cellValue board {row := c.row - 1, col := c.col} + 
                        cellValue board {row := c.row, col := c.col - 1}) :
  cellValue board {row := 2012, col := 2012} ≡ 2 [MOD 2011] :=
sorry

end NUMINAMATH_CALUDE_bottom_right_value_mod_2011_l2430_243003


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2430_243015

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2430_243015


namespace NUMINAMATH_CALUDE_sin_sqrt3_over_2_solution_set_l2430_243034

theorem sin_sqrt3_over_2_solution_set (θ : ℝ) : 
  Real.sin θ = (Real.sqrt 3) / 2 ↔ 
  ∃ k : ℤ, θ = π / 3 + 2 * k * π ∨ θ = 2 * π / 3 + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_sin_sqrt3_over_2_solution_set_l2430_243034


namespace NUMINAMATH_CALUDE_triangle_identity_l2430_243035

/-- Operation △ between ordered pairs of real numbers -/
def triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, a*d + b*c)

/-- Theorem: If (u,v) △ (x,y) = (u,v) for all real u and v, then (x,y) = (1,0) -/
theorem triangle_identity (x y : ℝ) :
  (∀ u v : ℝ, triangle u v x y = (u, v)) → (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l2430_243035


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2430_243017

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 75 →
  E = 4 * F - 15 →
  D + E + F = 180 →
  F = 24 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2430_243017


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l2430_243013

/-- The number of unique two-scoop sundaes that can be made from n types of ice cream -/
def two_scoop_sundaes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem: Given 6 types of ice cream, the number of unique two-scoop sundaes is 15 -/
theorem ice_cream_sundaes :
  two_scoop_sundaes 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l2430_243013


namespace NUMINAMATH_CALUDE_pet_food_discount_l2430_243049

theorem pet_food_discount (msrp : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ)
  (h1 : msrp = 30)
  (h2 : regular_discount_max = 0.3)
  (h3 : additional_discount = 0.2) :
  msrp * (1 - regular_discount_max) * (1 - additional_discount) = 16.8 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_discount_l2430_243049


namespace NUMINAMATH_CALUDE_sasha_remaining_questions_l2430_243048

/-- Calculates the number of remaining questions given the completion rate, total questions, and work time. -/
def remaining_questions (completion_rate : ℕ) (total_questions : ℕ) (work_time : ℕ) : ℕ :=
  total_questions - completion_rate * work_time

/-- Proves that for Sasha's specific case, the number of remaining questions is 30. -/
theorem sasha_remaining_questions :
  remaining_questions 15 60 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sasha_remaining_questions_l2430_243048


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2430_243001

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}
def B : Set ℤ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2430_243001


namespace NUMINAMATH_CALUDE_fall_spending_calculation_l2430_243067

/-- Represents the spending of River Town government in millions of dollars -/
structure RiverTownSpending where
  july_start : ℝ
  october_end : ℝ

/-- Calculates the spending during September and October -/
def fall_spending (s : RiverTownSpending) : ℝ :=
  s.october_end - s.july_start

/-- Theorem stating that for the given spending data, the fall spending is 3.4 million dollars -/
theorem fall_spending_calculation (s : RiverTownSpending) 
  (h1 : s.july_start = 3.1)
  (h2 : s.october_end = 6.5) : 
  fall_spending s = 3.4 := by
  sorry

#eval fall_spending { july_start := 3.1, october_end := 6.5 }

end NUMINAMATH_CALUDE_fall_spending_calculation_l2430_243067


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_four_l2430_243060

/-- Given a function f(x) = x - 1 + a*exp(x), prove that the sum of its roots is greater than 4 -/
theorem sum_of_roots_greater_than_four (a : ℝ) :
  let f := λ x : ℝ => x - 1 + a * Real.exp x
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_four_l2430_243060


namespace NUMINAMATH_CALUDE_range_of_function_l2430_243069

theorem range_of_function (x : ℝ) : 
  (1/2 : ℝ) ≤ ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ∧ 
  ((14 * Real.cos (2 * x) + 28 * Real.sin x + 15) * Real.pi / 108) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2430_243069


namespace NUMINAMATH_CALUDE_fixed_point_and_equal_intercept_line_l2430_243066

/-- The fixed point through which all lines of the form ax + y - a - 2 = 0 pass -/
def fixed_point : ℝ × ℝ := (1, 2)

/-- The line equation with parameter a -/
def line_equation (a x y : ℝ) : Prop := a * x + y - a - 2 = 0

/-- A line with equal intercepts on both axes passing through a point -/
def equal_intercept_line (p : ℝ × ℝ) (x y : ℝ) : Prop := x + y = p.1 + p.2

theorem fixed_point_and_equal_intercept_line :
  (∀ a : ℝ, ∃ x y : ℝ, line_equation a x y ∧ (x, y) = fixed_point) ∧
  equal_intercept_line fixed_point = λ x y => x + y = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_and_equal_intercept_line_l2430_243066


namespace NUMINAMATH_CALUDE_expand_expression_l2430_243090

theorem expand_expression (x : ℝ) : (11 * x + 5) * 3 * x^3 = 33 * x^4 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2430_243090


namespace NUMINAMATH_CALUDE_pet_store_hamsters_l2430_243084

/-- Given a pet store with rabbits and hamsters, prove the number of hamsters
    when the ratio of rabbits to hamsters is 3:4 and there are 18 rabbits. -/
theorem pet_store_hamsters (rabbit_count : ℕ) (hamster_count : ℕ) : 
  (rabbit_count : ℚ) / hamster_count = 3 / 4 →
  rabbit_count = 18 →
  hamster_count = 24 := by
sorry


end NUMINAMATH_CALUDE_pet_store_hamsters_l2430_243084


namespace NUMINAMATH_CALUDE_not_square_sum_ceil_l2430_243094

theorem not_square_sum_ceil (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + ⌈(4 * (a : ℤ)^2) / (b : ℤ)⌉ = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_sum_ceil_l2430_243094


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2430_243088

/-- The eccentricity of a hyperbola with equation 16x^2 - 9y^2 = 144 is 5/3 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := (a^2 + b^2).sqrt
  let e : ℝ := c / a
  16 * x^2 - 9 * y^2 = 144 → e = 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2430_243088


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l2430_243000

/-- Given a right triangle with legs a and b, hypotenuse c, and a circle of radius ρb
    touching leg b externally and extending to the other sides, prove that b + c = a + 2ρb -/
theorem right_triangle_circle_theorem
  (a b c ρb : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ρb > 0)
  (circle_property : ∃ (x y : ℝ), x^2 + y^2 = ρb^2 ∧ x + y = b ∧ (a - x)^2 + y^2 = c^2) :
  b + c = a + 2*ρb :=
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l2430_243000


namespace NUMINAMATH_CALUDE_tree_spacing_l2430_243096

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 350) (h2 : num_trees = 26) :
  let num_segments := num_trees - 1
  let spacing := yard_length / num_segments
  spacing = 14 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l2430_243096


namespace NUMINAMATH_CALUDE_angle_sum_in_special_polygon_l2430_243031

theorem angle_sum_in_special_polygon (x y : ℝ) : 
  34 + 80 + 90 + (360 - x) + (360 - y) = 540 → x + y = 144 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_polygon_l2430_243031


namespace NUMINAMATH_CALUDE_kareem_has_largest_result_l2430_243091

def jose_result (x : ℕ) : ℕ := ((x - 1) * 2) + 2

def thuy_result (x : ℕ) : ℕ := ((x * 2) - 1) + 2

def kareem_result (x : ℕ) : ℕ := ((x - 1) + 2) * 2

theorem kareem_has_largest_result :
  let start := 10
  kareem_result start > jose_result start ∧ kareem_result start > thuy_result start :=
by sorry

end NUMINAMATH_CALUDE_kareem_has_largest_result_l2430_243091


namespace NUMINAMATH_CALUDE_complex_modulus_l2430_243068

theorem complex_modulus (z : ℂ) : (1 - Complex.I) * z = 1 + 2 * Complex.I → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2430_243068


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l2430_243042

/-- Represents a die with opposite sides summing to 7 -/
structure Die where
  sides : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents a 4x4x4 cube made of 64 dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Function to calculate the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum of visible faces -/
theorem smallest_visible_sum (cube : LargeCube) : 
  visibleSum cube ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l2430_243042


namespace NUMINAMATH_CALUDE_average_of_numbers_is_eleven_l2430_243044

theorem average_of_numbers_is_eleven : ∃ (M N : ℝ), 
  10 < N ∧ N < 20 ∧ 
  M = N - 4 ∧ 
  (8 + M + N) / 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_eleven_l2430_243044


namespace NUMINAMATH_CALUDE_equation_satisfied_at_five_l2430_243092

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- Define the constant c
def c : ℝ := 11

-- Theorem statement
theorem equation_satisfied_at_five :
  2 * (f 5) - c = f (5 - 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_five_l2430_243092


namespace NUMINAMATH_CALUDE_undefined_values_count_l2430_243062

theorem undefined_values_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) = 0) ∧ 
  (∀ x ∉ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) ≠ 0) ∧ 
  Finset.card S = 4 := by
sorry

end NUMINAMATH_CALUDE_undefined_values_count_l2430_243062


namespace NUMINAMATH_CALUDE_house_number_theorem_l2430_243022

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop := sorry

/-- A function that converts a pair of digits to a two-digit number -/
def twoDigitNumber (a b : ℕ) : ℕ := sorry

/-- The set of valid house numbers -/
def validHouseNumbers : Set (ℕ × ℕ × ℕ × ℕ) := sorry

/-- The set of valid prime pairs -/
def validPrimePairs : Set (ℕ × ℕ) := sorry

theorem house_number_theorem :
  ∃ (f : (ℕ × ℕ × ℕ × ℕ) → (ℕ × ℕ)), 
    Function.Bijective f ∧
    (∀ a b c d, (a, b, c, d) ∈ validHouseNumbers ↔ 
      (twoDigitNumber a b, twoDigitNumber c d) ∈ validPrimePairs) := by
  sorry

#check house_number_theorem

end NUMINAMATH_CALUDE_house_number_theorem_l2430_243022


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2430_243010

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem complement_of_intersection (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4}) 
  (hM : M = {1, 2, 3}) 
  (hN : N = {2, 3, 4}) : 
  (U \ (M ∩ N)) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2430_243010


namespace NUMINAMATH_CALUDE_solve_euro_equation_l2430_243061

-- Define the € operation
def euro (x y : ℝ) := 3 * x * y

-- State the theorem
theorem solve_euro_equation (y : ℝ) (h1 : euro y (euro x 5) = 540) (h2 : y = 3) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l2430_243061


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2430_243023

-- Define the polynomial
def f (x m : ℝ) : ℝ := 3 * x^2 - 5 * x + m

-- Theorem statement
theorem polynomial_divisibility (m : ℝ) : 
  (∀ x : ℝ, f x m = 0 → x = 2) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2430_243023


namespace NUMINAMATH_CALUDE_partial_pressure_of_compound_l2430_243016

/-- Represents the partial pressure of a compound in a gas mixture. -/
def partial_pressure (mole_fraction : ℝ) (total_pressure : ℝ) : ℝ :=
  mole_fraction * total_pressure

/-- Theorem stating that the partial pressure of a compound in a gas mixture
    is 0.375 atm, given specific conditions. -/
theorem partial_pressure_of_compound (mole_fraction : ℝ) (total_pressure : ℝ) 
  (h1 : mole_fraction = 0.15)
  (h2 : total_pressure = 2.5) :
  partial_pressure mole_fraction total_pressure = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_partial_pressure_of_compound_l2430_243016


namespace NUMINAMATH_CALUDE_expression_problem_l2430_243093

theorem expression_problem (a : ℝ) (h : 5 * a = 3125) :
  ∃ b : ℝ, 5 * b = 25 ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_problem_l2430_243093


namespace NUMINAMATH_CALUDE_largest_prime_divisor_for_primality_test_l2430_243037

theorem largest_prime_divisor_for_primality_test :
  ∀ n : ℕ, 950 ≤ n → n ≤ 1000 →
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) →
  Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_for_primality_test_l2430_243037


namespace NUMINAMATH_CALUDE_acute_angled_triangle_with_acute_pedals_l2430_243051

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def Angle.toSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Checks if an angle is acute (less than 90 degrees) -/
def Angle.isAcute (a : Angle) : Prop :=
  a.toSeconds < 90 * 3600

/-- Calculates the i-th pedal angle given an original angle -/
def pedalAngle (a : Angle) (i : ℕ) : Angle :=
  sorry -- Implementation not required for the statement

/-- Theorem statement for the acute-angled triangle problem -/
theorem acute_angled_triangle_with_acute_pedals :
  ∃ (α β γ : Angle),
    α.toSeconds < β.toSeconds ∧
    β.toSeconds < γ.toSeconds ∧
    Angle.isAcute α ∧
    Angle.isAcute β ∧
    Angle.isAcute γ ∧
    α.toSeconds + β.toSeconds + γ.toSeconds = 180 * 3600 ∧
    (∀ i : ℕ, i > 0 → i ≤ 15 →
      Angle.isAcute (pedalAngle α i) ∧
      Angle.isAcute (pedalAngle β i) ∧
      Angle.isAcute (pedalAngle γ i)) :=
by sorry

end NUMINAMATH_CALUDE_acute_angled_triangle_with_acute_pedals_l2430_243051


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2430_243026

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 3)*x - k + 6 > 0) ↔ -3 < k ∧ k < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2430_243026


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2430_243041

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2430_243041


namespace NUMINAMATH_CALUDE_team_E_not_played_B_l2430_243064

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → ℕ
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- We don't know F's matches, so we set it to 0

/-- Theorem stating that team E has not played against team B -/
theorem team_E_not_played_B :
  ∀ (t : Team), matches_played Team.E = 1 → matches_played Team.B = 4 →
  matches_played Team.A = 5 → t ≠ Team.B → t ≠ Team.E → 
  ∃ (opponent : Team), opponent ≠ Team.E ∧ opponent ≠ Team.B :=
by sorry

end NUMINAMATH_CALUDE_team_E_not_played_B_l2430_243064


namespace NUMINAMATH_CALUDE_zero_of_f_necessary_not_sufficient_for_decreasing_g_l2430_243004

noncomputable def f (m : ℝ) (x : ℝ) := 2^x + m - 1
noncomputable def g (m : ℝ) (x : ℝ) := Real.log x / Real.log m

theorem zero_of_f_necessary_not_sufficient_for_decreasing_g :
  (∀ m : ℝ, (∃ x : ℝ, f m x = 0) → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) ∧
  (∃ m : ℝ, (∃ x : ℝ, f m x = 0) ∧ 
    ¬(∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → g m x₁ > g m x₂)) :=
by sorry

end NUMINAMATH_CALUDE_zero_of_f_necessary_not_sufficient_for_decreasing_g_l2430_243004


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2430_243098

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) : ℝ :=
  let firstRebound := initialHeight * reboundFactor
  let secondRebound := firstRebound * reboundFactor
  initialHeight + firstRebound + (initialHeight - firstRebound) + 
  secondRebound + (firstRebound - secondRebound) + secondRebound

/-- Theorem: The total distance traveled by a ball dropped from 80 cm 
    with a 50% rebound factor is 200 cm when it touches the floor for the third time -/
theorem ball_bounce_distance :
  totalDistance 80 0.5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2430_243098


namespace NUMINAMATH_CALUDE_circle_condition_exclusive_shape_condition_l2430_243072

/-- Represents a circle equation -/
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 4*x + a^2 = 0

/-- Represents an ellipse equation -/
def is_ellipse (a : ℝ) : Prop :=
  a > 0 ∧ ∃ (x y : ℝ), (y^2 / 3) + (x^2 / a) = 1

/-- The ellipse has its focus on the y-axis -/
def focus_on_y_axis (a : ℝ) : Prop :=
  is_ellipse a → a < 3

theorem circle_condition (a : ℝ) :
  is_circle a ↔ -2 < a ∧ a < 2 :=
sorry

theorem exclusive_shape_condition (a : ℝ) :
  (is_circle a ∨ is_ellipse a) ∧ ¬(is_circle a ∧ is_ellipse a) ↔
  ((-2 < a ∧ a ≤ 0) ∨ (2 ≤ a ∧ a < 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_exclusive_shape_condition_l2430_243072


namespace NUMINAMATH_CALUDE_min_value_theorem_l2430_243047

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2430_243047


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2430_243095

theorem regular_polygon_interior_angle (n : ℕ) (h : n - 3 = 5) :
  (180 * (n - 2) : ℝ) / n = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l2430_243095


namespace NUMINAMATH_CALUDE_externally_tangent_circle_radius_l2430_243039

/-- The radius of a circle externally tangent to three circles in a right triangle -/
theorem externally_tangent_circle_radius (A B C : ℝ × ℝ) (h_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (h_AB : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3)
  (h_AC : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 6)
  (r_A : ℝ) (r_B : ℝ) (r_C : ℝ)
  (h_r_A : r_A = 1) (h_r_B : r_B = 2) (h_r_C : r_C = 3) :
  ∃ R : ℝ, R = (8 * Real.sqrt 11 - 19) / 7 ∧
    ∀ O : ℝ × ℝ, (Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2) = R + r_A) ∧
                 (Real.sqrt ((O.1 - B.1)^2 + (O.2 - B.2)^2) = R + r_B) ∧
                 (Real.sqrt ((O.1 - C.1)^2 + (O.2 - C.2)^2) = R + r_C) :=
by sorry

end NUMINAMATH_CALUDE_externally_tangent_circle_radius_l2430_243039


namespace NUMINAMATH_CALUDE_power_divisibility_implies_equality_l2430_243097

theorem power_divisibility_implies_equality (m n : ℕ) : 
  m > 1 → n > 1 → (4^m - 1) % n = 0 → (n - 1) % (2^m) = 0 → n = 2^m + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_divisibility_implies_equality_l2430_243097


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2430_243089

theorem sqrt_fraction_simplification : 
  Real.sqrt (16 / 25 + 9 / 4) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l2430_243089


namespace NUMINAMATH_CALUDE_expression_value_l2430_243071

theorem expression_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  (x - 3)^2 + (x + 4)*(x - 4) = -9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2430_243071


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2430_243059

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3)^2 = a 1 * a 9 →
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2430_243059


namespace NUMINAMATH_CALUDE_smallest_maximal_arrangement_l2430_243076

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (total_squares : ℕ := size * size)

/-- Represents a Γ piece -/
structure GammaPiece :=
  (squares_covered : ℕ := 3)

/-- Represents an arrangement of Γ pieces on a chessboard -/
structure Arrangement (board : Chessboard) :=
  (pieces : ℕ)
  (is_maximal : Bool)

/-- The theorem stating the smallest number of Γ pieces in a maximal arrangement -/
theorem smallest_maximal_arrangement (board : Chessboard) (piece : GammaPiece) :
  board.size = 8 →
  ∃ (arr : Arrangement board), 
    arr.pieces = 16 ∧ 
    arr.is_maximal = true ∧
    ∀ (arr' : Arrangement board), arr'.is_maximal = true → arr'.pieces ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_maximal_arrangement_l2430_243076


namespace NUMINAMATH_CALUDE_chips_price_increase_l2430_243032

/-- The cost of a pack of pretzels in dollars -/
def pretzel_cost : ℝ := 4

/-- The number of packets of chips bought -/
def chips_bought : ℕ := 2

/-- The number of packets of pretzels bought -/
def pretzels_bought : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := 22

/-- The percentage increase in the price of chips compared to pretzels -/
def price_increase_percentage : ℝ := 75

theorem chips_price_increase :
  let chips_cost := pretzel_cost * (1 + price_increase_percentage / 100)
  chips_bought * chips_cost + pretzels_bought * pretzel_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chips_price_increase_l2430_243032


namespace NUMINAMATH_CALUDE_least_n_divisible_by_77_l2430_243007

theorem least_n_divisible_by_77 (n : ℕ) : 
  (n ≥ 100 ∧ 
   77 ∣ (2^(n+1) - 1) ∧ 
   ∀ m, m ≥ 100 ∧ m < n → ¬(77 ∣ (2^(m+1) - 1))) → 
  n = 119 :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisible_by_77_l2430_243007


namespace NUMINAMATH_CALUDE_five_digit_number_puzzle_l2430_243038

theorem five_digit_number_puzzle :
  ∃! N : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    ∃ (x y : ℕ),
      0 ≤ x ∧ x < 10 ∧
      1000 ≤ y ∧ y < 10000 ∧
      N = 10 * y + x ∧
      N + y = 54321 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_puzzle_l2430_243038


namespace NUMINAMATH_CALUDE_min_arg_z_l2430_243075

/-- Given a complex number z satisfying |z+3-√3i| = √3, 
    the minimum value of arg z is 5π/6 -/
theorem min_arg_z (z : ℂ) (h : Complex.abs (z + 3 - Complex.I * Real.sqrt 3) = Real.sqrt 3) :
  ∃ (min_arg : ℝ), min_arg = 5 * Real.pi / 6 ∧ 
    ∀ (θ : ℝ), Complex.arg z = θ → θ ≥ min_arg :=
by sorry

end NUMINAMATH_CALUDE_min_arg_z_l2430_243075


namespace NUMINAMATH_CALUDE_negation_of_existence_less_than_zero_l2430_243080

theorem negation_of_existence_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_less_than_zero_l2430_243080


namespace NUMINAMATH_CALUDE_exists_m_even_function_l2430_243054

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

-- State the theorem
theorem exists_m_even_function :
  ∃ m : ℝ, ∀ x : ℝ, f m x = f m (-x) :=
sorry

end NUMINAMATH_CALUDE_exists_m_even_function_l2430_243054


namespace NUMINAMATH_CALUDE_brendas_age_l2430_243009

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 7/3 years old. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 7)  -- Janet is seven years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins
  : B = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2430_243009


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l2430_243040

/-- The number of triangles formed from vertices of a regular decagon -/
def triangles_in_decagon : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of triangles in a regular decagon is 120 -/
theorem triangles_in_decagon_count : triangles_in_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l2430_243040


namespace NUMINAMATH_CALUDE_one_match_among_withdrawn_l2430_243082

/-- Represents a table tennis singles competition. -/
structure TableTennisCompetition where
  n : ℕ  -- Total number of players excluding the 3 who withdrew
  x : ℕ  -- Number of matches played among the 3 withdrawn players

/-- Conditions for the competition -/
def validCompetition (comp : TableTennisCompetition) : Prop :=
  comp.n * (comp.n - 1) / 2 + (6 - comp.x) = 50

theorem one_match_among_withdrawn (comp : TableTennisCompetition) :
  validCompetition comp → comp.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_match_among_withdrawn_l2430_243082


namespace NUMINAMATH_CALUDE_restaurant_customers_l2430_243081

theorem restaurant_customers (total : ℕ) : 
  (3 : ℚ) / 5 * total + 10 = total → total = 25 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_customers_l2430_243081


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2430_243052

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l2430_243052


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2430_243029

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2430_243029


namespace NUMINAMATH_CALUDE_chameleon_painter_cannot_create_checkerboard_l2430_243077

structure Board :=
  (size : Nat)
  (initial_color : Bool)

structure Painter :=
  (initial_color : Bool)
  (can_change_self : Bool)
  (can_change_square : Bool)

def is_checkerboard (board : Board) : Prop :=
  sorry

theorem chameleon_painter_cannot_create_checkerboard 
  (board : Board) 
  (painter : Painter) : 
  board.size = 8 ∧ 
  board.initial_color = false ∧ 
  painter.initial_color = true ∧
  painter.can_change_self = true ∧
  painter.can_change_square = true →
  ¬ (is_checkerboard board) :=
sorry

end NUMINAMATH_CALUDE_chameleon_painter_cannot_create_checkerboard_l2430_243077


namespace NUMINAMATH_CALUDE_polygon_ABCDE_perimeter_l2430_243053

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perimeter of a polygon given its vertices -/
def perimeter (vertices : List Point) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem polygon_ABCDE_perimeter :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨4, 8⟩
  let C : Point := ⟨4, 4⟩
  let D : Point := ⟨8, 0⟩
  let E : Point := ⟨0, 0⟩
  perimeter [A, B, C, D, E] = 12 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_ABCDE_perimeter_l2430_243053


namespace NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2430_243055

theorem nellie_gift_wrap_sales (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ) :
  total_goal = 45 →
  sold_to_uncle = 10 →
  sold_to_neighbor = 6 →
  remaining_to_sell = 28 →
  total_goal - remaining_to_sell - (sold_to_uncle + sold_to_neighbor) = 1 :=
by sorry

end NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2430_243055


namespace NUMINAMATH_CALUDE_robins_water_consumption_l2430_243074

theorem robins_water_consumption (bottles_bought : ℕ) (extra_bottles : ℕ) :
  bottles_bought = 617 →
  extra_bottles = 4 →
  ∃ (days : ℕ) (daily_consumption : ℕ) (last_day_consumption : ℕ),
    days = bottles_bought ∧
    daily_consumption = 1 ∧
    last_day_consumption = daily_consumption + extra_bottles ∧
    bottles_bought + extra_bottles = days * daily_consumption + extra_bottles :=
by
  sorry

#check robins_water_consumption

end NUMINAMATH_CALUDE_robins_water_consumption_l2430_243074


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l2430_243063

/-- Given functions f and g, prove that if for any x₁ in [0, 2], 
    there exists x₂ in [1, 2] such that f(x₁) ≥ g(x₂), then m ≥ 1/4 -/
theorem function_inequality_implies_m_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l2430_243063


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2430_243024

/-- The cost of a taxi ride given the base fare, cost per mile, and distance traveled. -/
def taxi_cost (base_fare : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  base_fare + cost_per_mile * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 base fare and $0.30 per mile is $5.00. -/
theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2430_243024


namespace NUMINAMATH_CALUDE_digit_A_value_l2430_243005

theorem digit_A_value : ∃ (A : ℕ), A < 10 ∧ 2 * 1000000 * A + 299561 = (3 * (523 + A))^2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_A_value_l2430_243005


namespace NUMINAMATH_CALUDE_base_conversion_sum_l2430_243079

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value of 254 in base 8 --/
def num1 : Nat := to_base_10 [4, 5, 2] 8

/-- The value of 16 in base 4 --/
def num2 : Nat := to_base_10 [6, 1] 4

/-- The value of 232 in base 7 --/
def num3 : Nat := to_base_10 [2, 3, 2] 7

/-- The value of 34 in base 5 --/
def num4 : Nat := to_base_10 [4, 3] 5

/-- The main theorem to prove --/
theorem base_conversion_sum :
  (num1 : ℚ) / num2 + (num3 : ℚ) / num4 = 23.6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l2430_243079


namespace NUMINAMATH_CALUDE_range_of_a_l2430_243019

-- Define propositions A and B
def propA (x : ℝ) : Prop := |x - 1| < 3
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, propA x → propB x a) ∧ 
  (∃ x, propB x a ∧ ¬propA x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ a < -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2430_243019


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l2430_243070

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_three : a 1 + a 2 + a 3 = 7
  product_first_three : a 1 * a 2 * a 3 = 8
  increasing : ∀ n : ℕ, a n < a (n + 1)

/-- The fourth term of the geometric sequence is 8 -/
theorem fourth_term_is_eight (seq : GeometricSequence) : seq.a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l2430_243070


namespace NUMINAMATH_CALUDE_triangle_area_l2430_243011

theorem triangle_area (a b c : ℝ) (h1 : a = 18) (h2 : b = 80) (h3 : c = 82) :
  (1/2) * a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2430_243011


namespace NUMINAMATH_CALUDE_parking_lot_ratio_l2430_243050

/-- Given the initial number of cars in the front parking lot, the total number of cars at the end,
    and the number of cars added during the play, prove the ratio of cars in the back to front parking lot. -/
theorem parking_lot_ratio
  (front_initial : ℕ)
  (total_end : ℕ)
  (added_during : ℕ)
  (h1 : front_initial = 100)
  (h2 : total_end = 700)
  (h3 : added_during = 300) :
  (total_end - added_during - front_initial) / front_initial = 3 := by
  sorry

#check parking_lot_ratio

end NUMINAMATH_CALUDE_parking_lot_ratio_l2430_243050


namespace NUMINAMATH_CALUDE_final_state_of_B_l2430_243006

/-- Represents a memory unit with a number of data pieces -/
structure MemoryUnit where
  data : ℕ

/-- Represents the state of all three memory units -/
structure MemoryState where
  A : MemoryUnit
  B : MemoryUnit
  C : MemoryUnit

/-- Performs the first operation: storing N data pieces in each unit -/
def firstOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { A := ⟨N⟩, B := ⟨N⟩, C := ⟨N⟩ }

/-- Performs the second operation: moving 2 data pieces from A to B -/
def secondOperation (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data - 2⟩
    B := ⟨state.B.data + 2⟩ }

/-- Performs the third operation: moving 2 data pieces from C to B -/
def thirdOperation (state : MemoryState) : MemoryState :=
  { state with
    B := ⟨state.B.data + 2⟩
    C := ⟨state.C.data - 2⟩ }

/-- Performs the fourth operation: moving N-2 data pieces from B to A -/
def fourthOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data + (N - 2)⟩
    B := ⟨state.B.data - (N - 2)⟩ }

/-- The main theorem stating that after all operations, B has 6 data pieces -/
theorem final_state_of_B (N : ℕ) (h : N ≥ 3) :
  let initialState : MemoryState := ⟨⟨0⟩, ⟨0⟩, ⟨0⟩⟩
  let finalState := fourthOperation N (thirdOperation (secondOperation (firstOperation N initialState)))
  finalState.B.data = 6 := by sorry

end NUMINAMATH_CALUDE_final_state_of_B_l2430_243006


namespace NUMINAMATH_CALUDE_power_multiplication_l2430_243033

theorem power_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2430_243033


namespace NUMINAMATH_CALUDE_red_yellow_peach_difference_l2430_243046

theorem red_yellow_peach_difference (red_peaches yellow_peaches : ℕ) 
  (h1 : red_peaches = 19) 
  (h2 : yellow_peaches = 11) : 
  red_peaches - yellow_peaches = 8 := by
sorry

end NUMINAMATH_CALUDE_red_yellow_peach_difference_l2430_243046


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l2430_243008

/-- The side length of square EFGH -/
def side_length : ℕ := 7

/-- The area of square EFGH -/
def total_area : ℕ := side_length ^ 2

/-- The area of the first shaded region (2x2 square) -/
def shaded_area_1 : ℕ := 2 ^ 2

/-- The area of the second shaded region (5x5 square minus 3x3 square) -/
def shaded_area_2 : ℕ := 5 ^ 2 - 3 ^ 2

/-- The area of the third shaded region (7x1 rectangle) -/
def shaded_area_3 : ℕ := 7 * 1

/-- The total shaded area -/
def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

/-- Theorem: The ratio of shaded area to total area is 33/49 -/
theorem shaded_area_ratio :
  (total_shaded_area : ℚ) / total_area = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l2430_243008


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l2430_243043

theorem divisibility_by_eight (n : ℤ) (h : Even n) :
  ∃ k₁ k₂ k₃ k₄ : ℤ,
    n * (n^2 + 20) = 8 * k₁ ∧
    n * (n^2 - 20) = 8 * k₂ ∧
    n * (n^2 + 4) = 8 * k₃ ∧
    n * (n^2 - 4) = 8 * k₄ :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l2430_243043


namespace NUMINAMATH_CALUDE_painting_price_increase_l2430_243045

theorem painting_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 0.15) = 1.0625 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l2430_243045


namespace NUMINAMATH_CALUDE_area_invariant_under_translation_l2430_243083

/-- Represents a rectangle in a 2D plane --/
structure Rectangle where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- Represents a quadrilateral formed by intersection points of two rectangles --/
structure IntersectionQuadrilateral where
  points : Fin 4 → ℝ × ℝ

/-- Calculates the area of a quadrilateral given its four vertices --/
def quadrilateralArea (q : IntersectionQuadrilateral) : ℝ :=
  sorry

/-- Translates a rectangle by a given vector --/
def translateRectangle (r : Rectangle) (v : ℝ × ℝ) : Rectangle :=
  sorry

/-- Finds the intersection points of two rectangles --/
def findIntersectionPoints (r1 r2 : Rectangle) : Fin 8 → ℝ × ℝ :=
  sorry

/-- Forms a quadrilateral from alternating intersection points --/
def formQuadrilateral (points : Fin 8 → ℝ × ℝ) : IntersectionQuadrilateral :=
  sorry

/-- The main theorem: area invariance under rectangle translation --/
theorem area_invariant_under_translation 
  (r1 r2 : Rectangle) 
  (v : ℝ × ℝ) : 
  let points := findIntersectionPoints r1 r2
  let q1 := formQuadrilateral points
  let r2_translated := translateRectangle r2 v
  let points_after := findIntersectionPoints r1 r2_translated
  let q2 := formQuadrilateral points_after
  quadrilateralArea q1 = quadrilateralArea q2 := by
  sorry

end NUMINAMATH_CALUDE_area_invariant_under_translation_l2430_243083


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2430_243030

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 20 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2430_243030


namespace NUMINAMATH_CALUDE_expression_evaluation_l2430_243036

theorem expression_evaluation :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 21^1006 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2430_243036
