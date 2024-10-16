import Mathlib

namespace NUMINAMATH_CALUDE_find_a_min_value_g_l3342_334284

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1: Prove that a = 3
theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 := by sorry

-- Define g(x) = f(2x) + f(x + 2) where f(x) = |x - 3|
def g (x : ℝ) : ℝ := |2*x - 3| + |x + 2 - 3|

-- Theorem 2: Prove that the minimum value of g(x) is 1/2
theorem min_value_g : 
  ∀ x : ℝ, g x ≥ 1/2 ∧ ∃ y : ℝ, g y = 1/2 := by sorry

end NUMINAMATH_CALUDE_find_a_min_value_g_l3342_334284


namespace NUMINAMATH_CALUDE_red_bead_cost_l3342_334222

/-- The cost of a box of red beads -/
def red_cost : ℝ := 2.30

/-- The cost of a box of yellow beads -/
def yellow_cost : ℝ := 2.00

/-- The number of boxes of each color used -/
def boxes_per_color : ℕ := 4

/-- The total number of mixed boxes -/
def total_boxes : ℕ := 10

/-- The cost per box of mixed beads -/
def mixed_cost : ℝ := 1.72

theorem red_bead_cost :
  red_cost * boxes_per_color + yellow_cost * boxes_per_color = mixed_cost * total_boxes := by
  sorry

#check red_bead_cost

end NUMINAMATH_CALUDE_red_bead_cost_l3342_334222


namespace NUMINAMATH_CALUDE_michael_small_balls_l3342_334295

/-- Represents the number of rubber bands in a pack --/
def total_rubber_bands : ℕ := 5000

/-- Represents the number of rubber bands needed for a small ball --/
def small_ball_rubber_bands : ℕ := 50

/-- Represents the number of rubber bands needed for a large ball --/
def large_ball_rubber_bands : ℕ := 300

/-- Represents the number of large balls that can be made with remaining rubber bands --/
def remaining_large_balls : ℕ := 13

/-- Calculates the number of small balls Michael made --/
def small_balls_made : ℕ :=
  (total_rubber_bands - remaining_large_balls * large_ball_rubber_bands) / small_ball_rubber_bands

theorem michael_small_balls :
  small_balls_made = 22 :=
sorry

end NUMINAMATH_CALUDE_michael_small_balls_l3342_334295


namespace NUMINAMATH_CALUDE_max_visible_cubes_10x10x10_l3342_334294

/-- Represents a cube made of unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_cubes := cube.size * cube.size
  let edge_cubes := cube.size - 1
  3 * face_cubes - 3 * edge_cubes + 1

/-- Theorem stating that for a 10x10x10 cube, the maximum number of visible unit cubes is 274 -/
theorem max_visible_cubes_10x10x10 :
  max_visible_cubes (UnitCube.mk 10) = 274 := by sorry

end NUMINAMATH_CALUDE_max_visible_cubes_10x10x10_l3342_334294


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3342_334208

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 36)
  (sum2 : b + c = 55)
  (sum3 : c + a = 60) : 
  a + b + c = 75.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3342_334208


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3342_334263

theorem fraction_equation_solution (x : ℝ) : (x - 3) / (x + 3) = 2 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3342_334263


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3342_334241

/-- Given two incorrect solutions to a quadratic inequality, prove the correct solution -/
theorem quadratic_inequality_solution 
  (b c : ℝ) 
  (h1 : ∀ x, x^2 + b*x + c < 0 ↔ -6 < x ∧ x < 2)
  (h2 : ∃ c', ∀ x, x^2 + b*x + c' < 0 ↔ -3 < x ∧ x < 2) :
  ∀ x, x^2 + b*x + c < 0 ↔ -4 < x ∧ x < 3 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3342_334241


namespace NUMINAMATH_CALUDE_no_factors_l3342_334261

/-- The main polynomial -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

/-- Potential factor 1 -/
def f1 (x : ℝ) : ℝ := x^2 + 4

/-- Potential factor 2 -/
def f2 (x : ℝ) : ℝ := x - 2

/-- Potential factor 3 -/
def f3 (x : ℝ) : ℝ := x^2 - 4

/-- Potential factor 4 -/
def f4 (x : ℝ) : ℝ := x^2 + 2*x + 4

/-- Theorem stating that none of the given polynomials are factors of p -/
theorem no_factors : 
  (∀ x, p x ≠ 0 → f1 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f2 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f3 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f4 x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_factors_l3342_334261


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3342_334206

/-- Given two lines in the plane, this theorem states that if one line passes through 
    a specific point and is perpendicular to the other line, then it has a specific equation. -/
theorem perpendicular_line_equation 
  (l₁ : Real → Real → Prop) 
  (l₂ : Real → Real → Prop) 
  (h₁ : l₁ = fun x y ↦ 2 * x - 3 * y + 4 = 0) 
  (h₂ : l₂ = fun x y ↦ 3 * x + 2 * y - 1 = 0) : 
  (∀ x y, l₂ x y ↔ (x = -1 ∧ y = 2 ∨ 
    ∃ m : Real, m * (2 : Real) / 3 = -1 ∧ 
    y - 2 = m * (x + 1))) := by 
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3342_334206


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l3342_334242

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(3*n) * y^(m+4) = -3 * x^9 * y^(2*n)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l3342_334242


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l3342_334291

def base2_to_decimal (n : List Bool) : ℕ :=
  n.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem base2_to_base4_conversion :
  let base2 : List Bool := [true, false, true, false, true, false, true, false, true]
  let base4 : List (Fin 4) := [1, 1, 1, 1, 1]
  decimal_to_base4 (base2_to_decimal base2) = base4 := by
  sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l3342_334291


namespace NUMINAMATH_CALUDE_dog_cleaner_amount_l3342_334293

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dog stains -/
def num_dogs : ℕ := 6

/-- The number of cat stains -/
def num_cats : ℕ := 3

/-- The number of rabbit stains -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

theorem dog_cleaner_amount :
  num_dogs * dog_cleaner + num_cats * cat_cleaner + num_rabbits * rabbit_cleaner = total_cleaner :=
by sorry

end NUMINAMATH_CALUDE_dog_cleaner_amount_l3342_334293


namespace NUMINAMATH_CALUDE_circle_radius_from_parabola_tangency_l3342_334219

/-- The radius of a circle given specific tangency conditions of a parabola -/
theorem circle_radius_from_parabola_tangency : ∃ (r : ℝ), 
  (∀ x y : ℝ, y = x^2 + r → y ≤ x) ∧ 
  (∃ x : ℝ, x^2 + r = x) ∧
  r = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_from_parabola_tangency_l3342_334219


namespace NUMINAMATH_CALUDE_painted_cube_probability_l3342_334218

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  paintedFaces : ℕ

/-- The number of unit cubes in a larger cube -/
def totalUnitCubes (size : ℕ) : ℕ := size ^ 3

/-- The number of ways to choose 2 cubes from a set -/
def waysToChooseTwoCubes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unit cubes with exactly three painted faces -/
def cubesWithThreePaintedFaces (size : ℕ) : ℕ := 8

/-- The number of unit cubes with no painted faces -/
def cubesWithNoPaintedFaces (size : ℕ) : ℕ := (size - 2) ^ 3

/-- The probability of selecting one cube with three painted faces and one with no painted faces -/
def probabilityOfSelection (size : ℕ) : ℚ :=
  let total := totalUnitCubes size
  let ways := waysToChooseTwoCubes total
  let threePainted := cubesWithThreePaintedFaces size
  let noPainted := cubesWithNoPaintedFaces size
  (threePainted * noPainted : ℚ) / ways

theorem painted_cube_probability :
  probabilityOfSelection 5 = 72 / 2583 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l3342_334218


namespace NUMINAMATH_CALUDE_tournament_handshakes_count_l3342_334248

/-- Calculates the total number of handshakes in a basketball tournament -/
def tournament_handshakes (num_teams : Nat) (players_per_team : Nat) (num_referees : Nat) : Nat :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem: In a tournament with 3 teams of 7 players each and 3 referees, 
    there are 210 handshakes in total -/
theorem tournament_handshakes_count :
  tournament_handshakes 3 7 3 = 210 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_count_l3342_334248


namespace NUMINAMATH_CALUDE_triangle_area_fraction_l3342_334280

/-- The area of triangle ABC with vertices A(1,3), B(5,1), and C(4,4) is 1/6 of the area of a 6 × 5 rectangle. -/
theorem triangle_area_fraction (A B C : ℝ × ℝ) (h_A : A = (1, 3)) (h_B : B = (5, 1)) (h_C : C = (4, 4)) :
  let triangle_area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  let rectangle_area := 6 * 5
  triangle_area / rectangle_area = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l3342_334280


namespace NUMINAMATH_CALUDE_martin_tv_purchase_l3342_334231

/-- The initial amount Martin decided to spend on a TV -/
def initial_amount : ℝ := 1000

/-- The discount amount applied before the percentage discount -/
def initial_discount : ℝ := 100

/-- The percentage discount applied after the initial discount -/
def percentage_discount : ℝ := 0.20

/-- The difference between the initial amount and the final price -/
def price_difference : ℝ := 280

theorem martin_tv_purchase :
  initial_amount = 1000 ∧
  initial_amount - (initial_amount - initial_discount - 
    percentage_discount * (initial_amount - initial_discount)) = price_difference := by
  sorry

end NUMINAMATH_CALUDE_martin_tv_purchase_l3342_334231


namespace NUMINAMATH_CALUDE_walter_fall_distance_l3342_334276

/-- The distance Walter fell before passing David -/
def distance_fallen (d : ℝ) : ℝ := 2 * d

theorem walter_fall_distance (d : ℝ) (h_positive : d > 0) :
  distance_fallen d = 2 * d :=
by sorry

end NUMINAMATH_CALUDE_walter_fall_distance_l3342_334276


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3342_334227

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3342_334227


namespace NUMINAMATH_CALUDE_intersecting_linear_function_k_range_l3342_334238

/-- A linear function passing through (2, 2) and intersecting y = -x + 3 within [0, 3] -/
structure IntersectingLinearFunction where
  k : ℝ
  b : ℝ
  passes_through_2_2 : 2 * k + b = 2
  intersects_in_domain : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ k * x + b = -x + 3

/-- The range of k values for the intersecting linear function -/
def k_range (f : IntersectingLinearFunction) : Prop :=
  (f.k ≤ -2 ∨ f.k ≥ -1/2) ∧ f.k ≠ 0

theorem intersecting_linear_function_k_range (f : IntersectingLinearFunction) :
  k_range f := by sorry

end NUMINAMATH_CALUDE_intersecting_linear_function_k_range_l3342_334238


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3342_334271

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3342_334271


namespace NUMINAMATH_CALUDE_father_sons_ages_l3342_334283

theorem father_sons_ages (father_age : ℕ) (youngest_son_age : ℕ) (years_until_equal : ℕ) :
  father_age = 33 →
  youngest_son_age = 2 →
  years_until_equal = 12 →
  ∃ (middle_son_age oldest_son_age : ℕ),
    (father_age + years_until_equal = youngest_son_age + years_until_equal + 
                                      middle_son_age + years_until_equal + 
                                      oldest_son_age + years_until_equal) ∧
    (middle_son_age = 3 ∧ oldest_son_age = 4) :=
by sorry

end NUMINAMATH_CALUDE_father_sons_ages_l3342_334283


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3342_334270

theorem rectangle_dimensions (x : ℝ) : 
  x > 3 →
  (x - 3) * (3 * x + 6) = 9 * x - 9 →
  x = (21 + Real.sqrt 549) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3342_334270


namespace NUMINAMATH_CALUDE_decreasing_cubic_function_parameter_bound_l3342_334239

/-- Given a function f(x) = ax³ - x that is decreasing on ℝ, prove that a ≤ 0 -/
theorem decreasing_cubic_function_parameter_bound (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) (3 * a * x^2 - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) →
  a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_cubic_function_parameter_bound_l3342_334239


namespace NUMINAMATH_CALUDE_distance_to_origin_l3342_334285

theorem distance_to_origin (x y : ℝ) (h1 : y = 16) (h2 : x > 3) 
  (h3 : Real.sqrt ((x - 3)^2 + (y - 6)^2) = 14) : 
  Real.sqrt (x^2 + y^2) = 19 + 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3342_334285


namespace NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l3342_334281

theorem cube_volume_from_circumscribed_sphere (V_sphere : ℝ) :
  V_sphere = (32 / 3) * Real.pi →
  ∃ (V_cube : ℝ), V_cube = (64 * Real.sqrt 3) / 9 ∧ 
  (∃ (a : ℝ), V_cube = a^3 ∧ V_sphere = (4 / 3) * Real.pi * ((a * Real.sqrt 3) / 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_circumscribed_sphere_l3342_334281


namespace NUMINAMATH_CALUDE_intersection_M_N_l3342_334230

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.log (2*x + 1) > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3342_334230


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3342_334235

theorem sum_of_three_numbers (A B C : ℚ) : 
  B = 30 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  A + B + C = 98 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3342_334235


namespace NUMINAMATH_CALUDE_vector_subtraction_l3342_334257

/-- Given two vectors OA and OB in ℝ², prove that the vector AB is their difference. -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (1, -2)) (h2 : OB = (-3, 1)) :
  OB - OA = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3342_334257


namespace NUMINAMATH_CALUDE_red_subsequence_2009_l3342_334264

/-- Represents the coloring rule for the red subsequence -/
def red_subsequence : ℕ → ℕ → ℕ → ℕ
| 0, _, _ => 1
| (n+1), count, last =>
  if n % 2 = 0 then
    if count < n + 1 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence n 0 (last + 1)
  else
    if count < n + 2 then red_subsequence n (count + 1) (last + 2)
    else red_subsequence (n + 1) 0 last

/-- The 2009th number in the red subsequence is 3953 -/
theorem red_subsequence_2009 :
  (red_subsequence 1000 0 1) = 3953 := by sorry

end NUMINAMATH_CALUDE_red_subsequence_2009_l3342_334264


namespace NUMINAMATH_CALUDE_different_color_probability_l3342_334228

def shorts_colors : ℕ := 3
def jersey_colors : ℕ := 4

theorem different_color_probability :
  let total_combinations := shorts_colors * jersey_colors
  let different_color_combinations := total_combinations - shorts_colors
  (different_color_combinations : ℚ) / total_combinations = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l3342_334228


namespace NUMINAMATH_CALUDE_part1_part2_part3_l3342_334240

-- Define the system of linear equations
def system (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6 ∧ 3 * x + 2 * y = 4

-- Define the new operation *
def star (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c

-- Theorem for part 1
theorem part1 (x y : ℝ) (h : system x y) : x + y = 2 ∧ x - y = -2 := by
  sorry

-- Theorem for part 2
theorem part2 : ∃ x y : ℝ, 2024 * x + 2025 * y = 2023 ∧ 2022 * x + 2023 * y = 2021 ∧ x = 2 ∧ y = -1 := by
  sorry

-- Theorem for part 3
theorem part3 (a b c : ℝ) (h1 : star a b c 2 4 = 15) (h2 : star a b c 3 7 = 27) : 
  star a b c 1 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l3342_334240


namespace NUMINAMATH_CALUDE_glycerin_solution_problem_l3342_334200

/-- Proves that given a solution with an initial volume of 4 gallons, 
    adding 0.8 gallons of water to achieve a 75% glycerin solution 
    implies that the initial percentage of glycerin was 90%. -/
theorem glycerin_solution_problem (initial_volume : ℝ) (water_added : ℝ) (final_percentage : ℝ) :
  initial_volume = 4 →
  water_added = 0.8 →
  final_percentage = 0.75 →
  (initial_volume * (initial_volume / (initial_volume + water_added))) / initial_volume = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_glycerin_solution_problem_l3342_334200


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3342_334268

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 13) 
  (sum_prod_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1027 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3342_334268


namespace NUMINAMATH_CALUDE_popsicle_stick_count_l3342_334243

/-- Represents the number of popsicle sticks in Gino's problem -/
structure PopsicleSticks where
  initial : ℕ
  given_away : ℕ
  left : ℕ

/-- Theorem stating that the initial number of popsicle sticks 
    is equal to the sum of those given away and those left -/
theorem popsicle_stick_count (p : PopsicleSticks) 
    (h1 : p.given_away = 50)
    (h2 : p.left = 13)
    : p.initial = p.given_away + p.left := by
  sorry

#check popsicle_stick_count

end NUMINAMATH_CALUDE_popsicle_stick_count_l3342_334243


namespace NUMINAMATH_CALUDE_quadratic_factorization_count_l3342_334224

theorem quadratic_factorization_count :
  ∃! (S : Finset Int), 
    (∀ k ∈ S, ∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) ∧
    (∀ k : Int, (∃ a b c d : Int, 2 * X^2 - k * X + 6 = (a * X + b) * (c * X + d)) → k ∈ S) ∧
    Finset.card S = 6 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_factorization_count_l3342_334224


namespace NUMINAMATH_CALUDE_g_range_l3342_334269

noncomputable def g (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem g_range : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, 
  ∃ y ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32), 
  g x = y ∧ 
  ∀ z, g x = z → z ∈ Set.Icc (Real.pi^4 / 16) ((3 * Real.pi^4) / 32) :=
sorry

end NUMINAMATH_CALUDE_g_range_l3342_334269


namespace NUMINAMATH_CALUDE_divisors_of_72_l3342_334212

def divisors (n : ℕ) : Set ℕ := {d | d ∣ n ∧ d > 0}

theorem divisors_of_72 : 
  divisors 72 = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := by sorry

end NUMINAMATH_CALUDE_divisors_of_72_l3342_334212


namespace NUMINAMATH_CALUDE_function_inequality_l3342_334277

open Real

theorem function_inequality (f g : ℝ → ℝ) (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (hpos_f : ∀ x, f x > 0) (hpos_g : ∀ x, g x > 0)
  (h_inequality : ∀ x, (deriv^[2] f) x * g x - f x * (deriv^[2] g) x < 0)
  (a b x : ℝ) (hx : b < x ∧ x < a) :
  f x * g a > f a * g x :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l3342_334277


namespace NUMINAMATH_CALUDE_playground_area_l3342_334207

/-- Proves that a rectangular playground with given conditions has an area of 29343.75 square feet -/
theorem playground_area : 
  ∀ (width length : ℝ),
  length = 3 * width + 40 →
  2 * (width + length) = 820 →
  width * length = 29343.75 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l3342_334207


namespace NUMINAMATH_CALUDE_no_real_solutions_l3342_334289

theorem no_real_solutions : ∀ x : ℝ, (x^3 - 8) / (x - 2) ≠ 3*x :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3342_334289


namespace NUMINAMATH_CALUDE_min_red_chips_is_72_l3342_334255

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def valid_counts (c : ChipCounts) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 72

/-- The minimum number of red chips required -/
def min_red_chips : ℕ := 72

/-- Theorem stating that the minimum number of red chips is 72 -/
theorem min_red_chips_is_72 :
  ∀ c : ChipCounts, valid_counts c → c.red ≥ min_red_chips :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_is_72_l3342_334255


namespace NUMINAMATH_CALUDE_inequality_proof_l3342_334204

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_eq : a + b + c = a * b + b * c + c * a) : 
  3 + (((a^3 + 1) / 2)^(1/3) + ((b^3 + 1) / 2)^(1/3) + ((c^3 + 1) / 2)^(1/3)) ≤ 2 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3342_334204


namespace NUMINAMATH_CALUDE_sum_interior_angles_formula_l3342_334266

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: The sum of the interior angles of an n-sided polygon is (n-2) × 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_formula_l3342_334266


namespace NUMINAMATH_CALUDE_max_marks_calculation_l3342_334297

theorem max_marks_calculation (passing_percentage : ℝ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_percentage = 0.40 →
  scored_marks = 212 →
  short_marks = 44 →
  ∃ max_marks : ℕ, max_marks = 640 ∧ 
    (scored_marks + short_marks : ℝ) / max_marks = passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l3342_334297


namespace NUMINAMATH_CALUDE_set_properties_l3342_334267

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

-- State the theorem
theorem set_properties (x : ℝ) (h : A x ⊆ B) :
  (x = 1 ∨ x = -1) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) :=
by sorry

end NUMINAMATH_CALUDE_set_properties_l3342_334267


namespace NUMINAMATH_CALUDE_new_average_after_modification_l3342_334258

def consecutive_integers (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i)

def modified_sequence (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i - (9 - i))

theorem new_average_after_modification (start : ℤ) :
  (consecutive_integers start).sum / 10 = 20 →
  (modified_sequence start).sum / 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_modification_l3342_334258


namespace NUMINAMATH_CALUDE_perfect_cubes_difference_l3342_334233

theorem perfect_cubes_difference (n : ℕ) : 
  (∃ x y : ℕ, (n + 195 = x^3) ∧ (n - 274 = y^3)) ↔ n = 2002 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_difference_l3342_334233


namespace NUMINAMATH_CALUDE_exists_valid_marking_l3342_334211

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a marking of squares on the chessboard -/
def BoardMarking := Position → Bool

/-- Calculates the minimum number of rook moves between two positions given a board marking -/
def minRookMoves (start finish : Position) (marking : BoardMarking) : ℕ :=
  sorry

/-- Theorem stating the existence of a board marking satisfying the given conditions -/
theorem exists_valid_marking : 
  ∃ (marking : BoardMarking),
    (minRookMoves ⟨0, 0⟩ ⟨2, 3⟩ marking = 3) ∧ 
    (minRookMoves ⟨2, 3⟩ ⟨7, 7⟩ marking = 2) ∧
    (minRookMoves ⟨0, 0⟩ ⟨7, 7⟩ marking = 4) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_marking_l3342_334211


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_formula_implies_t_equals_5_l3342_334287

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), n ≥ 1 → a (n + 1) = r * a n

-- Define the sum formula for the first n terms
def sum_formula (S : ℕ → ℝ) (t : ℝ) : Prop :=
  ∀ (n : ℕ), S n = t * 5^n - 2

-- Theorem statement
theorem geometric_sequence_sum_formula_implies_t_equals_5 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sum_formula S t) : 
  t = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_formula_implies_t_equals_5_l3342_334287


namespace NUMINAMATH_CALUDE_unique_special_number_l3342_334292

/-- A three-digit number ending with 2 that, when the 2 is moved to the front,
    results in a number 18 greater than the original. -/
def special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧  -- ends with 2
  200 + (n / 10) = n + 18  -- moving 2 to front increases by 18

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 202 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l3342_334292


namespace NUMINAMATH_CALUDE_addition_multiplication_equality_l3342_334282

theorem addition_multiplication_equality : 300 + 5 * 8 = 340 := by
  sorry

end NUMINAMATH_CALUDE_addition_multiplication_equality_l3342_334282


namespace NUMINAMATH_CALUDE_range_of_a_complete_theorem_l3342_334236

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}
def Q : Set ℝ := {x | -5 < x ∧ x < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (ha : 0 < a) (h_union : P a ∪ Q = Q) : a ≤ 1 := by
  sorry

-- The complete theorem combining all conditions
theorem complete_theorem :
  ∃ a : ℝ, 0 < a ∧ P a ∪ Q = Q ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_complete_theorem_l3342_334236


namespace NUMINAMATH_CALUDE_vector_position_at_negative_two_l3342_334288

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  position : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfying the problem conditions -/
def given_line : ParametricLine :=
  { position := sorry }

theorem vector_position_at_negative_two :
  let l := given_line
  (l.position 1 = (2, 0, -3)) →
  (l.position 2 = (7, -2, 1)) →
  (l.position 4 = (17, -6, 9)) →
  l.position (-2) = (-1, 3, -9) := by
  sorry

end NUMINAMATH_CALUDE_vector_position_at_negative_two_l3342_334288


namespace NUMINAMATH_CALUDE_speed_limit_violation_percentage_l3342_334229

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percentage : ℝ := 14

/-- Represents the percentage of all motorists who receive speeding tickets -/
def receive_ticket_percentage : ℝ := 10

/-- Represents the percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 30

theorem speed_limit_violation_percentage :
  (receive_ticket_percentage / (100 - no_ticket_percentage) * 100) = exceed_limit_percentage := by
  sorry

end NUMINAMATH_CALUDE_speed_limit_violation_percentage_l3342_334229


namespace NUMINAMATH_CALUDE_first_three_digits_of_large_number_l3342_334237

-- Define the expression
def large_number : ℝ := (10^100 + 1)^(5/3)

-- Define a function to extract the first three decimal digits
def first_three_decimal_digits (x : ℝ) : ℕ × ℕ × ℕ := sorry

-- State the theorem
theorem first_three_digits_of_large_number :
  first_three_decimal_digits large_number = (6, 6, 6) := by sorry

end NUMINAMATH_CALUDE_first_three_digits_of_large_number_l3342_334237


namespace NUMINAMATH_CALUDE_return_trip_duration_l3342_334220

/-- Represents the flight scenario with given conditions -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  time_against_wind : ℝ -- time flying against the wind
  time_diff_still_air : ℝ -- time difference compared to still air for return trip

/-- The possible durations for the return trip -/
def possible_return_times : Set ℝ := {60, 40}

/-- Theorem stating that the return trip duration is either 60 or 40 minutes -/
theorem return_trip_duration (scenario : FlightScenario) 
  (h1 : scenario.time_against_wind = 120)
  (h2 : scenario.time_diff_still_air = 20)
  (h3 : scenario.d > 0)
  (h4 : scenario.p > scenario.w)
  (h5 : scenario.w > 0) :
  ∃ (t : ℝ), t ∈ possible_return_times ∧ 
    scenario.d / (scenario.p + scenario.w) = t := by
  sorry


end NUMINAMATH_CALUDE_return_trip_duration_l3342_334220


namespace NUMINAMATH_CALUDE_problem_solution_l3342_334250

theorem problem_solution (a b c : ℝ) : 
  8 = 0.06 * a → 
  6 = 0.08 * b → 
  c = b / a → 
  c = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3342_334250


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_form_l3342_334245

theorem sqrt_sum_rational_form :
  ∃ (p q r : ℕ+), 
    (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p * Real.sqrt 6 + q * Real.sqrt 8) / r) ∧
    (∀ (p' q' r' : ℕ+), 
      (Real.sqrt 6 + (Real.sqrt 6)⁻¹ + Real.sqrt 8 + (Real.sqrt 8)⁻¹ = (p' * Real.sqrt 6 + q' * Real.sqrt 8) / r') →
      r ≤ r') ∧
    (p + q + r = 19) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_form_l3342_334245


namespace NUMINAMATH_CALUDE_equation_rewrite_l3342_334290

theorem equation_rewrite (x y : ℝ) : 
  (2 * x - y = 4) → (y = 2 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l3342_334290


namespace NUMINAMATH_CALUDE_product_equals_sum_implies_x_value_l3342_334221

theorem product_equals_sum_implies_x_value (x : ℝ) : 
  let S : Set ℝ := {3, 6, 9, x}
  (∃ (a b : ℝ), a ∈ S ∧ b ∈ S ∧ (∀ y ∈ S, a ≤ y ∧ y ≤ b) ∧ a * b = (3 + 6 + 9 + x)) →
  x = 9/4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_implies_x_value_l3342_334221


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_l3342_334217

theorem largest_n_for_square_sum : ∃ (n : ℕ), n = 1490 ∧ 
  (∀ m : ℕ, m > n → ¬ ∃ k : ℕ, 4^995 + 4^1500 + 4^m = k^2) ∧
  (∃ k : ℕ, 4^995 + 4^1500 + 4^n = k^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_l3342_334217


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3342_334265

theorem sum_of_solutions (x : ℝ) : 
  (9 * x / 45 = 6 / x) → (x = 0 ∨ x = 6 / 5) ∧ (0 + 6 / 5 = 6 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3342_334265


namespace NUMINAMATH_CALUDE_probability_one_triple_one_pair_l3342_334203

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def favorable_outcomes : ℕ := faces_per_die * (num_dice.choose 3) * (faces_per_die - 1) * 1

def total_outcomes : ℕ := faces_per_die ^ num_dice

theorem probability_one_triple_one_pair :
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 648 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_triple_one_pair_l3342_334203


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3342_334216

theorem trig_identity_proof : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3342_334216


namespace NUMINAMATH_CALUDE_abc_sum_problem_l3342_334252

theorem abc_sum_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 12) :
  c + 1 / b = 21 / 83 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l3342_334252


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l3342_334246

theorem sum_of_roots_quadratic_equation :
  let a : ℝ := 6 + 3 * Real.sqrt 3
  let b : ℝ := 3 + Real.sqrt 3
  let c : ℝ := -3
  let sum_of_roots := -b / a
  sum_of_roots = -1 + Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_equation_l3342_334246


namespace NUMINAMATH_CALUDE_bed_weight_problem_l3342_334226

theorem bed_weight_problem (single_bed_weight : ℝ) (double_bed_weight : ℝ) : 
  (5 * single_bed_weight = 50) →
  (double_bed_weight = single_bed_weight + 10) →
  (2 * single_bed_weight + 4 * double_bed_weight = 100) :=
by
  sorry

end NUMINAMATH_CALUDE_bed_weight_problem_l3342_334226


namespace NUMINAMATH_CALUDE_combined_solution_x_percentage_l3342_334232

/-- Represents a solution composed of liquid X and water -/
structure Solution where
  total_mass : ℝ
  x_percentage : ℝ

/-- The initial solution Y1 -/
def Y1 : Solution :=
  { total_mass := 12
  , x_percentage := 0.3 }

/-- The mass of water that evaporates -/
def evaporated_water : ℝ := 3

/-- The solution Y2 after evaporation -/
def Y2 : Solution :=
  { total_mass := Y1.total_mass - evaporated_water
  , x_percentage := 0.4 }

/-- The mass of Y2 added to the remaining solution -/
def added_Y2_mass : ℝ := 4

/-- Calculates the mass of liquid X in a given solution -/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.x_percentage

/-- Calculates the mass of water in a given solution -/
def water_mass (s : Solution) : ℝ :=
  s.total_mass * (1 - s.x_percentage)

/-- The combined solution after adding Y2 -/
def combined_solution : Solution :=
  { total_mass := Y2.total_mass + added_Y2_mass
  , x_percentage := 0 }  -- Placeholder value, to be proved

theorem combined_solution_x_percentage :
  combined_solution.x_percentage = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_combined_solution_x_percentage_l3342_334232


namespace NUMINAMATH_CALUDE_senate_committee_seating_l3342_334272

/-- The number of unique circular arrangements of n distinguishable objects -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating :
  circularArrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_l3342_334272


namespace NUMINAMATH_CALUDE_quadratic_eq_implies_quartic_eq_l3342_334275

theorem quadratic_eq_implies_quartic_eq (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) :
  4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_eq_implies_quartic_eq_l3342_334275


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3342_334210

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3342_334210


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3342_334244

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3342_334244


namespace NUMINAMATH_CALUDE_green_hats_count_l3342_334213

/-- The number of green hard hats initially in the truck -/
def initial_green_hats : ℕ := sorry

/-- The number of pink hard hats initially in the truck -/
def initial_pink_hats : ℕ := 26

/-- The number of yellow hard hats in the truck -/
def yellow_hats : ℕ := 24

/-- The number of pink hard hats Carl takes away -/
def carl_pink_hats : ℕ := 4

/-- The number of pink hard hats John takes away -/
def john_pink_hats : ℕ := 6

/-- The number of green hard hats John takes away -/
def john_green_hats : ℕ := 2 * john_pink_hats

/-- The total number of hard hats remaining in the truck -/
def remaining_hats : ℕ := 43

theorem green_hats_count : initial_green_hats = 15 :=
  by sorry

end NUMINAMATH_CALUDE_green_hats_count_l3342_334213


namespace NUMINAMATH_CALUDE_angle_system_solutions_l3342_334279

theorem angle_system_solutions :
  ∀ x y : ℝ,
  0 ≤ x ∧ x < 2 * Real.pi ∧ 0 ≤ y ∧ y < 2 * Real.pi →
  Real.sin x + Real.cos y = 0 ∧ Real.cos x * Real.sin y = -1/2 →
  (x = Real.pi/4 ∧ y = 5*Real.pi/4) ∨
  (x = 3*Real.pi/4 ∧ y = 3*Real.pi/4) ∨
  (x = 5*Real.pi/4 ∧ y = Real.pi/4) ∨
  (x = 7*Real.pi/4 ∧ y = 7*Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_angle_system_solutions_l3342_334279


namespace NUMINAMATH_CALUDE_solution_verification_l3342_334286

/-- Proves that (3, 2020, 4) and (-1, 2018, -2) are solutions to the given system of equations -/
theorem solution_verification :
  (∃ (x y z : ℤ), 
    (x + y - 2018 = (y - 2019) * x) ∧
    (y + z - 2017 = (y - 2019) * z) ∧
    (x + z + 5 = x * z) ∧
    ((x = 3 ∧ y = 2020 ∧ z = 4) ∨ (x = -1 ∧ y = 2018 ∧ z = -2))) := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l3342_334286


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3342_334254

theorem complex_arithmetic_equality : 
  2004 - (2003 - 2004 * (2003 - 2002 * (2003 - 2004)^2004)) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3342_334254


namespace NUMINAMATH_CALUDE_lens_savings_l3342_334215

theorem lens_savings (original_price : ℝ) (discount_rate : ℝ) (cheaper_price : ℝ) : 
  original_price = 300 ∧ 
  discount_rate = 0.20 ∧ 
  cheaper_price = 220 → 
  original_price * (1 - discount_rate) - cheaper_price = 20 := by
sorry

end NUMINAMATH_CALUDE_lens_savings_l3342_334215


namespace NUMINAMATH_CALUDE_cloth_gain_proof_l3342_334205

/-- 
Given:
- A shop owner sells 40 meters of cloth
- The gain percentage is 33.33333333333333%

Prove that the gain is equivalent to the selling price of 10 meters of cloth
-/
theorem cloth_gain_proof (total_meters : ℝ) (gain_percentage : ℝ) 
  (h1 : total_meters = 40)
  (h2 : gain_percentage = 33.33333333333333) :
  (gain_percentage / 100 * total_meters) / (1 + gain_percentage / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloth_gain_proof_l3342_334205


namespace NUMINAMATH_CALUDE_exp_7pi_i_div_3_rectangular_form_l3342_334225

theorem exp_7pi_i_div_3_rectangular_form :
  Complex.exp (7 * Real.pi * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_exp_7pi_i_div_3_rectangular_form_l3342_334225


namespace NUMINAMATH_CALUDE_angle_c_possibilities_l3342_334234

theorem angle_c_possibilities : ∃ (s : Finset ℕ), 
  (∀ c ∈ s, ∃ d : ℕ, 
    c > 0 ∧ d > 0 ∧ 
    c + d = 180 ∧ 
    ∃ k : ℕ, k > 0 ∧ c = k * d) ∧
  (∀ c : ℕ, 
    (∃ d : ℕ, c > 0 ∧ d > 0 ∧ c + d = 180 ∧ ∃ k : ℕ, k > 0 ∧ c = k * d) → 
    c ∈ s) ∧
  s.card = 17 :=
sorry

end NUMINAMATH_CALUDE_angle_c_possibilities_l3342_334234


namespace NUMINAMATH_CALUDE_quiz_logic_l3342_334247

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answers_all_correctly : Student → Prop)
variable (passes_quiz : Student → Prop)

-- State the theorem
theorem quiz_logic (s : Student) 
  (h : ∀ x : Student, answers_all_correctly x → passes_quiz x) :
  ¬(passes_quiz s) → ¬(answers_all_correctly s) :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_logic_l3342_334247


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_two_l3342_334278

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_equals_negative_two :
  (Function.invFun g) 8 + (Function.invFun g) (-64) = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_two_l3342_334278


namespace NUMINAMATH_CALUDE_delaney_travel_time_l3342_334298

/-- The time (in minutes) when the bus leaves, relative to midnight -/
def bus_departure_time : ℕ := 8 * 60

/-- The time (in minutes) when Delaney left home, relative to midnight -/
def delaney_departure_time : ℕ := 7 * 60 + 50

/-- The time (in minutes) that Delaney missed the bus by -/
def missed_by : ℕ := 20

/-- The time (in minutes) it takes Delaney to reach the pick-up point -/
def travel_time : ℕ := bus_departure_time + missed_by - delaney_departure_time

theorem delaney_travel_time : travel_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_delaney_travel_time_l3342_334298


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3342_334214

/-- An arithmetic sequence {aₙ} satisfying aₙ₊₁ + aₙ = 4n for all n has a₁ = 1 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)  -- arithmetic sequence condition
  (h_sum : ∀ n, a (n + 1) + a n = 4 * n)                    -- given condition
  : a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3342_334214


namespace NUMINAMATH_CALUDE_q_zero_at_sqrt2_l3342_334251

-- Define the polynomial q
def q (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) (x y : ℝ) : ℝ :=
  b₀ + b₁*x + b₂*y + b₃*x^2 + b₄*x*y + b₅*y^2 + b₆*x^3 + b₈*x*y^2 + b₉*y^3

-- State the theorem
theorem q_zero_at_sqrt2 (b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ : ℝ) :
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-1) 0 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 0 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 1 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (-2) 1 = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ 3 (-1) = 0 →
  q b₀ b₁ b₂ b₃ b₄ b₅ b₆ b₈ b₉ (Real.sqrt 2) (Real.sqrt 2) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_q_zero_at_sqrt2_l3342_334251


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3342_334273

theorem inequalities_theorem (a b m : ℝ) :
  (b < a ∧ a < 0 → 1 / a < 1 / b) ∧
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3342_334273


namespace NUMINAMATH_CALUDE_correct_costs_l3342_334223

/-- The cost of a pen in yuan -/
def pen_cost : ℝ := 10

/-- The cost of an exercise book in yuan -/
def book_cost : ℝ := 1

/-- The total cost of 2 exercise books and 1 pen in yuan -/
def total_cost : ℝ := 12

theorem correct_costs :
  (2 * book_cost + pen_cost = total_cost) ∧
  (book_cost = 0.1 * pen_cost) ∧
  (pen_cost = 10) ∧
  (book_cost = 1) := by sorry

end NUMINAMATH_CALUDE_correct_costs_l3342_334223


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_200_l3342_334299

/-- Given a train traveling at 72 kmph that crosses a platform in 30 seconds and a man in 20 seconds, 
    the length of the platform is 200 meters. -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) 
  (h1 : train_speed = 72) 
  (h2 : time_platform = 30) 
  (h3 : time_man = 20) : ℝ := by
  
  -- Convert train speed from kmph to m/s
  let train_speed_ms := train_speed * 1000 / 3600

  -- Calculate length of train
  let train_length := train_speed_ms * time_man

  -- Calculate total distance (train + platform)
  let total_distance := train_speed_ms * time_platform

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that platform_length = 200
  sorry

/-- The length of the platform is 200 meters -/
theorem platform_length_is_200 : platform_length 72 30 20 rfl rfl rfl = 200 := by sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_200_l3342_334299


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3342_334253

/-- An isosceles triangle with perimeter 13 and one side 3 has a base of 3 -/
theorem isosceles_triangle_base_length :
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 13 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 3 ∨ b = 3 ∨ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3342_334253


namespace NUMINAMATH_CALUDE_field_length_calculation_l3342_334202

/-- Given a rectangular field wrapped with tape, calculate its length. -/
theorem field_length_calculation (total_tape : ℕ) (width : ℕ) (leftover_tape : ℕ) :
  total_tape = 250 →
  width = 20 →
  leftover_tape = 90 →
  2 * (width + (total_tape - leftover_tape) / 2) = total_tape - leftover_tape →
  (total_tape - leftover_tape) / 2 - width = 60 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l3342_334202


namespace NUMINAMATH_CALUDE_square_difference_l3342_334296

theorem square_difference (a b : ℝ) :
  ∃ A : ℝ, (5*a + 3*b)^2 = (5*a - 3*b)^2 + A ∧ A = 60*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3342_334296


namespace NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l3342_334201

theorem factorization_of_x2y_minus_4y (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x2y_minus_4y_l3342_334201


namespace NUMINAMATH_CALUDE_journal_pages_per_session_l3342_334260

/-- Given the number of journal-writing sessions per week and the total number of pages written
    in a certain number of weeks, calculate the number of pages written per session. -/
def pages_per_session (sessions_per_week : ℕ) (total_pages : ℕ) (num_weeks : ℕ) : ℕ :=
  total_pages / (sessions_per_week * num_weeks)

/-- Theorem stating that under the given conditions, each student writes 4 pages per session. -/
theorem journal_pages_per_session :
  pages_per_session 3 72 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_journal_pages_per_session_l3342_334260


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l3342_334256

theorem reciprocal_of_negative_two_thirds :
  let x : ℚ := -2/3
  let reciprocal (y : ℚ) : ℚ := if y ≠ 0 then 1 / y else 0
  reciprocal x = -3/2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l3342_334256


namespace NUMINAMATH_CALUDE_weaving_problem_l3342_334262

/-- Represents the daily cloth production in a geometric sequence --/
def cloth_sequence (n : ℕ) : ℚ := sorry

/-- The sum of the first 5 terms of the sequence --/
def sum_5_days : ℚ := sorry

theorem weaving_problem :
  (∀ n, cloth_sequence (n + 1) = 2 * cloth_sequence n) →  -- Doubling each day
  sum_5_days = 5 →                                       -- Total of 5 feet in 5 days
  cloth_sequence 2 = 10 / 31 :=                          -- Second day's production
by sorry

end NUMINAMATH_CALUDE_weaving_problem_l3342_334262


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3342_334274

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 - 2 * x₂ - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3342_334274


namespace NUMINAMATH_CALUDE_largest_non_sum_is_correct_l3342_334259

/-- The largest natural number not exceeding 50 that cannot be expressed as a sum of 5s and 6s -/
def largest_non_sum : ℕ := 19

/-- A predicate that checks if a natural number can be expressed as a sum of 5s and 6s -/
def is_sum_of_5_and_6 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_non_sum_is_correct :
  (largest_non_sum ≤ 50) ∧
  ¬(is_sum_of_5_and_6 largest_non_sum) ∧
  ∀ (m : ℕ), m > largest_non_sum → m ≤ 50 → is_sum_of_5_and_6 m :=
by sorry

end NUMINAMATH_CALUDE_largest_non_sum_is_correct_l3342_334259


namespace NUMINAMATH_CALUDE_nina_has_24_dollars_l3342_334209

/-- The amount of money Nina has -/
def nina_money : ℝ := 24

/-- The original price of a widget -/
def original_price : ℝ := 4

/-- Nina can purchase exactly 6 widgets at the original price -/
axiom nina_purchase_original : nina_money = 6 * original_price

/-- If each widget's price is reduced by $1, Nina can purchase exactly 8 widgets -/
axiom nina_purchase_reduced : nina_money = 8 * (original_price - 1)

/-- Proof that Nina has $24 -/
theorem nina_has_24_dollars : nina_money = 24 := by sorry

end NUMINAMATH_CALUDE_nina_has_24_dollars_l3342_334209


namespace NUMINAMATH_CALUDE_residue_mod_17_l3342_334249

theorem residue_mod_17 : (195 * 15 - 18 * 8 + 4) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_17_l3342_334249
