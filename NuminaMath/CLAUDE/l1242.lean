import Mathlib

namespace NUMINAMATH_CALUDE_cube_root_properties_l1242_124269

theorem cube_root_properties :
  let n : ℕ := 59319
  let a : ℕ := 6859
  let b : ℕ := 19683
  let c : ℕ := 110592
  ∃ (x y z : ℕ),
    (10 ≤ x ∧ x < 100) ∧
    x^3 = n ∧
    x = 39 ∧
    y^3 = a ∧ y = 19 ∧
    z^3 = b ∧ z = 27 ∧
    (∃ w : ℕ, w^3 = c ∧ w = 48) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_properties_l1242_124269


namespace NUMINAMATH_CALUDE_koi_fish_problem_l1242_124216

theorem koi_fish_problem (num_koi : ℕ) (subtracted_num : ℕ) : 
  num_koi = 39 → 
  2 * num_koi - subtracted_num = 64 → 
  subtracted_num = 14 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_problem_l1242_124216


namespace NUMINAMATH_CALUDE_no_solution_to_system_l1242_124212

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l1242_124212


namespace NUMINAMATH_CALUDE_first_group_number_is_9_l1242_124230

/-- Represents a systematic sampling method -/
structure SystematicSampling where
  population : ℕ
  sample_size : ℕ
  group_number : ℕ → ℕ
  h_population : population > 0
  h_sample_size : sample_size > 0
  h_sample_size_le_population : sample_size ≤ population

/-- The number drawn by the first group in a systematic sampling -/
def first_group_number (s : SystematicSampling) : ℕ :=
  s.group_number 1

/-- Theorem stating that the first group number is 9 given the problem conditions -/
theorem first_group_number_is_9 (s : SystematicSampling)
    (h_population : s.population = 960)
    (h_sample_size : s.sample_size = 32)
    (h_fifth_group : s.group_number 5 = 129) :
    first_group_number s = 9 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_is_9_l1242_124230


namespace NUMINAMATH_CALUDE_jerry_age_l1242_124295

/-- Given that Mickey's age is 8 years less than 200% of Jerry's age,
    and Mickey is 16 years old, prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 16)
  (h2 : mickey_age = 2 * jerry_age - 8) : 
  jerry_age = 12 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l1242_124295


namespace NUMINAMATH_CALUDE_distance_to_triangle_plane_l1242_124278

-- Define the sphere and points
def Sphere : Type := ℝ × ℝ × ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the center and radius of the sphere
def S : Sphere := sorry
def radius : ℝ := 25

-- Define the points on the sphere
def P : Point := sorry
def Q : Point := sorry
def R : Point := sorry

-- Define the distances between points
def PQ : ℝ := 15
def QR : ℝ := 20
def RP : ℝ := 25

-- Define the distance function
def distance (a b : Point) : ℝ := sorry

-- Define the function to calculate the distance from a point to a plane
def distToPlane (point : Point) (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem distance_to_triangle_plane :
  distToPlane S P Q R = 25 * Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_triangle_plane_l1242_124278


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1242_124214

/-- 
Given a quadratic equation (a-1)x^2 - 4x - 1 = 0, where 'a' is a parameter,
this theorem states the conditions on 'a' for the equation to have two real roots.
-/
theorem quadratic_two_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 4*x - 1 = 0 ∧ (a - 1) * y^2 - 4*y - 1 = 0) ↔ 
  (a ≥ -3 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1242_124214


namespace NUMINAMATH_CALUDE_conservation_center_turtles_l1242_124249

/-- The number of green turtles -/
def green_turtles : ℕ := 800

/-- The number of hawksbill turtles -/
def hawksbill_turtles : ℕ := 2 * green_turtles + green_turtles

/-- The total number of turtles in the conservation center -/
def total_turtles : ℕ := green_turtles + hawksbill_turtles

theorem conservation_center_turtles : total_turtles = 3200 := by
  sorry

end NUMINAMATH_CALUDE_conservation_center_turtles_l1242_124249


namespace NUMINAMATH_CALUDE_a_fraction_of_b_and_c_l1242_124296

def total_amount : ℝ := 300
def a_share : ℝ := 120.00000000000001

theorem a_fraction_of_b_and_c (b_share c_share : ℝ) :
  (a_share = (2/3 : ℝ) * (b_share + c_share)) →
  (b_share = (6/9 : ℝ) * (a_share + c_share)) →
  (a_share + b_share + c_share = total_amount) →
  (a_share / (b_share + c_share) = 2/3) := by
sorry

end NUMINAMATH_CALUDE_a_fraction_of_b_and_c_l1242_124296


namespace NUMINAMATH_CALUDE_max_sum_on_ellipse_l1242_124273

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the sum function S
def S (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem max_sum_on_ellipse :
  (∀ x y : ℝ, on_ellipse x y → S x y ≤ 2) ∧
  (∃ x y : ℝ, on_ellipse x y ∧ S x y = 2) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_on_ellipse_l1242_124273


namespace NUMINAMATH_CALUDE_probability_two_number_cards_sum_15_l1242_124254

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of number cards (2 through 10) in each suit -/
def numberCardsPerSuit : ℕ := 9

/-- The number of suits in a standard deck -/
def numberOfSuits : ℕ := 4

/-- The total number of number cards (2 through 10) in a standard deck -/
def totalNumberCards : ℕ := numberCardsPerSuit * numberOfSuits

/-- The possible first card values that can sum to 15 with another number card -/
def validFirstCards : List ℕ := [5, 6, 7, 8, 9]

/-- The number of ways to choose two number cards that sum to 15 -/
def waysToSum15 : ℕ := validFirstCards.length * numberOfSuits

theorem probability_two_number_cards_sum_15 :
  (waysToSum15 : ℚ) / (standardDeckSize * (standardDeckSize - 1)) = 100 / 663 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_number_cards_sum_15_l1242_124254


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1242_124207

def is_valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  a^2 + c^2 > b^2 ∧
  b^2 + c^2 > a^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 9) :=
by sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1242_124207


namespace NUMINAMATH_CALUDE_picture_area_l1242_124234

/-- The area of a picture on a sheet of paper with given dimensions and margins. -/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_area_l1242_124234


namespace NUMINAMATH_CALUDE_marble_selection_with_blue_l1242_124202

def total_marbles : ℕ := 10
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 3
def selection_size : ℕ := 4

theorem marble_selection_with_blue (total_marbles red_marbles blue_marbles green_marbles selection_size : ℕ) 
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : total_marbles = 10)
  (h3 : red_marbles = 3)
  (h4 : blue_marbles = 4)
  (h5 : green_marbles = 3)
  (h6 : selection_size = 4) :
  (Nat.choose total_marbles selection_size) - (Nat.choose (total_marbles - blue_marbles) selection_size) = 195 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_with_blue_l1242_124202


namespace NUMINAMATH_CALUDE_container_capacity_l1242_124258

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 9 = 0.75 * C) : C = 20 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l1242_124258


namespace NUMINAMATH_CALUDE_disinfectant_sales_l1242_124263

/-- Disinfectant sales problem -/
theorem disinfectant_sales 
  (cost_A : ℕ) (cost_B : ℕ) (total_cost : ℕ) 
  (initial_price_A : ℕ) (initial_volume_A : ℕ) 
  (price_change : ℕ) (volume_change : ℕ)
  (price_B : ℕ) (x : ℕ) :
  cost_A = 20 →
  cost_B = 30 →
  total_cost = 2000 →
  initial_price_A = 30 →
  initial_volume_A = 100 →
  price_change = 1 →
  volume_change = 5 →
  price_B = 60 →
  x > 30 →
  (∃ (volume_A : ℕ → ℕ) (cost_price_B : ℕ → ℕ) (volume_B : ℕ → ℚ) 
      (max_profit : ℕ) (valid_prices : List ℕ),
    (∀ y : ℕ, volume_A y = 250 - 5 * y) ∧
    (∀ y : ℕ, cost_price_B y = 100 * y - 3000) ∧
    (∀ y : ℕ, volume_B y = (10 * y : ℚ) / 3 - 100) ∧
    max_profit = 2125 ∧
    valid_prices = [39, 42, 45, 48] ∧
    (∀ p ∈ valid_prices, 
      (-5 * (p - 45)^2 + 2125 : ℚ) ≥ 1945 ∧ 
      p ≤ 50 ∧ 
      p % 3 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_sales_l1242_124263


namespace NUMINAMATH_CALUDE_cubic_function_tangent_line_l1242_124261

/-- Given a cubic function f(x) = x^3 + ax + b, prove that if its tangent line
    at x = 1 has the equation 2x - y - 5 = 0, then a = -1 and b = -3. -/
theorem cubic_function_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 + a*x + b
  let f' : ℝ → ℝ := λ x => 3*x^2 + a
  (f' 1 = 2 ∧ f 1 = -3) → (a = -1 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_tangent_line_l1242_124261


namespace NUMINAMATH_CALUDE_integer_solution_system_l1242_124248

theorem integer_solution_system (m n : ℤ) : 
  m * (m + n) = n * 12 ∧ n * (m + n) = m * 3 → m = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_system_l1242_124248


namespace NUMINAMATH_CALUDE_hypergeom_problem_l1242_124283

/-- Hypergeometric distribution parameters -/
structure HyperGeomParams where
  N : ℕ  -- Population size
  M : ℕ  -- Number of successes in the population
  n : ℕ  -- Number of draws
  h1 : M ≤ N
  h2 : n ≤ N

/-- Probability of k successes in n draws -/
def prob_k_successes (p : HyperGeomParams) (k : ℕ) : ℚ :=
  (Nat.choose p.M k * Nat.choose (p.N - p.M) (p.n - k)) / Nat.choose p.N p.n

/-- Expected value of hypergeometric distribution -/
def expected_value (p : HyperGeomParams) : ℚ :=
  (p.n * p.M : ℚ) / p.N

/-- Theorem for the specific problem -/
theorem hypergeom_problem (p : HyperGeomParams) 
    (h3 : p.N = 10) (h4 : p.M = 5) (h5 : p.n = 4) : 
    prob_k_successes p 3 = 5 / 21 ∧ expected_value p = 2 := by
  sorry


end NUMINAMATH_CALUDE_hypergeom_problem_l1242_124283


namespace NUMINAMATH_CALUDE_friends_total_earnings_l1242_124276

/-- The total earnings of four friends selling electronics on eBay -/
def total_earnings (lauryn_earnings : ℝ) : ℝ :=
  let aurelia_earnings := 0.7 * lauryn_earnings
  let jackson_earnings := 1.5 * aurelia_earnings
  let maya_earnings := 0.4 * jackson_earnings
  lauryn_earnings + aurelia_earnings + jackson_earnings + maya_earnings

/-- Theorem stating that the total earnings of the four friends is $6340 -/
theorem friends_total_earnings :
  total_earnings 2000 = 6340 := by
  sorry

end NUMINAMATH_CALUDE_friends_total_earnings_l1242_124276


namespace NUMINAMATH_CALUDE_linear_function_point_sum_l1242_124251

/-- If the point A(m, n) lies on the line y = -2x + 1, then 4m + 2n + 2022 = 2024 -/
theorem linear_function_point_sum (m n : ℝ) : n = -2 * m + 1 → 4 * m + 2 * n + 2022 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_sum_l1242_124251


namespace NUMINAMATH_CALUDE_two_numbers_product_l1242_124286

theorem two_numbers_product (n : ℕ) (h : n = 34) : ∃ x y : ℕ, 
  x ∈ Finset.range (n + 1) ∧ 
  y ∈ Finset.range (n + 1) ∧ 
  x ≠ y ∧
  (Finset.sum (Finset.range (n + 1)) id) - x - y = 22 * (y - x) ∧
  x * y = 416 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_product_l1242_124286


namespace NUMINAMATH_CALUDE_range_of_m_l1242_124297

/-- Given two predicates p and q on real numbers x and m, prove that m ≥ 8 -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|1 - (x - 1) / 3| ≤ 2 → x^2 - 4*x + 4 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, |1 - (x - 1) / 3| ≤ 2 ∧ x^2 - 4*x + 4 - m^2 > 0)) →
  m > 0 →
  m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1242_124297


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l1242_124262

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

-- Theorem statement
theorem binary_110_equals_6 :
  binary_to_decimal [true, true, false] = 6 := by
  sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l1242_124262


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_25_l1242_124226

theorem smallest_four_digit_divisible_by_25 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 → n ≥ 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_25_l1242_124226


namespace NUMINAMATH_CALUDE_part_one_part_two_l1242_124228

-- Define the conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one : ∀ x : ℝ, (p 1 x ∨ q x) → (1 < x ∧ x < 3) := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, q x → p a x) ∧ 
  (∃ x : ℝ, p a x ∧ ¬q x) ∧ 
  (a > 0) → 
  (1 ≤ a ∧ a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1242_124228


namespace NUMINAMATH_CALUDE_sara_picked_37_peaches_l1242_124237

/-- The number of peaches Sara picked -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sara picked 37 peaches -/
theorem sara_picked_37_peaches (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 24)
  (h2 : final_peaches = 61) :
  peaches_picked initial_peaches final_peaches = 37 := by
  sorry

#check sara_picked_37_peaches

end NUMINAMATH_CALUDE_sara_picked_37_peaches_l1242_124237


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1242_124209

def ω : ℂ := Complex.I

theorem no_integer_solutions : ∀ a b : ℤ, (Complex.abs (a • ω + b) ≠ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1242_124209


namespace NUMINAMATH_CALUDE_composition_injective_implies_first_injective_l1242_124206

theorem composition_injective_implies_first_injective
  (f g : ℝ → ℝ) (h : Function.Injective (g ∘ f)) :
  Function.Injective f := by
  sorry

end NUMINAMATH_CALUDE_composition_injective_implies_first_injective_l1242_124206


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l1242_124211

theorem sqrt_eight_div_sqrt_two_eq_two : Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l1242_124211


namespace NUMINAMATH_CALUDE_quadratic_positive_range_l1242_124293

def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 3 → quadratic_function a x > 0) ↔ 
  ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_range_l1242_124293


namespace NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l1242_124201

-- Define the properties of triangles
def IsScalene (triangle : Type) : Prop := sorry
def IsEquilateral (triangle : Type) : Prop := sorry
def IsRight (triangle : Type) : Prop := sorry

-- Theorem stating that scalene equilateral triangles cannot exist
theorem no_scalene_equilateral_triangle (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) := by sorry

-- Theorem stating that equilateral right triangles cannot exist
theorem no_equilateral_right_triangle (triangle : Type) :
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

-- Main theorem combining both impossible triangle types
theorem impossible_triangles (triangle : Type) :
  ¬(IsScalene triangle ∧ IsEquilateral triangle) ∧
  ¬(IsEquilateral triangle ∧ IsRight triangle) := by sorry

end NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_equilateral_right_triangle_impossible_triangles_l1242_124201


namespace NUMINAMATH_CALUDE_price_of_33kg_apples_l1242_124280

/-- The price of apples for a given weight, where the first 30 kg have a different price than additional kg. -/
def applePrice (l q : ℚ) (weight : ℚ) : ℚ :=
  if weight ≤ 30 then l * weight
  else l * 30 + q * (weight - 30)

/-- Theorem stating the price of 33 kg of apples -/
theorem price_of_33kg_apples (l q : ℚ) :
  (applePrice l q 15 = 150) →
  (applePrice l q 36 = 366) →
  (applePrice l q 33 = 333) := by
  sorry

end NUMINAMATH_CALUDE_price_of_33kg_apples_l1242_124280


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1242_124229

/-- The minimum value of (x-2)^2 + (y-2)^2 given x-y-1=0 -/
theorem min_distance_to_line (x y : ℝ) (h : x - y - 1 = 0) :
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ 
  ∀ (x' y' : ℝ), x' - y' - 1 = 0 → (x' - 2)^2 + (y' - 2)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1242_124229


namespace NUMINAMATH_CALUDE_equalSideToWidthRatio_l1242_124271

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equalSide : ℝ
  base : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the perimeter of an isosceles triangle -/
def IsoscelesTriangle.perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equalSide + t.base

/-- Theorem: The ratio of the equal side of an isosceles triangle to the width of a rectangle
    is 5/2, given that both shapes have a perimeter of 60 and the rectangle's length is twice its width -/
theorem equalSideToWidthRatio :
  ∀ (r : Rectangle) (t : IsoscelesTriangle),
    r.perimeter = 60 →
    t.perimeter = 60 →
    r.length = 2 * r.width →
    t.equalSide / r.width = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equalSideToWidthRatio_l1242_124271


namespace NUMINAMATH_CALUDE_fraction_simplification_l1242_124272

theorem fraction_simplification (a x : ℝ) (h : a^2 + x^2 ≠ 0) :
  (Real.sqrt (a^2 + x^2) + (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) =
  2 * x^2 / (a^2 + x^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1242_124272


namespace NUMINAMATH_CALUDE_first_tree_groups_count_l1242_124281

/-- Represents the number of years in one ring group -/
def years_per_group : ℕ := 6

/-- Represents the number of ring groups in the second tree -/
def second_tree_groups : ℕ := 40

/-- Represents the age difference between the first and second tree in years -/
def age_difference : ℕ := 180

/-- Calculates the number of ring groups in the first tree -/
def first_tree_groups : ℕ := 
  (second_tree_groups * years_per_group + age_difference) / years_per_group

theorem first_tree_groups_count : first_tree_groups = 70 := by
  sorry

end NUMINAMATH_CALUDE_first_tree_groups_count_l1242_124281


namespace NUMINAMATH_CALUDE_A_divisibility_l1242_124223

/-- Definition of A_l for a prime p > 3 -/
def A (p : ℕ) (l : ℕ) : ℕ :=
  sorry

/-- Theorem statement -/
theorem A_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  (∀ l, 1 ≤ l ∧ l ≤ p - 2 → p ∣ A p l) ∧
  (∀ l, 1 < l ∧ l < p ∧ Odd l → p^2 ∣ A p l) :=
by sorry

end NUMINAMATH_CALUDE_A_divisibility_l1242_124223


namespace NUMINAMATH_CALUDE_test_scores_l1242_124245

theorem test_scores (joao_score claudia_score : ℕ) : 
  (10 ≤ joao_score ∧ joao_score < 100) →  -- João's score is a two-digit number
  (10 ≤ claudia_score ∧ claudia_score < 100) →  -- Cláudia's score is a two-digit number
  claudia_score = joao_score + 13 →  -- Cláudia scored 13 points more than João
  joao_score + claudia_score = 149 →  -- Their combined score is 149
  joao_score = 68 ∧ claudia_score = 81 :=
by sorry

end NUMINAMATH_CALUDE_test_scores_l1242_124245


namespace NUMINAMATH_CALUDE_cos_inequality_solution_set_l1242_124231

theorem cos_inequality_solution_set (x : ℝ) : 
  (Real.cos x + 1/2 ≤ 0) ↔ 
  (∃ k : ℤ, 2*k*Real.pi + 2*Real.pi/3 ≤ x ∧ x ≤ 2*k*Real.pi + 4*Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_cos_inequality_solution_set_l1242_124231


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l1242_124266

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

def probability_all_white : ℚ :=
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

theorem probability_three_white_balls :
  probability_all_white = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l1242_124266


namespace NUMINAMATH_CALUDE_iron_aluminum_weight_difference_l1242_124292

/-- The weight difference between two metal pieces -/
def weight_difference (iron_weight aluminum_weight : Float) : Float :=
  iron_weight - aluminum_weight

/-- Theorem stating the weight difference between iron and aluminum pieces -/
theorem iron_aluminum_weight_difference :
  let iron_weight : Float := 11.17
  let aluminum_weight : Float := 0.83
  weight_difference iron_weight aluminum_weight = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_iron_aluminum_weight_difference_l1242_124292


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1242_124285

theorem complex_fraction_calculation :
  |-(7/2)| * (12/7) / (4/3) / (-3)^2 = 1/2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1242_124285


namespace NUMINAMATH_CALUDE_dining_bill_share_l1242_124244

/-- Given a total bill, number of people, and tip percentage, calculates each person's share --/
def calculate_share (total_bill : ℚ) (num_people : ℕ) (tip_percentage : ℚ) : ℚ :=
  (total_bill * (1 + tip_percentage)) / num_people

/-- Proves that the calculated share for the given conditions is approximately $48.53 --/
theorem dining_bill_share :
  let total_bill : ℚ := 211
  let num_people : ℕ := 5
  let tip_percentage : ℚ := 15 / 100
  abs (calculate_share total_bill num_people tip_percentage - 48.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_share_l1242_124244


namespace NUMINAMATH_CALUDE_quadratic_equation_with_roots_as_coefficients_l1242_124267

/-- A quadratic equation with coefficients a, b, and c, represented as ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a quadratic equation -/
structure Roots where
  x₁ : ℝ
  x₂ : ℝ

/-- Checks if the given roots satisfy the quadratic equation -/
def satisfiesEquation (eq : QuadraticEquation) (roots : Roots) : Prop :=
  eq.a * roots.x₁^2 + eq.b * roots.x₁ + eq.c = 0 ∧
  eq.a * roots.x₂^2 + eq.b * roots.x₂ + eq.c = 0

/-- The theorem stating that given a quadratic equation with its roots as coefficients,
    only two specific equations are valid -/
theorem quadratic_equation_with_roots_as_coefficients
  (eq : QuadraticEquation)
  (roots : Roots)
  (h : satisfiesEquation eq roots)
  (h_coeff : eq.a = 1 ∧ eq.b = roots.x₁ ∧ eq.c = roots.x₂) :
  (eq.a = 1 ∧ eq.b = 0 ∧ eq.c = 0) ∨
  (eq.a = 1 ∧ eq.b = 1 ∧ eq.c = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_roots_as_coefficients_l1242_124267


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l1242_124291

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ((-1 + Complex.I * Real.sqrt 3) / 2)^8 + ((-1 - Complex.I * Real.sqrt 3) / 2)^8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l1242_124291


namespace NUMINAMATH_CALUDE_fraction_zeros_count_l1242_124220

/-- The number of zeros immediately following the decimal point in 1/((6 * 10)^10) -/
def zeros_after_decimal : ℕ := 17

/-- The fraction we're analyzing -/
def fraction : ℚ := 1 / ((6 * 10)^10)

/-- Theorem stating that the number of zeros after the decimal point in the 
    decimal representation of the fraction is equal to zeros_after_decimal -/
theorem fraction_zeros_count : 
  (∃ (n : ℕ) (r : ℚ), fraction * 10^zeros_after_decimal = n + r ∧ 0 < r ∧ r < 1) ∧ 
  (∀ (m : ℕ), m > zeros_after_decimal → ∃ (n : ℕ) (r : ℚ), fraction * 10^m = n + r ∧ r = 0) :=
sorry

end NUMINAMATH_CALUDE_fraction_zeros_count_l1242_124220


namespace NUMINAMATH_CALUDE_no_prime_with_consecutive_squares_l1242_124277

theorem no_prime_with_consecutive_squares (n : ℕ) : 
  Prime n → ¬(∃ a b : ℕ, (2 * n + 1 = a^2) ∧ (3 * n + 1 = b^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_with_consecutive_squares_l1242_124277


namespace NUMINAMATH_CALUDE_opposite_number_problem_l1242_124200

theorem opposite_number_problem (x : ℤ) : (x + 1 = -(- 10)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_problem_l1242_124200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1242_124274

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  first_term : a 1 = 3
  is_arithmetic : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d
  is_geometric : ∃ r ≠ 0, (a 4) ^ 2 = (a 1) * (a 13)

/-- The theorem stating the general formula for the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) : 
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, seq.a n = 2 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1242_124274


namespace NUMINAMATH_CALUDE_number_of_girls_l1242_124242

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36) :
  total_pupils - boys - teachers = 272 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l1242_124242


namespace NUMINAMATH_CALUDE_solar_panel_installation_l1242_124239

theorem solar_panel_installation
  (total_homes : ℕ)
  (panels_per_home : ℕ)
  (shortage : ℕ)
  (h1 : total_homes = 20)
  (h2 : panels_per_home = 10)
  (h3 : shortage = 50)
  : (total_homes * panels_per_home - shortage) / panels_per_home = 15 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_installation_l1242_124239


namespace NUMINAMATH_CALUDE_blood_donation_selection_l1242_124243

theorem blood_donation_selection (m n k : ℕ) (hm : m = 3) (hn : n = 6) (hk : k = 5) :
  (Nat.choose (m + n) k) - (Nat.choose n k) = 120 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_l1242_124243


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_tangent_line_l1242_124240

/-- The ellipse E -/
def E (x y : ℝ) : Prop := y^2 / 8 + x^2 / 4 = 1

/-- The line l -/
def l (x y : ℝ) : Prop := x + y - 3 = 0

/-- The function f -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4

theorem ellipse_line_intersection (A B : ℝ × ℝ) :
  E A.1 A.2 ∧ E B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B ∧
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 2 →
  ∀ x y, l x y ↔ x + y - 3 = 0 :=
sorry

theorem tangent_line (P : ℝ × ℝ) :
  P = (1, 2) ∧ (∀ x, f x = x^2 - 3*x + 4) ∧
  (∀ x y, l x y ↔ x + y - 3 = 0) →
  ∃ a b, f P.1 = P.2 ∧ (deriv f) P.1 = -1 ∧
  ∀ x, f x = x^2 - a*x + b :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_tangent_line_l1242_124240


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_of_10_15_25_l1242_124275

theorem gcf_lcm_sum_of_10_15_25 : ∃ (A B : ℕ),
  (A = Nat.gcd 10 (Nat.gcd 15 25)) ∧
  (B = Nat.lcm 10 (Nat.lcm 15 25)) ∧
  (A + B = 155) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_of_10_15_25_l1242_124275


namespace NUMINAMATH_CALUDE_officer_hopps_ticket_goal_l1242_124238

theorem officer_hopps_ticket_goal :
  let days_in_may : ℕ := 31
  let first_period_days : ℕ := 15
  let first_period_average : ℕ := 8
  let second_period_average : ℕ := 5
  let second_period_days : ℕ := days_in_may - first_period_days
  let first_period_tickets : ℕ := first_period_days * first_period_average
  let second_period_tickets : ℕ := second_period_days * second_period_average
  let total_tickets : ℕ := first_period_tickets + second_period_tickets
  total_tickets = 200 := by
sorry

end NUMINAMATH_CALUDE_officer_hopps_ticket_goal_l1242_124238


namespace NUMINAMATH_CALUDE_sum_of_weighted_variables_l1242_124236

theorem sum_of_weighted_variables (x y z : ℝ) 
  (eq1 : x + y + z = 20) 
  (eq2 : x + 2*y + 3*z = 16) : 
  x + 3*y + 5*z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_weighted_variables_l1242_124236


namespace NUMINAMATH_CALUDE_smallest_number_from_digits_l1242_124204

def digits : List Nat := [2, 0, 1, 6]

def isValidPermutation (n : Nat) : Bool :=
  let digits_n := n.digits 10
  digits_n.length == 4 && digits_n.head? != some 0 && digits_n.toFinset == digits.toFinset

theorem smallest_number_from_digits :
  ∀ n : Nat, isValidPermutation n → 1026 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_from_digits_l1242_124204


namespace NUMINAMATH_CALUDE_combined_share_A_and_C_l1242_124287

def total_amount : ℚ := 15800
def charity_percentage : ℚ := 10 / 100
def savings_percentage : ℚ := 8 / 100
def distribution_ratio : List ℚ := [5, 9, 6, 5]

def remaining_amount : ℚ := total_amount * (1 - charity_percentage - savings_percentage)

def share (ratio : ℚ) : ℚ := (ratio / (distribution_ratio.sum)) * remaining_amount

theorem combined_share_A_and_C : 
  share (distribution_ratio[0]!) + share (distribution_ratio[2]!) = 5700.64 := by
  sorry

end NUMINAMATH_CALUDE_combined_share_A_and_C_l1242_124287


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l1242_124252

theorem sum_of_digits_of_large_number : ∃ S : ℕ, 
  S = 10^2021 - 2021 ∧ 
  (∃ digits : List ℕ, 
    digits.sum = 18185 ∧ 
    digits.all (λ d => d < 10) ∧
    S = digits.foldr (λ d acc => d + 10 * acc) 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l1242_124252


namespace NUMINAMATH_CALUDE_fraction_equality_l1242_124227

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1242_124227


namespace NUMINAMATH_CALUDE_prime_factors_of_n_smallest_prime_factors_difference_l1242_124257

def n : ℕ := 172561

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define the prime factors of n
theorem prime_factors_of_n :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r :=
sorry

-- Prove that the positive difference between the two smallest prime factors is 26
theorem smallest_prime_factors_difference :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r ∧ q - p = 26 :=
sorry

end NUMINAMATH_CALUDE_prime_factors_of_n_smallest_prime_factors_difference_l1242_124257


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1242_124290

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (2 * a 2^2 - 7 * a 2 + 6 = 0) →  -- a_2 is a root of 2x^2 - 7x + 6 = 0
  (2 * a 8^2 - 7 * a 8 + 6 = 0) →  -- a_8 is a root of 2x^2 - 7x + 6 = 0
  (a 1 * a 3 * a 5 * a 7 * a 9 = 9 * Real.sqrt 3 ∨ 
   a 1 * a 3 * a 5 * a 7 * a 9 = -9 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1242_124290


namespace NUMINAMATH_CALUDE_rod_length_proof_l1242_124284

/-- The length of a rod in meters, given the number of pieces and the length of each piece in centimeters. -/
def rod_length_meters (num_pieces : ℕ) (piece_length_cm : ℕ) : ℚ :=
  (num_pieces * piece_length_cm : ℚ) / 100

/-- Theorem stating that a rod from which 45 pieces of 85 cm can be cut is 38.25 meters long. -/
theorem rod_length_proof : rod_length_meters 45 85 = 38.25 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_proof_l1242_124284


namespace NUMINAMATH_CALUDE_pension_fund_strategy_optimizes_portfolio_l1242_124282

/-- Represents different types of assets --/
inductive AssetType
  | DebtAsset
  | EquityAsset

/-- Represents an investment portfolio --/
structure Portfolio where
  debtAssets : ℝ
  equityAssets : ℝ

/-- Represents the investment strategy --/
structure InvestmentStrategy where
  portfolio : Portfolio
  maxEquityProportion : ℝ

/-- Defines the concept of a balanced portfolio --/
def isBalanced (s : InvestmentStrategy) : Prop :=
  s.portfolio.equityAssets / (s.portfolio.debtAssets + s.portfolio.equityAssets) ≤ s.maxEquityProportion

/-- Defines the concept of an optimized portfolio --/
def isOptimized (s : InvestmentStrategy) : Prop :=
  isBalanced s ∧ s.portfolio.equityAssets > 0 ∧ s.portfolio.debtAssets > 0

/-- Main theorem: The investment strategy optimizes the portfolio and balances returns and risks --/
theorem pension_fund_strategy_optimizes_portfolio (s : InvestmentStrategy) 
  (h1 : s.portfolio.debtAssets > 0)
  (h2 : s.portfolio.equityAssets > 0)
  (h3 : s.maxEquityProportion = 0.3)
  (h4 : isBalanced s) :
  isOptimized s :=
sorry


end NUMINAMATH_CALUDE_pension_fund_strategy_optimizes_portfolio_l1242_124282


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l1242_124264

theorem second_term_of_geometric_series 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  (h4 : second_term = a * r) : second_term = 1.875 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l1242_124264


namespace NUMINAMATH_CALUDE_craig_travel_difference_l1242_124260

theorem craig_travel_difference :
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by sorry

end NUMINAMATH_CALUDE_craig_travel_difference_l1242_124260


namespace NUMINAMATH_CALUDE_inequality_proof_l1242_124215

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^5 + b^5 + c^2)) + (1 / (b^5 + c^5 + a^2)) + (1 / (c^5 + a^5 + b^2)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1242_124215


namespace NUMINAMATH_CALUDE_element_in_set_l1242_124222

open Set

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}

theorem element_in_set (M : Set ℕ) (h : (U \ M) = {1, 3, 5}) : 7 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1242_124222


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l1242_124205

theorem sqrt_32_div_sqrt_8_eq_2 : Real.sqrt 32 / Real.sqrt 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_8_eq_2_l1242_124205


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_sqrt3_over_2_l1242_124294

theorem sine_cosine_sum_equals_sqrt3_over_2 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_sqrt3_over_2_l1242_124294


namespace NUMINAMATH_CALUDE_quadratic_shift_l1242_124233

/-- Represents a quadratic function of the form y = (x + a)^2 + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a - shift, b := f.b }

/-- Shifts a quadratic function vertically -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b + shift }

/-- The main theorem stating that shifting y = (x+2)^2 - 3 by 1 unit left and 2 units up
    results in y = (x+3)^2 - 1 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 2 (-3)
  let g := verticalShift (horizontalShift f 1) 2
  g = QuadraticFunction.mk 3 (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l1242_124233


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l1242_124288

/-- Proof that X is paid 120% of Y's pay given the conditions -/
theorem employee_pay_percentage (total pay_y : ℕ) (pay_x : ℕ) 
  (h1 : total = 880)
  (h2 : pay_y = 400)
  (h3 : pay_x + pay_y = total) :
  (pay_x : ℚ) / pay_y = 120 / 100 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l1242_124288


namespace NUMINAMATH_CALUDE_conditional_probability_b_given_a_and_c_l1242_124210

-- Define the sample space and probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Define events as measurable sets
variable (a b c : Set Ω)

-- Define probabilities
variable (pa pb pc pab pac pbc pabc : ℝ)

-- State the theorem
theorem conditional_probability_b_given_a_and_c
  (h_pa : P a = pa)
  (h_pb : P b = pb)
  (h_pc : P c = pc)
  (h_pab : P (a ∩ b) = pab)
  (h_pac : P (a ∩ c) = pac)
  (h_pbc : P (b ∩ c) = pbc)
  (h_pabc : P (a ∩ b ∩ c) = pabc)
  (h_pa_val : pa = 5/23)
  (h_pb_val : pb = 7/23)
  (h_pc_val : pc = 1/23)
  (h_pab_val : pab = 2/23)
  (h_pac_val : pac = 1/23)
  (h_pbc_val : pbc = 1/23)
  (h_pabc_val : pabc = 1/23)
  : P (b ∩ (a ∩ c)) / P (a ∩ c) = 1 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_b_given_a_and_c_l1242_124210


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1242_124298

/-- The infinite series ∑(n=1 to ∞) (n³ + 2n² - n) / (n+3)! converges to 1/6 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (n^3 + 2*n^2 - n : ℚ) / (Nat.factorial (n+3)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1242_124298


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l1242_124225

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 96 * 2^n ∧ ∀ m : ℕ, m > 0 ∧ m < n → 3^(m+1) ≠ 96 * 2^m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l1242_124225


namespace NUMINAMATH_CALUDE_max_perimeter_of_third_rectangle_l1242_124256

-- Define the rectangles
structure Rectangle where
  width : ℕ
  height : ℕ

-- Define the problem setup
def rectangle1 : Rectangle := ⟨70, 110⟩
def rectangle2 : Rectangle := ⟨40, 80⟩

-- Function to calculate perimeter
def perimeter (r : Rectangle) : ℕ :=
  2 * (r.width + r.height)

-- Function to check if three rectangles can form a larger rectangle
def canFormLargerRectangle (r1 r2 r3 : Rectangle) : Prop :=
  (r1.width + r2.width = r3.width ∧ max r1.height r2.height = r3.height) ∨
  (r1.height + r2.height = r3.height ∧ max r1.width r2.width = r3.width) ∨
  (r1.width + r2.height = r3.width ∧ r1.height + r2.width = r3.height) ∨
  (r1.height + r2.width = r3.width ∧ r1.width + r2.height = r3.height)

-- Theorem statement
theorem max_perimeter_of_third_rectangle :
  ∃ (r3 : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r3 ∧
    perimeter r3 = 300 ∧
    ∀ (r : Rectangle), canFormLargerRectangle rectangle1 rectangle2 r →
      perimeter r ≤ 300 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_of_third_rectangle_l1242_124256


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1242_124255

theorem problem_1 : (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (Real.pi / 3) - |1 - Real.sqrt 3| = 3 := by
  sorry

theorem problem_2 : ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (2 / (x + 1) + 1 = x / (x - 1) ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1242_124255


namespace NUMINAMATH_CALUDE_sector_arc_length_l1242_124246

/-- Given a circular sector with area 4 and central angle 2 radians, prove that the length of the arc is 4. -/
theorem sector_arc_length (S : ℝ) (θ : ℝ) (l : ℝ) : 
  S = 4 → θ = 2 → l = S * θ / 2 → l = 4 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1242_124246


namespace NUMINAMATH_CALUDE_min_k_for_inequality_l1242_124213

theorem min_k_for_inequality (x y : ℝ) : 
  x * (x - 1) ≤ y * (1 - y) → 
  (∃ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) ∧ 
   (∀ k' : ℝ, k' < k → ∃ x y : ℝ, x * (x - 1) ≤ y * (1 - y) ∧ x^2 + y^2 > k')) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) → k ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_min_k_for_inequality_l1242_124213


namespace NUMINAMATH_CALUDE_selling_price_correct_l1242_124253

/-- Calculates the selling price of a television after applying discounts -/
def selling_price (a : ℝ) : ℝ :=
  0.9 * (a - 100)

/-- Theorem stating that the selling price function correctly applies the discounts -/
theorem selling_price_correct (a : ℝ) : 
  selling_price a = 0.9 * (a - 100) := by
  sorry

end NUMINAMATH_CALUDE_selling_price_correct_l1242_124253


namespace NUMINAMATH_CALUDE_heathers_weight_l1242_124219

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) 
  (h1 : emily_weight = 9)
  (h2 : weight_difference = 78) :
  emily_weight + weight_difference = 87 := by
  sorry

end NUMINAMATH_CALUDE_heathers_weight_l1242_124219


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1242_124270

theorem inequality_solution_set :
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1242_124270


namespace NUMINAMATH_CALUDE_lcm_18_27_l1242_124268

theorem lcm_18_27 : Nat.lcm 18 27 = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_27_l1242_124268


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1242_124224

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1242_124224


namespace NUMINAMATH_CALUDE_coin_division_l1242_124299

theorem coin_division (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5)
  (h3 : ∀ m : ℕ, m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) :
  n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_coin_division_l1242_124299


namespace NUMINAMATH_CALUDE_remi_water_consumption_l1242_124241

/-- The amount of water Remi drinks in a week, given his bottle capacity, refill frequency, and spills. -/
def water_consumed (bottle_capacity : ℕ) (refills_per_day : ℕ) (days : ℕ) (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  bottle_capacity * refills_per_day * days - (spill1 + spill2)

/-- Theorem stating that Remi drinks 407 ounces of water in 7 days under the given conditions. -/
theorem remi_water_consumption :
  water_consumed 20 3 7 5 8 = 407 := by
  sorry

#eval water_consumed 20 3 7 5 8

end NUMINAMATH_CALUDE_remi_water_consumption_l1242_124241


namespace NUMINAMATH_CALUDE_compound_interest_rate_problem_l1242_124259

theorem compound_interest_rate_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 17640) 
  (h2 : P * (1 + r)^3 = 18522) : 
  (1 + r)^3 / (1 + r)^2 = 18522 / 17640 := by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_problem_l1242_124259


namespace NUMINAMATH_CALUDE_altered_solution_detergent_volume_l1242_124289

/-- Given a cleaning solution with initial ratio of bleach:detergent:disinfectant:water as 2:40:10:100,
    and after altering the solution such that:
    1) The ratio of bleach to detergent is tripled
    2) The ratio of detergent to water is halved
    3) The ratio of disinfectant to bleach is doubled
    If the altered solution contains 300 liters of water, prove that it contains 60 liters of detergent. -/
theorem altered_solution_detergent_volume (b d f w : ℚ) : 
  b / d = 2 / 40 →
  d / w = 40 / 100 →
  f / b = 10 / 2 →
  (3 * b) / d = 3 * (2 / 40) →
  d / w = (1 / 2) * (40 / 100) →
  f / (3 * b) = 2 * (10 / 2) →
  w = 300 →
  d = 60 := by
sorry

end NUMINAMATH_CALUDE_altered_solution_detergent_volume_l1242_124289


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l1242_124250

def matrix1 (a b c d e : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![a, 1, b, 2],
    ![2, 3, 4, 3],
    ![c, 5, d, 3],
    ![2, 4, 1, e]]

def matrix2 (f g h i j k : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![-7, f, -13, 3],
    ![g, -15, h, 2],
    ![3, i, 5, 1],
    ![2, j, 4, k]]

theorem inverse_matrices_sum (a b c d e f g h i j k : ℝ) :
  (matrix1 a b c d e) * (matrix2 f g h i j k) = 1 →
  a + b + c + d + e + f + g + h + i + j + k = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l1242_124250


namespace NUMINAMATH_CALUDE_line_perp_plane_necessity_not_sufficiency_l1242_124247

-- Define the types for lines and planes
variable (L : Type*) [NormedAddCommGroup L] [InnerProductSpace ℝ L]
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]

-- Define the perpendicular relation between lines and between a line and a plane
variable (perpendicular_lines : L → L → Prop)
variable (perpendicular_line_plane : L → P → Prop)

-- Define the containment relation between a line and a plane
variable (contained_in : L → P → Prop)

-- State the theorem
theorem line_perp_plane_necessity_not_sufficiency
  (m n : L) (α : P) (h_contained : contained_in n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ∃ (m' n' : L) (α' : P),
    contained_in n' α' ∧
    perpendicular_lines m' n' ∧
    ¬perpendicular_line_plane m' α' :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_necessity_not_sufficiency_l1242_124247


namespace NUMINAMATH_CALUDE_product_of_four_integers_l1242_124279

theorem product_of_four_integers (A B C D : ℕ+) 
  (sum_eq : A + B + C + D = 100)
  (relation : A + 4 = B + 4 ∧ B + 4 = C + 4 ∧ C + 4 = D * 2) : 
  A * B * C * D = 351232 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l1242_124279


namespace NUMINAMATH_CALUDE_potato_fetching_time_l1242_124203

/-- Represents the problem of calculating how long it takes a dog to fetch a launched potato. -/
theorem potato_fetching_time 
  (football_fields : ℕ) -- number of football fields the potato is launched
  (yards_per_field : ℕ) -- length of a football field in yards
  (dog_speed : ℕ) -- dog's speed in feet per minute
  (h1 : football_fields = 6)
  (h2 : yards_per_field = 200)
  (h3 : dog_speed = 400) :
  (football_fields * yards_per_field * 3) / dog_speed = 9 := by
  sorry

#check potato_fetching_time

end NUMINAMATH_CALUDE_potato_fetching_time_l1242_124203


namespace NUMINAMATH_CALUDE_secretary_work_time_l1242_124218

theorem secretary_work_time 
  (ratio : Fin 3 → ℕ)
  (total_time : ℕ) :
  ratio 0 = 2 →
  ratio 1 = 3 →
  ratio 2 = 5 →
  total_time = 110 →
  (ratio 0 + ratio 1 + ratio 2) * (total_time / (ratio 0 + ratio 1 + ratio 2)) = total_time →
  ratio 2 * (total_time / (ratio 0 + ratio 1 + ratio 2)) = 55 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_time_l1242_124218


namespace NUMINAMATH_CALUDE_sugar_loss_calculation_l1242_124208

/-- Given an initial amount of sugar, number of bags, and loss percentage,
    calculate the remaining amount of sugar. -/
def remaining_sugar (initial_sugar : ℝ) (num_bags : ℕ) (loss_percent : ℝ) : ℝ :=
  initial_sugar * (1 - loss_percent)

/-- Theorem: Given 24 kilos of sugar divided equally into 4 bags,
    with 15% loss in each bag, the total remaining sugar is 20.4 kilos. -/
theorem sugar_loss_calculation : remaining_sugar 24 4 0.15 = 20.4 := by
  sorry

#check sugar_loss_calculation

end NUMINAMATH_CALUDE_sugar_loss_calculation_l1242_124208


namespace NUMINAMATH_CALUDE_max_sin_sum_60_degrees_l1242_124232

open Real

theorem max_sin_sum_60_degrees (x y : ℝ) : 
  0 < x → x < π/2 →
  0 < y → y < π/2 →
  x + y = π/3 →
  (∀ a b : ℝ, 0 < a → a < π/2 → 0 < b → b < π/2 → a + b = π/3 → sin a + sin b ≤ sin x + sin y) →
  sin x + sin y = 1 := by
sorry


end NUMINAMATH_CALUDE_max_sin_sum_60_degrees_l1242_124232


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l1242_124265

theorem ratio_of_fractions (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 7) * B) : 
  A / B = 9 / 14 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l1242_124265


namespace NUMINAMATH_CALUDE_range_of_a_l1242_124217

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1242_124217


namespace NUMINAMATH_CALUDE_new_apples_grown_l1242_124235

theorem new_apples_grown (initial_apples picked_apples current_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : picked_apples = 7)
  (h3 : current_apples = 6) :
  current_apples - (initial_apples - picked_apples) = 2 :=
by sorry

end NUMINAMATH_CALUDE_new_apples_grown_l1242_124235


namespace NUMINAMATH_CALUDE_room_dimension_l1242_124221

/-- Proves that a square room with an area of 14400 square inches has sides of length 10 feet, given that there are 12 inches in a foot. -/
theorem room_dimension (inches_per_foot : ℕ) (area_sq_inches : ℕ) : 
  inches_per_foot = 12 → 
  area_sq_inches = 14400 → 
  ∃ (side_length : ℕ), side_length * side_length * (inches_per_foot * inches_per_foot) = area_sq_inches ∧ 
                        side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_dimension_l1242_124221
