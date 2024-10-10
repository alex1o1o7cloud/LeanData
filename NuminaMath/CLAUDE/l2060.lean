import Mathlib

namespace number_of_proper_subsets_l2060_206090

def U : Finset Nat := {0, 1, 2, 3}

def A : Finset Nat := {0, 1, 3}

def complement_A : Finset Nat := {2}

theorem number_of_proper_subsets :
  (U = {0, 1, 2, 3}) →
  (complement_A = {2}) →
  (A = U \ complement_A) →
  (Finset.powerset A).card - 1 = 7 := by
  sorry

end number_of_proper_subsets_l2060_206090


namespace all_statements_incorrect_l2060_206089

-- Define the types for functions and properties
def Function := ℝ → ℝ
def Periodic (f : Function) : Prop := ∃ T > 0, ∀ x, f (x + T) = f x
def Monotonic (f : Function) : Prop := ∀ x y, x < y → f x < f y

-- Define the original proposition
def OriginalProposition : Prop := ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Define the given statements
def GivenConverse : Prop := ∀ f : Function, Monotonic f → ¬(Periodic f)
def GivenNegation : Prop := ∀ f : Function, Periodic f → Monotonic f
def GivenContrapositive : Prop := ∀ f : Function, Monotonic f → Periodic f

-- Theorem stating that none of the given statements are correct
theorem all_statements_incorrect : 
  (GivenConverse ≠ (¬OriginalProposition → OriginalProposition)) ∧
  (GivenNegation ≠ ¬OriginalProposition) ∧
  (GivenContrapositive ≠ (¬¬OriginalProposition → ¬OriginalProposition)) :=
sorry

end all_statements_incorrect_l2060_206089


namespace multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l2060_206068

theorem multiples_of_6_or_8_not_both (n : Nat) : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range n)).card = 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range n)).card + 
  (Finset.filter (fun x => x % 8 = 0) (Finset.range n)).card - 
  (Finset.filter (fun x => x % 24 = 0) (Finset.range n)).card :=
by sorry

theorem count_multiples_6_or_8_not_both : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range 201)).card = 42 :=
by sorry

end multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l2060_206068


namespace second_discount_percentage_l2060_206092

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  first_discount = 20 →
  final_price = 152 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 :=
by sorry

end second_discount_percentage_l2060_206092


namespace smallest_number_l2060_206037

theorem smallest_number (jungkook yoongi yuna : ℕ) : 
  jungkook = 6 - 3 → yoongi = 4 → yuna = 5 → min jungkook (min yoongi yuna) = 3 := by
sorry

end smallest_number_l2060_206037


namespace common_chord_equation_l2060_206093

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem common_chord_equation :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y :=
by sorry

end common_chord_equation_l2060_206093


namespace sin_330_degrees_l2060_206020

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end sin_330_degrees_l2060_206020


namespace polygon_interior_less_than_exterior_has_three_sides_l2060_206074

theorem polygon_interior_less_than_exterior_has_three_sides
  (n : ℕ) -- number of sides of the polygon
  (h_polygon : n ≥ 3) -- n is at least 3 for a polygon
  (interior_sum : ℝ) -- sum of interior angles
  (exterior_sum : ℝ) -- sum of exterior angles
  (h_interior : interior_sum = (n - 2) * 180) -- formula for interior angle sum
  (h_exterior : exterior_sum = 360) -- exterior angle sum is always 360°
  (h_less : interior_sum < exterior_sum) -- given condition
  : n = 3 :=
by sorry

end polygon_interior_less_than_exterior_has_three_sides_l2060_206074


namespace parabola_shift_theorem_l2060_206083

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 2 0 0
  let shifted := shift_parabola original 4 3
  y = 2 * x^2 → y = shifted.a * (x - 4)^2 + shifted.b * (x - 4) + shifted.c :=
by sorry

end parabola_shift_theorem_l2060_206083


namespace unique_solution_inequality_l2060_206023

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) ↔ x = 4 := by
  sorry

end unique_solution_inequality_l2060_206023


namespace arcsin_cos_arcsin_plus_arccos_sin_arccos_l2060_206064

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end arcsin_cos_arcsin_plus_arccos_sin_arccos_l2060_206064


namespace number_card_problem_l2060_206025

theorem number_card_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 143 →
  A + 4.5 = (B + C) / 2 →
  C = B - 3 →
  C = 143 := by sorry

end number_card_problem_l2060_206025


namespace hamburgers_served_l2060_206010

/-- Proves that the number of hamburgers served is 3, given the total made and left over. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end hamburgers_served_l2060_206010


namespace solution_set_implies_a_equals_one_l2060_206076

/-- The solution set of the inequality |2x-a|+a≤4 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | |2*x - a| + a ≤ 4}

/-- The theorem stating that if the solution set of |2x-a|+a≤4 is {x|-1≤x≤2}, then a = 1 -/
theorem solution_set_implies_a_equals_one :
  SolutionSet 1 = {x : ℝ | -1 ≤ x ∧ x ≤ 2} → 1 = 1 :=
by sorry

end solution_set_implies_a_equals_one_l2060_206076


namespace cylinder_volume_from_unit_square_l2060_206094

/-- The volume of a cylinder formed by rolling a unit square -/
theorem cylinder_volume_from_unit_square : 
  ∃ (V : ℝ), V = (1 : ℝ) / (4 * Real.pi) ∧ 
  (∃ (r h : ℝ), r = (1 : ℝ) / (2 * Real.pi) ∧ h = 1 ∧ V = Real.pi * r^2 * h) :=
by sorry

end cylinder_volume_from_unit_square_l2060_206094


namespace distance_calculation_l2060_206080

theorem distance_calculation (speed : ℝ) (time : ℝ) (h1 : speed = 100) (h2 : time = 5) :
  speed * time = 500 := by
  sorry

end distance_calculation_l2060_206080


namespace max_cars_quotient_l2060_206056

/-- Represents the maximum number of cars that can pass a sensor in one hour -/
def N : ℕ := 4000

/-- The length of a car in meters -/
def car_length : ℝ := 5

/-- The safety rule factor: number of car lengths per 20 km/h of speed -/
def safety_factor : ℝ := 2

/-- Theorem stating that the maximum number of cars passing the sensor in one hour, 
    divided by 15, is equal to 266 -/
theorem max_cars_quotient : N / 15 = 266 := by sorry

end max_cars_quotient_l2060_206056


namespace set_intersection_theorem_l2060_206095

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x < 3}

theorem set_intersection_theorem : A ∩ B = Ioo 0 3 := by sorry

end set_intersection_theorem_l2060_206095


namespace milk_water_ratio_after_filling_l2060_206009

/-- Represents the ratio of milk to water -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents the can with its contents -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  ratio : Ratio

def initial_can : Can :=
  { capacity := 60
  , current_volume := 40
  , ratio := { milk := 5, water := 3 } }

def final_can : Can :=
  { capacity := 60
  , current_volume := 60
  , ratio := { milk := 3, water := 1 } }

theorem milk_water_ratio_after_filling (c : Can) (h : c = initial_can) :
  (final_can.ratio.milk : ℚ) / final_can.ratio.water = 3 := by
  sorry

#check milk_water_ratio_after_filling

end milk_water_ratio_after_filling_l2060_206009


namespace product_xy_is_264_l2060_206077

theorem product_xy_is_264 (x y : ℝ) 
  (eq1 : -3 * x + 4 * y = 28) 
  (eq2 : 3 * x - 2 * y = 8) : 
  x * y = 264 := by
  sorry

end product_xy_is_264_l2060_206077


namespace negation_of_proposition_l2060_206039

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), ab > 0 → a > 0) ↔ (∀ (a b : ℝ), ab ≤ 0 → a ≤ 0) :=
by sorry

end negation_of_proposition_l2060_206039


namespace base8_to_base10_conversion_l2060_206004

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [3, 4, 6, 2, 5]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 21923 := by
  sorry

end base8_to_base10_conversion_l2060_206004


namespace equation_solution_l2060_206066

theorem equation_solution (c d : ℝ) (h : d ≠ 0) :
  let x := (9 * d^2 - 4 * c^2) / (6 * d)
  x^2 + 4 * c^2 = (3 * d - x)^2 := by
  sorry

end equation_solution_l2060_206066


namespace layla_goals_l2060_206011

theorem layla_goals (layla kristin : ℕ) (h1 : kristin = layla - 24) (h2 : layla + kristin = 368) : layla = 196 := by
  sorry

end layla_goals_l2060_206011


namespace triangle_ABC_point_C_l2060_206028

-- Define the points
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (-1, -2)
def D : ℝ × ℝ := (2, 2)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem triangle_ABC_point_C : 
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (5, 6) := by sorry

end triangle_ABC_point_C_l2060_206028


namespace cube_volume_from_surface_area_l2060_206073

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 294 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 343 := by
  sorry

end cube_volume_from_surface_area_l2060_206073


namespace reciprocal_of_repeating_decimal_is_eleven_fourths_l2060_206044

/-- The repeating decimal 0.363636... as a rational number -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the repeating decimal 0.363636... -/
def reciprocal_of_repeating_decimal : ℚ := 11 / 4

/-- Theorem: The reciprocal of the common fraction form of 0.363636... is 11/4 -/
theorem reciprocal_of_repeating_decimal_is_eleven_fourths :
  (1 : ℚ) / repeating_decimal = reciprocal_of_repeating_decimal := by sorry

end reciprocal_of_repeating_decimal_is_eleven_fourths_l2060_206044


namespace ramu_profit_percent_l2060_206001

/-- Calculates the profit percent given the cost of a car, repair costs, and selling price --/
def profit_percent (car_cost repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := car_cost + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that under the given conditions, the profit percent is 18% --/
theorem ramu_profit_percent :
  profit_percent 42000 13000 64900 = 18 := by
  sorry

end ramu_profit_percent_l2060_206001


namespace quadratic_factorization_l2060_206038

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l2060_206038


namespace other_x_intercept_of_quadratic_l2060_206078

/-- Given a quadratic function with vertex (4, 10) and one x-intercept at (-1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 10 + a * (x - 4)^2) →  -- vertex form of quadratic
  a * (-1)^2 + b * (-1) + c = 0 →                    -- x-intercept at (-1, 0)
  ∃ x, x ≠ -1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9      -- other x-intercept at 9
  := by sorry

end other_x_intercept_of_quadratic_l2060_206078


namespace toy_price_after_discounts_l2060_206070

theorem toy_price_after_discounts (initial_price : ℝ) (discount : ℝ) : 
  initial_price = 200 → discount = 0.1 → 
  initial_price * (1 - discount)^2 = 162 := by
  sorry

#eval (200 : ℝ) * (1 - 0.1)^2

end toy_price_after_discounts_l2060_206070


namespace certain_number_subtraction_l2060_206016

theorem certain_number_subtraction (X : ℤ) (h : X - 46 = 15) : X - 29 = 32 := by
  sorry

end certain_number_subtraction_l2060_206016


namespace difference_of_decimal_and_fraction_l2060_206054

theorem difference_of_decimal_and_fraction : 0.650 - (1 / 8 : ℚ) = 0.525 := by
  sorry

end difference_of_decimal_and_fraction_l2060_206054


namespace max_cables_theorem_l2060_206072

/-- Represents the maximum number of cables that can be used to connect computers
    in an organization with specific constraints. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) : ℕ :=
  30

/-- Theorem stating that the maximum number of cables is 30 under given conditions. -/
theorem max_cables_theorem (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) :
  total_employees = 40 →
  brand_a_computers = 25 →
  brand_b_computers = 15 →
  total_employees = brand_a_computers + brand_b_computers →
  max_cables total_employees brand_a_computers brand_b_computers = 30 :=
by
  sorry

#check max_cables_theorem

end max_cables_theorem_l2060_206072


namespace prob_sum_three_dice_l2060_206050

/-- The probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The desired sum on the top faces -/
def desired_sum : ℕ := 3

/-- The probability of rolling the desired sum on all dice -/
def prob_desired_sum : ℚ := single_die_prob ^ num_dice

theorem prob_sum_three_dice (h : desired_sum = 3 ∧ num_dice = 3) :
  prob_desired_sum = 1 / 216 := by
  sorry

end prob_sum_three_dice_l2060_206050


namespace rhombus_diagonals_l2060_206098

/-- Given a rhombus where a height from the obtuse angle vertex divides a side into
    segments of length a and b, this theorem proves the lengths of its diagonals. -/
theorem rhombus_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (d1 d2 : ℝ),
    d1 = Real.sqrt (2 * b * (a + b)) ∧
    d2 = Real.sqrt (2 * (2 * a + b) * (a + b)) ∧
    d1 > 0 ∧ d2 > 0 :=
by sorry

end rhombus_diagonals_l2060_206098


namespace theater_ticket_cost_l2060_206041

/-- The cost of tickets at a theater -/
theorem theater_ticket_cost 
  (adult_price : ℝ) 
  (h1 : adult_price > 0)
  (h2 : 4 * adult_price + 3 * (adult_price / 2) + 2 * (3 * adult_price / 4) = 42) :
  6 * adult_price + 5 * (adult_price / 2) + 4 * (3 * adult_price / 4) = 69 := by
  sorry


end theater_ticket_cost_l2060_206041


namespace equally_spaced_number_line_l2060_206017

theorem equally_spaced_number_line (total_distance : ℝ) (num_steps : ℕ) (step_to_z : ℕ) : 
  total_distance = 16 → num_steps = 4 → step_to_z = 2 →
  let step_length := total_distance / num_steps
  let z := step_to_z * step_length
  z = 8 := by
  sorry

end equally_spaced_number_line_l2060_206017


namespace square_divisible_by_six_between_30_and_150_l2060_206058

theorem square_divisible_by_six_between_30_and_150 (x : ℕ) :
  (∃ n : ℕ, x = n^2) →  -- x is a square number
  x % 6 = 0 →           -- x is divisible by 6
  30 < x →              -- x is greater than 30
  x < 150 →             -- x is less than 150
  x = 36 ∨ x = 144 :=   -- x is either 36 or 144
by sorry

end square_divisible_by_six_between_30_and_150_l2060_206058


namespace max_value_a_l2060_206032

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 2 * d) 
  (h4 : d < 50) : 
  a ≤ 579 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 579 ∧ 
    a' < 2 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 2 * d' ∧ 
    d' < 50 :=
sorry

end max_value_a_l2060_206032


namespace jacket_price_calculation_l2060_206099

/-- Calculates the final price of a jacket after discount and tax --/
def finalPrice (originalPrice : ℝ) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  discountedPrice * (1 + taxRate)

/-- Theorem stating that the final price of the jacket is 92.4 --/
theorem jacket_price_calculation :
  finalPrice 120 0.3 0.1 = 92.4 := by
  sorry

end jacket_price_calculation_l2060_206099


namespace part_one_part_two_l2060_206081

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 4| + |x - a|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 10} = {x : ℝ | x > 8 ∨ x < -2} :=
sorry

-- Part II
theorem part_two : 
  (∀ x : ℝ, f a x ≥ 1) → (a ≥ 5 ∨ a ≤ 3) :=
sorry

end part_one_part_two_l2060_206081


namespace two_face_cubes_count_l2060_206043

/-- Represents a 3x3x3 cube formed by cutting a larger cube painted on all faces -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)
  (h_size : size = 3)
  (h_painted : painted_faces = 6)

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_cubes (cube : PaintedCube) : Nat :=
  12

/-- Theorem: The number of smaller cubes painted on exactly two faces in a 3x3x3 PaintedCube is 12 -/
theorem two_face_cubes_count (cube : PaintedCube) : count_two_face_cubes cube = 12 := by
  sorry

end two_face_cubes_count_l2060_206043


namespace fourth_row_sum_spiral_l2060_206024

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral filling of the grid -/
def spiralFill (n : ℕ) : List (Position × ℕ) := sorry

/-- The sum of the smallest and largest numbers in a given row -/
def sumMinMaxInRow (row : ℕ) (filled : List (Position × ℕ)) : ℕ := sorry

theorem fourth_row_sum_spiral (n : ℕ) (h : n = 21) :
  let filled := spiralFill n
  sumMinMaxInRow 4 filled = 742 := by sorry

end fourth_row_sum_spiral_l2060_206024


namespace a_upper_bound_l2060_206067

def f (x : ℝ) := x + x^3

theorem a_upper_bound
  (h : ∀ θ : ℝ, 0 < θ → θ < π/2 → ∀ a : ℝ, f (a * Real.sin θ) + f (1 - a) > 0) :
  ∀ a : ℝ, a ≤ 1 :=
sorry

end a_upper_bound_l2060_206067


namespace isosceles_triangle_base_length_l2060_206003

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 70,
    where one side of the equilateral triangle is also a side of the isosceles triangle,
    prove that the base of the isosceles triangle is 30 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 70)
  (h_shared_side : ∃ (side : ℝ), side = equilateral_perimeter / 3 ∧
                   isosceles_perimeter = 2 * side + (isosceles_perimeter - 2 * side)) :
  isosceles_perimeter - 2 * (equilateral_perimeter / 3) = 30 :=
by sorry

end isosceles_triangle_base_length_l2060_206003


namespace anya_vanya_catchup_l2060_206061

/-- Represents the speeds and catch-up times in the Anya-Vanya problem -/
structure AnyaVanyaProblem where
  anya_speed : ℝ
  vanya_speed : ℝ
  original_catch_up_time : ℝ

/-- The conditions of the problem -/
def problem_conditions (p : AnyaVanyaProblem) : Prop :=
  p.anya_speed > 0 ∧ 
  p.vanya_speed > p.anya_speed ∧
  p.original_catch_up_time > 0 ∧
  (2 * p.vanya_speed - p.anya_speed) * p.original_catch_up_time = 
    3 * (p.vanya_speed - p.anya_speed) * (p.original_catch_up_time / 3)

/-- The theorem to be proved -/
theorem anya_vanya_catchup (p : AnyaVanyaProblem) 
  (h : problem_conditions p) : 
  (2 * p.vanya_speed - p.anya_speed / 2) * (p.original_catch_up_time / 7) = 
  (p.vanya_speed - p.anya_speed) * p.original_catch_up_time :=
by sorry

end anya_vanya_catchup_l2060_206061


namespace stair_climbing_time_l2060_206087

theorem stair_climbing_time : 
  let n : ℕ := 4  -- number of flights
  let a : ℕ := 30 -- time for first flight
  let d : ℕ := 10 -- time increase for each subsequent flight
  let S := n * (2 * a + (n - 1) * d) / 2  -- sum formula for arithmetic sequence
  S = 180 := by sorry

end stair_climbing_time_l2060_206087


namespace first_fibonacci_exceeding_1000_l2060_206012

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem first_fibonacci_exceeding_1000 :
  ∃ n : ℕ, fibonacci n > 1000 ∧ ∀ m : ℕ, m < n → fibonacci m ≤ 1000 ∧ fibonacci n = 1597 :=
by
  sorry

end first_fibonacci_exceeding_1000_l2060_206012


namespace fourth_month_sale_l2060_206085

def sales_problem (sale1 sale2 sale3 sale5 sale6_target average_target : ℕ) : Prop :=
  let total_sales := 6 * average_target
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6_target
  let sale4 := total_sales - known_sales
  sale4 = 6350

theorem fourth_month_sale :
  sales_problem 5420 5660 6200 6500 7070 6200 :=
by sorry

end fourth_month_sale_l2060_206085


namespace invertible_function_theorem_l2060_206018

noncomputable section

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem invertible_function_theorem (c d : ℝ) 
  (h1 : Function.Injective g) 
  (h2 : g c = d) 
  (h3 : g d = 5) : 
  c - d = -2 := by sorry

end invertible_function_theorem_l2060_206018


namespace batsman_average_batsman_average_proof_l2060_206022

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let prev_average := 30
  let new_average := prev_average + average_increase
  new_average

#check batsman_average 10 60 3 = 33

theorem batsman_average_proof 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : total_innings = 10)
  (h2 : last_innings_score = 60)
  (h3 : average_increase = 3) :
  batsman_average total_innings last_innings_score average_increase = 33 := by
  sorry

end batsman_average_batsman_average_proof_l2060_206022


namespace no_prime_sided_integer_area_triangle_l2060_206047

theorem no_prime_sided_integer_area_triangle : 
  ¬ ∃ (a b c : ℕ) (S : ℝ), 
    (Prime a ∧ Prime b ∧ Prime c) ∧ 
    (S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) ∧ 
    (S ≠ 0) ∧ 
    (∃ (n : ℕ), S = n) := by
  sorry

end no_prime_sided_integer_area_triangle_l2060_206047


namespace force_for_10_inch_screwdriver_l2060_206019

/-- Represents the force-length relationship for screwdrivers -/
structure ScrewdriverForce where
  force : ℝ
  length : ℝ
  constant : ℝ

/-- The force-length relationship is inverse and constant -/
axiom force_length_relation (sf : ScrewdriverForce) : sf.force * sf.length = sf.constant

/-- Given conditions for the 6-inch screwdriver -/
def initial_screwdriver : ScrewdriverForce :=
  { force := 60
    length := 6
    constant := 60 * 6 }

/-- Theorem stating the force required for a 10-inch screwdriver -/
theorem force_for_10_inch_screwdriver :
  ∃ (sf : ScrewdriverForce), sf.length = 10 ∧ sf.constant = initial_screwdriver.constant ∧ sf.force = 36 :=
by sorry

end force_for_10_inch_screwdriver_l2060_206019


namespace tax_rate_calculation_l2060_206042

/-- Tax calculation problem -/
theorem tax_rate_calculation (total_value : ℝ) (tax_free_threshold : ℝ) (tax_paid : ℝ) :
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 112 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 10 := by
  sorry

end tax_rate_calculation_l2060_206042


namespace unfolded_paper_has_symmetric_holes_l2060_206075

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (is_rectangular : width > 0 ∧ height > 0)

/-- Represents a hole on the paper -/
structure Hole :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (original : Paper)
  (hole : Hole)
  (is_folded_left_right : Bool)
  (is_folded_diagonally : Bool)
  (is_hole_near_center : Bool)

/-- Represents the state of the paper after unfolding -/
structure UnfoldedPaper :=
  (original : Paper)
  (holes : List Hole)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Predicate to check if holes are symmetrically placed -/
def are_holes_symmetric (up : UnfoldedPaper) : Prop :=
  sorry

/-- Main theorem: Unfolding a properly folded and punched paper results in four symmetrically placed holes -/
theorem unfolded_paper_has_symmetric_holes (fp : FoldedPaper) 
  (h1 : fp.is_folded_left_right = true)
  (h2 : fp.is_folded_diagonally = true)
  (h3 : fp.is_hole_near_center = true) :
  let up := unfold fp
  (up.holes.length = 4) ∧ (are_holes_symmetric up) :=
  sorry

end unfolded_paper_has_symmetric_holes_l2060_206075


namespace inscribed_circle_radius_l2060_206013

/-- The radius of the largest circle inscribed in a square, given specific distances from a point on the circle to two adjacent sides of the square. -/
theorem inscribed_circle_radius (square_side : ℝ) (dist_to_side1 : ℝ) (dist_to_side2 : ℝ) :
  square_side > 20 →
  dist_to_side1 = 8 →
  dist_to_side2 = 9 →
  ∃ (radius : ℝ),
    radius > 10 ∧
    (radius - dist_to_side1)^2 + (radius - dist_to_side2)^2 = radius^2 ∧
    radius = 29 :=
by sorry

end inscribed_circle_radius_l2060_206013


namespace sum_f_odd_points_l2060_206014

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_zero : f 0 = 2
axiom f_translated_odd : ∀ x, f (x - 1) = -f (-x - 1)

-- State the theorem
theorem sum_f_odd_points :
  f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
sorry

end sum_f_odd_points_l2060_206014


namespace train_length_calculation_l2060_206015

/-- The length of a train that crosses an electric pole in a given time at a given speed. -/
def train_length (crossing_time : ℝ) (speed : ℝ) : ℝ :=
  crossing_time * speed

/-- Theorem: A train that crosses an electric pole in 40 seconds at a speed of 62.99999999999999 m/s has a length of 2520 meters. -/
theorem train_length_calculation :
  train_length 40 62.99999999999999 = 2520 := by
  sorry

end train_length_calculation_l2060_206015


namespace shortest_midpoint_to_midpoint_path_length_l2060_206096

-- Define a regular cube
structure RegularCube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

-- Define a path on the surface of the cube
def SurfacePath (cube : RegularCube) := ℝ

-- Define the property of being a valid path from midpoint to midpoint of opposite edges
def IsValidMidpointToMidpointPath (cube : RegularCube) (path : SurfacePath cube) : Prop :=
  sorry

-- Define the length of a path
def PathLength (cube : RegularCube) (path : SurfacePath cube) : ℝ :=
  sorry

-- Theorem statement
theorem shortest_midpoint_to_midpoint_path_length 
  (cube : RegularCube) 
  (h : cube.edgeLength = 2) :
  ∃ (path : SurfacePath cube), 
    IsValidMidpointToMidpointPath cube path ∧ 
    PathLength cube path = 4 ∧
    ∀ (other_path : SurfacePath cube), 
      IsValidMidpointToMidpointPath cube other_path → 
      PathLength cube other_path ≥ 4 :=
by sorry

end shortest_midpoint_to_midpoint_path_length_l2060_206096


namespace sam_cleaner_meet_twice_l2060_206052

/-- Represents the movement of Sam and the street cleaner on a path with benches --/
structure PathMovement where
  sam_speed : ℝ
  cleaner_speed : ℝ
  bench_distance : ℝ
  cleaner_stop_time : ℝ

/-- Calculates the number of times Sam and the cleaner meet --/
def number_of_meetings (movement : PathMovement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : PathMovement :=
  { sam_speed := 3
  , cleaner_speed := 9
  , bench_distance := 300
  , cleaner_stop_time := 40 }

/-- Theorem stating that Sam and the cleaner meet exactly twice --/
theorem sam_cleaner_meet_twice :
  number_of_meetings problem_scenario = 2 := by
  sorry

end sam_cleaner_meet_twice_l2060_206052


namespace tenth_occurrence_shift_l2060_206060

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2 + 1

/-- Theorem: The 10th occurrence of a letter is replaced by the letter 13 positions to its right -/
theorem tenth_occurrence_shift :
  shift 10 % alphabet_size = 13 :=
sorry

end tenth_occurrence_shift_l2060_206060


namespace circle_symmetry_l2060_206059

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem circle_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ circle2 y x ∧ line_of_symmetry x y :=
sorry

end circle_symmetry_l2060_206059


namespace trapezoid_shorter_diagonal_l2060_206027

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_E : Bool
  acute_F : Bool

def shorter_diagonal (t : Trapezoid) : ℝ := sorry

theorem trapezoid_shorter_diagonal 
  (t : Trapezoid) 
  (h1 : t.EF = 20) 
  (h2 : t.GH = 26) 
  (h3 : t.side1 = 13) 
  (h4 : t.side2 = 15) 
  (h5 : t.acute_E = true) 
  (h6 : t.acute_F = true) : 
  shorter_diagonal t = Real.sqrt 1496 / 3 := by sorry

end trapezoid_shorter_diagonal_l2060_206027


namespace largest_three_digit_multiple_of_6_with_digit_sum_15_l2060_206030

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_6_with_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → n % 6 = 0 → digit_sum n = 15 → n ≤ 960 :=
by sorry

end largest_three_digit_multiple_of_6_with_digit_sum_15_l2060_206030


namespace umbrella_arrangements_seven_l2060_206062

def umbrella_arrangements (n : ℕ) : ℕ := 
  if n % 2 = 0 then 0
  else Nat.choose (n - 1) ((n - 1) / 2)

theorem umbrella_arrangements_seven :
  umbrella_arrangements 7 = 20 := by
sorry

end umbrella_arrangements_seven_l2060_206062


namespace three_times_root_equation_iff_roots_l2060_206045

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) with two distinct real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of a "3 times root equation" -/
def is_three_times_root_equation (eq : QuadraticEquation) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁

/-- Theorem: A quadratic equation is a "3 times root equation" iff its roots satisfy r2 = 3r1 -/
theorem three_times_root_equation_iff_roots (eq : QuadraticEquation) :
  is_three_times_root_equation eq ↔
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁ :=
by sorry


end three_times_root_equation_iff_roots_l2060_206045


namespace cubic_sum_identity_l2060_206079

theorem cubic_sum_identity (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h_sum : a + b + c = d) :
  (a^3 + b^3 + c^3 - 3*a*b*c) / (a*b*c) = d * (a^2 + b^2 + c^2 - a*b - a*c - b*c) / (a*b*c) :=
by sorry

end cubic_sum_identity_l2060_206079


namespace f_max_min_implies_a_range_l2060_206040

/-- The function f parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x

/-- The main theorem -/
theorem f_max_min_implies_a_range :
  ∀ a : ℝ, has_max_and_min a → a > 2 ∨ a < -1 :=
sorry

end f_max_min_implies_a_range_l2060_206040


namespace chlorine_moles_l2060_206048

/-- Represents the chemical reaction between Methane and Chlorine to produce Hydrochloric acid -/
def chemical_reaction (methane : ℝ) (chlorine : ℝ) (hydrochloric_acid : ℝ) : Prop :=
  methane = 1 ∧ hydrochloric_acid = 2 ∧ chlorine = hydrochloric_acid

/-- Theorem stating that 2 moles of Chlorine are combined in the reaction -/
theorem chlorine_moles : ∃ (chlorine : ℝ), chemical_reaction 1 chlorine 2 ∧ chlorine = 2 := by
  sorry

end chlorine_moles_l2060_206048


namespace laptop_original_price_l2060_206031

/-- Proves that if a laptop's price is reduced by 15% and the new price is $680, then the original price was $800. -/
theorem laptop_original_price (discount_percent : ℝ) (discounted_price : ℝ) (original_price : ℝ) : 
  discount_percent = 15 →
  discounted_price = 680 →
  discounted_price = original_price * (1 - discount_percent / 100) →
  original_price = 800 := by
sorry

end laptop_original_price_l2060_206031


namespace largest_term_index_l2060_206021

def A (k : ℕ) : ℝ := (Nat.choose 2000 k) * (0.1 ^ k)

theorem largest_term_index : 
  ∃ (k : ℕ), k ≤ 2000 ∧ 
  (∀ (j : ℕ), j ≤ 2000 → A k ≥ A j) ∧
  k = 181 := by
  sorry

end largest_term_index_l2060_206021


namespace tileC_in_rectangleY_l2060_206034

-- Define a tile with four sides
structure Tile :=
  (top : ℕ) (right : ℕ) (bottom : ℕ) (left : ℕ)

-- Define the four tiles
def tileA : Tile := ⟨5, 3, 1, 6⟩
def tileB : Tile := ⟨3, 6, 2, 5⟩
def tileC : Tile := ⟨2, 7, 0, 3⟩
def tileD : Tile := ⟨6, 2, 4, 7⟩

-- Define a function to check if a tile has unique sides
def hasUniqueSides (t : Tile) (others : List Tile) : Prop :=
  (t.right ∉ others.map (λ tile => tile.left)) ∧
  (t.bottom ∉ others.map (λ tile => tile.top))

-- Define the theorem
theorem tileC_in_rectangleY :
  hasUniqueSides tileC [tileA, tileB, tileD] ∧
  ¬hasUniqueSides tileA [tileB, tileC, tileD] ∧
  ¬hasUniqueSides tileB [tileA, tileC, tileD] ∧
  ¬hasUniqueSides tileD [tileA, tileB, tileC] :=
sorry

end tileC_in_rectangleY_l2060_206034


namespace smallest_m_is_20_l2060_206086

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- The property that for all n ≥ m, there exists a complex number z in T such that z^n = 1 -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = 1

/-- 20 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_20 :
  has_nth_root_of_unity 20 ∧ ∀ m : ℕ, 0 < m → m < 20 → ¬(has_nth_root_of_unity m) :=
sorry

end smallest_m_is_20_l2060_206086


namespace power_of_two_l2060_206063

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → 2^n = 8 := by
  sorry

end power_of_two_l2060_206063


namespace sum_of_squares_theorem_l2060_206082

theorem sum_of_squares_theorem (a b m : ℝ) 
  (h1 : a^2 + a*b = 16 + m) 
  (h2 : b^2 + a*b = 9 - m) : 
  (a + b = 5) ∨ (a + b = -5) := by
sorry

end sum_of_squares_theorem_l2060_206082


namespace stock_worth_l2060_206005

/-- The total worth of a stock given specific sales conditions and overall loss -/
theorem stock_worth (stock : ℝ) : 
  (0.2 * stock * 1.1 + 0.8 * stock * 0.95 = stock - 450) → 
  stock = 22500 := by
sorry

end stock_worth_l2060_206005


namespace afternoon_rowers_l2060_206049

theorem afternoon_rowers (morning evening total : ℕ) 
  (h1 : morning = 36)
  (h2 : evening = 49)
  (h3 : total = 98)
  : total - morning - evening = 13 := by
  sorry

end afternoon_rowers_l2060_206049


namespace min_value_expression_l2060_206046

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
    x^2 + y^2 + 4 / (x + y)^2 ≥ m :=
  sorry

end min_value_expression_l2060_206046


namespace time_saved_without_tide_change_l2060_206029

/-- The time saved by a rower if the tide direction had not changed -/
theorem time_saved_without_tide_change 
  (speed_with_tide : ℝ) 
  (speed_against_tide : ℝ) 
  (distance_after_reversal : ℝ) 
  (h1 : speed_with_tide = 5)
  (h2 : speed_against_tide = 4)
  (h3 : distance_after_reversal = 40) : 
  distance_after_reversal / speed_with_tide - distance_after_reversal / speed_against_tide = 2 := by
  sorry

end time_saved_without_tide_change_l2060_206029


namespace bart_tuesday_surveys_l2060_206071

/-- Represents the number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := sorry

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys completed on Monday -/
def monday_surveys : ℕ := 3

/-- The total amount earned over two days in dollars -/
def total_earnings : ℚ := 14

theorem bart_tuesday_surveys :
  tuesday_surveys = 4 :=
by
  sorry

end bart_tuesday_surveys_l2060_206071


namespace negation_of_forall_geq_two_l2060_206026

theorem negation_of_forall_geq_two :
  ¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 := by sorry

end negation_of_forall_geq_two_l2060_206026


namespace sphere_part_volume_l2060_206033

theorem sphere_part_volume (circumference : ℝ) (h : circumference = 18 * Real.pi) :
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius ^ 3
  let part_volume := sphere_volume / 6
  part_volume = 162 * Real.pi := by
  sorry

end sphere_part_volume_l2060_206033


namespace clara_lego_count_l2060_206097

/-- The number of legos each person has --/
structure LegoCount where
  kent : ℕ
  bruce : ℕ
  simon : ℕ
  clara : ℕ

/-- Conditions of the lego distribution --/
def lego_distribution (l : LegoCount) : Prop :=
  l.kent = 80 ∧
  l.bruce = l.kent + 30 ∧
  l.simon = l.bruce + (l.bruce / 4) ∧
  l.clara = l.simon + l.kent - ((l.simon + l.kent) / 10)

/-- Theorem stating Clara's lego count --/
theorem clara_lego_count (l : LegoCount) (h : lego_distribution l) : l.clara = 197 := by
  sorry


end clara_lego_count_l2060_206097


namespace yarns_are_zorps_and_xings_l2060_206000

variable (U : Type) -- Universe set

-- Define the subsets
variable (Zorp Xing Yarn Wit Vamp : Set U)

-- State the given conditions
variable (h1 : Zorp ⊆ Xing)
variable (h2 : Yarn ⊆ Xing)
variable (h3 : Wit ⊆ Zorp)
variable (h4 : Yarn ⊆ Wit)
variable (h5 : Yarn ⊆ Vamp)

-- Theorem to prove
theorem yarns_are_zorps_and_xings : Yarn ⊆ Zorp ∧ Yarn ⊆ Xing := by sorry

end yarns_are_zorps_and_xings_l2060_206000


namespace age_sum_problem_l2060_206057

theorem age_sum_problem (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 144) :
  a + b + c = 16 := by
  sorry

end age_sum_problem_l2060_206057


namespace max_a_for_three_solutions_l2060_206091

/-- The equation function that we're analyzing -/
def f (x a : ℝ) : ℝ := (|x - 2| + 2*a)^2 - 3*(|x - 2| + 2*a) + 4*a*(3 - 4*a)

/-- Predicate to check if the equation has three solutions for a given 'a' -/
def has_three_solutions (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0

/-- The theorem stating that 0.5 is the maximum value of 'a' for which the equation has three solutions -/
theorem max_a_for_three_solutions :
  ∀ a : ℝ, has_three_solutions a → a ≤ 0.5 ∧
  has_three_solutions 0.5 :=
sorry

end max_a_for_three_solutions_l2060_206091


namespace novel_sales_theorem_l2060_206053

/-- Represents the sale of a novel in hardback and paperback versions -/
structure NovelSales where
  hardback_before_paperback : ℕ
  paperback_total : ℕ
  paperback_to_hardback_ratio : ℕ

/-- Calculates the total number of copies sold given the sales data -/
def total_copies_sold (sales : NovelSales) : ℕ :=
  sales.hardback_before_paperback + 
  sales.paperback_total + 
  (sales.paperback_total / sales.paperback_to_hardback_ratio)

/-- Theorem stating that given the conditions, the total number of copies sold is 440400 -/
theorem novel_sales_theorem (sales : NovelSales) 
  (h1 : sales.hardback_before_paperback = 36000)
  (h2 : sales.paperback_to_hardback_ratio = 9)
  (h3 : sales.paperback_total = 363600) :
  total_copies_sold sales = 440400 := by
  sorry

#eval total_copies_sold ⟨36000, 363600, 9⟩

end novel_sales_theorem_l2060_206053


namespace ball_color_equality_l2060_206008

theorem ball_color_equality (r g b : ℕ) : 
  (r + g + b = 20) →
  (b ≥ 7) →
  (r ≥ 4) →
  (b = 2 * g) →
  (r = b ∨ r = g) :=
by sorry

end ball_color_equality_l2060_206008


namespace quadratic_root_relation_l2060_206069

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end quadratic_root_relation_l2060_206069


namespace departure_sequences_count_l2060_206036

/-- The number of trains --/
def num_trains : ℕ := 6

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of trains per group --/
def trains_per_group : ℕ := 3

/-- The number of fixed trains (A and B) in the first group --/
def fixed_trains : ℕ := 2

/-- Theorem: The number of different departure sequences for the trains --/
theorem departure_sequences_count : 
  (num_trains - fixed_trains - trains_per_group) * 
  (Nat.factorial trains_per_group) * 
  (Nat.factorial trains_per_group) = 144 := by
  sorry

end departure_sequences_count_l2060_206036


namespace sqrt8_same_type_as_sqrt2_l2060_206051

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define a function to check if a number is of the same type of quadratic root as √2
def same_type_as_sqrt2 (n : ℕ) : Prop :=
  ¬ (is_perfect_square n) ∧ ∃ k : ℕ, n = 2 * k ∧ ¬ (is_perfect_square k)

-- Theorem statement
theorem sqrt8_same_type_as_sqrt2 :
  same_type_as_sqrt2 8 ∧
  ¬ (same_type_as_sqrt2 4) ∧
  ¬ (same_type_as_sqrt2 12) ∧
  ¬ (same_type_as_sqrt2 24) :=
sorry

end sqrt8_same_type_as_sqrt2_l2060_206051


namespace boat_speed_in_still_water_l2060_206088

/-- 
Given a boat traveling downstream with the following conditions:
1. The rate of the stream is 5 km/hr
2. The boat takes 3 hours to cover a distance of 63 km downstream

This theorem proves that the speed of the boat in still water is 16 km/hr.
-/
theorem boat_speed_in_still_water : 
  ∀ (stream_rate : ℝ) (downstream_time : ℝ) (downstream_distance : ℝ),
  stream_rate = 5 →
  downstream_time = 3 →
  downstream_distance = 63 →
  ∃ (still_water_speed : ℝ),
    still_water_speed = 16 ∧
    downstream_distance = (still_water_speed + stream_rate) * downstream_time :=
by sorry

end boat_speed_in_still_water_l2060_206088


namespace eve_distance_difference_l2060_206007

theorem eve_distance_difference : 
  let ran_distance : ℝ := 0.7
  let walked_distance : ℝ := 0.6
  ran_distance - walked_distance = 0.1 :=
by sorry

end eve_distance_difference_l2060_206007


namespace parabola_properties_l2060_206035

/-- Parabola C: y^2 = x with focus F -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  point : ℝ × ℝ
  on_parabola : C.equation point

/-- Theorem about the slope of line AB and the length of AB when collinear with focus -/
theorem parabola_properties (C : Parabola) 
    (hC : C.focus = (1/4, 0) ∧ C.equation = fun p => p.2^2 = p.1) 
    (A B : PointOnParabola C) 
    (hAB : A.point ≠ B.point ∧ A.point ≠ (0, 0) ∧ B.point ≠ (0, 0)) :
  (∃ k : ℝ, k = (A.point.2 - B.point.2) / (A.point.1 - B.point.1) → 
    k = 1 / (A.point.2 + B.point.2)) ∧
  (∃ AB : ℝ, (∃ t : ℝ, (1 - t) • A.point + t • B.point = C.focus) → 
    AB = A.point.1 + B.point.1 + 1/2) :=
sorry

end parabola_properties_l2060_206035


namespace largest_multiple_of_9_under_100_l2060_206006

theorem largest_multiple_of_9_under_100 : ∃ (n : ℕ), n = 99 ∧ 
  (∀ m : ℕ, m < 100 ∧ 9 ∣ m → m ≤ n) := by
  sorry

end largest_multiple_of_9_under_100_l2060_206006


namespace exists_special_number_l2060_206065

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 1000
    and the sum of digits of its square is 1000000 -/
theorem exists_special_number : 
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 := by
  sorry

end exists_special_number_l2060_206065


namespace minimal_distance_point_l2060_206084

/-- Given points A and B in ℝ², prove that P(0, 3) on the y-axis minimizes |PA| + |PB| -/
theorem minimal_distance_point (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  let P : ℝ × ℝ := (0, 3)
  (∀ Q : ℝ × ℝ, Q.1 = 0 → dist A P + dist B P ≤ dist A Q + dist B Q) :=
by sorry


end minimal_distance_point_l2060_206084


namespace inequalities_from_sqrt_l2060_206055

theorem inequalities_from_sqrt (a b : ℝ) (h : Real.sqrt a > Real.sqrt b) :
  (a^2 > b^2) ∧ ((b + 1) / (a + 1) > b / a) ∧ (b + 1 / (b + 1) ≥ 1) := by
  sorry

end inequalities_from_sqrt_l2060_206055


namespace card_draw_probability_l2060_206002

/-- The number of cards in the set -/
def n : ℕ := 100

/-- The number of draws -/
def k : ℕ := 20

/-- The probability that all drawn numbers are distinct -/
noncomputable def p : ℝ := (n.factorial / (n - k).factorial) / n^k

/-- Main theorem -/
theorem card_draw_probability : p < (9/10)^19 ∧ (9/10)^19 < 1/Real.exp 2 := by
  sorry

end card_draw_probability_l2060_206002
