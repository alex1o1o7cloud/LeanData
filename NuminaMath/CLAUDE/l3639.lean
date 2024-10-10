import Mathlib

namespace smallest_n_congruence_l3639_363979

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 5 * n ≡ 980 [ZMOD 33] ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬(5 * m ≡ 980 [ZMOD 33])) ↔ n = 19 :=
by sorry

end smallest_n_congruence_l3639_363979


namespace star_equation_solution_l3639_363980

-- Define the star operation
def star (a b : ℝ) : ℝ := 2*a*b + 3*b - 2*a

-- Theorem statement
theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 60 → x = 22/3 :=
by
  sorry

end star_equation_solution_l3639_363980


namespace at_least_95_buildings_collapsed_l3639_363974

/-- Represents the number of buildings that collapsed in each earthquake --/
structure EarthquakeCollapses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating that at least 95 buildings collapsed after five earthquakes --/
theorem at_least_95_buildings_collapsed
  (initial_buildings : ℕ)
  (collapses : EarthquakeCollapses)
  (h_initial : initial_buildings = 100)
  (h_first : collapses.first = 5)
  (h_second : collapses.second = 6)
  (h_third : collapses.third = 13)
  (h_fourth : collapses.fourth = 24)
  (h_handful : ∀ n : ℕ, n ≤ 5 → n ≤ initial_buildings - (collapses.first + collapses.second + collapses.third + collapses.fourth)) :
  95 ≤ collapses.first + collapses.second + collapses.third + collapses.fourth :=
sorry

end at_least_95_buildings_collapsed_l3639_363974


namespace units_digit_of_7_pow_6_pow_5_l3639_363993

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the exponential function
def pow (base exponent : ℕ) : ℕ := base ^ exponent

-- Theorem statement
theorem units_digit_of_7_pow_6_pow_5 : unitsDigit (pow 7 (pow 6 5)) = 1 := by
  sorry

end units_digit_of_7_pow_6_pow_5_l3639_363993


namespace david_crunches_count_l3639_363949

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 62

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 17

/-- The number of crunches David did -/
def david_crunches : ℕ := zachary_crunches - crunch_difference

theorem david_crunches_count : david_crunches = 45 := by
  sorry

end david_crunches_count_l3639_363949


namespace paint_remaining_is_three_eighths_l3639_363903

/-- The fraction of paint remaining after three days of usage --/
def paint_remaining (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount / 2
  let day2_remaining := day1_remaining * 3/4
  let day3_remaining := day2_remaining / 2
  day3_remaining / initial_amount

/-- Theorem stating that the fraction of paint remaining after three days is 3/8 --/
theorem paint_remaining_is_three_eighths (initial_amount : ℚ) :
  paint_remaining initial_amount = 3/8 := by
  sorry

#eval paint_remaining 2  -- To check the result

end paint_remaining_is_three_eighths_l3639_363903


namespace average_marks_proof_l3639_363995

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def num_subjects : ℕ := 5

theorem average_marks_proof :
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / num_subjects = 75 := by
  sorry

end average_marks_proof_l3639_363995


namespace smoothie_time_theorem_l3639_363943

/-- Represents the time in minutes to chop each fruit type -/
structure ChoppingTimes where
  apple : ℕ
  banana : ℕ
  strawberry : ℕ
  mango : ℕ
  pineapple : ℕ

/-- Calculates the total time to make smoothies -/
def totalSmoothieTime (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) : ℕ :=
  (ct.apple + ct.banana + ct.strawberry + ct.mango + ct.pineapple + blendTime) * numSmoothies

/-- Theorem: The total time to make 5 smoothies is 115 minutes -/
theorem smoothie_time_theorem (ct : ChoppingTimes) (blendTime : ℕ) (numSmoothies : ℕ) :
  ct.apple = 2 →
  ct.banana = 3 →
  ct.strawberry = 4 →
  ct.mango = 5 →
  ct.pineapple = 6 →
  blendTime = 3 →
  numSmoothies = 5 →
  totalSmoothieTime ct blendTime numSmoothies = 115 := by
  sorry

end smoothie_time_theorem_l3639_363943


namespace empty_quadratic_set_implies_m_greater_than_one_l3639_363953

theorem empty_quadratic_set_implies_m_greater_than_one (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) → m > 1 := by
  sorry

end empty_quadratic_set_implies_m_greater_than_one_l3639_363953


namespace geometric_sequence_sum_l3639_363947

theorem geometric_sequence_sum (a : ℕ) : 
  let seq := [a, 2*a, 4*a, 8*a, 16*a, 32*a]
  ∀ (x y z w : ℕ), x ∈ seq → y ∈ seq → z ∈ seq → w ∈ seq →
  x ≠ y ∧ z ≠ w ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w →
  x + y = 136 →
  z + w = 272 →
  ∃ (p q : ℕ), p ∈ seq ∧ q ∈ seq ∧ p ≠ q ∧ p + q = 96 :=
by sorry

end geometric_sequence_sum_l3639_363947


namespace square_root_plus_square_eq_zero_l3639_363959

theorem square_root_plus_square_eq_zero (x y : ℝ) :
  Real.sqrt (x + 2) + (x + y)^2 = 0 → x^2 - x*y = 8 := by
  sorry

end square_root_plus_square_eq_zero_l3639_363959


namespace officer_assignment_count_l3639_363938

def group_size : ℕ := 4
def roles : ℕ := 3

theorem officer_assignment_count : (group_size.choose roles) * (Nat.factorial roles) = 24 := by
  sorry

end officer_assignment_count_l3639_363938


namespace milk_for_six_cookies_l3639_363967

/-- Calculates the amount of milk needed for a given number of cookies. -/
def milk_needed (cookies : ℕ) : ℚ :=
  (5000 : ℚ) * cookies / 24

theorem milk_for_six_cookies :
  milk_needed 6 = 1250 := by sorry


end milk_for_six_cookies_l3639_363967


namespace no_real_solutions_for_equation_l3639_363920

theorem no_real_solutions_for_equation :
  ∀ x : ℝ, x + Real.sqrt (2 * x - 3) ≠ 5 := by
  sorry

end no_real_solutions_for_equation_l3639_363920


namespace area_difference_equals_target_l3639_363917

/-- A right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  base : Real
  height : Real
  hypotenuse : Real
  is_right : base = 3 ∧ height = 4 ∧ hypotenuse = 5

/-- The set Xₙ as defined in the problem -/
def X (n : ℕ) (t : RightTriangle) : Set (Real × Real) :=
  sorry

/-- The area of the region outside X₂₀ but inside X₂₁ -/
def area_difference (t : RightTriangle) : Real :=
  sorry

/-- The main theorem to prove -/
theorem area_difference_equals_target (t : RightTriangle) :
  area_difference t = (41 * Real.pi / 2) + 12 := by
  sorry

end area_difference_equals_target_l3639_363917


namespace sin_cos_extrema_l3639_363946

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∃ z w : ℝ, Real.sin z + Real.sin w = 1/3 ∧ 
    Real.sin w + (Real.cos z)^2 = 19/12) ∧
  (∀ a b : ℝ, Real.sin a + Real.sin b = 1/3 → 
    Real.sin b + (Real.cos a)^2 ≤ 19/12) ∧
  (∃ u v : ℝ, Real.sin u + Real.sin v = 1/3 ∧ 
    Real.sin v + (Real.cos u)^2 = -2/3) ∧
  (∀ c d : ℝ, Real.sin c + Real.sin d = 1/3 → 
    Real.sin d + (Real.cos c)^2 ≥ -2/3) :=
by sorry

end sin_cos_extrema_l3639_363946


namespace inscribed_squares_ratio_l3639_363972

/-- Given two right triangles with sides 5, 12, and 13, where a square of side length x
    is inscribed in the first triangle with a vertex coinciding with the right angle,
    and a square of side length y is inscribed in the second triangle with a side lying
    on the hypotenuse, the ratio x/y equals 12/13. -/
theorem inscribed_squares_ratio (x y : ℝ) : 
  (x > 0 ∧ y > 0) →
  (5^2 + 12^2 = 13^2) →
  (x / 12 = x / 5) →
  ((12 - y) / y = (5 - y) / y) →
  x / y = 12 / 13 := by sorry

end inscribed_squares_ratio_l3639_363972


namespace octagon_lines_l3639_363923

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The number of triangles Bill drew -/
def num_triangles : ℕ := 12

/-- The number of squares Bill drew -/
def num_squares : ℕ := 8

/-- The number of pentagons Bill drew -/
def num_pentagons : ℕ := 4

/-- The number of hexagons Bill drew -/
def num_hexagons : ℕ := 6

/-- The number of octagons Bill drew -/
def num_octagons : ℕ := 2

/-- The number of lines shared between triangles and squares -/
def shared_triangle_square : ℕ := 5

/-- The number of lines shared between pentagons and hexagons -/
def shared_pentagon_hexagon : ℕ := 3

/-- The number of lines shared between hexagons and octagons -/
def shared_hexagon_octagon : ℕ := 1

/-- Theorem: The number of lines drawn with the purple marker (for octagons) is 15 -/
theorem octagon_lines : 
  num_octagons * octagon_sides - shared_hexagon_octagon = 15 := by sorry

end octagon_lines_l3639_363923


namespace polygon_labeling_exists_l3639_363937

/-- A labeling of a polygon is a function that assigns a unique label to each vertex and midpoint -/
def Labeling (n : ℕ) := Fin (4*n+2) → Fin (4*n+2)

/-- The sum of labels for a side is the sum of the labels of its two vertices and midpoint -/
def sideSum (f : Labeling n) (i : Fin (2*n+1)) : ℕ :=
  f i + f (i+1) + f (i+2*n+1)

theorem polygon_labeling_exists (n : ℕ) :
  ∃ (f : Labeling n), Function.Injective f ∧
    ∀ (i j : Fin (2*n+1)), sideSum f i = sideSum f j :=
sorry

end polygon_labeling_exists_l3639_363937


namespace emily_age_is_23_l3639_363921

-- Define the ages as natural numbers
def uncle_bob_age : ℕ := 54
def daniel_age : ℕ := uncle_bob_age / 2
def emily_age : ℕ := daniel_age - 4
def zoe_age : ℕ := emily_age * 3 / 2

-- Theorem statement
theorem emily_age_is_23 : emily_age = 23 := by
  sorry

end emily_age_is_23_l3639_363921


namespace triangle_ratio_l3639_363984

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_ratio (t : Triangle) 
  (h1 : t.A = π / 3)
  (h2 : Real.sin (t.B + t.C) = 6 * Real.cos t.B * Real.sin t.C) :
  t.b / t.c = (1 + Real.sqrt 21) / 2 := by
  sorry


end triangle_ratio_l3639_363984


namespace grocer_coffee_stock_theorem_l3639_363919

/-- Represents the amount of coffee in pounds and its decaffeinated percentage -/
structure CoffeeStock where
  amount : ℝ
  decaf_percent : ℝ

/-- Calculates the new coffee stock after a purchase or sale -/
def update_stock (current : CoffeeStock) (transaction : CoffeeStock) (is_sale : Bool) : CoffeeStock :=
  sorry

/-- Calculates the final percentage of decaffeinated coffee -/
def final_decaf_percentage (transactions : List (CoffeeStock × Bool)) : ℝ :=
  sorry

theorem grocer_coffee_stock_theorem (initial_stock : CoffeeStock) 
  (transactions : List (CoffeeStock × Bool)) : 
  let final_percent := final_decaf_percentage transactions
  ∃ ε > 0, |final_percent - 28.88| < ε :=
by sorry

end grocer_coffee_stock_theorem_l3639_363919


namespace total_cost_theorem_l3639_363936

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := num_parents + num_brothers + num_brothers + (num_brothers * children_per_brother)

theorem total_cost_theorem : cost_per_package * total_relatives = 70 := by
  sorry

end total_cost_theorem_l3639_363936


namespace geometric_sequence_ratio_sum_l3639_363930

theorem geometric_sequence_ratio_sum 
  (m x y : ℝ) 
  (h_m : m ≠ 0) 
  (h_x_ne_y : x ≠ y) 
  (h_nonconstant : x ≠ 1 ∧ y ≠ 1) 
  (h_eq : m * x^2 - m * y^2 = 3 * (m * x - m * y)) : 
  x + y = 3 := by
sorry

end geometric_sequence_ratio_sum_l3639_363930


namespace trigonometric_identity_l3639_363929

theorem trigonometric_identity (α : Real) 
  (h : Real.sqrt 2 * Real.sin (α + π / 4) = 4 * Real.cos α) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := by
  sorry

end trigonometric_identity_l3639_363929


namespace hyperbola_eccentricity_l3639_363971

/-- Given a hyperbola with equation x²/m² - y² = 4 where m > 0 and focal distance 8,
    prove that its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) :
  let focal_distance : ℝ := 8
  let a : ℝ := m * 2
  let b : ℝ := 2
  let c : ℝ := focal_distance / 2
  let eccentricity : ℝ := c / a
  eccentricity = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l3639_363971


namespace store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l3639_363982

/-- Represents the cost of purchasing from Store A or B -/
def store_cost (x : ℝ) (is_store_a : Bool) : ℝ :=
  if is_store_a then 20 * x + 2400 else 18 * x + 2700

/-- Theorem stating the conditions under which Store A is cheaper than Store B -/
theorem store_a_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x true < store_cost x false ↔ x < 150 :=
sorry

/-- Theorem stating the conditions under which Store B is cheaper than Store A -/
theorem store_b_cheaper_condition (x : ℝ) (h1 : x > 30) :
  store_cost x false < store_cost x true ↔ x > 150 :=
sorry

/-- Theorem proving that for x = 100, Store A is cheaper -/
theorem store_a_cheaper_at_100 :
  store_cost 100 true < store_cost 100 false :=
sorry

/-- Definition of the cost for the most cost-effective plan when x = 100 -/
def most_cost_effective_plan : ℝ := 3000 + 20 * 70 * 0.9

/-- Theorem proving that the most cost-effective plan is cheaper than both Store A and B when x = 100 -/
theorem most_cost_effective_plan_is_best :
  most_cost_effective_plan < store_cost 100 true ∧
  most_cost_effective_plan < store_cost 100 false :=
sorry

end store_a_cheaper_condition_store_b_cheaper_condition_store_a_cheaper_at_100_most_cost_effective_plan_is_best_l3639_363982


namespace lcm_of_3_5_7_18_l3639_363904

theorem lcm_of_3_5_7_18 : Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 18)) = 630 := by
  sorry

end lcm_of_3_5_7_18_l3639_363904


namespace smallest_n_for_perfect_square_l3639_363988

theorem smallest_n_for_perfect_square : ∃ (n : ℕ), 
  (n = 12) ∧ 
  (∃ (k : ℕ), (2^n + 2^8 + 2^11 : ℕ) = k^2) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), (2^m + 2^8 + 2^11 : ℕ) = k^2) :=
by sorry

end smallest_n_for_perfect_square_l3639_363988


namespace thirteenth_digit_of_sum_l3639_363927

def decimal_sum (a b : ℚ) : ℚ := a + b

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem thirteenth_digit_of_sum :
  let sum := decimal_sum (1/8) (1/11)
  nth_digit_after_decimal sum 13 = 9 := by
  sorry

end thirteenth_digit_of_sum_l3639_363927


namespace no_solution_range_l3639_363970

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + a + 1| + |x + a^2 - 2| ≥ 3) ↔ 
  (a ≤ -2 ∨ (0 ≤ a ∧ a ≤ 1) ∨ 3 ≤ a) :=
sorry

end no_solution_range_l3639_363970


namespace quadratic_always_nonnegative_implies_a_bounded_l3639_363950

theorem quadratic_always_nonnegative_implies_a_bounded (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end quadratic_always_nonnegative_implies_a_bounded_l3639_363950


namespace total_books_collected_l3639_363968

def books_first_week : ℕ := 9
def weeks_collecting : ℕ := 6
def multiplier : ℕ := 10

theorem total_books_collected :
  (books_first_week + (weeks_collecting - 1) * (books_first_week * multiplier)) = 459 :=
by sorry

end total_books_collected_l3639_363968


namespace original_number_is_27_l3639_363951

theorem original_number_is_27 (x : ℕ) :
  (Odd (3 * x)) →
  (∃ k : ℕ, 3 * x = 9 * k) →
  (∃ y : ℕ, x * y = 108) →
  x = 27 := by
sorry

end original_number_is_27_l3639_363951


namespace diophantine_equation_solution_l3639_363932

theorem diophantine_equation_solution (x y : ℤ) :
  y^2 = x^3 + 16 → x = 0 ∧ y = 4 := by
  sorry

end diophantine_equation_solution_l3639_363932


namespace quadratic_polynomial_solutions_l3639_363963

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ y : ℝ, f y ^ 2 = 1 ↔ y = y₁ ∨ y = y₂) ∧ y₁ ≠ y₂

-- State the theorem
theorem quadratic_polynomial_solutions 
  (f : QuadraticPolynomial ℝ) 
  (h : HasThreeSolutionsCubicMinusFour f) : 
  HasTwoSolutionsSquaredEqualsOne f :=
sorry

end quadratic_polynomial_solutions_l3639_363963


namespace apple_cost_l3639_363948

theorem apple_cost (initial_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) : 
  initial_cost * (target_dozen / initial_dozen) = 54.60 :=
by
  sorry

#check apple_cost 39.00 5 7

end apple_cost_l3639_363948


namespace solution_set_x_squared_geq_four_l3639_363926

theorem solution_set_x_squared_geq_four :
  {x : ℝ | x^2 ≥ 4} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} := by
sorry

end solution_set_x_squared_geq_four_l3639_363926


namespace population_change_l3639_363987

/-- The population change problem --/
theorem population_change (P : ℝ) : 
  P > 0 → 
  P * 1.15 * 0.90 * 1.20 * 0.75 = 7575 → 
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) ≤ P ∧ P < (n : ℝ) + 1 :=
by sorry

end population_change_l3639_363987


namespace solution_difference_l3639_363910

theorem solution_difference (p q : ℝ) : 
  ((3 * p - 9) / (p^2 + 3*p - 18) = p + 3) →
  ((3 * q - 9) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 2 := by sorry

end solution_difference_l3639_363910


namespace triangle_perimeter_l3639_363908

theorem triangle_perimeter (x : ℕ+) : 
  (1 < x) ∧ (x < 5) ∧ (1 + x > 4) ∧ (x + 4 > 1) ∧ (4 + 1 > x) → 
  1 + x + 4 = 9 := by
sorry

end triangle_perimeter_l3639_363908


namespace nonreal_roots_product_l3639_363957

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ 
   (x^4 - 4*x^3 + 6*x^2 - 4*x + 4 = 4036) ∧
   (a * b = 1 + Real.sqrt 4033)) :=
by sorry

end nonreal_roots_product_l3639_363957


namespace min_product_of_three_l3639_363958

def S : Finset Int := {-9, -7, -1, 2, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  x * y * z = -432 ∧ (∀ (p q r : Int), p ∈ S → q ∈ S → r ∈ S →
  p ≠ q → q ≠ r → p ≠ r → p * q * r ≥ -432) :=
sorry

end min_product_of_three_l3639_363958


namespace regular_polygon_sides_l3639_363994

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (((n - 2) * 180) / n = 150) → n = 12 := by
  sorry

end regular_polygon_sides_l3639_363994


namespace cube_plus_reciprocal_cube_l3639_363925

theorem cube_plus_reciprocal_cube (a : ℝ) (h : (a + 1/a)^2 = 3) :
  a^3 + 1/a^3 = 0 := by
  sorry

end cube_plus_reciprocal_cube_l3639_363925


namespace cylinder_height_given_cone_volume_ratio_l3639_363985

theorem cylinder_height_given_cone_volume_ratio (base_area : ℝ) (cone_height : ℝ) :
  cone_height = 4.5 →
  (1 / 3 * base_area * cone_height) / (base_area * cylinder_height) = 1 / 6 →
  cylinder_height = 9 :=
by
  sorry

end cylinder_height_given_cone_volume_ratio_l3639_363985


namespace bus_driver_hours_l3639_363911

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriver where
  regularRate : ℝ
  regularHours : ℝ
  overtimeRate : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total hours worked by a bus driver --/
def totalHours (driver : BusDriver) : ℝ :=
  driver.regularHours + driver.overtimeHours

/-- Theorem stating the conditions and the result to be proved --/
theorem bus_driver_hours (driver : BusDriver) 
  (h1 : driver.regularRate = 16)
  (h2 : driver.regularHours = 40)
  (h3 : driver.overtimeRate = driver.regularRate * 1.75)
  (h4 : driver.totalCompensation = 976)
  (h5 : driver.totalCompensation = driver.regularRate * driver.regularHours + 
                                   driver.overtimeRate * driver.overtimeHours) :
  totalHours driver = 52 := by
  sorry

#eval 40 + 12 -- Expected output: 52

end bus_driver_hours_l3639_363911


namespace girls_not_adjacent_arrangements_l3639_363940

theorem girls_not_adjacent_arrangements :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let num_spaces : ℕ := num_boys + 1
  
  (num_boys.factorial * num_spaces.factorial) = 2880 :=
by sorry

end girls_not_adjacent_arrangements_l3639_363940


namespace amoeba_count_after_10_days_l3639_363955

def amoeba_count (day : ℕ) : ℕ :=
  if day = 0 then 3
  else if (day % 3 = 0) ∧ (day ≥ 3) then
    amoeba_count (day - 1)
  else
    2 * amoeba_count (day - 1)

theorem amoeba_count_after_10_days :
  amoeba_count 10 = 384 :=
sorry

end amoeba_count_after_10_days_l3639_363955


namespace existence_of_zero_crossing_l3639_363922

open Function Set

theorem existence_of_zero_crossing (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), ContinuousOn f (Icc a b) ∧ 
  f a * f b > 0 ∧ 
  ∃ c ∈ Ioo a b, f c = 0 := by
  sorry

end existence_of_zero_crossing_l3639_363922


namespace correct_coverings_8x8_l3639_363928

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a covering of the grid with colored triangles -/
def Covering (g : Grid) := Unit

/-- Checks if a covering is correct (adjacent triangles have different colors) -/
def is_correct_covering (g : Grid) (c : Covering g) : Prop := sorry

/-- Counts the number of correct coverings for a given grid -/
def count_correct_coverings (g : Grid) : ℕ := sorry

/-- Theorem: The number of correct coverings for an 8x8 grid is 2^16 -/
theorem correct_coverings_8x8 :
  let g : Grid := ⟨8⟩
  count_correct_coverings g = 2^16 := by sorry

end correct_coverings_8x8_l3639_363928


namespace g_eval_l3639_363969

def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 11

theorem g_eval : 3 * g 2 + 2 * g (-4) = 147 := by sorry

end g_eval_l3639_363969


namespace constant_value_proof_l3639_363989

theorem constant_value_proof (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end constant_value_proof_l3639_363989


namespace floor_times_self_eq_72_l3639_363941

theorem floor_times_self_eq_72 (x : ℝ) :
  x > 0 ∧ ⌊x⌋ * x = 72 → x = 9 := by sorry

end floor_times_self_eq_72_l3639_363941


namespace choir_problem_l3639_363924

/-- A choir problem involving singers joining in different verses -/
theorem choir_problem (total_singers : ℕ) (first_verse_singers : ℕ) 
  (second_verse_joiners : ℕ) (third_verse_joiners : ℕ) : 
  total_singers = 30 →
  first_verse_singers = total_singers / 2 →
  third_verse_joiners = 10 →
  first_verse_singers + second_verse_joiners + third_verse_joiners = total_singers →
  (second_verse_joiners : ℚ) / (total_singers - first_verse_singers : ℚ) = 1 / 3 := by
  sorry

#check choir_problem

end choir_problem_l3639_363924


namespace line_translation_proof_l3639_363901

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translateLine (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- The original line y = 4x -/
def originalLine : Line :=
  { slope := 4, intercept := 0 }

/-- The amount of downward translation -/
def translationAmount : ℝ := 5

theorem line_translation_proof :
  translateLine originalLine translationAmount = { slope := 4, intercept := -5 } := by
  sorry

end line_translation_proof_l3639_363901


namespace cube_vertex_sum_difference_l3639_363918

theorem cube_vertex_sum_difference (a b c d e f g h : ℝ) 
  (ha : 3 * a = b + e + d)
  (hb : 3 * b = c + f + a)
  (hc : 3 * c = d + g + b)
  (hd : 3 * d = a + h + c)
  (he : 3 * e = f + a + h)
  (hf : 3 * f = g + b + e)
  (hg : 3 * g = h + c + f)
  (hh : 3 * h = e + d + g) :
  (a + b + c + d) - (e + f + g + h) = 0 := by
  sorry

end cube_vertex_sum_difference_l3639_363918


namespace favorite_color_survey_l3639_363976

theorem favorite_color_survey (total_students : ℕ) (total_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : total_girls = 18)
  (h3 : total_students / 2 = total_students - total_girls + total_girls / 3 + 9) :
  9 = total_students - (total_students / 2 + total_girls / 3) :=
by sorry

end favorite_color_survey_l3639_363976


namespace arrangement_schemes_eq_twelve_l3639_363931

/-- The number of ways to divide 2 teachers and 4 students into 2 groups -/
def arrangement_schemes : ℕ :=
  (Nat.choose 2 1) * (Nat.choose 4 2)

/-- Theorem stating that the number of arrangement schemes is 12 -/
theorem arrangement_schemes_eq_twelve :
  arrangement_schemes = 12 := by
  sorry

end arrangement_schemes_eq_twelve_l3639_363931


namespace green_chips_count_l3639_363934

theorem green_chips_count (total : ℕ) (red : ℕ) (h1 : total = 60) (h2 : red = 34) :
  total - (total / 6) - red = 16 := by
  sorry

end green_chips_count_l3639_363934


namespace steel_bar_length_l3639_363905

/-- Given three types of steel bars A, B, and C with lengths x, y, and z respectively,
    prove that the total length of 1 bar of type A, 2 bars of type B, and 3 bars of type C
    is x + 2y + 3z, given the conditions. -/
theorem steel_bar_length (x y z : ℝ) 
  (h1 : 2 * x + y + 3 * z = 23) 
  (h2 : x + 4 * y + 5 * z = 36) : 
  x + 2 * y + 3 * z = (7 * x + 14 * y + 21 * z) / 7 := by
  sorry

end steel_bar_length_l3639_363905


namespace count_multiples_count_multiples_equals_1002_l3639_363978

theorem count_multiples : ℕ :=
  let range_start := 1
  let range_end := 2005
  let count_multiples_of_3 := (range_end / 3 : ℕ)
  let count_multiples_of_4 := (range_end / 4 : ℕ)
  let count_multiples_of_12 := (range_end / 12 : ℕ)
  count_multiples_of_3 + count_multiples_of_4 - count_multiples_of_12

theorem count_multiples_equals_1002 : count_multiples = 1002 := by
  sorry

end count_multiples_count_multiples_equals_1002_l3639_363978


namespace fraction_equivalence_l3639_363964

theorem fraction_equivalence (x y z : ℝ) (h1 : 2*x - z ≠ 0) (h2 : z ≠ 0) :
  (2*x + y) / (2*x - z) = y / (-z) ↔ y = -z :=
by sorry

end fraction_equivalence_l3639_363964


namespace triangle_perimeter_l3639_363914

/-- Given a triangle with inradius 2.5 cm and area 35 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 35)
  (h3 : A = r * (p / 2)) :
  p = 28 := by
  sorry

end triangle_perimeter_l3639_363914


namespace trapezoid_parallel_sides_l3639_363907

/-- Trapezoid properties and parallel sides calculation -/
theorem trapezoid_parallel_sides 
  (t : ℝ) 
  (m : ℝ) 
  (n : ℝ) 
  (E : ℝ) 
  (h_t : t = 204) 
  (h_m : m = 14) 
  (h_n : n = 2) 
  (h_E : E = 59 + 29/60 + 23/3600) : 
  ∃ (a c : ℝ), 
    a - c = m ∧ 
    t = (a + c) / 2 * (2 * t / (a + c)) ∧ 
    a = 24 ∧ 
    c = 10 := by
  sorry

end trapezoid_parallel_sides_l3639_363907


namespace shoe_price_calculation_l3639_363961

theorem shoe_price_calculation (discount_rate : ℝ) (savings : ℝ) (original_price : ℝ) : 
  discount_rate = 0.30 →
  savings = 46 →
  original_price = savings / discount_rate →
  original_price = 153.33 := by
sorry

end shoe_price_calculation_l3639_363961


namespace exists_line_not_through_lattice_points_l3639_363913

-- Define a 2D point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a lattice point (grid point)
def isLatticePoint (p : Point) : Prop :=
  ∃ (m n : ℤ), p.x = m ∧ p.y = n

-- Define when a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Theorem statement
theorem exists_line_not_through_lattice_points :
  ∃ (l : Line), ∀ (p : Point), isLatticePoint p → ¬ pointOnLine p l :=
sorry

end exists_line_not_through_lattice_points_l3639_363913


namespace parabola_c_value_l3639_363996

/-- A parabola passing through two specific points has a determined c-value. -/
theorem parabola_c_value (b c : ℝ) : 
  (2 = 1^2 + b*1 + c) ∧ (2 = 5^2 + b*5 + c) → c = 7 := by
  sorry

end parabola_c_value_l3639_363996


namespace negation_of_implication_is_false_l3639_363960

theorem negation_of_implication_is_false : 
  ¬(∃ a b : ℝ, (a ≤ 1 ∨ b ≤ 1) ∧ (a + b ≤ 2)) := by
  sorry

end negation_of_implication_is_false_l3639_363960


namespace quadratic_form_equivalence_l3639_363997

theorem quadratic_form_equivalence (b m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 56 = (x + m)^2 + 20) → 
  b = 12 := by
sorry

end quadratic_form_equivalence_l3639_363997


namespace arithmetic_sequence_perfect_square_sum_l3639_363965

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_perfect_square_sum (a d : ℕ) :
  (∀ n : ℕ, is_perfect_square (sum_arithmetic_sequence a d n)) ↔
  (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) :=
sorry

end arithmetic_sequence_perfect_square_sum_l3639_363965


namespace bug_path_length_l3639_363956

theorem bug_path_length (a b c : ℝ) (h1 : a = 120) (h2 : b = 90) (h3 : c = 150) : 
  ∃ (d : ℝ), (a^2 + b^2 = c^2) ∧ (c + c + d = 390) ∧ (d = a ∨ d = b) :=
sorry

end bug_path_length_l3639_363956


namespace convex_lattice_polygon_vertices_l3639_363944

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex polygon defined by its vertices -/
structure ConvexPolygon where
  vertices : List LatticePoint
  is_convex : Bool  -- Assume this is true for our polygon

/-- Checks if a point is inside or on the sides of a polygon -/
def is_inside_or_on_sides (point : LatticePoint) (polygon : ConvexPolygon) : Bool :=
  sorry  -- Implementation details omitted

theorem convex_lattice_polygon_vertices (polygon : ConvexPolygon) :
  (∀ point : LatticePoint, point ∉ polygon.vertices → ¬(is_inside_or_on_sides point polygon)) →
  List.length polygon.vertices ≤ 4 :=
sorry

end convex_lattice_polygon_vertices_l3639_363944


namespace regular_polygon_on_grid_l3639_363975

/-- A grid in the plane formed by two families of equally spaced parallel lines -/
structure Grid where
  -- We don't need to define the internal structure of the grid

/-- A point in the plane -/
structure Point where
  -- We don't need to define the internal structure of the point

/-- A regular convex n-gon -/
structure RegularPolygon where
  vertices : List Point
  n : Nat
  is_regular : Bool
  is_convex : Bool

/-- Predicate to check if a point is on the grid -/
def Point.on_grid (p : Point) (g : Grid) : Prop := sorry

/-- Predicate to check if all vertices of a polygon are on the grid -/
def RegularPolygon.vertices_on_grid (p : RegularPolygon) (g : Grid) : Prop :=
  ∀ v ∈ p.vertices, v.on_grid g

/-- The main theorem -/
theorem regular_polygon_on_grid (g : Grid) (p : RegularPolygon) :
  p.n ≥ 3 ∧ p.is_regular ∧ p.is_convex ∧ p.vertices_on_grid g → p.n = 3 ∨ p.n = 4 ∨ p.n = 6 := by
  sorry

end regular_polygon_on_grid_l3639_363975


namespace divisor_problem_l3639_363977

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 165 →
  quotient = 9 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end divisor_problem_l3639_363977


namespace james_marbles_left_james_final_marbles_l3639_363900

/-- Represents the number of marbles in each bag -/
structure Bags where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Calculates the total number of marbles in all bags -/
def totalMarbles (bags : Bags) : Nat :=
  bags.a + bags.b + bags.c + bags.d + bags.e + bags.f + bags.g

/-- Represents James' marble collection -/
structure MarbleCollection where
  initialTotal : Nat
  bags : Bags
  forgottenBag : Nat

/-- Theorem stating that James will have 20 marbles left -/
theorem james_marbles_left (collection : MarbleCollection) : Nat :=
  if collection.initialTotal = 28 ∧
     collection.bags.a = 4 ∧
     collection.bags.b = 3 ∧
     collection.bags.c = 5 ∧
     collection.bags.d = 2 * collection.bags.c - 1 ∧
     collection.bags.e = collection.bags.a / 2 ∧
     collection.bags.f = 3 ∧
     collection.bags.g = collection.bags.e ∧
     collection.forgottenBag = 4 ∧
     totalMarbles collection.bags = collection.initialTotal
  then
    collection.initialTotal - (collection.bags.d + collection.bags.f) + collection.forgottenBag
  else
    0

/-- Main theorem to prove -/
theorem james_final_marbles (collection : MarbleCollection) :
  james_marbles_left collection = 20 := by
  sorry

end james_marbles_left_james_final_marbles_l3639_363900


namespace negation_of_universal_quantifier_l3639_363973

theorem negation_of_universal_quantifier (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end negation_of_universal_quantifier_l3639_363973


namespace segment_length_is_zero_l3639_363916

/-- Triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle : ℝ)

/-- The problem setup -/
def problem : Prop :=
  ∃ (ABC DEF : Triangle),
    ABC.a = 8 ∧ ABC.b = 12 ∧ ABC.c = 10 ∧
    DEF.a = 4 ∧ DEF.b = 6 ∧ DEF.c = 5 ∧
    ABC.angle = 100 ∧ DEF.angle = 100 ∧
    ∀ (BD : ℝ), BD = 0

/-- The theorem to be proved -/
theorem segment_length_is_zero : problem := by sorry

end segment_length_is_zero_l3639_363916


namespace midpoint_locus_l3639_363999

/-- Given real numbers a, b, c forming an arithmetic sequence,
    prove that the locus of the midpoint of the chord of the line
    bx + ay + c = 0 intersecting the parabola y^2 = -1/2 x
    is described by the equation x + 1 = -(2y - 1)^2 -/
theorem midpoint_locus (a b c : ℝ) :
  (2 * b = a + c) →
  ∃ (x y : ℝ), 
    (∃ (x₁ y₁ : ℝ), 
      b * x₁ + a * y₁ + c = 0 ∧ 
      y₁^2 = -1/2 * x₁ ∧
      x = (x₁ - 2) / 2 ∧
      y = (y₁ + 1) / 2) →
    x + 1 = -(2 * y - 1)^2 :=
by sorry

end midpoint_locus_l3639_363999


namespace rotation_90_degrees_l3639_363990

-- Define the original line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the rotated line l₁
def line_l₁ (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem rotation_90_degrees :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ 
    (x - 2) = -(y₀ - 3) ∧ 
    (y - 3) = (x₀ - 2)) →
  line_l₁ x y := by sorry

end rotation_90_degrees_l3639_363990


namespace randy_store_trips_l3639_363992

theorem randy_store_trips (initial_amount : ℕ) (final_amount : ℕ) (amount_per_trip : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  amount_per_trip = 2 →
  months_per_year = 12 →
  (initial_amount - final_amount) / amount_per_trip / months_per_year = 4 := by
  sorry

end randy_store_trips_l3639_363992


namespace odd_periodic_two_at_one_l3639_363966

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x + 2) = f(x) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

/-- For a function f: ℝ → ℝ, if f is odd and has a period of 2, then f(1) = 0 -/
theorem odd_periodic_two_at_one (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  f 1 = 0 := by
  sorry

end odd_periodic_two_at_one_l3639_363966


namespace apple_basket_problem_l3639_363942

theorem apple_basket_problem (small_basket_capacity : ℕ) (small_basket_count : ℕ) 
  (large_basket_count : ℕ) (leftover_weight : ℕ) :
  small_basket_capacity = 25 →
  small_basket_count = 28 →
  large_basket_count = 10 →
  leftover_weight = 50 →
  (small_basket_capacity * small_basket_count - leftover_weight) / large_basket_count = 65 := by
  sorry

end apple_basket_problem_l3639_363942


namespace proportion_solution_l3639_363935

theorem proportion_solution (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end proportion_solution_l3639_363935


namespace trajectory_and_max_dot_product_l3639_363945

/-- Trajectory of point P satisfying given conditions -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line segment AB with A on x-axis and B on y-axis -/
def lineSegmentAB (xA yB : ℝ) : Prop :=
  xA^2 + yB^2 = 9 ∧ xA ≥ 0 ∧ yB ≥ 0

/-- Point P satisfies BP = 2PA -/
def pointPCondition (xA yB x y : ℝ) : Prop :=
  (x - 0)^2 + (y - yB)^2 = 4 * ((x - xA)^2 + y^2)

/-- Line passing through (1,0) -/
def lineThroughOneZero (t x y : ℝ) : Prop :=
  x = t * y + 1

/-- Theorem stating the trajectory equation and maximum dot product -/
theorem trajectory_and_max_dot_product :
  ∀ xA yB x y t x1 y1 x2 y2 : ℝ,
  lineSegmentAB xA yB →
  pointPCondition xA yB x y →
  trajectory x y →
  lineThroughOneZero t x1 y1 →
  lineThroughOneZero t x2 y2 →
  trajectory x1 y1 →
  trajectory x2 y2 →
  (∀ x' y' : ℝ, trajectory x' y' → lineThroughOneZero t x' y' → 
    x1 * x2 + y1 * y2 ≥ x' * x' + y' * y') →
  x1 * x2 + y1 * y2 ≤ 1/4 :=
by sorry

end trajectory_and_max_dot_product_l3639_363945


namespace expansion_terms_count_l3639_363991

def expandedTerms (N : ℕ) : ℕ := Nat.choose N 4

theorem expansion_terms_count : expandedTerms 14 = 1001 := by
  sorry

end expansion_terms_count_l3639_363991


namespace hundredth_number_is_hundred_l3639_363962

def counting_sequence (n : ℕ) : ℕ := n

theorem hundredth_number_is_hundred :
  counting_sequence 100 = 100 := by
  sorry

end hundredth_number_is_hundred_l3639_363962


namespace problem_solution_l3639_363902

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (m - 4) + y^2 / (6 - m) = 1 ∧ 
  ((m - 4) * (6 - m) < 0)

theorem problem_solution (m : ℝ) :
  (¬ proposition_p m ↔ m < 1) ∧
  (¬(proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) ↔ 
    m < 1 ∨ (4 ≤ m ∧ m ≤ 6)) :=
by sorry

end problem_solution_l3639_363902


namespace cylinder_volume_from_lateral_surface_l3639_363981

/-- The volume of a cylinder whose lateral surface is a square with side length 2 * (π^(1/3)) is 2 -/
theorem cylinder_volume_from_lateral_surface (π : ℝ) (h : π > 0) :
  let lateral_surface_side := 2 * π^(1/3)
  let cylinder_height := lateral_surface_side
  let cylinder_radius := lateral_surface_side / (2 * π)
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  cylinder_volume = 2 := by
  sorry

end cylinder_volume_from_lateral_surface_l3639_363981


namespace mary_remaining_stickers_l3639_363933

/-- Calculates the number of remaining stickers for Mary --/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (stickers_per_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * stickers_per_page)

/-- Proves that Mary has 44 stickers remaining --/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

#eval remaining_stickers 89 3 6 7

end mary_remaining_stickers_l3639_363933


namespace x_14_plus_inverse_l3639_363909

theorem x_14_plus_inverse (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + 1/x^14 = -1 := by
  sorry

end x_14_plus_inverse_l3639_363909


namespace number_125_with_digit_sum_5_l3639_363915

/-- A function that calculates the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth number in the sequence of natural numbers with digit sum 5 -/
def nthNumberWithDigitSum5 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 125th number in the sequence is 41000 -/
theorem number_125_with_digit_sum_5 : nthNumberWithDigitSum5 125 = 41000 := by sorry

end number_125_with_digit_sum_5_l3639_363915


namespace age_ratio_proof_l3639_363954

/-- Given three people A, B, and C with the following conditions:
    - A is two years older than B
    - The total of the ages of A, B, and C is 22
    - B is 8 years old
    Prove that the ratio of B's age to C's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end age_ratio_proof_l3639_363954


namespace quadratic_integer_solutions_l3639_363906

theorem quadratic_integer_solutions (p q : ℝ) : 
  p + q = 1998 ∧ 
  (∃ a b : ℤ, ∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  (p = 1998 ∧ q = 0) ∨ (p = -2002 ∧ q = 4000) :=
sorry

end quadratic_integer_solutions_l3639_363906


namespace arithmetic_calculation_l3639_363983

theorem arithmetic_calculation : (18 / (8 - 2 * 3)) + 4 = 13 := by
  sorry

end arithmetic_calculation_l3639_363983


namespace aunt_may_milk_problem_l3639_363986

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem 
  (morning_milk : ℕ) 
  (evening_milk : ℕ) 
  (sold_milk : ℕ) 
  (leftover_milk : ℕ) 
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 := by
sorry

end aunt_may_milk_problem_l3639_363986


namespace tangent_circles_exist_l3639_363912

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a circle -/
def IsOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is on a line -/
def IsOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if two circles are externally tangent -/
def AreExternallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- Predicate to check if a circle touches another circle at a point -/
def CircleTouchesCircleAt (c1 c2 : Circle) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c1 ∧ IsOnCircle p c2 ∧ AreExternallyTangent c1 c2

/-- Predicate to check if a circle touches a line at a point -/
def CircleTouchesLineAt (c : Circle) (l : Line) (p : ℝ × ℝ) : Prop :=
  IsOnCircle p c ∧ IsOnLine p l

/-- The main theorem -/
theorem tangent_circles_exist (k : Circle) (e : Line) (P Q : ℝ × ℝ)
    (h_P : IsOnCircle P k) (h_Q : IsOnLine Q e) :
    ∃ (c1 c2 : Circle),
      c1.radius = c2.radius ∧
      AreExternallyTangent c1 c2 ∧
      CircleTouchesCircleAt c1 k P ∧
      CircleTouchesLineAt c2 e Q := by
  sorry

end tangent_circles_exist_l3639_363912


namespace solution_set_inequality_l3639_363939

-- Define the determinant function
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Theorem statement
theorem solution_set_inequality (x : ℝ) :
  (log_sqrt2 (det 1 1 1 x) < 0) ↔ (1 < x ∧ x < 2) :=
sorry

end solution_set_inequality_l3639_363939


namespace yuan_yuan_delivery_cost_l3639_363998

def express_delivery_cost (weight : ℕ) : ℕ :=
  let base_fee := 13
  let weight_limit := 5
  let additional_fee := 2
  if weight ≤ weight_limit then
    base_fee
  else
    base_fee + (weight - weight_limit) * additional_fee

theorem yuan_yuan_delivery_cost :
  express_delivery_cost 7 = 17 := by sorry

end yuan_yuan_delivery_cost_l3639_363998


namespace medicine_supply_duration_l3639_363952

theorem medicine_supply_duration (pills : ℕ) (consumption_rate : ℚ) (consumption_days : ℕ) (days_per_month : ℕ) : 
  pills = 90 → 
  consumption_rate = 1/3 → 
  consumption_days = 3 → 
  days_per_month = 30 → 
  (pills * consumption_days / consumption_rate) / days_per_month = 27 := by
  sorry

end medicine_supply_duration_l3639_363952
