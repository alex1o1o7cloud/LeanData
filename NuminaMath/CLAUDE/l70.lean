import Mathlib

namespace NUMINAMATH_CALUDE_number_of_inequalities_l70_7005

-- Define a function to check if an expression is an inequality
def isInequality (expr : String) : Bool :=
  match expr with
  | "3 < 5" => true
  | "x > 0" => true
  | "2x ≠ 3" => true
  | "a = 3" => false
  | "2a + 1" => false
  | "(1-x)/5 > 1" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3 < 5", "x > 0", "2x ≠ 3", "a = 3", "2a + 1", "(1-x)/5 > 1"]

-- Theorem stating that the number of inequalities is 4
theorem number_of_inequalities :
  (expressions.filter isInequality).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_inequalities_l70_7005


namespace NUMINAMATH_CALUDE_two_tangent_circles_l70_7053

/-- The parabola y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- The directrix of the parabola -/
def directrix : ℝ → ℝ := λ x => -2

/-- The point M -/
def point_M : ℝ × ℝ := (3, 3)

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_focus : dist center focus = radius
  passes_through_M : dist center point_M = radius
  tangent_to_directrix : abs (center.2 - directrix center.1) = radius

/-- The main theorem -/
theorem two_tangent_circles : 
  ∃! (circles : Finset TangentCircle), circles.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_tangent_circles_l70_7053


namespace NUMINAMATH_CALUDE_stock_price_change_l70_7073

theorem stock_price_change (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = (total_stocks - higher_price_stocks) * 6 / 5) :
  higher_price_stocks = 1080 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l70_7073


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l70_7065

/-- Proves that the new concentration of the mixture is 29% given the initial conditions --/
theorem alcohol_mixture_concentration
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 40)
  (h5 : total_liquid = 8)
  (h6 : final_vessel_capacity = 10) :
  (vessel1_capacity * vessel1_alcohol_percentage / 100 +
   vessel2_capacity * vessel2_alcohol_percentage / 100) /
  final_vessel_capacity * 100 = 29 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l70_7065


namespace NUMINAMATH_CALUDE_yellow_balls_count_l70_7067

theorem yellow_balls_count (red blue yellow green : ℕ) : 
  red + blue + yellow + green = 531 →
  red + blue = yellow + green + 31 →
  yellow = green + 22 →
  yellow = 136 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l70_7067


namespace NUMINAMATH_CALUDE_sparkling_juice_bottles_l70_7066

def total_guests : ℕ := 120
def champagne_percentage : ℚ := 60 / 100
def wine_percentage : ℚ := 30 / 100
def juice_percentage : ℚ := 10 / 100

def champagne_glasses_per_guest : ℕ := 2
def wine_glasses_per_guest : ℕ := 1
def juice_glasses_per_guest : ℕ := 1

def champagne_servings_per_bottle : ℕ := 6
def wine_servings_per_bottle : ℕ := 5
def juice_servings_per_bottle : ℕ := 4

theorem sparkling_juice_bottles (
  total_guests : ℕ)
  (juice_percentage : ℚ)
  (juice_glasses_per_guest : ℕ)
  (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : juice_percentage = 10 / 100)
  (h3 : juice_glasses_per_guest = 1)
  (h4 : juice_servings_per_bottle = 4)
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_sparkling_juice_bottles_l70_7066


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l70_7069

/-- Represents a systematic sampling of examination rooms -/
structure SystematicSampling where
  totalRooms : Nat
  sampleSize : Nat
  firstRoom : Nat
  interval : Nat

/-- Checks if a room number is part of the systematic sample -/
def isSelected (s : SystematicSampling) (room : Nat) : Prop :=
  ∃ k : Nat, room = s.firstRoom + k * s.interval ∧ room ≤ s.totalRooms

/-- The set of selected room numbers in a systematic sampling -/
def selectedRooms (s : SystematicSampling) : Set Nat :=
  {room | isSelected s room}

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalRooms = 64)
  (h2 : s.sampleSize = 8)
  (h3 : s.firstRoom = 5)
  (h4 : isSelected s 21)
  (h5 : s.interval = s.totalRooms / s.sampleSize) :
  selectedRooms s = {5, 13, 21, 29, 37, 45, 53, 61} := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_theorem_l70_7069


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l70_7098

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2/5)^(1/5) :=
by sorry

theorem min_value_equality :
  let x := (2/5)^(1/5)
  3 * x + 5 + 2 / x^5 = 10 + 3 * (2/5)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l70_7098


namespace NUMINAMATH_CALUDE_haircut_tip_percentage_l70_7045

theorem haircut_tip_percentage (womens_haircut_cost : ℝ) (childrens_haircut_cost : ℝ) 
  (num_children : ℕ) (tip_amount : ℝ) :
  womens_haircut_cost = 48 →
  childrens_haircut_cost = 36 →
  num_children = 2 →
  tip_amount = 24 →
  (tip_amount / (womens_haircut_cost + num_children * childrens_haircut_cost)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_haircut_tip_percentage_l70_7045


namespace NUMINAMATH_CALUDE_sqrt_product_equals_thirty_l70_7068

theorem sqrt_product_equals_thirty (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (5 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_thirty_l70_7068


namespace NUMINAMATH_CALUDE_remainder_b_96_mod_50_l70_7017

theorem remainder_b_96_mod_50 : (7^96 + 9^96) % 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_96_mod_50_l70_7017


namespace NUMINAMATH_CALUDE_tan_two_implies_expression_eq_neg_two_l70_7077

theorem tan_two_implies_expression_eq_neg_two (θ : Real) (h : Real.tan θ = 2) :
  (2 * Real.cos θ) / (Real.sin (π / 2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_expression_eq_neg_two_l70_7077


namespace NUMINAMATH_CALUDE_solution_set_inequality_l70_7060

theorem solution_set_inequality (x : ℝ) :
  (Set.Ioo (-2 : ℝ) 0) = {x | |1 + x + x^2/2| < 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l70_7060


namespace NUMINAMATH_CALUDE_f_inequality_l70_7096

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) : 
  f 2013 < Real.exp 2013 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l70_7096


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l70_7013

theorem real_roots_quadratic_equation (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x + 1 = 0) → m ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l70_7013


namespace NUMINAMATH_CALUDE_non_similar_1200_pointed_stars_l70_7044

/-- Definition of a regular n-pointed star (placeholder) -/
def RegularStar (n : ℕ) : Type := sorry

/-- Counts the number of non-similar regular n-pointed stars -/
def countNonSimilarStars (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

theorem non_similar_1200_pointed_stars :
  countNonSimilarStars 1200 = 160 :=
by sorry

end NUMINAMATH_CALUDE_non_similar_1200_pointed_stars_l70_7044


namespace NUMINAMATH_CALUDE_set_intersection_complement_l70_7006

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {1, 2, 3, 5}

-- Define set B
def B : Set Nat := {2, 4, 6}

-- Theorem statement
theorem set_intersection_complement : B ∩ (U \ A) = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l70_7006


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l70_7061

theorem unique_sequence_existence :
  ∃! (a : ℕ → ℤ), 
    a 1 = 1 ∧ 
    a 2 = 2 ∧ 
    ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l70_7061


namespace NUMINAMATH_CALUDE_men_seated_on_bus_l70_7040

theorem men_seated_on_bus (total_passengers : ℕ) 
  (h1 : total_passengers = 48) 
  (h2 : (2 : ℕ) * (total_passengers - total_passengers / 3) = total_passengers / 3) 
  (h3 : (8 : ℕ) * ((total_passengers - total_passengers / 3) / 8) = total_passengers - total_passengers / 3) :
  total_passengers - total_passengers / 3 - ((total_passengers - total_passengers / 3) / 8) = 14 := by
  sorry

#check men_seated_on_bus

end NUMINAMATH_CALUDE_men_seated_on_bus_l70_7040


namespace NUMINAMATH_CALUDE_tangent_ratio_equals_three_l70_7035

theorem tangent_ratio_equals_three (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_equals_three_l70_7035


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l70_7064

def eggplants : ℕ := 12
def carrots : ℕ := 9
def potatoes : ℕ := 8
def onions : ℕ := 15
def zucchinis : ℕ := 7
def work_days : ℕ := 6

def vegetables_per_day : ℕ := eggplants + carrots + potatoes + onions + zucchinis

theorem conor_weekly_vegetables :
  vegetables_per_day * work_days = 306 := by sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l70_7064


namespace NUMINAMATH_CALUDE_fifth_rectangle_is_square_l70_7007

-- Define the structure of a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure of a square
structure Square where
  side : ℝ

-- Define the division of a square into rectangles
def squareDivision (s : Square) (r1 r2 r3 r4 r5 : Rectangle) : Prop :=
  -- The sum of widths and heights of corner rectangles equals the square's side
  r1.width + r2.width = s.side ∧
  r1.height + r3.height = s.side ∧
  -- The areas of the four corner rectangles are equal
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height ∧
  -- The fifth rectangle doesn't touch the sides of the square
  r5.width < s.side - r1.width ∧
  r5.height < s.side - r1.height

-- Theorem statement
theorem fifth_rectangle_is_square 
  (s : Square) (r1 r2 r3 r4 r5 : Rectangle) 
  (h : squareDivision s r1 r2 r3 r4 r5) : 
  r5.width = r5.height :=
sorry

end NUMINAMATH_CALUDE_fifth_rectangle_is_square_l70_7007


namespace NUMINAMATH_CALUDE_book_pages_count_l70_7081

/-- Given a book with 24 chapters that Frank read in 6 days at a rate of 102 pages per day,
    prove that the total number of pages in the book is 612. -/
theorem book_pages_count (chapters : ℕ) (days : ℕ) (pages_per_day : ℕ) 
  (h1 : chapters = 24)
  (h2 : days = 6)
  (h3 : pages_per_day = 102) :
  chapters * (days * pages_per_day) / chapters = 612 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l70_7081


namespace NUMINAMATH_CALUDE_test_failure_probability_l70_7046

theorem test_failure_probability
  (total : ℕ)
  (passed_first : ℕ)
  (passed_second : ℕ)
  (passed_third : ℕ)
  (passed_first_and_second : ℕ)
  (passed_second_and_third : ℕ)
  (passed_first_and_third : ℕ)
  (passed_all : ℕ)
  (h_total : total = 200)
  (h_first : passed_first = 110)
  (h_second : passed_second = 80)
  (h_third : passed_third = 70)
  (h_first_second : passed_first_and_second = 35)
  (h_second_third : passed_second_and_third = 30)
  (h_first_third : passed_first_and_third = 40)
  (h_all : passed_all = 20) :
  (total - (passed_first + passed_second + passed_third
          - passed_first_and_second - passed_second_and_third - passed_first_and_third
          + passed_all)) / total = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_test_failure_probability_l70_7046


namespace NUMINAMATH_CALUDE_bowling_team_score_l70_7075

/-- Represents the scores of a bowling team with three members -/
structure BowlingTeam where
  first_bowler : ℕ
  second_bowler : ℕ
  third_bowler : ℕ

/-- Calculates the total score of a bowling team -/
def total_score (team : BowlingTeam) : ℕ :=
  team.first_bowler + team.second_bowler + team.third_bowler

/-- Theorem stating the total score of the bowling team under given conditions -/
theorem bowling_team_score :
  ∃ (team : BowlingTeam),
    team.third_bowler = 162 ∧
    team.second_bowler = 3 * team.third_bowler ∧
    team.first_bowler = team.second_bowler / 3 ∧
    total_score team = 810 := by
  sorry


end NUMINAMATH_CALUDE_bowling_team_score_l70_7075


namespace NUMINAMATH_CALUDE_lucky_larry_calculation_l70_7020

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 16 ∧ b = 2 ∧ c = 3 ∧ d = 12 → 
  (a / (b / (c * (d / e))) = a / b / c * d / e) → 
  e = 9 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_calculation_l70_7020


namespace NUMINAMATH_CALUDE_fraction_equality_l70_7086

theorem fraction_equality (A B : ℤ) (x : ℝ) :
  (A / (x - 2) + B / (x^2 - 4*x + 8) = (x^2 - 4*x + 18) / (x^3 - 6*x^2 + 16*x - 16)) →
  (x ≠ 2 ∧ x ≠ 4 ∧ x^2 - 4*x + 8 ≠ 0) →
  B / A = -4 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l70_7086


namespace NUMINAMATH_CALUDE_large_pizza_price_l70_7049

/-- The price of a large pizza given the sales information -/
theorem large_pizza_price (small_price : ℕ) (total_sales : ℕ) (small_sold : ℕ) (large_sold : ℕ)
  (h1 : small_price = 2)
  (h2 : total_sales = 40)
  (h3 : small_sold = 8)
  (h4 : large_sold = 3) :
  (total_sales - small_price * small_sold) / large_sold = 8 := by
  sorry

#check large_pizza_price

end NUMINAMATH_CALUDE_large_pizza_price_l70_7049


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l70_7090

theorem complex_multiplication_result : (1 + 2 * Complex.I) * (1 - Complex.I) = 3 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l70_7090


namespace NUMINAMATH_CALUDE_shaded_quadrilateral_area_l70_7084

theorem shaded_quadrilateral_area : 
  let small_square_side : ℝ := 3
  let medium_square_side : ℝ := 5
  let large_square_side : ℝ := 7
  let total_base : ℝ := small_square_side + medium_square_side + large_square_side
  let diagonal_slope : ℝ := large_square_side / total_base
  let small_triangle_height : ℝ := small_square_side * diagonal_slope
  let medium_triangle_height : ℝ := (small_square_side + medium_square_side) * diagonal_slope
  let trapezoid_area : ℝ := (medium_square_side * (small_triangle_height + medium_triangle_height)) / 2
  trapezoid_area = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_shaded_quadrilateral_area_l70_7084


namespace NUMINAMATH_CALUDE_smallest_sum_prime_set_l70_7032

/-- A set of natural numbers uses each digit exactly once -/
def uses_each_digit_once (s : Finset ℕ) : Prop :=
  ∃ (digits : Finset ℕ), digits.card = 10 ∧
    ∀ d ∈ digits, 0 ≤ d ∧ d < 10 ∧
    ∀ n ∈ s, ∀ k, 0 ≤ k ∧ k < 10 → (n / 10^k % 10) ∈ digits

/-- The sum of a set of natural numbers -/
def set_sum (s : Finset ℕ) : ℕ := s.sum id

/-- The theorem to be proved -/
theorem smallest_sum_prime_set :
  ∃ (s : Finset ℕ),
    (∀ n ∈ s, Nat.Prime n) ∧
    uses_each_digit_once s ∧
    set_sum s = 4420 ∧
    (∀ t : Finset ℕ, (∀ n ∈ t, Nat.Prime n) → uses_each_digit_once t → set_sum s ≤ set_sum t) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_prime_set_l70_7032


namespace NUMINAMATH_CALUDE_snowman_volume_snowman_volume_calculation_l70_7051

theorem snowman_volume (π : ℝ) : ℝ → ℝ → ℝ → ℝ :=
  fun r₁ r₂ r₃ =>
    let sphere_volume := fun r : ℝ => (4 / 3) * π * r^3
    sphere_volume r₁ + sphere_volume r₂ + sphere_volume r₃

theorem snowman_volume_calculation (π : ℝ) :
  snowman_volume π 4 5 6 = (1620 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_snowman_volume_snowman_volume_calculation_l70_7051


namespace NUMINAMATH_CALUDE_min_value_theorem_l70_7022

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4*x + 3*y = 1) :
  ∀ z : ℝ, z = (1 / (2*x - y)) + (2 / (x + 2*y)) → z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l70_7022


namespace NUMINAMATH_CALUDE_apple_pie_calculation_l70_7038

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) :
  total_apples = 34 →
  unripe_apples = 6 →
  apples_per_pie = 4 →
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_calculation_l70_7038


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l70_7039

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + 2kx + 9 is a perfect square trinomial, then k = ±6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 4 (2 * k) 9 → k = 6 ∨ k = -6 :=
by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l70_7039


namespace NUMINAMATH_CALUDE_projection_theorem_l70_7010

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

theorem projection_theorem (Q : Plane) :
  project (7, 1, 8) Q = (6, 3, 2) →
  project (6, 2, 9) Q = (9/2, 5, 9/2) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l70_7010


namespace NUMINAMATH_CALUDE_bicycle_distance_theorem_l70_7034

/-- Represents a bicycle wheel -/
structure Wheel where
  perimeter : ℝ

/-- Represents a bicycle with two wheels -/
structure Bicycle where
  backWheel : Wheel
  frontWheel : Wheel

/-- Calculates the distance traveled by a wheel given the number of revolutions -/
def distanceTraveled (wheel : Wheel) (revolutions : ℝ) : ℝ :=
  wheel.perimeter * revolutions

theorem bicycle_distance_theorem (bike : Bicycle) 
    (h1 : bike.backWheel.perimeter = 9)
    (h2 : bike.frontWheel.perimeter = 7)
    (h3 : ∃ (r : ℝ), distanceTraveled bike.backWheel r = distanceTraveled bike.frontWheel (r + 10)) :
  ∃ (d : ℝ), d = 315 ∧ ∃ (r : ℝ), d = distanceTraveled bike.backWheel r ∧ d = distanceTraveled bike.frontWheel (r + 10) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_theorem_l70_7034


namespace NUMINAMATH_CALUDE_hundred_with_five_twos_l70_7041

theorem hundred_with_five_twos :
  ∃ (a b c d e : ℕ), a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  (a * b * c / d - e / d = 100) := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_five_twos_l70_7041


namespace NUMINAMATH_CALUDE_range_of_a_l70_7048

-- Define the functions f and g
def f (x : ℝ) := 3 * abs (x - 1) + abs (3 * x + 1)
def g (a : ℝ) (x : ℝ) := abs (x + 2) + abs (x - a)

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, f x = y}
def B (a : ℝ) : Set ℝ := {y | ∃ x, g a x = y}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (a ∈ Set.Icc (-6) 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l70_7048


namespace NUMINAMATH_CALUDE_hotel_reunion_attendees_l70_7055

theorem hotel_reunion_attendees (total_guests dates_attendees hall_attendees : ℕ) 
  (h1 : total_guests = 50)
  (h2 : dates_attendees = 50)
  (h3 : hall_attendees = 60)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ dates_attendees ∨ g ≤ hall_attendees)) :
  dates_attendees + hall_attendees - total_guests = 60 := by
  sorry

end NUMINAMATH_CALUDE_hotel_reunion_attendees_l70_7055


namespace NUMINAMATH_CALUDE_range_of_a_l70_7028

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, a ≥ -x^2 + 2*x - 2/3) ∧ 
  (∃ x : ℝ, x^2 + 4*x + a = 0) ↔ 
  a ∈ Set.Icc (1/3) 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l70_7028


namespace NUMINAMATH_CALUDE_find_genuine_coin_l70_7083

/-- Represents a coin, which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : ℕ → Coin

/-- Represents the result of weighing two coins -/
inductive WeighingResult
| equal : WeighingResult
| unequal : WeighingResult

/-- Represents a collection of coins -/
def CoinSet := List Coin

/-- Represents a weighing action -/
def Weighing := Coin → Coin → WeighingResult

/-- Represents a strategy to find a genuine coin -/
def Strategy := CoinSet → Weighing → Option Coin

theorem find_genuine_coin 
  (coins : CoinSet) 
  (h_total : coins.length = 9)
  (h_counterfeit : (coins.filter (λ c => match c with 
    | Coin.counterfeit _ => true 
    | _ => false)).length = 4)
  (h_genuine_equal : ∀ c1 c2, c1 = Coin.genuine ∧ c2 = Coin.genuine → 
    (λ _ _ => WeighingResult.equal) c1 c2 = WeighingResult.equal)
  (h_counterfeit_differ : ∀ c1 c2, c1 ≠ c2 → 
    (c1 = Coin.genuine ∨ (∃ n, c1 = Coin.counterfeit n)) ∧ 
    (c2 = Coin.genuine ∨ (∃ m, c2 = Coin.counterfeit m)) → 
    (λ _ _ => WeighingResult.unequal) c1 c2 = WeighingResult.unequal)
  : ∃ (s : Strategy), ∀ w : Weighing, 
    (∃ c, s coins w = some c ∧ c = Coin.genuine) ∧ 
    (s coins w).isSome → (Nat.card {p : Coin × Coin | w p.1 p.2 ≠ WeighingResult.equal}) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_find_genuine_coin_l70_7083


namespace NUMINAMATH_CALUDE_boys_cant_score_double_l70_7059

/-- Represents a participant in the chess tournament -/
inductive Participant
| Boy
| Girl

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- Calculates the number of games in a round-robin tournament -/
def numGames (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total points from a list of game results -/
def totalPoints (results : List GameResult) : Rat :=
  results.foldl (fun acc res => acc + match res with
    | GameResult.Win => 1
    | GameResult.Draw => 1/2
    | GameResult.Loss => 0) 0

/-- Represents the tournament results -/
structure TournamentResults where
  boyResults : List GameResult
  girlResults : List GameResult

/-- The main theorem to be proved -/
theorem boys_cant_score_double : 
  ∀ (results : TournamentResults), 
  (numGames 6 = results.boyResults.length + results.girlResults.length) →
  (totalPoints results.boyResults ≠ 2 * totalPoints results.girlResults) :=
sorry

end NUMINAMATH_CALUDE_boys_cant_score_double_l70_7059


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorial_and_square_10_l70_7063

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def sum_factorial_and_square (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + factorial (i + 1) + (i + 1)^2) 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorial_and_square_10 : 
  units_digit (sum_factorial_and_square 10) = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorial_and_square_10_l70_7063


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l70_7076

theorem max_value_sqrt_sum (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l70_7076


namespace NUMINAMATH_CALUDE_license_plate_count_l70_7070

-- Define the number of possible digits (0-9)
def num_digits : ℕ := 10

-- Define the number of possible letters (A-Z)
def num_letters : ℕ := 26

-- Define the number of digits in the license plate
def digits_in_plate : ℕ := 4

-- Define the number of letters in the license plate
def letters_in_plate : ℕ := 2

-- Define the number of positions where the letter pair can be placed
def letter_pair_positions : ℕ := digits_in_plate + 1

-- Theorem statement
theorem license_plate_count :
  letter_pair_positions * num_digits ^ digits_in_plate * num_letters ^ letters_in_plate = 33800000 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l70_7070


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l70_7089

theorem absolute_value_theorem (x : ℝ) (h : x < 1) : 
  |x - Real.sqrt ((x - 2)^2)| = 2 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l70_7089


namespace NUMINAMATH_CALUDE_equation_solutions_l70_7027

theorem equation_solutions :
  (∀ x : ℚ, 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2) ∧
  (∀ x : ℚ, 2 * x^2 + 3 * x - 5 = 0 ↔ x = -5/2 ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l70_7027


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l70_7016

theorem simplify_complex_fraction : 
  1 / ((1 / (Real.sqrt 3 + 2)) + (2 / (Real.sqrt 5 - 2))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l70_7016


namespace NUMINAMATH_CALUDE_inequality_proof_l70_7014

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l70_7014


namespace NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l70_7030

-- Define the geometric bodies
inductive GeometricBody
| Pyramid
| Prism
| Frustum
| Cuboid

-- Define a function to count parallel surfaces
def parallelSurfaces : GeometricBody → ℕ
| GeometricBody.Pyramid => 0
| GeometricBody.Prism => 6
| GeometricBody.Frustum => 2
| GeometricBody.Cuboid => 6

-- Theorem: Only the frustum has exactly two parallel surfaces
theorem only_frustum_has_two_parallel_surfaces :
  ∀ b : GeometricBody, parallelSurfaces b = 2 ↔ b = GeometricBody.Frustum :=
by sorry

end NUMINAMATH_CALUDE_only_frustum_has_two_parallel_surfaces_l70_7030


namespace NUMINAMATH_CALUDE_impossible_to_cover_l70_7000

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (initial_difference : ℤ)

/-- Represents a piece that can be placed on the chessboard -/
inductive Piece
  | horizontal : Piece  -- 1 × 2 piece
  | vertical : Piece    -- 1 × 3 piece

/-- Defines the effect of placing a piece on the chessboard -/
def place_piece (p : Piece) : ℤ :=
  match p with
  | Piece.horizontal => 0
  | Piece.vertical => 3

/-- Theorem stating that it's impossible to cover the chessboard -/
theorem impossible_to_cover (board : Chessboard) 
  (h1 : board.size = 2003)
  (h2 : board.initial_difference = 2003) :
  ¬ ∃ (arrangement : List Piece), 
    (arrangement.foldl (λ acc p => acc + place_piece p) 0 = -board.initial_difference) :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_cover_l70_7000


namespace NUMINAMATH_CALUDE_pta_fundraiser_l70_7002

theorem pta_fundraiser (initial_amount : ℝ) (school_supplies_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 400 →
  school_supplies_fraction = 1/4 →
  food_fraction = 1/2 →
  initial_amount * (1 - school_supplies_fraction) * (1 - food_fraction) = 150 := by
sorry

end NUMINAMATH_CALUDE_pta_fundraiser_l70_7002


namespace NUMINAMATH_CALUDE_luke_final_sticker_count_l70_7018

/-- Calculates the final number of stickers Luke has after various transactions -/
def final_sticker_count (initial : ℕ) (bought : ℕ) (from_friend : ℕ) (birthday : ℕ) 
                        (traded_out : ℕ) (traded_in : ℕ) (to_sister : ℕ) 
                        (for_card : ℕ) (to_charity : ℕ) : ℕ :=
  initial + bought + from_friend + birthday - traded_out + traded_in - to_sister - for_card - to_charity

/-- Theorem stating that Luke ends up with 67 stickers -/
theorem luke_final_sticker_count :
  final_sticker_count 20 12 25 30 10 15 5 8 12 = 67 := by
  sorry

end NUMINAMATH_CALUDE_luke_final_sticker_count_l70_7018


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l70_7009

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (46 * x + 8) % 24 = 4 ∧ ∀ (y : ℕ), y > 0 ∧ (46 * y + 8) % 24 = 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l70_7009


namespace NUMINAMATH_CALUDE_polynomial_with_triple_roots_l70_7031

def p : ℝ → ℝ := fun x ↦ 12 * x^5 - 30 * x^4 + 20 * x^3 - 1

theorem polynomial_with_triple_roots :
  (∀ x : ℝ, (∃ q : ℝ → ℝ, p x + 1 = x^3 * q x)) ∧
  (∀ x : ℝ, (∃ r : ℝ → ℝ, p x - 1 = (x - 1)^3 * r x)) →
  ∀ x : ℝ, p x = 12 * x^5 - 30 * x^4 + 20 * x^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_with_triple_roots_l70_7031


namespace NUMINAMATH_CALUDE_product_equals_square_minus_one_l70_7021

theorem product_equals_square_minus_one (r : ℕ) (hr : r > 5) :
  let a := r^3 + r^2 + r
  (a) * (a + 1) * (a + 2) * (a + 3) = (r^6 + 2*r^5 + 3*r^4 + 5*r^3 + 4*r^2 + 3*r + 1)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_product_equals_square_minus_one_l70_7021


namespace NUMINAMATH_CALUDE_course_failure_implies_question_failure_l70_7085

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed_course : Student → Prop)
variable (failed_no_questions : Student → Prop)

-- Ms. Johnson's statement
variable (johnsons_statement : ∀ s : Student, failed_no_questions s → passed_course s)

-- Theorem to prove
theorem course_failure_implies_question_failure :
  ∀ s : Student, ¬(passed_course s) → ¬(failed_no_questions s) :=
by sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_course_failure_implies_question_failure_l70_7085


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l70_7019

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4 * i) / (1 + 2 * i) = 11 / 5 - 2 / 5 * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l70_7019


namespace NUMINAMATH_CALUDE_intersection_distance_l70_7047

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 2.5

-- Define the shared focus
def shared_focus : ℝ × ℝ := (5, 0)

-- Define the directrix of the parabola
def parabola_directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ,
    hyperbola p1.1 p1.2 ∧
    hyperbola p2.1 p2.2 ∧
    parabola p1.1 p1.2 ∧
    parabola p2.1 p2.2 ∧
    p1 ≠ p2 ∧
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 4 * Real.sqrt 218 / 15 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l70_7047


namespace NUMINAMATH_CALUDE_rainfall_third_week_l70_7072

theorem rainfall_third_week (total : ℝ) (week1 : ℝ) (week2 : ℝ) (week3 : ℝ)
  (h_total : total = 45)
  (h_week2 : week2 = 1.5 * week1)
  (h_week3 : week3 = 2 * week2)
  (h_sum : week1 + week2 + week3 = total) :
  week3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_third_week_l70_7072


namespace NUMINAMATH_CALUDE_mission_duration_l70_7052

theorem mission_duration (planned_duration : ℝ) : 
  (1.6 * planned_duration + 3 = 11) → planned_duration = 5 := by
  sorry

end NUMINAMATH_CALUDE_mission_duration_l70_7052


namespace NUMINAMATH_CALUDE_domain_of_g_l70_7087

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-3) 5

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 1) + f (x - 2)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-1) 4

-- Theorem statement
theorem domain_of_g :
  ∀ x ∈ domain_g, (x + 1 ∈ domain_f ∧ x - 2 ∈ domain_f) ∧
  ∀ x ∉ domain_g, (x + 1 ∉ domain_f ∨ x - 2 ∉ domain_f) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l70_7087


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l70_7093

-- Define the function f with domain [-1, 2]
def f : Set ℝ := Set.Icc (-1) 2

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Icc 0 (3/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l70_7093


namespace NUMINAMATH_CALUDE_speed_increase_ratio_l70_7079

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) :
  (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_ratio_l70_7079


namespace NUMINAMATH_CALUDE_area_of_right_trapezoid_l70_7057

/-- 
Given a horizontally placed right trapezoid whose oblique axonometric projection
is an isosceles trapezoid with a bottom angle of 45°, legs of length 1, and 
top base of length 1, the area of the original right trapezoid is 2 + √2.
-/
theorem area_of_right_trapezoid (h : ℝ) (w : ℝ) :
  h = 2 →
  w = 1 + Real.sqrt 2 →
  (1 / 2 : ℝ) * (w + 1) * h = 2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_area_of_right_trapezoid_l70_7057


namespace NUMINAMATH_CALUDE_line_intercept_l70_7036

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line is the x-coordinate where the line crosses the x-axis -/
def x_intercept (l : Line) : ℝ := sorry

/-- Theorem: The line passing through (7, -3) and (3, 1) intersects the x-axis at (4, 0) -/
theorem line_intercept : 
  let l : Line := { x₁ := 7, y₁ := -3, x₂ := 3, y₂ := 1 }
  x_intercept l = 4 := by sorry

end NUMINAMATH_CALUDE_line_intercept_l70_7036


namespace NUMINAMATH_CALUDE_loanYears_correct_l70_7001

/-- Calculates the number of years for which the first part of a loan is lent, given the following conditions:
  * The total sum is 2704
  * The second part of the loan is 1664
  * The interest rate for the first part is 3% per annum
  * The interest rate for the second part is 5% per annum
  * The interest period for the second part is 3 years
  * The interest on the first part equals the interest on the second part
-/
def loanYears : ℕ :=
  let totalSum : ℕ := 2704
  let secondPart : ℕ := 1664
  let firstPartRate : ℚ := 3 / 100
  let secondPartRate : ℚ := 5 / 100
  let secondPartPeriod : ℕ := 3
  let firstPart : ℕ := totalSum - secondPart
  8

theorem loanYears_correct : loanYears = 8 := by sorry

end NUMINAMATH_CALUDE_loanYears_correct_l70_7001


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_pow_nineteen_l70_7012

/-- The sum of the tens digit and the ones digit of 7^19 is 7 -/
theorem sum_of_digits_seven_pow_nineteen : 
  (((7^19) / 10) % 10) + ((7^19) % 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_pow_nineteen_l70_7012


namespace NUMINAMATH_CALUDE_heejin_drinks_most_l70_7094

-- Define the drinking habits
def dongguk_frequency : ℕ := 5
def dongguk_amount : ℝ := 0.2

def yoonji_frequency : ℕ := 6
def yoonji_amount : ℝ := 0.3

def heejin_frequency : ℕ := 4
def heejin_amount : ℝ := 0.5  -- 500 ml = 0.5 L

-- Calculate total daily water intake for each person
def dongguk_total : ℝ := dongguk_frequency * dongguk_amount
def yoonji_total : ℝ := yoonji_frequency * yoonji_amount
def heejin_total : ℝ := heejin_frequency * heejin_amount

-- Theorem stating Heejin drinks the most water
theorem heejin_drinks_most : 
  heejin_total > dongguk_total ∧ heejin_total > yoonji_total :=
by sorry

end NUMINAMATH_CALUDE_heejin_drinks_most_l70_7094


namespace NUMINAMATH_CALUDE_room_width_calculation_l70_7062

theorem room_width_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) :
  room_length = 13 →
  carpet_width = 0.75 →
  carpet_cost_per_sqm = 12 →
  total_cost = 1872 →
  room_length * (total_cost / (room_length * carpet_cost_per_sqm)) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l70_7062


namespace NUMINAMATH_CALUDE_polygon_area_l70_7026

-- Define the polygon as a list of points
def polygon : List (ℕ × ℕ) := [(0, 0), (5, 0), (5, 2), (3, 2), (3, 3), (2, 3), (2, 2), (0, 2), (0, 0)]

-- Define a function to calculate the area of a polygon given its vertices
def calculatePolygonArea (vertices : List (ℕ × ℕ)) : ℕ := sorry

-- Theorem statement
theorem polygon_area : calculatePolygonArea polygon = 11 := by sorry

end NUMINAMATH_CALUDE_polygon_area_l70_7026


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l70_7050

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a4 : a 4 = 7) 
  (h_sum : a 3 + a 6 = 16) : 
  ∃ d : ℝ, (∀ n, a (n + 1) - a n = d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l70_7050


namespace NUMINAMATH_CALUDE_square_binomial_expansion_l70_7091

theorem square_binomial_expansion (x : ℝ) : (x - 2)^2 = x^2 - 4*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_expansion_l70_7091


namespace NUMINAMATH_CALUDE_wade_tips_theorem_l70_7095

def tips_per_customer : ℕ := 2
def friday_customers : ℕ := 28
def sunday_customers : ℕ := 36

def saturday_customers : ℕ := 3 * friday_customers

def total_tips : ℕ := tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

theorem wade_tips_theorem : total_tips = 296 := by
  sorry

end NUMINAMATH_CALUDE_wade_tips_theorem_l70_7095


namespace NUMINAMATH_CALUDE_division_problem_l70_7003

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = 172 →
  divisor = 17 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l70_7003


namespace NUMINAMATH_CALUDE_chef_nuts_weight_l70_7023

/-- The total weight of nuts bought by a chef -/
def total_weight (almonds pecans walnuts cashews pistachios : ℝ) : ℝ :=
  almonds + pecans + walnuts + cashews + pistachios

/-- Theorem stating that the total weight of nuts is 1.50 kg -/
theorem chef_nuts_weight :
  let almonds : ℝ := 0.14
  let pecans : ℝ := 0.38
  let walnuts : ℝ := 0.22
  let cashews : ℝ := 0.47
  let pistachios : ℝ := 0.29
  total_weight almonds pecans walnuts cashews pistachios = 1.50 := by
  sorry

end NUMINAMATH_CALUDE_chef_nuts_weight_l70_7023


namespace NUMINAMATH_CALUDE_smallest_number_proof_l70_7078

def smallest_number : ℕ := 271562

theorem smallest_number_proof :
  smallest_number = 271562 ∧
  ∃ k : ℕ, (smallest_number - 18) = k * lcm 14 (lcm 26 28) ∧
  k = 746 ∧
  ∀ y : ℕ, y < smallest_number →
    ¬(∃ m : ℕ, (y - 18) = m * lcm 14 (lcm 26 28) ∧ m = 746) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l70_7078


namespace NUMINAMATH_CALUDE_apple_capacity_l70_7099

def bookbag_capacity : ℕ := 20
def other_fruit_weight : ℕ := 3

theorem apple_capacity : bookbag_capacity - other_fruit_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_apple_capacity_l70_7099


namespace NUMINAMATH_CALUDE_outfit_choices_l70_7025

/-- The number of color options for each item type -/
def num_colors : ℕ := 6

/-- The number of item types in an outfit -/
def num_items : ℕ := 4

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_colors ^ num_items

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- The number of valid outfits (excluding those with all items of the same color) -/
def valid_outfits : ℕ := total_combinations - same_color_outfits

theorem outfit_choices :
  valid_outfits = 1290 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l70_7025


namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l70_7042

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 10 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ d : ℝ, d = (5 : ℝ) / 2 ∧ 
    ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ line1 → Q ∈ line2 → 
      d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l70_7042


namespace NUMINAMATH_CALUDE_boys_in_second_grade_l70_7088

/-- The number of students in the 3rd grade -/
def third_grade : ℕ := 19

/-- The number of students in the 4th grade -/
def fourth_grade : ℕ := 2 * third_grade

/-- The number of girls in the 2nd grade -/
def second_grade_girls : ℕ := 19

/-- The total number of students across all three grades -/
def total_students : ℕ := 86

/-- The number of boys in the 2nd grade -/
def second_grade_boys : ℕ := total_students - fourth_grade - third_grade - second_grade_girls

theorem boys_in_second_grade : second_grade_boys = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_second_grade_l70_7088


namespace NUMINAMATH_CALUDE_equal_distribution_of_stickers_l70_7008

/-- The number of stickers Haley has -/
def total_stickers : ℕ := 72

/-- The number of Haley's friends -/
def num_friends : ℕ := 9

/-- The number of stickers each friend will receive -/
def stickers_per_friend : ℕ := total_stickers / num_friends

theorem equal_distribution_of_stickers :
  stickers_per_friend * num_friends = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_of_stickers_l70_7008


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l70_7056

theorem polynomial_factorization_sum (a₁ a₂ c₁ b₂ c₂ : ℝ) 
  (h : ∀ x : ℝ, x^5 - x^4 + x^3 - x^2 + x - 1 = (x^3 + a₁*x^2 + a₂*x + c₁)*(x^2 + b₂*x + c₂)) :
  a₁*c₁ + b₂*c₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l70_7056


namespace NUMINAMATH_CALUDE_tv_price_proof_l70_7058

theorem tv_price_proof (X : ℝ) : 
  X * (1 + 0.4) * 0.8 - X = 270 → X = 2250 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_proof_l70_7058


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l70_7092

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l70_7092


namespace NUMINAMATH_CALUDE_class_gender_ratio_l70_7074

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls = boys + 6 →
  girls + boys = 36 →
  (girls : ℚ) / (boys : ℚ) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l70_7074


namespace NUMINAMATH_CALUDE_max_profit_at_90_l70_7080

noncomputable section

-- Define the cost function
def cost (x : ℝ) : ℝ :=
  if x < 90 then 0.5 * x^2 + 60 * x + 5
  else 121 * x + 8100 / x - 2180 + 5

-- Define the revenue function
def revenue (x : ℝ) : ℝ := 1.2 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := revenue x - cost x

-- Theorem statement
theorem max_profit_at_90 :
  ∀ x > 0, profit x ≤ profit 90 ∧ profit 90 = 1500 := by sorry

end

end NUMINAMATH_CALUDE_max_profit_at_90_l70_7080


namespace NUMINAMATH_CALUDE_expression_evaluation_l70_7043

theorem expression_evaluation : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l70_7043


namespace NUMINAMATH_CALUDE_equivalent_expression_l70_7097

theorem equivalent_expression (a b c d : ℝ) (h1 : a = 0.37) (h2 : b = 15) (h3 : c = 3.7) (h4 : d = 1.5) (h5 : c = a * 10) (h6 : d = b / 10) : c * d = a * b := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l70_7097


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l70_7015

/-- Definition of a pentagonal prism -/
structure PentagonalPrism where
  bases : ℕ := 2
  connecting_faces : ℕ := 5
  edges_per_base : ℕ := 5
  vertices_per_base : ℕ := 5

/-- Theorem: The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : 
  (p.bases + p.connecting_faces) + 
  (p.edges_per_base * 2 + p.edges_per_base) + 
  (p.vertices_per_base * 2) = 32 := by
  sorry

#check pentagonal_prism_sum

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l70_7015


namespace NUMINAMATH_CALUDE_delta_value_l70_7011

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l70_7011


namespace NUMINAMATH_CALUDE_sum_of_local_values_equals_number_l70_7037

/-- The local value of a digit in a number -/
def local_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- The number we're considering -/
def number : ℕ := 2345

/-- Theorem: The sum of local values of digits in 2345 equals 2345 -/
theorem sum_of_local_values_equals_number :
  local_value 2 3 + local_value 3 2 + local_value 4 1 + local_value 5 0 = number := by
  sorry

end NUMINAMATH_CALUDE_sum_of_local_values_equals_number_l70_7037


namespace NUMINAMATH_CALUDE_sam_container_capacity_l70_7029

/-- Represents a rectangular container with dimensions and marble capacity. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℕ

/-- Calculates the volume of a container. -/
def containerVolume (c : Container) : ℝ :=
  c.length * c.width * c.height

/-- Theorem: Given Ellie's container dimensions and capacity, and the relative dimensions
    of Sam's container, Sam's container holds 1200 marbles. -/
theorem sam_container_capacity
  (ellie : Container)
  (h_ellie_dims : ellie.length = 2 ∧ ellie.width = 3 ∧ ellie.height = 4)
  (h_ellie_capacity : ellie.capacity = 200)
  (sam : Container)
  (h_sam_dims : sam.length = ellie.length ∧ 
                sam.width = 2 * ellie.width ∧ 
                sam.height = 3 * ellie.height) :
  sam.capacity = 1200 := by
sorry


end NUMINAMATH_CALUDE_sam_container_capacity_l70_7029


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l70_7054

theorem abs_sum_minimum (x : ℝ) : 
  |x - 4| + |x - 6| ≥ 2 ∧ ∃ y : ℝ, |y - 4| + |y - 6| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l70_7054


namespace NUMINAMATH_CALUDE_smallest_a_value_l70_7033

/-- Given two quadratic equations with integer roots less than -1, find the smallest possible 'a' -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) →
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1) →
    a' ≥ a) →
  a = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l70_7033


namespace NUMINAMATH_CALUDE_square_of_difference_l70_7024

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l70_7024


namespace NUMINAMATH_CALUDE_carson_total_stars_l70_7082

/-- The number of gold stars Carson earned yesterday -/
def stars_yesterday : ℕ := 6

/-- The number of gold stars Carson earned today -/
def stars_today : ℕ := 9

/-- The total number of gold stars Carson earned -/
def total_stars : ℕ := stars_yesterday + stars_today

theorem carson_total_stars : total_stars = 15 := by
  sorry

end NUMINAMATH_CALUDE_carson_total_stars_l70_7082


namespace NUMINAMATH_CALUDE_max_value_quadratic_l70_7004

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 9) : 
  x^2 + 2*x*y + 3*y^2 ≤ 18 + 6*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l70_7004


namespace NUMINAMATH_CALUDE_no_real_roots_l70_7071

def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(17*(n+1)) - P n x

theorem no_real_roots : ∀ (n : ℕ) (x : ℝ), P n x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l70_7071
