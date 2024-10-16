import Mathlib

namespace NUMINAMATH_CALUDE_dawn_at_6am_l2831_283131

/-- Represents the time of dawn in hours before noon -/
def dawn_time : ℝ := 6

/-- Represents the time (in hours after noon) when the first pedestrian arrives at B -/
def arrival_time_B : ℝ := 4

/-- Represents the time (in hours after noon) when the second pedestrian arrives at A -/
def arrival_time_A : ℝ := 9

/-- The theorem states that given the conditions of the problem, dawn occurred at 6 AM -/
theorem dawn_at_6am :
  dawn_time * arrival_time_B = arrival_time_A * dawn_time ∧
  dawn_time + arrival_time_B + dawn_time + arrival_time_A = 24 →
  dawn_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_dawn_at_6am_l2831_283131


namespace NUMINAMATH_CALUDE_bus_intersection_percentages_l2831_283189

theorem bus_intersection_percentages : 
  let first_intersection_entrants : ℝ := 12
  let second_intersection_entrants : ℝ := 18
  let third_intersection_entrants : ℝ := 15
  (0.3 * first_intersection_entrants + 
   0.5 * second_intersection_entrants + 
   0.2 * third_intersection_entrants) = 15.6 := by
  sorry

end NUMINAMATH_CALUDE_bus_intersection_percentages_l2831_283189


namespace NUMINAMATH_CALUDE_remaining_blue_fraction_after_four_changes_l2831_283166

/-- The fraction of a square's area that remains blue after one change -/
def blue_fraction_after_one_change : ℚ := 8 / 9

/-- The number of changes applied to the square -/
def num_changes : ℕ := 4

/-- The fraction of the original area that remains blue after the specified number of changes -/
def remaining_blue_fraction : ℚ := blue_fraction_after_one_change ^ num_changes

/-- Theorem stating that the remaining blue fraction after four changes is 4096/6561 -/
theorem remaining_blue_fraction_after_four_changes :
  remaining_blue_fraction = 4096 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_remaining_blue_fraction_after_four_changes_l2831_283166


namespace NUMINAMATH_CALUDE_school_outing_buses_sufficient_l2831_283109

/-- Proves that the total capacity of 6 large buses is sufficient to accommodate 298 students. -/
theorem school_outing_buses_sufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 298 → 
  bus_capacity = 52 → 
  num_buses = 6 → 
  num_buses * bus_capacity ≥ students := by
sorry

end NUMINAMATH_CALUDE_school_outing_buses_sufficient_l2831_283109


namespace NUMINAMATH_CALUDE_correct_proposition_l2831_283111

-- Define proposition p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define proposition p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

-- Theorem statement
theorem correct_proposition : (¬p₁) ∧ p₂ := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l2831_283111


namespace NUMINAMATH_CALUDE_function_monotonicity_l2831_283174

/-- f is an odd function -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- g is an even function -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

/-- The main theorem -/
theorem function_monotonicity (f g : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_even : IsEven g) 
    (h_sum : ∀ x, f x + g x = 3^x) :
    ∀ a b, a > b → f a > f b := by
  sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2831_283174


namespace NUMINAMATH_CALUDE_expression_evaluation_l2831_283108

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -2
  (a + 2*b) * (a - b) + (a^3*b + 4*a*b^3) / (a*b) = 15/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2831_283108


namespace NUMINAMATH_CALUDE_f_properties_l2831_283127

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem f_properties :
  -- Monotonicity properties
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- Maximum and minimum values
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 59) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -49) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2831_283127


namespace NUMINAMATH_CALUDE_bc_is_one_twelfth_of_ad_l2831_283175

/-- Given a line segment AD with points B and C on it, prove that BC is 1/12 of AD -/
theorem bc_is_one_twelfth_of_ad (A B C D : ℝ) : 
  (B ≤ C) →  -- B is before or at C on the line
  (C ≤ D) →  -- C is before or at D on the line
  (A ≤ B) →  -- A is before or at B on the line
  (B - A = 3 * (D - B)) →  -- AB is 3 times BD
  (C - A = 5 * (D - C)) →  -- AC is 5 times CD
  (C - B = (D - A) / 12) := by  -- BC is 1/12 of AD
sorry

end NUMINAMATH_CALUDE_bc_is_one_twelfth_of_ad_l2831_283175


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l2831_283105

theorem rectangular_field_perimeter : ∀ (length breadth : ℝ),
  breadth = 0.6 * length →
  length * breadth = 37500 →
  2 * (length + breadth) = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l2831_283105


namespace NUMINAMATH_CALUDE_parabola_reflection_intersection_l2831_283184

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a standard parabola
structure StandardParabola where
  vertex : Point

-- Define the reflection with respect to the x-axis
def reflect (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Define the intersection of two standard parabolas
def intersect (p1 p2 : StandardParabola) : Point :=
  sorry

-- Theorem statement
theorem parabola_reflection_intersection 
  (V1 V2 V3 A1 A2 A3 : Point)
  (P1 : StandardParabola)
  (P2 : StandardParabola)
  (P3 : StandardParabola)
  (h1 : P1.vertex = V1)
  (h2 : P2.vertex = V2)
  (h3 : P3.vertex = V3)
  (h4 : intersect P2 P3 = A1)
  (h5 : intersect P3 P1 = A2)
  (h6 : intersect P1 P2 = A3)
  : ∃ (Q1 Q2 Q3 : StandardParabola),
    Q1.vertex = reflect A1 ∧
    Q2.vertex = reflect A2 ∧
    Q3.vertex = reflect A3 ∧
    intersect Q2 Q3 = reflect V1 ∧
    intersect Q3 Q1 = reflect V2 ∧
    intersect Q1 Q2 = reflect V3 :=
  sorry

end NUMINAMATH_CALUDE_parabola_reflection_intersection_l2831_283184


namespace NUMINAMATH_CALUDE_equally_spaced_posts_l2831_283153

/-- Given a sequence of 8 equally spaced posts, if the distance between the first and fifth post
    is 100 meters, then the distance between the first and last post is 175 meters. -/
theorem equally_spaced_posts (posts : Fin 8 → ℝ) 
  (equally_spaced : ∀ i j k : Fin 8, i.val + 1 = j.val → j.val + 1 = k.val → 
    posts k - posts j = posts j - posts i)
  (first_to_fifth : posts 4 - posts 0 = 100) :
  posts 7 - posts 0 = 175 :=
sorry

end NUMINAMATH_CALUDE_equally_spaced_posts_l2831_283153


namespace NUMINAMATH_CALUDE_fred_marbles_l2831_283116

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (red : ℕ) :
  total = 63 →
  dark_blue ≥ total / 3 →
  green = 4 →
  total = dark_blue + green + red →
  red = 38 := by
sorry

end NUMINAMATH_CALUDE_fred_marbles_l2831_283116


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2831_283171

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2831_283171


namespace NUMINAMATH_CALUDE_racing_car_A_time_l2831_283140

/-- The time taken by racing car A to complete the track -/
def time_A : ℕ := 7

/-- The time taken by racing car B to complete the track -/
def time_B : ℕ := 24

/-- The time after which both cars are side by side again -/
def side_by_side_time : ℕ := 168

/-- Theorem stating that the time taken by racing car A is correct -/
theorem racing_car_A_time :
  (time_A = 7) ∧ 
  (time_B = 24) ∧
  (side_by_side_time = 168) ∧
  (Nat.lcm time_A time_B = side_by_side_time) :=
sorry

end NUMINAMATH_CALUDE_racing_car_A_time_l2831_283140


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_seventeen_tenths_l2831_283132

theorem at_least_one_greater_than_seventeen_tenths
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a * b * c) :
  max a (max b c) > 17/10 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_seventeen_tenths_l2831_283132


namespace NUMINAMATH_CALUDE_chord_length_sum_l2831_283137

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Main theorem -/
theorem chord_length_sum (C1 C2 C3 : Circle) (m n p : ℕ) : 
  C1.radius = 4 →
  C2.radius = 10 →
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  are_collinear C1.center C2.center C3.center →
  (m.gcd p = 1) →
  (∀ q : ℕ, Prime q → n % (q^2) ≠ 0) →
  (∃ (chord_length : ℝ), chord_length = m * Real.sqrt n / p ∧ 
    chord_length^2 = 4 * (C3.radius^2 - ((C3.radius - C1.radius) * (C3.radius - C2.radius) / (C1.radius + C2.radius))^2)) →
  m + n + p = 405 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_sum_l2831_283137


namespace NUMINAMATH_CALUDE_circle_properties_l2831_283187

def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

theorem circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, circle_equation x y → x = 0 ∧ y = k) ∧
  (∀ x y : ℝ, circle_equation x y → (x - 0)^2 + (y - 2)^2 = 1) ∧
  circle_equation 1 2 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l2831_283187


namespace NUMINAMATH_CALUDE_cards_eaten_ratio_l2831_283113

theorem cards_eaten_ratio (initial_cards new_cards remaining_cards : ℕ) :
  initial_cards = 84 →
  new_cards = 8 →
  remaining_cards = 46 →
  (initial_cards + new_cards - remaining_cards) * 2 = initial_cards + new_cards :=
by
  sorry

end NUMINAMATH_CALUDE_cards_eaten_ratio_l2831_283113


namespace NUMINAMATH_CALUDE_stack_thickness_proof_l2831_283101

/-- Calculates the thickness of a stack of books in inches -/
def stack_thickness (num_books : ℕ) (pages_per_book : ℕ) (pages_per_inch : ℕ) : ℚ :=
  (num_books * pages_per_book : ℚ) / pages_per_inch

/-- Proves that the thickness of a stack of 6 books, each with an average of 160 pages, 
    is 12 inches, given that 80 pages is equivalent to 1 inch of thickness -/
theorem stack_thickness_proof :
  stack_thickness 6 160 80 = 12 := by
sorry

end NUMINAMATH_CALUDE_stack_thickness_proof_l2831_283101


namespace NUMINAMATH_CALUDE_train_length_calculation_l2831_283115

theorem train_length_calculation (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ) :
  platform_crossing_time = 55 →
  pole_crossing_time = 40 →
  platform_length = 159.375 →
  ∃ train_length : ℝ,
    train_length = 425 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2831_283115


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_three_l2831_283198

/-- Given vectors a, b, c in ℝ², if a + c is parallel to b + c, then the x-coordinate of c is 3. -/
theorem parallel_vectors_imply_x_equals_three :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![3, -1]
  let c : Fin 2 → ℝ := ![x, 4]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + c) = k • (b + c)) →
  x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_equals_three_l2831_283198


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2831_283114

theorem quadratic_equation_result : ∀ y : ℝ, 6 * y^2 + 7 = 2 * y + 12 → (12 * y - 4)^2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2831_283114


namespace NUMINAMATH_CALUDE_flour_calculation_l2831_283106

/-- The amount of flour Katie needs in pounds -/
def katie_flour : ℕ := 3

/-- The additional amount of flour Sheila needs compared to Katie in pounds -/
def sheila_extra_flour : ℕ := 2

/-- The total amount of flour needed by both Katie and Sheila -/
def total_flour : ℕ := katie_flour + (katie_flour + sheila_extra_flour)

theorem flour_calculation :
  total_flour = 8 :=
sorry

end NUMINAMATH_CALUDE_flour_calculation_l2831_283106


namespace NUMINAMATH_CALUDE_ravi_coins_l2831_283141

def coin_problem (nickels quarters dimes : ℕ) (nickel_value quarter_value dime_value : ℚ) : Prop :=
  nickels = 6 ∧
  quarters = nickels + 2 ∧
  dimes = quarters + 4 ∧
  nickel_value = 5/100 ∧
  quarter_value = 25/100 ∧
  dime_value = 10/100 ∧
  nickels * nickel_value + quarters * quarter_value + dimes * dime_value = 7/2

theorem ravi_coins :
  ∃ (nickels quarters dimes : ℕ) (nickel_value quarter_value dime_value : ℚ),
    coin_problem nickels quarters dimes nickel_value quarter_value dime_value :=
by
  sorry

#check ravi_coins

end NUMINAMATH_CALUDE_ravi_coins_l2831_283141


namespace NUMINAMATH_CALUDE_expression_factorization_l2831_283150

theorem expression_factorization (b : ℝ) : 
  (10 * b^4 - 27 * b^3 + 18 * b^2) - (-6 * b^4 + 4 * b^3 - 3 * b^2) = 
  b^2 * (16 * b - 7) * (b - 3) := by
sorry

end NUMINAMATH_CALUDE_expression_factorization_l2831_283150


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l2831_283104

theorem division_multiplication_problem : 
  let number : ℚ := 4
  let divisor : ℚ := 6
  let multiplier : ℚ := 12
  let result : ℚ := 8
  (number / divisor) * multiplier = result := by sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l2831_283104


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l2831_283179

theorem square_reciprocal_sum_implies_fourth_power_reciprocal_sum
  (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l2831_283179


namespace NUMINAMATH_CALUDE_fifth_observation_value_l2831_283147

theorem fifth_observation_value (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) :
  (x1 + x2 + x3 + x4 + x5) / 5 = 10 →
  (x5 + x6 + x7 + x8 + x9) / 5 = 8 →
  (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9 = 8 →
  x5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fifth_observation_value_l2831_283147


namespace NUMINAMATH_CALUDE_given_circles_are_externally_tangent_l2831_283112

/-- Two circles in a 2D plane --/
structure TwoCircles where
  c1 : (ℝ × ℝ) → Prop
  c2 : (ℝ × ℝ) → Prop

/-- Definition of the given circles --/
def givenCircles : TwoCircles where
  c1 := fun (x, y) ↦ x^2 + y^2 - 4*x - 6*y + 9 = 0
  c2 := fun (x, y) ↦ x^2 + y^2 + 12*x + 6*y - 19 = 0

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii --/
def areExternallyTangent (circles : TwoCircles) : Prop :=
  ∃ (x1 y1 x2 y2 r1 r2 : ℝ),
    (∀ (x y : ℝ), circles.c1 (x, y) ↔ (x - x1)^2 + (y - y1)^2 = r1^2) ∧
    (∀ (x y : ℝ), circles.c2 (x, y) ↔ (x - x2)^2 + (y - y2)^2 = r2^2) ∧
    (x2 - x1)^2 + (y2 - y1)^2 = (r1 + r2)^2

/-- Theorem stating that the given circles are externally tangent --/
theorem given_circles_are_externally_tangent :
  areExternallyTangent givenCircles := by sorry

end NUMINAMATH_CALUDE_given_circles_are_externally_tangent_l2831_283112


namespace NUMINAMATH_CALUDE_least_homeowners_l2831_283145

theorem least_homeowners (total_members : ℕ) (men_percentage : ℚ) (women_percentage : ℚ)
  (h_total : total_members = 150)
  (h_men_percentage : men_percentage = 1/10)
  (h_women_percentage : women_percentage = 1/5) :
  ∃ (men women : ℕ),
    men + women = total_members ∧
    ∃ (men_homeowners women_homeowners : ℕ),
      men_homeowners = ⌈men_percentage * men⌉ ∧
      women_homeowners = ⌈women_percentage * women⌉ ∧
      men_homeowners + women_homeowners = 16 ∧
      ∀ (other_men other_women : ℕ),
        other_men + other_women = total_members →
        ∃ (other_men_homeowners other_women_homeowners : ℕ),
          other_men_homeowners = ⌈men_percentage * other_men⌉ ∧
          other_women_homeowners = ⌈women_percentage * other_women⌉ →
          other_men_homeowners + other_women_homeowners ≥ 16 :=
by
  sorry

end NUMINAMATH_CALUDE_least_homeowners_l2831_283145


namespace NUMINAMATH_CALUDE_no_integer_solution_l2831_283120

theorem no_integer_solution :
  ¬ ∃ (a b x y : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧
    a * x - b * y = 16 ∧
    a * y + b * x = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2831_283120


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2831_283125

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 3) :
  2 * (a^2 - a*b) - 3 * ((2/3) * a^2 - a*b - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2831_283125


namespace NUMINAMATH_CALUDE_product_digit_sum_l2831_283164

def a : ℕ := 70707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def b : ℕ := 60606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 10000) % 10) = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2831_283164


namespace NUMINAMATH_CALUDE_washing_time_proof_l2831_283144

def shirts : ℕ := 18
def pants : ℕ := 12
def sweaters : ℕ := 17
def jeans : ℕ := 13
def max_items_per_cycle : ℕ := 15
def minutes_per_cycle : ℕ := 45

def total_items : ℕ := shirts + pants + sweaters + jeans

def cycles_needed : ℕ := (total_items + max_items_per_cycle - 1) / max_items_per_cycle

def total_minutes : ℕ := cycles_needed * minutes_per_cycle

theorem washing_time_proof : 
  total_minutes / 60 = 3 := by sorry

end NUMINAMATH_CALUDE_washing_time_proof_l2831_283144


namespace NUMINAMATH_CALUDE_walk_time_proof_l2831_283173

/-- Ajay's walking speed in km/hour -/
def walking_speed : ℝ := 6

/-- The time taken to walk a certain distance in hours -/
def time_taken : ℝ := 11.666666666666666

/-- Theorem stating that the time taken to walk the distance is 11.666666666666666 hours -/
theorem walk_time_proof : time_taken = 11.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_walk_time_proof_l2831_283173


namespace NUMINAMATH_CALUDE_male_students_count_l2831_283155

def scienceGroup (x : ℕ) : Prop :=
  ∃ (total : ℕ), total = x + 2 ∧ 
  (Nat.choose x 2) * (Nat.choose 2 1) = 20

theorem male_students_count :
  ∀ x : ℕ, scienceGroup x → x = 5 := by sorry

end NUMINAMATH_CALUDE_male_students_count_l2831_283155


namespace NUMINAMATH_CALUDE_angle_FDB_is_40_l2831_283154

-- Define the points
variable (A B C D E F : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define isosceles triangle
def isosceles (P Q R : Point) : Prop :=
  angle P Q R = angle P R Q

-- State the theorem
theorem angle_FDB_is_40 :
  isosceles A D E →
  isosceles A B C →
  angle D F C = 150 →
  angle F D B = 40 := by sorry

end NUMINAMATH_CALUDE_angle_FDB_is_40_l2831_283154


namespace NUMINAMATH_CALUDE_f_bijective_iff_power_of_two_l2831_283139

/-- The set of all possible lamp configurations for n lamps -/
def Ψ (n : ℕ) := Fin (2^n)

/-- The cool procedure function -/
def f (n : ℕ) : Ψ n → Ψ n := sorry

/-- Theorem stating that f is bijective if and only if n is a power of 2 -/
theorem f_bijective_iff_power_of_two (n : ℕ) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end NUMINAMATH_CALUDE_f_bijective_iff_power_of_two_l2831_283139


namespace NUMINAMATH_CALUDE_fathers_age_problem_l2831_283161

/-- The father's age problem -/
theorem fathers_age_problem (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 12 = (father_age + 12) / 2 →
  father_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_problem_l2831_283161


namespace NUMINAMATH_CALUDE_more_probability_papa_probability_l2831_283160

/- Define the words as lists of characters -/
def remontWord : List Char := ['р', 'е', 'м', 'о', 'н', 'т']
def moreWord : List Char := ['м', 'о', 'р', 'е']
def papakhaWord : List Char := ['п', 'а', 'п', 'а', 'х', 'а']
def papaWord : List Char := ['п', 'а', 'п', 'а']

/- Define the number of letters to draw -/
def numDrawn : Nat := 4

/- Define the probability calculation function -/
def calculateProbability (sourceWord : List Char) (targetWord : List Char) : Rat :=
  sorry

/- Theorem statements -/
theorem more_probability : calculateProbability remontWord moreWord = 1 / 360 := by sorry

theorem papa_probability : calculateProbability papakhaWord papaWord = 1 / 30 := by sorry

end NUMINAMATH_CALUDE_more_probability_papa_probability_l2831_283160


namespace NUMINAMATH_CALUDE_damaged_potatoes_calculation_l2831_283159

/-- Calculates the amount of damaged potatoes during transport -/
def damaged_potatoes (initial_amount : ℕ) (bag_size : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) : ℕ :=
  initial_amount - (total_sales / price_per_bag * bag_size)

/-- Theorem stating the amount of damaged potatoes -/
theorem damaged_potatoes_calculation :
  damaged_potatoes 6500 50 72 9144 = 150 := by
  sorry

#eval damaged_potatoes 6500 50 72 9144

end NUMINAMATH_CALUDE_damaged_potatoes_calculation_l2831_283159


namespace NUMINAMATH_CALUDE_dosage_for_package_l2831_283165

-- Define the dosage function
def dosage (x : ℝ) : ℝ := 10 * x + 10

-- Define the weight range
def valid_weight (x : ℝ) : Prop := 5 ≤ x ∧ x ≤ 50

-- Define the safe dosage range for a 300 mg package
def safe_dosage (y : ℝ) : Prop := 250 ≤ y ∧ y ≤ 300

-- Theorem statement
theorem dosage_for_package (x : ℝ) (h1 : 24 ≤ x) (h2 : x ≤ 29) (h3 : valid_weight x) :
  safe_dosage (dosage x) :=
sorry

end NUMINAMATH_CALUDE_dosage_for_package_l2831_283165


namespace NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l2831_283186

theorem subset_of_intersection_eq_union {A B C : Set α} 
  (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) 
  (h : A ∩ B = B ∪ C) : C ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_subset_of_intersection_eq_union_l2831_283186


namespace NUMINAMATH_CALUDE_remaining_segment_length_l2831_283177

/-- Represents an equilateral triangle with segments drawn from vertices to opposite sides. -/
structure SegmentedEquilateralTriangle where
  /-- Length of the first segment on one side -/
  a : ℝ
  /-- Length of the second segment on one side -/
  b : ℝ
  /-- Length of the third segment on one side -/
  c : ℝ
  /-- Length of the shortest segment on another side -/
  d : ℝ
  /-- Length of the segment adjacent to the shortest segment -/
  e : ℝ
  /-- Assumption that the triangle is equilateral and segments form a complete side -/
  side_length : a + b + c = d + e + (a + b + c - (d + e))

/-- Theorem stating that the remaining segment length is 4 cm given the conditions -/
theorem remaining_segment_length
  (triangle : SegmentedEquilateralTriangle)
  (h1 : triangle.a = 5)
  (h2 : triangle.b = 10)
  (h3 : triangle.c = 2)
  (h4 : triangle.d = 1.5)
  (h5 : triangle.e = 11.5) :
  triangle.a + triangle.b + triangle.c - (triangle.d + triangle.e) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_segment_length_l2831_283177


namespace NUMINAMATH_CALUDE_water_spilled_is_eight_quarts_l2831_283100

/-- Represents the water supply problem from the shipwreck scenario -/
structure WaterSupply where
  initial_people : ℕ
  initial_days : ℕ
  spill_day : ℕ
  quart_per_person_per_day : ℕ

/-- The amount of water spilled in the shipwreck scenario -/
def water_spilled (ws : WaterSupply) : ℕ :=
  ws.initial_people + 7

/-- Theorem stating that the amount of water spilled is 8 quarts -/
theorem water_spilled_is_eight_quarts (ws : WaterSupply) 
  (h1 : ws.initial_days = 13)
  (h2 : ws.quart_per_person_per_day = 1)
  (h3 : ws.spill_day = 5)
  (h4 : ws.initial_people > 0)
  : water_spilled ws = 8 := by
  sorry

#check water_spilled_is_eight_quarts

end NUMINAMATH_CALUDE_water_spilled_is_eight_quarts_l2831_283100


namespace NUMINAMATH_CALUDE_k_value_proof_l2831_283188

theorem k_value_proof (k : ℤ) 
  (h1 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 1000000)
  (h2 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) < 10000000) : 
  k = 11 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l2831_283188


namespace NUMINAMATH_CALUDE_work_completion_time_l2831_283130

/-- Represents the time taken to complete a work -/
structure WorkTime where
  days : ℝ
  hours_per_day : ℝ := 24
  total_hours : ℝ := days * hours_per_day

/-- Represents the rate of work -/
def WorkRate := ℝ

/-- The problem setup -/
structure WorkProblem where
  total_work : ℝ
  a_alone_time : WorkTime
  ab_initial_time : WorkTime
  a_final_time : WorkTime
  ab_together_time : WorkTime

/-- The main theorem to prove -/
theorem work_completion_time 
  (w : WorkProblem)
  (h1 : w.a_alone_time.days = 20)
  (h2 : w.ab_initial_time.days = 10)
  (h3 : w.a_final_time.days = 15)
  (h4 : w.total_work = (w.ab_initial_time.days / w.ab_together_time.days + 
                        w.a_final_time.days / w.a_alone_time.days) * w.total_work) :
  w.ab_together_time.days = 40 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2831_283130


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l2831_283124

theorem cube_sum_inequality (a b c : ℤ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l2831_283124


namespace NUMINAMATH_CALUDE_fish_rice_equivalence_l2831_283126

/-- Represents the value of one fish in terms of bags of rice -/
def fish_to_rice_ratio : ℚ := 21 / 20

theorem fish_rice_equivalence (fish bread rice : ℚ) 
  (h1 : 4 * fish = 3 * bread) 
  (h2 : 5 * bread = 7 * rice) : 
  fish = fish_to_rice_ratio * rice := by
  sorry

#check fish_rice_equivalence

end NUMINAMATH_CALUDE_fish_rice_equivalence_l2831_283126


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_subset_condition_l2831_283182

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1 - 2*a ≤ x ∧ x ≤ a - 2}

-- Statement for part (1)
theorem sufficient_not_necessary (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) ↔ a ≥ 7 :=
sorry

-- Statement for part (2)
theorem subset_condition (a : ℝ) :
  B a ⊆ A ↔ a < 1/3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_subset_condition_l2831_283182


namespace NUMINAMATH_CALUDE_parabola_directrix_l2831_283172

theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = 1 ↔ x^2 = 1 / (4 * k))) → 
  a = -1/4 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2831_283172


namespace NUMINAMATH_CALUDE_fourth_power_of_one_minus_i_l2831_283195

theorem fourth_power_of_one_minus_i :
  (1 - Complex.I) ^ 4 = -4 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_one_minus_i_l2831_283195


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2831_283162

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n = (360 : ℝ) / exterior_angle →
  interior_angle = 180 - exterior_angle →
  n = 20 ∧ interior_angle = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2831_283162


namespace NUMINAMATH_CALUDE_multiplicative_inverse_of_110_mod_667_l2831_283167

-- Define the triangle
def leg1 : ℕ := 65
def leg2 : ℕ := 156
def hypotenuse : ℕ := 169

-- Define the relation C = A + B
def relation (A B C : ℕ) : Prop := C = A + B

-- Define the modulus
def modulus : ℕ := 667

-- Define the number we're finding the inverse for
def num : ℕ := 110

-- Theorem statement
theorem multiplicative_inverse_of_110_mod_667 :
  (∃ (A B : ℕ), relation A B hypotenuse ∧ leg1^2 + leg2^2 = hypotenuse^2) →
  ∃ (n : ℕ), n < modulus ∧ (num * n) % modulus = 1 ∧ n = 608 :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_of_110_mod_667_l2831_283167


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2831_283129

theorem quadratic_inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2831_283129


namespace NUMINAMATH_CALUDE_battleship_max_ships_l2831_283181

/-- Represents a game board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a ship --/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Calculates the maximum number of ships that can be placed on a board --/
def maxShips (board : Board) (ship : Ship) : Nat :=
  (board.rows * board.cols) / (ship.length * ship.width)

theorem battleship_max_ships :
  let board : Board := ⟨10, 10⟩
  let ship : Ship := ⟨4, 1⟩
  maxShips board ship = 25 := by
  sorry

#eval maxShips ⟨10, 10⟩ ⟨4, 1⟩

end NUMINAMATH_CALUDE_battleship_max_ships_l2831_283181


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2831_283169

def calculate_total_cost (tv_price sound_price warranty_price install_price : ℝ)
  (tv_discount1 tv_discount2 sound_discount warranty_discount : ℝ)
  (tv_sound_tax warranty_install_tax : ℝ) : ℝ :=
  let tv_after_discounts := tv_price * (1 - tv_discount1) * (1 - tv_discount2)
  let sound_after_discount := sound_price * (1 - sound_discount)
  let warranty_after_discount := warranty_price * (1 - warranty_discount)
  let tv_with_tax := tv_after_discounts * (1 + tv_sound_tax)
  let sound_with_tax := sound_after_discount * (1 + tv_sound_tax)
  let warranty_with_tax := warranty_after_discount * (1 + warranty_install_tax)
  let install_with_tax := install_price * (1 + warranty_install_tax)
  tv_with_tax + sound_with_tax + warranty_with_tax + install_with_tax

theorem total_cost_calculation :
  calculate_total_cost 600 400 100 150 0.1 0.15 0.2 0.3 0.08 0.05 = 1072.32 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2831_283169


namespace NUMINAMATH_CALUDE_remainder_problem_l2831_283194

theorem remainder_problem (n a b c d : ℕ) : 
  n = 102 * a + b ∧ 
  0 ≤ b ∧ b < 102 ∧
  n = 103 * c + d ∧ 
  0 ≤ d ∧ d < 103 ∧
  a + d = 20 
  → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2831_283194


namespace NUMINAMATH_CALUDE_even_coverings_for_odd_height_l2831_283134

/-- Represents a covering of the lateral surface of a rectangular parallelepiped -/
def Covering (a b c : ℕ) := Unit

/-- Count the number of valid coverings for a rectangular parallelepiped -/
def countCoverings (a b c : ℕ) : ℕ := sorry

/-- Theorem: The number of valid coverings is even when the height is odd -/
theorem even_coverings_for_odd_height (a b c : ℕ) (h : c % 2 = 1) :
  ∃ k : ℕ, countCoverings a b c = 2 * k := by sorry

end NUMINAMATH_CALUDE_even_coverings_for_odd_height_l2831_283134


namespace NUMINAMATH_CALUDE_sally_eggs_l2831_283119

-- Define what a dozen is
def dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_bought : ℕ := 4

-- Theorem: Sally bought 48 eggs
theorem sally_eggs : dozens_bought * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_sally_eggs_l2831_283119


namespace NUMINAMATH_CALUDE_problem_statement_l2831_283196

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : x^2 - y^2 - z^2 = 2*a*y*z)
  (h2 : -x^2 + y^2 - z^2 = 2*b*z*x)
  (h3 : -x^2 - y^2 + z^2 = 2*c*x*y)
  (h4 : x*y*z ≠ 0) :
  a^2 + b^2 + c^2 - 2*a*b*c = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2831_283196


namespace NUMINAMATH_CALUDE_scientific_notation_of_8200000_l2831_283157

theorem scientific_notation_of_8200000 :
  (8200000 : ℝ) = 8.2 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_8200000_l2831_283157


namespace NUMINAMATH_CALUDE_max_trig_product_l2831_283176

theorem max_trig_product (x y z : ℝ) : 
  (Real.sin x + Real.sin (2*y) + Real.sin (3*z)) * 
  (Real.cos x + Real.cos (2*y) + Real.cos (3*z)) ≤ 4.5 := by
sorry

end NUMINAMATH_CALUDE_max_trig_product_l2831_283176


namespace NUMINAMATH_CALUDE_odd_increasing_function_property_l2831_283135

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing if x < y implies f(x) < f(y) -/
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem odd_increasing_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_mono : IsMonoIncreasing f) :
    (∀ a b : ℝ, f a + f (b - 1) = 0) → 
    (∀ a b : ℝ, a + b = 1) :=
  sorry

end NUMINAMATH_CALUDE_odd_increasing_function_property_l2831_283135


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l2831_283170

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

/-- The point on the curve -/
def point : ℝ × ℝ := (-1, -3)

/-- The proposed tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 3*x + y + 6 = 0

theorem tangent_line_is_correct : 
  tangent_line point.1 point.2 ∧ 
  (∀ x : ℝ, tangent_line x (f x) → x = point.1) ∧
  f' point.1 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l2831_283170


namespace NUMINAMATH_CALUDE_red_face_probability_l2831_283163

/-- The volume of the original cube in cubic centimeters -/
def original_volume : ℝ := 27

/-- The number of small cubes the original cube is sawn into -/
def num_small_cubes : ℕ := 27

/-- The volume of each small cube in cubic centimeters -/
def small_cube_volume : ℝ := 1

/-- The number of small cubes with at least one red face -/
def num_red_cubes : ℕ := 26

/-- The probability of selecting a cube with at least one red face -/
def prob_red_face : ℚ := 26 / 27

theorem red_face_probability :
  original_volume = num_small_cubes * small_cube_volume →
  (num_red_cubes : ℚ) / num_small_cubes = prob_red_face := by
  sorry

end NUMINAMATH_CALUDE_red_face_probability_l2831_283163


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2831_283146

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l2831_283146


namespace NUMINAMATH_CALUDE_sam_has_sixteen_dimes_l2831_283148

/-- The number of dimes Sam has after receiving some from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Sam has 16 dimes after receiving some from his dad -/
theorem sam_has_sixteen_dimes : total_dimes 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_sixteen_dimes_l2831_283148


namespace NUMINAMATH_CALUDE_ted_speed_l2831_283191

theorem ted_speed (frank_speed : ℝ) (h1 : frank_speed > 0) : 
  let ted_speed := (2 / 3) * frank_speed
  2 * frank_speed = 2 * ted_speed + 8 →
  ted_speed = 8 := by
sorry

end NUMINAMATH_CALUDE_ted_speed_l2831_283191


namespace NUMINAMATH_CALUDE_student_sport_signup_ways_l2831_283152

theorem student_sport_signup_ways :
  let num_students : ℕ := 4
  let num_sports : ℕ := 3
  num_sports ^ num_students = 81 :=
by sorry

end NUMINAMATH_CALUDE_student_sport_signup_ways_l2831_283152


namespace NUMINAMATH_CALUDE_circle_equation_l2831_283110

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola : Set Point :=
  {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point :=
  ⟨1, 0⟩

/-- The line passing through the focus with slope angle 30° -/
def Line : Set Point :=
  {p : Point | p.y = (Real.sqrt 3 / 3) * (p.x - 1)}

/-- Intersection points of the parabola and the line -/
def intersectionPoints : Set Point :=
  Parabola ∩ Line

/-- The circle with AB as diameter -/
def Circle (A B : Point) : Set Point :=
  {p : Point | (p.x - (A.x + B.x) / 2)^2 + (p.y - (A.y + B.y) / 2)^2 = ((A.x - B.x)^2 + (A.y - B.y)^2) / 4}

theorem circle_equation (A B : Point) 
  (hA : A ∈ intersectionPoints) (hB : B ∈ intersectionPoints) (hAB : A ≠ B) :
  Circle A B = {p : Point | (p.x - 7)^2 + (p.y - 2 * Real.sqrt 3)^2 = 64} :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2831_283110


namespace NUMINAMATH_CALUDE_managers_in_sample_l2831_283192

structure StaffUnit where
  total : ℕ
  managers : ℕ
  sample_size : ℕ

def stratified_sample_size (unit : StaffUnit) (stratum_size : ℕ) : ℕ :=
  (stratum_size * unit.sample_size) / unit.total

theorem managers_in_sample (unit : StaffUnit) 
    (h1 : unit.total = 160)
    (h2 : unit.managers = 32)
    (h3 : unit.sample_size = 20) :
  stratified_sample_size unit unit.managers = 4 := by
  sorry

end NUMINAMATH_CALUDE_managers_in_sample_l2831_283192


namespace NUMINAMATH_CALUDE_m_range_theorem_l2831_283123

/-- Proposition p: The solution set of the inequality |x-1| > m-1 is ℝ -/
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1

/-- Proposition q: f(x) = -(5-2m)x is a decreasing function -/
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5 - 2*m)*x > -(5 - 2*m)*y

/-- Either p or q is true -/
def either_p_or_q (m : ℝ) : Prop := p m ∨ q m

/-- Both p and q are false propositions -/
def both_p_and_q_false (m : ℝ) : Prop := ¬(p m) ∧ ¬(q m)

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 2

theorem m_range_theorem :
  ∀ m : ℝ, (either_p_or_q m ∧ ¬(both_p_and_q_false m)) ↔ m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2831_283123


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2831_283121

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 4 / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2831_283121


namespace NUMINAMATH_CALUDE_dance_attendance_l2831_283193

/-- The number of men at the dance -/
def num_men : ℕ := 15

/-- The number of women each man dances with -/
def dances_per_man : ℕ := 4

/-- The number of men each woman dances with -/
def dances_per_woman : ℕ := 3

/-- The number of women at the dance -/
def num_women : ℕ := num_men * dances_per_man / dances_per_woman

theorem dance_attendance : num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_dance_attendance_l2831_283193


namespace NUMINAMATH_CALUDE_passing_train_speed_is_50_l2831_283136

/-- The speed of the passing train in km/h -/
def passing_train_speed : ℝ := 50

/-- The speed of the passenger's train in km/h -/
def passenger_train_speed : ℝ := 40

/-- The time taken for the passing train to pass completely in seconds -/
def passing_time : ℝ := 3

/-- The length of the passing train in meters -/
def passing_train_length : ℝ := 75

/-- Theorem stating that the speed of the passing train is 50 km/h -/
theorem passing_train_speed_is_50 :
  passing_train_speed = 50 :=
sorry

end NUMINAMATH_CALUDE_passing_train_speed_is_50_l2831_283136


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2831_283128

/-- An ellipse with given foci and passing through specific points has the standard equation x²/8 + y²/4 = 1 -/
theorem ellipse_standard_equation (f1 f2 p1 p2 p3 : ℝ × ℝ) : 
  f1 = (0, -2) →
  f2 = (0, 2) →
  p1 = (-3/2, 5/2) →
  p2 = (2, -Real.sqrt 2) →
  p3 = (-1, Real.sqrt 14 / 2) →
  ∃ (ellipse : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), ellipse (x, y) ↔ x^2/8 + y^2/4 = 1) ∧
    (ellipse f1 ∧ ellipse f2 ∧ ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2831_283128


namespace NUMINAMATH_CALUDE_total_subscription_is_50000_l2831_283142

/-- Represents the subscription amounts and profit distribution for a business venture -/
structure BusinessSubscription where
  /-- Subscription amount for C -/
  c : ℕ
  /-- Total profit -/
  totalProfit : ℕ
  /-- A's share of the profit -/
  aProfit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def totalSubscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c + 14000

/-- Theorem stating that the total subscription amount is 50000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.totalProfit = 70000)
  (h2 : bs.aProfit = 29400)
  (h3 : bs.aProfit * (3 * bs.c + 14000) = bs.totalProfit * (bs.c + 9000)) :
  totalSubscription bs = 50000 := by
  sorry

end NUMINAMATH_CALUDE_total_subscription_is_50000_l2831_283142


namespace NUMINAMATH_CALUDE_shooting_competition_probability_l2831_283151

theorem shooting_competition_probability (p : ℝ) (n : ℕ) (k : ℕ) : 
  p = 0.4 → n = 3 → k = 2 →
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => Nat.choose n (n - i) * p^(n - i) * (1 - p)^i)) = 0.352 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_probability_l2831_283151


namespace NUMINAMATH_CALUDE_correct_conclusions_l2831_283103

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the parallel relation
def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- Define the dot product
variable (dot : V → V → ℝ)

-- Statement of the theorem
theorem correct_conclusions :
  (∀ (a b c : V), a = b → b = c → a = c) ∧ 
  (∃ (a b c : V), parallel a b → parallel b c → ¬ parallel a c) ∧
  (∃ (a b : V), |dot a b| ≠ |dot a (1 • b)|) ∧
  (∀ (a b c : V), b = c → dot a b = dot a c) :=
sorry

end NUMINAMATH_CALUDE_correct_conclusions_l2831_283103


namespace NUMINAMATH_CALUDE_sun_moon_volume_ratio_l2831_283158

/-- The ratio of the Sun-Earth distance to the Moon-Earth distance -/
def distance_ratio : ℝ := 387

/-- The ratio of the Sun's volume to the Moon's volume -/
def volume_ratio : ℝ := distance_ratio ^ 3

theorem sun_moon_volume_ratio : 
  volume_ratio = distance_ratio ^ 3 := by sorry

end NUMINAMATH_CALUDE_sun_moon_volume_ratio_l2831_283158


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_specific_sum_of_squares_2009_l2831_283199

theorem sum_of_squares_representation (n : ℕ) :
  ∃ (a b c d : ℕ), 2 * n^2 + 2 * (n + 1)^2 = a^2 + b^2 ∧ 2 * n^2 + 2 * (n + 1)^2 = c^2 + d^2 ∧ (a ≠ c ∨ b ≠ d) := by
  sorry

-- Specific case for n = 2009
theorem specific_sum_of_squares_2009 :
  ∃ (a b c d : ℕ), 2 * 2009^2 + 2 * 2010^2 = a^2 + b^2 ∧ 2 * 2009^2 + 2 * 2010^2 = c^2 + d^2 ∧ (a ≠ c ∨ b ≠ d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_specific_sum_of_squares_2009_l2831_283199


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_half_5060_l2831_283180

theorem fraction_of_three_fourths_half_5060 : 
  let total := (3/4 : ℚ) * (1/2 : ℚ) * 5060
  759.0000000000001 / total = 0.4 := by sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_half_5060_l2831_283180


namespace NUMINAMATH_CALUDE_book_pages_count_l2831_283143

theorem book_pages_count : 
  let days : ℕ := 10
  let first_four_days_avg : ℕ := 20
  let first_four_days_count : ℕ := 4
  let break_day_count : ℕ := 1
  let next_four_days_avg : ℕ := 30
  let next_four_days_count : ℕ := 4
  let last_day_pages : ℕ := 15
  (first_four_days_avg * first_four_days_count) + 
  (next_four_days_avg * next_four_days_count) + 
  last_day_pages = 215 := by
sorry

end NUMINAMATH_CALUDE_book_pages_count_l2831_283143


namespace NUMINAMATH_CALUDE_sample_definition_l2831_283107

/-- Represents a student's math score -/
def MathScore : Type := ℝ

/-- Represents a sample of math scores -/
def Sample : Type := List MathScore

structure SurveyData where
  totalStudents : ℕ
  sampleSize : ℕ
  scores : Sample
  h_sampleSize : sampleSize ≤ totalStudents

/-- Definition of a valid sample for the survey -/
def isValidSample (data : SurveyData) : Prop :=
  data.scores.length = data.sampleSize

theorem sample_definition (data : SurveyData) 
  (h_total : data.totalStudents = 960)
  (h_sample : data.sampleSize = 120)
  (h_valid : isValidSample data) :
  ∃ (sample : Sample), sample = data.scores ∧ sample.length = 120 :=
sorry

end NUMINAMATH_CALUDE_sample_definition_l2831_283107


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l2831_283149

def satisfies_inequality (x : ℝ) : Prop :=
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4

def solution_set : Set ℝ :=
  {x | x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x}

theorem inequality_solution_equivalence :
  ∀ x : ℝ, satisfies_inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l2831_283149


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2831_283138

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 2 → b = 4 → c = 4 →  -- Two sides are equal (isosceles) and one side is 2
  a + b + c = 10 :=         -- The perimeter is 10
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2831_283138


namespace NUMINAMATH_CALUDE_exponential_inequality_supremum_l2831_283178

theorem exponential_inequality_supremum : 
  (∃ a : ℝ, (∀ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) → m < a) ∧ 
   (∀ ε > 0, ∃ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) ∧ m > a - ε)) → 
  (∃ a : ℝ, a = 1 ∧ 
   (∀ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) → m < a) ∧ 
   (∀ ε > 0, ∃ m : ℝ, (¬∃ x : ℝ, Real.exp (|x - 1|) - m ≤ 0) ∧ m > a - ε)) :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_supremum_l2831_283178


namespace NUMINAMATH_CALUDE_no_integer_solution_l2831_283102

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 5 ≠ y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2831_283102


namespace NUMINAMATH_CALUDE_alice_walk_distance_l2831_283197

theorem alice_walk_distance (grass_miles : ℝ) : 
  (∀ (day : Fin 5), grass_miles > 0) →  -- Alice walks a positive distance through grass each weekday
  (∀ (day : Fin 5), 12 > 0) →  -- Alice walks 12 miles through forest each weekday
  (5 * grass_miles + 5 * 12 = 110) →  -- Total weekly distance is 110 miles
  grass_miles = 10 := by
  sorry

end NUMINAMATH_CALUDE_alice_walk_distance_l2831_283197


namespace NUMINAMATH_CALUDE_min_cos_C_in_triangle_l2831_283168

theorem min_cos_C_in_triangle (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A + 2 * Real.sin B = 3 * Real.sin C) : 
  (2 * Real.sqrt 10 - 2) / 9 ≤ Real.cos C :=
sorry

end NUMINAMATH_CALUDE_min_cos_C_in_triangle_l2831_283168


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l2831_283185

theorem adult_tickets_sold (adult_price child_price total_tickets total_receipts : ℕ) 
  (h1 : adult_price = 12)
  (h2 : child_price = 4)
  (h3 : total_tickets = 130)
  (h4 : total_receipts = 840) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_receipts ∧
    adult_tickets = 40 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l2831_283185


namespace NUMINAMATH_CALUDE_germs_left_is_thirty_percent_l2831_283117

/-- The percentage of germs killed by spray A -/
def spray_a_kill_rate : ℝ := 50

/-- The percentage of germs killed by spray B -/
def spray_b_kill_rate : ℝ := 25

/-- The percentage of germs killed by both sprays -/
def overlap_kill_rate : ℝ := 5

/-- The percentage of germs left after using both sprays -/
def germs_left : ℝ := 100 - (spray_a_kill_rate + spray_b_kill_rate - overlap_kill_rate)

theorem germs_left_is_thirty_percent :
  germs_left = 30 := by sorry

end NUMINAMATH_CALUDE_germs_left_is_thirty_percent_l2831_283117


namespace NUMINAMATH_CALUDE_road_area_in_square_park_l2831_283190

/-- 
Given a square park with a road inside, this theorem proves that
if the road is 3 meters wide and the perimeter along its outer edge is 600 meters,
then the area occupied by the road is 1836 square meters.
-/
theorem road_area_in_square_park (park_side : ℝ) (road_width : ℝ) (outer_perimeter : ℝ) 
  (h1 : road_width = 3)
  (h2 : outer_perimeter = 600)
  (h3 : 4 * (park_side - 2 * road_width) = outer_perimeter) :
  park_side^2 - (park_side - 2 * road_width)^2 = 1836 := by
  sorry

end NUMINAMATH_CALUDE_road_area_in_square_park_l2831_283190


namespace NUMINAMATH_CALUDE_lcm_gcd_220_126_l2831_283183

theorem lcm_gcd_220_126 :
  (Nat.lcm 220 126 = 13860) ∧ (Nat.gcd 220 126 = 2) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_220_126_l2831_283183


namespace NUMINAMATH_CALUDE_total_cost_one_large_three_small_l2831_283133

/-- The cost of a large puzzle, in dollars -/
def large_puzzle_cost : ℕ := 15

/-- The cost of a small puzzle and a large puzzle together, in dollars -/
def combined_cost : ℕ := 23

/-- The cost of a small puzzle, in dollars -/
def small_puzzle_cost : ℕ := combined_cost - large_puzzle_cost

theorem total_cost_one_large_three_small :
  large_puzzle_cost + 3 * small_puzzle_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_one_large_three_small_l2831_283133


namespace NUMINAMATH_CALUDE_vanessa_score_l2831_283122

/-- Vanessa's basketball game score calculation -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (other_avg : ℕ) : 
  total_score = 65 → other_players = 7 → other_avg = 5 → 
  total_score - (other_players * other_avg) = 30 := by
sorry

end NUMINAMATH_CALUDE_vanessa_score_l2831_283122


namespace NUMINAMATH_CALUDE_fixed_point_theorem_a_value_theorem_minimum_point_theorem_l2831_283118

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

theorem fixed_point_theorem (a : ℝ) :
  f a 0 = 1 := by sorry

theorem a_value_theorem (a : ℝ) :
  (∀ x, f_deriv a x ≥ -a * x - 1) → a = 1 := by sorry

theorem minimum_point_theorem :
  ∃ x₀, (∀ x, f 1 x ≥ f 1 x₀) ∧ -2 < f 1 x₀ ∧ f 1 x₀ < -1/4 := by sorry

end

end NUMINAMATH_CALUDE_fixed_point_theorem_a_value_theorem_minimum_point_theorem_l2831_283118


namespace NUMINAMATH_CALUDE_triangle_right_angle_l2831_283156

theorem triangle_right_angle (a b : ℝ) (A B : Real) (h : a + b = a / Real.tan A + b / Real.tan B) :
  A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l2831_283156
