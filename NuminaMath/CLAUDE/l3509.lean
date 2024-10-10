import Mathlib

namespace property_set_characterization_l3509_350901

/-- The property that a^(n+1) ≡ a (mod n) holds for all integers a -/
def has_property (n : ℕ) : Prop :=
  ∀ a : ℤ, (a^(n+1) : ℤ) ≡ a [ZMOD n]

/-- The set of integers satisfying the property -/
def property_set : Set ℕ := {n | has_property n}

/-- Theorem stating that the set of integers satisfying the property is exactly {1, 2, 6, 42, 1806} -/
theorem property_set_characterization :
  property_set = {1, 2, 6, 42, 1806} := by sorry

end property_set_characterization_l3509_350901


namespace polynomial_expansion_and_sum_l3509_350933

theorem polynomial_expansion_and_sum (A B C D E : ℤ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6) = A * x^4 + B * x^3 + C * x^2 + D * x + E) →
  A = 4 ∧ B = 10 ∧ C = 1 ∧ D = 15 ∧ E = -18 ∧ A + B + C + D + E = 12 := by
sorry

end polynomial_expansion_and_sum_l3509_350933


namespace isosceles_triangle_angle_measure_l3509_350939

/-- 
An isosceles triangle with one angle 20% smaller than a right angle 
has its two largest angles measuring 54 degrees each.
-/
theorem isosceles_triangle_angle_measure : 
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle (c) is 20% smaller than a right angle (90°)
  c = 0.8 * 90 →
  -- Each of the two largest angles (a and b) measures 54°
  a = 54 ∧ b = 54 := by
sorry

end isosceles_triangle_angle_measure_l3509_350939


namespace exponent_of_5_in_30_factorial_is_7_l3509_350999

/-- The exponent of 5 in the prime factorization of 30! -/
def exponent_of_5_in_30_factorial : ℕ :=
  (30 / 5) + (30 / 25)

/-- Theorem stating that the exponent of 5 in the prime factorization of 30! is 7 -/
theorem exponent_of_5_in_30_factorial_is_7 :
  exponent_of_5_in_30_factorial = 7 := by
  sorry

end exponent_of_5_in_30_factorial_is_7_l3509_350999


namespace sum_of_fractions_l3509_350945

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5)
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l3509_350945


namespace height_difference_l3509_350922

def elm_height : ℚ := 35 / 3
def oak_height : ℚ := 107 / 6

theorem height_difference : oak_height - elm_height = 37 / 6 := by
  sorry

end height_difference_l3509_350922


namespace triangle_area_with_squares_l3509_350928

/-- Given a scalene triangle with adjoining squares, prove its area -/
theorem triangle_area_with_squares (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 = 100 → b^2 = 64 → c^2 = 49 → h^2 = 81 →
  (1/2 : ℝ) * a * h = 45 := by
sorry

end triangle_area_with_squares_l3509_350928


namespace other_person_money_l3509_350938

/-- If Mia has $110 and this amount is $20 more than twice as much money as someone else, then that person has $45. -/
theorem other_person_money (mia_money : ℕ) (other_money : ℕ) : 
  mia_money = 110 → mia_money = 2 * other_money + 20 → other_money = 45 := by
  sorry

end other_person_money_l3509_350938


namespace cos_180_degrees_l3509_350988

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l3509_350988


namespace intersection_and_system_solution_l3509_350950

theorem intersection_and_system_solution :
  ∀ (m n : ℝ),
  (∃ (x y : ℝ), y = -x + 4 ∧ y = 2*x + m ∧ x = 3 ∧ y = n) →
  (∀ (x y : ℝ), x + y - 4 = 0 ∧ 2*x - y + m = 0 ↔ x = 3 ∧ y = 1) :=
by sorry

end intersection_and_system_solution_l3509_350950


namespace pizza_slices_pizza_has_eight_slices_l3509_350963

theorem pizza_slices : ℕ → Prop :=
  fun total_slices =>
    let remaining_after_friend := total_slices - 2
    let james_slices := remaining_after_friend / 2
    james_slices = 3 → total_slices = 8

/-- The pizza has 8 slices. -/
theorem pizza_has_eight_slices : ∃ (n : ℕ), pizza_slices n :=
  sorry

end pizza_slices_pizza_has_eight_slices_l3509_350963


namespace quadratic_equation_solution_l3509_350985

theorem quadratic_equation_solution :
  let equation := fun x : ℝ => 2 * x^2 + 6 * x - 1
  let solution1 := -3/2 + Real.sqrt 11 / 2
  let solution2 := -3/2 - Real.sqrt 11 / 2
  equation solution1 = 0 ∧ equation solution2 = 0 :=
by sorry

end quadratic_equation_solution_l3509_350985


namespace nine_sided_polygon_diagonals_l3509_350961

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular 9-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3509_350961


namespace billy_ferris_wheel_rides_l3509_350995

/-- The number of times Billy rode the ferris wheel -/
def F : ℕ := sorry

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

theorem billy_ferris_wheel_rides : F = 7 := by
  sorry

end billy_ferris_wheel_rides_l3509_350995


namespace min_value_ab_l3509_350919

theorem min_value_ab (a b : ℝ) (h : (4 / a) + (1 / b) = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, ((4 / x) + (1 / y) = Real.sqrt (x * y)) → (a * b) ≤ (x * y) :=
by
  sorry

end min_value_ab_l3509_350919


namespace octagon_triangle_side_ratio_l3509_350902

theorem octagon_triangle_side_ratio : 
  ∀ (s_o s_t : ℝ), s_o > 0 → s_t > 0 →
  (2 * Real.sqrt 2) * s_o^2 = (Real.sqrt 3 / 4) * s_t^2 →
  s_t / s_o = 2 * (2 : ℝ)^(1/4) :=
by
  sorry

end octagon_triangle_side_ratio_l3509_350902


namespace twelve_sided_polygon_equilateral_triangles_l3509_350925

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Function to check if a triangle is equilateral -/
def isEquilateral (t : EquilateralTriangle) : Prop := sorry

/-- Function to check if a triangle has at least two vertices from a given set -/
def hasAtLeastTwoVerticesFrom (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

/-- The main theorem -/
theorem twelve_sided_polygon_equilateral_triangles 
  (p : RegularPolygon 12) : 
  ∃ (ts : Finset EquilateralTriangle), 
    (∀ t ∈ ts, isEquilateral t ∧ 
      hasAtLeastTwoVerticesFrom t (Set.range p.vertices)) ∧ 
    ts.card ≥ 12 := by sorry

end twelve_sided_polygon_equilateral_triangles_l3509_350925


namespace decimal_89_to_binary_l3509_350909

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_89_to_binary :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end decimal_89_to_binary_l3509_350909


namespace log_x3y2_equals_2_l3509_350916

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x * y^2) = 2) 
  (h2 : Real.log (x^2 * y^3) = 3) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end log_x3y2_equals_2_l3509_350916


namespace negation_equivalence_l3509_350990

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negation_equivalence_l3509_350990


namespace card_value_decrease_l3509_350983

/-- Represents the percent decrease in the first year -/
def first_year_decrease : ℝ := sorry

/-- Represents the percent decrease in the second year -/
def second_year_decrease : ℝ := 10

/-- Represents the total percent decrease over two years -/
def total_decrease : ℝ := 55

theorem card_value_decrease :
  (1 - first_year_decrease / 100) * (1 - second_year_decrease / 100) = 1 - total_decrease / 100 ∧
  first_year_decrease = 50 := by sorry

end card_value_decrease_l3509_350983


namespace knitting_time_theorem_l3509_350964

/-- Represents the time in hours to knit each item -/
structure KnittingTime where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  mittens : ℝ
  socks : ℝ

/-- Calculates the total time to knit multiple sets of clothes -/
def totalKnittingTime (time : KnittingTime) (sets : ℕ) : ℝ :=
  (time.hat + time.scarf + time.sweater + time.mittens + time.socks) * sets

/-- Theorem: The total time to knit 3 sets of clothes with given knitting times is 48 hours -/
theorem knitting_time_theorem (time : KnittingTime) 
  (h_hat : time.hat = 2)
  (h_scarf : time.scarf = 3)
  (h_sweater : time.sweater = 6)
  (h_mittens : time.mittens = 2)
  (h_socks : time.socks = 3) :
  totalKnittingTime time 3 = 48 := by
  sorry

#check knitting_time_theorem

end knitting_time_theorem_l3509_350964


namespace bijection_probability_l3509_350993

/-- The probability of establishing a bijection from a subset of a 4-element set to a 5-element set -/
theorem bijection_probability (A : Finset α) (B : Finset β) 
  (hA : Finset.card A = 4) (hB : Finset.card B = 5) : 
  (Finset.card (Finset.powersetCard 4 B) / (Finset.card B ^ Finset.card A) : ℚ) = 24/125 :=
sorry

end bijection_probability_l3509_350993


namespace exists_non_zero_sign_function_l3509_350989

/-- Given functions on a blackboard -/
def f₁ (x : ℝ) : ℝ := x + 1
def f₂ (x : ℝ) : ℝ := x^2 + 1
def f₃ (x : ℝ) : ℝ := x^3 + 1
def f₄ (x : ℝ) : ℝ := x^4 + 1

/-- The set of functions that can be constructed from the given functions -/
inductive ConstructibleFunction : (ℝ → ℝ) → Prop
  | base₁ : ConstructibleFunction f₁
  | base₂ : ConstructibleFunction f₂
  | base₃ : ConstructibleFunction f₃
  | base₄ : ConstructibleFunction f₄
  | sub (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x - g x)
  | mul (f g : ℝ → ℝ) : ConstructibleFunction f → ConstructibleFunction g → ConstructibleFunction (λ x => f x * g x)

/-- The theorem to be proved -/
theorem exists_non_zero_sign_function :
  ∃ (f : ℝ → ℝ), ConstructibleFunction f ∧ f ≠ 0 ∧
  (∀ x > 0, f x ≥ 0) ∧ (∀ x < 0, f x ≤ 0) := by
  sorry

end exists_non_zero_sign_function_l3509_350989


namespace no_perfect_square_in_range_l3509_350942

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 12 ∧ ∃ (m : ℕ), 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end no_perfect_square_in_range_l3509_350942


namespace exists_polygon_different_centers_l3509_350900

/-- A polygon is represented by a list of its vertices --/
def Polygon := List (ℝ × ℝ)

/-- Calculate the center of gravity of a polygon's vertices --/
noncomputable def centerOfGravityVertices (p : Polygon) : ℝ × ℝ := sorry

/-- Calculate the center of gravity of a polygon plate --/
noncomputable def centerOfGravityPlate (p : Polygon) : ℝ × ℝ := sorry

/-- The theorem stating that there exists a polygon where the centers of gravity don't coincide --/
theorem exists_polygon_different_centers : 
  ∃ (p : Polygon), centerOfGravityVertices p ≠ centerOfGravityPlate p := by sorry

end exists_polygon_different_centers_l3509_350900


namespace polynomial_equality_l3509_350986

theorem polynomial_equality : 2090^3 + 2089 * 2090^2 - 2089^2 * 2090 + 2089^3 = 4179 := by
  sorry

end polynomial_equality_l3509_350986


namespace no_solution_implies_a_leq_two_l3509_350977

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end no_solution_implies_a_leq_two_l3509_350977


namespace sum_of_common_ratios_is_three_l3509_350904

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 3. -/
theorem sum_of_common_ratios_is_three
  (k p r : ℝ)
  (h_nonconstant : k ≠ 0)
  (h_different_ratios : p ≠ r)
  (h_condition : k * p^2 - k * r^2 = 3 * (k * p - k * r)) :
  p + r = 3 := by
sorry

end sum_of_common_ratios_is_three_l3509_350904


namespace planes_parallel_from_intersecting_lines_l3509_350982

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_in : Line → Plane → Prop)  -- A line lies in a plane
variable (parallel : Line → Plane → Prop)  -- A line is parallel to a plane
variable (intersect_at : Line → Line → Point → Prop)  -- Two lines intersect at a point
variable (plane_parallel : Plane → Plane → Prop)  -- Two planes are parallel

-- State the theorem
theorem planes_parallel_from_intersecting_lines 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →  -- l and m are distinct lines
  α ≠ β →  -- α and β are different planes
  lies_in l α →
  lies_in m α →
  intersect_at l m P →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end planes_parallel_from_intersecting_lines_l3509_350982


namespace sum_of_distinct_integers_l3509_350994

theorem sum_of_distinct_integers (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72 →
  p + q + r + s + t = 25 := by
sorry

end sum_of_distinct_integers_l3509_350994


namespace arithmetic_sequence_a6_l3509_350926

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SequenceCondition a) : 
  a 6 = 3 := by sorry

end arithmetic_sequence_a6_l3509_350926


namespace initial_markup_percentage_l3509_350906

theorem initial_markup_percentage (initial_price : ℝ) (additional_increase : ℝ) : 
  initial_price = 45 →
  additional_increase = 5 →
  initial_price + additional_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 8)))) →
  (initial_price - (initial_price / (1 + 8))) / (initial_price / (1 + 8)) = 8 := by
  sorry

end initial_markup_percentage_l3509_350906


namespace probability_two_girls_chosen_l3509_350972

def total_members : ℕ := 12
def num_boys : ℕ := 7
def num_girls : ℕ := 5

theorem probability_two_girls_chosen (total_members num_boys num_girls : ℕ) 
  (h1 : total_members = 12)
  (h2 : num_boys = 7)
  (h3 : num_girls = 5)
  (h4 : total_members = num_boys + num_girls) :
  (Nat.choose num_girls 2 : ℚ) / (Nat.choose total_members 2) = 5 / 33 := by
sorry

end probability_two_girls_chosen_l3509_350972


namespace complex_2_minus_3i_in_fourth_quadrant_l3509_350908

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_2_minus_3i_in_fourth_quadrant :
  is_in_fourth_quadrant (2 - 3*I) := by
  sorry

end complex_2_minus_3i_in_fourth_quadrant_l3509_350908


namespace line_parametrization_l3509_350967

/-- The slope of the line -/
def m : ℚ := 3/4

/-- The y-intercept of the line -/
def b : ℚ := 3

/-- The x-coordinate of the point on the line when t = 0 -/
def x₀ : ℚ := -9

/-- The y-coordinate of the direction vector -/
def v : ℚ := -7

/-- The equation of the line -/
def line_eq (x y : ℚ) : Prop := y = m * x + b

/-- The parametric equations of the line -/
def param_eq (x y s l t : ℚ) : Prop :=
  x = x₀ + t * l ∧ y = s + t * v

theorem line_parametrization (s l : ℚ) : 
  (∀ t, line_eq (x₀ + t * l) (s + t * v)) ↔ s = -15/4 ∧ l = -28/3 :=
sorry

end line_parametrization_l3509_350967


namespace problem_statement_l3509_350915

theorem problem_statement (a b c d : ℝ) :
  Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3 →
  a - b + c - d = -7 := by
sorry

end problem_statement_l3509_350915


namespace cyclists_meet_time_l3509_350965

/-- The time (in hours after 8:00 AM) when Cassie and Brian meet -/
def meeting_time : ℝ := 2.68333333

/-- The total distance of the route in miles -/
def total_distance : ℝ := 75

/-- Cassie's speed in miles per hour -/
def cassie_speed : ℝ := 15

/-- Brian's speed in miles per hour -/
def brian_speed : ℝ := 18

/-- The time difference between Cassie and Brian's departure in hours -/
def time_difference : ℝ := 0.75

theorem cyclists_meet_time :
  cassie_speed * meeting_time + brian_speed * (meeting_time - time_difference) = total_distance :=
sorry

end cyclists_meet_time_l3509_350965


namespace missing_number_is_sixty_l3509_350973

/-- Given that the average of 20, 40, and 60 is 5 more than the average of 10, x, and 35,
    prove that x = 60. -/
theorem missing_number_is_sixty :
  ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 := by
sorry

end missing_number_is_sixty_l3509_350973


namespace circumcircle_intersection_l3509_350992

-- Define a point in a plane
structure Point : Type :=
  (x : ℝ) (y : ℝ)

-- Define a circle
structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define a function to check if a point lies on a circle
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define a function to create a circumcircle of a triangle
def circumcircle (a b c : Point) : Circle :=
  sorry

-- Define the main theorem
theorem circumcircle_intersection
  (A₁ A₂ B₁ B₂ C₁ C₂ : Point)
  (h : ∃ (P : Point),
    pointOnCircle P (circumcircle A₁ B₁ C₁) ∧
    pointOnCircle P (circumcircle A₁ B₂ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₁ C₂) ∧
    pointOnCircle P (circumcircle A₂ B₂ C₁)) :
  ∃ (Q : Point),
    pointOnCircle Q (circumcircle A₂ B₂ C₂) ∧
    pointOnCircle Q (circumcircle A₂ B₁ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₂ C₁) ∧
    pointOnCircle Q (circumcircle A₁ B₁ C₂) :=
sorry

end circumcircle_intersection_l3509_350992


namespace inequalities_solution_l3509_350918

theorem inequalities_solution :
  (∀ x : ℝ, x * (9 - x) > 0 ↔ 0 < x ∧ x < 9) ∧
  (∀ x : ℝ, 16 - x^2 ≤ 0 ↔ x ≤ -4 ∨ x ≥ 4) := by sorry

end inequalities_solution_l3509_350918


namespace b_investment_l3509_350959

/-- Represents the investment and profit share of a person in the business. -/
structure Participant where
  investment : ℝ
  profitShare : ℝ

/-- Proves that given the conditions of the problem, b's investment is 10000. -/
theorem b_investment (a b c : Participant)
  (h1 : b.profitShare = 3500)
  (h2 : c.profitShare - a.profitShare = 1399.9999999999998)
  (h3 : a.investment = 8000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare / a.investment = b.profitShare / b.investment)
  (h6 : c.profitShare / c.investment = b.profitShare / b.investment) :
  b.investment = 10000 := by
  sorry


end b_investment_l3509_350959


namespace download_speed_scientific_notation_l3509_350935

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The theoretical download speed of the Huawei phone MateX on a 5G network in B/s -/
def download_speed : ℕ := 603000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem download_speed_scientific_notation :
  to_scientific_notation download_speed = ScientificNotation.mk 6.03 8 (by sorry) :=
sorry

end download_speed_scientific_notation_l3509_350935


namespace weight_of_steel_ingot_l3509_350948

/-- Given a weight vest and purchase conditions, prove the weight of each steel ingot. -/
theorem weight_of_steel_ingot
  (original_weight : ℝ)
  (weight_increase_percent : ℝ)
  (ingot_cost : ℝ)
  (discount_percent : ℝ)
  (final_cost : ℝ)
  (h1 : original_weight = 60)
  (h2 : weight_increase_percent = 0.60)
  (h3 : ingot_cost = 5)
  (h4 : discount_percent = 0.20)
  (h5 : final_cost = 72)
  : ∃ (num_ingots : ℕ), 
    num_ingots > 10 ∧ 
    (2 : ℝ) = (original_weight * weight_increase_percent) / num_ingots :=
by sorry

end weight_of_steel_ingot_l3509_350948


namespace sum_of_30th_set_l3509_350978

/-- Defines the first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + (n * (n - 1)) / 2

/-- Defines the last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Defines the sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

theorem sum_of_30th_set : S 30 = 13515 := by
  sorry

end sum_of_30th_set_l3509_350978


namespace vertical_shift_graph_l3509_350927

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define vertical shift operation
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- Theorem statement
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x ↔ (y + k) = (verticalShift f k) x :=
sorry

end vertical_shift_graph_l3509_350927


namespace largest_n_satisfying_conditions_l3509_350911

theorem largest_n_satisfying_conditions : ∃ (m k : ℤ),
  181^2 = (m + 1)^3 - m^3 ∧
  2 * 181 + 79 = k^2 ∧
  ∀ (n : ℤ), n > 181 → ¬(∃ (m' k' : ℤ), n^2 = (m' + 1)^3 - m'^3 ∧ 2 * n + 79 = k'^2) :=
by sorry

end largest_n_satisfying_conditions_l3509_350911


namespace probability_diamond_then_ace_l3509_350929

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamonds in a standard deck -/
def DiamondCount : ℕ := 13

/-- Represents the number of aces in a standard deck -/
def AceCount : ℕ := 4

/-- Represents the remaining deck after one card (not a diamond ace) has been dealt -/
def RemainingDeck : ℕ := StandardDeck - 1

/-- Represents the number of diamonds (excluding ace) in the remaining deck -/
def RemainingDiamonds : ℕ := DiamondCount - 1

theorem probability_diamond_then_ace :
  (RemainingDiamonds : ℚ) / RemainingDeck * AceCount / (RemainingDeck - 1) = 24 / 1275 := by
  sorry

end probability_diamond_then_ace_l3509_350929


namespace inequality_relation_l3509_350921

theorem inequality_relation (x y : ℝ) :
  (x^3 + x > x^2*y + y) → (x - y > -1) ∧
  ¬(∀ x y : ℝ, x - y > -1 → x^3 + x > x^2*y + y) :=
by sorry

end inequality_relation_l3509_350921


namespace point_upper_right_of_line_l3509_350944

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the concept of a point being to the upper right of a line
def upper_right_of_line (x y : ℝ) : Prop := x + y - 3 > 0

-- Theorem statement
theorem point_upper_right_of_line (a : ℝ) :
  upper_right_of_line (-1) a → a > 4 := by
  sorry

end point_upper_right_of_line_l3509_350944


namespace roses_cut_equality_l3509_350991

/-- Represents the number of roses in various states --/
structure RoseCount where
  initial : ℕ
  thrown : ℕ
  given : ℕ
  final : ℕ

/-- Calculates the number of roses cut --/
def rosesCut (r : RoseCount) : ℕ :=
  r.final - r.initial + r.thrown + r.given

/-- Theorem stating that the number of roses cut is equal to the sum of
    the difference between final and initial roses, roses thrown away, and roses given away --/
theorem roses_cut_equality (r : RoseCount) :
  rosesCut r = r.final - r.initial + r.thrown + r.given :=
by sorry

end roses_cut_equality_l3509_350991


namespace probability_at_least_one_correct_l3509_350996

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 1 - (1 - 1 / choices) ^ n = 31031 / 46656 := by
  sorry

end probability_at_least_one_correct_l3509_350996


namespace brayden_gavin_touchdowns_l3509_350931

theorem brayden_gavin_touchdowns :
  let touchdown_points : ℕ := 7
  let cole_freddy_touchdowns : ℕ := 9
  let point_difference : ℕ := 14
  let brayden_gavin_touchdowns : ℕ := 7

  touchdown_points * cole_freddy_touchdowns = 
  touchdown_points * brayden_gavin_touchdowns + point_difference :=
by
  sorry

end brayden_gavin_touchdowns_l3509_350931


namespace parabola_translation_l3509_350947

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := 2 * (x - 2)^2 + 3

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = translated_parabola x ↔ y - 3 = original_parabola (x - 2) :=
sorry

end parabola_translation_l3509_350947


namespace taxi_arrangement_count_l3509_350976

/-- The number of ways to divide 6 people into two taxis, where each taxi can carry up to 4 people -/
def taxi_arrangements : ℕ := 50

/-- The number of people to be divided -/
def num_people : ℕ := 6

/-- The maximum capacity of each taxi -/
def max_capacity : ℕ := 4

/-- Theorem stating that the number of ways to divide 6 people into two taxis, 
    where each taxi can carry up to 4 people, is equal to 50 -/
theorem taxi_arrangement_count : 
  taxi_arrangements = 50 ∧ 
  num_people = 6 ∧ 
  max_capacity = 4 := by
  sorry

end taxi_arrangement_count_l3509_350976


namespace kenny_sunday_jumping_jacks_l3509_350930

/-- Represents the number of jumping jacks Kenny did on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def totalJumpingJacks (week : WeeklyJumpingJacks) : ℕ :=
  week.sunday + week.monday + week.tuesday + week.wednesday + week.thursday + week.friday + week.saturday

theorem kenny_sunday_jumping_jacks 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (h1 : lastWeek = 324)
  (h2 : thisWeek.tuesday = 0)
  (h3 : thisWeek.wednesday = 123)
  (h4 : thisWeek.thursday = 64)
  (h5 : thisWeek.friday = 23)
  (h6 : thisWeek.saturday = 61)
  (h7 : thisWeek.monday = 20 ∨ thisWeek.sunday = 20)
  (h8 : totalJumpingJacks thisWeek > lastWeek) :
  thisWeek.sunday = 33 := by
  sorry

end kenny_sunday_jumping_jacks_l3509_350930


namespace return_flight_speed_l3509_350905

/-- Proves that given a round trip flight with specified conditions, the return flight speed is 500 mph -/
theorem return_flight_speed 
  (total_distance : ℝ) 
  (outbound_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 3000) 
  (h2 : outbound_speed = 300) 
  (h3 : total_time = 8) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / outbound_speed) = 500 := by
  sorry

#check return_flight_speed

end return_flight_speed_l3509_350905


namespace stating_sum_of_sides_approx_11_2_l3509_350956

/-- Represents a right triangle with angles 40°, 50°, and 90° -/
structure RightTriangle40_50_90 where
  /-- The side opposite to the 50° angle -/
  side_a : ℝ
  /-- The side opposite to the 40° angle -/
  side_b : ℝ
  /-- The hypotenuse -/
  side_c : ℝ
  /-- Constraint that side_a is 8 units long -/
  side_a_eq_8 : side_a = 8

/-- 
Theorem stating that the sum of the two sides (opposite to 40° and 90°) 
in a 40-50-90 right triangle with hypotenuse of 8 units 
is approximately 11.2 units
-/
theorem sum_of_sides_approx_11_2 (t : RightTriangle40_50_90) :
  ∃ ε > 0, abs (t.side_b + t.side_c - 11.2) < ε := by
  sorry


end stating_sum_of_sides_approx_11_2_l3509_350956


namespace age_sum_proof_l3509_350969

/-- Tom's age in years -/
def tom_age : ℕ := 9

/-- Tom's sister's age in years -/
def sister_age : ℕ := tom_age / 2 + 1

/-- The sum of Tom's and his sister's ages -/
def sum_ages : ℕ := tom_age + sister_age

theorem age_sum_proof : sum_ages = 14 := by
  sorry

end age_sum_proof_l3509_350969


namespace lillians_candies_l3509_350914

theorem lillians_candies (initial_candies final_candies : ℕ) 
  (h1 : initial_candies = 88)
  (h2 : final_candies = 93) :
  final_candies - initial_candies = 5 := by
  sorry

end lillians_candies_l3509_350914


namespace camping_trip_percentage_l3509_350920

theorem camping_trip_percentage
  (total_students : ℕ)
  (h1 : (14 : ℚ) / 100 * total_students = (25 : ℚ) / 100 * (56 : ℚ) / 100 * total_students)
  (h2 : (75 : ℚ) / 100 * (56 : ℚ) / 100 * total_students + (14 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students) :
  (56 : ℚ) / 100 * total_students = (56 : ℚ) / 100 * total_students :=
by sorry

end camping_trip_percentage_l3509_350920


namespace clock_angle_at_7pm_l3509_350971

/-- The number of hours on a clock face. -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle_degrees : ℕ := 360

/-- The time in hours (7 p.m. is represented as 19). -/
def time : ℕ := 19

/-- The angle between hour marks on a clock face. -/
def angle_per_hour : ℚ := full_circle_degrees / clock_hours

/-- The number of hour marks between the hour hand and 12 o'clock at the given time. -/
def hour_hand_position : ℕ := time % clock_hours

/-- The angle between the hour and minute hands at the given time. -/
def clock_angle : ℚ := angle_per_hour * hour_hand_position

/-- The smaller angle between the hour and minute hands. -/
def smaller_angle : ℚ := min clock_angle (full_circle_degrees - clock_angle)

/-- 
Theorem: The measure of the smaller angle formed by the hour and minute hands 
of a clock at 7 p.m. is 150°.
-/
theorem clock_angle_at_7pm : smaller_angle = 150 := by sorry

end clock_angle_at_7pm_l3509_350971


namespace cubic_equation_solution_l3509_350937

theorem cubic_equation_solution (x : ℝ) (hx : x^3 + 6 * (x / (x - 3))^3 = 135) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 0 ∨ y = 23382 / 122 := by
sorry


end cubic_equation_solution_l3509_350937


namespace bookstore_calculator_sales_l3509_350934

theorem bookstore_calculator_sales
  (price1 : ℕ) (price2 : ℕ) (total_sales : ℕ) (quantity2 : ℕ)
  (h1 : price1 = 15)
  (h2 : price2 = 67)
  (h3 : total_sales = 3875)
  (h4 : quantity2 = 35)
  (h5 : ∃ quantity1 : ℕ, price1 * quantity1 + price2 * quantity2 = total_sales) :
  ∃ total_quantity : ℕ, total_quantity = quantity2 + (total_sales - price2 * quantity2) / price1 ∧
                        total_quantity = 137 := by
  sorry

end bookstore_calculator_sales_l3509_350934


namespace apples_bought_by_junhyeok_and_jihyun_l3509_350998

/-- The number of apple boxes Junhyeok bought -/
def junhyeok_boxes : ℕ := 7

/-- The number of apples in each of Junhyeok's boxes -/
def junhyeok_apples_per_box : ℕ := 16

/-- The number of apple boxes Jihyun bought -/
def jihyun_boxes : ℕ := 6

/-- The number of apples in each of Jihyun's boxes -/
def jihyun_apples_per_box : ℕ := 25

/-- The total number of apples bought by Junhyeok and Jihyun -/
def total_apples : ℕ := junhyeok_boxes * junhyeok_apples_per_box + jihyun_boxes * jihyun_apples_per_box

theorem apples_bought_by_junhyeok_and_jihyun : total_apples = 262 := by
  sorry

end apples_bought_by_junhyeok_and_jihyun_l3509_350998


namespace product_remainder_mod_five_l3509_350946

theorem product_remainder_mod_five : (1236 * 7483 * 53) % 5 = 4 := by
  sorry

end product_remainder_mod_five_l3509_350946


namespace four_inch_cube_worth_l3509_350980

/-- The worth of a cube of gold in dollars -/
def worth (side_length : ℝ) : ℝ :=
  300 * side_length^3

/-- Theorem: The worth of a 4-inch cube of gold is $19200 -/
theorem four_inch_cube_worth : worth 4 = 19200 := by
  sorry

end four_inch_cube_worth_l3509_350980


namespace star_six_three_l3509_350966

-- Define the binary operation *
def star (x y : ℝ) : ℝ := 4*x + 5*y - x*y

-- Theorem statement
theorem star_six_three : star 6 3 = 21 := by sorry

end star_six_three_l3509_350966


namespace counterexample_exists_l3509_350936

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n) % 9 = 0 ∧ 
  n % 3 = 0 ∧ 
  n % 9 ≠ 0 := by sorry

end counterexample_exists_l3509_350936


namespace decagon_cuts_to_two_regular_polygons_l3509_350970

/-- A regular polygon with n sides -/
structure RegularPolygon where
  sides : Nat
  isRegular : sides ≥ 3

/-- A decagon is a regular polygon with 10 sides -/
def Decagon : RegularPolygon where
  sides := 10
  isRegular := by norm_num

/-- Represent a cut of a polygon along its diagonals -/
structure DiagonalCut (p : RegularPolygon) where
  pieces : List RegularPolygon
  sum_sides : (pieces.map RegularPolygon.sides).sum = p.sides

/-- Theorem: A regular decagon can be cut into two regular polygons -/
theorem decagon_cuts_to_two_regular_polygons : 
  ∃ (cut : DiagonalCut Decagon), cut.pieces.length = 2 := by
  sorry

end decagon_cuts_to_two_regular_polygons_l3509_350970


namespace max_value_of_sum_cube_roots_l3509_350997

open Real

theorem max_value_of_sum_cube_roots (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + 
           (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7 ^ (1/3) := by
sorry

end max_value_of_sum_cube_roots_l3509_350997


namespace log_49_x_equals_half_log_7_x_l3509_350974

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_49_x_equals_half_log_7_x (x : ℝ) (h : log 7 (x + 6) = 2) :
  log 49 x = (log 7 x) / 2 := by sorry

end log_49_x_equals_half_log_7_x_l3509_350974


namespace bobbit_worm_predation_l3509_350987

/-- Calculates the number of fish remaining in an aquarium after a Bobbit worm's predation --/
theorem bobbit_worm_predation 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (added_fish : ℕ) 
  (days_after_adding : ℕ) :
  initial_fish = 60 →
  daily_eaten = 2 →
  days_before_adding = 14 →
  added_fish = 8 →
  days_after_adding = 7 →
  initial_fish + added_fish - (daily_eaten * (days_before_adding + days_after_adding)) = 26 :=
by sorry

end bobbit_worm_predation_l3509_350987


namespace perpendicular_lines_slope_l3509_350941

-- Define the slopes of two lines
def slope1 (k : ℝ) := k
def slope2 : ℝ := 2

-- Define the condition for perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_slope (k : ℝ) :
  perpendicular (slope1 k) slope2 → k = -1/2 := by
  sorry

end perpendicular_lines_slope_l3509_350941


namespace product_in_N_l3509_350949

-- Define set M
def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}

-- Define set N
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

-- Theorem statement
theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end product_in_N_l3509_350949


namespace sin_cos_sum_special_angles_l3509_350953

theorem sin_cos_sum_special_angles :
  Real.sin (36 * π / 180) * Real.cos (24 * π / 180) + 
  Real.cos (36 * π / 180) * Real.sin (156 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_special_angles_l3509_350953


namespace stratified_sampling_third_major_l3509_350979

/-- Given a college with three majors and stratified sampling, prove the number of students
    to be drawn from the third major. -/
theorem stratified_sampling_third_major
  (total_students : ℕ)
  (major_a_students : ℕ)
  (major_b_students : ℕ)
  (total_sample : ℕ)
  (h1 : total_students = 1200)
  (h2 : major_a_students = 380)
  (h3 : major_b_students = 420)
  (h4 : total_sample = 120) :
  (total_students - (major_a_students + major_b_students)) * total_sample / total_students = 40 :=
by sorry

end stratified_sampling_third_major_l3509_350979


namespace isosceles_triangle_perimeter_l3509_350923

/-- Given a quadratic equation and an isosceles triangle, prove the perimeter is 5 -/
theorem isosceles_triangle_perimeter (k : ℝ) : 
  let equation := fun x : ℝ => x^2 - (k+2)*x + 2*k
  ∃ (b c : ℝ), 
    equation b = 0 ∧ 
    equation c = 0 ∧ 
    b = c ∧ 
    b + c + 1 = 5 := by
  sorry

end isosceles_triangle_perimeter_l3509_350923


namespace cylinder_no_triangular_cross_section_l3509_350960

-- Define the types of geometric solids
inductive GeometricSolid
  | Cylinder
  | Cone
  | TriangularPrism
  | Cube

-- Define a function to check if a solid can have a triangular cross-section
def canHaveTriangularCrossSection (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_no_triangular_cross_section :
  ∀ (solid : GeometricSolid),
    ¬(canHaveTriangularCrossSection solid) ↔ solid = GeometricSolid.Cylinder :=
by sorry

end cylinder_no_triangular_cross_section_l3509_350960


namespace work_completion_time_l3509_350913

/-- Given that two workers 'a' and 'b' can complete a job together in 4 days,
    and 'a' alone can complete the job in 12 days, prove that 'b' alone
    can complete the job in 6 days. -/
theorem work_completion_time (work_rate_a : ℚ) (work_rate_b : ℚ) :
  work_rate_a + work_rate_b = 1 / 4 →
  work_rate_a = 1 / 12 →
  work_rate_b = 1 / 6 := by
sorry

end work_completion_time_l3509_350913


namespace jerry_piercing_earnings_l3509_350955

theorem jerry_piercing_earnings :
  let nose_price : ℚ := 20
  let ear_price : ℚ := nose_price * (1 + 1/2)
  let nose_count : ℕ := 6
  let ear_count : ℕ := 9
  nose_price * nose_count + ear_price * ear_count = 390 := by
  sorry

end jerry_piercing_earnings_l3509_350955


namespace max_value_cyclic_expression_l3509_350910

theorem max_value_cyclic_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27/8 ∧ 
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 3 ∧
    (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) = 27/8 :=
by sorry

end max_value_cyclic_expression_l3509_350910


namespace sequence_solution_l3509_350940

def sequence_problem (b : Fin 6 → ℝ) : Prop :=
  (∀ n : Fin 3, b (2 * n) = b (2 * n - 1) ^ 2) ∧
  (∀ n : Fin 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧
  b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧
  (∀ i : Fin 6, 0 ≤ b i)

theorem sequence_solution (b : Fin 6 → ℝ) (h : sequence_problem b) : b 1 = 1/2 := by
  sorry

end sequence_solution_l3509_350940


namespace team_score_l3509_350981

def basketball_game (tobee jay sean remy alex : ℕ) : Prop :=
  tobee = 4 ∧
  jay = 2 * tobee + 6 ∧
  sean = jay / 2 ∧
  remy = tobee + jay - 3 ∧
  alex = sean + remy + 4

theorem team_score (tobee jay sean remy alex : ℕ) :
  basketball_game tobee jay sean remy alex →
  tobee + jay + sean + remy + alex = 66 := by
  sorry

end team_score_l3509_350981


namespace total_fish_count_l3509_350943

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  morning_catch + afternoon_catch - thrown_back + dad_catch

/-- Theorem stating the total number of fish caught by Brendan and his dad -/
theorem total_fish_count :
  total_fish 8 3 5 13 = 23 := by
  sorry

end total_fish_count_l3509_350943


namespace number_equation_solution_l3509_350951

theorem number_equation_solution : 
  ∃ x : ℝ, (42 - 3 * x = 12) ∧ (x = 10) := by
sorry

end number_equation_solution_l3509_350951


namespace rectangle_least_area_l3509_350903

theorem rectangle_least_area (l w : ℕ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * (l + w) = 100) : 
  l * w ≥ 49 := by
sorry

end rectangle_least_area_l3509_350903


namespace same_duration_trips_l3509_350954

/-- Proves that two trips with given distances and speed ratio have the same duration -/
theorem same_duration_trips (distance1 : ℝ) (distance2 : ℝ) (speed_ratio : ℝ) 
  (h1 : distance1 = 90) 
  (h2 : distance2 = 360) 
  (h3 : speed_ratio = 4) : 
  (distance1 / 1) = (distance2 / speed_ratio) := by
  sorry

#check same_duration_trips

end same_duration_trips_l3509_350954


namespace area_max_cyclic_l3509_350932

/-- A quadrilateral with sides a, b, c, d and diagonals e, f -/
structure Quadrilateral (α : Type*) [LinearOrderedField α] :=
  (a b c d e f : α)

/-- The area of a quadrilateral -/
def area {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) : α :=
  ((q.b + q.d - q.a + q.c) * (q.b + q.d + q.a - q.c) * 
   (q.a + q.c - q.b + q.d) * (q.a + q.b + q.c - q.d) - 
   4 * (q.a * q.c + q.b * q.d - q.e * q.f) * (q.a * q.c + q.b * q.d + q.e * q.f)) / 16

/-- The theorem stating that the area is maximized when ef = ac + bd -/
theorem area_max_cyclic {α : Type*} [LinearOrderedField α] (q : Quadrilateral α) :
  area q ≤ area { q with e := (q.a * q.c + q.b * q.d) / q.f, f := q.f } :=
sorry

end area_max_cyclic_l3509_350932


namespace hourglass_problem_l3509_350984

/-- Given two hourglasses that can measure exactly 15 minutes, 
    where one measures 7 minutes, the other measures 2 minutes. -/
theorem hourglass_problem :
  ∀ (x : ℕ), 
    (∃ (n m k : ℕ), n * 7 + m * x + k * (x - 1) = 15 ∧ 
                     n > 0 ∧ m ≥ 0 ∧ k ≥ 0 ∧ 
                     (m = 0 ∨ k = 0)) → 
    x = 2 :=
by sorry

end hourglass_problem_l3509_350984


namespace shells_added_l3509_350907

/-- The amount of shells added to Jovana's bucket -/
theorem shells_added (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 5.75)
  (h2 : final_amount = 28.3) : 
  final_amount - initial_amount = 22.55 := by
  sorry

end shells_added_l3509_350907


namespace fencing_cost_calculation_l3509_350975

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem stating the total cost of fencing for a specific rectangular plot -/
theorem fencing_cost_calculation :
  let length : ℝ := 62
  let breadth : ℝ := 38
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
sorry

end fencing_cost_calculation_l3509_350975


namespace quadratic_function_bound_l3509_350924

def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

theorem quadratic_function_bound (p q : ℝ) :
  (max (|f p q 1|) (max (|f p q 2|) (|f p q 3|))) ≥ (1/2 : ℝ) := by sorry

end quadratic_function_bound_l3509_350924


namespace quadratic_equation_solution_l3509_350917

theorem quadratic_equation_solution (x : ℝ) : x^2 + 2*x - 8 = 0 ↔ x = -4 ∨ x = 2 := by sorry

end quadratic_equation_solution_l3509_350917


namespace fixed_point_exponential_function_l3509_350912

/-- For any positive real number a, the function f(x) = 2 + a^(x-1) always passes through the point (1, 3) -/
theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 := by sorry

end fixed_point_exponential_function_l3509_350912


namespace linear_function_decreasing_l3509_350957

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
sorry

end linear_function_decreasing_l3509_350957


namespace parallelogram_height_l3509_350968

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 33.3) (h2 : base = 9) 
    (h3 : area = base * height) : height = 3.7 := by
  sorry

end parallelogram_height_l3509_350968


namespace regular_polygon_with_30_degree_exterior_angle_has_12_sides_l3509_350952

/-- A regular polygon with an exterior angle of 30° has 12 sides. -/
theorem regular_polygon_with_30_degree_exterior_angle_has_12_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n ≥ 3 →
    exterior_angle = 30 * (π / 180) →
    (360 : ℝ) * (π / 180) = n * exterior_angle →
    n = 12 :=
by sorry

end regular_polygon_with_30_degree_exterior_angle_has_12_sides_l3509_350952


namespace infinite_solutions_abs_value_equation_l3509_350958

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end infinite_solutions_abs_value_equation_l3509_350958


namespace stratified_sampling_arrangements_l3509_350962

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4

theorem stratified_sampling_arrangements :
  (Nat.choose total_balls black_balls) = number_of_arrangements :=
by sorry

#check stratified_sampling_arrangements

end stratified_sampling_arrangements_l3509_350962
