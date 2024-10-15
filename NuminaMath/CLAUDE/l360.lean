import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solutions_l360_36032

theorem inequality_solutions :
  (∀ x : ℝ, 5 * x + 3 < 11 + x ↔ x < 2) ∧
  (∀ x : ℝ, 2 * x + 1 < 3 * x + 3 ∧ (x + 1) / 2 ≤ (1 - x) / 6 + 1 ↔ -2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l360_36032


namespace NUMINAMATH_CALUDE_brick_length_calculation_brick_length_is_20cm_l360_36091

/-- Given a courtyard and brick specifications, calculate the length of each brick. -/
theorem brick_length_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width * 10000 -- Convert to cm²
  let brick_area := courtyard_area / total_bricks
  brick_area / brick_width

/-- Prove that for the given specifications, the brick length is 20 cm. -/
theorem brick_length_is_20cm : 
  brick_length_calculation 30 16 10 24000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_brick_length_is_20cm_l360_36091


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l360_36055

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 70) 
  (h2 : average_speed = 80) : 
  (2 * average_speed - speed_first_hour) = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l360_36055


namespace NUMINAMATH_CALUDE_trig_identity_proof_l360_36039

theorem trig_identity_proof : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.sin (70 * π / 180)) = 4 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l360_36039


namespace NUMINAMATH_CALUDE_green_marble_probability_l360_36020

/-- Represents a bag of marbles with specific colors and quantities -/
structure MarbleBag where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def drawProbability (bag : MarbleBag) (color : String) : ℚ :=
  if color == bag.color1 then
    bag.count1 / (bag.count1 + bag.count2)
  else if color == bag.color2 then
    bag.count2 / (bag.count1 + bag.count2)
  else
    0

/-- The main theorem stating the probability of drawing a green marble -/
theorem green_marble_probability
  (bagX : MarbleBag)
  (bagY : MarbleBag)
  (bagZ : MarbleBag)
  (hX : bagX = ⟨"white", 5, "black", 5⟩)
  (hY : bagY = ⟨"green", 4, "red", 6⟩)
  (hZ : bagZ = ⟨"green", 3, "purple", 7⟩) :
  let probWhiteX := drawProbability bagX "white"
  let probGreenY := drawProbability bagY "green"
  let probBlackX := drawProbability bagX "black"
  let probGreenZ := drawProbability bagZ "green"
  probWhiteX * probGreenY + probBlackX * probGreenZ = 7 / 20 := by
  sorry


end NUMINAMATH_CALUDE_green_marble_probability_l360_36020


namespace NUMINAMATH_CALUDE_percentage_needed_to_pass_l360_36024

def total_marks : ℕ := 2075
def pradeep_score : ℕ := 390
def failed_by : ℕ := 25

def passing_mark : ℕ := pradeep_score + failed_by

def percentage_to_pass : ℚ := (passing_mark : ℚ) / (total_marks : ℚ) * 100

theorem percentage_needed_to_pass :
  ∃ (ε : ℚ), abs (percentage_to_pass - 20) < ε ∧ ε > 0 :=
sorry

end NUMINAMATH_CALUDE_percentage_needed_to_pass_l360_36024


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l360_36068

theorem fraction_simplest_form (a b c : ℝ) :
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) =
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) := by sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l360_36068


namespace NUMINAMATH_CALUDE_negative_integers_abs_not_greater_than_4_l360_36011

def negativeIntegersWithAbsNotGreaterThan4 : Set ℤ :=
  {x : ℤ | x < 0 ∧ |x| ≤ 4}

theorem negative_integers_abs_not_greater_than_4 :
  negativeIntegersWithAbsNotGreaterThan4 = {-1, -2, -3, -4} := by
  sorry

end NUMINAMATH_CALUDE_negative_integers_abs_not_greater_than_4_l360_36011


namespace NUMINAMATH_CALUDE_last_digit_is_11_l360_36047

def fibonacci_mod_12 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibonacci_mod_12 (n + 1) + fibonacci_mod_12 n) % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod_12 k = d

theorem last_digit_is_11 :
  ∀ d : ℕ, d < 12 →
    ∃ n : ℕ, digit_appears d n ∧
      ¬∃ m : ℕ, m > n ∧ digit_appears 11 m ∧ ¬digit_appears 11 n :=
by sorry

end NUMINAMATH_CALUDE_last_digit_is_11_l360_36047


namespace NUMINAMATH_CALUDE_six_digit_number_rotation_l360_36067

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def rotate_last_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let r := n / 10
  d * 100000 + r

theorem six_digit_number_rotation (n : ℕ) :
  is_six_digit n ∧ rotate_last_to_first n = n / 3 → n = 428571 ∨ n = 857142 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_rotation_l360_36067


namespace NUMINAMATH_CALUDE_brians_pencils_l360_36028

/-- Given Brian's initial pencil count, the number he gives away, and the number he buys,
    prove that his final pencil count is equal to the initial count minus the number given away
    plus the number bought. -/
theorem brians_pencils (initial : ℕ) (given_away : ℕ) (bought : ℕ) :
  initial - given_away + bought = initial - given_away + bought :=
by sorry

end NUMINAMATH_CALUDE_brians_pencils_l360_36028


namespace NUMINAMATH_CALUDE_car_travel_distance_l360_36071

/-- Represents the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

theorem car_travel_distance :
  let initial_distance : ℚ := 3
  let initial_time : ℚ := 4
  let total_time : ℚ := 120
  distance_traveled (initial_distance / initial_time) total_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l360_36071


namespace NUMINAMATH_CALUDE_triangle_inradius_l360_36009

/-- Given a triangle with perimeter 40 and area 50, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : P = 40) 
  (h2 : A = 50) 
  (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l360_36009


namespace NUMINAMATH_CALUDE_counterexample_non_coprime_l360_36008

theorem counterexample_non_coprime :
  ∃ (a n : ℕ+), (Nat.gcd a.val n.val ≠ 1) ∧ (a.val ^ n.val % n.val ≠ a.val % n.val) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_non_coprime_l360_36008


namespace NUMINAMATH_CALUDE_frame_price_ratio_l360_36073

/-- Calculates the ratio of the price of a smaller frame to the price of an initially intended frame given specific conditions --/
theorem frame_price_ratio (budget : ℚ) (initial_frame_markup : ℚ) (remaining : ℚ) : 
  budget = 60 →
  initial_frame_markup = 0.2 →
  remaining = 6 →
  let initial_frame_price := budget * (1 + initial_frame_markup)
  let smaller_frame_price := budget - remaining
  let ratio := smaller_frame_price / initial_frame_price
  ratio = 3/4 := by
    sorry


end NUMINAMATH_CALUDE_frame_price_ratio_l360_36073


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l360_36036

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (2,4) to (6,10), prove that b = 11. -/
theorem perpendicular_bisector_b_value : 
  let point1 : ℝ × ℝ := (2, 4)
  let point2 : ℝ × ℝ := (6, 10)
  let midpoint : ℝ × ℝ := ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)
  ∃ b : ℝ, (∀ (x y : ℝ), x + y = b ↔ ((x - midpoint.1) ^ 2 + (y - midpoint.2) ^ 2 = 
    (point1.1 - midpoint.1) ^ 2 + (point1.2 - midpoint.2) ^ 2)) → b = 11 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l360_36036


namespace NUMINAMATH_CALUDE_similar_triangles_solution_l360_36023

/-- Two similar right triangles, one with legs 12 and 9, the other with legs x and 6 -/
def similar_triangles (x : ℝ) : Prop :=
  (12 : ℝ) / x = 9 / 6

theorem similar_triangles_solution :
  ∃ x : ℝ, similar_triangles x ∧ x = 8 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_solution_l360_36023


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l360_36044

/-- The number of valid combinations for a wizard's elixir --/
def validCombinations (herbs : ℕ) (gems : ℕ) (incompatible : ℕ) : ℕ :=
  herbs * gems - incompatible

/-- Theorem: Given 4 herbs, 6 gems, and 3 invalid combinations, 
    the number of valid combinations is 21 --/
theorem wizard_elixir_combinations : 
  validCombinations 4 6 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l360_36044


namespace NUMINAMATH_CALUDE_cone_base_radius_l360_36049

/-- Given a sector with radius 15 cm and central angle 120 degrees used to form a cone without seam loss, 
    the radius of the base of the cone is 5 cm. -/
theorem cone_base_radius (sector_radius : ℝ) (central_angle : ℝ) (base_radius : ℝ) : 
  sector_radius = 15 → 
  central_angle = 120 → 
  base_radius = (central_angle / 360) * sector_radius → 
  base_radius = 5 := by
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l360_36049


namespace NUMINAMATH_CALUDE_satisfaction_ratings_properties_l360_36034

def satisfaction_ratings : List ℝ := [5, 7, 8, 9, 7, 5, 10, 8, 4, 7]

def mode (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem satisfaction_ratings_properties :
  mode satisfaction_ratings = 7 ∧
  range satisfaction_ratings = 6 ∧
  variance satisfaction_ratings = 3.2 :=
by sorry

end NUMINAMATH_CALUDE_satisfaction_ratings_properties_l360_36034


namespace NUMINAMATH_CALUDE_bell_ringing_fraction_l360_36005

theorem bell_ringing_fraction :
  let big_bell_rings : ℕ := 36
  let total_rings : ℕ := 52
  let small_bell_rings (f : ℚ) : ℚ := f * big_bell_rings + 4

  ∃ f : ℚ, f = 1/3 ∧ (↑big_bell_rings : ℚ) + small_bell_rings f = total_rings := by
  sorry

end NUMINAMATH_CALUDE_bell_ringing_fraction_l360_36005


namespace NUMINAMATH_CALUDE_area_triangle_ABG_l360_36035

/-- Given a rectangle ABCD and a square AEFG, where AB = 6, AD = 4, and the area of triangle ADE is 2,
    prove that the area of triangle ABG is 3. -/
theorem area_triangle_ABG (A B C D E F G : ℝ × ℝ) : 
  (∀ X Y, X ≠ Y → (X = A ∧ Y = B) ∨ (X = B ∧ Y = C) ∨ (X = C ∧ Y = D) ∨ (X = D ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - X.1)^2 + (Y.2 - X.2)^2) →  -- ABCD is a rectangle
  (∀ X Y, X ≠ Y → (X = A ∧ Y = E) ∨ (X = E ∧ Y = F) ∨ (X = F ∧ Y = G) ∨ (X = G ∧ Y = A) → 
    (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (E.1 - A.1)^2 + (E.2 - A.2)^2) →  -- AEFG is a square
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →  -- AB = 6
  (D.1 - A.1)^2 + (D.2 - A.2)^2 = 16 →  -- AD = 4
  abs ((E.1 - A.1) * (D.2 - A.2) - (E.2 - A.2) * (D.1 - A.1)) / 2 = 2 →  -- Area of triangle ADE = 2
  abs ((G.1 - A.1) * (B.2 - A.2) - (G.2 - A.2) * (B.1 - A.1)) / 2 = 3  -- Area of triangle ABG = 3
  := by sorry

end NUMINAMATH_CALUDE_area_triangle_ABG_l360_36035


namespace NUMINAMATH_CALUDE_inequality_relation_l360_36095

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_inequality_relation_l360_36095


namespace NUMINAMATH_CALUDE_distinct_power_representations_l360_36059

theorem distinct_power_representations : ∃ (N : ℕ) 
  (a₁ a₂ b₁ b₂ c₁ c₂ d₁ d₂ : ℕ), 
  (∃ (x y : ℕ), a₁ = x^2 ∧ a₂ = y^2) ∧
  (∃ (x y : ℕ), b₁ = x^3 ∧ b₂ = y^3) ∧
  (∃ (x y : ℕ), c₁ = x^5 ∧ c₂ = y^5) ∧
  (∃ (x y : ℕ), d₁ = x^7 ∧ d₂ = y^7) ∧
  N = a₁ - a₂ ∧
  N = b₁ - b₂ ∧
  N = c₁ - c₂ ∧
  N = d₁ - d₂ ∧
  a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ c₁ ≠ d₁ :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_power_representations_l360_36059


namespace NUMINAMATH_CALUDE_percentage_difference_l360_36080

theorem percentage_difference (x y z : ℝ) (hx : x = 5 * y) (hz : z = 1.2 * y) :
  (z - y) / x = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l360_36080


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_10_l360_36022

theorem natural_numbers_less_than_10 : 
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_10_l360_36022


namespace NUMINAMATH_CALUDE_range_of_n_l360_36045

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n m : ℝ) : Set ℝ := {x | n - m < x ∧ x < n + m}

theorem range_of_n (h : ∀ n : ℝ, (∃ x, x ∈ A ∩ B n 1) → ∃ x, x ∈ A ∩ B n 1) :
  ∀ n : ℝ, n ∈ Set.Ioo (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_n_l360_36045


namespace NUMINAMATH_CALUDE_sqrt_three_divided_by_sum_l360_36043

theorem sqrt_three_divided_by_sum : 
  Real.sqrt 3 / (Real.sqrt (1/3) + Real.sqrt (3/16)) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_divided_by_sum_l360_36043


namespace NUMINAMATH_CALUDE_expression_evaluation_l360_36006

theorem expression_evaluation : (((2200 - 2081)^2 + 100) : ℚ) / 196 = 73 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l360_36006


namespace NUMINAMATH_CALUDE_max_value_xyz_l360_36060

theorem max_value_xyz (x y z : ℝ) (h1 : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h2 : x + y + z = 1) (h3 : x^2 + y^2 + z^2 = 1) : 
  x + y^3 + z^4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_l360_36060


namespace NUMINAMATH_CALUDE_quadratic_solution_l360_36004

theorem quadratic_solution (h : 108 * (3/4)^2 - 35 * (3/4) - 77 = 0) :
  108 * (-23/54)^2 - 35 * (-23/54) - 77 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l360_36004


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_circles_lines_theorem_l360_36084

-- Ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

-- Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 1 = 0

-- Lines
def line1 (a x y : ℝ) : Prop := a^2 * x - y + 6 = 0
def line2 (a x y : ℝ) : Prop := 4 * x - (a - 3) * y + 9 = 0

theorem ellipse_hyperbola_circles_lines_theorem :
  (∃ (F₁ F₂ P : ℝ × ℝ), ellipse P.1 P.2 ∧ |P.1 - F₁.1| + |P.2 - F₁.2| = 3 ∧ |P.1 - F₂.1| + |P.2 - F₂.2| ≠ 1) ∧
  (∀ (x y : ℝ), hyperbola x y → (|y| - |3/4 * x| = 12/5)) ∧
  (∃ (t₁ t₂ : ℝ × ℝ), t₁ ≠ t₂ ∧ (∀ (x y : ℝ), circle1 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₁.1)^2 + (y - t₁.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle1 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0) ∧
                                 (∀ (x y : ℝ), circle2 x y → (x - t₂.1)^2 + (y - t₂.2)^2 = 0)) ∧
  (∃ (a : ℝ), a ≠ -1 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), line1 a x₁ y₁ ∧ line2 a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_circles_lines_theorem_l360_36084


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l360_36053

theorem absolute_value_equation_solutions (x : ℝ) :
  |5 * x - 4| = 29 ↔ x = -5 ∨ x = 33/5 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l360_36053


namespace NUMINAMATH_CALUDE_salary_increase_l360_36017

/-- Prove that adding a manager's salary increases the average salary by 100 --/
theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  avg_salary = 1700 →
  manager_salary = 3800 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l360_36017


namespace NUMINAMATH_CALUDE_max_ice_creams_l360_36010

/-- Given a budget and costs of items, calculate the maximum number of ice creams that can be bought -/
theorem max_ice_creams (budget : ℕ) (pancake_cost ice_cream_cost pancakes_bought : ℕ) : 
  budget = 60 →
  pancake_cost = 5 →
  ice_cream_cost = 8 →
  pancakes_bought = 5 →
  (budget - pancake_cost * pancakes_bought) / ice_cream_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_ice_creams_l360_36010


namespace NUMINAMATH_CALUDE_largest_invalid_sum_l360_36079

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b % 6 ≠ 0 ∧ n = 42 * a + b

theorem largest_invalid_sum : 
  (∀ m : ℕ, m > 252 → is_valid_sum m) ∧ ¬ is_valid_sum 252 :=
sorry

end NUMINAMATH_CALUDE_largest_invalid_sum_l360_36079


namespace NUMINAMATH_CALUDE_concert_revenue_l360_36000

theorem concert_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_tickets : total_tickets = 200)
  (h_revenue : total_revenue = 3000) : ℕ :=
by
  -- Let f be the number of full-price tickets
  -- Let d be the number of discount tickets
  -- Let p be the price of a full-price ticket
  have h1 : ∃ (f d p : ℕ), 
    f + d = total_tickets ∧ 
    f * p + d * (p / 3) = total_revenue ∧ 
    f * p = 1500
  sorry

  exact 1500


end NUMINAMATH_CALUDE_concert_revenue_l360_36000


namespace NUMINAMATH_CALUDE_equation_standard_form_and_coefficients_l360_36090

theorem equation_standard_form_and_coefficients :
  ∀ x : ℝ, x * (x + 1) = 2 * x - 1 ↔ x^2 - x + 1 = 0 ∧
  1 = 1 ∧ -1 = -1 ∧ 1 = 1 := by sorry

end NUMINAMATH_CALUDE_equation_standard_form_and_coefficients_l360_36090


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l360_36081

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l360_36081


namespace NUMINAMATH_CALUDE_work_time_ratio_l360_36066

/-- Given two workers A and B who can complete a job together in 6 days,
    and B can complete the job alone in 36 days,
    prove that the ratio of the time A takes to complete the job alone
    to the time B takes is 1:5. -/
theorem work_time_ratio
  (time_together : ℝ)
  (time_B : ℝ)
  (h_together : time_together = 6)
  (h_B : time_B = 36)
  (time_A : ℝ)
  (h_combined_rate : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A / time_B = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_work_time_ratio_l360_36066


namespace NUMINAMATH_CALUDE_fathers_digging_time_l360_36012

/-- The father's digging rate in feet per hour -/
def fathersRate : ℝ := 4

/-- The depth difference between Michael's hole and twice his father's hole depth in feet -/
def depthDifference : ℝ := 400

/-- Michael's digging time in hours -/
def michaelsTime : ℝ := 700

/-- Father's digging time in hours -/
def fathersTime : ℝ := 400

theorem fathers_digging_time :
  ∀ (fathersDepth michaelsDepth : ℝ),
  michaelsDepth = 2 * fathersDepth - depthDifference →
  michaelsDepth = fathersRate * michaelsTime →
  fathersDepth = fathersRate * fathersTime :=
by sorry

end NUMINAMATH_CALUDE_fathers_digging_time_l360_36012


namespace NUMINAMATH_CALUDE_tenth_root_unity_sum_l360_36031

theorem tenth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (3 * Real.pi * Complex.I / 5) →
  z / (1 + z^2) + z^3 / (1 + z^6) + z^5 / (1 + z^10) = (z + z^3 - 1/2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_root_unity_sum_l360_36031


namespace NUMINAMATH_CALUDE_initial_birds_count_l360_36025

/-- The number of birds initially sitting in a tree -/
def initial_birds : ℕ := sorry

/-- The number of birds that flew up to join the initial birds -/
def additional_birds : ℕ := 81

/-- The total number of birds after additional birds joined -/
def total_birds : ℕ := 312

/-- Theorem stating that the number of birds initially sitting in the tree is 231 -/
theorem initial_birds_count : initial_birds = 231 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l360_36025


namespace NUMINAMATH_CALUDE_simplify_expression_l360_36058

theorem simplify_expression (x y : ℝ) : (3*x - 5*y) + (4*x + 5*y) = 7*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l360_36058


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l360_36082

-- Define the cube structure
structure Cube where
  faces : Fin 6 → Fin 4

-- Define the probability of a continuous stripe
def probability_continuous_stripe (c : Cube) : ℚ :=
  3 * (1 / 4) ^ 12

-- Theorem statement
theorem continuous_stripe_probability :
  ∀ c : Cube, probability_continuous_stripe c = 3 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l360_36082


namespace NUMINAMATH_CALUDE_wire_cutting_l360_36063

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 140 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (1 + ratio) * shorter_piece = total_length →
  shorter_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l360_36063


namespace NUMINAMATH_CALUDE_equation_solution_l360_36021

theorem equation_solution : ∃! x : ℚ, (3 / 5) * (1 / 9) * x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l360_36021


namespace NUMINAMATH_CALUDE_vector_problems_l360_36099

def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (3, -1)

theorem vector_problems :
  (∃ t : ℝ, (∀ s : ℝ, ‖a + s • b‖ ≥ ‖a + t • b‖) ∧ ‖a + t • b‖ = 7 / Real.sqrt 5 ∧ t = 4/5) ∧
  (∃ t : ℝ, ∃ k : ℝ, a - t • b = k • c ∧ t = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_vector_problems_l360_36099


namespace NUMINAMATH_CALUDE_first_group_size_l360_36074

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The number of acres that can be reaped by M men in 15 days -/
def acres_first_group : ℕ := 120

/-- The number of days it takes M men to reap 120 acres -/
def days_first_group : ℕ := 15

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of acres that can be reaped by 20 men in 30 days -/
def acres_second_group : ℕ := 480

/-- The number of days it takes 20 men to reap 480 acres -/
def days_second_group : ℕ := 30

theorem first_group_size :
  M = 10 :=
sorry

end NUMINAMATH_CALUDE_first_group_size_l360_36074


namespace NUMINAMATH_CALUDE_extreme_value_condition_l360_36087

/-- The function f(x) defined as x^3 + ax^2 + 3x - 9 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

/-- The derivative of f(x) with respect to x -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem extreme_value_condition (a : ℝ) : 
  (∀ x : ℝ, f a x = f_prime a x) → 
  f_prime a (-3) = 0 → 
  a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l360_36087


namespace NUMINAMATH_CALUDE_max_removable_marbles_l360_36002

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCount where
  yellow : Nat
  red : Nat
  black : Nat

/-- The initial number of marbles in the bag -/
def initialMarbles : MarbleCount := ⟨8, 7, 5⟩

/-- The condition that must be satisfied after removing marbles -/
def satisfiesCondition (mc : MarbleCount) : Prop :=
  (mc.yellow ≥ 4 ∧ (mc.red ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.red ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.black ≥ 3)) ∨
  (mc.black ≥ 4 ∧ (mc.yellow ≥ 3 ∨ mc.red ≥ 3))

/-- The maximum number of marbles that can be removed -/
def maxRemovable : Nat := 7

theorem max_removable_marbles :
  (∀ (removed : Nat), removed ≤ maxRemovable →
    ∀ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed →
      satisfiesCondition remaining) ∧
  (∀ (removed : Nat), removed > maxRemovable →
    ∃ (remaining : MarbleCount),
      remaining.yellow + remaining.red + remaining.black = initialMarbles.yellow + initialMarbles.red + initialMarbles.black - removed ∧
      ¬satisfiesCondition remaining) := by
  sorry

end NUMINAMATH_CALUDE_max_removable_marbles_l360_36002


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l360_36016

/-- Given a geometric sequence {a_n} with a_1 = 1, q = 1/2, and a_n = 1/64, prove that the number of terms n is 7. -/
theorem geometric_sequence_term_count (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →
  (∃ n : ℕ, a n = 1/64) →
  ∃ n : ℕ, n = 7 ∧ a n = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l360_36016


namespace NUMINAMATH_CALUDE_two_true_propositions_l360_36007

theorem two_true_propositions :
  let original := ∀ x : ℝ, x > 0 → x^2 > 0
  let converse := ∀ x : ℝ, x^2 > 0 → x > 0
  let negation := ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0
  let contrapositive := ∀ x : ℝ, x^2 ≤ 0 → x ≤ 0
  (original ∧ ¬converse ∧ ¬negation ∧ contrapositive) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l360_36007


namespace NUMINAMATH_CALUDE_simplify_expression_l360_36057

theorem simplify_expression (a b : ℝ) : (a + b)^2 - a*(a + 2*b) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l360_36057


namespace NUMINAMATH_CALUDE_playground_boys_count_l360_36075

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 117) 
  (h2 : girls = 77) : 
  total_children - girls = 40 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l360_36075


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l360_36078

/-- Given two points P and P' that are symmetric with respect to the origin,
    prove that 2a+b = -3 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (2*a + 1 = -1 ∧ 4 = -(3*b - 1)) → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l360_36078


namespace NUMINAMATH_CALUDE_train_journey_time_l360_36027

/-- Proves that if a train moving at 6/7 of its usual speed arrives 10 minutes late, then its usual journey time is 7 hours -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0) 
  (h3 : (6 / 7 * usual_speed) * (usual_time + 10 / 60) = usual_speed * usual_time) : 
  usual_time = 7 := by
  sorry

#check train_journey_time

end NUMINAMATH_CALUDE_train_journey_time_l360_36027


namespace NUMINAMATH_CALUDE_pool_depth_calculation_l360_36085

/-- Calculates the depth of a rectangular pool given its dimensions and draining specifications. -/
theorem pool_depth_calculation (width : ℝ) (length : ℝ) (drain_rate : ℝ) (drain_time : ℝ) (capacity_percentage : ℝ) :
  width = 50 →
  length = 150 →
  drain_rate = 60 →
  drain_time = 1000 →
  capacity_percentage = 0.8 →
  (width * length * (drain_rate * drain_time / capacity_percentage)) / (width * length) = 10 :=
by
  sorry

#check pool_depth_calculation

end NUMINAMATH_CALUDE_pool_depth_calculation_l360_36085


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l360_36086

theorem no_real_solutions_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 1/(a+b) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l360_36086


namespace NUMINAMATH_CALUDE_alex_phone_bill_l360_36040

/-- Calculates the total cost of a cell phone plan based on usage --/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (extra_data_cost : ℚ) (texts_sent : ℕ) (hours_used : ℚ) (data_used : ℚ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_used - 35) 0 * 60
  let minute_charge := extra_minute_cost * extra_minutes
  let extra_data := max (data_used - 2) 0 * 1000
  let data_charge := extra_data_cost * extra_data
  base_cost + text_charge + minute_charge + data_charge

/-- The total cost of Alex's cell phone plan in February is $126.30 --/
theorem alex_phone_bill : 
  calculate_phone_bill 30 0.07 0.12 0.15 150 36.5 2.5 = 126.30 := by
  sorry

end NUMINAMATH_CALUDE_alex_phone_bill_l360_36040


namespace NUMINAMATH_CALUDE_possible_polynomials_g_l360_36096

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 + 12 * x + 4

-- Theorem statement
theorem possible_polynomials_g :
  ∀ g : ℝ → ℝ, satisfies_condition g ↔ (∀ x, g x = 3 * x + 2 ∨ g x = -3 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_possible_polynomials_g_l360_36096


namespace NUMINAMATH_CALUDE_complex_calculation_l360_36041

theorem complex_calculation : ((7 - 3 * Complex.I) - 3 * (2 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -23 + 14 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l360_36041


namespace NUMINAMATH_CALUDE_triangle_area_product_l360_36056

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → (1/2 * (24/p) * (24/q) = 48) → p * q = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l360_36056


namespace NUMINAMATH_CALUDE_toy_production_difference_l360_36019

/-- The difference in daily toy production between actual and planned rates --/
theorem toy_production_difference (total_toys : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_toys = 10080)
  (h2 : planned_days = 14)
  (h3 : actual_days = planned_days - 2) :
  (total_toys / actual_days) - (total_toys / planned_days) = 120 := by
  sorry

end NUMINAMATH_CALUDE_toy_production_difference_l360_36019


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_point_l360_36052

-- Define the plane
variable (P : ℝ × ℝ)

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the foot of the perpendicular Q
def Q (P : ℝ × ℝ) : ℝ × ℝ := (-1, P.2)

-- Define the dot product of 2D vectors
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem trajectory_and_fixed_point (P : ℝ × ℝ) : 
  (dot (P.1 + 1, P.2) (2, -P.2) = dot (P.1 - 1, P.2) (-2, P.2)) →
  (∃ (C : Set (ℝ × ℝ)) (E : ℝ × ℝ), 
    C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧
    E = (1, 0) ∧
    (∀ (k m : ℝ), 
      let M := (m^2, 2*m)
      let N := (-1, -1/m + m)
      (M ∈ C ∧ N.1 = -1) →
      (∃ (r : ℝ), (M.1 - E.1)^2 + (M.2 - E.2)^2 = r^2 ∧
                  (N.1 - E.1)^2 + (N.2 - E.2)^2 = r^2))) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_point_l360_36052


namespace NUMINAMATH_CALUDE_investment_time_is_two_years_l360_36070

/-- Calculates the time period of investment given the principal, interest rates, and interest difference. -/
def calculate_investment_time (principal : ℚ) (rate_high : ℚ) (rate_low : ℚ) (interest_diff : ℚ) : ℚ :=
  interest_diff / (principal * (rate_high - rate_low))

theorem investment_time_is_two_years 
  (principal : ℚ) 
  (rate_high : ℚ) 
  (rate_low : ℚ) 
  (interest_diff : ℚ) :
  principal = 2500 ∧ 
  rate_high = 18 / 100 ∧ 
  rate_low = 12 / 100 ∧ 
  interest_diff = 300 → 
  calculate_investment_time principal rate_high rate_low interest_diff = 2 :=
by
  sorry

#eval calculate_investment_time 2500 (18/100) (12/100) 300

end NUMINAMATH_CALUDE_investment_time_is_two_years_l360_36070


namespace NUMINAMATH_CALUDE_coin_sum_problem_l360_36076

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins. -/
def total_sum_in_rupees (coins_20_paise : ℕ) (coins_25_paise : ℕ) : ℚ :=
  (coins_20_paise * 20 + coins_25_paise * 25) / 100

/-- Proves that given 336 total coins, with 260 coins of 20 paise and the rest being 25 paise coins, 
    the total sum of money is 71 rupees. -/
theorem coin_sum_problem (total_coins : ℕ) (coins_20_paise : ℕ) 
  (h1 : total_coins = 336)
  (h2 : coins_20_paise = 260)
  (h3 : total_coins = coins_20_paise + (total_coins - coins_20_paise)) :
  total_sum_in_rupees coins_20_paise (total_coins - coins_20_paise) = 71 := by
  sorry

#eval total_sum_in_rupees 260 76

end NUMINAMATH_CALUDE_coin_sum_problem_l360_36076


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l360_36050

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l360_36050


namespace NUMINAMATH_CALUDE_dodecagon_hexagon_area_ratio_l360_36013

/-- Given a regular dodecagon with area n and a hexagon ACEGIK formed by
    connecting every second vertex with area m, prove that m/n = √3 - 3/2 -/
theorem dodecagon_hexagon_area_ratio (n m : ℝ) : 
  n > 0 → -- Assuming positive area for the dodecagon
  (∃ (a : ℝ), a > 0 ∧ n = 3 * a^2 * (2 + Real.sqrt 3)) → -- Area formula for dodecagon
  (∃ (s : ℝ), s > 0 ∧ m = (3 * Real.sqrt 3 / 2) * s^2) → -- Area formula for hexagon
  m / n = Real.sqrt 3 - 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_hexagon_area_ratio_l360_36013


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_500_l360_36015

/-- Given a natural number n, returns the set of exponents of 2 in its binary representation -/
def binaryExponents (n : ℕ) : Finset ℕ :=
  sorry

/-- The sum of exponents of 2 in the binary representation of n -/
def sumOfExponents (n : ℕ) : ℕ :=
  (binaryExponents n).sum id

/-- Checks if a set of exponents represents a valid sum of powers of 2 for a given number -/
def isValidRepresentation (n : ℕ) (exponents : Finset ℕ) : Prop :=
  (exponents.sum (fun i => 2^i) = n) ∧ (exponents.card ≥ 3)

theorem least_exponent_sum_for_500 :
  ∀ (exponents : Finset ℕ),
    isValidRepresentation 500 exponents →
    sumOfExponents 500 ≤ (exponents.sum id) :=
by sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_500_l360_36015


namespace NUMINAMATH_CALUDE_books_sold_l360_36033

theorem books_sold (initial_books : Real) (bought_books : Real) (current_books : Real)
  (h1 : initial_books = 4.5)
  (h2 : bought_books = 175.3)
  (h3 : current_books = 62.8) :
  initial_books + bought_books - current_books = 117 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l360_36033


namespace NUMINAMATH_CALUDE_complex_magnitude_l360_36069

theorem complex_magnitude (z : ℂ) : z = 5 / (2 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l360_36069


namespace NUMINAMATH_CALUDE_root_product_equation_l360_36054

theorem root_product_equation (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 3 = 0) →
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 3 = 0) →
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 3 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 4046 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l360_36054


namespace NUMINAMATH_CALUDE_total_crayons_l360_36030

/-- Given that each child has 6 crayons and there are 12 children, prove that the total number of crayons is 72. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) (h1 : crayons_per_child = 6) (h2 : num_children = 12) :
  crayons_per_child * num_children = 72 := by
  sorry

#check total_crayons

end NUMINAMATH_CALUDE_total_crayons_l360_36030


namespace NUMINAMATH_CALUDE_max_handshakers_l360_36093

/-- Given a room with N people, where N > 4, and at least two people have not shaken
    hands with everyone else, the maximum number of people who could have shaken
    hands with everyone else is N-2. -/
theorem max_handshakers (N : ℕ) (h1 : N > 4) (h2 : ∃ (a b : ℕ), a ≠ b ∧ a < N ∧ b < N ∧ 
  (∃ (c : ℕ), c < N ∧ c ≠ a ∧ c ≠ b)) : 
  ∃ (M : ℕ), M = N - 2 ∧ 
  (∀ (k : ℕ), k ≤ N → (∃ (S : Finset ℕ), S.card = k ∧ 
    (∀ (i j : ℕ), i ∈ S → j ∈ S → i ≠ j → (∃ (H : Prop), H)) → k ≤ M)) :=
sorry

end NUMINAMATH_CALUDE_max_handshakers_l360_36093


namespace NUMINAMATH_CALUDE_orange_profit_l360_36083

theorem orange_profit : 
  let buy_rate : ℚ := 10 / 11  -- Cost in r per orange when buying
  let sell_rate : ℚ := 11 / 10  -- Revenue in r per orange when selling
  let num_oranges : ℕ := 110
  let cost : ℚ := buy_rate * num_oranges
  let revenue : ℚ := sell_rate * num_oranges
  let profit : ℚ := revenue - cost
  profit = 21 := by sorry

end NUMINAMATH_CALUDE_orange_profit_l360_36083


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_locus_l360_36001

/-- Given a hyperbola x^2 - y^2/4 = 1, if two perpendicular lines through its center O
    intersect the hyperbola at points A and B, then the locus of the midpoint P of chord AB
    satisfies the equation 3(4x^2 - y^2)^2 = 4(16x^2 + y^2). -/
theorem hyperbola_midpoint_locus (x y : ℝ) :
  (∃ (m n : ℝ),
    -- A and B are on the hyperbola
    4 * (x - m)^2 - (y - n)^2 = 4 ∧
    4 * (x + m)^2 - (y + n)^2 = 4 ∧
    -- OA ⊥ OB
    x^2 + y^2 = m^2 + n^2 ∧
    -- (x, y) is the midpoint of AB
    (x - m, y - n) = (-x - m, -y - n)) →
  3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) := by
sorry


end NUMINAMATH_CALUDE_hyperbola_midpoint_locus_l360_36001


namespace NUMINAMATH_CALUDE_distribute_negative_two_over_parentheses_l360_36018

theorem distribute_negative_two_over_parentheses (x : ℝ) : -2 * (x - 3) = -2 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_two_over_parentheses_l360_36018


namespace NUMINAMATH_CALUDE_nicoles_age_l360_36089

theorem nicoles_age (nicole_age sally_age : ℕ) : 
  nicole_age = 3 * sally_age →
  nicole_age + sally_age + 8 = 40 →
  nicole_age = 24 := by
sorry

end NUMINAMATH_CALUDE_nicoles_age_l360_36089


namespace NUMINAMATH_CALUDE_no_integer_solution_l360_36061

theorem no_integer_solution : ∀ x y : ℤ, x^5 + y^5 + 1 ≠ (x+2)^5 + (y-3)^5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l360_36061


namespace NUMINAMATH_CALUDE_max_value_expression_l360_36046

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a * b + b * c + 2 * c * a ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_expression_l360_36046


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l360_36029

theorem regular_polygon_properties :
  ∀ n : ℕ,
  (n ≥ 3) →
  (n - 2) * 180 = 3 * 360 + 180 →
  n = 9 ∧ ((n - 2) * 180 / n : ℚ) = 140 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l360_36029


namespace NUMINAMATH_CALUDE_tv_show_watch_time_l360_36092

theorem tv_show_watch_time : 
  let regular_seasons : ℕ := 9
  let episodes_per_regular_season : ℕ := 22
  let episodes_in_last_season : ℕ := 26
  let episode_duration : ℚ := 1/2
  
  let total_episodes : ℕ := 
    regular_seasons * episodes_per_regular_season + episodes_in_last_season
  
  let total_watch_time : ℚ := total_episodes * episode_duration
  
  total_watch_time = 112 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_watch_time_l360_36092


namespace NUMINAMATH_CALUDE_largest_geometric_digit_sequence_l360_36037

/-- Checks if the given three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = r * a ∧ c = r * b

/-- Checks if the given number is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- Distinct digits
  is_geometric_sequence a b c ∧  -- Geometric sequence
  b % 2 = 0  -- Tens digit is even

theorem largest_geometric_digit_sequence :
  ∀ n : ℕ, is_valid_solution n → n ≤ 964 :=
sorry

end NUMINAMATH_CALUDE_largest_geometric_digit_sequence_l360_36037


namespace NUMINAMATH_CALUDE_solve_system_for_b_l360_36051

theorem solve_system_for_b :
  ∀ (x y b : ℝ),
  (4 * x + 2 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -15) := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_b_l360_36051


namespace NUMINAMATH_CALUDE_expression_evaluation_l360_36097

theorem expression_evaluation : 4 * (8 - 2)^2 - 6 = 138 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l360_36097


namespace NUMINAMATH_CALUDE_unique_base_representation_l360_36064

theorem unique_base_representation : ∃! n : ℕ+, 
  ∃ A B : ℕ, 
    (0 ≤ A ∧ A < 7) ∧ 
    (0 ≤ B ∧ B < 7) ∧
    (0 ≤ A ∧ A < 5) ∧ 
    (0 ≤ B ∧ B < 5) ∧
    (n : ℕ) = 7 * A + B ∧
    (n : ℕ) = 5 * B + A ∧
    (n : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l360_36064


namespace NUMINAMATH_CALUDE_stream_speed_l360_36065

theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 11)
  (h2 : upstream_speed = 8) :
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 1.5 := by sorry

end NUMINAMATH_CALUDE_stream_speed_l360_36065


namespace NUMINAMATH_CALUDE_amara_clothing_donation_l360_36038

theorem amara_clothing_donation :
  ∀ (initial remaining thrown_away : ℕ) (first_donation : ℕ),
    initial = 100 →
    remaining = 65 →
    thrown_away = 15 →
    initial - remaining = first_donation + 3 * first_donation + thrown_away →
    first_donation = 5 := by
  sorry

end NUMINAMATH_CALUDE_amara_clothing_donation_l360_36038


namespace NUMINAMATH_CALUDE_intersection_plane_sphere_sum_l360_36048

/-- Given a plane x + 2y + 3z = 6 that passes through a point (a, b, c) and intersects
    the coordinate axes at points A, B, C distinct from the origin O,
    prove that a/p + b/q + c/r = 2, where (p, q, r) is the center of the sphere
    passing through A, B, C, and O. -/
theorem intersection_plane_sphere_sum (a b c p q r : ℝ) : 
  (∃ (x y z : ℝ), x + 2*y + 3*z = 6 ∧ 
                   a + 2*b + 3*c = 6 ∧
                   (x = 0 ∨ y = 0 ∨ z = 0) ∧
                   (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) →
  (p^2 + q^2 + r^2 = (p - 6)^2 + q^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + (q - 3)^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + q^2 + (r - 2)^2) →
  a/p + b/q + c/r = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_plane_sphere_sum_l360_36048


namespace NUMINAMATH_CALUDE_remaining_half_speed_l360_36072

-- Define the given conditions
def total_time : ℝ := 11
def total_distance : ℝ := 300
def first_half_speed : ℝ := 30

-- Define the theorem
theorem remaining_half_speed :
  let first_half_distance : ℝ := total_distance / 2
  let first_half_time : ℝ := first_half_distance / first_half_speed
  let remaining_time : ℝ := total_time - first_half_time
  let remaining_distance : ℝ := total_distance / 2
  (remaining_distance / remaining_time) = 25 := by
  sorry


end NUMINAMATH_CALUDE_remaining_half_speed_l360_36072


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_verify_conditions_l360_36003

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 450

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 15) * (sales_volume x)

-- Define the maximum allowed price
def max_price : ℝ := 28

-- Theorem statement
theorem max_profit_at_max_price :
  (∀ x, x ≤ max_price → profit x ≤ profit max_price) ∧
  profit max_price = 2210 := by
  sorry

-- Verify the conditions given in the problem
theorem verify_conditions :
  sales_volume 20 = 250 ∧
  profit 25 = 2000 ∧
  (∃ k b, ∀ x, sales_volume x = k * x + b) := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_verify_conditions_l360_36003


namespace NUMINAMATH_CALUDE_intersection_complement_equals_two_l360_36042

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 3}

theorem intersection_complement_equals_two :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_two_l360_36042


namespace NUMINAMATH_CALUDE_area_of_triangle_pqs_l360_36062

/-- Represents a trapezoid PQRS -/
structure Trapezoid where
  pq : ℝ
  rs : ℝ
  area : ℝ

/-- Theorem: Given a trapezoid PQRS with an area of 20, where RS is three times the length of PQ,
    the area of triangle PQS is 5. -/
theorem area_of_triangle_pqs (t : Trapezoid) 
    (h1 : t.area = 20)
    (h2 : t.rs = 3 * t.pq) : 
    t.area / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_pqs_l360_36062


namespace NUMINAMATH_CALUDE_tampa_bay_bucs_players_l360_36014

/-- The initial number of football players in the Tampa Bay Bucs team. -/
def initial_football_players : ℕ := 13

/-- The initial number of cheerleaders in the Tampa Bay Bucs team. -/
def initial_cheerleaders : ℕ := 16

/-- The number of football players who quit. -/
def quitting_football_players : ℕ := 10

/-- The number of cheerleaders who quit. -/
def quitting_cheerleaders : ℕ := 4

/-- The total number of people left after some quit. -/
def remaining_total : ℕ := 15

theorem tampa_bay_bucs_players :
  initial_football_players = 13 ∧
  (initial_football_players - quitting_football_players) +
  (initial_cheerleaders - quitting_cheerleaders) = remaining_total :=
sorry

end NUMINAMATH_CALUDE_tampa_bay_bucs_players_l360_36014


namespace NUMINAMATH_CALUDE_cubic_inequality_for_negative_numbers_l360_36088

theorem cubic_inequality_for_negative_numbers (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a^3 > b^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_for_negative_numbers_l360_36088


namespace NUMINAMATH_CALUDE_panthers_games_count_l360_36094

theorem panthers_games_count : ∀ (initial_games : ℕ) (initial_wins : ℕ),
  initial_wins = (60 * initial_games) / 100 →
  (initial_wins + 4) * 2 = initial_games + 8 →
  initial_games + 8 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_panthers_games_count_l360_36094


namespace NUMINAMATH_CALUDE_weight_problem_l360_36098

/-- Given three weights a, b, and c, prove that their average weights satisfy certain conditions -/
theorem weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 31 →
  (a + b) / 2 = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_weight_problem_l360_36098


namespace NUMINAMATH_CALUDE_sams_remaining_dimes_l360_36077

theorem sams_remaining_dimes 
  (initial_dimes : ℕ) 
  (borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 8) 
  (h2 : borrowed_dimes = 4) :
  initial_dimes - borrowed_dimes = 4 :=
by sorry

end NUMINAMATH_CALUDE_sams_remaining_dimes_l360_36077


namespace NUMINAMATH_CALUDE_only_5008300_has_no_zeros_l360_36026

/-- Represents a natural number and how it's pronounced in English --/
structure NumberPronunciation where
  value : Nat
  pronunciation : String

/-- Counts the number of times "zero" appears in a string --/
def countZeros (s : String) : Nat :=
  s.split (· = ' ') |>.filter (· = "zero") |>.length

/-- The main theorem stating that only 5008300 has no zeros when pronounced --/
theorem only_5008300_has_no_zeros (numbers : List NumberPronunciation) 
    (h1 : NumberPronunciation.mk 5008300 "five million eight thousand three hundred" ∈ numbers)
    (h2 : NumberPronunciation.mk 500800 "five hundred thousand eight hundred" ∈ numbers)
    (h3 : NumberPronunciation.mk 5080000 "five million eighty thousand" ∈ numbers) :
    ∃! n : NumberPronunciation, n ∈ numbers ∧ countZeros n.pronunciation = 0 :=
  sorry

end NUMINAMATH_CALUDE_only_5008300_has_no_zeros_l360_36026
