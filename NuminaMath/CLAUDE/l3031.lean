import Mathlib

namespace revenue_growth_equation_l3031_303101

/-- Represents the average monthly growth rate of revenue -/
def x : ℝ := sorry

/-- Represents the revenue in January in thousands of dollars -/
def january_revenue : ℝ := 36

/-- Represents the revenue in March in thousands of dollars -/
def march_revenue : ℝ := 48

/-- Theorem stating that the equation representing the revenue growth is 36(1+x)^2 = 48 -/
theorem revenue_growth_equation : 
  january_revenue * (1 + x)^2 = march_revenue := by sorry

end revenue_growth_equation_l3031_303101


namespace match_rectangle_properties_l3031_303142

/-- Represents a rectangle made of matches -/
structure MatchRectangle where
  m : ℕ
  n : ℕ
  h : m > n

/-- Total number of matches used to construct the rectangle -/
def totalMatches (r : MatchRectangle) : ℕ :=
  2 * r.m * r.n + r.m + r.n

/-- Total number of possible rectangles in the figure -/
def totalRectangles (r : MatchRectangle) : ℚ :=
  (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4

/-- Total number of possible squares in the figure -/
def totalSquares (r : MatchRectangle) : ℚ :=
  (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6

theorem match_rectangle_properties (r : MatchRectangle) :
  (totalMatches r = 2 * r.m * r.n + r.m + r.n) ∧
  (totalRectangles r = (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4) ∧
  (totalSquares r = (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6) := by
  sorry

end match_rectangle_properties_l3031_303142


namespace shopping_trip_cost_l3031_303176

/-- Calculates the total cost of a shopping trip including discounts, taxes, and fees -/
def calculate_total_cost (items : List (ℕ × ℚ)) (discount_rate : ℚ) (sales_tax_rate : ℚ) (local_tax_rate : ℚ) (sustainability_fee : ℚ) : ℚ :=
  let total_before_discount := (items.map (λ (q, p) => q * p)).sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_tax_rate := sales_tax_rate + local_tax_rate
  let tax_amount := discounted_total * total_tax_rate
  let total_with_tax := discounted_total + tax_amount
  total_with_tax + sustainability_fee

theorem shopping_trip_cost :
  let items := [(3, 18), (2, 11), (4, 22), (6, 9), (5, 14), (2, 30), (3, 25)]
  let discount_rate := 0.15
  let sales_tax_rate := 0.05
  let local_tax_rate := 0.02
  let sustainability_fee := 5
  calculate_total_cost items discount_rate sales_tax_rate local_tax_rate sustainability_fee = 389.72 := by
  sorry

end shopping_trip_cost_l3031_303176


namespace rectangle_area_l3031_303126

/-- The area of a rectangle with perimeter 200 cm, which can be divided into five identical squares -/
theorem rectangle_area (side : ℝ) (h1 : side > 0) (h2 : 12 * side = 200) : 
  5 * side^2 = 12500 / 9 := by
  sorry

end rectangle_area_l3031_303126


namespace tan_double_angle_special_point_l3031_303188

/-- Given a point P(1, -2) in the plane, and an angle α whose terminal side passes through P,
    prove that tan(2α) = 4/3 -/
theorem tan_double_angle_special_point (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = -2 ∧ Real.tan α = P.2 / P.1) →
  Real.tan (2 * α) = 4/3 := by
sorry

end tan_double_angle_special_point_l3031_303188


namespace cubic_increasing_implies_positive_a_l3031_303125

/-- A cubic function f(x) = ax^3 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The property of f being increasing on all real numbers -/
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: If f(x) = ax^3 + x is increasing on all real numbers, then a > 0 -/
theorem cubic_increasing_implies_positive_a (a : ℝ) :
  increasing_on_reals (f a) → a > 0 :=
by sorry

end cubic_increasing_implies_positive_a_l3031_303125


namespace equilateral_triangle_exists_l3031_303156

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertex : Point
  base1 : Point
  base2 : Point

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d1 := ((t.vertex.x - t.base1.x)^2 + (t.vertex.y - t.base1.y)^2)
  let d2 := ((t.vertex.x - t.base2.x)^2 + (t.vertex.y - t.base2.y)^2)
  let d3 := ((t.base1.x - t.base2.x)^2 + (t.base1.y - t.base2.y)^2)
  d1 = d2 ∧ d2 = d3

-- Theorem statement
theorem equilateral_triangle_exists (P : Point) (l : Line) :
  ∃ (t : EquilateralTriangle), t.vertex = P ∧ 
    isPointOnLine t.base1 l ∧ isPointOnLine t.base2 l ∧ 
    isEquilateral t :=
sorry

end equilateral_triangle_exists_l3031_303156


namespace alipay_growth_rate_l3031_303166

theorem alipay_growth_rate (initial : ℕ) (final : ℕ) (years : ℕ) (rate : ℝ) : 
  initial = 45000 →
  final = 64800 →
  years = 2 →
  (initial : ℝ) * (1 + rate) ^ years = final →
  rate = 0.2 := by
  sorry

end alipay_growth_rate_l3031_303166


namespace quadrilateral_area_inequalities_l3031_303162

/-- Properties of a quadrilateral -/
structure Quadrilateral where
  S : ℝ  -- Area
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  d : ℝ  -- Side length
  e : ℝ  -- Diagonal length
  f : ℝ  -- Diagonal length
  m : ℝ  -- Midpoint segment length
  n : ℝ  -- Midpoint segment length
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hd : 0 < d
  he : 0 < e
  hf : 0 < f
  hm : 0 < m
  hn : 0 < n
  hS : 0 < S

/-- Theorem: Area inequalities for a quadrilateral -/
theorem quadrilateral_area_inequalities (q : Quadrilateral) :
  q.S ≤ (1/4) * (q.e^2 + q.f^2) ∧
  q.S ≤ (1/2) * (q.m^2 + q.n^2) ∧
  q.S ≤ (1/4) * (q.a + q.c) * (q.b + q.d) := by
  sorry

end quadrilateral_area_inequalities_l3031_303162


namespace caiden_roofing_cost_l3031_303146

-- Define the parameters
def total_feet : ℕ := 300
def cost_per_foot : ℚ := 8
def discount_rate : ℚ := 0.1
def shipping_fee : ℚ := 150
def sales_tax_rate : ℚ := 0.05
def free_feet : ℕ := 250

-- Define the calculation steps
def paid_feet : ℕ := total_feet - free_feet
def base_cost : ℚ := paid_feet * cost_per_foot
def discounted_cost : ℚ := base_cost * (1 - discount_rate)
def cost_with_shipping : ℚ := discounted_cost + shipping_fee
def total_cost : ℚ := cost_with_shipping * (1 + sales_tax_rate)

-- Theorem to prove
theorem caiden_roofing_cost :
  total_cost = 535.5 := by sorry

end caiden_roofing_cost_l3031_303146


namespace cycling_distance_conversion_l3031_303194

/-- Converts a list of digits in base 9 to a number in base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

/-- The cycling distance in base 9 -/
def cyclingDistanceBase9 : List Nat := [3, 6, 1, 8]

theorem cycling_distance_conversion :
  base9ToBase10 cyclingDistanceBase9 = 2690 := by
  sorry

end cycling_distance_conversion_l3031_303194


namespace point_on_h_graph_coordinate_sum_l3031_303186

theorem point_on_h_graph_coordinate_sum : 
  ∀ (g h : ℝ → ℝ),
  g 4 = -5 →
  (∀ x, h x = (g x)^2 + 3) →
  4 + h 4 = 32 := by
sorry

end point_on_h_graph_coordinate_sum_l3031_303186


namespace rational_solutions_quadratic_l3031_303104

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
sorry

end rational_solutions_quadratic_l3031_303104


namespace table_price_is_56_l3031_303118

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end table_price_is_56_l3031_303118


namespace total_money_divided_l3031_303152

/-- Proves that the total amount of money divided is 1600, given the specified conditions. -/
theorem total_money_divided (x : ℝ) (T : ℝ) : 
  x + (T - x) = T →  -- The money is divided into two parts
  0.06 * x + 0.05 * (T - x) = 85 →  -- The whole annual interest from both parts is 85
  T - x = 1100 →  -- 1100 was lent at approximately 5%
  T = 1600 := by
  sorry

end total_money_divided_l3031_303152


namespace combination_not_equal_permutation_div_n_factorial_l3031_303116

/-- The number of combinations of n things taken m at a time -/
def C (n m : ℕ) : ℕ := sorry

/-- The number of permutations of n things taken m at a time -/
def A (n m : ℕ) : ℕ := sorry

theorem combination_not_equal_permutation_div_n_factorial (n m : ℕ) :
  C n m ≠ A n m / n! :=
sorry

end combination_not_equal_permutation_div_n_factorial_l3031_303116


namespace inequality_solution_set_l3031_303173

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} :=
sorry

end inequality_solution_set_l3031_303173


namespace f_of_g_10_l3031_303110

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 6
def f (x : ℝ) : ℝ := 4 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 96 := by
  sorry

end f_of_g_10_l3031_303110


namespace correct_quotient_proof_l3031_303117

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 49) : N / 21 = 28 := by
  sorry

end correct_quotient_proof_l3031_303117


namespace right_triangle_semicircles_l3031_303180

theorem right_triangle_semicircles (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at Q
  (1/2) * Real.pi * (pq/2)^2 = 50 * Real.pi →  -- area of semicircle on PQ
  Real.pi * (pr/2) = 18 * Real.pi →  -- circumference of semicircle on PR
  qr/2 = 20.6 ∧  -- radius of semicircle on QR
  ∃ (C : ℝ × ℝ), (C.1 - P.1)^2 + (C.2 - P.2)^2 = (pr/2)^2 ∧
                 (C.1 - R.1)^2 + (C.2 - R.2)^2 = (pr/2)^2 ∧
                 (C.1 - Q.1) * (R.1 - P.1) + (C.2 - Q.2) * (R.2 - P.2) = 0  -- 90° angle at Q in semicircle on PR
  := by sorry

end right_triangle_semicircles_l3031_303180


namespace volunteer_allocation_schemes_l3031_303148

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * (k.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end volunteer_allocation_schemes_l3031_303148


namespace dice_probability_l3031_303192

def probability_less_than_6 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose number_of_dice target_count : ℚ) * probability_less_than_6^number_of_dice = 5 / 16 := by
  sorry

end dice_probability_l3031_303192


namespace min_value_implies_a_bound_l3031_303155

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a*x^2 - x + 2

/-- Theorem stating that if the minimum value of f(x) is -1, then a ≥ 1/12 --/
theorem min_value_implies_a_bound (a : ℝ) :
  (∀ x, f a x ≥ -1) ∧ (∃ x, f a x = -1) → a ≥ 1/12 :=
by sorry

end min_value_implies_a_bound_l3031_303155


namespace boat_current_rate_l3031_303113

/-- Proves that the rate of the current is 4 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 12 →
  downstream_distance = 4.8 →
  downstream_time = 18 / 60 →
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by
  sorry


end boat_current_rate_l3031_303113


namespace seth_yogurt_purchase_l3031_303164

theorem seth_yogurt_purchase (ice_cream_cartons : ℕ) (ice_cream_cost : ℕ) (yogurt_cost : ℕ) (difference : ℕ) :
  ice_cream_cartons = 20 →
  ice_cream_cost = 6 →
  yogurt_cost = 1 →
  ice_cream_cartons * ice_cream_cost = difference + yogurt_cost * (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost →
  (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost = 2 :=
by sorry

end seth_yogurt_purchase_l3031_303164


namespace computer_preference_ratio_l3031_303197

theorem computer_preference_ratio : 
  ∀ (total mac no_pref equal : ℕ),
    total = 210 →
    mac = 60 →
    no_pref = 90 →
    equal = total - (mac + no_pref) →
    equal = mac →
    (equal : ℚ) / mac = 1 := by
  sorry

end computer_preference_ratio_l3031_303197


namespace cube_face_area_l3031_303169

/-- Given a cube with surface area 36 square centimeters, 
    prove that the area of one face is 6 square centimeters. -/
theorem cube_face_area (surface_area : ℝ) (h : surface_area = 36) :
  surface_area / 6 = 6 := by
  sorry

end cube_face_area_l3031_303169


namespace quadratic_roots_condition_l3031_303182

theorem quadratic_roots_condition (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
sorry

end quadratic_roots_condition_l3031_303182


namespace triangle_angle_proof_l3031_303145

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  B = π / 4 →
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  A + B + C = π →
  A = π / 3 :=
by sorry

end triangle_angle_proof_l3031_303145


namespace events_A_D_independent_l3031_303175

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | Odd ω.1}
def D : Set Ω := {ω | ω.1 + ω.2 = 7}

-- State the theorem
theorem events_A_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end events_A_D_independent_l3031_303175


namespace divisibility_of_fourth_power_minus_one_l3031_303137

theorem divisibility_of_fourth_power_minus_one (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^4 - 1)) := by
  sorry

end divisibility_of_fourth_power_minus_one_l3031_303137


namespace max_dot_product_in_trapezoid_l3031_303127

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is inside or on the boundary of a trapezoid -/
def isInTrapezoid (t : Trapezoid) (p : Point) : Prop := sorry

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Main theorem -/
theorem max_dot_product_in_trapezoid (t : Trapezoid) :
  t.A = Point.mk 0 0 →
  t.B = Point.mk 3 0 →
  t.C = Point.mk 2 2 →
  t.D = Point.mk 0 2 →
  let N := Point.mk 1 2
  ∀ M : Point, isInTrapezoid t M →
  dotProduct (Point.mk (M.x - t.A.x) (M.y - t.A.y)) (Point.mk (N.x - t.A.x) (N.y - t.A.y)) ≤ 6 := by
  sorry

end max_dot_product_in_trapezoid_l3031_303127


namespace unique_satisfying_polynomial_l3031_303111

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {r : ℝ | p.a * r^2 + p.b * r + p.c = 0}

/-- The coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {p.a, p.b, p.c}

/-- Predicate for a polynomial satisfying the problem conditions -/
def satisfies_conditions (p : QuadraticPolynomial) : Prop :=
  roots p = coefficients p ∧
  (p.a < 0 ∨ p.b < 0 ∨ p.c < 0)

/-- The main theorem stating that exactly one quadratic polynomial satisfies the conditions -/
theorem unique_satisfying_polynomial :
  ∃! p : QuadraticPolynomial, satisfies_conditions p :=
sorry

end unique_satisfying_polynomial_l3031_303111


namespace usual_time_to_catch_bus_l3031_303139

theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time > 0 → usual_speed > 0 →
  (4/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 20 := by
  sorry

end usual_time_to_catch_bus_l3031_303139


namespace tangent_line_equation_l3031_303147

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (-2, -2)

-- Define a tangent point
def TangentPoint (x y : ℝ) : Prop :=
  Circle x y ∧ ((x + 2) * x + (y + 2) * y = 0)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    TangentPoint x₁ y₁ → TangentPoint x₂ y₂ →
    (2 * x₁ + 2 * y₁ + 1 = 0) ∧ (2 * x₂ + 2 * y₂ + 1 = 0) :=
by sorry

end tangent_line_equation_l3031_303147


namespace oplus_four_two_l3031_303189

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem oplus_four_two : oplus 4 2 = 26 := by
  sorry

end oplus_four_two_l3031_303189


namespace question_selection_ways_eq_13838400_l3031_303171

/-- The number of ways to select questions from a question paper with three parts -/
def questionSelectionWays : ℕ :=
  let partA := Nat.choose 12 8
  let partB := Nat.choose 10 5
  let partC := Nat.choose 8 3
  partA * partB * partC

/-- Theorem stating the correct number of ways to select questions -/
theorem question_selection_ways_eq_13838400 : questionSelectionWays = 13838400 := by
  sorry

end question_selection_ways_eq_13838400_l3031_303171


namespace valid_outfit_choices_l3031_303196

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where pants and hat are the same color -/
def matching_pants_hat_outfits : ℕ := num_colors * num_shirts

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_outfits - matching_pants_hat_outfits

theorem valid_outfit_choices :
  valid_outfits = 240 :=
by sorry

end valid_outfit_choices_l3031_303196


namespace cow_husk_consumption_l3031_303119

/-- Given that 45 cows eat 45 bags of husk in 45 days, 
    prove that 1 cow will eat 1 bag of husk in 45 days -/
theorem cow_husk_consumption 
  (cows : ℕ) (bags : ℕ) (days : ℕ) 
  (h : cows = 45 ∧ bags = 45 ∧ days = 45) : 
  1 * bags / cows = days :=
sorry

end cow_husk_consumption_l3031_303119


namespace nancy_hourly_wage_l3031_303144

/-- Calculates the hourly wage needed to cover remaining expenses for one semester --/
def hourly_wage_needed (tuition housing meal_plan textbooks merit_scholarship need_scholarship work_hours : ℕ) : ℚ :=
  let total_cost := tuition + housing + meal_plan + textbooks
  let parents_contribution := tuition / 2
  let student_loan := 2 * merit_scholarship
  let total_support := parents_contribution + merit_scholarship + need_scholarship + student_loan
  let remaining_expenses := total_cost - total_support
  (remaining_expenses : ℚ) / work_hours

/-- Theorem stating that Nancy needs to earn $49 per hour --/
theorem nancy_hourly_wage :
  hourly_wage_needed 22000 6000 2500 800 3000 1500 200 = 49 := by
  sorry

end nancy_hourly_wage_l3031_303144


namespace S_periodic_l3031_303108

def S (x y z : ℤ) : ℤ × ℤ × ℤ := (x*y - x*z, y*z - y*x, z*x - z*y)

def S_power (n : ℕ) (a b c : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => S (S_power n a b c).1 (S_power n a b c).2.1 (S_power n a b c).2.2

def congruent_triple (u v : ℤ × ℤ × ℤ) (m : ℤ) : Prop :=
  u.1 % m = v.1 % m ∧ u.2.1 % m = v.2.1 % m ∧ u.2.2 % m = v.2.2 % m

theorem S_periodic (a b c : ℤ) (h : a * b * c > 1) :
  ∃ (n₀ k : ℕ), 0 < k ∧ k ≤ a * b * c ∧
  ∀ n ≥ n₀, congruent_triple (S_power (n + k) a b c) (S_power n a b c) (a * b * c) :=
sorry

end S_periodic_l3031_303108


namespace total_morning_afternoon_emails_l3031_303179

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- Theorem: The total number of emails Jack received in the morning and afternoon is 13 -/
theorem total_morning_afternoon_emails : morning_emails + afternoon_emails = 13 := by
  sorry

end total_morning_afternoon_emails_l3031_303179


namespace geometric_sequence_product_l3031_303177

/-- A geometric sequence with five terms where the first term is -1 and the last term is -2 -/
def GeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -2 = z * r

/-- The product of the middle three terms of the geometric sequence equals ±2√2 -/
theorem geometric_sequence_product (x y z : ℝ) :
  GeometricSequence x y z → x * y * z = 2 * Real.sqrt 2 ∨ x * y * z = -2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_product_l3031_303177


namespace robot_path_area_l3031_303114

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a closed path on a 2D plane -/
structure ClosedPath where
  vertices : List Point

/-- Function to calculate the area of a closed path -/
noncomputable def areaOfClosedPath (path : ClosedPath) : ℝ :=
  sorry

/-- The specific closed path followed by the robot -/
def robotPath : ClosedPath :=
  sorry

/-- Theorem stating that the area of the robot's path is 13√3/4 -/
theorem robot_path_area :
  areaOfClosedPath robotPath = (13 * Real.sqrt 3) / 4 := by
  sorry

end robot_path_area_l3031_303114


namespace population_change_factors_l3031_303132

-- Define the factors that can affect population change
inductive PopulationFactor
  | NaturalGrowth
  | Migration
  | Mortality
  | BirthRate

-- Define a function that determines if a factor affects population change
def affectsPopulationChange (factor : PopulationFactor) : Prop :=
  match factor with
  | PopulationFactor.NaturalGrowth => true
  | PopulationFactor.Migration => true
  | _ => false

-- Theorem stating that population change is determined by natural growth and migration
theorem population_change_factors :
  ∀ (factor : PopulationFactor),
    affectsPopulationChange factor ↔
      (factor = PopulationFactor.NaturalGrowth ∨ factor = PopulationFactor.Migration) :=
by
  sorry


end population_change_factors_l3031_303132


namespace decimal_multiplication_division_l3031_303115

theorem decimal_multiplication_division : (0.5 * 0.6) / 0.2 = 1.5 := by
  sorry

end decimal_multiplication_division_l3031_303115


namespace M_inter_compl_N_l3031_303170

/-- The set M defined by the square root function -/
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}

/-- The set N defined by a quadratic inequality -/
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

/-- The theorem stating the intersection of M and the complement of N -/
theorem M_inter_compl_N : M ∩ (Set.univ \ N) = {x | 0 ≤ x ∧ x < 2 ∨ x > 4} := by sorry

end M_inter_compl_N_l3031_303170


namespace problem_solution_l3031_303174

theorem problem_solution : 
  (27 / 8) ^ (-1/3 : ℝ) + Real.log 3 / Real.log 2 * Real.log 4 / Real.log 3 + 
  Real.log 2 / Real.log 10 + Real.log 50 / Real.log 10 = 14/3 := by
  sorry

end problem_solution_l3031_303174


namespace element_value_l3031_303198

theorem element_value (a : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) → 
  a ∉ ({0, 1, 2} : Set ℕ) → 
  a = 3 :=
by
  sorry

end element_value_l3031_303198


namespace arithmetic_evaluation_l3031_303191

theorem arithmetic_evaluation : 6 / 3 - 2 - 8 + 2 * 8 = 8 := by sorry

end arithmetic_evaluation_l3031_303191


namespace liar_count_l3031_303184

/-- Represents a candidate's statement about the number of lies told before their turn. -/
structure CandidateStatement where
  position : Nat
  claimed_lies : Nat
  is_truthful : Bool

/-- The debate scenario with 12 candidates. -/
def debate_scenario (statements : Vector CandidateStatement 12) : Prop :=
  (∀ i : Fin 12, (statements.get i).position = i.val + 1) ∧
  (∀ i : Fin 12, (statements.get i).claimed_lies = i.val + 1) ∧
  (∃ i : Fin 12, (statements.get i).is_truthful)

/-- The theorem to be proved. -/
theorem liar_count (statements : Vector CandidateStatement 12) 
  (h : debate_scenario statements) : 
  (statements.toList.filter (fun s => !s.is_truthful)).length = 11 := by
  sorry


end liar_count_l3031_303184


namespace sqrt_equation_solution_l3031_303102

theorem sqrt_equation_solution (x : ℚ) :
  Real.sqrt (2 - 5 * x) = 8 → x = -62 / 5 := by
  sorry

end sqrt_equation_solution_l3031_303102


namespace sum_of_squares_constant_l3031_303138

/-- A regular polygon with n vertices and circumradius r -/
structure RegularPolygon where
  n : ℕ
  r : ℝ
  h_n : n ≥ 3
  h_r : r > 0

/-- The sum of squares of distances from a point on the circumcircle to all vertices -/
def sum_of_squares (poly : RegularPolygon) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating that the sum of squares is constant for any point on the circumcircle -/
theorem sum_of_squares_constant (poly : RegularPolygon) :
  ∀ P : ℝ × ℝ, (P.1 - poly.r)^2 + P.2^2 = poly.r^2 →
    sum_of_squares poly P = 2 * poly.n * poly.r^2 :=
  sorry

end sum_of_squares_constant_l3031_303138


namespace time_to_hear_second_blast_l3031_303158

/-- The time taken for a man to hear a second blast, given specific conditions -/
theorem time_to_hear_second_blast 
  (speed_of_sound : ℝ) 
  (time_between_blasts : ℝ) 
  (distance_at_second_blast : ℝ) 
  (h1 : speed_of_sound = 330)
  (h2 : time_between_blasts = 30 * 60)
  (h3 : distance_at_second_blast = 4950) :
  speed_of_sound * (time_between_blasts + distance_at_second_blast / speed_of_sound) = 1815 * speed_of_sound :=
by sorry

end time_to_hear_second_blast_l3031_303158


namespace jeremy_payment_l3031_303159

theorem jeremy_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 5 / 2) :
  rate * rooms = 65 / 6 := by sorry

end jeremy_payment_l3031_303159


namespace prime_pair_from_quadratic_roots_l3031_303128

theorem prime_pair_from_quadratic_roots (p q : ℕ) (hp : p.Prime) (hq : q.Prime) 
  (x₁ x₂ : ℤ) (h_sum : x₁ + x₂ = -p) (h_prod : x₁ * x₂ = q) : p = 3 ∧ q = 2 := by
  sorry

end prime_pair_from_quadratic_roots_l3031_303128


namespace next_simultaneous_occurrence_l3031_303154

def town_hall_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def university_bell_interval : ℕ := 30

def simultaneous_occurrence (t : ℕ) : Prop :=
  t % town_hall_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % university_bell_interval = 0

theorem next_simultaneous_occurrence :
  ∃ t : ℕ, t > 0 ∧ t ≤ 360 ∧ simultaneous_occurrence t ∧
  ∀ s : ℕ, 0 < s ∧ s < t → ¬simultaneous_occurrence s :=
sorry

end next_simultaneous_occurrence_l3031_303154


namespace sock_time_correct_l3031_303160

/-- Represents the time (in hours) to knit each sock -/
def sock_time : ℝ := 1.5

/-- Represents the number of grandchildren -/
def num_grandchildren : ℕ := 3

/-- Time (in hours) to knit a hat -/
def hat_time : ℝ := 2

/-- Time (in hours) to knit a scarf -/
def scarf_time : ℝ := 3

/-- Time (in hours) to knit a sweater -/
def sweater_time : ℝ := 6

/-- Time (in hours) to knit each mitten -/
def mitten_time : ℝ := 1

/-- Total time (in hours) to knit all outfits -/
def total_time : ℝ := 48

/-- Theorem stating that the calculated sock_time satisfies the given conditions -/
theorem sock_time_correct : 
  num_grandchildren * (hat_time + scarf_time + sweater_time + 2 * mitten_time + 2 * sock_time) = total_time := by
  sorry

end sock_time_correct_l3031_303160


namespace spinner_probabilities_l3031_303181

theorem spinner_probabilities : ∃ (x y : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + x + y = 1 ∧
  x + y = 5 / 12 ∧
  x = 1 / 4 ∧
  y = 1 / 6 :=
by sorry

end spinner_probabilities_l3031_303181


namespace fraction_sum_and_divide_l3031_303107

theorem fraction_sum_and_divide : (3/20 + 5/200 + 7/2000) / 2 = 0.08925 := by
  sorry

end fraction_sum_and_divide_l3031_303107


namespace original_speed_before_training_l3031_303151

/-- Represents the skipping speed of a person -/
structure SkippingSpeed :=
  (skips : ℕ)
  (minutes : ℕ)

/-- Calculates the skips per minute -/
def skipsPerMinute (speed : SkippingSpeed) : ℚ :=
  speed.skips / speed.minutes

theorem original_speed_before_training
  (after_training : SkippingSpeed)
  (h_doubles : after_training.skips = 700 ∧ after_training.minutes = 5) :
  let before_training := SkippingSpeed.mk (after_training.skips / 2) after_training.minutes
  skipsPerMinute before_training = 70 := by
sorry

end original_speed_before_training_l3031_303151


namespace largest_consecutive_even_integer_l3031_303187

theorem largest_consecutive_even_integer (n : ℕ) : 
  n % 2 = 0 ∧ 
  n * (n + 2) * (n + 4) * (n + 6) = 6720 →
  n + 6 = 14 :=
by
  sorry

end largest_consecutive_even_integer_l3031_303187


namespace sector_radius_l3031_303103

theorem sector_radius (α : Real) (S : Real) (r : Real) : 
  α = 3/4 * Real.pi → 
  S = 3/2 * Real.pi → 
  S = 1/2 * r^2 * α → 
  r = 2 := by
  sorry

end sector_radius_l3031_303103


namespace solution_set_for_a_eq_2_range_of_a_l3031_303161

-- Define the function f
def f (a x : ℝ) : ℝ := |a - 3*x| - |2 + x|

-- Theorem for part (1)
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 3} = {x : ℝ | -3/4 ≤ x ∧ x ≤ 7/2} := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x ≥ 1 ∧ ∃ y, a + 2*|2 + y| = 0} = {a : ℝ | a ≥ -5/2} := by sorry

end solution_set_for_a_eq_2_range_of_a_l3031_303161


namespace count_male_students_l3031_303183

theorem count_male_students (total : ℕ) (girls : ℕ) (h1 : total = 13) (h2 : girls = 6) :
  total - girls = 7 := by
  sorry

end count_male_students_l3031_303183


namespace exists_x_iff_b_gt_min_sum_l3031_303112

/-- The minimum value of the sum of absolute differences -/
def min_sum : ℝ := 4

/-- The function representing the sum of absolute differences -/
def f (x : ℝ) : ℝ := |x - 5| + |x - 3| + |x - 2|

/-- Theorem stating the condition for the existence of x satisfying the inequality -/
theorem exists_x_iff_b_gt_min_sum (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, f x < b) ↔ b > min_sum :=
sorry

end exists_x_iff_b_gt_min_sum_l3031_303112


namespace min_point_of_translated_abs_value_l3031_303168

-- Define the function representing the translated graph
def f (x : ℝ) : ℝ := |x - 3| - 1

-- Theorem stating that the minimum point of the graph is (3, -1)
theorem min_point_of_translated_abs_value :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = 0 :=
sorry

end min_point_of_translated_abs_value_l3031_303168


namespace books_read_during_trip_l3031_303185

-- Define the travel distance in miles
def travel_distance : ℕ := 6760

-- Define the reading rate in miles per book
def miles_per_book : ℕ := 450

-- Theorem to prove
theorem books_read_during_trip : travel_distance / miles_per_book = 15 := by
  sorry

end books_read_during_trip_l3031_303185


namespace range_of_m_for_inequality_l3031_303121

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(-x^2 - x) > (1/2 : ℝ)^(2*x^2 - m*x + m + 4)) ↔ 
  -3 < m ∧ m < 5 := by
sorry

end range_of_m_for_inequality_l3031_303121


namespace friendship_divisibility_criterion_l3031_303100

/-- Represents a friendship relation between students -/
def FriendshipRelation (n : ℕ) := Fin n → Fin n → Prop

/-- The friendship relation is symmetric -/
def symmetric {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i j, r i j ↔ r j i

/-- The friendship relation is irreflexive -/
def irreflexive {n : ℕ} (r : FriendshipRelation n) :=
  ∀ i, ¬(r i i)

/-- Theorem: For any finite set of students with a friendship relation,
    there exists a positive integer N and an assignment of integers to students
    such that two students are friends if and only if N divides the product of their assigned integers -/
theorem friendship_divisibility_criterion
  {n : ℕ} (r : FriendshipRelation n) (h_sym : symmetric r) (h_irr : irreflexive r) :
  ∃ (N : ℕ) (N_pos : 0 < N) (a : Fin n → ℤ),
    ∀ i j, r i j ↔ (N : ℤ) ∣ (a i * a j) :=
sorry

end friendship_divisibility_criterion_l3031_303100


namespace jason_current_is_sum_jason_has_63_dollars_l3031_303131

/-- Represents the money situation for Fred and Jason --/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_current : ℕ
  jason_earned : ℕ

/-- Calculates Jason's current amount of money --/
def jason_current (s : MoneySituation) : ℕ := s.jason_initial + s.jason_earned

/-- Theorem stating that Jason's current amount is the sum of his initial and earned amounts --/
theorem jason_current_is_sum (s : MoneySituation) :
  jason_current s = s.jason_initial + s.jason_earned := by sorry

/-- The specific money situation from the problem --/
def problem_situation : MoneySituation :=
  { fred_initial := 49
  , jason_initial := 3
  , fred_current := 112
  , jason_earned := 60 }

/-- Theorem proving that Jason now has 63 dollars --/
theorem jason_has_63_dollars :
  jason_current problem_situation = 63 := by sorry

end jason_current_is_sum_jason_has_63_dollars_l3031_303131


namespace sum_a_b_equals_one_l3031_303136

theorem sum_a_b_equals_one (a b : ℝ) (h : Real.sqrt (a - b - 3) + |2 * a - 4| = 0) : a + b = 1 := by
  sorry

end sum_a_b_equals_one_l3031_303136


namespace value_of_S_l3031_303106

/-- Given S = 6 × 10000 + 5 × 1000 + 4 × 10 + 3 × 1, prove that S = 65043 -/
theorem value_of_S : 
  let S := 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1
  S = 65043 := by
  sorry

end value_of_S_l3031_303106


namespace paths_through_B_l3031_303178

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The position of point A -/
def point_A : ℕ × ℕ := (0, 0)

/-- The position of point B relative to A -/
def A_to_B : ℕ × ℕ := (4, 2)

/-- The position of point C relative to B -/
def B_to_C : ℕ × ℕ := (3, 2)

/-- The total number of steps from A to C -/
def total_steps : ℕ := A_to_B.1 + A_to_B.2 + B_to_C.1 + B_to_C.2

theorem paths_through_B : 
  grid_paths A_to_B.1 A_to_B.2 * grid_paths B_to_C.1 B_to_C.2 = 150 ∧ 
  total_steps = 11 := by
  sorry

end paths_through_B_l3031_303178


namespace paiges_math_problems_l3031_303190

theorem paiges_math_problems (total_problems math_problems science_problems finished_problems left_problems : ℕ) :
  science_problems = 12 →
  finished_problems = 44 →
  left_problems = 11 →
  total_problems = math_problems + science_problems →
  total_problems = finished_problems + left_problems →
  math_problems = 43 := by
sorry

end paiges_math_problems_l3031_303190


namespace square_division_and_triangle_area_l3031_303135

/-- The area of the remaining part after cutting off squares from a unit square -/
def S (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / (2 * n)

/-- The area of triangle ABP formed by the intersection of y = (1/2)x and y = 1/(2x) -/
def triangle_area (n : ℕ) : ℚ :=
  1 / 2 + (1 / 2) * (1 / 2)

theorem square_division_and_triangle_area (n : ℕ) (h : n ≥ 2) :
  S n = (n + 1 : ℚ) / (2 * n) ∧ 
  triangle_area n = 1 ∧
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |triangle_area n - 1| < ε :=
sorry

end square_division_and_triangle_area_l3031_303135


namespace smallest_with_24_factors_div_by_18_and_30_is_360_l3031_303195

/-- The smallest integer with 24 positive factors that is divisible by both 18 and 30 -/
def smallest_with_24_factors_div_by_18_and_30 : ℕ := 360

/-- Proposition: The smallest integer with 24 positive factors that is divisible by both 18 and 30 is 360 -/
theorem smallest_with_24_factors_div_by_18_and_30_is_360 :
  ∀ y : ℕ, 
    (Finset.card (Nat.divisors y) = 24) → 
    (18 ∣ y) → 
    (30 ∣ y) → 
    y ≥ smallest_with_24_factors_div_by_18_and_30 :=
sorry

end smallest_with_24_factors_div_by_18_and_30_is_360_l3031_303195


namespace cantaloupe_total_l3031_303163

theorem cantaloupe_total (fred_cantaloupes tim_cantaloupes : ℕ) 
  (h1 : fred_cantaloupes = 38) 
  (h2 : tim_cantaloupes = 44) : 
  fred_cantaloupes + tim_cantaloupes = 82 := by
sorry

end cantaloupe_total_l3031_303163


namespace divisibility_by_133_l3031_303130

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end divisibility_by_133_l3031_303130


namespace cube_root_inequality_l3031_303172

theorem cube_root_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a + 1) * (b + 1) * (c + 1)) (1/3) ≥ Real.rpow (a * b * c) (1/3) + 1 := by
  sorry

end cube_root_inequality_l3031_303172


namespace sum_of_three_numbers_l3031_303165

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 48)
  (sum_ca : c + a = 60) :
  a + b + c = 71.5 := by
sorry

end sum_of_three_numbers_l3031_303165


namespace investment_problem_l3031_303141

theorem investment_problem (amount_at_5_percent : ℝ) (total_with_interest : ℝ) :
  amount_at_5_percent = 600 →
  total_with_interest = 1054 →
  ∃ (total_investment : ℝ),
    total_investment = 1034 ∧
    amount_at_5_percent + amount_at_5_percent * 0.05 +
    (total_investment - amount_at_5_percent) +
    (total_investment - amount_at_5_percent) * 0.06 = total_with_interest :=
by sorry

end investment_problem_l3031_303141


namespace class_test_percentages_l3031_303193

theorem class_test_percentages (total : ℝ) (first : ℝ) (second : ℝ) (both : ℝ) 
  (h_total : total = 100)
  (h_first : first = 75)
  (h_second : second = 30)
  (h_both : both = 25) :
  total - (first + second - both) = 20 := by
  sorry

end class_test_percentages_l3031_303193


namespace conic_is_hyperbola_l3031_303120

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 - 8 * x + 36 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end conic_is_hyperbola_l3031_303120


namespace largest_divisible_n_l3031_303149

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 8) ∣ (m^3 + 64))) ∧ 
  ((n + 8) ∣ (n^3 + 64)) ∧ 
  n = 440 := by
  sorry

end largest_divisible_n_l3031_303149


namespace triangle_problem_l3031_303133

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if √3 * sin(C) + c * cos(A) = a + b, c = 2, and the area is √3,
    then C = π/3 and the perimeter is 6. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * Real.sin C + c * Real.cos A = a + b →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  C = π/3 ∧ a + b + c = 6 := by sorry

end triangle_problem_l3031_303133


namespace triangle_division_theorem_l3031_303129

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define a point inside a triangle
def PointInside (T : Triangle) (P : Point) : Prop :=
  -- Placeholder for the condition that P is inside triangle T
  sorry

-- Define a point on a side of a triangle
def PointOnSide (T : Triangle) (Q : Point) : Prop :=
  -- Placeholder for the condition that Q is on a side of triangle T
  sorry

-- Define the property of not sharing an entire side
def NotShareEntireSide (T1 T2 : Triangle) : Prop :=
  -- Placeholder for the condition that T1 and T2 do not share an entire side
  sorry

theorem triangle_division_theorem (T : Triangle) :
  ∃ (P Q : Point) (T1 T2 T3 T4 : Triangle),
    PointInside T P ∧
    PointOnSide T Q ∧
    NotShareEntireSide T1 T2 ∧
    NotShareEntireSide T1 T3 ∧
    NotShareEntireSide T1 T4 ∧
    NotShareEntireSide T2 T3 ∧
    NotShareEntireSide T2 T4 ∧
    NotShareEntireSide T3 T4 :=
  sorry

end triangle_division_theorem_l3031_303129


namespace power_seven_twelve_mod_hundred_l3031_303105

theorem power_seven_twelve_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end power_seven_twelve_mod_hundred_l3031_303105


namespace quadratic_two_roots_l3031_303134

theorem quadratic_two_roots (c : ℝ) (h : c < 4) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0 :=
by sorry

end quadratic_two_roots_l3031_303134


namespace polyhedron_volume_theorem_l3031_303123

/-- A polyhedron consisting of a prism and two pyramids -/
structure Polyhedron where
  prism_volume : ℝ
  pyramid_volume : ℝ
  prism_volume_eq : prism_volume = Real.sqrt 2 - 1
  pyramid_volume_eq : pyramid_volume = 1 / 6

/-- The total volume of the polyhedron -/
def total_volume (p : Polyhedron) : ℝ :=
  p.prism_volume + 2 * p.pyramid_volume

theorem polyhedron_volume_theorem (p : Polyhedron) :
  total_volume p = Real.sqrt 2 - 2 / 3 := by
  sorry

end polyhedron_volume_theorem_l3031_303123


namespace michelle_needs_one_more_rack_l3031_303140

/-- Represents the pasta making scenario for Michelle -/
structure PastaMaking where
  flour_per_pound : ℕ  -- cups of flour needed per pound of pasta
  pounds_per_rack : ℕ  -- pounds of pasta that can fit on one rack
  owned_racks : ℕ     -- number of racks Michelle currently owns
  flour_bags : ℕ      -- number of flour bags
  cups_per_bag : ℕ    -- cups of flour in each bag

/-- Calculates the number of additional racks Michelle needs -/
def additional_racks_needed (pm : PastaMaking) : ℕ :=
  let total_flour := pm.flour_bags * pm.cups_per_bag
  let total_pounds := total_flour / pm.flour_per_pound
  let total_racks_needed := (total_pounds + pm.pounds_per_rack - 1) / pm.pounds_per_rack
  (total_racks_needed - pm.owned_racks).max 0

/-- Theorem stating that Michelle needs one more rack -/
theorem michelle_needs_one_more_rack :
  let pm : PastaMaking := {
    flour_per_pound := 2,
    pounds_per_rack := 3,
    owned_racks := 3,
    flour_bags := 3,
    cups_per_bag := 8
  }
  additional_racks_needed pm = 1 := by sorry

end michelle_needs_one_more_rack_l3031_303140


namespace min_value_sum_reciprocals_l3031_303109

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ + 2*b₀ + 3*c₀ = 1 ∧ 1/a₀ + 2/b₀ + 3/c₀ = 36 := by
  sorry

end min_value_sum_reciprocals_l3031_303109


namespace exists_quadrilateral_with_adjacent_colors_l3031_303122

/-- Represents the color of a vertex -/
inductive Color
| Black
| White

/-- Represents a convex polygon -/
structure ConvexPolygon where
  vertices : ℕ
  coloring : ℕ → Color

/-- Represents a quadrilateral formed by dividing the polygon -/
structure Quadrilateral where
  v1 : ℕ
  v2 : ℕ
  v3 : ℕ
  v4 : ℕ

/-- The specific coloring pattern of the 2550-gon -/
def specific_coloring : ℕ → Color := sorry

/-- The 2550-gon with the specific coloring -/
def polygon_2550 : ConvexPolygon :=
  { vertices := 2550,
    coloring := specific_coloring }

/-- Predicate to check if a quadrilateral has two adjacent black vertices and two adjacent white vertices -/
def has_adjacent_colors (q : Quadrilateral) (p : ConvexPolygon) : Prop := sorry

/-- A division of the polygon into quadrilaterals -/
def division : List Quadrilateral := sorry

/-- Theorem stating that there exists a quadrilateral with the required color pattern -/
theorem exists_quadrilateral_with_adjacent_colors :
  ∃ q ∈ division, has_adjacent_colors q polygon_2550 := by sorry

end exists_quadrilateral_with_adjacent_colors_l3031_303122


namespace inequality_solution_l3031_303157

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧ 
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < 1 ↔ 
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by sorry

end inequality_solution_l3031_303157


namespace quadratic_inequality_range_l3031_303199

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, -x^2 + 2*x + 3 ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
sorry

end quadratic_inequality_range_l3031_303199


namespace intersection_orthogonal_l3031_303153

/-- The ellipse E with equation x²/8 + y²/4 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The line L with equation y = √5*x + 4 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 5 * p.1 + 4}

/-- The intersection points of E and L -/
def intersection := E ∩ L

/-- Theorem: If A and B are the intersection points of E and L, then OA ⊥ OB -/
theorem intersection_orthogonal (A B : ℝ × ℝ) 
  (hA : A ∈ intersection) (hB : B ∈ intersection) (hAB : A ≠ B) :
  (A.1 * B.1 + A.2 * B.2 = 0) := by sorry

end intersection_orthogonal_l3031_303153


namespace x_value_proof_l3031_303143

theorem x_value_proof (x : ℝ) (h1 : x^2 - 5*x = 0) (h2 : x ≠ 0) : x = 5 := by
  sorry

end x_value_proof_l3031_303143


namespace sound_distance_at_10C_l3031_303150

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the table

-- Theorem statement
theorem sound_distance_at_10C (temp : Int) (time : Int) :
  temp = 10 ∧ time = 4 → speed_of_sound temp * time = 1344 := by
  sorry

end sound_distance_at_10C_l3031_303150


namespace negation_of_universal_positive_square_plus_one_l3031_303124

theorem negation_of_universal_positive_square_plus_one (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_universal_positive_square_plus_one_l3031_303124


namespace a_cubed_congruence_l3031_303167

theorem a_cubed_congruence (n : ℕ+) (a : ℤ) 
  (h1 : a * a ≡ 1 [ZMOD n])
  (h2 : a ≡ -1 [ZMOD n]) :
  a^3 ≡ -1 [ZMOD n] := by
  sorry

end a_cubed_congruence_l3031_303167
