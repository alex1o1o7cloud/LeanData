import Mathlib

namespace order_of_sqrt_differences_l2529_252958

theorem order_of_sqrt_differences :
  let a := Real.sqrt 3 - Real.sqrt 2
  let b := Real.sqrt 6 - Real.sqrt 5
  let c := Real.sqrt 7 - Real.sqrt 6
  a > b ∧ b > c := by
  sorry

end order_of_sqrt_differences_l2529_252958


namespace linear_function_through_points_and_m_l2529_252979

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℚ
  b : ℚ

/-- Check if a point lies on a linear function -/
def pointOnFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.k * p.x + f.b

theorem linear_function_through_points_and_m
  (A : Point)
  (B : Point)
  (C : Point)
  (h1 : A.x = 3 ∧ A.y = 5)
  (h2 : B.x = -4 ∧ B.y = -9)
  (h3 : C.y = 2) :
  ∃ (f : LinearFunction),
    pointOnFunction A f ∧
    pointOnFunction B f ∧
    pointOnFunction C f ∧
    f.k = 2 ∧
    f.b = -1 ∧
    C.x = 3/2 := by
  sorry

end linear_function_through_points_and_m_l2529_252979


namespace abs_five_point_five_minus_pi_l2529_252992

theorem abs_five_point_five_minus_pi :
  |5.5 - Real.pi| = 5.5 - Real.pi :=
by sorry

end abs_five_point_five_minus_pi_l2529_252992


namespace greatest_common_multiple_10_15_under_90_l2529_252943

theorem greatest_common_multiple_10_15_under_90 : 
  ∃ (n : ℕ), n = 60 ∧ 
  (∀ m : ℕ, m < 90 ∧ 10 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  10 ∣ n ∧ 15 ∣ n ∧ n < 90 :=
by sorry

end greatest_common_multiple_10_15_under_90_l2529_252943


namespace visitors_growth_rate_l2529_252986

theorem visitors_growth_rate (x : ℝ) : 
  (420000 : ℝ) * (1 + x)^2 = 1339100 ↔ 42 * (1 + x)^2 = 133.91 :=
by sorry

end visitors_growth_rate_l2529_252986


namespace intersection_implies_a_value_l2529_252980

def set_A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def set_B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A a ∩ set_B a = {9} → a = -3 :=
by
  sorry

end intersection_implies_a_value_l2529_252980


namespace complex_calculations_l2529_252997

theorem complex_calculations : 
  (∀ x : ℝ, x^2 = 3 → (1 + x) * (2 - x) = -1 + x) ∧
  (Real.sqrt 36 * Real.sqrt 12 / Real.sqrt 3 = 12) ∧
  (Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4) ∧
  ((3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 = 2) :=
by sorry

end complex_calculations_l2529_252997


namespace range_of_a_l2529_252939

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0))) →
  (∃ x : ℝ, ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)) ∧ 
    (x^2 - 4*a*x + 3*a^2 ≥ 0)) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
by sorry

end range_of_a_l2529_252939


namespace not_always_parallel_if_perpendicular_to_same_plane_l2529_252948

-- Define a type for planes
axiom Plane : Type

-- Define a relation for perpendicularity between planes
axiom perpendicular : Plane → Plane → Prop

-- Define a relation for parallelism between planes
axiom parallel : Plane → Plane → Prop

-- State the theorem
theorem not_always_parallel_if_perpendicular_to_same_plane :
  ¬ (∀ (P Q R : Plane), perpendicular P R → perpendicular Q R → parallel P Q) :=
sorry

end not_always_parallel_if_perpendicular_to_same_plane_l2529_252948


namespace female_officers_count_l2529_252999

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_total_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/2 →
  female_total_ratio = 1/4 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_total_ratio = 1000 := by
  sorry

end female_officers_count_l2529_252999


namespace f_properties_l2529_252912

def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

theorem f_properties :
  (∀ x b, f x b 0 = -f (-x) b 0) ∧
  (∀ c, c > 0 → ∃! x, f x 0 c = 0) ∧
  (∀ x b c, f (x - 0) b c - c = -(f (-x - 0) b c - c)) ∧
  (∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end f_properties_l2529_252912


namespace min_value_theorem_l2529_252970

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 2/(y + 1) = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 2/(b + 1) = 2 → 2*x + y ≤ 2*a + b :=
by sorry

end min_value_theorem_l2529_252970


namespace cos_2alpha_plus_pi_third_l2529_252989

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end cos_2alpha_plus_pi_third_l2529_252989


namespace candy_ratio_is_three_l2529_252931

/-- The ratio of Jennifer's candies to Bob's candies -/
def candy_ratio (emily_candies bob_candies : ℕ) : ℚ :=
  (2 * emily_candies) / bob_candies

/-- Theorem: The ratio of Jennifer's candies to Bob's candies is 3 -/
theorem candy_ratio_is_three :
  candy_ratio 6 4 = 3 := by
  sorry

end candy_ratio_is_three_l2529_252931


namespace ratio_problem_l2529_252938

theorem ratio_problem (x y : ℚ) (h : (8*x - 5*y) / (10*x - 3*y) = 4/7) : x/y = 23/16 := by
  sorry

end ratio_problem_l2529_252938


namespace largest_n_when_floor_sqrt_n_is_5_l2529_252966

/-- Floor function: largest integer not greater than x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

theorem largest_n_when_floor_sqrt_n_is_5 :
  ∀ n : ℕ, (floor (Real.sqrt n) = 5) → (∀ m : ℕ, m ≤ n → m ≤ 35) ∧ n ≤ 35 :=
sorry

end largest_n_when_floor_sqrt_n_is_5_l2529_252966


namespace value_of_expression_l2529_252965

theorem value_of_expression (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end value_of_expression_l2529_252965


namespace perpendicular_probability_l2529_252901

/-- The set of positive integers less than 6 -/
def A : Set ℕ := {n | n < 6 ∧ n > 0}

/-- The line l: x + 2y + 1 = 0 -/
def l (x y : ℝ) : Prop := x + 2*y + 1 = 0

/-- The condition for the line from (a,b) to (0,0) being perpendicular to l -/
def perpendicular (a b : ℕ) : Prop := (b : ℝ) / (a : ℝ) = 2

/-- The number of ways to select 3 different elements from A -/
def total_outcomes : ℕ := Nat.choose 5 3

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 6

/-- The main theorem -/
theorem perpendicular_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 10 := by sorry

end perpendicular_probability_l2529_252901


namespace cube_edge_length_in_water_l2529_252913

/-- Theorem: Edge length of a cube immersed in water --/
theorem cube_edge_length_in_water 
  (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) (a : ℝ) :
  base_length = 20 →
  base_width = 15 →
  water_rise = 11.25 →
  a^3 = base_length * base_width * water_rise →
  a = 15 :=
by sorry

end cube_edge_length_in_water_l2529_252913


namespace polar_to_rectangular_conversion_l2529_252907

theorem polar_to_rectangular_conversion (r : ℝ) (θ : ℝ) :
  r = 6 ∧ θ = π / 3 →
  (r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 * Real.sqrt 3) := by
  sorry

end polar_to_rectangular_conversion_l2529_252907


namespace average_speed_round_trip_l2529_252974

/-- Calculates the average speed for a round trip journey between two points -/
theorem average_speed_round_trip (d : ℝ) (uphill_speed downhill_speed : ℝ) 
  (h1 : uphill_speed > 0)
  (h2 : downhill_speed > 0)
  (h3 : uphill_speed = 60)
  (h4 : downhill_speed = 36) :
  (2 * d) / (d / uphill_speed + d / downhill_speed) = 45 := by
  sorry

end average_speed_round_trip_l2529_252974


namespace intersection_count_l2529_252964

/-- The number of intersections between the line 3x + 4y = 12 and the circle x^2 + y^2 = 16 -/
def num_intersections : ℕ := 2

/-- The line equation 3x + 4y = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 16 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Theorem stating that the number of intersections between the given line and circle is 2 -/
theorem intersection_count :
  ∃ (p q : ℝ × ℝ),
    line_equation p.1 p.2 ∧ circle_equation p.1 p.2 ∧
    line_equation q.1 q.2 ∧ circle_equation q.1 q.2 ∧
    p ≠ q ∧
    (∀ (r : ℝ × ℝ), line_equation r.1 r.2 ∧ circle_equation r.1 r.2 → r = p ∨ r = q) :=
by sorry

end intersection_count_l2529_252964


namespace ribbon_segment_length_l2529_252903

theorem ribbon_segment_length :
  let total_length : ℚ := 4/5
  let num_segments : ℕ := 3
  let segment_fraction : ℚ := 1/3
  let segment_length : ℚ := total_length * segment_fraction
  segment_length = 4/15 := by
  sorry

end ribbon_segment_length_l2529_252903


namespace right_focus_coordinates_l2529_252945

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 64 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (10, 0)

/-- Theorem: The right focus of the given hyperbola is (10, 0) -/
theorem right_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → right_focus = (10, 0) := by
  sorry

end right_focus_coordinates_l2529_252945


namespace original_salary_is_twenty_thousand_l2529_252911

/-- Calculates the original salary of employees given the conditions of Emily's salary change --/
def calculate_original_salary (emily_original_salary : ℕ) (emily_new_salary : ℕ) (num_employees : ℕ) (new_employee_salary : ℕ) : ℕ :=
  let salary_difference := emily_original_salary - emily_new_salary
  let salary_increase_per_employee := salary_difference / num_employees
  new_employee_salary - salary_increase_per_employee

/-- Theorem stating that given the problem conditions, the original salary of each employee was $20,000 --/
theorem original_salary_is_twenty_thousand :
  calculate_original_salary 1000000 850000 10 35000 = 20000 := by
  sorry

end original_salary_is_twenty_thousand_l2529_252911


namespace complement_of_union_l2529_252927

open Set

universe u

def U : Finset ℕ := {1,2,3,4,5,6,7,8}
def A : Finset ℕ := {1,2,3}
def B : Finset ℕ := {3,4,5,6}

theorem complement_of_union :
  (U \ (A ∪ B) : Finset ℕ) = {7,8} := by sorry

end complement_of_union_l2529_252927


namespace complement_of_60_degrees_l2529_252951

def angle : ℝ := 60

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_60_degrees :
  complement angle = 30 := by
  sorry

end complement_of_60_degrees_l2529_252951


namespace litter_patrol_collection_l2529_252988

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 := by
  sorry

end litter_patrol_collection_l2529_252988


namespace triangle_side_relationship_l2529_252985

/-- Given a triangle with perimeter 12 and one side 5, prove the relationship between the other two sides -/
theorem triangle_side_relationship (x y : ℝ) : 
  (0 < x ∧ x < 6) → 
  (0 < y ∧ y < 6) → 
  (5 + x + y = 12) → 
  y = 7 - x :=
by sorry

end triangle_side_relationship_l2529_252985


namespace hyperbola_focus_distance_l2529_252950

/-- The hyperbola with equation x²/16 - y²/20 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focus_distance (P : ℝ × ℝ) :
  hyperbola P.1 P.2 → distance P F₁ = 9 → distance P F₂ = 17 := by sorry

end hyperbola_focus_distance_l2529_252950


namespace orchids_after_planting_l2529_252976

/-- The number of orchid bushes in the park after planting -/
def total_orchids (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 orchid bushes after planting -/
theorem orchids_after_planting :
  total_orchids 2 4 = 6 := by
sorry

end orchids_after_planting_l2529_252976


namespace investment_of_c_l2529_252962

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℚ
  profitShare : ℚ

/-- Represents a business partnership -/
def Partnership (a b c : Partner) : Prop :=
  -- Profit shares are proportional to investments
  a.profitShare / a.investment = b.profitShare / b.investment ∧
  b.profitShare / b.investment = c.profitShare / c.investment ∧
  -- Given conditions
  b.profitShare = 1800 ∧
  a.profitShare - c.profitShare = 720 ∧
  a.investment = 8000 ∧
  b.investment = 10000

theorem investment_of_c (a b c : Partner) 
  (h : Partnership a b c) : c.investment = 4000 := by
  sorry

end investment_of_c_l2529_252962


namespace subtract_negative_numbers_l2529_252933

theorem subtract_negative_numbers : -5 - 9 = -14 := by
  sorry

end subtract_negative_numbers_l2529_252933


namespace consecutive_even_sum_l2529_252949

theorem consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ a : ℤ, n * (n - 1)^(k - 1) = n * (2 * a + (n - 1)) :=
by sorry

end consecutive_even_sum_l2529_252949


namespace round_0_689_to_two_places_l2529_252973

/-- Rounds a real number to the specified number of decimal places. -/
def round_to_decimal_places (x : ℝ) (places : ℕ) : ℝ := 
  sorry

/-- The given number to be rounded -/
def given_number : ℝ := 0.689

/-- Theorem stating that rounding 0.689 to two decimal places results in 0.69 -/
theorem round_0_689_to_two_places :
  round_to_decimal_places given_number 2 = 0.69 := by
  sorry

end round_0_689_to_two_places_l2529_252973


namespace dan_has_five_limes_l2529_252968

/-- The number of limes Dan has after giving some to Sara -/
def dans_remaining_limes (initial_limes : ℕ) (limes_given : ℕ) : ℕ :=
  initial_limes - limes_given

/-- Theorem stating that Dan has 5 limes after giving 4 to Sara -/
theorem dan_has_five_limes :
  dans_remaining_limes 9 4 = 5 := by
  sorry

end dan_has_five_limes_l2529_252968


namespace smallest_n_for_eq1_smallest_n_for_eq2_l2529_252967

-- Define the properties for the equations
def satisfies_eq1 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^2

def satisfies_eq2 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^3

-- Define the smallest n for each equation
def smallest_n1 : ℕ := 3
def smallest_n2 : ℕ := 2

-- Theorem for the first equation
theorem smallest_n_for_eq1 :
  satisfies_eq1 smallest_n1 ∧
  ∀ m : ℕ, m < smallest_n1 → ¬(satisfies_eq1 m) :=
by sorry

-- Theorem for the second equation
theorem smallest_n_for_eq2 :
  satisfies_eq2 smallest_n2 ∧
  ∀ m : ℕ, m < smallest_n2 → ¬(satisfies_eq2 m) :=
by sorry

end smallest_n_for_eq1_smallest_n_for_eq2_l2529_252967


namespace f_is_even_f_is_decreasing_f_minimum_on_interval_l2529_252998

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: f(x) is even iff a = 0
theorem f_is_even (a : ℝ) : (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

-- Theorem 2: f(x) is decreasing on (-∞, 4] iff a ≥ 4
theorem f_is_decreasing (a : ℝ) : (∀ x y, x ≤ y ∧ y ≤ 4 → f a x ≥ f a y) ↔ a ≥ 4 := by sorry

-- Theorem 3: Minimum value of f(x) on [1, 2]
theorem f_minimum_on_interval (a : ℝ) :
  (∀ x ∈ [1, 2], f a x ≥ 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) ∧
  (∃ x ∈ [1, 2], f a x = 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) := by sorry

end f_is_even_f_is_decreasing_f_minimum_on_interval_l2529_252998


namespace rectangle_with_three_tangent_circles_l2529_252942

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Checks if two circles are tangent to each other -/
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the sides of a rectangle -/
def is_circle_tangent_to_rectangle (c : Circle) (r : Rectangle) : Prop :=
  c.radius ≤ r.width / 2 ∧ c.radius ≤ r.length / 2

/-- Main theorem: If a rectangle contains three tangent circles (two smaller equal ones and one larger),
    and the width of the rectangle is 4, then its length is 3 + √8 -/
theorem rectangle_with_three_tangent_circles 
  (r : Rectangle) 
  (c1 c2 c3 : Circle) : 
  r.width = 4 →
  c1.radius = c2.radius →
  c1.radius < c3.radius →
  are_circles_tangent c1 c2 →
  are_circles_tangent c1 c3 →
  are_circles_tangent c2 c3 →
  is_circle_tangent_to_rectangle c1 r →
  is_circle_tangent_to_rectangle c2 r →
  is_circle_tangent_to_rectangle c3 r →
  r.length = 3 + Real.sqrt 8 := by
  sorry


end rectangle_with_three_tangent_circles_l2529_252942


namespace quadratic_root_in_interval_l2529_252932

theorem quadratic_root_in_interval (a b : ℝ) (hb : b > 0) 
  (h_distinct : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ r₁^2 + a*r₁ + b = 0 ∧ r₂^2 + a*r₂ + b = 0)
  (h_one_in_interval : ∃! r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Icc (-1) 1) :
  ∃ r : ℝ, r^2 + a*r + b = 0 ∧ r ∈ Set.Ioo (-b) b :=
sorry

end quadratic_root_in_interval_l2529_252932


namespace eleventh_number_with_digit_sum_13_l2529_252990

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem stating that the 11th number with digit sum 13 is 145 -/
theorem eleventh_number_with_digit_sum_13 :
  nth_number_with_digit_sum_13 11 = 145 :=
sorry

end eleventh_number_with_digit_sum_13_l2529_252990


namespace comic_arrangement_count_l2529_252923

def arrange_comics (spiderman : Nat) (archie : Nat) (garfield : Nat) : Nat :=
  Nat.factorial spiderman * (Nat.factorial archie * Nat.factorial garfield * Nat.factorial 2)

theorem comic_arrangement_count :
  arrange_comics 7 6 5 = 871219200 := by
  sorry

end comic_arrangement_count_l2529_252923


namespace quadratic_inequality_l2529_252977

theorem quadratic_inequality (x : ℝ) : -15 * x^2 + 10 * x + 5 > 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end quadratic_inequality_l2529_252977


namespace inscribed_circle_radius_l2529_252905

/-- A sector that is one-third of a circle --/
structure ThirdCircleSector where
  /-- The radius of the full circle --/
  R : ℝ
  /-- Assumption that R is positive --/
  R_pos : 0 < R

/-- An inscribed circle in the sector --/
structure InscribedCircle (S : ThirdCircleSector) where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- Assumption that r is positive --/
  r_pos : 0 < r

/-- The theorem stating the radius of the inscribed circle --/
theorem inscribed_circle_radius (S : ThirdCircleSector) (C : InscribedCircle S) 
    (h : S.R = 6) : C.r = 6 * Real.sqrt 3 / 5 := by
  sorry

end inscribed_circle_radius_l2529_252905


namespace fixed_point_on_all_parabolas_l2529_252984

/-- The parabola family defined by a real parameter t -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (3, 36)

/-- Theorem stating that the fixed point lies on all parabolas in the family -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end fixed_point_on_all_parabolas_l2529_252984


namespace rectangle_x_value_l2529_252926

/-- Given a rectangle with vertices (x, 1), (1, 1), (1, -2), and (x, -2) and area 12, prove that x = -3 -/
theorem rectangle_x_value (x : ℝ) : 
  let vertices := [(x, 1), (1, 1), (1, -2), (x, -2)]
  let width := 1 - (-2)
  let area := 12
  let length := area / width
  x = 1 - length := by
  sorry

#check rectangle_x_value

end rectangle_x_value_l2529_252926


namespace solution_set_f_less_than_2_range_of_a_for_solution_exists_l2529_252978

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a where f(x) ≤ a - a²/2 has a solution
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 := by sorry

end solution_set_f_less_than_2_range_of_a_for_solution_exists_l2529_252978


namespace alex_calculation_l2529_252993

theorem alex_calculation (x : ℝ) : x / 6 - 18 = 24 → x * 6 + 18 = 1530 := by
  sorry

end alex_calculation_l2529_252993


namespace codecracker_combinations_l2529_252918

/-- The number of different colors of pegs available in CodeCracker -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in CodeCracker -/
def num_codes : ℕ := num_colors ^ num_slots

theorem codecracker_combinations : num_codes = 7776 := by
  sorry

end codecracker_combinations_l2529_252918


namespace fox_speed_l2529_252902

/-- Given a constant speed where 100 kilometers are covered in 120 minutes, 
    prove that the speed in kilometers per hour is 50. -/
theorem fox_speed (distance : ℝ) (time_minutes : ℝ) (speed_km_per_hour : ℝ)
  (h1 : distance = 100)
  (h2 : time_minutes = 120)
  (h3 : speed_km_per_hour = distance / time_minutes * 60) :
  speed_km_per_hour = 50 := by
  sorry

end fox_speed_l2529_252902


namespace intersection_x_coordinate_l2529_252921

-- Define the two curves
def curve1 (x y : ℝ) : Prop := y = 8 / (x^2 + 4)
def curve2 (x y : ℝ) : Prop := x + y = 2

-- Theorem stating that the x-coordinate of the intersection point is 0
theorem intersection_x_coordinate :
  ∃ y : ℝ, curve1 0 y ∧ curve2 0 y :=
sorry

end intersection_x_coordinate_l2529_252921


namespace max_sum_distance_to_line_l2529_252975

theorem max_sum_distance_to_line (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁^2 + y₁^2 = 1)
  (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₁*x₂ + y₁*y₂ = 1/2) :
  (|x₁ + y₁ - 1| / Real.sqrt 2) + (|x₂ + y₂ - 1| / Real.sqrt 2) ≤ 1 :=
sorry

end max_sum_distance_to_line_l2529_252975


namespace T_is_three_rays_with_common_point_l2529_252963

/-- The set T of points (x,y) in the coordinate plane satisfying the given conditions -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2
    (x + 3 = 5 ∧ y - 2 ≤ 5) ∨
    (y - 2 = 5 ∧ x + 3 ≤ 5) ∨
    (x + 3 = y - 2 ∧ 5 ≤ x + 3)}

/-- The common point of the three rays -/
def common_point : ℝ × ℝ := (2, 7)

/-- The three rays that form set T -/
def ray1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 ∧ p.2 ≤ 7}
def ray2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≤ 2 ∧ p.2 = 7}
def ray3 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 2 ∧ p.2 = p.1 + 5}

/-- Theorem stating that T consists of three rays with a common point -/
theorem T_is_three_rays_with_common_point :
  T = ray1 ∪ ray2 ∪ ray3 ∧
  common_point ∈ ray1 ∧ common_point ∈ ray2 ∧ common_point ∈ ray3 :=
sorry

end T_is_three_rays_with_common_point_l2529_252963


namespace beavers_help_l2529_252946

theorem beavers_help (initial_beavers : Real) (current_beavers : Nat) 
  (h1 : initial_beavers = 2.0) 
  (h2 : current_beavers = 3) : 
  (current_beavers : Real) - initial_beavers = 1 := by
  sorry

end beavers_help_l2529_252946


namespace inscribed_rectangle_area_l2529_252908

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area (a b h x : ℝ) (hb : b > a) (hh : h > 0) (hx : 0 < x ∧ x < h) :
  let rectangle_area := (b - a) * x * (h - x) / h
  rectangle_area = (b - a) * x * (h - x) / h := by sorry

end inscribed_rectangle_area_l2529_252908


namespace first_player_advantage_l2529_252922

/-- A game board configuration -/
structure BoardConfig where
  spaces : ℕ
  s₁ : ℕ
  s₂ : ℕ

/-- The probability of a player winning -/
def winProbability (player : ℕ) (config : BoardConfig) : ℝ :=
  sorry

/-- The theorem stating that the first player has a higher probability of winning -/
theorem first_player_advantage (config : BoardConfig) 
    (h : config.spaces ≥ 12) 
    (h_start : config.s₁ = config.s₂) : 
  winProbability 1 config > 1/2 :=
sorry

end first_player_advantage_l2529_252922


namespace scientific_notation_361000000_l2529_252994

/-- Express 361000000 in scientific notation -/
theorem scientific_notation_361000000 : ∃ (a : ℝ) (n : ℤ), 
  361000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.61 ∧ n = 8 := by
  sorry

end scientific_notation_361000000_l2529_252994


namespace purchasing_power_increase_l2529_252960

theorem purchasing_power_increase (original_price : ℝ) (money : ℝ) (h : money > 0) :
  let new_price := 0.8 * original_price
  let original_quantity := money / original_price
  let new_quantity := money / new_price
  new_quantity = 1.25 * original_quantity :=
by sorry

end purchasing_power_increase_l2529_252960


namespace circle_center_and_radius_l2529_252934

/-- Given a circle with equation x^2 + y^2 + 4x = 0, its center is at (-2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x = 0 → ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 0) ∧ radius = 2 ∧
    (x + 2)^2 + y^2 = 4 :=
by sorry

end circle_center_and_radius_l2529_252934


namespace logarithm_inequality_l2529_252961

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log c^2 / Real.log (a + b) + Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (c + a) ≥ 3 := by
  sorry

end logarithm_inequality_l2529_252961


namespace surface_area_of_revolution_l2529_252917

theorem surface_area_of_revolution (S α : ℝ) (h1 : S > 0) (h2 : 0 < α ∧ α < 2 * π) :
  let surface_area := (8 * π * S * Real.sin (α / 4)^2 * (1 + Real.cos (α / 4)^2)) / (α - Real.sin α)
  ∃ (R : ℝ), R > 0 ∧
    S = R^2 / 2 * (α - Real.sin α) ∧
    surface_area = 2 * π * R * (R * (1 - Real.cos (α / 2))) + π * (R * Real.sin (α / 2))^2 :=
by sorry

end surface_area_of_revolution_l2529_252917


namespace correct_paintball_spending_l2529_252982

/-- Represents the paintball spending calculation for John --/
def paintball_spending (regular_plays_per_month : ℕ) 
                       (boxes_per_play : ℕ) 
                       (price_1_5 : ℚ) 
                       (price_6_11 : ℚ) 
                       (price_12_plus : ℚ) 
                       (discount_12_plus : ℚ) 
                       (regular_maintenance : ℚ) 
                       (peak_maintenance : ℚ) 
                       (travel_week1 : ℚ) 
                       (travel_week2 : ℚ) 
                       (travel_week3 : ℚ) 
                       (travel_week4 : ℚ) : ℚ × ℚ :=
  let regular_boxes := regular_plays_per_month * boxes_per_play
  let peak_boxes := 2 * regular_boxes
  let travel_cost := travel_week1 + travel_week2 + travel_week3 + travel_week4
  
  let regular_paintball_cost := 
    if regular_boxes ≤ 5 then regular_boxes * price_1_5
    else if regular_boxes ≤ 11 then regular_boxes * price_6_11
    else let cost := regular_boxes * price_12_plus
         cost - (cost * discount_12_plus)
  
  let peak_paintball_cost := 
    let cost := peak_boxes * price_12_plus
    cost - (cost * discount_12_plus)
  
  let regular_total := regular_paintball_cost + regular_maintenance + travel_cost
  let peak_total := peak_paintball_cost + peak_maintenance + travel_cost
  
  (regular_total, peak_total)

/-- Theorem stating the correct paintball spending for John --/
theorem correct_paintball_spending :
  paintball_spending 3 3 25 23 22 (1/10) 40 60 10 15 12 8 = (292, 461.4) :=
sorry

end correct_paintball_spending_l2529_252982


namespace expand_and_simplify_l2529_252956

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a ^ 2 := by
  sorry

end expand_and_simplify_l2529_252956


namespace sales_difference_prove_sales_difference_l2529_252983

def morning_sales (remy_bottles : ℕ) (nick_bottles : ℕ) (price_per_bottle : ℚ) : ℚ :=
  (remy_bottles + nick_bottles) * price_per_bottle

theorem sales_difference (remy_morning_bottles : ℕ) (evening_sales : ℚ) : ℚ :=
  let nick_morning_bottles : ℕ := remy_morning_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_total : ℚ := morning_sales remy_morning_bottles nick_morning_bottles price_per_bottle
  evening_sales - morning_total

theorem prove_sales_difference : sales_difference 55 55 = 3 := by
  sorry

end sales_difference_prove_sales_difference_l2529_252983


namespace general_equation_pattern_l2529_252900

theorem general_equation_pattern (n : ℝ) : n ≠ 4 ∧ (8 - n) ≠ 4 →
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end general_equation_pattern_l2529_252900


namespace joker_prob_is_one_twentyseventh_l2529_252915

/-- A standard deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = 54)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of the deck -/
def joker_probability (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem: The probability of drawing a joker from a standard 54-card deck with 2 jokers is 1/27 -/
theorem joker_prob_is_one_twentyseventh (d : Deck) : joker_probability d = 1 / 27 := by
  sorry

end joker_prob_is_one_twentyseventh_l2529_252915


namespace equation_is_linear_l2529_252909

def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (A B C : ℝ), ∀ x y, f x y = A * x + B * y + C

def equation (x y : ℝ) : ℝ := 2 * x + 3 * y - 4

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation :=
sorry

end equation_is_linear_l2529_252909


namespace semicircle_sum_limit_l2529_252924

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of semicircles
    constructed on equal parts of a circle's diameter approaches the semi-circumference of the original circle. -/
theorem semicircle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * (D / n) / 2) - π * D / 2| < ε :=
sorry

end semicircle_sum_limit_l2529_252924


namespace rectangle_area_with_circles_l2529_252936

/-- The area of a rectangle containing 8 circles arranged in a 2x4 grid, 
    where each circle has a radius of 3 inches. -/
theorem rectangle_area_with_circles (radius : ℝ) (width_circles : ℕ) (length_circles : ℕ) :
  radius = 3 →
  width_circles = 2 →
  length_circles = 4 →
  (2 * radius * width_circles) * (2 * radius * length_circles) = 288 := by
  sorry

end rectangle_area_with_circles_l2529_252936


namespace trig_expression_max_value_l2529_252954

theorem trig_expression_max_value (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
by sorry

end trig_expression_max_value_l2529_252954


namespace exists_monochromatic_trapezoid_l2529_252914

/-- A color is represented as a natural number -/
def Color := ℕ

/-- A point on a circle -/
structure CirclePoint where
  angle : ℝ
  color : Color

/-- A circle with colored points -/
structure ColoredCircle where
  points : Set CirclePoint
  num_colors : ℕ
  color_bound : num_colors ≥ 2

/-- A trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint
  trapezoid_condition : (p2.angle - p1.angle) = (p4.angle - p3.angle)

/-- The main theorem -/
theorem exists_monochromatic_trapezoid (c : ColoredCircle) :
  ∃ t : InscribedTrapezoid, 
    t.p1 ∈ c.points ∧ 
    t.p2 ∈ c.points ∧ 
    t.p3 ∈ c.points ∧ 
    t.p4 ∈ c.points ∧
    t.p1.color = t.p2.color ∧ 
    t.p2.color = t.p3.color ∧ 
    t.p3.color = t.p4.color :=
  sorry

end exists_monochromatic_trapezoid_l2529_252914


namespace nested_fraction_evaluation_l2529_252996

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_evaluation_l2529_252996


namespace sum_of_roots_zero_l2529_252981

theorem sum_of_roots_zero (z₁ z₂ z₃ : ℝ) : 
  (4096 * z₁^3 + 16 * z₁ - 9 = 0) →
  (4096 * z₂^3 + 16 * z₂ - 9 = 0) →
  (4096 * z₃^3 + 16 * z₃ - 9 = 0) →
  z₁ + z₂ + z₃ = 0 :=
by sorry

end sum_of_roots_zero_l2529_252981


namespace ann_total_blocks_l2529_252940

/-- Ann's initial number of blocks -/
def initial_blocks : ℕ := 9

/-- Number of blocks Ann finds -/
def found_blocks : ℕ := 44

/-- Theorem stating the total number of blocks Ann ends with -/
theorem ann_total_blocks : initial_blocks + found_blocks = 53 := by
  sorry

end ann_total_blocks_l2529_252940


namespace f_of_3_equals_9_l2529_252953

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end f_of_3_equals_9_l2529_252953


namespace modular_arithmetic_problem_l2529_252930

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 120) + 9 * (13⁻¹ : ZMod 120) + 4 * (17⁻¹ : ZMod 120)) = (86 : ZMod 120) := by
  sorry

end modular_arithmetic_problem_l2529_252930


namespace points_for_level_completion_l2529_252969

/-- Given a game scenario, prove the points earned for completing the level -/
theorem points_for_level_completion 
  (enemies_defeated : ℕ) 
  (points_per_enemy : ℕ) 
  (total_points : ℕ) 
  (h1 : enemies_defeated = 6)
  (h2 : points_per_enemy = 9)
  (h3 : total_points = 62) :
  total_points - (enemies_defeated * points_per_enemy) = 8 :=
by sorry

end points_for_level_completion_l2529_252969


namespace ratio_simplification_and_increase_l2529_252971

def original_ratio : List Nat := [4, 16, 20, 12]

def gcd_list (l : List Nat) : Nat :=
  l.foldl Nat.gcd 0

def simplify_ratio (l : List Nat) : List Nat :=
  let gcd := gcd_list l
  l.map (·/gcd)

def percentage_increase (first last : Nat) : Nat :=
  ((last - first) * 100) / first

theorem ratio_simplification_and_increase :
  let simplified := simplify_ratio original_ratio
  simplified = [1, 4, 5, 3] ∧
  percentage_increase simplified.head! simplified.getLast! = 200 := by
  sorry

end ratio_simplification_and_increase_l2529_252971


namespace second_polygon_sides_l2529_252919

theorem second_polygon_sides (perimeter : ℝ) (sides_first : ℕ) (length_ratio : ℝ) (sides_second : ℕ) : 
  perimeter > 0 →
  sides_first = 50 →
  length_ratio = 3 →
  (perimeter = sides_first * (length_ratio * (perimeter / (sides_second * length_ratio)))) →
  (perimeter = sides_second * (perimeter / (sides_second * length_ratio))) →
  sides_second = 150 := by
sorry

end second_polygon_sides_l2529_252919


namespace rectangular_garden_area_rectangular_garden_area_proof_l2529_252995

/-- A rectangular garden with length three times its width and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_garden_area : ℝ → Prop :=
  fun w : ℝ =>
    w > 0 →                   -- width is positive
    2 * (w + 3 * w) = 72 →    -- perimeter is 72 meters
    w * (3 * w) = 243         -- area is 243 square meters

/-- Proof of the rectangular_garden_area theorem -/
theorem rectangular_garden_area_proof : ∃ w : ℝ, rectangular_garden_area w :=
  sorry

end rectangular_garden_area_rectangular_garden_area_proof_l2529_252995


namespace union_of_reduced_rectangles_l2529_252972

-- Define a reduced rectangle as a set in ℝ²
def ReducedRectangle : Set (ℝ × ℝ) → Prop :=
  sorry

-- Define a family of reduced rectangles
def FamilyOfReducedRectangles : Set (Set (ℝ × ℝ)) → Prop :=
  sorry

-- The main theorem
theorem union_of_reduced_rectangles 
  (F : Set (Set (ℝ × ℝ))) 
  (h : FamilyOfReducedRectangles F) :
  ∃ (C : Set (Set (ℝ × ℝ))), 
    (C ⊆ F) ∧ 
    (Countable C) ∧ 
    (⋃₀ F = ⋃₀ C) :=
  sorry

end union_of_reduced_rectangles_l2529_252972


namespace triangular_prism_volume_l2529_252935

/-- The volume of a triangular prism with given dimensions -/
theorem triangular_prism_volume 
  (thickness : ℝ) 
  (side1 side2 side3 : ℝ) 
  (h_thickness : thickness = 2)
  (h_side1 : side1 = 7)
  (h_side2 : side2 = 24)
  (h_side3 : side3 = 25)
  (h_right_triangle : side1^2 + side2^2 = side3^2) :
  thickness * (1/2 * side1 * side2) = 168 := by
sorry

end triangular_prism_volume_l2529_252935


namespace tan_alpha_and_fraction_l2529_252920

theorem tan_alpha_and_fraction (α : Real) 
  (h : Real.tan (α + π / 4) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = -1 / 9 := by
  sorry

end tan_alpha_and_fraction_l2529_252920


namespace real_part_of_complex_expression_l2529_252916

theorem real_part_of_complex_expression :
  Complex.re ((1 - 2 * Complex.I)^2 + Complex.I) = -3 := by
  sorry

end real_part_of_complex_expression_l2529_252916


namespace initial_ratio_proof_l2529_252928

/-- Proves that given a 30-liter mixture of liquids p and q, if adding 12 liters of liquid q
    results in a 3:4 ratio of p to q, then the initial ratio of p to q was 3:2. -/
theorem initial_ratio_proof (p q : ℝ) 
  (h1 : p + q = 30)  -- Initial mixture is 30 liters
  (h2 : p / (q + 12) = 3 / 4)  -- After adding 12 liters of q, the ratio becomes 3:4
  : p / q = 3 / 2 := by
  sorry

end initial_ratio_proof_l2529_252928


namespace gwen_spent_l2529_252947

theorem gwen_spent (received : ℕ) (left : ℕ) (spent : ℕ) : 
  received = 7 → left = 5 → spent = received - left → spent = 2 := by
  sorry

end gwen_spent_l2529_252947


namespace sufficient_not_necessary_l2529_252929

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a ≥ 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a < 0) :=
sorry

end sufficient_not_necessary_l2529_252929


namespace princess_pear_cherries_l2529_252959

def jester_height (i : ℕ) : ℕ := i

def is_valid_group (group : Finset ℕ) : Prop :=
  group.card = 6 ∧ ∃ (n : ℕ), n ≤ 100 ∧
  (∃ (lower upper : Finset ℕ),
    lower.card = 3 ∧ upper.card = 3 ∧
    lower ∪ upper = group ∧
    ∀ i ∈ lower, ∀ j ∈ upper, jester_height i < jester_height j)

def number_of_cherries : ℕ := (Nat.choose 50 3) ^ 2 * 2

theorem princess_pear_cherries :
  number_of_cherries = 384160000 := by sorry

end princess_pear_cherries_l2529_252959


namespace duke_dvd_count_l2529_252991

/-- Represents the number of DVDs Duke found in the first box -/
def first_box_count : ℕ := sorry

/-- Represents the price of each DVD in the first box -/
def first_box_price : ℚ := 2

/-- Represents the number of DVDs Duke found in the second box -/
def second_box_count : ℕ := 5

/-- Represents the price of each DVD in the second box -/
def second_box_price : ℚ := 5

/-- Represents the average price of all DVDs bought -/
def average_price : ℚ := 3

theorem duke_dvd_count : first_box_count = 5 := by
  sorry

end duke_dvd_count_l2529_252991


namespace triangle_medians_inequality_l2529_252937

/-- Given a triangle with sides a, b, c, medians ta, tb, tc, and circumcircle diameter D,
    the sum of the ratios of the squared sides to their opposite medians
    is less than or equal to 6 times the diameter of the circumcircle. -/
theorem triangle_medians_inequality (a b c ta tb tc D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_ta : 0 < ta) (h_pos_tb : 0 < tb) (h_pos_tc : 0 < tc)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_medians : ta^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧ 
               tb^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧ 
               tc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_circumcircle : D = (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (h_s : s = (a + b + c) / 2) :
  (a^2 + b^2) / tc + (b^2 + c^2) / ta + (c^2 + a^2) / tb ≤ 6 * D :=
sorry

end triangle_medians_inequality_l2529_252937


namespace profession_assignment_l2529_252941

/-- Represents the three people mentioned in the problem -/
inductive Person
  | Kondratyev
  | Davydov
  | Fedorov

/-- Represents the three professions mentioned in the problem -/
inductive Profession
  | Carpenter
  | Painter
  | Plumber

/-- Represents the age relation between two people -/
def OlderThan (a b : Person) : Prop := sorry

/-- Represents that one person has never heard of another -/
def NeverHeardOf (a b : Person) : Prop := sorry

/-- Represents the assignment of professions to people -/
def ProfessionAssignment := Person → Profession

/-- The carpenter was repairing the plumber's house -/
def CarpenterRepairingPlumbersHouse (assignment : ProfessionAssignment) : Prop := sorry

/-- The painter needed help from the carpenter -/
def PainterNeededHelpFromCarpenter (assignment : ProfessionAssignment) : Prop := sorry

/-- Main theorem: Given the conditions, prove the correct profession assignment -/
theorem profession_assignment :
  ∀ (assignment : ProfessionAssignment),
    (∀ p : Profession, ∃! person : Person, assignment person = p) →
    OlderThan Person.Davydov Person.Kondratyev →
    NeverHeardOf Person.Fedorov Person.Davydov →
    CarpenterRepairingPlumbersHouse assignment →
    PainterNeededHelpFromCarpenter assignment →
    (∀ p1 p2 : Person, assignment p1 = Profession.Plumber ∧ assignment p2 = Profession.Painter → OlderThan p1 p2) →
    assignment Person.Kondratyev = Profession.Carpenter ∧
    assignment Person.Davydov = Profession.Painter ∧
    assignment Person.Fedorov = Profession.Plumber := by
  sorry


end profession_assignment_l2529_252941


namespace range_of_b_l2529_252944

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * Real.log (x + 1) - x - b
def g (x : ℝ) : ℝ := Real.exp x

-- Define the point P
def P (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- State the theorem
theorem range_of_b (a : ℝ) (x₀ : ℝ) (h1 : 0 < x₀ ∧ x₀ < Real.exp 1 - 1) :
  ∃ b : ℝ, 0 < b ∧ b < 1 - 1 / Real.exp 1 ∧
  ∃ y₀ : ℝ, 
    -- P is on the curve f
    y₀ = f a b x₀ ∧
    -- OP is the tangent line of f
    (deriv (f a b) x₀ = y₀ / x₀) ∧
    -- OP is perpendicular to a tangent line of g passing through the origin
    ∃ m : ℝ, deriv g m * (y₀ / x₀) = -1 ∧ g m = m * (deriv g m) :=
sorry

end range_of_b_l2529_252944


namespace diana_statue_painting_l2529_252955

/-- The number of statues that can be painted given a certain amount of paint and paint required per statue -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Given 1/2 gallon of paint and 1/4 gallon required per statue, 2 statues can be painted -/
theorem diana_statue_painting :
  statues_paintable (1/2) (1/4) = 2 := by
  sorry

end diana_statue_painting_l2529_252955


namespace jack_two_queens_probability_l2529_252987

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of Jacks in a standard deck
def numJacks : ℕ := 4

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the probability of the specific draw
def probJackTwoQueens : ℚ := 2 / 5525

-- Theorem statement
theorem jack_two_queens_probability :
  (numJacks / standardDeck) * (numQueens / (standardDeck - 1)) * ((numQueens - 1) / (standardDeck - 2)) = probJackTwoQueens := by
  sorry

end jack_two_queens_probability_l2529_252987


namespace vasya_number_exists_l2529_252925

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 8) ∧
  (digits.count 1 = 2) ∧ (digits.count 2 = 2) ∧ (digits.count 3 = 2) ∧ (digits.count 4 = 2) ∧
  (∃ i, digits.get? i = some 1 ∧ digits.get? (i + 2) = some 1) ∧
  (∃ i, digits.get? i = some 2 ∧ digits.get? (i + 3) = some 2) ∧
  (∃ i, digits.get? i = some 3 ∧ digits.get? (i + 4) = some 3) ∧
  (∃ i, digits.get? i = some 4 ∧ digits.get? (i + 5) = some 4)

theorem vasya_number_exists : ∃ n : ℕ, is_valid_number n := by
  sorry

end vasya_number_exists_l2529_252925


namespace paige_goldfish_l2529_252904

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := sorry

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := 12

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

theorem paige_goldfish :
  initial_goldfish = 7 :=
by sorry

end paige_goldfish_l2529_252904


namespace binomial_expansion_theorem_l2529_252910

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) :
  (∃ k : ℕ, k ≥ 2 ∧
    Nat.choose n k * x^(n - k) * a^k = 210 ∧
    Nat.choose n (k + 1) * x^(n - k - 1) * a^(k + 1) = 504 ∧
    Nat.choose n (k + 2) * x^(n - k - 2) * a^(k + 2) = 1260) →
  n = 7 := by
sorry

end binomial_expansion_theorem_l2529_252910


namespace project_nap_duration_l2529_252957

theorem project_nap_duration 
  (project_days : ℕ) 
  (hours_per_day : ℕ) 
  (work_hours : ℕ) 
  (num_naps : ℕ) 
  (h1 : project_days = 4) 
  (h2 : hours_per_day = 24) 
  (h3 : work_hours = 54) 
  (h4 : num_naps = 6) : 
  (project_days * hours_per_day - work_hours) / num_naps = 7 := by
  sorry

end project_nap_duration_l2529_252957


namespace max_regions_six_chords_l2529_252906

/-- The number of regions created by drawing k chords in a circle -/
def num_regions (k : ℕ) : ℕ := 1 + k * (k + 1) / 2

/-- Theorem: The maximum number of regions created by drawing 6 chords in a circle is 22 -/
theorem max_regions_six_chords : num_regions 6 = 22 := by
  sorry

end max_regions_six_chords_l2529_252906


namespace beth_crayons_l2529_252952

/-- Given the number of crayon packs, crayons per pack, and extra crayons,
    calculate the total number of crayons Beth has. -/
def total_crayons (packs : ℕ) (crayons_per_pack : ℕ) (extra_crayons : ℕ) : ℕ :=
  packs * crayons_per_pack + extra_crayons

/-- Prove that Beth has 46 crayons in total. -/
theorem beth_crayons : total_crayons 4 10 6 = 46 := by
  sorry

end beth_crayons_l2529_252952
