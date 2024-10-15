import Mathlib

namespace NUMINAMATH_CALUDE_javier_exercise_minutes_l2279_227954

/-- Proves that Javier exercised for 50 minutes each day given the conditions of the problem -/
theorem javier_exercise_minutes : ℕ → Prop :=
  fun x => 
    (∀ d : ℕ, d ≤ 7 → x > 0) →  -- Javier exercised some minutes every day for one week
    (3 * 90 + 7 * x = 620) →   -- Total exercise time for both Javier and Sanda
    x = 50                     -- Javier exercised for 50 minutes each day

/-- Proof of the theorem -/
lemma prove_javier_exercise_minutes : ∃ x : ℕ, javier_exercise_minutes x :=
  sorry

end NUMINAMATH_CALUDE_javier_exercise_minutes_l2279_227954


namespace NUMINAMATH_CALUDE_min_value_of_function_l2279_227992

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 31 ∧
  (x^2 + 6*x + 36/x^2 = 31 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2279_227992


namespace NUMINAMATH_CALUDE_airport_distance_solution_l2279_227964

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistanceProblem (initial_speed : ℝ) (speed_increase : ℝ) (stop_time : ℝ) (early_arrival : ℝ) : Prop :=
  ∃ (distance : ℝ) (initial_time : ℝ),
    -- The first portion is driven at the initial speed
    initial_speed * initial_time = 40 ∧
    -- The total time includes the initial drive, stop time, and the rest of the journey
    (distance - 40) / (initial_speed + speed_increase) + initial_time + stop_time = 
      (distance / initial_speed) - early_arrival ∧
    -- The total distance is 190 miles
    distance = 190

/-- Theorem stating that the solution to the problem is 190 miles --/
theorem airport_distance_solution :
  AirportDistanceProblem 40 20 0.25 0.25 := by
  sorry

#check airport_distance_solution

end NUMINAMATH_CALUDE_airport_distance_solution_l2279_227964


namespace NUMINAMATH_CALUDE_pair_and_triplet_count_two_pairs_count_l2279_227975

/- Define the structure of a deck of cards -/
def numSuits : Nat := 4
def numRanks : Nat := 13
def deckSize : Nat := numSuits * numRanks

/- Define the combination function -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Theorem for part 1 -/
theorem pair_and_triplet_count :
  choose numRanks 1 * choose numSuits 2 * choose (numRanks - 1) 1 * choose numSuits 3 = 3744 :=
by sorry

/- Theorem for part 2 -/
theorem two_pairs_count :
  choose numRanks 2 * (choose numSuits 2)^2 * choose (numRanks - 2) 1 * choose numSuits 1 = 123552 :=
by sorry

end NUMINAMATH_CALUDE_pair_and_triplet_count_two_pairs_count_l2279_227975


namespace NUMINAMATH_CALUDE_main_theorem_l2279_227974

/-- The set S of ordered triples satisfying the given conditions -/
def S (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | ∃ x y z, t = (x, y, z) ∧ 
       x ∈ Finset.range n ∧ y ∈ Finset.range n ∧ z ∈ Finset.range n ∧
       ((x < y ∧ y < z) ∨ (y < z ∧ z < x) ∨ (z < x ∧ x < y)) ∧
       ¬((x < y ∧ y < z) ∧ (y < z ∧ z < x)) ∧
       ¬((y < z ∧ z < x) ∧ (z < x ∧ x < y)) ∧
       ¬((z < x ∧ x < y) ∧ (x < y ∧ y < z))}

/-- The main theorem -/
theorem main_theorem (n : ℕ) (h : n ≥ 4) 
  (x y z w : ℕ) (hxyz : (x, y, z) ∈ S n) (hzwx : (z, w, x) ∈ S n) :
  (y, z, w) ∈ S n ∧ (x, y, w) ∈ S n := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l2279_227974


namespace NUMINAMATH_CALUDE_zeros_of_specific_f_graph_above_line_implies_b_gt_2_solution_set_when_b_eq_2_l2279_227910

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (2*a + 1) * x + b

-- Statement 1
theorem zeros_of_specific_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = 1 ∧
  ∀ (x : ℝ), f 1 (-4) x = 0 ↔ (x = x₁ ∨ x = x₂) :=
sorry

-- Statement 2
theorem graph_above_line_implies_b_gt_2 (a b : ℝ) :
  (∀ x : ℝ, f a b x > x + 2) → b > 2 :=
sorry

-- Statement 3
theorem solution_set_when_b_eq_2 (a : ℝ) :
  let S := {x : ℝ | f a 2 x < 0}
  if a < 0 then
    S = {x : ℝ | x < -2 ∨ x > -1/a}
  else if a = 0 then
    S = {x : ℝ | x < -2}
  else if 0 < a ∧ a < 1/2 then
    S = {x : ℝ | -1/a < x ∧ x < -2}
  else if a = 1/2 then
    S = ∅
  else -- a > 1/2
    S = {x : ℝ | -2 < x ∧ x < -1/a} :=
sorry

end NUMINAMATH_CALUDE_zeros_of_specific_f_graph_above_line_implies_b_gt_2_solution_set_when_b_eq_2_l2279_227910


namespace NUMINAMATH_CALUDE_percentage_problem_l2279_227905

/-- The percentage P that satisfies the equation (1/10 * 8000) - (P/100 * 8000) = 796 -/
theorem percentage_problem (P : ℝ) : (1/10 * 8000) - (P/100 * 8000) = 796 ↔ P = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2279_227905


namespace NUMINAMATH_CALUDE_side_b_value_triangle_area_l2279_227900

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  -- Add conditions here
  a = 3 ∧ 
  Real.cos A = Real.sqrt 6 / 3 ∧
  B = A + Real.pi / 2

-- Theorem for the value of side b
theorem side_b_value (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : b = 3 * Real.sqrt 2 := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : 
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_side_b_value_triangle_area_l2279_227900


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2279_227922

theorem fraction_evaluation : (3 - (-3)) / (2 - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2279_227922


namespace NUMINAMATH_CALUDE_special_triangle_third_side_l2279_227906

/-- A triangle with two sides of lengths 2 and 3, and the third side length satisfying a quadratic equation. -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a = 2
  h2 : b = 3
  h3 : c^2 - 10*c + 21 = 0
  h4 : a + b > c ∧ b + c > a ∧ c + a > b  -- Triangle inequality

/-- The third side of the SpecialTriangle has length 3. -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_third_side_l2279_227906


namespace NUMINAMATH_CALUDE_restaurant_meal_cost_l2279_227976

theorem restaurant_meal_cost 
  (total_people : ℕ) 
  (num_kids : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_people = 11) 
  (h2 : num_kids = 2) 
  (h3 : total_cost = 72) :
  (total_cost : ℚ) / (total_people - num_kids : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_meal_cost_l2279_227976


namespace NUMINAMATH_CALUDE_min_unheard_lines_l2279_227967

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio read -/
def sonnets_read : ℕ := 7

/-- Represents the minimum number of additional sonnets Horatio prepared -/
def min_additional_sonnets : ℕ := 1

/-- Theorem stating the minimum number of unheard lines -/
theorem min_unheard_lines :
  min_additional_sonnets * lines_per_sonnet = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_unheard_lines_l2279_227967


namespace NUMINAMATH_CALUDE_math_team_selection_ways_l2279_227924

/-- The number of ways to select r items from n items --/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The total number of students in the math club --/
def total_students : ℕ := 14

/-- The number of students to be selected for the team --/
def team_size : ℕ := 6

/-- Theorem stating that the number of ways to select the team is 3003 --/
theorem math_team_selection_ways :
  binomial total_students team_size = 3003 := by sorry

end NUMINAMATH_CALUDE_math_team_selection_ways_l2279_227924


namespace NUMINAMATH_CALUDE_missing_number_equation_l2279_227989

theorem missing_number_equation (x : ℤ) : 10111 - 10 * 2 * x = 10011 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2279_227989


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2279_227941

theorem geometric_progression_ratio (b₁ q : ℕ) (h_sum : b₁ * q^2 + b₁ * q^4 + b₁ * q^6 = 7371 * 2^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2279_227941


namespace NUMINAMATH_CALUDE_line_intercepts_l2279_227997

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts :
  let line := (fun (x y : ℝ) => x/4 - y/3 = 1)
  (∃ x : ℝ, line x 0 ∧ x = 4) ∧
  (∃ y : ℝ, line 0 y ∧ y = -3) := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_l2279_227997


namespace NUMINAMATH_CALUDE_extremum_properties_l2279_227904

noncomputable section

variable (x : ℝ)

def f (x : ℝ) : ℝ := x * Real.log x + (1/2) * x^2

def is_extremum_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, x > 0 → f x ≤ f x₀ ∨ f x ≥ f x₀

theorem extremum_properties (x₀ : ℝ) 
  (h₁ : x₀ > 0) 
  (h₂ : is_extremum_point f x₀) : 
  (0 < x₀ ∧ x₀ < Real.exp (-1)) ∧ 
  (f x₀ + x₀ < 0) := by
  sorry

end

end NUMINAMATH_CALUDE_extremum_properties_l2279_227904


namespace NUMINAMATH_CALUDE_problem_statement_l2279_227943

theorem problem_statement : 112 * 5^4 * 3^2 = 630000 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2279_227943


namespace NUMINAMATH_CALUDE_min_tests_for_16_people_l2279_227937

/-- Represents the number of people in the group -/
def total_people : ℕ := 16

/-- Represents the number of infected people -/
def infected_people : ℕ := 1

/-- The function that calculates the minimum number of tests required -/
def min_tests (n : ℕ) : ℕ := Nat.log2 n + 1

/-- Theorem stating that the minimum number of tests for 16 people is 4 -/
theorem min_tests_for_16_people :
  min_tests total_people = 4 :=
sorry

end NUMINAMATH_CALUDE_min_tests_for_16_people_l2279_227937


namespace NUMINAMATH_CALUDE_tan_function_property_l2279_227961

/-- 
Given positive constants a and b, if the function y = a * tan(b * x) 
has a period of π/2 and passes through the point (π/8, 1), then ab = 2.
-/
theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = π / 2) → 
  (a * Real.tan (b * π / 8) = 1) → 
  a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l2279_227961


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2279_227996

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2279_227996


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2279_227934

/-- An isosceles triangle with perimeter 24 and a median that divides the perimeter in a 5:3 ratio -/
structure IsoscelesTriangle where
  /-- Length of each equal side -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The perimeter is 24 -/
  perimeter_eq : 2 * x + y = 24
  /-- The median divides the perimeter in a 5:3 ratio -/
  median_ratio : 3 * x / (x + y) = 5 / 3

/-- The base of the isosceles triangle is 4 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2279_227934


namespace NUMINAMATH_CALUDE_vector_equation_l2279_227998

/-- Given vectors a, b, c, and e in a vector space, 
    prove that 2a - 3b + c = 23e under certain conditions. -/
theorem vector_equation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2279_227998


namespace NUMINAMATH_CALUDE_polygon_sequence_limit_l2279_227987

/-- Represents the sequence of polygons formed by cutting corners -/
def polygon_sequence (n : ℕ) : ℝ :=
  sorry

/-- The area of the triangle cut from each corner in the nth iteration -/
def cut_triangle_area (n : ℕ) : ℝ :=
  sorry

/-- The number of corners in the nth polygon -/
def num_corners (n : ℕ) : ℕ :=
  sorry

theorem polygon_sequence_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |polygon_sequence n - 5/7| < ε :=
sorry

end NUMINAMATH_CALUDE_polygon_sequence_limit_l2279_227987


namespace NUMINAMATH_CALUDE_angle_symmetry_l2279_227970

theorem angle_symmetry (α β : Real) :
  (∃ k : ℤ, α + β = 2 * k * Real.pi) →
  ∃ k : ℤ, α = 2 * k * Real.pi - β :=
by sorry

end NUMINAMATH_CALUDE_angle_symmetry_l2279_227970


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2279_227977

-- Proposition 1
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by sorry

-- Proposition 2
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Proposition 3
theorem negation_equivalence :
  (¬∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2279_227977


namespace NUMINAMATH_CALUDE_violets_family_ticket_cost_l2279_227986

/-- Calculates the total cost of tickets for a family visit to the aquarium. -/
def total_ticket_cost (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Proves that the total cost for Violet's family to buy separate tickets is $155. -/
theorem violets_family_ticket_cost :
  total_ticket_cost 35 20 1 6 = 155 := by
  sorry

#eval total_ticket_cost 35 20 1 6

end NUMINAMATH_CALUDE_violets_family_ticket_cost_l2279_227986


namespace NUMINAMATH_CALUDE_lemonade_theorem_l2279_227909

/-- Represents the number of glasses of lemonade that can be made -/
def lemonade_glasses (lemons oranges grapefruits : ℕ) : ℕ :=
  let lemon_glasses := lemons / 2
  let orange_glasses := oranges
  let citrus_glasses := min lemon_glasses orange_glasses
  let grapefruit_glasses := grapefruits
  citrus_glasses + grapefruit_glasses

/-- Theorem stating that with given ingredients, 15 glasses of lemonade can be made -/
theorem lemonade_theorem : lemonade_glasses 18 10 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_theorem_l2279_227909


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2279_227947

theorem shaded_fraction_of_rectangle (length width : ℕ) 
  (h_length : length = 15)
  (h_width : width = 20)
  (section_fraction : ℚ)
  (h_section : section_fraction = 1 / 5)
  (shaded_fraction : ℚ)
  (h_shaded : shaded_fraction = 1 / 4) :
  (shaded_fraction * section_fraction : ℚ) = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2279_227947


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2279_227955

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 38621

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The processing fee in dollars -/
def processing_fee : ℝ := 2500

/-- The total amount paid in taxes and fees in dollars -/
def total_paid : ℝ := 16500

theorem inheritance_calculation (x : ℝ) (h : x = inheritance) :
  federal_tax_rate * x + state_tax_rate * (1 - federal_tax_rate) * x + processing_fee = total_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2279_227955


namespace NUMINAMATH_CALUDE_a_total_share_l2279_227946

def total_profit : ℚ := 9600
def a_investment : ℚ := 15000
def b_investment : ℚ := 25000
def management_fee_percentage : ℚ := 10 / 100

def total_investment : ℚ := a_investment + b_investment

def management_fee (profit : ℚ) : ℚ := management_fee_percentage * profit

def remaining_profit (profit : ℚ) : ℚ := profit - management_fee profit

def a_share_ratio : ℚ := a_investment / total_investment

theorem a_total_share :
  management_fee total_profit + (a_share_ratio * remaining_profit total_profit) = 4200 := by
  sorry

end NUMINAMATH_CALUDE_a_total_share_l2279_227946


namespace NUMINAMATH_CALUDE_class_size_problem_l2279_227948

/-- Given a class where:
    - The average mark of all students is 80
    - If 5 students with an average mark of 60 are excluded, the remaining students' average is 90
    Prove that the total number of students is 15 -/
theorem class_size_problem (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 15 ∧ 
  N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
sorry

end NUMINAMATH_CALUDE_class_size_problem_l2279_227948


namespace NUMINAMATH_CALUDE_test_average_l2279_227911

theorem test_average (male_count : ℕ) (female_count : ℕ) 
  (male_avg : ℝ) (female_avg : ℝ) : 
  male_count = 8 → 
  female_count = 32 → 
  male_avg = 82 → 
  female_avg = 92 → 
  (male_count * male_avg + female_count * female_avg) / (male_count + female_count) = 90 := by
  sorry

end NUMINAMATH_CALUDE_test_average_l2279_227911


namespace NUMINAMATH_CALUDE_polynomial_product_zero_l2279_227980

theorem polynomial_product_zero (a : ℚ) (h : a = 5/3) :
  (6*a^3 - 11*a^2 + 3*a - 2) * (3*a - 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_zero_l2279_227980


namespace NUMINAMATH_CALUDE_linear_function_through_origin_l2279_227930

/-- A linear function y = (m-1)x + m^2 - 1 passing through the origin has m = -1 -/
theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x + m^2 - 1) →
  (m - 1 ≠ 0) →
  (0 : ℝ) = (m - 1) * 0 + m^2 - 1 →
  m = -1 := by
  sorry

#check linear_function_through_origin

end NUMINAMATH_CALUDE_linear_function_through_origin_l2279_227930


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2279_227949

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2279_227949


namespace NUMINAMATH_CALUDE_triangle_transformation_l2279_227914

-- Define the points of the original triangle
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (8, 9)
def C : ℝ × ℝ := (-3, 7)

-- Define the points of the transformed triangle
def A' : ℝ × ℝ := (-2, -6)
def B' : ℝ × ℝ := (-7, -11)
def C' : ℝ × ℝ := (2, -9)

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-(x - 0.5) - 5.5, -y - 2)

-- Theorem stating that the transformation maps the original triangle to the new one
theorem triangle_transformation :
  transform A = A' ∧ transform B = B' ∧ transform C = C' := by
  sorry


end NUMINAMATH_CALUDE_triangle_transformation_l2279_227914


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l2279_227917

/-- Given a boat traveling downstream, calculate the speed of the stream. -/
theorem stream_speed_calculation 
  (boat_speed : ℝ)           -- Speed of the boat in still water
  (distance : ℝ)             -- Distance traveled downstream
  (time : ℝ)                 -- Time taken to travel downstream
  (h1 : boat_speed = 5)      -- Boat speed is 5 km/hr
  (h2 : distance = 100)      -- Distance is 100 km
  (h3 : time = 10)           -- Time taken is 10 hours
  : ∃ (stream_speed : ℝ), 
    stream_speed = 5 ∧ 
    distance = (boat_speed + stream_speed) * time :=
by sorry


end NUMINAMATH_CALUDE_stream_speed_calculation_l2279_227917


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l2279_227958

theorem quadratic_root_k_value : ∀ k : ℝ, 
  ((-1 : ℝ)^2 + 3*(-1) + k = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l2279_227958


namespace NUMINAMATH_CALUDE_sequence_problem_l2279_227968

theorem sequence_problem (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_relation : ∀ n, (a (n + 1))^2 + (a n)^2 = 2 * n * ((a (n + 1))^2 - (a n)^2)) :
  a 113 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2279_227968


namespace NUMINAMATH_CALUDE_smallest_possible_students_l2279_227953

/-- Represents the number of students in each of the four equal-sized groups -/
def n : ℕ := 7

/-- The total number of students in the drama club -/
def total_students : ℕ := 4 * n + 2 * (n + 1)

/-- The drama club has six groups -/
axiom six_groups : ℕ

/-- Four groups have the same number of students -/
axiom four_equal_groups : ℕ

/-- Two groups have one more student than the other four -/
axiom two_larger_groups : ℕ

/-- The total number of groups is six -/
axiom total_groups : six_groups = 4 + 2

/-- The total number of students exceeds 40 -/
axiom exceeds_forty : total_students > 40

/-- 44 is the smallest number of students satisfying all conditions -/
theorem smallest_possible_students : total_students = 44 ∧ 
  ∀ m : ℕ, m < n → 4 * m + 2 * (m + 1) ≤ 40 := by sorry

end NUMINAMATH_CALUDE_smallest_possible_students_l2279_227953


namespace NUMINAMATH_CALUDE_cake_muffin_buyers_l2279_227929

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 16) (h4 : prob_neither = 26/100) : 
  ∃ total_buyers : ℕ, 
    (total_buyers : ℚ) - ((cake_buyers : ℚ) + (muffin_buyers : ℚ) - (both_buyers : ℚ)) = 
    prob_neither * (total_buyers : ℚ) ∧ total_buyers = 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_buyers_l2279_227929


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2279_227928

/-- Given line passing through (1,2) and perpendicular to 2x - 6y + 1 = 0 -/
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

/-- Point that the perpendicular line passes through -/
def point : ℝ × ℝ := (1, 2)

/-- Equation of the perpendicular line -/
def perpendicular_line (x y : ℝ) : Prop := 3 * x + y - 5 = 0

/-- Theorem stating that the perpendicular line passing through (1,2) 
    has the equation 3x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ x y : ℝ, given_line x y → 
  (perpendicular_line x y ↔ 
   (perpendicular_line point.1 point.2 ∧ 
    (x - point.1) * (x - point.1) + (y - point.2) * (y - point.2) = 
    ((x - 1) * 2 + (y - 2) * (-6))^2 / (2^2 + (-6)^2))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2279_227928


namespace NUMINAMATH_CALUDE_forty_platforms_required_l2279_227918

/-- The minimum number of platforms required to transport granite slabs -/
def min_platforms (num_slabs_7ton : ℕ) (num_slabs_9ton : ℕ) (max_platform_capacity : ℕ) : ℕ :=
  let total_weight := num_slabs_7ton * 7 + num_slabs_9ton * 9
  (total_weight + max_platform_capacity - 1) / max_platform_capacity

/-- Theorem stating that 40 platforms are required for the given conditions -/
theorem forty_platforms_required :
  min_platforms 120 80 40 = 40 ∧
  ∀ n : ℕ, n < 40 → ¬ (120 * 7 + 80 * 9 ≤ n * 40) :=
by sorry

end NUMINAMATH_CALUDE_forty_platforms_required_l2279_227918


namespace NUMINAMATH_CALUDE_sqrt_a_is_integer_l2279_227988

theorem sqrt_a_is_integer (a b : ℕ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ k : ℤ, (Real.sqrt (Real.sqrt a + Real.sqrt b) + Real.sqrt (Real.sqrt a - Real.sqrt b)) = k) :
  ∃ n : ℕ, Real.sqrt a = n :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_is_integer_l2279_227988


namespace NUMINAMATH_CALUDE_lamp_cost_l2279_227972

theorem lamp_cost (lamp_cost bulb_cost : ℝ) : 
  (bulb_cost = lamp_cost - 4) →
  (2 * lamp_cost + 6 * bulb_cost = 32) →
  lamp_cost = 7 := by
sorry

end NUMINAMATH_CALUDE_lamp_cost_l2279_227972


namespace NUMINAMATH_CALUDE_expression_value_l2279_227982

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = 2) : 3 * a + 4 * b - 5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2279_227982


namespace NUMINAMATH_CALUDE_ratio_equals_2021_l2279_227944

def numerator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => numerator_sum n + (2021 - n) / (n + 1)

def denominator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => denominator_sum n + 1 / (n + 3)

theorem ratio_equals_2021 : 
  (numerator_sum 2016) / (denominator_sum 2016) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_2021_l2279_227944


namespace NUMINAMATH_CALUDE_log_158489_between_integers_l2279_227960

theorem log_158489_between_integers : ∃ p q : ℤ,
  (p : ℝ) < Real.log 158489 / Real.log 10 ∧
  Real.log 158489 / Real.log 10 < (q : ℝ) ∧
  q = p + 1 ∧
  p + q = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_158489_between_integers_l2279_227960


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2279_227927

theorem triangle_max_perimeter :
  ∀ (x y : ℕ),
    x > 0 →
    y > 0 →
    y = 2 * x →
    (x + y > 20 ∧ x + 20 > y ∧ y + 20 > x) →
    x + y + 20 ≤ 77 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2279_227927


namespace NUMINAMATH_CALUDE_pie_fraction_not_eaten_l2279_227919

theorem pie_fraction_not_eaten
  (lara_ate : ℚ)
  (ryan_ate : ℚ)
  (cassie_ate_remaining : ℚ)
  (h1 : lara_ate = 1/4)
  (h2 : ryan_ate = 3/10)
  (h3 : cassie_ate_remaining = 2/3)
  : 1 - (lara_ate + ryan_ate + cassie_ate_remaining * (1 - lara_ate - ryan_ate)) = 3/20 := by
  sorry

#check pie_fraction_not_eaten

end NUMINAMATH_CALUDE_pie_fraction_not_eaten_l2279_227919


namespace NUMINAMATH_CALUDE_faye_bought_48_books_l2279_227959

/-- The number of coloring books Faye initially had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := 3

/-- The total number of coloring books Faye had after buying more -/
def final_total : ℕ := 79

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := final_total - (initial_books - books_given_away)

theorem faye_bought_48_books : books_bought = 48 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_48_books_l2279_227959


namespace NUMINAMATH_CALUDE_class_fund_problem_l2279_227926

theorem class_fund_problem (m : ℕ) (x : ℕ) (y : ℚ) :
  m < 400 →
  38 ≤ x →
  x < 50 →
  x * y = m →
  (x + 12) * (y - 2) = m →
  x = 42 ∧ y = 9 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_problem_l2279_227926


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2279_227951

/-- 
Given three numbers forming an arithmetic sequence where the first number is 3,
and when the middle term is reduced by 6 it forms a geometric sequence,
prove that the third number (the unknown number) is either 3 or 27.
-/
theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 3 + b) →  -- arithmetic sequence condition
  ((a - 6)^2 = 3 * b) →  -- geometric sequence condition after reduction
  (b = 3 ∨ b = 27) :=  -- conclusion: the unknown number is either 3 or 27
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2279_227951


namespace NUMINAMATH_CALUDE_expression_simplification_l2279_227939

theorem expression_simplification (a : ℝ) (h : a = 3 + Real.sqrt 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (a^2 - 2*a)) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2279_227939


namespace NUMINAMATH_CALUDE_large_pepperoni_has_14_slices_l2279_227942

/-- The number of slices in a large pepperoni pizza -/
def large_pepperoni_slices (total_eaten : ℕ) (total_left : ℕ) (small_cheese_slices : ℕ) : ℕ :=
  total_eaten + total_left - small_cheese_slices

/-- Theorem stating that the large pepperoni pizza has 14 slices -/
theorem large_pepperoni_has_14_slices : 
  large_pepperoni_slices 18 4 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_large_pepperoni_has_14_slices_l2279_227942


namespace NUMINAMATH_CALUDE_wire_length_l2279_227950

/-- The length of a wire stretched between two poles -/
theorem wire_length (d h₁ h₂ : ℝ) (hd : d = 20) (hh₁ : h₁ = 8) (hh₂ : h₂ = 9) :
  Real.sqrt ((d ^ 2) + ((h₂ - h₁) ^ 2)) = Real.sqrt 401 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l2279_227950


namespace NUMINAMATH_CALUDE_last_three_average_l2279_227936

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l2279_227936


namespace NUMINAMATH_CALUDE_candidate_fail_marks_l2279_227983

theorem candidate_fail_marks (max_marks : ℝ) (passing_percentage : ℝ) (candidate_score : ℝ) :
  max_marks = 153.84615384615384 →
  passing_percentage = 52 →
  candidate_score = 45 →
  ⌈passing_percentage / 100 * max_marks⌉ - candidate_score = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_fail_marks_l2279_227983


namespace NUMINAMATH_CALUDE_trig_identity_l2279_227962

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2279_227962


namespace NUMINAMATH_CALUDE_josh_book_cost_l2279_227993

/-- Represents the cost of items and quantities purchased by Josh --/
structure ShoppingTrip where
  numFilms : ℕ
  filmCost : ℕ
  numBooks : ℕ
  numCDs : ℕ
  cdCost : ℕ
  totalSpent : ℕ

/-- Calculates the cost of each book given the shopping trip details --/
def bookCost (trip : ShoppingTrip) : ℕ :=
  (trip.totalSpent - trip.numFilms * trip.filmCost - trip.numCDs * trip.cdCost) / trip.numBooks

/-- Theorem stating that the cost of each book in Josh's shopping trip is 4 --/
theorem josh_book_cost :
  let trip : ShoppingTrip := {
    numFilms := 9,
    filmCost := 5,
    numBooks := 4,
    numCDs := 6,
    cdCost := 3,
    totalSpent := 79
  }
  bookCost trip = 4 := by sorry

end NUMINAMATH_CALUDE_josh_book_cost_l2279_227993


namespace NUMINAMATH_CALUDE_triangle_side_range_l2279_227963

theorem triangle_side_range (A B C : ℝ × ℝ) (x : ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 16 ∧ AC = 7 ∧ BC = x →
  9 < x ∧ x < 23 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2279_227963


namespace NUMINAMATH_CALUDE_students_per_van_l2279_227940

/-- Given five coaster vans transporting 60 boys and 80 girls, prove that each van carries 28 students. -/
theorem students_per_van (num_vans : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_vans = 5)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / num_vans = 28 := by
  sorry

end NUMINAMATH_CALUDE_students_per_van_l2279_227940


namespace NUMINAMATH_CALUDE_kim_status_update_time_l2279_227965

/-- Kim's morning routine -/
def morning_routine (coffee_time : ℕ) (payroll_time : ℕ) (num_employees : ℕ) (total_time : ℕ) (status_time : ℕ) : Prop :=
  coffee_time + num_employees * status_time + num_employees * payroll_time = total_time

/-- Theorem: Kim spends 2 minutes per employee getting a status update -/
theorem kim_status_update_time :
  ∃ (status_time : ℕ),
    morning_routine 5 3 9 50 status_time ∧
    status_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_kim_status_update_time_l2279_227965


namespace NUMINAMATH_CALUDE_thousandth_special_number_l2279_227952

/-- A function that returns true if n is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns true if n is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- The sequence of positive integers that are neither perfect squares nor perfect cubes --/
def specialSequence : ℕ → ℕ :=
  fun n => sorry

theorem thousandth_special_number :
  specialSequence 1000 = 1039 := by sorry

end NUMINAMATH_CALUDE_thousandth_special_number_l2279_227952


namespace NUMINAMATH_CALUDE_additional_cars_needed_l2279_227920

def cars_per_row : ℕ := 8
def current_cars : ℕ := 37

theorem additional_cars_needed : 
  ∃ (n : ℕ), (n > 0) ∧ (cars_per_row * n ≥ current_cars) ∧ 
  (cars_per_row * n - current_cars = 3) ∧
  (∀ m : ℕ, m > 0 → cars_per_row * m ≥ current_cars → 
    cars_per_row * m - current_cars ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l2279_227920


namespace NUMINAMATH_CALUDE_triangle_area_with_perpendicular_medians_l2279_227971

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define a median of a triangle
def Median (A B C M : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def Perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def Length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the altitude of a triangle
def Altitude (A B C H : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_with_perpendicular_medians 
  (X Y Z U V : ℝ × ℝ) 
  (h1 : Triangle X Y Z)
  (h2 : Median X Y Z U)
  (h3 : Median Y Z X V)
  (h4 : Perpendicular X U Y V)
  (h5 : Length X U = 10)
  (h6 : Length Y V = 24)
  (h7 : ∃ H, Altitude Z X Y H ∧ Length Z H = 16) :
  TriangleArea X Y Z = 160 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_perpendicular_medians_l2279_227971


namespace NUMINAMATH_CALUDE_remainder_5032_div_28_l2279_227923

theorem remainder_5032_div_28 : 5032 % 28 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5032_div_28_l2279_227923


namespace NUMINAMATH_CALUDE_milk_yogurt_quantities_l2279_227913

/-- Represents the quantities and prices of milk and yogurt --/
structure MilkYogurtData where
  milk_cost : ℝ
  yogurt_cost : ℝ
  yogurt_quantity_ratio : ℝ
  price_difference : ℝ
  milk_selling_price : ℝ
  yogurt_markup : ℝ
  yogurt_discount : ℝ
  total_profit : ℝ

/-- Theorem stating the quantities of milk and discounted yogurt --/
theorem milk_yogurt_quantities (data : MilkYogurtData) 
  (h_milk_cost : data.milk_cost = 2000)
  (h_yogurt_cost : data.yogurt_cost = 4800)
  (h_ratio : data.yogurt_quantity_ratio = 1.5)
  (h_price_diff : data.price_difference = 30)
  (h_milk_price : data.milk_selling_price = 80)
  (h_yogurt_markup : data.yogurt_markup = 0.25)
  (h_yogurt_discount : data.yogurt_discount = 0.1)
  (h_total_profit : data.total_profit = 2150) :
  ∃ (milk_quantity yogurt_discounted : ℕ),
    milk_quantity = 40 ∧ yogurt_discounted = 25 := by
  sorry

end NUMINAMATH_CALUDE_milk_yogurt_quantities_l2279_227913


namespace NUMINAMATH_CALUDE_minimum_value_and_range_l2279_227979

theorem minimum_value_and_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) → -2 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_l2279_227979


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2279_227973

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧
  (∃ x, x ∈ (M ∪ P) ∧ x ∉ (M ∩ P)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2279_227973


namespace NUMINAMATH_CALUDE_valid_triplet_configurations_l2279_227931

/-- A structure representing a configuration of triplet subsets satisfying the given conditions -/
structure TripletConfiguration (n : ℕ) :=
  (m : ℕ)
  (subsets : Fin m → Finset (Fin n))
  (cover_pairs : ∀ (i j : Fin n), i ≠ j → ∃ (k : Fin m), {i, j} ⊆ subsets k)
  (subset_size : ∀ (k : Fin m), (subsets k).card = 3)
  (intersect_one : ∀ (k₁ k₂ : Fin m), k₁ ≠ k₂ → (subsets k₁ ∩ subsets k₂).card = 1)

/-- The theorem stating that the only valid configurations are (1, 3) and (7, 7) -/
theorem valid_triplet_configurations :
  {n : ℕ | ∃ (c : TripletConfiguration n), True} = {3, 7} :=
sorry

end NUMINAMATH_CALUDE_valid_triplet_configurations_l2279_227931


namespace NUMINAMATH_CALUDE_division_problem_l2279_227938

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) : 
  dividend = 55053 → 
  quotient = 120 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  divisor = 456 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2279_227938


namespace NUMINAMATH_CALUDE_invertible_product_l2279_227907

def is_invertible (f : ℕ → Bool) : Prop := f 1 = false ∧ f 2 = true ∧ f 3 = true ∧ f 4 = true

theorem invertible_product (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [1, 2, 3, 4]).prod = 24 := by
  sorry

end NUMINAMATH_CALUDE_invertible_product_l2279_227907


namespace NUMINAMATH_CALUDE_eraser_ratio_l2279_227991

/-- The number of erasers each person has -/
structure EraserCounts where
  hanna : ℕ
  rachel : ℕ
  tanya : ℕ
  tanya_red : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : EraserCounts) : Prop :=
  counts.hanna = 2 * counts.rachel ∧
  counts.rachel = counts.tanya_red - 3 ∧
  counts.tanya = 20 ∧
  counts.tanya_red = counts.tanya / 2 ∧
  counts.hanna = 4

/-- The theorem to be proved -/
theorem eraser_ratio (counts : EraserCounts) 
  (h : problem_conditions counts) : 
  counts.rachel / counts.tanya_red = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_eraser_ratio_l2279_227991


namespace NUMINAMATH_CALUDE_henrys_room_books_l2279_227978

/-- The number of books Henry had in the room to donate -/
def books_in_room (initial_books : ℕ) (bookshelf_boxes : ℕ) (books_per_box : ℕ)
  (coffee_table_books : ℕ) (kitchen_books : ℕ) (free_books_taken : ℕ) (final_books : ℕ) : ℕ :=
  initial_books - (bookshelf_boxes * books_per_box + coffee_table_books + kitchen_books - free_books_taken)

/-- Theorem stating the number of books Henry had in the room to donate -/
theorem henrys_room_books :
  books_in_room 99 3 15 4 18 12 23 = 44 := by
  sorry

end NUMINAMATH_CALUDE_henrys_room_books_l2279_227978


namespace NUMINAMATH_CALUDE_line_equation_l2279_227966

/-- Circle C with center (3, 5) and radius sqrt(5) -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

/-- Line l passing through the center of circle C -/
structure Line_l where
  slope : ℝ
  equation : ℝ → ℝ → Prop
  passes_through_center : equation 3 5

/-- Point on the circle -/
structure Point_on_circle where
  x : ℝ
  y : ℝ
  on_circle : circle_C x y

/-- Point on the y-axis -/
structure Point_on_y_axis where
  y : ℝ

/-- Midpoint condition -/
def is_midpoint (A B P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line_l) 
  (A B : Point_on_circle) 
  (P : Point_on_y_axis)
  (h_A_on_l : l.equation A.x A.y)
  (h_B_on_l : l.equation B.x B.y)
  (h_P_on_l : l.equation 0 P.y)
  (h_midpoint : is_midpoint (A.x, A.y) (B.x, B.y) (0, P.y)) :
  (∃ k : ℝ, k = 2 ∨ k = -2) ∧ 
  (∀ x y, l.equation x y ↔ y - 5 = k * (x - 3)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2279_227966


namespace NUMINAMATH_CALUDE_am_gm_inequality_l2279_227903

theorem am_gm_inequality (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c < d) :
  ((c + d) / 2 - Real.sqrt (c * d)) < (d - c)^3 / (8 * c) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l2279_227903


namespace NUMINAMATH_CALUDE_recipe_cereal_cups_l2279_227912

/-- Given a recipe calling for 18.0 servings of cereal, where each serving is 2.0 cups,
    the total number of cups needed is 36.0. -/
theorem recipe_cereal_cups : 
  let servings : ℝ := 18.0
  let cups_per_serving : ℝ := 2.0
  servings * cups_per_serving = 36.0 := by
sorry

end NUMINAMATH_CALUDE_recipe_cereal_cups_l2279_227912


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l2279_227915

theorem complete_square_with_integer (x : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), x^2 - 6*x + 20 = (x - a)^2 + k ∧ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l2279_227915


namespace NUMINAMATH_CALUDE_line_within_plane_is_subset_l2279_227969

-- Define a type for geometric objects
inductive GeometricObject
| Line : GeometricObject
| Plane : GeometricObject

-- Define a relation for "is within"
def isWithin (x y : GeometricObject) : Prop := sorry

-- Define the subset relation
def subset (x y : GeometricObject) : Prop := sorry

-- Theorem statement
theorem line_within_plane_is_subset (a α : GeometricObject) :
  a = GeometricObject.Line → α = GeometricObject.Plane → isWithin a α → subset a α := by
  sorry

end NUMINAMATH_CALUDE_line_within_plane_is_subset_l2279_227969


namespace NUMINAMATH_CALUDE_sum_of_products_negative_max_greater_or_equal_cube_root_four_l2279_227984

-- Define the conditions
def sum_zero (a b c : ℝ) : Prop := a + b + c = 0
def product_one (a b c : ℝ) : Prop := a * b * c = 1

-- Define the theorems to prove
theorem sum_of_products_negative (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  a * b + b * c + c * a < 0 :=
sorry

theorem max_greater_or_equal_cube_root_four (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  max a (max b c) ≥ (4 : ℝ) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_products_negative_max_greater_or_equal_cube_root_four_l2279_227984


namespace NUMINAMATH_CALUDE_first_expression_value_l2279_227921

theorem first_expression_value (E a : ℝ) : 
  (E + (3 * a - 8)) / 2 = 84 → a = 32 → E = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_expression_value_l2279_227921


namespace NUMINAMATH_CALUDE_pentagonal_pyramid_edges_l2279_227935

-- Define a pentagonal pyramid
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle
  common_vertex : Point

-- Define the number of edges in a pentagonal pyramid
def num_edges_pentagonal_pyramid (pp : PentagonalPyramid) : ℕ := 10

-- Theorem statement
theorem pentagonal_pyramid_edges (pp : PentagonalPyramid) :
  num_edges_pentagonal_pyramid pp = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_pyramid_edges_l2279_227935


namespace NUMINAMATH_CALUDE_min_value_theorem_l2279_227956

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  ∃ (min : ℝ), min = 4 ∧ ∀ z, z = 1/x + 1/(3*y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2279_227956


namespace NUMINAMATH_CALUDE_kylie_jewelry_beads_l2279_227945

/-- The number of beaded necklaces Kylie makes on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces Kylie makes on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets Kylie makes on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie makes on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie uses to make her jewelry -/
def total_beads : ℕ := 
  (monday_necklaces + tuesday_necklaces) * beads_per_necklace +
  wednesday_bracelets * beads_per_bracelet +
  wednesday_earrings * beads_per_earring

theorem kylie_jewelry_beads : total_beads = 325 := by
  sorry

end NUMINAMATH_CALUDE_kylie_jewelry_beads_l2279_227945


namespace NUMINAMATH_CALUDE_slab_cost_l2279_227995

/-- The cost of a slab of beef given the conditions of kabob stick production -/
theorem slab_cost (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (sticks : ℕ) (total_cost : ℕ) : 
  cubes_per_stick = 4 →
  cubes_per_slab = 80 →
  sticks = 40 →
  total_cost = 50 →
  (total_cost : ℚ) / ((sticks * cubes_per_stick : ℕ) / cubes_per_slab : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_slab_cost_l2279_227995


namespace NUMINAMATH_CALUDE_equation_solution_l2279_227901

theorem equation_solution (x : ℝ) :
  (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4) →
  x = 257 / 16 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2279_227901


namespace NUMINAMATH_CALUDE_tank_difference_l2279_227994

theorem tank_difference (total : ℕ) (german allied sanchalian : ℕ) 
  (h1 : total = 115)
  (h2 : german = 2 * allied + 2)
  (h3 : allied = 3 * sanchalian + 1)
  (h4 : total = german + allied + sanchalian) :
  german - sanchalian = 59 := by
  sorry

end NUMINAMATH_CALUDE_tank_difference_l2279_227994


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2279_227990

-- Problem 1
theorem problem_1 : 24 - |(-2)| + (-16) - 8 = -2 := by sorry

-- Problem 2
theorem problem_2 : (-2) * (3/2) / (-3/4) * 4 = 4 := by sorry

-- Problem 3
theorem problem_3 : (-1)^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2279_227990


namespace NUMINAMATH_CALUDE_no_real_with_negative_sum_of_abs_and_square_l2279_227932

theorem no_real_with_negative_sum_of_abs_and_square :
  ¬ (∃ x : ℝ, abs x + x^2 < 0) := by
sorry

end NUMINAMATH_CALUDE_no_real_with_negative_sum_of_abs_and_square_l2279_227932


namespace NUMINAMATH_CALUDE_parabola_focus_l2279_227985

-- Define the parabola
def parabola (x y : ℝ) : Prop := 2 * x^2 = -y

-- Define the focus of a parabola
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, p x y → (x - f.1)^2 = 4 * f.2 * (y - f.2)

-- Theorem statement
theorem parabola_focus :
  focus (0, -1/8) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2279_227985


namespace NUMINAMATH_CALUDE_minimum_opponents_l2279_227933

/-- 
Given two integers h ≥ 1 and p ≥ 2, this theorem states that the minimum number of 
pairs of opponents in an hp-member parliament, such that in every partition into h 
houses of p members each, some house contains at least one pair of opponents, 
is equal to min((h-1)p + 1, (h+1 choose 2)).
-/
theorem minimum_opponents (h p : ℕ) (h_ge_one : h ≥ 1) (p_ge_two : p ≥ 2) :
  let parliament_size := h * p
  let min_opponents := min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)
  ∀ (opponents : Finset (Finset (Fin parliament_size))),
    (∀ partition : Finset (Finset (Fin parliament_size)),
      (partition.card = h ∧ 
       ∀ house ∈ partition, house.card = p ∧
       partition.sup id = Finset.univ) →
      ∃ house ∈ partition, ∃ pair ∈ opponents, pair ⊆ house) →
    opponents.card ≥ min_opponents ∧
    ∃ opponents_min : Finset (Finset (Fin parliament_size)),
      opponents_min.card = min_opponents ∧
      (∀ partition : Finset (Finset (Fin parliament_size)),
        (partition.card = h ∧ 
         ∀ house ∈ partition, house.card = p ∧
         partition.sup id = Finset.univ) →
        ∃ house ∈ partition, ∃ pair ∈ opponents_min, pair ⊆ house) :=
by sorry

end NUMINAMATH_CALUDE_minimum_opponents_l2279_227933


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2279_227925

/-- The common ratio of the infinite geometric series 8/10 - 6/15 + 36/225 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 36 / 225
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2279_227925


namespace NUMINAMATH_CALUDE_cube_monotone_l2279_227981

theorem cube_monotone (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_monotone_l2279_227981


namespace NUMINAMATH_CALUDE_train_length_l2279_227999

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 42 / 3600 → speed * time * 1000 = 700 := by sorry

end NUMINAMATH_CALUDE_train_length_l2279_227999


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2279_227902

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 150 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2279_227902


namespace NUMINAMATH_CALUDE_largest_circle_radius_on_chessboard_l2279_227916

/-- Represents a chessboard with the usual coloring of fields -/
structure Chessboard where
  size : Nat
  is_black : Nat → Nat → Bool

/-- Represents a circle on the chessboard -/
structure Circle where
  center : Nat × Nat
  radius : Real

/-- Check if a circle intersects a white field on the chessboard -/
def intersects_white (board : Chessboard) (circle : Circle) : Bool :=
  sorry

/-- The largest possible circle radius on a chessboard without intersecting white fields -/
def largest_circle_radius (board : Chessboard) : Real :=
  sorry

/-- Theorem stating the largest possible circle radius on a standard chessboard -/
theorem largest_circle_radius_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    (∀ i j, board.is_black i j = ((i + j) % 2 = 1)) →
    largest_circle_radius board = (1 / 2) * Real.sqrt 10 :=
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_on_chessboard_l2279_227916


namespace NUMINAMATH_CALUDE_money_division_l2279_227908

theorem money_division (a b c : ℚ) :
  a = (1/2 : ℚ) * (b + c) →
  b = (2/3 : ℚ) * (a + c) →
  a = 122 →
  a + b + c = 366 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2279_227908


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l2279_227957

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l2279_227957
