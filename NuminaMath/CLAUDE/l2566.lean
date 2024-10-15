import Mathlib

namespace NUMINAMATH_CALUDE_exponent_multiplication_l2566_256685

theorem exponent_multiplication (b : ℝ) : b * b^3 = b^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2566_256685


namespace NUMINAMATH_CALUDE_circle_center_l2566_256639

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The original equation of the circle -/
def OriginalEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 4

theorem circle_center :
  ∃ (r : ℝ), ∀ (x y : ℝ), OriginalEquation x y ↔ CircleEquation (-4) 2 r x y :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2566_256639


namespace NUMINAMATH_CALUDE_movie_theater_total_movies_l2566_256666

/-- Calculates the total number of movies shown in a movie theater. -/
def total_movies_shown (num_screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  num_screens * (open_hours / movie_duration)

/-- Proves that a movie theater with 6 screens, open for 8 hours, showing 2-hour movies, shows 24 movies total. -/
theorem movie_theater_total_movies :
  total_movies_shown 6 8 2 = 24 := by
  sorry

#eval total_movies_shown 6 8 2

end NUMINAMATH_CALUDE_movie_theater_total_movies_l2566_256666


namespace NUMINAMATH_CALUDE_debate_club_committee_selection_l2566_256656

theorem debate_club_committee_selection (n : ℕ) : 
  (n.choose 3 = 21) → (n.choose 4 = 126) := by
  sorry

end NUMINAMATH_CALUDE_debate_club_committee_selection_l2566_256656


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2566_256657

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2566_256657


namespace NUMINAMATH_CALUDE_speed_of_sound_l2566_256610

/-- The speed of sound given specific conditions --/
theorem speed_of_sound (travel_time : Real) (blast_interval : Real) (distance : Real) :
  travel_time = 30.0 + 25.0 / 60 → -- 30 minutes and 25 seconds in hours
  blast_interval = 0.5 → -- 30 minutes in hours
  distance = 8250 → -- distance in meters
  (distance / (travel_time - blast_interval)) * (1 / 3600) = 330 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_sound_l2566_256610


namespace NUMINAMATH_CALUDE_proposition_p_and_q_l2566_256655

def is_ellipse (m : ℝ) : Prop :=
  1 < m ∧ m < 3 ∧ m ≠ 2

def no_common_points (m : ℝ) : Prop :=
  m > Real.sqrt 5 / 2 ∨ m < -Real.sqrt 5 / 2

theorem proposition_p_and_q (m : ℝ) :
  (is_ellipse m ∧ no_common_points m) ↔ 
  (Real.sqrt 5 / 2 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_l2566_256655


namespace NUMINAMATH_CALUDE_division_problem_l2566_256695

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 301)
  (h2 : quotient = 14)
  (h3 : remainder = 7)
  : ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 21 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2566_256695


namespace NUMINAMATH_CALUDE_polynomial_division_l2566_256668

theorem polynomial_division (x : ℝ) :
  8 * x^4 - 4 * x^3 + 5 * x^2 - 9 * x + 3 = (x - 1) * (8 * x^3 - 4 * x^2 + 9 * x - 18) + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2566_256668


namespace NUMINAMATH_CALUDE_counterexample_exists_l2566_256686

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n - 2) :=
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2566_256686


namespace NUMINAMATH_CALUDE_toy_car_cost_price_l2566_256684

/-- The cost price of a toy car given specific pricing conditions --/
theorem toy_car_cost_price :
  ∀ (cost_price : ℝ),
  let initial_price := 2 * cost_price
  let second_day_price := 0.9 * initial_price
  let final_price := second_day_price - 360
  (final_price = 1.44 * cost_price) →
  cost_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_toy_car_cost_price_l2566_256684


namespace NUMINAMATH_CALUDE_base5_43102_equals_2902_l2566_256661

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5^(digits.length - 1 - i))) 0

theorem base5_43102_equals_2902 :
  base5_to_decimal [4, 3, 1, 0, 2] = 2902 := by
  sorry

end NUMINAMATH_CALUDE_base5_43102_equals_2902_l2566_256661


namespace NUMINAMATH_CALUDE_multiplication_mistake_l2566_256643

theorem multiplication_mistake (x : ℕ) (h : 53 * x - 35 * x = 540) : 53 * x = 1590 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l2566_256643


namespace NUMINAMATH_CALUDE_largest_band_size_l2566_256615

theorem largest_band_size : ∃ (m r x : ℕ),
  m < 150 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 2) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 150 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 2) = m' →
    m' ≤ m ∧
  m = 107 := by
sorry

end NUMINAMATH_CALUDE_largest_band_size_l2566_256615


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2566_256671

-- Define the polynomial expression
def poly (x : ℝ) : ℝ := 5 * (x - 2 * x^3) - 4 * (2 * x^2 - x^3 + 3 * x^6) + 3 * (5 * x^2 - 2 * x^8)

-- Theorem stating that the coefficient of x^2 in the polynomial is 7
theorem x_squared_coefficient : (deriv (deriv poly)) 0 / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2566_256671


namespace NUMINAMATH_CALUDE_subset_condition_l2566_256644

/-- The set A defined by the given condition -/
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (a + 1)) < 0}

/-- The set B defined by the given condition -/
def B (a : ℝ) : Set ℝ := {x | (x - 2*a) / (x - (a^2 + 1)) < 0}

/-- The theorem stating the relationship between a and the subset property -/
theorem subset_condition (a : ℝ) : 
  B a ⊆ A a ↔ a ∈ Set.Icc (-1/2) (-1/2) ∪ Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2566_256644


namespace NUMINAMATH_CALUDE_first_option_cost_is_68_l2566_256612

/-- Represents the car rental problem with given conditions -/
def CarRentalProblem (trip_distance : ℝ) (second_option_cost : ℝ) 
  (gas_efficiency : ℝ) (gas_cost_per_liter : ℝ) (savings : ℝ) : Prop :=
  let total_distance := 2 * trip_distance
  let gas_needed := total_distance / gas_efficiency
  let gas_cost := gas_needed * gas_cost_per_liter
  let first_option_cost := second_option_cost - savings
  first_option_cost = 68

/-- Theorem stating that the first option costs $68 per day -/
theorem first_option_cost_is_68 :
  CarRentalProblem 150 90 15 0.9 22 := by
  sorry

#check first_option_cost_is_68

end NUMINAMATH_CALUDE_first_option_cost_is_68_l2566_256612


namespace NUMINAMATH_CALUDE_find_k_l2566_256622

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 3 else n / 2

theorem find_k (k : ℤ) : 
  k % 2 = 1 → f (f (f k)) = 27 → k = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2566_256622


namespace NUMINAMATH_CALUDE_scientific_notation_conversion_l2566_256698

theorem scientific_notation_conversion :
  (2.61 * 10^(-5) = 0.0000261) ∧ (0.00068 = 6.8 * 10^(-4)) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_conversion_l2566_256698


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l2566_256663

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_intersection_properties :
  -- 1. The equation of the line containing AB is x - y = 0
  (∀ (x y : ℝ), (x - y = 0) ↔ (∃ (t : ℝ), x = t * A.1 + (1 - t) * B.1 ∧ y = t * A.2 + (1 - t) * B.2)) ∧
  -- 2. The equation of the perpendicular bisector of AB is x + y - 1 = 0
  (∀ (x y : ℝ), (x + y - 1 = 0) ↔ ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2)) ∧
  -- 3. The maximum distance from a point P on O₂ to the line AB is (3√2)/2 + √5
  (∃ (P : ℝ × ℝ), circle_O2 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ), circle_O2 Q.1 Q.2 →
      abs ((Q.1 - Q.2) / Real.sqrt 2) ≤ (3 * Real.sqrt 2) / 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l2566_256663


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2566_256621

theorem polynomial_factorization (x : ℤ) : 
  x^5 + x^4 + 1 = (x^3 - x + 1) * (x^2 + x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2566_256621


namespace NUMINAMATH_CALUDE_inequality_solution_l2566_256636

theorem inequality_solution (x : ℝ) : 
  (1 / 6 : ℝ) + |x - 1 / 3| < 1 / 2 ↔ 0 < x ∧ x < 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2566_256636


namespace NUMINAMATH_CALUDE_inequality_implies_bound_l2566_256635

theorem inequality_implies_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, Real.exp x - x > a * x) → a < Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bound_l2566_256635


namespace NUMINAMATH_CALUDE_johns_age_l2566_256681

def johns_age_problem (j d : ℕ) : Prop :=
  (j = d - 30) ∧ (j + d = 80)

theorem johns_age : ∃ j d : ℕ, johns_age_problem j d ∧ j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l2566_256681


namespace NUMINAMATH_CALUDE_peach_probability_l2566_256645

/-- A fruit type in the basket -/
inductive Fruit
| apple
| pear
| peach

/-- The number of fruits in the basket -/
def basket : Fruit → ℕ
| Fruit.apple => 5
| Fruit.pear => 3
| Fruit.peach => 2

/-- The total number of fruits in the basket -/
def total_fruits : ℕ := basket Fruit.apple + basket Fruit.pear + basket Fruit.peach

/-- The probability of picking a specific fruit -/
def prob_pick (f : Fruit) : ℚ := basket f / total_fruits

theorem peach_probability :
  prob_pick Fruit.peach = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_peach_probability_l2566_256645


namespace NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2566_256614

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  ((x1 = x3 ∧ x2 = x4) ∨ (x1 = x2 ∧ x3 = x4)) ∧
  ((y1 = y2 ∧ y3 = y4) ∨ (y1 = y4 ∧ y2 = y3))

/-- Theorem stating that given three vertices of a rectangle, the fourth vertex is determined --/
theorem fourth_vertex_of_rectangle (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1 = 2 ∧ y1 = 1)
  (h2 : x2 = 4 ∧ y2 = 1)
  (h3 : x3 = 2 ∧ y3 = 5) :
  ∃ (r : Rectangle), is_rectangle r ∧ 
    r.v1 = (x1, y1) ∧ r.v2 = (x2, y2) ∧ r.v3 = (x3, y3) ∧ r.v4 = (4, 5) := by
  sorry

#check fourth_vertex_of_rectangle

end NUMINAMATH_CALUDE_fourth_vertex_of_rectangle_l2566_256614


namespace NUMINAMATH_CALUDE_robert_kicks_before_break_l2566_256662

/-- The number of kicks Robert took before the break -/
def kicks_before_break (total : ℕ) (after_break : ℕ) (remaining : ℕ) : ℕ :=
  total - (after_break + remaining)

/-- Theorem stating that Robert took 43 kicks before the break -/
theorem robert_kicks_before_break :
  kicks_before_break 98 36 19 = 43 := by
  sorry

end NUMINAMATH_CALUDE_robert_kicks_before_break_l2566_256662


namespace NUMINAMATH_CALUDE_expression_equals_two_l2566_256693

theorem expression_equals_two :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l2566_256693


namespace NUMINAMATH_CALUDE_photo_arrangements_l2566_256676

/-- The number of ways 7 students can stand in a line for a photo, 
    given specific constraints on their positions. -/
theorem photo_arrangements (n : Nat) (h1 : n = 7) : 
  (∃ (arrangement_count : Nat), 
    (∀ (A B C : Nat) (others : Finset Nat), 
      A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
      others.card = n - 3 ∧
      (∀ x ∈ others, x ≠ A ∧ x ≠ B ∧ x ≠ C) ∧
      (∀ perm : List Nat, perm.length = n →
        (perm.indexOf A).succ ≠ perm.indexOf B ∧
        (perm.indexOf A).pred ≠ perm.indexOf B ∧
        ((perm.indexOf B).succ = perm.indexOf C ∨
         (perm.indexOf B).pred = perm.indexOf C)) →
    arrangement_count = 1200)) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2566_256676


namespace NUMINAMATH_CALUDE_cone_radius_theorem_l2566_256650

/-- For a cone with radius r, slant height 2r, and lateral surface area equal to half its volume, prove that r = 4√3 -/
theorem cone_radius_theorem (r : ℝ) (h : ℝ) : 
  r > 0 → 
  h > 0 → 
  (2 * r)^2 = r^2 + h^2 →  -- Pythagorean theorem for the slant height
  π * r * (2 * r) = (1/2) * ((1/3) * π * r^2 * h) →  -- Lateral surface area = 1/2 * Volume
  r = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_theorem_l2566_256650


namespace NUMINAMATH_CALUDE_probability_two_primary_schools_l2566_256692

/-- Represents the types of schools in the region -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → ℕ
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in the stratified sample -/
def selectedSchools : SchoolType → ℕ
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools in the stratified sample -/
def totalSampleSize : ℕ := 6

/-- The number of schools to be randomly selected from the sample -/
def selectionSize : ℕ := 2

/-- Theorem stating that the probability of selecting two primary schools
    from the stratified sample is 1/5 -/
theorem probability_two_primary_schools :
  (selectedSchools SchoolType.Primary).choose selectionSize /
  (totalSampleSize.choose selectionSize) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_primary_schools_l2566_256692


namespace NUMINAMATH_CALUDE_solution_implies_result_l2566_256637

theorem solution_implies_result (a b x y : ℝ) 
  (h1 : x = 1)
  (h2 : y = -2)
  (h3 : 2*a*x - 3*y = 10 - b)
  (h4 : a*x - b*y = -1) :
  (b - a)^3 = -125 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_result_l2566_256637


namespace NUMINAMATH_CALUDE_largest_integer_cube_less_than_triple_square_l2566_256689

theorem largest_integer_cube_less_than_triple_square :
  ∀ n : ℤ, n > 2 → n^3 ≥ 3*n^2 ∧ 2^3 < 3*2^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_cube_less_than_triple_square_l2566_256689


namespace NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l2566_256634

theorem tan_plus_3sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  (sin_30 / cos_30) + 3 * sin_30 = 2 + 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l2566_256634


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2566_256669

-- Define the present ages
def sons_present_age : ℕ := 24
def mans_present_age : ℕ := sons_present_age + 26

-- Define the ages in two years
def sons_future_age : ℕ := sons_present_age + 2
def mans_future_age : ℕ := mans_present_age + 2

-- Define the ratio
def age_ratio : ℚ := mans_future_age / sons_future_age

theorem age_ratio_is_two_to_one : age_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2566_256669


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2566_256688

theorem smallest_solution_of_equation :
  let x : ℝ := (5 - Real.sqrt 33) / 2
  (1 / (x - 1) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 1) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2566_256688


namespace NUMINAMATH_CALUDE_not_necessarily_parallel_l2566_256649

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the theorem
theorem not_necessarily_parallel
  (m : Line) (α β : Plane)
  (h1 : parallel_plane_plane α β)
  (h2 : parallel_line_plane m α) :
  ¬ (∀ m α β, parallel_plane_plane α β → parallel_line_plane m α → parallel_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_parallel_l2566_256649


namespace NUMINAMATH_CALUDE_all_lines_pass_through_common_point_l2566_256673

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Checks if three numbers form a geometric progression -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = a * r^2

theorem all_lines_pass_through_common_point :
  ∀ l : Line, isGeometricProgression l.a l.b l.c →
  l.contains (-1) 1 := by sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_common_point_l2566_256673


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_with_fifth_power_less_than_2000_l2566_256604

theorem greatest_multiple_of_four_with_fifth_power_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^5 < 2000 ∧ ∀ y : ℕ, y > 0 → 4 ∣ y → y^5 < 2000 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_with_fifth_power_less_than_2000_l2566_256604


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2566_256605

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ¬(1/a < 1 → a > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2566_256605


namespace NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l2566_256677

/-- Given that 30 glasses of lemonade were served from 6 pitchers, 
    prove that each pitcher can serve 5 glasses. -/
theorem lemonade_pitcher_capacity 
  (total_glasses : ℕ) 
  (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) 
  (h2 : total_pitchers = 6) : 
  total_glasses / total_pitchers = 5 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l2566_256677


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2566_256642

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

/-- The theorem stating the relationship between the maximum value of f and the value of a -/
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) ↔
  a = Real.sqrt 6 + 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2566_256642


namespace NUMINAMATH_CALUDE_harry_photo_reorganization_l2566_256654

/-- Represents a photo album organization system -/
structure PhotoAlbumSystem where
  initialAlbums : Nat
  pagesPerAlbum : Nat
  initialPhotosPerPage : Nat
  newPhotosPerPage : Nat
  filledAlbums : Nat

/-- Calculates the number of photos on the last page of the partially filled album -/
def photosOnLastPage (system : PhotoAlbumSystem) : Nat :=
  let totalPhotos := system.initialAlbums * system.pagesPerAlbum * system.initialPhotosPerPage
  let totalPagesNeeded := (totalPhotos + system.newPhotosPerPage - 1) / system.newPhotosPerPage
  let pagesInFilledAlbums := system.filledAlbums * system.pagesPerAlbum
  let remainingPhotos := totalPhotos - pagesInFilledAlbums * system.newPhotosPerPage
  remainingPhotos % system.newPhotosPerPage

theorem harry_photo_reorganization :
  let system : PhotoAlbumSystem := {
    initialAlbums := 10,
    pagesPerAlbum := 35,
    initialPhotosPerPage := 4,
    newPhotosPerPage := 8,
    filledAlbums := 6
  }
  photosOnLastPage system = 0 := by
  sorry

end NUMINAMATH_CALUDE_harry_photo_reorganization_l2566_256654


namespace NUMINAMATH_CALUDE_cannot_reach_goal_l2566_256617

/-- Represents the types of donuts --/
inductive DonutType
  | Plain
  | Glazed
  | Chocolate

/-- Represents the cost and price information for donuts --/
structure DonutInfo where
  costPerDozen : ℝ
  sellingPrice : ℝ

/-- The goal amount to be raised --/
def goalAmount : ℝ := 96

/-- The maximum number of dozens that can be bought --/
def maxDozens : ℕ := 6

/-- The number of donut types --/
def numTypes : ℕ := 3

/-- The donut information for each type --/
def donutData : DonutType → DonutInfo
  | DonutType.Plain => { costPerDozen := 2.4, sellingPrice := 1 }
  | DonutType.Glazed => { costPerDozen := 3.6, sellingPrice := 1.5 }
  | DonutType.Chocolate => { costPerDozen := 4.8, sellingPrice := 2 }

/-- Calculate the profit for a given number of dozens of a specific donut type --/
def profitForType (t : DonutType) (dozens : ℝ) : ℝ :=
  let info := donutData t
  dozens * (12 * info.sellingPrice - info.costPerDozen)

/-- The main theorem stating that the goal cannot be reached --/
theorem cannot_reach_goal :
  ∀ x : ℝ, x > 0 → x ≤ (maxDozens / numTypes : ℝ) →
  (profitForType DonutType.Plain x +
   profitForType DonutType.Glazed x +
   profitForType DonutType.Chocolate x) < goalAmount :=
sorry

end NUMINAMATH_CALUDE_cannot_reach_goal_l2566_256617


namespace NUMINAMATH_CALUDE_constant_d_value_l2566_256696

theorem constant_d_value (x y d : ℚ) 
  (h1 : (7 * x + 4 * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = d / 2) : 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_constant_d_value_l2566_256696


namespace NUMINAMATH_CALUDE_product_equality_l2566_256694

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 21.3 = 34.08 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2566_256694


namespace NUMINAMATH_CALUDE_unique_right_triangle_l2566_256665

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Theorem stating that among the given sets, only {1, 1, √2} forms a right triangle --/
theorem unique_right_triangle :
  ¬ is_right_triangle 4 5 6 ∧
  is_right_triangle 1 1 (Real.sqrt 2) ∧
  ¬ is_right_triangle 6 8 11 ∧
  ¬ is_right_triangle 5 12 23 :=
by sorry

#check unique_right_triangle

end NUMINAMATH_CALUDE_unique_right_triangle_l2566_256665


namespace NUMINAMATH_CALUDE_b_minus_c_equals_one_l2566_256640

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h2 : A = 9 - 4)
  (h3 : B = A + 5)
  (h4 : C - 8 = 1) : 
  B - C = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_c_equals_one_l2566_256640


namespace NUMINAMATH_CALUDE_max_value_theorem_l2566_256647

theorem max_value_theorem (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) 
  (h3 : ∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 → 1 ≤ x + y ∧ x + y ≤ 4 → x - 2*y ≤ a - 2*b) :
  8*a + 2002*b = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2566_256647


namespace NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_three_l2566_256600

/-- The number of integers from 1 to 50 inclusive -/
def total_numbers : ℕ := 50

/-- The number of multiples of 3 from 1 to 50 inclusive -/
def multiples_of_three : ℕ := 16

/-- The probability of choosing a number that is not a multiple of 3 -/
def prob_not_multiple : ℚ := (total_numbers - multiples_of_three) / total_numbers

/-- The probability of choosing at least one multiple of 3 in two selections -/
def prob_at_least_one_multiple : ℚ := 1 - prob_not_multiple ^ 2

theorem prob_at_least_one_multiple_of_three :
  prob_at_least_one_multiple = 336 / 625 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_three_l2566_256600


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2566_256679

-- Define ticket prices
def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

-- Define discounts
def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10

-- Define group composition
def num_adults : ℕ := 4
def num_children : ℕ := 2
def num_seniors : ℕ := 3
def num_teens : ℕ := 1
def num_adult_nephews : ℕ := 1

-- Define the total cost function
def total_cost : ℝ :=
  (num_adults * adult_price) +
  (num_children * child_price) +
  (num_seniors * senior_price) +
  (num_teens * adult_price) +
  (num_adult_nephews * adult_price) -
  (husband_discount * adult_price) -
  (parents_discount * (2 * senior_price)) -
  (nephew_discount * adult_price)

-- Theorem statement
theorem total_cost_is_correct :
  total_cost = 110.45 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2566_256679


namespace NUMINAMATH_CALUDE_infimum_of_expression_l2566_256658

theorem infimum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (2 / b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / (2 * a₀)) + (2 / b₀) = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_infimum_of_expression_l2566_256658


namespace NUMINAMATH_CALUDE_simplify_square_root_difference_l2566_256659

theorem simplify_square_root_difference : (Real.sqrt 8 - Real.sqrt (4 + 1/2))^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_difference_l2566_256659


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_range_l2566_256618

/-- A function f(x) = 2ax² - x - 1 has exactly one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 1) ∧ 2 * a * x^2 - x - 1 = 0

/-- The theorem stating that if f(x) = 2ax² - x - 1 has exactly one zero in (0, 1), 
    then a is in the interval (1, +∞) -/
theorem unique_zero_implies_a_range :
  ∀ a : ℝ, has_unique_zero_in_interval a → a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_range_l2566_256618


namespace NUMINAMATH_CALUDE_ella_toast_combinations_l2566_256678

/-- The number of different kinds of spreads -/
def num_spreads : ℕ := 12

/-- The number of different kinds of toppings -/
def num_toppings : ℕ := 8

/-- The number of types of bread -/
def num_breads : ℕ := 3

/-- The number of spreads chosen for each toast -/
def spreads_per_toast : ℕ := 1

/-- The number of toppings chosen for each toast -/
def toppings_per_toast : ℕ := 2

/-- The number of breads chosen for each toast -/
def breads_per_toast : ℕ := 1

/-- The total number of different toasts Ella can make -/
def total_toasts : ℕ := num_spreads * (num_toppings.choose toppings_per_toast) * num_breads

theorem ella_toast_combinations :
  total_toasts = 1008 := by sorry

end NUMINAMATH_CALUDE_ella_toast_combinations_l2566_256678


namespace NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l2566_256626

/-- Two concentric circles with center Q -/
structure ConcentricCircles where
  center : Point
  smallerRadius : ℝ
  largerRadius : ℝ
  smallerRadius_pos : 0 < smallerRadius
  largerRadius_pos : 0 < largerRadius
  smallerRadius_lt_largerRadius : smallerRadius < largerRadius

/-- The arc length of a circle given its radius and central angle (in radians) -/
def arcLength (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

theorem area_ratio_of_concentric_circles 
  (circles : ConcentricCircles) 
  (h : arcLength circles.smallerRadius (π/3) = arcLength circles.largerRadius (π/6)) : 
  (circles.smallerRadius^2) / (circles.largerRadius^2) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_area_ratio_of_concentric_circles_l2566_256626


namespace NUMINAMATH_CALUDE_library_book_count_l2566_256608

/-- Represents the library with its bookshelves and books. -/
structure Library where
  num_bookshelves : Nat
  floors_per_bookshelf : Nat
  left_position : Nat
  right_position : Nat

/-- Calculates the total number of books in the library. -/
def total_books (lib : Library) : Nat :=
  let books_per_floor := lib.left_position + lib.right_position - 1
  let books_per_bookshelf := books_per_floor * lib.floors_per_bookshelf
  books_per_bookshelf * lib.num_bookshelves

/-- Theorem stating the total number of books in the library. -/
theorem library_book_count :
  ∀ (lib : Library),
    lib.num_bookshelves = 28 →
    lib.floors_per_bookshelf = 6 →
    lib.left_position = 9 →
    lib.right_position = 11 →
    total_books lib = 3192 := by
  sorry

#eval total_books ⟨28, 6, 9, 11⟩

end NUMINAMATH_CALUDE_library_book_count_l2566_256608


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_white_l2566_256603

/-- The probability of drawing a ball that is neither red nor white from a bag containing
    2 red balls, 3 white balls, and 5 yellow balls. -/
theorem prob_neither_red_nor_white :
  let total_balls : ℕ := 2 + 3 + 5
  let yellow_balls : ℕ := 5
  (yellow_balls : ℚ) / total_balls = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_white_l2566_256603


namespace NUMINAMATH_CALUDE_square_area_problem_l2566_256699

theorem square_area_problem (x : ℝ) (h : 4 * x^2 = 240) : x^2 + (2*x)^2 + x^2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l2566_256699


namespace NUMINAMATH_CALUDE_points_five_units_from_negative_three_l2566_256601

theorem points_five_units_from_negative_three (x : ℝ) : 
  (|x - (-3)| = 5) ↔ (x = -8 ∨ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_points_five_units_from_negative_three_l2566_256601


namespace NUMINAMATH_CALUDE_cube_difference_given_difference_l2566_256652

theorem cube_difference_given_difference (x : ℝ) (h : x - 1/x = 5) :
  x^3 - 1/x^3 = 140 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_given_difference_l2566_256652


namespace NUMINAMATH_CALUDE_dogs_with_spots_l2566_256682

theorem dogs_with_spots (total_dogs : ℚ) (pointy_ears : ℚ) : ℚ :=
  by
  have h1 : pointy_ears = total_dogs / 5 := by sorry
  have h2 : total_dogs = pointy_ears * 5 := by sorry
  have h3 : total_dogs / 2 = (pointy_ears * 5) / 2 := by sorry
  exact (pointy_ears * 5) / 2

#check dogs_with_spots

end NUMINAMATH_CALUDE_dogs_with_spots_l2566_256682


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2566_256611

theorem cubic_root_sum_cubes (a b c r s t : ℝ) : 
  r^3 - a*r^2 + b*r - c = 0 →
  s^3 - a*s^2 + b*s - c = 0 →
  t^3 - a*t^2 + b*t - c = 0 →
  r^3 + s^3 + t^3 = a^3 - 3*a*b + 3*c :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2566_256611


namespace NUMINAMATH_CALUDE_third_term_value_l2566_256607

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the specific sequence with given conditions
def specific_sequence (a : ℕ → ℝ) : Prop :=
  geometric_sequence a ∧ a 1 = -2 ∧ a 5 = -8

-- Theorem statement
theorem third_term_value (a : ℕ → ℝ) (h : specific_sequence a) : a 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l2566_256607


namespace NUMINAMATH_CALUDE_acid_dilution_l2566_256633

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution 
    results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l2566_256633


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2566_256667

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- Define the variables
variables {a t x₁ x₂ : ℝ}

-- State the theorem
theorem isosceles_right_triangle_property
  (h1 : f a x₁ = 0)
  (h2 : f a x₂ = 0)
  (h3 : x₁ < x₂)
  (h4 : ∃ (c : ℝ), f a c = (x₂ - x₁) / 2 ∧ c = (x₁ + x₂) / 2)
  (h5 : t = Real.sqrt ((x₂ - 1) / (x₁ - 1)))
  : a * t - (a + t) = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_isosceles_right_triangle_property_l2566_256667


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2566_256638

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - (k - 1) = 0) ↔ k ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2566_256638


namespace NUMINAMATH_CALUDE_fourth_angle_is_85_l2566_256674

/-- A quadrilateral with three known angles -/
structure Quadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  sum_360 : angle1 + angle2 + angle3 + angle4 = 360

/-- The theorem stating that the fourth angle is 85° -/
theorem fourth_angle_is_85 (q : Quadrilateral) 
  (h1 : q.angle1 = 75) 
  (h2 : q.angle2 = 80) 
  (h3 : q.angle3 = 120) : 
  q.angle4 = 85 := by
  sorry


end NUMINAMATH_CALUDE_fourth_angle_is_85_l2566_256674


namespace NUMINAMATH_CALUDE_value_of_fraction_l2566_256683

-- Define the real numbers
variable (a₁ a₂ b₁ b₂ : ℝ)

-- Define the arithmetic sequence condition
def is_arithmetic_sequence : Prop :=
  ∃ d : ℝ, a₁ - (-1) = d ∧ a₂ - a₁ = d ∧ (-4) - a₂ = d

-- Define the geometric sequence condition
def is_geometric_sequence : Prop :=
  ∃ r : ℝ, b₁ / (-1) = r ∧ b₂ / b₁ = r ∧ (-8) / b₂ = r

-- Theorem statement
theorem value_of_fraction (h1 : is_arithmetic_sequence a₁ a₂)
                          (h2 : is_geometric_sequence b₁ b₂) :
  (a₂ - a₁) / b₂ = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_fraction_l2566_256683


namespace NUMINAMATH_CALUDE_frog_jump_parity_l2566_256602

def frog_jump (n : ℕ) (t : ℕ) : ℕ :=
  (t * (t + 1) / 2 - 1) % n

theorem frog_jump_parity (n : ℕ) (h1 : n > 1) :
  (∀ r : ℕ, r < n → ∃ t : ℕ, frog_jump n t = r) →
  Even n :=
sorry

end NUMINAMATH_CALUDE_frog_jump_parity_l2566_256602


namespace NUMINAMATH_CALUDE_line_and_circle_equations_l2566_256625

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4*x - 3*y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point of line1 and line2
def intersection_point : ℝ × ℝ := (2, 1)

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Main theorem
theorem line_and_circle_equations :
  ∃ (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop),
    -- l passes through the intersection of line1 and line2
    (l (intersection_point.1) (intersection_point.2)) ∧
    -- l is perpendicular to line3
    (∀ x y, l x y → line3 x y → (x + 1 = y)) ∧
    -- C passes through (1,0)
    (C 1 0) ∧
    -- Center of C is on positive x-axis
    (∃ a > 0, ∀ x y, C x y ↔ (x - a)^2 + y^2 = a^2) ∧
    -- Chord intercepted by l on C has length 2√2
    (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) ∧
    -- l is the line y = x - 1
    (∀ x y, l x y ↔ line_l x y) ∧
    -- C is the circle (x-3)^2 + y^2 = 4
    (∀ x y, C x y ↔ circle_C x y) :=
by sorry

end NUMINAMATH_CALUDE_line_and_circle_equations_l2566_256625


namespace NUMINAMATH_CALUDE_inverse_proportion_constant_difference_l2566_256632

/-- Given two inverse proportion functions and points satisfying certain conditions, 
    prove that the difference of their constants is 4. -/
theorem inverse_proportion_constant_difference 
  (k₁ k₂ : ℝ) 
  (f₁ : ℝ → ℝ) 
  (f₂ : ℝ → ℝ) 
  (a b : ℝ) 
  (h₁ : ∀ x, f₁ x = k₁ / x) 
  (h₂ : ∀ x, f₂ x = k₂ / x) 
  (h₃ : |f₁ a - f₂ a| = 2) 
  (h₄ : |f₂ b - f₁ b| = 3) 
  (h₅ : |b - a| = 10/3) : 
  k₂ - k₁ = 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_constant_difference_l2566_256632


namespace NUMINAMATH_CALUDE_orange_shirt_cost_l2566_256670

-- Define the number of students in each grade
def kindergartners : ℕ := 101
def first_graders : ℕ := 113
def second_graders : ℕ := 107
def third_graders : ℕ := 108

-- Define the cost of shirts for each grade (in cents to avoid floating-point issues)
def yellow_shirt_cost : ℕ := 500  -- $5.00
def blue_shirt_cost : ℕ := 560    -- $5.60
def green_shirt_cost : ℕ := 525   -- $5.25

-- Define the total amount spent by P.T.O. (in cents)
def total_spent : ℕ := 231700  -- $2,317.00

-- Theorem to prove
theorem orange_shirt_cost :
  (total_spent
    - (first_graders * yellow_shirt_cost
    + second_graders * blue_shirt_cost
    + third_graders * green_shirt_cost))
  / kindergartners = 580 := by
  sorry

end NUMINAMATH_CALUDE_orange_shirt_cost_l2566_256670


namespace NUMINAMATH_CALUDE_modified_rectangle_area_l2566_256623

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem about the area of a modified rectangle --/
theorem modified_rectangle_area 
  (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (modified : Rectangle), 
    (modified.length = original.length ∧ modified.width = original.width - 2) ∨
    (modified.length = original.length - 2 ∧ modified.width = original.width) ∧
    area modified = 15) :
  ∃ (final : Rectangle), 
    ((h2.choose.length = original.length ∧ h2.choose.width = original.width - 2) →
      final.length = original.length - 2 ∧ final.width = original.width) ∧
    ((h2.choose.length = original.length - 2 ∧ h2.choose.width = original.width) →
      final.length = original.length ∧ final.width = original.width - 2) ∧
    area final = 7 := by
  sorry

end NUMINAMATH_CALUDE_modified_rectangle_area_l2566_256623


namespace NUMINAMATH_CALUDE_building_height_ratio_l2566_256628

/-- Given a flagpole and two buildings under similar shadow conditions, 
    this theorem proves that the ratio of the heights of Building A to Building B is 5:6. -/
theorem building_height_ratio 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_a_shadow : ℝ) 
  (building_b_shadow : ℝ) 
  (flagpole_height_pos : 0 < flagpole_height)
  (flagpole_shadow_pos : 0 < flagpole_shadow)
  (building_a_shadow_pos : 0 < building_a_shadow)
  (building_b_shadow_pos : 0 < building_b_shadow)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_a_shadow : building_a_shadow = 60)
  (h_building_b_shadow : building_b_shadow = 72) :
  (flagpole_height / flagpole_shadow * building_a_shadow) / 
  (flagpole_height / flagpole_shadow * building_b_shadow) = 5 / 6 := by
  sorry

#check building_height_ratio

end NUMINAMATH_CALUDE_building_height_ratio_l2566_256628


namespace NUMINAMATH_CALUDE_divisibility_condition_l2566_256672

theorem divisibility_condition (m n : ℕ) : 
  m ≥ 1 → n ≥ 1 → 
  (m * n) ∣ (3^m + 1) → 
  (m * n) ∣ (3^n + 1) → 
  (m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2566_256672


namespace NUMINAMATH_CALUDE_hobby_gender_independence_l2566_256619

/-- Represents the contingency table data -/
structure ContingencyTable where
  total : ℕ
  male_hobby : ℕ
  female_no_hobby : ℕ

/-- Calculates the chi-square value for the independence test -/
def chi_square (ct : ContingencyTable) : ℝ :=
  sorry

/-- Calculates the probability of selecting k males from those without a hobby -/
def prob_select_males (ct : ContingencyTable) (k : ℕ) : ℚ :=
  sorry

/-- Calculates the expected number of males selected -/
def expected_males (ct : ContingencyTable) : ℚ :=
  sorry

/-- Main theorem encompassing all parts of the problem -/
theorem hobby_gender_independence (ct : ContingencyTable) 
  (h1 : ct.total = 100) 
  (h2 : ct.male_hobby = 30) 
  (h3 : ct.female_no_hobby = 10) : 
  chi_square ct < 6.635 ∧ 
  prob_select_males ct 0 = 3/29 ∧ 
  prob_select_males ct 1 = 40/87 ∧ 
  prob_select_males ct 2 = 38/87 ∧
  expected_males ct = 4/3 :=
sorry

end NUMINAMATH_CALUDE_hobby_gender_independence_l2566_256619


namespace NUMINAMATH_CALUDE_incorrect_elimination_process_l2566_256631

/-- Given a system of two linear equations in two variables, 
    prove that a specific elimination process is incorrect. -/
theorem incorrect_elimination_process 
  (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  ¬ (∃ (k : ℝ), 2 * a + b + 2 * (a - b) = 7 + 2 * k ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_incorrect_elimination_process_l2566_256631


namespace NUMINAMATH_CALUDE_odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l2566_256629

open Set
open Function
open BigOperators
open Filter

def diverges_to_infinity (s : ℕ → ℝ) : Prop :=
  ∀ M : ℝ, M > 0 → ∃ N : ℕ, ∀ n : ℕ, n > N → s n > M

theorem odd_reciprocal_sum_diverges :
  diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, 1 / (2 * k + 1 : ℝ)) :=
sorry

theorem exists_rearrangement_alternating_harmonic_diverges :
  ∃ f : ℕ → ℕ, Bijective f ∧
    diverges_to_infinity (λ n : ℕ => ∑ k in Finset.range n, (-1 : ℝ)^(f.invFun k - 1) / f.invFun k) :=
sorry

end NUMINAMATH_CALUDE_odd_reciprocal_sum_diverges_exists_rearrangement_alternating_harmonic_diverges_l2566_256629


namespace NUMINAMATH_CALUDE_lineup_count_l2566_256651

/-- The number of ways to choose a lineup from a basketball team with specific constraints. -/
def chooseLineup (totalPlayers : ℕ) (twinCount : ℕ) (tripletCount : ℕ) (lineupSize : ℕ) : ℕ :=
  let nonSpecialPlayers := totalPlayers - twinCount - tripletCount
  let noSpecial := Nat.choose nonSpecialPlayers lineupSize
  let oneTriplet := tripletCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTwin := twinCount * Nat.choose nonSpecialPlayers (lineupSize - 1)
  let oneTripletOneTwin := tripletCount * twinCount * Nat.choose nonSpecialPlayers (lineupSize - 2)
  noSpecial + oneTriplet + oneTwin + oneTripletOneTwin

/-- The theorem stating the number of ways to choose the lineup under given constraints. -/
theorem lineup_count :
  chooseLineup 16 2 3 5 = 3102 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l2566_256651


namespace NUMINAMATH_CALUDE_max_constant_inequality_l2566_256687

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → Real.sqrt x + Real.sqrt y ≤ b) → b ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l2566_256687


namespace NUMINAMATH_CALUDE_largest_of_three_l2566_256646

theorem largest_of_three (a b c : ℝ) : 
  let x₁ := a
  let x₂ := if b > x₁ then b else x₁
  let x₃ := if c > x₂ then c else x₂
  x₃ = max a (max b c) := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_l2566_256646


namespace NUMINAMATH_CALUDE_village_x_current_population_l2566_256697

/-- The current population of Village X -/
def village_x_population : ℕ := sorry

/-- The yearly decrease in Village X's population -/
def village_x_decrease_rate : ℕ := 1200

/-- The current population of Village Y -/
def village_y_population : ℕ := 42000

/-- The yearly increase in Village Y's population -/
def village_y_increase_rate : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 16

theorem village_x_current_population :
  village_x_population = 
    village_y_population + 
    village_y_increase_rate * years_until_equal + 
    village_x_decrease_rate * years_until_equal := by
  sorry

end NUMINAMATH_CALUDE_village_x_current_population_l2566_256697


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2566_256609

variable (a : ℝ)

theorem polynomial_simplification :
  ((-a^3)^2 * a^3 - 4*a^2 * a^7 = -3*a^9) ∧
  ((2*a + 1) * (-2*a + 1) = 4*a^2 - 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2566_256609


namespace NUMINAMATH_CALUDE_lifeguard_swim_time_l2566_256675

/-- Proves the time spent swimming front crawl given total distance, speeds, and total time -/
theorem lifeguard_swim_time 
  (total_distance : ℝ) 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : breaststroke_speed = 35)
  (h4 : total_time = 12) :
  ∃ (front_crawl_time : ℝ), 
    front_crawl_time * front_crawl_speed + 
    (total_time - front_crawl_time) * breaststroke_speed = total_distance ∧ 
    front_crawl_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_lifeguard_swim_time_l2566_256675


namespace NUMINAMATH_CALUDE_star_equation_solutions_l2566_256606

-- Define the * operation
def star (a b : ℝ) : ℝ := a * (a + b) + b

-- Theorem statement
theorem star_equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧ 
  star a₁ 2.5 = 28.5 ∧ 
  star a₂ 2.5 = 28.5 ∧
  (a₁ = 4 ∨ a₁ = -13/2) ∧
  (a₂ = 4 ∨ a₂ = -13/2) :=
sorry

end NUMINAMATH_CALUDE_star_equation_solutions_l2566_256606


namespace NUMINAMATH_CALUDE_jared_car_count_l2566_256690

theorem jared_car_count : ∀ (j a f : ℕ),
  (j : ℝ) = 0.85 * a →
  a = f + 7 →
  j + a + f = 983 →
  j = 295 :=
by sorry

end NUMINAMATH_CALUDE_jared_car_count_l2566_256690


namespace NUMINAMATH_CALUDE_welders_proof_l2566_256624

/-- Represents the initial number of welders -/
def initial_welders : ℕ := 12

/-- Represents the number of days initially needed to complete the order -/
def initial_days : ℕ := 3

/-- Represents the number of welders that leave after the first day -/
def welders_left : ℕ := 9

/-- Represents the additional days needed by remaining welders to complete the order -/
def additional_days : ℕ := 8

/-- Proves that the initial number of welders is correct given the conditions -/
theorem welders_proof :
  (initial_welders - welders_left) * additional_days = initial_welders * (initial_days - 1) :=
by sorry

end NUMINAMATH_CALUDE_welders_proof_l2566_256624


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l2566_256691

theorem min_value_quadratic_sum (a b c t k : ℝ) (hsum : a + b + c = t) (hk : k > 0) :
  k * a^2 + b^2 + k * c^2 ≥ k * t^2 / (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l2566_256691


namespace NUMINAMATH_CALUDE_motorbike_time_difference_l2566_256620

theorem motorbike_time_difference :
  let distance : ℝ := 960
  let speed_slow : ℝ := 60
  let speed_fast : ℝ := 64
  let time_slow : ℝ := distance / speed_slow
  let time_fast : ℝ := distance / speed_fast
  time_slow - time_fast = 1 := by
  sorry

end NUMINAMATH_CALUDE_motorbike_time_difference_l2566_256620


namespace NUMINAMATH_CALUDE_black_cards_count_l2566_256664

theorem black_cards_count (total_cards : Nat) (red_cards : Nat) (clubs : Nat)
  (h_total : total_cards = 13)
  (h_red : red_cards = 6)
  (h_clubs : clubs = 6)
  (h_suits : ∃ (spades diamonds hearts : Nat), 
    spades + diamonds + hearts + clubs = total_cards ∧
    diamonds = 2 * spades ∧
    hearts = 2 * diamonds) :
  clubs + (total_cards - red_cards - clubs) = 7 := by
  sorry

end NUMINAMATH_CALUDE_black_cards_count_l2566_256664


namespace NUMINAMATH_CALUDE_class_size_calculation_l2566_256627

theorem class_size_calculation (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 13 → average_increase = 1/2 → 
  (mark_increase : ℚ) / average_increase = 26 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l2566_256627


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l2566_256660

/-- Given two lines that pass through a common point, prove that the line
    passing through the points defined by their coefficients has a specific equation. -/
theorem line_through_coefficient_points
  (a₁ b₁ a₂ b₂ : ℝ)
  (h₁ : 2 * a₁ + 3 * b₁ + 1 = 0)
  (h₂ : 2 * a₂ + 3 * b₂ + 1 = 0) :
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l2566_256660


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2566_256641

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3

/-- The scaled quadratic function -/
def g (x : ℝ) : ℝ := 4 * f x

/-- The vertex form of a quadratic function -/
def vertex_form (m h p : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + p

theorem quadratic_vertex_form_h :
  ∃ (m p : ℝ), ∀ x, g x = vertex_form m (-5/4) p x :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l2566_256641


namespace NUMINAMATH_CALUDE_max_value_of_g_l2566_256630

/-- The function g(x) = 4x - x^4 --/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The theorem stating that the maximum value of g(x) on [0, √4] is 3 --/
theorem max_value_of_g :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  g x = 3 ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ g x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l2566_256630


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l2566_256680

/-- Represents a person who always tells the truth -/
structure TruthfulPerson where
  answer : Prop → Bool
  always_truthful : ∀ p, answer p = p

/-- Represents a question that can be asked -/
structure Question where
  ask : TruthfulPerson → Bool

/-- Represents the state of a day, including whether any questions have been asked -/
structure DayState where
  question_asked : Bool

/-- The theorem stating that there exists a question that yields different answers when asked twice -/
theorem exists_question_with_different_answers :
  ∃ (q : Question), ∀ (p : TruthfulPerson),
    ∃ (d1 d2 : DayState),
      d1.question_asked = false ∧
      d2.question_asked = true ∧
      q.ask p ≠ q.ask p :=
sorry

end NUMINAMATH_CALUDE_exists_question_with_different_answers_l2566_256680


namespace NUMINAMATH_CALUDE_negative_three_squared_l2566_256648

theorem negative_three_squared : (-3 : ℤ) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_l2566_256648


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2566_256653

/-- The y-intercept of the line 3x + 5y = 20 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x + 5 * y = 20 → x = 0 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2566_256653


namespace NUMINAMATH_CALUDE_six_lines_intersections_l2566_256613

/-- The maximum number of intersection points between n straight lines -/
def max_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The maximum number of intersection points between 6 straight lines is 15 -/
theorem six_lines_intersections :
  max_intersections 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_six_lines_intersections_l2566_256613


namespace NUMINAMATH_CALUDE_other_divisor_is_57_l2566_256616

theorem other_divisor_is_57 : 
  ∃ (x : ℕ), x ≠ 38 ∧ 
  114 % x = 0 ∧ 
  115 % x = 1 ∧
  115 % 38 = 1 ∧
  (∀ y : ℕ, y > x → 114 % y = 0 → y = 38 ∨ y = 114) :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_is_57_l2566_256616
