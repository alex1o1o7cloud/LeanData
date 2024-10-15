import Mathlib

namespace NUMINAMATH_CALUDE_milkshake_leftover_l226_22694

/-- Calculates the amount of milk left over after making milkshakes -/
theorem milkshake_leftover (milk_per_shake ice_cream_per_shake total_milk total_ice_cream : ℕ) :
  milk_per_shake = 4 →
  ice_cream_per_shake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  total_milk - (total_ice_cream / ice_cream_per_shake * milk_per_shake) = 8 := by
  sorry

#check milkshake_leftover

end NUMINAMATH_CALUDE_milkshake_leftover_l226_22694


namespace NUMINAMATH_CALUDE_age_ratio_six_years_ago_l226_22615

/-- Given Henry and Jill's ages, prove their age ratio 6 years ago -/
theorem age_ratio_six_years_ago 
  (henry_age : ℕ) 
  (jill_age : ℕ) 
  (henry_age_eq : henry_age = 20)
  (jill_age_eq : jill_age = 13)
  (sum_ages : henry_age + jill_age = 33)
  (past_multiple : ∃ k : ℕ, henry_age - 6 = k * (jill_age - 6)) :
  (henry_age - 6) / (jill_age - 6) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_six_years_ago_l226_22615


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l226_22628

/-- The lateral surface area of a cone with base radius 3 and slant height 4 is 12π. -/
theorem cone_lateral_surface_area :
  ∀ (r l : ℝ), r = 3 → l = 4 → π * r * l = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l226_22628


namespace NUMINAMATH_CALUDE_equal_reading_time_l226_22612

/-- Represents the reading scenario in Mrs. Reed's English class -/
structure ReadingScenario where
  total_pages : ℕ
  mia_speed : ℕ  -- seconds per page
  leo_speed : ℕ  -- seconds per page
  mia_pages : ℕ

/-- The specific reading scenario from the problem -/
def problem_scenario : ReadingScenario :=
  { total_pages := 840
  , mia_speed := 60
  , leo_speed := 40
  , mia_pages := 336 }

/-- Calculates the total reading time for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Theorem stating that Mia and Leo spend equal time reading in the given scenario -/
theorem equal_reading_time (s : ReadingScenario) (h : s = problem_scenario) :
  reading_time s.mia_pages s.mia_speed = reading_time (s.total_pages - s.mia_pages) s.leo_speed := by
  sorry

#check equal_reading_time

end NUMINAMATH_CALUDE_equal_reading_time_l226_22612


namespace NUMINAMATH_CALUDE_crocus_to_daffodil_ratio_l226_22642

/-- Represents the number of flower bulbs of each type planted by Jane. -/
structure FlowerBulbs where
  tulips : ℕ
  irises : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs. -/
def earnings (bulbs : FlowerBulbs) : ℚ :=
  0.5 * (bulbs.tulips + bulbs.irises + bulbs.daffodils + bulbs.crocus)

/-- Proves that given the conditions, the ratio of crocus bulbs to daffodil bulbs is 3:1. -/
theorem crocus_to_daffodil_ratio 
  (bulbs : FlowerBulbs)
  (h1 : bulbs.tulips = 20)
  (h2 : bulbs.irises = bulbs.tulips / 2)
  (h3 : bulbs.daffodils = 30)
  (h4 : earnings bulbs = 75) :
  bulbs.crocus / bulbs.daffodils = 3 := by
  sorry


end NUMINAMATH_CALUDE_crocus_to_daffodil_ratio_l226_22642


namespace NUMINAMATH_CALUDE_sphere_properties_l226_22635

/-- Proves surface area and volume of a sphere with diameter 10 inches -/
theorem sphere_properties :
  let d : ℝ := 10  -- diameter
  let r : ℝ := d / 2  -- radius
  ∀ (S V : ℝ),  -- surface area and volume
  S = 4 * Real.pi * r^2 →
  V = (4/3) * Real.pi * r^3 →
  S = 100 * Real.pi ∧ V = (500/3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_properties_l226_22635


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l226_22681

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) ≥ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15 ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' → 
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l226_22681


namespace NUMINAMATH_CALUDE_matchstick_length_theorem_l226_22667

/-- Represents a figure made of matchsticks -/
structure MatchstickFigure where
  smallSquareCount : ℕ
  largeSquareCount : ℕ
  totalArea : ℝ

/-- Calculates the total length of matchsticks used in the figure -/
def totalMatchstickLength (figure : MatchstickFigure) : ℝ :=
  sorry

/-- Theorem stating the total length of matchsticks in the given figure -/
theorem matchstick_length_theorem (figure : MatchstickFigure) 
  (h1 : figure.smallSquareCount = 8)
  (h2 : figure.largeSquareCount = 1)
  (h3 : figure.totalArea = 300) :
  totalMatchstickLength figure = 140 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_length_theorem_l226_22667


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l226_22695

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l226_22695


namespace NUMINAMATH_CALUDE_median_in_70_74_interval_l226_22617

/-- Represents a score interval with its lower bound and number of students -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (num_students : ℕ)

/-- Finds the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) : Option ScoreInterval :=
  sorry

theorem median_in_70_74_interval :
  let intervals : List ScoreInterval := [
    ⟨55, 4⟩,
    ⟨60, 8⟩,
    ⟨65, 15⟩,
    ⟨70, 20⟩,
    ⟨75, 18⟩,
    ⟨80, 10⟩
  ]
  let total_students : ℕ := 75
  median_interval intervals = some ⟨70, 20⟩ := by
    sorry

end NUMINAMATH_CALUDE_median_in_70_74_interval_l226_22617


namespace NUMINAMATH_CALUDE_horseshoe_selling_price_l226_22656

/-- Proves that the selling price per set of horseshoes is $50 given the specified conditions. -/
theorem horseshoe_selling_price
  (initial_outlay : ℕ)
  (cost_per_set : ℕ)
  (num_sets : ℕ)
  (profit : ℕ)
  (h1 : initial_outlay = 10000)
  (h2 : cost_per_set = 20)
  (h3 : num_sets = 500)
  (h4 : profit = 5000) :
  ∃ (selling_price : ℕ),
    selling_price * num_sets = initial_outlay + cost_per_set * num_sets + profit ∧
    selling_price = 50 :=
by sorry

end NUMINAMATH_CALUDE_horseshoe_selling_price_l226_22656


namespace NUMINAMATH_CALUDE_divisible_by_five_l226_22631

theorem divisible_by_five (a b : ℕ) : 
  (∃ k : ℕ, a * b = 5 * k) → (∃ m : ℕ, a = 5 * m) ∨ (∃ n : ℕ, b = 5 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l226_22631


namespace NUMINAMATH_CALUDE_red_square_density_l226_22643

/-- A standard rectangle is a rectangle on the coordinate plane with vertices at integer points and edges parallel to the coordinate axes. -/
def StandardRectangle (w h : ℕ) : Prop := sorry

/-- A unit square is a standard rectangle with an area of 1. -/
def UnitSquare : Prop := StandardRectangle 1 1

/-- A coloring of unit squares on the coordinate plane. -/
def Coloring := ℕ → ℕ → Bool

/-- The number of red squares in a standard rectangle. -/
def RedSquares (c : Coloring) (x y w h : ℕ) : ℕ := sorry

theorem red_square_density (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a) 
  (c : Coloring) (h4 : ∀ x y, RedSquares c x y a b + RedSquares c x y b a > 0) :
  ∀ N : ℕ, ∃ x y : ℕ, RedSquares c x y N N ≥ N * N * N / (N - 1) := by sorry

end NUMINAMATH_CALUDE_red_square_density_l226_22643


namespace NUMINAMATH_CALUDE_ship_departure_theorem_l226_22648

/-- Represents the total transit time for a cargo shipment -/
def total_transit_time (navigation_time customs_time delivery_time : ℕ) : ℕ :=
  navigation_time + customs_time + delivery_time

/-- Calculates the departure date given the expected arrival and total transit time -/
def departure_date (days_until_arrival total_transit : ℕ) : ℕ :=
  days_until_arrival + total_transit

/-- Theorem: Given the specified conditions, the ship should have departed 34 days ago -/
theorem ship_departure_theorem (navigation_time customs_time delivery_time days_until_arrival : ℕ)
  (h1 : navigation_time = 21)
  (h2 : customs_time = 4)
  (h3 : delivery_time = 7)
  (h4 : days_until_arrival = 2) :
  departure_date days_until_arrival (total_transit_time navigation_time customs_time delivery_time) = 34 := by
  sorry

#check ship_departure_theorem

end NUMINAMATH_CALUDE_ship_departure_theorem_l226_22648


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l226_22657

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3 → x ≤ y) ∧
  (x = 17) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l226_22657


namespace NUMINAMATH_CALUDE_current_trees_count_l226_22627

/-- The number of dogwood trees to be planted today -/
def trees_planted_today : ℕ := 41

/-- The number of dogwood trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The total number of dogwood trees after planting -/
def total_trees_after_planting : ℕ := 100

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := total_trees_after_planting - trees_planted_today - trees_planted_tomorrow

theorem current_trees_count : current_trees = 39 := by
  sorry

end NUMINAMATH_CALUDE_current_trees_count_l226_22627


namespace NUMINAMATH_CALUDE_triangle_point_distance_l226_22680

-- Define the triangle and points
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 4 ∧ d B C = 5 ∧ d C A = 6

def OnRay (A B D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

def CircumCircle (A B C P : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A P = d B P ∧ d B P = d C P

-- State the theorem
theorem triangle_point_distance (A B C D E F : ℝ × ℝ) :
  Triangle A B C →
  OnRay A B D →
  OnRay A B E →
  CircumCircle A C D F →
  CircumCircle E B C F →
  F ≠ C →
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d D F = 2 →
  d E F = 7 →
  d B E = (5 + 21 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l226_22680


namespace NUMINAMATH_CALUDE_chris_savings_proof_l226_22605

def chris_birthday_savings (grandmother_gift aunt_uncle_gift parents_gift chores_money friend_gift_cost total_after : ℝ) : Prop :=
  let total_received := grandmother_gift + aunt_uncle_gift + parents_gift + chores_money
  let additional_amount := total_received - friend_gift_cost
  let savings_before := total_after - additional_amount
  let percentage_increase := (additional_amount / savings_before) * 100
  savings_before = 144 ∧ percentage_increase = 93.75

theorem chris_savings_proof :
  chris_birthday_savings 25 20 75 30 15 279 :=
sorry

end NUMINAMATH_CALUDE_chris_savings_proof_l226_22605


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l226_22678

/-- A polynomial of degree 5 with coefficients in ℝ -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x + 9

/-- The condition that the polynomial is divisible by (x + 3)(x - 2) -/
def is_divisible (p q : ℝ) : Prop :=
  ∀ x : ℝ, (x + 3 = 0 ∨ x - 2 = 0) → polynomial p q x = 0

/-- The main theorem stating that if the polynomial is divisible by (x + 3)(x - 2),
    then p = -130.5 and q = -277.5 -/
theorem polynomial_divisibility (p q : ℝ) :
  is_divisible p q → p = -130.5 ∧ q = -277.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l226_22678


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l226_22686

/-- Represents the ratio of time spent on different activities -/
structure ActivityRatio :=
  (swimming : ℕ)
  (running : ℕ)
  (gym : ℕ)
  (biking : ℕ)
  (yoga : ℕ)

/-- Calculates the time spent on yoga given the activity ratio and biking time -/
def yoga_time (ratio : ActivityRatio) (biking_time : ℕ) : ℕ :=
  (biking_time * ratio.yoga) / ratio.biking

/-- Theorem stating that given the specific activity ratio and 30 minutes of biking, 
    the time spent on yoga is 24 minutes -/
theorem yoga_time_calculation :
  let ratio : ActivityRatio := {
    swimming := 1,
    running := 2,
    gym := 3,
    biking := 5,
    yoga := 4
  }
  let biking_time : ℕ := 30
  yoga_time ratio biking_time = 24 := by sorry

end NUMINAMATH_CALUDE_yoga_time_calculation_l226_22686


namespace NUMINAMATH_CALUDE_max_permissible_length_l226_22606

/-- A word is permissible if all adjacent letters are different and 
    it's not possible to obtain a word of the form abab by deleting letters, 
    where a and b are different. -/
def Permissible (word : List Char) (alphabet : List Char) : Prop := sorry

/-- The maximum length of a permissible word for an alphabet with n letters -/
def MaxPermissibleLength (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum length of a permissible word for an alphabet with n letters is 2n - 1 -/
theorem max_permissible_length (n : ℕ) : MaxPermissibleLength n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_max_permissible_length_l226_22606


namespace NUMINAMATH_CALUDE_final_solid_properties_l226_22629

/-- Represents a solid shape with faces, edges, and vertices -/
structure Solid where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents a pyramid attached to a face -/
structure Pyramid where
  base_edges : ℕ

/-- Attaches a pyramid to a solid, updating its properties -/
def attach_pyramid (s : Solid) (p : Pyramid) : Solid :=
  { faces := s.faces + p.base_edges - 1
  , edges := s.edges + p.base_edges
  , vertices := s.vertices + 1 }

/-- The initial triangular prism -/
def initial_prism : Solid :=
  { faces := 5, edges := 9, vertices := 6 }

/-- Pyramid attached to triangular face -/
def triangular_pyramid : Pyramid :=
  { base_edges := 3 }

/-- Pyramid attached to quadrilateral face -/
def quadrilateral_pyramid : Pyramid :=
  { base_edges := 4 }

theorem final_solid_properties :
  let s1 := attach_pyramid initial_prism triangular_pyramid
  let final_solid := attach_pyramid s1 quadrilateral_pyramid
  final_solid.faces = 10 ∧
  final_solid.edges = 16 ∧
  final_solid.vertices = 8 ∧
  final_solid.faces + final_solid.edges + final_solid.vertices = 34 := by
  sorry


end NUMINAMATH_CALUDE_final_solid_properties_l226_22629


namespace NUMINAMATH_CALUDE_blocks_standing_final_value_l226_22670

/-- The number of blocks left standing in the final tower -/
def blocks_standing_final (first_stack : ℕ) (second_stack_diff : ℕ) (final_stack_diff : ℕ) 
  (blocks_standing_second : ℕ) (total_fallen : ℕ) : ℕ :=
  let second_stack := first_stack + second_stack_diff
  let final_stack := second_stack + final_stack_diff
  let fallen_first := first_stack
  let fallen_second := second_stack - blocks_standing_second
  let fallen_final := total_fallen - fallen_first - fallen_second
  final_stack - fallen_final

theorem blocks_standing_final_value :
  blocks_standing_final 7 5 7 2 33 = 3 := by sorry

end NUMINAMATH_CALUDE_blocks_standing_final_value_l226_22670


namespace NUMINAMATH_CALUDE_close_interval_for_f_and_g_l226_22685

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being "close functions" on an interval
def are_close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_for_f_and_g :
  are_close_functions f g 2 3 ∧
  ∀ a b, a < 2 ∨ b > 3 → ¬(are_close_functions f g a b) :=
sorry

end NUMINAMATH_CALUDE_close_interval_for_f_and_g_l226_22685


namespace NUMINAMATH_CALUDE_min_value_problem_l226_22623

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^3 = 16/9) :
  3 * x + y ≥ 8/3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀^3 = 16/9 ∧ 3 * x₀ + y₀ = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l226_22623


namespace NUMINAMATH_CALUDE_rice_left_calculation_l226_22644

/-- Calculates the amount of rice left in grams after cooking -/
def rice_left (initial : ℚ) (morning_cooked : ℚ) (evening_fraction : ℚ) : ℚ :=
  let remaining_after_morning := initial - morning_cooked
  let evening_cooked := remaining_after_morning * evening_fraction
  let final_remaining := remaining_after_morning - evening_cooked
  final_remaining * 1000  -- Convert to grams

/-- Theorem stating the amount of rice left after cooking -/
theorem rice_left_calculation :
  rice_left 10 (9/10 * 10) (1/4) = 750 := by
  sorry

#eval rice_left 10 (9/10 * 10) (1/4)

end NUMINAMATH_CALUDE_rice_left_calculation_l226_22644


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l226_22640

-- Define a triangle in Euclidean space
def Triangle : Type := ℝ × ℝ × ℝ

-- Define the function that calculates the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := sorry

-- Theorem stating that the sum of interior angles of any triangle is 180°
theorem sum_of_interior_angles_is_180 (t : Triangle) :
  sum_of_interior_angles t = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_180_l226_22640


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l226_22660

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l226_22660


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l226_22646

/-- The function f(x) = ax^2 + (b+1)x + b - 2 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- A point x is a fixed point of f if f(x) = x -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point 2 (-2) x ↔ x = -1 ∨ x = 2 := by sorry

theorem range_for_two_distinct_fixed_points :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) →
  (0 < a ∧ a < 2) := by sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l226_22646


namespace NUMINAMATH_CALUDE_age_problem_l226_22697

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l226_22697


namespace NUMINAMATH_CALUDE_restaurant_bill_l226_22654

theorem restaurant_bill (n : ℕ) (extra : ℝ) (discount : ℝ) (original_bill : ℝ) :
  n = 10 →
  extra = 3 →
  discount = 10 →
  (n - 1) * ((original_bill - discount) / n + extra) = original_bill - discount →
  original_bill = 180 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_l226_22654


namespace NUMINAMATH_CALUDE_factorial_gcd_property_l226_22621

theorem factorial_gcd_property (m n : ℕ) (h : m > n) :
  Nat.gcd (Nat.factorial n) (Nat.factorial m) = Nat.factorial n := by
  sorry

end NUMINAMATH_CALUDE_factorial_gcd_property_l226_22621


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_minus_one_l226_22632

theorem divisibility_of_power_plus_minus_one (n : ℕ) (h : ¬ 17 ∣ n) :
  17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_minus_one_l226_22632


namespace NUMINAMATH_CALUDE_hyperbola_equation_l226_22674

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is parallel to the line x + 3y + 2√5 = 0
    and one of its foci lies on this line, then a² = 18 and b² = 2 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (-b / a = -1 / 3) →
  (∃ (x : ℝ), x + 3 * 0 + 2 * Real.sqrt 5 = 0 ∧ x^2 = 4 * 5) →
  a^2 = 18 ∧ b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l226_22674


namespace NUMINAMATH_CALUDE_min_value_of_function_l226_22645

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + (x + 1)⁻¹ ≥ 1 ∧ (x + (x + 1)⁻¹ = 1 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l226_22645


namespace NUMINAMATH_CALUDE_eighteen_gon_symmetry_sum_l226_22690

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry of a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ :=
  360 / n

theorem eighteen_gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_gon_symmetry_sum_l226_22690


namespace NUMINAMATH_CALUDE_exactly_two_linear_functions_l226_22614

/-- Two quadratic trinomials -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Linear function -/
structure LinearFunction where
  m : ℝ
  n : ℝ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate_quadratic (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Evaluate a linear function at a given x -/
def evaluate_linear (l : LinearFunction) (x : ℝ) : ℝ :=
  l.m * x + l.n

/-- The main theorem -/
theorem exactly_two_linear_functions (P Q : QuadraticTrinomial) :
  ∃! (l₁ l₂ : LinearFunction),
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₁ x)) ∧
    (∀ x : ℝ, evaluate_quadratic P x = evaluate_quadratic Q (evaluate_linear l₂ x)) ∧
    l₁ ≠ l₂ :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_linear_functions_l226_22614


namespace NUMINAMATH_CALUDE_cake_mix_buyers_l226_22652

theorem cake_mix_buyers (total : ℕ) (muffin : ℕ) (both : ℕ) (neither_prob : ℚ) :
  total = 100 →
  muffin = 40 →
  both = 18 →
  neither_prob = 28/100 →
  ∃ cake : ℕ, cake = 50 ∧ cake + muffin - both = (1 - neither_prob) * total := by
  sorry

end NUMINAMATH_CALUDE_cake_mix_buyers_l226_22652


namespace NUMINAMATH_CALUDE_earnings_difference_proof_l226_22684

/-- Calculates the difference in annual earnings between two jobs --/
def annual_earnings_difference (
  new_wage : ℕ
  ) (new_hours : ℕ
  ) (old_wage : ℕ
  ) (old_hours : ℕ
  ) (weeks_per_year : ℕ
  ) : ℕ :=
  (new_wage * new_hours * weeks_per_year) - (old_wage * old_hours * weeks_per_year)

/-- Proves that the difference in annual earnings is $20,800 --/
theorem earnings_difference_proof :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_proof_l226_22684


namespace NUMINAMATH_CALUDE_zach_rental_cost_l226_22693

/-- Calculates the total cost of a car rental given the base cost, cost per mile, and miles driven. -/
def rental_cost (base_cost : ℚ) (cost_per_mile : ℚ) (miles_driven : ℚ) : ℚ :=
  base_cost + cost_per_mile * miles_driven

/-- Proves that the total cost of Zach's car rental is $832. -/
theorem zach_rental_cost :
  let base_cost : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let monday_miles : ℚ := 620
  let thursday_miles : ℚ := 744
  let total_miles : ℚ := monday_miles + thursday_miles
  rental_cost base_cost cost_per_mile total_miles = 832 := by
  sorry

end NUMINAMATH_CALUDE_zach_rental_cost_l226_22693


namespace NUMINAMATH_CALUDE_point_A_coordinates_l226_22608

/-- Given a point A with coordinates (2a-9, 1-2a), prove that if A is moved 5 units
    to the right and lands on the y-axis, then its new coordinates are (-5, -3) -/
theorem point_A_coordinates (a : ℝ) :
  let initial_A : ℝ × ℝ := (2*a - 9, 1 - 2*a)
  let moved_A : ℝ × ℝ := (2*a - 4, 1 - 2*a)  -- Moved 5 units to the right
  moved_A.1 = 0 →  -- Lands on y-axis
  moved_A = (-5, -3) :=
by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l226_22608


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l226_22622

def decimal_representation (n : ℕ) : ℚ → ℕ := sorry

theorem digit_150_of_one_thirteenth : decimal_representation 150 (1/13) = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l226_22622


namespace NUMINAMATH_CALUDE_notebooks_given_to_mike_l226_22666

theorem notebooks_given_to_mike (jack_original : ℕ) (gerald : ℕ) (to_paula : ℕ) (jack_final : ℕ) : 
  jack_original = gerald + 13 →
  gerald = 8 →
  to_paula = 5 →
  jack_final = 10 →
  jack_original - to_paula - jack_final = 6 := by
sorry

end NUMINAMATH_CALUDE_notebooks_given_to_mike_l226_22666


namespace NUMINAMATH_CALUDE_matrix_power_four_l226_22637

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_power_four :
  A ^ 4 = !![(-1 : ℤ), 1; -1, 0] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l226_22637


namespace NUMINAMATH_CALUDE_cubic_root_implies_h_value_l226_22679

theorem cubic_root_implies_h_value :
  ∀ h : ℝ, ((-3 : ℝ)^3 + h * (-3) - 18 = 0) → h = -15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_implies_h_value_l226_22679


namespace NUMINAMATH_CALUDE_union_of_sets_l226_22673

theorem union_of_sets (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 1}) :
  A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l226_22673


namespace NUMINAMATH_CALUDE_crease_lines_form_annulus_l226_22655

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the folding operation
def Fold (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
    p = (center.1 + t * (point.1 - center.1), center.2 + t * (point.2 - center.2))}

-- Define the set of all crease lines
def CreaseLines (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  ⋃ (point ∈ Circle center radius), Fold center radius point

-- Define the annulus
def Annulus (center : ℝ × ℝ) (innerRadius outerRadius : ℝ) : Set (ℝ × ℝ) :=
  {p | innerRadius^2 ≤ (p.1 - center.1)^2 + (p.2 - center.2)^2 ∧ 
       (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ outerRadius^2}

-- The theorem to prove
theorem crease_lines_form_annulus (center : ℝ × ℝ) :
  CreaseLines center 10 = Annulus center 5 10 := by sorry

end NUMINAMATH_CALUDE_crease_lines_form_annulus_l226_22655


namespace NUMINAMATH_CALUDE_first_courier_speed_l226_22688

/-- The speed of the first courier in km/h -/
def v : ℝ := 30

/-- The distance between cities A and B in km -/
def distance : ℝ := 120

/-- The speed of the second courier in km/h -/
def speed_second : ℝ := 50

/-- The time delay of the second courier in hours -/
def delay : ℝ := 1

theorem first_courier_speed :
  (distance / v = (3 * speed_second) / (v - speed_second)) ∧
  (v > 0) ∧ (v < speed_second) := by
  sorry

#check first_courier_speed

end NUMINAMATH_CALUDE_first_courier_speed_l226_22688


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l226_22692

/-- The side length of the largest square that can be inscribed in a square with side length 12,
    given two congruent equilateral triangles are inscribed as described in the problem. -/
theorem largest_inscribed_square_side_length : ℝ :=
  let outer_square_side : ℝ := 12
  let triangle_side : ℝ := 4 * Real.sqrt 6
  6 - Real.sqrt 6

/-- Proof that the calculated side length is correct -/
theorem largest_inscribed_square_side_length_proof :
  largest_inscribed_square_side_length = 6 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l226_22692


namespace NUMINAMATH_CALUDE_distance_covered_l226_22611

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) :
  time_minutes = 24 →
  speed_km_per_hour = 10 →
  (time_minutes / 60) * speed_km_per_hour = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_covered_l226_22611


namespace NUMINAMATH_CALUDE_distance_to_park_is_correct_l226_22698

/-- The distance from point A to the forest amusement park -/
def distance_to_park : ℕ := 2370

/-- The rabbit's starting time in minutes after midnight -/
def rabbit_start : ℕ := 9 * 60

/-- The turtle's starting time in minutes after midnight -/
def turtle_start : ℕ := 6 * 60 + 40

/-- The rabbit's speed in meters per minute -/
def rabbit_speed : ℕ := 40

/-- The turtle's speed in meters per minute -/
def turtle_speed : ℕ := 10

/-- The rabbit's jumping time in minutes -/
def rabbit_jump_time : ℕ := 3

/-- The rabbit's resting time in minutes -/
def rabbit_rest_time : ℕ := 2

/-- The time difference between rabbit and turtle arrival in seconds -/
def arrival_time_diff : ℕ := 15

theorem distance_to_park_is_correct : 
  ∀ (t : ℕ), 
  t * turtle_speed = distance_to_park ∧ 
  t = (rabbit_start - turtle_start) + 
      (distance_to_park - (rabbit_start - turtle_start) * turtle_speed) / 
      (rabbit_speed * rabbit_jump_time / (rabbit_jump_time + rabbit_rest_time) - turtle_speed) + 
      arrival_time_diff / 60 :=
sorry

end NUMINAMATH_CALUDE_distance_to_park_is_correct_l226_22698


namespace NUMINAMATH_CALUDE_dummies_remainder_l226_22625

theorem dummies_remainder (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dummies_remainder_l226_22625


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l226_22641

theorem quadratic_solution_property :
  ∀ p q : ℝ,
  (5 * p^2 - 20 * p + 15 = 0) →
  (5 * q^2 - 20 * q + 15 = 0) →
  p ≠ q →
  (p * q - 3)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l226_22641


namespace NUMINAMATH_CALUDE_scientific_notation_of_40_9_billion_l226_22647

theorem scientific_notation_of_40_9_billion :
  (40.9 : ℝ) * 1000000000 = 4.09 * (10 : ℝ)^9 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_40_9_billion_l226_22647


namespace NUMINAMATH_CALUDE_age_of_replaced_man_l226_22659

theorem age_of_replaced_man
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (age_increase : ℝ)
  (replaced_man1_age : ℝ)
  (women_avg_age : ℝ)
  (h1 : n = 7)
  (h2 : new_avg = initial_avg + age_increase)
  (h3 : age_increase = 4)
  (h4 : replaced_man1_age = 30)
  (h5 : women_avg_age = 42)
  : ∃ (replaced_man2_age : ℝ),
    n * new_avg = n * initial_avg - replaced_man1_age - replaced_man2_age + 2 * women_avg_age
    ∧ replaced_man2_age = 26 :=
by sorry

end NUMINAMATH_CALUDE_age_of_replaced_man_l226_22659


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00001_l226_22649

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00001 :
  toScientificNotation 0.00001 = ScientificNotation.mk 1 (-5) sorry :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00001_l226_22649


namespace NUMINAMATH_CALUDE_circle_center_sum_l226_22604

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (x + y = -1) := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l226_22604


namespace NUMINAMATH_CALUDE_xiao_ming_calculation_l226_22664

theorem xiao_ming_calculation (a : ℚ) : 
  (37 + 31 * a = 37 + 31 + a) → (a = 31 / 30) := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_calculation_l226_22664


namespace NUMINAMATH_CALUDE_product_absolute_value_l226_22610

theorem product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq : x + 2 / y = y + 2 / z ∧ y + 2 / z = z + 2 / x) :
  |x * y * z| = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_product_absolute_value_l226_22610


namespace NUMINAMATH_CALUDE_peanut_price_in_mixed_nuts_l226_22682

/-- Calculates the price per pound of peanuts in a mixed nut blend --/
theorem peanut_price_in_mixed_nuts
  (total_weight : ℝ)
  (mixed_price : ℝ)
  (cashew_weight : ℝ)
  (cashew_price : ℝ)
  (h1 : total_weight = 100)
  (h2 : mixed_price = 2.5)
  (h3 : cashew_weight = 60)
  (h4 : cashew_price = 4) :
  (total_weight * mixed_price - cashew_weight * cashew_price) / (total_weight - cashew_weight) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_peanut_price_in_mixed_nuts_l226_22682


namespace NUMINAMATH_CALUDE_identity_function_l226_22662

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end NUMINAMATH_CALUDE_identity_function_l226_22662


namespace NUMINAMATH_CALUDE_totalCost_equals_64_l226_22600

-- Define the side length of each square
def squareSide : ℝ := 4

-- Define the number of squares
def numSquares : ℕ := 4

-- Define the areas of overlap
def centralOverlap : ℕ := 1
def tripleOverlap : ℕ := 6
def doubleOverlap : ℕ := 12
def singleArea : ℕ := 18

-- Define the cost function
def costFunction (overlappingSquares : ℕ) : ℕ := overlappingSquares

-- Theorem statement
theorem totalCost_equals_64 :
  (centralOverlap * costFunction numSquares) +
  (tripleOverlap * costFunction 3) +
  (doubleOverlap * costFunction 2) +
  (singleArea * costFunction 1) = 64 := by
  sorry

end NUMINAMATH_CALUDE_totalCost_equals_64_l226_22600


namespace NUMINAMATH_CALUDE_chord_length_dot_product_value_l226_22607

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 0)^2 + (y - 1)^2 = 5

-- Define point P
def point_P : ℝ × ℝ := (0, -2)

-- Theorem for the length of the chord
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 10 :=
sorry

-- Theorem for the dot product
theorem dot_product_value :
  ∀ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B ∧
    ∃ (t : ℝ), A.1 = point_P.1 + t * (A.1 - point_P.1) ∧
                A.2 = point_P.2 + t * (A.2 - point_P.2) ∧
                B.1 = point_P.1 + t * (B.1 - point_P.1) ∧
                B.2 = point_P.2 + t * (B.2 - point_P.2) →
    ((A.1 - point_P.1) * (B.1 - point_P.1) + (A.2 - point_P.2) * (B.2 - point_P.2))^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_chord_length_dot_product_value_l226_22607


namespace NUMINAMATH_CALUDE_opposite_reciprocal_theorem_l226_22601

theorem opposite_reciprocal_theorem (a b c d m : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : |m| = 2) :
  (a + b) / (4 * m) + m^2 - 3 * c * d = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_theorem_l226_22601


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l226_22661

theorem linear_function_not_in_fourth_quadrant (b : ℝ) (h : b ≥ 0) :
  ∀ x y : ℝ, y = 2 * x + b → ¬(x > 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l226_22661


namespace NUMINAMATH_CALUDE_average_height_combined_groups_l226_22616

theorem average_height_combined_groups (n₁ n₂ : ℕ) (h₁ h₂ : ℝ) :
  n₁ = 35 →
  n₂ = 25 →
  h₁ = 22 →
  h₂ = 18 →
  (n₁ * h₁ + n₂ * h₂) / (n₁ + n₂ : ℝ) = 20.33 :=
by sorry

end NUMINAMATH_CALUDE_average_height_combined_groups_l226_22616


namespace NUMINAMATH_CALUDE_equation_a_is_quadratic_l226_22626

-- Define what a quadratic equation is
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation we want to prove is quadratic
def f (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem equation_a_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_a_is_quadratic_l226_22626


namespace NUMINAMATH_CALUDE_hanna_roses_to_friends_l226_22677

/-- Calculates the number of roses Hanna gives to her friends --/
def roses_given_to_friends (total_money : ℚ) (rose_price : ℚ) 
  (jenna_fraction : ℚ) (imma_fraction : ℚ) : ℚ :=
  let total_roses := total_money / rose_price
  let jenna_roses := jenna_fraction * total_roses
  let imma_roses := imma_fraction * total_roses
  jenna_roses + imma_roses

/-- Theorem stating the number of roses Hanna gives to her friends --/
theorem hanna_roses_to_friends : 
  roses_given_to_friends 300 2 (1/3) (1/2) = 125 := by
  sorry

end NUMINAMATH_CALUDE_hanna_roses_to_friends_l226_22677


namespace NUMINAMATH_CALUDE_tan_alpha_value_l226_22638

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/3) = 2 * Real.sqrt 3) :
  Real.tan α = Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l226_22638


namespace NUMINAMATH_CALUDE_all_truth_probability_l226_22603

def alice_truth_prob : ℝ := 0.7
def bob_truth_prob : ℝ := 0.6
def carol_truth_prob : ℝ := 0.8
def david_truth_prob : ℝ := 0.5

theorem all_truth_probability :
  alice_truth_prob * bob_truth_prob * carol_truth_prob * david_truth_prob = 0.168 := by
  sorry

end NUMINAMATH_CALUDE_all_truth_probability_l226_22603


namespace NUMINAMATH_CALUDE_prism_lateral_faces_are_parallelograms_l226_22633

/-- A prism is a polyhedron with two congruent and parallel faces (called bases) 
    and all other faces (called lateral faces) are parallelograms. -/
structure Prism where
  -- We don't need to define the internal structure for this problem
  mk :: 

/-- A face of a polyhedron -/
structure Face where
  -- We don't need to define the internal structure for this problem
  mk ::

/-- Predicate to check if a face is a lateral face of a prism -/
def is_lateral_face (p : Prism) (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Predicate to check if a face is a parallelogram -/
def is_parallelogram (f : Face) : Prop :=
  -- Definition omitted for brevity
  sorry

theorem prism_lateral_faces_are_parallelograms (p : Prism) :
  ∀ (f : Face), is_lateral_face p f → is_parallelogram f := by
  sorry

end NUMINAMATH_CALUDE_prism_lateral_faces_are_parallelograms_l226_22633


namespace NUMINAMATH_CALUDE_election_winner_percentage_l226_22687

theorem election_winner_percentage (total_votes : ℕ) (winning_margin : ℕ) :
  total_votes = 900 →
  winning_margin = 360 →
  ∃ (winning_percentage : ℚ),
    winning_percentage = 70 / 100 ∧
    (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_margin :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l226_22687


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l226_22618

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l226_22618


namespace NUMINAMATH_CALUDE_egg_sale_remainder_l226_22683

theorem egg_sale_remainder : (53 + 65 + 26) % 15 = 9 := by sorry

end NUMINAMATH_CALUDE_egg_sale_remainder_l226_22683


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l226_22658

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 = 2) 
  (eq2 : 2 * b - 3 * a = 4) : 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l226_22658


namespace NUMINAMATH_CALUDE_triangle_altitude_slopes_l226_22671

/-- Given a triangle ABC with vertices A(-1,0), B(1,1), and C(0,2),
    prove that the slopes of the altitudes on sides AB, AC, and BC
    are -2, -1/2, and 1 respectively. -/
theorem triangle_altitude_slopes :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (0, 2)
  let slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  let perpendicular_slope (m : ℝ) : ℝ := -1 / m
  (perpendicular_slope (slope A B) = -2) ∧
  (perpendicular_slope (slope A C) = -1/2) ∧
  (perpendicular_slope (slope B C) = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_slopes_l226_22671


namespace NUMINAMATH_CALUDE_k_value_max_value_on_interval_l226_22691

-- Define the function f(x) with parameter k
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 3)*x + k^2 - 7

-- Theorem 1: k = 3 given the zeros of f(x)
theorem k_value (k : ℝ) : (f k (-1) = 0 ∧ f k (-2) = 0) → k = 3 := by sorry

-- Define the specific function f(x) = x^2 + 3x + 2
def f_specific (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Theorem 2: Maximum value of f_specific on [-2, 2] is 12
theorem max_value_on_interval : 
  ∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f_specific x ≤ 12 ∧ ∃ y ∈ Set.Icc (-2) 2, f_specific y = 12 := by sorry

end NUMINAMATH_CALUDE_k_value_max_value_on_interval_l226_22691


namespace NUMINAMATH_CALUDE_subtracted_value_l226_22696

theorem subtracted_value (x y : ℤ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l226_22696


namespace NUMINAMATH_CALUDE_binomial_10_3_l226_22620

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem binomial_10_3 : binomial 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l226_22620


namespace NUMINAMATH_CALUDE_tiles_crossed_specific_floor_l226_22676

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ) (length : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (width : ℕ) (length : ℕ)

/-- Counts the number of tiles crossed by a diagonal line on a floor -/
def tilesCrossedByDiagonal (f : Floor) (t : Tile) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

theorem tiles_crossed_specific_floor :
  let floor := Floor.mk 12 19
  let tile := Tile.mk 1 2
  tilesCrossedByDiagonal floor tile = 30 := by
  sorry

#eval tilesCrossedByDiagonal (Floor.mk 12 19) (Tile.mk 1 2)

end NUMINAMATH_CALUDE_tiles_crossed_specific_floor_l226_22676


namespace NUMINAMATH_CALUDE_green_sweets_count_l226_22665

/-- Given the number of blue and yellow sweets, and the total number of sweets,
    calculate the number of green sweets. -/
theorem green_sweets_count 
  (blue_sweets : ℕ) 
  (yellow_sweets : ℕ) 
  (total_sweets : ℕ) 
  (h1 : blue_sweets = 310) 
  (h2 : yellow_sweets = 502) 
  (h3 : total_sweets = 1024) : 
  total_sweets - (blue_sweets + yellow_sweets) = 212 := by
sorry

#eval 1024 - (310 + 502)  -- This should output 212

end NUMINAMATH_CALUDE_green_sweets_count_l226_22665


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l226_22619

theorem ratio_a_to_c (a b c : ℚ) 
  (h1 : a / b = 8 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l226_22619


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l226_22669

/-- Given vectors OA, OB, OC in ℝ², prove that if A, B, C are collinear, then the x-coordinate of OA is 6. -/
theorem collinear_points_k_value (OA OB OC : ℝ × ℝ) :
  OA.1 = k ∧ OA.2 = 11 ∧
  OB = (4, 5) ∧
  OC = (5, 8) ∧
  ∃ (t : ℝ), (OC.1 - OB.1, OC.2 - OB.2) = t • (OB.1 - OA.1, OB.2 - OA.2) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l226_22669


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l226_22653

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage : ℚ) (hindu_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  hindu_percentage = 14 / 100 →
  other_boys = 272 →
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l226_22653


namespace NUMINAMATH_CALUDE_correct_mixture_ratio_l226_22689

/-- Represents a salt solution with a given concentration and amount -/
structure SaltSolution :=
  (concentration : ℚ)
  (amount : ℚ)

/-- Represents a mixture of two salt solutions -/
def mix (s1 s2 : SaltSolution) (r1 r2 : ℚ) : SaltSolution :=
  { concentration := (s1.concentration * r1 + s2.concentration * r2) / (r1 + r2),
    amount := r1 + r2 }

theorem correct_mixture_ratio :
  let solutionA : SaltSolution := ⟨2/5, 30⟩
  let solutionB : SaltSolution := ⟨4/5, 60⟩
  let mixedSolution := mix solutionA solutionB 3 1
  mixedSolution.concentration = 1/2 ∧ mixedSolution.amount = 50 :=
by sorry


end NUMINAMATH_CALUDE_correct_mixture_ratio_l226_22689


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l226_22602

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 4x^2 - 6x + 9 has discriminant -108 -/
theorem quadratic_discriminant :
  discriminant 4 (-6) 9 = -108 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l226_22602


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_approx_l226_22636

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGP where
  a : ℝ
  r : ℝ
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a SpecialGP satisfies a cubic equation -/
theorem special_gp_ratio_equation (gp : SpecialGP) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

/-- The solution to the cubic equation is approximately 0.5437 -/
theorem special_gp_ratio_approx (gp : SpecialGP) :
  ∃ ε > 0, |gp.r - 0.5437| < ε := by
  sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_special_gp_ratio_approx_l226_22636


namespace NUMINAMATH_CALUDE_abs_neg_six_l226_22651

theorem abs_neg_six : |(-6 : ℤ)| = 6 := by sorry

end NUMINAMATH_CALUDE_abs_neg_six_l226_22651


namespace NUMINAMATH_CALUDE_slope_at_one_l226_22650

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem slope_at_one (α : ℝ) :
  (deriv f 1 = α) →
  (Real.cos α / (Real.sin α - 4 * Real.cos α) = -1) :=
by sorry

end NUMINAMATH_CALUDE_slope_at_one_l226_22650


namespace NUMINAMATH_CALUDE_asha_win_probability_l226_22634

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_prob_val : lose_prob = 5/12)
  (tie_prob_val : tie_prob = 1/6)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l226_22634


namespace NUMINAMATH_CALUDE_reverse_digits_sum_l226_22672

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem reverse_digits_sum (x y : ℕ) (hx : TwoDigitInt x) (hy : TwoDigitInt y)
  (h_reverse : y = 10 * (x % 10) + (x / 10))
  (a b : ℕ) (hx_digits : x = 10 * a + b)
  (hab : a - b = 3)
  (m : ℕ) (hm : x^2 - y^2 = m^2) :
  x + y + m = 178 := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_sum_l226_22672


namespace NUMINAMATH_CALUDE_distinct_pairs_of_twelve_students_l226_22668

-- Define the number of students
def num_students : ℕ := 12

-- Define the function to calculate the number of distinct pairs
def num_distinct_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem distinct_pairs_of_twelve_students :
  num_distinct_pairs num_students = 66 := by
  sorry

end NUMINAMATH_CALUDE_distinct_pairs_of_twelve_students_l226_22668


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l226_22630

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) (h3 : d = 13) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 20 / 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l226_22630


namespace NUMINAMATH_CALUDE_product_expansion_sum_l226_22613

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4*x^2 - 6*x + 5) * (8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  8*a + 4*b + 2*c + d = 18 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l226_22613


namespace NUMINAMATH_CALUDE_male_students_in_school_l226_22624

/-- Represents the number of students in a school population --/
structure SchoolPopulation where
  total : Nat
  sample : Nat
  females_in_sample : Nat

/-- Calculates the number of male students in the school --/
def male_students (pop : SchoolPopulation) : Nat :=
  pop.total - (pop.total * pop.females_in_sample / pop.sample)

/-- Theorem stating the number of male students in the given scenario --/
theorem male_students_in_school (pop : SchoolPopulation) 
  (h1 : pop.total = 1600)
  (h2 : pop.sample = 200)
  (h3 : pop.females_in_sample = 95) :
  male_students pop = 840 := by
  sorry

#eval male_students { total := 1600, sample := 200, females_in_sample := 95 }

end NUMINAMATH_CALUDE_male_students_in_school_l226_22624


namespace NUMINAMATH_CALUDE_intersection_A_B_l226_22675

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x^2 + 4*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l226_22675


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l226_22609

theorem perfect_cube_units_digits : 
  ∃ (S : Finset ℕ), (∀ n : ℕ, ∃ k : ℕ, n ^ 3 % 10 ∈ S) ∧ S.card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l226_22609


namespace NUMINAMATH_CALUDE_limit_at_one_l226_22663

noncomputable def f (x : ℝ) : ℝ := (5/3) * x - Real.log (2*x + 1)

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_one_l226_22663


namespace NUMINAMATH_CALUDE_plane_line_propositions_l226_22699

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines are skew -/
def skew (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects a plane -/
def intersects_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- The angles formed by two lines with a plane are equal -/
def equal_angles_with_plane (l1 l2 : Line) (p : Plane) : Prop :=
  sorry

theorem plane_line_propositions (α : Plane) (m n : Line) :
  (∃! prop : Prop, prop = true ∧
    (prop = (parallel m n → equal_angles_with_plane m n α) ∨
     prop = (parallel_to_plane m α → parallel_to_plane n α → parallel m n) ∨
     prop = (perpendicular_to_plane m α → perpendicular m n → parallel_to_plane n α) ∨
     prop = (skew m n → parallel_to_plane m α → intersects_plane n α))) :=
  sorry

end NUMINAMATH_CALUDE_plane_line_propositions_l226_22699


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l226_22639

/-- Given that the inequality ax^2 + x + 1 < 0 has a non-empty solution set for x,
    prove that the range of a is a < 1/4 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + x + 1 < 0) → a < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l226_22639
