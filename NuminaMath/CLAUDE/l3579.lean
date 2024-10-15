import Mathlib

namespace NUMINAMATH_CALUDE_factorial_of_factorial_l3579_357985

theorem factorial_of_factorial (n : ℕ) : (n.factorial.factorial) / n.factorial = (n.factorial - 1).factorial := by
  sorry

end NUMINAMATH_CALUDE_factorial_of_factorial_l3579_357985


namespace NUMINAMATH_CALUDE_unit_digit_of_15_power_l3579_357934

theorem unit_digit_of_15_power (X : ℕ+) : ∃ n : ℕ, 15^(X : ℕ) ≡ 5 [MOD 10] :=
sorry

end NUMINAMATH_CALUDE_unit_digit_of_15_power_l3579_357934


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3579_357966

/-- A quadratic function satisfying given conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_minimum_value
  (a b c : ℝ)
  (h1 : f a b c (-7) = -9)
  (h2 : f a b c (-5) = -4)
  (h3 : f a b c (-3) = -1)
  (h4 : f a b c (-1) = 0)
  (h5 : f a b c 1 = -1) :
  ∀ x ∈ Set.Icc (-7 : ℝ) 7, f a b c x ≥ -16 ∧ ∃ x₀ ∈ Set.Icc (-7 : ℝ) 7, f a b c x₀ = -16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3579_357966


namespace NUMINAMATH_CALUDE_chocolate_bars_in_box_l3579_357907

/-- The weight of a single chocolate bar in grams -/
def bar_weight : ℕ := 125

/-- The weight of the box in kilograms -/
def box_weight : ℕ := 2

/-- The number of chocolate bars in the box -/
def num_bars : ℕ := (box_weight * 1000) / bar_weight

/-- Theorem stating that the number of chocolate bars in the box is 16 -/
theorem chocolate_bars_in_box : num_bars = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_box_l3579_357907


namespace NUMINAMATH_CALUDE_negation_equivalence_l3579_357982

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3579_357982


namespace NUMINAMATH_CALUDE_cost_of_candies_in_dollars_l3579_357949

-- Define the cost of one piece of candy in cents
def cost_per_candy : ℕ := 2

-- Define the number of pieces of candy
def number_of_candies : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_candies_in_dollars :
  (number_of_candies * cost_per_candy) / cents_per_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_candies_in_dollars_l3579_357949


namespace NUMINAMATH_CALUDE_tom_marble_groups_l3579_357971

/-- Represents the types of marbles Tom has --/
inductive MarbleType
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents Tom's marble collection --/
structure MarbleCollection where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Counts the number of different groups of 3 marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom can choose 8 different groups of 3 marbles --/
theorem tom_marble_groups (tom_marbles : MarbleCollection) 
  (h_red : tom_marbles.red = 1)
  (h_blue : tom_marbles.blue = 1)
  (h_green : tom_marbles.green = 2)
  (h_yellow : tom_marbles.yellow = 3) :
  countDifferentGroups tom_marbles = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_marble_groups_l3579_357971


namespace NUMINAMATH_CALUDE_beans_to_seitan_ratio_l3579_357964

/-- Represents the number of dishes with a specific protein combination -/
structure DishCount where
  total : ℕ
  beansAndLentils : ℕ
  beansAndSeitan : ℕ
  withLentils : ℕ
  onlyBeans : ℕ
  onlySeitan : ℕ

/-- The conditions of the problem -/
def restaurantMenu : DishCount where
  total := 10
  beansAndLentils := 2
  beansAndSeitan := 2
  withLentils := 4
  onlyBeans := 2
  onlySeitan := 2

/-- The theorem to prove -/
theorem beans_to_seitan_ratio (menu : DishCount) 
  (h1 : menu.total = 10)
  (h2 : menu.beansAndLentils = 2)
  (h3 : menu.beansAndSeitan = 2)
  (h4 : menu.withLentils = 4)
  (h5 : menu.onlyBeans + menu.onlySeitan = menu.total - menu.beansAndLentils - menu.beansAndSeitan - (menu.withLentils - menu.beansAndLentils))
  (h6 : menu.onlyBeans = menu.onlySeitan) :
  menu.onlyBeans = menu.onlySeitan := by
  sorry

end NUMINAMATH_CALUDE_beans_to_seitan_ratio_l3579_357964


namespace NUMINAMATH_CALUDE_m_range_l3579_357973

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m))^x > (-(5 - 2*m))^y

-- Define the theorem
theorem m_range (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3579_357973


namespace NUMINAMATH_CALUDE_total_songs_bought_l3579_357900

theorem total_songs_bought (country_albums : ℕ) (pop_albums : ℕ) 
  (songs_per_country_album : ℕ) (songs_per_pop_album : ℕ) : 
  country_albums = 4 → pop_albums = 7 → 
  songs_per_country_album = 5 → songs_per_pop_album = 6 → 
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album = 62 := by
  sorry

#check total_songs_bought

end NUMINAMATH_CALUDE_total_songs_bought_l3579_357900


namespace NUMINAMATH_CALUDE_f_inequality_l3579_357987

open Real

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 1 + log x

theorem f_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂) :
  (f x₂ - f x₁) / (x₂ - x₁) < f_deriv ((x₁ + x₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l3579_357987


namespace NUMINAMATH_CALUDE_ned_bomb_diffusion_l3579_357993

/-- Represents the problem of Ned racing to deactivate a time bomb --/
def bomb_diffusion_problem (total_flights : ℕ) (seconds_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let total_time := total_flights * seconds_per_flight
  let remaining_time := total_time - time_spent
  let time_left := bomb_timer - remaining_time
  time_left = 84

/-- Theorem stating that Ned will have 84 seconds to diffuse the bomb --/
theorem ned_bomb_diffusion :
  bomb_diffusion_problem 40 13 58 273 := by
  sorry

#check ned_bomb_diffusion

end NUMINAMATH_CALUDE_ned_bomb_diffusion_l3579_357993


namespace NUMINAMATH_CALUDE_tourist_travel_time_l3579_357962

theorem tourist_travel_time (boat_distance : ℝ) (walk_distance : ℝ) 
  (h1 : boat_distance = 90) 
  (h2 : walk_distance = 10) : ∃ (walk_time boat_time : ℝ),
  walk_time + 4 = boat_time ∧ 
  walk_distance / walk_time * boat_time = boat_distance / boat_time * walk_time ∧
  walk_time = 2 ∧ 
  boat_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_tourist_travel_time_l3579_357962


namespace NUMINAMATH_CALUDE_milk_container_problem_l3579_357923

theorem milk_container_problem (capacity_A : ℝ) : 
  (capacity_A > 0) →
  (0.375 * capacity_A + 156 = 0.625 * capacity_A - 156) →
  capacity_A = 1248 := by
sorry

end NUMINAMATH_CALUDE_milk_container_problem_l3579_357923


namespace NUMINAMATH_CALUDE_sine_cosine_relation_l3579_357912

theorem sine_cosine_relation (x : ℝ) : 
  Real.sin (2 * x + π / 6) = -1 / 3 → Real.cos (π / 3 - 2 * x) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_relation_l3579_357912


namespace NUMINAMATH_CALUDE_solids_of_revolution_l3579_357989

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define the property of being a solid of revolution
def isSolidOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem solids_of_revolution :
  ∀ s : GeometricSolid,
    isSolidOfRevolution s ↔ (s = GeometricSolid.Cylinder ∨ s = GeometricSolid.Sphere) :=
by
  sorry

#check solids_of_revolution

end NUMINAMATH_CALUDE_solids_of_revolution_l3579_357989


namespace NUMINAMATH_CALUDE_rectangle_area_l3579_357929

theorem rectangle_area (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) 
  (h_pythagorean : a^2 + b^2 = c^2) : a * b = a * b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3579_357929


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3579_357913

-- Define the bowties operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 10 ∧ h = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3579_357913


namespace NUMINAMATH_CALUDE_revenue_calculation_l3579_357902

/-- The total revenue from selling apples and oranges -/
def total_revenue (z t : ℕ) (a b : ℚ) : ℚ :=
  z * a + t * b

/-- Theorem: The total revenue from selling 200 apples at $0.50 each and 75 oranges at $0.75 each is $156.25 -/
theorem revenue_calculation :
  total_revenue 200 75 (1/2) (3/4) = 156.25 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l3579_357902


namespace NUMINAMATH_CALUDE_f_properties_l3579_357904

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem stating the properties of f and the inequality
theorem f_properties :
  (∃ (t : ℝ), t = 3 ∧ ∀ x, f x ≤ t) ∧
  (∀ x, x ≥ 2 → f x = 3) ∧
  (∀ a b : ℝ, a^2 + 2*b = 1 → 2*a^2 + b^2 ≥ 1/4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3579_357904


namespace NUMINAMATH_CALUDE_orthocenter_centroid_perpendicular_l3579_357906

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if two points are not equal -/
def notEqual (p q : ℝ × ℝ) : Prop := p ≠ q

/-- Calculates the angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_centroid_perpendicular (t : Triangle) :
  isAcuteAngled t →
  notEqual t.A t.B →
  notEqual t.A t.C →
  let H := orthocenter t
  let G := centroid t
  1 / triangleArea H t.A t.B + 1 / triangleArea H t.A t.C = 1 / triangleArea H t.B t.C →
  angle t.A G H = 90 := by sorry

end NUMINAMATH_CALUDE_orthocenter_centroid_perpendicular_l3579_357906


namespace NUMINAMATH_CALUDE_two_suits_cost_l3579_357911

def off_the_rack_cost : ℕ := 300
def tailoring_cost : ℕ := 200

def total_cost (off_the_rack : ℕ) (tailoring : ℕ) : ℕ :=
  off_the_rack + (3 * off_the_rack + tailoring)

theorem two_suits_cost :
  total_cost off_the_rack_cost tailoring_cost = 1400 := by
  sorry

end NUMINAMATH_CALUDE_two_suits_cost_l3579_357911


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l3579_357948

theorem sqrt_equation_condition (a : ℝ) : 
  Real.sqrt (a^2 - 4*a + 4) = 2 - a ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l3579_357948


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3579_357925

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line l -/
structure Line where
  l : ℝ × ℝ → Prop

/-- The distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distancePointToLine c.O l < c.r

theorem line_circle_intersection (c : Circle) (l : Line) :
  distancePointToLine c.O l < c.r → intersects c l :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3579_357925


namespace NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_equality_iff_k_range_l3579_357965

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem for part I
theorem union_when_k_neg_one :
  A ∪ B (-1) = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem intersection_equality_iff_k_range :
  ∀ k : ℝ, A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_equality_iff_k_range_l3579_357965


namespace NUMINAMATH_CALUDE_crayon_calculation_l3579_357992

/-- Calculates the final number of crayons and their percentage of the total items -/
theorem crayon_calculation (initial_crayons : ℕ) (initial_pencils : ℕ) 
  (removed_crayons : ℕ) (added_crayons : ℕ) (increase_percentage : ℚ) :
  initial_crayons = 41 →
  initial_pencils = 26 →
  removed_crayons = 8 →
  added_crayons = 12 →
  increase_percentage = 1/10 →
  let intermediate_crayons := initial_crayons - removed_crayons + added_crayons
  let final_crayons := (intermediate_crayons : ℚ) * (1 + increase_percentage)
  let rounded_final_crayons := round final_crayons
  let total_items := rounded_final_crayons + initial_pencils
  let percentage_crayons := (rounded_final_crayons : ℚ) / (total_items : ℚ) * 100
  rounded_final_crayons = 50 ∧ abs (percentage_crayons - 65.79) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_crayon_calculation_l3579_357992


namespace NUMINAMATH_CALUDE_abs_diff_two_equiv_interval_l3579_357922

theorem abs_diff_two_equiv_interval (x : ℝ) : (1 < x ∧ x < 3) ↔ |x - 2| < 1 := by sorry

end NUMINAMATH_CALUDE_abs_diff_two_equiv_interval_l3579_357922


namespace NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l3579_357954

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (givenLine : Line2D) 
  (point : Point2D) 
  (resultLine : Line2D) : Prop :=
  parallelLines givenLine resultLine ∧ 
  pointOnLine point resultLine

/-- The main theorem to prove -/
theorem line_equation_proof : 
  let givenLine : Line2D := { a := 2, b := 3, c := 5 }
  let point : Point2D := { x := 1, y := -4 }
  let resultLine : Line2D := { a := 2, b := 3, c := 10 }
  parallel_line_through_point givenLine point resultLine := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l3579_357954


namespace NUMINAMATH_CALUDE_manufacturing_department_percentage_l3579_357984

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 162) : 
  (manufacturing_degrees / total_degrees) * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_department_percentage_l3579_357984


namespace NUMINAMATH_CALUDE_shaded_area_circle_and_tangents_l3579_357953

theorem shaded_area_circle_and_tangents (r : ℝ) (θ : ℝ) :
  r = 3 →
  θ = Real.pi / 3 →
  let circle_area := π * r^2
  let sector_angle := 2 * θ
  let sector_area := (sector_angle / (2 * Real.pi)) * circle_area
  let triangle_area := r^2 * Real.tan θ
  sector_area + 2 * triangle_area = 6 * π + 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_and_tangents_l3579_357953


namespace NUMINAMATH_CALUDE_extra_chairs_added_l3579_357947

/-- The number of extra chairs added to a wedding seating arrangement -/
theorem extra_chairs_added (rows : ℕ) (chairs_per_row : ℕ) (total_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → total_chairs = 95 → 
  total_chairs - (rows * chairs_per_row) = 11 := by
  sorry

end NUMINAMATH_CALUDE_extra_chairs_added_l3579_357947


namespace NUMINAMATH_CALUDE_total_time_knife_and_vegetables_l3579_357974

/-- Proves that the total time spent on knife sharpening and vegetable peeling is 40 minutes -/
theorem total_time_knife_and_vegetables (knife_time vegetable_time total_time : ℕ) : 
  knife_time = 10 →
  vegetable_time = 3 * knife_time →
  total_time = knife_time + vegetable_time →
  total_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_time_knife_and_vegetables_l3579_357974


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3579_357961

theorem absolute_value_equation (x : ℝ) :
  |x - 25| + |x - 15| = |2*x - 40| ↔ x ≤ 15 ∨ x ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3579_357961


namespace NUMINAMATH_CALUDE_total_distance_flown_l3579_357958

theorem total_distance_flown (trip_distance : ℝ) (num_trips : ℝ) 
  (h1 : trip_distance = 256.0) 
  (h2 : num_trips = 32.0) : 
  trip_distance * num_trips = 8192.0 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_flown_l3579_357958


namespace NUMINAMATH_CALUDE_function_form_from_inequality_l3579_357938

/-- A function satisfying the given inequality property. -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y) - f (x - y) - y| ≤ y^2

/-- The main theorem stating that a function satisfying the inequality
    must be of the form f(x) = x/2 + c for some constant c. -/
theorem function_form_from_inequality (f : ℝ → ℝ) 
    (h : SatisfiesInequality f) : 
    ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end NUMINAMATH_CALUDE_function_form_from_inequality_l3579_357938


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3579_357978

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3579_357978


namespace NUMINAMATH_CALUDE_bread_making_time_is_375_l3579_357937

/-- Represents the duration of each step in Mark's bread-making process -/
def bread_making_steps : List ℕ := [30, 120, 20, 120, 10, 30, 30, 15]

/-- The total time Mark spends making bread -/
def total_bread_making_time : ℕ := bread_making_steps.sum

/-- Theorem stating that the total time Mark spends making bread is 375 minutes -/
theorem bread_making_time_is_375 : total_bread_making_time = 375 := by
  sorry

#eval total_bread_making_time

end NUMINAMATH_CALUDE_bread_making_time_is_375_l3579_357937


namespace NUMINAMATH_CALUDE_initial_machines_l3579_357997

/-- The number of machines working initially -/
def N : ℕ := sorry

/-- The number of units produced by N machines in 5 days -/
def x : ℝ := sorry

/-- Machines work at a constant rate -/
axiom constant_rate : ∀ (m : ℕ) (u t : ℝ), m ≠ 0 → t ≠ 0 → u / (m * t) = x / (N * 5)

theorem initial_machines :
  N * (x / 5) = 12 * (x / 30) → N = 2 :=
sorry

end NUMINAMATH_CALUDE_initial_machines_l3579_357997


namespace NUMINAMATH_CALUDE_sqrt_224_range_l3579_357919

theorem sqrt_224_range : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_224_range_l3579_357919


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l3579_357955

-- Part 1
theorem simplify_expression_1 (x : ℝ) : 2 * x^2 + 3 * x - 3 * x^2 + 4 * x = -x^2 + 7 * x := by sorry

-- Part 2
theorem simplify_expression_2 (a : ℝ) : 3 * a - 5 * (a + 1) + 4 * (2 + a) = 2 * a + 3 := by sorry

-- Part 3
theorem calculate_expression_3 : (-2/3 + 5/8 - 1/6) * (-24) = 5 := by sorry

-- Part 4
theorem calculate_expression_4 : -(1^4) + 16 / ((-2)^3) * |(-3) - 1| = -9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l3579_357955


namespace NUMINAMATH_CALUDE_right_triangle_tan_y_l3579_357999

theorem right_triangle_tan_y (X Y Z : ℝ × ℝ) :
  -- Right triangle condition
  (Y.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (Y.2 - X.2) →
  -- XY = 30 condition
  Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 30 →
  -- XZ = 40 condition (derived from the solution)
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 40 →
  -- Conclusion: tan Y = 4/3
  (Z.2 - X.2) / (Y.1 - X.1) = 4 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_tan_y_l3579_357999


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_properties_l3579_357932

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Theorem stating the properties of the sum of areas for the given trapezoid -/
theorem trapezoid_area_sum_properties :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t := Trapezoid.mk 4 6 8 10
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 742 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_properties_l3579_357932


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3579_357916

theorem larger_integer_problem :
  ∃ (x : ℕ+) (y : ℕ+), (4 * x)^2 - 2 * x = 8100 ∧ x + 10 = 2 * y ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3579_357916


namespace NUMINAMATH_CALUDE_janessa_baseball_cards_l3579_357998

/-- Janessa's baseball card collection problem -/
theorem janessa_baseball_cards
  (initial_cards : ℕ)
  (father_cards : ℕ)
  (ebay_cards : ℕ)
  (bad_cards : ℕ)
  (cards_given_to_dexter : ℕ)
  (h1 : initial_cards = 4)
  (h2 : father_cards = 13)
  (h3 : ebay_cards = 36)
  (h4 : bad_cards = 4)
  (h5 : cards_given_to_dexter = 29) :
  initial_cards + father_cards + ebay_cards - bad_cards - cards_given_to_dexter = 20 := by
  sorry

#check janessa_baseball_cards

end NUMINAMATH_CALUDE_janessa_baseball_cards_l3579_357998


namespace NUMINAMATH_CALUDE_intercept_sum_l3579_357914

/-- The modulus of the congruence -/
def m : ℕ := 17

/-- The congruence relation -/
def congruence (x y : ℕ) : Prop :=
  (7 * x) % m = (3 * y + 2) % m

/-- The x-intercept of the congruence -/
def x_intercept : ℕ := 10

/-- The y-intercept of the congruence -/
def y_intercept : ℕ := 5

/-- Theorem stating that the sum of x and y intercepts is 15 -/
theorem intercept_sum :
  x_intercept + y_intercept = 15 ∧
  congruence x_intercept 0 ∧
  congruence 0 y_intercept ∧
  x_intercept < m ∧
  y_intercept < m :=
sorry

end NUMINAMATH_CALUDE_intercept_sum_l3579_357914


namespace NUMINAMATH_CALUDE_closer_to_cottage_l3579_357975

theorem closer_to_cottage (c m p : ℝ) 
  (hc : c > 0)
  (hm : m + 3/2 * (1/2 * m) = c)
  (hp : 2*p + 1/3 * (2*p) = c) : 
  m/c > p/c := by
sorry

end NUMINAMATH_CALUDE_closer_to_cottage_l3579_357975


namespace NUMINAMATH_CALUDE_equation_equivalence_l3579_357924

theorem equation_equivalence : ∀ x : ℝ, 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3579_357924


namespace NUMINAMATH_CALUDE_contractor_problem_l3579_357909

/-- Represents the problem of calculating the number of days to complete 1/4 of the work --/
theorem contractor_problem (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) :
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  let remaining_workers := initial_workers - fired_workers
  let work_per_day := (1 : ℚ) / initial_workers
  let days_to_quarter := (1 / 4 : ℚ) / work_per_day
  let remaining_work := (3 / 4 : ℚ) / (remaining_workers : ℚ) / (remaining_days : ℚ)
  (1 : ℚ) = days_to_quarter * work_per_day + remaining_work * (remaining_workers : ℚ) * (remaining_days : ℚ) →
  days_to_quarter = 20 := by
  sorry


end NUMINAMATH_CALUDE_contractor_problem_l3579_357909


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l3579_357941

/-- Given two parabolas that intersect at points A and B, prove that if the sum of the x-coordinate
    and y-coordinate of the midpoint of AB is 2017, then c = 4031. -/
theorem parabola_intersection_midpoint (c : ℝ) : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let g (x : ℝ) := -x^2 + 4*x + c
  ∃ A B : ℝ × ℝ, 
    (f A.1 = A.2 ∧ g A.1 = A.2) ∧ 
    (f B.1 = B.2 ∧ g B.1 = B.2) ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 2017) →
  c = 4031 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l3579_357941


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3579_357967

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3579_357967


namespace NUMINAMATH_CALUDE_sequence_equality_l3579_357996

theorem sequence_equality (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + (a (n + 1)) - (a n))) :
  ∀ n : ℕ, a n = n := by
sorry

end NUMINAMATH_CALUDE_sequence_equality_l3579_357996


namespace NUMINAMATH_CALUDE_cab_driver_income_theorem_l3579_357950

/-- Represents the weather condition for a day --/
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

/-- Represents a day's income data --/
structure DayData where
  income : ℝ
  weather : Weather
  isPeakHours : Bool

/-- Calculates the adjusted income for a day based on weather and peak hours --/
def adjustedIncome (day : DayData) : ℝ :=
  match day.weather with
  | Weather.Rainy => day.income * 1.1
  | Weather.Cloudy => day.income * 0.95
  | Weather.Sunny => 
    if day.isPeakHours then day.income * 1.2
    else day.income

/-- The income data for 12 days --/
def incomeData : List DayData := [
  ⟨200, Weather.Rainy, false⟩,
  ⟨150, Weather.Sunny, false⟩,
  ⟨750, Weather.Sunny, false⟩,
  ⟨400, Weather.Sunny, false⟩,
  ⟨500, Weather.Cloudy, false⟩,
  ⟨300, Weather.Rainy, false⟩,
  ⟨650, Weather.Sunny, false⟩,
  ⟨350, Weather.Cloudy, false⟩,
  ⟨600, Weather.Sunny, true⟩,
  ⟨450, Weather.Sunny, false⟩,
  ⟨530, Weather.Sunny, false⟩,
  ⟨480, Weather.Cloudy, false⟩
]

theorem cab_driver_income_theorem :
  let totalIncome := (incomeData.map adjustedIncome).sum
  let averageIncome := totalIncome / incomeData.length
  totalIncome = 4963.5 ∧ averageIncome = 413.625 := by
  sorry


end NUMINAMATH_CALUDE_cab_driver_income_theorem_l3579_357950


namespace NUMINAMATH_CALUDE_sequence_properties_l3579_357903

def geometric_sequence (i : ℕ) : ℕ := 7 * 3^(16 - i) * 5^(i - 1)

theorem sequence_properties :
  (∀ i ∈ Finset.range 16, geometric_sequence (i + 1) > 0) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 1) ≥ 10^8 ∧ geometric_sequence (i + 1) < 10^9) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 6) ≥ 10^9 ∧ geometric_sequence (i + 6) < 10^10) ∧
  (∀ i ∈ Finset.range 4, geometric_sequence (i + 11) ≥ 10^10 ∧ geometric_sequence (i + 11) < 10^11) ∧
  (∀ i ∈ Finset.range 2, geometric_sequence (i + 15) ≥ 10^11 ∧ geometric_sequence (i + 15) < 10^12) ∧
  (∀ i ∈ Finset.range 15, geometric_sequence (i + 2) / geometric_sequence (i + 1) = geometric_sequence 2 / geometric_sequence 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3579_357903


namespace NUMINAMATH_CALUDE_cos_theta_equals_sqrt2_over_2_l3579_357920

/-- Given vectors a and b with an angle θ between them, 
    if a = (1,1) and b - a = (-1,1), then cos θ = √2/2 -/
theorem cos_theta_equals_sqrt2_over_2 (a b : ℝ × ℝ) (θ : ℝ) :
  a = (1, 1) →
  b - a = (-1, 1) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_equals_sqrt2_over_2_l3579_357920


namespace NUMINAMATH_CALUDE_three_W_seven_l3579_357988

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 5 * a - 3 * a^2

-- Theorem to prove
theorem three_W_seven : W 3 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_three_W_seven_l3579_357988


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l3579_357972

/-- Given a function f(x) = x^3 - ax^2 + x, prove that if its tangent line at x=1 
    is parallel to y=2x, then a = 1. -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 + x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x + 1
  (f' 1 = 2) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l3579_357972


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3579_357946

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation has exactly one solution iff a = 4 or a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, log10 (a * x) = 2 * log10 (x + 1) ∧ a * x > 0 ∧ x + 1 > 0) ↔ 
  (a = 4 ∨ a < 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3579_357946


namespace NUMINAMATH_CALUDE_max_parrots_in_zoo_l3579_357959

/-- Represents the number of parrots of each color in the zoo -/
structure ParrotCount where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The conditions of the zoo problem -/
def ZooConditions (p : ParrotCount) : Prop :=
  p.red > 0 ∧ p.yellow > 0 ∧ p.green > 0 ∧
  ∀ (s : Finset ℕ), s.card = 10 → (∃ i ∈ s, i < p.red) ∧
  ∀ (s : Finset ℕ), s.card = 12 → (∃ i ∈ s, i < p.yellow)

/-- The theorem stating the maximum number of parrots in the zoo -/
theorem max_parrots_in_zoo :
  ∃ (max : ℕ), max = 19 ∧
  (∃ (p : ParrotCount), ZooConditions p ∧ p.red + p.yellow + p.green = max) ∧
  ∀ (p : ParrotCount), ZooConditions p → p.red + p.yellow + p.green ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_parrots_in_zoo_l3579_357959


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3579_357963

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ 1 / (x + 2) + 1 / (y + 2) = 4 / 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3579_357963


namespace NUMINAMATH_CALUDE_pure_imaginary_solution_l3579_357942

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_solution (z : ℂ) (a : ℝ) 
  (h1 : IsPureImaginary z) 
  (h2 : (1 - Complex.I) * z = 1 + a * Complex.I) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solution_l3579_357942


namespace NUMINAMATH_CALUDE_probability_is_one_fourth_l3579_357976

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 1

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 4

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem probability_is_one_fourth : probability = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_fourth_l3579_357976


namespace NUMINAMATH_CALUDE_map_distance_theorem_l3579_357968

/-- Given a map scale and an actual distance, calculate the distance on the map --/
def map_distance (scale : ℚ) (actual_distance_km : ℚ) : ℚ :=
  (actual_distance_km * 100000) / (1 / scale)

/-- Theorem: The distance between two points on a map with scale 1/250000 and actual distance 5 km is 2 cm --/
theorem map_distance_theorem :
  map_distance (1 / 250000) 5 = 2 := by
  sorry

#eval map_distance (1 / 250000) 5

end NUMINAMATH_CALUDE_map_distance_theorem_l3579_357968


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3579_357927

theorem least_subtraction_for_divisibility (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  ∃ k, k ≥ 0 ∧ k < m ∧ (n ^ 1000 - k) % m = 0 ∧ 
  ∀ j, 0 ≤ j ∧ j < k → (n ^ 1000 - j) % m ≠ 0 :=
by sorry

#check least_subtraction_for_divisibility 10 97

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3579_357927


namespace NUMINAMATH_CALUDE_right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l3579_357944

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < r_a ∧ 0 < r_b ∧ 0 < r_c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

theorem right_angled_iff_sum_radii (t : Triangle) :
  is_right_angled t ↔ t.r + t.r_a + t.r_b + t.r_c = t.a + t.b + t.c :=
sorry

theorem right_angled_iff_sum_squared_radii (t : Triangle) :
  is_right_angled t ↔ t.r^2 + t.r_a^2 + t.r_b^2 + t.r_c^2 = t.a^2 + t.b^2 + t.c^2 :=
sorry

end NUMINAMATH_CALUDE_right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l3579_357944


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3579_357921

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence (a 5) (a 9) (a 15)) :
  a 15 / a 9 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3579_357921


namespace NUMINAMATH_CALUDE_simplify_expression_l3579_357957

theorem simplify_expression (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  18 * x^3 * y^2 * z^2 / (9 * x^2 * y * z^3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3579_357957


namespace NUMINAMATH_CALUDE_max_value_expression_l3579_357917

theorem max_value_expression (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3579_357917


namespace NUMINAMATH_CALUDE_total_notes_l3579_357951

/-- Calculates the total number of notes on a communal board -/
theorem total_notes (red_rows : Nat) (red_per_row : Nat) (blue_per_red : Nat) (extra_blue : Nat) :
  red_rows = 5 →
  red_per_row = 6 →
  blue_per_red = 2 →
  extra_blue = 10 →
  red_rows * red_per_row + red_rows * red_per_row * blue_per_red + extra_blue = 100 := by
  sorry


end NUMINAMATH_CALUDE_total_notes_l3579_357951


namespace NUMINAMATH_CALUDE_extreme_points_imply_a_range_and_negative_min_l3579_357930

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem statement
theorem extreme_points_imply_a_range_and_negative_min 
  (a : ℝ) (x₁ x₂ : ℝ) (h_extreme : x₁ < x₂ ∧ 
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧
    (∀ x, x₁ < x → x < x₂ → f_deriv a x ≠ 0)) :
  (0 < a ∧ a < Real.exp (-1)) ∧ f a x₁ < 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_imply_a_range_and_negative_min_l3579_357930


namespace NUMINAMATH_CALUDE_complex_power_4_30_degrees_l3579_357960

theorem complex_power_4_30_degrees : 
  (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 4 = -8 + 8 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_4_30_degrees_l3579_357960


namespace NUMINAMATH_CALUDE_imaginary_power_l3579_357936

theorem imaginary_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_l3579_357936


namespace NUMINAMATH_CALUDE_palindrome_square_base_l3579_357986

theorem palindrome_square_base (r : ℕ) (x : ℕ) : 
  x = r^3 + r^2 + r + 1 →
  Even r →
  ∃ (a b c d : ℕ), 
    (x^2 = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧
    (b + c = 24) →
  r = 26 :=
sorry

end NUMINAMATH_CALUDE_palindrome_square_base_l3579_357986


namespace NUMINAMATH_CALUDE_equal_area_division_l3579_357910

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral using the shoelace formula -/
def area (q : Quadrilateral) : ℚ :=
  let det := q.A.x * q.B.y + q.B.x * q.C.y + q.C.x * q.D.y + q.D.x * q.A.y -
             (q.B.x * q.A.y + q.C.x * q.B.y + q.D.x * q.C.y + q.A.x * q.D.y)
  (1/2) * abs det

/-- Represents the intersection point of the dividing line with CD -/
structure IntersectionPoint where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The main theorem -/
theorem equal_area_division (q : Quadrilateral) (i : IntersectionPoint) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 3⟩ →
  q.C = ⟨4, 4⟩ →
  q.D = ⟨5, 0⟩ →
  area { A := q.A, B := q.B, C := ⟨i.p / i.q, i.r / i.s⟩, D := q.D } = 
  area { A := q.A, B := ⟨i.p / i.q, i.r / i.s⟩, C := q.C, D := q.D } →
  i.p + i.q + i.r + i.s = 13 := by sorry

end NUMINAMATH_CALUDE_equal_area_division_l3579_357910


namespace NUMINAMATH_CALUDE_pizza_combinations_l3579_357943

def num_toppings : ℕ := 8

def one_topping_pizzas (n : ℕ) : ℕ := n

def two_topping_pizzas (n : ℕ) : ℕ := n.choose 2

def three_topping_pizzas (n : ℕ) : ℕ := n.choose 3

theorem pizza_combinations :
  one_topping_pizzas num_toppings +
  two_topping_pizzas num_toppings +
  three_topping_pizzas num_toppings = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3579_357943


namespace NUMINAMATH_CALUDE_range_of_a_l3579_357933

theorem range_of_a (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + a * (y - 2 * e * x) * (Real.log y - Real.log x) = 0) ↔
  a < 0 ∨ a ≥ 2 / e := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3579_357933


namespace NUMINAMATH_CALUDE_different_winning_scores_l3579_357931

def cross_country_meet (n : ℕ) : Prop :=
  n = 12 ∧ ∃ (team_size : ℕ), team_size = 6

def sum_of_positions (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def winning_score (total_score : ℕ) (score : ℕ) : Prop :=
  score ≤ total_score / 2

def min_winning_score (team_size : ℕ) : ℕ :=
  sum_of_positions team_size

theorem different_winning_scores (total_runners : ℕ) (team_size : ℕ) : 
  cross_country_meet total_runners →
  (winning_score (sum_of_positions total_runners) (sum_of_positions total_runners / 2) ∧
   min_winning_score team_size = sum_of_positions team_size) →
  (sum_of_positions total_runners / 2 - min_winning_score team_size + 1 = 19) :=
by sorry

end NUMINAMATH_CALUDE_different_winning_scores_l3579_357931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l3579_357901

theorem arithmetic_sequence_sum_product (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- Arithmetic sequence property
  a 7 < 0 →
  a 8 > 0 →
  a 8 > |a 7| →
  S 13 * S 14 < 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_product_l3579_357901


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l3579_357979

/-- Proves the number of females with advanced degrees in a company --/
theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : employees_with_advanced_degrees = 90)
  (h4 : males_with_college_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 55 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l3579_357979


namespace NUMINAMATH_CALUDE_system_solution_l3579_357935

theorem system_solution : 
  ∃! (x y : ℝ), x = 4 * y ∧ x + 2 * y = -12 ∧ x = -8 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3579_357935


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l3579_357940

/-- The perimeter of a figure formed by cutting a square and rearranging it -/
theorem square_cut_perimeter (s : ℝ) (h : s = 100) : 
  let rect_length : ℝ := s
  let rect_width : ℝ := s / 2
  let perimeter : ℝ := 3 * rect_length + 4 * rect_width
  perimeter = 500 := by sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l3579_357940


namespace NUMINAMATH_CALUDE_billy_brad_weight_difference_l3579_357994

-- Define the weights as natural numbers
def carl_weight : ℕ := 145
def billy_weight : ℕ := 159

-- Define Brad's weight in terms of Carl's
def brad_weight : ℕ := carl_weight + 5

-- State the theorem
theorem billy_brad_weight_difference :
  billy_weight - brad_weight = 9 :=
by sorry

end NUMINAMATH_CALUDE_billy_brad_weight_difference_l3579_357994


namespace NUMINAMATH_CALUDE_roberta_shopping_l3579_357945

def shopping_trip (initial_amount bag_price_difference : ℕ) : Prop :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price / 4
  let total_expenses := shoe_price + bag_price + lunch_price
  let money_left := initial_amount - total_expenses
  money_left = 78

theorem roberta_shopping :
  shopping_trip 158 17 := by
  sorry

end NUMINAMATH_CALUDE_roberta_shopping_l3579_357945


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l3579_357952

theorem sum_of_fractions_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h : a + b + 2*c = 0) : 
  1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l3579_357952


namespace NUMINAMATH_CALUDE_min_value_on_interval_l3579_357981

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l3579_357981


namespace NUMINAMATH_CALUDE_exactly_two_non_congruent_triangles_l3579_357983

/-- A triangle with integer side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle --/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Triangle inequality --/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Non-congruent triangles --/
def are_non_congruent (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of valid triangles with perimeter 12 --/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 12 ∧ is_valid_triangle t}

/-- The theorem to be proved --/
theorem exactly_two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    are_non_congruent t1 t2 ∧
    ∀ (t3 : IntTriangle),
      t3 ∈ valid_triangles →
      (t3 = t1 ∨ t3 = t2 ∨ ¬(are_non_congruent t1 t3 ∧ are_non_congruent t2 t3)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_non_congruent_triangles_l3579_357983


namespace NUMINAMATH_CALUDE_system_solution_l3579_357915

theorem system_solution :
  ∃ (x y : ℤ), 
    (x + 9773 = 13200) ∧
    (2 * x - 3 * y = 1544) ∧
    (x = 3427) ∧
    (y = 1770) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3579_357915


namespace NUMINAMATH_CALUDE_simplify_expression_l3579_357918

theorem simplify_expression (w : ℝ) : 
  4*w + 6*w + 8*w + 10*w + 12*w + 14*w + 16 = 54*w + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3579_357918


namespace NUMINAMATH_CALUDE_clock_malfunction_l3579_357977

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit either increases or decreases by 1 -/
def is_malfunctioned (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem clock_malfunction (displayed : Time) (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunctioned original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end NUMINAMATH_CALUDE_clock_malfunction_l3579_357977


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3579_357990

theorem simplify_and_evaluate (a : ℝ) (h : a = 2023) :
  (a + 1) / a / (a - 1 / a) = 1 / 2022 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3579_357990


namespace NUMINAMATH_CALUDE_sin_cos_product_l3579_357980

theorem sin_cos_product (α : Real) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l3579_357980


namespace NUMINAMATH_CALUDE_change_calculation_l3579_357926

/-- Given an apple cost of $0.75 and a payment of $5, the change returned is $4.25. -/
theorem change_calculation (apple_cost payment : ℚ) (h1 : apple_cost = 0.75) (h2 : payment = 5) :
  payment - apple_cost = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_change_calculation_l3579_357926


namespace NUMINAMATH_CALUDE_mary_seashells_l3579_357970

theorem mary_seashells (jessica_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : jessica_seashells = 41)
  (h2 : total_seashells = 59) :
  total_seashells - jessica_seashells = 18 :=
by sorry

end NUMINAMATH_CALUDE_mary_seashells_l3579_357970


namespace NUMINAMATH_CALUDE_min_sum_m_n_l3579_357905

theorem min_sum_m_n (m n : ℕ+) (h1 : 45 * m = n ^ 3) (h2 : ∃ k : ℕ+, n = 5 * k) :
  (∀ m' n' : ℕ+, 45 * m' = n' ^ 3 → (∃ k' : ℕ+, n' = 5 * k') → m + n ≤ m' + n') →
  m + n = 90 := by
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l3579_357905


namespace NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l3579_357928

theorem continuous_additive_function_is_linear 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_additive : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_continuous_additive_function_is_linear_l3579_357928


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3579_357969

theorem complex_magnitude_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 6) 
  (h2 : z + 9 / z = r) : 
  Complex.abs z = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3579_357969


namespace NUMINAMATH_CALUDE_volunteers_in_2002_l3579_357991

/-- The number of volunteers after n years, given an initial number and annual increase rate -/
def volunteers (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

/-- Theorem: The number of volunteers in 2002 will be 6075, given the initial conditions -/
theorem volunteers_in_2002 :
  volunteers 1200 (1/2) 4 = 6075 := by
  sorry

#eval volunteers 1200 (1/2) 4

end NUMINAMATH_CALUDE_volunteers_in_2002_l3579_357991


namespace NUMINAMATH_CALUDE_min_value_fraction_l3579_357956

theorem min_value_fraction (x : ℝ) (h : x > 7) :
  (x^2 + 49) / (x - 7) ≥ 7 + 14 * Real.sqrt 2 ∧
  ∃ y > 7, (y^2 + 49) / (y - 7) = 7 + 14 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3579_357956


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3579_357908

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (6 - 5 * z) = 7 :=
by
  -- The unique solution is z = -43/5
  use -43/5
  constructor
  · -- Prove that -43/5 satisfies the equation
    sorry
  · -- Prove that any solution must equal -43/5
    sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3579_357908


namespace NUMINAMATH_CALUDE_simplify_fraction_l3579_357939

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3579_357939


namespace NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_proof_l3579_357995

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The smallest four-digit palindrome divisible by 3 -/
def smallest_four_digit_palindrome_div_by_3 : ℕ := 2112

theorem smallest_four_digit_palindrome_div_by_3_proof :
  is_four_digit_palindrome smallest_four_digit_palindrome_div_by_3 ∧
  smallest_four_digit_palindrome_div_by_3 % 3 = 0 ∧
  ∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 → n ≥ smallest_four_digit_palindrome_div_by_3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_palindrome_div_by_3_proof_l3579_357995
