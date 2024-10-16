import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2351_235114

theorem regular_polygon_diagonals (n : ℕ) : n > 2 →
  (n * (n - 3)) / 2 = 20 → n = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2351_235114


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l2351_235139

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l2351_235139


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l2351_235136

def number_of_people : ℕ := 8

-- Define a function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (restricted_pair : ℕ) : ℕ :=
  Nat.factorial n - Nat.factorial (n - 1) * restricted_pair

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements number_of_people 2 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l2351_235136


namespace NUMINAMATH_CALUDE_arrangement_speeches_not_adjacent_l2351_235175

theorem arrangement_speeches_not_adjacent (n : ℕ) (m : ℕ) :
  n = 5 ∧ m = 3 →
  (n.factorial * (n + 1).factorial / ((n + 1 - m).factorial)) = 14400 :=
sorry

end NUMINAMATH_CALUDE_arrangement_speeches_not_adjacent_l2351_235175


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2351_235158

-- Define the sets A and B
def A : Set ℝ := {x | x < -3}
def B : Set ℝ := {x | x > -4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -4 < x ∧ x < -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2351_235158


namespace NUMINAMATH_CALUDE_equation_solution_l2351_235199

theorem equation_solution : 
  ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2351_235199


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_sum_of_fraction_parts_l2351_235143

/-- Recurrence relation for sequences without three consecutive 1s -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element binary sequence not containing three consecutive 1s -/
theorem probability_no_three_consecutive_ones : 
  (b 12 : ℚ) / 2^12 = 927 / 4096 := by sorry

/-- The sum of numerator and denominator of the probability fraction -/
theorem sum_of_fraction_parts : 927 + 4096 = 5023 := by sorry

end NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_sum_of_fraction_parts_l2351_235143


namespace NUMINAMATH_CALUDE_infinite_commuting_functions_l2351_235190

/-- Given a bijective function f from R to R, there exists an infinite number of functions g 
    from R to R such that f(g(x)) = g(f(x)) for all x in R. -/
theorem infinite_commuting_functions 
  (f : ℝ → ℝ) 
  (hf : Function.Bijective f) : 
  ∃ (S : Set (ℝ → ℝ)), Set.Infinite S ∧ ∀ g ∈ S, ∀ x, f (g x) = g (f x) := by
  sorry

end NUMINAMATH_CALUDE_infinite_commuting_functions_l2351_235190


namespace NUMINAMATH_CALUDE_manicure_cost_l2351_235130

/-- The cost of a manicure before tip, given the total amount paid and tip percentage. -/
theorem manicure_cost (total_paid : ℝ) (tip_percentage : ℝ) (cost : ℝ) : 
  total_paid = 39 → 
  tip_percentage = 0.30 → 
  cost * (1 + tip_percentage) = total_paid → 
  cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_manicure_cost_l2351_235130


namespace NUMINAMATH_CALUDE_willy_stuffed_animals_l2351_235155

/-- The number of stuffed animals Willy's mom gave him for his birthday -/
def moms_gift : ℕ := 2

/-- Willy's initial number of stuffed animals -/
def initial_count : ℕ := 10

/-- The factor by which Willy's dad increases his stuffed animal count -/
def dad_factor : ℕ := 3

/-- The total number of stuffed animals Willy has at the end -/
def final_count : ℕ := 48

theorem willy_stuffed_animals :
  initial_count + moms_gift + dad_factor * (initial_count + moms_gift) = final_count :=
by sorry

end NUMINAMATH_CALUDE_willy_stuffed_animals_l2351_235155


namespace NUMINAMATH_CALUDE_initial_crayons_l2351_235173

theorem initial_crayons (crayons_left : ℕ) (crayons_lost : ℕ) 
  (h1 : crayons_left = 134) 
  (h2 : crayons_lost = 345) : 
  crayons_left + crayons_lost = 479 := by
  sorry

end NUMINAMATH_CALUDE_initial_crayons_l2351_235173


namespace NUMINAMATH_CALUDE_mutter_lagaan_payment_l2351_235137

theorem mutter_lagaan_payment (total_lagaan : ℝ) (mutter_percentage : ℝ) :
  total_lagaan = 344000 →
  mutter_percentage = 0.23255813953488372 →
  mutter_percentage / 100 * total_lagaan = 800 := by
sorry

end NUMINAMATH_CALUDE_mutter_lagaan_payment_l2351_235137


namespace NUMINAMATH_CALUDE_polynomial_intersection_l2351_235118

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_intersection (a b c d : ℝ) :
  (∃ (x : ℝ), f a b x = g c d x ∧ x = 50 ∧ f a b x = -200) →
  (g c d (-a/2) = 0) →
  (f a b (-c/2) = 0) →
  (∃ (m : ℝ), (∀ (x : ℝ), f a b x ≥ m) ∧ (∀ (x : ℝ), g c d x ≥ m) ∧
               (∃ (x₁ x₂ : ℝ), f a b x₁ = m ∧ g c d x₂ = m)) →
  a + c = -200 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l2351_235118


namespace NUMINAMATH_CALUDE_erica_ride_percentage_longer_l2351_235119

-- Define the ride times for Dave, Chuck, and Erica
def dave_ride_time : ℕ := 10
def chuck_ride_time : ℕ := 5 * dave_ride_time
def erica_ride_time : ℕ := 65

-- Define the percentage difference
def percentage_difference : ℚ := (erica_ride_time - chuck_ride_time : ℚ) / chuck_ride_time * 100

-- Theorem statement
theorem erica_ride_percentage_longer :
  percentage_difference = 30 := by sorry

end NUMINAMATH_CALUDE_erica_ride_percentage_longer_l2351_235119


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2351_235168

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + x₂^3 + x₃^3 = 11 ∧ 
  x₁ + x₂ + x₃ = 2 ∧
  x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -1 ∧
  x₁ * x₂ * x₃ = -1 ∧
  x₁^3 - 2*x₁^2 - x₁ + 1 = 0 ∧
  x₂^3 - 2*x₂^2 - x₂ + 1 = 0 ∧
  x₃^3 - 2*x₃^2 - x₃ + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2351_235168


namespace NUMINAMATH_CALUDE_line_AB_equation_l2351_235182

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- Point P -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line AB -/
def on_line_AB (x y : ℝ) : Prop := ∃ (t : ℝ), x = 8 + t ∧ y = 1 + 2*t

/-- A and B are intersection points of line AB and the hyperbola -/
def A_B_intersection (A B : ℝ × ℝ) : Prop :=
  on_line_AB A.1 A.2 ∧ on_line_AB B.1 B.2 ∧
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2

/-- P is the midpoint of AB -/
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

/-- The main theorem -/
theorem line_AB_equation :
  ∃ (A B : ℝ × ℝ), A_B_intersection A B ∧ P_is_midpoint A B →
  ∀ (x y : ℝ), on_line_AB x y ↔ 2*x - y - 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_l2351_235182


namespace NUMINAMATH_CALUDE_fencing_cost_for_specific_plot_l2351_235101

/-- Represents a rectangular plot with its dimensions in meters -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Calculates the total cost of fencing a plot given the cost per meter -/
def fencingCost (plot : RectangularPlot) (costPerMeter : ℝ) : ℝ :=
  costPerMeter * perimeter plot

/-- Theorem stating the total cost of fencing for a specific rectangular plot -/
theorem fencing_cost_for_specific_plot :
  let plot : RectangularPlot := { length := 60, breadth := 40 }
  let costPerMeter : ℝ := 26.5
  fencingCost plot costPerMeter = 5300 := by
  sorry

#check fencing_cost_for_specific_plot

end NUMINAMATH_CALUDE_fencing_cost_for_specific_plot_l2351_235101


namespace NUMINAMATH_CALUDE_log_division_simplification_l2351_235172

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2351_235172


namespace NUMINAMATH_CALUDE_cos_negative_1320_degrees_l2351_235180

theorem cos_negative_1320_degrees : Real.cos (-(1320 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_1320_degrees_l2351_235180


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2351_235192

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2351_235192


namespace NUMINAMATH_CALUDE_solve_equation_l2351_235131

theorem solve_equation (x : ℝ) :
  3 * (x - 5) = 3 * (18 - 5) → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2351_235131


namespace NUMINAMATH_CALUDE_candy_probability_l2351_235128

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def same_color_probability : ℚ :=
  678 / 1735

theorem candy_probability :
  let first_pick := 2
  let second_pick := 2
  let remaining_candies := total_candies - first_pick
  (red_candies.choose first_pick * (red_candies - first_pick).choose second_pick +
   blue_candies.choose first_pick * (blue_candies - first_pick).choose second_pick +
   (red_candies.choose 1 * blue_candies.choose 1) * 
   ((red_candies - 1).choose 1 * (blue_candies - 1).choose 1)) /
  (total_candies.choose first_pick * remaining_candies.choose second_pick) =
  same_color_probability := by
sorry

end NUMINAMATH_CALUDE_candy_probability_l2351_235128


namespace NUMINAMATH_CALUDE_lineup_organization_l2351_235176

/-- The number of ways to organize a football lineup -/
def organize_lineup (total_members : ℕ) (defensive_linemen : ℕ) : ℕ :=
  defensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem: The number of ways to organize a lineup for a team with 7 members,
    of which 4 can play defensive lineman, is 480 -/
theorem lineup_organization :
  organize_lineup 7 4 = 480 := by
  sorry

end NUMINAMATH_CALUDE_lineup_organization_l2351_235176


namespace NUMINAMATH_CALUDE_pencil_count_l2351_235147

-- Define the number of items in the pencil case
def total_items : ℕ := 13

-- Define the relationship between pens and pencils
def pen_pencil_relation (pencils : ℕ) : ℕ := 2 * pencils

-- Define the number of erasers
def erasers : ℕ := 1

-- Theorem statement
theorem pencil_count : 
  ∃ (pencils : ℕ), 
    pencils + pen_pencil_relation pencils + erasers = total_items ∧ 
    pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2351_235147


namespace NUMINAMATH_CALUDE_a2_value_l2351_235132

theorem a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 = 
    a₀ + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + 
    a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7) →
  a₂ = 56 := by
sorry

end NUMINAMATH_CALUDE_a2_value_l2351_235132


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_set_l2351_235184

theorem sin_cos_equation_solution_set (x : ℝ) : 
  Real.sin (x / 2) - Real.cos (x / 2) = 1 ↔ 
  (∃ k : ℤ, x = π * (1 + 4 * k) ∨ x = 2 * π * (1 + 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_set_l2351_235184


namespace NUMINAMATH_CALUDE_min_pieces_to_find_both_l2351_235159

-- Define the grid
def Grid := Fin 8 → Fin 8 → Bool

-- Define the properties of the grid
def has_fish (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_sausage (g : Grid) (i j : Fin 8) : Prop := g i j = true
def has_both (g : Grid) (i j : Fin 8) : Prop := has_fish g i j ∧ has_sausage g i j

-- Define the conditions
def valid_grid (g : Grid) : Prop :=
  (∃ i j k l m n : Fin 8, has_fish g i j ∧ has_fish g k l ∧ has_fish g m n ∧ 
    ¬(i = k ∧ j = l) ∧ ¬(i = m ∧ j = n) ∧ ¬(k = m ∧ l = n)) ∧
  (∃ i j k l : Fin 8, has_sausage g i j ∧ has_sausage g k l ∧ ¬(i = k ∧ j = l)) ∧
  (∃! i j : Fin 8, has_both g i j) ∧
  (∀ i j : Fin 6, ∃ k l m n : Fin 8, k ≥ i ∧ k < i + 6 ∧ l ≥ j ∧ l < j + 6 ∧
    m ≥ i ∧ m < i + 6 ∧ n ≥ j ∧ n < j + 6 ∧ has_fish g k l ∧ has_fish g m n ∧ ¬(k = m ∧ l = n)) ∧
  (∀ i j : Fin 6, ∃! k l : Fin 8, k ≥ i ∧ k < i + 3 ∧ l ≥ j ∧ l < j + 3 ∧ has_sausage g k l)

-- Define the theorem
theorem min_pieces_to_find_both (g : Grid) (h : valid_grid g) :
  ∃ s : Finset (Fin 8 × Fin 8), s.card = 5 ∧
    (∀ t : Finset (Fin 8 × Fin 8), t.card < 5 → 
      ∃ i j : Fin 8, has_both g i j ∧ (i, j) ∉ t) ∧
    (∀ i j : Fin 8, has_both g i j → (i, j) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_min_pieces_to_find_both_l2351_235159


namespace NUMINAMATH_CALUDE_quadratic_roots_complex_l2351_235183

theorem quadratic_roots_complex (x : ℂ) :
  x^2 - 6*x + 25 = 0 ↔ x = 3 + 4*I ∨ x = 3 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_complex_l2351_235183


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l2351_235167

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l2351_235167


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2351_235120

theorem sum_of_fifth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 4)
  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2351_235120


namespace NUMINAMATH_CALUDE_intersection_point_l2351_235196

/-- The function f(x) = x^2 + x - 2 -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- The point (0, -2) -/
def point : ℝ × ℝ := (0, -2)

/-- Theorem: The point (0, -2) is the intersection point of f(x) with the y-axis -/
theorem intersection_point :
  (point.1 = 0 ∧ point.2 = f point.1) ∧
  ∀ y : ℝ, (0, y) ≠ point → f 0 ≠ y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l2351_235196


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2351_235191

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-3) - (-2) + 3 * k - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2351_235191


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_m_range_l2351_235160

theorem intersection_in_fourth_quadrant_implies_m_range 
  (m : ℝ) 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y, line1 x y ↔ x + y - 3*m = 0)
  (h2 : ∀ x y, line2 x y ↔ 2*x - y + 2*m - 1 = 0)
  (h_intersect : ∃ x y, line1 x y ∧ line2 x y ∧ x > 0 ∧ y < 0) :
  -1 < m ∧ m < 1/8 := by
sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_m_range_l2351_235160


namespace NUMINAMATH_CALUDE_large_to_medium_ratio_l2351_235127

/-- Represents the number of ceiling lights of each size -/
structure CeilingLights where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of bulbs needed for the given ceiling lights -/
def totalBulbs (lights : CeilingLights) : ℕ :=
  lights.small + 2 * lights.medium + 3 * lights.large

/-- The main theorem stating the ratio of large to medium ceiling lights -/
theorem large_to_medium_ratio (lights : CeilingLights) : 
  lights.medium = 12 →
  lights.small = lights.medium + 10 →
  totalBulbs lights = 118 →
  lights.large = 2 * lights.medium :=
by sorry

end NUMINAMATH_CALUDE_large_to_medium_ratio_l2351_235127


namespace NUMINAMATH_CALUDE_max_value_theorem_range_of_a_l2351_235157

-- Define the constraint function
def constraint (x y z : ℝ) : Prop := x^2 + y^2 + z^2 = 1

-- Define the objective function
def objective (x y z : ℝ) : ℝ := x + 2*y + 2*z

-- Theorem 1: Maximum value of the objective function
theorem max_value_theorem (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_constraint : constraint x y z) :
  objective x y z ≤ 3 :=
sorry

-- Theorem 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → constraint x y z → |a - 3| ≥ objective x y z) ↔
  a ≤ 0 ∨ a ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_range_of_a_l2351_235157


namespace NUMINAMATH_CALUDE_min_shirts_for_acme_savings_l2351_235195

def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

theorem min_shirts_for_acme_savings : ∃ n : ℕ, n = 13 ∧ ∀ x : ℕ, x ≥ n → acme_cost x < gamma_cost x :=
sorry

end NUMINAMATH_CALUDE_min_shirts_for_acme_savings_l2351_235195


namespace NUMINAMATH_CALUDE_village_male_population_l2351_235170

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_parts : ℕ
  male_parts : ℕ

/-- Calculates the number of males in the village -/
def num_males (v : Village) : ℕ :=
  v.total_population * v.male_parts / v.num_parts

theorem village_male_population (v : Village) 
  (h1 : v.total_population = 600)
  (h2 : v.num_parts = 4)
  (h3 : v.male_parts = 2) : 
  num_males v = 300 := by
  sorry

#check village_male_population

end NUMINAMATH_CALUDE_village_male_population_l2351_235170


namespace NUMINAMATH_CALUDE_tangent_through_origin_l2351_235108

theorem tangent_through_origin (x : ℝ) :
  (∃ y : ℝ, y = Real.exp x ∧ 
   (Real.exp x) * (0 - x) = 0 - y) →
  x = 1 ∧ Real.exp x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l2351_235108


namespace NUMINAMATH_CALUDE_max_last_place_wins_theorem_l2351_235178

/-- Represents a baseball league -/
structure BaseballLeague where
  teams : ℕ
  gamesPerPair : ℕ
  noTies : Bool
  constantDifference : Bool

/-- Calculates the maximum number of games the last-place team could have won -/
def maxLastPlaceWins (league : BaseballLeague) : ℕ :=
  if league.teams = 14 ∧ league.gamesPerPair = 10 ∧ league.noTies ∧ league.constantDifference then
    52
  else
    0  -- Default value for other cases

/-- Theorem stating the maximum number of games the last-place team could have won -/
theorem max_last_place_wins_theorem (league : BaseballLeague) :
  league.teams = 14 ∧ 
  league.gamesPerPair = 10 ∧ 
  league.noTies ∧ 
  league.constantDifference →
  maxLastPlaceWins league = 52 := by
  sorry

#eval maxLastPlaceWins { teams := 14, gamesPerPair := 10, noTies := true, constantDifference := true }

end NUMINAMATH_CALUDE_max_last_place_wins_theorem_l2351_235178


namespace NUMINAMATH_CALUDE_quadrilateral_problem_l2351_235177

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Find the intersection point of two lines -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_problem (PQRS : Quadrilateral) (T : Point) :
  isConvex PQRS →
  isPerpendicular PQRS.R PQRS.S PQRS.P PQRS.Q →
  isPerpendicular PQRS.P PQRS.Q PQRS.R PQRS.S →
  distance PQRS.R PQRS.S = 52 →
  distance PQRS.P PQRS.Q = 39 →
  isPerpendicular PQRS.Q T PQRS.P PQRS.S →
  T = lineIntersection PQRS.P PQRS.Q PQRS.Q T →
  distance PQRS.P T = 25 →
  distance PQRS.Q T = 14 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_problem_l2351_235177


namespace NUMINAMATH_CALUDE_largest_root_of_g_l2351_235152

def g (x : ℝ) : ℝ := 24 * x^4 - 34 * x^2 + 6

theorem largest_root_of_g :
  ∃ (r : ℝ), r = 1/2 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_largest_root_of_g_l2351_235152


namespace NUMINAMATH_CALUDE_game_draw_fraction_l2351_235189

theorem game_draw_fraction (jack_win : ℚ) (emma_win : ℚ) (draw : ℚ) : 
  jack_win = 4/9 → emma_win = 5/14 → draw = 1 - (jack_win + emma_win) → draw = 25/126 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l2351_235189


namespace NUMINAMATH_CALUDE_garden_items_cost_l2351_235126

/-- Proves the costs of three items given their total cost and price relationships -/
theorem garden_items_cost (total : ℝ) (bench : ℝ) (table : ℝ) (umbrella : ℝ)
  (h1 : total = 765)
  (h2 : table = 2 * bench)
  (h3 : umbrella = 3 * bench)
  (h4 : total = bench + table + umbrella) :
  bench = 127.5 ∧ table = 255 ∧ umbrella = 382.5 := by
  sorry

end NUMINAMATH_CALUDE_garden_items_cost_l2351_235126


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2351_235112

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x^2 ∧ 6*s^2 = 4*x) → x = 1/216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2351_235112


namespace NUMINAMATH_CALUDE_distance_PQ_is_2_25_l2351_235169

/-- The distance between two points on a ruler -/
def distance_on_ruler (p q : ℚ) : ℚ := q - p

/-- The position of point P on the ruler -/
def P : ℚ := 1/2

/-- The position of point Q on the ruler -/
def Q : ℚ := 2 + 3/4

theorem distance_PQ_is_2_25 : distance_on_ruler P Q = 2.25 := by sorry

end NUMINAMATH_CALUDE_distance_PQ_is_2_25_l2351_235169


namespace NUMINAMATH_CALUDE_mobile_phone_cost_l2351_235111

def refrigerator_cost : ℝ := 15000
def refrigerator_loss_percent : ℝ := 0.04
def mobile_profit_percent : ℝ := 0.10
def overall_profit : ℝ := 200

theorem mobile_phone_cost (mobile_cost : ℝ) : 
  (refrigerator_cost * (1 - refrigerator_loss_percent) + mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = overall_profit →
  mobile_cost = 6000 := by
sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_l2351_235111


namespace NUMINAMATH_CALUDE_largest_m_is_nine_l2351_235125

/-- A quadratic function f(x) = ax² + bx + c satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  lower_bound : ∀ x : ℝ, a * x^2 + b * x + c ≥ x
  upper_bound : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating that the largest m > 1 satisfying the given conditions is 9 -/
theorem largest_m_is_nine (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  ∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_m_is_nine_l2351_235125


namespace NUMINAMATH_CALUDE_fraction_defined_iff_not_five_l2351_235106

theorem fraction_defined_iff_not_five (x : ℝ) : IsRegular (x - 5)⁻¹ ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_fraction_defined_iff_not_five_l2351_235106


namespace NUMINAMATH_CALUDE_ship_passengers_l2351_235124

theorem ship_passengers :
  let total : ℕ := 900
  let north_america : ℚ := 1/4
  let europe : ℚ := 2/15
  let africa : ℚ := 1/5
  let asia : ℚ := 1/6
  let south_america : ℚ := 1/12
  let oceania : ℚ := 1/20
  let other_regions : ℕ := 105
  (north_america + europe + africa + asia + south_america + oceania) * total + other_regions = total :=
by sorry

end NUMINAMATH_CALUDE_ship_passengers_l2351_235124


namespace NUMINAMATH_CALUDE_point_a_coordinates_l2351_235142

/-- A point on the x-axis at a distance of 3 units from the origin -/
structure PointA where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_from_origin : x^2 + y^2 = 3^2

theorem point_a_coordinates (A : PointA) : (A.x = 3 ∧ A.y = 0) ∨ (A.x = -3 ∧ A.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_a_coordinates_l2351_235142


namespace NUMINAMATH_CALUDE_calculation_proof_l2351_235154

theorem calculation_proof : (30 / (8 + 2 - 5)) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2351_235154


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2351_235161

theorem angle_C_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) : 
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  C = π / 6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2351_235161


namespace NUMINAMATH_CALUDE_opposite_numbers_and_unit_absolute_value_l2351_235116

theorem opposite_numbers_and_unit_absolute_value 
  (a b c : ℝ) 
  (h1 : a + b = 0) 
  (h2 : abs c = 1) : 
  a + b - c = 1 ∨ a + b - c = -1 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_unit_absolute_value_l2351_235116


namespace NUMINAMATH_CALUDE_score_difference_l2351_235146

def sammy_score : ℝ := 20

def gab_score : ℝ := 2 * sammy_score

def cher_score : ℝ := 2 * gab_score

def alex_score : ℝ := cher_score * 1.1

def team1_score : ℝ := sammy_score + gab_score + cher_score + alex_score

def opponent_initial_score : ℝ := 85

def opponent_final_score : ℝ := opponent_initial_score * 1.5

theorem score_difference :
  team1_score - opponent_final_score = 100.5 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l2351_235146


namespace NUMINAMATH_CALUDE_total_candies_l2351_235113

theorem total_candies (red_candies blue_candies : ℕ) 
  (h1 : red_candies = 145) 
  (h2 : blue_candies = 3264) : 
  red_candies + blue_candies = 3409 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l2351_235113


namespace NUMINAMATH_CALUDE_ali_wallet_final_amount_l2351_235164

def initial_wallet_value : ℕ := 7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1

def grocery_spending : ℕ := 65

def change_received : ℕ := 1 * 5 + 5 * 1

def friend_payment : ℕ := 2 * 20 + 2 * 1

theorem ali_wallet_final_amount :
  initial_wallet_value - grocery_spending + change_received + friend_payment = 150 := by
  sorry

end NUMINAMATH_CALUDE_ali_wallet_final_amount_l2351_235164


namespace NUMINAMATH_CALUDE_phone_number_probability_l2351_235162

theorem phone_number_probability :
  let first_three_options : ℕ := 2
  let last_four_arrangements : ℕ := 24
  let total_numbers : ℕ := first_three_options * last_four_arrangements
  let correct_numbers : ℕ := 1
  (correct_numbers : ℚ) / total_numbers = 1 / 48 := by
sorry

end NUMINAMATH_CALUDE_phone_number_probability_l2351_235162


namespace NUMINAMATH_CALUDE_apple_ratio_is_one_to_two_l2351_235117

/-- Represents the number of golden delicious apples needed for one pint of cider -/
def golden_delicious_per_pint : ℕ := 20

/-- Represents the number of pink lady apples needed for one pint of cider -/
def pink_lady_per_pint : ℕ := 40

/-- Represents the number of farmhands -/
def num_farmhands : ℕ := 6

/-- Represents the number of apples a farmhand can pick per hour -/
def apples_per_hour : ℕ := 240

/-- Represents the number of hours worked -/
def hours_worked : ℕ := 5

/-- Represents the number of pints of cider that can be made -/
def pints_of_cider : ℕ := 120

/-- Theorem stating that the ratio of golden delicious apples to pink lady apples gathered is 1:2 -/
theorem apple_ratio_is_one_to_two :
  (golden_delicious_per_pint * pints_of_cider) / (pink_lady_per_pint * pints_of_cider) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_is_one_to_two_l2351_235117


namespace NUMINAMATH_CALUDE_museum_groups_l2351_235179

/-- Given a class trip to a museum with the following conditions:
  * The class has 18 students in total
  * Each student takes 4 minutes to go through the museum
  * It takes each group 24 minutes to go through the museum
  Prove that the number of groups Howard split the class into is 3 -/
theorem museum_groups (total_students : ℕ) (student_time : ℕ) (group_time : ℕ)
  (h1 : total_students = 18)
  (h2 : student_time = 4)
  (h3 : group_time = 24) :
  total_students / (group_time / student_time) = 3 :=
sorry

end NUMINAMATH_CALUDE_museum_groups_l2351_235179


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l2351_235198

/-- Given a polynomial function f(x) = px³ - qx² + rx - s, 
    if f(1) = 4, then 2p + q - 3r + 2s = -8 -/
theorem polynomial_value_theorem (p q r s : ℝ) : 
  let f := fun (x : ℝ) => p * x^3 - q * x^2 + r * x - s
  (f 1 = 4) → (2*p + q - 3*r + 2*s = -8) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l2351_235198


namespace NUMINAMATH_CALUDE_factorization_3y_squared_minus_12_l2351_235121

theorem factorization_3y_squared_minus_12 (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3y_squared_minus_12_l2351_235121


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l2351_235163

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l2351_235163


namespace NUMINAMATH_CALUDE_joshua_share_l2351_235187

def total_amount : ℚ := 123.50
def joshua_multiplier : ℚ := 3.5
def jasmine_multiplier : ℚ := 0.75

theorem joshua_share :
  ∃ (justin_share : ℚ),
    justin_share + joshua_multiplier * justin_share + jasmine_multiplier * justin_share = total_amount ∧
    joshua_multiplier * justin_share = 82.32 :=
by sorry

end NUMINAMATH_CALUDE_joshua_share_l2351_235187


namespace NUMINAMATH_CALUDE_triangle_inradius_l2351_235153

/-- Given a triangle with perimeter 24 cm and area 30 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 24 → area = 30 → inradius * (perimeter / 2) = area → inradius = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2351_235153


namespace NUMINAMATH_CALUDE_bridget_apples_l2351_235110

theorem bridget_apples (x : ℕ) : 
  (x / 3 : ℚ) + 5 + 2 + 8 = x → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_l2351_235110


namespace NUMINAMATH_CALUDE_inequality_contradiction_l2351_235174

theorem inequality_contradiction (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l2351_235174


namespace NUMINAMATH_CALUDE_max_base_seven_digit_sum_l2351_235165

/-- Represents a positive integer in base 7 --/
def BaseSevenDigits := List Nat

/-- Converts a positive integer to its base 7 representation --/
def toBaseSeven (n : Nat) : BaseSevenDigits :=
  sorry

/-- Calculates the sum of digits in a base 7 representation --/
def sumBaseSevenDigits (digits : BaseSevenDigits) : Nat :=
  sorry

/-- Checks if a base 7 representation is valid (all digits < 7) --/
def isValidBaseSeven (digits : BaseSevenDigits) : Prop :=
  sorry

/-- Converts a base 7 representation back to a natural number --/
def fromBaseSeven (digits : BaseSevenDigits) : Nat :=
  sorry

/-- The main theorem --/
theorem max_base_seven_digit_sum :
  ∀ n : Nat, n > 0 → n < 3000 →
    ∃ (max : Nat),
      max = 24 ∧
      sumBaseSevenDigits (toBaseSeven n) ≤ max ∧
      (∀ m : Nat, m > 0 → m < 3000 →
        sumBaseSevenDigits (toBaseSeven m) ≤ max) :=
by sorry

end NUMINAMATH_CALUDE_max_base_seven_digit_sum_l2351_235165


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2351_235149

theorem smallest_sum_of_sequence (A B C D : ℕ+) : 
  (∃ (k : ℕ), k > 0 ∧ 
    A.val = 9 * k - 12 ∧ 
    B.val = 9 * k ∧ 
    C.val = 15 * k ∧ 
    D.val = 25 * k) →
  (C.val : ℚ) / (B.val : ℚ) = 5 / 3 →
  (∀ n : ℕ+, n < A → 
    ¬∃ (k : ℕ), k > 0 ∧ 
      n.val = 9 * k - 12 ∧ 
      B.val = 9 * k ∧ 
      C.val = 15 * k ∧ 
      D.val = 25 * k) →
  A.val + B.val + C.val + D.val = 104 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2351_235149


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_directrix_l2351_235141

/-- An ellipse with equation x^2 + k*y^2 = 1 -/
def ellipse (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + k * p.2^2 = 1}

/-- A hyperbola with equation x^2/4 - y^2/5 = 1 -/
def hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 5 = 1}

/-- The directrix of a conic section -/
def directrix (c : Set (ℝ × ℝ)) : Set ℝ := sorry

theorem ellipse_hyperbola_same_directrix (k : ℝ) :
  directrix (ellipse k) = directrix hyperbola → k = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_directrix_l2351_235141


namespace NUMINAMATH_CALUDE_balloons_in_park_l2351_235104

/-- The number of balloons Allan and Jake have in the park -/
def total_balloons (allan_initial : ℕ) (jake : ℕ) (allan_bought : ℕ) : ℕ :=
  (allan_initial + allan_bought) + jake

/-- Theorem: Allan and Jake have 10 balloons in total -/
theorem balloons_in_park :
  total_balloons 3 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l2351_235104


namespace NUMINAMATH_CALUDE_students_in_neither_course_l2351_235138

theorem students_in_neither_course (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : coding = 90)
  (h3 : robotics = 70)
  (h4 : both = 25) :
  total - (coding + robotics - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_in_neither_course_l2351_235138


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2351_235194

/-- Given a sale where the gain is $20 and the gain percentage is 25%, 
    prove that the selling price is $100. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 20 →
  gain_percentage = 25 →
  ∃ (cost_price selling_price : ℝ),
    gain = gain_percentage / 100 * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2351_235194


namespace NUMINAMATH_CALUDE_clinton_meal_days_l2351_235148

/-- The number of days Clinton buys the meal -/
def days_buying_meal (meal_cost : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent / meal_cost

/-- Theorem: Given the meal cost and total spent, prove the number of days is 5 -/
theorem clinton_meal_days :
  let meal_cost : ℚ := 7
  let total_spent : ℚ := 35
  days_buying_meal meal_cost total_spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_clinton_meal_days_l2351_235148


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2351_235102

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a * b ≠ 0) → ¬(a ≠ 0 ∧ b ≠ 0)) ↔ ((a = 0 ∨ b = 0) → a * b = 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2351_235102


namespace NUMINAMATH_CALUDE_product_digit_sum_l2351_235109

def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

theorem product_digit_sum :
  let product := number1 * number2
  let tens_digit := (product / 10) % 10
  let units_digit := product % 10
  tens_digit + units_digit = 9 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2351_235109


namespace NUMINAMATH_CALUDE_flag_design_count_l2351_235123

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flags : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  total_flags = 27 :=
by sorry

end NUMINAMATH_CALUDE_flag_design_count_l2351_235123


namespace NUMINAMATH_CALUDE_parabola_focus_l2351_235156

/-- Represents a parabola with equation y^2 = ax and directrix x = -1 -/
structure Parabola where
  a : ℝ
  directrix : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Theorem stating that the focus of the given parabola is at (1, 0) -/
theorem parabola_focus (p : Parabola) (h1 : p.directrix = -1) : focus p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2351_235156


namespace NUMINAMATH_CALUDE_complex_power_225_deg_18_l2351_235188

theorem complex_power_225_deg_18 : 
  (Complex.exp (Complex.I * Real.pi * (5 / 4)))^18 = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_225_deg_18_l2351_235188


namespace NUMINAMATH_CALUDE_car_bike_speed_ratio_l2351_235134

/-- Proves that the ratio of the average speed of a car to the average speed of a bike is 1.8 -/
theorem car_bike_speed_ratio :
  let tractor_distance : ℝ := 575
  let tractor_time : ℝ := 25
  let car_distance : ℝ := 331.2
  let car_time : ℝ := 4
  let tractor_speed : ℝ := tractor_distance / tractor_time
  let bike_speed : ℝ := 2 * tractor_speed
  let car_speed : ℝ := car_distance / car_time
  car_speed / bike_speed = 1.8 := by
sorry


end NUMINAMATH_CALUDE_car_bike_speed_ratio_l2351_235134


namespace NUMINAMATH_CALUDE_correct_answer_l2351_235140

theorem correct_answer (x : ℤ) (h : x - 8 = 32) : x + 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l2351_235140


namespace NUMINAMATH_CALUDE_common_difference_is_one_l2351_235151

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem common_difference_is_one
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 1)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 9)) :
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_one_l2351_235151


namespace NUMINAMATH_CALUDE_negation_disjunction_true_l2351_235193

theorem negation_disjunction_true (p q : Prop) : 
  (p ∧ q) = False → (¬p ∨ ¬q) = True := by sorry

end NUMINAMATH_CALUDE_negation_disjunction_true_l2351_235193


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l2351_235150

-- Define the compositions of mixtures X and Y
def x_ryegrass : ℝ := 0.4
def x_bluegrass : ℝ := 0.6
def y_ryegrass : ℝ := 0.25
def y_fescue : ℝ := 0.75

-- Define the proportion of X in the final mixture
def x_proportion : ℝ := 0.3333333333333333

-- Define the proportion of Y in the final mixture
def y_proportion : ℝ := 1 - x_proportion

-- Theorem statement
theorem ryegrass_percentage_in_mixture :
  x_ryegrass * x_proportion + y_ryegrass * y_proportion = 0.3 := by sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l2351_235150


namespace NUMINAMATH_CALUDE_set_theory_propositions_l2351_235105

theorem set_theory_propositions (A B : Set α) : 
  (∀ a, a ∈ A → a ∈ A ∪ B) ∧
  (A ⊆ B → A ∪ B = B) ∧
  (A ∪ B = B → A ∩ B = A) ∧
  ¬(∀ a, a ∈ B → a ∈ A ∩ B) ∧
  ¬(∀ C, A ∪ B = B ∪ C → A = C) :=
by sorry

end NUMINAMATH_CALUDE_set_theory_propositions_l2351_235105


namespace NUMINAMATH_CALUDE_f_fixed_point_exists_f_fixed_point_19_pow_86_l2351_235185

def f (A : ℕ) : ℕ :=
  let digits := Nat.digits 10 A
  List.sum (List.zipWith (·*·) (List.reverse digits) (List.map (2^·) (List.range digits.length)))

theorem f_fixed_point_exists (A : ℕ) : ∃ k : ℕ, f (f^[k] A) = f^[k] A :=
sorry

theorem f_fixed_point_19_pow_86 : ∃ k : ℕ, f^[k] (19^86) = 19 :=
sorry

end NUMINAMATH_CALUDE_f_fixed_point_exists_f_fixed_point_19_pow_86_l2351_235185


namespace NUMINAMATH_CALUDE_basketball_probability_l2351_235171

theorem basketball_probability (p_free_throw p_high_school p_pro : ℚ) 
  (h1 : p_free_throw = 4/5)
  (h2 : p_high_school = 1/2)
  (h3 : p_pro = 1/3) :
  1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l2351_235171


namespace NUMINAMATH_CALUDE_B_power_difference_l2351_235135

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_difference : 
  B^30 - B^29 = !![2, 4; 0, 1] := by sorry

end NUMINAMATH_CALUDE_B_power_difference_l2351_235135


namespace NUMINAMATH_CALUDE_semicircle_area_l2351_235107

theorem semicircle_area (d : ℝ) (h : d = 11) : 
  (1/2) * π * (d/2)^2 = (121/8) * π := by sorry

end NUMINAMATH_CALUDE_semicircle_area_l2351_235107


namespace NUMINAMATH_CALUDE_middle_circle_radius_l2351_235103

/-- Given three circles in a geometric sequence with radii r₁, r₂, and r₃,
    where r₁ = 5 cm and r₃ = 20 cm, prove that r₂ = 10 cm. -/
theorem middle_circle_radius (r₁ r₂ r₃ : ℝ) 
    (h_geom_seq : r₂^2 = r₁ * r₃)
    (h_r₁ : r₁ = 5)
    (h_r₃ : r₃ = 20) : 
  r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l2351_235103


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l2351_235166

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : Nat :=
  13

/-- Theorem stating that for the given gumball machine configuration,
    the minimum number of gumballs needed to guarantee four of the same color is 13 -/
theorem min_gumballs_for_four_same_color_is_13 (machine : GumballMachine)
    (h : machine = { red := 12, white := 15, blue := 10, green := 7 }) :
    minGumballsForFourSameColor machine = 13 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_four_same_color_is_13_l2351_235166


namespace NUMINAMATH_CALUDE_monopoly_prefers_durable_coffee_machine_production_decision_l2351_235122

/-- Represents the type of coffee machine -/
inductive CoffeeMachineType
| Durable
| LowQuality

/-- Represents the market structure -/
inductive MarketStructure
| Monopoly
| PerfectlyCompetitive

/-- Represents a coffee machine -/
structure CoffeeMachine where
  type : CoffeeMachineType
  productionCost : ℝ

/-- Represents the consumer's utility from using a coffee machine -/
def consumerUtility : ℝ := 20

/-- Represents the lifespan of a coffee machine in periods -/
def machineLifespan (t : CoffeeMachineType) : ℕ :=
  match t with
  | CoffeeMachineType.Durable => 2
  | CoffeeMachineType.LowQuality => 1

/-- Calculates the profit for a monopolist selling a coffee machine -/
def monopolyProfit (m : CoffeeMachine) : ℝ :=
  (consumerUtility * machineLifespan m.type) - m.productionCost

/-- Theorem: In a monopoly, durable machines are produced when low-quality machine cost exceeds 6 -/
theorem monopoly_prefers_durable (c : ℝ) :
  let durableMachine : CoffeeMachine := ⟨CoffeeMachineType.Durable, 12⟩
  let lowQualityMachine : CoffeeMachine := ⟨CoffeeMachineType.LowQuality, c⟩
  monopolyProfit durableMachine > 2 * monopolyProfit lowQualityMachine ↔ c > 6 := by
  sorry

/-- Main theorem combining all conditions -/
theorem coffee_machine_production_decision 
  (marketStructure : MarketStructure) 
  (c : ℝ) : 
  (marketStructure = MarketStructure.Monopoly ∧ c > 6) ↔ 
  (∃ (d : CoffeeMachine), d.type = CoffeeMachineType.Durable ∧ 
   ∀ (l : CoffeeMachine), l.type = CoffeeMachineType.LowQuality → 
   monopolyProfit d > monopolyProfit l) := by
  sorry

end NUMINAMATH_CALUDE_monopoly_prefers_durable_coffee_machine_production_decision_l2351_235122


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2351_235129

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℤ) 
  (h1 : QuadraticFunction a b c 2 = 5)
  (h2 : ∀ x, QuadraticFunction a b c x ≥ QuadraticFunction a b c 1)
  (h3 : QuadraticFunction a b c 1 = 3) :
  a - b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2351_235129


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l2351_235115

theorem diophantine_equation_solution (k ℓ : ℤ) :
  5 * k + 3 * ℓ = 32 ↔ ∃ x : ℤ, k = -32 + 3 * x ∧ ℓ = 64 - 5 * x :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l2351_235115


namespace NUMINAMATH_CALUDE_gcd_7854_13843_l2351_235186

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7854_13843_l2351_235186


namespace NUMINAMATH_CALUDE_pythagorean_triples_l2351_235100

theorem pythagorean_triples (n m : ℕ) : 
  (n ≥ 3 ∧ Odd n) → 
  ((n^2 - 1) / 2)^2 + n^2 = ((n^2 + 1) / 2)^2 ∧
  (m > 1) →
  (m^2 - 1)^2 + (2*m)^2 = (m^2 + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_triples_l2351_235100


namespace NUMINAMATH_CALUDE_find_number_l2351_235181

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2351_235181


namespace NUMINAMATH_CALUDE_supplement_of_complement_35_l2351_235145

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (β : ℝ) : ℝ := 180 - β

/-- The original angle in degrees -/
def original_angle : ℝ := 35

/-- Theorem: The degree measure of the supplement of the complement of a 35-degree angle is 125° -/
theorem supplement_of_complement_35 : 
  supplement (complement original_angle) = 125 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_35_l2351_235145


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2351_235133

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → 
    ∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2) ↔ 
  m^2 = 1/3 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2351_235133


namespace NUMINAMATH_CALUDE_constant_area_l2351_235197

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point P on C₂
def P : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the line OP
def OP (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | ∃ t : ℝ, q.1 = t * p.1 ∧ q.2 = t * p.2}

-- Define the points A and B
def A (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 2 * p.2)
def B : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define B explicitly

-- Define the tangent line l to C₂ at P
def l (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | p.1 * q.1 + 4 * p.2 * q.2 = 4}

-- Define the points C and D
def C : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define C explicitly
def D : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define D explicitly

-- Define the area of quadrilateral ACBD
def area_ACBD (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem constant_area (p : ℝ × ℝ) (hp : P p) :
  area_ACBD p = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_constant_area_l2351_235197


namespace NUMINAMATH_CALUDE_tan_sum_special_angle_l2351_235144

theorem tan_sum_special_angle (θ : Real) (h : Real.tan θ = 1/3) :
  Real.tan (θ + π/4) = 2 := by sorry

end NUMINAMATH_CALUDE_tan_sum_special_angle_l2351_235144
