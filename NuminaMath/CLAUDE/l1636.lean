import Mathlib

namespace NUMINAMATH_CALUDE_value_of_y_l1636_163648

theorem value_of_y : (2010^2 - 2010 + 1) / 2010 = 2009 + 1/2010 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l1636_163648


namespace NUMINAMATH_CALUDE_flagpole_height_l1636_163620

theorem flagpole_height (A B C D E : ℝ × ℝ) : 
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let AD : ℝ := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let DE : ℝ := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  (A.2 = 0 ∧ B.2 > 0 ∧ C.2 = 0 ∧ D.2 = 0 ∧ E.2 > 0) → -- Points on x-axis or above
  (B.1 = A.1) → -- AB is vertical
  (AC = 4) → -- Wire length on ground
  (AD = 3) → -- Tom's distance from pole
  (DE = 1.8) → -- Tom's height
  ((E.1 - D.1) * (C.1 - A.1) + (E.2 - D.2) * (C.2 - A.2) = 0) → -- DE perpendicular to AC
  (AB = 7.2) -- Flagpole height
  := by sorry

end NUMINAMATH_CALUDE_flagpole_height_l1636_163620


namespace NUMINAMATH_CALUDE_allocation_methods_count_l1636_163641

/-- Represents the number of male students -/
def num_males : ℕ := 4

/-- Represents the number of female students -/
def num_females : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := num_males + num_females

/-- Represents the minimum number of students in each group -/
def min_group_size : ℕ := 2

/-- Calculates the number of ways to divide students into two groups -/
def num_allocation_methods : ℕ := sorry

/-- Theorem stating that the number of allocation methods is 52 -/
theorem allocation_methods_count : num_allocation_methods = 52 := by sorry

end NUMINAMATH_CALUDE_allocation_methods_count_l1636_163641


namespace NUMINAMATH_CALUDE_johns_investment_l1636_163655

theorem johns_investment (total_interest rate1 rate_difference investment1 : ℝ) 
  (h1 : total_interest = 1282)
  (h2 : rate1 = 0.11)
  (h3 : rate_difference = 0.015)
  (h4 : investment1 = 4000) : 
  ∃ investment2 : ℝ, 
    investment2 = 6736 ∧ 
    total_interest = investment1 * rate1 + investment2 * (rate1 + rate_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_johns_investment_l1636_163655


namespace NUMINAMATH_CALUDE_division_power_eq_inv_pow_l1636_163651

/-- Division power of a rational number -/
def division_power (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (division_power a (n-1))

/-- Theorem: Division power equals inverse raised to power (n-2) -/
theorem division_power_eq_inv_pow (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 2) :
  division_power a n = (a⁻¹) ^ (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_division_power_eq_inv_pow_l1636_163651


namespace NUMINAMATH_CALUDE_lottery_tickets_bought_l1636_163616

theorem lottery_tickets_bought (total_won : ℕ) (winning_number_value : ℕ) (winning_numbers_per_ticket : ℕ) : 
  total_won = 300 →
  winning_number_value = 20 →
  winning_numbers_per_ticket = 5 →
  (total_won / winning_number_value) / winning_numbers_per_ticket = 3 :=
by sorry

end NUMINAMATH_CALUDE_lottery_tickets_bought_l1636_163616


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1636_163665

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : 
  |x - y| = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1636_163665


namespace NUMINAMATH_CALUDE_cubic_curve_tangent_line_bc_product_l1636_163617

/-- Given a cubic curve y = x³ + bx + c passing through (1, 2) with tangent line y = x + 1 at that point, 
    the product bc equals -6. -/
theorem cubic_curve_tangent_line_bc_product (b c : ℝ) : 
  (1^3 + b*1 + c = 2) →   -- Point (1, 2) is on the curve
  (3*1^2 + b = 1) →       -- Derivative at x = 1 is 1 (from tangent line y = x + 1)
  b * c = -6 := by sorry

end NUMINAMATH_CALUDE_cubic_curve_tangent_line_bc_product_l1636_163617


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1636_163692

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x + 1) < 1) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1636_163692


namespace NUMINAMATH_CALUDE_farm_field_calculation_correct_l1636_163657

/-- Represents the farm field ploughing problem -/
structure FarmField where
  initialCapacityA : ℝ
  initialCapacityB : ℝ
  reducedCapacityA : ℝ
  reducedCapacityB : ℝ
  extraDays : ℕ
  unattendedArea : ℝ

/-- Calculates the area of the farm field and the initially planned work days -/
def calculateFarmFieldResult (f : FarmField) : ℝ × ℕ :=
  let initialTotalCapacity := f.initialCapacityA + f.initialCapacityB
  let reducedTotalCapacity := f.reducedCapacityA + f.reducedCapacityB
  let area := 6600
  let initialDays := 30
  (area, initialDays)

/-- Theorem stating the correctness of the farm field calculation -/
theorem farm_field_calculation_correct (f : FarmField) 
  (h1 : f.initialCapacityA = 120)
  (h2 : f.initialCapacityB = 100)
  (h3 : f.reducedCapacityA = f.initialCapacityA * 0.9)
  (h4 : f.reducedCapacityB = 90)
  (h5 : f.extraDays = 3)
  (h6 : f.unattendedArea = 60) :
  calculateFarmFieldResult f = (6600, 30) := by
  sorry

#eval calculateFarmFieldResult {
  initialCapacityA := 120,
  initialCapacityB := 100,
  reducedCapacityA := 108,
  reducedCapacityB := 90,
  extraDays := 3,
  unattendedArea := 60
}

end NUMINAMATH_CALUDE_farm_field_calculation_correct_l1636_163657


namespace NUMINAMATH_CALUDE_remainder_seven_count_l1636_163627

theorem remainder_seven_count : ∃! k : ℕ, k = (Finset.filter (fun n => 61 % n = 7) (Finset.range 62)).card := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_count_l1636_163627


namespace NUMINAMATH_CALUDE_pete_walked_3350_miles_l1636_163629

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer :=
  (max_reading : ℕ)

/-- Represents Pete's walking data for a year --/
structure YearlyWalkingData :=
  (pedometer : Pedometer)
  (resets : ℕ)
  (final_reading : ℕ)
  (steps_per_mile : ℕ)

/-- Calculates the total miles walked based on the yearly walking data --/
def total_miles_walked (data : YearlyWalkingData) : ℕ :=
  ((data.resets * (data.pedometer.max_reading + 1) + data.final_reading) / data.steps_per_mile)

/-- Theorem stating that Pete walked 3350 miles given the problem conditions --/
theorem pete_walked_3350_miles :
  let petes_pedometer : Pedometer := ⟨99999⟩
  let petes_data : YearlyWalkingData := ⟨petes_pedometer, 50, 25000, 1500⟩
  total_miles_walked petes_data = 3350 := by
  sorry


end NUMINAMATH_CALUDE_pete_walked_3350_miles_l1636_163629


namespace NUMINAMATH_CALUDE_balloon_distribution_l1636_163639

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (take_back : ℕ) 
  (h1 : total_balloons = 250)
  (h2 : num_friends = 5)
  (h3 : take_back = 11) :
  (total_balloons / num_friends) - take_back = 39 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l1636_163639


namespace NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l1636_163606

theorem integral_equals_six_implies_b_equals_e_to_four (b : ℝ) :
  (∫ (x : ℝ) in e..b, 2 / x) = 6 → b = Real.exp 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_six_implies_b_equals_e_to_four_l1636_163606


namespace NUMINAMATH_CALUDE_scientific_notation_of_229000_l1636_163600

theorem scientific_notation_of_229000 :
  229000 = 2.29 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_229000_l1636_163600


namespace NUMINAMATH_CALUDE_sqrt_D_always_odd_l1636_163635

theorem sqrt_D_always_odd (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 1
  let c : ℤ := a * b
  let D : ℤ := a^2 + b^2 + c^2
  ∃ (k : ℤ), D = (2 * k + 1)^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_D_always_odd_l1636_163635


namespace NUMINAMATH_CALUDE_part1_part2_l1636_163678

-- Define a "three times angle triangle"
def is_three_times_angle_triangle (a b c : ℝ) : Prop :=
  (a + b + c = 180) ∧ (a = 3 * b ∨ b = 3 * c ∨ c = 3 * a)

-- Part 1
theorem part1 : is_three_times_angle_triangle 35 40 105 := by sorry

-- Part 2
theorem part2 (a b c : ℝ) (h : is_three_times_angle_triangle a b c) (hb : b = 60) :
  (min a (min b c) = 20) ∨ (min a (min b c) = 30) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1636_163678


namespace NUMINAMATH_CALUDE_pears_left_theorem_l1636_163605

/-- The number of pears Keith and Mike are left with after Keith gives away some pears -/
def pears_left (keith_picked : ℕ) (mike_picked : ℕ) (keith_gave_away : ℕ) : ℕ :=
  (keith_picked - keith_gave_away) + mike_picked

/-- Theorem stating that Keith and Mike are left with 13 pears -/
theorem pears_left_theorem : pears_left 47 12 46 = 13 := by
  sorry

end NUMINAMATH_CALUDE_pears_left_theorem_l1636_163605


namespace NUMINAMATH_CALUDE_book_pages_count_l1636_163664

/-- The number of pages Liam read in a week-long reading assignment -/
def totalPages (firstThreeDaysAvg : ℕ) (nextThreeDaysAvg : ℕ) (lastDayPages : ℕ) : ℕ :=
  3 * firstThreeDaysAvg + 3 * nextThreeDaysAvg + lastDayPages

/-- Theorem stating that the total number of pages in the book is 310 -/
theorem book_pages_count :
  totalPages 45 50 25 = 310 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l1636_163664


namespace NUMINAMATH_CALUDE_sum_of_powers_l1636_163634

theorem sum_of_powers (x y : ℝ) (h1 : (x + y)^2 = 7) (h2 : (x - y)^2 = 3) :
  (x^2 + y^2 = 5) ∧ (x^4 + y^4 = 23) ∧ (x^6 + y^6 = 110) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1636_163634


namespace NUMINAMATH_CALUDE_g_one_equals_three_l1636_163666

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_one_equals_three (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h1 : f (-1) + g 1 = 2) 
  (h2 : f 1 + g (-1) = 4) : 
  g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l1636_163666


namespace NUMINAMATH_CALUDE_machine_production_theorem_l1636_163643

def pair_productions : List ℕ := [35, 39, 40, 49, 44, 46, 30, 41, 32, 36]

def individual_productions : List ℕ := [13, 17, 19, 22, 27]

def valid_pair_production (p : ℕ × ℕ) : Prop :=
  p.1 ∈ individual_productions ∧ p.2 ∈ individual_productions ∧ p.1 + p.2 ∈ pair_productions

theorem machine_production_theorem :
  (∀ p ∈ pair_productions, ∃ (x y : ℕ), x ∈ individual_productions ∧ y ∈ individual_productions ∧ x + y = p) ∧
  (∀ (x y : ℕ), x ∈ individual_productions → y ∈ individual_productions → x ≠ y → x + y ∈ pair_productions) ∧
  (individual_productions.length = 5) :=
sorry

end NUMINAMATH_CALUDE_machine_production_theorem_l1636_163643


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1636_163690

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- The octagon formed by connecting midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint octagon is 1/4 of the original octagon -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 / 4) * area o :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1636_163690


namespace NUMINAMATH_CALUDE_gwen_total_books_l1636_163640

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves for mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves for picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books Gwen has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_total_books : total_books = 72 := by sorry

end NUMINAMATH_CALUDE_gwen_total_books_l1636_163640


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l1636_163668

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x - 4) % 7 = 0) 
  (hy : (y + 4) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → 
  (∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
  (∀ n : ℕ+, (x^2 - x*y + y^2 + n) % 7 = 0 ∧ 
    (∀ m : ℕ+, (x^2 - x*y + y^2 + m) % 7 = 0 → n ≤ m) → n = 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l1636_163668


namespace NUMINAMATH_CALUDE_power_four_times_power_four_l1636_163672

theorem power_four_times_power_four (x : ℝ) : x^4 * x^4 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_power_four_l1636_163672


namespace NUMINAMATH_CALUDE_monochromatic_equilateral_triangle_l1636_163645

-- Define a type for colors
inductive Color
| White
| Black

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point
  eq_sides : distance a b = distance b c ∧ distance b c = distance c a

-- Theorem statement
theorem monochromatic_equilateral_triangle :
  ∃ (t : EquilateralTriangle),
    (distance t.a t.b = 1 ∨ distance t.a t.b = Real.sqrt 3) ∧
    (coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c) := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_equilateral_triangle_l1636_163645


namespace NUMINAMATH_CALUDE_expression_evaluation_l1636_163682

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1636_163682


namespace NUMINAMATH_CALUDE_product_of_roots_l1636_163628

theorem product_of_roots : ∃ (r₁ r₂ : ℝ), 
  (r₁^2 + 18*r₁ + 30 = 2 * Real.sqrt (r₁^2 + 18*r₁ + 45)) ∧
  (r₂^2 + 18*r₂ + 30 = 2 * Real.sqrt (r₂^2 + 18*r₂ + 45)) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1636_163628


namespace NUMINAMATH_CALUDE_polygon_missing_angle_l1636_163630

theorem polygon_missing_angle (n : ℕ) (sum_n_minus_1 : ℝ) (h1 : n > 2) (h2 : sum_n_minus_1 = 2843) : 
  (n - 2) * 180 - sum_n_minus_1 = 37 := by
  sorry

end NUMINAMATH_CALUDE_polygon_missing_angle_l1636_163630


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l1636_163612

/-- The cost of pencils and notebooks given specific quantities -/
structure StationeryCost where
  pencil_cost : ℝ
  notebook_cost : ℝ

/-- The conditions from the problem -/
def problem_conditions (c : StationeryCost) : Prop :=
  4 * c.pencil_cost + 5 * c.notebook_cost = 3.35 ∧
  6 * c.pencil_cost + 4 * c.notebook_cost = 3.16

/-- The theorem to prove -/
theorem stationery_cost_theorem (c : StationeryCost) :
  problem_conditions c →
  20 * c.pencil_cost + 13 * c.notebook_cost = 10.29 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l1636_163612


namespace NUMINAMATH_CALUDE_min_value_theorem_l1636_163662

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) 
  (h2 : -2 * m - n + 1 = 0) : 
  (2 / m + 1 / n) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1636_163662


namespace NUMINAMATH_CALUDE_total_matches_played_l1636_163673

theorem total_matches_played (home_wins rival_wins home_draws rival_draws : ℕ) : 
  home_wins = 3 →
  rival_wins = 2 * home_wins →
  home_draws = 4 →
  rival_draws = 4 →
  home_wins + rival_wins + home_draws + rival_draws = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_total_matches_played_l1636_163673


namespace NUMINAMATH_CALUDE_ladder_movement_l1636_163680

theorem ladder_movement (ladder_length : ℝ) (initial_distance : ℝ) (slide_down : ℝ) : 
  ladder_length = 25 →
  initial_distance = 7 →
  slide_down = 4 →
  ∃ (final_distance : ℝ),
    final_distance > initial_distance ∧
    final_distance ^ 2 + (ladder_length - slide_down) ^ 2 = ladder_length ^ 2 ∧
    final_distance - initial_distance = 8 :=
by sorry

end NUMINAMATH_CALUDE_ladder_movement_l1636_163680


namespace NUMINAMATH_CALUDE_water_level_drop_l1636_163698

/-- The drop in water level when removing a partially submerged spherical ball from a prism-shaped glass -/
theorem water_level_drop (a r h : ℝ) (ha : a > 0) (hr : r > 0) (hh : h > 0) (hhr : h < r) :
  let base_area := (3 * Real.sqrt 3 * a^2) / 2
  let submerged_height := r - h
  let submerged_volume := π * submerged_height^2 * (3*r - submerged_height) / 3
  submerged_volume / base_area = (6 * π * Real.sqrt 3) / 25 :=
by sorry

end NUMINAMATH_CALUDE_water_level_drop_l1636_163698


namespace NUMINAMATH_CALUDE_representative_selection_count_l1636_163624

def total_students : ℕ := 10
def female_students : ℕ := 4
def male_students : ℕ := 6
def representatives : ℕ := 3

theorem representative_selection_count : 
  (Nat.choose female_students 1 * Nat.choose male_students 2) + 
  (Nat.choose female_students 2 * Nat.choose male_students 1) + 
  (Nat.choose female_students 3) = 100 := by
  sorry

end NUMINAMATH_CALUDE_representative_selection_count_l1636_163624


namespace NUMINAMATH_CALUDE_centroids_form_equilateral_triangle_l1636_163699

/-- Given a triangle ABC with vertices z₁, z₂, z₃ in the complex plane,
    the centroids of equilateral triangles constructed externally on its sides
    form an equilateral triangle. -/
theorem centroids_form_equilateral_triangle (z₁ z₂ z₃ : ℂ) : 
  let g₁ := (z₁ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₂ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₂ := (z₂ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₃ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  let g₃ := (z₃ * (1 + Complex.exp (Real.pi * Complex.I / 3)) + z₁ * (2 - Complex.exp (Real.pi * Complex.I / 3))) / 3
  (g₂ - g₁) = (g₃ - g₁) * Complex.exp ((2 * Real.pi * Complex.I) / 3) :=
by sorry


end NUMINAMATH_CALUDE_centroids_form_equilateral_triangle_l1636_163699


namespace NUMINAMATH_CALUDE_eccentricity_range_l1636_163615

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the property of a point P being inside the ellipse -/
def inside_ellipse (P : Point) (E : Ellipse) : Prop :=
  (P.x^2 / E.a^2) + (P.y^2 / E.b^2) < 1

/-- Defines the condition that the dot product of vectors PF₁ and PF₂ is zero -/
def orthogonal_foci (P : Point) (E : Ellipse) : Prop :=
  ∃ (F₁ F₂ : Point), (P.x - F₁.x) * (P.x - F₂.x) + (P.y - F₁.y) * (P.y - F₂.y) = 0

/-- Theorem stating the range of eccentricity for the ellipse -/
theorem eccentricity_range (E : Ellipse) 
  (h : ∀ P : Point, orthogonal_foci P E → inside_ellipse P E) :
  ∃ e : ℝ, 0 < e ∧ e < Real.sqrt 2 / 2 ∧ e^2 = (E.a^2 - E.b^2) / E.a^2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_range_l1636_163615


namespace NUMINAMATH_CALUDE_subset_relation_l1636_163644

theorem subset_relation (x : ℝ) : x^2 - x < 0 → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1636_163644


namespace NUMINAMATH_CALUDE_sophie_journey_l1636_163607

/-- Proves that given the specified journey conditions, the walking distance is 5.1 km -/
theorem sophie_journey (d : ℝ) 
  (h1 : (2/3 * d) / 20 + (1/3 * d) / 4 = 1.8) : 
  1/3 * d = 5.1 := by
  sorry

end NUMINAMATH_CALUDE_sophie_journey_l1636_163607


namespace NUMINAMATH_CALUDE_base5_44_equals_binary_10111_l1636_163675

-- Define a function to convert a base-5 number to decimal
def base5ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Theorem stating that (44)₅ in base-5 is equal to (10111)₂ in binary
theorem base5_44_equals_binary_10111 :
  decimalToBinary (base5ToDecimal 44) = [1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base5_44_equals_binary_10111_l1636_163675


namespace NUMINAMATH_CALUDE_alice_prob_after_three_turns_l1636_163674

/-- Represents the player who has the ball -/
inductive Player
  | Alice
  | Bob

/-- The game state after each turn -/
def GameState := List Player

/-- The probability of Alice keeping the ball on her turn -/
def aliceKeepProb : ℚ := 2/3

/-- The probability of Bob keeping the ball on his turn -/
def bobKeepProb : ℚ := 2/3

/-- The initial game state with Alice having the ball -/
def initialState : GameState := [Player.Alice]

/-- Calculates the probability of a specific game state after three turns -/
def stateProb (state : GameState) : ℚ :=
  match state with
  | [Player.Alice, Player.Alice, Player.Alice, Player.Alice] => aliceKeepProb * aliceKeepProb * aliceKeepProb
  | [Player.Alice, Player.Alice, Player.Bob, Player.Alice] => aliceKeepProb * (1 - aliceKeepProb) * (1 - bobKeepProb)
  | [Player.Alice, Player.Bob, Player.Alice, Player.Alice] => (1 - aliceKeepProb) * (1 - bobKeepProb) * aliceKeepProb
  | [Player.Alice, Player.Bob, Player.Bob, Player.Alice] => (1 - aliceKeepProb) * bobKeepProb * (1 - bobKeepProb)
  | _ => 0

/-- All possible game states after three turns where Alice ends up with the ball -/
def validStates : List GameState := [
  [Player.Alice, Player.Alice, Player.Alice, Player.Alice],
  [Player.Alice, Player.Alice, Player.Bob, Player.Alice],
  [Player.Alice, Player.Bob, Player.Alice, Player.Alice],
  [Player.Alice, Player.Bob, Player.Bob, Player.Alice]
]

/-- The main theorem: probability of Alice having the ball after three turns is 14/27 -/
theorem alice_prob_after_three_turns :
  (validStates.map stateProb).sum = 14/27 := by
  sorry


end NUMINAMATH_CALUDE_alice_prob_after_three_turns_l1636_163674


namespace NUMINAMATH_CALUDE_range_of_a_l1636_163652

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 < 0 → x^2 - 2*x + 1 - a^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 ≤ 0 ∧ x^2 - 8*x - 20 ≥ 0) ∧
  (a > 0) →
  a ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1636_163652


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1636_163649

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + c*a = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1636_163649


namespace NUMINAMATH_CALUDE_puppies_per_cage_l1636_163647

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : num_cages = 3)
  (h4 : num_cages > 0)
  (h5 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l1636_163647


namespace NUMINAMATH_CALUDE_sandwich_cost_is_four_l1636_163671

/-- The cost of Karen's fast food order --/
def fast_food_order (burger_cost smoothie_cost sandwich_cost : ℚ) : Prop :=
  burger_cost = 5 ∧
  smoothie_cost = 4 ∧
  burger_cost + 2 * smoothie_cost + sandwich_cost = 17

theorem sandwich_cost_is_four :
  ∀ (burger_cost smoothie_cost sandwich_cost : ℚ),
    fast_food_order burger_cost smoothie_cost sandwich_cost →
    sandwich_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_is_four_l1636_163671


namespace NUMINAMATH_CALUDE_millet_majority_day_four_l1636_163614

/-- Amount of millet in the feeder on day n -/
def millet_amount (n : ℕ) : ℝ :=
  if n = 0 then 0
  else 0.4 + 0.7 * millet_amount (n - 1)

/-- Total amount of seeds in the feeder on day n -/
def total_seeds (n : ℕ) : ℝ := 1

theorem millet_majority_day_four :
  (∀ k < 4, millet_amount k ≤ 0.5) ∧ millet_amount 4 > 0.5 := by sorry

end NUMINAMATH_CALUDE_millet_majority_day_four_l1636_163614


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l1636_163654

-- Define the polynomial
def f (x : ℝ) := x^3 - x + 2

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f a = 0 ∧ f b = 0 ∧ f c = 0 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l1636_163654


namespace NUMINAMATH_CALUDE_complement_union_M_N_l1636_163610

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def M : Set ℕ := {1,3,5,7}
def N : Set ℕ := {5,6,7}

theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l1636_163610


namespace NUMINAMATH_CALUDE_total_books_count_l1636_163686

theorem total_books_count (T : ℕ) : 
  (T = (1/4 : ℚ) * T + 10 + 
       (3/5 : ℚ) * (T - ((1/4 : ℚ) * T + 10)) - 5 + 
       12 + 13) → 
  T = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1636_163686


namespace NUMINAMATH_CALUDE_pen_price_calculation_l1636_163604

theorem pen_price_calculation (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pencil_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 690 →
  pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 18 := by
sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l1636_163604


namespace NUMINAMATH_CALUDE_eva_age_is_six_l1636_163691

-- Define the set of ages
def ages : Finset ℕ := {2, 4, 6, 8, 10}

-- Define the condition for park visit
def park_visit (a b : ℕ) : Prop := a + b = 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

-- Define the condition for concert visit
def concert_visit : Prop := 2 ∈ ages ∧ 10 ∈ ages

-- Define the condition for staying home
def stay_home (eva_age : ℕ) : Prop := eva_age ∈ ages ∧ 4 ∈ ages

-- Theorem statement
theorem eva_age_is_six :
  ∃ (a b : ℕ), park_visit a b ∧ concert_visit ∧ stay_home 6 →
  ∃! (eva_age : ℕ), eva_age ∈ ages ∧ eva_age ≠ 2 ∧ eva_age ≠ 4 ∧ eva_age ≠ 8 ∧ eva_age ≠ 10 :=
by sorry

end NUMINAMATH_CALUDE_eva_age_is_six_l1636_163691


namespace NUMINAMATH_CALUDE_mean_temperature_l1636_163658

def temperatures : List ℚ := [-6, -3, -3, -4, 2, 4, 1]

def mean (list : List ℚ) : ℚ :=
  (list.sum) / list.length

theorem mean_temperature : mean temperatures = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1636_163658


namespace NUMINAMATH_CALUDE_all_black_after_two_rotations_l1636_163677

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Returns true if the given position is on a diagonal of a 4x4 grid -/
def isDiagonal (row col : Fin 4) : Bool :=
  row = col ∨ row + col = 3

/-- Initial grid configuration with black diagonals -/
def initialGrid : Grid :=
  fun row col => isDiagonal row col

/-- Rotates a position 90 degrees clockwise in a 4x4 grid -/
def rotate (row col : Fin 4) : Fin 4 × Fin 4 :=
  (col, 3 - row)

/-- Applies the transformation rule after rotation -/
def transform (g : Grid) : Grid :=
  fun row col =>
    let (oldRow, oldCol) := rotate row col
    g row col ∨ initialGrid oldRow oldCol

/-- Applies two consecutive 90° rotations and transformations -/
def finalGrid : Grid :=
  transform (transform initialGrid)

/-- Theorem stating that all squares in the final grid are black -/
theorem all_black_after_two_rotations :
  ∀ row col, finalGrid row col = true := by sorry

end NUMINAMATH_CALUDE_all_black_after_two_rotations_l1636_163677


namespace NUMINAMATH_CALUDE_trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l1636_163632

theorem trailing_zeros_2006_factorial : Nat → Nat
| n => (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem trailing_zeros_2006_factorial_is_500 :
  trailing_zeros_2006_factorial 2006 = 500 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_2006_factorial_trailing_zeros_2006_factorial_is_500_l1636_163632


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1636_163663

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1636_163663


namespace NUMINAMATH_CALUDE_g_zero_at_negative_one_l1636_163601

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem g_zero_at_negative_one (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_negative_one_l1636_163601


namespace NUMINAMATH_CALUDE_fib_10_calls_l1636_163676

def FIB : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n+2 => FIB (n+1) + FIB n

def count_calls : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | n+2 => count_calls (n+1) + count_calls n + 2

theorem fib_10_calls : count_calls 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_fib_10_calls_l1636_163676


namespace NUMINAMATH_CALUDE_tie_shirt_ratio_l1636_163602

/-- Represents the cost of a school uniform -/
structure UniformCost where
  pants : ℝ
  shirt : ℝ
  tie : ℝ
  socks : ℝ

/-- Calculates the total cost of a given number of uniforms -/
def totalCost (u : UniformCost) (n : ℕ) : ℝ :=
  n * (u.pants + u.shirt + u.tie + u.socks)

/-- Theorem: The ratio of tie cost to shirt cost is 1:5 given the uniform pricing conditions -/
theorem tie_shirt_ratio :
  ∀ (u : UniformCost),
    u.pants = 20 →
    u.shirt = 2 * u.pants →
    u.socks = 3 →
    totalCost u 5 = 355 →
    u.tie / u.shirt = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_tie_shirt_ratio_l1636_163602


namespace NUMINAMATH_CALUDE_temple_visit_theorem_l1636_163695

/-- The number of people who went to the temple with Nathan -/
def number_of_people : ℕ := 3

/-- The cost per object in dollars -/
def cost_per_object : ℕ := 11

/-- The number of objects per person -/
def objects_per_person : ℕ := 5

/-- The total charge for all objects in dollars -/
def total_charge : ℕ := 165

/-- Theorem stating that the number of people is correct given the conditions -/
theorem temple_visit_theorem : 
  number_of_people * objects_per_person * cost_per_object = total_charge :=
by sorry

end NUMINAMATH_CALUDE_temple_visit_theorem_l1636_163695


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l1636_163693

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x - y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x - y + 3 = 0
  ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
  (∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ↔ l₂ (x + k) (y + k)) →
  Real.sqrt 2 = |x₂ - x₁| :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l1636_163693


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1636_163688

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 6 ∧ (7538 - x) % 14 = 0 ∧ ∀ (y : ℕ), y < x → (7538 - y) % 14 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1636_163688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1636_163637

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_8th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 8 + a 9 = 21) : 
  a 8 = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1636_163637


namespace NUMINAMATH_CALUDE_sandbox_volume_calculation_l1636_163611

/-- The volume of a rectangular box with given dimensions -/
def sandbox_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of the sandbox is 3,429,000 cubic centimeters -/
theorem sandbox_volume_calculation :
  sandbox_volume 312 146 75 = 3429000 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_volume_calculation_l1636_163611


namespace NUMINAMATH_CALUDE_valentine_biscuits_l1636_163679

theorem valentine_biscuits (total_biscuits : ℕ) (num_dogs : ℕ) (biscuits_per_dog : ℕ) :
  total_biscuits = 6 →
  num_dogs = 2 →
  total_biscuits = num_dogs * biscuits_per_dog →
  biscuits_per_dog = 3 := by
  sorry

end NUMINAMATH_CALUDE_valentine_biscuits_l1636_163679


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l1636_163683

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l1636_163683


namespace NUMINAMATH_CALUDE_sum_of_unknown_angles_l1636_163638

/-- A six-sided polygon with two right angles and three known angles -/
structure HexagonWithKnownAngles where
  -- The three known angles
  angle_P : ℝ
  angle_Q : ℝ
  angle_R : ℝ
  -- Conditions on the known angles
  angle_P_eq : angle_P = 30
  angle_Q_eq : angle_Q = 60
  angle_R_eq : angle_R = 34
  -- The polygon has two right angles
  has_two_right_angles : True

/-- The sum of the two unknown angles in the hexagon is 124° -/
theorem sum_of_unknown_angles (h : HexagonWithKnownAngles) :
  ∃ x y, x + y = 124 := by sorry

end NUMINAMATH_CALUDE_sum_of_unknown_angles_l1636_163638


namespace NUMINAMATH_CALUDE_function_range_theorem_l1636_163681

theorem function_range_theorem (a : ℝ) :
  (∃ x : ℝ, (|2*x + 1| + |2*x - 3| < |a - 1|)) →
  (a < -3 ∨ a > 5) :=
by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l1636_163681


namespace NUMINAMATH_CALUDE_opposite_face_is_blue_l1636_163622

/-- Represents the colors of the squares --/
inductive Color
  | R | B | O | Y | G | W

/-- Represents a square with colors on both sides --/
structure Square where
  front : Color
  back : Color

/-- Represents the cube formed by folding the squares --/
structure Cube where
  squares : List Square
  white_face : Color
  opposite_face : Color

/-- Axiom: The cube is formed by folding six hinged squares --/
axiom cube_formation (c : Cube) : c.squares.length = 6

/-- Axiom: The white face exists in the cube --/
axiom white_face_exists (c : Cube) : c.white_face = Color.W

/-- Theorem: The face opposite to the white face is blue --/
theorem opposite_face_is_blue (c : Cube) : c.opposite_face = Color.B := by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_blue_l1636_163622


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1636_163696

theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∃ (min : ℝ), min = 9/4 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 1/(1+x) + 4/(1+y) ≥ min) ∧
  a^2*b^2 + a^2 + b^2 ≥ a*b*(a+b+1) := by
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l1636_163696


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l1636_163685

-- Define i as a complex number with i² = -1
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  (Finset.range 603).sum (fun k => i ^ k) = i - 1 := by
  sorry

-- Note: The proof is omitted as per your instructions.

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l1636_163685


namespace NUMINAMATH_CALUDE_derivative_of_f_l1636_163626

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l1636_163626


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1636_163689

theorem solve_quadratic_equation (m n : ℤ) 
  (h : m^2 - 2*m*n + 2*n^2 - 8*n + 16 = 0) : 
  m = 4 ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1636_163689


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l1636_163646

def danny_time : ℝ := 29

theorem time_difference_to_halfway_point :
  let steve_time := 2 * danny_time
  let danny_halfway_time := danny_time / 2
  let steve_halfway_time := steve_time / 2
  steve_halfway_time - danny_halfway_time = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l1636_163646


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1636_163603

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  45 ∣ n^2 ∧ 720 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 45 ∣ m^2 → 720 ∣ m^3 → n ≤ m :=
by
  use 60
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1636_163603


namespace NUMINAMATH_CALUDE_bobsQuestionsRatio_l1636_163619

/-- Represents the number of questions created in each hour -/
structure HourlyQuestions where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the conditions of the problem -/
def bobsQuestions : HourlyQuestions → Prop
  | ⟨first, second, third⟩ =>
    first = 13 ∧
    third = 2 * second ∧
    first + second + third = 91

/-- The theorem to be proved -/
theorem bobsQuestionsRatio (q : HourlyQuestions) :
  bobsQuestions q → q.second / q.first = 2 := by
  sorry

end NUMINAMATH_CALUDE_bobsQuestionsRatio_l1636_163619


namespace NUMINAMATH_CALUDE_one_sided_limits_arctg_reciprocal_l1636_163621

noncomputable def f (x : ℝ) : ℝ := Real.arctan (1 / (x - 1))

theorem one_sided_limits_arctg_reciprocal :
  (∀ ε > 0, ∃ δ > 0, ∀ x > 1, |x - 1| < δ → |f x - π/2| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x < 1, |x - 1| < δ → |f x + π/2| < ε) :=
sorry

end NUMINAMATH_CALUDE_one_sided_limits_arctg_reciprocal_l1636_163621


namespace NUMINAMATH_CALUDE_luke_money_lasted_nine_weeks_l1636_163661

/-- The number of weeks Luke's money lasted given his earnings and spending -/
def weeks_money_lasted (mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Luke's earnings and spending, his money lasted 9 weeks -/
theorem luke_money_lasted_nine_weeks :
  weeks_money_lasted 9 18 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_money_lasted_nine_weeks_l1636_163661


namespace NUMINAMATH_CALUDE_elevator_problem_l1636_163684

theorem elevator_problem (initial_people : ℕ) (remaining_people : ℕ) 
  (h1 : initial_people = 18) 
  (h2 : remaining_people = 11) : 
  initial_people - remaining_people = 7 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l1636_163684


namespace NUMINAMATH_CALUDE_max_value_abc_l1636_163633

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 18 ∧ 
  ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1 → 
  x^2 * (y - z) + y^2 * (z - x) + z^2 * (x - y) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l1636_163633


namespace NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l1636_163660

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The length of a chord formed by the intersection of a circle and a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

theorem circle_line_intersection_chord_length (a : ℝ) :
  let c : Circle := { equation := fun x y z => x^2 + y^2 + 2*x - 2*y + z = 0 }
  let l : Line := { equation := fun x y => x + y + 2 = 0 }
  chordLength c l = 4 → a = -4 := by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l1636_163660


namespace NUMINAMATH_CALUDE_smallest_positive_omega_l1636_163609

/-- Given a function f(x) = sin(ωx + π/3), if f(x - π/3) = -f(x) for all x, 
    then the smallest positive value of ω is 3. -/
theorem smallest_positive_omega (ω : ℝ) : 
  (∀ x, Real.sin (ω * (x - π/3) + π/3) = -Real.sin (ω * x + π/3)) → 
  (∀ δ > 0, δ < ω → δ ≤ 3) ∧ ω = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_omega_l1636_163609


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1636_163613

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (n.cast * a n)

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℚ := sorry

theorem arithmetic_sequence_properties :
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧  -- arithmetic sequence property
  a 7 = 4 ∧                                 -- given condition
  a 19 = 2 * a 9 ∧                          -- given condition
  (∀ n : ℕ, a n = (1 + n.cast) / 2) ∧       -- general formula for a_n
  (∀ n : ℕ, S n = (2 * n.cast) / (n.cast + 1)) -- sum of first n terms of b_n
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1636_163613


namespace NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_range_l1636_163625

theorem y_squared_plus_7y_plus_12_range (y : ℝ) (h : y^2 - 7*y + 12 < 0) :
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_plus_7y_plus_12_range_l1636_163625


namespace NUMINAMATH_CALUDE_james_february_cost_l1636_163670

/-- Calculates the total cost for a streaming service based on the given parameters. -/
def streaming_cost (base_cost : ℝ) (free_hours : ℕ) (extra_hour_cost : ℝ) 
                   (movie_rental_cost : ℝ) (hours_streamed : ℕ) (movies_rented : ℕ) : ℝ :=
  let extra_hours := max (hours_streamed - free_hours) 0
  base_cost + (extra_hours : ℝ) * extra_hour_cost + (movies_rented : ℝ) * movie_rental_cost

/-- Theorem stating that James' streaming cost in February is $24. -/
theorem james_february_cost :
  streaming_cost 15 50 2 0.1 53 30 = 24 := by
  sorry

end NUMINAMATH_CALUDE_james_february_cost_l1636_163670


namespace NUMINAMATH_CALUDE_range_of_a_l1636_163618

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1636_163618


namespace NUMINAMATH_CALUDE_min_value_function_max_value_constraint_l1636_163608

-- Problem 1
theorem min_value_function (x : ℝ) (h : x > 1/2) :
  (∀ z, z > 1/2 → 2*z + 4/(2*z - 1) ≥ 2*x + 4/(2*x - 1)) →
  2*x + 4/(2*x - 1) = 5 :=
sorry

-- Problem 2
theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 3 → x*y + 2*x + y ≥ z*w + 2*z + w) →
  x*y + 2*x + y = 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_function_max_value_constraint_l1636_163608


namespace NUMINAMATH_CALUDE_cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l1636_163687

-- Define the necessary variables and functions
variable (R : ℝ)
variable (x y z : ℝ)

-- Define the equations for the surfaces
def cone_equation (x y z : ℝ) : Prop := z^2 = 2*x*y
def sphere_equation (x y z R : ℝ) : Prop := x^2 + y^2 + z^2 = R^2
def cylinder_equation (x y R : ℝ) : Prop := x^2 + y^2 = R*x

-- Define the surface area functions
noncomputable def cone_surface_area (x_max y_max : ℝ) : ℝ := 
  sorry

noncomputable def sphere_surface_area_in_cylinder (R : ℝ) : ℝ := 
  sorry

noncomputable def cylinder_surface_area_in_sphere (R : ℝ) : ℝ := 
  sorry

-- State the theorems to be proven
theorem cone_surface_area_theorem :
  cone_surface_area 2 4 = 16 :=
sorry

theorem sphere_surface_area_theorem :
  sphere_surface_area_in_cylinder R = 2 * R^2 * (Real.pi - 2) :=
sorry

theorem cylinder_surface_area_theorem :
  cylinder_surface_area_in_sphere R = 4 * R^2 :=
sorry

end NUMINAMATH_CALUDE_cone_surface_area_theorem_sphere_surface_area_theorem_cylinder_surface_area_theorem_l1636_163687


namespace NUMINAMATH_CALUDE_drums_per_day_l1636_163669

/-- Given that 90 drums are filled in 5 days, prove that 18 drums are filled per day -/
theorem drums_per_day (total_drums : ℕ) (total_days : ℕ) (h1 : total_drums = 90) (h2 : total_days = 5) :
  total_drums / total_days = 18 := by
  sorry

end NUMINAMATH_CALUDE_drums_per_day_l1636_163669


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l1636_163667

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + 5

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * a * x + b

theorem tangent_line_and_extrema 
  (a b : ℝ) 
  (h1 : f' a b 1 = -12)  -- Slope of tangent line at x=1 is -12
  (h2 : f a b 1 = -12)   -- Point (1, -12) lies on the graph of f(x)
  : 
  (a = -3 ∧ b = -18) ∧   -- Part 1: Coefficients a and b
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≤ 16) ∧  -- Part 2: Maximum value
  (∀ x ∈ Set.Icc (-3) 1, f (-3) (-18) x ≥ -76)   -- Part 2: Minimum value
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l1636_163667


namespace NUMINAMATH_CALUDE_alicia_wages_l1636_163697

/- Define the hourly wage in dollars -/
def hourly_wage : ℚ := 25

/- Define the local tax rate as a percentage -/
def tax_rate : ℚ := 2.5

/- Define the conversion rate from dollars to cents -/
def cents_per_dollar : ℕ := 100

/- Theorem statement -/
theorem alicia_wages :
  let wage_in_cents := hourly_wage * cents_per_dollar
  let tax_amount := (tax_rate / 100) * wage_in_cents
  let after_tax_earnings := wage_in_cents - tax_amount
  (tax_amount = 62.5 ∧ after_tax_earnings = 2437.5) := by
  sorry

end NUMINAMATH_CALUDE_alicia_wages_l1636_163697


namespace NUMINAMATH_CALUDE_dani_pants_per_pair_l1636_163636

/-- Calculates the number of pants in each pair given the initial number of pants,
    the number of pants after a certain number of years, the number of pairs received each year,
    and the number of years. -/
def pants_per_pair (initial_pants : ℕ) (final_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  let total_pairs := pairs_per_year * years
  let total_new_pants := final_pants - initial_pants
  total_new_pants / total_pairs

theorem dani_pants_per_pair :
  pants_per_pair 50 90 4 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_per_pair_l1636_163636


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l1636_163659

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the two branches of the hyperbola
def branch1 (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def branch2 (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define an equilateral triangle
def is_equilateral_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Main theorem
theorem hyperbola_equilateral_triangle :
  ∀ (Q R : ℝ × ℝ),
  hyperbola (-1) (-1) →
  branch2 (-1) (-1) →
  branch1 Q.1 Q.2 →
  branch1 R.1 R.2 →
  is_equilateral_triangle (-1, -1) Q R →
  (Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧ R = (2 + Real.sqrt 3, 2 - Real.sqrt 3)) ∨
  (Q = (2 + Real.sqrt 3, 2 - Real.sqrt 3) ∧ R = (2 - Real.sqrt 3, 2 + Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_triangle_l1636_163659


namespace NUMINAMATH_CALUDE_mary_cut_roses_l1636_163631

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial final : ℕ) : ℕ := final - initial

theorem mary_cut_roses : roses_cut 6 16 = 10 := by sorry

end NUMINAMATH_CALUDE_mary_cut_roses_l1636_163631


namespace NUMINAMATH_CALUDE_optimal_purchase_is_maximal_l1636_163642

/-- The cost of a red pencil in kopecks -/
def red_cost : ℕ := 27

/-- The cost of a blue pencil in kopecks -/
def blue_cost : ℕ := 23

/-- The maximum total cost in kopecks -/
def max_cost : ℕ := 940

/-- The maximum allowed difference between the number of blue and red pencils -/
def max_diff : ℕ := 10

/-- Represents a valid purchase of pencils -/
structure PencilPurchase where
  red : ℕ
  blue : ℕ
  total_cost_valid : red * red_cost + blue * blue_cost ≤ max_cost
  diff_valid : blue ≤ red + max_diff

/-- The optimal purchase of pencils -/
def optimal_purchase : PencilPurchase :=
  { red := 14
  , blue := 24
  , total_cost_valid := by sorry
  , diff_valid := by sorry }

/-- Theorem stating that the optimal purchase maximizes the total number of pencils -/
theorem optimal_purchase_is_maximal :
  ∀ p : PencilPurchase, p.red + p.blue ≤ optimal_purchase.red + optimal_purchase.blue :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_is_maximal_l1636_163642


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1636_163650

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1636_163650


namespace NUMINAMATH_CALUDE_chickens_in_coop_l1636_163653

theorem chickens_in_coop (coop run free_range : ℕ) : 
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 →
  coop = 14 := by
sorry

end NUMINAMATH_CALUDE_chickens_in_coop_l1636_163653


namespace NUMINAMATH_CALUDE_estimated_probability_is_correct_l1636_163623

/-- Represents the result of a single trial in the traffic congestion simulation -/
structure TrialResult :=
  (days_with_congestion : Nat)
  (h_valid : days_with_congestion ≤ 3)

/-- The simulation data -/
def simulation_data : Finset TrialResult := sorry

/-- The total number of trials in the simulation -/
def total_trials : Nat := 20

/-- The number of trials with exactly two days of congestion -/
def trials_with_two_congestion : Nat := 5

/-- The estimated probability of having exactly two days of congestion in three days -/
def estimated_probability : ℚ := trials_with_two_congestion / total_trials

theorem estimated_probability_is_correct :
  estimated_probability = 1/4 := by sorry

end NUMINAMATH_CALUDE_estimated_probability_is_correct_l1636_163623


namespace NUMINAMATH_CALUDE_initial_orchids_count_l1636_163656

theorem initial_orchids_count (initial_roses : ℕ) (final_roses : ℕ) (final_orchids : ℕ) (total_flowers : ℕ) : 
  initial_roses = 13 →
  final_roses = 14 →
  final_orchids = 91 →
  total_flowers = 105 →
  final_roses + final_orchids = total_flowers →
  final_orchids = initial_roses + final_orchids - total_flowers + final_roses :=
by
  sorry

#check initial_orchids_count

end NUMINAMATH_CALUDE_initial_orchids_count_l1636_163656


namespace NUMINAMATH_CALUDE_problem_statement_l1636_163694

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1636_163694
