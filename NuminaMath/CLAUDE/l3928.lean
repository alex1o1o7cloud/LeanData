import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_division_l3928_392832

/-- Given a square divided into four congruent rectangles, each with its longer side
    parallel to the sides of the square and having a perimeter of 40 inches,
    the perimeter of the square is 64 inches. -/
theorem square_perimeter_from_rectangle_division (s : ℝ) :
  s > 0 →
  (2 * (s + s/4) = 40) →
  (4 * s = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_division_l3928_392832


namespace NUMINAMATH_CALUDE_brick_width_calculation_l3928_392881

/-- Proves that the width of each brick is 11.25 cm given the wall and brick dimensions and the number of bricks needed. -/
theorem brick_width_calculation (brick_length : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_height = 6 →
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  num_bricks = 6000 →
  ∃ (brick_width : ℝ), brick_width = 11.25 ∧ 
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l3928_392881


namespace NUMINAMATH_CALUDE_complex_power_to_rectangular_l3928_392815

theorem complex_power_to_rectangular : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
    Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complex_power_to_rectangular_l3928_392815


namespace NUMINAMATH_CALUDE_sum_product_inequality_l3928_392891

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l3928_392891


namespace NUMINAMATH_CALUDE_perfume_cost_calculation_l3928_392868

/-- The cost of a bottle of perfume given initial savings, earnings from jobs, and additional amount needed --/
def perfume_cost (christian_initial : ℕ) (sue_initial : ℕ) 
                 (yards_mowed : ℕ) (yard_price : ℕ) 
                 (dogs_walked : ℕ) (dog_price : ℕ) 
                 (additional_needed : ℕ) : ℕ :=
  christian_initial + sue_initial + 
  yards_mowed * yard_price + 
  dogs_walked * dog_price + 
  additional_needed

/-- Theorem stating the cost of the perfume given the problem conditions --/
theorem perfume_cost_calculation : 
  perfume_cost 5 7 4 5 6 2 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_perfume_cost_calculation_l3928_392868


namespace NUMINAMATH_CALUDE_candy_bar_profit_l3928_392838

/-- Calculates the profit from selling candy bars -/
def candy_profit (
  num_bars : ℕ
  ) (purchase_price : ℚ)
    (selling_price : ℚ)
    (sales_fee : ℚ) : ℚ :=
  num_bars * selling_price - num_bars * purchase_price - num_bars * sales_fee

/-- Theorem stating the profit from the candy bar sale -/
theorem candy_bar_profit :
  candy_profit 800 (3/4) (2/3) (1/20) = -533/5 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l3928_392838


namespace NUMINAMATH_CALUDE_remainder_sum_l3928_392856

theorem remainder_sum (p q : ℤ) (hp : p % 60 = 47) (hq : q % 45 = 36) : (p + q) % 30 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3928_392856


namespace NUMINAMATH_CALUDE_second_largest_part_l3928_392851

theorem second_largest_part (total : ℚ) (a b c d : ℚ) : 
  total = 120 → 
  a + b + c + d = total →
  a / 3 = b / 2 →
  a / 3 = c / 4 →
  a / 3 = d / 5 →
  (max b (min c d)) = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_part_l3928_392851


namespace NUMINAMATH_CALUDE_product_of_fractions_l3928_392819

theorem product_of_fractions : 
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3928_392819


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l3928_392884

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (fraction_replaced : ℝ) :
  original_intensity = 0.5 →
  added_intensity = 0.2 →
  fraction_replaced = 2/3 →
  let new_intensity := (1 - fraction_replaced) * original_intensity + fraction_replaced * added_intensity
  new_intensity = 0.3 := by
sorry


end NUMINAMATH_CALUDE_paint_mixture_intensity_l3928_392884


namespace NUMINAMATH_CALUDE_polynomial_equality_l3928_392802

theorem polynomial_equality (x : ℝ) (h : 3 * x^3 - x = 1) :
  9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3928_392802


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l3928_392855

/-- The number of corn cobs in a row -/
def cobs_per_row : ℕ := 4

/-- The number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- The number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- The total number of corn cobs grown on the farm -/
def total_cobs : ℕ := rows_field1 * cobs_per_row + rows_field2 * cobs_per_row

theorem farm_corn_cobs : total_cobs = 116 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l3928_392855


namespace NUMINAMATH_CALUDE_triangle_properties_l3928_392820

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2 →
  -- Radius of circumcircle
  2 * Real.sqrt 3 = 2 * a / Real.sin A →
  -- Prove A = π/3
  A = π/3 ∧
  -- Prove maximum area is 9√3
  (1/2) * b * c * Real.sin A ≤ 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3928_392820


namespace NUMINAMATH_CALUDE_thermodynamic_cycle_efficiency_l3928_392888

/-- Represents a thermodynamic cycle with three stages -/
structure ThermodynamicCycle where
  P₀ : ℝ
  ρ₀ : ℝ
  stage1_isochoric : ℝ → ℝ → Prop
  stage2_isobaric : ℝ → ℝ → Prop
  stage3_return : ℝ → ℝ → Prop

/-- Efficiency of a thermodynamic cycle -/
def efficiency (cycle : ThermodynamicCycle) : ℝ := sorry

/-- Maximum possible efficiency for given temperature range -/
def max_efficiency (T_min T_max : ℝ) : ℝ := sorry

/-- Theorem stating the efficiency of the described thermodynamic cycle -/
theorem thermodynamic_cycle_efficiency (cycle : ThermodynamicCycle) 
  (h1 : cycle.stage1_isochoric (3 * cycle.P₀) cycle.P₀)
  (h2 : cycle.stage2_isobaric cycle.ρ₀ (3 * cycle.ρ₀))
  (h3 : cycle.stage3_return 1 1)
  (h4 : ∃ T_min T_max, efficiency cycle = (1 / 8) * max_efficiency T_min T_max) :
  efficiency cycle = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_thermodynamic_cycle_efficiency_l3928_392888


namespace NUMINAMATH_CALUDE_pizza_area_increase_l3928_392828

/-- The radius of the larger pizza in inches -/
def r1 : ℝ := 5

/-- The radius of the smaller pizza in inches -/
def r2 : ℝ := 2

/-- The percentage increase in area from the smaller pizza to the larger pizza -/
def M : ℝ := 525

theorem pizza_area_increase :
  (π * r1^2 - π * r2^2) / (π * r2^2) * 100 = M :=
sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l3928_392828


namespace NUMINAMATH_CALUDE_age_ratio_simplified_l3928_392880

theorem age_ratio_simplified (kul_age saras_age : ℕ) 
  (h1 : kul_age = 22) 
  (h2 : saras_age = 33) : 
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ saras_age * b = kul_age * a :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_simplified_l3928_392880


namespace NUMINAMATH_CALUDE_b_equals_seven_l3928_392875

-- Define the functions f and F
def f (a : ℝ) (x : ℝ) : ℝ := x - a

def F (x y : ℝ) : ℝ := y^2 + x

-- Define b as F(3, f(4))
def b (a : ℝ) : ℝ := F 3 (f a 4)

-- Theorem to prove
theorem b_equals_seven (a : ℝ) : b a = 7 := by
  sorry

end NUMINAMATH_CALUDE_b_equals_seven_l3928_392875


namespace NUMINAMATH_CALUDE_range_of_m_l3928_392883

/-- The piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x else x^2 + 4*x

/-- The property that for all p < m, there exists q ≥ m such that f(p) + f(q) = 0 -/
def property (m : ℝ) : Prop :=
  ∀ p < m, ∃ q ≥ m, f m p + f m q = 0

/-- The theorem stating the range of m -/
theorem range_of_m : ∀ m : ℝ, property m ↔ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l3928_392883


namespace NUMINAMATH_CALUDE_solve_triangle_l3928_392837

noncomputable def triangle_problem (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := (1/2) * a * b * Real.sin C
  (a = 3) ∧
  (Real.cos A = Real.sqrt 6 / 3) ∧
  (B = A + Real.pi / 2) →
  (b = 3 * Real.sqrt 2) ∧
  (S = (3/2) * Real.sqrt 2)

theorem solve_triangle : ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_problem A B C a b c :=
by
  sorry

end NUMINAMATH_CALUDE_solve_triangle_l3928_392837


namespace NUMINAMATH_CALUDE_exp_ge_e_l3928_392804

theorem exp_ge_e (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_exp_ge_e_l3928_392804


namespace NUMINAMATH_CALUDE_madelines_score_l3928_392806

theorem madelines_score (madeline_mistakes : ℕ) (leo_mistakes : ℕ) (brent_score : ℕ) (brent_mistakes : ℕ) 
  (h1 : madeline_mistakes = 2)
  (h2 : madeline_mistakes * 2 = leo_mistakes)
  (h3 : brent_score = 25)
  (h4 : brent_mistakes = leo_mistakes + 1) :
  30 - madeline_mistakes = 28 := by
  sorry

end NUMINAMATH_CALUDE_madelines_score_l3928_392806


namespace NUMINAMATH_CALUDE_drevlandia_roads_l3928_392821

-- Define the number of cities
def num_cities : ℕ := 101

-- Define the function to calculate the number of roads
def num_roads (n : ℕ) : ℕ := n * (n - 1) / 2

-- Theorem statement
theorem drevlandia_roads : num_roads num_cities = 5050 := by
  sorry

end NUMINAMATH_CALUDE_drevlandia_roads_l3928_392821


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l3928_392833

/-- Three points are collinear if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_b_value :
  ∀ b : ℚ, collinear 5 (-3) (-b + 3) 5 (3*b + 1) 4 → b = 18/31 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l3928_392833


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l3928_392889

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l3928_392889


namespace NUMINAMATH_CALUDE_unique_solution_l3928_392886

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x * f y) = y * f x) ∧
  (Filter.Tendsto f Filter.atTop (nhds 0))

/-- The theorem stating that the function f(x) = 1/x is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) : 
  ∀ x, x > 0 → f x = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3928_392886


namespace NUMINAMATH_CALUDE_max_condition_implies_a_range_l3928_392843

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f has a maximum at x = a, then a is in the open interval (-1, 0) -/
theorem max_condition_implies_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_max_condition_implies_a_range_l3928_392843


namespace NUMINAMATH_CALUDE_f_properties_l3928_392841

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  (x₂ * f x₁ < x₁ * f x₂) ∧
  (x₁ > Real.exp (-1) → x₁ * f x₁ + x₂ * f x₂ > x₂ * f x₁ + x₁ * f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3928_392841


namespace NUMINAMATH_CALUDE_journey_fraction_by_rail_l3928_392853

theorem journey_fraction_by_rail 
  (total_journey : ℝ) 
  (bus_fraction : ℝ) 
  (foot_distance : ℝ) : 
  total_journey = 130 ∧ 
  bus_fraction = 17/20 ∧ 
  foot_distance = 6.5 → 
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1/10 := by
sorry

end NUMINAMATH_CALUDE_journey_fraction_by_rail_l3928_392853


namespace NUMINAMATH_CALUDE_fuel_station_cost_fuel_station_cost_example_l3928_392898

/-- Calculates the total cost of filling up vehicles at a fuel station -/
theorem fuel_station_cost (service_cost : ℝ) (fuel_cost : ℝ) (minivan_count : ℕ) (truck_count : ℕ) 
  (minivan_tank : ℝ) (truck_tank_increase : ℝ) : ℝ :=
  let truck_tank := minivan_tank * (1 + truck_tank_increase)
  let minivan_fuel_cost := minivan_count * minivan_tank * fuel_cost
  let truck_fuel_cost := truck_count * truck_tank * fuel_cost
  let total_service_cost := (minivan_count + truck_count) * service_cost
  minivan_fuel_cost + truck_fuel_cost + total_service_cost

/-- Proves that the total cost for filling up 3 mini-vans and 2 trucks is $347.20 -/
theorem fuel_station_cost_example : 
  fuel_station_cost 2.10 0.70 3 2 65 1.20 = 347.20 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_cost_fuel_station_cost_example_l3928_392898


namespace NUMINAMATH_CALUDE_eight_lines_theorem_l3928_392890

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Representation of a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- The given set of lines -/
def given_lines : LineSet :=
  { num_lines := 8
  , no_parallel := true
  , no_concurrent := true }

theorem eight_lines_theorem (lines : LineSet) (h1 : lines.num_lines = 8) 
    (h2 : lines.no_parallel) (h3 : lines.no_concurrent) : 
  num_regions lines.num_lines = 37 := by
  sorry

#eval num_regions 8

end NUMINAMATH_CALUDE_eight_lines_theorem_l3928_392890


namespace NUMINAMATH_CALUDE_f_inequality_l3928_392863

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shift : ∀ x, f (x + 2) = f (2 - x)

-- State the theorem
theorem f_inequality : f 3.5 < f 1 ∧ f 1 < f 2.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3928_392863


namespace NUMINAMATH_CALUDE_birds_on_branch_l3928_392867

theorem birds_on_branch (initial_parrots : ℕ) (remaining_parrots : ℕ) (remaining_crows : ℕ) :
  initial_parrots = 7 →
  remaining_parrots = 2 →
  remaining_crows = 1 →
  ∃ (initial_crows : ℕ) (flew_away : ℕ),
    flew_away = initial_parrots - remaining_parrots ∧
    flew_away = initial_crows - remaining_crows ∧
    initial_parrots + initial_crows = 13 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_branch_l3928_392867


namespace NUMINAMATH_CALUDE_johns_final_push_l3928_392847

/-- John's final push in a speed walking race -/
theorem johns_final_push (john_pace : ℝ) : 
  john_pace * 34 = 3.7 * 34 + (15 + 2) → john_pace = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_final_push_l3928_392847


namespace NUMINAMATH_CALUDE_frisbee_sales_theorem_l3928_392854

/-- Represents the total number of frisbees sold -/
def total_frisbees : ℕ := 60

/-- Represents the number of $3 frisbees sold -/
def frisbees_3 : ℕ := 36

/-- Represents the number of $4 frisbees sold -/
def frisbees_4 : ℕ := 24

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 204

/-- Theorem stating that the total number of frisbees sold is 60 -/
theorem frisbee_sales_theorem :
  (frisbees_3 * 3 + frisbees_4 * 4 = total_receipts) ∧
  (frisbees_4 ≥ 24) ∧
  (total_frisbees = frisbees_3 + frisbees_4) :=
by sorry

end NUMINAMATH_CALUDE_frisbee_sales_theorem_l3928_392854


namespace NUMINAMATH_CALUDE_banana_production_l3928_392887

theorem banana_production (x : ℕ) : 
  x + 10 * x = 99000 → x = 9000 := by
  sorry

end NUMINAMATH_CALUDE_banana_production_l3928_392887


namespace NUMINAMATH_CALUDE_intersection_point_product_l3928_392811

noncomputable section

-- Define the curves in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.cos θ
def C₂ (θ : Real) : Real := 3 / (Real.cos θ + Real.sin θ)

-- Define the condition for α
def valid_α (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- State the theorem
theorem intersection_point_product (α : Real) 
  (h₁ : valid_α α) 
  (h₂ : C₁ α * C₂ α = 3) : 
  α = Real.pi / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_point_product_l3928_392811


namespace NUMINAMATH_CALUDE_square_of_threes_and_four_exist_three_digits_for_infinite_squares_l3928_392813

/-- Represents a number with n threes followed by a four -/
def number_with_threes_and_four (n : ℕ) : ℕ :=
  (3 * (10^n - 1) / 9) * 10 + 4

/-- Represents a number with n+1 ones, followed by n fives, and ending with a six -/
def number_with_ones_fives_and_six (n : ℕ) : ℕ :=
  (10^(2*n + 2) - 1) / 9 * 10^n * 5 + 6

/-- Theorem stating that the square of number_with_threes_and_four is equal to number_with_ones_fives_and_six -/
theorem square_of_threes_and_four (n : ℕ) :
  (number_with_threes_and_four n)^2 = number_with_ones_fives_and_six n := by
  sorry

/-- Corollary stating that there exist three non-zero digits that can be used to form
    an infinite number of decimal representations of squares of different integers -/
theorem exist_three_digits_for_infinite_squares :
  ∃ (d₁ d₂ d₃ : ℕ), d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ d₃ ≠ 0 ∧
    ∀ (n : ℕ), ∃ (m : ℕ), m^2 = number_with_ones_fives_and_six n ∧
    (∀ (k : ℕ), k < n → number_with_ones_fives_and_six k ≠ number_with_ones_fives_and_six n) := by
  sorry

end NUMINAMATH_CALUDE_square_of_threes_and_four_exist_three_digits_for_infinite_squares_l3928_392813


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3928_392805

/-- The volume ratio of two right circular cylinders -/
theorem cylinder_volume_ratio :
  let r1 := 4 / Real.pi
  let h1 := 10
  let r2 := 5 / Real.pi
  let h2 := 8
  let v1 := Real.pi * r1^2 * h1
  let v2 := Real.pi * r2^2 * h2
  v1 / v2 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3928_392805


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3928_392814

theorem greatest_perimeter_of_special_triangle :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    b = 2 * a →
    17 + a > b →
    b + 17 > a →
    a + b > 17 →
    a + b + 17 ≤ 65 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3928_392814


namespace NUMINAMATH_CALUDE_simplify_expression_l3928_392896

theorem simplify_expression (w : ℝ) : 4*w + 6*w + 8*w + 10*w + 12*w + 24 = 40*w + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3928_392896


namespace NUMINAMATH_CALUDE_f1_not_unique_l3928_392857

-- Define the type of our functions
def F := ℝ → ℝ

-- Define the recursive relationship
def recursive_relation (f₁ : F) (n : ℕ) : F :=
  match n with
  | 0 => id
  | 1 => f₁
  | n + 2 => f₁ ∘ (recursive_relation f₁ (n + 1))

-- State the theorem
theorem f1_not_unique :
  ∃ (f₁ g₁ : F),
    f₁ ≠ g₁ ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation f₁ n) = f₁ ∘ (recursive_relation f₁ (n - 1))) ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation g₁ n) = g₁ ∘ (recursive_relation g₁ (n - 1))) ∧
    (recursive_relation f₁ 5) 2 = 33 ∧
    (recursive_relation g₁ 5) 2 = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_f1_not_unique_l3928_392857


namespace NUMINAMATH_CALUDE_jaden_final_cars_l3928_392865

/-- The number of toy cars Jaden has after various changes --/
def final_car_count (initial : ℕ) (bought : ℕ) (birthday : ℕ) (sister : ℕ) (friend : ℕ) : ℕ :=
  initial + bought + birthday - sister - friend

/-- Theorem stating that Jaden's final car count is 43 --/
theorem jaden_final_cars : 
  final_car_count 14 28 12 8 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_jaden_final_cars_l3928_392865


namespace NUMINAMATH_CALUDE_unique_magnitude_for_quadratic_roots_l3928_392877

theorem unique_magnitude_for_quadratic_roots (z : ℂ) : 
  z^2 - 8*z + 37 = 0 → ∃! m : ℝ, ∃ w : ℂ, w^2 - 8*w + 37 = 0 ∧ Complex.abs w = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_quadratic_roots_l3928_392877


namespace NUMINAMATH_CALUDE_distance_by_sea_l3928_392836

/-- The distance traveled by sea is the difference between the total distance and the distance by land -/
theorem distance_by_sea (total_distance land_distance : ℕ) (h1 : total_distance = 601) (h2 : land_distance = 451) :
  total_distance - land_distance = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_by_sea_l3928_392836


namespace NUMINAMATH_CALUDE_stating_correct_equation_representation_l3928_392845

/-- Represents the distribution of people in a campus beautification activity -/
def campus_beautification (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) 
  (support_weeding : ℕ) : Prop :=
  let final_weeding := initial_weeding + support_weeding
  let final_planting := initial_planting + (total_support - support_weeding)
  final_weeding = 2 * final_planting

/-- 
Theorem stating that the equation correctly represents the final distribution
of people in the campus beautification activity.
-/
theorem correct_equation_representation 
  (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) (support_weeding : ℕ) :
  campus_beautification initial_weeding initial_planting total_support support_weeding →
  initial_weeding + support_weeding = 2 * (initial_planting + (total_support - support_weeding)) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_correct_equation_representation_l3928_392845


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3928_392899

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 4 * x = y^2) → (∃ (z : ℕ), 5 * x = z^3) → x ≥ n) ∧
  n = 400 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3928_392899


namespace NUMINAMATH_CALUDE_raj_ate_ten_bananas_l3928_392872

/-- The number of bananas Raj ate -/
def bananas_eaten (initial_bananas : ℕ) (remaining_bananas : ℕ) : ℕ :=
  initial_bananas - remaining_bananas - 2 * remaining_bananas

/-- Theorem stating that Raj ate 10 bananas -/
theorem raj_ate_ten_bananas :
  bananas_eaten 310 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_raj_ate_ten_bananas_l3928_392872


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3928_392842

/-- The area of the shaded region in a configuration where a 5x5 square adjoins a 15x15 square,
    with a line drawn from the top left corner of the larger square to the bottom right corner
    of the smaller square, is 175/8 square inches. -/
theorem shaded_area_calculation : 
  let large_square_side : ℝ := 15
  let small_square_side : ℝ := 5
  let total_width : ℝ := large_square_side + small_square_side
  let triangle_base : ℝ := large_square_side * small_square_side / total_width
  let triangle_area : ℝ := 1/2 * triangle_base * small_square_side
  let small_square_area : ℝ := small_square_side ^ 2
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 175/8 := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3928_392842


namespace NUMINAMATH_CALUDE_sum_of_evens_between_1_and_31_l3928_392846

def sumOfEvens : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 ∧ n + 1 < 31 then n + 1 + sumOfEvens n else sumOfEvens n

theorem sum_of_evens_between_1_and_31 : sumOfEvens 30 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_evens_between_1_and_31_l3928_392846


namespace NUMINAMATH_CALUDE_math_club_team_selection_l3928_392850

theorem math_club_team_selection (boys girls : ℕ) (h1 : boys = 7) (h2 : girls = 9) :
  (Finset.sum (Finset.range 3) (λ i =>
    Nat.choose boys (i + 2) * Nat.choose girls (4 - i))) = 6846 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l3928_392850


namespace NUMINAMATH_CALUDE_trisha_money_theorem_l3928_392882

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog's food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha brought at the beginning -/
def total_money : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + money_left

theorem trisha_money_theorem : total_money = 167 := by
  sorry

end NUMINAMATH_CALUDE_trisha_money_theorem_l3928_392882


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_abs_equation_l3928_392823

theorem smallest_x_satisfying_abs_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |5*y + 2| = 28 → x ≤ y) ∧ |5*x + 2| = 28 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_abs_equation_l3928_392823


namespace NUMINAMATH_CALUDE_ratio_adjustment_l3928_392858

theorem ratio_adjustment (x : ℚ) : x = 29 ↔ (4 + x) / (15 + x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_adjustment_l3928_392858


namespace NUMINAMATH_CALUDE_remainder_equality_l3928_392852

theorem remainder_equality (a b c : ℤ) (hc : c ≠ 0) :
  c ∣ (a - b) → a ≡ b [ZMOD c] :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l3928_392852


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3928_392801

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41/20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3928_392801


namespace NUMINAMATH_CALUDE_two_digit_sum_to_four_digit_sum_l3928_392897

/-- Given two two-digit numbers that sum to 137, prove that the sum of the four-digit numbers
    formed by concatenating these digits in order and in reverse order is 13837. -/
theorem two_digit_sum_to_four_digit_sum
  (A B C D : ℕ)
  (h_AB_two_digit : A * 10 + B < 100)
  (h_CD_two_digit : C * 10 + D < 100)
  (h_sum : A * 10 + B + C * 10 + D = 137) :
  (A * 1000 + B * 100 + C * 10 + D) + (C * 1000 + D * 100 + A * 10 + B) = 13837 := by
  sorry


end NUMINAMATH_CALUDE_two_digit_sum_to_four_digit_sum_l3928_392897


namespace NUMINAMATH_CALUDE_first_group_size_l3928_392831

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of men in the second group -/
def second_group_men : ℕ := 25

/-- The number of days it takes the second group to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the first group -/
def first_group_men : ℕ := first_group_days * second_group_men * second_group_days / first_group_days

theorem first_group_size :
  first_group_men = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l3928_392831


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l3928_392859

/-- Two fair six-sided dice are thrown. -/
def dice_space : Type := Fin 6 × Fin 6

/-- Event A: "the number of points on die A is greater than 4" -/
def event_A : Set dice_space :=
  {x | x.1 > 4}

/-- Event B: "the sum of the number of points on dice A and B is equal to 7" -/
def event_B : Set dice_space :=
  {x | x.1.val + x.2.val = 7}

/-- The probability measure on the dice space -/
def P : Set dice_space → ℝ :=
  sorry

/-- Theorem: The conditional probability P(B|A) = 1/6 -/
theorem conditional_probability_B_given_A :
  P event_B / P event_A = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l3928_392859


namespace NUMINAMATH_CALUDE_truck_load_problem_l3928_392826

/-- Proves that the number of crates loaded yesterday is 10 --/
theorem truck_load_problem :
  let truck_capacity : ℕ := 13500
  let box_weight : ℕ := 100
  let box_count : ℕ := 100
  let crate_weight : ℕ := 60
  let sack_weight : ℕ := 50
  let sack_count : ℕ := 50
  let bag_weight : ℕ := 40
  let bag_count : ℕ := 10

  let total_box_weight := box_weight * box_count
  let total_sack_weight := sack_weight * sack_count
  let total_bag_weight := bag_weight * bag_count

  let remaining_weight := truck_capacity - (total_box_weight + total_sack_weight + total_bag_weight)

  ∃ crate_count : ℕ, crate_count * crate_weight = remaining_weight ∧ crate_count = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_truck_load_problem_l3928_392826


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3928_392827

theorem triangle_abc_properties (a b c A B C m : ℝ) :
  0 < A → A ≤ 2 * Real.pi / 3 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  a^2 + b^2 - c^2 = Real.sqrt 3 * a * b →
  m = 2 * (Real.cos (A / 2))^2 - Real.sin B - 1 →
  (C = Real.pi / 6 ∧ -1 ≤ m ∧ m < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3928_392827


namespace NUMINAMATH_CALUDE_power_inequality_l3928_392876

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^6 + b^6 > a^4*b^2 + a^2*b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3928_392876


namespace NUMINAMATH_CALUDE_peach_difference_l3928_392824

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  steven_peaches = jill_peaches + 15 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches - jake_peaches = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3928_392824


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3928_392885

theorem quadratic_expression_value (x : ℝ) : 
  let a : ℝ := 2010 * x + 2010
  let b : ℝ := 2010 * x + 2011
  let c : ℝ := 2010 * x + 2012
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3928_392885


namespace NUMINAMATH_CALUDE_game_probability_result_l3928_392869

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) : ℚ :=
  let chelsea_prob := (1 - alex_prob) / (1 + mel_chelsea_ratio)
  let mel_prob := chelsea_prob * mel_chelsea_ratio
  let specific_sequence_prob := alex_prob^4 * mel_prob^2 * chelsea_prob
  let arrangements := (Nat.factorial total_rounds) / 
                      ((Nat.factorial 4) * (Nat.factorial 2) * (Nat.factorial 1))
  arrangements * specific_sequence_prob

theorem game_probability_result : 
  game_probability 7 (1/2) 2 = 35/288 := by sorry

end NUMINAMATH_CALUDE_game_probability_result_l3928_392869


namespace NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l3928_392848

def triple_number (n : ℕ) : ℕ := 3 * n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_triple_4567 :
  ones_digit (triple_number 4567) = 1 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_triple_4567_l3928_392848


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3928_392812

theorem basketball_free_throws (two_points three_points free_throws : ℕ) : 
  (3 * three_points = 2 * two_points) →  -- Points from three-point shots are twice the points from two-point shots
  (free_throws = 2 * two_points - 1) →   -- Number of free throws is twice the number of two-point shots minus one
  (2 * two_points + 3 * three_points + free_throws = 89) →  -- Total score is 89 points
  free_throws = 29 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3928_392812


namespace NUMINAMATH_CALUDE_power_of_power_l3928_392808

theorem power_of_power (k : ℕ+) : (k^5)^3 = k^15 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l3928_392808


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3928_392893

theorem first_discount_percentage (original_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : original_price = 480) 
  (h2 : final_price = 306) (h3 : second_discount = 25) : 
  ∃ (first_discount : ℝ), 
    first_discount = 15 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3928_392893


namespace NUMINAMATH_CALUDE_unique_solution_l3928_392818

def is_valid_number (α β : ℕ) : Prop :=
  0 ≤ α ∧ α ≤ 9 ∧ 0 ≤ β ∧ β ≤ 9

def number_value (α β : ℕ) : ℕ :=
  62000000 + α * 10000 + β * 1000 + 427

theorem unique_solution (α β : ℕ) :
  is_valid_number α β →
  (number_value α β) % 99 = 0 →
  α = 2 ∧ β = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3928_392818


namespace NUMINAMATH_CALUDE_challenge_probability_challenge_probability_value_l3928_392810

/-- The probability of selecting all letters from "CHALLENGE" when choosing 3 letters from "FARM", 
    4 letters from "BENCHES", and 2 letters from "GLOVE" -/
theorem challenge_probability : ℚ := by
  -- Define the number of letters in each word
  let farm_letters : ℕ := 4
  let benches_letters : ℕ := 7
  let glove_letters : ℕ := 5

  -- Define the number of letters to be selected from each word
  let farm_select : ℕ := 3
  let benches_select : ℕ := 4
  let glove_select : ℕ := 2

  -- Define the number of required letters from each word
  let farm_required : ℕ := 2  -- A and L
  let benches_required : ℕ := 3  -- C, H, and E
  let glove_required : ℕ := 2  -- G and E

  -- Calculate the probability
  sorry

-- The theorem states that the probability is 2/350
theorem challenge_probability_value : challenge_probability = 2 / 350 := by sorry

end NUMINAMATH_CALUDE_challenge_probability_challenge_probability_value_l3928_392810


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3928_392822

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_product : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3928_392822


namespace NUMINAMATH_CALUDE_extended_box_volume_sum_l3928_392803

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volume_extended_box (b : Box) : ℝ := sorry

/-- Checks if two natural numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_box_volume_sum (b : Box) (m n p : ℕ) :
  b.length = 3 ∧ b.width = 4 ∧ b.height = 5 →
  volume_extended_box b = (m : ℝ) + (n : ℝ) * Real.pi / (p : ℝ) →
  m > 0 ∧ n > 0 ∧ p > 0 →
  are_relatively_prime n p →
  m + n + p = 505 := by
  sorry

end NUMINAMATH_CALUDE_extended_box_volume_sum_l3928_392803


namespace NUMINAMATH_CALUDE_wind_on_rainy_day_probability_l3928_392809

/-- Given probabilities in a meteorological context -/
structure WeatherProbabilities where
  rain : ℚ
  wind : ℚ
  both : ℚ

/-- The probability of wind on a rainy day -/
def windOnRainyDay (wp : WeatherProbabilities) : ℚ :=
  wp.both / wp.rain

/-- Theorem stating the probability of wind on a rainy day -/
theorem wind_on_rainy_day_probability (wp : WeatherProbabilities) 
  (h1 : wp.rain = 4/15)
  (h2 : wp.wind = 2/15)
  (h3 : wp.both = 1/10) :
  windOnRainyDay wp = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_wind_on_rainy_day_probability_l3928_392809


namespace NUMINAMATH_CALUDE_apple_distribution_l3928_392816

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 3

/-- The number of apples Kevin has -/
def kevins_apples : ℕ := 2 * jackies_apples

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 8

/-- The total number of apples Adam, Jackie, and Kevin have -/
def total_apples : ℕ := jackies_apples + kevins_apples + adams_apples

/-- The number of apples He has -/
def his_apples : ℕ := 3 * total_apples

theorem apple_distribution :
  total_apples = 20 ∧ his_apples = 60 :=
sorry

end NUMINAMATH_CALUDE_apple_distribution_l3928_392816


namespace NUMINAMATH_CALUDE_intersection_M_N_l3928_392870

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3928_392870


namespace NUMINAMATH_CALUDE_discount_rates_sum_l3928_392879

/-- The discount rate for Fox jeans -/
def fox_discount_rate : ℝ := sorry

/-- The discount rate for Pony jeans -/
def pony_discount_rate : ℝ := 0.1

/-- The regular price of Fox jeans -/
def fox_regular_price : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_regular_price : ℝ := 18

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase -/
def total_savings : ℝ := 9

theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by
  have h1 : fox_quantity * fox_regular_price * fox_discount_rate +
            pony_quantity * pony_regular_price * pony_discount_rate = total_savings :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_discount_rates_sum_l3928_392879


namespace NUMINAMATH_CALUDE_sqrt_b_minus_3_domain_l3928_392807

theorem sqrt_b_minus_3_domain : {b : ℝ | ∃ x : ℝ, x^2 = b - 3} = {b : ℝ | b ≥ 3} := by sorry

end NUMINAMATH_CALUDE_sqrt_b_minus_3_domain_l3928_392807


namespace NUMINAMATH_CALUDE_a_salary_is_5250_l3928_392862

/-- Proof that A's salary is $5250 given the conditions of the problem -/
theorem a_salary_is_5250 (a b : ℝ) : 
  a + b = 7000 →                   -- Total salary is $7000
  0.05 * a = 0.15 * b →            -- Savings are equal
  a = 5250 := by
    sorry

end NUMINAMATH_CALUDE_a_salary_is_5250_l3928_392862


namespace NUMINAMATH_CALUDE_power_product_equals_l3928_392844

theorem power_product_equals : (3 : ℕ)^6 * (4 : ℕ)^6 = 2985984 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_l3928_392844


namespace NUMINAMATH_CALUDE_line_with_equal_intercepts_through_intersection_l3928_392892

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection point of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := sorry

/-- Check if a line has equal intercepts on coordinate axes -/
def equal_intercepts (l : Line) : Prop := sorry

theorem line_with_equal_intercepts_through_intersection 
  (l1 l2 : Line) 
  (h1 : l1 = Line.mk 1 2 (-11)) 
  (h2 : l2 = Line.mk 2 1 (-10)) :
  ∃ (l : Line), 
    on_line (intersection l1 l2) l ∧ 
    equal_intercepts l ∧ 
    (l = Line.mk 4 (-3) 0 ∨ l = Line.mk 1 1 (-7)) := by
  sorry

end NUMINAMATH_CALUDE_line_with_equal_intercepts_through_intersection_l3928_392892


namespace NUMINAMATH_CALUDE_no_arithmetic_sqrt_neg_nine_l3928_392873

-- Define the concept of an arithmetic square root
def arithmetic_sqrt (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x ∧ y ≥ 0

-- Theorem stating that the arithmetic square root of -9 does not exist
theorem no_arithmetic_sqrt_neg_nine :
  ¬ arithmetic_sqrt (-9) :=
sorry

end NUMINAMATH_CALUDE_no_arithmetic_sqrt_neg_nine_l3928_392873


namespace NUMINAMATH_CALUDE_cindy_walking_speed_l3928_392834

/-- Cindy's running speed in miles per hour -/
def running_speed : ℝ := 3

/-- Distance Cindy runs in miles -/
def run_distance : ℝ := 0.5

/-- Distance Cindy walks in miles -/
def walk_distance : ℝ := 0.5

/-- Total time for the journey in minutes -/
def total_time : ℝ := 40

/-- Cindy's walking speed in miles per hour -/
def walking_speed : ℝ := 1

theorem cindy_walking_speed :
  running_speed = 3 ∧
  run_distance = 0.5 ∧
  walk_distance = 0.5 ∧
  total_time = 40 →
  walking_speed = 1 := by sorry

end NUMINAMATH_CALUDE_cindy_walking_speed_l3928_392834


namespace NUMINAMATH_CALUDE_perimeter_of_C_l3928_392835

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 36

-- Define the side length of C as the difference between side lengths of A and B
def side_C (a b : Real) : Real := b - a

-- Theorem statement
theorem perimeter_of_C : 
  ∀ (a b : Real),
  square_A a = perimeter_A →
  square_B b = perimeter_B →
  square_C (side_C a b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_C_l3928_392835


namespace NUMINAMATH_CALUDE_preimage_of_four_l3928_392829

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_l3928_392829


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3928_392860

theorem partial_fraction_decomposition (N₁ N₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (60 * x - 46) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -1036 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3928_392860


namespace NUMINAMATH_CALUDE_adjacent_teacher_performances_probability_l3928_392871

-- Define the number of student performances
def num_student_performances : ℕ := 5

-- Define the number of teacher performances
def num_teacher_performances : ℕ := 2

-- Define the total number of performances
def total_performances : ℕ := num_student_performances + num_teacher_performances

-- Define the function to calculate the probability
def probability_adjacent_teacher_performances : ℚ :=
  (num_student_performances + 1 : ℚ) * 2 / ((total_performances : ℚ) * (total_performances - 1 : ℚ) / 2)

-- Theorem statement
theorem adjacent_teacher_performances_probability :
  probability_adjacent_teacher_performances = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_teacher_performances_probability_l3928_392871


namespace NUMINAMATH_CALUDE_polynomial_root_magnitude_implies_a_range_l3928_392878

/-- A polynomial of degree 4 with real coefficients -/
structure Polynomial4 (a : ℝ) where
  coeff : Fin 5 → ℝ
  coeff_0 : coeff 0 = 2
  coeff_1 : coeff 1 = a
  coeff_2 : coeff 2 = 9
  coeff_3 : coeff 3 = a
  coeff_4 : coeff 4 = 2

/-- The roots of a polynomial -/
def roots (p : Polynomial4 a) : Finset ℂ := sorry

/-- Predicate to check if all roots are complex -/
def allRootsComplex (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, r.im ≠ 0

/-- Predicate to check if all root magnitudes are not equal to 1 -/
def allRootMagnitudesNotOne (p : Polynomial4 a) : Prop :=
  ∀ r ∈ roots p, Complex.abs r ≠ 1

/-- The main theorem -/
theorem polynomial_root_magnitude_implies_a_range (a : ℝ) (p : Polynomial4 a) 
    (h1 : allRootsComplex p) (h2 : allRootMagnitudesNotOne p) : 
    a ∈ Set.Ioo (-2 * Real.sqrt 10) (2 * Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_magnitude_implies_a_range_l3928_392878


namespace NUMINAMATH_CALUDE_knitting_time_for_two_pairs_l3928_392894

/-- Given A's and B's knitting rates, prove the time needed to knit two pairs of socks together -/
theorem knitting_time_for_two_pairs 
  (rate_A : ℚ) -- A's knitting rate in pairs per day
  (rate_B : ℚ) -- B's knitting rate in pairs per day
  (h_rate_A : rate_A = 1/3) -- A can knit a pair in 3 days
  (h_rate_B : rate_B = 1/6) -- B can knit a pair in 6 days
  : (2 : ℚ) / (rate_A + rate_B) = 4 := by
  sorry

end NUMINAMATH_CALUDE_knitting_time_for_two_pairs_l3928_392894


namespace NUMINAMATH_CALUDE_triangle_area_is_six_l3928_392895

-- Define the triangle vertices
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, 3)

-- Define the line on which C lies
def line_C (x y : ℝ) : Prop := x + y = 7

-- Define the area of the triangle
def triangle_area (C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_is_six :
  ∀ C : ℝ × ℝ, line_C C.1 C.2 → triangle_area C = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_six_l3928_392895


namespace NUMINAMATH_CALUDE_smallest_result_l3928_392861

def S : Finset ℕ := {4, 5, 7, 11, 13, 17}

theorem smallest_result (a b c : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x + y) * z = 48 ∧ (a + b) * c ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l3928_392861


namespace NUMINAMATH_CALUDE_divisible_by_ten_l3928_392825

theorem divisible_by_ten (n : ℕ) : ∃ k : ℤ, 3^(n+2) - 2^(n+2) + 3^n - 2^n = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l3928_392825


namespace NUMINAMATH_CALUDE_sum_of_roots_is_six_l3928_392830

-- Define the quadratic polynomials
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem sum_of_roots_is_six 
  (a b c d : ℝ) 
  (hf : ∃ r₁ r₂ : ℝ, ∀ x, f a b x = (x - r₁) * (x - r₂))
  (hg : ∃ s₁ s₂ : ℝ, ∀ x, g c d x = (x - s₁) * (x - s₂))
  (h_eq1 : f a b 1 = g c d 2)
  (h_eq2 : g c d 1 = f a b 2) :
  ∃ r₁ r₂ s₁ s₂ : ℝ, r₁ + r₂ + s₁ + s₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_six_l3928_392830


namespace NUMINAMATH_CALUDE_school_relationship_l3928_392840

/-- In a school with teachers and students, prove the relationship between
    the number of teachers, students, students per teacher, and teachers per student. -/
theorem school_relationship (m n k ℓ : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : k > 0) 
  (h4 : ℓ > 0) 
  (teacher_students : ∀ t, t ≤ m → (∃ s, s ≤ n ∧ s = k))
  (student_teachers : ∀ s, s ≤ n → (∃ t, t ≤ m ∧ t = ℓ)) :
  m * k = n * ℓ := by
  sorry


end NUMINAMATH_CALUDE_school_relationship_l3928_392840


namespace NUMINAMATH_CALUDE_one_pepperoni_fell_off_l3928_392849

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)
  (num_slices : ℕ)
  (pepperoni_on_given_slice : ℕ)

/-- Calculates the number of pepperoni slices that fell off -/
def pepperoni_fell_off (p : Pizza) : ℕ :=
  (p.total_pepperoni / p.num_slices) - p.pepperoni_on_given_slice

/-- Theorem stating that one pepperoni slice fell off -/
theorem one_pepperoni_fell_off (p : Pizza) 
    (h1 : p.total_pepperoni = 40)
    (h2 : p.num_slices = 4)
    (h3 : p.pepperoni_on_given_slice = 9) : 
  pepperoni_fell_off p = 1 := by
  sorry

#eval pepperoni_fell_off { total_pepperoni := 40, num_slices := 4, pepperoni_on_given_slice := 9 }

end NUMINAMATH_CALUDE_one_pepperoni_fell_off_l3928_392849


namespace NUMINAMATH_CALUDE_square_even_implies_even_l3928_392800

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end NUMINAMATH_CALUDE_square_even_implies_even_l3928_392800


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3928_392866

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the "contained in" relation
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (a : Line) (α β : Plane) 
  (h_contained : contained_in a α) :
  (∀ β, perpendicular a β → perpendicular_planes α β) ∧ 
  (∃ β, perpendicular_planes α β ∧ ¬perpendicular a β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3928_392866


namespace NUMINAMATH_CALUDE_area_difference_square_rectangle_l3928_392864

/-- Given a square and a rectangle with the same perimeter, this theorem proves
    the difference between their areas when specific dimensions are provided. -/
theorem area_difference_square_rectangle (square_perimeter : ℝ) (rect_perimeter : ℝ) (rect_length : ℝ)
  (h1 : square_perimeter = 52)
  (h2 : rect_perimeter = 52)
  (h3 : rect_length = 15) :
  (square_perimeter / 4) ^ 2 - rect_length * ((rect_perimeter / 2) - rect_length) = 4 :=
by sorry


end NUMINAMATH_CALUDE_area_difference_square_rectangle_l3928_392864


namespace NUMINAMATH_CALUDE_matthews_crackers_l3928_392874

theorem matthews_crackers (initial_cakes : ℕ) (num_friends : ℕ) (cakes_eaten_per_person : ℕ)
  (h1 : initial_cakes = 30)
  (h2 : num_friends = 2)
  (h3 : cakes_eaten_per_person = 15)
  : initial_cakes = num_friends * cakes_eaten_per_person :=
by
  sorry

#check matthews_crackers

end NUMINAMATH_CALUDE_matthews_crackers_l3928_392874


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3928_392839

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (7 * x^3 + 3 * x^2 - 5 * x - 8) = (x + 2) * (7 * x^2 - 11 * x + 17) + (-42) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3928_392839


namespace NUMINAMATH_CALUDE_child_sold_seven_apples_l3928_392817

/-- Represents the number of apples sold by a child given the initial conditions and final count -/
def apples_sold (num_children : ℕ) (apples_per_child : ℕ) (eating_children : ℕ) (apples_eaten_each : ℕ) (apples_left : ℕ) : ℕ :=
  num_children * apples_per_child - eating_children * apples_eaten_each - apples_left

/-- Theorem stating that given the conditions in the problem, the child sold 7 apples -/
theorem child_sold_seven_apples :
  apples_sold 5 15 2 4 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_child_sold_seven_apples_l3928_392817
