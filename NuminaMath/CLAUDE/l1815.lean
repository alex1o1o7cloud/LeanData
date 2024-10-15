import Mathlib

namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l1815_181503

theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → k = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l1815_181503


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1815_181566

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1815_181566


namespace NUMINAMATH_CALUDE_f_negative_two_l1815_181563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + x^3 + 1

theorem f_negative_two (a : ℝ) (h : f a 2 = 3) : f a (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_two_l1815_181563


namespace NUMINAMATH_CALUDE_problem_solving_probability_l1815_181575

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l1815_181575


namespace NUMINAMATH_CALUDE_rectangle_area_around_square_total_area_of_rectangles_l1815_181535

theorem rectangle_area_around_square (l₁ l₂ : ℝ) : 
  l₁ + l₂ = 11 → 
  2 * (6 * l₁ + 6 * l₂) = 132 := by
  sorry

theorem total_area_of_rectangles : 
  ∀ (l₁ l₂ : ℝ), 
  (4 * (12 + l₁ + l₂) = 92) → 
  (2 * (6 * l₁ + 6 * l₂) = 132) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_around_square_total_area_of_rectangles_l1815_181535


namespace NUMINAMATH_CALUDE_sisters_ages_l1815_181510

theorem sisters_ages (a b c : ℕ+) 
  (middle_age : c = 10)
  (age_relation : a * 10 - 9 * b = 89) :
  a = 17 ∧ b = 9 ∧ c = 10 := by
sorry

end NUMINAMATH_CALUDE_sisters_ages_l1815_181510


namespace NUMINAMATH_CALUDE_min_value_of_x_l1815_181523

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 + (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l1815_181523


namespace NUMINAMATH_CALUDE_total_carpets_l1815_181505

theorem total_carpets (house1 house2 house3 house4 : ℕ) : 
  house1 = 12 → 
  house2 = 20 → 
  house3 = 10 → 
  house4 = 2 * house3 → 
  house1 + house2 + house3 + house4 = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_carpets_l1815_181505


namespace NUMINAMATH_CALUDE_birthday_candles_cost_l1815_181559

/-- Calculates the total cost of blue and green candles given the ratio and number of red candles -/
def total_cost_blue_green_candles (ratio_red blue green : ℕ) (num_red : ℕ) (cost_blue cost_green : ℕ) : ℕ :=
  let units_per_ratio := num_red / ratio_red
  let num_blue := units_per_ratio * blue
  let num_green := units_per_ratio * green
  num_blue * cost_blue + num_green * cost_green

/-- Theorem stating that the total cost of blue and green candles is $333 given the problem conditions -/
theorem birthday_candles_cost : 
  total_cost_blue_green_candles 5 3 7 45 3 4 = 333 := by
  sorry

end NUMINAMATH_CALUDE_birthday_candles_cost_l1815_181559


namespace NUMINAMATH_CALUDE_function_monotonicity_l1815_181544

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2) * x + 2

theorem function_monotonicity (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  4 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l1815_181544


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1815_181589

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1815_181589


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1815_181539

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1815_181539


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1815_181508

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  width > 0 → 
  length > 0 → 
  width = length / 3 → 
  perimeter = 2 * (width + length) → 
  perimeter = 72 → 
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1815_181508


namespace NUMINAMATH_CALUDE_boundaries_hit_count_l1815_181557

/-- Represents the number of runs scored by a boundary -/
def boundary_runs : ℕ := 4

/-- Represents the number of runs scored by a six -/
def six_runs : ℕ := 6

/-- Represents the total runs scored by the batsman -/
def total_runs : ℕ := 120

/-- Represents the number of sixes hit by the batsman -/
def sixes_hit : ℕ := 8

/-- Represents the fraction of runs scored by running between wickets -/
def running_fraction : ℚ := 1/2

theorem boundaries_hit_count :
  ∃ (boundaries : ℕ),
    boundaries * boundary_runs + 
    sixes_hit * six_runs + 
    (running_fraction * total_runs).num = total_runs ∧
    boundaries = 3 := by sorry

end NUMINAMATH_CALUDE_boundaries_hit_count_l1815_181557


namespace NUMINAMATH_CALUDE_three_chords_when_sixty_degrees_l1815_181550

/-- Represents a configuration of concentric circles with tangent chords -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection on the larger circle -/
  angle : ℝ
  /-- The number of chords needed to form a closed polygon -/
  num_chords : ℕ

/-- Theorem stating that when the angle between chords is 60°, exactly 3 chords are needed -/
theorem three_chords_when_sixty_degrees (config : ConcentricCirclesWithChords) :
  config.angle = 60 → config.num_chords = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_chords_when_sixty_degrees_l1815_181550


namespace NUMINAMATH_CALUDE_laura_to_ken_ratio_l1815_181573

/-- The number of tiles Don can paint per minute -/
def D : ℕ := 3

/-- The number of tiles Ken can paint per minute -/
def K : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def L : ℕ := 10

/-- The number of tiles Kim can paint per minute -/
def Kim : ℕ := L - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

/-- The theorem stating that the ratio of Laura's painting rate to Ken's painting rate is 2:1 -/
theorem laura_to_ken_ratio :
  (L : ℚ) / K = 2 / 1 ∧ 15 * (D + K + L + Kim) = total_tiles :=
by sorry

end NUMINAMATH_CALUDE_laura_to_ken_ratio_l1815_181573


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1815_181591

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 7 + a 13 = 4 →
  a 2 + a 12 = 8/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1815_181591


namespace NUMINAMATH_CALUDE_parent_age_problem_l1815_181500

/-- Given the conditions about the relationship between a parent's age and their daughter's age,
    prove that the parent's current age is 40 years. -/
theorem parent_age_problem (Y D : ℕ) : 
  Y = 4 * D →                 -- You are 4 times your daughter's age today
  Y - 7 = 11 * (D - 7) →      -- 7 years earlier, you were 11 times her age
  Y = 40                      -- Your current age is 40
:= by sorry

end NUMINAMATH_CALUDE_parent_age_problem_l1815_181500


namespace NUMINAMATH_CALUDE_probability_of_selection_for_given_sizes_l1815_181579

/-- Simple random sampling without replacement -/
structure SimpleRandomSampling where
  population_size : ℕ
  sample_size : ℕ
  sample_size_le_population : sample_size ≤ population_size

/-- The probability of an individual being selected in simple random sampling -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population_size

theorem probability_of_selection_for_given_sizes :
  ∀ (srs : SimpleRandomSampling),
    srs.population_size = 6 →
    srs.sample_size = 3 →
    probability_of_selection srs = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selection_for_given_sizes_l1815_181579


namespace NUMINAMATH_CALUDE_olivia_initial_wallet_l1815_181519

/-- The amount of money Olivia spent at the supermarket -/
def amount_spent : ℕ := 15

/-- The amount of money Olivia has left after spending -/
def amount_left : ℕ := 63

/-- The initial amount of money in Olivia's wallet -/
def initial_amount : ℕ := amount_spent + amount_left

theorem olivia_initial_wallet : initial_amount = 78 := by
  sorry

end NUMINAMATH_CALUDE_olivia_initial_wallet_l1815_181519


namespace NUMINAMATH_CALUDE_log_comparison_l1815_181515

theorem log_comparison : 
  (Real.log 4 / Real.log 3) > (0.3 ^ 4) ∧ (0.3 ^ 4) > (Real.log 0.9 / Real.log 1.1) := by
sorry

end NUMINAMATH_CALUDE_log_comparison_l1815_181515


namespace NUMINAMATH_CALUDE_ice_cream_scoops_l1815_181501

/-- The number of ice cream scoops served to a family at Ice Cream Palace -/
def total_scoops (single_cone waffle_bowl banana_split double_cone : ℕ) : ℕ :=
  single_cone + waffle_bowl + banana_split + double_cone

/-- Theorem: Given the conditions of the ice cream orders, the total number of scoops served is 10 -/
theorem ice_cream_scoops :
  ∀ (single_cone waffle_bowl banana_split double_cone : ℕ),
    single_cone = 1 →
    banana_split = 3 * single_cone →
    waffle_bowl = banana_split + 1 →
    double_cone = 2 →
    total_scoops single_cone waffle_bowl banana_split double_cone = 10 :=
by
  sorry

#check ice_cream_scoops

end NUMINAMATH_CALUDE_ice_cream_scoops_l1815_181501


namespace NUMINAMATH_CALUDE_florist_roses_l1815_181564

/-- A problem about a florist's roses -/
theorem florist_roses (initial : ℕ) (sold : ℕ) (final : ℕ) (picked : ℕ) : 
  initial = 11 → sold = 2 → final = 41 → picked = final - (initial - sold) → picked = 32 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_l1815_181564


namespace NUMINAMATH_CALUDE_people_eating_both_veg_nonveg_l1815_181580

/-- The number of people who eat only vegetarian food -/
def only_veg : ℕ := 13

/-- The total number of people who eat vegetarian food -/
def total_veg : ℕ := 21

/-- The number of people who eat both vegetarian and non-vegetarian food -/
def both_veg_nonveg : ℕ := total_veg - only_veg

theorem people_eating_both_veg_nonveg : both_veg_nonveg = 8 := by
  sorry

end NUMINAMATH_CALUDE_people_eating_both_veg_nonveg_l1815_181580


namespace NUMINAMATH_CALUDE_cars_distance_theorem_l1815_181516

/-- The distance between two cars after their movements on a main road -/
def final_distance (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem stating the final distance between two cars -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  final_distance initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval final_distance 113 50 35

end NUMINAMATH_CALUDE_cars_distance_theorem_l1815_181516


namespace NUMINAMATH_CALUDE_barbaras_savings_l1815_181538

/-- Calculates the current savings given the total cost, weekly allowance, and remaining weeks to save. -/
def currentSavings (totalCost : ℕ) (weeklyAllowance : ℕ) (remainingWeeks : ℕ) : ℕ :=
  totalCost - (weeklyAllowance * remainingWeeks)

/-- Proves that given the specific conditions, Barbara's current savings is $20. -/
theorem barbaras_savings :
  let watchCost : ℕ := 100
  let weeklyAllowance : ℕ := 5
  let remainingWeeks : ℕ := 16
  currentSavings watchCost weeklyAllowance remainingWeeks = 20 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_savings_l1815_181538


namespace NUMINAMATH_CALUDE_x_varies_as_three_fifths_of_z_l1815_181587

-- Define the relationships between x, y, and z
def varies_as_cube (x y : ℝ) : Prop := ∃ k : ℝ, x = k * y^3

def varies_as_fifth_root (y z : ℝ) : Prop := ∃ j : ℝ, y = j * z^(1/5)

def varies_as_power (x z : ℝ) (n : ℝ) : Prop := ∃ m : ℝ, x = m * z^n

-- State the theorem
theorem x_varies_as_three_fifths_of_z (x y z : ℝ) :
  varies_as_cube x y → varies_as_fifth_root y z → varies_as_power x z (3/5) :=
by sorry

end NUMINAMATH_CALUDE_x_varies_as_three_fifths_of_z_l1815_181587


namespace NUMINAMATH_CALUDE_constant_value_proof_l1815_181543

/-- Given consecutive integers x, y, and z where x > y > z, z = 2, 
    and 2x + 3y + 3z = 5y + C, prove that C = 8 -/
theorem constant_value_proof (x y z : ℤ) (C : ℤ) 
    (h1 : x = z + 2)
    (h2 : y = z + 1)
    (h3 : x > y ∧ y > z)
    (h4 : z = 2)
    (h5 : 2*x + 3*y + 3*z = 5*y + C) : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l1815_181543


namespace NUMINAMATH_CALUDE_dance_class_theorem_l1815_181592

theorem dance_class_theorem (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 40 →
  Finset.card A = 18 →
  Finset.card B = 22 →
  Finset.card (A ∩ B) = 10 →
  Finset.card (U \ (A ∪ B)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_dance_class_theorem_l1815_181592


namespace NUMINAMATH_CALUDE_annual_yield_improvement_l1815_181570

/-- The percentage improvement in annual yield given last year's and this year's ranges -/
theorem annual_yield_improvement (last_year_range this_year_range : ℝ) 
  (h1 : last_year_range = 10000)
  (h2 : this_year_range = 11500) :
  (this_year_range - last_year_range) / last_year_range * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_annual_yield_improvement_l1815_181570


namespace NUMINAMATH_CALUDE_negative_seven_times_sum_l1815_181555

theorem negative_seven_times_sum : -7 * 45 + (-7) * 55 = -700 := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_times_sum_l1815_181555


namespace NUMINAMATH_CALUDE_final_population_theorem_l1815_181533

/-- Calculates the population after two years of change -/
def population_after_two_years (initial_population : ℕ) : ℕ :=
  let after_increase := initial_population * 130 / 100
  let after_decrease := after_increase * 70 / 100
  after_decrease

/-- Theorem stating the final population after two years of change -/
theorem final_population_theorem :
  population_after_two_years 15000 = 13650 := by
  sorry

end NUMINAMATH_CALUDE_final_population_theorem_l1815_181533


namespace NUMINAMATH_CALUDE_escalator_rate_calculation_l1815_181598

/-- The rate at which the escalator moves, in feet per second -/
def escalator_rate : ℝ := 11

/-- The length of the escalator, in feet -/
def escalator_length : ℝ := 140

/-- The rate at which the person walks, in feet per second -/
def person_walking_rate : ℝ := 3

/-- The time taken by the person to cover the entire length, in seconds -/
def time_taken : ℝ := 10

theorem escalator_rate_calculation :
  (person_walking_rate + escalator_rate) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_rate_calculation_l1815_181598


namespace NUMINAMATH_CALUDE_tan_ratio_inequality_l1815_181572

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_inequality_l1815_181572


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1815_181561

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1815_181561


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1815_181531

/-- Given a circle and a line, find the equations of lines tangent to the circle and parallel to the given line -/
theorem tangent_line_to_circle (x y : ℝ) : 
  let circle := {(x, y) | x^2 + y^2 + 2*y = 0}
  let l2 := {(x, y) | 3*x + 4*y - 6 = 0}
  ∃ (b : ℝ), (b = -1 ∨ b = 9) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ {(x, y) | 3*x + 4*y + b = 0} → 
      (∃ (x0 y0 : ℝ), (x0, y0) ∈ circle ∧ 
        ((x - x0)^2 + (y - y0)^2 = 1 ∧
         ∀ (x1 y1 : ℝ), (x1, y1) ∈ circle → (x1 - x0)^2 + (y1 - y0)^2 ≤ 1)))
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1815_181531


namespace NUMINAMATH_CALUDE_james_total_score_l1815_181520

theorem james_total_score (field_goals : ℕ) (two_point_shots : ℕ) 
  (h1 : field_goals = 13) (h2 : two_point_shots = 20) : 
  field_goals * 3 + two_point_shots * 2 = 79 := by
sorry

end NUMINAMATH_CALUDE_james_total_score_l1815_181520


namespace NUMINAMATH_CALUDE_bryce_raisins_l1815_181507

theorem bryce_raisins : ∃ (b c : ℕ), b = c + 8 ∧ c = b / 3 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_bryce_raisins_l1815_181507


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l1815_181540

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |x - b|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f x (-2) 1 < 6 ↔ -1 < x ∧ x < 3 := by sorry

-- Part 2
theorem min_value_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x : ℝ, f x a b = 1 ∧ ∀ y : ℝ, f y a b ≥ 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2/c + 1/d ≥ 4) ∧
  (∃ e g : ℝ, e > 0 ∧ g > 0 ∧ 2/e + 1/g = 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l1815_181540


namespace NUMINAMATH_CALUDE_rational_roots_of_polynomial_l1815_181581

theorem rational_roots_of_polynomial (x : ℚ) :
  (3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0) ↔ (x = 1 ∨ x = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_polynomial_l1815_181581


namespace NUMINAMATH_CALUDE_trig_identity_l1815_181576

theorem trig_identity (θ : Real) (h : Real.tan θ = Real.sqrt 3) :
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1815_181576


namespace NUMINAMATH_CALUDE_puzzle_sum_l1815_181571

theorem puzzle_sum (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * 1000 + B - (5000 + C * 10 + 9) = 1000 + D * 100 + 90 + 3 →
  A + B + C + D = 18 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_sum_l1815_181571


namespace NUMINAMATH_CALUDE_inequality_proof_l1815_181526

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 + 1/x) * (1 + 1/y) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1815_181526


namespace NUMINAMATH_CALUDE_hexagon_angle_A_l1815_181599

/-- A hexagon is a polygon with 6 sides -/
def Hexagon (A B C D E F : ℝ) : Prop :=
  A + B + C + D + E + F = 720

/-- The theorem states that in a hexagon ABCDEF where B = 134°, C = 98°, D = 120°, E = 139°, and F = 109°, the measure of angle A is 120° -/
theorem hexagon_angle_A (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) : 
  A = 120 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_A_l1815_181599


namespace NUMINAMATH_CALUDE_flag_paint_cost_l1815_181552

/-- Calculates the cost of paint for a flag given its dimensions and paint properties -/
theorem flag_paint_cost (width height : ℝ) (paint_cost_per_quart : ℝ) (coverage_per_quart : ℝ) : 
  width = 5 → height = 4 → paint_cost_per_quart = 2 → coverage_per_quart = 4 → 
  (2 * width * height / coverage_per_quart) * paint_cost_per_quart = 20 := by
sorry


end NUMINAMATH_CALUDE_flag_paint_cost_l1815_181552


namespace NUMINAMATH_CALUDE_sine_function_period_l1815_181595

/-- Given a sinusoidal function y = 2sin(ωx + φ) with ω > 0,
    if the maximum value 2 occurs at x = π/6 and
    the minimum value -2 occurs at x = 2π/3,
    then ω = 2. -/
theorem sine_function_period (ω φ : ℝ) (h_ω_pos : ω > 0) :
  (∀ x : ℝ, 2 * Real.sin (ω * x + φ) ≤ 2) ∧
  (2 * Real.sin (ω * (π / 6) + φ) = 2) ∧
  (2 * Real.sin (ω * (2 * π / 3) + φ) = -2) →
  ω = 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_period_l1815_181595


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1815_181506

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 1500 →
  percentage = 20 →
  final = initial * (1 + percentage / 100) →
  final = 1800 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1815_181506


namespace NUMINAMATH_CALUDE_moon_temperature_difference_l1815_181528

/-- The temperature difference between day and night on the moon's surface. -/
def moonTemperatureDifference (dayTemp : ℝ) (nightTemp : ℝ) : ℝ :=
  dayTemp - nightTemp

/-- Theorem stating the temperature difference on the moon's surface. -/
theorem moon_temperature_difference :
  moonTemperatureDifference 127 (-183) = 310 := by
  sorry

end NUMINAMATH_CALUDE_moon_temperature_difference_l1815_181528


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1815_181504

theorem rectangle_triangle_area_ratio : 
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l1815_181504


namespace NUMINAMATH_CALUDE_handshake_theorem_l1815_181596

theorem handshake_theorem (n : ℕ) (h : n = 8) :
  let total_people := n
  let num_teams := n / 2
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l1815_181596


namespace NUMINAMATH_CALUDE_log_inequality_implies_sum_nonnegative_l1815_181565

theorem log_inequality_implies_sum_nonnegative (x y : ℝ) :
  (Real.log 3 / Real.log 2)^x + (Real.log 5 / Real.log 3)^y ≥ 
  (Real.log 2 / Real.log 3)^y + (Real.log 3 / Real.log 5)^x →
  x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_sum_nonnegative_l1815_181565


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1815_181522

theorem right_triangle_acute_angles (α β : ℝ) : 
  α + β = 90 →  -- sum of acute angles in a right triangle is 90°
  α = 54 →      -- one acute angle is 54°
  β = 36 :=     -- the other acute angle is 36°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1815_181522


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1815_181578

theorem arctan_sum_equals_pi_over_four :
  ∃ (n : ℕ), n > 0 ∧ Real.arctan (1/3) + Real.arctan (1/5) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/4 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l1815_181578


namespace NUMINAMATH_CALUDE_sum_convergence_l1815_181594

/-- Given two sequences of real numbers (a_n) and (b_n) satisfying the condition
    (3 - 2i)^n = a_n + b_ni for all integers n ≥ 0, where i = √(-1),
    prove that the sum ∑(n=0 to ∞) (a_n * b_n) / 8^n converges to 4/5. -/
theorem sum_convergence (a b : ℕ → ℝ) 
    (h : ∀ n : ℕ, Complex.I ^ 2 = -1 ∧ (3 - 2 * Complex.I) ^ n = a n + b n * Complex.I) :
    HasSum (λ n => (a n * b n) / 8^n) (4/5) :=
by sorry

end NUMINAMATH_CALUDE_sum_convergence_l1815_181594


namespace NUMINAMATH_CALUDE_power_of_two_multiple_one_two_l1815_181517

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every power of 2, there exists a multiple of it that only contains digits 1 and 2 -/
theorem power_of_two_multiple_one_two :
  ∀ k : ℕ, ∃ n : ℕ, 2^k ∣ n ∧ onlyOneAndTwo n := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_multiple_one_two_l1815_181517


namespace NUMINAMATH_CALUDE_remainder_theorem_l1815_181585

theorem remainder_theorem (x : ℝ) : 
  ∃ (P : ℝ → ℝ) (S : ℝ → ℝ), 
    (∀ x, x^105 = (x^2 - 4*x + 3) * P x + S x) ∧ 
    (∀ x, S x = (3^105 * (x - 1) - (x - 2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1815_181585


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1815_181524

/-- Represents a configuration of unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  num_outlying : ℕ
  num_exposed_faces : ℕ

/-- Calculates the volume of a cube configuration -/
def volume (config : CubeConfiguration) : ℕ := config.num_cubes

/-- Calculates the surface area of a cube configuration -/
def surface_area (config : CubeConfiguration) : ℕ := config.num_exposed_faces

/-- The specific configuration described in the problem -/
def problem_config : CubeConfiguration :=
  { num_cubes := 8
  , num_outlying := 7
  , num_exposed_faces := 33 }

/-- Theorem stating the ratio of volume to surface area for the given configuration -/
theorem volume_to_surface_area_ratio (config : CubeConfiguration) :
  config = problem_config →
  (volume config : ℚ) / (surface_area config : ℚ) = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1815_181524


namespace NUMINAMATH_CALUDE_least_positive_integer_divisibility_l1815_181577

theorem least_positive_integer_divisibility : ∃ d : ℕ+, 
  (∀ k : ℕ+, k < d → ¬(13 ∣ (k^3 + 1000))) ∧ (13 ∣ (d^3 + 1000)) ∧ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisibility_l1815_181577


namespace NUMINAMATH_CALUDE_anna_ate_three_cupcakes_l1815_181574

def total_cupcakes : ℕ := 60
def fraction_given_away : ℚ := 4/5
def cupcakes_left : ℕ := 9

theorem anna_ate_three_cupcakes :
  total_cupcakes - (fraction_given_away * total_cupcakes).floor - cupcakes_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_anna_ate_three_cupcakes_l1815_181574


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1815_181521

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + x + y = 120) : 
  x^2 + y^2 = 11980 / 121 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l1815_181521


namespace NUMINAMATH_CALUDE_rectangle_circle_equality_l1815_181502

/-- The length of a rectangle with width 3 units whose perimeter equals the circumference of a circle with radius 5 units is 5π - 3. -/
theorem rectangle_circle_equality (l : ℝ) : 
  (2 * (l + 3) = 2 * π * 5) → l = 5 * π - 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_equality_l1815_181502


namespace NUMINAMATH_CALUDE_combined_shape_perimeter_l1815_181545

/-- Given a shape consisting of a rectangle and a right triangle sharing one side,
    where the rectangle has sides of length 6 and x, and the triangle has legs of length x and 6,
    the perimeter of the combined shape is 18 + 2x + √(x^2 + 36). -/
theorem combined_shape_perimeter (x : ℝ) :
  let rectangle_perimeter := 2 * (6 + x)
  let triangle_hypotenuse := Real.sqrt (x^2 + 36)
  let shared_side := x
  rectangle_perimeter + x + 6 + triangle_hypotenuse - shared_side = 18 + 2*x + Real.sqrt (x^2 + 36) := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_perimeter_l1815_181545


namespace NUMINAMATH_CALUDE_max_triangle_area_l1815_181593

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 6 = 0

/-- Circle C in the xy-plane -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

/-- Point on circle C -/
def point_on_C (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

/-- Intersection points of line l and circle C -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

/-- Area of triangle PAB -/
noncomputable def triangle_area (P A B : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Maximum area of triangle PAB -/
theorem max_triangle_area :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  ∃ max_area : ℝ, max_area = (27 * Real.sqrt 3) / 4 ∧
  ∀ P : ℝ × ℝ, point_on_C P → triangle_area P A B ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1815_181593


namespace NUMINAMATH_CALUDE_min_value_theorem_l1815_181509

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 2 / y) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1815_181509


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1815_181548

/-- Given two lines l₁: ax + y + 1 = 0 and l₂: x - 2y + 1 = 0,
    if they are perpendicular, then a = 2 -/
theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y, ax + y + 1 = 0 ∧ x - 2*y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂, ax₁ + y₁ + 1 = 0 ∧ x₁ - 2*y₁ + 1 = 0 ∧
                   ax₂ + y₂ + 1 = 0 ∧ x₂ - 2*y₂ + 1 = 0 →
                   (x₂ - x₁) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1815_181548


namespace NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l1815_181541

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_seven_eight_l1815_181541


namespace NUMINAMATH_CALUDE_probability_one_defective_six_two_two_l1815_181553

/-- The probability of selecting exactly one defective product from a set of items. -/
def probability_one_defective (total_items defective_items items_selected : ℕ) : ℚ :=
  let favorable_outcomes := (total_items - defective_items).choose (items_selected - 1) * defective_items.choose 1
  let total_outcomes := total_items.choose items_selected
  favorable_outcomes / total_outcomes

/-- Theorem: The probability of selecting exactly one defective product when taking 2 items at random from 6 items with 2 defective products is 8/15. -/
theorem probability_one_defective_six_two_two :
  probability_one_defective 6 2 2 = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_defective_six_two_two_l1815_181553


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1815_181529

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1815_181529


namespace NUMINAMATH_CALUDE_regular_ticket_price_l1815_181568

/-- Calculates the price of each regular ticket given the initial savings,
    VIP ticket information, number of regular tickets, and remaining money. -/
theorem regular_ticket_price
  (initial_savings : ℕ)
  (vip_ticket_count : ℕ)
  (vip_ticket_price : ℕ)
  (regular_ticket_count : ℕ)
  (remaining_money : ℕ)
  (h1 : initial_savings = 500)
  (h2 : vip_ticket_count = 2)
  (h3 : vip_ticket_price = 100)
  (h4 : regular_ticket_count = 3)
  (h5 : remaining_money = 150)
  (h6 : initial_savings ≥ vip_ticket_count * vip_ticket_price + remaining_money) :
  (initial_savings - (vip_ticket_count * vip_ticket_price + remaining_money)) / regular_ticket_count = 50 :=
by sorry

end NUMINAMATH_CALUDE_regular_ticket_price_l1815_181568


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1815_181583

/-- The circumference of the base of a right circular cone formed from a circular piece of paper 
    with radius 6 inches, after removing a 180-degree sector, is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) : 
  let full_circumference := 2 * π * r
  let removed_angle := π  -- 180 degrees in radians
  let remaining_angle := 2 * π - removed_angle
  let base_circumference := (remaining_angle / (2 * π)) * full_circumference
  base_circumference = 6 * π := by
sorry


end NUMINAMATH_CALUDE_cone_base_circumference_l1815_181583


namespace NUMINAMATH_CALUDE_basketball_game_scores_l1815_181513

/-- Represents the scores of a basketball team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    s.q2 = s.q1 * r ∧
    s.q3 = s.q1 * r^2 ∧
    s.q4 = s.q1 * r^3

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧
    s.q2 = s.q1 + d ∧
    s.q3 = s.q1 + 2*d ∧
    s.q4 = s.q1 + 3*d

/-- The main theorem about the basketball game -/
theorem basketball_game_scores 
  (eagles lions : TeamScores)
  (h1 : eagles.q1 = lions.q1)
  (h2 : is_increasing_geometric eagles)
  (h3 : is_increasing_arithmetic lions)
  (h4 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 = 
        lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2)
  (h5 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100) :
  eagles.q1 + eagles.q2 + lions.q1 + lions.q2 = 43 :=
sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l1815_181513


namespace NUMINAMATH_CALUDE_total_money_value_l1815_181536

-- Define the number of nickels (and quarters)
def num_coins : ℕ := 40

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem total_money_value : 
  (num_coins * nickel_value + num_coins * quarter_value) / cents_per_dollar = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_money_value_l1815_181536


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l1815_181511

open Real

/-- The derivative of x * ln(x) is 1 + ln(x) -/
theorem derivative_x_ln_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => x * log x) x = 1 + log x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l1815_181511


namespace NUMINAMATH_CALUDE_sin_720_equals_0_l1815_181527

theorem sin_720_equals_0 (n : ℤ) (h1 : -90 ≤ n ∧ n ≤ 90) (h2 : Real.sin (n * π / 180) = Real.sin (720 * π / 180)) : n = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_720_equals_0_l1815_181527


namespace NUMINAMATH_CALUDE_nadia_mistakes_l1815_181582

/-- Represents Nadia's piano playing statistics -/
structure PianoStats where
  mistakes_per_40_notes : ℕ
  notes_per_minute : ℕ
  playing_time : ℕ

/-- Calculates the number of mistakes Nadia makes given her piano playing statistics -/
def calculate_mistakes (stats : PianoStats) : ℕ :=
  let total_notes := stats.notes_per_minute * stats.playing_time
  let blocks_of_40 := total_notes / 40
  blocks_of_40 * stats.mistakes_per_40_notes

/-- Theorem stating that Nadia makes 36 mistakes in 8 minutes of playing -/
theorem nadia_mistakes (stats : PianoStats)
  (h1 : stats.mistakes_per_40_notes = 3)
  (h2 : stats.notes_per_minute = 60)
  (h3 : stats.playing_time = 8) :
  calculate_mistakes stats = 36 := by
  sorry


end NUMINAMATH_CALUDE_nadia_mistakes_l1815_181582


namespace NUMINAMATH_CALUDE_city_population_ratio_l1815_181554

/-- The population of Lake View -/
def lake_view_population : ℕ := 24000

/-- The difference between Lake View and Seattle populations -/
def population_difference : ℕ := 4000

/-- The total population of the three cities -/
def total_population : ℕ := 56000

/-- The ratio of Boise's population to Seattle's population -/
def population_ratio : ℚ := 3 / 5

theorem city_population_ratio :
  ∃ (boise seattle : ℕ),
    boise + seattle + lake_view_population = total_population ∧
    lake_view_population = seattle + population_difference ∧
    population_ratio = boise / seattle := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l1815_181554


namespace NUMINAMATH_CALUDE_result_is_fifty_l1815_181551

-- Define the original number
def x : ℝ := 150

-- Define the percentage
def percentage : ℝ := 0.60

-- Define the subtracted value
def subtracted : ℝ := 40

-- Theorem to prove
theorem result_is_fifty : percentage * x - subtracted = 50 := by
  sorry

end NUMINAMATH_CALUDE_result_is_fifty_l1815_181551


namespace NUMINAMATH_CALUDE_number_increased_by_twenty_percent_l1815_181546

theorem number_increased_by_twenty_percent (x : ℝ) : x * 1.2 = 1080 ↔ x = 900 := by sorry

end NUMINAMATH_CALUDE_number_increased_by_twenty_percent_l1815_181546


namespace NUMINAMATH_CALUDE_fraction_product_cubes_evaluate_fraction_product_l1815_181537

theorem fraction_product_cubes (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = (a * c / (b * d)) ^ 3 :=
by sorry

theorem evaluate_fraction_product :
  (8 / 9 : ℚ) ^ 3 * (3 / 4 : ℚ) ^ 3 = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cubes_evaluate_fraction_product_l1815_181537


namespace NUMINAMATH_CALUDE_exponent_fraction_simplification_l1815_181512

theorem exponent_fraction_simplification :
  (3^100 + 3^98) / (3^100 - 3^98) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_simplification_l1815_181512


namespace NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l1815_181558

theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l1815_181558


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l1815_181542

def f (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -1) ∧ (∀ x : ℝ, f x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l1815_181542


namespace NUMINAMATH_CALUDE_erdos_szekeres_101_l1815_181597

theorem erdos_szekeres_101 (σ : Fin 101 → Fin 101) :
  ∃ (s : Finset (Fin 101)) (f : Fin 11 → Fin 101),
    s.card = 11 ∧ 
    (∀ i j : Fin 11, i < j → (f i : ℕ) < (f j : ℕ) ∨ (f i : ℕ) > (f j : ℕ)) ∧
    (∀ i : Fin 11, f i ∈ s) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_101_l1815_181597


namespace NUMINAMATH_CALUDE_chips_in_bag_is_24_l1815_181569

/-- The number of chips in a bag, given the calorie and cost information --/
def chips_in_bag (calories_per_chip : ℕ) (cost_per_bag : ℕ) (total_calories : ℕ) (total_cost : ℕ) : ℕ :=
  (total_calories / calories_per_chip) / (total_cost / cost_per_bag)

/-- Theorem stating that there are 24 chips in a bag --/
theorem chips_in_bag_is_24 :
  chips_in_bag 10 2 480 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_chips_in_bag_is_24_l1815_181569


namespace NUMINAMATH_CALUDE_expression_value_l1815_181556

def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1815_181556


namespace NUMINAMATH_CALUDE_intersection_M_N_l1815_181562

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1815_181562


namespace NUMINAMATH_CALUDE_parcel_cost_correct_l1815_181567

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  12 + 5 * P

/-- Theorem stating the correctness of the parcel cost function -/
theorem parcel_cost_correct (P : ℕ) (h : P ≥ 1) :
  parcel_cost P = 15 + 5 * (P - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_correct_l1815_181567


namespace NUMINAMATH_CALUDE_jay_painting_time_l1815_181525

theorem jay_painting_time (bong_time : ℝ) (combined_time : ℝ) (jay_time : ℝ) : 
  bong_time = 3 → 
  combined_time = 1.2 → 
  (1 / jay_time) + (1 / bong_time) = (1 / combined_time) → 
  jay_time = 2 := by
sorry

end NUMINAMATH_CALUDE_jay_painting_time_l1815_181525


namespace NUMINAMATH_CALUDE_problem_1_l1815_181584

theorem problem_1 : 2 * Real.tan (π / 3) - |Real.sqrt 3 - 2| - 3 * Real.sqrt 3 + (1 / 3)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1815_181584


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l1815_181586

/-- Calculates the percentage of a workday spent in meetings given the workday length and meeting durations. -/
theorem workday_meeting_percentage 
  (workday_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_multiplier : ℕ) : 
  workday_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 3 →
  (first_meeting_minutes + first_meeting_minutes * second_meeting_multiplier) / (workday_hours * 60) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l1815_181586


namespace NUMINAMATH_CALUDE_slinkums_shipment_correct_l1815_181560

/-- The total number of Mr. Slinkums in the initial shipment -/
def total_slinkums : ℕ := 200

/-- The percentage of Mr. Slinkums on display -/
def display_percentage : ℚ := 30 / 100

/-- The number of Mr. Slinkums in storage -/
def storage_slinkums : ℕ := 140

/-- Theorem stating that the total number of Mr. Slinkums is correct given the conditions -/
theorem slinkums_shipment_correct : 
  (1 - display_percentage) * (total_slinkums : ℚ) = storage_slinkums := by
  sorry

end NUMINAMATH_CALUDE_slinkums_shipment_correct_l1815_181560


namespace NUMINAMATH_CALUDE_april_greatest_drop_l1815_181549

/-- Represents the months from January to June --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- Returns the price of the smartphone at the end of the given month --/
def price (m : Month) : Int :=
  match m with
  | Month.January => 350
  | Month.February => 330
  | Month.March => 370
  | Month.April => 340
  | Month.May => 320
  | Month.June => 300

/-- Calculates the price drop from one month to the next --/
def priceDrop (m : Month) : Int :=
  match m with
  | Month.January => price Month.January - price Month.February
  | Month.February => price Month.February - price Month.March
  | Month.March => price Month.March - price Month.April
  | Month.April => price Month.April - price Month.May
  | Month.May => price Month.May - price Month.June
  | Month.June => 0  -- No next month defined

/-- Theorem stating that April had the greatest monthly drop in price --/
theorem april_greatest_drop :
  ∀ m : Month, m ≠ Month.April → priceDrop Month.April ≥ priceDrop m :=
by sorry

end NUMINAMATH_CALUDE_april_greatest_drop_l1815_181549


namespace NUMINAMATH_CALUDE_cello_viola_pairs_count_l1815_181547

/-- The number of cello-viola pairs in a music store, where each pair consists of
    a cello and a viola made from the same tree. -/
def cello_viola_pairs : ℕ := 70

theorem cello_viola_pairs_count (total_cellos : ℕ) (total_violas : ℕ) 
  (prob_same_tree : ℚ) (h1 : total_cellos = 800) (h2 : total_violas = 600) 
  (h3 : prob_same_tree = 14583333333333335 / 100000000000000000) : 
  cello_viola_pairs = (prob_same_tree * total_cellos * total_violas : ℚ).num := by
  sorry

end NUMINAMATH_CALUDE_cello_viola_pairs_count_l1815_181547


namespace NUMINAMATH_CALUDE_smallest_five_digit_in_pascal_l1815_181530

/-- Pascal's triangle function -/
def pascal (n k : ℕ) : ℕ := sorry

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem smallest_five_digit_in_pascal :
  ∃ (n k : ℕ), pascal n k = 10000 ∧
  (∀ (m l : ℕ), pascal m l < 10000 ∨ (pascal m l = 10000 ∧ m ≥ n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_in_pascal_l1815_181530


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1815_181514

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop := sorry

theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1815_181514


namespace NUMINAMATH_CALUDE_sammys_offer_per_record_l1815_181518

theorem sammys_offer_per_record (total_records : ℕ) 
  (bryans_offer_high : ℕ) (bryans_offer_low : ℕ) (profit_difference : ℕ) :
  total_records = 200 →
  bryans_offer_high = 6 →
  bryans_offer_low = 1 →
  profit_difference = 100 →
  (total_records / 2 * bryans_offer_high + total_records / 2 * bryans_offer_low + profit_difference) / total_records = 4 := by
  sorry

end NUMINAMATH_CALUDE_sammys_offer_per_record_l1815_181518


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1815_181532

-- Define the inequality
def inequality (x : ℝ) : Prop := x^2 < -2*x + 15

-- Define the solution set
def solution_set : Set ℝ := {x | -5 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1815_181532


namespace NUMINAMATH_CALUDE_a_upper_bound_l1815_181534

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * a (n / 2) + 3 * a (n / 3) + 6 * a (n / 6)

theorem a_upper_bound : ∀ n : ℕ, a n ≤ 10 * n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l1815_181534


namespace NUMINAMATH_CALUDE_theater_ticket_cost_is_3320_l1815_181590

/-- Calculates the total cost of theater tickets sold given the following conditions:
    - Total tickets sold: 370
    - Orchestra ticket price: $12
    - Balcony ticket price: $8
    - 190 more balcony tickets sold than orchestra tickets
-/
def theater_ticket_cost : ℕ := by
  -- Define the total number of tickets sold
  let total_tickets : ℕ := 370
  -- Define the price of orchestra tickets
  let orchestra_price : ℕ := 12
  -- Define the price of balcony tickets
  let balcony_price : ℕ := 8
  -- Define the difference between balcony and orchestra tickets sold
  let balcony_orchestra_diff : ℕ := 190
  
  -- Calculate the number of orchestra tickets sold
  let orchestra_tickets : ℕ := (total_tickets - balcony_orchestra_diff) / 2
  -- Calculate the number of balcony tickets sold
  let balcony_tickets : ℕ := total_tickets - orchestra_tickets
  
  -- Calculate and return the total cost
  exact orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

/-- Theorem stating that the total cost of theater tickets is $3320 -/
theorem theater_ticket_cost_is_3320 : theater_ticket_cost = 3320 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_is_3320_l1815_181590


namespace NUMINAMATH_CALUDE_median_of_class_distribution_l1815_181588

/-- Represents the distribution of weekly reading times for students -/
structure ReadingTimeDistribution where
  six_hours : Nat
  seven_hours : Nat
  eight_hours : Nat
  nine_hours : Nat

/-- Calculates the median of a given reading time distribution -/
def median (d : ReadingTimeDistribution) : Real :=
  sorry

/-- The specific distribution of reading times for the 30 students -/
def class_distribution : ReadingTimeDistribution :=
  { six_hours := 7
  , seven_hours := 8
  , eight_hours := 5
  , nine_hours := 10 }

/-- The theorem stating that the median of the given distribution is 7.5 -/
theorem median_of_class_distribution :
  median class_distribution = 7.5 := by sorry

end NUMINAMATH_CALUDE_median_of_class_distribution_l1815_181588
