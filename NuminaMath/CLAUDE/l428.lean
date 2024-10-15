import Mathlib

namespace NUMINAMATH_CALUDE_permutation_combination_sum_l428_42829

/-- Given that A_n^m = 272 and C_n^m = 136, prove that m + n = 19 -/
theorem permutation_combination_sum (m n : ℕ) 
  (h1 : m.factorial * (n - m).factorial * 272 = n.factorial)
  (h2 : m.factorial * (n - m).factorial * 136 = n.factorial) : 
  m + n = 19 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l428_42829


namespace NUMINAMATH_CALUDE_base6_multiplication_l428_42845

/-- Converts a base-6 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base6_multiplication :
  toBase6 (toBase10 [6] * toBase10 [1, 2]) = [2, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base6_multiplication_l428_42845


namespace NUMINAMATH_CALUDE_slope_intercept_product_l428_42898

theorem slope_intercept_product (m b : ℚ) 
  (h1 : m = 3/4)
  (h2 : b = -5/3)
  (h3 : m > 0)
  (h4 : b < 0) : 
  m * b < -1 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_product_l428_42898


namespace NUMINAMATH_CALUDE_correct_parentheses_removal_l428_42808

theorem correct_parentheses_removal (a b c d : ℝ) : 
  (a^2 - (1 - 2*a) ≠ a^2 - 1 - 2*a) ∧ 
  (a^2 + (-1 - 2*a) ≠ a^2 - 1 + 2*a) ∧ 
  (a - (5*b - (2*c - 1)) = a - 5*b + 2*c - 1) ∧ 
  (-(a + b) + (c - d) ≠ -a - b + c + d) :=
by sorry

end NUMINAMATH_CALUDE_correct_parentheses_removal_l428_42808


namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l428_42848

/-- The line x = k intersects the parabola x = -2y^2 - 3y + 5 at exactly one point if and only if k = 49/8 -/
theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, k = -2 * y^2 - 3 * y + 5) ↔ k = 49/8 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l428_42848


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l428_42814

theorem smallest_number_of_eggs : ∀ (n : ℕ),
  n > 200 ∧
  ∃ (c : ℕ), n = 15 * c - 3 ∧
  c ≥ 14 →
  n ≥ 207 ∧
  ∃ (m : ℕ), m = 207 ∧ m > 200 ∧ ∃ (d : ℕ), m = 15 * d - 3 ∧ d ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l428_42814


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l428_42868

/-- The equation (x-3)^2 = (3y+4)^2 - 75 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x - 3)^2 = (3*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l428_42868


namespace NUMINAMATH_CALUDE_company_employees_l428_42882

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ).floor)
  (h2 : (20 : ℚ) / 100 * total = ((40 : ℚ) / 100 * total).floor / 2)
  (h3 : (60 : ℚ) / 100 * total = (20 : ℚ) / 100 * total + 40) :
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l428_42882


namespace NUMINAMATH_CALUDE_arc_square_region_area_coefficients_sum_l428_42875

/-- Represents a circular arc --/
structure CircularArc where
  radius : ℝ
  centralAngle : ℝ

/-- Represents the region formed by three circular arcs and a square --/
structure ArcSquareRegion where
  arcs : Fin 3 → CircularArc
  squareSideLength : ℝ

/-- The area of the region inside the arcs but outside the square --/
noncomputable def regionArea (r : ArcSquareRegion) : ℝ :=
  sorry

/-- Coefficients of the area expression a√b + cπ - d --/
structure AreaCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem arc_square_region_area_coefficients_sum :
  ∀ r : ArcSquareRegion,
  (∀ i : Fin 3, (r.arcs i).radius = 6 ∧ (r.arcs i).centralAngle = 45 * π / 180) →
  r.squareSideLength = 12 →
  ∃ coeff : AreaCoefficients,
    regionArea r = coeff.c * π - coeff.d ∧
    coeff.a + coeff.b + coeff.c + coeff.d = 174 :=
sorry

end NUMINAMATH_CALUDE_arc_square_region_area_coefficients_sum_l428_42875


namespace NUMINAMATH_CALUDE_fair_coin_tails_probability_l428_42851

-- Define a fair coin
def FairCoin : Type := Unit

-- Define the possible outcomes of a coin flip
inductive CoinOutcome : Type
| Heads : CoinOutcome
| Tails : CoinOutcome

-- Define the probability of getting tails for a fair coin
def probTails (coin : FairCoin) : ℚ := 1 / 2

-- Theorem statement
theorem fair_coin_tails_probability (coin : FairCoin) (previous_flips : List CoinOutcome) :
  probTails coin = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_tails_probability_l428_42851


namespace NUMINAMATH_CALUDE_linear_function_properties_l428_42892

/-- A linear function y = ax + b satisfying specific conditions -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x + b

theorem linear_function_properties (a b : ℝ) :
  (LinearFunction a b 1 = 1 ∧ LinearFunction a b 2 = -5) →
  (a = -6 ∧ b = 7 ∧
   LinearFunction a b 0 = 7 ∧
   ∀ x, LinearFunction a b x > 0 ↔ x < 7/6) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l428_42892


namespace NUMINAMATH_CALUDE_midpoint_polar_specific_points_l428_42895

/-- The midpoint of a line segment in polar coordinates --/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_polar_specific_points :
  let A : ℝ × ℝ := (9, π/3)
  let B : ℝ × ℝ := (9, 2*π/3)
  let M := midpoint_polar A.1 A.2 B.1 B.2
  (0 < A.1 ∧ 0 ≤ A.2 ∧ A.2 < 2*π) ∧
  (0 < B.1 ∧ 0 ≤ B.2 ∧ B.2 < 2*π) →
  M = (9 * Real.sqrt 3 / 2, π/2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_polar_specific_points_l428_42895


namespace NUMINAMATH_CALUDE_ajax_exercise_hours_per_day_l428_42874

/-- Calculates the number of hours Ajax needs to exercise per day to reach his weight loss goal. -/
theorem ajax_exercise_hours_per_day 
  (initial_weight_kg : ℝ)
  (weight_loss_per_hour_lbs : ℝ)
  (kg_to_lbs_conversion : ℝ)
  (final_weight_lbs : ℝ)
  (days_of_exercise : ℕ)
  (h1 : initial_weight_kg = 80)
  (h2 : weight_loss_per_hour_lbs = 1.5)
  (h3 : kg_to_lbs_conversion = 2.2)
  (h4 : final_weight_lbs = 134)
  (h5 : days_of_exercise = 14) :
  (initial_weight_kg * kg_to_lbs_conversion - final_weight_lbs) / (weight_loss_per_hour_lbs * days_of_exercise) = 2 := by
  sorry

#check ajax_exercise_hours_per_day

end NUMINAMATH_CALUDE_ajax_exercise_hours_per_day_l428_42874


namespace NUMINAMATH_CALUDE_product_equals_difference_of_squares_l428_42822

theorem product_equals_difference_of_squares (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_squares_l428_42822


namespace NUMINAMATH_CALUDE_class_size_l428_42896

theorem class_size (total_budget : ℕ) (souvenir_cost : ℕ) (remaining : ℕ) : 
  total_budget = 730 →
  souvenir_cost = 17 →
  remaining = 16 →
  (total_budget - remaining) / souvenir_cost = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l428_42896


namespace NUMINAMATH_CALUDE_g_values_l428_42860

/-- The real-valued function f -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 4)

/-- The complex-valued function g -/
def g (x y : ℝ) : ℂ := (f (2 * x + 3) : ℂ) + Complex.I * y

/-- Theorem stating the values of g(29,k) for k = 1, 2, 3 -/
theorem g_values : ∀ k ∈ ({1, 2, 3} : Set ℕ), g 29 k = (858 : ℂ) + k * Complex.I :=
sorry

end NUMINAMATH_CALUDE_g_values_l428_42860


namespace NUMINAMATH_CALUDE_bingo_first_column_count_l428_42864

/-- The number of ways to choose 5 distinct numbers from 1 to 15 -/
def bingo_first_column : ℕ :=
  (15 * 14 * 13 * 12 * 11)

/-- Theorem: The number of distinct possibilities for the first column
    of a MODIFIED SHORT BINGO card is 360360 -/
theorem bingo_first_column_count : bingo_first_column = 360360 := by
  sorry

end NUMINAMATH_CALUDE_bingo_first_column_count_l428_42864


namespace NUMINAMATH_CALUDE_travel_theorem_l428_42837

-- Define the cities and distances
def XY : ℝ := 4500
def XZ : ℝ := 4000

-- Define travel costs
def bus_cost_per_km : ℝ := 0.20
def plane_cost_per_km : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the theorem
theorem travel_theorem :
  let YZ : ℝ := Real.sqrt (XY^2 - XZ^2)
  let total_distance : ℝ := XY + YZ + XZ
  let bus_total_cost : ℝ := bus_cost_per_km * total_distance
  let plane_total_cost : ℝ := plane_booking_fee + plane_cost_per_km * total_distance
  total_distance = 10562 ∧ plane_total_cost < bus_total_cost := by
  sorry

end NUMINAMATH_CALUDE_travel_theorem_l428_42837


namespace NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_positive_mn_l428_42836

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x ≤ 10 - |x - 3| ↔ x ∈ Set.Icc (-8/3) 4 := by sorry

-- Theorem for part 2
theorem inequality_for_positive_mn (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_mn : m + 2 * n = m * n) :
  f m + f (-2 * n) ≥ 16 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_inequality_for_positive_mn_l428_42836


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l428_42844

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {2, 3, 4}

-- Define set B
def B : Finset Nat := {4, 5}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l428_42844


namespace NUMINAMATH_CALUDE_ratio_equivalence_l428_42872

theorem ratio_equivalence (x : ℚ) : 
  (12 : ℚ) / 8 = 6 / (x * 60) → x = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l428_42872


namespace NUMINAMATH_CALUDE_seed_germination_requires_water_l428_42853

-- Define a seed
structure Seed where
  water_content : ℝ
  germinated : Bool

-- Define the germination process
def germinate (s : Seed) : Prop :=
  s.germinated = true

-- Theorem: A seed cannot germinate without water
theorem seed_germination_requires_water (s : Seed) :
  germinate s → s.water_content > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_seed_germination_requires_water_l428_42853


namespace NUMINAMATH_CALUDE_at_least_one_good_part_l428_42847

theorem at_least_one_good_part (total : ℕ) (good : ℕ) (defective : ℕ) (pick : ℕ) :
  total = 20 →
  good = 16 →
  defective = 4 →
  pick = 3 →
  total = good + defective →
  (Nat.choose total pick) - (Nat.choose defective pick) = 1136 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_good_part_l428_42847


namespace NUMINAMATH_CALUDE_function_non_negative_implies_k_range_l428_42887

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + k + 3

/-- The theorem statement -/
theorem function_non_negative_implies_k_range (k : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f k x ≥ 0) → k ≥ -3/13 := by
  sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_k_range_l428_42887


namespace NUMINAMATH_CALUDE_no_such_function_l428_42877

theorem no_such_function : ¬∃ f : ℝ → ℝ, f 0 > 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l428_42877


namespace NUMINAMATH_CALUDE_line_equation_l428_42820

/-- A line parameterized by (x,y) = (3t + 6, 5t - 7) where t is a real number -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem line_equation :
  ∀ (t x y : ℝ), parameterized_line t = (x, y) →
  y = slope_intercept_form (5/3) (-17) x := by
sorry

end NUMINAMATH_CALUDE_line_equation_l428_42820


namespace NUMINAMATH_CALUDE_polygon_sides_l428_42807

/-- The number of diagonals that can be drawn from one vertex of an n-sided polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: If 2018 diagonals can be drawn from one vertex of an n-sided polygon, then n = 2021 -/
theorem polygon_sides (n : ℕ) (h : diagonals_from_vertex n = 2018) : n = 2021 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l428_42807


namespace NUMINAMATH_CALUDE_brownie_problem_l428_42883

theorem brownie_problem (initial_brownies : ℕ) 
  (h1 : initial_brownies = 16) 
  (children_ate_percent : ℚ) 
  (h2 : children_ate_percent = 1/4) 
  (family_ate_percent : ℚ) 
  (h3 : family_ate_percent = 1/2) 
  (lorraine_ate : ℕ) 
  (h4 : lorraine_ate = 1) : 
  initial_brownies - 
  (initial_brownies * children_ate_percent).floor - 
  ((initial_brownies - (initial_brownies * children_ate_percent).floor) * family_ate_percent).floor - 
  lorraine_ate = 5 := by
sorry


end NUMINAMATH_CALUDE_brownie_problem_l428_42883


namespace NUMINAMATH_CALUDE_faucet_leak_approx_l428_42809

/-- The volume of water leaked by an untightened faucet in 4 hours -/
def faucet_leak_volume : ℝ :=
  let drops_per_second : ℝ := 2
  let milliliters_per_drop : ℝ := 0.05
  let hours : ℝ := 4
  let seconds_per_hour : ℝ := 3600
  drops_per_second * milliliters_per_drop * hours * seconds_per_hour

/-- Assertion that the faucet leak volume is approximately 1.4 × 10^3 milliliters -/
theorem faucet_leak_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 10 ∧ |faucet_leak_volume - 1.4e3| < ε :=
sorry

end NUMINAMATH_CALUDE_faucet_leak_approx_l428_42809


namespace NUMINAMATH_CALUDE_faulty_key_theorem_l428_42879

/-- Represents a sequence of digits -/
def DigitSequence := List Nat

/-- Checks if a digit is valid (0-9) -/
def isValidDigit (d : Nat) : Bool := d ≤ 9

/-- Represents the count of each digit in a sequence -/
def DigitCount := Nat → Nat

/-- Counts the occurrences of each digit in a sequence -/
def countDigits (seq : DigitSequence) : DigitCount :=
  fun d => seq.filter (· = d) |>.length

/-- Checks if a digit meets the criteria for being faulty -/
def isFaultyCandidate (count : DigitCount) (d : Nat) : Bool :=
  isValidDigit d ∧ count d ≥ 5

/-- The main theorem -/
theorem faulty_key_theorem (attempted : DigitSequence) (registered : DigitSequence) :
  attempted.length = 10 →
  registered.length = 7 →
  (∃ d, isFaultyCandidate (countDigits attempted) d) →
  (∃ d, d ∈ [7, 9] ∧ isFaultyCandidate (countDigits attempted) d) :=
by sorry

end NUMINAMATH_CALUDE_faulty_key_theorem_l428_42879


namespace NUMINAMATH_CALUDE_fraction_calculation_l428_42841

theorem fraction_calculation : (900^2 : ℝ) / (264^2 - 256^2) = 194.711 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l428_42841


namespace NUMINAMATH_CALUDE_simplify_sqrt_450_l428_42876

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_450_l428_42876


namespace NUMINAMATH_CALUDE_smallest_sphere_and_largest_cylinder_radius_l428_42886

/-- Three identical cylindrical surfaces with radius R and mutually perpendicular axes that touch each other in pairs -/
structure PerpendicularCylinders (R : ℝ) :=
  (radius : ℝ := R)
  (perpendicular_axes : Prop)
  (touch_in_pairs : Prop)

theorem smallest_sphere_and_largest_cylinder_radius 
  (R : ℝ) 
  (h : R > 0) 
  (cylinders : PerpendicularCylinders R) : 
  ∃ (smallest_sphere_radius largest_cylinder_radius : ℝ),
    smallest_sphere_radius = (Real.sqrt 2 - 1) * R ∧
    largest_cylinder_radius = (Real.sqrt 2 - 1) * R :=
by sorry

end NUMINAMATH_CALUDE_smallest_sphere_and_largest_cylinder_radius_l428_42886


namespace NUMINAMATH_CALUDE_target_line_correct_l428_42801

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - y + 1 = 0 -/
def given_line : Line2D :=
  { a := 2, b := -1, c := 1 }

/-- Point A (-1, 0) -/
def point_A : Point2D :=
  { x := -1, y := 0 }

/-- The line we need to prove -/
def target_line : Line2D :=
  { a := 2, b := -1, c := 2 }

theorem target_line_correct :
  point_on_line point_A target_line ∧
  parallel_lines target_line given_line := by
  sorry

end NUMINAMATH_CALUDE_target_line_correct_l428_42801


namespace NUMINAMATH_CALUDE_f_difference_f_equation_solution_l428_42855

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2
theorem f_equation_solution : {x : ℝ | f x = x + 3} = {-1, 2} := by
  sorry

end NUMINAMATH_CALUDE_f_difference_f_equation_solution_l428_42855


namespace NUMINAMATH_CALUDE_complex_equation_solution_l428_42863

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2*I : ℂ)*a + b = 2*I → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l428_42863


namespace NUMINAMATH_CALUDE_pyramid_max_volume_l428_42889

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem pyramid_max_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 / 8) * Real.sqrt 3 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_max_volume_l428_42889


namespace NUMINAMATH_CALUDE_championship_outcomes_l428_42894

theorem championship_outcomes (n : ℕ) (m : ℕ) : 
  n = 5 → m = 3 → n ^ m = 125 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l428_42894


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l428_42824

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l428_42824


namespace NUMINAMATH_CALUDE_shaded_area_of_partitioned_isosceles_right_triangle_l428_42881

theorem shaded_area_of_partitioned_isosceles_right_triangle 
  (leg_length : ℝ) 
  (num_partitions : ℕ) 
  (num_shaded : ℕ) : 
  leg_length = 8 → 
  num_partitions = 16 → 
  num_shaded = 10 → 
  (1 / 2 * leg_length * leg_length) * (num_shaded / num_partitions) = 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_of_partitioned_isosceles_right_triangle_l428_42881


namespace NUMINAMATH_CALUDE_unique_solution_condition_l428_42805

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0) ↔ b < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l428_42805


namespace NUMINAMATH_CALUDE_petya_can_buy_ice_cream_l428_42834

theorem petya_can_buy_ice_cream (total : ℕ) (kolya vasya petya : ℕ) : 
  total = 2200 →
  kolya * 18 = vasya →
  total = kolya + vasya + petya →
  petya ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_petya_can_buy_ice_cream_l428_42834


namespace NUMINAMATH_CALUDE_difference_of_place_values_l428_42827

def numeral : ℕ := 7669

def place_value (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

theorem difference_of_place_values : 
  place_value 6 2 - place_value 6 1 = 540 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_place_values_l428_42827


namespace NUMINAMATH_CALUDE_consistency_comparison_l428_42866

/-- Represents a student's performance in a series of games -/
structure StudentPerformance where
  numGames : ℕ
  avgScore : ℝ
  stdDev : ℝ

/-- Defines what it means for a student to perform more consistently -/
def MoreConsistent (a b : StudentPerformance) : Prop :=
  a.numGames = b.numGames ∧ a.avgScore = b.avgScore ∧ a.stdDev < b.stdDev

theorem consistency_comparison (a b : StudentPerformance) 
  (h1 : a.numGames = b.numGames) 
  (h2 : a.avgScore = b.avgScore) 
  (h3 : a.stdDev < b.stdDev) : 
  MoreConsistent a b :=
sorry

end NUMINAMATH_CALUDE_consistency_comparison_l428_42866


namespace NUMINAMATH_CALUDE_point_ordering_l428_42861

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem point_ordering :
  let y₁ := f (-3)
  let y₂ := f 1
  let y₃ := f (-1/2)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_point_ordering_l428_42861


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l428_42850

theorem arithmetic_simplification : 2 - (-3) * 2 - 4 - (-5) - 6 - (-7) * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l428_42850


namespace NUMINAMATH_CALUDE_hcd_4760_280_minus_12_l428_42859

theorem hcd_4760_280_minus_12 : Nat.gcd 4760 280 - 12 = 268 := by
  sorry

end NUMINAMATH_CALUDE_hcd_4760_280_minus_12_l428_42859


namespace NUMINAMATH_CALUDE_prob_15th_roll_last_correct_l428_42856

/-- The probability of the 15th roll being the last roll when rolling an
    eight-sided die until getting the same number on consecutive rolls. -/
def prob_15th_roll_last : ℚ :=
  (7 ^ 13 : ℚ) / (8 ^ 14 : ℚ)

/-- The number of sides on the die. -/
def num_sides : ℕ := 8

/-- The number of rolls. -/
def num_rolls : ℕ := 15

theorem prob_15th_roll_last_correct :
  prob_15th_roll_last = (7 ^ (num_rolls - 2) : ℚ) / (num_sides ^ (num_rolls - 1) : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_15th_roll_last_correct_l428_42856


namespace NUMINAMATH_CALUDE_f_always_positive_l428_42862

def f (x : ℝ) : ℝ := x^2 + 3*x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l428_42862


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l428_42826

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (h : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l428_42826


namespace NUMINAMATH_CALUDE_expression_evaluation_l428_42888

theorem expression_evaluation : 3^4 - 4 * 3^3 + 6 * 3^2 - 4 * 3 + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l428_42888


namespace NUMINAMATH_CALUDE_inscribed_triangle_with_parallel_sides_l428_42842

/-- A line in the plane -/
structure Line where
  -- Add necessary fields for a line

/-- A circle in the plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- A point in the plane -/
structure Point where
  -- Add necessary fields for a point

/-- A triangle in the plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Check if a line is parallel to a side of a triangle -/
def line_parallel_to_side (l : Line) (t : Triangle) : Prop :=
  sorry

/-- Main theorem: Given three pairwise non-parallel lines and a circle,
    there exists a triangle inscribed in the circle with sides parallel to the given lines -/
theorem inscribed_triangle_with_parallel_sides
  (l1 l2 l3 : Line) (c : Circle)
  (h1 : ¬ are_parallel l1 l2)
  (h2 : ¬ are_parallel l2 l3)
  (h3 : ¬ are_parallel l3 l1) :
  ∃ (t : Triangle),
    point_on_circle t.A c ∧
    point_on_circle t.B c ∧
    point_on_circle t.C c ∧
    line_parallel_to_side l1 t ∧
    line_parallel_to_side l2 t ∧
    line_parallel_to_side l3 t :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_with_parallel_sides_l428_42842


namespace NUMINAMATH_CALUDE_maurice_cookout_invites_l428_42831

/-- The number of people Maurice can invite to the cookout --/
def people_invited : ℕ := by sorry

theorem maurice_cookout_invites :
  let packages : ℕ := 4
  let pounds_per_package : ℕ := 5
  let pounds_per_burger : ℕ := 2
  let total_pounds : ℕ := packages * pounds_per_package
  let total_burgers : ℕ := total_pounds / pounds_per_burger
  people_invited = total_burgers - 1 := by sorry

end NUMINAMATH_CALUDE_maurice_cookout_invites_l428_42831


namespace NUMINAMATH_CALUDE_polynomial_roots_l428_42858

def polynomial (x : ℝ) : ℝ :=
  x^6 - 2*x^5 - 9*x^4 + 14*x^3 + 24*x^2 - 20*x - 20

def has_zero_sum_pairs (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ p a = 0 ∧ p (-a) = 0 ∧ p b = 0 ∧ p (-b) = 0

theorem polynomial_roots : 
  has_zero_sum_pairs polynomial →
  (∀ x : ℝ, polynomial x = 0 ↔ 
    x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨
    x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∨
    x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l428_42858


namespace NUMINAMATH_CALUDE_percent_calculation_l428_42870

theorem percent_calculation :
  (0.02 / 100) * 12356 = 2.4712 := by sorry

end NUMINAMATH_CALUDE_percent_calculation_l428_42870


namespace NUMINAMATH_CALUDE_wedding_attendance_l428_42854

/-- The number of people Laura invited to her wedding. -/
def invited : ℕ := 220

/-- The percentage of people who typically don't show up. -/
def no_show_percentage : ℚ := 5 / 100

/-- The number of people expected to attend Laura's wedding. -/
def expected_attendance : ℕ := 209

/-- Proves that the expected attendance at Laura's wedding is 209 people. -/
theorem wedding_attendance : 
  (invited : ℚ) * (1 - no_show_percentage) = expected_attendance := by
  sorry

end NUMINAMATH_CALUDE_wedding_attendance_l428_42854


namespace NUMINAMATH_CALUDE_moon_radius_scientific_notation_l428_42843

/-- The radius of the moon in meters -/
def moon_radius : ℝ := 1738000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the moon's radius is correctly expressed in scientific notation -/
theorem moon_radius_scientific_notation :
  ∃ (sn : ScientificNotation), moon_radius = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_moon_radius_scientific_notation_l428_42843


namespace NUMINAMATH_CALUDE_min_height_for_box_l428_42835

/-- Represents the dimensions of a rectangular box with square base --/
structure BoxDimensions where
  base : ℕ  -- side length of the square base
  height : ℕ -- height of the box

/-- Calculates the surface area of the box --/
def surfaceArea (d : BoxDimensions) : ℕ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the height condition --/
def satisfiesHeightCondition (d : BoxDimensions) : Prop :=
  d.height = 2 * d.base + 1

/-- Checks if the box dimensions satisfy the surface area condition --/
def satisfiesSurfaceAreaCondition (d : BoxDimensions) : Prop :=
  surfaceArea d ≥ 130

/-- The main theorem stating the minimum height that satisfies all conditions --/
theorem min_height_for_box : 
  ∃ (d : BoxDimensions), 
    satisfiesHeightCondition d ∧ 
    satisfiesSurfaceAreaCondition d ∧ 
    (∀ (d' : BoxDimensions), 
      satisfiesHeightCondition d' ∧ 
      satisfiesSurfaceAreaCondition d' → 
      d.height ≤ d'.height) ∧
    d.height = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_height_for_box_l428_42835


namespace NUMINAMATH_CALUDE_inequality_solution_set_l428_42840

def solution_set : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ x^2 - 5*x + 6 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l428_42840


namespace NUMINAMATH_CALUDE_system_solution_l428_42899

theorem system_solution (x y : ℝ) (eq1 : x + y = 2) (eq2 : 3 * x - y = 8) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l428_42899


namespace NUMINAMATH_CALUDE_set_operation_equality_l428_42812

def U : Finset Nat := {1,2,3,4,5}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,2,5}

theorem set_operation_equality : 
  (U \ M) ∩ N = {1,2} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l428_42812


namespace NUMINAMATH_CALUDE_min_vertical_distance_l428_42823

-- Define the two functions
def f (x : ℝ) : ℝ := abs x
def g (x : ℝ) : ℝ := -x^2 - 3*x - 2

-- Define the vertical distance between the two functions
def verticalDistance (x : ℝ) : ℝ := abs (f x - g x)

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), verticalDistance x = 1 ∧ ∀ (y : ℝ), verticalDistance y ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l428_42823


namespace NUMINAMATH_CALUDE_smallest_k_with_remainders_l428_42825

theorem smallest_k_with_remainders : ∃ k : ℕ, 
  k > 1 ∧
  k % 17 = 1 ∧
  k % 11 = 1 ∧
  k % 6 = 2 ∧
  ∀ m : ℕ, m > 1 → m % 17 = 1 → m % 11 = 1 → m % 6 = 2 → k ≤ m :=
by
  use 188
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainders_l428_42825


namespace NUMINAMATH_CALUDE_gecko_eats_hundred_crickets_l428_42800

/-- The number of crickets a gecko eats over three days -/
def gecko_crickets : ℕ → Prop
| C => 
  -- Day 1: 30% of total
  let day1 : ℚ := 0.3 * C
  -- Day 2: 6 less than day 1
  let day2 : ℚ := day1 - 6
  -- Day 3: 34 crickets
  let day3 : ℕ := 34
  -- Total crickets eaten equals sum of three days
  C = day1.ceil + day2.ceil + day3

theorem gecko_eats_hundred_crickets : 
  ∃ C : ℕ, gecko_crickets C ∧ C = 100 := by sorry

end NUMINAMATH_CALUDE_gecko_eats_hundred_crickets_l428_42800


namespace NUMINAMATH_CALUDE_total_cars_produced_l428_42880

/-- Given that a car company produced 3,884 cars in North America and 2,871 cars in Europe,
    prove that the total number of cars produced is 6,755. -/
theorem total_cars_produced (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_produced_l428_42880


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l428_42803

theorem min_value_trig_expression (A : Real) (h : 0 < A ∧ A < Real.pi / 2) :
  Real.sqrt (Real.sin A ^ 4 + 1) + Real.sqrt (Real.cos A ^ 4 + 4) ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l428_42803


namespace NUMINAMATH_CALUDE_backyard_area_l428_42890

theorem backyard_area (length width : ℝ) 
  (h1 : length * 50 = 2000)
  (h2 : (2 * length + 2 * width) * 20 = 2000) :
  length * width = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l428_42890


namespace NUMINAMATH_CALUDE_least_value_theorem_l428_42846

theorem least_value_theorem (x y z : ℕ+) (h : 2 * x.val = 5 * y.val ∧ 5 * y.val = 6 * z.val) :
  ∃ n : ℤ, x.val + y.val + n = 26 ∧ ∀ m : ℤ, x.val + y.val + m = 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_value_theorem_l428_42846


namespace NUMINAMATH_CALUDE_orthocenter_PQR_l428_42869

/-- The orthocenter of a triangle PQR in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle PQR with given coordinates is (1/2, 13/2, 15/2). -/
theorem orthocenter_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (1/2, 13/2, 15/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_PQR_l428_42869


namespace NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l428_42804

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of odd positive integer divisors of n -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen positive integer divisor of n being odd -/
def prob_odd_divisor (n : ℕ) : ℚ :=
  (num_odd_divisors n : ℚ) / (num_divisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  prob_odd_divisor (factorial 15) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_15_factorial_l428_42804


namespace NUMINAMATH_CALUDE_option_C_most_suitable_for_comprehensive_survey_l428_42885

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Understanding the service life of a batch of light bulbs

/-- Defines what makes a survey comprehensive -/
def isComprehensive (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable_for_comprehensive_survey :
  ∀ (option : SurveyOption), isComprehensive option → option = SurveyOption.C :=
by sorry

end NUMINAMATH_CALUDE_option_C_most_suitable_for_comprehensive_survey_l428_42885


namespace NUMINAMATH_CALUDE_max_distance_between_vectors_l428_42884

theorem max_distance_between_vectors (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  (∀ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) → 
    ‖a - b‖ ≤ Real.sqrt 5 + 1) ∧
  (∃ a b : ℝ × ℝ, a = (x, y) ∧ b = (1, 2) ∧ 
    ‖a - b‖ = Real.sqrt 5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_vectors_l428_42884


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l428_42828

/-- Given a triangle with one side of length 10 and the opposite angle of 45°,
    the diameter of its circumscribed circle is 10√2. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 10) (h_angle : angle = Real.pi / 4) :
  (side / Real.sin angle) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l428_42828


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l428_42810

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 24| + |x - 30| = |3*x - 72| :=
by
  -- The unique solution is x = 26
  use 26
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l428_42810


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l428_42871

/-- Calculates the number of games in a single-elimination tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The number of teams in the tournament -/
def num_teams : ℕ := 24

theorem single_elimination_tournament_games :
  tournament_games num_teams = 23 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l428_42871


namespace NUMINAMATH_CALUDE_min_value_of_f_l428_42821

-- Define second-order product sum
def second_order_sum (a b c d : ℤ) : ℤ := a * d + b * c

-- Define third-order product sum
def third_order_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℤ) : ℤ :=
  a1 * (second_order_sum b2 b3 c2 c3) +
  a2 * (second_order_sum b1 b3 c1 c3) +
  a3 * (second_order_sum b1 b2 c1 c2)

-- Define the function f
def f (n : ℕ+) : ℤ := third_order_sum n 2 (-9) n 1 n 1 2 n

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℤ), m = -21 ∧ ∀ (n : ℕ+), f n ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l428_42821


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l428_42830

theorem overtime_hours_calculation (regular_rate overtime_rate total_pay : ℚ) 
  (h1 : regular_rate = 3)
  (h2 : overtime_rate = 2 * regular_rate)
  (h3 : total_pay = 186) : 
  (total_pay - 40 * regular_rate) / overtime_rate = 11 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l428_42830


namespace NUMINAMATH_CALUDE_intersection_point_determines_m_l428_42811

-- Define the two lines
def line1 (x y m : ℝ) : Prop := 3 * x - 2 * y = m
def line2 (x y : ℝ) : Prop := -x - 2 * y = -10

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y 6 ∧ line2 x y

-- Theorem statement
theorem intersection_point_determines_m :
  ∃ y : ℝ, intersection 4 y → (∀ m : ℝ, line1 4 y m → m = 6) := by sorry

end NUMINAMATH_CALUDE_intersection_point_determines_m_l428_42811


namespace NUMINAMATH_CALUDE_fifty_third_term_is_2_to_53_l428_42867

def double_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * double_sequence n

theorem fifty_third_term_is_2_to_53 :
  double_sequence 52 = 2^53 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_term_is_2_to_53_l428_42867


namespace NUMINAMATH_CALUDE_adult_ticket_price_l428_42817

theorem adult_ticket_price
  (total_tickets : ℕ)
  (senior_price : ℚ)
  (total_receipts : ℚ)
  (senior_tickets : ℕ)
  (h1 : total_tickets = 529)
  (h2 : senior_price = 15)
  (h3 : total_receipts = 9745)
  (h4 : senior_tickets = 348) :
  (total_receipts - senior_price * senior_tickets) / (total_tickets - senior_tickets) = 25 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l428_42817


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l428_42897

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = x + 3 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  (x^2 - 4*x + 4) / (x^2 - 4) / ((x - 2) / (x^2 + 2*x)) + 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l428_42897


namespace NUMINAMATH_CALUDE_total_animal_sightings_l428_42819

def week1_sightings : List Nat := [8, 7, 8, 11, 8, 7, 13]
def week2_sightings : List Nat := [7, 9, 10, 21, 11, 7, 17]

theorem total_animal_sightings :
  (week1_sightings.sum + week2_sightings.sum) = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_animal_sightings_l428_42819


namespace NUMINAMATH_CALUDE_tangent_point_min_value_on_interval_l428_42873

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_point (a : ℝ) :
  (∃ x > 0, f a x = 0 ∧ (deriv (f a)) x = 0) → a = 1 / Real.exp 1 :=
sorry

theorem min_value_on_interval (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_min_value_on_interval_l428_42873


namespace NUMINAMATH_CALUDE_probability_at_most_six_distinct_numbers_l428_42833

theorem probability_at_most_six_distinct_numbers : 
  let n_dice : ℕ := 8
  let n_faces : ℕ := 6
  let total_outcomes : ℕ := n_faces ^ n_dice
  let favorable_outcomes : ℕ := 3628800
  (favorable_outcomes : ℚ) / total_outcomes = 45 / 52 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_most_six_distinct_numbers_l428_42833


namespace NUMINAMATH_CALUDE_square_pyramid_sphere_ratio_l428_42802

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  -- Length of the edge of the square base
  baseEdge : ℝ
  -- Height of the pyramid (perpendicular distance from apex to base)
  height : ℝ

/-- Calculates the ratio of surface areas of circumscribed to inscribed spheres for a square pyramid -/
def sphereAreaRatio (p : SquarePyramid) : ℝ :=
  -- This function would contain the actual calculation
  sorry

theorem square_pyramid_sphere_ratio :
  let p := SquarePyramid.mk 8 6
  sphereAreaRatio p = 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sphere_ratio_l428_42802


namespace NUMINAMATH_CALUDE_green_ball_probability_l428_42839

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of selecting a green ball from three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

/-- Theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  let c1 : Container := ⟨8, 4⟩
  let c2 : Container := ⟨2, 4⟩
  let c3 : Container := ⟨2, 6⟩
  totalGreenProbability c1 c2 c3 = 7 / 12 := by
  sorry


end NUMINAMATH_CALUDE_green_ball_probability_l428_42839


namespace NUMINAMATH_CALUDE_students_present_l428_42857

theorem students_present (total : ℕ) (absent_fraction : ℚ) (present : ℕ) : 
  total = 28 → 
  absent_fraction = 2/7 → 
  present = total - (total * absent_fraction).floor → 
  present = 20 := by
sorry

end NUMINAMATH_CALUDE_students_present_l428_42857


namespace NUMINAMATH_CALUDE_parallel_vectors_dot_product_l428_42818

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The dot product of two 2D vectors -/
def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem parallel_vectors_dot_product :
  ∀ x : ℝ, 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  parallel a b → dot_product a b = -10 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_dot_product_l428_42818


namespace NUMINAMATH_CALUDE_det_equation_roots_l428_42865

/-- The determinant equation has either one or three real roots -/
theorem det_equation_roots (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let det := fun x => x * (x * x + a * a) + c * (b * x + a * b) - b * (a * c - b * x)
  ∃ (n : Fin 2), (n = 0 ∧ (∃! x, det x = d)) ∨ (n = 1 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ det x = d ∧ det y = d ∧ det z = d)) :=
by sorry

end NUMINAMATH_CALUDE_det_equation_roots_l428_42865


namespace NUMINAMATH_CALUDE_base_k_equivalence_l428_42893

/-- 
Given a natural number k, this function converts a number from base k to decimal.
The input is a list of digits in reverse order (least significant digit first).
-/
def baseKToDecimal (k : ℕ) (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * k^i) 0

/-- 
This theorem states that if 26 in decimal is equal to 32 in base-k, then k must be 8.
-/
theorem base_k_equivalence :
  ∀ k : ℕ, k > 1 → baseKToDecimal k [2, 3] = 26 → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_k_equivalence_l428_42893


namespace NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l428_42806

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are equilateral triangles -/
  adjacent_faces_equilateral : Bool
  /-- Side length of the equilateral triangular faces -/
  side_length : ℝ
  /-- Dihedral angle between the two adjacent equilateral faces -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the maximum projection area for a specific tetrahedron -/
theorem max_projection_area_specific_tetrahedron :
  ∀ t : Tetrahedron,
    t.adjacent_faces_equilateral = true →
    t.side_length = 1 →
    t.dihedral_angle = π / 3 →
    max_projection_area t = Real.sqrt 3 / 4 :=
  sorry

end NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l428_42806


namespace NUMINAMATH_CALUDE_initial_average_problem_l428_42838

theorem initial_average_problem (n : ℕ) (A : ℝ) (added_value : ℝ) (new_average : ℝ) 
  (h1 : n = 15)
  (h2 : added_value = 14)
  (h3 : new_average = 54)
  (h4 : (n : ℝ) * A + n * added_value = n * new_average) :
  A = 40 := by
sorry

end NUMINAMATH_CALUDE_initial_average_problem_l428_42838


namespace NUMINAMATH_CALUDE_cyclist_round_trip_time_l428_42852

/-- Calculates the total time for a cyclist's round trip given specific conditions. -/
theorem cyclist_round_trip_time 
  (total_distance : ℝ)
  (first_segment_distance : ℝ)
  (second_segment_distance : ℝ)
  (first_segment_speed : ℝ)
  (second_segment_speed : ℝ)
  (return_speed : ℝ)
  (h1 : total_distance = first_segment_distance + second_segment_distance)
  (h2 : first_segment_distance = 12)
  (h3 : second_segment_distance = 24)
  (h4 : first_segment_speed = 8)
  (h5 : second_segment_speed = 12)
  (h6 : return_speed = 9)
  : (first_segment_distance / first_segment_speed + 
     second_segment_distance / second_segment_speed + 
     total_distance / return_speed) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_round_trip_time_l428_42852


namespace NUMINAMATH_CALUDE_f_min_value_l428_42849

def f (x : ℝ) : ℝ := |x - 1| + |x + 4| - 5

theorem f_min_value :
  ∀ x : ℝ, f x ≥ 0 ∧ ∃ y : ℝ, f y = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l428_42849


namespace NUMINAMATH_CALUDE_parallelogram_area_l428_42891

/-- Proves that a parallelogram with base 16 cm and where 2 times the sum of its base and height is 56, has an area of 192 square centimeters. -/
theorem parallelogram_area (b h : ℝ) : 
  b = 16 → 2 * (b + h) = 56 → b * h = 192 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l428_42891


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l428_42816

theorem sum_of_reciprocals_bound (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 1) : 
  1 / (4*a + 3*b + c) + 1 / (3*a + b + 4*d) + 
  1 / (a + 4*c + 3*d) + 1 / (4*b + 3*c + d) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l428_42816


namespace NUMINAMATH_CALUDE_equation_holds_l428_42878

theorem equation_holds : (5 - 2) + 6 - (4 - 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l428_42878


namespace NUMINAMATH_CALUDE_unique_function_solution_l428_42832

theorem unique_function_solution :
  ∃! f : ℝ → ℝ, (∀ x y : ℝ, f (x + f y - 1) = x + y) ∧ (∀ x : ℝ, f x = x + 1/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l428_42832


namespace NUMINAMATH_CALUDE_equation_solution_l428_42815

theorem equation_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l428_42815


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_m_leq_neg_one_l428_42813

def A : Set (ℝ × ℝ) := {p | p.2 = Real.log (p.1 + 1) - 1}

def B (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 = m}

theorem disjoint_sets_imply_m_leq_neg_one (m : ℝ) :
  A ∩ B m = ∅ → m ≤ -1 := by sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_m_leq_neg_one_l428_42813
