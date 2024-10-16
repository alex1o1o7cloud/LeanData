import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_range_l1568_156848

open Real

theorem triangle_side_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- ABC is an acute triangle
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  Real.sqrt 3 * (a * cos B + b * cos A) = 2 * c * sin C ∧  -- Given equation
  b = 1 →  -- Given condition
  sqrt 3 / 2 < c ∧ c < sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1568_156848


namespace NUMINAMATH_CALUDE_M_intersect_N_l1568_156862

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem M_intersect_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1568_156862


namespace NUMINAMATH_CALUDE_determinant_inequality_solution_l1568_156841

-- Define the determinant
def det (a b c d : ℝ) : ℝ := |a * d - b * c|

-- Define the logarithm base sqrt(2)
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioo 1 2)}

-- State the theorem
theorem determinant_inequality_solution :
  {x : ℝ | log_sqrt2 (det 1 11 1 x) < 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_determinant_inequality_solution_l1568_156841


namespace NUMINAMATH_CALUDE_radio_selling_price_l1568_156808

theorem radio_selling_price 
  (purchase_price : ℝ) 
  (overhead_expenses : ℝ) 
  (profit_percent : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 20)
  (h3 : profit_percent = 22.448979591836732) : 
  ∃ (selling_price : ℝ), selling_price = 300 ∧ 
  selling_price = purchase_price + overhead_expenses + 
  (profit_percent / 100) * (purchase_price + overhead_expenses) :=
sorry

end NUMINAMATH_CALUDE_radio_selling_price_l1568_156808


namespace NUMINAMATH_CALUDE_xy_value_l1568_156892

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(6*y) = 81) :
  x * y = 644 / 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1568_156892


namespace NUMINAMATH_CALUDE_power_mod_500_l1568_156859

theorem power_mod_500 : 7^(7^(7^7)) % 500 = 343 := by sorry

end NUMINAMATH_CALUDE_power_mod_500_l1568_156859


namespace NUMINAMATH_CALUDE_square_difference_equality_l1568_156826

theorem square_difference_equality (m n : ℝ) :
  9 * m^2 - (m - 2*n)^2 = 4 * (2*m - n) * (m + n) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1568_156826


namespace NUMINAMATH_CALUDE_solution_difference_l1568_156814

theorem solution_difference (r s : ℝ) : 
  (r - 4) * (r + 4) = 24 * r - 96 →
  (s - 4) * (s + 4) = 24 * s - 96 →
  r ≠ s →
  r > s →
  r - s = 16 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l1568_156814


namespace NUMINAMATH_CALUDE_least_integer_square_64_more_than_double_l1568_156817

theorem least_integer_square_64_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 64 ∧ ∀ y : ℤ, y^2 = 2*y + 64 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_64_more_than_double_l1568_156817


namespace NUMINAMATH_CALUDE_sqrt_15_range_l1568_156853

theorem sqrt_15_range : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_range_l1568_156853


namespace NUMINAMATH_CALUDE_sqrt_difference_is_two_l1568_156807

theorem sqrt_difference_is_two (x : ℝ) : 
  Real.sqrt (x + 2 + 2 * Real.sqrt (x + 1)) - Real.sqrt (x + 2 - 2 * Real.sqrt (x + 1)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_difference_is_two_l1568_156807


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1568_156833

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) (h4 : a 4 = 5) :
  a 1 * a 7 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1568_156833


namespace NUMINAMATH_CALUDE_f_deriv_l1568_156893

/-- The function f(x) = 2x + 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

/-- Theorem: The derivative of f(x) = 2x + 3 is equal to 2 -/
theorem f_deriv : deriv f = λ _ => 2 := by sorry

end NUMINAMATH_CALUDE_f_deriv_l1568_156893


namespace NUMINAMATH_CALUDE_germs_per_dish_l1568_156844

theorem germs_per_dish :
  let total_germs : ℝ := 5.4 * 10^6
  let total_dishes : ℝ := 10800
  let germs_per_dish : ℝ := total_germs / total_dishes
  germs_per_dish = 500 :=
by sorry

end NUMINAMATH_CALUDE_germs_per_dish_l1568_156844


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1568_156824

theorem imaginary_part_of_product : Complex.im ((3 - 4*Complex.I) * (1 + 2*Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1568_156824


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1568_156871

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + k*y - 2 = 0 ∧ y = 1 ∧ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1568_156871


namespace NUMINAMATH_CALUDE_suv_highway_efficiency_l1568_156895

/-- Represents the fuel efficiency of an SUV -/
structure SUVFuelEfficiency where
  city_mpg : ℝ
  highway_mpg : ℝ
  max_distance : ℝ
  tank_capacity : ℝ

/-- Theorem stating the highway fuel efficiency of the SUV -/
theorem suv_highway_efficiency (suv : SUVFuelEfficiency)
  (h1 : suv.city_mpg = 7.6)
  (h2 : suv.max_distance = 268.4)
  (h3 : suv.tank_capacity = 22) :
  suv.highway_mpg = 12.2 := by
  sorry

#check suv_highway_efficiency

end NUMINAMATH_CALUDE_suv_highway_efficiency_l1568_156895


namespace NUMINAMATH_CALUDE_tangent_line_at_2_l1568_156809

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_2 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) → y = 13 * x - 32 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_l1568_156809


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l1568_156865

/-- Represents the weight of different types of candy in pounds -/
structure CandyWeights where
  chocolate : ℝ
  gummyBears : ℝ
  caramels : ℝ
  hardCandy : ℝ

/-- Calculates the total weight of candy -/
def totalWeight (cw : CandyWeights) : ℝ :=
  cw.chocolate + cw.gummyBears + cw.caramels + cw.hardCandy

/-- Frank's candy weights -/
def frankCandy : CandyWeights := {
  chocolate := 3,
  gummyBears := 2,
  caramels := 1,
  hardCandy := 4
}

/-- Gwen's candy weights -/
def gwenCandy : CandyWeights := {
  chocolate := 2,
  gummyBears := 2.5,
  caramels := 1,
  hardCandy := 1.5
}

/-- Theorem stating that the total combined weight of Frank and Gwen's Halloween candy is 17 pounds -/
theorem halloween_candy_weight :
  totalWeight frankCandy + totalWeight gwenCandy = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l1568_156865


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l1568_156845

/-- The time taken for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (train_length_positive : 0 < train_length)
  (time_to_cross_point_positive : 0 < time_to_cross_point)
  (platform_length_positive : 0 < platform_length)
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 1200) : 
  (train_length + platform_length) / (train_length / time_to_cross_point) = 240 := by
sorry


end NUMINAMATH_CALUDE_train_platform_passing_time_l1568_156845


namespace NUMINAMATH_CALUDE_max_d_value_l1568_156854

def a (n : ℕ) : ℕ := 150 + (n + 1)^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (k : ℕ), d k = 2 ∧ ∀ (n : ℕ), d n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1568_156854


namespace NUMINAMATH_CALUDE_kanul_total_amount_l1568_156803

/-- The total amount Kanul had -/
def T : ℝ := 93750

/-- The amount spent on raw materials -/
def raw_materials : ℝ := 35000

/-- The amount spent on machinery -/
def machinery : ℝ := 40000

/-- The percentage of total amount spent as cash -/
def cash_percentage : ℝ := 0.20

theorem kanul_total_amount :
  raw_materials + machinery + cash_percentage * T = T := by sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l1568_156803


namespace NUMINAMATH_CALUDE_pyramid_side_length_l1568_156825

/-- Regular triangular pyramid with specific properties -/
structure RegularPyramid where
  -- Base triangle side length
  a : ℝ
  -- Angle of inclination of face to base
  α : ℝ
  -- Height of the pyramid
  h : ℝ
  -- Condition that α is arctan(3/4)
  angle_condition : α = Real.arctan (3/4)
  -- Relation between height, side length, and angle
  height_relation : h = (a * Real.sqrt 3) / 2

/-- Polyhedron formed by intersecting prism with pyramid -/
structure Polyhedron (p : RegularPyramid) where
  -- Surface area of the polyhedron
  surface_area : ℝ
  -- Condition that surface area is 53√3
  area_condition : surface_area = 53 * Real.sqrt 3

/-- Theorem stating the side length of the base triangle -/
theorem pyramid_side_length (p : RegularPyramid) (poly : Polyhedron p) :
  p.a = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_side_length_l1568_156825


namespace NUMINAMATH_CALUDE_season_games_count_l1568_156805

/-- The number of teams in the league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team -/
def games_per_matchup : ℕ := 2

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 5

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_matchup + num_teams * non_league_games

theorem season_games_count : total_games = 192 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l1568_156805


namespace NUMINAMATH_CALUDE_athlete_subgrid_exists_l1568_156818

/-- Represents a grid of athletes -/
def AthleteGrid := Fin 5 → Fin 49 → Bool

/-- Theorem: In any 5x49 grid of athletes, there exists a 3x3 subgrid of the same gender -/
theorem athlete_subgrid_exists (grid : AthleteGrid) :
  ∃ (i j : Fin 3) (r c : Fin 5),
    (∀ x y, x < i → y < j → grid (r + x) (c + y) = grid r c) :=
  sorry

end NUMINAMATH_CALUDE_athlete_subgrid_exists_l1568_156818


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1568_156887

theorem min_value_of_expression (x y z : ℝ) :
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10 ≥ -7/2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀^2 + 3 * x₀ * y₀ + y₀^2 - 3 * x₀ + 3 * y₀ + z₀^2 + 6 * z₀ + 10 = -7/2 ∧
    x₀ = 3/2 ∧ y₀ = -3/2 ∧ z₀ = -3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1568_156887


namespace NUMINAMATH_CALUDE_triathlon_bicycle_speed_l1568_156832

/-- Triathlon problem -/
theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_distance swim_speed : ℝ)
  (run_distance run_speed : ℝ)
  (bike_distance : ℝ) :
  total_time = 2 →
  swim_distance = 0.5 →
  swim_speed = 3 →
  run_distance = 5 →
  run_speed = 10 →
  bike_distance = 20 →
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 15 := by
  sorry


end NUMINAMATH_CALUDE_triathlon_bicycle_speed_l1568_156832


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l1568_156860

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 2 * x + 1 > 0}
def B : Set ℝ := {x : ℝ | |x - 1| < 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_open_interval :
  A_intersect_B = {x : ℝ | -1/2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l1568_156860


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1568_156847

theorem fraction_to_decimal (h : 243 = 3^5) : 7 / 243 = 0.00224 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1568_156847


namespace NUMINAMATH_CALUDE_topsoil_cost_l1568_156822

/-- The cost of topsoil in dollars per cubic foot -/
def topsoil_cost_per_cubic_foot : ℝ := 6

/-- The number of cubic feet in one cubic yard -/
def cubic_feet_per_cubic_yard : ℝ := 27

/-- The number of cubic yards of topsoil -/
def cubic_yards_of_topsoil : ℝ := 5

/-- The theorem stating the cost of the given amount of topsoil -/
theorem topsoil_cost : 
  cubic_yards_of_topsoil * cubic_feet_per_cubic_yard * topsoil_cost_per_cubic_foot = 810 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_l1568_156822


namespace NUMINAMATH_CALUDE_complex_fraction_max_value_l1568_156858

theorem complex_fraction_max_value (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((Complex.I * Real.sqrt 3 - z) / (Real.sqrt 2 - z)) ≤ Real.sqrt 7 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_max_value_l1568_156858


namespace NUMINAMATH_CALUDE_product_inequality_l1568_156861

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1568_156861


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1568_156891

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x / Real.sqrt (1 - x)) + (y / Real.sqrt (1 - y)) ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1568_156891


namespace NUMINAMATH_CALUDE_fraction_simplification_l1568_156812

theorem fraction_simplification (a : ℝ) (h : a ≠ 0) : (a - 1) / a + 1 / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1568_156812


namespace NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg150_l1568_156876

theorem largest_multiple_12_negation_gt_neg150 :
  ∀ n : ℤ, (12 ∣ n) → -n > -150 → n ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_12_negation_gt_neg150_l1568_156876


namespace NUMINAMATH_CALUDE_bob_initial_bushels_bob_extra_ears_l1568_156872

/-- Represents the number of ears of corn in a bushel -/
def ears_per_bushel : ℕ := 14

/-- Represents the number of ears of corn Bob has left after giving some away -/
def ears_left : ℕ := 357

/-- Represents the minimum number of full bushels Bob has left -/
def min_bushels_left : ℕ := ears_left / ears_per_bushel

/-- Theorem stating that Bob initially had at least 25 bushels of corn -/
theorem bob_initial_bushels :
  min_bushels_left ≥ 25 := by
  sorry

/-- Theorem stating that Bob has some extra ears that don't make up a full bushel -/
theorem bob_extra_ears :
  ears_left % ears_per_bushel > 0 := by
  sorry

end NUMINAMATH_CALUDE_bob_initial_bushels_bob_extra_ears_l1568_156872


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l1568_156857

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 12 ∧ b = 12 ∧ c = 17) →  -- Two sides are 12, third side is 17
    (a = b)  →                    -- Isosceles triangle condition
    (a + b + c = 41)              -- Perimeter is 41

/-- The theorem holds for the given triangle. -/
theorem isosceles_triangle_perimeter_holds : isosceles_triangle_perimeter 12 12 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_holds_l1568_156857


namespace NUMINAMATH_CALUDE_average_of_nine_numbers_l1568_156866

theorem average_of_nine_numbers (numbers : Fin 9 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4) / 5 = 99)
  (h2 : (numbers 4 + numbers 5 + numbers 6 + numbers 7 + numbers 8) / 5 = 100)
  (h3 : numbers 4 = 59) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + 
   numbers 5 + numbers 6 + numbers 7 + numbers 8) / 9 = 104 := by
sorry

end NUMINAMATH_CALUDE_average_of_nine_numbers_l1568_156866


namespace NUMINAMATH_CALUDE_abs_func_even_and_increasing_l1568_156898

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_func_even_and_increasing :
  (∀ x : ℝ, abs_func (-x) = abs_func x) ∧
  (∀ x y : ℝ, 0 < x → x < y → abs_func x < abs_func y) :=
by sorry

end NUMINAMATH_CALUDE_abs_func_even_and_increasing_l1568_156898


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l1568_156811

theorem complex_equation_solutions (x y : ℝ) :
  x^2 - y^2 + (2*x*y : ℂ)*I = (2 : ℂ)*I →
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l1568_156811


namespace NUMINAMATH_CALUDE_rebecca_eggs_l1568_156873

theorem rebecca_eggs (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : eggs_per_group = 2) : 
  num_groups * eggs_per_group = 22 := by
sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l1568_156873


namespace NUMINAMATH_CALUDE_mans_age_percentage_l1568_156838

/-- Given a man's age satisfying certain conditions, prove that his present age is 125% of what it was 10 years ago. -/
theorem mans_age_percentage (present_age : ℕ) (future_age : ℕ) (past_age : ℕ) : 
  present_age = 50 ∧ 
  present_age = (5 : ℚ) / 6 * future_age ∧ 
  present_age = past_age + 10 →
  (present_age : ℚ) / past_age = 5 / 4 := by
sorry


end NUMINAMATH_CALUDE_mans_age_percentage_l1568_156838


namespace NUMINAMATH_CALUDE_cube_root_two_not_expressible_l1568_156881

theorem cube_root_two_not_expressible : ¬ ∃ (p q r : ℚ), (2 : ℝ)^(1/3) = p + q * (r^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_two_not_expressible_l1568_156881


namespace NUMINAMATH_CALUDE_divisibility_problem_l1568_156867

theorem divisibility_problem (x q : ℤ) (hx : x > 0) (h_pos : q * x + 197 > 0) 
  (h_197 : 197 % x = 3) : (q * x + 197) % x = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1568_156867


namespace NUMINAMATH_CALUDE_comic_pages_calculation_l1568_156800

theorem comic_pages_calculation (total_pages : ℕ) (extra_pages : ℕ) : 
  total_pages = 220 → 
  extra_pages = 4 → 
  ∃ (first_issue : ℕ), 
    first_issue * 2 + (first_issue + extra_pages) = total_pages ∧ 
    first_issue = 72 := by
  sorry

end NUMINAMATH_CALUDE_comic_pages_calculation_l1568_156800


namespace NUMINAMATH_CALUDE_number_of_factors_19368_l1568_156886

theorem number_of_factors_19368 : Nat.card (Nat.divisors 19368) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_19368_l1568_156886


namespace NUMINAMATH_CALUDE_M_mod_45_l1568_156836

def M : ℕ := sorry

theorem M_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_mod_45_l1568_156836


namespace NUMINAMATH_CALUDE_popsicle_stick_difference_l1568_156855

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks : ℕ := num_boys * sticks_per_boy
  let total_girls_sticks : ℕ := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_difference_l1568_156855


namespace NUMINAMATH_CALUDE_strawberry_fraction_remaining_l1568_156827

theorem strawberry_fraction_remaining 
  (num_hedgehogs : ℕ) 
  (num_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (strawberries_eaten_per_hedgehog : ℕ) : 
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_basket = 900 →
  strawberries_eaten_per_hedgehog = 1050 →
  (num_baskets * strawberries_per_basket - num_hedgehogs * strawberries_eaten_per_hedgehog : ℚ) / 
  (num_baskets * strawberries_per_basket) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_fraction_remaining_l1568_156827


namespace NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_five_l1568_156894

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_def (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -5 is 5 -/
theorem opposite_of_neg_five : opposite (-5 : ℝ) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_def_opposite_of_neg_five_l1568_156894


namespace NUMINAMATH_CALUDE_cookie_recipe_ratio_l1568_156877

-- Define the total amount of sugar needed for the recipe
def total_sugar : ℚ := 3

-- Define the amount of sugar Katie still needs to add
def sugar_to_add : ℚ := 2.5

-- Define the amount of sugar Katie has already added
def sugar_already_added : ℚ := total_sugar - sugar_to_add

-- Define the ratio of sugar already added to total sugar needed
def sugar_ratio : ℚ × ℚ := (sugar_already_added, total_sugar)

-- Theorem to prove
theorem cookie_recipe_ratio :
  sugar_ratio = (1, 6) := by sorry

end NUMINAMATH_CALUDE_cookie_recipe_ratio_l1568_156877


namespace NUMINAMATH_CALUDE_max_areas_circular_disk_l1568_156820

/-- 
Given a circular disk divided by 2n equally spaced radii and two secant lines 
that do not intersect at the same point on the circumference, the maximum number 
of non-overlapping areas into which the disk can be divided is 4n + 4.
-/
theorem max_areas_circular_disk (n : ℕ) : ℕ := by
  sorry

#check max_areas_circular_disk

end NUMINAMATH_CALUDE_max_areas_circular_disk_l1568_156820


namespace NUMINAMATH_CALUDE_quadrilateral_complex_point_l1568_156852

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  z : ℂ

/-- Represents a quadrilateral with vertices A, B, C, D -/
structure Quadrilateral where
  A : ComplexPoint
  B : ComplexPoint
  C : ComplexPoint
  D : ComplexPoint

/-- Theorem: In quadrilateral ABCD, if A, B, and C correspond to given complex numbers,
    then D corresponds to 1+3i -/
theorem quadrilateral_complex_point (q : Quadrilateral)
    (hA : q.A.z = 2 + I)
    (hB : q.B.z = 4 + 3*I)
    (hC : q.C.z = 3 + 5*I) :
    q.D.z = 1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_complex_point_l1568_156852


namespace NUMINAMATH_CALUDE_sum_greater_than_twice_a_l1568_156846

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 + 2 * Real.cos x

def g (x : ℝ) : ℝ := (deriv f) x - 5 * x + 5 * a * Real.log x

theorem sum_greater_than_twice_a (h₁ : x₁ ≠ x₂) (h₂ : g a x₁ = g a x₂) : 
  x₁ + x₂ > 2 * a := by
  sorry

end

end NUMINAMATH_CALUDE_sum_greater_than_twice_a_l1568_156846


namespace NUMINAMATH_CALUDE_quadratic_trinomial_with_integral_roots_l1568_156851

theorem quadratic_trinomial_with_integral_roots :
  ∃ (a b c : ℕ+),
    (∃ (x : ℤ), a * x^2 + b * x + c = 0) ∧
    (∃ (y : ℤ), (a + 1) * y^2 + (b + 1) * y + (c + 1) = 0) ∧
    (∃ (z : ℤ), (a + 2) * z^2 + (b + 2) * z + (c + 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_with_integral_roots_l1568_156851


namespace NUMINAMATH_CALUDE_perpendicular_line_l1568_156830

/-- Given a line l: mx - m²y - 1 = 0, prove that the line perpendicular to l 
    passing through the point (2, 1) has the equation x + y - 3 = 0 -/
theorem perpendicular_line (m : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y => m * x - m^2 * y - 1 = 0
  let p : ℝ × ℝ := (2, 1)
  let perpendicular : ℝ → ℝ → Prop := λ x y => x + y - 3 = 0
  (∀ x y, perpendicular x y ↔ 
    (l x y → False) ∧ 
    (x - p.1) * m + (y - p.2) * (-m^2) = 0 ∧
    perpendicular p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_l1568_156830


namespace NUMINAMATH_CALUDE_area_geometric_mean_l1568_156839

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define a point on a line
def pointOnLine (p1 p2 : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

theorem area_geometric_mean 
  (ABC : Triangle) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (h1 : area ABC = S₁) 
  (O : ℝ × ℝ) 
  (h2 : O = orthocenter ABC) 
  (AOB : Triangle) 
  (h3 : area AOB = S₂) 
  (K : ℝ × ℝ) 
  (h4 : ∃ k, K = pointOnLine O ABC.C k) 
  (ABK : Triangle) 
  (h5 : isRightAngled ABK) : 
  area ABK = Real.sqrt (S₁ * S₂) := 
by sorry

end NUMINAMATH_CALUDE_area_geometric_mean_l1568_156839


namespace NUMINAMATH_CALUDE_catch_in_park_l1568_156896

-- Define the square park
structure Park :=
  (side_length : ℝ)
  (has_diagonal_walkways : Bool)

-- Define the participants
structure Participant :=
  (speed : ℝ)
  (position : ℝ × ℝ)

-- Define the catching condition
def can_catch (pursuer1 pursuer2 target : Participant) (park : Park) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer1.position = target.position ∨ pursuer2.position = target.position)

-- Theorem statement
theorem catch_in_park (park : Park) (pursuer1 pursuer2 target : Participant) :
  park.side_length > 0 ∧
  park.has_diagonal_walkways = true ∧
  pursuer1.speed > 0 ∧
  pursuer2.speed > 0 ∧
  target.speed = 3 * pursuer1.speed ∧
  target.speed = 3 * pursuer2.speed →
  can_catch pursuer1 pursuer2 target park :=
sorry

end NUMINAMATH_CALUDE_catch_in_park_l1568_156896


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1568_156849

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - 1)}

theorem intersection_of_M_and_N : M ∩ N = {x | 1/2 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1568_156849


namespace NUMINAMATH_CALUDE_great_circle_bisects_angle_l1568_156882

/-- A point on a sphere -/
structure SpherePoint where
  -- Add necessary fields

/-- A great circle on a sphere -/
structure GreatCircle where
  -- Add necessary fields

/-- The North Pole -/
def NorthPole : SpherePoint :=
  sorry

/-- The equator -/
def Equator : GreatCircle :=
  sorry

/-- Check if a point is on a great circle -/
def isOnGreatCircle (p : SpherePoint) (gc : GreatCircle) : Prop :=
  sorry

/-- Check if two points are equidistant from a third point -/
def areEquidistant (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Check if a point is on the equator -/
def isOnEquator (p : SpherePoint) : Prop :=
  sorry

/-- The great circle through two points -/
def greatCircleThrough (p1 p2 : SpherePoint) : GreatCircle :=
  sorry

/-- Check if a great circle bisects an angle in a spherical triangle -/
def bisectsAngle (gc : GreatCircle) (p1 p2 p3 : SpherePoint) : Prop :=
  sorry

/-- Main theorem -/
theorem great_circle_bisects_angle (A B C : SpherePoint) :
  isOnGreatCircle A (greatCircleThrough NorthPole B) →
  isOnGreatCircle B (greatCircleThrough NorthPole A) →
  areEquidistant A B NorthPole →
  isOnEquator C →
  bisectsAngle (greatCircleThrough C NorthPole) A C B :=
by
  sorry

end NUMINAMATH_CALUDE_great_circle_bisects_angle_l1568_156882


namespace NUMINAMATH_CALUDE_one_less_than_three_times_l1568_156888

/-- The number that is 1 less than 3 times a real number a can be expressed as 3a - 1. -/
theorem one_less_than_three_times (a : ℝ) : ∃ x : ℝ, x = 3 * a - 1 ∧ x + 1 = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_three_times_l1568_156888


namespace NUMINAMATH_CALUDE_max_product_constraint_l1568_156801

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (hsum : m + n = 4) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → x * y ≤ m * n → m * n = 4 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l1568_156801


namespace NUMINAMATH_CALUDE_vector_subtraction_l1568_156883

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1568_156883


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l1568_156813

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 9 * k = 0) ↔ k = 4 :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l1568_156813


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l1568_156802

/-- Check if a number uses only the digits 1, 2, 3, 4, 5 --/
def usesValidDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- Check if two numbers use all the digits 1, 2, 3, 4, 5 exactly once between them --/
def useAllDigitsOnce (a b : ℕ) : Prop :=
  (a.digits 10 ++ b.digits 10).toFinset = {1, 2, 3, 4, 5}

theorem existence_of_special_numbers : ∃ a b : ℕ,
  10 ≤ a ∧ a < 100 ∧
  100 ≤ b ∧ b < 1000 ∧
  usesValidDigits a ∧
  usesValidDigits b ∧
  useAllDigitsOnce a b ∧
  b % a = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l1568_156802


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1568_156874

/-- 
If the intersection point of the lines y = 2x + 4 and y = -2x + m 
is in the second quadrant, then -4 < m < 4.
-/
theorem intersection_in_second_quadrant (m : ℝ) : 
  (∃ x y : ℝ, y = 2*x + 4 ∧ y = -2*x + m ∧ x < 0 ∧ y > 0) → 
  -4 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1568_156874


namespace NUMINAMATH_CALUDE_total_retail_price_proof_l1568_156828

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) : ℝ :=
  wholesale_price * (1 + profit_margin)

theorem total_retail_price_proof 
  (P Q R : ℝ)
  (discount1 discount2 discount3 : ℝ)
  (profit_margin1 profit_margin2 profit_margin3 : ℝ)
  (h1 : P = 90)
  (h2 : Q = 120)
  (h3 : R = 150)
  (h4 : discount1 = 0.10)
  (h5 : discount2 = 0.15)
  (h6 : discount3 = 0.20)
  (h7 : profit_margin1 = 0.20)
  (h8 : profit_margin2 = 0.25)
  (h9 : profit_margin3 = 0.30) :
  calculate_retail_price P profit_margin1 +
  calculate_retail_price Q profit_margin2 +
  calculate_retail_price R profit_margin3 = 453 := by
sorry

end NUMINAMATH_CALUDE_total_retail_price_proof_l1568_156828


namespace NUMINAMATH_CALUDE_star_operation_result_l1568_156899

-- Define the sets A and B
def A : Set ℝ := {y | y ≥ 0}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define the set difference operation
def set_difference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define the * operation
def star_operation (X Y : Set ℝ) : Set ℝ := 
  (set_difference X Y) ∪ (set_difference Y X)

-- Theorem statement
theorem star_operation_result : 
  star_operation A B = {x | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l1568_156899


namespace NUMINAMATH_CALUDE_percentage_with_both_colors_l1568_156829

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  totalFlags : ℕ
  bluePercentage : ℚ
  redPercentage : ℚ
  bothPercentage : ℚ

/-- Theorem stating the percentage of children with both color flags -/
theorem percentage_with_both_colors (fd : FlagDistribution) :
  fd.totalFlags % 2 = 0 ∧
  fd.bluePercentage = 60 / 100 ∧
  fd.redPercentage = 45 / 100 ∧
  fd.bluePercentage + fd.redPercentage > 1 →
  fd.bothPercentage = 5 / 100 := by
  sorry

#check percentage_with_both_colors

end NUMINAMATH_CALUDE_percentage_with_both_colors_l1568_156829


namespace NUMINAMATH_CALUDE_expand_expression_l1568_156878

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1568_156878


namespace NUMINAMATH_CALUDE_transformed_point_difference_l1568_156821

def rotate90CounterClockwise (x y xc yc : ℝ) : ℝ × ℝ :=
  (xc - (y - yc), yc + (x - xc))

def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformed_point_difference (a b : ℝ) :
  let (x1, y1) := rotate90CounterClockwise a b 2 3
  let (x2, y2) := reflectAboutYEqualsX x1 y1
  (x2 = 4 ∧ y2 = 1) → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_transformed_point_difference_l1568_156821


namespace NUMINAMATH_CALUDE_race_time_proof_l1568_156897

/-- Represents the race times of two runners -/
structure RaceTimes where
  total : ℕ
  difference : ℕ

/-- Calculates the longer race time given the total time and the difference between runners -/
def longerTime (times : RaceTimes) : ℕ :=
  (times.total + times.difference) / 2

theorem race_time_proof (times : RaceTimes) (h1 : times.total = 112) (h2 : times.difference = 4) :
  longerTime times = 58 := by
  sorry

end NUMINAMATH_CALUDE_race_time_proof_l1568_156897


namespace NUMINAMATH_CALUDE_equation_solution_l1568_156816

theorem equation_solution : 
  ∃ y : ℝ, 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) ∧ y = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1568_156816


namespace NUMINAMATH_CALUDE_trapezoid_median_l1568_156835

/-- Given a triangle and a trapezoid with equal areas and the same altitude,
    where the base of the triangle is 24 inches and the sum of the bases of
    the trapezoid is 40 inches, the median of the trapezoid is 20 inches. -/
theorem trapezoid_median (h : ℝ) (triangle_area trapezoid_area : ℝ) 
  (triangle_base trapezoid_base_sum : ℝ) (trapezoid_median : ℝ) :
  h > 0 →
  triangle_area = trapezoid_area →
  triangle_base = 24 →
  trapezoid_base_sum = 40 →
  triangle_area = (1 / 2) * triangle_base * h →
  trapezoid_area = trapezoid_median * h →
  trapezoid_median = trapezoid_base_sum / 2 →
  trapezoid_median = 20 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_median_l1568_156835


namespace NUMINAMATH_CALUDE_largest_common_term_l1568_156806

theorem largest_common_term (n m : ℕ) : 
  (147 = 2 + 5 * n) ∧ 
  (147 = 3 + 8 * m) ∧ 
  (147 ≤ 150) ∧ 
  (∀ k : ℕ, k > 147 → k ≤ 150 → (k - 2) % 5 ≠ 0 ∨ (k - 3) % 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1568_156806


namespace NUMINAMATH_CALUDE_student_number_choice_l1568_156889

theorem student_number_choice (x : ℝ) : 2 * x - 138 = 104 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_choice_l1568_156889


namespace NUMINAMATH_CALUDE_sum_of_ages_l1568_156884

theorem sum_of_ages (petra_age mother_age : ℕ) : 
  petra_age = 11 → 
  mother_age = 36 → 
  mother_age = 2 * petra_age + 14 → 
  petra_age + mother_age = 47 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1568_156884


namespace NUMINAMATH_CALUDE_fifth_page_stickers_l1568_156879

def sticker_sequence (n : ℕ) : ℕ := 8 * n

theorem fifth_page_stickers : sticker_sequence 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_page_stickers_l1568_156879


namespace NUMINAMATH_CALUDE_cheryl_different_colors_probability_l1568_156815

/-- Represents the number of marbles of each color in the box -/
def initial_marbles : Nat := 2

/-- Represents the total number of colors -/
def total_colors : Nat := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : Nat := initial_marbles * total_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : Nat := 2

/-- Calculates the probability of Cheryl not getting two marbles of the same color -/
theorem cheryl_different_colors_probability :
  let total_outcomes := (total_marbles.choose marbles_drawn) * 
                        ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                        ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  let favorable_outcomes := (total_colors.choose marbles_drawn) * 
                            ((total_marbles - marbles_drawn).choose marbles_drawn) * 
                            ((total_marbles - 2*marbles_drawn).choose marbles_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

#eval initial_marbles -- 2
#eval total_colors -- 4
#eval total_marbles -- 8
#eval marbles_drawn -- 2

end NUMINAMATH_CALUDE_cheryl_different_colors_probability_l1568_156815


namespace NUMINAMATH_CALUDE_river_road_ratio_l1568_156831

def river_road_vehicles (cars : ℕ) (bus_difference : ℕ) : ℕ × ℕ :=
  (cars - bus_difference, cars)

def simplify_ratio (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem river_road_ratio :
  let (buses, cars) := river_road_vehicles 100 90
  simplify_ratio buses cars = (1, 10) := by
sorry

end NUMINAMATH_CALUDE_river_road_ratio_l1568_156831


namespace NUMINAMATH_CALUDE_complex_magnitude_l1568_156843

theorem complex_magnitude (w : ℂ) (h : w^2 + 2*w = 11 - 16*I) : 
  Complex.abs w = 17 ∨ Complex.abs w = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1568_156843


namespace NUMINAMATH_CALUDE_pages_read_day5_l1568_156863

def pages_day1 : ℕ := 63
def pages_day2 : ℕ := 95 -- Rounded up from 94.5
def pages_day3 : ℕ := pages_day2 + 20
def pages_day4 : ℕ := 86 -- Rounded down from 86.25
def total_pages : ℕ := 480

theorem pages_read_day5 : 
  total_pages - (pages_day1 + pages_day2 + pages_day3 + pages_day4) = 121 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_day5_l1568_156863


namespace NUMINAMATH_CALUDE_julie_school_year_hours_l1568_156819

/-- Calculates the required weekly hours for Julie to earn a target amount during the school year,
    given her summer work details and school year duration. -/
theorem julie_school_year_hours
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (summer_earnings : ℚ)
  (school_year_weeks : ℕ)
  (school_year_target : ℚ)
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 7500)
  (h4 : school_year_weeks = 50)
  (h5 : school_year_target = 7500) :
  (school_year_target / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_year_weeks = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_hours_l1568_156819


namespace NUMINAMATH_CALUDE_student_preferences_l1568_156834

/-- In a class of 30 students, prove that the sum of students who like maths and history is 15,
    given the distribution of student preferences. -/
theorem student_preferences (total : ℕ) (maths_ratio science_ratio history_ratio : ℚ) : 
  total = 30 ∧ 
  maths_ratio = 3/10 ∧ 
  science_ratio = 1/4 ∧ 
  history_ratio = 2/5 → 
  ∃ (maths science history literature : ℕ),
    maths = ⌊maths_ratio * total⌋ ∧
    science = ⌊science_ratio * (total - maths)⌋ ∧
    history = ⌊history_ratio * (total - maths - science)⌋ ∧
    literature = total - maths - science - history ∧
    maths + history = 15 :=
by sorry


end NUMINAMATH_CALUDE_student_preferences_l1568_156834


namespace NUMINAMATH_CALUDE_mean_temperature_is_negative_point_six_l1568_156850

def temperatures : List ℝ := [-8, -5, -5, -2, 0, 4, 5, 3, 6, 1]

theorem mean_temperature_is_negative_point_six :
  (temperatures.sum / temperatures.length : ℝ) = -0.6 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_negative_point_six_l1568_156850


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1568_156840

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - I) / (2 + 3*I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1568_156840


namespace NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l1568_156810

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ (∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_equation_smallest_solution_l1568_156810


namespace NUMINAMATH_CALUDE_complex_multiplication_l1568_156842

theorem complex_multiplication (i : ℂ) : i * i = -1 →
  (1/2 : ℂ) + (Real.sqrt 3/2 : ℂ) * i * ((Real.sqrt 3/2 : ℂ) + (1/2 : ℂ) * i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1568_156842


namespace NUMINAMATH_CALUDE_expression_value_l1568_156804

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1568_156804


namespace NUMINAMATH_CALUDE_theater_revenue_l1568_156869

theorem theater_revenue (n : ℕ) (C : ℝ) :
  (∃ R : ℝ, R = 1.20 * C) →
  (∃ R_95 : ℝ, R_95 = 0.95 * 1.20 * C ∧ R_95 = 1.14 * C) :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l1568_156869


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1568_156868

/-- The distance from the focus of the parabola x^2 = 8y to the asymptotes of the hyperbola x^2 - y^2/9 = 1 is √10 / 5 -/
theorem distance_focus_to_asymptotes :
  let parabola := {p : ℝ × ℝ | p.1^2 = 8 * p.2}
  let hyperbola := {p : ℝ × ℝ | p.1^2 - p.2^2 / 9 = 1}
  let focus : ℝ × ℝ := (0, 2)
  let asymptote (x : ℝ) := {p : ℝ × ℝ | p.2 = 3 * p.1 ∨ p.2 = -3 * p.1}
  let distance (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) := 
    Real.sqrt (10) / 5
  ∀ p ∈ parabola, p.1^2 = 8 * p.2 →
  ∀ h ∈ hyperbola, h.1^2 - h.2^2 / 9 = 1 →
  distance focus (asymptote 0) = Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_focus_to_asymptotes_l1568_156868


namespace NUMINAMATH_CALUDE_min_diff_y_x_l1568_156890

theorem min_diff_y_x (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : Even x)
  (h3 : Odd y ∧ Odd z)
  (h4 : ∀ w, (w : ℤ) ≥ x ∧ Odd w → w - x ≥ 9) :
  ∃ (d : ℤ), d = y - x ∧ ∀ (d' : ℤ), y - x ≤ d' := by
  sorry

end NUMINAMATH_CALUDE_min_diff_y_x_l1568_156890


namespace NUMINAMATH_CALUDE_right_triangle_other_leg_l1568_156837

theorem right_triangle_other_leg 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- right triangle condition
  (h_a : a = 9)                -- one leg is 9 cm
  (h_c : c = 15)               -- hypotenuse is 15 cm
  : b = 12 := by               -- prove other leg is 12 cm
  sorry

end NUMINAMATH_CALUDE_right_triangle_other_leg_l1568_156837


namespace NUMINAMATH_CALUDE_opposite_number_theorem_l1568_156870

theorem opposite_number_theorem (m : ℤ) : (m + 1 = -(-4)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_theorem_l1568_156870


namespace NUMINAMATH_CALUDE_heights_academy_music_problem_l1568_156875

/-- The Heights Academy music problem -/
theorem heights_academy_music_problem
  (total_students : ℕ)
  (females_band : ℕ)
  (males_band : ℕ)
  (females_orchestra : ℕ)
  (males_orchestra : ℕ)
  (females_both : ℕ)
  (h1 : total_students = 260)
  (h2 : females_band = 120)
  (h3 : males_band = 90)
  (h4 : females_orchestra = 100)
  (h5 : males_orchestra = 130)
  (h6 : females_both = 80) :
  males_band - (males_band + males_orchestra - (total_students - (females_band + females_orchestra - females_both))) = 30 := by
  sorry


end NUMINAMATH_CALUDE_heights_academy_music_problem_l1568_156875


namespace NUMINAMATH_CALUDE_square_nailing_theorem_l1568_156880

/-- Represents a paper square on the table -/
structure Square where
  color : Nat
  position : Real × Real

/-- Represents the arrangement of squares on the table -/
def Arrangement := List Square

/-- Checks if two squares can be nailed with one nail -/
def can_nail_together (s1 s2 : Square) : Prop := sorry

/-- The main theorem to be proved -/
theorem square_nailing_theorem (k : Nat) (arrangement : Arrangement) :
  (∀ (distinct_squares : List Square),
    distinct_squares.length = k →
    distinct_squares.Pairwise (λ s1 s2 => s1.color ≠ s2.color) →
    distinct_squares.Sublist arrangement →
    ∃ (s1 s2 : Square), s1 ∈ distinct_squares ∧ s2 ∈ distinct_squares ∧ can_nail_together s1 s2) →
  ∃ (color : Nat),
    let squares_of_color := arrangement.filter (λ s => s.color = color)
    ∃ (nails : List (Real × Real)), nails.length ≤ 2 * k - 2 ∧
      ∀ (s : Square), s ∈ squares_of_color →
        ∃ (nail : Real × Real), nail ∈ nails ∧ s.position = nail :=
sorry

end NUMINAMATH_CALUDE_square_nailing_theorem_l1568_156880


namespace NUMINAMATH_CALUDE_complement_N_Nstar_is_finite_l1568_156885

def complement_N_Nstar : Set ℕ := {0}

theorem complement_N_Nstar_is_finite :
  Set.Finite complement_N_Nstar :=
sorry

end NUMINAMATH_CALUDE_complement_N_Nstar_is_finite_l1568_156885


namespace NUMINAMATH_CALUDE_complex_subtraction_reciprocal_l1568_156823

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_subtraction_reciprocal : i - (1 : ℂ) / i = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_reciprocal_l1568_156823


namespace NUMINAMATH_CALUDE_sandwich_composition_ham_cost_is_correct_l1568_156864

/-- The cost of a slice of ham in a sandwich -/
def ham_cost : ℚ := 25 / 100

/-- The selling price of a sandwich -/
def sandwich_price : ℚ := 150 / 100

/-- The cost of a slice of bread -/
def bread_cost : ℚ := 15 / 100

/-- The cost of a slice of cheese -/
def cheese_cost : ℚ := 35 / 100

/-- The total cost to make a sandwich -/
def sandwich_cost : ℚ := 90 / 100

/-- A sandwich contains 2 slices of bread, 1 slice of ham, and 1 slice of cheese -/
theorem sandwich_composition (h : ℚ) :
  sandwich_cost = 2 * bread_cost + h + cheese_cost :=
sorry

/-- The cost of a slice of ham is $0.25 -/
theorem ham_cost_is_correct :
  ham_cost = sandwich_cost - 2 * bread_cost - cheese_cost :=
sorry

end NUMINAMATH_CALUDE_sandwich_composition_ham_cost_is_correct_l1568_156864


namespace NUMINAMATH_CALUDE_quadrant_passing_implies_negative_m_l1568_156856

/-- A linear function passing through the second, third, and fourth quadrants -/
structure QuadrantPassingFunction where
  m : ℝ
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = -3 * x + m
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = -3 * x + m
  passes_fourth_quadrant : ∃ x y, x > 0 ∧ y < 0 ∧ y = -3 * x + m

/-- Theorem: If a linear function y = -3x + m passes through the second, third, and fourth quadrants, then m is negative -/
theorem quadrant_passing_implies_negative_m (f : QuadrantPassingFunction) : f.m < 0 :=
  sorry

end NUMINAMATH_CALUDE_quadrant_passing_implies_negative_m_l1568_156856
