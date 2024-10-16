import Mathlib

namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3780_378085

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (fun x => -1/2 * (2 - 3*x) * Real.cos (2*x) - 3/4 * Real.sin (2*x))) x 
  = (2 - 3*x) * Real.sin (2*x) := by
sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3780_378085


namespace NUMINAMATH_CALUDE_triangle_area_l3780_378068

/-- Given a triangle ABC with sides a, b, c corresponding to angles A, B, C,
    where a = √2, c = √6, and C = 2π/3, prove that its area S is √3/2 -/
theorem triangle_area (a b c A B C S : Real) : 
  a = Real.sqrt 2 →
  c = Real.sqrt 6 →
  C = 2 * Real.pi / 3 →
  S = (1 / 2) * a * c * Real.sin B →
  S = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3780_378068


namespace NUMINAMATH_CALUDE_top_quality_soccer_balls_l3780_378083

/-- Given a batch of soccer balls, calculate the number of top-quality balls -/
theorem top_quality_soccer_balls 
  (total : ℕ) 
  (frequency : ℝ) 
  (h_total : total = 10000)
  (h_frequency : frequency = 0.975) :
  ⌊(total : ℝ) * frequency⌋ = 9750 := by
  sorry

end NUMINAMATH_CALUDE_top_quality_soccer_balls_l3780_378083


namespace NUMINAMATH_CALUDE_factor_calculation_l3780_378069

theorem factor_calculation (f : ℚ) : f * (2 * 16 + 5) = 111 ↔ f = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l3780_378069


namespace NUMINAMATH_CALUDE_stormi_lawn_mowing_price_l3780_378045

/-- Stormi's lawn mowing price calculation -/
theorem stormi_lawn_mowing_price 
  (cars_washed : ℕ) 
  (car_wash_price : ℚ) 
  (lawns_mowed : ℕ) 
  (bicycle_price : ℚ) 
  (additional_money_needed : ℚ)
  (h1 : cars_washed = 3)
  (h2 : car_wash_price = 10)
  (h3 : lawns_mowed = 2)
  (h4 : bicycle_price = 80)
  (h5 : additional_money_needed = 24) :
  (bicycle_price - additional_money_needed - cars_washed * car_wash_price) / lawns_mowed = 13 := by
  sorry

end NUMINAMATH_CALUDE_stormi_lawn_mowing_price_l3780_378045


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_target_l3780_378075

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := Set.Ioc 1 (3/2)

-- Theorem statement
theorem M_intersect_N_equals_target : M_intersect_N = target_set := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_target_l3780_378075


namespace NUMINAMATH_CALUDE_computer_printer_price_l3780_378027

/-- The total price of a basic computer and printer -/
def total_price (basic_computer_price printer_price : ℝ) : ℝ :=
  basic_computer_price + printer_price

/-- The price of an enhanced computer -/
def enhanced_computer_price (basic_computer_price : ℝ) : ℝ :=
  basic_computer_price + 500

/-- Condition for printer price with enhanced computer -/
def printer_price_condition (basic_computer_price printer_price : ℝ) : Prop :=
  printer_price = (1/3) * (enhanced_computer_price basic_computer_price + printer_price)

theorem computer_printer_price :
  ∃ (printer_price : ℝ),
    let basic_computer_price := 1500
    printer_price_condition basic_computer_price printer_price ∧
    total_price basic_computer_price printer_price = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_printer_price_l3780_378027


namespace NUMINAMATH_CALUDE_disjoint_subset_pairs_mod_1000_l3780_378000

def S : Finset Nat := Finset.range 10

def disjointSubsetPairs (X : Finset Nat) : Nat :=
  (3^X.card - 2 * 2^X.card + 1) / 2

theorem disjoint_subset_pairs_mod_1000 :
  disjointSubsetPairs S % 1000 = 501 := by sorry

end NUMINAMATH_CALUDE_disjoint_subset_pairs_mod_1000_l3780_378000


namespace NUMINAMATH_CALUDE_car_storm_distance_time_l3780_378086

/-- The time when a car traveling north at 3/4 mile per minute is 30 miles away from the center of a storm
    moving southeast at 3/4√2 mile per minute, given that at t=0 the storm's center is 150 miles due east of the car. -/
theorem car_storm_distance_time : ∃ t : ℝ,
  (27 / 32 : ℝ) * t^2 - (450 * Real.sqrt 2 / 2) * t + 21600 = 0 :=
by sorry

end NUMINAMATH_CALUDE_car_storm_distance_time_l3780_378086


namespace NUMINAMATH_CALUDE_impossible_tiling_l3780_378096

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile that can be placed on the board -/
inductive Tile
  | Domino    : Tile  -- 1 × 2 horizontal domino
  | Rectangle : Tile  -- 1 × 3 vertical rectangle

/-- Represents a tiling of the board -/
def Tiling := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Check if a tiling is valid for the given board -/
def is_valid_tiling (board : Board) (tiling : Tiling) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to tile the 2003 × 2003 board -/
theorem impossible_tiling :
  ∀ (tiling : Tiling), ¬(is_valid_tiling (Board.mk 2003 2003) tiling) :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l3780_378096


namespace NUMINAMATH_CALUDE_distribute_a_over_sum_l3780_378009

theorem distribute_a_over_sum (a b c : ℝ) : a * (a + b - c) = a^2 + a*b - a*c := by sorry

end NUMINAMATH_CALUDE_distribute_a_over_sum_l3780_378009


namespace NUMINAMATH_CALUDE_log_inequality_l3780_378032

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem log_inequality (x : ℝ) (h : f (Real.log x) < 0) : 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3780_378032


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3780_378061

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 0 else 2 / (2 - x)

theorem unique_function_satisfying_conditions :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) ∧
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x : ℝ, x ≥ 0 → g x ≥ 0) ∧
     (g 2 = 0) ∧
     (∀ x : ℝ, 0 ≤ x ∧ x < 2 → g x ≠ 0) ∧
     (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → g (x * g y) * g y = g (x + y))) →
    (∀ x : ℝ, x ≥ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3780_378061


namespace NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l3780_378019

-- Define the functions
def f (x : ℝ) := x * |x|
def g (x : ℝ) := x^3

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_odd_and_increasing_l3780_378019


namespace NUMINAMATH_CALUDE_weight_change_result_l3780_378098

/-- Calculate the final weight after weight loss and gain -/
def final_weight (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : ℝ :=
  initial_weight - (initial_weight * loss_percentage) + weight_gain

/-- Theorem stating that the given weight changes result in a final weight of 200 pounds -/
theorem weight_change_result : 
  final_weight 220 0.1 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_weight_change_result_l3780_378098


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3780_378066

/-- If the complex number lg(m^2-2m-2) + (m^2+3m+2)i is purely imaginary and m is real, then m = 3 -/
theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).im ≠ 0 ∧ 
  (Complex.log (m^2 - 2*m - 2) + Complex.I * (m^2 + 3*m + 2)).re = 0 → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3780_378066


namespace NUMINAMATH_CALUDE_square_fraction_count_l3780_378010

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    S.card = 2 ∧ 
    (∀ n ∈ S, 0 ≤ n ∧ n < 25 ∧ ∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) ∧
    (∀ n : ℤ, 0 ≤ n → n < 25 → (∃ k : ℤ, (n : ℚ) / (25 - n) = k^2) → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3780_378010


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3780_378004

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 3 * π / 2
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) := by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l3780_378004


namespace NUMINAMATH_CALUDE_a_a_a_zero_l3780_378055

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_a_a_zero : a (a (a 0)) = 343 := by sorry

end NUMINAMATH_CALUDE_a_a_a_zero_l3780_378055


namespace NUMINAMATH_CALUDE_kitten_food_consumption_l3780_378050

/-- Proves that given the conditions, each kitten eats 0.75 cans of food per day -/
theorem kitten_food_consumption
  (num_kittens : ℕ)
  (num_adult_cats : ℕ)
  (initial_food : ℕ)
  (additional_food : ℕ)
  (days : ℕ)
  (adult_cat_consumption : ℚ)
  (h1 : num_kittens = 4)
  (h2 : num_adult_cats = 3)
  (h3 : initial_food = 7)
  (h4 : additional_food = 35)
  (h5 : days = 7)
  (h6 : adult_cat_consumption = 1)
  : (initial_food + additional_food - num_adult_cats * adult_cat_consumption * days) / (num_kittens * days) = 0.75 := by
  sorry


end NUMINAMATH_CALUDE_kitten_food_consumption_l3780_378050


namespace NUMINAMATH_CALUDE_real_part_of_z_l3780_378080

theorem real_part_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : 
  z.re = 3 / 25 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3780_378080


namespace NUMINAMATH_CALUDE_one_in_M_l3780_378044

def M : Set ℕ := {0, 1, 2}

theorem one_in_M : 1 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_in_M_l3780_378044


namespace NUMINAMATH_CALUDE_optimal_speed_theorem_l3780_378071

theorem optimal_speed_theorem (d t : ℝ) 
  (h1 : d = 45 * (t + 1/15))
  (h2 : d = 75 * (t - 1/15)) :
  d / t = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_speed_theorem_l3780_378071


namespace NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l3780_378041

/-- 
Given a quadratic function f(x) = 3x^2 + 2x + 5, when shifted 7 units to the right,
it results in a new quadratic function g(x) = ax^2 + bx + c.
This theorem proves that the sum of the coefficients a + b + c equals 101.
-/
theorem shifted_quadratic_coefficient_sum :
  ∀ (a b c : ℝ),
  (∀ x, (3 * (x - 7)^2 + 2 * (x - 7) + 5) = (a * x^2 + b * x + c)) →
  a + b + c = 101 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_coefficient_sum_l3780_378041


namespace NUMINAMATH_CALUDE_problem_solution_l3780_378092

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_solution (x y : ℝ) :
  (∀ x, (f x)^2 - (g x)^2 = -4) ∧
  (f x * f y = 4 ∧ g x * g y = 8 → g (x + y) / g (x - y) = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3780_378092


namespace NUMINAMATH_CALUDE_intersection_S_T_l3780_378008

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_S_T : S ∩ T = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3780_378008


namespace NUMINAMATH_CALUDE_log_equation_solution_l3780_378017

theorem log_equation_solution (p q : ℝ) (h1 : p > q) (h2 : q > 0) :
  Real.log p + Real.log q = Real.log (p - q) ↔ p = q / (1 - q) ∧ q < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3780_378017


namespace NUMINAMATH_CALUDE_harvest_time_calculation_l3780_378091

theorem harvest_time_calculation (initial_harvesters initial_days initial_area final_harvesters final_area : ℕ) 
  (h1 : initial_harvesters = 2)
  (h2 : initial_days = 3)
  (h3 : initial_area = 450)
  (h4 : final_harvesters = 7)
  (h5 : final_area = 2100) :
  (initial_harvesters * initial_days * final_area) / (initial_area * final_harvesters) = 4 := by
  sorry

end NUMINAMATH_CALUDE_harvest_time_calculation_l3780_378091


namespace NUMINAMATH_CALUDE_repeating_decimal_equality_l3780_378060

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℚ
  repeatingPart : ℚ
  repeatingPartLength : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def RepeatingDecimal.toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + x.repeatingPart / (1 - (1 / 10 ^ x.repeatingPartLength))

/-- Theorem stating that 0.3̅206̅ is equal to 5057/9990 -/
theorem repeating_decimal_equality : 
  let x : RepeatingDecimal := ⟨3/10, 206/1000, 3⟩
  x.toRational = 5057 / 9990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equality_l3780_378060


namespace NUMINAMATH_CALUDE_shaded_area_of_semicircle_l3780_378038

theorem shaded_area_of_semicircle (total_area : ℝ) (h : total_area > 0) :
  let num_parts : ℕ := 6
  let excluded_fraction : ℝ := 2 / 3
  let shaded_area : ℝ := total_area * (1 - excluded_fraction)
  shaded_area = total_area / 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_semicircle_l3780_378038


namespace NUMINAMATH_CALUDE_range_f_is_closed_interval_l3780_378093

/-- The quadratic function f(x) = -x^2 + 4x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x + 1

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

/-- The range of f over the interval I -/
def range_f : Set ℝ := { y | ∃ x ∈ I, f x = y }

theorem range_f_is_closed_interval :
  range_f = { y | 1 ≤ y ∧ y ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_range_f_is_closed_interval_l3780_378093


namespace NUMINAMATH_CALUDE_square_operation_l3780_378016

theorem square_operation (x y : ℝ) (h1 : y = 68.70953354520753) (h2 : y^2 - x^2 = 4321) :
  ∃ (z : ℝ), z^2 = x^2 ∧ z = x :=
sorry

end NUMINAMATH_CALUDE_square_operation_l3780_378016


namespace NUMINAMATH_CALUDE_acme_savings_threshold_l3780_378077

/-- Acme T-Shirt Plus Company's pricing structure -/
def acme_cost (x : ℕ) : ℚ := 75 + 8 * x

/-- Gamma T-shirt Company's pricing structure -/
def gamma_cost (x : ℕ) : ℚ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_savings : ℕ := 19

theorem acme_savings_threshold :
  (∀ x : ℕ, x ≥ min_shirts_for_acme_savings → acme_cost x < gamma_cost x) ∧
  (∀ x : ℕ, x < min_shirts_for_acme_savings → acme_cost x ≥ gamma_cost x) :=
sorry

end NUMINAMATH_CALUDE_acme_savings_threshold_l3780_378077


namespace NUMINAMATH_CALUDE_diamond_ratio_l3780_378067

def diamond (n m : ℝ) : ℝ := n^4 * m^3

theorem diamond_ratio : (diamond 3 2) / (diamond 2 3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_diamond_ratio_l3780_378067


namespace NUMINAMATH_CALUDE_reflect_F_l3780_378025

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Theorem: Reflecting point F(3, 3) over y-axis then x-axis results in F''(-3, -3) -/
theorem reflect_F : 
  let F : ℝ × ℝ := (3, 3)
  reflect_x (reflect_y F) = (-3, -3) := by
sorry

end NUMINAMATH_CALUDE_reflect_F_l3780_378025


namespace NUMINAMATH_CALUDE_expression_evaluation_l3780_378051

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x = z / y) :
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3780_378051


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3780_378014

theorem min_value_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x ≥ 2 * Real.sqrt 3 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x^2 + y^2 + 4/x^2 + 2*y/x = 2 * Real.sqrt 3 ↔ 
  (x = Real.sqrt (Real.sqrt 3) ∨ x = -Real.sqrt (Real.sqrt 3)) ∧
  (y = -1 / x) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3780_378014


namespace NUMINAMATH_CALUDE_blue_balls_count_l3780_378031

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5 + blue_balls + 4

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 5 / total_balls * 4 / (total_balls - 1)

theorem blue_balls_count : 
  (5 : ℕ) > 0 ∧ 
  (4 : ℕ) > 0 ∧ 
  prob_two_red = 0.09523809523809523 →
  blue_balls = 6 := by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l3780_378031


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3780_378064

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * Real.pi - Real.pi / 4 < α ∧ α < k * Real.pi

theorem half_angle_quadrant (α : Real) :
  is_in_fourth_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3780_378064


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l3780_378076

/-- The probability of selecting at least one female student when choosing two students from a group of three male and two female students is 7/10. -/
theorem prob_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (select : ℕ) :
  total = male + female →
  total = 5 →
  male = 3 →
  female = 2 →
  select = 2 →
  (1 : ℚ) - (Nat.choose male select : ℚ) / (Nat.choose total select : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l3780_378076


namespace NUMINAMATH_CALUDE_milk_mixture_water_content_l3780_378049

theorem milk_mixture_water_content 
  (initial_water_percentage : ℝ)
  (initial_milk_volume : ℝ)
  (pure_milk_volume : ℝ)
  (h1 : initial_water_percentage = 5)
  (h2 : initial_milk_volume = 10)
  (h3 : pure_milk_volume = 15) :
  let total_water := initial_water_percentage / 100 * initial_milk_volume
  let total_volume := initial_milk_volume + pure_milk_volume
  let final_water_percentage := total_water / total_volume * 100
  final_water_percentage = 2 := by
sorry

end NUMINAMATH_CALUDE_milk_mixture_water_content_l3780_378049


namespace NUMINAMATH_CALUDE_johns_allowance_problem_l3780_378030

/-- The problem of calculating the fraction of John's remaining allowance spent at the toy store -/
theorem johns_allowance_problem (allowance : ℚ) :
  allowance = 345/100 →
  let arcade_spent := (3/5) * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_spent := 92/100
  let toy_spent := remaining_after_arcade - candy_spent
  (toy_spent / remaining_after_arcade) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_problem_l3780_378030


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3780_378006

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3780_378006


namespace NUMINAMATH_CALUDE_N_equals_one_l3780_378070

theorem N_equals_one :
  let N := (Real.sqrt (Real.sqrt 5 + 2) + Real.sqrt (Real.sqrt 5 - 2)) / Real.sqrt (Real.sqrt 5 + 1) - Real.sqrt (3 - 2 * Real.sqrt 2)
  N = 1 := by sorry

end NUMINAMATH_CALUDE_N_equals_one_l3780_378070


namespace NUMINAMATH_CALUDE_correct_answers_range_l3780_378054

/-- Represents the scoring system and conditions of the test --/
structure TestScoring where
  total_questions : Nat
  correct_points : Int
  wrong_points : Int
  min_score : Int

/-- Represents Xiaoyu's test result --/
structure TestResult (scoring : TestScoring) where
  correct_answers : Nat
  no_missed_questions : correct_answers ≤ scoring.total_questions

/-- Calculates the total score based on the number of correct answers --/
def calculate_score (scoring : TestScoring) (result : TestResult scoring) : Int :=
  result.correct_answers * scoring.correct_points + 
  (scoring.total_questions - result.correct_answers) * scoring.wrong_points

/-- Theorem stating the range of possible values for correct answers --/
theorem correct_answers_range (scoring : TestScoring) 
  (h_total : scoring.total_questions = 25)
  (h_correct : scoring.correct_points = 4)
  (h_wrong : scoring.wrong_points = -2)
  (h_min_score : scoring.min_score = 70) :
  ∀ (result : TestResult scoring), 
    calculate_score scoring result ≥ scoring.min_score →
    (20 : Nat) ≤ result.correct_answers ∧ result.correct_answers ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_range_l3780_378054


namespace NUMINAMATH_CALUDE_max_cups_in_kitchen_l3780_378029

theorem max_cups_in_kitchen (a b : ℕ) : 
  (a.choose 2) * (b.choose 3) = 1200 → a + b ≤ 29 :=
by sorry

end NUMINAMATH_CALUDE_max_cups_in_kitchen_l3780_378029


namespace NUMINAMATH_CALUDE_clarinet_tryouts_l3780_378020

theorem clarinet_tryouts 
  (total_band : ℕ)
  (flutes_tried : ℕ)
  (flutes_ratio : ℚ)
  (trumpets_tried : ℕ)
  (trumpets_ratio : ℚ)
  (pianists_tried : ℕ)
  (pianists_ratio : ℚ)
  (clarinets_ratio : ℚ)
  (h1 : total_band = 53)
  (h2 : flutes_tried = 20)
  (h3 : flutes_ratio = 4/5)
  (h4 : trumpets_tried = 60)
  (h5 : trumpets_ratio = 1/3)
  (h6 : pianists_tried = 20)
  (h7 : pianists_ratio = 1/10)
  (h8 : clarinets_ratio = 1/2)
  : ℕ := by
  sorry

#check clarinet_tryouts

end NUMINAMATH_CALUDE_clarinet_tryouts_l3780_378020


namespace NUMINAMATH_CALUDE_problem_solution_l3780_378058

def f (n : ℤ) : ℤ := 3 * n^6 + 26 * n^4 + 33 * n^2 + 1

def valid_k (k : ℕ) : Prop :=
  k ≤ 100 ∧ ∃ n : ℤ, f n % k = 0

def solution_set : Finset ℕ :=
  {9, 21, 27, 39, 49, 57, 63, 81, 87, 91, 93}

theorem problem_solution :
  ∀ k : ℕ, valid_k k ↔ k ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3780_378058


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3780_378095

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 2 →
  a 4 + a 5 = 4 →
  a 10 + a 11 = 16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3780_378095


namespace NUMINAMATH_CALUDE_seven_digit_sum_2015_l3780_378024

theorem seven_digit_sum_2015 :
  ∃ (a b c d e f g : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
    e ≠ f ∧ e ≠ g ∧
    f ≠ g ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧
    (1000 * a + 100 * b + 10 * c + d) + (10 * e + f) + g = 2015 :=
by sorry

end NUMINAMATH_CALUDE_seven_digit_sum_2015_l3780_378024


namespace NUMINAMATH_CALUDE_paving_problem_l3780_378074

/-- Represents a worker paving paths in a park -/
structure Worker where
  speed : ℝ
  path_length : ℝ

/-- Represents the paving scenario in the park -/
structure PavingScenario where
  worker1 : Worker
  worker2 : Worker
  total_time : ℝ

/-- The theorem statement for the paving problem -/
theorem paving_problem (scenario : PavingScenario) :
  scenario.worker1.speed > 0 ∧
  scenario.worker2.speed = 1.2 * scenario.worker1.speed ∧
  scenario.total_time = 9 ∧
  scenario.worker1.path_length * scenario.worker1.speed = scenario.worker2.path_length * scenario.worker2.speed ∧
  scenario.worker2.path_length = scenario.worker1.path_length + 2 * (scenario.worker2.path_length / 12) →
  (scenario.worker2.path_length / 12) / scenario.worker2.speed * 60 = 45 := by
  sorry

#check paving_problem

end NUMINAMATH_CALUDE_paving_problem_l3780_378074


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3780_378023

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ -1) (h3 : x ≠ 1) :
  (1 - 3 / (x + 2)) / ((x^2 - 1) / (x + 2)) = 1 / (x + 1) ∧
  (2 - 3 / (2 + 2)) / ((2^2 - 1) / (2 + 2)) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3780_378023


namespace NUMINAMATH_CALUDE_all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l3780_378052

def probability_live_to_75 : ℝ := 0.60

-- Probability that all three policyholders live to 75
theorem all_three_live_to_75 : 
  probability_live_to_75 ^ 3 = 0.216 := by sorry

-- Probability that exactly one out of three policyholders lives to 75
theorem exactly_one_lives_to_75 : 
  3 * probability_live_to_75 * (1 - probability_live_to_75) ^ 2 = 0.288 := by sorry

-- Probability that at least one out of three policyholders lives to 75
theorem at_least_one_lives_to_75 : 
  1 - (1 - probability_live_to_75) ^ 3 = 0.936 := by sorry

end NUMINAMATH_CALUDE_all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l3780_378052


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l3780_378018

theorem curve_to_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 5) (h2 : y = 5 * t - 3) : 
  y = (5 * x - 34) / 3 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l3780_378018


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3780_378015

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The line equation -/
def line_equation (a x y : ℝ) : Prop :=
  a*x + y - 5 = 0

/-- The chord length when the line intersects the circle -/
def chord_length : ℝ := 4

/-- The theorem statement -/
theorem line_intersects_circle (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y ∧ line_equation a x y) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ line_equation a x₁ y₁ ∧
    circle_equation x₂ y₂ ∧ line_equation a x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3780_378015


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l3780_378022

theorem largest_prime_factors_difference (n : Nat) (h : n = 180181) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l3780_378022


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3780_378012

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  g 0 = 1 ∧ ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y

/-- The main theorem stating that g(x) = 5^x - 3^x is the unique solution -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    ∀ x : ℝ, g x = 5^x - 3^x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3780_378012


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3780_378007

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3780_378007


namespace NUMINAMATH_CALUDE_can_finish_typing_l3780_378094

/-- Proves that given a passage of 300 characters and a typing speed of 52 characters per minute, 
    it is possible to finish typing the passage in 6 minutes. -/
theorem can_finish_typing (passage_length : ℕ) (typing_speed : ℕ) (time : ℕ) : 
  passage_length = 300 → 
  typing_speed = 52 → 
  time = 6 → 
  typing_speed * time ≥ passage_length := by
sorry

end NUMINAMATH_CALUDE_can_finish_typing_l3780_378094


namespace NUMINAMATH_CALUDE_number_equals_sixteen_l3780_378087

theorem number_equals_sixteen (x y : ℝ) (h1 : |x| = 9*x - y) (h2 : x = 2) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_sixteen_l3780_378087


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l3780_378062

theorem shirt_tie_outfits (shirts : ℕ) (ties : ℕ) :
  shirts = 7 → ties = 4 → shirts * ties = 28 := by
sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l3780_378062


namespace NUMINAMATH_CALUDE_exponent_product_equality_l3780_378057

theorem exponent_product_equality : 
  (10 ^ 0.4) * (10 ^ 0.1) * (10 ^ 0.7) * (10 ^ 0.2) * (10 ^ 0.6) * (5 ^ 2) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_equality_l3780_378057


namespace NUMINAMATH_CALUDE_expression_simplification_l3780_378081

theorem expression_simplification (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3780_378081


namespace NUMINAMATH_CALUDE_rectangular_box_side_area_l3780_378072

theorem rectangular_box_side_area 
  (l w h : ℝ) 
  (front_top : w * h = 0.5 * (l * w))
  (top_side : l * w = 1.5 * (l * h))
  (volume : l * w * h = 3000) :
  l * h = 200 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_side_area_l3780_378072


namespace NUMINAMATH_CALUDE_same_number_probability_l3780_378099

def max_number : ℕ := 250
def billy_multiple : ℕ := 20
def bobbi_multiple : ℕ := 30

theorem same_number_probability :
  let billy_choices := (max_number - 1) / billy_multiple
  let bobbi_choices := (max_number - 1) / bobbi_multiple
  let common_choices := (max_number - 1) / (lcm billy_multiple bobbi_multiple)
  (common_choices : ℚ) / (billy_choices * bobbi_choices) = 1 / 24 := by
sorry

end NUMINAMATH_CALUDE_same_number_probability_l3780_378099


namespace NUMINAMATH_CALUDE_total_book_pairs_l3780_378043

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 4

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The main theorem stating the total number of possible book pairs -/
theorem total_book_pairs : 
  (choose_two num_genres * books_per_genre * books_per_genre) + 
  (choose_two books_per_genre) = 54 := by sorry

end NUMINAMATH_CALUDE_total_book_pairs_l3780_378043


namespace NUMINAMATH_CALUDE_binomial_30_3_l3780_378082

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3780_378082


namespace NUMINAMATH_CALUDE_swimmer_distance_l3780_378011

/-- Proves that a swimmer covers 8 km when swimming against a current for 5 hours -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) :
  swimmer_speed = 3 →
  current_speed = 1.4 →
  time = 5 →
  (swimmer_speed - current_speed) * time = 8 := by
sorry

end NUMINAMATH_CALUDE_swimmer_distance_l3780_378011


namespace NUMINAMATH_CALUDE_ellipse_axis_ratio_l3780_378079

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where 0 < m < 3,
    if the length of its major axis is twice that of its minor axis,
    then m = 3/2 -/
theorem ellipse_axis_ratio (m : ℝ) 
  (h1 : 0 < m) (h2 : m < 3) 
  (h3 : ∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → 6 = 2*(2*m)) : 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_ratio_l3780_378079


namespace NUMINAMATH_CALUDE_distance_between_B_and_C_l3780_378056

/-- The distance between two locations in kilometers -/
def distance_between (x y : ℝ) : ℝ := |x - y|

/-- The position of an individual after traveling for a given time -/
def position_after_time (initial_position velocity time : ℝ) : ℝ :=
  initial_position + velocity * time

/-- Arithmetic sequence of four speeds -/
structure ArithmeticSpeedSequence (v₁ v₂ v₃ v₄ : ℝ) : Prop where
  decreasing : v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > v₄
  arithmetic : ∃ d : ℝ, v₁ - v₂ = d ∧ v₂ - v₃ = d ∧ v₃ - v₄ = d

theorem distance_between_B_and_C
  (vA vB vC vD : ℝ)  -- Speeds of individuals A, B, C, and D
  (n : ℝ)            -- Time when B and C meet
  (h1 : ArithmeticSpeedSequence vA vB vC vD)
  (h2 : position_after_time 0 vB n = position_after_time 60 (-vC) n)  -- B and C meet after n hours
  (h3 : position_after_time 0 vA (2*n) = position_after_time 60 vD (2*n))  -- A catches up with D after 2n hours
  : distance_between 60 (position_after_time 60 (-vC) n) = 30 :=
sorry

end NUMINAMATH_CALUDE_distance_between_B_and_C_l3780_378056


namespace NUMINAMATH_CALUDE_arithmetic_statement_not_basic_unique_non_basic_statement_l3780_378097

/-- The set of basic algorithmic statements -/
def BasicAlgorithmicStatements : Set String :=
  {"input statement", "output statement", "assignment statement", "conditional statement", "loop statement"}

/-- The list of options given in the problem -/
def Options : List String :=
  ["assignment statement", "arithmetic statement", "conditional statement", "loop statement"]

/-- Theorem: The arithmetic statement is not a member of the set of basic algorithmic statements -/
theorem arithmetic_statement_not_basic : "arithmetic statement" ∉ BasicAlgorithmicStatements := by
  sorry

/-- Theorem: The arithmetic statement is the only option not in the set of basic algorithmic statements -/
theorem unique_non_basic_statement :
  ∀ s ∈ Options, s ∉ BasicAlgorithmicStatements → s = "arithmetic statement" := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_statement_not_basic_unique_non_basic_statement_l3780_378097


namespace NUMINAMATH_CALUDE_cruise_liner_travelers_l3780_378059

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end NUMINAMATH_CALUDE_cruise_liner_travelers_l3780_378059


namespace NUMINAMATH_CALUDE_function_nonnegative_implies_inequalities_l3780_378026

/-- Given real constants a, b, A, B, and a function f(θ) = 1 - a cos θ - b sin θ - A sin 2θ - B cos 2θ,
    if f(θ) ≥ 0 for all real θ, then a² + b² ≤ 2 and A² + B² ≤ 1. -/
theorem function_nonnegative_implies_inequalities (a b A B : ℝ) :
  (∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.sin (2 * θ) - B * Real.cos (2 * θ) ≥ 0) →
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_nonnegative_implies_inequalities_l3780_378026


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3780_378048

theorem no_integer_solutions : ¬ ∃ (m n : ℤ), m + 2*n = 2*m*n - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3780_378048


namespace NUMINAMATH_CALUDE_remainder_equality_l3780_378013

theorem remainder_equality (P P' D : ℕ) (hP : P > P') : 
  let R := P % D
  let R' := P' % D
  let r := (P * P') % D
  let r' := (R * R') % D
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l3780_378013


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3780_378028

theorem geometric_series_common_ratio :
  ∀ (a r : ℚ),
    a = 4/7 →
    a * r = 16/21 →
    r = 4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3780_378028


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l3780_378088

/-- A rectangular parallelepiped with face diagonals √3, √5, and 2 has volume √6 -/
theorem parallelepiped_volume (a b c : ℝ) 
  (h1 : a^2 + b^2 = 3)
  (h2 : a^2 + c^2 = 5)
  (h3 : b^2 + c^2 = 4) :
  a * b * c = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l3780_378088


namespace NUMINAMATH_CALUDE_distributions_five_balls_three_boxes_l3780_378035

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def total_distributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least one specific box is empty -/
def distributions_with_empty_box (n k : ℕ) : ℕ := (k - 1)^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least two specific boxes are empty -/
def distributions_with_two_empty_boxes (n k : ℕ) : ℕ := 1

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that no box remains empty -/
def distributions_no_empty_boxes (n k : ℕ) : ℕ :=
  total_distributions n k - 
  (k * distributions_with_empty_box n k) +
  (Nat.choose k 2 * distributions_with_two_empty_boxes n k)

theorem distributions_five_balls_three_boxes :
  distributions_no_empty_boxes 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distributions_five_balls_three_boxes_l3780_378035


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3780_378065

/-- The lateral surface area of a cone with base radius 6 and slant height 15 is 90π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 6 → l = 15 → π * r * l = 90 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3780_378065


namespace NUMINAMATH_CALUDE_odd_not_div_by_three_square_plus_five_div_by_six_l3780_378090

theorem odd_not_div_by_three_square_plus_five_div_by_six (n : ℤ) 
  (h_odd : Odd n) (h_not_div_three : ¬(3 ∣ n)) : 
  6 ∣ (n^2 + 5) := by
  sorry

end NUMINAMATH_CALUDE_odd_not_div_by_three_square_plus_five_div_by_six_l3780_378090


namespace NUMINAMATH_CALUDE_Z_equals_S_l3780_378002

-- Define the set of functions F
def F : Set (ℝ → ℝ) := {f | ∀ x y, f (x + f y) = f x + f y}

-- Define the set of rational numbers q
def Z : Set ℚ := {q | ∀ f ∈ F, ∃ z : ℝ, f z = q * z}

-- Define the set S
def S : Set ℚ := {q | ∃ n : ℤ, n ≠ 0 ∧ q = (n + 1) / n}

-- State the theorem
theorem Z_equals_S : Z = S := by sorry

end NUMINAMATH_CALUDE_Z_equals_S_l3780_378002


namespace NUMINAMATH_CALUDE_fraction_comparison_l3780_378063

theorem fraction_comparison : 
  (2.00000000004 / ((1.00000000004)^2 + 2.00000000004)) < 
  (2.00000000002 / ((1.00000000002)^2 + 2.00000000002)) := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3780_378063


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3780_378005

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3780_378005


namespace NUMINAMATH_CALUDE_claire_pets_l3780_378046

theorem claire_pets (total_pets : ℕ) (male_pets : ℕ) 
  (h_total : total_pets = 90)
  (h_male : male_pets = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (1 : ℚ) / 4 * gerbils + (1 : ℚ) / 3 * hamsters = male_pets ∧
    gerbils = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_pets_l3780_378046


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l3780_378089

theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (hired_waiters : ℕ) :
  initial_cooks = 9 →
  initial_cooks * 11 = initial_waiters * 3 →
  hired_waiters = 12 →
  initial_cooks * 5 = (initial_waiters + hired_waiters) * 1 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l3780_378089


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3780_378037

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 40 →
  (1/2) * a * b = 30 →
  a^2 + b^2 = c^2 →
  c = 18.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3780_378037


namespace NUMINAMATH_CALUDE_triangle_area_l3780_378033

/-- Given a point A(a, 0) where a > 0, a line with 30° inclination tangent to circle O: x^2 + y^2 = r^2 
    at point B, and |AB| = √3, prove that the area of triangle OAB is √3/2 -/
theorem triangle_area (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let A : ℝ × ℝ := (a, 0)
  let O : ℝ × ℝ := (0, 0)
  let line_slope : ℝ := Real.sqrt 3 / 3
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let tangent_line (x y : ℝ) := y = line_slope * (x - a)
  ∃ (B : ℝ × ℝ), 
    circle B.1 B.2 ∧ 
    tangent_line B.1 B.2 ∧ 
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 3 →
    (1/2 : ℝ) * r * Real.sqrt 3 = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3780_378033


namespace NUMINAMATH_CALUDE_exam_venue_problem_l3780_378084

/-- Given a group of students, calculates the number not good at either of two subjects. -/
def students_not_good_at_either (total : ℕ) (good_at_english : ℕ) (good_at_chinese : ℕ) (good_at_both : ℕ) : ℕ :=
  total - (good_at_english + good_at_chinese - good_at_both)

/-- Proves that in a group of 45 students, if 35 are good at English, 31 are good at Chinese,
    and 24 are good at both, then 3 students are not good at either subject. -/
theorem exam_venue_problem :
  students_not_good_at_either 45 35 31 24 = 3 := by
  sorry

end NUMINAMATH_CALUDE_exam_venue_problem_l3780_378084


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3780_378078

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 26)
  (h3 : r - p = 32) :
  (p + q) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3780_378078


namespace NUMINAMATH_CALUDE_polyhedron_volume_l3780_378042

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the polyhedron formed by cutting a regular quadrangular prism -/
structure Polyhedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  C1 : Point3D
  D1 : Point3D
  O : Point3D  -- Center of the base

/-- The volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ := sorry

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 plane2 : Set Point3D) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Main theorem stating the volume of the polyhedron -/
theorem polyhedron_volume (p : Polyhedron) :
  (distance p.A p.B = 1) →  -- AB = 1
  (distance p.A p.A1 = distance p.O p.C1) →  -- AA₁ = OC₁
  (dihedralAngle {p.A, p.B, p.C, p.D} {p.A1, p.B, p.C1, p.D1} = π/4) →  -- 45° dihedral angle
  (volume p = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l3780_378042


namespace NUMINAMATH_CALUDE_math_test_score_l3780_378053

theorem math_test_score (korean_score english_score : ℕ)
  (h1 : (korean_score + english_score) / 2 = 92)
  (h2 : (korean_score + english_score + math_score) / 3 = 94)
  : math_score = 98 := by
  sorry

end NUMINAMATH_CALUDE_math_test_score_l3780_378053


namespace NUMINAMATH_CALUDE_difference_of_squares_l3780_378047

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3780_378047


namespace NUMINAMATH_CALUDE_visible_friends_count_l3780_378073

theorem visible_friends_count : 
  (Finset.sum (Finset.range 10) (λ i => 
    (Finset.filter (λ j => Nat.gcd (i + 1) j = 1) (Finset.range 6)).card
  )) + 10 = 36 := by
  sorry

end NUMINAMATH_CALUDE_visible_friends_count_l3780_378073


namespace NUMINAMATH_CALUDE_popped_kernel_red_probability_l3780_378003

theorem popped_kernel_red_probability
  (total_kernels : ℝ)
  (red_ratio : ℝ)
  (green_ratio : ℝ)
  (red_pop_ratio : ℝ)
  (green_pop_ratio : ℝ)
  (h1 : red_ratio = 3/4)
  (h2 : green_ratio = 1/4)
  (h3 : red_pop_ratio = 3/5)
  (h4 : green_pop_ratio = 3/4)
  (h5 : red_ratio + green_ratio = 1) :
  let red_kernels := red_ratio * total_kernels
  let green_kernels := green_ratio * total_kernels
  let popped_red := red_pop_ratio * red_kernels
  let popped_green := green_pop_ratio * green_kernels
  let total_popped := popped_red + popped_green
  (popped_red / total_popped) = 12/17 :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_red_probability_l3780_378003


namespace NUMINAMATH_CALUDE_managers_wage_l3780_378039

/-- Proves that the manager's hourly wage is $7.50 given the wage relationships between manager, chef, and dishwasher -/
theorem managers_wage (manager chef dishwasher : ℝ) 
  (h1 : chef = dishwasher * 1.2)
  (h2 : dishwasher = manager / 2)
  (h3 : chef = manager - 3) :
  manager = 7.5 := by
sorry

end NUMINAMATH_CALUDE_managers_wage_l3780_378039


namespace NUMINAMATH_CALUDE_vessel_base_length_l3780_378040

/-- Given a cube immersed in a rectangular vessel, calculate the length of the vessel's base. -/
theorem vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : 
  cube_edge = 15 →
  vessel_width = 14 →
  water_rise = 12.053571428571429 →
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end NUMINAMATH_CALUDE_vessel_base_length_l3780_378040


namespace NUMINAMATH_CALUDE_laurie_kurt_difference_l3780_378034

/-- The number of marbles each person has -/
structure Marbles where
  dennis : ℕ
  kurt : ℕ
  laurie : ℕ

/-- The conditions of the problem -/
def marble_problem (m : Marbles) : Prop :=
  m.dennis = 70 ∧ 
  m.kurt = m.dennis - 45 ∧
  m.laurie = 37

/-- The theorem to prove -/
theorem laurie_kurt_difference (m : Marbles) 
  (h : marble_problem m) : m.laurie - m.kurt = 12 := by
  sorry

end NUMINAMATH_CALUDE_laurie_kurt_difference_l3780_378034


namespace NUMINAMATH_CALUDE_sqrt_seven_sixth_power_l3780_378036

theorem sqrt_seven_sixth_power : (Real.sqrt 7) ^ 6 = 343 := by sorry

end NUMINAMATH_CALUDE_sqrt_seven_sixth_power_l3780_378036


namespace NUMINAMATH_CALUDE_number_divided_multiplied_l3780_378021

theorem number_divided_multiplied : ∃! x : ℚ, (x / 6) * 12 = 18 := by sorry

end NUMINAMATH_CALUDE_number_divided_multiplied_l3780_378021


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l3780_378001

-- Define the triangle and its properties
structure Triangle where
  A : Real
  B : Real
  C_1 : Real
  C_2 : Real
  B_ext : Real
  h_B_gt_A : B > A
  h_angle_sum : A + B + C_1 + C_2 = 180
  h_ext_angle : B_ext = 180 - B

-- Theorem statement
theorem triangle_angle_relation (t : Triangle) :
  t.C_1 - t.C_2 = t.A + t.B_ext - 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l3780_378001
