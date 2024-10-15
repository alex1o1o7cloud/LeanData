import Mathlib

namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2807_280737

theorem quadrilateral_angle_measure (A B C D : ℝ) : 
  A = 105 → B = C → A + B + C + D = 360 → D = 180 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l2807_280737


namespace NUMINAMATH_CALUDE_basketball_team_chances_l2807_280753

/-- The starting percentage for making the basketball team for a 66-inch tall player -/
def starting_percentage : ℝ := 10

/-- The increase in percentage chance per inch above 66 inches -/
def increase_per_inch : ℝ := 10

/-- The height of the player with known chances -/
def known_height : ℝ := 68

/-- The chances of making the team for the player with known height -/
def known_chances : ℝ := 30

/-- The baseline height for the starting percentage -/
def baseline_height : ℝ := 66

theorem basketball_team_chances :
  starting_percentage =
    known_chances - (increase_per_inch * (known_height - baseline_height)) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_chances_l2807_280753


namespace NUMINAMATH_CALUDE_negative_abs_negative_eight_l2807_280749

theorem negative_abs_negative_eight : -|-8| = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_eight_l2807_280749


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l2807_280746

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 8 → isPalindromeInBase n 2 → isPalindromeInBase n 8 → n ≥ 63 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l2807_280746


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l2807_280768

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l2807_280768


namespace NUMINAMATH_CALUDE_smallest_multiple_l2807_280774

theorem smallest_multiple (n : ℕ) : n = 1628 ↔ 
  (∃ k : ℕ, n = 37 * k) ∧ 
  (∃ m : ℕ, n - 3 = 101 * m) ∧ 
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 37 * k) ∧ (∃ m : ℕ, x - 3 = 101 * m))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2807_280774


namespace NUMINAMATH_CALUDE_min_bullseyes_is_52_l2807_280727

/-- The number of shots in the archery tournament -/
def total_shots : ℕ := 120

/-- Chelsea's minimum score on each shot -/
def chelsea_min_score : ℕ := 5

/-- Score for a bullseye -/
def bullseye_score : ℕ := 12

/-- Chelsea's lead at halfway point -/
def chelsea_lead : ℕ := 60

/-- The number of shots taken so far -/
def shots_taken : ℕ := total_shots / 2

/-- Function to calculate the minimum number of bullseyes Chelsea needs to guarantee victory -/
def min_bullseyes_for_victory : ℕ :=
  let max_opponent_score := shots_taken * bullseye_score + chelsea_lead
  let chelsea_non_bullseye_score := (total_shots - shots_taken) * chelsea_min_score
  ((max_opponent_score - chelsea_non_bullseye_score) / (bullseye_score - chelsea_min_score)) + 1

/-- Theorem stating that the minimum number of bullseyes Chelsea needs is 52 -/
theorem min_bullseyes_is_52 : min_bullseyes_for_victory = 52 := by
  sorry

end NUMINAMATH_CALUDE_min_bullseyes_is_52_l2807_280727


namespace NUMINAMATH_CALUDE_municipal_hiring_problem_l2807_280747

theorem municipal_hiring_problem (U P : Finset ℕ) 
  (h1 : U.card = 120)
  (h2 : P.card = 98)
  (h3 : (U ∩ P).card = 40) :
  (U ∪ P).card = 218 := by
sorry

end NUMINAMATH_CALUDE_municipal_hiring_problem_l2807_280747


namespace NUMINAMATH_CALUDE_euler_dedekind_divisibility_l2807_280785

-- Define the Euler totient function
def Φ : ℕ → ℕ := sorry

-- Define the Dedekind's totient function
def Ψ : ℕ → ℕ := sorry

-- Define the set of numbers of the form 2^n₁, 2^n₁3^n₂, or 2^n₁5^n₂
def S : Set ℕ :=
  {n : ℕ | n = 1 ∨ (∃ n₁ n₂ : ℕ, n = 2^n₁ ∨ n = 2^n₁ * 3^n₂ ∨ n = 2^n₁ * 5^n₂)}

-- State the theorem
theorem euler_dedekind_divisibility (n : ℕ) :
  (n ∈ S) ↔ (Φ n ∣ n + Ψ n) := by sorry

end NUMINAMATH_CALUDE_euler_dedekind_divisibility_l2807_280785


namespace NUMINAMATH_CALUDE_full_seasons_count_l2807_280712

/-- The number of days until the final season premiere -/
def days_until_premiere : ℕ := 10

/-- The number of episodes per season -/
def episodes_per_season : ℕ := 15

/-- The number of episodes Joe watches per day -/
def episodes_per_day : ℕ := 6

/-- The number of full seasons already aired -/
def full_seasons : ℕ := (days_until_premiere * episodes_per_day) / episodes_per_season

theorem full_seasons_count : full_seasons = 4 := by
  sorry

end NUMINAMATH_CALUDE_full_seasons_count_l2807_280712


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l2807_280792

/-- Initial percentage of alcohol in the solution -/
def initial_percentage : ℝ := 5

/-- Initial volume of the solution in liters -/
def initial_volume : ℝ := 40

/-- Volume of alcohol added in liters -/
def added_alcohol : ℝ := 3.5

/-- Volume of water added in liters -/
def added_water : ℝ := 6.5

/-- Final percentage of alcohol in the solution -/
def final_percentage : ℝ := 11

theorem initial_alcohol_percentage :
  initial_percentage = 5 :=
by
  have h1 : initial_volume + added_alcohol + added_water = 50 := by sorry
  have h2 : (initial_percentage / 100) * initial_volume + added_alcohol =
            (final_percentage / 100) * (initial_volume + added_alcohol + added_water) := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l2807_280792


namespace NUMINAMATH_CALUDE_expression_factorization_l2807_280728

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 + 9) = 6 * (2 * y^6 + 7 * y^4 - 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2807_280728


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2807_280791

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0,
    this function returns true if they are perpendicular. -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The slope-intercept form of l₁: (m-2)x + 3y + 2m = 0 -/
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m - 2) * x + 3 * y + 2 * m = 0

/-- The slope-intercept form of l₂: x + my + 6 = 0 -/
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  x + m * y + 6 = 0

theorem perpendicular_lines_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, are_perpendicular (m - 2) 3 1 m) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l2807_280791


namespace NUMINAMATH_CALUDE_least_difference_l2807_280752

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem least_difference (x y z : ℕ) : 
  x < y → 
  y < z → 
  y - x > 5 → 
  Even x → 
  x % 3 = 0 → 
  Odd y → 
  Odd z → 
  is_prime y → 
  y > 20 → 
  z % 5 = 0 → 
  1 < x → 
  x < 30 → 
  (∀ x' y' z' : ℕ, 
    x' < y' → 
    y' < z' → 
    y' - x' > 5 → 
    Even x' → 
    x' % 3 = 0 → 
    Odd y' → 
    Odd z' → 
    is_prime y' → 
    y' > 20 → 
    z' % 5 = 0 → 
    1 < x' → 
    x' < 30 → 
    z - x ≤ z' - x') → 
  z - x = 19 := by
sorry

end NUMINAMATH_CALUDE_least_difference_l2807_280752


namespace NUMINAMATH_CALUDE_fraction_simplification_l2807_280720

theorem fraction_simplification (c : ℝ) : (5 + 6 * c) / 9 + 3 = (32 + 6 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2807_280720


namespace NUMINAMATH_CALUDE_gnomon_shadow_length_l2807_280723

/-- Given a candle and a gnomon, this theorem calculates the length of the shadow cast by the gnomon. -/
theorem gnomon_shadow_length 
  (h : ℝ) -- height of the candle
  (H : ℝ) -- height of the gnomon
  (d : ℝ) -- distance between the bases of the candle and gnomon
  (h_pos : h > 0)
  (H_pos : H > 0)
  (d_pos : d > 0)
  (H_gt_h : H > h) :
  ∃ x : ℝ, x = (h * d) / (H - h) ∧ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_gnomon_shadow_length_l2807_280723


namespace NUMINAMATH_CALUDE_derivative_at_one_equals_one_l2807_280798

theorem derivative_at_one_equals_one 
  (f : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^3 - (deriv f 1) * x^2 + 1) : 
  deriv f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_equals_one_l2807_280798


namespace NUMINAMATH_CALUDE_michael_cleaning_count_l2807_280778

/-- The number of times Michael takes a bath per week -/
def baths_per_week : ℕ := 2

/-- The number of times Michael takes a shower per week -/
def showers_per_week : ℕ := 1

/-- The number of weeks in the given time period -/
def weeks : ℕ := 52

/-- The total number of times Michael cleans himself in the given time period -/
def total_cleanings : ℕ := weeks * (baths_per_week + showers_per_week)

theorem michael_cleaning_count : total_cleanings = 156 := by
  sorry

end NUMINAMATH_CALUDE_michael_cleaning_count_l2807_280778


namespace NUMINAMATH_CALUDE_hcl_formed_equals_c2h6_available_l2807_280741

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the stoichiometric coefficients
def stoichiometric_ratio : Reaction :=
  { c2h6 := 1, cl2 := 1, c2h5cl := 1, hcl := 1 }

-- Define the available moles of reactants
def available_reactants : Reaction :=
  { c2h6 := 3, cl2 := 6, c2h5cl := 0, hcl := 0 }

-- Theorem: The number of moles of HCl formed is equal to the number of moles of C2H6 available
theorem hcl_formed_equals_c2h6_available :
  available_reactants.hcl = available_reactants.c2h6 :=
by
  sorry


end NUMINAMATH_CALUDE_hcl_formed_equals_c2h6_available_l2807_280741


namespace NUMINAMATH_CALUDE_mike_earnings_l2807_280722

/-- Mike's total earnings for the week -/
def total_earnings (first_job_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  first_job_wages + second_job_hours * second_job_rate

/-- Theorem stating Mike's total earnings for the week -/
theorem mike_earnings : 
  total_earnings 52 12 9 = 160 := by
  sorry

end NUMINAMATH_CALUDE_mike_earnings_l2807_280722


namespace NUMINAMATH_CALUDE_divisible_by_six_sum_powers_divisible_by_seven_l2807_280767

-- Part (a)
theorem divisible_by_six (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

-- Part (b)
theorem sum_powers_divisible_by_seven :
  7 ∣ (1^2015 + 2^2015 + 3^2015 + 4^2015 + 5^2015 + 6^2015) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_sum_powers_divisible_by_seven_l2807_280767


namespace NUMINAMATH_CALUDE_abc_equation_solutions_l2807_280754

/-- Given integers a, b, c ≥ 2, prove that a b c - 1 = (a - 1)(b - 1)(c - 1) 
    if and only if (a, b, c) is a permutation of (2, 2, 2), (2, 2, 4), (2, 4, 8), or (3, 5, 15) -/
theorem abc_equation_solutions (a b c : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) ↔ 
  List.Perm [a, b, c] [2, 2, 2] ∨ 
  List.Perm [a, b, c] [2, 2, 4] ∨ 
  List.Perm [a, b, c] [2, 4, 8] ∨ 
  List.Perm [a, b, c] [3, 5, 15] :=
by sorry


end NUMINAMATH_CALUDE_abc_equation_solutions_l2807_280754


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2807_280736

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) := by sorry

theorem negation_of_proposition : 
  (¬∃ x₀ : ℝ, (2 : ℝ)^x₀ ≠ 1) ↔ (∀ x₀ : ℝ, (2 : ℝ)^x₀ = 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l2807_280736


namespace NUMINAMATH_CALUDE_expression_equality_l2807_280763

theorem expression_equality (x y z : ℝ) 
  (h1 : x * y = 6)
  (h2 : x - z = 2)
  (h3 : x + y + z = 9) :
  x / y - z / x - z^2 / (x * y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2807_280763


namespace NUMINAMATH_CALUDE_positive_correlation_from_arrangement_l2807_280721

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A scatter plot is a list of points -/
def ScatterPlot := List Point

/-- 
  A function that determines if a scatter plot has a general 
  bottom-left to top-right arrangement 
-/
def isBottomLeftToTopRight (plot : ScatterPlot) : Prop :=
  sorry

/-- 
  A function that calculates the correlation coefficient 
  between x and y coordinates in a scatter plot
-/
def correlationCoefficient (plot : ScatterPlot) : ℝ :=
  sorry

/-- 
  Theorem: If a scatter plot has a general bottom-left to top-right arrangement,
  then the correlation between x and y coordinates is positive
-/
theorem positive_correlation_from_arrangement (plot : ScatterPlot) :
  isBottomLeftToTopRight plot → correlationCoefficient plot > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_correlation_from_arrangement_l2807_280721


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l2807_280779

/-- The amount of flour in cups that Mary has already added to the recipe. -/
def flour_already_added : ℕ := 2

/-- The amount of flour in cups that Mary needs to add to the recipe. -/
def flour_to_be_added : ℕ := 5

/-- The total amount of flour in cups that the recipe calls for. -/
def total_flour : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount :
  total_flour = 7 :=
by sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l2807_280779


namespace NUMINAMATH_CALUDE_gradient_and_magnitude_at_point_l2807_280735

/-- The function z(x, y) = 3x^2 - 2y^2 -/
def z (x y : ℝ) : ℝ := 3 * x^2 - 2 * y^2

/-- The gradient of z at point (x, y) -/
def grad_z (x y : ℝ) : ℝ × ℝ := (6 * x, -4 * y)

theorem gradient_and_magnitude_at_point :
  let p : ℝ × ℝ := (1, 2)
  (grad_z p.1 p.2 = (6, -8)) ∧
  (Real.sqrt ((grad_z p.1 p.2).1^2 + (grad_z p.1 p.2).2^2) = 10) := by
  sorry

end NUMINAMATH_CALUDE_gradient_and_magnitude_at_point_l2807_280735


namespace NUMINAMATH_CALUDE_exponent_equality_l2807_280756

theorem exponent_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 7) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2807_280756


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l2807_280765

/-- Given two plane vectors a and b with the angle between them π/2 and magnitudes 1,
    prove that the magnitude of 3a - 2b is 1. -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  ((3*a.1 - 2*b.1)^2 + (3*a.2 - 2*b.2)^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l2807_280765


namespace NUMINAMATH_CALUDE_fibonacci_square_equality_l2807_280776

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_square_equality :
  ∃! n : ℕ, n > 0 ∧ fib n = n^2 ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_fibonacci_square_equality_l2807_280776


namespace NUMINAMATH_CALUDE_pizza_combinations_l2807_280788

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  (n.choose 1) + (n.choose 2) + (n.choose 3) = 92 :=
by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2807_280788


namespace NUMINAMATH_CALUDE_perimeter_F₂MN_is_8_l2807_280775

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the foci F₁ and F₂
variable (F₁ F₂ : ℝ × ℝ)

-- Define points M and N on the ellipse
variable (M N : ℝ × ℝ)

-- Axiom: F₁ and F₂ are foci of the ellipse C
axiom foci_of_C : ∀ (x y : ℝ), C x y → ∃ (a : ℝ), dist (x, y) F₁ + dist (x, y) F₂ = 2 * a

-- Axiom: M and N are on the ellipse C
axiom M_on_C : C M.1 M.2
axiom N_on_C : C N.1 N.2

-- Axiom: M, N, and F₁ are collinear
axiom collinear_MNF₁ : ∃ (t : ℝ), N = F₁ + t • (M - F₁)

-- Theorem: The perimeter of triangle F₂MN is 8
theorem perimeter_F₂MN_is_8 : dist M N + dist M F₂ + dist N F₂ = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_F₂MN_is_8_l2807_280775


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2807_280799

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_02 : ℚ := 2/99

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_02 = 35/99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2807_280799


namespace NUMINAMATH_CALUDE_river_road_bus_car_ratio_l2807_280758

/-- The ratio of buses to cars on River Road -/
def busCarRatio (numBuses : ℕ) (numCars : ℕ) : ℚ :=
  numBuses / numCars

theorem river_road_bus_car_ratio : 
  let numCars : ℕ := 60
  let numBuses : ℕ := numCars - 40
  busCarRatio numBuses numCars = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_river_road_bus_car_ratio_l2807_280758


namespace NUMINAMATH_CALUDE_square_difference_l2807_280782

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2807_280782


namespace NUMINAMATH_CALUDE_square_plot_poles_l2807_280760

/-- The number of fence poles needed for a square plot -/
def total_poles (poles_per_side : ℕ) : ℕ :=
  poles_per_side * 4 - 4

/-- Theorem: For a square plot with 27 fence poles on each side, 
    the total number of poles needed is 104 -/
theorem square_plot_poles : total_poles 27 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_poles_l2807_280760


namespace NUMINAMATH_CALUDE_pens_per_student_l2807_280701

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (num_students : ℕ) : 
  total_pens = 1001 → total_pencils = 910 → num_students = 91 →
  total_pens / num_students = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_per_student_l2807_280701


namespace NUMINAMATH_CALUDE_new_speed_calculation_l2807_280793

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) 
  (h1 : distance = 469)
  (h2 : original_time = 6)
  (h3 : original_time > 0) :
  let new_time := original_time * (3/2)
  let new_speed := distance / new_time
  new_speed = distance / (original_time * (3/2)) := by
sorry

end NUMINAMATH_CALUDE_new_speed_calculation_l2807_280793


namespace NUMINAMATH_CALUDE_x_greater_than_e_l2807_280729

theorem x_greater_than_e (x : ℝ) (h1 : Real.log x > 0) (h2 : x > 1) : x > Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_e_l2807_280729


namespace NUMINAMATH_CALUDE_right_triangle_height_l2807_280781

theorem right_triangle_height (a b h : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  a = 1 → b = 4 → h^2 = a * b → h = 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_height_l2807_280781


namespace NUMINAMATH_CALUDE_badminton_cost_comparison_l2807_280700

/-- Cost calculation for badminton equipment purchase --/
theorem badminton_cost_comparison 
  (x : ℝ) 
  (h_x : x ≥ 3) 
  (yA yB : ℝ) 
  (h_yA : yA = 32 * x + 320) 
  (h_yB : yB = 40 * x + 280) : 
  (yA = yB ↔ x = 5) ∧ 
  (3 ≤ x ∧ x < 5 → yB < yA) ∧ 
  (x > 5 → yA < yB) := by
sorry

end NUMINAMATH_CALUDE_badminton_cost_comparison_l2807_280700


namespace NUMINAMATH_CALUDE_milk_drinking_problem_l2807_280777

theorem milk_drinking_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (max_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  max_fraction = 1/3 →
  max_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_drinking_problem_l2807_280777


namespace NUMINAMATH_CALUDE_square_area_ratio_l2807_280702

theorem square_area_ratio (small_side : ℝ) (large_side : ℝ) 
  (h1 : small_side = 2) 
  (h2 : large_side = 5) : 
  (small_side^2) / ((large_side^2 / 2) - (small_side^2 / 2)) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2807_280702


namespace NUMINAMATH_CALUDE_abs_diff_segments_of_cyclic_quad_with_incircle_l2807_280738

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadWithIncircle where
  -- Side lengths of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Conditions for a valid quadrilateral
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  -- Condition for cyclic quadrilateral (sum of opposite sides are equal)
  cyclic : a + c = b + d
  -- Additional condition for having an inscribed circle
  has_incircle : True

/-- Theorem stating the absolute difference between segments -/
theorem abs_diff_segments_of_cyclic_quad_with_incircle 
  (q : CyclicQuadWithIncircle) 
  (h1 : q.a = 80) 
  (h2 : q.b = 100) 
  (h3 : q.c = 140) 
  (h4 : q.d = 120) 
  (x y : ℝ) 
  (h5 : x + y = q.c) : 
  |x - y| = 166.36 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_segments_of_cyclic_quad_with_incircle_l2807_280738


namespace NUMINAMATH_CALUDE_candidate_A_percentage_l2807_280731

def total_votes : ℕ := 560000
def invalid_vote_percentage : ℚ := 15 / 100
def valid_votes_for_A : ℕ := 380800

theorem candidate_A_percentage :
  (valid_votes_for_A : ℚ) / ((1 - invalid_vote_percentage) * total_votes) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_candidate_A_percentage_l2807_280731


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l2807_280717

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- Define the given conditions
variable (l₁ l₂ : Line) (α β : Plane)
variable (h1 : perpendicular l₁ α)
variable (h2 : contains β l₂)

-- Theorem to prove
theorem line_plane_perpendicular_parallel 
  (h3 : parallel α β) : perpendicularLines l₁ l₂ :=
sorry

theorem parallel_lines_perpendicular_planes 
  (h4 : perpendicularLines l₁ l₂) : perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_parallel_lines_perpendicular_planes_l2807_280717


namespace NUMINAMATH_CALUDE_simplify_expression_l2807_280742

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2807_280742


namespace NUMINAMATH_CALUDE_ellen_yogurt_amount_l2807_280708

/-- The amount of yogurt used in Ellen's smoothie -/
def yogurt_amount (strawberries orange_juice total : ℝ) : ℝ :=
  total - (strawberries + orange_juice)

/-- Theorem: Ellen used 0.1 cup of yogurt in her smoothie -/
theorem ellen_yogurt_amount :
  yogurt_amount 0.2 0.2 0.5 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ellen_yogurt_amount_l2807_280708


namespace NUMINAMATH_CALUDE_power_product_cube_l2807_280705

theorem power_product_cube (R : Type*) [CommRing R] (x y : R) :
  (x * y^2)^3 = x^3 * y^6 := by sorry

end NUMINAMATH_CALUDE_power_product_cube_l2807_280705


namespace NUMINAMATH_CALUDE_masters_percentage_is_76_l2807_280740

/-- Represents a sports team with juniors and masters -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- Calculates the percentage of masters in a team -/
def percentageMasters (team : Team) : ℚ :=
  (team.masters : ℚ) / ((team.juniors + team.masters) : ℚ) * 100

/-- Theorem stating that under the given conditions, the percentage of masters is 76% -/
theorem masters_percentage_is_76 (team : Team) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  percentageMasters team = 76 := by
  sorry

#eval (76 : ℚ)

end NUMINAMATH_CALUDE_masters_percentage_is_76_l2807_280740


namespace NUMINAMATH_CALUDE_movies_watched_l2807_280714

theorem movies_watched (total : ℕ) (book_movie_diff : ℕ) : 
  total = 13 ∧ book_movie_diff = 1 → 
  ∃ (books movies : ℕ), books = movies + book_movie_diff ∧ 
                         books + movies = total ∧ 
                         movies = 6 := by
  sorry

end NUMINAMATH_CALUDE_movies_watched_l2807_280714


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l2807_280710

/-- The number of paint colors available. -/
def num_colors : ℕ := 5

/-- The number of painting methods available. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l2807_280710


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2807_280759

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2807_280759


namespace NUMINAMATH_CALUDE_scarves_per_box_l2807_280743

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  mittens_per_box = 6 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * mittens_per_box) / num_boxes = 4 := by
sorry

end NUMINAMATH_CALUDE_scarves_per_box_l2807_280743


namespace NUMINAMATH_CALUDE_K_3_15_5_l2807_280730

def K (x y z : ℚ) : ℚ := x / y + y / z + z / x

theorem K_3_15_5 : K 3 15 5 = 73 / 15 := by
  sorry

end NUMINAMATH_CALUDE_K_3_15_5_l2807_280730


namespace NUMINAMATH_CALUDE_unique_p_q_for_inequality_l2807_280703

theorem unique_p_q_for_inequality :
  ∀ (p q : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |Real.sqrt (1 - x^2) - p*x - q| ≤ (Real.sqrt 2 - 1) / 2) →
    p = -1 ∧ q = (1 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_p_q_for_inequality_l2807_280703


namespace NUMINAMATH_CALUDE_farey_sequence_properties_l2807_280797

/-- Farey sequence of order n -/
def farey_sequence (n : ℕ) : List (ℚ) := sorry

/-- Sum of numerators in a Farey sequence -/
def sum_numerators (seq : List ℚ) : ℚ := sorry

/-- Sum of denominators in a Farey sequence -/
def sum_denominators (seq : List ℚ) : ℚ := sorry

/-- Sum of fractions in a Farey sequence -/
def sum_fractions (seq : List ℚ) : ℚ := sorry

theorem farey_sequence_properties (n : ℕ) :
  let seq := farey_sequence n
  (sum_denominators seq = 2 * sum_numerators seq) ∧
  (sum_fractions seq = (seq.length : ℚ) / 2) := by sorry

end NUMINAMATH_CALUDE_farey_sequence_properties_l2807_280797


namespace NUMINAMATH_CALUDE_fallen_striped_tiles_count_l2807_280739

/-- Represents the type of a tile -/
inductive TileType
| Striped
| Plain

/-- Represents the state of a tile position -/
inductive TileState
| Present
| Fallen

/-- Represents the initial checkerboard pattern -/
def initialPattern : List (List TileType) :=
  List.replicate 7 (List.replicate 7 TileType.Striped)

/-- Represents the current state of the wall after some tiles have fallen -/
def currentState : List (List TileState) :=
  [
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Fallen, TileState.Fallen],
    [TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Fallen, TileState.Fallen],
    [TileState.Fallen, TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Fallen, TileState.Present, TileState.Present],
    [TileState.Fallen, TileState.Fallen, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present],
    [TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present, TileState.Present]
  ]

/-- Counts the number of fallen striped tiles -/
def countFallenStripedTiles (initial : List (List TileType)) (current : List (List TileState)) : Nat :=
  sorry

/-- Theorem: The number of fallen striped tiles is 15 -/
theorem fallen_striped_tiles_count :
  countFallenStripedTiles initialPattern currentState = 15 := by
  sorry

end NUMINAMATH_CALUDE_fallen_striped_tiles_count_l2807_280739


namespace NUMINAMATH_CALUDE_special_sequence_14th_term_l2807_280715

/-- A sequence of positive real numbers satisfying certain conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 2 = 2) ∧ 
  (a 8 = 8) ∧ 
  (∀ n ≥ 2, Real.sqrt (a (n - 1)) * Real.sqrt (a (n + 1)) = a n)

/-- The 14th term of the special sequence is 32 -/
theorem special_sequence_14th_term (a : ℕ → ℝ) (h : SpecialSequence a) : a 14 = 32 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_14th_term_l2807_280715


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2807_280794

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 5 →
  (red_balls : ℚ) / total_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2807_280794


namespace NUMINAMATH_CALUDE_rotated_square_top_vertex_distance_l2807_280713

/-- The distance of the top vertex of a rotated square from the base line -/
theorem rotated_square_top_vertex_distance 
  (square_side : ℝ) 
  (rotation_angle : ℝ) :
  square_side = 2 →
  rotation_angle = π/4 →
  let diagonal := square_side * Real.sqrt 2
  let center_height := square_side / 2
  let vertical_shift := Real.sqrt 2 / 2 * diagonal
  center_height + vertical_shift = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rotated_square_top_vertex_distance_l2807_280713


namespace NUMINAMATH_CALUDE_equation_solutions_l2807_280783

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (2 * (x₁ - 3) = 3 * x₁ * (x₁ - 3) ∧ x₁ = 3) ∧ 
                (2 * (x₂ - 3) = 3 * x₂ * (x₂ - 3) ∧ x₂ = 2/3)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ + 1 = 0 ∧ y₁ = 1) ∧ 
                (2 * y₂^2 - 3 * y₂ + 1 = 0 ∧ y₂ = 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l2807_280783


namespace NUMINAMATH_CALUDE_max_sections_five_l2807_280789

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_l2807_280789


namespace NUMINAMATH_CALUDE_max_value_of_z_l2807_280704

theorem max_value_of_z (x y : ℝ) (h1 : |x| + |y| ≤ 4) (h2 : 2*x + y - 4 ≤ 0) :
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 20/3 ∧ ∃ (x' y' : ℝ), |x'| + |y'| ≤ 4 ∧ 2*x' + y' - 4 ≤ 0 ∧ 2*x' - y' = 20/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2807_280704


namespace NUMINAMATH_CALUDE_counterexample_square_inequality_l2807_280719

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_square_inequality_l2807_280719


namespace NUMINAMATH_CALUDE_simplify_expression_l2807_280766

theorem simplify_expression (a b : ℝ) : a + b - (a - b) = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2807_280766


namespace NUMINAMATH_CALUDE_not_proportional_six_nine_nine_twelve_l2807_280773

/-- Two ratios a:b and c:d are proportional if a/b = c/d -/
def proportional (a b c d : ℚ) : Prop := a / b = c / d

/-- The ratios 6:9 and 9:12 -/
def ratio1 : ℚ := 6 / 9
def ratio2 : ℚ := 9 / 12

/-- Theorem stating that 6:9 and 9:12 are not proportional -/
theorem not_proportional_six_nine_nine_twelve : ¬(proportional 6 9 9 12) := by
  sorry

end NUMINAMATH_CALUDE_not_proportional_six_nine_nine_twelve_l2807_280773


namespace NUMINAMATH_CALUDE_product_ab_l2807_280771

theorem product_ab (a b : ℝ) (h1 : a - b = 2) (h2 : a^2 + b^2 = 25) : a * b = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l2807_280771


namespace NUMINAMATH_CALUDE_cuboid_volume_l2807_280761

/-- A cuboid with integer edge lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℕ :=
  c.length * c.width * c.height

/-- Theorem: The volume of a cuboid with edges 6 cm, 5 cm, and 6 cm is 180 cubic centimeters -/
theorem cuboid_volume : volume { length := 6, width := 5, height := 6 } = 180 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2807_280761


namespace NUMINAMATH_CALUDE_mistake_correction_l2807_280790

theorem mistake_correction (x : ℝ) : 8 * x + 8 = 56 → x / 8 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l2807_280790


namespace NUMINAMATH_CALUDE_complement_of_46_35_l2807_280764

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- The theorem stating that the complement of 46°35' is 43°25' -/
theorem complement_of_46_35 :
  complement { degrees := 46, minutes := 35 } = { degrees := 43, minutes := 25 } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_46_35_l2807_280764


namespace NUMINAMATH_CALUDE_parabola_directrix_theorem_l2807_280724

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- A parabola opens upward if a > 0 -/
def opens_upward (p : Parabola) : Prop := p.a > 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_directrix_theorem (p : Parabola) :
  p.a = 1/4 ∧ p.b = 0 ∧ p.c = 0 ∧ opens_upward p ∧ vertex p = (0, 0) →
  directrix p = -1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_theorem_l2807_280724


namespace NUMINAMATH_CALUDE_amanda_friends_count_l2807_280772

def total_tickets : ℕ := 80
def tickets_per_friend : ℕ := 4
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_friends_count :
  ∃ (friends : ℕ), 
    friends * tickets_per_friend + second_day_tickets + third_day_tickets = total_tickets ∧
    friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_amanda_friends_count_l2807_280772


namespace NUMINAMATH_CALUDE_total_silk_dyed_l2807_280733

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end NUMINAMATH_CALUDE_total_silk_dyed_l2807_280733


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l2807_280787

def n : ℕ := 5040000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor 5 n = 315000000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l2807_280787


namespace NUMINAMATH_CALUDE_racecar_repair_discount_l2807_280709

/-- Calculates the discount percentage on a racecar repair --/
theorem racecar_repair_discount (original_cost prize keep_percentage profit : ℝ) :
  original_cost = 20000 →
  prize = 70000 →
  keep_percentage = 0.9 →
  profit = 47000 →
  (original_cost - (keep_percentage * prize - profit)) / original_cost = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_racecar_repair_discount_l2807_280709


namespace NUMINAMATH_CALUDE_initial_water_temp_l2807_280786

-- Define the constants
def total_time : ℕ := 73
def temp_increase_per_minute : ℕ := 3
def boiling_point : ℕ := 212
def pasta_cooking_time : ℕ := 12

-- Define the theorem
theorem initial_water_temp (mixing_time : ℕ) (boiling_time : ℕ) 
  (h1 : mixing_time = pasta_cooking_time / 3)
  (h2 : boiling_time = total_time - (pasta_cooking_time + mixing_time))
  (h3 : boiling_point = temp_increase_per_minute * boiling_time + 41) :
  41 = boiling_point - temp_increase_per_minute * boiling_time :=
by sorry

end NUMINAMATH_CALUDE_initial_water_temp_l2807_280786


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_l2807_280784

/-- The distance from a point in polar coordinates to a line in polar form -/
theorem distance_point_to_line_polar (ρ_A : ℝ) (θ_A : ℝ) (k : ℝ) :
  let l : ℝ × ℝ → Prop := λ (ρ, θ) ↦ 2 * ρ * Real.sin (θ - π/4) = Real.sqrt 2
  let A : ℝ × ℝ := (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)
  let d := abs (A.1 - A.2 + 1) / Real.sqrt 2
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 → d = 5 * Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_distance_point_to_line_polar_l2807_280784


namespace NUMINAMATH_CALUDE_tom_dimes_count_l2807_280796

/-- The number of dimes Tom initially had -/
def initial_dimes : ℕ := 15

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad : ℕ := 33

/-- The total number of dimes Tom has after receiving dimes from his dad -/
def total_dimes : ℕ := initial_dimes + dimes_from_dad

theorem tom_dimes_count : total_dimes = 48 := by
  sorry

end NUMINAMATH_CALUDE_tom_dimes_count_l2807_280796


namespace NUMINAMATH_CALUDE_direct_proportion_n_value_l2807_280732

/-- A direct proportion function passing through (n, -9) with decreasing y as x increases -/
def DirectProportionFunction (n : ℝ) : ℝ → ℝ := fun x ↦ -n * x

theorem direct_proportion_n_value (n : ℝ) :
  (DirectProportionFunction n n = -9) ∧  -- The graph passes through (n, -9)
  (∀ x₁ x₂, x₁ < x₂ → DirectProportionFunction n x₁ > DirectProportionFunction n x₂) →  -- y decreases as x increases
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_direct_proportion_n_value_l2807_280732


namespace NUMINAMATH_CALUDE_triangle_equality_l2807_280762

/-- Given a triangle ABC with sides a, b, and c satisfying a^2 + b^2 + c^2 = ab + bc + ac,
    prove that the triangle is equilateral. -/
theorem triangle_equality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (eq : a^2 + b^2 + c^2 = a*b + b*c + a*c) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2807_280762


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l2807_280726

/-- Triangle with positive integer side lengths --/
structure IsoscelesTriangle where
  pq : ℕ+
  qr : ℕ+

/-- Angle bisector intersection point --/
structure AngleBisectorIntersection where
  qj : ℝ

/-- Theorem statement for the smallest perimeter of the isosceles triangle --/
theorem smallest_perimeter_isosceles_triangle
  (t : IsoscelesTriangle)
  (j : AngleBisectorIntersection)
  (h1 : j.qj = 10) :
  2 * (t.pq + t.qr) ≥ 416 ∧
  ∃ (t' : IsoscelesTriangle), 2 * (t'.pq + t'.qr) = 416 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l2807_280726


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2807_280734

/-- Given a circle with the following properties:
  - A chord of length 18
  - The chord is intersected by a diameter at a point
  - The intersection point is 7 units from the center
  - The intersection point divides the chord in the ratio 2:1
  Prove that the radius of the circle is 11 -/
theorem circle_radius_proof (chord_length : ℝ) (intersection_distance : ℝ) 
  (h1 : chord_length = 18)
  (h2 : intersection_distance = 7)
  (h3 : ∃ (a b : ℝ), a + b = chord_length ∧ a = 2 * b) :
  ∃ (radius : ℝ), radius = 11 ∧ radius^2 = intersection_distance^2 + (chord_length^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2807_280734


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2807_280751

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 - 2*x < 0 → abs x < 2) ∧
  (∃ x : ℝ, abs x < 2 ∧ ¬(x^2 - 2*x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2807_280751


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l2807_280780

/-- The cost of tickets for a group attending a circus -/
def ticket_cost (adult_price : ℚ) (child_price : ℚ) (num_adults : ℕ) (num_children : ℕ) : ℚ :=
  (adult_price * num_adults) + (child_price * num_children)

/-- Theorem: The total cost of tickets for 2 adults at $44.00 each and 5 children at $28.00 each is $228.00 -/
theorem circus_ticket_cost :
  ticket_cost 44 28 2 5 = 228 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l2807_280780


namespace NUMINAMATH_CALUDE_inverse_sum_bound_l2807_280706

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := |x^2 - 1| + x^2 + k*x

-- State the theorem
theorem inverse_sum_bound 
  (k : ℝ) (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < 2)
  (h4 : f k α = 0) (h5 : f k β = 0) :
  1/α + 1/β < 4 :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_bound_l2807_280706


namespace NUMINAMATH_CALUDE_gcf_of_180_and_126_l2807_280707

theorem gcf_of_180_and_126 : Nat.gcd 180 126 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_and_126_l2807_280707


namespace NUMINAMATH_CALUDE_magic_box_theorem_l2807_280750

theorem magic_box_theorem (m : ℝ) : m^2 - 2*m - 1 = 2 → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_magic_box_theorem_l2807_280750


namespace NUMINAMATH_CALUDE_symmetry_plane_arrangement_l2807_280755

/-- A symmetry plane of a body. -/
structure SymmetryPlane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A body with symmetry planes. -/
structure Body where
  symmetry_planes : List SymmetryPlane
  exactly_three_planes : symmetry_planes.length = 3

/-- Angle between two symmetry planes. -/
def angle_between (p1 p2 : SymmetryPlane) : ℝ :=
  sorry

/-- Predicate to check if two planes are perpendicular. -/
def are_perpendicular (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 90

/-- Predicate to check if two planes intersect at 60 degrees. -/
def intersect_at_60 (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 60

/-- Theorem stating the possible arrangements of symmetry planes. -/
theorem symmetry_plane_arrangement (b : Body) :
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    are_perpendicular p1 p2) ∨
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    intersect_at_60 p1 p2) :=
  sorry

end NUMINAMATH_CALUDE_symmetry_plane_arrangement_l2807_280755


namespace NUMINAMATH_CALUDE_necklace_beads_l2807_280757

theorem necklace_beads (total : ℕ) (amethyst : ℕ) (amber : ℕ) (turquoise : ℕ) :
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  total = amethyst + amber + turquoise →
  turquoise = 19 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l2807_280757


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2807_280744

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Define points M and N
def point_M : ℝ × ℝ := (-2, 0)
def point_N : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition for point Q
def distance_ratio (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) / Real.sqrt ((x + 1)^2 + y^2) = Real.sqrt 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the perpendicularity condition
def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Part I: Prove that the trajectory of Q is the circle C
  (∀ x y : ℝ, distance_ratio x y ↔ circle_C x y) ∧
  -- Part II: Prove that when l intersects C at two points with perpendicular position vectors, a = ±√2
  (∀ a x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_l a x₁ y₁ ∧ line_l a x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    perpendicular_vectors x₁ y₁ x₂ y₂ →
    a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2807_280744


namespace NUMINAMATH_CALUDE_tan_half_angle_special_point_l2807_280770

/-- 
If the terminal side of angle α passes through the point (-1, 2),
then tan(α/2) = (1 + √5) / 2.
-/
theorem tan_half_angle_special_point (α : Real) :
  (Real.cos α = -1 / Real.sqrt 5 ∧ Real.sin α = 2 / Real.sqrt 5) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_special_point_l2807_280770


namespace NUMINAMATH_CALUDE_prime_power_plus_two_l2807_280745

theorem prime_power_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → Prime (p^3 + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_plus_two_l2807_280745


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2807_280748

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^7 + 5^7 + 5^7 + 5^7)^(1/3) → x = 100 * 10^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_powers_l2807_280748


namespace NUMINAMATH_CALUDE_expand_expression_l2807_280795

theorem expand_expression (x : ℝ) : (3*x^2 + 2*x - 4)*(x - 3) = 3*x^3 - 7*x^2 - 10*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2807_280795


namespace NUMINAMATH_CALUDE_square_in_S_l2807_280718

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

def S : Set ℕ :=
  {n | is_sum_of_two_squares (n - 1) ∧ 
       is_sum_of_two_squares n ∧ 
       is_sum_of_two_squares (n + 1)}

theorem square_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_square_in_S_l2807_280718


namespace NUMINAMATH_CALUDE_extreme_point_property_and_max_value_l2807_280711

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^3 - a * x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Theorem statement
theorem extreme_point_property_and_max_value (a : ℝ) :
  a > 0 →
  ∃ x₀ x₁ : ℝ,
    x₀ ≠ x₁ ∧
    (∀ x : ℝ, f a x₀ ≥ f a x ∨ f a x₀ ≤ f a x) ∧
    f a x₀ = f a x₁ →
    x₁ + 2 * x₀ = 6 ∧
    (∀ x : ℝ, x ∈ Set.Icc 0 6 → g a x ≤ 40) ∧
    (∃ x : ℝ, x ∈ Set.Icc 0 6 ∧ g a x = 40) →
    a = 4 ∨ a = 12 :=
by sorry

end NUMINAMATH_CALUDE_extreme_point_property_and_max_value_l2807_280711


namespace NUMINAMATH_CALUDE_min_value_of_function_l2807_280716

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  let y := 2*x + 4/(x-1) - 1
  ∀ z, z = 2*x + 4/(x-1) - 1 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2807_280716


namespace NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2807_280725

-- Define a structure for triangles
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.c < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop :=
  isPrime t.a ∧ isPrime t.b ∧ isPrime t.c

-- Define what it means for a triangle to be valid (sum of angles is 180)
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b + t.c = 180

-- Theorem statement
theorem unique_acute_prime_angled_triangle :
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.a = 2 ∧ t.b = 89 ∧ t.c = 89 :=
sorry

end NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2807_280725


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2807_280769

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2807_280769
