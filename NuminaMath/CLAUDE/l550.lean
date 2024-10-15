import Mathlib

namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l550_55066

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) :
  let circle_diameter := 2 * r
  let square_side := 2 * circle_diameter
  square_side ^ 2 = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l550_55066


namespace NUMINAMATH_CALUDE_square_sum_product_inequality_l550_55033

theorem square_sum_product_inequality (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_inequality_l550_55033


namespace NUMINAMATH_CALUDE_route_b_faster_l550_55055

/-- Represents a route with multiple segments, each with its own distance and speed. -/
structure Route where
  segments : List (Float × Float)
  total_distance : Float

/-- Calculates the total time taken to travel a route -/
def travel_time (r : Route) : Float :=
  r.segments.foldl (fun acc (d, s) => acc + d / s) 0

/-- Route A details -/
def route_a : Route :=
  { segments := [(8, 40)], total_distance := 8 }

/-- Route B details -/
def route_b : Route :=
  { segments := [(5.5, 45), (1, 25), (0.5, 15)], total_distance := 7 }

/-- The time difference between Route A and Route B in minutes -/
def time_difference : Float :=
  travel_time route_a - travel_time route_b

theorem route_b_faster : 
  0.26 < time_difference ∧ time_difference < 0.28 :=
sorry

end NUMINAMATH_CALUDE_route_b_faster_l550_55055


namespace NUMINAMATH_CALUDE_f_sum_zero_four_l550_55092

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem f_sum_zero_four (a b c d : ℝ) :
  f a b c d 1 = 1 →
  f a b c d 2 = 2 →
  f a b c d 3 = 3 →
  f a b c d 0 + f a b c d 4 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_f_sum_zero_four_l550_55092


namespace NUMINAMATH_CALUDE_sphere_volume_l550_55039

theorem sphere_volume (A : Real) (V : Real) :
  A = 9 * Real.pi →  -- area of the main view (circle)
  V = (4 / 3) * Real.pi * (3 ^ 3) →  -- volume formula with radius 3
  V = 36 * Real.pi :=  -- expected volume
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l550_55039


namespace NUMINAMATH_CALUDE_solution_sets_imply_a_minus_b_eq_neg_seven_l550_55015

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the coefficients a and b
def a : ℝ := -4
def b : ℝ := 3

-- Theorem statement
theorem solution_sets_imply_a_minus_b_eq_neg_seven :
  (A_intersect_B = {x | x^2 + a*x + b < 0}) →
  a - b = -7 := by sorry

end NUMINAMATH_CALUDE_solution_sets_imply_a_minus_b_eq_neg_seven_l550_55015


namespace NUMINAMATH_CALUDE_smallest_b_value_l550_55053

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l550_55053


namespace NUMINAMATH_CALUDE_square_ratio_l550_55043

theorem square_ratio (n m : ℝ) :
  (∃ a : ℝ, 9 * x^2 + n * x + 1 = (3 * x + a)^2) →
  (∃ b : ℝ, 4 * y^2 + 12 * y + m = (2 * y + b)^2) →
  n > 0 →
  n / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_l550_55043


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l550_55081

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_diameter : ℝ := 2
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 4
  let hole_depth : ℝ := 10
  let sphere_volume := (4/3) * π * (sphere_diameter/2)^3
  let hole1_volume := π * (hole1_diameter/2)^2 * hole_depth
  let hole2_volume := π * (hole2_diameter/2)^2 * hole_depth
  let hole3_volume := π * (hole3_diameter/2)^2 * hole_depth
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = (31710/3) * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l550_55081


namespace NUMINAMATH_CALUDE_least_number_of_pennies_l550_55052

theorem least_number_of_pennies :
  ∃ (p : ℕ), p > 0 ∧ p % 7 = 3 ∧ p % 4 = 1 ∧
  ∀ (q : ℕ), q > 0 ∧ q % 7 = 3 ∧ q % 4 = 1 → p ≤ q :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_of_pennies_l550_55052


namespace NUMINAMATH_CALUDE_power_inequality_l550_55030

theorem power_inequality : (1.7 : ℝ) ^ (0.3 : ℝ) > (0.9 : ℝ) ^ (0.3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_power_inequality_l550_55030


namespace NUMINAMATH_CALUDE_tobys_journey_l550_55080

/-- Toby's sled-pulling journey --/
theorem tobys_journey (unloaded_speed loaded_speed : ℝ)
  (distance1 distance2 distance3 distance4 : ℝ)
  (h1 : unloaded_speed = 20)
  (h2 : loaded_speed = 10)
  (h3 : distance1 = 180)
  (h4 : distance2 = 120)
  (h5 : distance3 = 80)
  (h6 : distance4 = 140) :
  distance1 / loaded_speed + distance2 / unloaded_speed +
  distance3 / loaded_speed + distance4 / unloaded_speed = 39 := by
  sorry

end NUMINAMATH_CALUDE_tobys_journey_l550_55080


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l550_55051

/-- For a parabola with equation y^2 = 4x, the distance from its focus to its directrix is 2 -/
theorem parabola_focus_directrix_distance : 
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ), 
    (f.1 = 1 ∧ f.2 = 0) ∧ -- focus coordinates
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧ -- directrix equation
    (f.1 - d.1 = 2) -- distance between focus and directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l550_55051


namespace NUMINAMATH_CALUDE_sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l550_55064

/-- Sum of the first n terms of a geometric sequence with a₁ = 2 and r = 1 -/
def geometricSum (n : ℕ) : ℝ := 2 * n

/-- The geometric sequence with a₁ = 2 and r = 1 -/
def geometricSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => geometricSequence n

theorem sum_first_10_terms :
  geometricSum 10 = 20 := by sorry

theorem sequence_is_constant (n : ℕ) :
  geometricSequence n = 2 := by sorry

theorem sum_equals_first_term_times_n (n : ℕ) :
  geometricSum n = 2 * n := by sorry

end NUMINAMATH_CALUDE_sum_first_10_terms_sequence_is_constant_sum_equals_first_term_times_n_l550_55064


namespace NUMINAMATH_CALUDE_lunch_cakes_count_l550_55016

def total_cakes : ℕ := 15
def dinner_cakes : ℕ := 9

theorem lunch_cakes_count : total_cakes - dinner_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_lunch_cakes_count_l550_55016


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l550_55060

theorem arithmetic_calculation : (180 / 6) * 2 + 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l550_55060


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l550_55082

theorem subtraction_puzzle :
  ∀ (A B C D E F H I J : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
     F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
     H ≠ I ∧ H ≠ J ∧
     I ≠ J) →
    (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧
    (1 ≤ D ∧ D ≤ 9) ∧ (1 ≤ E ∧ E ≤ 9) ∧ (1 ≤ F ∧ F ≤ 9) ∧
    (1 ≤ H ∧ H ≤ 9) ∧ (1 ≤ I ∧ I ≤ 9) ∧ (1 ≤ J ∧ J ≤ 9) →
    100 * A + 10 * B + C - (100 * D + 10 * E + F) = 100 * H + 10 * I + J →
    A + B + C + D + E + F + H + I + J = 45 →
    A + B + C = 18 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l550_55082


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l550_55044

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l550_55044


namespace NUMINAMATH_CALUDE_cube_plus_one_expansion_problem_solution_l550_55040

theorem cube_plus_one_expansion (n : ℕ) : 
  n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 :=
by sorry

theorem problem_solution : 
  98^3 + 3*(98^2) + 3*98 + 1 = 970299 :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_one_expansion_problem_solution_l550_55040


namespace NUMINAMATH_CALUDE_sum_of_derivatives_positive_l550_55017

def f (x : ℝ) : ℝ := -x^2 - x^4 - x^6

theorem sum_of_derivatives_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  (deriv f x₁) + (deriv f x₂) + (deriv f x₃) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_derivatives_positive_l550_55017


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l550_55007

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  i^255 + i^256 + i^257 + i^258 + i^259 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l550_55007


namespace NUMINAMATH_CALUDE_honey_market_optimization_l550_55094

/-- Represents the honey market in Milnlandia -/
structure HoneyMarket where
  /-- Inverse demand function: P = 310 - 3Q -/
  demand : ℝ → ℝ
  /-- Production cost per jar in milns -/
  cost : ℝ
  /-- Tax per jar in milns -/
  tax : ℝ

/-- Profit function for the honey producer -/
def profit (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  (market.demand quantity) * quantity - market.cost * quantity - market.tax * quantity

/-- Tax revenue function for the government -/
def taxRevenue (market : HoneyMarket) (quantity : ℝ) : ℝ :=
  market.tax * quantity

/-- The statement to be proved -/
theorem honey_market_optimization (market : HoneyMarket) 
    (h_demand : ∀ q, market.demand q = 310 - 3 * q)
    (h_cost : market.cost = 10) :
  (∃ q_max : ℝ, q_max = 50 ∧ 
    ∀ q, profit market q ≤ profit market q_max) ∧
  (∃ t_max : ℝ, t_max = 150 ∧
    ∀ t, market.tax = t → 
      taxRevenue { market with tax := t } 
        ((310 - t) / 6) ≤ 
      taxRevenue { market with tax := t_max } 
        ((310 - t_max) / 6)) := by
  sorry


end NUMINAMATH_CALUDE_honey_market_optimization_l550_55094


namespace NUMINAMATH_CALUDE_triangle_properties_l550_55099

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about the properties of a triangle -/
theorem triangle_properties (t : Triangle) :
  (Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) →
  (2 * t.a^2 = t.b^2 + t.c^2) ∧
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l550_55099


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l550_55076

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 2 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l550_55076


namespace NUMINAMATH_CALUDE_inequality_proof_l550_55042

theorem inequality_proof (a b c : ℝ) 
  (sum_cond : a + b + c = 3)
  (nonzero_cond : (6*a + b^2 + c^2) * (6*b + c^2 + a^2) * (6*c + a^2 + b^2) ≠ 0) :
  a / (6*a + b^2 + c^2) + b / (6*b + c^2 + a^2) + c / (6*c + a^2 + b^2) ≤ 3/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l550_55042


namespace NUMINAMATH_CALUDE_system_solutions_correct_l550_55003

theorem system_solutions_correct : 
  -- System 1
  (∃ (x y : ℝ), x - 2*y = 1 ∧ 3*x + 2*y = 7 ∧ x = 2 ∧ y = 1/2) ∧
  -- System 2
  (∃ (x y : ℝ), x - y = 3 ∧ (x - y - 3)/2 - y/3 = -1 ∧ x = 6 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l550_55003


namespace NUMINAMATH_CALUDE_min_value_a_l550_55004

theorem min_value_a (a b : ℤ) (m : ℕ) (h1 : a - b = m) (h2 : Nat.Prime m) 
  (h3 : ∃ n : ℕ, a * b = n * n) (h4 : a ≥ 2012) : 
  (∀ a' b' : ℤ, ∃ m' : ℕ, a' - b' = m' ∧ Nat.Prime m' ∧ (∃ n' : ℕ, a' * b' = n' * n') ∧ a' ≥ 2012 → a' ≥ a) ∧ 
  a = 2025 := by
sorry


end NUMINAMATH_CALUDE_min_value_a_l550_55004


namespace NUMINAMATH_CALUDE_parabola_directrix_l550_55038

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem: Given a parabola x^2 = 2py (p > 0) intersected by a line with slope 1 at points A and B,
    if the x-coordinate of the midpoint of AB is 2, then the equation of the directrix is y = -1 -/
theorem parabola_directrix (par : Parabola) (A B : Point) :
  (A.x^2 = 2 * par.p * A.y) →
  (B.x^2 = 2 * par.p * B.y) →
  (B.y - A.y = B.x - A.x) →
  ((A.x + B.x) / 2 = 2) →
  (∀ (x y : ℝ), y = -1 ↔ y = -par.p / 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l550_55038


namespace NUMINAMATH_CALUDE_base5_20314_equals_1334_l550_55085

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ (digits.length - 1 - i))) 0

theorem base5_20314_equals_1334 :
  base5_to_base10 [2, 0, 3, 1, 4] = 1334 := by
  sorry

end NUMINAMATH_CALUDE_base5_20314_equals_1334_l550_55085


namespace NUMINAMATH_CALUDE_lucys_cookies_l550_55093

/-- Lucy's grocery shopping problem -/
theorem lucys_cookies (total_packs cake_packs cookie_packs : ℕ) : 
  total_packs = 27 → cake_packs = 4 → total_packs = cookie_packs + cake_packs → cookie_packs = 23 := by
  sorry

end NUMINAMATH_CALUDE_lucys_cookies_l550_55093


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l550_55067

theorem simplify_and_evaluate (x y : ℝ) 
  (hx : x = 2 + 3 * Real.sqrt 3) 
  (hy : y = 2 - 3 * Real.sqrt 3) : 
  (x^2 / (x - y)) - (y^2 / (x - y)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l550_55067


namespace NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l550_55019

theorem chocolate_bars_in_large_box : 
  ∀ (num_small_boxes : ℕ) (bars_per_small_box : ℕ),
    num_small_boxes = 15 →
    bars_per_small_box = 20 →
    num_small_boxes * bars_per_small_box = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_in_large_box_l550_55019


namespace NUMINAMATH_CALUDE_rem_prime_specific_value_l550_55069

/-- Modified remainder function -/
def rem' (x y : ℚ) : ℚ := x - y * ⌊x / (2 * y)⌋

/-- Theorem stating the value of rem'(5/9, -3/7) -/
theorem rem_prime_specific_value : rem' (5/9) (-3/7) = 62/63 := by
  sorry

end NUMINAMATH_CALUDE_rem_prime_specific_value_l550_55069


namespace NUMINAMATH_CALUDE_project_work_time_difference_l550_55025

/-- Given three people working on a project with their working times in the ratio of 3:5:6,
    and a total project time of 140 hours, prove that the difference between the working hours
    of the person who worked the most and the person who worked the least is 30 hours. -/
theorem project_work_time_difference :
  ∀ (x : ℝ), 
  (3 * x + 5 * x + 6 * x = 140) →
  (6 * x - 3 * x = 30) :=
by sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l550_55025


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l550_55012

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : n > m) :
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l550_55012


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l550_55079

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    imaginary axis length of 4, and focal distance of 4√3,
    prove that its asymptotes are given by y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imaginary_axis : b = 2) 
  (h_focal_distance : 2 * Real.sqrt ((a^2 + b^2) : ℝ) = 4 * Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_l550_55079


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l550_55027

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (3 - x) + Real.sqrt (x - 2) = 2) ↔ (x = 3/4 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l550_55027


namespace NUMINAMATH_CALUDE_wizard_hat_theorem_l550_55062

/-- Represents a strategy for the wizard hat problem -/
def Strategy : Type := Unit

/-- Represents the outcome of applying a strategy -/
def Outcome (n : ℕ) : Type := Fin n → Bool

/-- A wizard can see hats in front but not their own -/
axiom can_see_forward (n : ℕ) (i : Fin n) : ∀ j : Fin n, i < j → Prop

/-- Each wizard says a unique number between 1 and 1001 -/
axiom unique_numbers (n : ℕ) (outcome : Outcome n) : 
  ∀ i j : Fin n, i ≠ j → outcome i ≠ outcome j

/-- Wizards speak from back to front -/
axiom speak_order (n : ℕ) (i j : Fin n) : i < j → Prop

/-- Applying a strategy produces an outcome -/
def apply_strategy (n : ℕ) (s : Strategy) : Outcome n := sorry

/-- Counts the number of correct identifications in an outcome -/
def count_correct (n : ℕ) (outcome : Outcome n) : ℕ := sorry

theorem wizard_hat_theorem (n : ℕ) (h : n > 1000) :
  ∃ (s : Strategy), 
    (count_correct n (apply_strategy n s) > 500) ∧ 
    (count_correct n (apply_strategy n s) ≥ 999) := by
  sorry

end NUMINAMATH_CALUDE_wizard_hat_theorem_l550_55062


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l550_55034

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3)) →
  A + B = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l550_55034


namespace NUMINAMATH_CALUDE_fraction_equality_l550_55041

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7^2) :
  d / a = 1 / 122.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l550_55041


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l550_55063

theorem sum_of_squares_theorem (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l550_55063


namespace NUMINAMATH_CALUDE_frustum_height_theorem_l550_55018

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  -- Height of the original pyramid
  h : ℝ
  -- Height of the smaller pyramid (cut off part)
  h_small : ℝ
  -- Ratio of upper to lower base areas
  area_ratio : ℝ

/-- The height of the frustum in a cut pyramid -/
def frustum_height (p : CutPyramid) : ℝ := p.h - p.h_small

/-- Theorem: If the ratio of upper to lower base areas is 1:4 and the height of the smaller pyramid is 3,
    then the height of the frustum is 3 -/
theorem frustum_height_theorem (p : CutPyramid) 
  (h_ratio : p.area_ratio = 1 / 4)
  (h_small : p.h_small = 3) :
  frustum_height p = 3 := by
  sorry

#check frustum_height_theorem

end NUMINAMATH_CALUDE_frustum_height_theorem_l550_55018


namespace NUMINAMATH_CALUDE_composite_numbers_1991_l550_55023

theorem composite_numbers_1991 : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = 1991^1991 + 1) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ c * d = 1991^1991 - 1) := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_1991_l550_55023


namespace NUMINAMATH_CALUDE_star_operation_divisors_l550_55035

-- Define the star operation
def star (a b : ℤ) : ℚ := (a^2 : ℚ) / b

-- Define the count of positive integer divisors of a number
def countPositiveDivisors (n : ℕ) : ℕ := sorry

-- Define the count of integer x for which (20 ★ x) is a positive integer
def countValidX : ℕ := sorry

-- Theorem statement
theorem star_operation_divisors : 
  countPositiveDivisors 400 = countValidX := by sorry

end NUMINAMATH_CALUDE_star_operation_divisors_l550_55035


namespace NUMINAMATH_CALUDE_problem_solution_l550_55010

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem problem_solution (a b : ℝ) :
  (A ∪ B a b = Set.univ) ∧ 
  (A ∩ B a b = Set.Ioc 3 4) →
  a = -3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l550_55010


namespace NUMINAMATH_CALUDE_cara_in_middle_groups_l550_55074

theorem cara_in_middle_groups (n : ℕ) (h : n = 6) : Nat.choose n 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cara_in_middle_groups_l550_55074


namespace NUMINAMATH_CALUDE_treehouse_planks_l550_55008

/-- The total number of planks Charlie and his father have -/
def total_planks (initial_planks charlie_planks father_planks : ℕ) : ℕ :=
  initial_planks + charlie_planks + father_planks

/-- Theorem stating that the total number of planks is 35 -/
theorem treehouse_planks : total_planks 15 10 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l550_55008


namespace NUMINAMATH_CALUDE_line_through_point_l550_55091

/-- Given a line 3x + ay - 5 = 0 that passes through the point (1, 2), prove that a = 1 --/
theorem line_through_point (a : ℝ) : (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l550_55091


namespace NUMINAMATH_CALUDE_asima_integer_possibilities_l550_55072

theorem asima_integer_possibilities (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : 4 * (2 * a - 10) + 4 * (2 * b - 10) = 440) :
  ∃ (n : ℕ), n = 64 ∧ (∀ x : ℕ, x > 0 ∧ x ≤ n → ∃ y : ℕ, y > 0 ∧ 4 * (2 * x - 10) + 4 * (2 * y - 10) = 440) :=
sorry

end NUMINAMATH_CALUDE_asima_integer_possibilities_l550_55072


namespace NUMINAMATH_CALUDE_joan_football_games_l550_55048

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to this year and last year -/
def total_games : ℕ := 9

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 5 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l550_55048


namespace NUMINAMATH_CALUDE_exist_point_W_l550_55026

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₂, y₂) := Y
  let (x₃, y₃) := Z
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 10^2 ∧
  (x₂ - x₃)^2 + (y₂ - y₃)^2 = 11^2 ∧
  (x₁ - x₃)^2 + (y₁ - y₃)^2 = 12^2

-- Define point P on XZ
def PointP (X Z P : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (x₃, y₃) := Z
  let (xp, yp) := P
  (xp - x₃)^2 + (yp - y₃)^2 = 6^2 ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = t * x₁ + (1 - t) * x₃ ∧ yp = t * y₁ + (1 - t) * y₃

-- Define point W on line PY
def PointW (Y P W : ℝ × ℝ) : Prop :=
  let (x₂, y₂) := Y
  let (xp, yp) := P
  let (xw, yw) := W
  ∃ t : ℝ, xw = t * xp + (1 - t) * x₂ ∧ yw = t * yp + (1 - t) * y₂

-- Define XW parallel to ZY
def Parallel (X W Z Y : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := X
  let (xw, yw) := W
  let (x₃, y₃) := Z
  let (x₂, y₂) := Y
  (xw - x₁) * (y₃ - y₂) = (yw - y₁) * (x₃ - x₂)

-- Define cyclic hexagon
def CyclicHexagon (Y X Y Z W X : ℝ × ℝ) : Prop :=
  -- This is a simplified definition, as the full condition for a cyclic hexagon is complex
  -- In reality, we would need to check if all six points lie on a circle
  true

-- Main theorem
theorem exist_point_W (X Y Z P : ℝ × ℝ) :
  Triangle X Y Z →
  PointP X Z P →
  ∃ W : ℝ × ℝ,
    PointW Y P W ∧
    Parallel X W Z Y ∧
    CyclicHexagon Y X Y Z W X ∧
    let (xp, yp) := P
    let (xw, yw) := W
    (xw - xp)^2 + (yw - yp)^2 = 10^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_point_W_l550_55026


namespace NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l550_55089

theorem pascal_triangle_51_numbers (n : ℕ) : 
  n = 50 → (n.choose 4) = 230150 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_numbers_l550_55089


namespace NUMINAMATH_CALUDE_grid_division_l550_55046

/-- Represents a cell in the grid -/
inductive Cell
| Shaded
| Unshaded

/-- Represents the 6x6 grid -/
def Grid := Matrix (Fin 6) (Fin 6) Cell

/-- Counts the number of shaded cells in a given region of the grid -/
def count_shaded (g : Grid) (start_row end_row start_col end_col : Fin 6) : Nat :=
  sorry

/-- Checks if a given 3x3 region of the grid contains exactly 3 shaded cells -/
def is_valid_part (g : Grid) (start_row start_col : Fin 6) : Prop :=
  count_shaded g start_row (start_row + 2) start_col (start_col + 2) = 3

/-- The main theorem to be proved -/
theorem grid_division (g : Grid) 
  (h1 : count_shaded g 0 5 0 5 = 12) : 
  (is_valid_part g 0 0) ∧ 
  (is_valid_part g 0 3) ∧ 
  (is_valid_part g 3 0) ∧ 
  (is_valid_part g 3 3) :=
sorry

end NUMINAMATH_CALUDE_grid_division_l550_55046


namespace NUMINAMATH_CALUDE_library_average_disk_space_per_hour_l550_55014

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  diskSpace : ℕ

/-- Calculates the average disk space usage per hour for a given music library -/
def averageDiskSpacePerHour (library : MusicLibrary) : ℚ :=
  library.diskSpace / (library.days * 24)

/-- Theorem stating that for the given library, the average disk space per hour is 50 MB -/
theorem library_average_disk_space_per_hour :
  let library : MusicLibrary := { days := 15, diskSpace := 18000 }
  averageDiskSpacePerHour library = 50 := by
  sorry

end NUMINAMATH_CALUDE_library_average_disk_space_per_hour_l550_55014


namespace NUMINAMATH_CALUDE_distance_from_origin_l550_55032

theorem distance_from_origin (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l550_55032


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l550_55054

theorem cousins_ages_sum : ∃ (a b c d : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧  -- single-digit
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧      -- positive
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧  -- distinct
  ((a * b = 20 ∧ c * d = 21) ∨ (a * c = 20 ∧ b * d = 21) ∨ 
   (a * d = 20 ∧ b * c = 21) ∨ (b * c = 20 ∧ a * d = 21) ∨ 
   (b * d = 20 ∧ a * c = 21) ∧ (c * d = 20 ∧ a * b = 21)) ∧
  (a + b + c + d = 19) :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l550_55054


namespace NUMINAMATH_CALUDE_expression_evaluation_l550_55029

theorem expression_evaluation :
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 8) / (x - 4)
  f 5 = 7 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l550_55029


namespace NUMINAMATH_CALUDE_equal_to_2x_6_l550_55049

theorem equal_to_2x_6 (x : ℝ) : 2 * x^7 / x = 2 * x^6 := by sorry

end NUMINAMATH_CALUDE_equal_to_2x_6_l550_55049


namespace NUMINAMATH_CALUDE_divisor_problem_l550_55037

theorem divisor_problem (n : ℤ) : ∃ (d : ℤ), d = 22 ∧ ∃ (k : ℤ), n = k * d + 12 ∧ ∃ (m : ℤ), 2 * n = 11 * m + 2 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l550_55037


namespace NUMINAMATH_CALUDE_exam_score_proof_l550_55057

/-- Given an exam with mean score 76, prove that the score 2 standard deviations
    below the mean is 60, knowing that 100 is 3 standard deviations above the mean. -/
theorem exam_score_proof (mean : ℝ) (score_above : ℝ) (std_dev_above : ℝ) (std_dev_below : ℝ) :
  mean = 76 →
  score_above = 100 →
  std_dev_above = 3 →
  std_dev_below = 2 →
  score_above = mean + std_dev_above * ((score_above - mean) / std_dev_above) →
  mean - std_dev_below * ((score_above - mean) / std_dev_above) = 60 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_proof_l550_55057


namespace NUMINAMATH_CALUDE_matrix_power_2023_l550_55031

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l550_55031


namespace NUMINAMATH_CALUDE_each_child_gets_twenty_cookies_l550_55009

/-- Represents the cookie distribution problem in Everlee's family -/
def cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  let adults_share := total_cookies / 3
  let remaining_cookies := total_cookies - adults_share
  remaining_cookies / num_children

/-- Theorem stating that each child gets 20 cookies -/
theorem each_child_gets_twenty_cookies :
  cookie_distribution 120 2 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_each_child_gets_twenty_cookies_l550_55009


namespace NUMINAMATH_CALUDE_initial_customers_count_l550_55087

/-- The number of customers who left -/
def customers_left : ℕ := 5

/-- The number of customers remaining -/
def customers_remaining : ℕ := 9

/-- The initial number of customers -/
def initial_customers : ℕ := customers_left + customers_remaining

theorem initial_customers_count : initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_customers_count_l550_55087


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l550_55090

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000 ∧ m % 5 = 0) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l550_55090


namespace NUMINAMATH_CALUDE_equation_solution_l550_55083

theorem equation_solution : ∃ x : ℝ, 
  (Real.sqrt (7 * x - 3) + Real.sqrt (2 * x - 2) = 5) ∧ 
  (7 * x - 3 ≥ 0) ∧ 
  (2 * x - 2 ≥ 0) ∧ 
  (abs (x - 20.14) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l550_55083


namespace NUMINAMATH_CALUDE_tiktok_twitter_ratio_l550_55059

/-- Represents the number of followers on different social media platforms --/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms --/
def total_followers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating the relationship between TikTok and Twitter followers --/
theorem tiktok_twitter_ratio (f : Followers) (x : ℕ) : 
  f.instagram = 240 →
  f.facebook = 500 →
  f.twitter = (f.instagram + f.facebook) / 2 →
  f.tiktok = x * f.twitter →
  f.youtube = f.tiktok + 510 →
  total_followers f = 3840 →
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_tiktok_twitter_ratio_l550_55059


namespace NUMINAMATH_CALUDE_danny_fish_tank_theorem_l550_55011

/-- The number of fish remaining after selling some from Danny's fish tank. -/
def remaining_fish (initial_guppies initial_angelfish initial_tiger_sharks initial_oscar_fish
                    sold_guppies sold_angelfish sold_tiger_sharks sold_oscar_fish : ℕ) : ℕ :=
  (initial_guppies - sold_guppies) +
  (initial_angelfish - sold_angelfish) +
  (initial_tiger_sharks - sold_tiger_sharks) +
  (initial_oscar_fish - sold_oscar_fish)

/-- Theorem stating the number of remaining fish in Danny's tank. -/
theorem danny_fish_tank_theorem :
  remaining_fish 94 76 89 58 30 48 17 24 = 198 := by
  sorry

end NUMINAMATH_CALUDE_danny_fish_tank_theorem_l550_55011


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l550_55073

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- a, b, c are positive
  c = 2 * b - a →          -- a, b, c form an arithmetic sequence
  a * b * c = 125 →        -- product condition
  b ≥ 5 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    c' = 2 * b' - a' ∧ a' * b' * c' = 125 ∧ b' = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l550_55073


namespace NUMINAMATH_CALUDE_same_wage_proportional_earnings_l550_55065

/-- Proves that maintaining the same hourly wage and weekly hours results in proportional earnings -/
theorem same_wage_proportional_earnings
  (seasonal_weeks : ℕ)
  (seasonal_earnings : ℝ)
  (new_weeks : ℕ)
  (new_earnings : ℝ)
  (h_seasonal_weeks : seasonal_weeks = 36)
  (h_seasonal_earnings : seasonal_earnings = 7200)
  (h_new_weeks : new_weeks = 18)
  (h_new_earnings : new_earnings = 3600)
  : (new_earnings / new_weeks) = (seasonal_earnings / seasonal_weeks) :=
by sorry

end NUMINAMATH_CALUDE_same_wage_proportional_earnings_l550_55065


namespace NUMINAMATH_CALUDE_cone_volume_l550_55071

/-- Given a cone with base radius 1 and lateral area 2π, its volume is (√3/3)π -/
theorem cone_volume (r h : ℝ) : 
  r = 1 → 
  π * r * (r^2 + h^2).sqrt = 2 * π → 
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l550_55071


namespace NUMINAMATH_CALUDE_fraction_evaluation_l550_55068

theorem fraction_evaluation : (2 + 1/2) / (1 - 3/4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l550_55068


namespace NUMINAMATH_CALUDE_increasing_function_range_increasing_function_and_hyperbola_range_l550_55045

/-- The function f(x) = x² + (a-1)x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x

/-- The property that f is increasing on (1, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → x < y → f a x < f a y

/-- The equation x² - ay² = 1 represents a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x^2 - a*y^2 = 1

theorem increasing_function_range (a : ℝ) :
  is_increasing_on_interval a → a > -1 :=
sorry

theorem increasing_function_and_hyperbola_range (a : ℝ) :
  is_increasing_on_interval a → is_hyperbola a → a > 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_range_increasing_function_and_hyperbola_range_l550_55045


namespace NUMINAMATH_CALUDE_players_satisfy_distances_l550_55084

/-- Represents the positions of four players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two player positions -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required distances between players -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem players_satisfy_distances : 
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_players_satisfy_distances_l550_55084


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l550_55036

/-- The sum of the first n terms of a sequence -/
def S (n : ℕ) : ℝ := 3 * n^2

/-- The n-th term of the sequence -/
def u (n : ℕ) : ℝ := S n - S (n-1)

theorem sequence_is_arithmetic_progression :
  ∃ (a d : ℝ), ∀ n : ℕ, u n = a + (n - 1) * d :=
sorry

theorem first_term_is_three : u 1 = 3 :=
sorry

theorem common_difference_is_six :
  ∀ n : ℕ, n > 1 → u n - u (n-1) = 6 :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_progression_first_term_is_three_common_difference_is_six_l550_55036


namespace NUMINAMATH_CALUDE_chef_potato_problem_chef_potato_solution_l550_55058

theorem chef_potato_problem (already_cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  let remaining_potatoes := remaining_cooking_time / cooking_time_per_potato
  let total_potatoes := already_cooked + remaining_potatoes
  total_potatoes

#check chef_potato_problem 8 9 63

theorem chef_potato_solution :
  chef_potato_problem 8 9 63 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_chef_potato_solution_l550_55058


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l550_55000

theorem smallest_value_theorem (v w : ℝ) 
  (h : ∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w) : 
  (∀ (v' w' : ℝ), (∀ (a b : ℝ), (2^(a+b) + 8)*(3^a + 3^b) ≤ v'*(12^(a-1) + 12^(b-1) - 2^(a+b-1)) + w') → 
    128*v^2 + w^2 ≤ 128*v'^2 + w'^2) ∧ 128*v^2 + w^2 = 62208 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l550_55000


namespace NUMINAMATH_CALUDE_goods_train_speed_l550_55013

theorem goods_train_speed 
  (man_train_speed : ℝ) 
  (goods_train_length : ℝ) 
  (passing_time : ℝ) 
  (h1 : man_train_speed = 70) 
  (h2 : goods_train_length = 0.28) 
  (h3 : passing_time = 9 / 3600) : 
  ∃ (goods_train_speed : ℝ), 
    goods_train_speed = 42 ∧ 
    (goods_train_speed + man_train_speed) * passing_time = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l550_55013


namespace NUMINAMATH_CALUDE_fifth_week_hours_l550_55096

-- Define the required average hours per week
def required_average : ℝ := 12

-- Define the number of weeks
def num_weeks : ℕ := 5

-- Define the study hours for the first four weeks
def week1_hours : ℝ := 10
def week2_hours : ℝ := 14
def week3_hours : ℝ := 9
def week4_hours : ℝ := 13

-- Define the sum of study hours for the first four weeks
def sum_first_four_weeks : ℝ := week1_hours + week2_hours + week3_hours + week4_hours

-- Theorem to prove
theorem fifth_week_hours : 
  ∃ (x : ℝ), (sum_first_four_weeks + x) / num_weeks = required_average ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_week_hours_l550_55096


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l550_55077

/-- The cost price of one metre of cloth given the selling price, quantity, and profit per metre -/
theorem cost_price_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℚ) 
  (profit_per_metre : ℚ) 
  (h1 : total_metres = 85)
  (h2 : total_selling_price = 8925)
  (h3 : profit_per_metre = 15) :
  (total_selling_price - total_metres * profit_per_metre) / total_metres = 90 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l550_55077


namespace NUMINAMATH_CALUDE_star_properties_l550_55020

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem statement
theorem star_properties :
  (∀ x ∈ T, star x (-1) ≠ x ∨ star (-1) x ≠ x) ∧
  (star 1 (-1/2) = -1 ∧ star (-1/2) 1 = -1) :=
sorry

end NUMINAMATH_CALUDE_star_properties_l550_55020


namespace NUMINAMATH_CALUDE_fred_final_balloons_l550_55078

def fred_balloons : ℕ → Prop
| n => ∃ (initial given received distributed : ℕ),
  initial = 1457 ∧
  given = 341 ∧
  received = 225 ∧
  distributed = ((initial - given + received) / 2) ∧
  n = initial - given + received - distributed

theorem fred_final_balloons : fred_balloons 671 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_balloons_l550_55078


namespace NUMINAMATH_CALUDE_exist_x_y_sequences_l550_55050

def sequence_a : ℕ → ℚ
  | 0 => 4
  | 1 => 22
  | (n + 2) => 6 * sequence_a (n + 1) - sequence_a n

theorem exist_x_y_sequences :
  ∃ (x y : ℕ → ℕ), ∀ n, 
    sequence_a n = (y n ^ 2 + 7 : ℚ) / (x n - y n : ℚ) ∧
    x n > y n ∧ 
    x n > 0 ∧ 
    y n > 0 :=
by sorry

end NUMINAMATH_CALUDE_exist_x_y_sequences_l550_55050


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l550_55022

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (4 * x + 15 - 6) = 12 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l550_55022


namespace NUMINAMATH_CALUDE_percentage_to_new_school_l550_55086

theorem percentage_to_new_school (total_students : ℕ) 
  (percent_to_A : ℚ) (percent_to_B : ℚ) 
  (percent_A_to_C : ℚ) (percent_B_to_C : ℚ) :
  percent_to_A = 60 / 100 →
  percent_to_B = 40 / 100 →
  percent_A_to_C = 30 / 100 →
  percent_B_to_C = 40 / 100 →
  let students_A := (percent_to_A * total_students).floor
  let students_B := (percent_to_B * total_students).floor
  let students_A_to_C := (percent_A_to_C * students_A).floor
  let students_B_to_C := (percent_B_to_C * students_B).floor
  let total_to_C := students_A_to_C + students_B_to_C
  ((total_to_C : ℚ) / total_students * 100).floor = 34 := by
sorry

end NUMINAMATH_CALUDE_percentage_to_new_school_l550_55086


namespace NUMINAMATH_CALUDE_even_odd_difference_3000_l550_55095

/-- Sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- Sum of the first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers and the sum of the first n odd numbers -/
def evenOddDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem even_odd_difference_3000 : evenOddDifference 3000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_difference_3000_l550_55095


namespace NUMINAMATH_CALUDE_matrix_power_4_l550_55021

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_4 : A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l550_55021


namespace NUMINAMATH_CALUDE_total_wristbands_distributed_l550_55088

/-- Represents the number of wristbands given to each spectator -/
def wristbands_per_spectator : ℕ := 2

/-- Represents the total number of wristbands distributed -/
def total_wristbands : ℕ := 125

/-- Theorem stating that the total number of wristbands distributed is 125 -/
theorem total_wristbands_distributed :
  total_wristbands = 125 := by sorry

end NUMINAMATH_CALUDE_total_wristbands_distributed_l550_55088


namespace NUMINAMATH_CALUDE_prob_select_AB_l550_55097

/-- The number of employees -/
def total_employees : ℕ := 4

/-- The number of employees to be selected -/
def selected_employees : ℕ := 2

/-- The probability of selecting at least one of A and B -/
def prob_at_least_one_AB : ℚ := 5/6

/-- Theorem stating the probability of selecting at least one of A and B -/
theorem prob_select_AB : 
  1 - (Nat.choose (total_employees - 2) selected_employees : ℚ) / (Nat.choose total_employees selected_employees : ℚ) = prob_at_least_one_AB :=
sorry

end NUMINAMATH_CALUDE_prob_select_AB_l550_55097


namespace NUMINAMATH_CALUDE_subset_condition_l550_55070

def A (x : ℝ) : Prop := |2 * x - 1| < 1

def B (a x : ℝ) : Prop := x^2 - 2*a*x + a^2 - 1 > 0

theorem subset_condition (a : ℝ) :
  (∀ x, A x → B a x) ↔ (a ≤ -1 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_l550_55070


namespace NUMINAMATH_CALUDE_polynomial_remainder_l550_55006

/-- Given a polynomial Q(x) such that Q(17) = 53 and Q(53) = 17,
    the remainder when Q(x) is divided by (x - 17)(x - 53) is -x + 70 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 53) (h2 : Q 53 = 17) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 53) * R x + (-x + 70) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l550_55006


namespace NUMINAMATH_CALUDE_first_chapter_pages_l550_55047

/-- Represents a book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- Theorem stating the number of pages in the first chapter of the book -/
theorem first_chapter_pages (b : Book) 
  (h1 : b.chapter2_pages = 11) 
  (h2 : b.chapter1_pages = b.chapter2_pages + 37) : 
  b.chapter1_pages = 48 := by
  sorry

end NUMINAMATH_CALUDE_first_chapter_pages_l550_55047


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l550_55028

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem statement
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : EvenFunction f) :
  EvenFunction (f ∘ f) := by
sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l550_55028


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l550_55056

theorem tan_theta_two_implies_expression_equals_negative_two (θ : Real) 
  (h : Real.tan θ = 2) : 
  (2 * Real.cos θ) / (Real.sin (π/2 + θ) + Real.sin (π + θ)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_negative_two_l550_55056


namespace NUMINAMATH_CALUDE_sector_arc_length_l550_55001

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (L : ℝ) : 
  r = 1 → θ_deg = 60 → L = r * (θ_deg * π / 180) → L = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l550_55001


namespace NUMINAMATH_CALUDE_hazel_lemonade_cups_l550_55098

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel gave to her friends -/
def cups_given_to_friends : ℕ := cups_sold_to_kids / 2

/-- The number of cups of lemonade Hazel drank herself -/
def cups_drunk_by_hazel : ℕ := 1

/-- The total number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  2 * (cups_sold_to_kids + cups_given_to_friends + cups_drunk_by_hazel) = total_cups := by
  sorry

#check hazel_lemonade_cups

end NUMINAMATH_CALUDE_hazel_lemonade_cups_l550_55098


namespace NUMINAMATH_CALUDE_contest_scores_l550_55061

theorem contest_scores (n k : ℕ) (hn : n ≥ 2) :
  (∀ (i : ℕ), i ≤ k → ∃! (f : ℕ → ℕ), (∀ x, x ≤ n → f x ≤ n) ∧ 
    (∀ x y, x ≠ y → f x ≠ f y) ∧ (Finset.sum (Finset.range n) f = Finset.sum (Finset.range n) id)) →
  (∀ x, x ≤ n → k * (Finset.sum (Finset.range n) id) = 26 * n) →
  (n = 25 ∧ k = 2) ∨ (n = 12 ∧ k = 4) ∨ (n = 3 ∧ k = 13) :=
by sorry

end NUMINAMATH_CALUDE_contest_scores_l550_55061


namespace NUMINAMATH_CALUDE_function_fits_data_l550_55024

def f (x : ℝ) : ℝ := 210 - 10*x - x^2 - 2*x^3

theorem function_fits_data : 
  (f 0 = 210) ∧ 
  (f 2 = 170) ∧ 
  (f 4 = 110) ∧ 
  (f 6 = 30) ∧ 
  (f 8 = -70) := by
  sorry

end NUMINAMATH_CALUDE_function_fits_data_l550_55024


namespace NUMINAMATH_CALUDE_students_attending_game_l550_55005

/-- Proves the number of students attending a football game -/
theorem students_attending_game (total_attendees : ℕ) (student_price non_student_price : ℕ) (total_revenue : ℕ) : 
  total_attendees = 3000 →
  student_price = 10 →
  non_student_price = 15 →
  total_revenue = 36250 →
  ∃ (students non_students : ℕ),
    students + non_students = total_attendees ∧
    students * student_price + non_students * non_student_price = total_revenue ∧
    students = 1750 :=
by sorry

end NUMINAMATH_CALUDE_students_attending_game_l550_55005


namespace NUMINAMATH_CALUDE_non_officers_count_l550_55075

/-- Proves that the number of non-officers is 525 given the salary information --/
theorem non_officers_count (total_avg : ℝ) (officer_avg : ℝ) (non_officer_avg : ℝ) (officer_count : ℕ) :
  total_avg = 120 →
  officer_avg = 470 →
  non_officer_avg = 110 →
  officer_count = 15 →
  ∃ (non_officer_count : ℕ),
    (officer_count * officer_avg + non_officer_count * non_officer_avg) / (officer_count + non_officer_count) = total_avg ∧
    non_officer_count = 525 :=
by sorry

end NUMINAMATH_CALUDE_non_officers_count_l550_55075


namespace NUMINAMATH_CALUDE_work_completion_equality_prove_new_group_size_l550_55002

/-- The number of persons in the original group -/
def original_group : ℕ := 15

/-- The number of days the original group takes to complete the work -/
def original_days : ℕ := 18

/-- The fraction of work done by the new group -/
def new_group_work_fraction : ℚ := 1/3

/-- The number of days the new group takes to complete their fraction of work -/
def new_group_days : ℕ := 21

/-- The multiplier for the number of persons in the new group -/
def new_group_multiplier : ℚ := 5/2

/-- The number of persons in the new group -/
def new_group_size : ℕ := 7

theorem work_completion_equality :
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * new_group_size) * new_group_work_fraction / new_group_days :=
by sorry

/-- The main theorem proving the size of the new group -/
theorem prove_new_group_size : 
  ∃ (n : ℕ), n = new_group_size ∧
  (original_group : ℚ) / original_days = 
  (new_group_multiplier * n) * new_group_work_fraction / new_group_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_prove_new_group_size_l550_55002
