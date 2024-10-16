import Mathlib

namespace NUMINAMATH_CALUDE_find_natural_number_A_l3850_385027

theorem find_natural_number_A : ∃ A : ℕ, 
  A > 0 ∧ 
  312 % A = 2 * (270 % A) ∧ 
  270 % A = 2 * (211 % A) ∧ 
  A = 19 := by
  sorry

end NUMINAMATH_CALUDE_find_natural_number_A_l3850_385027


namespace NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3850_385016

/-- Given a point P on the terminal side of angle 4π/3 with |OP| = 4,
    prove that the coordinates of P are (-2, -2√3) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = 4 * Real.cos (4 * Real.pi / 3) ∧ P.2 = 4 * Real.sin (4 * Real.pi / 3)) →
  P = (-2, -2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l3850_385016


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3850_385050

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
structure CircleConfiguration where
  O₁ : Circle
  O₂ : Circle
  O : Circle
  h₁ : O₁.radius ≠ O₂.radius
  h₂ : O₁.center ≠ O₂.center
  h₃ : ∀ p : ℝ × ℝ, (dist p O₁.center ≠ O₁.radius) ∨ (dist p O₂.center ≠ O₂.radius)
  h₄ : dist O.center O₁.center = O.radius + O₁.radius ∨ dist O.center O₁.center = abs (O.radius - O₁.radius)
  h₅ : dist O.center O₂.center = O.radius + O₂.radius ∨ dist O.center O₂.center = abs (O.radius - O₂.radius)

-- Define the trajectory types
inductive TrajectoryType
  | Hyperbola
  | Ellipse

-- State the theorem
theorem trajectory_of_moving_circle (config : CircleConfiguration) :
  ∃ t : TrajectoryType, t = TrajectoryType.Hyperbola ∨ t = TrajectoryType.Ellipse :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l3850_385050


namespace NUMINAMATH_CALUDE_line_segment_length_l3850_385077

theorem line_segment_length : Real.sqrt ((8 - 3)^2 + (16 - 4)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l3850_385077


namespace NUMINAMATH_CALUDE_oranges_used_proof_l3850_385063

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 12 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ)
  (h1 : oranges_per_glass = 2)
  (h2 : glasses = 6) :
  total_oranges oranges_per_glass glasses = 12 := by
  sorry

end NUMINAMATH_CALUDE_oranges_used_proof_l3850_385063


namespace NUMINAMATH_CALUDE_equal_inclination_l3850_385043

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf : Continuous f ∧ ContinuousDeriv f)

-- Define points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define that A and B are on the curve
variable (hA : A.2 = f A.1)
variable (hB : B.2 = f B.1)

-- Define that P is on the curve
variable (hP : P.2 = f P.1)

-- Define that P is between A and B
variable (hAP : A.1 ≤ P.1)
variable (hPB : P.1 ≤ B.1)

-- Define that the arc AB is concave to the chord AB
variable (hConcave : ∀ x ∈ Set.Icc A.1 B.1, f x ≤ (B.2 - A.2) / (B.1 - A.1) * (x - A.1) + A.2)

-- Define that AP + PB is maximal at P
variable (hMaximal : ∀ Q : ℝ × ℝ, Q.2 = f Q.1 → A.1 ≤ Q.1 → Q.1 ≤ B.1 → 
  Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) + Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) ≤
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2))

-- State the theorem
theorem equal_inclination :
  let tangent_slope := deriv f P.1
  let PA_slope := (A.2 - P.2) / (A.1 - P.1)
  let PB_slope := (B.2 - P.2) / (B.1 - P.1)
  abs ((tangent_slope - PA_slope) / (1 + tangent_slope * PA_slope)) =
  abs ((PB_slope - tangent_slope) / (1 + PB_slope * tangent_slope)) :=
by sorry

end NUMINAMATH_CALUDE_equal_inclination_l3850_385043


namespace NUMINAMATH_CALUDE_tile_arrangements_l3850_385037

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 2
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / (Nat.factorial brown_tiles * Nat.factorial purple_tiles * Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 630 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l3850_385037


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3850_385054

def A : Set ℝ := {x : ℝ | |x| ≤ 1}
def B : Set ℝ := {x : ℝ | x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3850_385054


namespace NUMINAMATH_CALUDE_larger_circle_radius_l3850_385028

-- Define the radii of the three inner circles
def r₁ : ℝ := 2
def r₂ : ℝ := 3
def r₃ : ℝ := 10

-- Define the centers of the three inner circles
variable (A B C : ℝ × ℝ)

-- Define the center and radius of the larger circle
variable (O : ℝ × ℝ)
variable (R : ℝ)

-- Define the condition that all circles are touching one another
def circles_touching (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist A B = r₁ + r₂) ∧ (dist B C = r₂ + r₃) ∧ (dist A C = r₁ + r₃)

-- Define the condition that the larger circle contains the three inner circles
def larger_circle_contains (O : ℝ × ℝ) (R : ℝ) (A B C : ℝ × ℝ) (r₁ r₂ r₃ : ℝ) : Prop :=
  (dist O A = R - r₁) ∧ (dist O B = R - r₂) ∧ (dist O C = R - r₃)

-- The main theorem
theorem larger_circle_radius 
  (h₁ : circles_touching A B C r₁ r₂ r₃)
  (h₂ : larger_circle_contains O R A B C r₁ r₂ r₃) :
  R = 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l3850_385028


namespace NUMINAMATH_CALUDE_function_inequality_l3850_385020

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo (-π/2) (π/2), (deriv f x) * cos x + f x * sin x > 0) :
  Real.sqrt 2 * f (-π/3) < f (-π/4) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3850_385020


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3850_385095

theorem sqrt_equation_solution (a b : ℝ) : 
  Real.sqrt (a - 5) + Real.sqrt (5 - a) = b + 3 → 
  a = 5 ∧ (Real.sqrt (a^2 - b^2) = 4 ∨ Real.sqrt (a^2 - b^2) = -4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3850_385095


namespace NUMINAMATH_CALUDE_factor_expression_l3850_385083

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3850_385083


namespace NUMINAMATH_CALUDE_car_speed_problem_l3850_385035

/-- Proves that given a 15-hour trip where a car travels at 30 mph for the first 5 hours
    and the overall average speed is 38 mph, the average speed for the remaining 10 hours is 42 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (5 * 30 + 10 * v) / 15 = 38 → v = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3850_385035


namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_l3850_385087

/-- The axis of symmetry of a parabola y^2 = mx has the equation x = -m/4 -/
def axis_of_symmetry (m : ℝ) : ℝ → Prop :=
  fun x ↦ x = -m/4

/-- A point (x, y) lies on the parabola y^2 = mx -/
def on_parabola (m : ℝ) : ℝ × ℝ → Prop :=
  fun p ↦ p.2^2 = m * p.1

theorem parabola_symmetry_axis (m : ℝ) :
  axis_of_symmetry m (-m^2) →
  on_parabola m (-m^2, 3) →
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_l3850_385087


namespace NUMINAMATH_CALUDE_nancy_wednesday_pots_l3850_385074

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPots where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot creation --/
def nancy_pots : ClayPots where
  monday := 12
  tuesday := 2 * 12
  wednesday := 50 - (12 + 2 * 12)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem nancy_wednesday_pots : nancy_pots.wednesday = 14 := by
  sorry

#eval nancy_pots.wednesday

end NUMINAMATH_CALUDE_nancy_wednesday_pots_l3850_385074


namespace NUMINAMATH_CALUDE_inequality_proof_l3850_385049

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3850_385049


namespace NUMINAMATH_CALUDE_chess_game_probability_l3850_385012

theorem chess_game_probability (prob_draw prob_B_win : ℝ) 
  (h1 : prob_draw = 1/2)
  (h2 : prob_B_win = 1/3) :
  prob_draw + prob_B_win = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l3850_385012


namespace NUMINAMATH_CALUDE_min_value_expression_l3850_385022

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4*a^2 + b^2)).sqrt) / (a * b) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3850_385022


namespace NUMINAMATH_CALUDE_minimum_value_range_l3850_385093

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The theorem stating the range of m for which f(x) has a minimum on (m, 6-m^2) --/
theorem minimum_value_range (m : ℝ) : 
  (∃ (c : ℝ), c ∈ Set.Ioo m (6 - m^2) ∧ 
    (∀ x ∈ Set.Ioo m (6 - m^2), f c ≤ f x)) ↔ 
  m ∈ Set.Icc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_minimum_value_range_l3850_385093


namespace NUMINAMATH_CALUDE_function_identification_l3850_385088

/-- A first-degree function -/
def first_degree_function (f : ℝ → ℝ) : Prop :=
  ∃ k m : ℝ, ∀ x, f x = k * x + m

/-- A second-degree function -/
def second_degree_function (g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

/-- Function composition equality -/
def composition_equality (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = g (f x)

/-- Tangent to x-axis -/
def tangent_to_x_axis (g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, y ≠ x → g y > 0

/-- Tangent to another function -/
def tangent_to_function (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ∀ y : ℝ, y ≠ x → f y ≠ g y

theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : first_degree_function f)
  (h2 : second_degree_function g)
  (h3 : composition_equality f g)
  (h4 : tangent_to_x_axis g)
  (h5 : tangent_to_function f g)
  (h6 : g 0 = 1/16) :
  (∀ x, f x = x) ∧ (∀ x, g x = x^2 + 1/2 * x + 1/16) := by
  sorry

end NUMINAMATH_CALUDE_function_identification_l3850_385088


namespace NUMINAMATH_CALUDE_apples_per_pie_l3850_385094

theorem apples_per_pie 
  (initial_apples : ℕ) 
  (handed_out : ℕ) 
  (num_pies : ℕ) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out = 19) 
  (h3 : num_pies = 7) :
  (initial_apples - handed_out) / num_pies = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3850_385094


namespace NUMINAMATH_CALUDE_storks_on_fence_storks_count_l3850_385075

theorem storks_on_fence (initial_birds : ℕ) (additional_birds : ℕ) (bird_stork_difference : ℕ) : ℕ :=
  let total_birds := initial_birds + additional_birds
  let storks := total_birds - bird_stork_difference
  storks

theorem storks_count : storks_on_fence 3 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_storks_count_l3850_385075


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l3850_385032

-- Define the triangle
def triangle_PQR (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = 37 ∧ dist Q R = 20 ∧ dist R P = 45

-- Define the circumcircle
def circumcircle (P Q R S : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), dist O P = r ∧ dist O Q = r ∧ dist O R = r ∧ dist O S = r

-- Define the perpendicular bisector
def perp_bisector (P R S : ℝ × ℝ) : Prop :=
  dist P S = dist R S ∧ (S.1 - P.1) * (R.1 - P.1) + (S.2 - P.2) * (R.2 - P.2) = 0

-- Main theorem
theorem triangle_circumcircle_intersection 
  (P Q R S : ℝ × ℝ) 
  (h_triangle : triangle_PQR P Q R)
  (h_circumcircle : circumcircle P Q R S)
  (h_perp_bisector : perp_bisector P R S)
  (h_opposite_side : (S.1 - P.1) * (Q.1 - P.1) + (S.2 - P.2) * (Q.2 - P.2) < 0) :
  ∃ (a b : ℕ), 
    a = 15 ∧ 
    b = 27 ∧ 
    dist P S = a * Real.sqrt b ∧
    ⌊a + Real.sqrt b⌋ = 20 :=
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l3850_385032


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3850_385014

theorem roots_of_polynomial (a b : ℝ) : 
  (a + 3 * Complex.I) * (b + 6 * Complex.I) = 52 + 105 * Complex.I ∧
  (a + 3 * Complex.I) + (b + 6 * Complex.I) = 12 + 15 * Complex.I →
  a = 23 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3850_385014


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3850_385030

/-- Given that sin(α) / (sin(α) - cos(α)) = -1, prove:
    1. tan(α) = 1/2
    2. (sin²(α) + 2sin(α)cos(α)) / (3sin²(α) + cos²(α)) = 5/7 -/
theorem trigonometric_identity (α : ℝ) 
    (h : Real.sin α / (Real.sin α - Real.cos α) = -1) : 
    Real.tan α = 1/2 ∧ 
    (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / 
    (3 * Real.sin α ^ 2 + Real.cos α ^ 2) = 5/7 := by
  sorry


end NUMINAMATH_CALUDE_trigonometric_identity_l3850_385030


namespace NUMINAMATH_CALUDE_vector_equality_l3850_385052

/-- Given vectors a, b, and c in ℝ², prove that c = 1/2 * a - 3/2 * b -/
theorem vector_equality (a b c : Fin 2 → ℝ) 
  (ha : a = ![1, 1])
  (hb : b = ![1, -1])
  (hc : c = ![-1, 2]) :
  c = 1/2 • a - 3/2 • b := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l3850_385052


namespace NUMINAMATH_CALUDE_eight_people_lineup_l3850_385084

theorem eight_people_lineup : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_lineup_l3850_385084


namespace NUMINAMATH_CALUDE_red_ball_removal_l3850_385089

theorem red_ball_removal (total : ℕ) (initial_red_percent : ℚ) (final_red_percent : ℚ) 
  (removed : ℕ) (h_total : total = 600) (h_initial_red : initial_red_percent = 70/100) 
  (h_final_red : final_red_percent = 60/100) (h_removed : removed = 150) : 
  (initial_red_percent * total - removed) / (total - removed) = final_red_percent := by
  sorry

end NUMINAMATH_CALUDE_red_ball_removal_l3850_385089


namespace NUMINAMATH_CALUDE_total_phones_sold_l3850_385080

/-- Calculates the total number of cell phones sold given the initial and final inventories, and the number of damaged/defective phones. -/
def cellPhonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIPhone : ℕ) (finalIPhone : ℕ) (damagedSamsung : ℕ) (defectiveIPhone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIPhone - defectiveIPhone - finalIPhone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific inventory and damage numbers. -/
theorem total_phones_sold :
  cellPhonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_total_phones_sold_l3850_385080


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3850_385051

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80*x + 1551) / (x^2 + 57*x - 2970)) →
  α + β = 137 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3850_385051


namespace NUMINAMATH_CALUDE_system_solution_l3850_385039

theorem system_solution : 
  ∃ (x y : ℝ), (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧
                4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔
               ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3850_385039


namespace NUMINAMATH_CALUDE_emily_candies_l3850_385076

theorem emily_candies (bob_candies : ℕ) (jennifer_candies : ℕ) (emily_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : bob_candies = 4) :
  emily_candies = 6 := by
sorry

end NUMINAMATH_CALUDE_emily_candies_l3850_385076


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l3850_385078

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l3850_385078


namespace NUMINAMATH_CALUDE_largest_root_of_cubic_equation_l3850_385023

theorem largest_root_of_cubic_equation :
  let f (x : ℝ) := 4 * x^3 - 17 * x^2 + x + 10
  ∃ (max_root : ℝ), max_root = (25 + Real.sqrt 545) / 8 ∧
    f max_root = 0 ∧
    ∀ (y : ℝ), f y = 0 → y ≤ max_root :=
by sorry

end NUMINAMATH_CALUDE_largest_root_of_cubic_equation_l3850_385023


namespace NUMINAMATH_CALUDE_max_silver_tokens_l3850_385071

/-- Represents the number of tokens Alex has --/
structure Tokens where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange at a booth --/
inductive Exchange
  | First  -- 3 red -> 1 silver + 2 blue
  | Second -- 4 blue -> 1 silver + 2 red

/-- Applies an exchange to the current token state --/
def applyExchange (t : Tokens) (e : Exchange) : Tokens :=
  match e with
  | Exchange.First => 
      { red := t.red - 3, blue := t.blue + 2, silver := t.silver + 1 }
  | Exchange.Second => 
      { red := t.red + 2, blue := t.blue - 4, silver := t.silver + 1 }

/-- Checks if an exchange is possible given the current token state --/
def canExchange (t : Tokens) (e : Exchange) : Bool :=
  match e with
  | Exchange.First => t.red ≥ 3
  | Exchange.Second => t.blue ≥ 4

/-- The main theorem to prove --/
theorem max_silver_tokens : 
  ∃ (exchanges : List Exchange), 
    let finalTokens := exchanges.foldl applyExchange { red := 100, blue := 90, silver := 0 }
    finalTokens.silver = 39 ∧ 
    (∀ e : Exchange, ¬(canExchange finalTokens e)) ∧
    (∀ otherExchanges : List Exchange, 
      let otherFinalTokens := otherExchanges.foldl applyExchange { red := 100, blue := 90, silver := 0 }
      otherFinalTokens.silver ≤ 39) := by
  sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l3850_385071


namespace NUMINAMATH_CALUDE_sum_to_k_is_triangular_square_k_values_l3850_385031

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem sum_to_k_is_triangular_square (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_to_k k = n^2 ∧ n < 150 ∧ is_perfect_square (triangular_number n)

theorem k_values : {k : ℕ | sum_to_k_is_triangular_square k} = {1, 8, 39, 92, 168} := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_is_triangular_square_k_values_l3850_385031


namespace NUMINAMATH_CALUDE_woodworker_chairs_l3850_385017

/-- Calculates the number of chairs built given the total number of furniture legs,
    number of tables, legs per table, and legs per chair. -/
def chairs_built (total_legs : ℕ) (num_tables : ℕ) (legs_per_table : ℕ) (legs_per_chair : ℕ) : ℕ :=
  (total_legs - num_tables * legs_per_table) / legs_per_chair

/-- Proves that given 40 total furniture legs, 4 tables, 4 legs per table,
    and 4 legs per chair, the number of chairs built is 6. -/
theorem woodworker_chairs : chairs_built 40 4 4 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_chairs_l3850_385017


namespace NUMINAMATH_CALUDE_complex_moduli_product_l3850_385007

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l3850_385007


namespace NUMINAMATH_CALUDE_arrangements_five_singers_l3850_385066

/-- The number of singers --/
def n : ℕ := 5

/-- The number of different arrangements for n singers with constraints --/
def arrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1) + (n - 2) * (n - 2) * Nat.factorial (n - 2)

/-- Theorem: The number of arrangements for 5 singers with constraints is 78 --/
theorem arrangements_five_singers : arrangements n = 78 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_five_singers_l3850_385066


namespace NUMINAMATH_CALUDE_second_transfer_amount_l3850_385061

/-- Calculates the amount of a bank transfer given the initial balance, 
    first transfer amount, final balance, and service charge rate. -/
def calculate_second_transfer (initial_balance : ℚ) (first_transfer : ℚ) 
  (final_balance : ℚ) (service_charge_rate : ℚ) : ℚ :=
  let first_transfer_with_charge := first_transfer * (1 + service_charge_rate)
  let total_deduction := initial_balance - final_balance
  (total_deduction - first_transfer_with_charge) / (service_charge_rate)

/-- Theorem stating that given the problem conditions, 
    the second transfer amount is $60. -/
theorem second_transfer_amount 
  (initial_balance : ℚ) 
  (first_transfer : ℚ)
  (final_balance : ℚ)
  (service_charge_rate : ℚ)
  (h1 : initial_balance = 400)
  (h2 : first_transfer = 90)
  (h3 : final_balance = 307)
  (h4 : service_charge_rate = 2/100) :
  calculate_second_transfer initial_balance first_transfer final_balance service_charge_rate = 60 := by
  sorry

#eval calculate_second_transfer 400 90 307 (2/100)

end NUMINAMATH_CALUDE_second_transfer_amount_l3850_385061


namespace NUMINAMATH_CALUDE_square_sum_eq_25_l3850_385072

theorem square_sum_eq_25 (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 7) : p^2 + q^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_25_l3850_385072


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l3850_385024

theorem cricket_bat_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (price_C : ℝ) 
  (h1 : profit_A_to_B = 0.20)
  (h2 : profit_B_to_C = 0.25)
  (h3 : price_C = 228) : 
  ∃ (cost_price_A : ℝ), cost_price_A = 152 ∧ 
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l3850_385024


namespace NUMINAMATH_CALUDE_range_of_a_l3850_385018

-- Define propositions p and q
def p (a : ℝ) : Prop := -3 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3850_385018


namespace NUMINAMATH_CALUDE_no_solution_exists_l3850_385057

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem no_solution_exists : ¬∃ (x y : ℕ), sumOfDigits ((10^x)^y - 64) = 279 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3850_385057


namespace NUMINAMATH_CALUDE_josh_marbles_l3850_385053

/-- The number of marbles Josh has after losing some -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: If Josh had 9 marbles initially and lost 5, he now has 4 marbles -/
theorem josh_marbles : remaining_marbles 9 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3850_385053


namespace NUMINAMATH_CALUDE_sum_irreducible_fractions_integer_l3850_385008

theorem sum_irreducible_fractions_integer (a b c d A : ℤ) 
  (h1 : b ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : Nat.gcd a.natAbs b.natAbs = 1) 
  (h4 : Nat.gcd c.natAbs d.natAbs = 1) 
  (h5 : a / b + c / d = A) : 
  b = d := by
sorry

end NUMINAMATH_CALUDE_sum_irreducible_fractions_integer_l3850_385008


namespace NUMINAMATH_CALUDE_no_true_propositions_l3850_385041

theorem no_true_propositions : 
  let prop1 := ∀ x : ℝ, x^2 - 3*x + 2 = 0
  let prop2 := ∃ x : ℚ, x^2 = 2
  let prop3 := ∃ x : ℝ, x^2 + 1 = 0
  let prop4 := ∀ x : ℝ, 4*x^2 > 2*x - 1 + 3*x^2
  ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 :=
by
  sorry

#check no_true_propositions

end NUMINAMATH_CALUDE_no_true_propositions_l3850_385041


namespace NUMINAMATH_CALUDE_or_sufficient_not_necessary_for_and_l3850_385033

theorem or_sufficient_not_necessary_for_and (p q : Prop) :
  (∃ (h : p ∨ q → p ∧ q), ¬(p ∧ q → p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_or_sufficient_not_necessary_for_and_l3850_385033


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11_terms_l3850_385010

theorem arithmetic_sequence_11_terms (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 12 →
  d = 6 →
  n = 11 →
  aₙ = a₁ + (n - 1) * d →
  aₙ = 72 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11_terms_l3850_385010


namespace NUMINAMATH_CALUDE_negation_distribution_l3850_385069

theorem negation_distribution (x : ℝ) : -(3*x - 2) = -3*x + 2 := by sorry

end NUMINAMATH_CALUDE_negation_distribution_l3850_385069


namespace NUMINAMATH_CALUDE_only_234_and_468_satisfy_l3850_385090

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def satisfiesCondition (n : Nat) : Prop :=
  n < 10000 ∧ n = 26 * sumOfDigits n

theorem only_234_and_468_satisfy :
  ∀ n : Nat, satisfiesCondition n ↔ n = 234 ∨ n = 468 := by
  sorry

end NUMINAMATH_CALUDE_only_234_and_468_satisfy_l3850_385090


namespace NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l3850_385001

/-- The number of pine trees -/
def pine_trees : ℕ := 4

/-- The number of cedar trees -/
def cedar_trees : ℕ := 5

/-- The number of fir trees -/
def fir_trees : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := pine_trees + cedar_trees + fir_trees

/-- The probability that no two fir trees are next to one another when planted in a random order -/
theorem fir_trees_not_adjacent_probability : 
  (Nat.choose (pine_trees + cedar_trees + 1) fir_trees : ℚ) / 
  (Nat.choose total_trees fir_trees) = 6 / 143 := by sorry

end NUMINAMATH_CALUDE_fir_trees_not_adjacent_probability_l3850_385001


namespace NUMINAMATH_CALUDE_fraction_difference_equals_two_l3850_385056

theorem fraction_difference_equals_two 
  (a b : ℝ) 
  (h1 : 2 * b = 1 + a * b) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_two_l3850_385056


namespace NUMINAMATH_CALUDE_cistern_length_is_ten_l3850_385006

/-- Represents a cistern with given dimensions and water level --/
structure Cistern where
  length : ℝ
  width : ℝ
  waterDepth : ℝ

/-- Calculates the total wet surface area of a cistern --/
def wetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.waterDepth + 2 * c.width * c.waterDepth

/-- Theorem stating that a cistern with given dimensions has a length of 10 meters --/
theorem cistern_length_is_ten :
  ∃ (c : Cistern), c.width = 8 ∧ c.waterDepth = 1.5 ∧ wetSurfaceArea c = 134 → c.length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cistern_length_is_ten_l3850_385006


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l3850_385038

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v1 v2 : Vector2D) : Prop := dot_product v1 v2 = 0

theorem perpendicular_lines_theorem (b c : ℝ) :
  let v1 : Vector2D := ⟨4, 1⟩
  let v2 : Vector2D := ⟨b, -8⟩
  let v3 : Vector2D := ⟨5, c⟩
  perpendicular v1 v3 ∧ perpendicular v2 v3 → b = 2 ∧ c = -20 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l3850_385038


namespace NUMINAMATH_CALUDE_expression_simplification_l3850_385097

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x / (x - 1) - 1) / ((x^2 + 2*x + 1) / (x^2 - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3850_385097


namespace NUMINAMATH_CALUDE_prob_five_dice_three_matching_l3850_385086

/-- The probability of rolling at least three matching dice out of five fair six-sided dice -/
def prob_at_least_three_matching (n : ℕ) (s : ℕ) : ℚ :=
  -- n is the number of dice
  -- s is the number of sides on each die
  sorry

/-- Theorem stating that the probability of rolling at least three matching dice
    out of five fair six-sided dice is equal to 23/108 -/
theorem prob_five_dice_three_matching :
  prob_at_least_three_matching 5 6 = 23 / 108 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_dice_three_matching_l3850_385086


namespace NUMINAMATH_CALUDE_ellipse_foci_range_ellipse_or_quadratic_range_l3850_385015

/-- Definition of an ellipse with semi-major axis 5 and semi-minor axis √a -/
def is_ellipse (a : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / 5 + y^2 / a = 1

/-- The foci of the ellipse are on the x-axis -/
def foci_on_x_axis (a : ℝ) : Prop :=
  is_ellipse a ∧ ∃ c : ℝ, c^2 = 5 - a ∧ c ≥ 0

/-- The quadratic inequality holds for all real x -/
def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + 2 * a * x + 3 ≥ 0

theorem ellipse_foci_range (a : ℝ) :
  foci_on_x_axis a → 0 < a ∧ a < 5 :=
sorry

theorem ellipse_or_quadratic_range (a : ℝ) :
  (foci_on_x_axis a ∨ quadratic_inequality_holds a) ∧
  ¬(foci_on_x_axis a ∧ quadratic_inequality_holds a) →
  (3 < a ∧ a < 5) ∨ (-3 ≤ a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_range_ellipse_or_quadratic_range_l3850_385015


namespace NUMINAMATH_CALUDE_boys_count_proof_l3850_385098

/-- Given a total number of eyes and the number of eyes per boy, 
    calculate the number of boys. -/
def number_of_boys (total_eyes : ℕ) (eyes_per_boy : ℕ) : ℕ :=
  total_eyes / eyes_per_boy

theorem boys_count_proof (total_eyes : ℕ) (eyes_per_boy : ℕ) 
  (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : 
  number_of_boys total_eyes eyes_per_boy = 23 := by
  sorry

#eval number_of_boys 46 2

end NUMINAMATH_CALUDE_boys_count_proof_l3850_385098


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3850_385034

/-- The speed of a boat in still water, given its downstream travel information and current rate. -/
theorem boat_speed_in_still_water 
  (current_rate : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : current_rate = 5)
  (h2 : distance_downstream = 11.25)
  (h3 : time_minutes = 27) :
  ∃ (speed_still_water : ℝ), 
    speed_still_water = 20 ∧ 
    distance_downstream = (speed_still_water + current_rate) * (time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3850_385034


namespace NUMINAMATH_CALUDE_cherry_pies_profit_independence_l3850_385067

/-- Proves that the number of cherry pies does not affect the profit in Benny's pie sale scenario -/
theorem cherry_pies_profit_independence (num_pumpkin : ℕ) (cost_pumpkin : ℚ) (cost_cherry : ℚ) (sell_price : ℚ) (target_profit : ℚ) :
  num_pumpkin = 10 →
  cost_pumpkin = 3 →
  cost_cherry = 5 →
  sell_price = 5 →
  target_profit = 20 →
  ∀ num_cherry : ℕ,
    sell_price * (num_pumpkin + num_cherry) - (num_pumpkin * cost_pumpkin + num_cherry * cost_cherry) = target_profit :=
by sorry


end NUMINAMATH_CALUDE_cherry_pies_profit_independence_l3850_385067


namespace NUMINAMATH_CALUDE_tangent_segment_region_area_l3850_385060

/-- The area of the region formed by all line segments of length 6 that are tangent to a circle of radius 3 at their midpoints -/
theorem tangent_segment_region_area : Real := by
  -- Define the circle radius
  let circle_radius : Real := 3
  
  -- Define the line segment length
  let segment_length : Real := 6
  
  -- Define the region area
  let region_area : Real := 9 * Real.pi
  
  -- State that the line segments are tangent to the circle at their midpoints
  -- (This is implicitly used in the proof, but we don't need to explicitly define it in Lean)
  
  -- Prove that the area of the region is equal to 9π
  sorry

#check tangent_segment_region_area

end NUMINAMATH_CALUDE_tangent_segment_region_area_l3850_385060


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3850_385091

theorem decimal_to_fraction : 
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3850_385091


namespace NUMINAMATH_CALUDE_cubic_integer_root_l3850_385068

theorem cubic_integer_root
  (a b c : ℚ)
  (h1 : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = 3 - Real.sqrt 5 ∨ x = 3 + Real.sqrt 5 ∨ (∃ n : ℤ, x = n)))
  (h2 : ∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5)
  (h3 : ∃ n : ℤ, (n : ℝ)^3 + a*(n : ℝ)^2 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, n^3 + a*n^2 + b*n + c = 0 ∧ n = -6 :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l3850_385068


namespace NUMINAMATH_CALUDE_m_range_l3850_385040

theorem m_range : ∃ m : ℝ, m = Real.sqrt 5 - 1 ∧ 1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3850_385040


namespace NUMINAMATH_CALUDE_iggy_wednesday_miles_l3850_385099

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running schedule --/
def IggySchedule : RunDay → ℕ
  | RunDay.Monday => 3
  | RunDay.Tuesday => 4
  | RunDay.Thursday => 8
  | RunDay.Friday => 3
  | RunDay.Wednesday => 0  -- We'll prove this should be 6

/-- Iggy's pace in minutes per mile --/
def IggyPace : ℕ := 10

/-- Total running time in hours --/
def TotalRunningTime : ℕ := 4

/-- Converts hours to minutes --/
def HoursToMinutes (hours : ℕ) : ℕ := hours * 60

theorem iggy_wednesday_miles :
  ∃ (wednesday_miles : ℕ),
    wednesday_miles = 6 ∧
    HoursToMinutes TotalRunningTime =
      (IggySchedule RunDay.Monday +
       IggySchedule RunDay.Tuesday +
       wednesday_miles +
       IggySchedule RunDay.Thursday +
       IggySchedule RunDay.Friday) * IggyPace :=
by sorry

end NUMINAMATH_CALUDE_iggy_wednesday_miles_l3850_385099


namespace NUMINAMATH_CALUDE_exists_a_satisfying_conditions_l3850_385062

def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem exists_a_satisfying_conditions :
  ∃ a : ℝ, a = -2 ∧ 
    (A a ∩ C = ∅) ∧ 
    (∅ ⊂ A a ∩ B) :=
by sorry

end NUMINAMATH_CALUDE_exists_a_satisfying_conditions_l3850_385062


namespace NUMINAMATH_CALUDE_range_of_a_l3850_385048

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) → 
  3 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3850_385048


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3850_385079

theorem line_segment_endpoint (x : ℝ) :
  (((x - 3)^2 + (4 + 2)^2).sqrt = 17) →
  (x < 0) →
  (x = 3 - Real.sqrt 253) := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3850_385079


namespace NUMINAMATH_CALUDE_S_25_equals_7825_l3850_385046

def first_element (n : ℕ) : ℕ :=
  1 + (n - 1) * n / 2

def last_element (n : ℕ) : ℕ :=
  first_element n + n - 1

def S (n : ℕ) : ℕ :=
  n * (first_element n + last_element n) / 2

theorem S_25_equals_7825 : S 25 = 7825 := by
  sorry

end NUMINAMATH_CALUDE_S_25_equals_7825_l3850_385046


namespace NUMINAMATH_CALUDE_all_expressions_completely_symmetric_l3850_385005

/-- A function is completely symmetric if it remains unchanged when any two of its variables are exchanged. -/
def CompletelySymmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c, f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

/-- Expression 1: a(b+c)+b(a+c)+c(a+b) -/
def expr1 (a b c : ℝ) : ℝ := a*(b+c) + b*(a+c) + c*(a+b)

/-- Expression 2: a²bc+b²ac+c²ab -/
def expr2 (a b c : ℝ) : ℝ := a^2*b*c + b^2*a*c + c^2*a*b

/-- Expression 3: a²+b²+c²-ab-bc-ac -/
def expr3 (a b c : ℝ) : ℝ := a^2 + b^2 + c^2 - a*b - b*c - a*c

theorem all_expressions_completely_symmetric :
  CompletelySymmetric expr1 ∧ CompletelySymmetric expr2 ∧ CompletelySymmetric expr3 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_completely_symmetric_l3850_385005


namespace NUMINAMATH_CALUDE_baseball_tickets_sold_l3850_385002

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 →
  2 * fair_tickets + 6 = baseball_tickets →
  baseball_tickets = 56 := by
  sorry

end NUMINAMATH_CALUDE_baseball_tickets_sold_l3850_385002


namespace NUMINAMATH_CALUDE_truck_distance_l3850_385096

theorem truck_distance (north_distance east_distance : ℝ) 
  (h1 : north_distance = 40)
  (h2 : east_distance = 30) :
  Real.sqrt (north_distance ^ 2 + east_distance ^ 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l3850_385096


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l3850_385019

/-- Proves that the length of a rectangular plot is 60 metres given the specified conditions -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_metre : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_metre = 26.50 →
  total_cost = 5300 →
  perimeter = total_cost / cost_per_metre →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_sixty_l3850_385019


namespace NUMINAMATH_CALUDE_money_distribution_solution_l3850_385021

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  ann_initial : ℕ
  bill_initial : ℕ
  charlie_initial : ℕ
  bill_to_ann : ℕ
  charlie_to_bill : ℕ

/-- Checks if the money distribution results in equal amounts --/
def isEqualDistribution (md : MoneyDistribution) : Prop :=
  let ann_final := md.ann_initial + md.bill_to_ann
  let bill_final := md.bill_initial - md.bill_to_ann + md.charlie_to_bill
  let charlie_final := md.charlie_initial - md.charlie_to_bill
  ann_final = bill_final ∧ bill_final = charlie_final

/-- Theorem stating the solution to the money distribution problem --/
theorem money_distribution_solution :
  let md : MoneyDistribution := {
    ann_initial := 777,
    bill_initial := 1111,
    charlie_initial := 1555,
    bill_to_ann := 371,
    charlie_to_bill := 408
  }
  isEqualDistribution md ∧ 
  (md.ann_initial + md.bill_to_ann = 1148) ∧
  (md.bill_initial - md.bill_to_ann + md.charlie_to_bill = 1148) ∧
  (md.charlie_initial - md.charlie_to_bill = 1148) :=
by
  sorry


end NUMINAMATH_CALUDE_money_distribution_solution_l3850_385021


namespace NUMINAMATH_CALUDE_power_of_product_l3850_385082

theorem power_of_product (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3850_385082


namespace NUMINAMATH_CALUDE_triangle_shape_l3850_385026

theorem triangle_shape (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  ∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ^ 2 + b ^ 2 - c ^ 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l3850_385026


namespace NUMINAMATH_CALUDE_solve_star_equation_l3850_385070

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - a - 2

-- Theorem statement
theorem solve_star_equation (y : ℝ) (h : star 3 y = 25) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l3850_385070


namespace NUMINAMATH_CALUDE_chords_for_full_rotation_l3850_385025

/-- The number of chords needed to complete a full rotation when drawing chords on a larger circle
    tangent to a smaller concentric circle, given that the angle between consecutive chords is 60°. -/
def numChords : ℕ := 3

theorem chords_for_full_rotation (angle : ℝ) (h : angle = 60) :
  (numChords : ℝ) * angle = 360 := by
  sorry

end NUMINAMATH_CALUDE_chords_for_full_rotation_l3850_385025


namespace NUMINAMATH_CALUDE_diamond_3_2_l3850_385044

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

/-- Theorem: The diamond operation applied to 3 and 2 equals 125 -/
theorem diamond_3_2 : diamond 3 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_diamond_3_2_l3850_385044


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3850_385011

theorem simplify_and_evaluate :
  ∀ x : ℤ, -2 < x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 →
  (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x)) = x^2 / (x - 1) ∧
  (x = 2 → x^2 / (x - 1) = 4) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3850_385011


namespace NUMINAMATH_CALUDE_composite_n4_plus_4_l3850_385065

theorem composite_n4_plus_4 (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_n4_plus_4_l3850_385065


namespace NUMINAMATH_CALUDE_sector_area_special_case_l3850_385073

/-- The area of a circular sector with central angle 2π/3 radians and radius 2 is 4π/3. -/
theorem sector_area_special_case :
  let central_angle : Real := (2 * Real.pi) / 3
  let radius : Real := 2
  let sector_area : Real := (1 / 2) * radius^2 * central_angle
  sector_area = (4 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_special_case_l3850_385073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3850_385059

/-- Given two arithmetic sequences {a_n} and {b_n}, S_n and T_n are the sums of their first n terms respectively -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of S_n to T_n is (7n + 2) / (n + 3) for all n -/
def ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : ratio_condition S T) :
  a 5 / b 5 = 65 / 12 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3850_385059


namespace NUMINAMATH_CALUDE_triangle_theorem_l3850_385029

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : t.a + t.c = 4) :
  t.B = π / 3 ∧ 
  ((t.a = 1 ∧ t.c = 3) ∨ (t.a = 3 ∧ t.c = 1)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3850_385029


namespace NUMINAMATH_CALUDE_equation_solution_l3850_385085

theorem equation_solution (y : ℚ) : 
  (∃ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21) → 
  (∀ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3850_385085


namespace NUMINAMATH_CALUDE_percent_relation_l3850_385045

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3850_385045


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l3850_385055

theorem fraction_of_powers_equals_five_fourths :
  (3^10 + 3^8) / (3^10 - 3^8) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l3850_385055


namespace NUMINAMATH_CALUDE_tangent_function_property_l3850_385036

theorem tangent_function_property (c d : ℝ) (h1 : c > 0) (h2 : d > 0) : 
  (∀ x, c * Real.tan (d * x) = c * Real.tan (d * (x + 3 * π / 4))) →
  c * Real.tan (d * π / 8) = 3 →
  c * d = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_function_property_l3850_385036


namespace NUMINAMATH_CALUDE_motorcycles_in_anytown_l3850_385058

/-- Represents the number of vehicles of each type in Anytown -/
structure VehicleCounts where
  trucks : ℕ
  sedans : ℕ
  motorcycles : ℕ

/-- The ratio of vehicles in Anytown -/
def vehicle_ratio : VehicleCounts := ⟨3, 7, 2⟩

/-- The actual number of sedans in Anytown -/
def actual_sedans : ℕ := 9100

/-- Theorem stating the number of motorcycles in Anytown -/
theorem motorcycles_in_anytown : 
  ∃ (vc : VehicleCounts), 
    vc.sedans = actual_sedans ∧ 
    vc.trucks * vehicle_ratio.sedans = vc.sedans * vehicle_ratio.trucks ∧
    vc.sedans * vehicle_ratio.motorcycles = vc.motorcycles * vehicle_ratio.sedans ∧
    vc.motorcycles = 2600 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_anytown_l3850_385058


namespace NUMINAMATH_CALUDE_car_speed_problem_l3850_385064

theorem car_speed_problem (D : ℝ) (D_pos : D > 0) : 
  let total_time := D / 40
  let first_part_time := (0.75 * D) / 60
  let second_part_time := total_time - first_part_time
  let s := (0.25 * D) / second_part_time
  s = 20 := by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3850_385064


namespace NUMINAMATH_CALUDE_sqrt_200_simplified_l3850_385003

theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplified_l3850_385003


namespace NUMINAMATH_CALUDE_unique_whole_number_between_l3850_385004

theorem unique_whole_number_between (M : ℕ) : 
  (5.25 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 5.75) → M = 22 :=
by sorry

end NUMINAMATH_CALUDE_unique_whole_number_between_l3850_385004


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3850_385042

theorem min_value_theorem (x : ℝ) (h : x > 3) : x + 4 / (x - 3) ≥ 7 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 3) : 
  x + 4 / (x - 3) = 7 ↔ x = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3850_385042


namespace NUMINAMATH_CALUDE_yellow_jacket_incident_l3850_385081

theorem yellow_jacket_incident (total_students : ℕ) 
  (initial_cafeteria_fraction : ℚ) (final_cafeteria_count : ℕ) 
  (cafeteria_to_outside : ℕ) : 
  total_students = 90 →
  initial_cafeteria_fraction = 2/3 →
  final_cafeteria_count = 67 →
  cafeteria_to_outside = 3 →
  (final_cafeteria_count - (initial_cafeteria_fraction * total_students).floor + cafeteria_to_outside) / 
  (total_students - (initial_cafeteria_fraction * total_students).floor) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_yellow_jacket_incident_l3850_385081


namespace NUMINAMATH_CALUDE_bowl_capacity_sum_l3850_385000

theorem bowl_capacity_sum : 
  let second_bowl : ℕ := 600
  let first_bowl : ℕ := (3 * second_bowl) / 4
  let third_bowl : ℕ := first_bowl / 2
  let fourth_bowl : ℕ := second_bowl / 3
  second_bowl + first_bowl + third_bowl + fourth_bowl = 1475 :=
by sorry

end NUMINAMATH_CALUDE_bowl_capacity_sum_l3850_385000


namespace NUMINAMATH_CALUDE_estimate_sqrt_19_l3850_385009

theorem estimate_sqrt_19 : 6 < 2 + Real.sqrt 19 ∧ 2 + Real.sqrt 19 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_19_l3850_385009


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l3850_385047

/-- Proves that the length of a train equals the length of a platform given specific conditions --/
theorem train_platform_length_equality (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_min : ℝ) :
  train_length = 750 →
  train_speed_kmh = 90 →
  crossing_time_min = 1 →
  ∃ (platform_length : ℝ),
    platform_length = train_length ∧
    platform_length + train_length = train_speed_kmh * (1000 / 3600) * (crossing_time_min * 60) :=
by sorry


end NUMINAMATH_CALUDE_train_platform_length_equality_l3850_385047


namespace NUMINAMATH_CALUDE_min_value_of_cosine_sum_l3850_385013

theorem min_value_of_cosine_sum (x y z : Real) 
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 2)
  (hy : 0 ≤ y ∧ y ≤ Real.pi / 2)
  (hz : 0 ≤ z ∧ z ≤ Real.pi / 2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_cosine_sum_l3850_385013


namespace NUMINAMATH_CALUDE_prob_A_truth_l3850_385092

-- Define the probabilities
def prob_B_truth : ℝ := 0.60
def prob_both_truth : ℝ := 0.45

-- Theorem statement
theorem prob_A_truth :
  ∃ (prob_A : ℝ),
    prob_A * prob_B_truth = prob_both_truth ∧
    prob_A = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_prob_A_truth_l3850_385092
