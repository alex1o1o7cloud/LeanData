import Mathlib

namespace NUMINAMATH_CALUDE_cost_of_500_cookies_l1240_124018

/-- The cost in dollars for buying a number of cookies -/
def cookie_cost (num_cookies : ℕ) : ℚ :=
  (num_cookies * 2) / 100

/-- Proof that buying 500 cookies costs 10 dollars -/
theorem cost_of_500_cookies : cookie_cost 500 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_cookies_l1240_124018


namespace NUMINAMATH_CALUDE_solve_x_solve_y_solve_pqr_l1240_124015

-- Define the structure of the diagram for parts (a) and (b)
structure Diagram :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom : ℤ)
  (top_sum : ℤ)
  (left_sum : ℤ)
  (right_sum : ℤ)

-- Define the diagram for part (a)
def diagram_a : Diagram :=
  { top_left := 9,  -- This is derived from the given information
    top_right := 4,
    bottom := 1,    -- This is derived from the given information
    top_sum := 13,
    left_sum := 10,
    right_sum := 5  -- This is x, which we need to prove
  }

-- Define the diagram for part (b)
def diagram_b : Diagram :=
  { top_left := 24,  -- This is 3w, where w = 8
    top_right := 24, -- This is also 3w
    bottom := 8,     -- This is w
    top_sum := 48,
    left_sum := 32,  -- This is y, which we need to prove
    right_sum := 32  -- This is also y
  }

-- Theorem for part (a)
theorem solve_x (d : Diagram) : 
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.bottom + d.top_right = d.right_sum →
  d.right_sum = 5 :=
sorry

-- Theorem for part (b)
theorem solve_y (d : Diagram) :
  d.top_left = d.top_right ∧
  d.top_left = 3 * d.bottom ∧
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.left_sum = d.right_sum →
  d.left_sum = 32 :=
sorry

-- Theorem for part (c)
theorem solve_pqr (p q r : ℤ) :
  p + r = 3 ∧
  p + q = 18 ∧
  q + r = 13 →
  p = 4 ∧ q = 14 ∧ r = -1 :=
sorry

end NUMINAMATH_CALUDE_solve_x_solve_y_solve_pqr_l1240_124015


namespace NUMINAMATH_CALUDE_stream_speed_l1240_124094

theorem stream_speed (upstream_distance : ℝ) (downstream_distance : ℝ) (time : ℝ) 
  (h1 : upstream_distance = 16)
  (h2 : downstream_distance = 24)
  (h3 : time = 4)
  (h4 : upstream_distance / time + downstream_distance / time = 10) :
  let stream_speed := (downstream_distance - upstream_distance) / (2 * time)
  stream_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1240_124094


namespace NUMINAMATH_CALUDE_sum_lower_bound_l1240_124055

theorem sum_lower_bound (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : a = 1 / b) :
  a + 2014 * b > 2015 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l1240_124055


namespace NUMINAMATH_CALUDE_complex_calculation_result_l1240_124007

theorem complex_calculation_result : (13.672 * 125 + 136.72 * 12.25 - 1367.2 * 1.875) / 17.09 = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_result_l1240_124007


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l1240_124069

open Real

theorem tangent_slope_angle (f : ℝ → ℝ) (x : ℝ) :
  f = (λ x => Real.log (x^2 + 1)) →
  x = 1 →
  let slope := (deriv f) x
  let angle := Real.arctan slope
  angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l1240_124069


namespace NUMINAMATH_CALUDE_triangle_properties_l1240_124012

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : Real.tan abc.A = 1/3) 
  (h2 : Real.tan abc.C = 1/2) :
  -- Part I: Angle B
  abc.B = 3 * Real.pi / 4 ∧ 
  -- Part II: Range of √2sinα - sinβ
  ∀ (α β : Real), 
    α > 0 → β > 0 → α + β = abc.B → 
    -Real.sqrt 2 / 2 < Real.sqrt 2 * Real.sin α - Real.sin β ∧
    Real.sqrt 2 * Real.sin α - Real.sin β < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1240_124012


namespace NUMINAMATH_CALUDE_specific_lot_volume_l1240_124003

/-- The volume of a rectangular lot -/
def lot_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem stating that the volume of the specific lot is 1600 cubic meters -/
theorem specific_lot_volume : lot_volume 40 20 2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_specific_lot_volume_l1240_124003


namespace NUMINAMATH_CALUDE_discount_calculation_l1240_124061

/-- The original price of a shirt before discount -/
def original_price : ℚ := 746.68

/-- The discounted price of the shirt -/
def discounted_price : ℚ := 560

/-- The discount rate applied to the shirt -/
def discount_rate : ℚ := 0.25

/-- Theorem stating that the discounted price is equal to the original price minus the discount -/
theorem discount_calculation (original : ℚ) (discount : ℚ) (discounted : ℚ) :
  original = discounted_price ∧ discount = discount_rate ∧ discounted = original * (1 - discount) →
  original = original_price :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1240_124061


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_less_than_100_eq_2500_l1240_124028

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def sum_of_odd_numbers_less_than_100 : ℕ := 
  (Finset.range 50).sum (λ i => 2*i + 1)

theorem sum_of_odd_numbers_less_than_100_eq_2500 : 
  sum_of_odd_numbers_less_than_100 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_less_than_100_eq_2500_l1240_124028


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1240_124034

theorem quadratic_inequality_solution_set (c : ℝ) (hc : c > 1) :
  {x : ℝ | x^2 - (c + 1/c)*x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1240_124034


namespace NUMINAMATH_CALUDE_total_distance_traveled_l1240_124058

/-- Calculate the total distance traveled given cycling, walking, and jogging durations and speeds -/
theorem total_distance_traveled
  (cycling_time : ℚ) (cycling_speed : ℚ)
  (walking_time : ℚ) (walking_speed : ℚ)
  (jogging_time : ℚ) (jogging_speed : ℚ)
  (h1 : cycling_time = 20 / 60)
  (h2 : cycling_speed = 12)
  (h3 : walking_time = 40 / 60)
  (h4 : walking_speed = 3)
  (h5 : jogging_time = 50 / 60)
  (h6 : jogging_speed = 7) :
  let total_distance := cycling_time * cycling_speed + walking_time * walking_speed + jogging_time * jogging_speed
  ∃ ε > 0, |total_distance - 11.8333| < ε :=
sorry

#eval (20/60 : ℚ) * 12 + (40/60 : ℚ) * 3 + (50/60 : ℚ) * 7

end NUMINAMATH_CALUDE_total_distance_traveled_l1240_124058


namespace NUMINAMATH_CALUDE_function_range_equivalence_l1240_124064

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * a + 1

-- State the theorem
theorem function_range_equivalence (a : ℝ) :
  (∃ x y : ℝ, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ f a x > 0 ∧ f a y < 0) ↔
  a ∈ Set.Ioo (-1) (-1/3) :=
sorry

end NUMINAMATH_CALUDE_function_range_equivalence_l1240_124064


namespace NUMINAMATH_CALUDE_donut_hole_count_donut_hole_count_proof_l1240_124021

/-- The number of donut holes Nira will have coated when all three workers finish simultaneously -/
theorem donut_hole_count : ℕ :=
  let nira_radius : ℝ := 5
  let theo_radius : ℝ := 7
  let kaira_side : ℝ := 6
  let nira_surface_area : ℝ := 4 * Real.pi * nira_radius ^ 2
  let theo_surface_area : ℝ := 4 * Real.pi * theo_radius ^ 2
  let kaira_surface_area : ℝ := 6 * kaira_side ^ 2
  5292

/-- Proof that Nira will have coated 5292 donut holes when all three workers finish simultaneously -/
theorem donut_hole_count_proof : donut_hole_count = 5292 := by
  sorry

end NUMINAMATH_CALUDE_donut_hole_count_donut_hole_count_proof_l1240_124021


namespace NUMINAMATH_CALUDE_total_lives_after_third_level_l1240_124037

-- Define the game parameters
def initial_lives : ℕ := 2
def enemies_defeated : ℕ := 5
def powerups_collected : ℕ := 4
def first_level_penalty : ℕ := 3
def second_level_modifier (x : ℕ) : ℕ := x / 2

-- Define the game rules
def first_level_lives (x : ℕ) : ℕ := initial_lives + 2 * x - first_level_penalty

def second_level_lives (first_level : ℕ) (y : ℕ) : ℕ :=
  first_level + 3 * y - second_level_modifier first_level

def third_level_bonus (x y : ℕ) : ℕ := x + 2 * y - 5

-- The main theorem
theorem total_lives_after_third_level :
  let first_level := first_level_lives enemies_defeated
  let second_level := second_level_lives first_level powerups_collected
  second_level + third_level_bonus enemies_defeated powerups_collected = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_after_third_level_l1240_124037


namespace NUMINAMATH_CALUDE_shooting_competition_l1240_124000

theorem shooting_competition (p_tie p_win : ℝ) 
  (h_tie : p_tie = 1/2)
  (h_win : p_win = 1/3) :
  p_tie + p_win = 5/6 := by
sorry

end NUMINAMATH_CALUDE_shooting_competition_l1240_124000


namespace NUMINAMATH_CALUDE_inequality_proof_l1240_124077

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = a*b) :
  (a + 2*b ≥ 8) ∧ (2*a + b ≥ 9) ∧ (a^2 + 4*b^2 + 5*a*b ≥ 72) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1240_124077


namespace NUMINAMATH_CALUDE_solution_for_F_l1240_124066

/-- Definition of function F --/
def F (a b c : ℝ) : ℝ := a * b^2 - c

/-- Theorem stating that 1/6 is the solution to F(a,5,10) = F(a,7,14) --/
theorem solution_for_F : ∃ a : ℝ, F a 5 10 = F a 7 14 ∧ a = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_F_l1240_124066


namespace NUMINAMATH_CALUDE_equation_solutions_l1240_124063

def equation (a b c d e x : ℝ) : Prop :=
  a * b^x * c^(2*x) = (d^(1/(3*x))) * (e^(1/(4*x)))

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation 2 3 5 7 11 x₁ ∧ equation 2 3 5 7 11 x₂) ∧
  (¬∃ x : ℝ, equation 5 3 2 (1/7) (1/11) x) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1240_124063


namespace NUMINAMATH_CALUDE_lcm_of_12_and_15_l1240_124099

theorem lcm_of_12_and_15 : 
  let a := 12
  let b := 15
  let hcf := 3
  let lcm := Nat.lcm a b
  lcm = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_15_l1240_124099


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1240_124024

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 4]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -43/14, 53/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1240_124024


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1240_124017

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1240_124017


namespace NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l1240_124041

theorem rectangle_triangle_perimeter (d : ℕ) : 
  let triangle_side := 3 * w - d
  let rectangle_width := w
  let rectangle_length := 3 * w
  let triangle_perimeter := 3 * triangle_side
  let rectangle_perimeter := 2 * (rectangle_width + rectangle_length)
  (∀ w : ℕ, 
    triangle_perimeter > 0 ∧ 
    rectangle_perimeter = triangle_perimeter + 1950 ∧
    rectangle_length - triangle_side = d ∧
    rectangle_length = 3 * rectangle_width) →
  d > 650 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l1240_124041


namespace NUMINAMATH_CALUDE_horner_polynomial_eval_l1240_124033

def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem horner_polynomial_eval :
  let coeffs := [7, 3, -5, 11]
  let x := 23
  horner_eval coeffs x = 86652 := by
sorry

end NUMINAMATH_CALUDE_horner_polynomial_eval_l1240_124033


namespace NUMINAMATH_CALUDE_sheila_cinnamon_balls_l1240_124042

/-- The number of family members -/
def family_members : ℕ := 5

/-- The number of days Sheila can place cinnamon balls in socks -/
def days : ℕ := 10

/-- The number of cinnamon balls Sheila bought -/
def cinnamon_balls : ℕ := family_members * days

theorem sheila_cinnamon_balls : cinnamon_balls = 50 := by
  sorry

end NUMINAMATH_CALUDE_sheila_cinnamon_balls_l1240_124042


namespace NUMINAMATH_CALUDE_interview_problem_l1240_124093

/-- The number of people to be hired -/
def people_hired : ℕ := 3

/-- The probability of two specific individuals being hired together -/
def prob_two_hired : ℚ := 1 / 70

/-- The total number of people interviewed -/
def total_interviewed : ℕ := 21

theorem interview_problem :
  (people_hired = 3) →
  (prob_two_hired = 1 / 70) →
  (total_interviewed = 21) := by
  sorry

end NUMINAMATH_CALUDE_interview_problem_l1240_124093


namespace NUMINAMATH_CALUDE_circle_properties_l1240_124031

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem circle_properties :
  ∃ (a : ℝ),
    -- P is on circle C
    C a (a + 1) ∧
    -- |PQ| = 2√10
    (a - (-2))^2 + ((a + 1) - 3)^2 = 40 ∧
    -- Slope of PQ is 1/3
    (3 - (a + 1)) / (-2 - a) = 1/3 ∧
    -- For any point M on C
    ∀ (m n : ℝ), C m n →
      -- Maximum value of |MQ| is 6√2
      (m - (-2))^2 + (n - 3)^2 ≤ 72 ∧
      -- Minimum value of |MQ| is 2√2
      (m - (-2))^2 + (n - 3)^2 ≥ 8 ∧
      -- Maximum value of (n-3)/(m+2) is 2 + √3
      (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 ∧
      -- Minimum value of (n-3)/(m+2) is 2 - √3
      (n - 3) / (m + 2) ≥ 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1240_124031


namespace NUMINAMATH_CALUDE_square_area_ratio_l1240_124083

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 3 * (4 * b)) : a^2 = 9 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1240_124083


namespace NUMINAMATH_CALUDE_genevieve_thermoses_l1240_124088

/-- Proves the number of thermoses Genevieve drank given the conditions -/
theorem genevieve_thermoses (total_coffee : ℚ) (num_thermoses : ℕ) (genevieve_consumption : ℚ) : 
  total_coffee = 4.5 ∧ num_thermoses = 18 ∧ genevieve_consumption = 6 →
  (genevieve_consumption / (total_coffee * 8 / num_thermoses) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_thermoses_l1240_124088


namespace NUMINAMATH_CALUDE_work_completion_time_l1240_124002

/-- The time it takes for worker c to complete the work alone -/
def time_c : ℝ := 12

/-- The time it takes for worker a to complete the work alone -/
def time_a : ℝ := 16

/-- The time it takes for worker b to complete the work alone -/
def time_b : ℝ := 6

/-- The time it takes for workers a, b, and c to complete the work together -/
def time_abc : ℝ := 3.2

theorem work_completion_time :
  1 / time_a + 1 / time_b + 1 / time_c = 1 / time_abc :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1240_124002


namespace NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_300_l1240_124074

theorem percentage_of_amount (amount : ℝ) (percentage : ℝ) :
  (percentage / 100) * amount = (percentage * amount) / 100 := by sorry

theorem twenty_five_percent_of_300 :
  (25 : ℝ) / 100 * 300 = 75 := by sorry

end NUMINAMATH_CALUDE_percentage_of_amount_twenty_five_percent_of_300_l1240_124074


namespace NUMINAMATH_CALUDE_f_properties_l1240_124075

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^2 + 12 * x - 15

-- Theorem statement
theorem f_properties :
  -- 1. Zeros of f(x)
  (∃ x : ℝ, f x = 0 ↔ x = -5 ∨ x = 1) ∧
  -- 2. Minimum and maximum values on [-3, 3]
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≥ -27) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = -27) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 3 → f x ≤ 48) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = 48) ∧
  -- 3. f(x) is increasing on [-2, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici (-2) → x₂ ∈ Set.Ici (-2) → x₁ < x₂ → f x₁ < f x₂) :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l1240_124075


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1240_124047

theorem polynomial_factorization (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 + 2 * y^3 = 2 * y * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1240_124047


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1240_124027

theorem arithmetic_equality : 202 - 101 + 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1240_124027


namespace NUMINAMATH_CALUDE_men_with_ac_at_least_12_l1240_124089

theorem men_with_ac_at_least_12 (total : ℕ) (married : ℕ) (tv : ℕ) (radio : ℕ) (all_four : ℕ) (ac : ℕ) :
  total = 100 →
  married = 82 →
  tv = 75 →
  radio = 85 →
  all_four = 12 →
  all_four ≤ ac →
  ac ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_men_with_ac_at_least_12_l1240_124089


namespace NUMINAMATH_CALUDE_next_term_is_512x4_l1240_124029

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 2
  | 1 => 8 * x
  | 2 => 32 * x^2
  | 3 => 128 * x^3
  | (n + 4) => geometric_sequence x n

theorem next_term_is_512x4 (x : ℝ) : geometric_sequence x 4 = 512 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_next_term_is_512x4_l1240_124029


namespace NUMINAMATH_CALUDE_sugar_calculation_l1240_124053

-- Define the original amount of sugar in the recipe
def original_sugar : ℚ := 5 + 3/4

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 1 + 11/12

-- Theorem statement
theorem sugar_calculation : 
  recipe_fraction * original_sugar = result := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1240_124053


namespace NUMINAMATH_CALUDE_unique_solution_l1240_124004

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, 2 * (f x) - 21 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1240_124004


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1240_124035

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1240_124035


namespace NUMINAMATH_CALUDE_single_elimination_512_players_games_l1240_124020

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- Calculates the number of games required to determine a champion in a single-elimination tournament. -/
def games_required (t : Tournament) : ℕ :=
  if t.single_elimination then t.num_players - 1 else 0

/-- Theorem stating that a single-elimination tournament with 512 players requires 511 games. -/
theorem single_elimination_512_players_games (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.single_elimination = true) : 
  games_required t = 511 := by
  sorry

#eval games_required ⟨512, true⟩

end NUMINAMATH_CALUDE_single_elimination_512_players_games_l1240_124020


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sequence_triangle_area_l1240_124046

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  3 * t.b^2 = 2 * t.a * t.c * (1 + Real.cos t.B)

-- Define arithmetic sequence property
def isArithmeticSequence (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Theorem 1
theorem triangle_arithmetic_sequence (t : Triangle) 
  (h : satisfiesCondition t) : isArithmeticSequence t.a t.b t.c := by
  sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 3) (h2 : t.b = 5) (h3 : satisfiesCondition t) : 
  (1/2 * t.a * t.b * Real.sin t.C) = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sequence_triangle_area_l1240_124046


namespace NUMINAMATH_CALUDE_shaded_probability_is_half_l1240_124081

/-- Represents an isosceles triangle with base 2 cm and height 4 cm -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  is_isosceles : base = 2 ∧ height = 4

/-- Represents the division of the triangle into 4 regions -/
structure TriangleDivision where
  triangle : IsoscelesTriangle
  num_regions : ℕ
  is_divided : num_regions = 4

/-- Represents the shading of two opposite regions -/
structure ShadedRegions where
  division : TriangleDivision
  num_shaded : ℕ
  are_opposite : num_shaded = 2

/-- The probability of the spinner landing on a shaded region -/
def shaded_probability (shaded : ShadedRegions) : ℚ :=
  shaded.num_shaded / shaded.division.num_regions

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem shaded_probability_is_half (shaded : ShadedRegions) :
  shaded_probability shaded = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_half_l1240_124081


namespace NUMINAMATH_CALUDE_total_rainfall_2004_l1240_124048

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total rainfall in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_2004_l1240_124048


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1240_124092

-- Define the common ratios
variable (p r : ℝ)

-- Define the first term of both sequences
variable (k : ℝ)

-- Define the geometric sequences
def a (n : ℕ) : ℝ := k * p^n
def b (n : ℕ) : ℝ := k * r^n

-- State the theorem
theorem sum_of_common_ratios_is_three 
  (h1 : p ≠ 1) 
  (h2 : r ≠ 1) 
  (h3 : p ≠ r) 
  (h4 : k ≠ 0) 
  (h5 : a 4 - b 4 = 4 * (a 2 - b 2)) : 
  p + r = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l1240_124092


namespace NUMINAMATH_CALUDE_egg_game_probabilities_l1240_124013

/-- Represents the color of an egg -/
inductive EggColor
| Yellow
| Red
| Blue

/-- Represents the game setup -/
structure EggGame where
  total_eggs : Nat
  yellow_eggs : Nat
  red_eggs : Nat
  blue_eggs : Nat
  game_fee : Int
  same_color_reward : Int
  diff_color_reward : Int

/-- Defines the game rules -/
def game : EggGame :=
  { total_eggs := 9
  , yellow_eggs := 3
  , red_eggs := 3
  , blue_eggs := 3
  , game_fee := 10
  , same_color_reward := 100
  , diff_color_reward := 10 }

/-- Event A: Picking a yellow egg on the first draw -/
def eventA (g : EggGame) : Rat :=
  g.yellow_eggs / g.total_eggs

/-- Event B: Winning the maximum reward -/
def eventB (g : EggGame) : Rat :=
  3 / Nat.choose g.total_eggs 3

/-- Probability of both events A and B occurring -/
def eventAB (g : EggGame) : Rat :=
  1 / Nat.choose g.total_eggs 3

/-- Expected profit from playing the game -/
def expectedProfit (g : EggGame) : Rat :=
  (g.same_color_reward - g.game_fee) * eventB g +
  (g.diff_color_reward - g.game_fee) * (9 / 28) +
  (-g.game_fee) * (18 / 28)

theorem egg_game_probabilities :
  eventA game = 1/3 ∧
  eventB game = 1/28 ∧
  eventAB game = eventA game * eventB game ∧
  expectedProfit game < 0 :=
sorry

end NUMINAMATH_CALUDE_egg_game_probabilities_l1240_124013


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1240_124062

theorem right_triangle_perimeter (a b c x : ℝ) : 
  a * b / 2 = 200 →  -- area condition
  b = 20 →  -- given leg length
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a = x →  -- other leg length
  x + b + c = 40 + 20 * Real.sqrt 2 :=  -- perimeter
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1240_124062


namespace NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1240_124040

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 35) :
  star a b = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_star_value_for_specific_conditions_l1240_124040


namespace NUMINAMATH_CALUDE_min_value_expression_l1240_124078

theorem min_value_expression (a b c d : ℝ) (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2 ≥ 9 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), b₀ > d₀ ∧ d₀ > c₀ ∧ c₀ > a₀ ∧ b₀ ≠ 0 ∧
    ((a₀ + b₀)^2 + (b₀ - c₀)^2 + (d₀ - c₀)^2 + (c₀ - a₀)^2) / b₀^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1240_124078


namespace NUMINAMATH_CALUDE_locker_labeling_cost_l1240_124068

/-- Calculates the cost of labeling lockers given the number of lockers and cost per digit -/
def labelingCost (numLockers : ℕ) (costPerDigit : ℚ) : ℚ :=
  let singleDigitCost := (min numLockers 9 : ℕ) * costPerDigit
  let doubleDigitCost := (min (numLockers - 9) 90 : ℕ) * 2 * costPerDigit
  let tripleDigitCost := (min (numLockers - 99) 900 : ℕ) * 3 * costPerDigit
  let quadrupleDigitCost := (max (numLockers - 999) 0 : ℕ) * 4 * costPerDigit
  singleDigitCost + doubleDigitCost + tripleDigitCost + quadrupleDigitCost

theorem locker_labeling_cost :
  labelingCost 2999 (3 / 100) = 32667 / 100 :=
by sorry

end NUMINAMATH_CALUDE_locker_labeling_cost_l1240_124068


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l1240_124032

def total_dogs : Nat := 12
def group1_size : Nat := 4
def group2_size : Nat := 5
def group3_size : Nat := 3

theorem dog_grouping_combinations :
  (total_dogs = group1_size + group2_size + group3_size) →
  (Nat.choose (total_dogs - 2) (group1_size - 1) * Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) = 5775) := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l1240_124032


namespace NUMINAMATH_CALUDE_sixth_plate_cookies_l1240_124067

def cookie_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 5
  | 1 => 7
  | k + 2 => cookie_sequence n (k + 1) + (k + 2)

theorem sixth_plate_cookies :
  cookie_sequence 5 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sixth_plate_cookies_l1240_124067


namespace NUMINAMATH_CALUDE_cornelias_asian_countries_l1240_124090

theorem cornelias_asian_countries 
  (total_countries : ℕ) 
  (european_countries : ℕ) 
  (south_american_countries : ℕ) 
  (h1 : total_countries = 42)
  (h2 : european_countries = 20)
  (h3 : south_american_countries = 10)
  (h4 : 2 * (total_countries - european_countries - south_american_countries) / 2 = 
       total_countries - european_countries - south_american_countries) :
  (total_countries - european_countries - south_american_countries) / 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cornelias_asian_countries_l1240_124090


namespace NUMINAMATH_CALUDE_football_throw_distance_l1240_124044

/-- Proves that Kyle threw the ball 24 yards farther than Parker -/
theorem football_throw_distance (parker_distance : ℝ) (grant_distance : ℝ) (kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_distance_l1240_124044


namespace NUMINAMATH_CALUDE_tree_planting_temperature_reduction_l1240_124097

theorem tree_planting_temperature_reduction 
  (initial_temp : ℝ) 
  (cost_per_tree : ℝ) 
  (temp_reduction_per_tree : ℝ) 
  (total_cost : ℝ) 
  (h1 : initial_temp = 80)
  (h2 : cost_per_tree = 6)
  (h3 : temp_reduction_per_tree = 0.1)
  (h4 : total_cost = 108) :
  initial_temp - (total_cost / cost_per_tree * temp_reduction_per_tree) = 78.2 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_temperature_reduction_l1240_124097


namespace NUMINAMATH_CALUDE_least_cookies_l1240_124059

theorem least_cookies (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 3 ∧ 
  b % 9 = 7 ∧
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 3 ∧ c % 9 = 7 → b ≤ c) → 
  b = 179 := by
sorry

end NUMINAMATH_CALUDE_least_cookies_l1240_124059


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1240_124030

/-- A line parallel to y = 1/2x - 1 passing through (0, 3) has equation y = 1/2x + 3 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- The line has equation y = kx + b
  k = 1/2 →                    -- The line is parallel to y = 1/2x - 1
  3 = b →                      -- The line passes through (0, 3)
  ∀ x y : ℝ, y = 1/2 * x + 3   -- The equation of the line is y = 1/2x + 3
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1240_124030


namespace NUMINAMATH_CALUDE_accumulate_small_steps_necessary_not_sufficient_l1240_124009

-- Define the concept of "reaching a thousand miles"
def reach_thousand_miles : Prop := sorry

-- Define the concept of "accumulating small steps"
def accumulate_small_steps : Prop := sorry

-- Xunzi's saying as an axiom
axiom xunzi_saying : ¬accumulate_small_steps → ¬reach_thousand_miles

-- Define what it means to be a necessary condition
def is_necessary_condition (condition goal : Prop) : Prop :=
  ¬condition → ¬goal

-- Define what it means to be a sufficient condition
def is_sufficient_condition (condition goal : Prop) : Prop :=
  condition → goal

-- Theorem to prove
theorem accumulate_small_steps_necessary_not_sufficient :
  (is_necessary_condition accumulate_small_steps reach_thousand_miles) ∧
  ¬(is_sufficient_condition accumulate_small_steps reach_thousand_miles) := by
  sorry

end NUMINAMATH_CALUDE_accumulate_small_steps_necessary_not_sufficient_l1240_124009


namespace NUMINAMATH_CALUDE_number_added_to_product_l1240_124006

theorem number_added_to_product (a b : Int) (h1 : a = -2) (h2 : b = -3) :
  ∃ x : Int, a * b + x = 65 ∧ x = 59 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_product_l1240_124006


namespace NUMINAMATH_CALUDE_sin_45_cos_15_plus_cos_45_sin_15_l1240_124023

theorem sin_45_cos_15_plus_cos_45_sin_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_45_cos_15_plus_cos_45_sin_15_l1240_124023


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l1240_124070

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : ℕ
  captain_age : ℕ
  team_average_age : ℕ
  wicket_keeper_age_difference : ℕ

/-- The age difference between the wicket keeper and the captain is correct
    if it satisfies the given conditions -/
def correct_age_difference (team : CricketTeam) : Prop :=
  let remaining_players := team.total_members - 2
  let remaining_average := team.team_average_age - 1
  let total_age := team.team_average_age * team.total_members
  let remaining_age := remaining_average * remaining_players
  total_age = remaining_age + team.captain_age + (team.captain_age + team.wicket_keeper_age_difference)

/-- The theorem stating that the wicket keeper is 3 years older than the captain -/
theorem wicket_keeper_age_difference (team : CricketTeam) 
  (h1 : team.total_members = 11)
  (h2 : team.captain_age = 26)
  (h3 : team.team_average_age = 23)
  : correct_age_difference team → team.wicket_keeper_age_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l1240_124070


namespace NUMINAMATH_CALUDE_taxi_fare_calculation_l1240_124080

/-- Represents the fare structure of a taxi service -/
structure TaxiFare where
  base_fare : ℝ
  per_mile_charge : ℝ

/-- Calculates the total fare for a given distance -/
def total_fare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.base_fare + tf.per_mile_charge * distance

theorem taxi_fare_calculation (tf : TaxiFare) 
  (h1 : tf.base_fare = 40)
  (h2 : total_fare tf 80 = 200) :
  total_fare tf 100 = 240 := by
sorry

end NUMINAMATH_CALUDE_taxi_fare_calculation_l1240_124080


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1240_124050

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l1240_124050


namespace NUMINAMATH_CALUDE_x_minus_y_equals_four_l1240_124095

theorem x_minus_y_equals_four (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_four_l1240_124095


namespace NUMINAMATH_CALUDE_real_part_of_inverse_difference_l1240_124016

theorem real_part_of_inverse_difference (z : ℂ) (h1 : z ≠ 0) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1/4 :=
sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_difference_l1240_124016


namespace NUMINAMATH_CALUDE_likelihood_number_is_probability_l1240_124019

/-- A number representing the likelihood of a random event occurring -/
def likelihood_number : ℝ := sorry

/-- The term for the number representing the likelihood of a random event occurring -/
def probability_term : String := sorry

/-- The theorem stating that the term for the number representing the likelihood of a random event occurring is "probability" -/
theorem likelihood_number_is_probability : probability_term = "probability" := by
  sorry

end NUMINAMATH_CALUDE_likelihood_number_is_probability_l1240_124019


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1240_124054

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1240_124054


namespace NUMINAMATH_CALUDE_remainder_2984_times_3998_mod_1000_l1240_124025

theorem remainder_2984_times_3998_mod_1000 : (2984 * 3998) % 1000 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2984_times_3998_mod_1000_l1240_124025


namespace NUMINAMATH_CALUDE_inequality_proof_l1240_124065

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1240_124065


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1240_124079

theorem expression_equals_negative_one :
  -5 * (2/3) + 6 * (2/7) + (1/3) * (-5) - (2/7) * (-8) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1240_124079


namespace NUMINAMATH_CALUDE_parallelogram_height_eq_two_thirds_rectangle_side_l1240_124071

/-- Given a rectangle with side length r and a parallelogram with base b = 1.5r,
    prove that the height of the parallelogram h = 2r/3 when their areas are equal. -/
theorem parallelogram_height_eq_two_thirds_rectangle_side 
  (r : ℝ) (b h : ℝ) (h_positive : r > 0) : 
  b = 1.5 * r → r * r = b * h → h = 2 * r / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_eq_two_thirds_rectangle_side_l1240_124071


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l1240_124001

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0. -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) →
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l1240_124001


namespace NUMINAMATH_CALUDE_inequality_solution_l1240_124008

theorem inequality_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (2 * x) / (x + 1) + (x - 3) / (3 * x) ≤ 4 ↔ x < -1 ∨ x > -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1240_124008


namespace NUMINAMATH_CALUDE_unique_solution_l1240_124087

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 3*x + 3*y + 1 = 0 ∧
  x^3 * y - x * y^3 - 3*x^2 + 3*y^2 + 3 = 0

/-- The theorem stating that (2, 1) is the only solution to the system -/
theorem unique_solution :
  ∀ x y : ℝ, system x y ↔ x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1240_124087


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_l1240_124049

theorem largest_multiple_of_seven (n : ℤ) : n = 147 ↔ 
  (∃ k : ℤ, n = 7 * k) ∧ 
  (-n > -150) ∧ 
  (∀ m : ℤ, (∃ j : ℤ, m = 7 * j) ∧ (-m > -150) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_l1240_124049


namespace NUMINAMATH_CALUDE_coffee_consumption_l1240_124011

/-- 
Given:
- A: number of times Maria goes to the coffee shop per day
- B: number of days in the week Maria orders X cups per visit
- X: number of cups Maria orders per visit on the first B days
- Y: number of cups Maria orders per visit on the remaining (7 - B) days
- Z: total number of cups of coffee Maria orders in a week

Prove that Z = (A * B * X) + (A * (7 - B) * Y)
-/
theorem coffee_consumption 
  (A B X Y Z : ℕ) 
  (h1 : B ≤ 7) 
  (h2 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := by
  sorry

end NUMINAMATH_CALUDE_coffee_consumption_l1240_124011


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1240_124098

theorem inequality_and_minimum_value 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hxy : x + y = 1) : 
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧ 
  (∃ (min : ℝ), min = 9 ∧ ∀ (z w : ℝ), 0 < z → 0 < w → z + w = 1 → 1/z + 4/w ≥ min) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1240_124098


namespace NUMINAMATH_CALUDE_valid_selections_eq_48_l1240_124096

/-- The number of ways to select k items from n items -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- The number of valid selections given the problem constraints -/
def valid_selections : ℕ :=
  arrangements 5 3 - arrangements 4 2

theorem valid_selections_eq_48 : valid_selections = 48 := by sorry

end NUMINAMATH_CALUDE_valid_selections_eq_48_l1240_124096


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l1240_124026

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(∀ x ∈ domain, f x = -f (-x)) ∧ ¬(∀ x ∈ domain, f x = f (-x)) :=
sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l1240_124026


namespace NUMINAMATH_CALUDE_sunset_time_l1240_124072

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def sunrise : Time := { hours := 6, minutes := 45 }
def daylight : Duration := { hours := 11, minutes := 12 }

theorem sunset_time :
  addTime sunrise daylight = { hours := 17, minutes := 57 } := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_l1240_124072


namespace NUMINAMATH_CALUDE_square_increase_l1240_124076

theorem square_increase (a : ℕ) : (a + 1)^2 - a^2 = 1001 → a = 500 := by
  sorry

end NUMINAMATH_CALUDE_square_increase_l1240_124076


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1240_124036

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 96 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1240_124036


namespace NUMINAMATH_CALUDE_seed_germination_probabilities_l1240_124057

/-- The number of seeds in each pit -/
def seeds_per_pit : ℕ := 3

/-- The probability of a single seed germinating -/
def germination_prob : ℝ := 0.5

/-- The number of pits -/
def num_pits : ℕ := 3

/-- The probability that at least one seed germinates in a pit -/
def prob_at_least_one_germinates : ℝ := 1 - (1 - germination_prob) ^ seeds_per_pit

/-- The probability that exactly two pits need replanting -/
def prob_exactly_two_need_replanting : ℝ := 
  (num_pits.choose 2) * (1 - prob_at_least_one_germinates) ^ 2 * prob_at_least_one_germinates

/-- The probability that at least one pit needs replanting -/
def prob_at_least_one_needs_replanting : ℝ := 1 - prob_at_least_one_germinates ^ num_pits

theorem seed_germination_probabilities :
  (prob_at_least_one_germinates = 0.875) ∧
  (prob_exactly_two_need_replanting = 0.713) ∧
  (prob_at_least_one_needs_replanting = 0.330) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_probabilities_l1240_124057


namespace NUMINAMATH_CALUDE_integer_root_values_l1240_124084

theorem integer_root_values (a : ℤ) : 
  (∃ x : ℤ, x^3 + 3*x^2 + a*x + 9 = 0) ↔ 
  a ∈ ({-109, -21, -13, 3, 11, 53} : Set ℤ) :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l1240_124084


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l1240_124082

theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 ∧ hole_depth = 10 → 
  sphere_radius^2 = (hole_width/2)^2 + (sphere_radius - hole_depth)^2 →
  sphere_radius = 16.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l1240_124082


namespace NUMINAMATH_CALUDE_exactly_two_true_l1240_124052

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Define equilateral triangle
def equilateral (t : Triangle) : Prop := 
  ∀ i : Fin 3, t.angles i = 60

-- Proposition 1
def prop1 : Prop := 
  ∀ t1 t2 : Triangle, t1.area = t2.area → congruent t1 t2

-- Proposition 2
def prop2 : Prop := 
  ∃ a b : ℝ, a * b = 0 ∧ a ≠ 0

-- Proposition 3
def prop3 : Prop := 
  ∀ t : Triangle, ¬equilateral t → ∃ i : Fin 3, t.angles i ≠ 60

-- Main theorem
theorem exactly_two_true : 
  (¬prop1 ∧ prop2 ∧ prop3) ∨
  (prop1 ∧ prop2 ∧ ¬prop3) ∨
  (prop1 ∧ ¬prop2 ∧ prop3) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_true_l1240_124052


namespace NUMINAMATH_CALUDE_ellipse_equation_constants_l1240_124038

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  passingPoint : Point
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Check if a point satisfies the ellipse equation -/
def satisfiesEllipseEquation (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.h)^2 / e.a^2 + (p.y - e.k)^2 / e.b^2 = 1

/-- The main theorem to prove -/
theorem ellipse_equation_constants : ∃ (e : Ellipse),
  e.focus1 = ⟨2, 2⟩ ∧
  e.focus2 = ⟨2, 6⟩ ∧
  e.passingPoint = ⟨14, -3⟩ ∧
  e.a > 0 ∧
  e.b > 0 ∧
  satisfiesEllipseEquation e e.passingPoint ∧
  e.a = 8 * Real.sqrt 3 ∧
  e.b = 14 ∧
  e.h = 2 ∧
  e.k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_constants_l1240_124038


namespace NUMINAMATH_CALUDE_cube_difference_prime_factor_l1240_124039

theorem cube_difference_prime_factor (a b p : ℕ) : 
  Nat.Prime p → a^3 - b^3 = 633 * p → a = 16 ∧ b = 13 :=
by sorry

end NUMINAMATH_CALUDE_cube_difference_prime_factor_l1240_124039


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l1240_124022

-- Define the total population size
def N : ℕ := 1200

-- Define the sample size
def n : ℕ := 30

-- Define the systematic sampling interval
def k : ℕ := N / n

-- Theorem to prove
theorem systematic_sampling_interval :
  k = 40 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l1240_124022


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l1240_124014

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l1240_124014


namespace NUMINAMATH_CALUDE_range_of_m_l1240_124005

/-- Given points A(1,0) and B(4,0) in the Cartesian plane, and a point P on the line x-y+m=0 
    such that 2PA = PB, the range of possible values for m is [-2√2, 2√2]. -/
theorem range_of_m (m : ℝ) : 
  ∃ (x y : ℝ), 
    (x - y + m = 0) ∧ 
    (2 * ((x - 1)^2 + y^2) = (x - 4)^2 + y^2) →
    -2 * Real.sqrt 2 ≤ m ∧ m ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1240_124005


namespace NUMINAMATH_CALUDE_choose_one_friend_from_ten_l1240_124086

def number_of_friends : ℕ := 10
def friends_to_choose : ℕ := 1

theorem choose_one_friend_from_ten :
  Nat.choose number_of_friends friends_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_choose_one_friend_from_ten_l1240_124086


namespace NUMINAMATH_CALUDE_inequality_preservation_l1240_124060

theorem inequality_preservation (x y : ℝ) : x < y → 3 - x > 3 - y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1240_124060


namespace NUMINAMATH_CALUDE_circle_area_from_square_perimeter_l1240_124085

/-- The area of a circle that shares a center with a square of perimeter 40 feet -/
theorem circle_area_from_square_perimeter : ∃ (circle_area : ℝ), 
  circle_area = 50 * Real.pi ∧ 
  ∃ (square_side : ℝ), 
    4 * square_side = 40 ∧
    circle_area = Real.pi * (square_side * Real.sqrt 2 / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_square_perimeter_l1240_124085


namespace NUMINAMATH_CALUDE_ksyusha_wednesday_travel_time_l1240_124073

/-- The time taken for Ksyusha to travel from home to school on Wednesday -/
theorem ksyusha_wednesday_travel_time :
  -- Given conditions
  ∀ (S : ℝ) (v : ℝ),
  S > 0 → v > 0 →
  -- Tuesday's scenario
  (2 * S / v + S / (2 * v) = 30) →
  -- Wednesday's scenario
  (S / v + 2 * S / (2 * v) = 24) :=
by sorry

end NUMINAMATH_CALUDE_ksyusha_wednesday_travel_time_l1240_124073


namespace NUMINAMATH_CALUDE_negative_comparison_l1240_124051

theorem negative_comparison : -2023 > -2024 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l1240_124051


namespace NUMINAMATH_CALUDE_amy_school_year_hours_l1240_124010

/-- Calculates the number of hours Amy needs to work per week during the school year --/
def school_year_hours_per_week (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℚ :=
  let summer_total_hours := summer_hours_per_week * summer_weeks
  let hourly_rate := summer_earnings / summer_total_hours
  let school_year_total_hours := school_year_earnings / hourly_rate
  school_year_total_hours / school_year_weeks

/-- Theorem stating that Amy needs to work 9 hours per week during the school year --/
theorem amy_school_year_hours : 
  school_year_hours_per_week 36 10 3000 40 3000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_hours_l1240_124010


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1240_124043

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1240_124043


namespace NUMINAMATH_CALUDE_point_on_line_l1240_124091

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨3, 0⟩
  let p2 : Point := ⟨11, 4⟩
  let p3 : Point := ⟨19, 8⟩
  collinear p1 p2 p3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1240_124091


namespace NUMINAMATH_CALUDE_min_ellipse_eccentricity_l1240_124045

/-- Given an ellipse C: x²/a² + y²/b² = 1 where a > b > 0, with foci F₁ and F₂,
    and right vertex A. A line l passing through F₁ intersects C at P and Q.
    AP · AQ = (1/2)(a+c)². The minimum eccentricity of C is 1 - √2/2. -/
theorem min_ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := c / a
  ∀ P Q : ℝ × ℝ,
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
  (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
  ∃ m : ℝ, (P.1 = m * P.2 - c) ∧ (Q.1 = m * Q.2 - c) →
  ((P.1 - a) * (Q.1 - a) + P.2 * Q.2 = (1/2) * (a + c)^2) →
  e ≥ 1 - Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_ellipse_eccentricity_l1240_124045


namespace NUMINAMATH_CALUDE_power_inequality_set_l1240_124056

theorem power_inequality_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | a^(x+3) > a^(2*x)} = {x : ℝ | x > 3} := by sorry

end NUMINAMATH_CALUDE_power_inequality_set_l1240_124056
