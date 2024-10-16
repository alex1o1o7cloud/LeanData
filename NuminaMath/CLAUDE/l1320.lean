import Mathlib

namespace NUMINAMATH_CALUDE_polygon_properties_l1320_132082

/-- A polygon with interior angle sum of 1080 degrees has 8 sides and exterior angle sum of 360 degrees -/
theorem polygon_properties (n : ℕ) (interior_sum : ℝ) (h : interior_sum = 1080) :
  (n - 2) * 180 = interior_sum ∧ n = 8 ∧ 360 = (n : ℝ) * (360 / n) := by
  sorry

end NUMINAMATH_CALUDE_polygon_properties_l1320_132082


namespace NUMINAMATH_CALUDE_cubic_not_prime_l1320_132015

theorem cubic_not_prime (n : ℕ+) : ¬ Nat.Prime (n.val^3 - 7*n.val^2 + 16*n.val - 12) := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_prime_l1320_132015


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1320_132090

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1320_132090


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l1320_132048

theorem sufficient_not_necessary_implies_necessary_not_sufficient 
  (A B : Prop) (h : (A → B) ∧ ¬(B → A)) : 
  ((¬B → ¬A) ∧ ¬(¬A → ¬B)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l1320_132048


namespace NUMINAMATH_CALUDE_ping_pong_balls_count_l1320_132081

/-- The number of ping-pong balls bought with tax -/
def B : ℕ := 60

/-- The sales tax rate -/
def tax_rate : ℚ := 16 / 100

theorem ping_pong_balls_count :
  (B : ℚ) * (1 + tax_rate) = (B + 3 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_balls_count_l1320_132081


namespace NUMINAMATH_CALUDE_sixth_term_is_half_l1320_132083

/-- Geometric sequence with first term 16 and common ratio 1/2 -/
def geometricSequence : ℕ → ℚ
  | 0 => 16
  | n + 1 => (geometricSequence n) / 2

/-- The sixth term of the geometric sequence is 1/2 -/
theorem sixth_term_is_half : geometricSequence 5 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_half_l1320_132083


namespace NUMINAMATH_CALUDE_chess_tournament_25_players_l1320_132064

/-- Calculate the number of games in a chess tournament -/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 25 players, where each player plays twice against every other player, the total number of games is 1200 -/
theorem chess_tournament_25_players :
  2 * chess_tournament_games 25 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_25_players_l1320_132064


namespace NUMINAMATH_CALUDE_composition_value_l1320_132019

/-- Given two functions f and g, prove that g(f(3)) = 1902 -/
theorem composition_value (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^3 - 2) 
  (hg : ∀ x, g x = 3*x^2 + x + 2) : 
  g (f 3) = 1902 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l1320_132019


namespace NUMINAMATH_CALUDE_dance_steps_l1320_132094

/-- Nancy and Jason are learning to dance. This function calculates the total number of times they step on each other's feet. -/
def total_steps (jason_steps : ℕ) (nancy_multiplier : ℕ) : ℕ :=
  jason_steps + nancy_multiplier * jason_steps

/-- Theorem stating that Nancy and Jason step on each other's feet 32 times in total. -/
theorem dance_steps : total_steps 8 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_dance_steps_l1320_132094


namespace NUMINAMATH_CALUDE_max_sum_after_pyramid_addition_l1320_132099

/-- Represents a polyhedron with faces, edges, and vertices -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Represents the result of adding a pyramid to a face of a polyhedron -/
structure PyramidAddition where
  newFaces : ℕ
  newEdges : ℕ
  newVertices : ℕ

/-- Calculates the sum of faces, edges, and vertices after adding a pyramid -/
def sumAfterPyramidAddition (p : Polyhedron) (pa : PyramidAddition) : ℕ :=
  (p.faces - 1 + pa.newFaces) + (p.edges + pa.newEdges) + (p.vertices + pa.newVertices)

/-- The pentagonal prism -/
def pentagonalPrism : Polyhedron :=
  { faces := 7, edges := 15, vertices := 10 }

/-- Adding a pyramid to a pentagonal face -/
def pentagonalFaceAddition : PyramidAddition :=
  { newFaces := 5, newEdges := 5, newVertices := 1 }

/-- Adding a pyramid to a quadrilateral face -/
def quadrilateralFaceAddition : PyramidAddition :=
  { newFaces := 4, newEdges := 4, newVertices := 1 }

/-- Theorem: The maximum sum of faces, edges, and vertices after adding a pyramid is 42 -/
theorem max_sum_after_pyramid_addition :
  (max 
    (sumAfterPyramidAddition pentagonalPrism pentagonalFaceAddition)
    (sumAfterPyramidAddition pentagonalPrism quadrilateralFaceAddition)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_after_pyramid_addition_l1320_132099


namespace NUMINAMATH_CALUDE_triangle_tan_A_l1320_132033

theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a / b = (b + Real.sqrt 3 * c) / a →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  Real.tan A = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l1320_132033


namespace NUMINAMATH_CALUDE_donut_circumference_ratio_l1320_132045

/-- The ratio of the outer circumference to the inner circumference of a donut-shaped object
    is equal to the ratio of their respective radii. -/
theorem donut_circumference_ratio (inner_radius outer_radius : ℝ)
  (h1 : inner_radius = 2)
  (h2 : outer_radius = 6) :
  (2 * Real.pi * outer_radius) / (2 * Real.pi * inner_radius) = outer_radius / inner_radius := by
  sorry

end NUMINAMATH_CALUDE_donut_circumference_ratio_l1320_132045


namespace NUMINAMATH_CALUDE_ainsley_win_probability_l1320_132012

/-- A fair six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

/-- The probability of rolling a specific outcome on a fair six-sided die -/
def prob_roll (outcome : Die) : ℚ := 1 / 6

/-- Whether a roll is a multiple of 3 -/
def is_multiple_of_three (roll : Die) : Prop :=
  roll = Die.three ∨ roll = Die.six

/-- The probability of rolling a multiple of 3 -/
def prob_multiple_of_three : ℚ :=
  (prob_roll Die.three) + (prob_roll Die.six)

/-- The probability of rolling a non-multiple of 3 -/
def prob_non_multiple_of_three : ℚ :=
  1 - prob_multiple_of_three

/-- The probability of Ainsley winning the game -/
theorem ainsley_win_probability :
  prob_multiple_of_three * prob_multiple_of_three = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ainsley_win_probability_l1320_132012


namespace NUMINAMATH_CALUDE_odd_function_sum_zero_l1320_132095

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum_zero (f : ℝ → ℝ) (h : IsOdd f) : 
  f (-1) + f 0 + f 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_sum_zero_l1320_132095


namespace NUMINAMATH_CALUDE_incident_ray_slope_l1320_132050

/-- Given a circle with center (2, -1) and a point P(-1, -3), prove that the slope
    of the line passing through P and the reflection of the circle's center
    across the x-axis is 4/3. -/
theorem incident_ray_slope (P : ℝ × ℝ) (C : ℝ × ℝ) :
  P = (-1, -3) →
  C = (2, -1) →
  let D : ℝ × ℝ := (C.1, -C.2)  -- Reflection of C across x-axis
  (D.2 - P.2) / (D.1 - P.1) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_incident_ray_slope_l1320_132050


namespace NUMINAMATH_CALUDE_class_point_system_l1320_132073

/-- Calculates the number of tasks required for a given number of points -/
def tasksRequired (points : ℕ) : ℕ :=
  let fullSets := (points - 1) / 3
  let taskMultiplier := min fullSets 2 + 1
  taskMultiplier * ((points + 2) / 3)

/-- The point-earning system for the class -/
theorem class_point_system (points : ℕ) :
  points = 18 → tasksRequired points = 10 :=
by
  sorry

#eval tasksRequired 18  -- Should output 10

end NUMINAMATH_CALUDE_class_point_system_l1320_132073


namespace NUMINAMATH_CALUDE_max_value_of_f_l1320_132006

def f (x : ℝ) : ℝ := -2 * x^2 + 16 * x - 14

theorem max_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = -14 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1320_132006


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1320_132088

theorem arithmetic_calculations :
  ((-0.9) + 1.5 = 0.6) ∧
  (1/2 + (-2/3) = -1/6) ∧
  (1 + (-1/2) + 1/3 + (-1/6) = 2/3) ∧
  (3 + 1/4 + (-2 - 3/5) + 5 + 3/4 + (-8 - 2/5) = -2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1320_132088


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l1320_132080

theorem sum_remainder_by_eight (n : ℤ) : (9 - n + (n + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l1320_132080


namespace NUMINAMATH_CALUDE_sample_size_equals_sampled_students_l1320_132013

/-- Represents a survey conducted on eighth-grade students -/
structure Survey where
  sampled_students : ℕ

/-- The sample size of a survey is equal to the number of sampled students -/
theorem sample_size_equals_sampled_students (s : Survey) : s.sampled_students = 1500 → s.sampled_students = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_sampled_students_l1320_132013


namespace NUMINAMATH_CALUDE_rectangle_split_area_l1320_132079

theorem rectangle_split_area (c : ℝ) : 
  let total_area : ℝ := 8
  let smaller_area : ℝ := total_area / 3
  let larger_area : ℝ := 2 * smaller_area
  let triangle_area : ℝ := 2 * (4 - c)
  (4 + total_area - triangle_area = larger_area) → c = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_split_area_l1320_132079


namespace NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l1320_132056

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ∃ (k : ℤ), (5 * n^2 + 10) ≠ k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l1320_132056


namespace NUMINAMATH_CALUDE_quartic_trinomial_m_value_l1320_132016

theorem quartic_trinomial_m_value (m : ℤ) : 
  (abs (m - 3) = 4) → (m - 7 ≠ 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quartic_trinomial_m_value_l1320_132016


namespace NUMINAMATH_CALUDE_num_regions_correct_l1320_132091

/-- The number of regions formed by n lines in a plane, where no two lines are parallel and no three lines are concurrent. -/
def num_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- Theorem stating that num_regions correctly calculates the number of regions. -/
theorem num_regions_correct (n : ℕ) :
  num_regions n = n * (n + 1) / 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_num_regions_correct_l1320_132091


namespace NUMINAMATH_CALUDE_probability_of_C_l1320_132062

def spinner_game (pA pB pC pD : ℚ) : Prop :=
  pA + pB + pC + pD = 1 ∧ pA ≥ 0 ∧ pB ≥ 0 ∧ pC ≥ 0 ∧ pD ≥ 0

theorem probability_of_C (pA pB pC pD : ℚ) :
  spinner_game pA pB pC pD → pA = 1/4 → pB = 1/3 → pC = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l1320_132062


namespace NUMINAMATH_CALUDE_break_time_is_30_minutes_l1320_132066

/-- Represents the travel scenario with three train stations -/
structure TravelScenario where
  /-- Time between each station in hours -/
  station_distance : ℝ
  /-- Total travel time including break in minutes -/
  total_time : ℝ

/-- Calculates the break time at the second station -/
def break_time (scenario : TravelScenario) : ℝ :=
  scenario.total_time - 2 * (scenario.station_distance * 60)

/-- Theorem stating that the break time is 30 minutes -/
theorem break_time_is_30_minutes (scenario : TravelScenario) 
  (h1 : scenario.station_distance = 2)
  (h2 : scenario.total_time = 270) : 
  break_time scenario = 30 := by
  sorry

end NUMINAMATH_CALUDE_break_time_is_30_minutes_l1320_132066


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_l1320_132032

theorem halloween_cleanup_time (
  egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (graffiti_cleanup_time : ℕ)
  (pumpkin_cleanup_time : ℕ)
  (num_eggs : ℕ)
  (num_tp_rolls : ℕ)
  (sq_ft_graffiti : ℕ)
  (num_pumpkins : ℕ)
  (h1 : egg_cleanup_time = 15)
  (h2 : tp_cleanup_time = 30)
  (h3 : graffiti_cleanup_time = 45)
  (h4 : pumpkin_cleanup_time = 10)
  (h5 : num_eggs = 60)
  (h6 : num_tp_rolls = 7)
  (h7 : sq_ft_graffiti = 8)
  (h8 : num_pumpkins = 5) :
  (num_eggs * egg_cleanup_time) / 60 +
  num_tp_rolls * tp_cleanup_time +
  sq_ft_graffiti * graffiti_cleanup_time +
  num_pumpkins * pumpkin_cleanup_time = 635 := by
sorry

end NUMINAMATH_CALUDE_halloween_cleanup_time_l1320_132032


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l1320_132089

/-- Proves that given a total distance of 24 kilometers, where half is traveled by foot
    and 6 kilometers by car, the fraction of the distance traveled by bus is 1/4. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_distance : ℝ) (car_distance : ℝ) :
  total_distance = 24 →
  foot_distance = total_distance / 2 →
  car_distance = 6 →
  (total_distance - (foot_distance + car_distance)) / total_distance = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_bus_travel_fraction_l1320_132089


namespace NUMINAMATH_CALUDE_sum_first_11_even_numbers_eq_132_l1320_132024

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  (first_n_even_numbers n).sum

theorem sum_first_11_even_numbers_eq_132 : sum_first_n_even_numbers 11 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_11_even_numbers_eq_132_l1320_132024


namespace NUMINAMATH_CALUDE_increasing_order_x_z_y_l1320_132022

theorem increasing_order_x_z_y (x : ℝ) (h : 0.8 < x ∧ x < 0.9) :
  x < x^(x^x) ∧ x^(x^x) < x^x := by
  sorry

end NUMINAMATH_CALUDE_increasing_order_x_z_y_l1320_132022


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l1320_132036

theorem subset_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {-1, 1}
  let B : Set ℝ := {x | a * x + 2 = 0}
  B ⊆ A → a ∈ ({-2, 0, 2} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l1320_132036


namespace NUMINAMATH_CALUDE_remainder_problem_l1320_132067

theorem remainder_problem (m n : ℕ) (h1 : m > n) (h2 : n % 6 = 3) (h3 : (m - n) % 6 = 5) :
  m % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1320_132067


namespace NUMINAMATH_CALUDE_nacl_formation_l1320_132007

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String
  coef_r1 : Nat
  coef_r2 : Nat
  coef_p1 : Nat
  coef_p2 : Nat

/-- Calculates the moles of product formed based on limiting reactant -/
def molesFormed (reaction : Reaction) (moles_r1 : Rat) (moles_r2 : Rat) : Rat :=
  min (moles_r1 * reaction.coef_p1 / reaction.coef_r1) (moles_r2 * reaction.coef_p1 / reaction.coef_r2)

/-- The main theorem to prove -/
theorem nacl_formation 
  (reaction1 : Reaction)
  (reaction2 : Reaction)
  (moles_naoh : Rat)
  (moles_cl2 : Rat)
  (moles_hcl : Rat)
  (h1 : reaction1 = {
    reactant1 := "NaOH", reactant2 := "Cl2", 
    product1 := "NaCl", product2 := "H2O",
    coef_r1 := 2, coef_r2 := 1, coef_p1 := 2, coef_p2 := 1
  })
  (h2 : reaction2 = {
    reactant1 := "NaOH", reactant2 := "HCl", 
    product1 := "NaCl", product2 := "H2O",
    coef_r1 := 1, coef_r2 := 1, coef_p1 := 1, coef_p2 := 1
  })
  (h3 : moles_naoh = 3)
  (h4 : moles_cl2 = 2)
  (h5 : moles_hcl = 4) :
  molesFormed reaction1 moles_naoh moles_cl2 + molesFormed reaction2 (moles_naoh - molesFormed reaction1 moles_naoh moles_cl2 * reaction1.coef_r1 / reaction1.coef_p1) moles_hcl = 7 :=
by sorry

end NUMINAMATH_CALUDE_nacl_formation_l1320_132007


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l1320_132026

/-- The number of ways to place 5 numbered balls into 5 numbered boxes, leaving one box empty -/
def ball_placement_count : ℕ := 1200

/-- The number of balls -/
def num_balls : ℕ := 5

/-- The number of boxes -/
def num_boxes : ℕ := 5

theorem ball_placement_theorem : 
  ball_placement_count = 1200 ∧ 
  num_balls = 5 ∧ 
  num_boxes = 5 := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l1320_132026


namespace NUMINAMATH_CALUDE_parabola_minimum_distance_l1320_132047

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with slope k passing through (2, 0) -/
def line (k x y : ℝ) : Prop := y = k*(x - 2)

/-- Distance between two x-coordinates on the parabola -/
def distance_on_parabola (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem parabola_minimum_distance (k1 k2 : ℝ) :
  k1 * k2 = -2 →
  ∃ (xA xC xB xD : ℝ),
    parabola xA (k1*(xA - 2)) ∧
    parabola xC (k1*(xC - 2)) ∧
    parabola xB (k2*(xB - 2)) ∧
    parabola xD (k2*(xD - 2)) ∧
    (∀ x1A x1C x2B x2D : ℝ,
      parabola x1A (k1*(x1A - 2)) →
      parabola x1C (k1*(x1C - 2)) →
      parabola x2B (k2*(x2B - 2)) →
      parabola x2D (k2*(x2D - 2)) →
      distance_on_parabola xA xC + distance_on_parabola xB xD ≤
      distance_on_parabola x1A x1C + distance_on_parabola x2B x2D) ∧
    distance_on_parabola xA xC + distance_on_parabola xB xD = 24 :=
by sorry


end NUMINAMATH_CALUDE_parabola_minimum_distance_l1320_132047


namespace NUMINAMATH_CALUDE_base8_536_equals_base7_1054_l1320_132001

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def base10_to_base7 (n : ℕ) : ℕ := sorry

/-- Theorem stating that 536 in base 8 is equal to 1054 in base 7 -/
theorem base8_536_equals_base7_1054 : 
  base10_to_base7 (base8_to_base10 536) = 1054 := by sorry

end NUMINAMATH_CALUDE_base8_536_equals_base7_1054_l1320_132001


namespace NUMINAMATH_CALUDE_candies_left_l1320_132000

def initial_candies : ℕ := 88
def candies_taken : ℕ := 6

theorem candies_left : initial_candies - candies_taken = 82 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_l1320_132000


namespace NUMINAMATH_CALUDE_marble_weight_l1320_132038

theorem marble_weight (marble_weight : ℚ) (car_weight : ℚ) : 
  9 * marble_weight = 5 * car_weight →
  4 * car_weight = 120 →
  marble_weight = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_l1320_132038


namespace NUMINAMATH_CALUDE_remaining_area_of_semicircle_l1320_132069

theorem remaining_area_of_semicircle (d : ℝ) (h : d > 0) :
  let r := d / 2
  let chord_length := 2 * Real.sqrt 7
  chord_length ^ 2 + r ^ 2 = d ^ 2 →
  (π * r ^ 2 / 2) - 2 * (π * (r / 2) ^ 2 / 2) = 7 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_area_of_semicircle_l1320_132069


namespace NUMINAMATH_CALUDE_remainder_theorem_l1320_132093

theorem remainder_theorem (n : ℤ) (h : n % 5 = 3) : (4 * n - 9) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1320_132093


namespace NUMINAMATH_CALUDE_cos_2a_over_1_plus_sin_2a_l1320_132072

theorem cos_2a_over_1_plus_sin_2a (a : ℝ) (h : 4 * Real.sin a = 3 * Real.cos a) :
  (Real.cos (2 * a)) / (1 + Real.sin (2 * a)) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cos_2a_over_1_plus_sin_2a_l1320_132072


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1320_132051

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 6}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1320_132051


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l1320_132063

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) +
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l1320_132063


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1320_132070

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 10) 
  (eq2 : x + 3 * y = 14) : 
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1320_132070


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l1320_132076

/-- The equation of a parabola -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- The slope of the tangent line at a given x-coordinate -/
def tangent_slope (x : ℝ) : ℝ := 8 * x

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (1, 4)

/-- The proposed equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_is_correct :
  let (x₀, y₀) := point_of_tangency
  tangent_line x₀ y₀ ∧
  y₀ = parabola x₀ ∧
  (∀ x y, tangent_line x y ↔ y - y₀ = tangent_slope x₀ * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l1320_132076


namespace NUMINAMATH_CALUDE_find_divisor_l1320_132057

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 176) (h2 : quotient = 9) (h3 : remainder = 5) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 19 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1320_132057


namespace NUMINAMATH_CALUDE_function_through_point_l1320_132020

/-- Given a function f(x) = x^α that passes through the point (2, √2), prove that f(9) = 3 -/
theorem function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
sorry

end NUMINAMATH_CALUDE_function_through_point_l1320_132020


namespace NUMINAMATH_CALUDE_sqrt_expression_equivalence_l1320_132086

theorem sqrt_expression_equivalence (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - x * (1 - 1 / (x + 1)))) = abs x :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equivalence_l1320_132086


namespace NUMINAMATH_CALUDE_P3_is_one_fourth_P4_is_three_fourths_l1320_132074

/-- The probability that the center of a circle is in the interior of the convex hull of n points
    selected independently with uniform distribution on the circle. -/
def P (n : ℕ) : ℝ := sorry

/-- Theorem: The probability P3 is 1/4 -/
theorem P3_is_one_fourth : P 3 = 1/4 := by sorry

/-- Theorem: The probability P4 is 3/4 -/
theorem P4_is_three_fourths : P 4 = 3/4 := by sorry

end NUMINAMATH_CALUDE_P3_is_one_fourth_P4_is_three_fourths_l1320_132074


namespace NUMINAMATH_CALUDE_jims_taxi_charge_l1320_132035

/-- Proves that the additional charge per 2/5 of a mile is $0.30 for Jim's taxi service -/
theorem jims_taxi_charge (initial_fee : ℚ) (total_charge : ℚ) (trip_distance : ℚ) :
  initial_fee = 2.25 →
  total_charge = 4.95 →
  trip_distance = 3.6 →
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.30 := by
sorry

end NUMINAMATH_CALUDE_jims_taxi_charge_l1320_132035


namespace NUMINAMATH_CALUDE_calculator_display_l1320_132049

/-- The special key function -/
def f (x : ℚ) : ℚ := 1 / (1 - x)

/-- Applies the function n times to the initial value -/
def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem calculator_display : iterate_f 120 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculator_display_l1320_132049


namespace NUMINAMATH_CALUDE_B_power_103_l1320_132042

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_103 : B^103 = B := by sorry

end NUMINAMATH_CALUDE_B_power_103_l1320_132042


namespace NUMINAMATH_CALUDE_paul_sold_63_books_l1320_132025

/-- The number of books Paul sold in a garage sale --/
def books_sold_in_garage_sale (initial_books donated_books exchanged_books given_to_friend remaining_books : ℕ) : ℕ :=
  initial_books - donated_books - given_to_friend - remaining_books

/-- Theorem stating that Paul sold 63 books in the garage sale --/
theorem paul_sold_63_books :
  books_sold_in_garage_sale 250 50 20 35 102 = 63 := by
  sorry

end NUMINAMATH_CALUDE_paul_sold_63_books_l1320_132025


namespace NUMINAMATH_CALUDE_maggies_earnings_l1320_132009

/-- Calculates the total earnings for Maggie's magazine subscription sales --/
theorem maggies_earnings 
  (family_commission : ℕ) 
  (neighbor_commission : ℕ)
  (bonus_threshold : ℕ)
  (bonus_base : ℕ)
  (bonus_per_extra : ℕ)
  (family_subscriptions : ℕ)
  (neighbor_subscriptions : ℕ)
  (h1 : family_commission = 7)
  (h2 : neighbor_commission = 6)
  (h3 : bonus_threshold = 10)
  (h4 : bonus_base = 10)
  (h5 : bonus_per_extra = 1)
  (h6 : family_subscriptions = 9)
  (h7 : neighbor_subscriptions = 6) :
  family_commission * family_subscriptions +
  neighbor_commission * neighbor_subscriptions +
  bonus_base +
  (if family_subscriptions + neighbor_subscriptions > bonus_threshold
   then (family_subscriptions + neighbor_subscriptions - bonus_threshold) * bonus_per_extra
   else 0) = 114 := by
  sorry


end NUMINAMATH_CALUDE_maggies_earnings_l1320_132009


namespace NUMINAMATH_CALUDE_ball_probabilities_l1320_132014

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 12

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing a yellow ball -/
def prob_yellow : ℚ := 0

/-- Represents the probability of drawing at least one red ball out of 9 balls drawn at once -/
def prob_at_least_one_red : ℚ := 1

/-- Represents the probability of drawing a red ball at random -/
def prob_red : ℚ := 3 / 5

/-- Represents the number of red balls removed and white balls added -/
def x : ℕ := 8

theorem ball_probabilities :
  (prob_yellow = 0) ∧
  (prob_at_least_one_red = 1) ∧
  (prob_red = 3 / 5) ∧
  (((initial_white_balls + x : ℚ) / total_balls) = 4 / 5 → x = 8) :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1320_132014


namespace NUMINAMATH_CALUDE_soccer_match_draw_probability_l1320_132017

theorem soccer_match_draw_probability 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.3) 
  (h2 : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_soccer_match_draw_probability_l1320_132017


namespace NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l1320_132071

theorem sqrt_fifth_power_sixth : (Real.sqrt ((Real.sqrt 5)^4))^6 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifth_power_sixth_l1320_132071


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l1320_132004

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + b

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a b : ℝ, f_derivative a 1 = -1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l1320_132004


namespace NUMINAMATH_CALUDE_fraction_equality_l1320_132077

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (b - a) / b = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1320_132077


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l1320_132030

/-- The length of a rectangular solid with given dimensions and surface area -/
theorem rectangular_solid_length
  (width : ℝ) (depth : ℝ) (surface_area : ℝ)
  (h_width : width = 9)
  (h_depth : depth = 6)
  (h_surface_area : surface_area = 408)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  length = 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l1320_132030


namespace NUMINAMATH_CALUDE_average_equals_black_dots_l1320_132041

/-- Represents the types of butterflies -/
inductive ButterflyType
  | A
  | B
  | C

/-- Returns the number of black dots for a given butterfly type -/
def blackDots (t : ButterflyType) : ℕ :=
  match t with
  | .A => 545
  | .B => 780
  | .C => 1135

/-- Returns the number of butterflies for a given type -/
def butterflyCount (t : ButterflyType) : ℕ :=
  match t with
  | .A => 15
  | .B => 25
  | .C => 35

/-- Calculates the average number of black dots per butterfly for a given type -/
def averageBlackDots (t : ButterflyType) : ℚ :=
  (blackDots t : ℚ) * (butterflyCount t : ℚ) / (butterflyCount t : ℚ)

theorem average_equals_black_dots (t : ButterflyType) :
  averageBlackDots t = blackDots t := by
  sorry

#eval averageBlackDots ButterflyType.A
#eval averageBlackDots ButterflyType.B
#eval averageBlackDots ButterflyType.C

end NUMINAMATH_CALUDE_average_equals_black_dots_l1320_132041


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_is_18_l1320_132005

/-- Pentagon with vertices A, B, C, D, E -/
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- Calculate the area of a pentagon -/
def pentagonArea (p : Pentagon) : ℝ := sorry

/-- Check if a pentagon has a vertical line of symmetry -/
def hasVerticalSymmetry (p : Pentagon) : Prop := sorry

/-- Theorem: The y-coordinate of vertex C in the given pentagon is 18 -/
theorem y_coordinate_of_C_is_18 (p : Pentagon) 
  (h1 : p.A = (0, 0))
  (h2 : p.B = (0, 6))
  (h3 : p.D = (6, 6))
  (h4 : p.E = (6, 0))
  (h5 : hasVerticalSymmetry p)
  (h6 : pentagonArea p = 72)
  : p.C.2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_is_18_l1320_132005


namespace NUMINAMATH_CALUDE_negative_decimal_greater_than_negative_fraction_l1320_132023

theorem negative_decimal_greater_than_negative_fraction : -0.6 > -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_decimal_greater_than_negative_fraction_l1320_132023


namespace NUMINAMATH_CALUDE_two_integers_sum_l1320_132027

theorem two_integers_sum (a b : ℕ+) : a - b = 4 → a * b = 96 → a + b = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l1320_132027


namespace NUMINAMATH_CALUDE_indefinite_integral_arctg_sqrt_2x_minus_1_l1320_132028

theorem indefinite_integral_arctg_sqrt_2x_minus_1 (x : ℝ) :
  HasDerivAt (fun x => x * Real.arctan (Real.sqrt (2 * x - 1)) - (1/2) * Real.sqrt (2 * x - 1))
             (Real.arctan (Real.sqrt (2 * x - 1)))
             x :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_arctg_sqrt_2x_minus_1_l1320_132028


namespace NUMINAMATH_CALUDE_largest_number_problem_l1320_132092

theorem largest_number_problem (a b c : ℝ) : 
  a < b → b < c →
  a + b + c = 100 →
  c - b = 8 →
  b - a = 4 →
  c = 40 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l1320_132092


namespace NUMINAMATH_CALUDE_jill_earnings_l1320_132053

def first_month_daily_wage : ℕ := 10
def days_per_month : ℕ := 30

def second_month_daily_wage : ℕ := 2 * first_month_daily_wage
def third_month_working_days : ℕ := days_per_month / 2

def first_month_earnings : ℕ := first_month_daily_wage * days_per_month
def second_month_earnings : ℕ := second_month_daily_wage * days_per_month
def third_month_earnings : ℕ := second_month_daily_wage * third_month_working_days

def total_earnings : ℕ := first_month_earnings + second_month_earnings + third_month_earnings

theorem jill_earnings : total_earnings = 1200 := by
  sorry

end NUMINAMATH_CALUDE_jill_earnings_l1320_132053


namespace NUMINAMATH_CALUDE_range_of_a_l1320_132098

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - 2*a - 5) > 0}
def B (a : ℝ) : Set ℝ := {x | (a^2 + 2 - x) * (2*a - x) < 0}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) → 
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) → 
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1320_132098


namespace NUMINAMATH_CALUDE_fraction_undefined_values_l1320_132060

def undefined_values (b : ℝ) : Prop :=
  b^2 - 9 = 0

theorem fraction_undefined_values :
  {b : ℝ | undefined_values b} = {-3, 3} := by
  sorry

end NUMINAMATH_CALUDE_fraction_undefined_values_l1320_132060


namespace NUMINAMATH_CALUDE_only_vegetarian_count_l1320_132043

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  total_veg : ℕ
  only_nonveg : ℕ
  both : ℕ

/-- Given the specified family diet, prove that the number of people who eat only vegetarian is 13 -/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.total_veg = 21)
  (h2 : f.only_nonveg = 7)
  (h3 : f.both = 8) :
  f.total_veg - f.both = 13 := by
  sorry

#check only_vegetarian_count

end NUMINAMATH_CALUDE_only_vegetarian_count_l1320_132043


namespace NUMINAMATH_CALUDE_xy_sum_difference_l1320_132044

theorem xy_sum_difference (x y : ℝ) 
  (h1 : x + Real.sqrt (x * y) + y = 9) 
  (h2 : x^2 + x*y + y^2 = 27) : 
  x - Real.sqrt (x * y) + y = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_difference_l1320_132044


namespace NUMINAMATH_CALUDE_angle_greater_than_120_degrees_l1320_132039

open Real Set

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- The theorem statement -/
theorem angle_greater_than_120_degrees (n : ℕ) (points : Finset Point) :
  points.card = n →
  ∃ (ordered_points : Fin n → Point),
    (∀ i : Fin n, ordered_points i ∈ points) ∧
    (∀ (i j k : Fin n), i < j → j < k →
      angle (ordered_points i) (ordered_points j) (ordered_points k) > 120 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_angle_greater_than_120_degrees_l1320_132039


namespace NUMINAMATH_CALUDE_correct_fraction_proof_l1320_132055

theorem correct_fraction_proof (x y : ℚ) : 
  (5 : ℚ) / 6 * 384 = x / y * 384 + 200 → x / y = (5 : ℚ) / 16 := by
sorry

end NUMINAMATH_CALUDE_correct_fraction_proof_l1320_132055


namespace NUMINAMATH_CALUDE_duck_park_population_l1320_132068

theorem duck_park_population (initial_ducks : ℕ) (arriving_ducks : ℕ) (leaving_geese : ℕ) : 
  initial_ducks = 25 →
  arriving_ducks = 4 →
  leaving_geese = 10 →
  (initial_ducks * 2 - 10) - leaving_geese - (initial_ducks + arriving_ducks) = 1 :=
by sorry

end NUMINAMATH_CALUDE_duck_park_population_l1320_132068


namespace NUMINAMATH_CALUDE_range_of_m_l1320_132046

-- Define p and q as functions of x and m
def p (x : ℝ) : Prop := |x - 3| ≤ 2

def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : Prop) : Prop :=
  (q → p) ∧ ¬(p → q)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, necessary_not_sufficient (¬(p x)) (¬(q x m))) →
  (m ≥ 2 ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1320_132046


namespace NUMINAMATH_CALUDE_equation_solutions_l1320_132078

theorem equation_solutions :
  (∃ x : ℚ, (5 / (x + 1) = 1 / (x - 3)) ∧ x = 4) ∧
  (∃ x : ℚ, ((2 - x) / (x - 3) + 2 = 1 / (3 - x)) ∧ x = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1320_132078


namespace NUMINAMATH_CALUDE_tank_capacity_l1320_132065

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 7/8 →
  (initial_fraction * C + added_amount = final_fraction * C) →
  C = 72 :=
by sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1320_132065


namespace NUMINAMATH_CALUDE_solve_for_q_l1320_132011

theorem solve_for_q (k l q : ℚ) 
  (eq1 : 3/4 = k/48)
  (eq2 : 3/4 = (k + l)/56)
  (eq3 : 3/4 = (q - l)/160) : 
  q = 126 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l1320_132011


namespace NUMINAMATH_CALUDE_convex_quad_polyhedron_16v_14f_l1320_132021

/-- A convex polyhedron with quadrilateral faces -/
structure ConvexQuadPolyhedron where
  vertices : ℕ
  faces : ℕ
  edges : ℕ
  convex : Bool
  all_faces_quadrilateral : Bool
  euler : vertices + faces - edges = 2
  quad_face_edge_relation : edges = 2 * faces

/-- Theorem: A convex polyhedron with 16 vertices and all quadrilateral faces has 14 faces -/
theorem convex_quad_polyhedron_16v_14f :
  ∀ (P : ConvexQuadPolyhedron), 
    P.vertices = 16 ∧ P.convex ∧ P.all_faces_quadrilateral → P.faces = 14 :=
by sorry

end NUMINAMATH_CALUDE_convex_quad_polyhedron_16v_14f_l1320_132021


namespace NUMINAMATH_CALUDE_triangular_array_digit_sum_l1320_132058

def triangular_array_sum (n : ℕ) : ℕ := 3 * n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem triangular_array_digit_sum :
  ∃ n : ℕ, triangular_array_sum n = 1575 ∧ sum_of_digits n = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_digit_sum_l1320_132058


namespace NUMINAMATH_CALUDE_plane_parallel_transitivity_l1320_132018

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_parallel_transitivity (α β γ : Plane) :
  parallel γ α → parallel γ β → parallel α β := by sorry

end NUMINAMATH_CALUDE_plane_parallel_transitivity_l1320_132018


namespace NUMINAMATH_CALUDE_reflected_line_x_intercept_l1320_132010

/-- The x-intercept of a line reflected in the y-axis -/
theorem reflected_line_x_intercept :
  let original_line : ℝ → ℝ := λ x => 2 * x - 6
  let reflected_line : ℝ → ℝ := λ x => -2 * x - 6
  let x_intercept : ℝ := -3
  (reflected_line x_intercept = 0) ∧ 
  (∀ y : ℝ, reflected_line y = 0 → y = x_intercept) :=
by sorry

end NUMINAMATH_CALUDE_reflected_line_x_intercept_l1320_132010


namespace NUMINAMATH_CALUDE_odd_function_product_nonpositive_l1320_132040

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_nonpositive (f : ℝ → ℝ) (h : odd_function f) :
  ∀ x : ℝ, f x * f (-x) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_product_nonpositive_l1320_132040


namespace NUMINAMATH_CALUDE_log_2_base_10_upper_bound_l1320_132034

theorem log_2_base_10_upper_bound (h1 : 10^3 = 1000) (h2 : 10^4 = 10000)
  (h3 : 2^9 = 512) (h4 : 2^11 = 2048) : Real.log 2 / Real.log 10 < 4/11 := by
  sorry

end NUMINAMATH_CALUDE_log_2_base_10_upper_bound_l1320_132034


namespace NUMINAMATH_CALUDE_alfonso_helmet_weeks_l1320_132003

/-- Calculates the number of weeks Alfonso needs to work to buy a helmet -/
def weeks_to_buy_helmet (daily_earnings : ℚ) (days_per_week : ℕ) (helmet_cost : ℚ) (savings : ℚ) : ℚ :=
  (helmet_cost - savings) / (daily_earnings * days_per_week)

/-- Proves that Alfonso needs 10 weeks to buy the helmet -/
theorem alfonso_helmet_weeks : 
  let daily_earnings : ℚ := 6
  let days_per_week : ℕ := 5
  let helmet_cost : ℚ := 340
  let savings : ℚ := 40
  weeks_to_buy_helmet daily_earnings days_per_week helmet_cost savings = 10 := by
sorry

#eval weeks_to_buy_helmet 6 5 340 40

end NUMINAMATH_CALUDE_alfonso_helmet_weeks_l1320_132003


namespace NUMINAMATH_CALUDE_train_crossing_time_l1320_132037

/-- Proves that a train with given length and speed takes a specific time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 900 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1320_132037


namespace NUMINAMATH_CALUDE_solution_is_three_l1320_132008

/-- A linear function passing through (-2, 0) with y-intercept 3 -/
structure LinearFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_through : k * (-2) + 3 = 0

/-- The solution to k(x-5)+3=0 is x=3 -/
theorem solution_is_three (f : LinearFunction) : 
  ∃ x : ℝ, f.k * (x - 5) + 3 = 0 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_three_l1320_132008


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l1320_132097

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_squared_l1320_132097


namespace NUMINAMATH_CALUDE_annes_height_l1320_132087

/-- Proves that Anne's height is 80 cm given the relationships between heights of Anne, her sister, and Bella -/
theorem annes_height (sister_height : ℝ) (anne_height : ℝ) (bella_height : ℝ) : 
  anne_height = 2 * sister_height →
  bella_height = 3 * anne_height →
  bella_height - sister_height = 200 →
  anne_height = 80 := by
sorry

end NUMINAMATH_CALUDE_annes_height_l1320_132087


namespace NUMINAMATH_CALUDE_other_ticket_price_l1320_132059

/-- Represents the ticket sales scenario for the Red Rose Theatre --/
def theatre_sales (other_price : ℝ) : Prop :=
  let total_tickets : ℕ := 380
  let cheap_tickets : ℕ := 205
  let cheap_price : ℝ := 4.50
  let total_revenue : ℝ := 1972.50
  (cheap_tickets : ℝ) * cheap_price + (total_tickets - cheap_tickets : ℝ) * other_price = total_revenue

/-- Theorem stating that the price of the other tickets is $6.00 --/
theorem other_ticket_price : ∃ (price : ℝ), theatre_sales price ∧ price = 6 := by
  sorry

end NUMINAMATH_CALUDE_other_ticket_price_l1320_132059


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1320_132002

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 9 * X^4 + 8 * X^3 - 12 * X^2 - 7 * X + 4
  let divisor : Polynomial ℚ := 3 * X^2 + 2 * X + 5
  let quotient : Polynomial ℚ := 3 * X^2 - 2 * X + 2
  (dividend.div divisor) = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1320_132002


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l1320_132054

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l1320_132054


namespace NUMINAMATH_CALUDE_correct_schedule_count_l1320_132029

/-- Represents a club with members and scheduling constraints -/
structure Club where
  totalMembers : Nat
  daysToSchedule : Nat
  membersPerDay : Nat

/-- Represents the scheduling constraints for specific members -/
structure SchedulingConstraints where
  mustBeTogetherPair : Fin 2 → Nat
  cannotBeTogether : Fin 2 → Nat

/-- Calculates the total number of possible schedules given the club and constraints -/
def totalPossibleSchedules (club : Club) (constraints : SchedulingConstraints) : Nat :=
  sorry

/-- The main theorem stating the correct number of schedules -/
theorem correct_schedule_count :
  let club := Club.mk 10 5 2
  let constraints := SchedulingConstraints.mk
    (fun i => if i.val = 0 then 0 else 1)  -- A and B (represented as 0 and 1)
    (fun i => if i.val = 0 then 2 else 3)  -- C and D (represented as 2 and 3)
  totalPossibleSchedules club constraints = 5400 := by sorry

end NUMINAMATH_CALUDE_correct_schedule_count_l1320_132029


namespace NUMINAMATH_CALUDE_inequality_proof_l1320_132031

theorem inequality_proof (a b : ℝ) (h1 : a < 0) (h2 : b > 0) (h3 : a + b < 0) :
  -a > b ∧ b > -b ∧ -b > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1320_132031


namespace NUMINAMATH_CALUDE_triangle_properties_l1320_132085

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def condition (t : Triangle) : Prop :=
  (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b

theorem triangle_properties (t : Triangle) (h : condition t) :
  Real.sin t.C / Real.sin t.A = 2 ∧
  (Real.cos t.B = 1/4 ∧ t.b = 2 → 
    1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 15 / 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1320_132085


namespace NUMINAMATH_CALUDE_card_area_theorem_l1320_132052

/-- Represents the dimensions of a rectangular card -/
structure Card where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card -/
def area (c : Card) : ℝ := c.length * c.width

/-- Theorem: Given a 5x7 card, if shortening one side by 2 inches results in 
    an area of 21 square inches, then shortening the other side by 2 inches 
    will result in an area of 25 square inches -/
theorem card_area_theorem (c : Card) 
    (h1 : c.length = 5 ∧ c.width = 7)
    (h2 : ∃ (shortened_card : Card), 
      (shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
      (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2))
    (h3 : ∃ (shortened_card : Card), 
      area shortened_card = 21 ∧
      ((shortened_card.length = c.length - 2 ∧ shortened_card.width = c.width) ∨
       (shortened_card.length = c.length ∧ shortened_card.width = c.width - 2)))
    : ∃ (other_shortened_card : Card), 
      area other_shortened_card = 25 ∧
      ((other_shortened_card.length = c.length - 2 ∧ other_shortened_card.width = c.width) ∨
       (other_shortened_card.length = c.length ∧ other_shortened_card.width = c.width - 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_card_area_theorem_l1320_132052


namespace NUMINAMATH_CALUDE_sally_pens_home_l1320_132084

/-- The number of pens Sally takes home given the initial conditions -/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_remaining := total_pens - pens_given
  pens_remaining / 2

theorem sally_pens_home :
  pens_taken_home 342 44 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sally_pens_home_l1320_132084


namespace NUMINAMATH_CALUDE_divisibility_relations_l1320_132075

theorem divisibility_relations (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1) :
  (¬ ((a ∣ b^2) ↔ (a ∣ b))) ∧ ((a^2 ∣ b^2) ↔ (a ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_relations_l1320_132075


namespace NUMINAMATH_CALUDE_total_cost_is_24_l1320_132096

/-- The cost of one gold ring in dollars -/
def ring_cost : ℕ := 12

/-- The number of index fingers a person has -/
def index_fingers : ℕ := 2

/-- The total cost of buying gold rings for all index fingers -/
def total_cost : ℕ := ring_cost * index_fingers

/-- Theorem: The total cost of buying gold rings for all index fingers is 24 dollars -/
theorem total_cost_is_24 : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l1320_132096


namespace NUMINAMATH_CALUDE_johns_star_wars_toys_cost_l1320_132061

theorem johns_star_wars_toys_cost (lightsaber_cost other_toys_cost total_spent : ℕ) : 
  lightsaber_cost = 2 * other_toys_cost →
  total_spent = lightsaber_cost + other_toys_cost →
  total_spent = 3000 →
  other_toys_cost = 1000 := by
sorry

end NUMINAMATH_CALUDE_johns_star_wars_toys_cost_l1320_132061
