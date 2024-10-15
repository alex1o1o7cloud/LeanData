import Mathlib

namespace NUMINAMATH_CALUDE_vector_subtraction_l147_14787

/-- Given two vectors a and b in ℝ², prove that their difference is (1, 2). -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (2, 3)) (hb : b = (1, 1)) :
  a - b = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l147_14787


namespace NUMINAMATH_CALUDE_polynomial_factorization_l147_14722

theorem polynomial_factorization (m : ℝ) : 
  (∀ x, x^2 + m*x - 6 = (x - 2) * (x + 3)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l147_14722


namespace NUMINAMATH_CALUDE_c_rent_share_is_72_l147_14754

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℚ) (oxen_a : ℕ) (months_a : ℕ) (oxen_b : ℕ) (months_b : ℕ) (oxen_c : ℕ) (months_c : ℕ) : ℚ :=
  let total_oxen_months := oxen_a * months_a + oxen_b * months_b + oxen_c * months_c
  let c_oxen_months := oxen_c * months_c
  (c_oxen_months : ℚ) / total_oxen_months * total_rent

/-- Theorem stating that C's share of the rent is approximately 72 -/
theorem c_rent_share_is_72 :
  let rent_share := RentShare 280 10 7 12 5 15 3
  ∃ ε > 0, abs (rent_share - 72) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_c_rent_share_is_72_l147_14754


namespace NUMINAMATH_CALUDE_unique_integer_perfect_square_l147_14788

theorem unique_integer_perfect_square : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_perfect_square_l147_14788


namespace NUMINAMATH_CALUDE_jessica_cut_forty_roses_l147_14768

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_vase : ℕ) (final_vase : ℕ) (returned_to_sarah : ℕ) (total_garden : ℕ) : ℕ :=
  (final_vase - initial_vase) + returned_to_sarah

/-- Theorem stating that Jessica cut 40 roses from her garden -/
theorem jessica_cut_forty_roses :
  roses_cut 7 37 10 84 = 40 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_forty_roses_l147_14768


namespace NUMINAMATH_CALUDE_middle_number_problem_l147_14712

theorem middle_number_problem (x y z : ℤ) 
  (sum_xy : x + y = 15)
  (sum_xz : x + z = 18)
  (sum_yz : y + z = 22) :
  y = (19 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l147_14712


namespace NUMINAMATH_CALUDE_log_sum_of_zeros_gt_two_l147_14792

open Real

/-- Given a function g(x) = ln x - bx, if it has two distinct positive zeros,
    then the sum of their natural logarithms is greater than 2. -/
theorem log_sum_of_zeros_gt_two (b : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂)
  (hz₁ : log x₁ - b * x₁ = 0) (hz₂ : log x₂ - b * x₂ = 0) :
  log x₁ + log x₂ > 2 := by
sorry


end NUMINAMATH_CALUDE_log_sum_of_zeros_gt_two_l147_14792


namespace NUMINAMATH_CALUDE_potato_ratio_l147_14706

def potato_distribution (initial : ℕ) (gina : ℕ) (remaining : ℕ) : Prop :=
  ∃ (tom anne : ℕ),
    tom = 2 * gina ∧
    initial = gina + tom + anne + remaining ∧
    anne * 3 = tom

theorem potato_ratio (initial : ℕ) (gina : ℕ) (remaining : ℕ) 
  (h : potato_distribution initial gina remaining) :
  potato_distribution 300 69 47 :=
by sorry

end NUMINAMATH_CALUDE_potato_ratio_l147_14706


namespace NUMINAMATH_CALUDE_quadratic_max_value_l147_14752

def f (x : ℝ) : ℝ := -x^2 - 3*x + 4

theorem quadratic_max_value :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l147_14752


namespace NUMINAMATH_CALUDE_inverse_B_cubed_l147_14728

theorem inverse_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![3, 7; -2, -4]) : 
  (B⁻¹)^3 = !![11, 17; -10, -18] := by
  sorry

end NUMINAMATH_CALUDE_inverse_B_cubed_l147_14728


namespace NUMINAMATH_CALUDE_system_solution_l147_14741

theorem system_solution (x y : ℝ) : 
  ((x = 0 ∧ y = 0) ∨ 
   (x = 1 ∧ y = 1) ∨ 
   (x = -(5/4)^(1/5) ∧ y = (-50)^(1/5))) → 
  (4 * x^2 - 3 * y = x * y^3 ∧ 
   x^2 + x^3 * y^2 = 2 * y) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l147_14741


namespace NUMINAMATH_CALUDE_probability_red_or_white_l147_14746

def total_marbles : ℕ := 20
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white :
  (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l147_14746


namespace NUMINAMATH_CALUDE_hexagonal_grid_path_theorem_l147_14775

/-- Represents a point in the hexagonal grid -/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- Represents a direction in the hexagonal grid -/
inductive HexDirection
  | Right
  | UpRight
  | UpLeft
  | Left
  | DownLeft
  | DownRight

/-- Represents a path in the hexagonal grid -/
def HexPath := List (HexPoint × HexDirection)

/-- Function to calculate the length of a path -/
def pathLength (path : HexPath) : ℕ := path.length

/-- Function to check if a path is valid in the hexagonal grid -/
def isValidPath (path : HexPath) : Prop := sorry

/-- Function to find the longest continuous segment in the same direction -/
def longestContinuousSegment (path : HexPath) : ℕ := sorry

/-- Theorem: In a hexagonal grid, if the shortest path between two points is 20 units,
    then there exists a continuous segment of at least 10 units in the same direction -/
theorem hexagonal_grid_path_theorem (A B : HexPoint) (path : HexPath) :
  isValidPath path →
  pathLength path = 20 →
  (∀ p : HexPath, isValidPath p → pathLength p ≥ 20) →
  longestContinuousSegment path ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_grid_path_theorem_l147_14775


namespace NUMINAMATH_CALUDE_ariel_fencing_years_l147_14738

theorem ariel_fencing_years (birth_year : ℕ) (fencing_start : ℕ) (current_age : ℕ) : 
  birth_year = 1992 → fencing_start = 2006 → current_age = 30 → 
  fencing_start - birth_year - current_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_ariel_fencing_years_l147_14738


namespace NUMINAMATH_CALUDE_meeting_participants_l147_14794

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 130 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 780 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l147_14794


namespace NUMINAMATH_CALUDE_square_area_diagonal_relation_l147_14793

theorem square_area_diagonal_relation (d : ℝ) (h : d > 0) :
  ∃ (A : ℝ), A > 0 ∧ A = (1/2) * d^2 ∧ 
  (∃ (s : ℝ), s > 0 ∧ A = s^2 ∧ d^2 = 2 * s^2) := by
  sorry

end NUMINAMATH_CALUDE_square_area_diagonal_relation_l147_14793


namespace NUMINAMATH_CALUDE_stone_slab_length_l147_14779

theorem stone_slab_length (total_area : Real) (num_slabs : Nat) (slab_length : Real) : 
  total_area = 58.8 ∧ 
  num_slabs = 30 ∧ 
  slab_length * slab_length * num_slabs = total_area * 10000 →
  slab_length = 140 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_length_l147_14779


namespace NUMINAMATH_CALUDE_train_length_l147_14743

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 72 → time = 25 → platform_length = 250.04 → 
  speed * (5/18) * time - platform_length = 249.96 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l147_14743


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l147_14783

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, 2 * b * x^2 + 16 * x + 5 = 0) :
  ∃ x, 2 * b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l147_14783


namespace NUMINAMATH_CALUDE_vacuum_tube_alignment_l147_14702

theorem vacuum_tube_alignment (f : Fin 7 → Fin 7) (h : Function.Bijective f) :
  ∃ x : Fin 7, f x = x := by
  sorry

end NUMINAMATH_CALUDE_vacuum_tube_alignment_l147_14702


namespace NUMINAMATH_CALUDE_smallest_value_fraction_achievable_value_l147_14709

theorem smallest_value_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) ≥ 10/3 :=
sorry

theorem achievable_value (a b : ℕ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (a b : ℕ), a > b ∧ b > 0 ∧ (a + 2*b : ℚ)/(a - b) + (a - b : ℚ)/(a + 2*b) = 10/3 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_fraction_achievable_value_l147_14709


namespace NUMINAMATH_CALUDE_part_one_part_two_l147_14716

-- Part 1
theorem part_one (f h : ℝ → ℝ) (m : ℝ) :
  (∀ x > 1, f x = x^2 - m * Real.log x) →
  (∀ x > 1, h x = x^2 - x) →
  (∀ x > 1, f x ≥ h x) →
  m ≤ Real.exp 1 :=
sorry

-- Part 2
theorem part_two (f h k : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^2 - 2 * Real.log x) →
  (∀ x, h x = x^2 - x + a) →
  (∀ x, k x = f x - h x) →
  (∃ x y, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x < y ∧ k x = 0 ∧ k y = 0 ∧ 
    ∀ z ∈ Set.Icc 1 3, k z = 0 → (z = x ∨ z = y)) →
  2 - 2 * Real.log 2 < a ∧ a ≤ 3 - 2 * Real.log 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l147_14716


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l147_14795

open Real

theorem function_inequality_implies_m_bound (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 ℯ ∧ m * (x₀ - 1 / x₀) - 2 * log x₀ < -m / x₀) →
  m < 2 / ℯ := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l147_14795


namespace NUMINAMATH_CALUDE_daily_harvest_l147_14720

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l147_14720


namespace NUMINAMATH_CALUDE_round_robin_tournament_l147_14784

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l147_14784


namespace NUMINAMATH_CALUDE_certain_number_is_900_l147_14759

theorem certain_number_is_900 :
  ∃ x : ℝ, (45 * 9 = 0.45 * x) ∧ (x = 900) :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_900_l147_14759


namespace NUMINAMATH_CALUDE_existence_of_equal_segments_l147_14715

/-- An acute-angled triangle is a triangle where all angles are less than 90 degrees -/
def AcuteTriangle (A B C : Point) : Prop := sorry

/-- Point X is on line segment AB -/
def OnSegment (X A B : Point) : Prop := sorry

/-- AX = XY = YC -/
def EqualSegments (A X Y C : Point) : Prop := sorry

/-- Theorem: In any acute-angled triangle, there exist points X and Y on its sides
    such that AX = XY = YC -/
theorem existence_of_equal_segments (A B C : Point) 
  (h : AcuteTriangle A B C) : 
  ∃ X Y, OnSegment X A B ∧ OnSegment Y B C ∧ EqualSegments A X Y C := by
  sorry

end NUMINAMATH_CALUDE_existence_of_equal_segments_l147_14715


namespace NUMINAMATH_CALUDE_cube_from_wire_l147_14733

/-- Given a wire of length 60 cm formed into a cube frame, prove that the volume is 125 cm³ and the surface area is 150 cm². -/
theorem cube_from_wire (wire_length : ℝ) (h_wire : wire_length = 60) :
  let edge_length : ℝ := wire_length / 12
  let volume : ℝ := edge_length ^ 3
  let surface_area : ℝ := 6 * edge_length ^ 2
  volume = 125 ∧ surface_area = 150 := by sorry

end NUMINAMATH_CALUDE_cube_from_wire_l147_14733


namespace NUMINAMATH_CALUDE_john_haircut_tip_percentage_l147_14730

/-- Represents the growth rate of John's hair in inches per month -/
def hair_growth_rate : ℝ := 1.5

/-- Represents the length of John's hair in inches when he gets a haircut -/
def hair_length_at_cut : ℝ := 9

/-- Represents the length of John's hair in inches after a haircut -/
def hair_length_after_cut : ℝ := 6

/-- Represents the cost of a single haircut in dollars -/
def haircut_cost : ℝ := 45

/-- Represents the total amount John spends on haircuts in a year in dollars -/
def annual_haircut_spend : ℝ := 324

/-- Theorem stating that the percentage of the tip John gives for a haircut is 20% -/
theorem john_haircut_tip_percentage :
  let hair_growth_between_cuts := hair_length_at_cut - hair_length_after_cut
  let months_between_cuts := hair_growth_between_cuts / hair_growth_rate
  let haircuts_per_year := 12 / months_between_cuts
  let total_cost_per_haircut := annual_haircut_spend / haircuts_per_year
  let tip_amount := total_cost_per_haircut - haircut_cost
  let tip_percentage := (tip_amount / haircut_cost) * 100
  tip_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_john_haircut_tip_percentage_l147_14730


namespace NUMINAMATH_CALUDE_bug_total_distance_l147_14750

def bug_path : List ℤ := [-3, 0, -8, 10]

def total_distance (path : List ℤ) : ℕ :=
  (path.zip (path.tail!)).foldl (fun acc (a, b) => acc + (a - b).natAbs) 0

theorem bug_total_distance :
  total_distance bug_path = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l147_14750


namespace NUMINAMATH_CALUDE_triangle_area_l147_14719

/-- The area of a triangle with side lengths 7, 7, and 5 is 2.5√42.75 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 7) (h2 : b = 7) (h3 : c = 5) :
  (1/2 : ℝ) * c * Real.sqrt ((a^2 - (c/2)^2) : ℝ) = 2.5 * Real.sqrt 42.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l147_14719


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l147_14778

theorem complex_modulus_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l147_14778


namespace NUMINAMATH_CALUDE_stating_remaining_pieces_l147_14707

/-- The number of pieces on a standard chessboard at the start of the game. -/
def initial_pieces : ℕ := 32

/-- The number of pieces Audrey lost. -/
def audrey_lost : ℕ := 6

/-- The number of pieces Thomas lost. -/
def thomas_lost : ℕ := 5

/-- The total number of pieces lost by both players. -/
def total_lost : ℕ := audrey_lost + thomas_lost

/-- 
  Theorem stating that the number of pieces remaining on the chessboard is 21,
  given the initial number of pieces and the number of pieces lost by each player.
-/
theorem remaining_pieces :
  initial_pieces - total_lost = 21 := by sorry

end NUMINAMATH_CALUDE_stating_remaining_pieces_l147_14707


namespace NUMINAMATH_CALUDE_chord_length_l147_14737

theorem chord_length (r : ℝ) (h : r = 15) : 
  ∃ (c : ℝ), c = 26 * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) := by
sorry

end NUMINAMATH_CALUDE_chord_length_l147_14737


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l147_14742

theorem shopkeeper_profit (total_apples : ℝ) (profit_rate1 profit_rate2 : ℝ) 
  (portion1 portion2 : ℝ) :
  total_apples = 280 ∧ 
  profit_rate1 = 0.1 ∧ 
  profit_rate2 = 0.3 ∧ 
  portion1 = 0.4 ∧ 
  portion2 = 0.6 ∧ 
  portion1 + portion2 = 1 →
  let selling_price1 := portion1 * total_apples * (1 + profit_rate1)
  let selling_price2 := portion2 * total_apples * (1 + profit_rate2)
  let total_selling_price := selling_price1 + selling_price2
  let total_profit := total_selling_price - total_apples
  let percentage_profit := (total_profit / total_apples) * 100
  percentage_profit = 22 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l147_14742


namespace NUMINAMATH_CALUDE_roots_are_eccentricities_l147_14723

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 4 * x + 1 = 0

def is_ellipse_eccentricity (e : ℝ) : Prop := 0 < e ∧ e < 1

def is_parabola_eccentricity (e : ℝ) : Prop := e = 1

theorem roots_are_eccentricities :
  ∃ (e₁ e₂ : ℝ),
    quadratic_equation e₁ ∧
    quadratic_equation e₂ ∧
    e₁ ≠ e₂ ∧
    ((is_ellipse_eccentricity e₁ ∧ is_parabola_eccentricity e₂) ∨
     (is_ellipse_eccentricity e₂ ∧ is_parabola_eccentricity e₁)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_eccentricities_l147_14723


namespace NUMINAMATH_CALUDE_pencil_price_theorem_l147_14727

/-- Calculates the final price of a pencil after applying discounts and taxes -/
def final_price (initial_cost christmas_discount seasonal_discount final_discount tax_rate : ℚ) : ℚ :=
  let price_after_christmas := initial_cost - christmas_discount
  let price_after_seasonal := price_after_christmas * (1 - seasonal_discount)
  let price_after_final := price_after_seasonal * (1 - final_discount)
  price_after_final * (1 + tax_rate)

/-- The final price of the pencil is approximately $3.17 -/
theorem pencil_price_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_price 4 0.63 0.07 0.05 0.065 - 3.17| < ε :=
sorry

end NUMINAMATH_CALUDE_pencil_price_theorem_l147_14727


namespace NUMINAMATH_CALUDE_vector_points_to_line_and_parallel_l147_14731

/-- The line is parameterized by x = 3t + 1, y = t + 1 -/
def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, t + 1)

/-- The direction vector -/
def direction : ℝ × ℝ := (3, 1)

/-- The vector we want to prove -/
def vector : ℝ × ℝ := (9, 3)

theorem vector_points_to_line_and_parallel :
  (∃ t : ℝ, line_param t = vector) ∧ 
  (∃ k : ℝ, vector = (k * direction.1, k * direction.2)) :=
sorry

end NUMINAMATH_CALUDE_vector_points_to_line_and_parallel_l147_14731


namespace NUMINAMATH_CALUDE_unique_real_solution_l147_14789

theorem unique_real_solution (a : ℝ) :
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) ↔ a < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_solution_l147_14789


namespace NUMINAMATH_CALUDE_race_heartbeats_l147_14772

/-- The number of heartbeats during a race, given the race distance, heart rate, and pace. -/
def heartbeats_during_race (distance : ℕ) (heart_rate : ℕ) (pace : ℕ) : ℕ :=
  distance * pace * heart_rate

/-- Theorem stating that the number of heartbeats during a 30-mile race
    with a heart rate of 160 beats per minute and a pace of 6 minutes per mile
    is equal to 28800. -/
theorem race_heartbeats :
  heartbeats_during_race 30 160 6 = 28800 := by
  sorry

#eval heartbeats_during_race 30 160 6

end NUMINAMATH_CALUDE_race_heartbeats_l147_14772


namespace NUMINAMATH_CALUDE_max_value_of_f_l147_14799

theorem max_value_of_f (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  (∃ (a b : ℝ), f 0 + f 2017 ≤ a ∧ f 0 + f 2017 ≥ b) ∧ 
  (∀ (y : ℝ), f 0 + f 2017 ≤ y → y ≤ 1 + Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l147_14799


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l147_14735

theorem christmas_tree_lights (total : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : total = 95)
  (h2 : yellow = 37)
  (h3 : blue = 32) :
  total - (yellow + blue) = 26 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l147_14735


namespace NUMINAMATH_CALUDE_complex_multiplication_l147_14770

theorem complex_multiplication (i : ℂ) : i * i = -1 → 2 * i * (1 + i) = -2 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l147_14770


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l147_14708

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 10) (h3 : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ ∧ c = Real.sqrt (181 + 90 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l147_14708


namespace NUMINAMATH_CALUDE_total_stairs_l147_14791

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2 + 18) ∧
  ravi = (veronica * 3 / 2 : ℕ) ∧  -- Using integer division
  samir + veronica + ravi = 761

theorem total_stairs : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
sorry

end NUMINAMATH_CALUDE_total_stairs_l147_14791


namespace NUMINAMATH_CALUDE_all_female_finalists_probability_l147_14765

-- Define the total number of participants
def total_participants : ℕ := 6

-- Define the number of female participants
def female_participants : ℕ := 4

-- Define the number of male participants
def male_participants : ℕ := 2

-- Define the number of finalists to be chosen
def finalists : ℕ := 3

-- Define the probability of selecting all female finalists
def prob_all_female_finalists : ℚ := (female_participants.choose finalists) / (total_participants.choose finalists)

-- Theorem statement
theorem all_female_finalists_probability :
  prob_all_female_finalists = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_all_female_finalists_probability_l147_14765


namespace NUMINAMATH_CALUDE_quadratic_value_l147_14740

/-- A quadratic function with specific properties -/
def f (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c

/-- Theorem stating the properties of the quadratic function and its value at x = 5 -/
theorem quadratic_value (a b c : ℚ) :
  (f a b c (-2) = 10) →  -- Maximum value is 10 at x = -2
  ((2 * a * (-2) + b) = 0) →  -- Derivative is 0 at x = -2 (maximum condition)
  (f a b c 0 = -8) →  -- Passes through (0, -8)
  (f a b c 1 = 0) →  -- Passes through (1, 0)
  (f a b c 5 = -400/9) :=  -- Value at x = 5
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l147_14740


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l147_14703

/-- A circle tangent to the parabola y^2 = 2x (y > 0), its axis, and the x-axis -/
structure TangentCircle where
  /-- Center of the circle -/
  center : ℝ × ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the parabola y^2 = 2x (y > 0) -/
  tangent_to_parabola : center.2^2 = 2 * center.1
  /-- The circle is tangent to the x-axis -/
  tangent_to_x_axis : center.2 = radius
  /-- The circle's center is on the axis of the parabola (x-axis) -/
  on_parabola_axis : center.1 ≥ 0

/-- The equation of the circle is x^2 + y^2 - x - 2y + 1/4 = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
  x^2 + y^2 - x - 2*y + 1/4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_equation_l147_14703


namespace NUMINAMATH_CALUDE_sequence_periodicity_l147_14790

def isEventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ (n k : ℕ), k > 0 ∧ ∀ m, m ≥ n → a (m + k) = a m

theorem sequence_periodicity (a : ℕ → ℕ) 
    (h : ∀ n : ℕ, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
    isEventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l147_14790


namespace NUMINAMATH_CALUDE_quadratic_value_at_3_l147_14774

/-- A quadratic function y = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_x : ℝ
  point_y : ℝ

/-- Properties of the quadratic function -/
def has_properties (f : QuadraticFunction) : Prop :=
  f.min_value = -8 ∧
  f.min_x = -2 ∧
  f.point_x = 1 ∧
  f.point_y = 5 ∧
  f.point_y = f.a * f.point_x^2 + f.b * f.point_x + f.c

/-- The value of y when x = 3 -/
def y_at_3 (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- The main theorem -/
theorem quadratic_value_at_3 (f : QuadraticFunction) (h : has_properties f) :
  y_at_3 f = 253/9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_value_at_3_l147_14774


namespace NUMINAMATH_CALUDE_sequence_matches_first_five_terms_general_term_formula_l147_14701

/-- The sequence a_n defined by the given first five terms and the general formula -/
def a : ℕ → ℕ := λ n => n^2 + 5

/-- The theorem stating that the sequence matches the given first five terms -/
theorem sequence_matches_first_five_terms :
  a 1 = 6 ∧ a 2 = 9 ∧ a 3 = 14 ∧ a 4 = 21 ∧ a 5 = 30 := by sorry

/-- The main theorem proving that a_n is the general term formula for the sequence -/
theorem general_term_formula (n : ℕ) (h : n > 0) : a n = n^2 + 5 := by sorry

end NUMINAMATH_CALUDE_sequence_matches_first_five_terms_general_term_formula_l147_14701


namespace NUMINAMATH_CALUDE_first_class_students_l147_14748

theorem first_class_students (avg_first : ℝ) (students_second : ℕ) (avg_second : ℝ) (avg_all : ℝ)
  (h1 : avg_first = 30)
  (h2 : students_second = 50)
  (h3 : avg_second = 60)
  (h4 : avg_all = 48.75) :
  ∃ students_first : ℕ,
    students_first * avg_first + students_second * avg_second =
    (students_first + students_second) * avg_all ∧
    students_first = 30 :=
by sorry

end NUMINAMATH_CALUDE_first_class_students_l147_14748


namespace NUMINAMATH_CALUDE_division_simplification_l147_14725

theorem division_simplification (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (8 * a^3 * b - 4 * a^2 * b^2) / (4 * a * b) = 2 * a^2 - a * b :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l147_14725


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l147_14766

def C : Set Nat := {65, 67, 68, 71, 74}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 68 C := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l147_14766


namespace NUMINAMATH_CALUDE_square_of_1007_l147_14755

theorem square_of_1007 : (1007 : ℕ) ^ 2 = 1014049 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1007_l147_14755


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l147_14797

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_sum_equals_two :
  f (Real.log 2 / Real.log 10) + f (Real.log (1 / 2) / Real.log 10) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l147_14797


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_four_l147_14721

theorem fraction_zero_implies_x_equals_four (x : ℝ) : 
  (16 - x^2) / (x + 4) = 0 ∧ x + 4 ≠ 0 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_four_l147_14721


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l147_14749

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l147_14749


namespace NUMINAMATH_CALUDE_stock_income_theorem_l147_14786

/-- Calculates the income from a stock investment given the rate, market value, and investment amount. -/
def calculate_income (rate : ℚ) (market_value : ℚ) (investment : ℚ) : ℚ :=
  (rate / 100) * (investment / market_value) * 100

/-- Theorem stating that given the specific conditions, the income is 650. -/
theorem stock_income_theorem (rate market_value investment : ℚ) 
  (h_rate : rate = 10)
  (h_market_value : market_value = 96)
  (h_investment : investment = 6240) :
  calculate_income rate market_value investment = 650 :=
by
  sorry

#eval calculate_income 10 96 6240

end NUMINAMATH_CALUDE_stock_income_theorem_l147_14786


namespace NUMINAMATH_CALUDE_greatest_integer_with_nonpositive_product_l147_14717

theorem greatest_integer_with_nonpositive_product (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_coprime : Nat.Coprime a b) :
  ∀ n : ℕ, n > a * b →
    ∃ x y : ℤ, (n : ℤ) = a * x + b * y ∧ x * y > 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_nonpositive_product_l147_14717


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l147_14710

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l147_14710


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_fifth_l147_14726

theorem units_digit_of_six_to_fifth (n : ℕ) : n = 6^5 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_fifth_l147_14726


namespace NUMINAMATH_CALUDE_minimum_games_for_90_percent_win_rate_min_additional_games_is_25_l147_14753

theorem minimum_games_for_90_percent_win_rate : ℕ → Prop :=
  fun n =>
    let initial_games : ℕ := 5
    let initial_eagles_wins : ℕ := 2
    let total_games : ℕ := initial_games + n
    let total_eagles_wins : ℕ := initial_eagles_wins + n
    (total_eagles_wins : ℚ) / (total_games : ℚ) ≥ 9/10 ∧
    ∀ m : ℕ, m < n → (initial_eagles_wins + m : ℚ) / (initial_games + m : ℚ) < 9/10

theorem min_additional_games_is_25 : 
  minimum_games_for_90_percent_win_rate 25 := by sorry

end NUMINAMATH_CALUDE_minimum_games_for_90_percent_win_rate_min_additional_games_is_25_l147_14753


namespace NUMINAMATH_CALUDE_last_number_crossed_out_l147_14714

/-- Represents the circular arrangement of numbers from 1 to 2016 -/
def CircularArrangement := Fin 2016

/-- The deletion process function -/
def deletionProcess (n : ℕ) : ℕ :=
  (n + 2) * (n - 1) / 2

/-- Theorem stating that 2015 is the last number to be crossed out -/
theorem last_number_crossed_out :
  ∃ (n : ℕ), deletionProcess n = 2015 ∧ 
  ∀ (m : ℕ), m > n → deletionProcess m > 2015 :=
sorry

end NUMINAMATH_CALUDE_last_number_crossed_out_l147_14714


namespace NUMINAMATH_CALUDE_box_volume_l147_14745

theorem box_volume (l w h : ℝ) 
  (side1 : l * w = 120)
  (side2 : w * h = 72)
  (top : l * h = 60) :
  l * w * h = 720 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l147_14745


namespace NUMINAMATH_CALUDE_four_digit_sum_gcd_quotient_l147_14780

theorem four_digit_sum_gcd_quotient
  (a b c d : Nat)
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h3 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  let S := a + b + c + d
  let G := Nat.gcd a (Nat.gcd b (Nat.gcd c d))
  (33 * S - S * G) / S = 33 - G :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_gcd_quotient_l147_14780


namespace NUMINAMATH_CALUDE_distinct_values_count_l147_14777

-- Define the expression
def base : ℕ := 3
def expr := base^(base^(base^base))

-- Define the possible parenthesizations
def p1 := base^(base^(base^base))
def p2 := base^((base^base)^base)
def p3 := ((base^base)^base)^base
def p4 := (base^(base^base))^base
def p5 := (base^base)^(base^base)

-- Theorem statement
theorem distinct_values_count :
  ∃ (s : Finset ℕ), (∀ x : ℕ, x ∈ s ↔ (x = p1 ∨ x = p2 ∨ x = p3 ∨ x = p4 ∨ x = p5)) ∧ Finset.card s = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_values_count_l147_14777


namespace NUMINAMATH_CALUDE_lines_perpendicular_l147_14767

-- Define the slopes of the lines
def slope_l1 : ℚ := -2
def slope_l2 : ℚ := 1/2

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_perpendicular : perpendicular slope_l1 slope_l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_l147_14767


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l147_14760

def canoes_per_month (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- January (0-indexed)
  | n + 1 => 3 * canoes_per_month n

theorem total_canoes_by_april : 
  (canoes_per_month 0) + (canoes_per_month 1) + (canoes_per_month 2) + (canoes_per_month 3) = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l147_14760


namespace NUMINAMATH_CALUDE_vehicle_y_speed_l147_14773

/-- Proves that the average speed of vehicle Y is 45 miles per hour given the problem conditions -/
theorem vehicle_y_speed
  (initial_distance : ℝ)
  (vehicle_x_speed : ℝ)
  (overtake_time : ℝ)
  (final_lead : ℝ)
  (h1 : initial_distance = 22)
  (h2 : vehicle_x_speed = 36)
  (h3 : overtake_time = 5)
  (h4 : final_lead = 23) :
  (initial_distance + final_lead + vehicle_x_speed * overtake_time) / overtake_time = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_y_speed_l147_14773


namespace NUMINAMATH_CALUDE_sqrt_difference_of_squares_l147_14785

theorem sqrt_difference_of_squares : 
  (Real.sqrt 2023 + Real.sqrt 23) * (Real.sqrt 2023 - Real.sqrt 23) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_of_squares_l147_14785


namespace NUMINAMATH_CALUDE_students_catching_up_on_homework_l147_14736

theorem students_catching_up_on_homework (total : ℕ) (silent_reading : ℕ) (board_games : ℕ) :
  total = 24 →
  silent_reading = total / 2 →
  board_games = total / 3 →
  total - (silent_reading + board_games) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_students_catching_up_on_homework_l147_14736


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l147_14732

/-- If 55 cows eat 55 bags of husk in 55 days, then one cow will eat one bag of husk in 55 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 55 ∧ bags = 55 ∧ days = 55) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l147_14732


namespace NUMINAMATH_CALUDE_gcd_of_specific_squares_l147_14704

theorem gcd_of_specific_squares : Nat.gcd (131^2 + 243^2 + 357^2) (130^2 + 242^2 + 358^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_squares_l147_14704


namespace NUMINAMATH_CALUDE_equal_areas_of_same_side_lengths_l147_14705

/-- A polygon inscribed in a circle -/
structure InscribedPolygon (n : ℕ) where
  sides : Fin n → ℝ
  inscribed : Bool

/-- The area of an inscribed polygon -/
noncomputable def area (p : InscribedPolygon n) : ℝ := sorry

/-- Two polygons have the same set of side lengths -/
def same_side_lengths (p1 p2 : InscribedPolygon n) : Prop :=
  ∃ (σ : Equiv (Fin n) (Fin n)), ∀ i, p1.sides i = p2.sides (σ i)

theorem equal_areas_of_same_side_lengths (n : ℕ) (p1 p2 : InscribedPolygon n) 
  (h1 : p1.inscribed) (h2 : p2.inscribed) (h3 : same_side_lengths p1 p2) : 
  area p1 = area p2 := by sorry

end NUMINAMATH_CALUDE_equal_areas_of_same_side_lengths_l147_14705


namespace NUMINAMATH_CALUDE_hyperbola_properties_l147_14798

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 24 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) :=
  {(0, 6), (0, -6)}

-- Define the asymptotes of the reference hyperbola
def reference_asymptotes (x y : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_equation x y → 
    (∃ f ∈ foci, (x - f.1)^2 + (y - f.2)^2 = 36)) ∧
  (∀ x y, hyperbola_equation x y → reference_asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l147_14798


namespace NUMINAMATH_CALUDE_integer_triangle_properties_l147_14751

/-- A triangle with positive integer side lengths and circumradius -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  R : ℕ+

/-- Properties of an integer triangle -/
theorem integer_triangle_properties (T : IntegerTriangle) :
  ∃ (r : ℕ+) (P : ℕ),
    (∃ (k : ℕ), P = 4 * k) ∧
    (∃ (m n l : ℕ), T.a = 2 * m ∧ T.b = 2 * n ∧ T.c = 2 * l) := by
  sorry


end NUMINAMATH_CALUDE_integer_triangle_properties_l147_14751


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l147_14729

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l147_14729


namespace NUMINAMATH_CALUDE_divide_by_eight_l147_14758

theorem divide_by_eight (x y z : ℕ) (h1 : x > 0) (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 3 * y * z + 3) (h4 : 13 * y - x = 1) : z = 8 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_eight_l147_14758


namespace NUMINAMATH_CALUDE_four_numbers_puzzle_l147_14747

theorem four_numbers_puzzle (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_puzzle_l147_14747


namespace NUMINAMATH_CALUDE_calories_burned_proof_l147_14734

/-- The number of times players run up and down the bleachers -/
def num_runs : ℕ := 40

/-- The number of stairs climbed in one direction -/
def stairs_one_way : ℕ := 32

/-- The number of calories burned per stair -/
def calories_per_stair : ℕ := 2

/-- Calculates the total calories burned during the exercise -/
def total_calories_burned : ℕ :=
  num_runs * (2 * stairs_one_way) * calories_per_stair

/-- Theorem stating that the total calories burned is 5120 -/
theorem calories_burned_proof : total_calories_burned = 5120 := by
  sorry

end NUMINAMATH_CALUDE_calories_burned_proof_l147_14734


namespace NUMINAMATH_CALUDE_solution_value_l147_14756

-- Define the function E
def E (a b c : ℚ) : ℚ := a * b^2 + c

-- State the theorem
theorem solution_value :
  ∃ (a : ℚ), E a 3 10 = E a 5 (-2) ∧ a = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_value_l147_14756


namespace NUMINAMATH_CALUDE_grandma_gift_amount_l147_14796

/-- Calculates the amount grandma gave each person given the initial amount, expenses, and remaining amount. -/
theorem grandma_gift_amount
  (initial_amount : ℝ)
  (gasoline_cost : ℝ)
  (lunch_cost : ℝ)
  (gift_cost_per_person : ℝ)
  (num_people : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : gasoline_cost = 8)
  (h3 : lunch_cost = 15.65)
  (h4 : gift_cost_per_person = 5)
  (h5 : num_people = 2)
  (h6 : remaining_amount = 36.35) :
  (remaining_amount - (initial_amount - (gasoline_cost + lunch_cost + gift_cost_per_person * num_people))) / num_people = 10 :=
by sorry

end NUMINAMATH_CALUDE_grandma_gift_amount_l147_14796


namespace NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l147_14724

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Condition for two circles to have exactly 2 common tangents -/
def has_two_common_tangents (a : ℝ) : Prop :=
  6 < Real.sqrt (16 + a^2) ∧ Real.sqrt (16 + a^2) < 10

/-- Theorem stating the existence of a positive integer a satisfying the conditions -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ+, has_two_common_tangents a.val := by sorry

end NUMINAMATH_CALUDE_exists_a_with_two_common_tangents_l147_14724


namespace NUMINAMATH_CALUDE_triangle_angle_and_vector_dot_product_l147_14763

theorem triangle_angle_and_vector_dot_product 
  (A B C : ℝ) (a b c : ℝ) (k : ℝ) :
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  k > 1 ∧
  (∀ t : ℝ, 0 < t ∧ t ≤ 1 → -2 * t^2 + 4 * k * t + 1 ≤ 5) ∧
  -2 + 4 * k + 1 = 5 →
  B = π / 3 ∧ k = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_and_vector_dot_product_l147_14763


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l147_14739

/-- Proves the equivalence between a polar equation and its Cartesian form --/
theorem polar_to_cartesian_equivalence (x y ρ θ : ℝ) :
  (ρ = -4 * Real.cos θ + Real.sin θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  x^2 + y^2 + 4*x - y = 0 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l147_14739


namespace NUMINAMATH_CALUDE_dunk_a_clown_tickets_l147_14764

/-- Proves the number of tickets spent at the 'dunk a clown' booth -/
theorem dunk_a_clown_tickets (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) :
  total_tickets - (num_rides * tickets_per_ride) =
  total_tickets - num_rides * tickets_per_ride :=
by sorry

/-- Calculates the number of tickets spent at the 'dunk a clown' booth -/
def tickets_at_dunk_a_clown (total_tickets : ℕ) (num_rides : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  total_tickets - (num_rides * tickets_per_ride)

#eval tickets_at_dunk_a_clown 79 8 7

end NUMINAMATH_CALUDE_dunk_a_clown_tickets_l147_14764


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l147_14718

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  (5 : ℕ) = Nat.minFac ((n + 5) % m) :=
sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l147_14718


namespace NUMINAMATH_CALUDE_reflection_sum_l147_14757

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of reflection y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if two points are reflections of each other across a given line -/
def areReflections (A B : Point) (L : Line) : Prop :=
  let midpoint : Point := ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩
  (midpoint.y = L.m * midpoint.x + L.b) ∧
  (L.m = -(B.x - A.x) / (B.y - A.y))

/-- The main theorem -/
theorem reflection_sum (A B : Point) (L : Line) :
  A = ⟨2, 3⟩ → B = ⟨10, 7⟩ → areReflections A B L → L.m + L.b = 15 := by
  sorry


end NUMINAMATH_CALUDE_reflection_sum_l147_14757


namespace NUMINAMATH_CALUDE_basketball_practice_time_ratio_l147_14769

theorem basketball_practice_time_ratio :
  ∀ (total_practice_time shooting_time weightlifting_time running_time : ℕ),
  total_practice_time = 120 →
  shooting_time = total_practice_time / 2 →
  weightlifting_time = 20 →
  running_time = total_practice_time - shooting_time - weightlifting_time →
  running_time / weightlifting_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_practice_time_ratio_l147_14769


namespace NUMINAMATH_CALUDE_largest_factorable_n_l147_14711

/-- The largest value of n for which 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 271

/-- A function that checks if a quadratic expression 3x^2 + nx + 90 can be factored as the product of two linear factors with integer coefficients -/
def is_factorable (n : ℤ) : Prop :=
  ∃ (a b : ℤ), (3 * a + b = n) ∧ (a * b = 90)

theorem largest_factorable_n :
  (is_factorable largest_n) ∧ 
  (∀ m : ℤ, m > largest_n → ¬(is_factorable m)) :=
sorry

end NUMINAMATH_CALUDE_largest_factorable_n_l147_14711


namespace NUMINAMATH_CALUDE_farmer_apples_final_apple_count_l147_14700

theorem farmer_apples (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial - given_away + harvested = initial + harvested - given_away :=
by sorry

theorem final_apple_count (initial : ℝ) (given_away : ℝ) (harvested : ℝ) :
  initial = 5708 → given_away = 2347.5 → harvested = 1526.75 →
  initial - given_away + harvested = 4887.25 :=
by sorry

end NUMINAMATH_CALUDE_farmer_apples_final_apple_count_l147_14700


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l147_14762

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → Complex.abs (-i) + i^2018 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l147_14762


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_l147_14761

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_l147_14761


namespace NUMINAMATH_CALUDE_bus_problem_l147_14713

theorem bus_problem (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 60 → num_stops = 5 → 
  (initial_students : ℚ) * (2/3)^num_stops = 640/81 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l147_14713


namespace NUMINAMATH_CALUDE_ellipse_equation_l147_14771

/-- The locus of points P such that |F₁F₂| is the arithmetic mean of |PF₁| and |PF₂| -/
def EllipseLocus (F₁ F₂ : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist F₁ F₂ = (dist P F₁ + dist P F₂) / 2}

theorem ellipse_equation (P : ℝ × ℝ) :
  P ∈ EllipseLocus (-2, 0) (2, 0) ↔ P.1^2 / 16 + P.2^2 / 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l147_14771


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l147_14782

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t + 1, (t - 1)^2)

-- Define the ray at θ = π/4
def ray (x : ℝ) : ℝ × ℝ := (x, x)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, curve t = p ∧ ray p.1 = p}

-- Theorem statement
theorem midpoint_of_intersection :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
  (A.1 + B.1) / 2 = 2.5 ∧ (A.2 + B.2) / 2 = 2.5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l147_14782


namespace NUMINAMATH_CALUDE_oakwood_academy_walking_students_l147_14776

theorem oakwood_academy_walking_students (total : ℚ) :
  let bus : ℚ := 1 / 3
  let car : ℚ := 1 / 5
  let cycle : ℚ := 1 / 8
  let walk : ℚ := total - (bus + car + cycle)
  walk = 41 / 120 := by
  sorry

end NUMINAMATH_CALUDE_oakwood_academy_walking_students_l147_14776


namespace NUMINAMATH_CALUDE_function_value_difference_bound_l147_14744

theorem function_value_difference_bound
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_difference_bound_l147_14744


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l147_14781

theorem neither_sufficient_nor_necessary :
  ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) → (x + y > 3)) ∧
  ¬(∀ x y : ℝ, (x + y > 3) → (x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l147_14781
