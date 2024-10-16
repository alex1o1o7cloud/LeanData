import Mathlib

namespace NUMINAMATH_CALUDE_bee_speed_l3854_385443

/-- The speed of a bee flying between flowers -/
theorem bee_speed (time_to_rose time_to_poppy : ℝ)
  (distance_difference speed_difference : ℝ)
  (h1 : time_to_rose = 10)
  (h2 : time_to_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_difference = 3) :
  ∃ (speed_to_rose : ℝ),
    speed_to_rose * time_to_rose = 
    (speed_to_rose + speed_difference) * time_to_poppy + distance_difference ∧
    speed_to_rose = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_l3854_385443


namespace NUMINAMATH_CALUDE_largest_sum_proof_l3854_385489

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11, 1/4 + 1/12, 1/4 + 1/13]
  (∀ x ∈ sums, x ≤ (1/4 + 1/9)) ∧ (1/4 + 1/9 = 13/36) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l3854_385489


namespace NUMINAMATH_CALUDE_hdtv_horizontal_length_l3854_385447

theorem hdtv_horizontal_length :
  ∀ (diagonal : ℝ) (aspect_width aspect_height : ℕ),
    diagonal = 42 →
    aspect_width = 16 →
    aspect_height = 9 →
    ∃ (horizontal : ℝ),
      horizontal = (aspect_width : ℝ) * diagonal / Real.sqrt ((aspect_width ^ 2 : ℝ) + (aspect_height ^ 2 : ℝ)) ∧
      horizontal = 672 / Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_hdtv_horizontal_length_l3854_385447


namespace NUMINAMATH_CALUDE_remaining_budget_for_accessories_l3854_385436

def total_budget : ℕ := 250
def frame_cost : ℕ := 85
def front_wheel_cost : ℕ := 35
def rear_wheel_cost : ℕ := 40
def seat_cost : ℕ := 25
def handlebar_tape_cost : ℕ := 15
def water_bottle_cage_cost : ℕ := 10
def bike_lock_cost : ℕ := 20
def future_expenses : ℕ := 10

def total_expenses : ℕ :=
  frame_cost + front_wheel_cost + rear_wheel_cost + seat_cost +
  handlebar_tape_cost + water_bottle_cage_cost + bike_lock_cost + future_expenses

theorem remaining_budget_for_accessories :
  total_budget - total_expenses = 10 := by sorry

end NUMINAMATH_CALUDE_remaining_budget_for_accessories_l3854_385436


namespace NUMINAMATH_CALUDE_avg_people_per_hour_rounded_l3854_385485

/-- The number of people moving to Texas in five days -/
def total_people : ℕ := 5000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving to Texas per hour -/
def avg_people_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_per_hour_rounded :
  round_to_nearest avg_people_per_hour = 42 := by
  sorry

end NUMINAMATH_CALUDE_avg_people_per_hour_rounded_l3854_385485


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3854_385452

/-- Given a line with equation 3x - 6y = 12, prove that its slope (and the slope of any parallel line) is 1/2. -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3854_385452


namespace NUMINAMATH_CALUDE_sector_central_angle_l3854_385467

/-- Given a circular sector with circumference 6 and area 2, prove that its central angle in radians is either 1 or 4. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 6) (h2 : (1 / 2) * l * r = 2) :
  let α := l / r
  α = 1 ∨ α = 4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3854_385467


namespace NUMINAMATH_CALUDE_no_safe_numbers_l3854_385481

def is_p_safe (n p : ℕ+) : Prop :=
  ∀ k : ℤ, |n.val - k * p.val| > 3

theorem no_safe_numbers : 
  ¬ ∃ n : ℕ+, n.val ≤ 15000 ∧ 
    is_p_safe n 5 ∧ 
    is_p_safe n 7 ∧ 
    is_p_safe n 11 :=
sorry

end NUMINAMATH_CALUDE_no_safe_numbers_l3854_385481


namespace NUMINAMATH_CALUDE_counterfeit_coin_identification_l3854_385417

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- Represents a weighing action on the balance scale -/
def weighing (c1 c2 : Coin) : WeighingResult :=
  match c1, c2 with
  | Coin.Real, Coin.Real => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighingResult.Unequal
  | Coin.Real, Coin.Counterfeit => WeighingResult.Unequal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal

/-- Theorem stating that the counterfeit coin can be identified in at most 2 weighings -/
theorem counterfeit_coin_identification
  (coins : Fin 4 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.Counterfeit) :
  ∃ (w1 w2 : Fin 4 × Fin 4),
    let r1 := weighing (coins w1.1) (coins w1.2)
    let r2 := weighing (coins w2.1) (coins w2.2)
    ∃ i, coins i = Coin.Counterfeit ∧
         ∀ j, j ≠ i → coins j = Coin.Real :=
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_identification_l3854_385417


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l3854_385424

/-- The area of a square containing four touching circles -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l3854_385424


namespace NUMINAMATH_CALUDE_parabola_focus_at_triangle_centroid_l3854_385413

/-- Given a triangle ABC with vertices A(-1,2), B(3,4), and C(4,-6),
    and a parabola y^2 = ax with focus at the centroid of ABC,
    prove that a = 8. -/
theorem parabola_focus_at_triangle_centroid :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (3, 4)
  let C : ℝ × ℝ := (4, -6)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  ∀ a : ℝ, (∀ x y : ℝ, y^2 = a*x → (x = a/4 ↔ (x, y) = centroid)) → a = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_at_triangle_centroid_l3854_385413


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3854_385487

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x - y) / (2*x + 3*y) + (2*x + 3*y) / (x - y) = 2) :
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 34/15 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3854_385487


namespace NUMINAMATH_CALUDE_loan_years_calculation_l3854_385441

/-- Calculates the number of years for a loan given the original amount, current balance, and annual reduction rate. -/
theorem loan_years_calculation (c V t : ℝ) (h : V = c / (1 + t)^(3 * n)) :
  n = Real.log (c / V) / (3 * Real.log (1 + t)) :=
sorry

end NUMINAMATH_CALUDE_loan_years_calculation_l3854_385441


namespace NUMINAMATH_CALUDE_central_cell_value_l3854_385421

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l3854_385421


namespace NUMINAMATH_CALUDE_star_equation_solution_l3854_385465

/-- Custom star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem: If 4 ⋆ x = 46, then x = 50/7 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 46) : x = 50/7 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3854_385465


namespace NUMINAMATH_CALUDE_base_edges_same_color_l3854_385426

/-- A color type representing red or green -/
inductive Color
| Red
| Green

/-- A vertex of the prism -/
structure Vertex where
  base : Bool  -- True for top base, False for bottom base
  index : Fin 5

/-- An edge of the prism -/
structure Edge where
  v1 : Vertex
  v2 : Vertex

/-- A prism with pentagonal bases -/
structure Prism where
  /-- The color of each edge -/
  edge_color : Edge → Color
  /-- Ensure that any triangle has edges of different colors -/
  triangle_property : ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1 →
    (edge_color ⟨v1, v2⟩ ≠ edge_color ⟨v2, v3⟩ ∨
     edge_color ⟨v2, v3⟩ ≠ edge_color ⟨v3, v1⟩ ∨
     edge_color ⟨v3, v1⟩ ≠ edge_color ⟨v1, v2⟩)

/-- The main theorem -/
theorem base_edges_same_color (p : Prism) :
  (∀ (i j : Fin 5), p.edge_color ⟨⟨true, i⟩, ⟨true, j⟩⟩ = p.edge_color ⟨⟨false, i⟩, ⟨false, j⟩⟩) :=
sorry

end NUMINAMATH_CALUDE_base_edges_same_color_l3854_385426


namespace NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3854_385433

theorem infinite_geometric_series_second_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 20) :
  let a := S * (1 - r)
  a * r = 15/4 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l3854_385433


namespace NUMINAMATH_CALUDE_three_Y_five_l3854_385484

-- Define the operation Y
def Y (a b : ℤ) : ℤ := 3*b + 8*a - a^2

-- Theorem to prove
theorem three_Y_five : Y 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_Y_five_l3854_385484


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3854_385408

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 ∧ y < 10 ∧ y = 2 * x ∧ (10 * x + y) - (x + y) = 8 → 
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3854_385408


namespace NUMINAMATH_CALUDE_tangent_line_theorem_intersecting_line_theorem_l3854_385498

-- Define the circle C
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 6
def tangent_line_2 (x y : ℝ) : Prop := 5*x + 12*y - 78 = 0

-- Define the intersecting line equation
def intersecting_line (x y : ℝ) : Prop := 
  ∃ k, y - 4 = k*(x - 6) ∧ (k = (4 + Real.sqrt 17)/3 ∨ k = (4 - Real.sqrt 17)/3)

theorem tangent_line_theorem :
  ∀ x y : ℝ, 
  (∃ l : ℝ → ℝ → Prop, (l x y ↔ tangent_line_1 x ∨ tangent_line_2 x y) ∧ 
    (l (point_P.1) (point_P.2) ∧ 
     ∀ a b : ℝ, circle_equation a b → (l a b → a = (point_P.1) ∧ b = (point_P.2)))) :=
sorry

theorem intersecting_line_theorem :
  ∀ x y : ℝ,
  (intersecting_line x y →
    (x = point_P.1 ∧ y = point_P.2 ∨
     (∃ a b : ℝ, circle_equation a b ∧ intersecting_line a b ∧
      ∃ c d : ℝ, circle_equation c d ∧ intersecting_line c d ∧
      (a - c)^2 + (b - d)^2 = 18))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_intersecting_line_theorem_l3854_385498


namespace NUMINAMATH_CALUDE_a_5_value_l3854_385456

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 7 = 2 →
  a 3 + a 7 = -4 →
  a 5 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l3854_385456


namespace NUMINAMATH_CALUDE_min_value_theorem_l3854_385462

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2/a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 2 → 3*x + y ≥ (7 + 2*Real.sqrt 6)/2) ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2/a₀ + 1/b₀ = 2 ∧ 3*a₀ + b₀ = (7 + 2*Real.sqrt 6)/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3854_385462


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3854_385416

/-- The inclination angle of a line with equation x + √3 * y + c = 0 is 5π/6 --/
theorem line_inclination_angle (c : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y + c = 0}
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < π ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → Real.tan θ = -(1 / Real.sqrt 3)) ∧
    θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3854_385416


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l3854_385448

theorem fourth_degree_polynomial_roots :
  let p : ℝ → ℝ := λ x => 3*x^4 - 19*x^3 + 34*x^2 - 19*x + 3
  (∀ x : ℝ, p x = 0 ↔ x = 2 + Real.sqrt 3 ∨ 
                      x = 2 - Real.sqrt 3 ∨ 
                      x = (7 + Real.sqrt 13) / 6 ∨ 
                      x = (7 - Real.sqrt 13) / 6) :=
by sorry

end NUMINAMATH_CALUDE_fourth_degree_polynomial_roots_l3854_385448


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l3854_385437

/-- The equation (x-3)^2 = (3y+4)^2 - 75 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
  ∀ (x y : ℝ), (x - 3)^2 = (3*y + 4)^2 - 75 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l3854_385437


namespace NUMINAMATH_CALUDE_particle_hit_probability_l3854_385471

/-- Probability of hitting (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

/-- The particle starts at (5,5) -/
def start_pos : ℕ × ℕ := (5, 5)

/-- The probability of hitting (0,0) is m/3^n -/
def hit_prob : ℚ := 1 / 3^5

theorem particle_hit_probability :
  P start_pos.1 start_pos.2 = hit_prob :=
sorry

end NUMINAMATH_CALUDE_particle_hit_probability_l3854_385471


namespace NUMINAMATH_CALUDE_theorem_A_theorem_B_theorem_C_theorem_D_l3854_385490

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- Axioms for the relations
axiom different_planes : α ≠ β
axiom different_lines : m ≠ n

-- Theorem A
theorem theorem_A : 
  parallel_planes α β → perpendicular_plane_line α m → perpendicular_plane_line β m :=
sorry

-- Theorem B
theorem theorem_B :
  perpendicular_plane_line α m → perpendicular_plane_line α n → parallel_lines m n :=
sorry

-- Theorem C
theorem theorem_C :
  perpendicular_planes α β → intersection α β = n → ¬parallel_line_plane m α → 
  perpendicular_lines m n → perpendicular_plane_line β m :=
sorry

-- Theorem D (which should be false)
theorem theorem_D :
  parallel_line_plane m α → parallel_line_plane n α → 
  parallel_line_plane m β → parallel_line_plane n β → 
  ¬(parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_theorem_A_theorem_B_theorem_C_theorem_D_l3854_385490


namespace NUMINAMATH_CALUDE_snowfall_rate_hamilton_l3854_385401

/-- Snowfall rates and depths in Kingston and Hamilton --/
theorem snowfall_rate_hamilton (
  kingston_initial : ℝ) (hamilton_initial : ℝ) 
  (duration : ℝ) (kingston_rate : ℝ) (hamilton_rate : ℝ) :
  kingston_initial = 12.1 →
  hamilton_initial = 18.6 →
  duration = 13 →
  kingston_rate = 2.6 →
  kingston_initial + kingston_rate * duration = hamilton_initial + hamilton_rate * duration →
  hamilton_rate = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_rate_hamilton_l3854_385401


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l3854_385483

theorem at_least_one_real_root (p q : ℕ) (h_distinct : p ≠ q) (h_positive_p : p > 0) (h_positive_q : q > 0) :
  (p^2 : ℝ) - 4*q ≥ 0 ∨ (q^2 : ℝ) - 4*p ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l3854_385483


namespace NUMINAMATH_CALUDE_unique_fixed_point_of_f_and_f_inv_l3854_385415

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 9

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (x + 9) / 4

-- Theorem statement
theorem unique_fixed_point_of_f_and_f_inv :
  ∃! x : ℝ, f x = f_inv x :=
sorry

end NUMINAMATH_CALUDE_unique_fixed_point_of_f_and_f_inv_l3854_385415


namespace NUMINAMATH_CALUDE_certain_number_problem_l3854_385478

theorem certain_number_problem (x : ℝ) : ((x + 20) * 2) / 2 - 2 = 88 / 2 ↔ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3854_385478


namespace NUMINAMATH_CALUDE_collinearity_condition_perpendicularity_condition_l3854_385418

-- Define the points as functions of a
def A (a : ℝ) : ℝ × ℝ := (1, -2*a)
def B (a : ℝ) : ℝ × ℝ := (2, a)
def C (a : ℝ) : ℝ × ℝ := (2+a, 0)
def D (a : ℝ) : ℝ × ℝ := (2*a, 1)

-- Define collinearity of three points
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define perpendicularity of two lines
def perpendicular (p1 q1 p2 q2 : ℝ × ℝ) : Prop :=
  (q1.2 - p1.2) * (q2.2 - p2.2) = -(q1.1 - p1.1) * (q2.1 - p2.1)

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ a : ℝ, collinear (A a) (B a) (C a) ↔ a = -1/3 :=
sorry

-- Theorem 2: Perpendicularity condition
theorem perpendicularity_condition :
  ∀ a : ℝ, perpendicular (A a) (B a) (C a) (D a) ↔ a = 1/2 :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_perpendicularity_condition_l3854_385418


namespace NUMINAMATH_CALUDE_matchstick_houses_l3854_385434

theorem matchstick_houses (initial_matchsticks : ℕ) (num_houses : ℕ) 
  (h1 : initial_matchsticks = 600)
  (h2 : num_houses = 30) :
  (initial_matchsticks / 2) / num_houses = 10 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_houses_l3854_385434


namespace NUMINAMATH_CALUDE_inequality_solution_solution_set_complete_l3854_385493

-- Define the inequality
def inequality (x : ℝ) : Prop := -x^2 + 3*x + 4 < 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > 4 ∨ x < -1}

-- Theorem stating that the solution set satisfies the inequality
theorem inequality_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

-- Theorem stating that the solution set is complete
theorem solution_set_complete :
  ∀ x : ℝ, inequality x → x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_solution_set_complete_l3854_385493


namespace NUMINAMATH_CALUDE_hcd_4760_280_minus_12_l3854_385414

theorem hcd_4760_280_minus_12 : Nat.gcd 4760 280 - 12 = 268 := by
  sorry

end NUMINAMATH_CALUDE_hcd_4760_280_minus_12_l3854_385414


namespace NUMINAMATH_CALUDE_sock_ratio_l3854_385479

theorem sock_ratio (total : ℕ) (blue : ℕ) (h1 : total = 180) (h2 : blue = 60) :
  (total - blue) / total = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sock_ratio_l3854_385479


namespace NUMINAMATH_CALUDE_inequality_proof_l3854_385469

theorem inequality_proof (x : ℝ) : (x - 2) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3854_385469


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l3854_385432

/-- Inverse variation relation between three quantities -/
def inverse_variation (r s t : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ r * s = k₁ ∧ r * t = k₂

theorem inverse_variation_solution (r₁ s₁ t₁ r₂ s₂ t₂ : ℝ) :
  inverse_variation r₁ s₁ t₁ →
  inverse_variation r₂ s₂ t₂ →
  r₁ = 1500 →
  s₁ = 0.25 →
  t₁ = 0.5 →
  r₂ = 3000 →
  s₂ = 0.125 ∧ t₂ = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_solution_l3854_385432


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3854_385451

theorem cubic_equation_root (a b : ℚ) : 
  (∃ x : ℂ, x^3 + a*x^2 + b*x + 15 = 0 ∧ x = -1 - 3*Real.sqrt 2) →
  a = 19/17 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3854_385451


namespace NUMINAMATH_CALUDE_a_plus_b_values_l3854_385499

theorem a_plus_b_values (a b : ℝ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_values_l3854_385499


namespace NUMINAMATH_CALUDE_orthocenter_PQR_l3854_385438

/-- The orthocenter of a triangle PQR in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle PQR with given coordinates is (1/2, 13/2, 15/2). -/
theorem orthocenter_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (1/2, 13/2, 15/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_PQR_l3854_385438


namespace NUMINAMATH_CALUDE_four_inch_gold_cube_value_l3854_385412

/-- The value of a cube of gold given its side length -/
def gold_value (side : ℕ) : ℚ :=
  let base_value : ℚ := 300
  let base_side : ℕ := 1
  let increase_rate : ℚ := 1.1
  let value_per_cubic_inch : ℚ := base_value * (increase_rate ^ (side - base_side))
  (side ^ 3 : ℚ) * value_per_cubic_inch

/-- Theorem stating the value of a 4-inch cube of gold -/
theorem four_inch_gold_cube_value :
  ⌊gold_value 4⌋ = 25555 :=
sorry

end NUMINAMATH_CALUDE_four_inch_gold_cube_value_l3854_385412


namespace NUMINAMATH_CALUDE_envelope_addressing_machines_l3854_385463

theorem envelope_addressing_machines (machine1_time machine2_time combined_time : ℚ) :
  machine1_time = 10 →
  combined_time = 4 →
  (1 / machine1_time + 1 / machine2_time = 1 / combined_time) →
  machine2_time = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_envelope_addressing_machines_l3854_385463


namespace NUMINAMATH_CALUDE_cars_between_black_and_white_l3854_385445

theorem cars_between_black_and_white :
  ∀ (n : ℕ) (black_pos_right : ℕ) (white_pos_left : ℕ),
    n = 20 →
    black_pos_right = 16 →
    white_pos_left = 11 →
    (n - black_pos_right) - (white_pos_left - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cars_between_black_and_white_l3854_385445


namespace NUMINAMATH_CALUDE_family_average_age_unchanged_l3854_385454

theorem family_average_age_unchanged 
  (initial_members : ℕ) 
  (initial_avg_age : ℝ) 
  (years_passed : ℕ) 
  (baby_age : ℝ) 
  (h1 : initial_members = 5)
  (h2 : initial_avg_age = 17)
  (h3 : years_passed = 3)
  (h4 : baby_age = 2) : 
  initial_avg_age = 
    (initial_members * (initial_avg_age + years_passed) + baby_age) / (initial_members + 1) := by
  sorry

#check family_average_age_unchanged

end NUMINAMATH_CALUDE_family_average_age_unchanged_l3854_385454


namespace NUMINAMATH_CALUDE_daily_profit_at_52_selling_price_for_profit_l3854_385427

/-- Represents the pricing and sales model of a company's craft. -/
structure CraftSalesModel where
  cost_per_unit : ℝ
  base_price : ℝ
  base_sales : ℝ
  price_sales_ratio : ℝ
  max_price : ℝ

/-- Calculates the daily sales profit for a given selling price. -/
def daily_profit (model : CraftSalesModel) (selling_price : ℝ) : ℝ :=
  let sales_volume := model.base_sales - model.price_sales_ratio * (selling_price - model.base_price)
  let profit_per_unit := selling_price - model.cost_per_unit
  sales_volume * profit_per_unit

/-- Theorem stating the daily profit for a specific selling price. -/
theorem daily_profit_at_52 (model : CraftSalesModel) 
  (h1 : model.cost_per_unit = 40)
  (h2 : model.base_price = 50)
  (h3 : model.base_sales = 100)
  (h4 : model.price_sales_ratio = 2)
  (h5 : model.max_price = 65) :
  daily_profit model 52 = 1152 := by
  sorry

/-- Theorem stating the selling price that results in a specific daily profit. -/
theorem selling_price_for_profit (model : CraftSalesModel)
  (h1 : model.cost_per_unit = 40)
  (h2 : model.base_price = 50)
  (h3 : model.base_sales = 100)
  (h4 : model.price_sales_ratio = 2)
  (h5 : model.max_price = 65) :
  ∃ (x : ℝ), x ≤ 65 ∧ daily_profit model x = 1350 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_daily_profit_at_52_selling_price_for_profit_l3854_385427


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l3854_385419

theorem scooter_repair_cost (initial_cost selling_price gain_percent : ℚ) 
  (h1 : initial_cost = 800)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) : 
  let repair_cost := selling_price / (1 + gain_percent / 100) - initial_cost
  repair_cost = 200 := by
  sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l3854_385419


namespace NUMINAMATH_CALUDE_interview_bounds_l3854_385409

theorem interview_bounds (students : ℕ) (junior_high : ℕ) (teachers : ℕ) (table_tennis : ℕ) (basketball : ℕ)
  (h1 : students = 6)
  (h2 : junior_high = 4)
  (h3 : teachers = 2)
  (h4 : table_tennis = 5)
  (h5 : basketball = 2)
  (h6 : junior_high ≤ students) :
  ∃ (min max : ℕ),
    (min = students + teachers) ∧
    (max = students - junior_high + teachers + table_tennis + basketball + junior_high) ∧
    (min = 8) ∧
    (max = 15) ∧
    (∀ n : ℕ, n ≥ min ∧ n ≤ max) := by
  sorry

end NUMINAMATH_CALUDE_interview_bounds_l3854_385409


namespace NUMINAMATH_CALUDE_symmetric_log4_implies_neg_exp4_l3854_385495

-- Define the logarithm base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Define the symmetry condition
def symmetric_about_line (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-y) = -x

-- State the theorem
theorem symmetric_log4_implies_neg_exp4 (f : ℝ → ℝ) :
  symmetric_about_line f log4 → ∀ x, f x = -4^(-x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_log4_implies_neg_exp4_l3854_385495


namespace NUMINAMATH_CALUDE_arithmetic_computation_l3854_385464

theorem arithmetic_computation : 2 + 3^2 * 4 - 5 + 6 / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l3854_385464


namespace NUMINAMATH_CALUDE_pool_water_increase_l3854_385491

theorem pool_water_increase (total_capacity : ℝ) (additional_water : ℝ) 
  (h1 : total_capacity = 1857.1428571428573)
  (h2 : additional_water = 300)
  (h3 : additional_water + (total_capacity * 0.7 - additional_water) = total_capacity * 0.7) :
  let initial_water := total_capacity * 0.7 - additional_water
  let percentage_increase := (additional_water / initial_water) * 100
  percentage_increase = 30 := by
sorry

end NUMINAMATH_CALUDE_pool_water_increase_l3854_385491


namespace NUMINAMATH_CALUDE_bridge_length_l3854_385400

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 350 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3854_385400


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l3854_385482

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^12 + 18*x^6 + 81) / (x^6 + 9) = 738 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l3854_385482


namespace NUMINAMATH_CALUDE_division_problem_l3854_385460

theorem division_problem (dividend : ℤ) (divisor : ℤ) (remainder : ℤ) (quotient : ℤ) :
  dividend = 12 →
  divisor = 17 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3854_385460


namespace NUMINAMATH_CALUDE_system_solutions_l3854_385455

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^3 + y^3 = 3*y + 3*z + 4 ∧
  y^3 + z^3 = 3*z + 3*x + 4 ∧
  z^3 + x^3 = 3*x + 3*y + 4

/-- The solutions to the system of equations -/
theorem system_solutions :
  (∀ x y z : ℝ, system x y z ↔ (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 2 ∧ y = 2 ∧ z = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3854_385455


namespace NUMINAMATH_CALUDE_exists_21_win_stretch_l3854_385446

/-- Represents the cumulative wins of a chess player over 77 days -/
def CumulativeWins := Fin 78 → ℕ

/-- The conditions for the chess player's winning record -/
def ValidWinningRecord (x : CumulativeWins) : Prop :=
  (∀ i : Fin 77, x (i + 1) > x i) ∧ 
  (∀ i : Fin 71, x (i + 7) - x i ≤ 12) ∧
  x 0 = 0 ∧ x 77 ≤ 132

/-- The theorem stating that there exists a stretch of consecutive days with exactly 21 wins -/
theorem exists_21_win_stretch (x : CumulativeWins) (h : ValidWinningRecord x) : 
  ∃ i j : Fin 78, i < j ∧ x j - x i = 21 := by
  sorry


end NUMINAMATH_CALUDE_exists_21_win_stretch_l3854_385446


namespace NUMINAMATH_CALUDE_white_balls_count_l3854_385425

theorem white_balls_count (yellow_balls : ℕ) (yellow_prob : ℚ) : 
  yellow_balls = 15 → yellow_prob = 3/4 → 
  ∃ (white_balls : ℕ), (yellow_balls : ℚ) / ((white_balls : ℚ) + yellow_balls) = yellow_prob ∧ white_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3854_385425


namespace NUMINAMATH_CALUDE_ben_homework_theorem_l3854_385405

/-- The time in minutes Ben has to work on homework -/
def total_time : ℕ := 60

/-- The time taken to solve the i-th problem -/
def problem_time (i : ℕ) : ℕ := i

/-- The sum of time taken to solve the first n problems -/
def total_problem_time (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- The maximum number of problems Ben can solve -/
def max_problems : ℕ := 10

theorem ben_homework_theorem :
  (∀ n : ℕ, n > max_problems → total_problem_time n > total_time) ∧
  total_problem_time max_problems ≤ total_time :=
sorry

end NUMINAMATH_CALUDE_ben_homework_theorem_l3854_385405


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3854_385450

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, (4 * x^2 + 3 = 3 * x - 9) ↔ (x = a + b * I ∨ x = a - b * I)) →
  a + b^2 = 207/64 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3854_385450


namespace NUMINAMATH_CALUDE_baker_production_theorem_l3854_385461

/-- Represents the baker's bread production over a period of time. -/
structure BakerProduction where
  loaves_per_oven_hour : ℕ
  num_ovens : ℕ
  weekday_hours : ℕ
  weekend_hours : ℕ
  num_weeks : ℕ

/-- Calculates the total number of loaves baked over the given period. -/
def total_loaves (bp : BakerProduction) : ℕ :=
  let loaves_per_hour := bp.loaves_per_oven_hour * bp.num_ovens
  let weekday_loaves := loaves_per_hour * bp.weekday_hours * 5
  let weekend_loaves := loaves_per_hour * bp.weekend_hours * 2
  (weekday_loaves + weekend_loaves) * bp.num_weeks

/-- Theorem stating that given the baker's production conditions, 
    the total number of loaves baked in 3 weeks is 1740. -/
theorem baker_production_theorem (bp : BakerProduction) 
  (h1 : bp.loaves_per_oven_hour = 5)
  (h2 : bp.num_ovens = 4)
  (h3 : bp.weekday_hours = 5)
  (h4 : bp.weekend_hours = 2)
  (h5 : bp.num_weeks = 3) :
  total_loaves bp = 1740 := by
  sorry

#eval total_loaves ⟨5, 4, 5, 2, 3⟩

end NUMINAMATH_CALUDE_baker_production_theorem_l3854_385461


namespace NUMINAMATH_CALUDE_sector_perimeter_l3854_385458

/-- The perimeter of a circular sector with a central angle of 180 degrees and a radius of 28.000000000000004 cm is 143.96459430079216 cm. -/
theorem sector_perimeter : 
  let r : ℝ := 28.000000000000004
  let θ : ℝ := 180
  let arc_length : ℝ := (θ / 360) * 2 * Real.pi * r
  let perimeter : ℝ := arc_length + 2 * r
  perimeter = 143.96459430079216 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l3854_385458


namespace NUMINAMATH_CALUDE_no_common_integer_solutions_l3854_385420

theorem no_common_integer_solutions : ¬∃ y : ℤ, (-3 * y ≥ y + 9) ∧ (2 * y ≥ 14) ∧ (-4 * y ≥ 2 * y + 21) := by
  sorry

end NUMINAMATH_CALUDE_no_common_integer_solutions_l3854_385420


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3854_385440

theorem quadratic_equation_solutions :
  let equation := fun y : ℝ => 3 * y * (y - 1) = 2 * (y - 1)
  (equation (2/3) ∧ equation 1) ∧
  ∀ y : ℝ, equation y → (y = 2/3 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3854_385440


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3854_385488

theorem unique_prime_with_prime_sums : ∀ p : ℕ, 
  Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) → p = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_sums_l3854_385488


namespace NUMINAMATH_CALUDE_round_73_26_repeating_l3854_385411

/-- Represents a repeating decimal number -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The specific number 73.2626... -/
def number : RepeatingDecimal :=
  { integerPart := 73,
    nonRepeatingPart := 26,
    repeatingPart := 26 }

theorem round_73_26_repeating :
  roundToHundredth number = 73.26 :=
sorry

end NUMINAMATH_CALUDE_round_73_26_repeating_l3854_385411


namespace NUMINAMATH_CALUDE_inequality_proof_l3854_385459

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.tan (23 * π / 180) / (1 - Real.tan (23 * π / 180)^2))
  (hb : b = 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180))
  (hc : c = Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)) :
  c < b ∧ b < a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3854_385459


namespace NUMINAMATH_CALUDE_sphere_cube_volume_ratio_l3854_385475

/-- Given a cube with its vertices on a spherical surface, 
    the ratio of the sphere's volume to the cube's volume is √3π/2 -/
theorem sphere_cube_volume_ratio : 
  ∀ (cube_edge : ℝ) (sphere_radius : ℝ),
  cube_edge > 0 →
  sphere_radius > 0 →
  sphere_radius = cube_edge * (Real.sqrt 3) / 2 →
  (4 / 3 * Real.pi * sphere_radius^3) / cube_edge^3 = Real.sqrt 3 * Real.pi / 2 :=
by sorry


end NUMINAMATH_CALUDE_sphere_cube_volume_ratio_l3854_385475


namespace NUMINAMATH_CALUDE_solve_bank_problem_l3854_385470

def bank_problem (initial_balance : ℚ) : Prop :=
  let tripled_balance := initial_balance * 3
  let balance_after_withdrawal := tripled_balance - 250
  balance_after_withdrawal = 950

theorem solve_bank_problem :
  ∃ (initial_balance : ℚ), bank_problem initial_balance ∧ initial_balance = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_bank_problem_l3854_385470


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l3854_385402

/-- The probability of drawing two chips of different colors from a bag containing
    6 green chips, 5 purple chips, and 4 orange chips, when drawing with replacement. -/
theorem two_different_color_chips_probability :
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let purple_chips : ℕ := 5
  let orange_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_purple : ℚ := purple_chips / total_chips
  let prob_orange : ℚ := orange_chips / total_chips
  let prob_not_green : ℚ := (purple_chips + orange_chips) / total_chips
  let prob_not_purple : ℚ := (green_chips + orange_chips) / total_chips
  let prob_not_orange : ℚ := (green_chips + purple_chips) / total_chips
  (prob_green * prob_not_green + prob_purple * prob_not_purple + prob_orange * prob_not_orange) = 148 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l3854_385402


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3854_385430

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

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3854_385430


namespace NUMINAMATH_CALUDE_log3_20_approximation_l3854_385444

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target value
def target_value : ℝ := 2.7

-- State the theorem
theorem log3_20_approximation :
  let log3_20 := (1 + log10_2_approx) / log10_3_approx
  abs (log3_20 - target_value) < 0.05 := by sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l3854_385444


namespace NUMINAMATH_CALUDE_function_composition_difference_l3854_385439

/-- Given functions f and g, prove that f(g(x)) - g(f(x)) = 5/2 for all x. -/
theorem function_composition_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x - 3
  let g : ℝ → ℝ := λ x ↦ x / 2 + 1
  f (g x) - g (f x) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_difference_l3854_385439


namespace NUMINAMATH_CALUDE_smiths_b_students_l3854_385494

theorem smiths_b_students (jacobs_total : ℕ) (jacobs_b : ℕ) (smiths_total : ℕ) :
  jacobs_total = 20 →
  jacobs_b = 8 →
  smiths_total = 30 →
  ∃ (smiths_b : ℕ), 
    (smiths_b : ℚ) / smiths_total = (jacobs_b : ℚ) / jacobs_total ∧
    smiths_b = 12 := by
  sorry

#check smiths_b_students

end NUMINAMATH_CALUDE_smiths_b_students_l3854_385494


namespace NUMINAMATH_CALUDE_fraction_equality_l3854_385497

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3854_385497


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3854_385453

theorem sum_of_a_and_b (a b : ℝ) : (a - 2)^2 + |b + 4| = 0 → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3854_385453


namespace NUMINAMATH_CALUDE_dance_lesson_cost_l3854_385492

/-- The cost of each dance lesson given the total number of lessons, 
    number of free lessons, and total cost paid. -/
theorem dance_lesson_cost 
  (total_lessons : ℕ) 
  (free_lessons : ℕ) 
  (total_cost : ℚ) : 
  total_lessons = 10 → 
  free_lessons = 2 → 
  total_cost = 80 → 
  (total_cost / (total_lessons - free_lessons : ℚ)) = 10 := by
sorry

end NUMINAMATH_CALUDE_dance_lesson_cost_l3854_385492


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_T_l3854_385473

def K' : ℚ := 1/1 + 1/2 + 1/3 + 1/4 + 1/5

def T (n : ℕ) : ℚ := n * (5^(n-1)) * K'

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_T :
  ∀ n : ℕ, n > 0 → (is_integer (T n) ↔ n ≥ 24) ∧
  ∀ m : ℕ, m < 24 → ¬ is_integer (T m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_T_l3854_385473


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3854_385474

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) : 
  (y = 4*x - 2) ∧ (y = -3*x + 9) ∧ (y = 2*x + k) → k = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3854_385474


namespace NUMINAMATH_CALUDE_base_nine_calculation_l3854_385477

/-- Represents a number in base 9 --/
def BaseNine : Type := Nat

/-- Addition operation for base 9 numbers --/
def add_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Multiplication operation for base 9 numbers --/
def mul_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Converts a natural number to its base 9 representation --/
def to_base_nine : Nat → BaseNine
| n => sorry

theorem base_nine_calculation :
  let a : BaseNine := to_base_nine 35
  let b : BaseNine := to_base_nine 273
  let c : BaseNine := to_base_nine 2
  let result : BaseNine := to_base_nine 620
  mul_base_nine (add_base_nine a b) c = result := by sorry

end NUMINAMATH_CALUDE_base_nine_calculation_l3854_385477


namespace NUMINAMATH_CALUDE_f_of_3_equals_8_l3854_385435

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 1) + 2

-- State the theorem
theorem f_of_3_equals_8 : f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_f_of_3_equals_8_l3854_385435


namespace NUMINAMATH_CALUDE_power_equation_solution_l3854_385428

theorem power_equation_solution :
  ∃ (x : ℕ), (2^(2*21) + 2^(2*21) + 2^(2*21) + 2^(2*21) = x^22) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3854_385428


namespace NUMINAMATH_CALUDE_weather_prediction_probabilities_l3854_385442

-- Define the number of days and the accuracy of prediction
def num_days : ℕ := 3
def accuracy : ℝ := 0.8

-- Define the probability of at least 2 days being accurately predicted
def prob_at_least_2_accurate : ℝ :=
  (Nat.choose num_days 2) * (accuracy ^ 2) * (1 - accuracy) +
  (Nat.choose num_days 3) * (accuracy ^ 3)

-- Define the probability of at least one instance of 2 consecutive days being accurately predicted
def prob_at_least_1_consecutive : ℝ :=
  2 * (accuracy ^ 2) * (1 - accuracy) + accuracy ^ 3

-- State the theorem
theorem weather_prediction_probabilities :
  (prob_at_least_2_accurate = 0.896) ∧
  (prob_at_least_1_consecutive = 0.768) := by
  sorry

end NUMINAMATH_CALUDE_weather_prediction_probabilities_l3854_385442


namespace NUMINAMATH_CALUDE_playground_area_l3854_385407

theorem playground_area (perimeter width length : ℝ) (h1 : perimeter = 80) 
  (h2 : length = 3 * width) (h3 : perimeter = 2 * (length + width)) : 
  length * width = 300 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l3854_385407


namespace NUMINAMATH_CALUDE_sum_remainder_l3854_385466

theorem sum_remainder (m : ℤ) : (10 - 3*m + (5*m + 6)) % 8 = (2*m) % 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3854_385466


namespace NUMINAMATH_CALUDE_max_value_under_constraints_l3854_385449

/-- Given real numbers x and y satisfying the given conditions, 
    the maximum value of 2x - y is 8. -/
theorem max_value_under_constraints (x y : ℝ) 
  (h1 : x + y - 7 ≤ 0) 
  (h2 : x - 3*y + 1 ≤ 0) 
  (h3 : 3*x - y - 5 ≥ 0) : 
  (∀ a b : ℝ, a + b - 7 ≤ 0 → a - 3*b + 1 ≤ 0 → 3*a - b - 5 ≥ 0 → 2*a - b ≤ 2*x - y) ∧ 
  2*x - y = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_under_constraints_l3854_385449


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3854_385410

-- Define the length of the train in meters
def train_length : ℝ := 310

-- Define the length of the platform in meters
def platform_length : ℝ := 210

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 26

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3854_385410


namespace NUMINAMATH_CALUDE_soccer_league_games_l3854_385429

theorem soccer_league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l3854_385429


namespace NUMINAMATH_CALUDE_optimal_purchase_l3854_385480

def budget : ℕ := 100
def basic_calc_cost : ℕ := 8
def battery_cost : ℕ := 2
def scientific_calc_cost : ℕ := 2 * basic_calc_cost
def graphing_calc_cost : ℕ := 3 * scientific_calc_cost

def total_basic_cost : ℕ := basic_calc_cost + battery_cost
def total_scientific_cost : ℕ := scientific_calc_cost + battery_cost
def total_graphing_cost : ℕ := graphing_calc_cost + battery_cost

def one_of_each_cost : ℕ := total_basic_cost + total_scientific_cost + total_graphing_cost

theorem optimal_purchase :
  ∀ (b s g : ℕ),
    b ≥ 1 → s ≥ 1 → g ≥ 1 →
    (b + s + g) % 3 = 0 →
    b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost ≤ budget →
    b + s + g ≤ 3 ∧
    budget - (b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost) ≤ budget - one_of_each_cost :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_l3854_385480


namespace NUMINAMATH_CALUDE_tangent_slope_at_half_l3854_385404

-- Define the function f(x) = x^3 - 2
def f (x : ℝ) : ℝ := x^3 - 2

-- State the theorem
theorem tangent_slope_at_half :
  (deriv f) (1/2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_half_l3854_385404


namespace NUMINAMATH_CALUDE_pass_through_walls_technique_l3854_385457

theorem pass_through_walls_technique (n : ℕ) :
  10 * Real.sqrt (10 / n) = Real.sqrt (10 * (10 / n)) ↔ n = 99 :=
sorry

end NUMINAMATH_CALUDE_pass_through_walls_technique_l3854_385457


namespace NUMINAMATH_CALUDE_stratified_sample_size_l3854_385496

/-- Represents the ratio of students in three schools -/
structure SchoolRatio where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total sample size given the number of students sampled from the smallest school -/
def totalSampleSize (ratio : SchoolRatio) (smallestSchoolSample : ℕ) : ℕ :=
  smallestSchoolSample * (ratio.a + ratio.b + ratio.c) / ratio.a

/-- Theorem: For schools with ratio 2:3:5, if 10 students are sampled from the smallest school, the total sample is 50 -/
theorem stratified_sample_size (ratio : SchoolRatio) (h1 : ratio.a = 2) (h2 : ratio.b = 3) (h3 : ratio.c = 5) :
  totalSampleSize ratio 10 = 50 := by
  sorry

#eval totalSampleSize ⟨2, 3, 5⟩ 10

end NUMINAMATH_CALUDE_stratified_sample_size_l3854_385496


namespace NUMINAMATH_CALUDE_town_x_employment_l3854_385476

structure TownPopulation where
  total_employed : Real
  employed_20_35 : Real
  employed_36_50 : Real
  employed_51_65 : Real
  employed_males : Real
  males_high_school : Real
  males_college : Real
  males_postgrad : Real

def employed_females (pop : TownPopulation) : Real :=
  pop.total_employed - pop.employed_males

theorem town_x_employment (pop : TownPopulation)
  (h1 : pop.total_employed = 0.96)
  (h2 : pop.employed_20_35 = 0.40 * pop.total_employed)
  (h3 : pop.employed_36_50 = 0.50 * pop.total_employed)
  (h4 : pop.employed_51_65 = 0.10 * pop.total_employed)
  (h5 : pop.employed_males = 0.24)
  (h6 : pop.males_high_school = 0.45 * pop.employed_males)
  (h7 : pop.males_college = 0.35 * pop.employed_males)
  (h8 : pop.males_postgrad = 0.20 * pop.employed_males) :
  let females := employed_females pop
  ∃ (f_20_35 f_36_50 f_51_65 f_high_school f_college f_postgrad : Real),
    f_20_35 = 0.288 ∧
    f_36_50 = 0.36 ∧
    f_51_65 = 0.072 ∧
    f_high_school = 0.324 ∧
    f_college = 0.252 ∧
    f_postgrad = 0.144 ∧
    f_20_35 = 0.40 * females ∧
    f_36_50 = 0.50 * females ∧
    f_51_65 = 0.10 * females ∧
    f_high_school = 0.45 * females ∧
    f_college = 0.35 * females ∧
    f_postgrad = 0.20 * females :=
by sorry

end NUMINAMATH_CALUDE_town_x_employment_l3854_385476


namespace NUMINAMATH_CALUDE_factorization_problem1_l3854_385486

theorem factorization_problem1 (a b : ℝ) :
  4 * a^2 + 12 * a * b + 9 * b^2 = (2*a + 3*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3854_385486


namespace NUMINAMATH_CALUDE_g_values_l3854_385406

/-- The real-valued function f -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 4)

/-- The complex-valued function g -/
def g (x y : ℝ) : ℂ := (f (2 * x + 3) : ℂ) + Complex.I * y

/-- Theorem stating the values of g(29,k) for k = 1, 2, 3 -/
theorem g_values : ∀ k ∈ ({1, 2, 3} : Set ℕ), g 29 k = (858 : ℂ) + k * Complex.I :=
sorry

end NUMINAMATH_CALUDE_g_values_l3854_385406


namespace NUMINAMATH_CALUDE_team_ate_96_point_5_slices_l3854_385422

/-- The total number of pizza slices initially bought -/
def total_slices : ℝ := 116

/-- The number of pizza slices left after eating -/
def slices_left : ℝ := 19.5

/-- The number of pizza slices eaten by the team -/
def slices_eaten : ℝ := total_slices - slices_left

theorem team_ate_96_point_5_slices : slices_eaten = 96.5 := by
  sorry

end NUMINAMATH_CALUDE_team_ate_96_point_5_slices_l3854_385422


namespace NUMINAMATH_CALUDE_energetic_time_proof_l3854_385403

def initial_speed : ℝ := 25
def tired_speed : ℝ := 15
def rest_time : ℝ := 0.5
def total_distance : ℝ := 132
def total_time : ℝ := 8

theorem energetic_time_proof :
  ∃ x : ℝ, 
    x ≥ 0 ∧
    x ≤ total_time - rest_time ∧
    initial_speed * x + tired_speed * (total_time - rest_time - x) = total_distance ∧
    x = 39 / 20 := by
  sorry

end NUMINAMATH_CALUDE_energetic_time_proof_l3854_385403


namespace NUMINAMATH_CALUDE_equation_solution_l3854_385472

theorem equation_solution (x : ℝ) : 
  (∀ z : ℝ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3854_385472


namespace NUMINAMATH_CALUDE_system_solution_l3854_385431

theorem system_solution (a : ℝ) (h : a ≠ 0) :
  ∃! (x : ℝ), 3 * x + 2 * x = 15 * a ∧ (1 / a) * x + x = 9 → x = 6 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3854_385431


namespace NUMINAMATH_CALUDE_one_not_in_set_l3854_385423

theorem one_not_in_set : 1 ∉ {x : ℝ | ∃ a : ℕ+, x = -a^2 + 1} := by sorry

end NUMINAMATH_CALUDE_one_not_in_set_l3854_385423


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_diff_l3854_385468

theorem abs_sum_zero_implies_diff (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_diff_l3854_385468
