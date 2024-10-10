import Mathlib

namespace seats_per_bus_is_60_field_trip_problem_l374_37474

/-- Represents the field trip scenario -/
structure FieldTrip where
  total_students : ℕ
  num_buses : ℕ
  all_accommodated : Bool

/-- Calculates the number of seats per bus -/
def seats_per_bus (trip : FieldTrip) : ℕ :=
  trip.total_students / trip.num_buses

/-- Theorem stating that the number of seats per bus is 60 -/
theorem seats_per_bus_is_60 (trip : FieldTrip) 
  (h1 : trip.total_students = 180)
  (h2 : trip.num_buses = 3)
  (h3 : trip.all_accommodated = true) : 
  seats_per_bus trip = 60 := by
  sorry

/-- Main theorem proving the field trip problem -/
theorem field_trip_problem : 
  ∃ (trip : FieldTrip), seats_per_bus trip = 60 ∧ 
    trip.total_students = 180 ∧ 
    trip.num_buses = 3 ∧ 
    trip.all_accommodated = true := by
  sorry

end seats_per_bus_is_60_field_trip_problem_l374_37474


namespace cross_country_winning_scores_l374_37428

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  runners_per_team : ℕ
  total_runners : ℕ
  min_score : ℕ
  max_winning_score : ℕ

/-- The number of different possible winning scores in a cross country meet -/
def winning_scores (meet : CrossCountryMeet) : ℕ :=
  meet.max_winning_score - meet.min_score + 1

/-- Theorem stating the number of different possible winning scores in the specific meet conditions -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.runners_per_team = 6 ∧
    meet.total_runners = 12 ∧
    meet.min_score = 21 ∧
    meet.max_winning_score = 38 ∧
    winning_scores meet = 18 :=
  sorry

end cross_country_winning_scores_l374_37428


namespace diagonal_length_is_13_l374_37458

/-- Represents an isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  AB : ℝ  -- Length of the longer parallel side
  CD : ℝ  -- Length of the shorter parallel side
  AD : ℝ  -- Length of a leg (equal to BC in an isosceles trapezoid)

/-- The diagonal length of the isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ := 
  sorry

/-- Theorem stating that for the given trapezoid dimensions, the diagonal length is 13 -/
theorem diagonal_length_is_13 :
  let t : IsoscelesTrapezoid := { AB := 24, CD := 10, AD := 13 }
  diagonal_length t = 13 := by
  sorry

end diagonal_length_is_13_l374_37458


namespace isosceles_triangle_count_l374_37423

-- Define the geoboard as a square grid
structure Geoboard :=
  (size : ℕ)

-- Define a point on the geoboard
structure Point :=
  (x : ℕ)
  (y : ℕ)

-- Define a triangle on the geoboard
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∨
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 ∨
  (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2

-- Theorem statement
theorem isosceles_triangle_count (g : Geoboard) (A B : Point) 
  (h1 : A.y = B.y) -- A and B are on the same horizontal line
  (h2 : B.x - A.x = 3) -- Distance between A and B is 3 units
  (h3 : A.x > 0 ∧ A.y > 0 ∧ B.x < g.size ∧ B.y < g.size) -- A and B are within the grid
  : ∃ (S : Finset Point), 
    (∀ C ∈ S, C ≠ A ∧ C ≠ B ∧ C.x > 0 ∧ C.y > 0 ∧ C.x ≤ g.size ∧ C.y ≤ g.size) ∧ 
    (∀ C ∈ S, isIsosceles ⟨A, B, C⟩) ∧
    S.card = 3 :=
sorry

end isosceles_triangle_count_l374_37423


namespace proposition_3_proposition_4_l374_37427

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)
variable (linePerpendicularToPlane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ᵖ " => planeParallel
local infix:50 " ⊥ᵖ " => planePerpendicular
local infix:50 " ∥ᵖˡ " => lineParallelToPlane
local infix:50 " ⊥ᵖˡ " => linePerpendicularToPlane

-- Theorem statements
theorem proposition_3 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ∥ᵖˡ β → α ∥ᵖ β → m ⊥ n := by sorry

theorem proposition_4 (m n : Line) (α β : Plane) :
  m ⊥ᵖˡ α → n ⊥ᵖˡ β → α ⊥ᵖ β → m ⊥ n := by sorry

end proposition_3_proposition_4_l374_37427


namespace cos_n_eq_sin_312_l374_37482

theorem cos_n_eq_sin_312 :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ (Real.cos (n * π / 180) = Real.sin (312 * π / 180)) ∧ n = 42 := by
  sorry

end cos_n_eq_sin_312_l374_37482


namespace largest_prime_less_than_5000_l374_37494

def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ d : Nat, d > 1 → d < p → ¬(p % d = 0)

def is_of_form (p : Nat) : Prop :=
  ∃ (a n : Nat), a > 0 ∧ n > 1 ∧ p = a^n - 1

theorem largest_prime_less_than_5000 :
  ∀ p : Nat, p < 5000 → is_prime p → is_of_form p →
  p ≤ 127 :=
sorry

end largest_prime_less_than_5000_l374_37494


namespace square_root_of_four_l374_37464

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l374_37464


namespace line_contains_point_l374_37416

/-- Proves that k = 11 for the line 3 - 3kx = 4y containing the point (1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (3 - 3 * k * (1/3) = 4 * (-2)) → k = 11 := by
  sorry

end line_contains_point_l374_37416


namespace road_signs_total_l374_37442

/-- The total number of road signs at six intersections -/
def total_road_signs (first second third fourth fifth sixth : ℕ) : ℕ :=
  first + second + third + fourth + fifth + sixth

/-- Theorem stating the total number of road signs at six intersections -/
theorem road_signs_total : ∃ (first second third fourth fifth sixth : ℕ),
  (first = 50) ∧
  (second = first + first / 5) ∧
  (third = 2 * second - 10) ∧
  (fourth = ((first + second) + 1) / 2) ∧
  (fifth = third - second) ∧
  (sixth = first + fourth - 15) ∧
  (total_road_signs first second third fourth fifth sixth = 415) :=
by sorry

end road_signs_total_l374_37442


namespace sum_of_reciprocals_lower_bound_l374_37480

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end sum_of_reciprocals_lower_bound_l374_37480


namespace inverse_function_domain_l374_37486

-- Define the function f(x) = -x(x+2)
def f (x : ℝ) : ℝ := -x * (x + 2)

-- State the theorem
theorem inverse_function_domain :
  {y : ℝ | ∃ x ≥ 0, f x = y} = Set.Iic 0 := by sorry

end inverse_function_domain_l374_37486


namespace smallest_composite_with_large_factors_l374_37410

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_with_large_factors :
  ∃ n : ℕ, is_composite n ∧
           has_no_small_prime_factors n ∧
           (∀ m, m < n → ¬(is_composite m ∧ has_no_small_prime_factors m)) ∧
           n = 529 ∧
           520 < n ∧ n ≤ 530 :=
sorry

end smallest_composite_with_large_factors_l374_37410


namespace balanced_digraph_has_valid_coloring_l374_37478

/-- A directed graph where each vertex has in-degree 2 and out-degree 2 -/
structure BalancedDigraph (V : Type) :=
  (edge : V → V → Prop)
  (in_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge u v ∧ edge w v) ∧ 
                        (∀ x y z, edge x v → edge y v → edge z v → (x = y ∨ x = z ∨ y = z)))
  (out_degree_two : ∀ v, (∃ u w, u ≠ w ∧ edge v u ∧ edge v w) ∧ 
                         (∀ x y z, edge v x → edge v y → edge v z → (x = y ∨ x = z ∨ y = z)))

/-- A valid coloring of edges in a balanced digraph -/
def ValidColoring (V : Type) (G : BalancedDigraph V) (color : V → V → Bool) : Prop :=
  ∀ v, (∃! u, G.edge v u ∧ color v u = true) ∧
       (∃! u, G.edge v u ∧ color v u = false) ∧
       (∃! u, G.edge u v ∧ color u v = true) ∧
       (∃! u, G.edge u v ∧ color u v = false)

/-- The main theorem: every balanced digraph has a valid coloring -/
theorem balanced_digraph_has_valid_coloring (V : Type) (G : BalancedDigraph V) :
  ∃ color : V → V → Bool, ValidColoring V G color := by
  sorry

end balanced_digraph_has_valid_coloring_l374_37478


namespace min_value_expression_l374_37421

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100) ≥ -2500 := by
  sorry

end min_value_expression_l374_37421


namespace first_player_wins_petya_wins_1000x2020_l374_37402

/-- Represents the state of the rectangular grid game -/
structure GameState where
  m : ℕ+
  n : ℕ+

/-- Determines if a player has a winning strategy based on the game state -/
def has_winning_strategy (state : GameState) : Prop :=
  ∃ (a b : ℕ), state.m = 2^a * (2 * state.m.val + 1) ∧ state.n = 2^b * (2 * state.n.val + 1) ∧ a ≠ b

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (initial_state : GameState) :
  has_winning_strategy initial_state ↔ 
  ∀ (strategy : GameState → GameState), 
    ∃ (counter_strategy : GameState → GameState),
      ∀ (game_length : ℕ),
        (game_length > 0 ∧ has_winning_strategy (counter_strategy (strategy initial_state))) ∨
        (game_length = 0 ∧ ¬ has_winning_strategy initial_state) :=
by sorry

/-- The specific case for the 1000 × 2020 grid -/
theorem petya_wins_1000x2020 :
  has_winning_strategy { m := 1000, n := 2020 } :=
by sorry

end first_player_wins_petya_wins_1000x2020_l374_37402


namespace road_length_l374_37493

/-- Given 10 trees planted on one side of a road at intervals of 10 meters,
    with trees at both ends, prove that the length of the road is 90 meters. -/
theorem road_length (num_trees : ℕ) (interval : ℕ) : 
  num_trees = 10 → interval = 10 → (num_trees - 1) * interval = 90 := by
  sorry

end road_length_l374_37493


namespace child_share_proof_l374_37498

theorem child_share_proof (total_money : ℕ) (ratio : List ℕ) : 
  total_money = 4500 →
  ratio = [2, 4, 5, 4] →
  (ratio[0]! + ratio[1]!) * total_money / ratio.sum = 1800 := by
  sorry

end child_share_proof_l374_37498


namespace height_comparison_l374_37459

theorem height_comparison (p q : ℝ) (h : p = 0.6 * q) :
  (q - p) / p = 2 / 3 := by
  sorry

end height_comparison_l374_37459


namespace store_visitor_count_l374_37441

/-- The number of people who entered the store in the first hour -/
def first_hour_entry : ℕ := 94

/-- The number of people who left the store in the first hour -/
def first_hour_exit : ℕ := 27

/-- The number of people who left the store in the second hour -/
def second_hour_exit : ℕ := 9

/-- The number of people remaining in the store after 2 hours -/
def remaining_after_two_hours : ℕ := 76

/-- The number of people who entered the store in the second hour -/
def second_hour_entry : ℕ := 18

theorem store_visitor_count :
  (first_hour_entry - first_hour_exit) + second_hour_entry - second_hour_exit = remaining_after_two_hours :=
by sorry

end store_visitor_count_l374_37441


namespace wheel_radius_increase_l374_37487

/-- Calculates the increase in wheel radius given the original and new odometer readings,
    and the original wheel radius. --/
theorem wheel_radius_increase (original_reading : ℝ) (new_reading : ℝ) (original_radius : ℝ) 
  (h1 : original_reading = 390)
  (h2 : new_reading = 380)
  (h3 : original_radius = 12)
  (h4 : original_reading > new_reading) :
  ∃ (increase : ℝ), 
    0.265 < increase ∧ increase < 0.275 ∧ 
    (2 * Real.pi * (original_radius + increase) * new_reading = 
     2 * Real.pi * original_radius * original_reading) :=
by sorry

end wheel_radius_increase_l374_37487


namespace vector_magnitude_l374_37461

/-- Given two vectors a and b in ℝ², where a is parallel to (a - b),
    prove that the magnitude of (a + b) is 3√5/2. -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end vector_magnitude_l374_37461


namespace modified_rectangle_remaining_length_l374_37434

/-- The total length of remaining segments after modifying a rectangle --/
def remaining_length (height width top_right_removed middle_left_removed bottom_removed top_left_removed : ℕ) : ℕ :=
  (height - middle_left_removed) + 
  (width - bottom_removed) + 
  (width - top_right_removed) + 
  (height - top_left_removed)

/-- Theorem stating the total length of remaining segments in the modified rectangle --/
theorem modified_rectangle_remaining_length :
  remaining_length 10 7 2 2 3 1 = 26 := by
  sorry

end modified_rectangle_remaining_length_l374_37434


namespace typist_salary_problem_l374_37415

theorem typist_salary_problem (original_salary : ℝ) : 
  let increased_salary := original_salary * 1.1
  let final_salary := increased_salary * 0.95
  final_salary = 5225 →
  original_salary = 5000 := by
sorry

end typist_salary_problem_l374_37415


namespace equation_solutions_l374_37417

/-- The equation from the original problem -/
def original_equation (x : ℝ) : Prop :=
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, original_equation x ↔ (x = 6 ∨ x = 8) :=
by sorry

end equation_solutions_l374_37417


namespace inequality_proof_l374_37485

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a + b + c = 2 * Real.sqrt (a * b * c)) : 
  b * c ≥ b + c := by
sorry

end inequality_proof_l374_37485


namespace stating_probability_reroll_two_dice_l374_37467

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The sum we're aiming for -/
def targetSum : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := 216

/-- 
Represents the optimal strategy for rerolling dice to achieve the target sum.
d1, d2, d3 are the values of the three dice.
Returns the number of dice to reroll (0, 1, 2, or 3).
-/
def optimalReroll (d1 d2 d3 : Die) : Fin 4 :=
  sorry

/-- 
The number of outcomes where rerolling exactly two dice is optimal.
-/
def twoRerollOutcomes : ℕ := 84

/-- 
Theorem stating that the probability of choosing to reroll exactly two dice
to optimize the chances of getting a sum of 9 is 7/18.
-/
theorem probability_reroll_two_dice :
  (twoRerollOutcomes : ℚ) / totalOutcomes = 7 / 18 := by
  sorry

end stating_probability_reroll_two_dice_l374_37467


namespace closest_point_and_area_l374_37401

def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

def line_l (x y : ℝ) : Prop := y = -x - 2

def point_P : ℝ × ℝ := (-2, 1)

def focus_C : ℝ × ℝ := (0, 1)

def is_centroid (G A B C : ℝ × ℝ) : Prop :=
  G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3

theorem closest_point_and_area :
  ∀ (A B : ℝ × ℝ),
    parabola_C point_P.1 point_P.2 →
    (∀ (Q : ℝ × ℝ), parabola_C Q.1 Q.2 →
      ∃ (d_P d_Q : ℝ),
        d_P = abs (point_P.2 + point_P.1 + 2) / Real.sqrt 2 ∧
        d_Q = abs (Q.2 + Q.1 + 2) / Real.sqrt 2 ∧
        d_P ≤ d_Q) →
    parabola_C A.1 A.2 →
    parabola_C B.1 B.2 →
    is_centroid focus_C point_P A B →
    ∃ (area : ℝ), area = (3 * Real.sqrt 3) / 2 := by sorry

end closest_point_and_area_l374_37401


namespace min_value_of_a_l374_37456

theorem min_value_of_a (a x y : ℤ) : 
  x ≠ y →
  x - y^2 = a →
  y - x^2 = a →
  |x| ≤ 10 →
  (∀ b : ℤ, (∃ x' y' : ℤ, x' ≠ y' ∧ x' - y'^2 = b ∧ y' - x'^2 = b ∧ |x'| ≤ 10) → b ≥ a) →
  a = -111 :=
by sorry

end min_value_of_a_l374_37456


namespace complex_modulus_product_l374_37496

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end complex_modulus_product_l374_37496


namespace max_sum_sqrt_inequality_l374_37407

theorem max_sum_sqrt_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 := by
  sorry

end max_sum_sqrt_inequality_l374_37407


namespace smallest_absolute_value_of_z_l374_37497

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22 ∧ 
             Complex.abs w ≤ Complex.abs z ∧
             Complex.abs w = 45/11 :=
by sorry

end smallest_absolute_value_of_z_l374_37497


namespace problem_statement_l374_37479

-- Define sets A, B, and C
def A : Set ℝ := {x | |3*x - 4| > 2}
def B : Set ℝ := {x | x^2 - x - 2 > 0}
def C (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 1) ≥ 0}

-- Define predicates p, q, and r
def p : Set ℝ := {x | 2/3 ≤ x ∧ x ≤ 2}
def q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def r (a : ℝ) : Set ℝ := {x | x ≤ a ∨ x ≥ a + 1}

theorem problem_statement :
  (∀ x : ℝ, x ∈ p → x ∈ q) ∧
  (∃ x : ℝ, x ∈ q ∧ x ∉ p) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ p → x ∈ r a) ∧ (∃ x : ℝ, x ∈ r a ∧ x ∉ p) ↔ a ≥ 2 ∨ a ≤ -1/3) :=
by sorry

end problem_statement_l374_37479


namespace two_digit_number_formation_l374_37408

theorem two_digit_number_formation (k : ℕ) 
  (h1 : k > 0)
  (h2 : k ≤ 9)
  (h3 : ∀ (S T : ℕ), S = 11 * T * (k - 1) → S / T = 22) :
  k = 3 := by
sorry

end two_digit_number_formation_l374_37408


namespace twenty_five_percent_problem_l374_37444

theorem twenty_five_percent_problem (x : ℚ) : x + (1/4) * x = 80 - (1/4) * 80 ↔ x = 48 := by
  sorry

end twenty_five_percent_problem_l374_37444


namespace sequence_ratio_l374_37481

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ -4 - b = r ∧ a - (-1) = r

-- Define the geometric sequence
def geometric_sequence (c d e : ℝ) : Prop :=
  ∃ q : ℝ, c = -1 * q ∧ d = c * q ∧ e = d * q ∧ -4 = e * q

-- State the theorem
theorem sequence_ratio (a b c d e : ℝ) 
  (h1 : arithmetic_sequence a b)
  (h2 : geometric_sequence c d e) :
  (b - a) / d = 1/2 := by sorry

end sequence_ratio_l374_37481


namespace fence_cost_for_square_plot_l374_37453

theorem fence_cost_for_square_plot (area : ℝ) (price_per_foot : ℝ) :
  area = 289 →
  price_per_foot = 58 →
  (4 * Real.sqrt area) * price_per_foot = 3944 := by
  sorry

end fence_cost_for_square_plot_l374_37453


namespace decimal_sum_equals_fraction_l374_37465

theorem decimal_sum_equals_fraction : 
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end decimal_sum_equals_fraction_l374_37465


namespace simplify_expression_l374_37426

theorem simplify_expression (a b : ℝ) : (2 * a^2)^3 * (-a * b) = -8 * a^7 * b := by
  sorry

end simplify_expression_l374_37426


namespace nested_square_root_value_l374_37438

theorem nested_square_root_value :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end nested_square_root_value_l374_37438


namespace average_after_17th_inning_l374_37476

def batsman_average (previous_innings : ℕ) (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / (previous_innings + 1)

theorem average_after_17th_inning 
  (previous_innings : ℕ) 
  (previous_total : ℕ) 
  (new_score : ℕ) 
  (average_increase : ℚ) :
  previous_innings = 16 →
  new_score = 88 →
  average_increase = 3 →
  batsman_average previous_innings previous_total new_score - 
    (previous_total / previous_innings) = average_increase →
  batsman_average previous_innings previous_total new_score = 40 :=
by
  sorry

#check average_after_17th_inning

end average_after_17th_inning_l374_37476


namespace john_walked_four_miles_l374_37495

/-- Represents the distance John traveled in miles -/
structure JohnTravel where
  initial_skate : ℝ
  total_skate : ℝ
  walk : ℝ

/-- The conditions of John's travel -/
def travel_conditions (j : JohnTravel) : Prop :=
  j.initial_skate = 10 ∧ 
  j.total_skate = 24 ∧
  j.total_skate = 2 * j.initial_skate + j.walk

/-- Theorem stating that John walked 4 miles to the park -/
theorem john_walked_four_miles (j : JohnTravel) 
  (h : travel_conditions j) : j.walk = 4 := by
  sorry

end john_walked_four_miles_l374_37495


namespace complex_fraction_simplification_l374_37451

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -1 + 2*I
  z₁ / z₂ = -5/3 + 10/3*I :=
by sorry

end complex_fraction_simplification_l374_37451


namespace black_raisins_amount_l374_37460

/-- The amount of yellow raisins added (in cups) -/
def yellow_raisins : ℝ := 0.3

/-- The total amount of raisins added (in cups) -/
def total_raisins : ℝ := 0.7

/-- The amount of black raisins added (in cups) -/
def black_raisins : ℝ := total_raisins - yellow_raisins

theorem black_raisins_amount : black_raisins = 0.4 := by
  sorry

end black_raisins_amount_l374_37460


namespace isle_of_misfortune_l374_37483

/-- Represents a person who is either a knight (truth-teller) or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the group -/
def total_people : Nat := 101

/-- A function that returns true if removing a person results in a majority of liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) (person : Person) : Prop :=
  match person with
  | Person.Knight => liars ≥ knights - 1
  | Person.Liar => liars - 1 ≥ knights

theorem isle_of_misfortune :
  ∀ (knights liars : Nat),
    knights + liars = total_people →
    (∀ (p : Person), majority_liars_if_removed knights liars p) →
    knights = 50 ∧ liars = 51 := by
  sorry

end isle_of_misfortune_l374_37483


namespace min_p_plus_q_l374_37446

theorem min_p_plus_q (p q : ℕ+) (h : 98 * p = q^3) : 
  ∀ (p' q' : ℕ+), 98 * p' = q'^3 → p + q ≤ p' + q' :=
by
  sorry

#check min_p_plus_q

end min_p_plus_q_l374_37446


namespace polyhedron_sum_l374_37473

structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  V : ℕ
  faces_sum : faces = triangles + pentagons
  faces_20 : faces = 20
  triangles_twice_pentagons : triangles = 2 * pentagons
  euler : V - ((3 * triangles + 5 * pentagons) / 2) + faces = 2

def vertex_sum (poly : Polyhedron) : ℕ := 100 * poly.P + 10 * poly.T + poly.V

theorem polyhedron_sum (poly : Polyhedron) (h1 : poly.T = 2) (h2 : poly.P = 2) : 
  vertex_sum poly = 238 := by
  sorry

end polyhedron_sum_l374_37473


namespace kenzo_round_tables_l374_37457

/-- The number of round tables Kenzo initially had -/
def num_round_tables : ℕ := 20

/-- The number of office chairs Kenzo initially had -/
def initial_chairs : ℕ := 80

/-- The number of legs each office chair has -/
def legs_per_chair : ℕ := 5

/-- The number of legs each round table has -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that were damaged and disposed of -/
def damaged_chair_percentage : ℚ := 40 / 100

/-- The total number of remaining legs of furniture -/
def total_remaining_legs : ℕ := 300

theorem kenzo_round_tables :
  num_round_tables * legs_per_table = 
    total_remaining_legs - 
    (initial_chairs * (1 - damaged_chair_percentage) : ℚ).num * legs_per_chair :=
by sorry

end kenzo_round_tables_l374_37457


namespace max_sum_given_sum_of_squares_and_product_l374_37433

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 18 → x + y ≤ Real.sqrt 166 := by
  sorry

end max_sum_given_sum_of_squares_and_product_l374_37433


namespace house_transaction_loss_l374_37439

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

def first_transaction (value : ℝ) (loss : ℝ) : ℝ :=
  value * (1 - loss)

def second_transaction (value : ℝ) (gain : ℝ) : ℝ :=
  value * (1 + gain)

theorem house_transaction_loss :
  second_transaction (first_transaction initial_value loss_percentage) gain_percentage - initial_value = 240 := by
  sorry

end house_transaction_loss_l374_37439


namespace geometric_sequence_problem_l374_37411

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 6 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
sorry

end geometric_sequence_problem_l374_37411


namespace tan_negative_3900_degrees_l374_37452

theorem tan_negative_3900_degrees : Real.tan ((-3900 : ℝ) * π / 180) = Real.sqrt 3 := by sorry

end tan_negative_3900_degrees_l374_37452


namespace floor_product_equality_l374_37443

theorem floor_product_equality (Y : ℝ) : ⌊(0.3242 * Y)⌋ = 0.3242 * Y := by
  sorry

end floor_product_equality_l374_37443


namespace david_trip_expenses_l374_37472

theorem david_trip_expenses (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 1800)
  (h2 : remaining_amount = 500)
  (h3 : initial_amount > remaining_amount) :
  initial_amount - remaining_amount - remaining_amount = 800 := by
  sorry

end david_trip_expenses_l374_37472


namespace tangent_sum_l374_37450

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is tangent to y = -x + 8 at (5, f(5))
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = -5 + 8 ∧ deriv f 5 = -1

-- State the theorem
theorem tangent_sum (f : ℝ → ℝ) (h : is_tangent_at_5 f) :
  f 5 + deriv f 5 = 2 := by
  sorry

end tangent_sum_l374_37450


namespace mixture_temperature_swap_l374_37437

theorem mixture_temperature_swap (a b c : ℝ) :
  let x := a + b - c
  ∃ (m_a m_b : ℝ), m_a > 0 ∧ m_b > 0 ∧
    (m_a * (a - c) + m_b * (b - c) = 0) ∧
    (m_b * (a - x) + m_a * (b - x) = 0) :=
by sorry

end mixture_temperature_swap_l374_37437


namespace inequality_proof_l374_37491

theorem inequality_proof (x y : ℝ) : 5 * x^2 + y^2 + 4 - 4 * x - 4 * x * y ≥ 0 := by
  sorry

end inequality_proof_l374_37491


namespace min_total_cost_l374_37455

-- Define the probabilities and costs
def prob_event : ℝ := 0.3
def loss : ℝ := 400 -- in ten thousand yuan
def cost_A : ℝ := 45 -- in ten thousand yuan
def cost_B : ℝ := 30 -- in ten thousand yuan
def prob_no_event_A : ℝ := 0.9
def prob_no_event_B : ℝ := 0.85

-- Define the total cost function for each scenario
def total_cost_none : ℝ := prob_event * loss
def total_cost_A : ℝ := cost_A + (1 - prob_no_event_A) * loss
def total_cost_B : ℝ := cost_B + (1 - prob_no_event_B) * loss
def total_cost_both : ℝ := cost_A + cost_B + (1 - prob_no_event_A * prob_no_event_B) * loss

-- Theorem: Implementing measure A results in the minimum total cost
theorem min_total_cost :
  total_cost_A = 85 ∧ 
  total_cost_A ≤ total_cost_none ∧ 
  total_cost_A ≤ total_cost_B ∧ 
  total_cost_A ≤ total_cost_both :=
sorry

end min_total_cost_l374_37455


namespace max_sum_of_squares_l374_37492

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 105 →
  a * d + b * c = 225 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 150 :=
by sorry

end max_sum_of_squares_l374_37492


namespace band_sections_fraction_l374_37425

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℚ) 
  (h1 : trumpet_fraction = 1/2)
  (h2 : trombone_fraction = 1/8) :
  trumpet_fraction + trombone_fraction = 5/8 := by
  sorry

end band_sections_fraction_l374_37425


namespace cos_equality_solutions_l374_37422

theorem cos_equality_solutions (n : ℤ) : 
  0 ≤ n ∧ n ≤ 360 ∧ Real.cos (n * π / 180) = Real.cos (340 * π / 180) → n = 20 ∨ n = 340 := by
  sorry

end cos_equality_solutions_l374_37422


namespace p_sufficient_not_necessary_l374_37405

/-- p is the condition a^2 + a ≠ 0 -/
def p (a : ℝ) : Prop := a^2 + a ≠ 0

/-- q is the condition a ≠ 0 -/
def q (a : ℝ) : Prop := a ≠ 0

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary : 
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by
  sorry

end p_sufficient_not_necessary_l374_37405


namespace product_def_l374_37471

theorem product_def (a b c d e f : ℝ) : 
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 500 →
  (a * f) / (c * d) = 1 →
  d * e * f = 250 := by
sorry

end product_def_l374_37471


namespace score_for_91_correct_out_of_100_l374_37435

/-- Calculates the score for a test based on the number of correct responses and total questions. -/
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℤ :=
  correctResponses - 2 * (totalQuestions - correctResponses)

/-- Proves that for a 100-question test with 91 correct responses, the calculated score is 73. -/
theorem score_for_91_correct_out_of_100 :
  calculateScore 100 91 = 73 := by
  sorry

end score_for_91_correct_out_of_100_l374_37435


namespace solution_range_l374_37466

theorem solution_range (x : ℝ) :
  (5 * x - 8 > 12 - 2 * x) ∧ (|x - 1| ≤ 3) → (20 / 7 < x ∧ x ≤ 4) :=
by sorry

end solution_range_l374_37466


namespace recipe_flour_amount_l374_37462

/-- Represents the recipe and Mary's baking progress -/
structure Recipe :=
  (total_sugar : ℕ)
  (flour_added : ℕ)
  (sugar_added : ℕ)
  (sugar_to_add : ℕ)

/-- The amount of flour required is independent of the amount of sugar -/
axiom flour_independent_of_sugar (r : Recipe) : 
  r.flour_added = r.flour_added

/-- Theorem: The recipe calls for 10 cups of flour -/
theorem recipe_flour_amount (r : Recipe) 
  (h1 : r.total_sugar = 14)
  (h2 : r.flour_added = 10)
  (h3 : r.sugar_added = 2)
  (h4 : r.sugar_to_add = 12) :
  r.flour_added = 10 := by
  sorry

end recipe_flour_amount_l374_37462


namespace system_solution_set_l374_37424

theorem system_solution_set (x : ℝ) : 
  (x - 1 < 1 ∧ x + 3 > 0) ↔ (-3 < x ∧ x < 2) := by
  sorry

end system_solution_set_l374_37424


namespace female_democrats_count_l374_37420

theorem female_democrats_count (total : ℕ) (male female : ℕ) (male_democrats female_democrats : ℕ) :
  total = 870 →
  male + female = total →
  female_democrats = female / 2 →
  male_democrats = male / 4 →
  female_democrats + male_democrats = total / 3 →
  female_democrats = 145 := by
  sorry

end female_democrats_count_l374_37420


namespace spherical_coordinate_shift_l374_37429

/-- Given a point with rectangular coordinates (3, -2, 5) and spherical coordinates (r, α, β),
    prove that the point with spherical coordinates (r, α+π, β) has rectangular coordinates (-3, 2, 5). -/
theorem spherical_coordinate_shift (r α β : ℝ) : 
  (3 = r * Real.sin β * Real.cos α) → 
  (-2 = r * Real.sin β * Real.sin α) → 
  (5 = r * Real.cos β) → 
  ((-3, 2, 5) : ℝ × ℝ × ℝ) = (
    r * Real.sin β * Real.cos (α + Real.pi),
    r * Real.sin β * Real.sin (α + Real.pi),
    r * Real.cos β
  ) := by sorry

end spherical_coordinate_shift_l374_37429


namespace fraction_product_squared_main_theorem_l374_37419

theorem fraction_product_squared (a b c d : ℚ) : 
  (a / b) ^ 2 * (c / d) ^ 2 = (a * c / (b * d)) ^ 2 :=
by sorry

theorem main_theorem : (6 / 7) ^ 2 * (1 / 2) ^ 2 = 9 / 49 :=
by sorry

end fraction_product_squared_main_theorem_l374_37419


namespace ellipse_eccentricity_range_l374_37489

/-- The eccentricity of an ellipse with given conditions is between 0 and √2/2 -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (a^2 - b^2) / a
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
sorry

end ellipse_eccentricity_range_l374_37489


namespace jane_crayons_l374_37447

/-- The number of crayons Jane ends up with after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: If Jane starts with 87 crayons and 7 are eaten by a hippopotamus,
    she will end up with 80 crayons. -/
theorem jane_crayons : crayons_left 87 7 = 80 := by
  sorry

end jane_crayons_l374_37447


namespace original_ratio_of_boarders_to_day_students_l374_37431

theorem original_ratio_of_boarders_to_day_students 
  (initial_boarders : ℕ) 
  (new_boarders : ℕ) 
  (final_ratio_boarders : ℕ) 
  (final_ratio_day_students : ℕ) : 
  initial_boarders = 150 →
  new_boarders = 30 →
  final_ratio_boarders = 1 →
  final_ratio_day_students = 2 →
  ∃ (original_ratio_boarders original_ratio_day_students : ℕ),
    original_ratio_boarders = 5 ∧ 
    original_ratio_day_students = 12 ∧
    (initial_boarders : ℚ) / (initial_boarders + new_boarders : ℚ) * final_ratio_day_students = 
      (original_ratio_boarders : ℚ) / (original_ratio_boarders + original_ratio_day_students : ℚ) :=
by sorry


end original_ratio_of_boarders_to_day_students_l374_37431


namespace probability_of_B_is_one_fourth_l374_37418

/-- The probability of choosing a specific letter from a bag of letters -/
def probability_of_letter (total_letters : ℕ) (target_letters : ℕ) : ℚ :=
  target_letters / total_letters

/-- The bag contains 8 letters in total -/
def total_letters : ℕ := 8

/-- The bag contains 2 B's -/
def number_of_Bs : ℕ := 2

/-- The probability of choosing a B is 1/4 -/
theorem probability_of_B_is_one_fourth :
  probability_of_letter total_letters number_of_Bs = 1 / 4 := by
  sorry

end probability_of_B_is_one_fourth_l374_37418


namespace lisa_pizza_meat_distribution_l374_37409

/-- The number of pieces of meat on each slice of Lisa's pizza --/
def pieces_per_slice : ℕ :=
  let pepperoni : ℕ := 30
  let ham : ℕ := 2 * pepperoni
  let sausage : ℕ := pepperoni + 12
  let total_meat : ℕ := pepperoni + ham + sausage
  let num_slices : ℕ := 6
  total_meat / num_slices

theorem lisa_pizza_meat_distribution :
  pieces_per_slice = 22 := by
  sorry

end lisa_pizza_meat_distribution_l374_37409


namespace owls_on_fence_l374_37475

/-- The number of owls that joined the fence -/
def owls_joined (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem owls_on_fence (initial : ℕ) (final : ℕ) 
  (h_initial : initial = 3) 
  (h_final : final = 5) : 
  owls_joined initial final = 2 := by
  sorry

end owls_on_fence_l374_37475


namespace series_calculation_l374_37403

def series_sum (n : ℕ) : ℤ :=
  (n + 1) * 3

theorem series_calculation : series_sum 32 = 1584 := by
  sorry

#eval series_sum 32

end series_calculation_l374_37403


namespace five_digit_permutations_l374_37490

/-- The number of permutations of a multiset with repeated elements -/
def multiset_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (repetitions.map Nat.factorial).prod

/-- The number of different five-digit integers formed using 1, 1, 1, 8, and 8 -/
theorem five_digit_permutations : multiset_permutations 5 [3, 2] = 10 := by
  sorry

end five_digit_permutations_l374_37490


namespace chord_length_is_four_l374_37499

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = x + 2

def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the chord length
def chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ :=
  4 -- The actual calculation is omitted, we just state the result

-- Theorem statement
theorem chord_length_is_four :
  chord_length C1 C2 = 4 := by sorry

end chord_length_is_four_l374_37499


namespace congruence_solution_l374_37469

theorem congruence_solution (x : ℤ) : 
  x ∈ Finset.Icc 20 50 ∧ (6 * x + 5) % 10 = 19 % 10 ↔ 
  x ∈ ({24, 29, 34, 39, 44, 49} : Finset ℤ) := by
sorry

end congruence_solution_l374_37469


namespace geometric_sequence_ratio_l374_37406

/-- Given a geometric sequence with common ratio -1/3, 
    prove that the sum of odd-indexed terms up to a₇ 
    divided by the sum of even-indexed terms up to a₈ equals -3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (-1/3)) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 := by
sorry


end geometric_sequence_ratio_l374_37406


namespace problem_solution_l374_37430

theorem problem_solution (x y : ℝ) :
  (4 * x + y = 1) →
  (y = 1 - 4 * x) ∧
  (y ≥ 0 → x ≤ 1/4) ∧
  (-1 < y ∧ y ≤ 2 → -1/4 ≤ x ∧ x < 1/2) :=
by sorry

end problem_solution_l374_37430


namespace exponent_calculations_l374_37488

theorem exponent_calculations :
  (16 ^ (1/2 : ℝ) + (1/81 : ℝ) ^ (-1/4 : ℝ) - (-1/2 : ℝ) ^ (0 : ℝ) = 10/3) ∧
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
    ((2 * a ^ (1/4 : ℝ) * b ^ (-1/3 : ℝ)) * (-3 * a ^ (-1/2 : ℝ) * b ^ (2/3 : ℝ))) / 
    (-1/4 * a ^ (-1/4 : ℝ) * b ^ (-2/3 : ℝ)) = 24 * b) := by
  sorry

end exponent_calculations_l374_37488


namespace video_streaming_cost_l374_37414

/-- Represents the monthly cost of a video streaming subscription -/
def monthly_cost : ℝ := 14

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the total cost paid by one person for a year -/
def total_cost_per_person : ℝ := 84

theorem video_streaming_cost : 
  monthly_cost * months_in_year = 2 * total_cost_per_person :=
by sorry

end video_streaming_cost_l374_37414


namespace t_in_possible_values_l374_37404

/-- The set of possible values for t given the conditions -/
def possible_t_values : Set ℝ :=
  {t | 3 < t ∧ t < 4}

/-- The point (1, t) is above the line 2x - y + 1 = 0 -/
def above_line (t : ℝ) : Prop :=
  2 * 1 - t + 1 < 0

/-- The inequality x^2 + (2t-4)x + 4 > 0 always holds -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ x, x^2 + (2*t-4)*x + 4 > 0

/-- Given the conditions, prove that t is in the set of possible values -/
theorem t_in_possible_values (t : ℝ) 
  (h1 : above_line t) 
  (h2 : inequality_holds t) : 
  t ∈ possible_t_values :=
sorry

end t_in_possible_values_l374_37404


namespace honey_servings_per_ounce_l374_37436

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := sorry

/-- Represents the number of servings used per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea consumed per night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of nights the honey lasts -/
def nights_lasted : ℕ := 48

theorem honey_servings_per_ounce :
  servings_per_ounce = 6 :=
sorry

end honey_servings_per_ounce_l374_37436


namespace x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l374_37468

theorem x_gt_5_sufficient_not_necessary_for_x_sq_gt_25 :
  (∀ x : ℝ, x > 5 → x^2 > 25) ∧
  (∃ x : ℝ, x^2 > 25 ∧ x ≤ 5) := by
  sorry

end x_gt_5_sufficient_not_necessary_for_x_sq_gt_25_l374_37468


namespace line_l_equation_line_l_prime_equation_l374_37448

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the point of symmetry
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem line_l_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧ l₂ x₀ y₀ ∧ x - 2 * y + m = 0) ∧
    (∀ (x₁ y₁ : ℝ), perp_line x₁ y₁ → (x - x₁) * 2 + (y - y₁) * 1 = 0) →
    x - 2 * y + 5 = 0 :=
sorry

-- Theorem for the equation of line l'
theorem line_l_prime_equation :
  ∀ (x y : ℝ),
    (∃ (x₀ y₀ : ℝ), l₁ x₀ y₀ ∧ 
      x₀ = 2 * sym_point.1 - x ∧
      y₀ = 2 * sym_point.2 - y) →
    3 * x + 4 * y + 7 = 0 :=
sorry

end line_l_equation_line_l_prime_equation_l374_37448


namespace cube_root_cube_identity_l374_37484

theorem cube_root_cube_identity (x : ℝ) : (x^3)^(1/3) = x := by
  sorry

end cube_root_cube_identity_l374_37484


namespace cape_may_less_than_double_daytona_main_result_l374_37445

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 40 - cape_may_sightings

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := 40

/-- Cape May has some less than double the number of shark sightings of Daytona Beach -/
theorem cape_may_less_than_double_daytona : cape_may_sightings < 2 * daytona_beach_sightings :=
sorry

/-- The difference between double the number of shark sightings in Daytona Beach and Cape May -/
def sightings_difference : ℕ := 2 * daytona_beach_sightings - cape_may_sightings

theorem main_result : sightings_difference = 8 := by
  sorry

end cape_may_less_than_double_daytona_main_result_l374_37445


namespace first_player_wins_l374_37470

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- Checks if a position is valid on the table -/
def isValidPosition (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Represents a move in the game -/
def makeMove (state : GameState) (pos : Position) : GameState :=
  { state with
    occupied := state.occupied ∪ {pos},
    currentPlayer := ¬state.currentPlayer
  }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ pos, pos ∈ state.table → pos ∈ state.occupied

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∀ (initialState : GameState),
  initialState.currentPlayer = true →
  ∃ (strategy : GameState → Position),
  (∀ state, isValidPosition state (strategy state)) →
  (∀ state, ¬isGameOver state → 
    ∃ (opponentMove : Position),
    isValidPosition state opponentMove →
    isGameOver (makeMove (makeMove state (strategy state)) opponentMove)) :=
sorry

end first_player_wins_l374_37470


namespace factor_expression_l374_37477

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 10*(x+2) + 2*(x+2) = (x+2)*(4*x+12) := by
  sorry

end factor_expression_l374_37477


namespace brent_candy_count_l374_37400

/-- Calculates the total number of candy pieces Brent has left after trick-or-treating and giving some to his sister. -/
def total_candy_left (kitkat : ℕ) (nerds : ℕ) (initial_lollipops : ℕ) (baby_ruth : ℕ) (given_lollipops : ℕ) : ℕ :=
  let hershey := 3 * kitkat
  let reese := baby_ruth / 2
  let remaining_lollipops := initial_lollipops - given_lollipops
  kitkat + hershey + nerds + baby_ruth + reese + remaining_lollipops

theorem brent_candy_count : 
  total_candy_left 5 8 11 10 5 = 49 := by
  sorry

end brent_candy_count_l374_37400


namespace bridge_length_calculation_l374_37413

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 240 := by
  sorry

end bridge_length_calculation_l374_37413


namespace ceiling_sqrt_244_l374_37463

theorem ceiling_sqrt_244 : ⌈Real.sqrt 244⌉ = 16 := by
  sorry

end ceiling_sqrt_244_l374_37463


namespace f_properties_l374_37412

noncomputable section

def f (x : ℝ) : ℝ := (2*x - x^2) * Real.exp x

theorem f_properties :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x ∧ max_x = Real.sqrt 2) ∧
  (¬ ∃ min_x, ∀ x, f min_x ≤ f x) ∧
  (∃ max_x, ∀ x, f x ≤ f max_x) :=
sorry

end

end f_properties_l374_37412


namespace exam_score_proof_l374_37432

/-- Given an examination with the following conditions:
  * There are 60 questions in total
  * Each correct answer scores 4 marks
  * Each wrong answer loses 1 mark
  * The total score is 130 marks
  This theorem proves that the number of correctly answered questions is 38. -/
theorem exam_score_proof (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers = 38 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry

end exam_score_proof_l374_37432


namespace unique_cds_l374_37449

theorem unique_cds (shared : ℕ) (alice_total : ℕ) (bob_unique : ℕ) 
  (h1 : shared = 12)
  (h2 : alice_total = 23)
  (h3 : bob_unique = 8) :
  alice_total - shared + bob_unique = 19 :=
by sorry

end unique_cds_l374_37449


namespace max_sum_on_circle_l374_37454

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 169) : x + y ≤ 17 :=
sorry

end max_sum_on_circle_l374_37454


namespace approximate_profit_percent_l374_37440

-- Define the selling price and cost price
def selling_price : Float := 2552.36
def cost_price : Float := 2400.0

-- Define the profit amount
def profit_amount : Float := selling_price - cost_price

-- Define the profit percent
def profit_percent : Float := (profit_amount / cost_price) * 100

-- Theorem to prove the approximate profit percent
theorem approximate_profit_percent :
  (Float.round (profit_percent * 100) / 100) = 6.35 := by
  sorry

end approximate_profit_percent_l374_37440
