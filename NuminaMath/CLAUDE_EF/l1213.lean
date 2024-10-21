import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cool_graphs_less_than_total_graphs_l1213_121356

/-- A graph G with n vertices is "cool" if we can label each vertex with a different positive integer 
    not greater than n²/4, and there exists a set D of non-negative integers such that there is an edge 
    between two vertices if and only if the difference between their labels is in D. -/
def is_cool_graph {n : ℕ} (G : SimpleGraph (Fin n)) : Prop :=
  ∃ (labeling : Fin n → ℕ+) (D : Set ℕ), 
    (∀ i j, i ≠ j → labeling i ≠ labeling j) ∧ 
    (∀ i, (labeling i : ℕ) ≤ n^2 / 4) ∧
    (∀ i j, G.Adj i j ↔ (((labeling i : ℕ) - (labeling j : ℕ)) ∈ D ∨ ((labeling j : ℕ) - (labeling i : ℕ)) ∈ D))

/-- The number of "cool" graphs with n vertices -/
noncomputable def num_cool_graphs (n : ℕ) : ℕ := sorry

/-- The total number of graphs with n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n * (n - 1) / 2)

theorem cool_graphs_less_than_total_graphs :
  ∃ N, ∀ n ≥ N, num_cool_graphs n < total_graphs n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cool_graphs_less_than_total_graphs_l1213_121356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_paths_count_l1213_121352

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move in the Cartesian plane -/
inductive Move
  | Right
  | Up
  | Diagonal

/-- A path is a list of moves -/
def PathMoves := List Move

/-- Checks if a path is valid according to the problem rules -/
def isValidPath (p : PathMoves) : Bool :=
  sorry

/-- Checks if a path ends at the target point (4,4) -/
def endsAtTarget (p : PathMoves) : Bool :=
  sorry

/-- Counts the number of unique valid paths from (0,0) to (4,4) -/
def countValidPaths : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 27 unique valid paths -/
theorem unique_paths_count : countValidPaths = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_paths_count_l1213_121352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_implies_bound_l1213_121361

/-- π(n) is the number of prime numbers less than or equal to n -/
def prime_counting_function (n : ℕ) : ℕ := (Finset.filter Nat.Prime (Finset.range (n + 1))).card

/-- Given a list of integers, returns true if one of them divides the product of the others -/
def one_divides_product_of_others (list : List ℕ) : Prop :=
  ∃ (i : ℕ), i < list.length ∧ (list.getD i 1 ∣ (list.removeNth i).prod)

theorem divisibility_property_implies_bound {n k : ℕ} (a : List ℕ) 
  (h_length : a.length = k)
  (h_sorted : a.Sorted (· < ·))
  (h_range : ∀ i, i ∈ a → 1 < i ∧ i ≤ n)
  (h_div : one_divides_product_of_others a) :
  k ≤ prime_counting_function n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_implies_bound_l1213_121361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_n_l1213_121380

theorem remainder_of_n (n : ℕ) (h : ∃ k : ℕ, (47 : ℚ) / 5 * (4 / 47 + n / 141) = k) : n % 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_n_l1213_121380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slip_25_in_cup_b_l1213_121335

noncomputable def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

def cups : List Char := ['A', 'B', 'C', 'D', 'E']

noncomputable def sum_slips : ℝ := slips.sum

noncomputable def avg_sum : ℝ := sum_slips / 5

def cup_sums : List ℝ := [11, 10, 9, 8, 7]

theorem slip_25_in_cup_b (h1 : ∃ (s : ℝ) (rest : List ℝ), s = 4 ∧ s::rest ⊆ slips ∧ (s::rest).sum = 11)
                         (h2 : ∃ (s : ℝ) (rest : List ℝ), s = 5 ∧ s::rest ⊆ slips ∧ (s::rest).sum = 8)
                         (h3 : ∀ i : Fin 5, cup_sums[i] = avg_sum - i.val / 10)
                         (h4 : cup_sums.sum = sum_slips) :
  ∃ (rest : List ℝ), 2.5::rest ⊆ slips ∧ (2.5::rest).sum = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slip_25_in_cup_b_l1213_121335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_panes_count_l1213_121347

/-- Represents a rectangular glass pane -/
structure GlassPane where
  length : ℕ
  width : ℕ

/-- Represents a window made up of glass panes -/
structure Window where
  pane : GlassPane
  totalArea : ℕ

/-- Calculates the number of panes in a window -/
def numberOfPanes (w : Window) : ℕ :=
  w.totalArea / (w.pane.length * w.pane.width)

theorem window_panes_count (w : Window) 
  (h1 : w.pane.length = 12)
  (h2 : w.pane.width = 8)
  (h3 : w.totalArea = 768) : 
  numberOfPanes w = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_panes_count_l1213_121347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l1213_121334

noncomputable section

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define eccentricity for a hyperbola
def eccentricity_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

theorem ellipse_and_hyperbola_equations :
  (∃ a b : ℝ, ellipse a b (-5) 0 ∧ ellipse a b 0 3 ∧ a = 5 ∧ b = 3) ∧
  (∃ a : ℝ, hyperbola a a (-5) 3 ∧ eccentricity_hyperbola a a = Real.sqrt 2 ∧ a = 4) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_hyperbola_equations_l1213_121334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l1213_121309

theorem cricket_average_increase (initial_average : ℝ) (initial_innings : ℕ) (next_innings_runs : ℕ) :
  initial_average = 30 ∧ 
  initial_innings = 10 ∧ 
  next_innings_runs = 74 →
  (initial_average * (initial_innings : ℝ) + (next_innings_runs : ℝ)) / ((initial_innings + 1) : ℝ) - initial_average = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l1213_121309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_for_given_scenario_l1213_121397

/-- Represents the manufacturing scenario for electronic components -/
structure ComponentManufacturing where
  productionCost : ℚ
  shippingCost : ℚ
  fixedMonthlyCost : ℚ
  monthlyProduction : ℕ

/-- Calculates the lowest price per component to cover all costs -/
def lowestPricePerComponent (cm : ComponentManufacturing) : ℚ :=
  cm.productionCost + cm.shippingCost + (cm.fixedMonthlyCost / cm.monthlyProduction)

/-- Theorem stating the lowest price per component for the given scenario -/
theorem lowest_price_for_given_scenario :
  let cm : ComponentManufacturing := {
    productionCost := 80,
    shippingCost := 6,
    fixedMonthlyCost := 16500,
    monthlyProduction := 150
  }
  lowestPricePerComponent cm = 196 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_price_for_given_scenario_l1213_121397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_is_descendant_of_all_l1213_121341

def successor (x : ℕ) : ℕ :=
  if x % 10 = 0 then x / 10
  else if x % 10 = 4 then x / 10
  else 2 * x

def is_descendant (n m : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate successor k) n = m

theorem four_is_descendant_of_all : ∀ n : ℕ, is_descendant n 4 := by
  sorry

#check four_is_descendant_of_all

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_is_descendant_of_all_l1213_121341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_zero_vector_parallel_l1213_121377

-- Define a vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define a point type
variable {P : Type*}

-- Define a function to create a vector from two points
variable (vector : P → P → V)

-- Define parallelism for vectors
def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w ∨ w = c • v

-- Theorem 1: For any two points A and B, vector AB + vector BA = 0
theorem vector_sum_zero (A B : P) : vector A B + vector B A = 0 := by sorry

-- Theorem 2: The zero vector is parallel to any vector
theorem zero_vector_parallel (v : V) : parallel 0 v := by
  use 0
  left
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_zero_zero_vector_parallel_l1213_121377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_min_distances_l1213_121308

/-- Two ships K₁ and K₂ with initial distance d, where K₂ sails at speed v perpendicular to K₁K₂ -/
structure ShipScenario where
  d : ℝ  -- Initial distance between ships
  v : ℝ  -- Speed of K₂
  u : ℝ  -- Speed of K₁ in part b
  h_positive_d : 0 < d
  h_positive_v : 0 < v
  h_positive_u : 0 < u
  h_u_less_v : u < v

/-- The minimum distance between K₁ and K₂ when K₁'s speed is equal to K₂'s -/
noncomputable def min_distance_equal_speed (s : ShipScenario) : ℝ := s.d / 2

/-- The minimum distance between K₁ and K₂ when K₁'s speed is less than K₂'s -/
noncomputable def min_distance_unequal_speed (s : ShipScenario) : ℝ := s.d * Real.sqrt (s.v^2 - s.u^2) / s.v

/-- Theorem stating the minimum distances for both scenarios -/
theorem ship_min_distances (s : ShipScenario) :
  (min_distance_equal_speed s = s.d / 2) ∧
  (min_distance_unequal_speed s = s.d * Real.sqrt (s.v^2 - s.u^2) / s.v) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_min_distances_l1213_121308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_minor_arc_line_l1213_121376

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- represents ax + by + c = 0

def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

def Line.passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

def Line.perpendicular_to_center_line (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop :=
  l.a * (c.center.1 - p.1) + l.b * (c.center.2 - p.2) = 0

theorem shortest_minor_arc_line (A : Circle) (M : ℝ × ℝ) (l : Line) :
  A.center = (2, 0) →
  A.radius = 3 →
  M = (1, 2) →
  Line.passes_through l M →
  Line.perpendicular_to_center_line l A M →
  l.a = 1 ∧ l.b = -2 ∧ l.c = 3 := by
  sorry

#check shortest_minor_arc_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_minor_arc_line_l1213_121376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_is_2500_l1213_121314

/-- Represents the profit share calculation for a business partnership -/
noncomputable def profit_share (total_investment : ℝ) (individual_investment : ℝ) (total_profit : ℝ) : ℝ :=
  (individual_investment / total_investment) * total_profit

/-- Theorem: Given the investments and A's profit share, B's profit share is 2500 -/
theorem b_profit_share_is_2500 
  (investment_A investment_B investment_C : ℝ)
  (profit_share_A : ℝ)
  (profit_difference_AC : ℝ)
  (h1 : investment_A = 8000)
  (h2 : investment_B = 10000)
  (h3 : investment_C = 12000)
  (h4 : profit_share_A = 2000)
  (h5 : profit_difference_AC = 999.9999999999998)
  : profit_share (investment_A + investment_B + investment_C) investment_B 
    ((profit_share_A * (investment_A + investment_B + investment_C)) / investment_A) = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_profit_share_is_2500_l1213_121314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_probability_l1213_121337

/-- The probability of selecting 1 blue, 1 green, and 2 red marbles 
    from a bag with 3 red, 2 blue, and 2 green marbles -/
theorem marble_selection_probability : 
  (Nat.choose 2 1 * Nat.choose 2 1 * Nat.choose 3 2) / Nat.choose 7 4 = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_probability_l1213_121337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_line_bisecting_chord_chord_length_45_degrees_l1213_121388

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 0)

-- Theorem 1: Line equation through P and circle center
theorem line_through_center (x y : ℝ) :
  circle_C x y → (2 * x - y - 2 = 0 ↔ ∃ t, x = 1 + t ∧ y = 2*t) :=
sorry

-- Theorem 2: Line equation when chord AB is bisected by P
theorem line_bisecting_chord (x y : ℝ) :
  circle_C x y → (x + 2*y - 6 = 0 ↔ ∃ t, x = 2 - 2*t ∧ y = 2 + t) :=
sorry

-- Theorem 3: Length of chord AB when line has 45° angle
theorem chord_length_45_degrees :
  ∃ (A B : ℝ × ℝ), 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧ 
    (∃ t, A.1 = 2 + t ∧ A.2 = 2 + t) ∧ 
    (∃ t, B.1 = 2 + t ∧ B.2 = 2 + t) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 34) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_center_line_bisecting_chord_chord_length_45_degrees_l1213_121388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorize_quadratic_find_m_value_triangle_perimeter_l1213_121363

-- Part 1
theorem factorize_quadratic (a : ℝ) : a^2 - 6*a + 5 = (a-1)*(a-5) := by sorry

-- Part 2
theorem find_m_value (a b m c : ℝ) 
  (h1 : a^2 + b^2 - 12*a - 6*b + 45 + |1/2 * m - c| = 0)
  (h2 : (2:ℝ)^a * (4:ℝ)^b = (8:ℝ)^m) : 
  m = 4 := by sorry

-- Part 3
theorem triangle_perimeter (a b m c : ℝ) 
  (h1 : a^2 + b^2 - 12*a - 6*b + 45 + |1/2 * m - c| = 0)
  (h2 : ∃ (n : ℕ), c = 2*n + 1) :
  a + b + c = 14 ∨ a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorize_quadratic_find_m_value_triangle_perimeter_l1213_121363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1213_121325

/-- Calculates the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for them to cross each other. -/
noncomputable def trainLength (trainSpeed : ℝ) (personSpeed : ℝ) (crossingTime : ℝ) : ℝ :=
  (trainSpeed + personSpeed) * (1000 / 3600) * crossingTime

/-- Theorem stating that given a train moving at 25 km/h and a person moving at 2 km/h in the opposite direction, if they cross each other in 20 seconds, then the length of the train is 150 meters. -/
theorem train_length_calculation :
  trainLength 25 2 20 = 150 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval trainLength 25 2 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1213_121325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l1213_121318

theorem infinite_geometric_series_sum :
  let a : ℚ := 5/4
  let r : ℚ := 1/3
  let S := (∑' n : ℕ, a * r^n)
  S = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l1213_121318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1213_121383

/-- Given a triangle ABC with point N on AC such that AN = 1/2 AC,
    and point P on BN, prove that if AP = m * AB + 3/8 * AC,
    then m = 1/4. -/
theorem triangle_vector_relation (A B C N P : EuclideanSpace ℝ (Fin 2)) (m : ℝ) : 
  (N - A) = (1/2 : ℝ) • (C - A) →
  ∃ t : ℝ, P = B + t • (N - B) →
  (P - A) = m • (B - A) + (3/8 : ℝ) • (C - A) →
  m = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_relation_l1213_121383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1213_121357

theorem division_problem (x y : ℕ) (hx : x > 0) (hy : y > 0)
  (h1 : x % y = 3) (h2 : (x : ℚ) / y = 96.15) : y = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l1213_121357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_l1213_121374

def Arrangement := Fin 9 → Fin 9

def validArrangement (arr : Arrangement) : Prop :=
  (arr 0 = 8) ∧ (arr 8 = 0) ∧
  (arr 1 > 5 ∨ arr 3 > 5) ∧
  (arr 5 > 3 ∧ arr 7 > 3) ∧
  (arr 1 = 7 ∨ arr 3 = 7)

-- Add instance for Fintype Arrangement
instance : Fintype Arrangement := by
  apply Fintype.ofEquiv (Fin 9 → Fin 9)
  exact Equiv.refl _

-- Add instance for DecidablePred validArrangement
instance : DecidablePred validArrangement := fun arr =>
  decidable_of_iff
    ((arr 0 = 8) ∧ (arr 8 = 0) ∧
     (arr 1 > 5 ∨ arr 3 > 5) ∧
     (arr 5 > 3 ∧ arr 7 > 3) ∧
     (arr 1 = 7 ∨ arr 3 = 7))
    (by simp [validArrangement])

theorem count_valid_arrangements :
  (Finset.filter validArrangement (Finset.univ : Finset Arrangement)).card = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_arrangements_l1213_121374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1213_121378

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 3 * t + 6
noncomputable def y (t : ℝ) : ℝ := 5 * t - 7

-- Define the slope and y-intercept of the line
noncomputable def m : ℝ := 5 / 3
noncomputable def b : ℝ := -17

-- Theorem statement
theorem line_equation :
  ∀ t : ℝ, y t = m * (x t) + b := by
  intro t
  -- Expand the definitions of x, y, m, and b
  simp [x, y, m, b]
  -- Prove the equality
  ring  -- This tactic should handle the algebraic simplification
  -- If 'ring' doesn't work, you can use 'sorry' as a placeholder
  -- sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1213_121378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_arrangements_eq_12_l1213_121358

/-- The number of different four-digit numbers that can be formed by arranging the digits in 2335 -/
def four_digit_arrangements : ℕ :=
  let digits : Multiset ℕ := {2, 3, 3, 5}
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 1)

theorem four_digit_arrangements_eq_12 : four_digit_arrangements = 12 := by
  rfl

#eval four_digit_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_arrangements_eq_12_l1213_121358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1213_121353

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 6/x - Real.log x / Real.log 2

-- State the theorem
theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l1213_121353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_ball_draws_min_draws_for_five_same_results_l1213_121393

theorem pigeonhole_ball_draws (n : ℕ) (h : n ≤ 24) :
  ∃ (f : Fin n → Fin 6), ∀ (i : Fin 6), (Finset.filter (λ x ↦ f x = i) Finset.univ).card < 5 :=
sorry

theorem min_draws_for_five_same_results :
  ∀ (f : ℕ → Fin 6), ∃ (i : Fin 6), (Finset.filter (λ x ↦ f x = i) (Finset.range 25)).card ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pigeonhole_ball_draws_min_draws_for_five_same_results_l1213_121393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1213_121326

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 5) ^ (1/3) + Real.sqrt (9 - x)

-- Define the domain of f
def domain_f : Set ℝ := { x | x ≤ 9 }

-- Theorem statement
theorem domain_of_f : 
  { x : ℝ | ∃ y, f x = y } = domain_f :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1213_121326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1213_121350

-- Define variables
variable (x y : ℂ)
variable (a b : ℝ)

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := (2 * x - 1) + Complex.I = y - (3 - y) * Complex.I
def equation2 (x y : ℂ) (a b : ℝ) : Prop := (2 * x + a * y) - (4 * x - y + b) * Complex.I = 9 - 8 * Complex.I

-- State the theorem
theorem system_solution (h1 : equation1 x y) (h2 : equation2 x y a b) (h3 : x.im = 0) (h4 : y.im = 0) :
  a = 1 ∧ b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1213_121350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1213_121313

theorem largest_lambda : 
  ∃ (lambda_max : ℝ), (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + lambda_max*b*c + c*d) ∧ 
  (∀ (lambda : ℝ), (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ a*b + lambda*b*c + c*d) → lambda ≤ lambda_max) ∧
  lambda_max = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lambda_l1213_121313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1213_121369

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The length of side CD of the trapezoid -/
def side_length (t : Trapezoid) : ℝ := sorry

/-- The distance from the midpoint of side AB to line CD -/
def midpoint_distance (t : Trapezoid) : ℝ := sorry

/-- The area of the trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of a trapezoid is equal to the product of the length of its parallel side
    and the distance from the midpoint of the other parallel side to the first parallel side -/
theorem trapezoid_area (t : Trapezoid) :
  area t = side_length t * midpoint_distance t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l1213_121369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_side_length_l1213_121365

/-- An equilateral triangle with vertices A, B, and C -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : ∀ (X Y : ℝ × ℝ), X ∈ ({A, B, C} : Set (ℝ × ℝ)) → Y ∈ ({A, B, C} : Set (ℝ × ℝ)) → X ≠ Y → 
    Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The distance from a point to a line -/
noncomputable def distance_to_line (P : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := 
  sorry

/-- The theorem stating the largest possible side length of the equilateral triangle -/
theorem largest_side_length (t : EquilateralTriangle) (l : ℝ → ℝ) 
  (h1 : distance_to_line t.A l = 39)
  (h2 : distance_to_line t.B l = 35)
  (h3 : distance_to_line t.C l = 13) :
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) ≤ 58 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_side_length_l1213_121365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_x_is_plus_minus_one_l1213_121396

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- State the theorem
theorem f_equals_one_iff_x_is_plus_minus_one :
  ∀ x : ℝ, f x = 1 ↔ x = 1 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_x_is_plus_minus_one_l1213_121396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_term_l1213_121340

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like_sequence (n + 1) + fibonacci_like_sequence n

theorem existence_of_divisible_term (m : ℕ) (hm : m > 0) :
  ∃ k : ℕ, k > 0 ∧ m ∣ (fibonacci_like_sequence k)^4 - (fibonacci_like_sequence k) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisible_term_l1213_121340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_extension_l1213_121348

/-- Given a quadrilateral WXYZ with specific extensions, prove that W can be expressed as a linear combination of W' and Z' -/
theorem quadrilateral_extension (W X Y Z W' X' Y' Z' : ℝ × ℝ) : 
  (∃ (WXYZ : Set (ℝ × ℝ)), WXYZ = {W, X, Y, Z}) →  -- WXYZ is a quadrilateral
  (W' - Z = 2 • (W - Z)) →  -- W'Z = 2WZ
  (X' - Y = 2 • (X - Y)) →  -- X'Y = 2XY
  (Y' - X = W - X) →        -- Y'X = WX
  (Z' - Z = Y - Z) →        -- Z'Z = YZ
  (W = (1/2 : ℝ) • W' + (1/2 : ℝ) • Z') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_extension_l1213_121348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_value_l1213_121370

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1. -/
def MonicQuarticPolynomial (R : Type*) [CommRing R] :=
  { f : Polynomial R // f.degree = 4 ∧ f.leadingCoeff = 1 }

/-- The theorem statement -/
theorem monic_quartic_polynomial_value (R : Type*) [CommRing R] [CharZero R] 
  (h : MonicQuarticPolynomial R) :
  h.val.eval (-2 : R) = -12 →
  h.val.eval 1 = -3 →
  h.val.eval 3 = -27 →
  h.val.eval 5 = -75 →
  h.val.eval 0 = -30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quartic_polynomial_value_l1213_121370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_generation_form_l1213_121354

/-- Generates the next number in the sequence based on two input numbers -/
noncomputable def generateNext (x y : ℝ) : ℝ := x * y + x + y

/-- Generates the nth number in the sequence -/
noncomputable def generateNth (p q : ℝ) : ℕ → ℝ
  | 0 => max p q
  | 1 => generateNext p q
  | (n+2) => let prev1 := generateNth p q (n+1)
             let prev2 := generateNth p q n
             generateNext prev1 prev2

theorem sixth_generation_form (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > q) :
  generateNth p q 6 = (q + 1)^8 * (p + 1)^13 - 1 := by
  sorry

#check sixth_generation_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_generation_form_l1213_121354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cost_is_18_l1213_121367

/-- Represents the balloon problem with given conditions --/
structure BalloonProblem where
  total_budget : ℚ
  sheet_cost : ℚ
  propane_cost : ℚ
  helium_cost_per_oz : ℚ
  height_per_oz : ℚ
  max_height : ℚ

/-- Calculates the cost of the rope in the balloon problem --/
def rope_cost (problem : BalloonProblem) : ℚ :=
  let helium_oz := problem.max_height / problem.height_per_oz
  let helium_cost := helium_oz * problem.helium_cost_per_oz
  problem.total_budget - (problem.sheet_cost + problem.propane_cost + helium_cost)

/-- Theorem stating that the rope cost is $18 for the given problem --/
theorem rope_cost_is_18 (problem : BalloonProblem) 
    (h1 : problem.total_budget = 200)
    (h2 : problem.sheet_cost = 42)
    (h3 : problem.propane_cost = 14)
    (h4 : problem.helium_cost_per_oz = 3/2)
    (h5 : problem.height_per_oz = 113)
    (h6 : problem.max_height = 9492) :
    rope_cost problem = 18 := by
  sorry

#eval rope_cost { 
  total_budget := 200, 
  sheet_cost := 42, 
  propane_cost := 14, 
  helium_cost_per_oz := 3/2, 
  height_per_oz := 113, 
  max_height := 9492 
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_cost_is_18_l1213_121367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_happy_people_theorem_l1213_121312

structure Institution where
  size : ℕ
  happiness : ℚ
  size_constraint : 100 ≤ size ∧ size ≤ 200
  happiness_constraint : 60 / 100 ≤ happiness ∧ happiness ≤ 95 / 100

def total_happy_people (institutions : List Institution) : ℚ :=
  (institutions.map (λ i => (i.size : ℚ) * i.happiness)).sum

theorem total_happy_people_theorem (institutions : List Institution) :
  institutions.length = 12 →
  total_happy_people institutions =
    (institutions.map (λ i => (i.size : ℚ) * i.happiness)).sum := by
  intro h
  rfl

#check total_happy_people_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_happy_people_theorem_l1213_121312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_one_half_l1213_121371

/-- The area of a triangle with vertices (0, 0), (1, 0), and (0, 1) is 1/2 -/
theorem triangle_area_one_half :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, 1)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_one_half_l1213_121371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equality_l1213_121304

theorem binomial_equality (k : ℕ) : (1 - 3 : ℤ) ^ (4 * k) = (1 - 5 : ℤ) ^ (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equality_l1213_121304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pairs_l1213_121319

/-- The number of distinct ordered pairs of positive integers (a,b) such that 1/a + 1/b = 1/6 -/
noncomputable def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card

/-- Theorem stating that there are exactly 9 such pairs -/
theorem nine_pairs : count_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pairs_l1213_121319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1213_121362

/-- For positive real numbers a, b, c with a > b > c, the sum of the infinite series
    1/(b*c) + 1/(c*(2*c - b)) + 1/((2*c - b)*(3*c - 2*b)) + 1/((3*c - 2*b)*(4*c - 3*b)) + ...
    is equal to 1/((c-b)*c). -/
theorem infinite_series_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a > b) (hbc : b > c) :
  let series := λ (n : ℕ) => 1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b))
  ∑' n, series n = 1 / ((c - b) * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l1213_121362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_parallel_edges_l1213_121368

/-- A rectangular prism is a three-dimensional shape with three pairs of parallel faces -/
structure RectangularPrism where
  faces : Fin 6 → Type
  parallel_pairs : Fin 3 → Fin 2 → Fin 6
  is_parallel : ∀ i : Fin 3, ∀ j k : Fin 2, j ≠ k → 
    faces (parallel_pairs i j) = faces (parallel_pairs i k)

/-- An edge is a line segment where two faces meet -/
def Edge (prism : RectangularPrism) := Type

/-- Two edges are parallel if they are formed by parallel faces -/
def ParallelEdges (prism : RectangularPrism) (e1 e2 : Edge prism) : Prop := sorry

/-- The number of pairs of parallel edges in a rectangular prism -/
def ParallelEdgePairs (prism : RectangularPrism) : ℕ := sorry

/-- Theorem: A rectangular prism has 6 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) : 
  ParallelEdgePairs prism = 6 := by sorry

#check rectangular_prism_parallel_edges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_parallel_edges_l1213_121368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1213_121339

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (6 + x - x^2)

-- State the theorem
theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/2 → f x₁ < f x₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l1213_121339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_normal_line_l1213_121330

-- Define a nondegenerate conic
class NonDegenerateConic (C : Type*) where
  is_nondegenerate : C → Prop

-- Define a point on a conic
def PointOnConic (C : Type*) [NonDegenerateConic C] (p : C) : Prop :=
  true  -- We assume the point is on the conic

-- Define a normal line to a conic at a point
def NormalLine (C : Type*) [NonDegenerateConic C] (c : C) (p : C) : Set C :=
  sorry

-- Define a right angle
def RightAngle (C : Type*) [NonDegenerateConic C] (vertex : C) (a : C) (b : C) : Prop :=
  sorry

-- Define a line passing through two points
def LineThroughPoints (C : Type*) [NonDegenerateConic C] (a : C) (b : C) : Set C :=
  sorry

-- Define a point lying on a line
def PointOnLine (C : Type*) [NonDegenerateConic C] (p : C) (l : Set C) : Prop :=
  p ∈ l

-- The main theorem
theorem fixed_point_on_normal_line (C : Type*) [NonDegenerateConic C] 
  (c : C) (O : C) (hO : PointOnConic C O) :
  ∃ (P : C), P ∈ NormalLine C c O ∧ 
  ∀ (A B : C), RightAngle C O A B → 
  PointOnLine C P (LineThroughPoints C A B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_normal_line_l1213_121330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_sequence_geometric_progression_l1213_121327

def is_generating_sequence (a b : ℕ → ℝ) (n : ℕ) : Prop :=
  b 1 = a n ∧ ∀ k ∈ Finset.range n, k ≥ 2 → b (k - 1) * b k = a (k - 1) * a k ∧ a (k - 1) * a k ≠ 0

theorem generating_sequence_geometric_progression (n : ℕ) (a b c : ℕ → ℝ) :
  Odd n →
  is_generating_sequence a b n →
  is_generating_sequence b c n →
  ∃ r : ℝ, r ≠ 0 ∧ b 1 = r * a 1 ∧ c 1 = r * b 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_sequence_geometric_progression_l1213_121327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1213_121315

-- Define the function as noncomputable due to its dependence on real numbers
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 + 3*x - x^2) + 1 / Real.sqrt (x - 1)

-- State the theorem about the domain of the function
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 1 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1213_121315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_journey_graph_correct_mary_journey_graph_correct_main_l1213_121323

/-- Represents a speed-time graph --/
structure SpeedTimeGraph where
  speeds : List ℝ
  stops : ℕ

/-- Checks if the speeds are in strictly decreasing order --/
def SpeedTimeGraph.decreasingSpeed (g : SpeedTimeGraph) : Prop :=
  g.speeds.length ≥ 3 ∧ g.speeds.Pairwise (· > ·)

/-- Checks if the graph has exactly one intermediate stop --/
def SpeedTimeGraph.oneIntermediateStop (g : SpeedTimeGraph) : Prop :=
  g.stops = 1

/-- Checks if the highest speed is between two lower speeds --/
def SpeedTimeGraph.highSpeedBetweenLow (g : SpeedTimeGraph) : Prop :=
  g.speeds.length ≥ 3 ∧ g.speeds[1]! > g.speeds[0]! ∧ g.speeds[1]! > g.speeds[2]!

/-- A graph correctly represents Mary's journey if and only if it satisfies all conditions --/
theorem mary_journey_graph_correct (g : SpeedTimeGraph) :
  (g.decreasingSpeed ∧ g.oneIntermediateStop ∧ g.highSpeedBetweenLow) ↔
  (g.decreasingSpeed ∧ g.oneIntermediateStop ∧ g.highSpeedBetweenLow) := by
  apply Iff.refl

/-- Helper function to determine if a graph correctly represents Mary's journey --/
def correctlyRepresentsMaryJourney (g : SpeedTimeGraph) : Prop :=
  g.decreasingSpeed ∧ g.oneIntermediateStop ∧ g.highSpeedBetweenLow

/-- The main theorem restated using the helper function --/
theorem mary_journey_graph_correct_main (g : SpeedTimeGraph) :
  correctlyRepresentsMaryJourney g ↔ 
  (g.decreasingSpeed ∧ g.oneIntermediateStop ∧ g.highSpeedBetweenLow) := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_journey_graph_correct_mary_journey_graph_correct_main_l1213_121323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1213_121387

-- Define the squares and their properties
def square_ABCD (side_length : ℝ) : Prop := side_length = Real.sqrt 50

-- Define the condition that each side of EFGH extends to a vertex of ABCD
def side_extends_to_vertex (side : ℝ) (vertex : ℝ × ℝ) : Prop := sorry

def square_EFGH (side_length : ℝ) (be_length : ℝ) : Prop :=
  be_length = 2 ∧ 
  ∀ (side : ℝ), side = side_length → 
    ∃ (vertex : ℝ × ℝ), side_extends_to_vertex side vertex

-- State the theorem
theorem inner_square_area 
  (abcd_side : ℝ) 
  (efgh_side : ℝ) 
  (be_length : ℝ) :
  square_ABCD abcd_side →
  square_EFGH efgh_side be_length →
  efgh_side^2 = 50 - 4 * Real.sqrt 46 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l1213_121387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_three_rays_with_common_endpoint_l1213_121385

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ), 
    ((a = 3 ∧ b = p.1 + 2) ∨ (a = 3 ∧ b = p.2 - 4) ∨ (a = p.1 + 2 ∧ b = p.2 - 4)) ∧
    (∀ c, c ∈ ({3, p.1 + 2, p.2 - 4} : Set ℝ) → c ≤ a)}

-- Define a ray
def Ray (origin : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (origin.1 + t * direction.1, origin.2 + t * direction.2)}

-- Theorem statement
theorem S_is_three_rays_with_common_endpoint :
  ∃ (origin : ℝ × ℝ) (dir1 dir2 dir3 : ℝ × ℝ),
    S = Ray origin dir1 ∪ Ray origin dir2 ∪ Ray origin dir3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_three_rays_with_common_endpoint_l1213_121385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_after_45_days_l1213_121321

/-- Calculates the amount of water in a pool after a given number of days -/
noncomputable def water_after_days (initial_water : ℝ) (evaporation_rate : ℝ) (addition_amount : ℝ) (addition_interval : ℕ) (days : ℕ) : ℝ :=
  initial_water - (evaporation_rate * days) + (addition_amount * (days / addition_interval))

/-- Theorem stating the amount of water in the pool after 45 days -/
theorem pool_water_after_45_days :
  water_after_days 500 0.7 5 3 45 = 543.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_water_after_45_days_l1213_121321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l1213_121390

def distribute_balls (total : ℕ) (num_boxes : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = total ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

def max3 (a b c : ℕ) : ℕ := max (max a b) c

theorem ball_distribution_theorem :
  let total_balls := 11
  let num_boxes := 3
  distribute_balls total_balls num_boxes →
  (∃ (max : ℕ), max ≤ 8 ∧ 
    (∀ (x y z : ℕ), x + y + z = total_balls ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      max ≥ x ∧ max ≥ y ∧ max ≥ z) ∧
    (∃ (a b c : ℕ), a + b + c = total_balls ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      (max = a ∨ max = b ∨ max = c))) ∧
  (∃ (min : ℕ), min ≥ 5 ∧ 
    (∀ (x y z : ℕ), x + y + z = total_balls ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      min ≤ max3 x y z) ∧
    (∃ (a b c : ℕ), a + b + c = total_balls ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
      min = max3 a b c)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_distribution_theorem_l1213_121390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1213_121360

noncomputable section

open Real

/-- A function f(x) with given properties -/
def f (A ω x : ℝ) : ℝ := A * sin (ω * x + π / 3)

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  ∀ (A ω : ℝ), A > 0 → ω > 0 →
  (∀ x, |f A ω x| ≤ 2) →  -- Amplitude is 2
  (∀ x, f A ω (x + π/ω) = f A ω x) →  -- Period is π
  (∃ (B : ℝ), ∀ x, f A ω x = 2 * sin (2*x + π/3)) ∧  -- Part 1
  (∀ (k : ℤ), ∀ x ∈ Set.Icc (-5*π/12 + k*π) (k*π + π/12),
    (∀ y ∈ Set.Icc (-5*π/12 + k*π) x, f A ω y ≤ f A ω x)) ∧  -- Part 2
  (Set.Icc (-2 : ℝ) (sqrt 3) = {y | ∃ x ∈ Set.Icc (-π/2) 0, f A ω x = y}) :=  -- Part 3
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1213_121360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1213_121372

/-- Represents a point in 3D space. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid with vertex S and base ABC. -/
structure TriangularPyramid where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Represents a dihedral angle in a triangular pyramid. -/
inductive DihedralAngle
  | ABCS
  | BACS

/-- The surface area of a sphere. -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Placeholder for perpendicular function -/
def perpendicular (v : Point3D → Point3D → ℝ × ℝ × ℝ) (p : Point3D → Point3D → Point3D → Prop) : Prop := sorry

/-- Placeholder for angle_eq_pi_div_two function -/
def angle_eq_pi_div_two (p1 p2 p3 : Point3D) : Prop := sorry

/-- Placeholder for distance function -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Placeholder for dihedral_angle_eq function -/
def dihedral_angle_eq (da : DihedralAngle) (angle : ℝ) : Prop := sorry

/-- Placeholder for circumscribed_sphere_radius function -/
def circumscribed_sphere_radius (pyramid : TriangularPyramid) : ℝ := sorry

/-- The main theorem about the surface area of the circumscribed sphere of a specific triangular pyramid. -/
theorem circumscribed_sphere_surface_area (pyramid : TriangularPyramid) 
  (h1 : perpendicular (λ p1 p2 => (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)) (λ p1 p2 p3 => True))
  (h2 : angle_eq_pi_div_two pyramid.B pyramid.S pyramid.C)
  (h3 : distance pyramid.S pyramid.C = 1)
  (h4 : dihedral_angle_eq DihedralAngle.ABCS (45 * Real.pi / 180))
  (h5 : dihedral_angle_eq DihedralAngle.BACS (60 * Real.pi / 180)) :
  sphereSurfaceArea (circumscribed_sphere_radius pyramid) = (5 * Real.pi) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1213_121372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_relative_errors_equal_relative_error_value_l1213_121344

-- Define the measurements and errors
noncomputable def measurement1 : ℝ := 25
noncomputable def measurement2 : ℝ := 75
noncomputable def measurement3 : ℝ := 125
noncomputable def error1 : ℝ := 0.05
noncomputable def error2 : ℝ := 0.15
noncomputable def error3 : ℝ := 0.25

-- Define relative error
noncomputable def relativeError (error measurement : ℝ) : ℝ := error / measurement

-- Theorem: All relative errors are equal
theorem all_relative_errors_equal :
  relativeError error1 measurement1 = relativeError error2 measurement2 ∧
  relativeError error2 measurement2 = relativeError error3 measurement3 := by
  sorry

-- Theorem: The relative error for each measurement is 0.002 (0.2%)
theorem relative_error_value :
  relativeError error1 measurement1 = 0.002 ∧
  relativeError error2 measurement2 = 0.002 ∧
  relativeError error3 measurement3 = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_relative_errors_equal_relative_error_value_l1213_121344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_grid_lines_l1213_121342

/-- A point in the 3D grid --/
structure GridPoint where
  x : Fin 5
  y : Fin 5
  z : Fin 5

/-- A line in the 3D grid --/
structure GridLine where
  points : Fin 4 → GridPoint
  distinct : ∀ i j, i ≠ j → points i ≠ points j

/-- The set of all valid grid lines --/
def allGridLines : Set GridLine := sorry

/-- Fintype instance for GridLine --/
instance : Fintype GridLine := sorry

/-- The number of grid lines is 156 --/
theorem count_grid_lines : Fintype.card GridLine = 156 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_grid_lines_l1213_121342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l1213_121392

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the circle M
def circleM (x y : ℝ) : Prop := x^2 + (y+4)^2 = 1

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

-- Define the minimum distance condition
def min_distance (p : ℝ) : Prop := 
  ∃ (x y : ℝ), circleM x y ∧ 
  Real.sqrt ((x - (focus p).1)^2 + (y - (focus p).2)^2) - 1 = 4

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circleM P.1 P.2

-- Define tangent points
def tangent_points (P A B : ℝ × ℝ) (p : ℝ) : Prop :=
  point_on_circle P ∧ parabola p A.1 A.2 ∧ parabola p B.1 B.2

-- Helper function to calculate triangle area (not implemented)
noncomputable def area_triangle (P A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem parabola_circle_theorem (p : ℝ) :
  (∀ x y, parabola p x y) →
  (∀ x y, circleM x y) →
  min_distance p →
  (p = 2) ∧
  (∃ (P A B : ℝ × ℝ), 
    tangent_points P A B p ∧
    (∀ P' A' B' : ℝ × ℝ, 
      tangent_points P' A' B' p →
      area_triangle P A B ≥ area_triangle P' A' B') ∧
    area_triangle P A B = 20 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_theorem_l1213_121392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1213_121384

/-- The time taken for a train to cross an electric pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_hr * (1000 / 3600)
  train_length / train_speed_m_s

/-- Theorem: A 100 m long train traveling at 144 km/hr takes 2.5 seconds to cross an electric pole -/
theorem train_crossing_pole_time :
  train_crossing_time 100 144 = 2.5 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l1213_121384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_inequality_l1213_121375

/-- The function f(x) with parameters a and b -/
noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x - Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 - b - 1 / x

theorem critical_point_implies_inequality (a b : ℝ) (ha : a > 0) :
  f_derivative a b 1 = 0 → Real.log a - b + 1 < 0 := by
  sorry

#check critical_point_implies_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_inequality_l1213_121375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_range_for_sin_l1213_121317

theorem arccos_range_for_sin (a : Real) (x : Real) :
  x = Real.sin a ∧ a ∈ Set.Icc (-Real.pi/4) (3*Real.pi/4) →
  Real.arccos x ∈ Set.Icc 0 (3*Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arccos_range_for_sin_l1213_121317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_max_marks_l1213_121333

/-- Calculates the maximum possible marks given the passing threshold percentage,
    a student's score, and the shortfall to pass. -/
def max_marks_calculation (passing_threshold : ℚ) (student_score : ℕ) (shortfall : ℕ) : ℕ :=
  let passing_score : ℕ := student_score + shortfall
  ⌊(passing_score : ℚ) / passing_threshold⌋.toNat

/-- Proves that given the conditions, the maximum possible marks is 770. -/
theorem prove_max_marks :
  let passing_threshold : ℚ := 3/10
  let student_score : ℕ := 212
  let shortfall : ℕ := 19
  max_marks_calculation passing_threshold student_score shortfall = 770 := by
  -- Unfold the definition and simplify
  unfold max_marks_calculation
  simp
  -- The rest of the proof
  sorry

#eval max_marks_calculation (3/10) 212 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_max_marks_l1213_121333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l1213_121381

def our_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 2) - a n = 2

theorem seventh_term_value (a : ℕ → ℤ) (h : our_sequence a) : a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_value_l1213_121381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_no_direct_solution_l1213_121398

/-- The equation z(z+2i)(z-3i) = 8048i -/
def complex_equation (z : ℂ) : Prop :=
  z * (z + 2*Complex.I) * (z - 3*Complex.I) = 8048 * Complex.I

theorem complex_equation_no_direct_solution :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  ¬∃ (a_value : ℝ), complex_equation (Complex.mk a_value b) ∧ a = a_value := by
  sorry

#check complex_equation_no_direct_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_no_direct_solution_l1213_121398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_decrease_rate_l1213_121301

/-- The annual decrease rate of a population --/
noncomputable def annual_decrease_rate (initial_population : ℝ) (final_population : ℝ) (years : ℝ) : ℝ :=
  1 - (final_population / initial_population) ^ (1 / years)

/-- Theorem: The annual decrease rate is 0.2 given the initial population of 20,000 and the population of 12,800 after 2 years --/
theorem population_decrease_rate :
  let initial_population : ℝ := 20000
  let final_population : ℝ := 12800
  let years : ℝ := 2
  annual_decrease_rate initial_population final_population years = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_decrease_rate_l1213_121301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_37_37_repeating_l1213_121355

/-- Represents a repeating decimal -/
def repeating_decimal (whole : ℕ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 99

/-- Rounds a rational number to the nearest hundredth -/
def round_to_hundredth (q : ℚ) : ℚ :=
  (q * 100).floor / 100 + if (q * 100 - (q * 100).floor ≥ 1/2) then 1/100 else 0

/-- The main theorem: rounding 37.37̄ to the nearest hundredth equals 37.38 -/
theorem round_37_37_repeating :
  round_to_hundredth (repeating_decimal 37 37) = 37 + 38 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_37_37_repeating_l1213_121355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_circumscribed_circles_l1213_121311

/-- A circle in the real plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are externally tangent -/
def ExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

/-- One circle circumscribes another -/
def Circumscribes (c1 c2 : Circle) : Prop :=
  sorry

/-- The area between a large circle and two smaller circles it circumscribes -/
noncomputable def AreaBetween (large small1 small2 : Circle) : ℝ :=
  sorry

/-- Given two externally tangent circles with radii 4 and 5 circumscribed by a third circle,
    the area of the region outside the smaller circles but inside the larger circle is 40π. -/
theorem shaded_area_of_circumscribed_circles : 
  ∀ (c1 c2 c3 : Circle),
    c1.radius = 4 →
    c2.radius = 5 →
    ExternallyTangent c1 c2 →
    Circumscribes c3 c1 →
    Circumscribes c3 c2 →
    AreaBetween c3 c1 c2 = 40 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_of_circumscribed_circles_l1213_121311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1213_121331

noncomputable def point_a : ℝ × ℝ := (1, 3)
noncomputable def point_b : ℝ × ℝ := (7, 3)
noncomputable def point_c : ℝ × ℝ := (-2, -4)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem triangle_perimeter :
  distance point_a point_b + distance point_b point_c + distance point_c point_a
  = 6 + Real.sqrt 58 + Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1213_121331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_concurrency_l1213_121305

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Check if four lines are concurrent -/
def are_concurrent (l1 l2 l3 l4 : Line) : Prop := sorry

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Check if a circle is tangent to a line at a point -/
def is_tangent_at (c : Circle) (l : Line) (p : Point) : Prop := sorry

/-- Main theorem -/
theorem incircle_tangent_concurrency 
  (ABC : Triangle) 
  (incircle : Circle) 
  (M N R S : Point) 
  (t : Line) 
  (h1 : is_tangent_at incircle (Line.mk 0 1 0) M)  -- AC
  (h2 : is_tangent_at incircle (Line.mk 1 0 0) N)  -- BC
  (h3 : is_tangent_at incircle (Line.mk 1 1 0) R)  -- AB
  (h4 : point_on_line S t)
  (h5 : ∃ P : Point, point_on_line P t ∧ point_on_line P (Line.mk 1 0 0))  -- NC
  (h6 : ∃ Q : Point, point_on_line Q t ∧ point_on_line Q (Line.mk 0 1 0))  -- MC
  : ∃ (P Q : Point), are_concurrent 
    (Line.mk 1 0 (-ABC.A.x))  -- AP
    (Line.mk 0 1 (-ABC.B.y))  -- BQ
    (Line.mk (S.y - R.y) (R.x - S.x) (S.x * R.y - R.x * S.y))  -- SR
    (Line.mk (N.y - M.y) (M.x - N.x) (N.x * M.y - M.x * N.y))  -- MN
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_tangent_concurrency_l1213_121305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1213_121306

noncomputable section

-- Define the vectors
def OA : ℝ × ℝ := (2, 1)
def OB (t : ℝ) : ℝ × ℝ := (t, -2)
def OC (t : ℝ) : ℝ × ℝ := (1, 2*t)

-- Define vector operations
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the conditions
def AB_length_is_5 (t : ℝ) : Prop :=
  vector_length (vector_sub (OB t) OA) = 5

def BOC_is_90_degrees (t : ℝ) : Prop :=
  dot_product (OB t) (OC t) = 0

def are_collinear (t : ℝ) : Prop :=
  ∃ (k : ℝ), vector_sub (OC t) OA = k • (vector_sub (OB t) OA)

-- The theorem to prove
theorem vector_problem :
  (∀ t : ℝ, AB_length_is_5 t ↔ t = 6 ∨ t = -2) ∧
  (∀ t : ℝ, BOC_is_90_degrees t ↔ t = 0) ∧
  (∀ t : ℝ, are_collinear t ↔ t = (3 - Real.sqrt 13) / 2 ∨ t = (3 + Real.sqrt 13) / 2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1213_121306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l1213_121310

-- Define the isosceles triangle sketch
noncomputable def isosceles_sketch (leg : ℝ) (base : ℝ) : Prop :=
  leg > 0 ∧ base > 0 ∧ leg = Real.sqrt 6 ∧ base = 4

-- Define the area ratio between original and sketch
noncomputable def area_ratio : ℝ := 2 * Real.sqrt 2

-- Define the area of the isosceles triangle sketch
noncomputable def sketch_area (leg : ℝ) (base : ℝ) : ℝ :=
  (1 / 2) * base * Real.sqrt (leg^2 - (base/2)^2)

-- State the theorem
theorem original_triangle_area 
  (h : isosceles_sketch (Real.sqrt 6) 4) : 
  ∃ (original_area : ℝ), 
    sketch_area (Real.sqrt 6) 4 * area_ratio = original_area ∧
    original_area = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_triangle_area_l1213_121310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1213_121316

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 3*x + 2 else x^2 - 3*x + 2

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧  -- f is even
  (∀ x, x ≥ 0 → f x = x^2 + 3*x + 2) →  -- given condition
  (∀ x, x < 0 → f x = x^2 - 3*x + 2) ∧  -- part 1
  (∀ x, f (2*x - 1) < 20 ↔ -1 < x ∧ x < 2)  -- part 2
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1213_121316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_neg_f_neg_y_l1213_121359

-- Define the function f
noncomputable def f (t : ℝ) : ℝ := t / (1 + t)

-- State the theorem
theorem x_equals_neg_f_neg_y (x y : ℝ) (h : x ≠ -1) :
  y = f x → x = -f (-y) := by
  intro h_y_eq_fx
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_neg_f_neg_y_l1213_121359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_average_daily_wage_l1213_121373

-- Define the parameters
noncomputable def a_days : ℝ := 12
noncomputable def b_days : ℝ := 15
noncomputable def days_worked_together : ℝ := 5
noncomputable def total_payment : ℝ := 810

-- Define the work done by A and B together
noncomputable def work_done_together : ℝ := days_worked_together * (1 / a_days + 1 / b_days)

-- Define B's share of the payment
noncomputable def b_share : ℝ := work_done_together * total_payment

-- Theorem to prove
theorem b_average_daily_wage :
  b_share / days_worked_together = 121.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_average_daily_wage_l1213_121373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l1213_121391

noncomputable def train_length : ℝ := 2500
noncomputable def tree_crossing_time : ℝ := 90
noncomputable def platform_length : ℝ := 1500

noncomputable def train_speed : ℝ := train_length / tree_crossing_time

noncomputable def total_distance : ℝ := train_length + platform_length

noncomputable def time_to_pass_platform : ℝ := total_distance / train_speed

theorem train_platform_passing_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |time_to_pass_platform - 143.88| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l1213_121391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_round_table_l1213_121349

/-- The number of distinct seating arrangements for n people around a round table,
    considering rotations of the same arrangement as equivalent -/
def roundTableArrangements (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else Nat.factorial (n - 1)

/-- Theorem: There are 24 distinct ways for 5 people to sit around a round table -/
theorem five_people_round_table : roundTableArrangements 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_people_round_table_l1213_121349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1213_121324

open Real

noncomputable def f (x : ℝ) := x - sin x

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 0  -- We define a_0 as 0 to make the indexing consistent
  | n + 1 => f (sequence_a n)

theorem problem_solution 
  (h1 : 0 < sequence_a 1) (h2 : sequence_a 1 < 1) :
  (∀ x ∈ Set.Ioo 0 1, StrictMono f) ∧ 
  (∀ n : ℕ, 0 < sequence_a (n + 1) ∧ sequence_a (n + 1) < 1) ∧
  (∀ n : ℕ, sequence_a (n + 2) < (1/6) * (sequence_a (n + 1))^3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1213_121324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformationPossible_l1213_121320

/-- A permutation of {0, 1, ..., n} -/
def Permutation (n : ℕ) := Fin (n + 1) → Fin (n + 1)

/-- The initial permutation (1, n, n-1, ..., 3, 2, 0) -/
def initialPerm (n : ℕ) : Permutation n := sorry

/-- The target permutation (1, 2, ..., n, 0) -/
def targetPerm (n : ℕ) : Permutation n := sorry

/-- A valid transformation on a permutation -/
def validTransformation (n : ℕ) (p : Permutation n) (p' : Permutation n) : Prop := 
  ∃ i j : Fin (n + 1), p i = 0 ∧ p j = p (i - 1) + 1 ∧ 
    (∀ k : Fin (n + 1), k ≠ i ∧ k ≠ j → p' k = p k) ∧
    p' i = p j ∧ p' j = p i

/-- A sequence of valid transformations -/
def validTransformationSeq (n : ℕ) (p p' : Permutation n) : Prop :=
  ∃ (l : ℕ) (seq : Fin (l + 1) → Permutation n), 
    seq 0 = p ∧ seq l = p' ∧
    ∀ i : Fin l, validTransformation n (seq i) (seq (i + 1))

/-- The main theorem -/
theorem transformationPossible (n : ℕ) : 
  (∃ m : ℕ, n = 2^m - 1) ∨ n = 2 ↔ 
  validTransformationSeq n (initialPerm n) (targetPerm n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformationPossible_l1213_121320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_tangent_line_l1213_121332

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - 1 / x

theorem f_derivative_and_tangent_line :
  (∀ x > 0, deriv f x = Real.log x + 1 + 1 / x^2) ∧
  (∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -3 ∧
    ∀ x y : ℝ, (x, y) ∈ Set.range (fun t => (t, f t + (f 1 - f t))) ↔ a * x + b * y + c = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_and_tangent_line_l1213_121332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_2023_mod_14_l1213_121300

/-- T(n) is the number of valid sequences of length n consisting of A and B,
    with no more than two consecutive A's or B's. -/
def T : ℕ → ℕ := sorry

/-- The remainder when T(2023) is divided by 14 is 8. -/
theorem T_2023_mod_14 : T 2023 % 14 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_2023_mod_14_l1213_121300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_progression_set_l1213_121329

/-- A set of integers from which both a geometric and an arithmetic progression of length 5 can be selected -/
def ProgressionSet (s : Finset ℤ) : Prop :=
  ∃ (a b c d e : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ e ∈ s ∧
  (∃ (r : ℚ), r ≠ 1 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r) ∧
  (∃ (q : ℤ), q ≠ 0 ∧ b = a + q ∧ c = b + q ∧ d = c + q ∧ e = d + q)

/-- The theorem stating that 6 is the smallest number of distinct integers satisfying the condition -/
theorem smallest_progression_set : 
  (∃ (s : Finset ℤ), s.card = 6 ∧ ProgressionSet s) ∧ 
  (∀ (s : Finset ℤ), s.card < 6 → ¬ProgressionSet s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_progression_set_l1213_121329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l1213_121303

def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (1, -1)

def circleEquation (P : ℝ × ℝ) : Prop := P.1^2 + P.2^2 = 2

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_ratio_on_circle :
  ∃ (max : ℝ), max = 2 ∧
  ∀ (P : ℝ × ℝ), circleEquation P →
    distance P B / distance P A ≤ max ∧
    ∃ (P' : ℝ × ℝ), circleEquation P' ∧ distance P' B / distance P' A = max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l1213_121303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_friendship_theorem_l1213_121328

/-- Represents a class with boys and girls -/
structure ClassComposition where
  boys : ℕ
  girls : ℕ

/-- Checks if the friendship conditions are satisfied -/
def satisfiesFriendshipConditions (c : ClassComposition) : Prop :=
  3 * c.boys = 2 * c.girls

/-- Checks if the class has a specific total number of students -/
def hasTotal (c : ClassComposition) (total : ℕ) : Prop :=
  c.boys + c.girls = total

theorem class_friendship_theorem :
  (∀ c : ClassComposition, satisfiesFriendshipConditions c → ¬hasTotal c 32) ∧
  (∃ c : ClassComposition, satisfiesFriendshipConditions c ∧ hasTotal c 30) := by
  sorry

#check class_friendship_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_friendship_theorem_l1213_121328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l1213_121351

/-- The work rate of one man -/
noncomputable def man_rate : ℝ := 1

/-- The work rate of one woman -/
noncomputable def woman_rate : ℝ := 1/2 * man_rate

/-- The number of men in the second group -/
def x : ℕ := 6

theorem work_rate_equality :
  (3 * man_rate + 8 * woman_rate = x * man_rate + 2 * woman_rate) ∧
  (2 * man_rate + 3 * woman_rate = (3 * man_rate + 8 * woman_rate) / 2) →
  x = 6 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_rate_equality_l1213_121351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1213_121395

/-- Given a hyperbola where the real axis is twice the length of the imaginary axis,
    prove that its eccentricity is √5/2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a = 2 * b) (h2 : c^2 = a^2 + b^2) :
  c / a = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1213_121395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_constant_sum_l1213_121394

/-- A line intersecting a parabola -/
structure IntersectingLine where
  α : ℝ  -- angle of the line
  t₁ : ℝ  -- parameter for point P
  t₂ : ℝ  -- parameter for point Q

/-- The theorem statement -/
theorem parabola_intersection_constant_sum (a p : ℝ) (h₁ : a > 0) (h₂ : p > 0) :
  (∀ l : IntersectingLine, 
    (l.t₁ * Real.sin l.α)^2 = 2*p*(a + l.t₁ * Real.cos l.α) ∧ 
    (l.t₂ * Real.sin l.α)^2 = 2*p*(a + l.t₂ * Real.cos l.α) ∧
    (1 / l.t₁^2 + 1 / l.t₂^2 = (p * (Real.cos l.α)^2 + a * (Real.sin l.α)^2) / (p * a^2))) →
  a = p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_constant_sum_l1213_121394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_log2_and_reciprocal_l1213_121366

-- Define the function f
noncomputable def f (x : ℝ) := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

-- State the theorem
theorem f_sum_log2_and_reciprocal : f (Real.log 2 / Real.log 10) + f (Real.log (1/2) / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_log2_and_reciprocal_l1213_121366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_sqrt_two_over_two_l1213_121364

/-- An isosceles right triangle with a randomly chosen point on the hypotenuse -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  M : ℝ × ℝ
  isIsosceles : dist A C = dist B C
  isRight : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  MOnHypotenuse : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

/-- The probability that AM < AC in an isosceles right triangle -/
noncomputable def probability (triangle : IsoscelesRightTriangle) : ℝ :=
  (dist triangle.A triangle.C) / (dist triangle.A triangle.B)

theorem probability_is_sqrt_two_over_two (triangle : IsoscelesRightTriangle) :
  probability triangle = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_sqrt_two_over_two_l1213_121364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1213_121302

/-- Definition of a line through two points -/
def line_through_points (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, (x, y) = ((1 - t) • P.1 + t • Q.1, (1 - t) • P.2 + t • Q.2)}

/-- Definition of a point on the perpendicular line -/
def perpendicular_point (A B C : ℝ × ℝ) : ℝ × ℝ :=
  (C.1 + (B.2 - A.2), C.2 - (B.1 - A.1))

/-- Given points A, B, and C in a 2D plane, prove the equations of line AB and the line perpendicular to AB passing through C. -/
theorem line_equations (A B C : ℝ × ℝ) (h1 : A = (2, -2)) (h2 : B = (4, 6)) (h3 : C = (-2, 0)) :
  (∃ (a b c : ℝ), a * 4 + b * (-1) + c = 0 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ line_through_points A B ↔ a * x + b * y + c = 0) ∧
  (∃ (d e f : ℝ), d * 1 + e * 4 + f = 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ line_through_points C (perpendicular_point A B C) ↔ d * x + e * y + f = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1213_121302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_together_is_ten_l1213_121386

/-- Represents the time (in days) it takes to complete the work -/
structure WorkTime where
  together : ℚ  -- Time for a and b to finish together
  a_alone : ℚ   -- Time for a to finish alone
  a_remaining : ℚ  -- Time a worked alone after b left
  mk_work_time : together > 0 ∧ a_alone > 0 ∧ a_remaining > 0

/-- Calculates the number of days a and b worked together -/
def days_worked_together (wt : WorkTime) : ℚ :=
  wt.together * (1 - wt.a_remaining / wt.a_alone)

/-- Theorem stating that a and b worked together for 10 days -/
theorem days_worked_together_is_ten (wt : WorkTime) 
  (h1 : wt.together = 40)
  (h2 : wt.a_alone = 20)
  (h3 : wt.a_remaining = 15) :
  days_worked_together wt = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_together_is_ten_l1213_121386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l1213_121389

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem (right triangle)
  c = 60 →                 -- hypotenuse is 60
  a + b = 84 →             -- sum of legs is 84
  ((a = 48 ∧ b = 36) ∨ (a = 36 ∧ b = 48)) := by
  sorry

#check right_triangle_legs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_legs_l1213_121389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l1213_121345

/-- Represents a segment of a train journey -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the total time of a train journey -/
noncomputable def totalTime (seg1 seg2 seg3 : Segment) : ℝ :=
  seg1.distance / seg1.speed + seg2.distance / seg2.speed + seg3.distance / seg3.speed

/-- Theorem stating the total time for the specific journey described in the problem -/
theorem journey_time (x y z : ℝ) :
  let seg1 : Segment := { distance := x, speed := 50 }
  let seg2 : Segment := { distance := 2 * y, speed := 30 }
  let seg3 : Segment := { distance := 3 * z, speed := 80 }
  totalTime seg1 seg2 seg3 = (12 * x + 40 * y + 22.5 * z) / 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l1213_121345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_and_angle_sum_l1213_121307

theorem tan_sum_and_angle_sum (α β : ℝ) : 
  (6 * (Real.tan α)^2 - 5 * (Real.tan α) + 1 = 0) →
  (6 * (Real.tan β)^2 - 5 * (Real.tan β) + 1 = 0) →
  (0 < α) → (α < π / 2) →
  (π < β) → (β < 3 * π / 2) →
  (Real.tan (α + β) = 1) ∧ (α + β = 5 * π / 4) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_and_angle_sum_l1213_121307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1213_121343

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |triangle_area 26 22 10 - 107.76| < ε :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l1213_121343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l1213_121382

/-- Calculates the percentage increase in rent for a group of friends -/
theorem rent_increase_percentage 
  (num_friends : ℕ) 
  (original_average : ℝ) 
  (new_average : ℝ) 
  (increased_original_rent : ℝ) 
  (h1 : num_friends = 4)
  (h2 : original_average = 800)
  (h3 : new_average = 880)
  (h4 : increased_original_rent = 1600) : 
  (new_average * (num_friends : ℝ) - original_average * (num_friends : ℝ)) / increased_original_rent * 100 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_percentage_l1213_121382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_intersect_inside_l1213_121346

/-- A polygon is a closed, bounded figure in a plane formed by straight lines. -/
structure Polygon where
  -- We don't need to define the full structure of a polygon,
  -- just its existence is enough for this problem
  bounded : Bool
  closed : Bool

/-- An axis of symmetry is a line that divides a polygon into two congruent parts. -/
structure AxisOfSymmetry (p : Polygon) where
  -- We don't need to define the full structure of an axis of symmetry,
  -- just its existence is enough for this problem

/-- A point represents a location in the plane. -/
structure Point where
  -- We don't need to define the full structure of a point,
  -- just its existence is enough for this problem

/-- Predicate to check if a point is inside a polygon -/
def isInside (point : Point) (polygon : Polygon) : Prop :=
  sorry

/-- Predicate to check if two lines intersect -/
def intersect (p : Polygon) (line1 line2 : AxisOfSymmetry p) : Prop :=
  sorry

/-- Theorem: For any polygon with two axes of symmetry, these axes must intersect inside the polygon -/
theorem axes_intersect_inside (p : Polygon) (axis1 axis2 : AxisOfSymmetry p) :
  ∃ (point : Point), isInside point p ∧ intersect p axis1 axis2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axes_intersect_inside_l1213_121346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1213_121322

/-- Definition of the complex number z in terms of real number m -/
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

/-- Theorem stating the conditions for z to be real, pure imaginary, or in the second quadrant -/
theorem z_properties (m : ℝ) :
  (z m ∈ Set.range Complex.ofReal ↔ m = -1 ∨ m = -2) ∧
  (z m ∈ Set.range (λ x : ℝ => Complex.I * x) ↔ m = 3) ∧
  (z m ∈ {w : ℂ | w.re < 0 ∧ w.im > 0} ↔ -1 < m ∧ m < 3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l1213_121322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_tan_theta_l1213_121336

theorem pure_imaginary_tan_theta (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_tan_theta_l1213_121336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_two_poles_l1213_121338

-- Define the diameters of the two poles
def small_diameter : ℝ := 4
def large_diameter : ℝ := 16

-- Define the function to calculate the shortest wire length
noncomputable def shortest_wire_length (d1 d2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let r2 := d2 / 2
  let center_distance := r1 + r2
  let straight_section := 2 * Real.sqrt (center_distance^2 - (r2 - r1)^2)
  let small_arc := Real.pi * r1 / 2
  let large_arc := 3 * Real.pi * r2 / 2
  straight_section + small_arc + large_arc

-- Theorem statement
theorem shortest_wire_for_two_poles :
  shortest_wire_length small_diameter large_diameter = 16 + 13 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_wire_for_two_poles_l1213_121338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1213_121399

noncomputable def α : ℝ := Real.arcsin (1/2)

theorem problem_solution :
  (Real.sin (2 * α) - Real.tan α = -Real.sqrt 3 / 6) ∧
  (let f : ℝ → ℝ := λ x => Real.cos (x - α) * Real.cos α - Real.sin (x - α) * Real.sin α
   let y : ℝ → ℝ := λ x => Real.sqrt 3 * f (π / 2 - 2 * x) - 2 * f x ^ 2
   ∀ x ∈ Set.Icc 0 (2 * π / 3), -2 ≤ y x ∧ y x ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1213_121399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1213_121379

-- Define the polynomial
def f (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 12*x^2 - 20*x + 5

-- Define the divisor
def g (x : ℝ) : ℝ := 4*x - 8

-- Theorem statement
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 61 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l1213_121379
