import Mathlib

namespace NUMINAMATH_CALUDE_all_events_probability_at_least_one_not_occurring_l3086_308645

-- Define the probabilities of each event
def P_A : ℝ := 0.8
def P_B : ℝ := 0.6
def P_C : ℝ := 0.5

-- Theorem for the probability of all three events occurring
theorem all_events_probability :
  P_A * P_B * P_C = 0.24 :=
sorry

-- Theorem for the probability of at least one event not occurring
theorem at_least_one_not_occurring :
  1 - (P_A * P_B * P_C) = 0.76 :=
sorry

end NUMINAMATH_CALUDE_all_events_probability_at_least_one_not_occurring_l3086_308645


namespace NUMINAMATH_CALUDE_melanie_dimes_l3086_308612

/-- The number of dimes Melanie initially had -/
def initial_dimes : ℕ := 7

/-- The number of dimes Melanie gave to her dad -/
def dimes_to_dad : ℕ := 8

/-- The number of dimes Melanie's mother gave her -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 3

theorem melanie_dimes : 
  initial_dimes - dimes_to_dad + dimes_from_mother = current_dimes :=
by sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3086_308612


namespace NUMINAMATH_CALUDE_lcm_36_100_l3086_308626

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l3086_308626


namespace NUMINAMATH_CALUDE_type_of_2004_least_type_B_after_2004_l3086_308638

/-- Represents the type of a number in the game -/
inductive NumberType
| A
| B

/-- Determines if a number is of type A or B in the game -/
def numberType (n : ℕ) : NumberType :=
  sorry

/-- Theorem stating that 2004 is of type A -/
theorem type_of_2004 : numberType 2004 = NumberType.A :=
  sorry

/-- Theorem stating that 2048 is the least number greater than 2004 of type B -/
theorem least_type_B_after_2004 :
  (numberType 2048 = NumberType.B) ∧
  (∀ m : ℕ, 2004 < m → m < 2048 → numberType m = NumberType.A) :=
  sorry

end NUMINAMATH_CALUDE_type_of_2004_least_type_B_after_2004_l3086_308638


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3086_308610

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_neg_first : a 1 < 0)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  ∃ q : ℝ, 0 < q ∧ q < 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3086_308610


namespace NUMINAMATH_CALUDE_remaining_amount_proof_l3086_308629

def initial_amount : ℕ := 87
def spent_amount : ℕ := 64

theorem remaining_amount_proof :
  initial_amount - spent_amount = 23 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_proof_l3086_308629


namespace NUMINAMATH_CALUDE_larry_win_probability_l3086_308661

/-- The probability of Larry winning a turn-based game against Julius, where:
  * Larry throws first
  * The probability of Larry knocking off the bottle is 3/5
  * The probability of Julius knocking off the bottle is 1/3
  * The winner is the first to knock off the bottle -/
theorem larry_win_probability (p_larry : ℝ) (p_julius : ℝ) 
  (h1 : p_larry = 3/5) 
  (h2 : p_julius = 1/3) :
  p_larry + (1 - p_larry) * (1 - p_julius) * p_larry / (1 - (1 - p_larry) * (1 - p_julius)) = 9/11 :=
by sorry

end NUMINAMATH_CALUDE_larry_win_probability_l3086_308661


namespace NUMINAMATH_CALUDE_mass_of_man_in_boat_l3086_308617

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth : Real) : Real :=
  boat_length * boat_breadth * sink_depth * 1000

/-- Theorem stating the mass of the man in the given problem. -/
theorem mass_of_man_in_boat : mass_of_man 3 2 0.02 = 120 := by
  sorry

#eval mass_of_man 3 2 0.02

end NUMINAMATH_CALUDE_mass_of_man_in_boat_l3086_308617


namespace NUMINAMATH_CALUDE_number_of_routes_P_to_Q_l3086_308678

/-- Represents the points in the diagram --/
inductive Point : Type
| P | Q | R | S | T

/-- Represents a direct path between two points --/
def DirectPath : Point → Point → Prop :=
  fun p q => match p, q with
  | Point.P, Point.R => True
  | Point.P, Point.S => True
  | Point.R, Point.T => True
  | Point.R, Point.Q => True
  | Point.S, Point.T => True
  | Point.T, Point.Q => True
  | _, _ => False

/-- Represents a route from one point to another --/
def Route : Point → Point → Type :=
  fun p q => List (Σ' x y : Point, DirectPath x y)

/-- Counts the number of routes between two points --/
def countRoutes : Point → Point → Nat :=
  fun p q => sorry

theorem number_of_routes_P_to_Q :
  countRoutes Point.P Point.Q = 3 := by sorry

end NUMINAMATH_CALUDE_number_of_routes_P_to_Q_l3086_308678


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3086_308681

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (101054 - k) % 10 = 0 ∧ ∀ (m : ℕ), m < k → (101054 - m) % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l3086_308681


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3086_308683

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3086_308683


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3086_308655

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3086_308655


namespace NUMINAMATH_CALUDE_no_integer_solution_l3086_308663

theorem no_integer_solution : ¬∃ (n : ℕ+), ∃ (k : ℤ), (n.val^(3*n.val - 2) - 3*n.val + 1) / (3*n.val - 2) = k := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3086_308663


namespace NUMINAMATH_CALUDE_complex_root_and_purely_imaginary_l3086_308658

/-- Given that 2-i is a root of x^2 - mx + n = 0 where m and n are real,
    prove that m = 4, n = 5, and if z = a^2 - na + m + (a-m)i is purely imaginary, then a = 1 -/
theorem complex_root_and_purely_imaginary (m n a : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((2 : ℂ) - Complex.I) ^ 2 - m * ((2 : ℂ) - Complex.I) + n = 0 →
  (∃ (b : ℝ), (a ^ 2 - n * a + m : ℂ) + (a - m) * Complex.I = b * Complex.I) →
  (m = 4 ∧ n = 5 ∧ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_root_and_purely_imaginary_l3086_308658


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l3086_308606

theorem integer_less_than_sqrt_23 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_23_l3086_308606


namespace NUMINAMATH_CALUDE_even_function_inequality_l3086_308695

open Real Set

theorem even_function_inequality 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, x ∈ Ioo (-π/2) (π/2) → f x = f (-x))
  (h_deriv : ∀ x, x ∈ Ioo 0 (π/2) → 
    (deriv^[2] f) x * cos x + f x * sin x < 0) :
  ∀ x, x ∈ (Ioo (-π/2) (-π/4) ∪ Ioo (π/4) (π/2)) → 
    f x < Real.sqrt 2 * f (π/4) * cos x := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l3086_308695


namespace NUMINAMATH_CALUDE_max_value_trig_sum_l3086_308639

open Real

theorem max_value_trig_sum (α β γ δ ε : ℝ) : 
  (∀ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α ≤ 5) ∧ 
  (∃ α β γ δ ε : ℝ, cos α * sin β + cos β * sin γ + cos γ * sin δ + cos δ * sin ε + cos ε * sin α = 5) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_sum_l3086_308639


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3086_308600

-- Define the condition function
def condition (x y : ℝ) : Prop := 3 * (x^2 + y^2) = x^2 + y

-- Define the objective function
def objective (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem max_x_minus_y :
  ∃ (max : ℝ), max = 1 / Real.sqrt 24 ∧
  ∀ (x y : ℝ), condition x y → objective x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3086_308600


namespace NUMINAMATH_CALUDE_trig_identity_l3086_308630

theorem trig_identity (x y : ℝ) :
  (Real.sin x)^2 + (Real.sin (x + y + π/4))^2 - 
  2 * (Real.sin x) * (Real.sin (y + π/4)) * (Real.sin (x + y + π/4)) = 
  1 - (1/2) * (Real.sin y)^2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3086_308630


namespace NUMINAMATH_CALUDE_equation_solutions_l3086_308601

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3086_308601


namespace NUMINAMATH_CALUDE_bedroom_size_problem_l3086_308688

theorem bedroom_size_problem (total_area : ℝ) (difference : ℝ) :
  total_area = 300 →
  difference = 60 →
  ∃ (smaller larger : ℝ),
    smaller + larger = total_area ∧
    larger = smaller + difference ∧
    smaller = 120 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_size_problem_l3086_308688


namespace NUMINAMATH_CALUDE_fraction_comparison_l3086_308604

theorem fraction_comparison : (291 : ℚ) / 730 > 29 / 73 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3086_308604


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l3086_308607

structure SeatingArrangement where
  rows_6 : Nat
  rows_8 : Nat
  rows_9 : Nat
  total_people : Nat
  max_rows : Nat

def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_6 * 6 + s.rows_8 * 8 + s.rows_9 * 9 = s.total_people ∧
  s.rows_6 + s.rows_8 + s.rows_9 ≤ s.max_rows

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement),
    s.total_people = 58 ∧
    s.max_rows = 7 ∧
    is_valid s ∧
    s.rows_9 = 4 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l3086_308607


namespace NUMINAMATH_CALUDE_bug_return_probability_l3086_308668

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The problem statement -/
theorem bug_return_probability : P 8 = 547/2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3086_308668


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3086_308632

theorem sum_of_roots_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a ≠ b) →
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3086_308632


namespace NUMINAMATH_CALUDE_suit_price_calculation_l3086_308682

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.3 →
  discount_rate = 0.3 →
  original_price * (1 + increase_rate) * (1 - discount_rate) = 182 := by
  sorry

#check suit_price_calculation

end NUMINAMATH_CALUDE_suit_price_calculation_l3086_308682


namespace NUMINAMATH_CALUDE_constant_function_satisfies_inequality_l3086_308656

theorem constant_function_satisfies_inequality :
  ∀ f : ℕ → ℝ,
  (∀ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 → f (a * c) + f (b * c) - f c * f (a * b) ≥ 1) →
  (∀ x : ℕ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_constant_function_satisfies_inequality_l3086_308656


namespace NUMINAMATH_CALUDE_grandfathers_age_l3086_308619

theorem grandfathers_age (grandfather_age : ℕ) (xiaoming_age : ℕ) : 
  grandfather_age > 7 * xiaoming_age →
  grandfather_age < 70 →
  ∃ (k : ℕ), grandfather_age - xiaoming_age = 60 * k →
  grandfather_age = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l3086_308619


namespace NUMINAMATH_CALUDE_circular_lake_diameter_l3086_308624

/-- The diameter of a circular lake with radius 7 meters is 14 meters. -/
theorem circular_lake_diameter (radius : ℝ) (h : radius = 7) : 2 * radius = 14 := by
  sorry

end NUMINAMATH_CALUDE_circular_lake_diameter_l3086_308624


namespace NUMINAMATH_CALUDE_inequality_solution_l3086_308652

theorem inequality_solution (x : ℝ) : 4 ≤ (2*x)/(3*x-7) ∧ (2*x)/(3*x-7) < 9 ↔ 63/25 < x ∧ x ≤ 2.8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3086_308652


namespace NUMINAMATH_CALUDE_charles_whistles_l3086_308660

/-- Given that Sean has 45 whistles and 32 more whistles than Charles,
    prove that Charles has 13 whistles. -/
theorem charles_whistles (sean_whistles : ℕ) (difference : ℕ) :
  sean_whistles = 45 →
  difference = 32 →
  sean_whistles - difference = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_charles_whistles_l3086_308660


namespace NUMINAMATH_CALUDE_art_collection_cost_l3086_308690

theorem art_collection_cost (price_first_three : ℝ) (price_fourth : ℝ) : 
  price_first_three = 45000 →
  price_fourth = (price_first_three / 3) * 1.5 →
  price_first_three + price_fourth = 67500 := by
sorry

end NUMINAMATH_CALUDE_art_collection_cost_l3086_308690


namespace NUMINAMATH_CALUDE_shirt_count_l3086_308627

theorem shirt_count (total_shirt_price : ℝ) (total_sweater_price : ℝ) (sweater_count : ℕ) (price_difference : ℝ) :
  total_shirt_price = 400 →
  total_sweater_price = 1500 →
  sweater_count = 75 →
  (total_sweater_price / sweater_count) = (total_shirt_price / (total_shirt_price / 16)) + price_difference →
  price_difference = 4 →
  (total_shirt_price / 16 : ℝ) = 25 :=
by sorry

end NUMINAMATH_CALUDE_shirt_count_l3086_308627


namespace NUMINAMATH_CALUDE_max_team_size_l3086_308623

/-- A function that represents a valid selection of team numbers -/
def ValidSelection (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, x ≤ 100 ∧
  ∀ y ∈ s, ∀ z ∈ s, x ≠ y + z ∧
  ∀ y ∈ s, x ≠ 2 * y

/-- The theorem stating the maximum size of a valid selection is 50 -/
theorem max_team_size :
  (∃ s : Finset ℕ, ValidSelection s ∧ s.card = 50) ∧
  ∀ s : Finset ℕ, ValidSelection s → s.card ≤ 50 := by sorry

end NUMINAMATH_CALUDE_max_team_size_l3086_308623


namespace NUMINAMATH_CALUDE_garden_outer_radius_l3086_308699

/-- Given a circular park with a central fountain and a surrounding garden ring,
    this theorem proves the radius of the garden's outer boundary. -/
theorem garden_outer_radius (fountain_diameter : ℝ) (garden_width : ℝ) :
  fountain_diameter = 12 →
  garden_width = 10 →
  fountain_diameter / 2 + garden_width = 16 := by
  sorry

end NUMINAMATH_CALUDE_garden_outer_radius_l3086_308699


namespace NUMINAMATH_CALUDE_number_puzzle_l3086_308628

theorem number_puzzle (x : ℝ) : (x / 2) / 2 = 85 + 45 → x - 45 = 475 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3086_308628


namespace NUMINAMATH_CALUDE_sixth_power_of_complex_root_of_unity_l3086_308616

theorem sixth_power_of_complex_root_of_unity (z : ℂ) : 
  z = (-1 + Complex.I * Real.sqrt 3) / 2 → z^6 = (1 : ℂ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_complex_root_of_unity_l3086_308616


namespace NUMINAMATH_CALUDE_unique_b_solution_l3086_308686

def base_83_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (83 ^ i)) 0

theorem unique_b_solution : ∃! b : ℤ, 
  (0 ≤ b ∧ b ≤ 20) ∧ 
  (∃ k : ℤ, base_83_to_decimal [2, 5, 7, 3, 6, 4, 5] - b = 17 * k) ∧
  b = 8 := by sorry

end NUMINAMATH_CALUDE_unique_b_solution_l3086_308686


namespace NUMINAMATH_CALUDE_no_perfect_square_pair_l3086_308672

theorem no_perfect_square_pair : ¬∃ (a b : ℕ+), 
  (∃ (k : ℕ+), (a.val ^ 2 + b.val : ℕ) = k.val ^ 2) ∧ 
  (∃ (m : ℕ+), (b.val ^ 2 + a.val : ℕ) = m.val ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_pair_l3086_308672


namespace NUMINAMATH_CALUDE_triangle_cut_theorem_l3086_308642

theorem triangle_cut_theorem (x : ℝ) : 
  (∀ y : ℝ, y ≥ x → (12 - y) + (18 - y) ≤ 24 - y) →
  (∀ z : ℝ, z < x → ∃ a b c : ℝ, 
    a + b > c ∧ 
    a + c > b ∧ 
    b + c > a ∧
    a = 12 - z ∧ 
    b = 18 - z ∧ 
    c = 24 - z) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_cut_theorem_l3086_308642


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3086_308676

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ) (b : ℕ), x = a * Real.sqrt b ∧ 
  (∀ (c : ℚ) (d : ℕ), x = c * Real.sqrt d → b ≤ d)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (-Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/7)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3086_308676


namespace NUMINAMATH_CALUDE_expression_value_l3086_308613

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) :
  8 - 6 * a + 9 * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3086_308613


namespace NUMINAMATH_CALUDE_puzzle_pieces_left_l3086_308679

theorem puzzle_pieces_left (total_pieces : ℕ) (num_children : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 500 →
  num_children = 4 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2*reyn_pieces + 3*reyn_pieces + 4*reyn_pieces) = 250 := by
sorry

end NUMINAMATH_CALUDE_puzzle_pieces_left_l3086_308679


namespace NUMINAMATH_CALUDE_triangle_max_area_l3086_308622

/-- Given a triangle ABC with side a = √2 and acosB + bsinA = c, 
    the maximum area of the triangle is (√2 + 1) / 2 -/
theorem triangle_max_area (a b c A B C : Real) :
  a = Real.sqrt 2 →
  a * Real.cos B + b * Real.sin A = c →
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
    S ≤ (Real.sqrt 2 + 1) / 2 ∧
    (S = (Real.sqrt 2 + 1) / 2 ↔ b = c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3086_308622


namespace NUMINAMATH_CALUDE_cow_starting_weight_l3086_308608

/-- The starting weight of a cow, given certain conditions about its weight gain and value increase. -/
theorem cow_starting_weight (W : ℝ) : W = 400 :=
  -- Given conditions
  have weight_increase : W * 1.5 = W + W * 0.5 := by sorry
  have price_per_pound : ℝ := 3
  have value_increase : W * 1.5 * price_per_pound - W * price_per_pound = 600 := by sorry

  -- Proof
  sorry

end NUMINAMATH_CALUDE_cow_starting_weight_l3086_308608


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l3086_308649

-- Define the function representing the curve
def f (x : ℝ) := x^2 + 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point : 
  f 2 = 10 → f' 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l3086_308649


namespace NUMINAMATH_CALUDE_age_sum_problem_l3086_308685

theorem age_sum_problem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 256 →
  a + b + c = 38 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3086_308685


namespace NUMINAMATH_CALUDE_square_of_trinomial_l3086_308641

theorem square_of_trinomial (a b c : ℝ) : 
  (a - 2*b - 3*c)^2 = a^2 - 4*a*b + 4*b^2 - 6*a*c + 12*b*c + 9*c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_trinomial_l3086_308641


namespace NUMINAMATH_CALUDE_investment_growth_l3086_308693

/-- Calculates the future value of an investment with compound interest -/
def future_value (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that an initial investment of $313,021.70 at 6% annual interest for 15 years 
    results in approximately $750,000 -/
theorem investment_growth :
  let initial_investment : ℝ := 313021.70
  let interest_rate : ℝ := 0.06
  let years : ℕ := 15
  let target_amount : ℝ := 750000
  
  abs (future_value initial_investment interest_rate years - target_amount) < 1 := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l3086_308693


namespace NUMINAMATH_CALUDE_estimate_nearsighted_students_l3086_308694

/-- Estimates the number of nearsighted students in a population based on a sample. -/
theorem estimate_nearsighted_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (nearsighted_in_sample : ℕ) 
  (h1 : total_students = 400) 
  (h2 : sample_size = 30) 
  (h3 : nearsighted_in_sample = 12) :
  ⌊(total_students : ℚ) * (nearsighted_in_sample : ℚ) / (sample_size : ℚ)⌋ = 160 :=
sorry

end NUMINAMATH_CALUDE_estimate_nearsighted_students_l3086_308694


namespace NUMINAMATH_CALUDE_minimum_additional_stickers_l3086_308640

def initial_stickers : ℕ := 29
def row_size : ℕ := 4
def group_size : ℕ := 5

theorem minimum_additional_stickers :
  let total_stickers := initial_stickers + 11
  (total_stickers % row_size = 0) ∧
  (total_stickers % group_size = 0) ∧
  (∀ n : ℕ, n < 11 →
    let test_total := initial_stickers + n
    (test_total % row_size ≠ 0) ∨ (test_total % group_size ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_additional_stickers_l3086_308640


namespace NUMINAMATH_CALUDE_randy_piggy_bank_l3086_308651

/-- Calculates the initial amount in Randy's piggy bank -/
def initial_amount (spend_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) (amount_left : ℕ) : ℕ :=
  spend_per_trip * trips_per_month * months_per_year + amount_left

/-- Proves that Randy initially had $200 in his piggy bank -/
theorem randy_piggy_bank : initial_amount 2 4 12 104 = 200 := by
  sorry

end NUMINAMATH_CALUDE_randy_piggy_bank_l3086_308651


namespace NUMINAMATH_CALUDE_train_length_proof_l3086_308646

/-- Given a train that crosses a 500-meter platform in 48 seconds and a signal pole in 18 seconds,
    prove that its length is 300 meters. -/
theorem train_length_proof (L : ℝ) : (L + 500) / 48 = L / 18 ↔ L = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l3086_308646


namespace NUMINAMATH_CALUDE_circle_equation_center_radius_l3086_308659

/-- Given a circle equation, prove its center and radius -/
theorem circle_equation_center_radius 
  (x y : ℝ) 
  (h : x^2 - 2*x + y^2 + 6*y = 6) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -3) ∧ 
    radius = 4 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_center_radius_l3086_308659


namespace NUMINAMATH_CALUDE_solution_set_nonempty_implies_a_range_l3086_308671

theorem solution_set_nonempty_implies_a_range 
  (h : ∃ x, |x - 3| + |x - a| < 4) : 
  -1 < a ∧ a < 7 := by
sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_implies_a_range_l3086_308671


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3086_308609

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry condition
axiom f_symmetry (x : ℝ) : f x = f (5 - x)

-- Define the line of symmetry
def line_of_symmetry : ℝ → Prop := λ x ↦ x = 2.5

-- Theorem stating that the line x = 2.5 is an axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y → f (5 - x) = y → line_of_symmetry ((x + (5 - x)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3086_308609


namespace NUMINAMATH_CALUDE_min_n_plus_d_for_arithmetic_sequence_l3086_308670

/-- An arithmetic sequence with positive integral terms -/
def ArithmeticSequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem min_n_plus_d_for_arithmetic_sequence :
  ∀ a : ℕ → ℕ,
  ∀ d : ℕ,
  ArithmeticSequence a d →
  a 1 = 1 →
  (∃ n : ℕ, a n = 51) →
  (∃ n d : ℕ, ArithmeticSequence a d ∧ a 1 = 1 ∧ a n = 51 ∧ n + d = 16 ∧
    ∀ m k : ℕ, ArithmeticSequence a k ∧ a 1 = 1 ∧ a m = 51 → m + k ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_min_n_plus_d_for_arithmetic_sequence_l3086_308670


namespace NUMINAMATH_CALUDE_problem_solution_l3086_308620

def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem problem_solution :
  (A = {x : ℝ | -3 ≤ x ∧ x ≤ 4}) ∧
  (A ∪ B 3 = {x : ℝ | -3 ≤ x ∧ x ≤ 5}) ∧
  (∀ m : ℝ, A ∪ B m = A ↔ m ≤ 5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3086_308620


namespace NUMINAMATH_CALUDE_triangle_side_length_l3086_308665

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The theorem stating the relationship between sides and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 3)
  (h3 : 3 * t.α + 2 * t.β = Real.pi)
  (h4 : t.α + t.β + t.γ = Real.pi)
  (h5 : 0 < t.a ∧ 0 < t.b ∧ 0 < t.c)
  (h6 : 0 < t.α ∧ 0 < t.β ∧ 0 < t.γ)
  (h7 : t.a / (Real.sin t.α) = t.b / (Real.sin t.β))
  (h8 : t.b / (Real.sin t.β) = t.c / (Real.sin t.γ)) :
  t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3086_308665


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3086_308687

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) :
  (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b ≥ Real.sqrt (8 / 3) := by
  sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a ≠ 0 ∧ (1 / a^2) + 2 * a^2 + 3 * b^2 + 4 * a * b = Real.sqrt (8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3086_308687


namespace NUMINAMATH_CALUDE_star_neg_two_three_l3086_308684

-- Define the new operation ※
def star (a b : ℤ) : ℤ := a^2 + 2*a*b

-- Theorem statement
theorem star_neg_two_three : star (-2) 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_two_three_l3086_308684


namespace NUMINAMATH_CALUDE_draw_balls_count_l3086_308650

/-- The number of ways to draw 4 balls from 20 balls numbered 1 through 20,
    where the sum of the first and last ball drawn is 21. -/
def draw_balls : ℕ :=
  let total_balls : ℕ := 20
  let balls_drawn : ℕ := 4
  let sum_first_last : ℕ := 21
  let valid_first_balls : ℕ := sum_first_last - 1
  let remaining_choices : ℕ := total_balls - 2
  valid_first_balls * remaining_choices * (remaining_choices - 1)

theorem draw_balls_count : draw_balls = 3060 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_count_l3086_308650


namespace NUMINAMATH_CALUDE_square_diagonal_less_than_twice_fg_l3086_308611

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B = (A.1 + s, A.2) ∧
    C = (A.1 + s, A.2 + s) ∧
    D = (A.1, A.2 + s)

-- Define that E is an internal point on side AD
def InternalPointOnSide (E A D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (A.1, A.2 + t * (D.2 - A.2))

-- Define F as the foot of the perpendicular from B to CE
def PerpendicularFoot (F B C E : ℝ × ℝ) : Prop :=
  (F.1 - C.1) * (E.1 - C.1) + (F.2 - C.2) * (E.2 - C.2) = 0 ∧
  (F.1 - B.1) * (E.1 - C.1) + (F.2 - B.2) * (E.2 - C.2) = 0

-- Define that BG = FG
def EqualDistances (B F G : ℝ × ℝ) : Prop :=
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = (G.1 - F.1)^2 + (G.2 - F.2)^2

-- Define that the line through G parallel to BC passes through the midpoint of EF
def ParallelThroughMidpoint (G B C E F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = ((E.1 + F.1)/2 + t*(C.1 - B.1), (E.2 + F.2)/2 + t*(C.2 - B.2))

-- State the theorem
theorem square_diagonal_less_than_twice_fg 
  (A B C D E F G : ℝ × ℝ) : 
  Square A B C D → 
  InternalPointOnSide E A D → 
  PerpendicularFoot F B C E → 
  EqualDistances B F G → 
  ParallelThroughMidpoint G B C E F → 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 < 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_less_than_twice_fg_l3086_308611


namespace NUMINAMATH_CALUDE_sqrt_product_equals_150_sqrt_3_l3086_308635

theorem sqrt_product_equals_150_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 45 * Real.sqrt 20 = 150 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_150_sqrt_3_l3086_308635


namespace NUMINAMATH_CALUDE_f_multiplicative_f_derivative_positive_f_derivative_odd_f_satisfies_properties_l3086_308657

/-- A function satisfying specific properties -/
def f (x : ℝ) : ℝ := x^2

/-- Property 1: f(x₁x₂) = f(x₁)f(x₂) for all x₁, x₂ -/
theorem f_multiplicative : ∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂ := by sorry

/-- Property 2: For x ∈ (0, +∞), f'(x) > 0 -/
theorem f_derivative_positive : ∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x := by sorry

/-- Property 3: f'(x) is an odd function -/
theorem f_derivative_odd : ∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ↔ HasDerivAt f (-2 * x) x := by sorry

/-- The main theorem stating that f satisfies all properties -/
theorem f_satisfies_properties : 
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x) ∧
  (∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ↔ HasDerivAt f (-2 * x) x) := by sorry

end NUMINAMATH_CALUDE_f_multiplicative_f_derivative_positive_f_derivative_odd_f_satisfies_properties_l3086_308657


namespace NUMINAMATH_CALUDE_calculation_proofs_l3086_308605

theorem calculation_proofs :
  (40 + (1/6 - 2/3 + 3/4) * 12 = 43) ∧
  ((-1)^2021 + |(-9)| * (2/3) + (-3) / (1/5) = -10) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l3086_308605


namespace NUMINAMATH_CALUDE_max_player_salary_l3086_308643

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  num_players = 25 →
  min_salary = 18000 →
  total_cap = 900000 →
  (num_players - 1) * min_salary + (total_cap - (num_players - 1) * min_salary) ≤ total_cap →
  (∀ (salaries : List ℕ), salaries.length = num_players → 
    (∀ s ∈ salaries, s ≥ min_salary) → 
    salaries.sum ≤ total_cap →
    ∀ s ∈ salaries, s ≤ 468000) :=
by sorry

#check max_player_salary

end NUMINAMATH_CALUDE_max_player_salary_l3086_308643


namespace NUMINAMATH_CALUDE_quadratic_inequality_boundary_l3086_308625

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 1) < c ↔ -5/2 < x ∧ x < 3) → c = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_boundary_l3086_308625


namespace NUMINAMATH_CALUDE_remainder_theorem_l3086_308637

/-- The polynomial f(x) = 4x^5 - 9x^4 + 7x^2 - x - 35 -/
def f (x : ℝ) : ℝ := 4 * x^5 - 9 * x^4 + 7 * x^2 - x - 35

/-- The theorem stating that the remainder when f(x) is divided by (x - 2.5) is 45.3125 -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = λ x => (x - 2.5) * q x + 45.3125 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3086_308637


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3086_308677

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ y : ℝ, y^2 - 1 > 0 ∧ ¬(y < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3086_308677


namespace NUMINAMATH_CALUDE_original_price_correct_l3086_308602

/-- The original price of a bag of mini peanut butter cups before discount -/
def original_price : ℝ := 6

/-- The discount percentage applied to the bags -/
def discount_percentage : ℝ := 0.75

/-- The number of bags purchased -/
def num_bags : ℕ := 2

/-- The total amount spent on the bags after discount -/
def total_spent : ℝ := 3

/-- Theorem stating that the original price is correct given the conditions -/
theorem original_price_correct : 
  (1 - discount_percentage) * (num_bags * original_price) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_original_price_correct_l3086_308602


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3086_308614

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^2 - 3*x + 1) - 4*(2*x^2 - 3*x + 5) = 8*x^3 - 14*x^2 + 14*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3086_308614


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l3086_308680

/-- The maximum area of a rectangular garden with a perimeter of 168 feet and natural number side lengths --/
theorem max_area_rectangular_garden :
  ∃ (w h : ℕ), 
    w + h = 84 ∧ 
    (∀ (x y : ℕ), x + y = 84 → x * y ≤ w * h) ∧
    w * h = 1764 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l3086_308680


namespace NUMINAMATH_CALUDE_classroom_problem_l3086_308618

/-- Calculates the final number of children in a classroom after some changes -/
def final_children_count (initial_boys initial_girls boys_left girls_entered : ℕ) : ℕ :=
  (initial_boys - boys_left) + (initial_girls + girls_entered)

/-- Proves that the final number of children in the classroom is 8 -/
theorem classroom_problem :
  let initial_boys : ℕ := 5
  let initial_girls : ℕ := 4
  let boys_left : ℕ := 3
  let girls_entered : ℕ := 2
  final_children_count initial_boys initial_girls boys_left girls_entered = 8 := by
  sorry

#eval final_children_count 5 4 3 2

end NUMINAMATH_CALUDE_classroom_problem_l3086_308618


namespace NUMINAMATH_CALUDE_parabola_points_m_range_l3086_308664

/-- The parabola equation -/
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

theorem parabola_points_m_range (a x₁ x₂ y₁ y₂ m : ℝ) : 
  a ≠ 0 →
  parabola a x₁ = y₁ →
  parabola a x₂ = y₂ →
  -2 < x₁ →
  x₁ < 0 →
  m < x₂ →
  x₂ < m + 1 →
  y₁ ≠ y₂ →
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_points_m_range_l3086_308664


namespace NUMINAMATH_CALUDE_accurate_to_hundreds_place_l3086_308667

/-- Represents a number with a specified precision --/
structure PreciseNumber where
  value : ℝ
  precision : ℕ

/-- Defines what it means for a number to be accurate to a certain place value --/
def accurate_to (n : PreciseNumber) (place : ℕ) : Prop :=
  ∃ (m : ℤ), n.value = (m : ℝ) * (10 : ℝ) ^ place ∧ n.precision = place

/-- The statement to be proved --/
theorem accurate_to_hundreds_place :
  let n : PreciseNumber := ⟨4.0 * 10^3, 2⟩
  accurate_to n 2 :=
sorry

end NUMINAMATH_CALUDE_accurate_to_hundreds_place_l3086_308667


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3086_308634

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ, 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3086_308634


namespace NUMINAMATH_CALUDE_equation_solution_l3086_308621

theorem equation_solution : ∃! x : ℝ, (x - 1) + 2 * Real.sqrt (x + 3) = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3086_308621


namespace NUMINAMATH_CALUDE_rectangle_area_doubling_l3086_308631

theorem rectangle_area_doubling (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  let new_length := 1.4 * l
  let new_width := (10/7) * w
  let original_area := l * w
  let new_area := new_length * new_width
  new_area = 2 * original_area := by sorry

end NUMINAMATH_CALUDE_rectangle_area_doubling_l3086_308631


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3086_308696

theorem complex_fraction_simplification :
  (Complex.I - 1) / (1 + Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3086_308696


namespace NUMINAMATH_CALUDE_steak_knife_cost_l3086_308675

/-- The number of steak knife sets -/
def num_sets : ℕ := 2

/-- The number of steak knives in each set -/
def knives_per_set : ℕ := 4

/-- The cost of each set in dollars -/
def cost_per_set : ℚ := 80

/-- The total number of steak knives -/
def total_knives : ℕ := num_sets * knives_per_set

/-- The total cost of all sets in dollars -/
def total_cost : ℚ := num_sets * cost_per_set

/-- The cost of each single steak knife in dollars -/
def cost_per_knife : ℚ := total_cost / total_knives

theorem steak_knife_cost : cost_per_knife = 20 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_cost_l3086_308675


namespace NUMINAMATH_CALUDE_cube_less_than_three_times_square_l3086_308674

theorem cube_less_than_three_times_square (x : ℤ) :
  x^3 < 3*x^2 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_three_times_square_l3086_308674


namespace NUMINAMATH_CALUDE_curve_C₁_and_constant_product_l3086_308653

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C₂ x' y' → (x - x')^2 + (y - y')^2 ≥ (y + 2)^2

-- Define the line y = -4
def line_y_neg4 (x y : ℝ) : Prop := y = -4

-- Define the tangent lines from a point to C₂
def tangent_to_C₂ (x₀ y₀ x y : ℝ) : Prop :=
  ∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ (x₀^2 - 9) * k^2 + 18 * x₀ * k + 72 = 0

-- Theorem statement
theorem curve_C₁_and_constant_product :
  (∀ x y : ℝ, C₁ x y ↔ x^2 = 20 * y) ∧
  (∀ x₀ : ℝ, x₀ ≠ 3 ∧ x₀ ≠ -3 →
    ∀ x₁ x₂ x₃ x₄ : ℝ,
    (∃ y₀, line_y_neg4 x₀ y₀) →
    (∃ y₁, C₁ x₁ y₁ ∧ tangent_to_C₂ x₀ (-4) x₁ y₁) →
    (∃ y₂, C₁ x₂ y₂ ∧ tangent_to_C₂ x₀ (-4) x₂ y₂) →
    (∃ y₃, C₁ x₃ y₃ ∧ tangent_to_C₂ x₀ (-4) x₃ y₃) →
    (∃ y₄, C₁ x₄ y₄ ∧ tangent_to_C₂ x₀ (-4) x₄ y₄) →
    x₁ * x₂ * x₃ * x₄ = 6400) :=
sorry

end NUMINAMATH_CALUDE_curve_C₁_and_constant_product_l3086_308653


namespace NUMINAMATH_CALUDE_min_groups_for_class_l3086_308698

theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ) (group_size : ℕ),
    num_groups * group_size = total_students ∧
    group_size ≤ max_group_size ∧
    (∀ (other_num_groups : ℕ) (other_group_size : ℕ),
      other_num_groups * other_group_size = total_students →
      other_group_size ≤ max_group_size →
      num_groups ≤ other_num_groups) ∧
    num_groups = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_class_l3086_308698


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l3086_308691

/-- For positive real numbers p and q where p > 2q, the infinite sum
    1/(pq) + 1/(p(3p-2q)) + 1/((3p-2q)(5p-4q)) + 1/((5p-4q)(7p-6q)) + ...
    is equal to 1/((p-2q)p). -/
theorem infinite_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > 2*q) :
  let f : ℕ → ℝ := λ n => 1 / ((2*n - 1)*p - (2*n - 2)*q) / ((2*n + 1)*p - 2*n*q)
  ∑' n, f n = 1 / ((p - 2*q) * p) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l3086_308691


namespace NUMINAMATH_CALUDE_largest_n_is_69_l3086_308666

/-- Represents a three-digit number in a given base --/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- Converts a three-digit number from a given base to base 10 --/
def to_base_10 (base : ℕ) (num : ThreeDigitNumber base) : ℕ :=
  num.hundreds * base^2 + num.tens * base + num.ones

theorem largest_n_is_69 :
  ∀ (n : ℕ) (base_5 : ThreeDigitNumber 5) (base_9 : ThreeDigitNumber 9),
    n > 0 →
    to_base_10 5 base_5 = n →
    to_base_10 9 base_9 = n →
    base_5.hundreds = base_9.ones →
    base_5.tens = base_9.tens →
    base_5.ones = base_9.hundreds →
    n ≤ 69 :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_69_l3086_308666


namespace NUMINAMATH_CALUDE_square_perimeter_l3086_308644

theorem square_perimeter (area : ℝ) (side : ℝ) : 
  area = 400 ∧ area = side * side → 4 * side = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3086_308644


namespace NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3086_308689

theorem percentage_of_women_in_study_group 
  (percentage_women_lawyers : Real) 
  (probability_selecting_woman_lawyer : Real) :
  let percentage_women := probability_selecting_woman_lawyer / percentage_women_lawyers
  percentage_women_lawyers = 0.4 →
  probability_selecting_woman_lawyer = 0.28 →
  percentage_women = 0.7 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l3086_308689


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l3086_308647

theorem midpoint_of_fractions : 
  let f1 : ℚ := 3/4
  let f2 : ℚ := 5/6
  let midpoint : ℚ := (f1 + f2) / 2
  midpoint = 19/24 := by sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l3086_308647


namespace NUMINAMATH_CALUDE_cost_for_haleighs_pets_l3086_308673

/-- Calculates the cost of leggings for Haleigh's pets -/
def cost_of_leggings (dogs cats spiders parrots chickens octopuses : ℕ)
  (dog_legs cat_legs spider_legs parrot_legs chicken_legs octopus_legs : ℕ)
  (bulk_price : ℚ) (bulk_quantity : ℕ) (regular_price : ℚ) : ℚ :=
  let total_legs := dogs * dog_legs + cats * cat_legs + spiders * spider_legs +
                    parrots * parrot_legs + chickens * chicken_legs + octopuses * octopus_legs
  let total_pairs := total_legs / 2
  let bulk_sets := total_pairs / bulk_quantity
  let remaining_pairs := total_pairs % bulk_quantity
  (bulk_sets * bulk_price) + (remaining_pairs * regular_price)

theorem cost_for_haleighs_pets :
  cost_of_leggings 4 3 2 1 5 3 4 4 8 2 2 8 18 12 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_haleighs_pets_l3086_308673


namespace NUMINAMATH_CALUDE_brick_length_is_20_l3086_308697

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 27

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required for the wall -/
def num_bricks : ℕ := 27000

/-- Conversion factor from cubic meters to cubic centimeters -/
def m3_to_cm3 : ℝ := 1000000

theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width = 10 ∧
  brick_height = 7.5 ∧
  wall_length = 27 ∧
  wall_width = 2 ∧
  wall_height = 0.75 ∧
  num_bricks = 27000 →
  wall_length * wall_width * wall_height * m3_to_cm3 =
    brick_length * brick_width * brick_height * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_20_l3086_308697


namespace NUMINAMATH_CALUDE_factorization_and_sum_of_coefficients_l3086_308633

theorem factorization_and_sum_of_coefficients :
  ∃ (a b c d e f : ℤ),
    (81 : ℚ) * x^4 - 256 * y^4 = (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) ∧
    (a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) = (3 * x - 4 * y) * (3 * x + 4 * y) * (9 * x^2 + 16 * y^2) ∧
    a + b + c + d + e + f = 31 :=
by sorry

end NUMINAMATH_CALUDE_factorization_and_sum_of_coefficients_l3086_308633


namespace NUMINAMATH_CALUDE_fruit_tree_ratio_l3086_308662

theorem fruit_tree_ratio (total_streets : ℕ) (plum_trees pear_trees apricot_trees : ℕ) : 
  total_streets = 18 →
  plum_trees = 3 →
  pear_trees = 3 →
  apricot_trees = 3 →
  (plum_trees + pear_trees + apricot_trees : ℚ) / total_streets = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_ratio_l3086_308662


namespace NUMINAMATH_CALUDE_smallest_odd_with_24_divisors_l3086_308654

/-- The number of divisors of a positive integer -/
def numDivisors (n : ℕ+) : ℕ := sorry

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem smallest_odd_with_24_divisors :
  ∃ (n : ℕ+),
    isOdd n.val ∧
    numDivisors n = 24 ∧
    (∀ (m : ℕ+), isOdd m.val ∧ numDivisors m = 24 → n ≤ m) ∧
    n = 3465 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_with_24_divisors_l3086_308654


namespace NUMINAMATH_CALUDE_scalene_triangle_unique_x_l3086_308603

/-- Represents a scalene triangle with specific properties -/
structure ScaleneTriangle where
  -- One angle is 45 degrees
  angle1 : ℝ
  angle1_eq : angle1 = 45
  -- Another angle is x degrees
  angle2 : ℝ
  -- The third angle
  angle3 : ℝ
  -- The sum of all angles is 180 degrees
  angle_sum : angle1 + angle2 + angle3 = 180
  -- The sides opposite angle1 and angle2 are equal
  equal_sides : True
  -- The triangle is scalene (all sides are different)
  is_scalene : True

/-- 
Theorem: In a scalene triangle with one angle of 45° and another angle of x°, 
where the side lengths opposite these two angles are equal, 
the only possible value for x is 45°.
-/
theorem scalene_triangle_unique_x (t : ScaleneTriangle) : t.angle2 = 45 := by
  sorry

#check scalene_triangle_unique_x

end NUMINAMATH_CALUDE_scalene_triangle_unique_x_l3086_308603


namespace NUMINAMATH_CALUDE_beth_crayons_l3086_308648

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 8

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 12

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 15

/-- The number of crayons Beth borrowed from her friend -/
def borrowed_crayons : ℕ := 7

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons + borrowed_crayons

theorem beth_crayons :
  total_crayons = 118 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l3086_308648


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3086_308669

theorem no_natural_numbers_satisfying_condition : 
  ¬∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y - (x + y) = 2021 :=
by sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l3086_308669


namespace NUMINAMATH_CALUDE_women_in_room_l3086_308615

theorem women_in_room (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 5 = 23 →
  3 * (initial_women - 4) = 57 :=
by sorry

end NUMINAMATH_CALUDE_women_in_room_l3086_308615


namespace NUMINAMATH_CALUDE_pug_cleaning_theorem_l3086_308692

/-- The number of pugs in the first scenario -/
def num_pugs : ℕ := 4

/-- The time taken by the unknown number of pugs to clean the house -/
def time1 : ℕ := 45

/-- The number of pugs in the second scenario -/
def num_pugs2 : ℕ := 15

/-- The time taken by the known number of pugs to clean the house -/
def time2 : ℕ := 12

/-- The theorem stating that the number of pugs in the first scenario is 4 -/
theorem pug_cleaning_theorem : 
  num_pugs * time1 = num_pugs2 * time2 := by sorry

end NUMINAMATH_CALUDE_pug_cleaning_theorem_l3086_308692


namespace NUMINAMATH_CALUDE_dancer_count_l3086_308636

theorem dancer_count (n : ℕ) : 
  (200 ≤ n ∧ n ≤ 300) ∧
  (∃ k : ℕ, n + 5 = 12 * k) ∧
  (∃ m : ℕ, n + 5 = 10 * m) →
  n = 235 ∨ n = 295 := by
sorry

end NUMINAMATH_CALUDE_dancer_count_l3086_308636
