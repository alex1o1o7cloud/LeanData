import Mathlib

namespace NUMINAMATH_CALUDE_sum_in_base4_is_1022_l670_67012

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The sum of 321₄, 32₄, and 3₄ in base 4 --/
def sumInBase4 : List Nat :=
  let sum := base4ToBase10 [1, 2, 3] + base4ToBase10 [2, 3] + base4ToBase10 [3]
  base10ToBase4 sum

theorem sum_in_base4_is_1022 : sumInBase4 = [1, 0, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base4_is_1022_l670_67012


namespace NUMINAMATH_CALUDE_proportion_solution_l670_67003

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l670_67003


namespace NUMINAMATH_CALUDE_count_pairs_eq_738_l670_67018

/-- The number of pairs (a, b) with 1 ≤ a < b ≤ 57 such that a^2 mod 57 < b^2 mod 57 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let (a, b) := p
    1 ≤ a ∧ a < b ∧ b ≤ 57 ∧ (a^2 % 57 < b^2 % 57))
    (Finset.product (Finset.range 58) (Finset.range 58))).card

theorem count_pairs_eq_738 : count_pairs = 738 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_738_l670_67018


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l670_67080

-- Define the conditions
def p (x : ℝ) : Prop := (x - 1) / (x + 3) ≥ 0
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the relationship between ¬p and ¬q
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (¬P → ¬Q) ∧ ¬(¬Q → ¬P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  sufficient_not_necessary (∃ x, p x) (∃ x, q x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l670_67080


namespace NUMINAMATH_CALUDE_intra_division_games_is_56_l670_67049

/-- Represents a basketball league with specific conditions -/
structure BasketballLeague where
  N : ℕ  -- Number of times teams within the same division play each other
  M : ℕ  -- Number of times teams from different divisions play each other
  division_size : ℕ  -- Number of teams in each division
  total_games : ℕ  -- Total number of games each team plays in the season
  h1 : 3 * N = 5 * M + 8
  h2 : M > 6
  h3 : division_size = 5
  h4 : total_games = 82
  h5 : (division_size - 1) * N + division_size * M = total_games

/-- The number of games a team plays within its own division -/
def intra_division_games (league : BasketballLeague) : ℕ :=
  (league.division_size - 1) * league.N

/-- Theorem stating that each team plays 56 games within its own division -/
theorem intra_division_games_is_56 (league : BasketballLeague) :
  intra_division_games league = 56 := by
  sorry

end NUMINAMATH_CALUDE_intra_division_games_is_56_l670_67049


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l670_67035

theorem consecutive_even_integers_sum (n : ℤ) :
  (n + (n + 6) = 160) →
  ((n + 2) + (n + 4) = 160) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l670_67035


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_l670_67055

theorem simplify_sqrt_product : 
  Real.sqrt (5 * 3) * Real.sqrt (3^4 * 5^2) = 225 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_l670_67055


namespace NUMINAMATH_CALUDE_coat_cost_price_l670_67076

theorem coat_cost_price (markup_percentage : ℝ) (final_price : ℝ) (cost_price : ℝ) : 
  markup_percentage = 0.25 →
  final_price = 275 →
  final_price = cost_price * (1 + markup_percentage) →
  cost_price = 220 := by
  sorry

end NUMINAMATH_CALUDE_coat_cost_price_l670_67076


namespace NUMINAMATH_CALUDE_elder_sister_savings_l670_67002

theorem elder_sister_savings (total : ℝ) (elder_donation_rate : ℝ) (younger_donation_rate : ℝ)
  (h_total : total = 108)
  (h_elder_rate : elder_donation_rate = 0.75)
  (h_younger_rate : younger_donation_rate = 0.8)
  (h_equal_remainder : ∃ (elder younger : ℝ), 
    elder + younger = total ∧ 
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate)) :
  ∃ (elder : ℝ), elder = 48 ∧ 
    ∃ (younger : ℝ), younger = total - elder ∧
    elder * (1 - elder_donation_rate) = younger * (1 - younger_donation_rate) := by
  sorry

end NUMINAMATH_CALUDE_elder_sister_savings_l670_67002


namespace NUMINAMATH_CALUDE_exists_alpha_congruence_l670_67021

theorem exists_alpha_congruence : ∃ α : ℤ, α ^ 2 ≡ 2 [ZMOD 7^3] ∧ α ≡ 3 [ZMOD 7] :=
sorry

end NUMINAMATH_CALUDE_exists_alpha_congruence_l670_67021


namespace NUMINAMATH_CALUDE_basket_replacement_theorem_l670_67087

/-- The number of people who entered the stadium before the basket needed replacement -/
def people_entered : ℕ :=
  sorry

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

theorem basket_replacement_theorem :
  people_entered = 411 ∧
  people_entered * placards_per_person < basket_capacity ∧
  (people_entered + 1) * placards_per_person > basket_capacity :=
by sorry

end NUMINAMATH_CALUDE_basket_replacement_theorem_l670_67087


namespace NUMINAMATH_CALUDE_all_points_on_line_l670_67089

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line defined by two other points -/
def isOnLine (p : Point) (p1 : Point) (p2 : Point) : Prop :=
  (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)

theorem all_points_on_line :
  let p1 : Point := ⟨8, 2⟩
  let p2 : Point := ⟨2, -10⟩
  let points : List Point := [⟨5, -4⟩, ⟨4, -6⟩, ⟨10, 6⟩, ⟨0, -14⟩, ⟨1, -12⟩]
  ∀ p ∈ points, isOnLine p p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_all_points_on_line_l670_67089


namespace NUMINAMATH_CALUDE_parabola_hyperbola_configuration_l670_67078

/-- Theorem: Value of 'a' for a specific parabola and hyperbola configuration -/
theorem parabola_hyperbola_configuration (p t a : ℝ) : 
  p > 0 → 
  t > 0 → 
  a > 0 → 
  t^2 = 2*p*1 → 
  (1 + p/2)^2 + t^2 = 5^2 → 
  (∃ k : ℝ, k = 4/(1+a) ∧ k = 3/a) → 
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_configuration_l670_67078


namespace NUMINAMATH_CALUDE_flea_count_l670_67006

/-- The total number of fleas on three chickens (Gertrude, Maud, and Olive) -/
def total_fleas (gertrude_fleas : ℕ) (olive_fleas : ℕ) (maud_fleas : ℕ) : ℕ :=
  gertrude_fleas + olive_fleas + maud_fleas

/-- Theorem stating the total number of fleas on the three chickens is 40 -/
theorem flea_count :
  ∀ (gertrude_fleas olive_fleas maud_fleas : ℕ),
  gertrude_fleas = 10 →
  olive_fleas = gertrude_fleas / 2 →
  maud_fleas = 5 * olive_fleas →
  total_fleas gertrude_fleas olive_fleas maud_fleas = 40 :=
by
  sorry

#check flea_count

end NUMINAMATH_CALUDE_flea_count_l670_67006


namespace NUMINAMATH_CALUDE_a_5_equals_9_l670_67004

-- Define the sequence a_n implicitly through S_n
def S (n : ℕ) : ℕ := n^2 - 1

-- Define a_n as the difference between consecutive S_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else (S n : ℤ) - (S (n-1) : ℤ)

-- State the theorem
theorem a_5_equals_9 : a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_a_5_equals_9_l670_67004


namespace NUMINAMATH_CALUDE_factory_composition_diagram_l670_67098

/-- Represents different types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

/-- Represents the purpose of a diagram --/
inductive DiagramPurpose
  | RepresentComposition
  | RepresentProcedures
  | RepresentKnowledge

/-- Associates a diagram type with its primary purpose --/
def diagramPurpose (d : Diagram) : DiagramPurpose :=
  match d with
  | Diagram.ProgramFlowchart => DiagramPurpose.RepresentProcedures
  | Diagram.ProcessFlow => DiagramPurpose.RepresentProcedures
  | Diagram.KnowledgeStructure => DiagramPurpose.RepresentKnowledge
  | Diagram.OrganizationalStructure => DiagramPurpose.RepresentComposition

/-- The theorem stating that the Organizational Structure Diagram 
    is used to represent the composition of a factory --/
theorem factory_composition_diagram :
  diagramPurpose Diagram.OrganizationalStructure = DiagramPurpose.RepresentComposition :=
by sorry


end NUMINAMATH_CALUDE_factory_composition_diagram_l670_67098


namespace NUMINAMATH_CALUDE_sine_cosine_values_l670_67079

def angle_on_line (α : Real) : Prop :=
  ∃ (x y : Real), y = Real.sqrt 3 * x ∧ 
  (Real.cos α = x / Real.sqrt (x^2 + y^2)) ∧
  (Real.sin α = y / Real.sqrt (x^2 + y^2))

theorem sine_cosine_values (α : Real) (h : angle_on_line α) :
  (Real.sin α = Real.sqrt 3 / 2 ∧ Real.cos α = 1 / 2) ∨
  (Real.sin α = -Real.sqrt 3 / 2 ∧ Real.cos α = -1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_values_l670_67079


namespace NUMINAMATH_CALUDE_candy_heating_rate_l670_67091

/-- Candy heating problem -/
theorem candy_heating_rate
  (initial_temp : ℝ)
  (max_temp : ℝ)
  (final_temp : ℝ)
  (cooling_rate : ℝ)
  (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : max_temp = 240)
  (h3 : final_temp = 170)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46)
  : ∃ (heating_rate : ℝ), heating_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_heating_rate_l670_67091


namespace NUMINAMATH_CALUDE_polynomial_inequality_solution_l670_67061

theorem polynomial_inequality_solution (x : ℝ) : 
  x^4 - 15*x^3 + 80*x^2 - 200*x > 0 ↔ (0 < x ∧ x < 5) ∨ x > 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_solution_l670_67061


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l670_67032

def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a * b = k

theorem inverse_variation_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : inverse_relation a₁ b₁ ∧ inverse_relation a₂ b₂)
  (h_a₁ : a₁ = 1500)
  (h_b₁ : b₁ = 0.25)
  (h_a₂ : a₂ = 3000) :
  b₂ = 0.125 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l670_67032


namespace NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l670_67051

theorem pascal_triangle_row_15_fifth_number : 
  let row := List.range 16
  let pascal_row := row.map (fun k => Nat.choose 15 k)
  pascal_row[4] = 1365 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_15_fifth_number_l670_67051


namespace NUMINAMATH_CALUDE_max_abs_diff_on_interval_l670_67099

open Real

-- Define the functions
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the absolute difference function
def abs_diff (x : ℝ) : ℝ := |f x - g x|

-- State the theorem
theorem max_abs_diff_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ 
  ∀ (x : ℝ), x ∈ Set.Icc 0 1 → abs_diff x ≤ abs_diff c ∧
  abs_diff c = 4/27 :=
sorry

end NUMINAMATH_CALUDE_max_abs_diff_on_interval_l670_67099


namespace NUMINAMATH_CALUDE_common_point_intersection_l670_67057

/-- The common point of intersection for a family of lines -/
def common_point : ℝ × ℝ := (-1, 1)

/-- The equation of lines in the family -/
def line_equation (a b c x y : ℝ) : Prop := a * x + b * y = c

/-- The arithmetic progression condition -/
def arithmetic_progression (a b c d : ℝ) : Prop := b = a - d ∧ c = a - 2 * d

theorem common_point_intersection :
  ∀ (a b c d x y : ℝ),
    arithmetic_progression a b c d →
    (x, y) = common_point ↔ line_equation a b c x y :=
by sorry

end NUMINAMATH_CALUDE_common_point_intersection_l670_67057


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_product_less_500_l670_67007

theorem greatest_sum_consecutive_odd_integers_product_less_500 : 
  (∃ (n : ℤ), 
    Odd n ∧ 
    n * (n + 2) < 500 ∧ 
    n + (n + 2) = 44 ∧ 
    (∀ (m : ℤ), Odd m → m * (m + 2) < 500 → m + (m + 2) ≤ 44)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_odd_integers_product_less_500_l670_67007


namespace NUMINAMATH_CALUDE_solution_implies_m_range_l670_67074

/-- A function representing the quadratic equation x^2 - mx + 2 = 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 2

/-- The theorem stating that if the equation x^2 - mx + 2 = 0 has a solution 
    in the interval [1, 2], then m is in the range [2√2, 3] -/
theorem solution_implies_m_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ f m x = 0) → 
  m ∈ Set.Icc (2 * Real.sqrt 2) 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_implies_m_range_l670_67074


namespace NUMINAMATH_CALUDE_bowl_capacity_ratio_l670_67069

theorem bowl_capacity_ratio :
  ∀ (capacity_1 capacity_2 : ℕ),
    capacity_1 < capacity_2 →
    capacity_2 = 600 →
    capacity_1 + capacity_2 = 1050 →
    (capacity_1 : ℚ) / capacity_2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bowl_capacity_ratio_l670_67069


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_double_factorial_l670_67023

/-- Double factorial of a natural number -/
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

/-- Sum of factorials from 1 to n -/
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => Nat.factorial (i + 1))

/-- Theorem: The units digit of the sum of factorials from 1 to 12 plus 12!! is 3 -/
theorem units_digit_sum_factorials_plus_double_factorial :
  (sum_factorials 12 + double_factorial 12) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_double_factorial_l670_67023


namespace NUMINAMATH_CALUDE_bank_savings_exceed_50_dollars_l670_67065

/-- The sum of a geometric sequence with first term 5 and ratio 2, after n terms -/
def geometric_sum (n : ℕ) : ℚ := 5 * (2^n - 1)

/-- The smallest number of days needed for the sum to exceed 5000 cents -/
def smallest_day : ℕ := 10

theorem bank_savings_exceed_50_dollars :
  (∀ k < smallest_day, geometric_sum k ≤ 5000) ∧
  geometric_sum smallest_day > 5000 := by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_50_dollars_l670_67065


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l670_67083

/-- Given a line L1 with equation x + 3y + 4 = 0, prove that the line L2 with equation 3x - y - 5 = 0
    passes through the point (2, 1) and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 3 * y + 4 = 0) →  -- Equation of line L1
  (3 * 2 - 1 - 5 = 0) →  -- L2 passes through (2, 1)
  (3 * (1 / 3) = -1) →   -- Slopes are negative reciprocals
  (3 * x - y - 5 = 0) -- Equation of line L2
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l670_67083


namespace NUMINAMATH_CALUDE_right_triangle_area_l670_67072

theorem right_triangle_area (a b c : ℝ) (h1 : b = (2/3) * a) (h2 : b = (2/3) * c) 
  (h3 : a^2 + b^2 = c^2) : (1/2) * a * b = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l670_67072


namespace NUMINAMATH_CALUDE_triangle_problem_l670_67013

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l670_67013


namespace NUMINAMATH_CALUDE_total_peaches_is_273_l670_67088

/-- The number of monkeys in the zoo --/
def num_monkeys : ℕ := 36

/-- The number of peaches each monkey receives in the first scenario --/
def peaches_per_monkey_scenario1 : ℕ := 6

/-- The number of peaches left over in the first scenario --/
def peaches_left_scenario1 : ℕ := 57

/-- The number of peaches each monkey should receive in the second scenario --/
def peaches_per_monkey_scenario2 : ℕ := 9

/-- The number of monkeys that get nothing in the second scenario --/
def monkeys_with_no_peaches : ℕ := 5

/-- The number of peaches the last monkey gets in the second scenario --/
def peaches_for_last_monkey : ℕ := 3

/-- The total number of peaches --/
def total_peaches : ℕ := num_monkeys * peaches_per_monkey_scenario1 + peaches_left_scenario1

theorem total_peaches_is_273 : total_peaches = 273 := by
  sorry

#eval total_peaches

end NUMINAMATH_CALUDE_total_peaches_is_273_l670_67088


namespace NUMINAMATH_CALUDE_set_equality_l670_67011

def U := Set ℝ

def M : Set ℝ := {x | x > -1}

def N : Set ℝ := {x | -2 < x ∧ x < 3}

theorem set_equality : {x : ℝ | x ≤ -2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l670_67011


namespace NUMINAMATH_CALUDE_system_solution_l670_67059

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (x + z = 0) ∧ (y + z = -1) ∧ 
  (x = 1) ∧ (y = 0) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l670_67059


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_3_l670_67073

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 3

theorem first_year_after_2010_with_digit_sum_3 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2100 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_3_l670_67073


namespace NUMINAMATH_CALUDE_customers_without_tip_greasy_spoon_tip_problem_l670_67000

/-- The number of customers who didn't leave a tip at 'The Greasy Spoon' restaurant --/
theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  initial_customers + additional_customers - customers_with_tip

/-- Proof that 34 customers didn't leave a tip --/
theorem greasy_spoon_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_greasy_spoon_tip_problem_l670_67000


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_plus_5a_l670_67068

theorem factorization_of_a_squared_plus_5a (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_plus_5a_l670_67068


namespace NUMINAMATH_CALUDE_east_to_north_ratio_l670_67020

/-- Represents the number of tents in different areas of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  center : ℕ
  south : ℕ
  east : ℕ

/-- The conditions of the campsite as described in the problem -/
def campsite_conditions (c : Campsite) : Prop :=
  c.total = 900 ∧
  c.north = 100 ∧
  c.center = 4 * c.north ∧
  c.south = 200 ∧
  c.total = c.north + c.center + c.south + c.east

/-- The theorem stating the ratio of tents on the east side to the northernmost part -/
theorem east_to_north_ratio (c : Campsite) 
  (h : campsite_conditions c) : c.east = 2 * c.north :=
sorry

end NUMINAMATH_CALUDE_east_to_north_ratio_l670_67020


namespace NUMINAMATH_CALUDE_lexiCement_is_10_l670_67001

/-- The amount of cement used for Lexi's street -/
def lexiCement : ℝ := sorry

/-- The amount of cement used for Tess's street -/
def tessCement : ℝ := 5.1

/-- The total amount of cement used -/
def totalCement : ℝ := 15.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexiCement_is_10 : lexiCement = 10 :=
by
  have h1 : lexiCement = totalCement - tessCement := sorry
  sorry


end NUMINAMATH_CALUDE_lexiCement_is_10_l670_67001


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l670_67075

/-- The number of days in a month -/
def days_per_month : ℕ := 28

/-- The number of computers produced in 30 minutes -/
def computers_per_interval : ℕ := 3

/-- The number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 24 * 2

/-- Calculates the number of computers produced in a month -/
def computers_per_month : ℕ :=
  days_per_month * intervals_per_day * computers_per_interval

/-- Theorem stating that the number of computers produced per month is 4032 -/
theorem computers_produced_per_month :
  computers_per_month = 4032 := by
  sorry


end NUMINAMATH_CALUDE_computers_produced_per_month_l670_67075


namespace NUMINAMATH_CALUDE_persistent_is_two_l670_67056

/-- A number T is persistent if for any a, b, c, d ≠ 0, 1:
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T implies 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def IsPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

theorem persistent_is_two (T : ℝ) : IsPersistent T → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_persistent_is_two_l670_67056


namespace NUMINAMATH_CALUDE_smallest_land_fraction_150_members_l670_67090

/-- Represents a noble family with land division rules -/
structure NobleFamily :=
  (total_members : ℕ)
  (founder_land : ℝ)
  (divide_land : ℝ → ℕ → ℝ)
  (transfer_to_state : ℝ → ℝ)

/-- The smallest possible fraction of land a family member could receive -/
def smallest_land_fraction (family : NobleFamily) : ℚ :=
  1 / (2 * 3^49)

/-- Theorem stating the smallest possible fraction of land for a family of 150 members -/
theorem smallest_land_fraction_150_members 
  (family : NobleFamily) 
  (h_members : family.total_members = 150) :
  smallest_land_fraction family = 1 / (2 * 3^49) :=
sorry

end NUMINAMATH_CALUDE_smallest_land_fraction_150_members_l670_67090


namespace NUMINAMATH_CALUDE_optimal_rate_maximizes_income_l670_67044

/-- Represents the hotel's room pricing and occupancy model -/
structure HotelModel where
  totalRooms : ℕ
  baseRate : ℕ
  occupancyDecrease : ℕ
  rateIncrease : ℕ

/-- Calculates the number of occupied rooms based on the new rate -/
def occupiedRooms (model : HotelModel) (newRate : ℕ) : ℤ :=
  model.totalRooms - (newRate - model.baseRate) / model.rateIncrease * model.occupancyDecrease

/-- Calculates the total daily income based on the new rate -/
def dailyIncome (model : HotelModel) (newRate : ℕ) : ℕ :=
  newRate * (occupiedRooms model newRate).toNat

/-- The optimal rate that maximizes daily income -/
def optimalRate (model : HotelModel) : ℕ := model.baseRate + model.rateIncrease * (model.totalRooms / model.occupancyDecrease) / 2

/-- Theorem stating that the optimal rate maximizes daily income -/
theorem optimal_rate_maximizes_income (model : HotelModel) :
  model.totalRooms = 300 →
  model.baseRate = 200 →
  model.occupancyDecrease = 10 →
  model.rateIncrease = 20 →
  ∀ rate, dailyIncome model (optimalRate model) ≥ dailyIncome model rate := by
  sorry

#eval optimalRate { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 }
#eval dailyIncome { totalRooms := 300, baseRate := 200, occupancyDecrease := 10, rateIncrease := 20 } 400

end NUMINAMATH_CALUDE_optimal_rate_maximizes_income_l670_67044


namespace NUMINAMATH_CALUDE_area_ratio_of_specific_trapezoid_l670_67036

/-- Represents a trapezoid with extended legs -/
structure ExtendedTrapezoid where
  -- Length of the shorter base
  pq : ℝ
  -- Length of the longer base
  rs : ℝ
  -- Point where extended legs meet
  t : Point

/-- Calculates the ratio of the area of triangle TPQ to the area of trapezoid PQRS -/
def areaRatio (trap : ExtendedTrapezoid) : ℚ :=
  100 / 429

/-- Theorem stating the area ratio for the given trapezoid -/
theorem area_ratio_of_specific_trapezoid :
  ∃ (trap : ExtendedTrapezoid),
    trap.pq = 10 ∧ trap.rs = 23 ∧ areaRatio trap = 100 / 429 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_specific_trapezoid_l670_67036


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l670_67063

theorem partial_fraction_decomposition (x : ℝ) 
  (h1 : x ≠ 7/8) (h2 : x ≠ 4/5) (h3 : x ≠ 1/2) :
  (306 * x^2 - 450 * x + 162) / ((8*x-7)*(5*x-4)*(2*x-1)) = 
  9 / (8*x-7) + 6 / (5*x-4) + 3 / (2*x-1) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l670_67063


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l670_67096

theorem complex_magnitude_one (n : ℕ) (a : ℝ) (z : ℂ)
  (h_n : n ≥ 2)
  (h_a : 0 < a ∧ a < (n + 1 : ℝ) / (n - 1 : ℝ))
  (h_z : z^(n+1) - a * z^n + a * z - 1 = 0) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l670_67096


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l670_67093

theorem least_positive_integer_with_given_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  x % 9 = 8 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 ∧ y % 9 = 8 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l670_67093


namespace NUMINAMATH_CALUDE_average_donation_l670_67009

theorem average_donation (total_people : ℝ) (h_total_positive : total_people > 0) : 
  let group1_fraction : ℝ := 1 / 10
  let group2_fraction : ℝ := 3 / 4
  let group3_fraction : ℝ := 1 - group1_fraction - group2_fraction
  let donation1 : ℝ := 200
  let donation2 : ℝ := 100
  let donation3 : ℝ := 50
  let total_donation : ℝ := 
    group1_fraction * donation1 * total_people + 
    group2_fraction * donation2 * total_people + 
    group3_fraction * donation3 * total_people
  total_donation / total_people = 102.5 := by
sorry

end NUMINAMATH_CALUDE_average_donation_l670_67009


namespace NUMINAMATH_CALUDE_card_distribution_proof_l670_67058

/-- Represents the number of cards each player has -/
structure CardDistribution :=
  (alfred : ℕ)
  (bruno : ℕ)
  (christophe : ℕ)
  (damien : ℕ)

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- Redistribution function for Alfred -/
def redistributeAlfred (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred - d.alfred / 2,
    bruno := d.bruno + d.alfred / 4,
    christophe := d.christophe + d.alfred / 4,
    damien := d.damien }

/-- Redistribution function for Bruno -/
def redistributeBruno (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.bruno / 4,
    bruno := d.bruno - d.bruno / 2,
    christophe := d.christophe + d.bruno / 4,
    damien := d.damien }

/-- Redistribution function for Christophe -/
def redistributeChristophe (d : CardDistribution) : CardDistribution :=
  { alfred := d.alfred + d.christophe / 4,
    bruno := d.bruno + d.christophe / 4,
    christophe := d.christophe - d.christophe / 2,
    damien := d.damien }

/-- The initial distribution of cards -/
def initialDistribution : CardDistribution :=
  { alfred := 4, bruno := 7, christophe := 13, damien := 8 }

theorem card_distribution_proof :
  let finalDist := redistributeChristophe (redistributeBruno (redistributeAlfred initialDistribution))
  (finalDist.alfred = finalDist.bruno) ∧
  (finalDist.bruno = finalDist.christophe) ∧
  (finalDist.christophe = finalDist.damien) ∧
  (finalDist.alfred + finalDist.bruno + finalDist.christophe + finalDist.damien = totalCards) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_proof_l670_67058


namespace NUMINAMATH_CALUDE_johnny_tables_l670_67042

/-- The number of tables that can be built given a total number of planks and planks required per table -/
def tables_built (total_planks : ℕ) (planks_per_table : ℕ) : ℕ :=
  total_planks / planks_per_table

/-- Theorem: Given 45 planks of wood and 9 planks required per table, 5 tables can be built -/
theorem johnny_tables : tables_built 45 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_tables_l670_67042


namespace NUMINAMATH_CALUDE_correct_number_of_pupils_l670_67086

/-- The number of pupils in a class where an error in one pupil's marks
    caused the class average to increase by half a mark. -/
def number_of_pupils : ℕ :=
  -- We define this as 20, which is the value we want to prove
  20

/-- The increase in one pupil's marks due to the error -/
def mark_increase : ℕ := 10

/-- The increase in the class average due to the error -/
def average_increase : ℚ := 1/2

theorem correct_number_of_pupils :
  mark_increase = (number_of_pupils : ℚ) * average_increase :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_pupils_l670_67086


namespace NUMINAMATH_CALUDE_problem_solution_l670_67043

theorem problem_solution : 
  (Real.sqrt 48 - Real.sqrt 27 + Real.sqrt (1/3) = (4 * Real.sqrt 3) / 3) ∧
  ((Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt 5 + Real.sqrt 2) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l670_67043


namespace NUMINAMATH_CALUDE_key_arrangement_count_l670_67082

/-- The number of keys on the keychain -/
def total_keys : ℕ := 6

/-- The number of effective units to arrange (treating the adjacent pair as one unit) -/
def effective_units : ℕ := total_keys - 1

/-- The number of ways to arrange the adjacent pair -/
def adjacent_pair_arrangements : ℕ := 2

/-- The number of distinct circular arrangements of n objects -/
def circular_arrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The total number of distinct arrangements -/
def total_arrangements : ℕ := circular_arrangements effective_units * adjacent_pair_arrangements

theorem key_arrangement_count : total_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_key_arrangement_count_l670_67082


namespace NUMINAMATH_CALUDE_fraction_is_one_fifth_l670_67017

/-- The total number of states in the collection -/
def total_states : ℕ := 50

/-- The number of states that joined the union between 1790 and 1809 -/
def states_1790_1809 : ℕ := 10

/-- The fraction of states that joined between 1790 and 1809 -/
def fraction_1790_1809 : ℚ := states_1790_1809 / total_states

theorem fraction_is_one_fifth : fraction_1790_1809 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_one_fifth_l670_67017


namespace NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l670_67077

/-- Given a circle and a line, proves that the radius of the circle is 11 
    when the chord formed by their intersection has length 6 -/
theorem circle_line_intersection_chord_length (a : ℝ) : 
  (∃ x y : ℝ, (x + 2)^2 + (y - 2)^2 = a ∧ x + y + 2 = 0) →  -- Circle intersects line
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 2)^2 + (y₁ - 2)^2 = a ∧ 
    (x₂ + 2)^2 + (y₂ - 2)^2 = a ∧ 
    x₁ + y₁ + 2 = 0 ∧ 
    x₂ + y₂ + 2 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) →  -- Chord length is 6
  a = 11 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_chord_length_l670_67077


namespace NUMINAMATH_CALUDE_probability_black_white_balls_l670_67095

/-- The probability of picking one black ball and one white ball from a jar -/
theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = black_balls + white_balls + green_balls)
  (h2 : black_balls = 3)
  (h3 : white_balls = 3)
  (h4 : green_balls = 1) :
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_probability_black_white_balls_l670_67095


namespace NUMINAMATH_CALUDE_fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l670_67024

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular : triangular_number 15 = 120 := by sorry

/-- The sum of the 15th and 16th triangular numbers is 256 -/
theorem sum_fifteenth_sixteenth_triangular : 
  triangular_number 15 + triangular_number 16 = 256 := by sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l670_67024


namespace NUMINAMATH_CALUDE_final_jellybean_count_l670_67067

def jellybean_count (initial : ℕ) (first_removal : ℕ) (addition : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + addition - second_removal

theorem final_jellybean_count :
  jellybean_count 37 15 5 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_jellybean_count_l670_67067


namespace NUMINAMATH_CALUDE_sphere_volume_from_intersection_l670_67045

/-- Given a sphere intersected by a plane at distance 1 from its center,
    creating a cross-sectional area of π, prove that its volume is (8√2π)/3. -/
theorem sphere_volume_from_intersection (r : ℝ) : 
  (r^2 - 1^2 = 1^2) →   -- Pythagorean theorem for the right triangle
  (π * 1^2 = π) →       -- Cross-sectional area is π
  ((4/3) * π * r^3 = (8 * Real.sqrt 2 * π) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_intersection_l670_67045


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l670_67008

/-- The area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area : ℝ :=
let A : ℝ × ℝ := (0, 0)
let B : ℝ × ℝ := (4, 2)
let C : ℝ × ℝ := (4, -4)
let triangle_area := Real.sqrt 5 * 4
triangle_area

/-- Proof that the area of the triangle formed by points (0, 0), (4, 2), and (4, -4) is 4√5 square units -/
theorem triangle_area_proof : triangle_area = Real.sqrt 5 * 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l670_67008


namespace NUMINAMATH_CALUDE_total_legs_l670_67010

/-- The number of bees -/
def num_bees : ℕ := 50

/-- The number of ants -/
def num_ants : ℕ := 35

/-- The number of spiders -/
def num_spiders : ℕ := 20

/-- The number of legs a bee has -/
def bee_legs : ℕ := 6

/-- The number of legs an ant has -/
def ant_legs : ℕ := 6

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- Theorem stating the total number of legs -/
theorem total_legs : 
  num_bees * bee_legs + num_ants * ant_legs + num_spiders * spider_legs = 670 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_l670_67010


namespace NUMINAMATH_CALUDE_remaining_pie_portion_l670_67038

theorem remaining_pie_portion (carlos_share : Real) (maria_fraction : Real) : 
  carlos_share = 0.8 →
  maria_fraction = 0.25 →
  (1 - carlos_share) * (1 - maria_fraction) = 0.15 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_pie_portion_l670_67038


namespace NUMINAMATH_CALUDE_valuable_files_count_l670_67015

def initial_download : ℕ := 800
def first_deletion_rate : ℚ := 70 / 100
def second_download : ℕ := 400
def second_deletion_rate : ℚ := 3 / 5

theorem valuable_files_count : 
  (initial_download - (initial_download * first_deletion_rate).floor) + 
  (second_download - (second_download * second_deletion_rate).floor) = 400 :=
by sorry

end NUMINAMATH_CALUDE_valuable_files_count_l670_67015


namespace NUMINAMATH_CALUDE_lab_capacity_l670_67071

theorem lab_capacity (total_capacity : ℕ) (total_stations : ℕ) (two_student_stations : ℕ) 
  (h1 : total_capacity = 38)
  (h2 : total_stations = 16)
  (h3 : two_student_stations = 10) :
  total_capacity - (2 * two_student_stations) = 18 := by
  sorry

end NUMINAMATH_CALUDE_lab_capacity_l670_67071


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l670_67084

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := side * h / 2
  area = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l670_67084


namespace NUMINAMATH_CALUDE_silver_cube_side_length_l670_67085

/-- Proves that a silver cube sold for $4455 at 110% of its silver value, 
    where a cubic inch of silver weighs 6 ounces and each ounce of silver 
    sells for $25, has a side length of 3 inches. -/
theorem silver_cube_side_length :
  let selling_price : ℝ := 4455
  let markup_percentage : ℝ := 1.10
  let weight_per_cubic_inch : ℝ := 6
  let price_per_ounce : ℝ := 25
  let side_length : ℝ := (selling_price / markup_percentage / price_per_ounce / weight_per_cubic_inch) ^ (1/3)
  side_length = 3 := by sorry

end NUMINAMATH_CALUDE_silver_cube_side_length_l670_67085


namespace NUMINAMATH_CALUDE_community_service_arrangements_l670_67041

def number_of_arrangements (n m k : ℕ) (a b : Fin n) : ℕ :=
  let without_ab := Nat.choose m k
  let with_one := 2 * Nat.choose (m - 1) (k - 1)
  without_ab + 2 * with_one

theorem community_service_arrangements :
  number_of_arrangements 8 6 3 0 1 = 80 := by
  sorry

end NUMINAMATH_CALUDE_community_service_arrangements_l670_67041


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l670_67039

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2

-- State the theorem
theorem derivative_f_at_2 : 
  (deriv f) 2 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l670_67039


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l670_67016

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x^2 + 1/x^2 = 5) : x^4 + 1/x^4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l670_67016


namespace NUMINAMATH_CALUDE_monotonic_function_characterization_l670_67014

-- Define the types of our functions
def MonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y

def StrictlyMonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem monotonic_function_characterization 
  (u : ℝ → ℝ) 
  (h_u_monotonic : MonotonicFunction u) 
  (h_exists_f : ∃ f : ℝ → ℝ, 
    StrictlyMonotonicFunction f ∧ 
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, u x = Real.exp (k * x) := by
sorry

end NUMINAMATH_CALUDE_monotonic_function_characterization_l670_67014


namespace NUMINAMATH_CALUDE_sin_sum_angles_l670_67048

theorem sin_sum_angles (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1/4)
  (h2 : Real.cos α + Real.sin β = -8/5) : 
  Real.sin (α + β) = 249/800 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_angles_l670_67048


namespace NUMINAMATH_CALUDE_remaining_truck_capacity_l670_67050

/-- Calculates the remaining capacity of a truck after loading lemons -/
theorem remaining_truck_capacity 
  (max_load : ℕ)           -- Maximum load capacity of the truck
  (bag_weight : ℕ)         -- Weight of each bag of lemons
  (num_bags : ℕ)           -- Number of bags of lemons
  (h1 : max_load = 900)    -- Given maximum load is 900 kg
  (h2 : bag_weight = 8)    -- Given weight of each bag is 8 kg
  (h3 : num_bags = 100)    -- Given number of bags is 100
  : max_load - (bag_weight * num_bags) = 100 := by
  sorry

#check remaining_truck_capacity

end NUMINAMATH_CALUDE_remaining_truck_capacity_l670_67050


namespace NUMINAMATH_CALUDE_largest_divisors_ratio_l670_67053

theorem largest_divisors_ratio (N : ℕ) (h1 : N > 1) 
  (h2 : ∃ (a : ℕ), a ∣ N ∧ 6 * a ∣ N ∧ a ≠ 1 ∧ 6 * a ≠ N) :
  (N / 2) / (N / 3) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_largest_divisors_ratio_l670_67053


namespace NUMINAMATH_CALUDE_dog_count_l670_67037

theorem dog_count (total : ℕ) (cats : ℕ) (h1 : total = 17) (h2 : cats = 8) :
  total - cats = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l670_67037


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l670_67052

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = a (n - 1) + a (n - 2)

theorem fourth_number_in_sequence
  (a : ℕ → ℕ)
  (h_fib : fibonacci_like_sequence a)
  (h_7 : a 7 = 42)
  (h_9 : a 9 = 110) :
  a 4 = 10 :=
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l670_67052


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l670_67022

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (2 - x) - 3 = x / (x - 2) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l670_67022


namespace NUMINAMATH_CALUDE_special_function_value_l670_67070

/-- A function satisfying f(x + y) = f(x) + f(y) + 2xy for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l670_67070


namespace NUMINAMATH_CALUDE_arrangements_count_l670_67064

/-- Represents the number of male students -/
def num_male_students : ℕ := 3

/-- Represents the number of female students -/
def num_female_students : ℕ := 3

/-- Represents the total number of students -/
def total_students : ℕ := num_male_students + num_female_students

/-- Represents whether female students can stand at the ends of the row -/
def female_at_ends : Prop := False

/-- Represents whether female students A and B can be adjacent to female student C -/
def female_AB_adjacent_C : Prop := False

/-- Calculates the number of different arrangements given the conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 given the conditions -/
theorem arrangements_count :
  num_male_students = 3 ∧
  num_female_students = 3 ∧
  total_students = num_male_students + num_female_students ∧
  ¬female_at_ends ∧
  ¬female_AB_adjacent_C →
  num_arrangements = 144 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_l670_67064


namespace NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_l670_67005

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_equality_sufficient_not_necessary :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
sorry

end NUMINAMATH_CALUDE_floor_equality_sufficient_not_necessary_l670_67005


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l670_67046

/-- A scalene triangle with given properties -/
structure ScaleneTriangle where
  -- A, B, C are the angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- rA, rB, rC are the exradii
  rA : ℝ
  rB : ℝ
  rC : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  exradii_condition : 20 * (rB^2 * rC^2 + rC^2 * rA^2 + rA^2 * rB^2) = 19 * (rA * rB * rC)^2
  tan_sum : Real.tan (A/2) + Real.tan (B/2) + Real.tan (C/2) = 2.019
  inradius : ℝ := 1

/-- The area of a scalene triangle with the given properties is 2019/25 -/
theorem area_of_special_triangle (t : ScaleneTriangle) : 
  (2 * t.inradius * (Real.tan (t.A/2) + Real.tan (t.B/2) + Real.tan (t.C/2))) = 2019/25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l670_67046


namespace NUMINAMATH_CALUDE_parallelogram_solution_l670_67034

-- Define the parallelogram EFGH
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ

-- Define the specific parallelogram from the problem
def specificParallelogram : Parallelogram where
  EF := 45
  FG := fun y ↦ 4 * y^2
  GH := fun x ↦ 3 * x + 6
  HE := 32

-- Theorem statement
theorem parallelogram_solution (p : Parallelogram) 
  (h1 : p = specificParallelogram) : 
  ∃ (x y : ℝ), p.GH x = p.EF ∧ p.FG y = p.HE ∧ x = 13 ∧ y = 2 * Real.sqrt 2 := by
  sorry

#check parallelogram_solution

end NUMINAMATH_CALUDE_parallelogram_solution_l670_67034


namespace NUMINAMATH_CALUDE_polar_bears_research_l670_67066

theorem polar_bears_research (time_per_round : ℕ) (sunday_rounds : ℕ) (total_time : ℕ) :
  time_per_round = 30 →
  sunday_rounds = 15 →
  total_time = 780 →
  ∃ (saturday_additional_rounds : ℕ),
    saturday_additional_rounds = 10 ∧
    total_time = time_per_round * (1 + saturday_additional_rounds + sunday_rounds) :=
by sorry

end NUMINAMATH_CALUDE_polar_bears_research_l670_67066


namespace NUMINAMATH_CALUDE_inverse_proportion_l670_67028

/-- Given that x is inversely proportional to y, prove that if x = 4 when y = 2, then x = -8/5 when y = -5 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  -5 * (-8/5 : ℝ) = k := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_l670_67028


namespace NUMINAMATH_CALUDE_magazines_per_box_l670_67094

theorem magazines_per_box (total_magazines : ℕ) (num_boxes : ℕ) (h1 : total_magazines = 63) (h2 : num_boxes = 7) :
  total_magazines / num_boxes = 9 := by
  sorry

end NUMINAMATH_CALUDE_magazines_per_box_l670_67094


namespace NUMINAMATH_CALUDE_B_roster_l670_67040

def A : Set ℤ := {-2, 2, 3, 4}

def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem B_roster : B = {4, 9, 16} := by
  sorry

end NUMINAMATH_CALUDE_B_roster_l670_67040


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l670_67033

/-- A function that calculates the probability of the given conditions for a given n -/
noncomputable def probability (n : ℕ) : ℝ :=
  ((n - 2)^3 + 3 * (n - 2) * (2 * n - 4)) / n^3

/-- The theorem stating that 12 is the smallest n satisfying the probability condition -/
theorem smallest_n_satisfying_condition :
  ∀ k : ℕ, k < 12 → probability k ≤ 3/4 ∧ probability 12 > 3/4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l670_67033


namespace NUMINAMATH_CALUDE_sum_in_terms_of_x_l670_67081

theorem sum_in_terms_of_x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) :
  x + y + z = 16 * x := by
sorry

end NUMINAMATH_CALUDE_sum_in_terms_of_x_l670_67081


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_p_and_q_l670_67025

theorem p_necessary_not_sufficient_for_p_and_q :
  (∃ p q : Prop, (p ∧ q → p) ∧ ¬(p → p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_p_and_q_l670_67025


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l670_67027

/-- Proves the volume of fuel A in a partially filled tank --/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 212 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), volume_a = 98 ∧
    ∃ (volume_b : ℝ), volume_a + volume_b = tank_capacity ∧
      ethanol_a * volume_a + ethanol_b * volume_b = total_ethanol :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l670_67027


namespace NUMINAMATH_CALUDE_grid_midpoint_theorem_l670_67062

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) 
  (h : points.card = 5) :
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ 
  (∃ m : ℤ × ℤ, m.1 * 2 = p1.1 + p2.1 ∧ m.2 * 2 = p1.2 + p2.2) :=
sorry

end NUMINAMATH_CALUDE_grid_midpoint_theorem_l670_67062


namespace NUMINAMATH_CALUDE_triangle_proof_l670_67097

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_proof (ABC : Triangle) 
  (h1 : ABC.B.sin = 1/3)
  (h2 : ABC.a^2 - ABC.b^2 + ABC.c^2 = 2 ∨ ABC.a * ABC.c * ABC.B.cos = -1)
  (h3 : ABC.A.sin * ABC.C.sin = Real.sqrt 2 / 3) :
  (ABC.a * ABC.c = 3 * Real.sqrt 2 / 4) ∧ 
  (ABC.b = 1/2) := by
sorry

end NUMINAMATH_CALUDE_triangle_proof_l670_67097


namespace NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l670_67031

theorem real_roots_iff_a_leq_two (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_a_leq_two_l670_67031


namespace NUMINAMATH_CALUDE_min_value_fraction_lower_bound_achievable_l670_67092

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (x + y) / (x * y * z) ≥ 16 := by
  sorry

theorem lower_bound_achievable : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ (x + y) / (x * y * z) = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_lower_bound_achievable_l670_67092


namespace NUMINAMATH_CALUDE_spinner_probability_l670_67047

theorem spinner_probability : 
  ∀ (p_C : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + p_C + p_C = 1 →
  p_C = 5 / 24 :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l670_67047


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l670_67019

/-- Configuration of squares and rectangles -/
structure SquareRectConfig where
  inner_square_side : ℝ
  rect_short_side : ℝ
  rect_long_side : ℝ

/-- The configuration satisfies the problem conditions -/
def valid_config (c : SquareRectConfig) : Prop :=
  c.inner_square_side > 0 ∧
  c.rect_short_side > 0 ∧
  c.rect_long_side > 0 ∧
  c.inner_square_side + 2 * c.rect_short_side = 3 * c.inner_square_side ∧
  c.inner_square_side + c.rect_long_side = 3 * c.inner_square_side

theorem rectangle_ratio_is_two (c : SquareRectConfig) (h : valid_config c) :
  c.rect_long_side / c.rect_short_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l670_67019


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l670_67060

/-- A quadratic function of the form f(x) = x^2 + ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the unique quadratic function satisfying the given condition -/
theorem unique_quadratic_function (a b : ℝ) :
  (∀ x, (f a b (f a b x - x)) / (f a b x) = x^2 + 2023*x + 1777) →
  a = 2025 ∧ b = 249 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l670_67060


namespace NUMINAMATH_CALUDE_clara_owes_mandy_l670_67029

/-- The amount Clara owes Mandy for cleaning rooms -/
def amount_owed (rate : ℚ) (rooms : ℚ) (discount_threshold : ℚ) (discount_rate : ℚ) : ℚ :=
  let base_amount := rate * rooms
  if rooms > discount_threshold then
    base_amount * (1 - discount_rate)
  else
    base_amount

/-- Theorem stating the amount Clara owes Mandy -/
theorem clara_owes_mandy :
  let rate : ℚ := 15 / 4
  let rooms : ℚ := 12 / 5
  let discount_threshold : ℚ := 2
  let discount_rate : ℚ := 1 / 10
  amount_owed rate rooms discount_threshold discount_rate = 81 / 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_owes_mandy_l670_67029


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l670_67030

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l670_67030


namespace NUMINAMATH_CALUDE_sin_135_degrees_l670_67026

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l670_67026


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_l670_67054

theorem sqrt_five_irrational : Irrational (Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_irrational_l670_67054
