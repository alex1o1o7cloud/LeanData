import Mathlib

namespace NUMINAMATH_CALUDE_montero_trip_feasibility_l1900_190012

/-- Represents the parameters of Mr. Montero's trip -/
structure TripParameters where
  normal_efficiency : Real
  traffic_efficiency_reduction : Real
  total_distance : Real
  traffic_distance : Real
  initial_gas : Real
  gas_price : Real
  price_increase : Real
  budget : Real

/-- Calculates whether Mr. Montero can complete his trip within budget -/
def can_complete_trip (params : TripParameters) : Prop :=
  let reduced_efficiency := params.normal_efficiency * (1 - params.traffic_efficiency_reduction)
  let normal_distance := params.total_distance - params.traffic_distance
  let gas_needed := normal_distance / params.normal_efficiency + 
                    params.traffic_distance / reduced_efficiency
  let gas_to_buy := gas_needed - params.initial_gas
  let half_trip_gas := (params.total_distance / 2) / params.normal_efficiency - params.initial_gas
  let first_half_cost := min half_trip_gas gas_to_buy * params.gas_price
  let second_half_cost := max 0 (gas_to_buy - half_trip_gas) * (params.gas_price * (1 + params.price_increase))
  first_half_cost + second_half_cost ≤ params.budget

theorem montero_trip_feasibility :
  let params : TripParameters := {
    normal_efficiency := 20,
    traffic_efficiency_reduction := 0.2,
    total_distance := 600,
    traffic_distance := 100,
    initial_gas := 8,
    gas_price := 2.5,
    price_increase := 0.1,
    budget := 75
  }
  can_complete_trip params := by sorry

end NUMINAMATH_CALUDE_montero_trip_feasibility_l1900_190012


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l1900_190003

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct / total) * 100 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l1900_190003


namespace NUMINAMATH_CALUDE_square_floor_tiles_l1900_190048

/-- Given a square floor with side length s, where tiles along the diagonals
    are marked blue, prove that if there are 225 blue tiles, then the total
    number of tiles on the floor is 12769. -/
theorem square_floor_tiles (s : ℕ) : 
  (2 * s - 1 = 225) → s^2 = 12769 := by sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l1900_190048


namespace NUMINAMATH_CALUDE_company_workforce_after_hiring_l1900_190070

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real)
  (additional_male_hires : Nat)
  (final_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_hires = 26 →
  final_female_percentage = 0.55 →
  ∃ (initial_total : Nat),
    (initial_total : Real) * initial_female_percentage + 
    (initial_total : Real) * (1 - initial_female_percentage) = initial_total ∧
    (initial_total + additional_male_hires : Real) * final_female_percentage + 
    ((initial_total : Real) * (1 - initial_female_percentage) + additional_male_hires) = 
    initial_total + additional_male_hires ∧
    initial_total + additional_male_hires = 312 :=
by sorry

end NUMINAMATH_CALUDE_company_workforce_after_hiring_l1900_190070


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1900_190002

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n = 4 ∧ 
  (15 ∣ (9679 - n)) ∧ 
  ∀ (m : ℕ), m < n → ¬(15 ∣ (9679 - m)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1900_190002


namespace NUMINAMATH_CALUDE_find_x_value_l1900_190020

theorem find_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-1, 1} → 
  B = {0, 1, x-1} → 
  A ⊆ B → 
  x = 0 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l1900_190020


namespace NUMINAMATH_CALUDE_f_max_at_zero_l1900_190071

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_max_at_zero :
  ∃ (a : ℝ), ∀ (x : ℝ), f x ≤ f a ∧ a = 0 :=
sorry

end NUMINAMATH_CALUDE_f_max_at_zero_l1900_190071


namespace NUMINAMATH_CALUDE_power_function_through_point_l1900_190081

theorem power_function_through_point (n : ℝ) : 
  (∀ x y : ℝ, y = x^n → (x = 2 ∧ y = 8) → n = 3) :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1900_190081


namespace NUMINAMATH_CALUDE_min_socks_in_box_min_socks_even_black_l1900_190068

/-- Represents a box of socks -/
structure SockBox where
  red : ℕ
  black : ℕ

/-- The probability of drawing two red socks from the box -/
def prob_two_red (box : SockBox) : ℚ :=
  (box.red / (box.red + box.black)) * ((box.red - 1) / (box.red + box.black - 1))

/-- The total number of socks in the box -/
def total_socks (box : SockBox) : ℕ := box.red + box.black

theorem min_socks_in_box :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 → total_socks box ≤ total_socks other :=
sorry

theorem min_socks_even_black :
  ∃ (box : SockBox), prob_two_red box = 1/2 ∧ box.black % 2 = 0 ∧
    ∀ (other : SockBox), prob_two_red other = 1/2 ∧ other.black % 2 = 0 →
      total_socks box ≤ total_socks other :=
sorry

end NUMINAMATH_CALUDE_min_socks_in_box_min_socks_even_black_l1900_190068


namespace NUMINAMATH_CALUDE_one_pair_percentage_l1900_190052

def five_digit_numbers : ℕ := 90000

def numbers_with_one_pair : ℕ := 10 * 10 * 9 * 8 * 7

theorem one_pair_percentage : 
  (numbers_with_one_pair : ℚ) / five_digit_numbers * 100 = 56 :=
by sorry

end NUMINAMATH_CALUDE_one_pair_percentage_l1900_190052


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1900_190006

theorem invalid_external_diagonals : ¬ ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (a^2 + b^2 = 5^2 ∧ b^2 + c^2 = 6^2 ∧ a^2 + c^2 = 8^2) :=
sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1900_190006


namespace NUMINAMATH_CALUDE_geometric_sequence_17th_term_l1900_190091

/-- Given a geometric sequence where a₅ = 5 and a₁₁ = 40, prove that a₁₇ = 320 -/
theorem geometric_sequence_17th_term (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h2 : a 5 = 5) (h3 : a 11 = 40) : a 17 = 320 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_17th_term_l1900_190091


namespace NUMINAMATH_CALUDE_figurine_cost_l1900_190022

/-- The cost of a single figurine given Annie's purchases -/
theorem figurine_cost (num_tvs : ℕ) (tv_cost : ℕ) (num_figurines : ℕ) (total_spent : ℕ) : 
  num_tvs = 5 → 
  tv_cost = 50 → 
  num_figurines = 10 → 
  total_spent = 260 → 
  (total_spent - num_tvs * tv_cost) / num_figurines = 1 := by
sorry

end NUMINAMATH_CALUDE_figurine_cost_l1900_190022


namespace NUMINAMATH_CALUDE_toothpick_grids_count_l1900_190082

/-- Calculates the number of toothpicks needed for a grid -/
def toothpicks_for_grid (length : ℕ) (width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- The total number of toothpicks for two separate grids -/
def total_toothpicks (outer_length outer_width inner_length inner_width : ℕ) : ℕ :=
  toothpicks_for_grid outer_length outer_width + toothpicks_for_grid inner_length inner_width

theorem toothpick_grids_count :
  total_toothpicks 80 40 30 20 = 7770 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grids_count_l1900_190082


namespace NUMINAMATH_CALUDE_intersection_point_correct_l1900_190051

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a line in 2D space defined by a point and a direction vector -/
structure Line2D where
  point : Point2D
  direction : Point2D

/-- The first line -/
def line1 : Line2D := {
  point := { x := 3, y := 0 },
  direction := { x := 1, y := 2 }
}

/-- The second line -/
def line2 : Line2D := {
  point := { x := -1, y := 4 },
  direction := { x := 3, y := -1 }
}

/-- The proposed intersection point -/
def intersectionPoint : Point2D := {
  x := 30 / 7,
  y := 18 / 7
}

/-- Function to check if a point lies on a line -/
def isPointOnLine (p : Point2D) (l : Line2D) : Prop :=
  ∃ t : ℚ, p.x = l.point.x + t * l.direction.x ∧ p.y = l.point.y + t * l.direction.y

/-- Theorem stating that the proposed intersection point lies on both lines -/
theorem intersection_point_correct :
  isPointOnLine intersectionPoint line1 ∧ isPointOnLine intersectionPoint line2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l1900_190051


namespace NUMINAMATH_CALUDE_games_missed_l1900_190030

theorem games_missed (total_games attended_games : ℕ) (h1 : total_games = 31) (h2 : attended_games = 13) :
  total_games - attended_games = 18 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l1900_190030


namespace NUMINAMATH_CALUDE_largest_size_percentage_longer_than_smallest_l1900_190049

-- Define the shoe size range
def min_size : ℕ := 8
def max_size : ℕ := 17

-- Define the length increase per size
def length_increase_per_size : ℚ := 1 / 5

-- Define the length of size 15 shoe
def size_15_length : ℚ := 21 / 2  -- 10.4 as a rational number

-- Function to calculate shoe length given size
def shoe_length (size : ℕ) : ℚ :=
  size_15_length + (size - 15 : ℚ) * length_increase_per_size

-- Theorem statement
theorem largest_size_percentage_longer_than_smallest :
  (shoe_length max_size - shoe_length min_size) / shoe_length min_size = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_size_percentage_longer_than_smallest_l1900_190049


namespace NUMINAMATH_CALUDE_bertolli_farm_produce_difference_l1900_190025

theorem bertolli_farm_produce_difference : 
  let tomatoes : ℕ := 2073
  let corn : ℕ := 4112
  let onions : ℕ := 985
  (tomatoes + corn) - onions = 5200 := by sorry

end NUMINAMATH_CALUDE_bertolli_farm_produce_difference_l1900_190025


namespace NUMINAMATH_CALUDE_alfred_ranking_bounds_l1900_190098

/-- Represents a participant in the Generic Math Tournament -/
structure Participant where
  algebra_rank : Nat
  combinatorics_rank : Nat
  geometry_rank : Nat

/-- The total number of participants in the tournament -/
def total_participants : Nat := 99

/-- Alfred's rankings in each subject -/
def alfred : Participant :=
  { algebra_rank := 16
  , combinatorics_rank := 30
  , geometry_rank := 23 }

/-- Calculate the total score of a participant -/
def total_score (p : Participant) : Nat :=
  p.algebra_rank + p.combinatorics_rank + p.geometry_rank

/-- The best possible ranking Alfred could achieve -/
def best_ranking : Nat := 1

/-- The worst possible ranking Alfred could achieve -/
def worst_ranking : Nat := 67

theorem alfred_ranking_bounds :
  (∀ p : Participant, p ≠ alfred → total_score p ≠ total_score alfred) →
  (best_ranking = 1 ∧ worst_ranking = 67) :=
by sorry

end NUMINAMATH_CALUDE_alfred_ranking_bounds_l1900_190098


namespace NUMINAMATH_CALUDE_fraction_less_than_two_l1900_190036

theorem fraction_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_two_l1900_190036


namespace NUMINAMATH_CALUDE_circus_show_acrobats_l1900_190054

/-- Represents the number of acrobats in the circus show. -/
def numAcrobats : ℕ := 2

/-- Represents the number of elephants in the circus show. -/
def numElephants : ℕ := 14

/-- Represents the number of clowns in the circus show. -/
def numClowns : ℕ := 14

/-- The total number of legs observed in the circus show. -/
def totalLegs : ℕ := 88

/-- The total number of heads observed in the circus show. -/
def totalHeads : ℕ := 30

theorem circus_show_acrobats :
  (2 * numAcrobats + 4 * numElephants + 2 * numClowns = totalLegs) ∧
  (numAcrobats + numElephants + numClowns = totalHeads) ∧
  (numAcrobats = 2) := by sorry

end NUMINAMATH_CALUDE_circus_show_acrobats_l1900_190054


namespace NUMINAMATH_CALUDE_largest_a_for_integer_solution_l1900_190087

theorem largest_a_for_integer_solution : 
  ∃ (a : ℝ), ∀ (b : ℝ), 
    (∃ (x y : ℤ), x - 4*y = 1 ∧ a*x + 3*y = 1) ∧
    (∀ (x y : ℤ), b*x + 3*y = 1 → x - 4*y = 1 → b ≤ a) ∧
    a = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_a_for_integer_solution_l1900_190087


namespace NUMINAMATH_CALUDE_second_replaced_man_age_l1900_190059

theorem second_replaced_man_age 
  (n : ℕ) 
  (age_increase : ℝ) 
  (first_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : n = 15)
  (h2 : age_increase = 2)
  (h3 : first_replaced_age = 21)
  (h4 : new_men_avg_age = 37) :
  ∃ (second_replaced_age : ℕ),
    (n : ℝ) * age_increase = 
      2 * new_men_avg_age - (first_replaced_age : ℝ) - (second_replaced_age : ℝ) ∧
    second_replaced_age = 23 :=
by sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_l1900_190059


namespace NUMINAMATH_CALUDE_picnic_blanket_side_length_l1900_190050

theorem picnic_blanket_side_length 
  (number_of_blankets : ℕ) 
  (folds : ℕ) 
  (total_folded_area : ℝ) 
  (L : ℝ) :
  number_of_blankets = 3 →
  folds = 4 →
  total_folded_area = 48 →
  (number_of_blankets : ℝ) * (L^2 / 2^folds) = total_folded_area →
  L = 16 :=
by sorry

end NUMINAMATH_CALUDE_picnic_blanket_side_length_l1900_190050


namespace NUMINAMATH_CALUDE_one_square_covered_l1900_190007

/-- Represents a square on the checkerboard -/
structure Square where
  x : ℕ
  y : ℕ

/-- Represents the circular disc -/
structure Disc where
  center : Square
  diameter : ℝ

/-- Determines if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Prop :=
  (s.x - d.center.x)^2 + (s.y - d.center.y)^2 ≤ (d.diameter / 2)^2

/-- The checkerboard -/
def checkerboard : Set Square :=
  {s | s.x ≤ 8 ∧ s.y ≤ 8}

theorem one_square_covered (d : Disc) :
  d.diameter = Real.sqrt 2 →
  d.center ∈ checkerboard →
  ∃! s : Square, s ∈ checkerboard ∧ is_covered s d :=
sorry

end NUMINAMATH_CALUDE_one_square_covered_l1900_190007


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l1900_190005

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let technicians_salary := num_technicians * avg_salary_technicians
  let remaining_salary := total_salary - technicians_salary
  (remaining_salary / remaining_workers : ℚ) = 6000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l1900_190005


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1900_190090

/-- Given a hyperbola with foci F₁(-√5,0) and F₂(√5,0), and a point P on the hyperbola
    such that PF₁ · PF₂ = 0 and |PF₁| · |PF₂| = 2, the standard equation of the hyperbola
    is x²/4 - y² = 1. -/
theorem hyperbola_equation (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-Real.sqrt 5, 0) →
  F₂ = (Real.sqrt 5, 0) →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
    Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 →
  ∃ (x y : ℝ), x^2 / 4 - y^2 = 1 ∧ 
    (x, y) = P :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1900_190090


namespace NUMINAMATH_CALUDE_box_volume_formula_l1900_190023

/-- The volume of a box formed by cutting squares from corners of a sheet -/
def boxVolume (x : ℝ) : ℝ := (16 - 2*x) * (12 - 2*x) * x

/-- The constraint on the side length of the cut squares -/
def sideConstraint (x : ℝ) : Prop := x ≤ 12/5

theorem box_volume_formula (x : ℝ) (h : sideConstraint x) :
  boxVolume x = 192*x - 56*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l1900_190023


namespace NUMINAMATH_CALUDE_tiling_pattern_ratio_l1900_190027

/-- The ratio of the area covered by triangles to the total area in a specific tiling pattern -/
theorem tiling_pattern_ratio : ∀ s : ℝ,
  s > 0 →
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let triangle_area := (Real.sqrt 3 / 16) * s^2
  let total_area := hexagon_area + 2 * triangle_area
  triangle_area / total_area = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_tiling_pattern_ratio_l1900_190027


namespace NUMINAMATH_CALUDE_solve_for_x_l1900_190034

theorem solve_for_x (x y : ℚ) (h1 : x / y = 9 / 5) (h2 : y = 25) : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1900_190034


namespace NUMINAMATH_CALUDE_max_sum_of_digits_of_sum_l1900_190011

/-- Represents a three-digit positive integer with distinct digits from 1 to 9 -/
structure ThreeDigitNumber :=
  (value : ℕ)
  (is_three_digit : 100 ≤ value ∧ value ≤ 999)
  (distinct_digits : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃)
  (digits_range : ∀ d₁ d₂ d₃, value = 100 * d₁ + 10 * d₂ + d₃ → 1 ≤ d₁ ∧ d₁ ≤ 9 ∧ 1 ≤ d₂ ∧ d₂ ≤ 9 ∧ 1 ≤ d₃ ∧ d₃ ≤ 9)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The main theorem -/
theorem max_sum_of_digits_of_sum (a b : ThreeDigitNumber) :
  let S := a.value + b.value
  100 ≤ S ∧ S ≤ 999 →
  sum_of_digits S ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_of_sum_l1900_190011


namespace NUMINAMATH_CALUDE_evaluate_expression_l1900_190043

theorem evaluate_expression : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1900_190043


namespace NUMINAMATH_CALUDE_existence_of_common_root_l1900_190024

-- Define the structure of a quadratic polynomial
structure QuadraticPolynomial (R : Type*) [Ring R] where
  a : R
  b : R
  c : R

-- Define the evaluation of a quadratic polynomial
def evaluate {R : Type*} [Ring R] (p : QuadraticPolynomial R) (x : R) : R :=
  p.a * x * x + p.b * x + p.c

-- Theorem statement
theorem existence_of_common_root 
  {R : Type*} [Field R] 
  (f g h : QuadraticPolynomial R)
  (no_roots : ∀ x, evaluate f x ≠ 0 ∧ evaluate g x ≠ 0 ∧ evaluate h x ≠ 0)
  (same_leading_coeff : f.a = g.a ∧ f.a = h.a)
  (diff_x_coeff : f.b ≠ g.b ∧ f.b ≠ h.b ∧ g.b ≠ h.b) :
  ∃ c x, evaluate f x + c * evaluate g x = 0 ∧ evaluate f x + c * evaluate h x = 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_root_l1900_190024


namespace NUMINAMATH_CALUDE_mom_in_middle_l1900_190075

-- Define the people in the lineup
inductive Person : Type
  | Dad : Person
  | Mom : Person
  | Brother : Person
  | Sister : Person
  | Me : Person

-- Define the concept of being next to someone in the lineup
def next_to (p1 p2 : Person) : Prop := sorry

-- Define the concept of being in the middle
def in_middle (p : Person) : Prop := sorry

-- State the theorem
theorem mom_in_middle :
  -- Conditions
  (next_to Person.Me Person.Dad) →
  (next_to Person.Me Person.Mom) →
  (next_to Person.Sister Person.Mom) →
  (next_to Person.Sister Person.Brother) →
  -- Conclusion
  in_middle Person.Mom := by sorry

end NUMINAMATH_CALUDE_mom_in_middle_l1900_190075


namespace NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l1900_190056

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + n

def sum_sequence (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sequence_sum :
  units_digit (sum_sequence 12) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l1900_190056


namespace NUMINAMATH_CALUDE_x_value_theorem_l1900_190004

theorem x_value_theorem (x y : ℝ) (h : (x - 1) / x = (y^3 + 3*y^2 - 4) / (y^3 + 3*y^2 - 5)) :
  x = y^3 + 3*y^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1900_190004


namespace NUMINAMATH_CALUDE_point_below_line_l1900_190032

theorem point_below_line (m : ℝ) : 
  ((-2 : ℝ) + m * (-1 : ℝ) - 1 < 0) ↔ (m < -3 ∨ m > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_below_line_l1900_190032


namespace NUMINAMATH_CALUDE_sqrt_over_thirteen_equals_four_l1900_190044

theorem sqrt_over_thirteen_equals_four :
  Real.sqrt 2704 / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_over_thirteen_equals_four_l1900_190044


namespace NUMINAMATH_CALUDE_min_qr_length_l1900_190072

/-- Given two triangles PQR and SQR sharing side QR, with known side lengths,
    prove that the least possible integral length of QR is 15 cm. -/
theorem min_qr_length (pq pr sr sq : ℝ) (h_pq : pq = 7)
                      (h_pr : pr = 15) (h_sr : sr = 10) (h_sq : sq = 25) :
  ∀ qr : ℝ, qr > pr - pq ∧ qr > sq - sr → qr ≥ 15 := by sorry

end NUMINAMATH_CALUDE_min_qr_length_l1900_190072


namespace NUMINAMATH_CALUDE_greenhill_soccer_kicks_l1900_190064

/-- Given a soccer team with total players and goalies, calculate the number of penalty kicks required --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: For a team with 25 players including 4 goalies, 96 penalty kicks are required --/
theorem greenhill_soccer_kicks : penalty_kicks 25 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_greenhill_soccer_kicks_l1900_190064


namespace NUMINAMATH_CALUDE_inhabitable_earth_fraction_l1900_190063

/-- Represents the fraction of Earth's surface not covered by water -/
def land_fraction : ℚ := 1 / 3

/-- Represents the fraction of exposed land that is inhabitable -/
def inhabitable_land_fraction : ℚ := 1 / 3

/-- Theorem stating the fraction of Earth's surface that is inhabitable for humans -/
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_inhabitable_earth_fraction_l1900_190063


namespace NUMINAMATH_CALUDE_chopping_percentage_difference_l1900_190015

/-- Represents the chopping rates and total amount for Tom and Tammy -/
structure ChoppingData where
  tom_rate : ℚ  -- Tom's chopping rate in lb/min
  tammy_rate : ℚ  -- Tammy's chopping rate in lb/min
  total_amount : ℚ  -- Total amount of salad chopped in lb

/-- Calculates the percentage difference between Tammy's and Tom's chopped quantities -/
def percentage_difference (data : ChoppingData) : ℚ :=
  let combined_rate := data.tom_rate + data.tammy_rate
  let tom_share := (data.tom_rate / combined_rate) * data.total_amount
  let tammy_share := (data.tammy_rate / combined_rate) * data.total_amount
  ((tammy_share - tom_share) / tom_share) * 100

/-- Theorem stating that the percentage difference is 125% for the given data -/
theorem chopping_percentage_difference :
  let data : ChoppingData := {
    tom_rate := 2 / 3,  -- 2 lb in 3 minutes
    tammy_rate := 3 / 2,  -- 3 lb in 2 minutes
    total_amount := 65
  }
  percentage_difference data = 125 := by sorry


end NUMINAMATH_CALUDE_chopping_percentage_difference_l1900_190015


namespace NUMINAMATH_CALUDE_carriage_sharing_problem_l1900_190066

theorem carriage_sharing_problem (x : ℕ) : 
  (x / 3 : ℚ) + 2 = (x - 9 : ℚ) / 2 ↔ 
  (∃ (total_carriages : ℕ), 
    (x / 3 : ℚ) + 2 = total_carriages ∧ 
    (x - 9 : ℚ) / 2 = total_carriages) :=
sorry

end NUMINAMATH_CALUDE_carriage_sharing_problem_l1900_190066


namespace NUMINAMATH_CALUDE_marble_count_l1900_190028

theorem marble_count (r : ℝ) (b g y : ℝ) : 
  r > 0 →
  b = r / 1.3 →
  g = 1.5 * r →
  y = 1.2 * g →
  r + b + g + y = 5.069 * r := by
sorry

end NUMINAMATH_CALUDE_marble_count_l1900_190028


namespace NUMINAMATH_CALUDE_yans_distance_ratio_l1900_190085

/-- Yan's problem statement -/
theorem yans_distance_ratio :
  ∀ (w x y : ℝ),
  w > 0 →  -- walking speed is positive
  x > 0 →  -- distance from Yan to home is positive
  y > 0 →  -- distance from Yan to stadium is positive
  x + y > 0 →  -- Yan is between home and stadium
  y / w = (x / w + (x + y) / (9 * w)) →  -- both choices take the same time
  x / y = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_yans_distance_ratio_l1900_190085


namespace NUMINAMATH_CALUDE_triangle_properties_l1900_190008

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) 
  (h1 : t.c ≥ t.a ∧ t.c ≥ t.b) 
  (h2 : t.b = Real.sqrt 3 * circumradius t)
  (h3 : t.b * Real.sin t.B = (t.a + t.c) * Real.sin t.A)
  (h4 : 0 < t.A ∧ t.A < Real.pi / 2)
  (h5 : 0 < t.B ∧ t.B < Real.pi / 2)
  (h6 : 0 < t.C ∧ t.C < Real.pi / 2) :
  t.B = Real.pi / 3 ∧ t.A = Real.pi / 6 ∧ t.C = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1900_190008


namespace NUMINAMATH_CALUDE_red_highest_probability_l1900_190093

/-- A color of a ball -/
inductive Color
  | Red
  | Yellow
  | White

/-- The number of balls of each color in the bag -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 6
  | Color.Yellow => 4
  | Color.White => 1

/-- The total number of balls in the bag -/
def totalBalls : ℕ := ballCount Color.Red + ballCount Color.Yellow + ballCount Color.White

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing a red ball is the highest -/
theorem red_highest_probability :
  probability Color.Red > probability Color.Yellow ∧
  probability Color.Red > probability Color.White :=
sorry

end NUMINAMATH_CALUDE_red_highest_probability_l1900_190093


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1900_190099

-- Define the function f(x) = x² + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem statement
theorem f_increasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l1900_190099


namespace NUMINAMATH_CALUDE_roots_equation_value_l1900_190074

theorem roots_equation_value (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 → β^2 - 2*β - 4 = 0 → α^3 + 8*β + 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l1900_190074


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1900_190088

def senate_committee_ways (total_republicans : ℕ) (total_democrats : ℕ) 
  (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * 
  Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count : 
  senate_committee_ways 10 8 4 3 = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1900_190088


namespace NUMINAMATH_CALUDE_f_three_quadrants_iff_a_range_l1900_190010

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x - 1
  else x^3 - a * x + |x - 2|

/-- The graph of f passes through exactly three quadrants -/
def passes_through_three_quadrants (a : ℝ) : Prop :=
  ∃ x y z : ℝ,
    (x < 0 ∧ f a x < 0) ∧
    (y > 0 ∧ f a y > 0) ∧
    ((z < 0 ∧ f a z > 0) ∨ (z > 0 ∧ f a z < 0)) ∧
    ∀ w : ℝ, (w < 0 ∧ f a w > 0) → (z < 0 ∧ f a z > 0) ∧
             (w > 0 ∧ f a w < 0) → (z > 0 ∧ f a z < 0)

/-- Main theorem: f passes through exactly three quadrants iff a < 0 or a > 2 -/
theorem f_three_quadrants_iff_a_range (a : ℝ) :
  passes_through_three_quadrants a ↔ a < 0 ∨ a > 2 := by
  sorry


end NUMINAMATH_CALUDE_f_three_quadrants_iff_a_range_l1900_190010


namespace NUMINAMATH_CALUDE_train_length_calculation_l1900_190097

/-- The length of a train given its speed and time to pass a point -/
def train_length (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 160 * 1000 / 3600) -- Speed in m/s
  (h2 : time = 2.699784017278618) : 
  ∃ (ε : Real), ε > 0 ∧ |train_length speed time - 120| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1900_190097


namespace NUMINAMATH_CALUDE_solution_l1900_190069

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 3) + B / (x^2 - 9*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 27*x)

-- State the theorem
theorem solution (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_solution_l1900_190069


namespace NUMINAMATH_CALUDE_initial_percent_problem_l1900_190076

theorem initial_percent_problem (x : ℝ) :
  (3 : ℝ) / 100 = (60 : ℝ) / 100 * x → x = (5 : ℝ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_percent_problem_l1900_190076


namespace NUMINAMATH_CALUDE_sum_of_cube_ratios_l1900_190055

open BigOperators

/-- Given a finite sequence of rational numbers x_t = i/101 for i = 0 to 101,
    the sum T = ∑(i=0 to 101) [x_i^3 / (3x_t^2 - 3x_t + 1)] is equal to 51. -/
theorem sum_of_cube_ratios (x : Fin 102 → ℚ) 
  (h : ∀ i : Fin 102, x i = (i : ℚ) / 101) : 
  ∑ i : Fin 102, (x i)^3 / (3 * (x i)^2 - 3 * (x i) + 1) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_ratios_l1900_190055


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1900_190018

theorem purely_imaginary_complex_number (a : ℝ) :
  (Complex.I * Complex.im (a * (1 + Complex.I) - 2) = a * (1 + Complex.I) - 2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1900_190018


namespace NUMINAMATH_CALUDE_equation_solution_l1900_190046

theorem equation_solution : 
  ∀ x : ℝ, (3*x - 1)*(2*x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1900_190046


namespace NUMINAMATH_CALUDE_det_special_matrix_l1900_190077

/-- The determinant of the matrix [[2x + 2, 2x, 2x], [2x, 2x + 2, 2x], [2x, 2x, 2x + 2]] is equal to 20x + 8 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![2*x + 2, 2*x, 2*x; 
                2*x, 2*x + 2, 2*x; 
                2*x, 2*x, 2*x + 2] = 20*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1900_190077


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1900_190033

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1900_190033


namespace NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1900_190021

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1 / 3

/-- The theorem stating that 0.3̄ * 9 = 3 -/
theorem product_repeating_third_and_nine : repeating_third * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_and_nine_l1900_190021


namespace NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l1900_190014

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + x * Real.log x

-- State the theorem
theorem f_derivative_and_tangent_line :
  -- The derivative of f(x)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2*x + Real.log x + 1) x) ∧
  -- The equation of the tangent line at x = 1
  (∃ A B C : ℝ, A = 3 ∧ B = -1 ∧ C = -2 ∧
    ∀ x y : ℝ, (x = 1 ∧ y = f 1) → (A*x + B*y + C = 0)) := by
  sorry

end

end NUMINAMATH_CALUDE_f_derivative_and_tangent_line_l1900_190014


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l1900_190060

theorem power_of_three_mod_eight : 3^2007 % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l1900_190060


namespace NUMINAMATH_CALUDE_henry_twice_jills_age_l1900_190037

/-- Proves that Henry was twice Jill's age 6 years ago given their present ages and sum. -/
theorem henry_twice_jills_age (henry_age jill_age : ℕ) (sum_ages : ℕ) : 
  henry_age = 20 → 
  jill_age = 13 → 
  sum_ages = henry_age + jill_age → 
  sum_ages = 33 → 
  ∃ (years_ago : ℕ), years_ago = 6 ∧ henry_age - years_ago = 2 * (jill_age - years_ago) := by
  sorry

end NUMINAMATH_CALUDE_henry_twice_jills_age_l1900_190037


namespace NUMINAMATH_CALUDE_picnic_basket_theorem_l1900_190057

/-- Calculate the total cost of a picnic basket given the number of people and item prices -/
def picnic_basket_cost (num_people : ℕ) (sandwich_price fruit_salad_price soda_price snack_price : ℚ) : ℚ :=
  let sandwich_cost := num_people * sandwich_price
  let fruit_salad_cost := num_people * fruit_salad_price
  let soda_cost := num_people * 2 * soda_price
  let snack_cost := 3 * snack_price
  sandwich_cost + fruit_salad_cost + soda_cost + snack_cost

/-- The total cost of the picnic basket is $60 -/
theorem picnic_basket_theorem :
  picnic_basket_cost 4 5 3 2 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_basket_theorem_l1900_190057


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1900_190083

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line y = ax + 1 -/
def slope1 (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x + 3 -/
def slope2 (a : ℝ) : ℝ := a - 2

/-- The theorem stating that a = 1 is the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope1 a) (slope2 a) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1900_190083


namespace NUMINAMATH_CALUDE_monkey_count_l1900_190045

theorem monkey_count : ∃! x : ℕ, x > 0 ∧ (x / 8)^2 + 12 = x := by
  sorry

end NUMINAMATH_CALUDE_monkey_count_l1900_190045


namespace NUMINAMATH_CALUDE_leadership_selection_count_l1900_190013

/-- The number of people in the group -/
def n : ℕ := 5

/-- The number of positions to be filled (leader and deputy) -/
def k : ℕ := 2

/-- The number of ways to select a leader and deputy with no restrictions -/
def total_selections : ℕ := n * (n - 1)

/-- The number of invalid selections (when the restricted person is deputy) -/
def invalid_selections : ℕ := n - 1

/-- The number of valid selections -/
def valid_selections : ℕ := total_selections - invalid_selections

theorem leadership_selection_count :
  valid_selections = 16 :=
sorry

end NUMINAMATH_CALUDE_leadership_selection_count_l1900_190013


namespace NUMINAMATH_CALUDE_expand_expression_l1900_190040

theorem expand_expression (x : ℝ) : 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1) = 10 * x^3 - 15 * x^2 + 20 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1900_190040


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1900_190019

/-- The length of the real axis of a hyperbola with equation x^2 - y^2/9 = 1 is 2 -/
theorem hyperbola_real_axis_length :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - y^2/9 = 1}
  ∃ (a : ℝ), a > 0 ∧ (∀ (p : ℝ × ℝ), p ∈ hyperbola → p.1 ≤ a) ∧ 2 * a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1900_190019


namespace NUMINAMATH_CALUDE_x_value_l1900_190061

theorem x_value (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 3 ∨ x = -9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1900_190061


namespace NUMINAMATH_CALUDE_fraction_simplification_l1900_190084

theorem fraction_simplification :
  5 / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1900_190084


namespace NUMINAMATH_CALUDE_set_equality_l1900_190029

def S : Set (ℕ × ℕ) := {p | p.1 + p.2 = 3}

theorem set_equality : S = {(1, 2), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1900_190029


namespace NUMINAMATH_CALUDE_fraction_simplification_l1900_190095

theorem fraction_simplification : (4 * 5) / 10 = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1900_190095


namespace NUMINAMATH_CALUDE_sum_f_neg1_0_1_l1900_190086

-- Define the function f
variable (f : ℝ → ℝ)

-- State the condition
axiom f_add (x y : ℝ) : f x + f y = f (x + y)

-- State the theorem to be proved
theorem sum_f_neg1_0_1 : f (-1) + f 0 + f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_neg1_0_1_l1900_190086


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_l1900_190031

theorem zoo_ticket_cost 
  (total_spent : ℚ)
  (family_size : ℕ)
  (adult_ticket_cost : ℚ)
  (adult_tickets : ℕ)
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : adult_ticket_cost = 21)
  (h4 : adult_tickets = 4) :
  let children_tickets := family_size - adult_tickets
  let children_total_cost := total_spent - (adult_ticket_cost * adult_tickets)
  children_total_cost / children_tickets = 35 / 3 :=
sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_l1900_190031


namespace NUMINAMATH_CALUDE_fraction_c_simplest_form_l1900_190001

/-- A fraction is in its simplest form if its numerator and denominator have no common factors other than 1 and -1. -/
def IsSimplestForm (num den : ℤ → ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, (∀ k : ℤ, k ≠ 1 ∧ k ≠ -1 → (k ∣ num a b ↔ k ∣ den a b) → False)

/-- The fraction (3a + b) / (a + b) is in its simplest form. -/
theorem fraction_c_simplest_form :
  IsSimplestForm (fun a b => 3*a + b) (fun a b => a + b) := by
  sorry

#check fraction_c_simplest_form

end NUMINAMATH_CALUDE_fraction_c_simplest_form_l1900_190001


namespace NUMINAMATH_CALUDE_pizza_area_increase_l1900_190073

theorem pizza_area_increase : 
  let r1 : ℝ := 2
  let r2 : ℝ := 5
  let area1 := π * r1^2
  let area2 := π * r2^2
  (area2 - area1) / area1 * 100 = 525 :=
by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l1900_190073


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1900_190096

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 5*x + c > 0 ↔ 1/3 < x ∧ x < 1/2) → 
  a + c = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1900_190096


namespace NUMINAMATH_CALUDE_parallelogram_center_not_axis_symmetric_l1900_190041

-- Define the shape types
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Rhombus

-- Define the symmetry properties
def isAxisSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.Rhombus => true

def isCenterSymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Rhombus => true

-- Theorem statement
theorem parallelogram_center_not_axis_symmetric :
  ∃ (s : Shape), isCenterSymmetric s ∧ ¬isAxisSymmetric s ∧
  ∀ (t : Shape), t ≠ s → ¬(isCenterSymmetric t ∧ ¬isAxisSymmetric t) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_center_not_axis_symmetric_l1900_190041


namespace NUMINAMATH_CALUDE_largest_coeff_sixth_term_l1900_190080

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℚ := (Nat.choose n k : ℚ)

-- Define the coefficient of the r-th term in the expansion
def coeff (r : ℕ) : ℚ := (1/2)^r * binomial_coeff 15 r

-- Theorem statement
theorem largest_coeff_sixth_term :
  ∀ k : ℕ, k ≠ 5 → coeff 5 ≥ coeff k :=
sorry

end NUMINAMATH_CALUDE_largest_coeff_sixth_term_l1900_190080


namespace NUMINAMATH_CALUDE_max_value_cos_sin_linear_combination_l1900_190017

theorem max_value_cos_sin_linear_combination (a b φ : ℝ) :
  (∀ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, a * Real.cos (θ - φ) + b * Real.sin (θ - φ) = Real.sqrt (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_linear_combination_l1900_190017


namespace NUMINAMATH_CALUDE_expression_evaluation_l1900_190009

theorem expression_evaluation : 12 - (-18) + (-7) - 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1900_190009


namespace NUMINAMATH_CALUDE_carol_weight_l1900_190053

/-- Given two people's weights satisfying certain conditions, prove that one person's weight is 165 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (sum_condition : alice_weight + carol_weight = 220)
  (difference_condition : carol_weight - alice_weight = (2/3) * carol_weight) :
  carol_weight = 165 := by
  sorry

end NUMINAMATH_CALUDE_carol_weight_l1900_190053


namespace NUMINAMATH_CALUDE_smallest_sum_of_roots_l1900_190089

theorem smallest_sum_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  a + b ≥ 6.5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_roots_l1900_190089


namespace NUMINAMATH_CALUDE_binomial_expression_approx_l1900_190094

/-- Calculates the binomial coefficient for real x and nonnegative integer k -/
def binomial (x : ℝ) (k : ℕ) : ℝ := sorry

/-- The main theorem stating that the given expression is approximately equal to -1.243 -/
theorem binomial_expression_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((binomial (3/2 : ℝ) 10) * 3^10) / (binomial 20 10) + 1.243| < ε :=
sorry

end NUMINAMATH_CALUDE_binomial_expression_approx_l1900_190094


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1900_190079

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 ∧
  Real.sin β = -1 / 3 ∧
  α - β = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1900_190079


namespace NUMINAMATH_CALUDE_irrationality_of_pi_and_rationality_of_others_l1900_190035

-- Define irrational numbers
def IsIrrational (x : ℝ) : Prop :=
  ∀ a b : ℤ, b ≠ 0 → x ≠ a / b

-- State the theorem
theorem irrationality_of_pi_and_rationality_of_others :
  IsIrrational Real.pi ∧ ¬IsIrrational 0 ∧ ¬IsIrrational (-1/3) ∧ ¬IsIrrational (3/2) :=
sorry

end NUMINAMATH_CALUDE_irrationality_of_pi_and_rationality_of_others_l1900_190035


namespace NUMINAMATH_CALUDE_sales_increase_price_reduction_for_target_profit_l1900_190042

/-- Represents the Asian Games mascot badge sales scenario -/
structure BadgeSales where
  originalProfit : ℝ  -- Original profit per set
  originalSold : ℝ    -- Original number of sets sold per day
  profitReduction : ℝ -- Reduction in profit per set
  saleIncrease : ℝ    -- Increase in sales per $1 reduction

/-- Calculates the increase in sets sold given a profit reduction -/
def increasedSales (s : BadgeSales) : ℝ :=
  s.profitReduction * s.saleIncrease

/-- Calculates the daily profit given a price reduction -/
def dailyProfit (s : BadgeSales) (priceReduction : ℝ) : ℝ :=
  (s.originalProfit - priceReduction) * (s.originalSold + priceReduction * s.saleIncrease)

/-- Theorem stating the increase in sales when profit is reduced by $2 -/
theorem sales_increase (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.profitReduction = 2 ∧ s.saleIncrease = 2) :
  increasedSales s = 4 := by sorry

/-- Theorem stating the price reduction needed for a daily profit of $1200 -/
theorem price_reduction_for_target_profit (s : BadgeSales) (h : s.originalProfit = 40 ∧ s.originalSold = 20 ∧ s.saleIncrease = 2) :
  ∃ x : ℝ, x = 20 ∧ dailyProfit s x = 1200 := by sorry

end NUMINAMATH_CALUDE_sales_increase_price_reduction_for_target_profit_l1900_190042


namespace NUMINAMATH_CALUDE_decimal_53_is_binary_110101_l1900_190026

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_53_is_binary_110101 :
  toBinary 53 = [true, false, true, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_decimal_53_is_binary_110101_l1900_190026


namespace NUMINAMATH_CALUDE_base4_even_digits_145_l1900_190092

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_even_digits_145 :
  countEvenDigits (toBase4 145) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base4_even_digits_145_l1900_190092


namespace NUMINAMATH_CALUDE_even_digits_529_base9_l1900_190039

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-9 representation of 529₁₀ is 2 -/
theorem even_digits_529_base9 : 
  countEvenDigits (toBase9 529) = 2 :=
sorry

end NUMINAMATH_CALUDE_even_digits_529_base9_l1900_190039


namespace NUMINAMATH_CALUDE_product_divisible_by_eight_l1900_190000

theorem product_divisible_by_eight (n : ℤ) (h : 1 ≤ n ∧ n ≤ 96) : 
  8 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_eight_l1900_190000


namespace NUMINAMATH_CALUDE_system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l1900_190067

/-- The system of equations has at least one solution for all real a -/
theorem system_has_solution (a : ℝ) : ∃ x y : ℝ, 
  (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

/-- Solutions for a ≤ 0 or a = 1 -/
theorem solutions_for_a_nonpositive_or_one (a : ℝ) (h : a ≤ 0 ∨ a = 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for 0 < a < 2, a ≠ 1 -/
theorem solutions_for_a_between_zero_and_two (a : ℝ) (h : 0 < a ∧ a < 2 ∧ a ≠ 1) : 
  (∃ x y : ℝ, x = 0 ∧ y = 1 - a / 2 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) ∧
  (∃ x y : ℝ, x = ((2 - a) / a)^2 ∧ y = (2 - a) / a ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x)) := by sorry

/-- Solutions for a ≥ 2 -/
theorem solutions_for_a_geq_two (a : ℝ) (h : a ≥ 2) : 
  ∃ x y : ℝ, x = 1 ∧ y = 1 ∧ 
    (2 * y - 2 = a * (x - 1)) ∧ (2 * x / (|y| + y) = Real.sqrt x) := by sorry

end NUMINAMATH_CALUDE_system_has_solution_solutions_for_a_nonpositive_or_one_solutions_for_a_between_zero_and_two_solutions_for_a_geq_two_l1900_190067


namespace NUMINAMATH_CALUDE_linear_equation_mn_l1900_190065

theorem linear_equation_mn (m n : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, x^(4-3*|m|) + y^(3*|n|) = a*x + b*y + c) →
  m * n < 0 →
  0 < m + n →
  m + n ≤ 3 →
  m - n = 4/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_mn_l1900_190065


namespace NUMINAMATH_CALUDE_segments_intersection_l1900_190016

-- Define the number of segments
def n : ℕ := 1977

-- Define the type for segments
def Segment : Type := ℕ → Set ℝ

-- Define the property of intersection
def intersects (s1 s2 : Set ℝ) : Prop := ∃ x, x ∈ s1 ∧ x ∈ s2

-- State the theorem
theorem segments_intersection 
  (A B : Segment) 
  (h1 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + n - 1) % n)))
  (h2 : ∀ k ∈ Finset.range n, intersects (A k) (B ((k + 1) % n)))
  (h3 : intersects (A (n - 1)) (B 0))
  (h4 : intersects (A 0) (B (n - 1)))
  : ∃ k ∈ Finset.range n, intersects (A k) (B k) :=
by sorry

end NUMINAMATH_CALUDE_segments_intersection_l1900_190016


namespace NUMINAMATH_CALUDE_total_loss_proof_l1900_190058

/-- Represents the capital and loss of an investor -/
structure Investor where
  capital : ℝ
  loss : ℝ

/-- Calculates the total loss given two investors -/
def totalLoss (investor1 investor2 : Investor) : ℝ :=
  investor1.loss + investor2.loss

/-- Theorem: Given two investors with capitals in ratio 1:9 and losses proportional to their investments,
    if one investor loses Rs 900, the total loss is Rs 1000 -/
theorem total_loss_proof (investor1 investor2 : Investor) 
    (h1 : investor1.capital = (1/9) * investor2.capital)
    (h2 : investor1.loss / investor2.loss = investor1.capital / investor2.capital)
    (h3 : investor2.loss = 900) :
    totalLoss investor1 investor2 = 1000 := by
  sorry

#eval totalLoss { capital := 1, loss := 100 } { capital := 9, loss := 900 }

end NUMINAMATH_CALUDE_total_loss_proof_l1900_190058


namespace NUMINAMATH_CALUDE_min_value_of_z_l1900_190047

theorem min_value_of_z (x y : ℝ) : 
  3 * x^2 + y^2 + 12 * x - 6 * y + 40 ≥ 19 ∧ 
  ∃ x y : ℝ, 3 * x^2 + y^2 + 12 * x - 6 * y + 40 = 19 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1900_190047


namespace NUMINAMATH_CALUDE_cube_difference_l1900_190062

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l1900_190062


namespace NUMINAMATH_CALUDE_xiao_ming_book_price_l1900_190038

/-- The price of Xiao Ming's book satisfies 15 < x < 20, given that:
    1. Classmate A guessed the price is at least 20.
    2. Classmate B guessed the price is at most 15.
    3. Xiao Ming said both classmates are wrong. -/
theorem xiao_ming_book_price (x : ℝ) 
  (hA : x < 20)  -- Xiao Ming said A is wrong, so price is less than 20
  (hB : x > 15)  -- Xiao Ming said B is wrong, so price is greater than 15
  : 15 < x ∧ x < 20 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_book_price_l1900_190038


namespace NUMINAMATH_CALUDE_product_congruence_l1900_190078

theorem product_congruence : 45 * 68 * 99 ≡ 15 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l1900_190078
