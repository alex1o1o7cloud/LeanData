import Mathlib

namespace NUMINAMATH_CALUDE_probability_isosceles_triangle_l590_59016

def roll_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_isosceles (a b : ℕ) : Bool :=
  a + b > 5

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (roll_die.product roll_die).filter (fun (a, b) => is_isosceles a b)

theorem probability_isosceles_triangle :
  (favorable_outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_isosceles_triangle_l590_59016


namespace NUMINAMATH_CALUDE_max_elephants_l590_59006

/-- The number of union members --/
def unionMembers : ℕ := 28

/-- The number of non-members --/
def nonMembers : ℕ := 37

/-- The total number of attendees --/
def totalAttendees : ℕ := unionMembers + nonMembers

/-- A function to check if a distribution is valid --/
def isValidDistribution (elephants : ℕ) : Prop :=
  ∃ (unionElephants nonUnionElephants : ℕ),
    elephants = unionElephants + nonUnionElephants ∧
    unionElephants % unionMembers = 0 ∧
    nonUnionElephants % nonMembers = 0 ∧
    unionElephants / unionMembers ≥ 1 ∧
    nonUnionElephants / nonMembers ≥ 1

/-- The theorem stating the maximum number of elephants --/
theorem max_elephants :
  ∃! (maxElephants : ℕ),
    isValidDistribution maxElephants ∧
    ∀ (n : ℕ), n > maxElephants → ¬isValidDistribution n :=
by
  sorry

end NUMINAMATH_CALUDE_max_elephants_l590_59006


namespace NUMINAMATH_CALUDE_cuboid_breadth_proof_l590_59065

/-- The surface area of a cuboid given its length, width, and height. -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with surface area 700 m², length 12 m, and height 7 m is 14 m. -/
theorem cuboid_breadth_proof :
  ∃ w : ℝ, cuboidSurfaceArea 12 w 7 = 700 ∧ w = 14 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_breadth_proof_l590_59065


namespace NUMINAMATH_CALUDE_alice_number_theorem_l590_59069

def smallest_prime_divisor (n : ℕ) : ℕ := sorry

def subtract_smallest_prime_divisor (n : ℕ) : ℕ := n - smallest_prime_divisor n

def iterate_subtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_subtraction (subtract_smallest_prime_divisor n) k

theorem alice_number_theorem (n : ℕ) :
  n > 0 ∧ Nat.Prime (iterate_subtraction n 2022) →
  n = 4046 ∨ n = 4047 :=
sorry

end NUMINAMATH_CALUDE_alice_number_theorem_l590_59069


namespace NUMINAMATH_CALUDE_smallest_sticker_count_l590_59060

theorem smallest_sticker_count (N : ℕ) : 
  N > 1 → 
  (∃ x y z : ℕ, N = 3 * x + 1 ∧ N = 5 * y + 1 ∧ N = 11 * z + 1) → 
  N ≥ 166 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_count_l590_59060


namespace NUMINAMATH_CALUDE_age_difference_l590_59050

/-- Given two people A and B, where B is currently 37 years old,
    and in 10 years A will be twice as old as B was 10 years ago,
    prove that A is currently 7 years older than B. -/
theorem age_difference (a b : ℕ) (h1 : b = 37) 
    (h2 : a + 10 = 2 * (b - 10)) : a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l590_59050


namespace NUMINAMATH_CALUDE_inequality_always_true_l590_59036

theorem inequality_always_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l590_59036


namespace NUMINAMATH_CALUDE_line_slope_l590_59003

-- Define the parametric equations of the line
def x (t : ℝ) : ℝ := 3 + 4 * t
def y (t : ℝ) : ℝ := 4 - 5 * t

-- State the theorem
theorem line_slope :
  ∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ m = -5/4 :=
sorry

end NUMINAMATH_CALUDE_line_slope_l590_59003


namespace NUMINAMATH_CALUDE_white_balls_count_l590_59094

theorem white_balls_count (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 4 →
  (white_balls : ℚ) / (red_balls + white_balls : ℚ) = 2 / 3 →
  white_balls = 8 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l590_59094


namespace NUMINAMATH_CALUDE_parabola_values_l590_59031

/-- A parabola passing through (1, 1) with a specific tangent line -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x - 7

theorem parabola_values (a b : ℝ) :
  (Parabola a b 1 = 1) ∧ 
  (4 * 1 - Parabola a b 1 - 3 = 0) ∧
  (2 * a * 1 + b = 4) →
  a = -4 ∧ b = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_values_l590_59031


namespace NUMINAMATH_CALUDE_color_film_fraction_l590_59043

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let bw_selected_percent := y / x
  let bw_selected := bw_selected_percent * total_bw / 100
  let color_selected := total_color
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l590_59043


namespace NUMINAMATH_CALUDE_max_fraction_over65_l590_59079

/-- Represents the number of people in a room with age-related conditions -/
structure RoomPopulation where
  total : ℕ
  under21 : ℕ
  over65 : ℕ
  h1 : under21 = (3 * total) / 7
  h2 : 50 < total
  h3 : total < 100
  h4 : under21 = 30

/-- The maximum fraction of people over 65 in the room is 4/7 -/
theorem max_fraction_over65 (room : RoomPopulation) :
  (room.over65 : ℚ) / room.total ≤ 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_over65_l590_59079


namespace NUMINAMATH_CALUDE_algorithm_finite_results_l590_59038

-- Define the properties of an algorithm
structure Algorithm where
  steps : ℕ
  inputs : ℕ
  deterministic : Bool
  unique_meaning : Bool
  definite : Bool
  finite : Bool
  orderly : Bool
  non_unique : Bool
  universal : Bool

-- Define the theorem
theorem algorithm_finite_results (a : Algorithm) : 
  a.definite → ¬(∃ (results : ℕ → Prop), (∀ n : ℕ, results n) ∧ (∀ m n : ℕ, m ≠ n → results m ≠ results n)) :=
by sorry

end NUMINAMATH_CALUDE_algorithm_finite_results_l590_59038


namespace NUMINAMATH_CALUDE_min_value_of_function_l590_59067

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  (x^2 + 4) / x ≥ 4 ∧ ∃ y > 0, (y^2 + 4) / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l590_59067


namespace NUMINAMATH_CALUDE_basketball_points_third_game_l590_59081

theorem basketball_points_third_game 
  (total_points : ℕ) 
  (first_game_fraction : ℚ) 
  (second_game_fraction : ℚ) 
  (h1 : total_points = 20) 
  (h2 : first_game_fraction = 1/2) 
  (h3 : second_game_fraction = 1/10) : 
  total_points - (first_game_fraction * total_points + second_game_fraction * total_points) = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_points_third_game_l590_59081


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l590_59012

/-- Given a quadratic function f(x) = ax² + bx + b²/(2a) where a > 0,
    prove that the graph of f has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), a * x^2 + b * x + b^2 / (2 * a) ≥ a * x_min^2 + b * x_min + b^2 / (2 * a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l590_59012


namespace NUMINAMATH_CALUDE_unique_triangle_set_l590_59020

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ 
  match sides with
  | [a, b, c] => triangle_inequality a b c
  | _ => False

theorem unique_triangle_set : 
  ¬ can_form_triangle [2, 3, 6] ∧
  ¬ can_form_triangle [3, 4, 8] ∧
  can_form_triangle [5, 6, 10] ∧
  ¬ can_form_triangle [5, 6, 11] :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_set_l590_59020


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l590_59088

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := robs_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := robs_grapes + allies_grapes + allyns_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l590_59088


namespace NUMINAMATH_CALUDE_A_share_of_profit_l590_59054

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_A_share_of_profit (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months_before_change : ℕ) (total_months : ℕ) (total_profit : ℕ) : ℚ :=
  let a_investment_months := a_initial * months_before_change + (a_initial - a_withdrawal) * (total_months - months_before_change)
  let b_investment_months := b_initial * months_before_change + (b_initial + b_addition) * (total_months - months_before_change)
  let total_investment_months := a_investment_months + b_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit

theorem A_share_of_profit :
  calculate_A_share_of_profit 3000 4000 1000 1000 8 12 630 = 240 := by
  sorry

end NUMINAMATH_CALUDE_A_share_of_profit_l590_59054


namespace NUMINAMATH_CALUDE_tiles_needed_for_room_main_tiling_theorem_l590_59077

/-- Represents the tiling pattern where n is the number of days and f(n) is the number of tiles placed on day n. -/
def tilingPattern (n : ℕ) : ℕ := n

/-- Represents the total number of tiles placed after n days. -/
def totalTiles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The surface area of the room in square units. -/
def roomArea : ℕ := 18144

/-- The theorem stating that 2016 tiles are needed to cover the room. -/
theorem tiles_needed_for_room :
  ∃ (sideLength : ℕ), sideLength > 0 ∧ totalTiles 63 = 2016 ∧ 2016 * sideLength^2 = roomArea :=
by
  sorry

/-- The main theorem proving that 2016 tiles are needed and follow the tiling pattern. -/
theorem main_tiling_theorem :
  ∃ (n : ℕ), totalTiles n = 2016 ∧
    (∀ (k : ℕ), k ≤ n → tilingPattern k = k) ∧
    (∃ (sideLength : ℕ), sideLength > 0 ∧ 2016 * sideLength^2 = roomArea) :=
by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_for_room_main_tiling_theorem_l590_59077


namespace NUMINAMATH_CALUDE_C_grazed_for_4_months_l590_59082

/-- The number of milkmen who rented the pasture -/
def num_milkmen : ℕ := 4

/-- The number of cows grazed by milkman A -/
def cows_A : ℕ := 24

/-- The number of months milkman A grazed his cows -/
def months_A : ℕ := 3

/-- The number of cows grazed by milkman B -/
def cows_B : ℕ := 10

/-- The number of months milkman B grazed his cows -/
def months_B : ℕ := 5

/-- The number of cows grazed by milkman C -/
def cows_C : ℕ := 35

/-- The number of cows grazed by milkman D -/
def cows_D : ℕ := 21

/-- The number of months milkman D grazed his cows -/
def months_D : ℕ := 3

/-- A's share of the rent in rupees -/
def share_A : ℕ := 720

/-- The total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- The theorem stating that C grazed his cows for 4 months -/
theorem C_grazed_for_4_months :
  ∃ (months_C : ℕ),
    months_C = 4 ∧
    total_rent = share_A +
      (cows_B * months_B * share_A / (cows_A * months_A)) +
      (cows_C * months_C * share_A / (cows_A * months_A)) +
      (cows_D * months_D * share_A / (cows_A * months_A)) :=
by sorry

end NUMINAMATH_CALUDE_C_grazed_for_4_months_l590_59082


namespace NUMINAMATH_CALUDE_sum_reciprocals_l590_59000

theorem sum_reciprocals (a b c d e : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 → e ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) + 1 / (e + ω) : ℂ) = 3 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) + 1 / (e + 1) : ℝ) = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l590_59000


namespace NUMINAMATH_CALUDE_siblings_age_sum_l590_59010

theorem siblings_age_sum (a b c : ℕ+) : 
  a < b ∧ b = c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l590_59010


namespace NUMINAMATH_CALUDE_special_function_inequality_l590_59071

open Set

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  f_diff : Differentiable ℝ f
  f_domain : ∀ x, x < 0 → f x ≠ 0
  f_ineq : ∀ x, x < 0 → 2 * (f x) + x * (deriv f x) > x^2

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  {x : ℝ | (x + 2016)^2 * sf.f (x + 2016) - 4 * sf.f (-2) > 0} = Iio (-2018) := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l590_59071


namespace NUMINAMATH_CALUDE_seating_probability_l590_59097

/-- The number of chairs in the row -/
def total_chairs : ℕ := 10

/-- The number of usable chairs -/
def usable_chairs : ℕ := total_chairs - 1

/-- The probability that Mary and James do not sit next to each other -/
def probability_not_adjacent : ℚ := 7/9

theorem seating_probability :
  (total_chairs : ℕ) = 10 →
  (usable_chairs : ℕ) = total_chairs - 1 →
  probability_not_adjacent = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l590_59097


namespace NUMINAMATH_CALUDE_folded_paper_distance_l590_59032

theorem folded_paper_distance (sheet_area : ℝ) (folded_leg : ℝ) : 
  sheet_area = 6 →
  folded_leg ^ 2 / 2 = sheet_area - folded_leg ^ 2 →
  Real.sqrt (2 * folded_leg ^ 2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l590_59032


namespace NUMINAMATH_CALUDE_banana_count_l590_59017

def fruit_bowl (apples pears bananas : ℕ) : Prop :=
  (pears = apples + 2) ∧
  (bananas = pears + 3) ∧
  (apples + pears + bananas = 19)

theorem banana_count :
  ∀ (a p b : ℕ), fruit_bowl a p b → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l590_59017


namespace NUMINAMATH_CALUDE_frog_riverbank_probability_l590_59055

/-- The probability of reaching the riverbank from stone N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of stones -/
def num_stones : ℕ := 7

theorem frog_riverbank_probability :
  -- The frog starts on stone 2
  -- There are 7 stones labeled from 0 to 6
  -- For stone N (0 < N < 6), the frog jumps to N-1 with probability N/6 and to N+1 with probability 1 - N/6
  -- If the frog reaches stone 0, it falls into the water (probability 0)
  -- If the frog reaches stone 6, it safely reaches the riverbank (probability 1)
  (∀ N, 0 < N → N < 6 → P N = (N / 6 : ℝ) * P (N - 1) + (1 - N / 6 : ℝ) * P (N + 1)) →
  P 0 = 0 →
  P 6 = 1 →
  P 2 = 4/9 :=
sorry

end NUMINAMATH_CALUDE_frog_riverbank_probability_l590_59055


namespace NUMINAMATH_CALUDE_max_surface_area_inscribed_sphere_l590_59027

/-- The maximum surface area of an inscribed sphere in a right triangular prism --/
theorem max_surface_area_inscribed_sphere (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 25) :
  ∃ (r : ℝ), r > 0 ∧ 
    r = (5/2) * (Real.sqrt 2 - 1) ∧
    4 * π * r^2 = 25 * (3 - 3 * Real.sqrt 2) * π ∧
    ∀ (r' : ℝ), r' > 0 → r' * (a + b + 5) ≤ a * b → 4 * π * r'^2 ≤ 25 * (3 - 3 * Real.sqrt 2) * π :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_inscribed_sphere_l590_59027


namespace NUMINAMATH_CALUDE_flight_cost_A_to_C_via_B_l590_59053

/-- Represents the cost of a flight with a given distance and number of stops -/
def flight_cost (distance : ℝ) (stops : ℕ) : ℝ :=
  120 + 0.15 * distance + 50 * stops

/-- The cities A, B, and C form a right-angled triangle -/
axiom right_triangle : ∃ (AB BC AC : ℝ), AB^2 + BC^2 = AC^2

/-- The distance between A and C is 2000 km -/
axiom AC_distance : ∃ AC : ℝ, AC = 2000

/-- The distance between A and B is 4000 km -/
axiom AB_distance : ∃ AB : ℝ, AB = 4000

/-- Theorem: The cost to fly from A to C with one stop at B is $1289.62 -/
theorem flight_cost_A_to_C_via_B : 
  ∃ (AB BC AC : ℝ), 
    AB^2 + BC^2 = AC^2 ∧ 
    AC = 2000 ∧ 
    AB = 4000 ∧ 
    flight_cost (AB + BC) 1 = 1289.62 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_A_to_C_via_B_l590_59053


namespace NUMINAMATH_CALUDE_stating_equilateral_triangle_condition_l590_59044

/-- 
A function that checks if a natural number n satisfies the condition
that sticks of lengths 1, 2, ..., n can form an equilateral triangle.
-/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- 
Theorem stating that sticks of lengths 1, 2, ..., n can form an equilateral triangle
if and only if n satisfies the condition defined in can_form_equilateral_triangle.
-/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔
  can_form_equilateral_triangle n :=
sorry

end NUMINAMATH_CALUDE_stating_equilateral_triangle_condition_l590_59044


namespace NUMINAMATH_CALUDE_bookkeeper_probability_l590_59098

def word_length : ℕ := 10

def num_e : ℕ := 3
def num_o : ℕ := 2
def num_k : ℕ := 2
def num_b : ℕ := 1
def num_p : ℕ := 1
def num_r : ℕ := 1

def adjacent_o : Prop := true
def two_adjacent_e : Prop := true
def no_o_e_at_beginning : Prop := true

def total_arrangements : ℕ := 9600

theorem bookkeeper_probability : 
  word_length = num_e + num_o + num_k + num_b + num_p + num_r →
  adjacent_o →
  two_adjacent_e →
  no_o_e_at_beginning →
  (1 : ℚ) / total_arrangements = (1 : ℚ) / 9600 :=
sorry

end NUMINAMATH_CALUDE_bookkeeper_probability_l590_59098


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l590_59046

/-- Proves that 6 members didn't show up to a trivia game --/
theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 15 →
  points_per_member = 3 →
  total_points = 27 →
  total_members - (total_points / points_per_member) = 6 := by
  sorry


end NUMINAMATH_CALUDE_trivia_team_absentees_l590_59046


namespace NUMINAMATH_CALUDE_solution_characterization_l590_59087

theorem solution_characterization (x y : ℝ) :
  (|x| + |y| = 1340) ∧ (x^3 + y^3 + 2010*x*y = 670^3) →
  (x + y = 670) ∧ (x * y = -673350) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l590_59087


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l590_59095

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m ≥ 24 ∧ (m - 24) % 5 = 0 ∧ (m - 24) % 10 = 0 ∧ (m - 24) % 15 = 0 ∧ (m - 24) % 20 = 0 → m ≥ n) ∧
  n ≥ 24 ∧ (n - 24) % 5 = 0 ∧ (n - 24) % 10 = 0 ∧ (n - 24) % 15 = 0 ∧ (n - 24) % 20 = 0 →
  n = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l590_59095


namespace NUMINAMATH_CALUDE_parallelogram_area_l590_59047

def v : Fin 2 → ℝ := ![5, -3]
def w : Fin 2 → ℝ := ![11, -2]

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 23 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l590_59047


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l590_59078

def f (x : ℝ) := x^2 - 6*x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l590_59078


namespace NUMINAMATH_CALUDE_inequality_proof_l590_59074

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  2 * m + 1 / (m^2 - 2*m*n + n^2) ≥ 2 * n + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l590_59074


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l590_59009

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 1) : ℂ).re = 0 ∧ (Complex.I * (x + 1) : ℂ).im ≠ 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l590_59009


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l590_59096

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (size : ℕ)
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Adds a border of white tiles to a given pattern -/
def add_border (pattern : TilePattern) : TilePattern :=
  { size := pattern.size + 2,
    black_tiles := pattern.black_tiles,
    white_tiles := pattern.white_tiles + (pattern.size + 2)^2 - pattern.size^2 }

/-- The ratio of black tiles to white tiles -/
def tile_ratio (pattern : TilePattern) : ℚ :=
  pattern.black_tiles / (pattern.black_tiles + pattern.white_tiles)

theorem extended_pattern_ratio :
  let initial_pattern : TilePattern := ⟨6, 12, 24⟩
  let extended_pattern := add_border initial_pattern
  tile_ratio extended_pattern = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_ratio_l590_59096


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l590_59058

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l590_59058


namespace NUMINAMATH_CALUDE_hyperbola_sum_l590_59002

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 5 ∧ 
  k = 0 ∧ 
  c = 10 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 10 + 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l590_59002


namespace NUMINAMATH_CALUDE_red_light_estimation_l590_59092

theorem red_light_estimation (total_surveyed : ℕ) (yes_answers : ℕ) :
  total_surveyed = 600 →
  yes_answers = 180 →
  let prob_odd_id := (1 : ℚ) / 2
  let prob_yes := (yes_answers : ℚ) / total_surveyed
  let prob_red_light := 2 * prob_yes - prob_odd_id
  ⌊total_surveyed * prob_red_light⌋ = 60 := by
  sorry

end NUMINAMATH_CALUDE_red_light_estimation_l590_59092


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l590_59083

theorem greatest_common_multiple_under_120 : 
  ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  9 ∣ n ∧ 15 ∣ n ∧ n < 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l590_59083


namespace NUMINAMATH_CALUDE_mark_sprint_distance_l590_59066

/-- The distance traveled by Mark given his sprint duration and speed -/
theorem mark_sprint_distance (duration : ℝ) (speed : ℝ) (h1 : duration = 24.0) (h2 : speed = 6.0) :
  duration * speed = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_distance_l590_59066


namespace NUMINAMATH_CALUDE_june_population_calculation_l590_59026

/-- Represents the fish population model in the reservoir --/
structure FishPopulation where
  june_population : ℕ
  tagged_fish : ℕ
  october_sample : ℕ
  tagged_in_sample : ℕ

/-- Calculates the number of fish in the reservoir on June 1 --/
def calculate_june_population (model : FishPopulation) : ℕ :=
  let remaining_tagged := model.tagged_fish * 7 / 10  -- 70% of tagged fish remain
  let october_old_fish := model.october_sample / 2    -- 50% of October fish are old
  (remaining_tagged * october_old_fish) / model.tagged_in_sample

/-- Theorem stating the correct number of fish in June based on the given model --/
theorem june_population_calculation (model : FishPopulation) :
  model.tagged_fish = 100 →
  model.october_sample = 90 →
  model.tagged_in_sample = 4 →
  calculate_june_population model = 1125 :=
by
  sorry

#eval calculate_june_population ⟨1125, 100, 90, 4⟩

end NUMINAMATH_CALUDE_june_population_calculation_l590_59026


namespace NUMINAMATH_CALUDE_probability_of_three_correct_l590_59023

/-- Two fair dice are thrown once each -/
def dice : ℕ := 2

/-- Each die has 6 faces -/
def faces_per_die : ℕ := 6

/-- The numbers facing up are different -/
def different_numbers : Prop := true

/-- The probability that one of the dice shows a 3 -/
def probability_of_three : ℚ := 1 / 3

/-- Theorem stating that the probability of getting a 3 on one die when two fair dice are thrown with different numbers is 1/3 -/
theorem probability_of_three_correct (h : different_numbers) : probability_of_three = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_correct_l590_59023


namespace NUMINAMATH_CALUDE_m_range_l590_59051

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem m_range (h1 : ∀ x, x ∈ Set.Icc (-2) 2 → f x ∈ Set.Icc (-2) 2)
                (h2 : StrictMono f)
                (h3 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l590_59051


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l590_59049

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l590_59049


namespace NUMINAMATH_CALUDE_least_cubes_for_6x9x12_block_l590_59042

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the least number of equal cubes that can be cut from a block -/
def leastNumberOfEqualCubes (d : BlockDimensions) : ℕ :=
  (d.length * d.width * d.height) / (Nat.gcd d.length (Nat.gcd d.width d.height))^3

/-- Theorem stating that for a 6x9x12 cm block, the least number of equal cubes is 24 -/
theorem least_cubes_for_6x9x12_block :
  leastNumberOfEqualCubes ⟨6, 9, 12⟩ = 24 := by
  sorry

#eval leastNumberOfEqualCubes ⟨6, 9, 12⟩

end NUMINAMATH_CALUDE_least_cubes_for_6x9x12_block_l590_59042


namespace NUMINAMATH_CALUDE_line_slope_l590_59035

/-- The slope of the line represented by the equation x/4 - y/3 = 1 is 3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 - y / 3 = 1) → (y = (3 / 4) * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l590_59035


namespace NUMINAMATH_CALUDE_percentage_of_female_students_l590_59057

theorem percentage_of_female_students 
  (total_students : ℕ) 
  (female_percentage : ℝ) 
  (brunette_percentage : ℝ) 
  (under_5ft_percentage : ℝ) 
  (under_5ft_count : ℕ) :
  total_students = 200 →
  brunette_percentage = 50 →
  under_5ft_percentage = 50 →
  under_5ft_count = 30 →
  (female_percentage / 100) * (brunette_percentage / 100) * (under_5ft_percentage / 100) * total_students = under_5ft_count →
  female_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_female_students_l590_59057


namespace NUMINAMATH_CALUDE_peter_mowing_time_l590_59013

/-- The time it takes Nancy to mow the yard alone (in hours) -/
def nancy_time : ℝ := 3

/-- The time it takes Nancy and Peter together to mow the yard (in hours) -/
def combined_time : ℝ := 1.71428571429

/-- The time it takes Peter to mow the yard alone (in hours) -/
def peter_time : ℝ := 4

/-- Theorem stating that given Nancy's time and the combined time, Peter's individual time is approximately 4 hours -/
theorem peter_mowing_time (ε : ℝ) (h_ε : ε > 0) :
  ∃ (t : ℝ), abs (t - peter_time) < ε ∧ 
  1 / nancy_time + 1 / t = 1 / combined_time :=
sorry


end NUMINAMATH_CALUDE_peter_mowing_time_l590_59013


namespace NUMINAMATH_CALUDE_population_growth_l590_59037

/-- Given an initial population and two consecutive percentage increases,
    calculate the final population after both increases. -/
def final_population (initial : ℕ) (increase1 : ℚ) (increase2 : ℚ) : ℚ :=
  initial * (1 + increase1) * (1 + increase2)

/-- Theorem stating that the population after two years of growth is 1320. -/
theorem population_growth : final_population 1000 (1/10) (1/5) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l590_59037


namespace NUMINAMATH_CALUDE_intersection_A_B_l590_59061

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l590_59061


namespace NUMINAMATH_CALUDE_quadratic_functions_problem_l590_59015

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a quadratic function -/
def x_intercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_problem 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (100 - x))
  (h2 : vertex f ∈ x_intercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : {x₁, x₂, x₃, x₄} ⊆ x_intercepts f ∪ x_intercepts g)
  (h4 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h5 : x₃ - x₂ = 150)
  : x₄ - x₁ = 450 + 300 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_problem_l590_59015


namespace NUMINAMATH_CALUDE_expression_simplification_l590_59052

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - m / (m + 3)) / ((m^2 - 9) / (m^2 + 6*m + 9)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l590_59052


namespace NUMINAMATH_CALUDE_count_100_digit_even_numbers_l590_59001

/-- A function that represents the count of n-digit even numbers where each digit is 0, 1, or 3 -/
def countEvenNumbers (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * 3^(n - 2)

/-- Theorem stating that the count of 100-digit even numbers where each digit is 0, 1, or 3 is 2 * 3^98 -/
theorem count_100_digit_even_numbers :
  countEvenNumbers 100 = 2 * 3^98 := by
  sorry


end NUMINAMATH_CALUDE_count_100_digit_even_numbers_l590_59001


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_36_l590_59021

theorem consecutive_integers_sum_36 : 
  ∃! (a : ℕ), a > 0 ∧ a + (a + 1) + (a + 2) = 36 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_36_l590_59021


namespace NUMINAMATH_CALUDE_cathy_cookies_l590_59022

theorem cathy_cookies (total : ℝ) (amy_fraction : ℝ) (bob_cookies : ℝ) : 
  total = 18 → 
  amy_fraction = 1/3 → 
  bob_cookies = 2.5 → 
  total - (amy_fraction * total + bob_cookies) = 9.5 := by
sorry

end NUMINAMATH_CALUDE_cathy_cookies_l590_59022


namespace NUMINAMATH_CALUDE_f_properties_l590_59041

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_properties :
  is_smallest_positive_period π f ∧
  is_monotonically_decreasing f (π / 3) (5 * π / 6) ∧
  ∀ α : ℝ, 
    (3 * π / 2 < α ∧ α < 2 * π) →  -- α in fourth quadrant
    Real.cos α = 3 / 5 → 
    f (α / 2 + 7 * π / 12) = 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l590_59041


namespace NUMINAMATH_CALUDE_series_sum_equals_inverse_sqrt5_minus_1_l590_59030

/-- The sum of the series $\sum_{k=0}^{\infty} \frac{5^{2^k}}{25^{2^k} - 1}$ is equal to $\frac{1}{\sqrt{5}-1}$ -/
theorem series_sum_equals_inverse_sqrt5_minus_1 :
  let series_term (k : ℕ) := (5 ^ (2 ^ k)) / ((25 ^ (2 ^ k)) - 1)
  ∑' (k : ℕ), series_term k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_inverse_sqrt5_minus_1_l590_59030


namespace NUMINAMATH_CALUDE_circle_condition_l590_59008

theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l590_59008


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l590_59005

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 →
  g = -a - c - e →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = -2 * Complex.I →
  d + f + h = -4 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l590_59005


namespace NUMINAMATH_CALUDE_natalies_height_l590_59073

/-- Prove that Natalie's height is 176 cm given the conditions -/
theorem natalies_height (h_natalie : ℝ) (h_harpreet : ℝ) (h_jiayin : ℝ) 
  (h_same_height : h_natalie = h_harpreet)
  (h_jiayin_height : h_jiayin = 161)
  (h_average : (h_natalie + h_harpreet + h_jiayin) / 3 = 171) :
  h_natalie = 176 := by
  sorry

end NUMINAMATH_CALUDE_natalies_height_l590_59073


namespace NUMINAMATH_CALUDE_will_toy_cost_l590_59048

def toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℚ :=
  (initial_money - game_cost) / num_toys

theorem will_toy_cost :
  toy_cost 57 27 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_toy_cost_l590_59048


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l590_59039

theorem quadratic_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + m - 1 = 0 ∧ 
                x₂^2 - 4*x₂ + m - 1 = 0 ∧ 
                3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l590_59039


namespace NUMINAMATH_CALUDE_solution_is_eight_l590_59085

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg (2^x + 2*x - 16) = x * (1 - lg 5)

-- Theorem statement
theorem solution_is_eight : 
  ∃ (x : ℝ), equation x ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_solution_is_eight_l590_59085


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l590_59089

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

/-- Theorem stating the minimum perimeter of two noncongruent isosceles triangles
    with the same area and bases in the ratio 3:2 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    t1.base * 2 = t2.base * 3 ∧
    perimeter t1 = perimeter t2 ∧
    perimeter t1 = 508 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      area s1 = area s2 →
      s1.base * 2 = s2.base * 3 →
      perimeter s1 = perimeter s2 →
      perimeter s1 ≥ 508) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l590_59089


namespace NUMINAMATH_CALUDE_flag_pole_height_l590_59062

/-- Given a tree and a flag pole casting shadows at the same time, 
    calculate the height of the flag pole. -/
theorem flag_pole_height 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (flag_shadow : ℝ) 
  (h_tree_height : tree_height = 12)
  (h_tree_shadow : tree_shadow = 8)
  (h_flag_shadow : flag_shadow = 100) :
  (tree_height / tree_shadow) * flag_shadow = 150 :=
by
  sorry

#check flag_pole_height

end NUMINAMATH_CALUDE_flag_pole_height_l590_59062


namespace NUMINAMATH_CALUDE_unique_number_with_property_l590_59063

/-- Calculate the total number of digits needed to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that the number, when doubled, equals the total number of digits -/
def hasProperty (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ 2 * x = totalDigits x

theorem unique_number_with_property :
  ∃! x : ℕ, hasProperty x ∧ x = 108 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_property_l590_59063


namespace NUMINAMATH_CALUDE_bakery_combinations_l590_59076

/-- The number of ways to distribute n identical items among k groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The number of remaining rolls to distribute -/
def remaining_rolls : ℕ := 2

theorem bakery_combinations :
  distribute remaining_rolls num_roll_types = 10 := by
  sorry

end NUMINAMATH_CALUDE_bakery_combinations_l590_59076


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l590_59011

/-- The probability of selecting 2 red balls from a bag containing 3 red, 2 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) 
  (h_red : red = 3) 
  (h_blue : blue = 2) 
  (h_green : green = 4) : 
  (red.choose 2 : ℚ) / ((red + blue + green).choose 2) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l590_59011


namespace NUMINAMATH_CALUDE_tangent_forms_345_triangle_l590_59072

/-- An isosceles triangle with leg 10 cm and base 12 cm -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ
  leg_positive : 0 < leg
  base_positive : 0 < base
  isosceles : leg = 10
  base_length : base = 12

/-- The inscribed circle of the triangle -/
def inscribed_circle (t : IsoscelesTriangle) : ℝ := sorry

/-- Tangent line to the inscribed circle parallel to the height of the triangle -/
def tangent_line (t : IsoscelesTriangle) (c : ℝ) : ℝ → ℝ := sorry

/-- Right triangle formed by the tangent line -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse_positive : 0 < hypotenuse
  leg1_positive : 0 < leg1
  leg2_positive : 0 < leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- The theorem to be proved -/
theorem tangent_forms_345_triangle (t : IsoscelesTriangle) (c : ℝ) :
  ∃ (rt : RightTriangle), rt.leg1 = 3 ∧ rt.leg2 = 4 ∧ rt.hypotenuse = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_forms_345_triangle_l590_59072


namespace NUMINAMATH_CALUDE_two_points_imply_line_in_plane_l590_59091

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define what it means for a point to be on a line
variable (on_line : Point → Line → Prop)

-- Define what it means for a point to be within a plane
variable (in_plane : Point → Plane → Prop)

-- Define what it means for a line to be within a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem two_points_imply_line_in_plane 
  (a : Line) (α : Plane) (A B : Point) 
  (h1 : A ≠ B) 
  (h2 : on_line A a) 
  (h3 : on_line B a) 
  (h4 : in_plane A α) 
  (h5 : in_plane B α) : 
  line_in_plane a α :=
sorry

end NUMINAMATH_CALUDE_two_points_imply_line_in_plane_l590_59091


namespace NUMINAMATH_CALUDE_cyclists_speed_l590_59093

theorem cyclists_speed (initial_distance : ℝ) (fly_speed : ℝ) (fly_distance : ℝ) :
  initial_distance = 50 ∧ 
  fly_speed = 15 ∧ 
  fly_distance = 37.5 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed = 10 ∧ 
    initial_distance = 2 * cyclist_speed * (fly_distance / fly_speed) :=
by sorry

end NUMINAMATH_CALUDE_cyclists_speed_l590_59093


namespace NUMINAMATH_CALUDE_handshaking_arrangements_mod_1000_l590_59019

/-- Represents a handshaking arrangement for a group of people -/
structure HandshakingArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j)

/-- The number of valid handshaking arrangements for 12 people -/
def M : ℕ := sorry

/-- Theorem stating that the number of handshaking arrangements M for 12 people,
    where each person shakes hands with exactly 3 others, satisfies M ≡ 50 (mod 1000) -/
theorem handshaking_arrangements_mod_1000 :
  M ≡ 50 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshaking_arrangements_mod_1000_l590_59019


namespace NUMINAMATH_CALUDE_fraction_product_l590_59024

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 * 17 / 19 = 1870 / 5187 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l590_59024


namespace NUMINAMATH_CALUDE_f_properties_l590_59090

def f (x : ℝ) := |2*x + 3| + |2*x - 1|

theorem f_properties :
  (∀ x : ℝ, f x < 10 ↔ x ∈ Set.Ioo (-3) 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ |a - 1|) ↔ a ∈ Set.Icc (-2) 5) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l590_59090


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l590_59064

/-- The decimal representation 0.142857142857... as a real number -/
def a : ℚ := 142857 / 999999

/-- The decimal representation 0.285714285714... as a real number -/
def b : ℚ := 285714 / 999999

/-- Theorem stating that the ratio of the two repeating decimals is 1/2 -/
theorem repeating_decimal_ratio : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l590_59064


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_equality_l590_59059

theorem largest_n_for_square_sum_equality : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ n → j ≤ n → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ m → j ≤ m → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_equality_l590_59059


namespace NUMINAMATH_CALUDE_angle_measure_problem_l590_59025

/-- Two angles are complementary if their measures sum to 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Given two complementary angles A and B, where A is 5 times B, prove A is 75 degrees -/
theorem angle_measure_problem (A B : ℝ) 
  (h1 : complementary A B) 
  (h2 : A = 5 * B) : 
  A = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l590_59025


namespace NUMINAMATH_CALUDE_attendance_proof_l590_59099

/-- Calculates the total attendance given the number of adults and children -/
def total_attendance (adults : ℕ) (children : ℕ) : ℕ :=
  adults + children

/-- Theorem: The total attendance for 280 adults and 120 children is 400 -/
theorem attendance_proof :
  total_attendance 280 120 = 400 := by
  sorry

end NUMINAMATH_CALUDE_attendance_proof_l590_59099


namespace NUMINAMATH_CALUDE_total_divisors_xyz_l590_59040

-- Define the variables and their properties
variable (p q r : ℕ) -- Natural numbers for primes
variable (hp : Prime p) -- p is prime
variable (hq : Prime q) -- q is prime
variable (hr : Prime r) -- r is prime
variable (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) -- p, q, and r are distinct

-- Define x, y, and z
def x : ℕ := p^2
def y : ℕ := q^2
def z : ℕ := r^4

-- State the theorem
theorem total_divisors_xyz (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  (Finset.card (Nat.divisors ((x p)^3 * (y q)^4 * (z r)^2))) = 567 := by
  sorry

end NUMINAMATH_CALUDE_total_divisors_xyz_l590_59040


namespace NUMINAMATH_CALUDE_function_value_proof_l590_59086

theorem function_value_proof (x : ℝ) (f : ℝ → ℝ) 
  (h1 : x > 0) 
  (h2 : x + 17 = 60 * f x) 
  (h3 : x = 3) : 
  f 3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_proof_l590_59086


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l590_59084

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l590_59084


namespace NUMINAMATH_CALUDE_f_ln_2_value_l590_59029

/-- A function f is monotonically decreasing on (0, +∞) if for any a, b ∈ (0, +∞) with a < b, f(a) ≥ f(b) -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 < a → a < b → f a ≥ f b

/-- The main theorem -/
theorem f_ln_2_value (f : ℝ → ℝ) 
  (h_mono : MonoDecreasing f)
  (h_domain : ∀ x, x > 0 → f x ≠ 0)
  (h_eq : ∀ x, x > 0 → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_ln_2_value_l590_59029


namespace NUMINAMATH_CALUDE_steak_price_per_pound_l590_59034

theorem steak_price_per_pound (steak_price : ℚ) : 
  4.5 * steak_price + 1.5 * 8 = 42 → steak_price = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_steak_price_per_pound_l590_59034


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l590_59018

variable (a : ℝ)

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l590_59018


namespace NUMINAMATH_CALUDE_rug_area_calculation_l590_59014

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 204 square meters given the specific dimensions -/
theorem rug_area_calculation :
  rugArea 25 20 4 = 204 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_calculation_l590_59014


namespace NUMINAMATH_CALUDE_average_age_of_three_l590_59045

/-- Given the ages of Omi, Kimiko, and Arlette, prove their average age is 35 --/
theorem average_age_of_three (kimiko_age : ℕ) (omi_age : ℕ) (arlette_age : ℕ) 
  (h1 : kimiko_age = 28) 
  (h2 : omi_age = 2 * kimiko_age) 
  (h3 : arlette_age = 3 * kimiko_age / 4) : 
  (kimiko_age + omi_age + arlette_age) / 3 = 35 := by
  sorry

#check average_age_of_three

end NUMINAMATH_CALUDE_average_age_of_three_l590_59045


namespace NUMINAMATH_CALUDE_complex_number_proof_l590_59075

theorem complex_number_proof : 
  ∀ (z : ℂ), (Complex.im ((1 + 2*Complex.I) * z) = 0) → (Complex.abs z = Real.sqrt 5) → 
  (z = 1 - 2*Complex.I ∨ z = -1 + 2*Complex.I) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l590_59075


namespace NUMINAMATH_CALUDE_bens_car_cost_ratio_l590_59056

theorem bens_car_cost_ratio :
  let old_car_cost : ℚ := 1800
  let new_car_cost : ℚ := 2000 + 1800
  (new_car_cost / old_car_cost) = 19 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bens_car_cost_ratio_l590_59056


namespace NUMINAMATH_CALUDE_line_intersects_circle_l590_59080

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through point P
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - point_P.1)

-- Theorem statement
theorem line_intersects_circle :
  ∃ (m : ℝ) (x y : ℝ), line_through_P m x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l590_59080


namespace NUMINAMATH_CALUDE_bill_insurance_cost_l590_59007

def monthly_plan_price : ℚ := 500
def hourly_rate : ℚ := 25
def weekly_hours : ℚ := 30
def weeks_per_month : ℚ := 4
def months_per_year : ℚ := 12

def annual_income (rate : ℚ) (hours : ℚ) (weeks : ℚ) (months : ℚ) : ℚ :=
  rate * hours * weeks * months

def subsidy_rate (income : ℚ) : ℚ :=
  if income < 10000 then 0.9
  else if income ≤ 40000 then 0.5
  else 0.2

def annual_insurance_cost (plan_price : ℚ) (income : ℚ) (months : ℚ) : ℚ :=
  plan_price * (1 - subsidy_rate income) * months

theorem bill_insurance_cost :
  let income := annual_income hourly_rate weekly_hours weeks_per_month months_per_year
  annual_insurance_cost monthly_plan_price income months_per_year = 3000 :=
by sorry

end NUMINAMATH_CALUDE_bill_insurance_cost_l590_59007


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l590_59004

theorem right_triangle_hypotenuse : 
  ∀ (hypotenuse : ℝ), 
  hypotenuse > 0 →
  (hypotenuse - 1)^2 + 7^2 = hypotenuse^2 →
  hypotenuse = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l590_59004


namespace NUMINAMATH_CALUDE_total_weight_of_four_l590_59028

theorem total_weight_of_four (jim steve stan tim : ℕ) : 
  jim = 110 →
  steve = jim - 8 →
  stan = steve + 5 →
  tim = stan + 12 →
  jim + steve + stan + tim = 438 := by
sorry

end NUMINAMATH_CALUDE_total_weight_of_four_l590_59028


namespace NUMINAMATH_CALUDE_officer_selection_ways_l590_59068

theorem officer_selection_ways (n : ℕ) (h : n = 8) : 
  (n.factorial / (n - 3).factorial) = 336 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l590_59068


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l590_59033

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  x + y = 40 →
  x - y = 8 →
  (∃ y' : ℝ, InverselyProportional 7 y' ∧ y' = 384 / 7) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l590_59033


namespace NUMINAMATH_CALUDE_real_solutions_exist_l590_59070

theorem real_solutions_exist : ∃ x : ℝ, x^4 - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_exist_l590_59070
