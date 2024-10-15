import Mathlib

namespace NUMINAMATH_CALUDE_shirt_cost_l3548_354809

theorem shirt_cost (num_shirts : ℕ) (num_jeans : ℕ) (total_earnings : ℕ) :
  num_shirts = 20 →
  num_jeans = 10 →
  total_earnings = 400 →
  ∃ (shirt_cost : ℕ),
    shirt_cost * num_shirts + (2 * shirt_cost) * num_jeans = total_earnings ∧
    shirt_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_l3548_354809


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3548_354810

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3548_354810


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3548_354854

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 60 → 
  correct_answers = 34 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong) = 110 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3548_354854


namespace NUMINAMATH_CALUDE_lindas_lunchbox_theorem_l3548_354838

/-- Represents the cost calculation at Linda's Lunchbox -/
def lindas_lunchbox_cost (sandwich_price : ℝ) (soda_price : ℝ) (discount_rate : ℝ) 
  (discount_threshold : ℕ) (num_sandwiches : ℕ) (num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items ≥ discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- Theorem: The cost of 7 sandwiches and 5 sodas at Linda's Lunchbox is $38.7 -/
theorem lindas_lunchbox_theorem : 
  lindas_lunchbox_cost 4 3 0.1 10 7 5 = 38.7 := by
  sorry

end NUMINAMATH_CALUDE_lindas_lunchbox_theorem_l3548_354838


namespace NUMINAMATH_CALUDE_rich_walk_distance_l3548_354868

/-- Calculates the total distance Rich walks based on the given conditions -/
def total_distance : ℝ :=
  let initial_distance := 20 + 200
  let left_turn_distance := 2 * initial_distance
  let halfway_distance := initial_distance + left_turn_distance
  let final_distance := halfway_distance + 0.5 * halfway_distance
  2 * final_distance

/-- Theorem stating that the total distance Rich walks is 1980 feet -/
theorem rich_walk_distance : total_distance = 1980 := by
  sorry

end NUMINAMATH_CALUDE_rich_walk_distance_l3548_354868


namespace NUMINAMATH_CALUDE_unique_solution_system_l3548_354850

theorem unique_solution_system (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * (x + y + z) = 26 ∧ y * (x + y + z) = 27 ∧ z * (x + y + z) = 28 →
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3548_354850


namespace NUMINAMATH_CALUDE_intersection_count_l3548_354817

/-- The complementary curve C₂ -/
def complementary_curve (x y : ℝ) : Prop := 1 / x^2 - 1 / y^2 = 1

/-- The hyperbola C₁ -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line MN passing through (m,0) and (0,n) -/
def line_mn (m n x y : ℝ) : Prop := y = -n/m * x + n

theorem intersection_count (m n : ℝ) :
  complementary_curve m n →
  ∃! p : ℝ × ℝ, hyperbola p.1 p.2 ∧ line_mn m n p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l3548_354817


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3548_354896

/-- A line passing through (2,4) and tangent to (x-1)^2 + (y-2)^2 = 1 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∨ (3 * p.1 - 4 * p.2 + 10 = 0)}

/-- The circle (x-1)^2 + (y-2)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1}

/-- The point (2,4) -/
def Point : ℝ × ℝ := (2, 4)

theorem tangent_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
    (Point ∈ L) →
    (∃! p, p ∈ L ∩ Circle) →
    L = TangentLine :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3548_354896


namespace NUMINAMATH_CALUDE_rectangle_tangent_circles_l3548_354824

/-- Given a rectangle ABCD with side lengths a and b, and two externally tangent circles
    inside the rectangle, one tangent to AB and AD, the other tangent to CB and CD,
    this theorem proves properties about the distance between circle centers and
    the locus of their tangency point. -/
theorem rectangle_tangent_circles
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let d := (Real.sqrt a - Real.sqrt b) ^ 2
  let m := min a b
  let p₁ := (a - m / 2, b - m / 2)
  let p₂ := (m / 2 + Real.sqrt (2 * a * b) - b, m / 2 + Real.sqrt (2 * a * b) - a)
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    -- c₁ and c₂ are the centers of the circles
    -- r₁ and r₂ are the radii of the circles
    -- The circles are inside the rectangle
    c₁.1 ∈ Set.Icc 0 a ∧ c₁.2 ∈ Set.Icc 0 b ∧
    c₂.1 ∈ Set.Icc 0 a ∧ c₂.2 ∈ Set.Icc 0 b ∧
    -- The circles are tangent to the sides of the rectangle
    c₁.1 = r₁ ∧ c₁.2 = r₁ ∧
    c₂.1 = a - r₂ ∧ c₂.2 = b - r₂ ∧
    -- The circles are externally tangent to each other
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = (r₁ + r₂) ^ 2 ∧
    -- The distance between the centers is d
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = d ^ 2 ∧
    -- The locus of the tangency point is a line segment
    ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
      let p := (1 - t) • p₁ + t • p₂
      p.1 = (r₁ * (a - r₁ - r₂)) / (r₁ + r₂) + r₁ ∧
      p.2 = (r₁ * (b - r₁ - r₂)) / (r₁ + r₂) + r₁ :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_tangent_circles_l3548_354824


namespace NUMINAMATH_CALUDE_symmetric_circle_l3548_354883

/-- Given a circle C1 and a line of symmetry, this theorem proves the equation of the symmetric circle C2. -/
theorem symmetric_circle (x y : ℝ) : 
  (∃ C1 : ℝ × ℝ → Prop, C1 = λ (x, y) ↦ (x - 3)^2 + (y + 1)^2 = 1) →
  (∃ L : ℝ × ℝ → Prop, L = λ (x, y) ↦ 2*x - y - 2 = 0) →
  (∃ C2 : ℝ × ℝ → Prop, C2 = λ (x, y) ↦ (x + 1)^2 + (y - 1)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l3548_354883


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3548_354821

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3548_354821


namespace NUMINAMATH_CALUDE_inverse_function_problem_l3548_354867

/-- Given a function h and its inverse f⁻¹, prove that 7c + 7d = 2 -/
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, (7 * x - 6 : ℝ) = (Function.invFun (fun x ↦ c * x + d) x - 5)) →
  7 * c + 7 * d = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l3548_354867


namespace NUMINAMATH_CALUDE_car_p_distance_l3548_354855

/-- The distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_p_distance (v : ℝ) : 
  let car_m_speed := v
  let car_m_time := 3
  let car_n_speed := 3 * v
  let car_n_time := 2
  let car_p_speed := 2 * v
  let car_p_time := 1.5
  distance car_p_speed car_p_time = 3 * v :=
by sorry

end NUMINAMATH_CALUDE_car_p_distance_l3548_354855


namespace NUMINAMATH_CALUDE_kenny_book_purchase_l3548_354888

def lawn_price : ℕ := 15
def video_game_price : ℕ := 45
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5

def total_earned : ℕ := lawn_price * lawns_mowed
def video_games_cost : ℕ := video_game_price * video_games_wanted
def remaining_money : ℕ := total_earned - video_games_cost

theorem kenny_book_purchase :
  remaining_money / book_price = 60 := by sorry

end NUMINAMATH_CALUDE_kenny_book_purchase_l3548_354888


namespace NUMINAMATH_CALUDE_notebook_distribution_l3548_354891

theorem notebook_distribution (total_notebooks : ℕ) 
  (h1 : total_notebooks = 512) : 
  ∃ (num_children : ℕ), 
    (num_children > 0) ∧ 
    (total_notebooks = num_children * (num_children / 8)) ∧
    (total_notebooks = (num_children / 2) * 16) := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3548_354891


namespace NUMINAMATH_CALUDE_sol_earnings_l3548_354825

def candy_sales (day : Nat) : Nat :=
  10 + 4 * (day - 1)

def total_sales : Nat :=
  (List.range 6).map (λ i => candy_sales (i + 1)) |>.sum

def earnings_cents : Nat :=
  total_sales * 10

theorem sol_earnings :
  earnings_cents / 100 = 12 := by sorry

end NUMINAMATH_CALUDE_sol_earnings_l3548_354825


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3548_354833

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters needed to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3548_354833


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3548_354813

/-- Represents the problem of calculating the profit percentage for a retailer --/
theorem retailer_profit_percentage 
  (monthly_sales : ℕ)
  (profit_per_item : ℚ)
  (discount_rate : ℚ)
  (break_even_sales : ℚ)
  (h1 : monthly_sales = 100)
  (h2 : profit_per_item = 30)
  (h3 : discount_rate = 0.05)
  (h4 : break_even_sales = 156.86274509803923)
  : ∃ (item_price : ℚ), 
    profit_per_item / item_price = 0.16 :=
by sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3548_354813


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3548_354804

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3548_354804


namespace NUMINAMATH_CALUDE_track_event_races_l3548_354826

/-- The number of races needed to determine a champion in a track event -/
def races_needed (total_athletes : ℕ) (lanes_per_race : ℕ) : ℕ :=
  let first_round := total_athletes / lanes_per_race
  let second_round := first_round / lanes_per_race
  let final_round := 1
  first_round + second_round + final_round

/-- Theorem stating that 43 races are needed for 216 athletes with 6 lanes per race -/
theorem track_event_races : races_needed 216 6 = 43 := by
  sorry

#eval races_needed 216 6

end NUMINAMATH_CALUDE_track_event_races_l3548_354826


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l3548_354839

/-- Represents a frustum with given height and two spheres placed inside it. -/
structure Frustum :=
  (height : ℝ)
  (sphere1_radius : ℝ)
  (sphere2_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can be placed in the frustum. -/
def max_additional_spheres (f : Frustum) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres. -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h1 : f.height = 8)
  (h2 : f.sphere1_radius = 2)
  (h3 : f.sphere2_radius = 3)
  : max_additional_spheres f = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l3548_354839


namespace NUMINAMATH_CALUDE_largest_difference_in_S_l3548_354815

def S : Set ℤ := {-20, -8, 0, 6, 10, 15, 25}

theorem largest_difference_in_S : 
  ∀ (a b : ℤ), a ∈ S → b ∈ S → (a - b) ≤ 45 ∧ ∃ (x y : ℤ), x ∈ S ∧ y ∈ S ∧ x - y = 45 :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_in_S_l3548_354815


namespace NUMINAMATH_CALUDE_melanie_remaining_plums_l3548_354861

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := 7

/-- The number of plums Melanie gave away -/
def plums_given_away : ℕ := 3

/-- Theorem: Melanie has 4 plums after giving some away -/
theorem melanie_remaining_plums : 
  initial_plums - plums_given_away = 4 := by sorry

end NUMINAMATH_CALUDE_melanie_remaining_plums_l3548_354861


namespace NUMINAMATH_CALUDE_family_size_family_size_proof_l3548_354852

theorem family_size : ℕ → Prop :=
  fun n =>
    ∀ (b : ℕ),
      -- Peter has b brothers and 3b sisters
      (3 * b = n - b - 1) →
      -- Louise has b + 1 brothers and 3b - 1 sisters
      (3 * b - 1 = 2 * (b + 1)) →
      n = 13

-- The proof is omitted
theorem family_size_proof : family_size 13 := by sorry

end NUMINAMATH_CALUDE_family_size_family_size_proof_l3548_354852


namespace NUMINAMATH_CALUDE_intersection_equality_l3548_354808

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem intersection_equality (m : ℝ) : 
  A m ∩ B m = B m → m = 3 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3548_354808


namespace NUMINAMATH_CALUDE_complex_equation_roots_l3548_354836

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 1 - I ∧ z₂ = -3 + I ∧ 
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ 
  (z₂^2 + 2*z₂ = 3 - 4*I) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l3548_354836


namespace NUMINAMATH_CALUDE_quadratic_one_solution_positive_m_value_l3548_354860

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

theorem positive_m_value (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ m > 0 → m = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_positive_m_value_l3548_354860


namespace NUMINAMATH_CALUDE_bob_discount_percentage_l3548_354844

def bob_bill : ℝ := 30
def kate_bill : ℝ := 25
def total_after_discount : ℝ := 53

theorem bob_discount_percentage :
  let total_before_discount := bob_bill + kate_bill
  let discount_amount := total_before_discount - total_after_discount
  let discount_percentage := (discount_amount / bob_bill) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |discount_percentage - 6.67| < ε :=
sorry

end NUMINAMATH_CALUDE_bob_discount_percentage_l3548_354844


namespace NUMINAMATH_CALUDE_no_y_term_in_polynomial_l3548_354862

theorem no_y_term_in_polynomial (x y k : ℝ) : 
  (2*x - 3*y + 4 + 3*k*x + 2*k*y - k = (2 + 3*k)*x + (-k + 4)) → k = 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_y_term_in_polynomial_l3548_354862


namespace NUMINAMATH_CALUDE_bulb_selection_problem_l3548_354801

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (prob_at_least_one_defective : ℝ) :
  total_bulbs = 22 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.33766233766233766 →
  ∃ n : ℕ, n = 2 ∧ 
    (1 - ((total_bulbs - defective_bulbs : ℝ) / total_bulbs) ^ n) = prob_at_least_one_defective :=
by sorry

end NUMINAMATH_CALUDE_bulb_selection_problem_l3548_354801


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3548_354834

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧ 
  a > 3 ∧ b > 6 ∧ c > 12 ∧
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 2143 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3548_354834


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l3548_354811

/-- Calculates the downstream speed of a rower given their upstream speed and still water speed. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that a rower with an upstream speed of 12 kmph and a still water speed of 25 kmph
    will have a downstream speed of 38 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 12 25 = 38 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l3548_354811


namespace NUMINAMATH_CALUDE_fifteenth_term_binomial_expansion_l3548_354877

theorem fifteenth_term_binomial_expansion : 
  let n : ℕ := 20
  let k : ℕ := 14
  let z : ℂ := -1 + Complex.I
  Nat.choose n k * (-1)^(n - k) * Complex.I^k = -38760 := by sorry

end NUMINAMATH_CALUDE_fifteenth_term_binomial_expansion_l3548_354877


namespace NUMINAMATH_CALUDE_chandra_pairings_l3548_354846

/-- Represents the number of valid pairings between bowls and glasses -/
def valid_pairings (num_bowls num_glasses num_unmatched : ℕ) : ℕ :=
  (num_bowls - num_unmatched) * num_glasses + num_unmatched * num_glasses

/-- Theorem: Given 5 bowls and 4 glasses, where one bowl doesn't have a matching glass,
    the total number of valid pairings is 20 -/
theorem chandra_pairings :
  valid_pairings 5 4 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chandra_pairings_l3548_354846


namespace NUMINAMATH_CALUDE_problem_solution_l3548_354882

noncomputable section

def f (x : ℝ) := 3 - 2 * Real.log x / Real.log 2
def g (x : ℝ) := Real.log x / Real.log 2
def h (x : ℝ) := (f x + 1) * g x
def M (x : ℝ) := max (g x) (f x)

theorem problem_solution :
  (∀ x ∈ Set.Icc 1 8, h x ∈ Set.Icc (-6) 2) ∧
  (∀ x > 0, M x ≤ 1) ∧
  (∃ x > 0, M x = 1) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc 1 8, f (x^2) * f (Real.sqrt x) ≥ k * g x) → k ≤ -3) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l3548_354882


namespace NUMINAMATH_CALUDE_a_values_l3548_354857

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l3548_354857


namespace NUMINAMATH_CALUDE_candy_bar_difference_l3548_354895

theorem candy_bar_difference : 
  ∀ (bob_candy : ℕ),
  let fred_candy : ℕ := 12
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  120 = (40 : ℕ) * jacqueline_candy / 100 →
  bob_candy - fred_candy = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l3548_354895


namespace NUMINAMATH_CALUDE_slope_through_origin_and_point_l3548_354894

/-- The slope of a line passing through (0, 0) and (5, 1) is 1/5 -/
theorem slope_through_origin_and_point :
  let x1 : ℝ := 0
  let y1 : ℝ := 0
  let x2 : ℝ := 5
  let y2 : ℝ := 1
  let slope : ℝ := (y2 - y1) / (x2 - x1)
  slope = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_slope_through_origin_and_point_l3548_354894


namespace NUMINAMATH_CALUDE_dolphin_shark_ratio_l3548_354865

/-- The ratio of buckets fed to dolphins compared to sharks -/
def R : ℚ := 1 / 2

/-- The number of buckets fed to sharks daily -/
def shark_buckets : ℕ := 4

/-- The number of days in 3 weeks -/
def days : ℕ := 21

/-- The total number of buckets lasting 3 weeks -/
def total_buckets : ℕ := 546

theorem dolphin_shark_ratio :
  R * shark_buckets * days +
  shark_buckets * days +
  (5 * shark_buckets) * days = total_buckets := by sorry

end NUMINAMATH_CALUDE_dolphin_shark_ratio_l3548_354865


namespace NUMINAMATH_CALUDE_blackboard_problem_l3548_354864

/-- Represents the state of the blackboard -/
structure BoardState where
  ones : ℕ
  twos : ℕ
  threes : ℕ
  fours : ℕ

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours + 2 }
  | Operation.erase_124_add_3 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes + 2, fours := state.fours - 1 }
  | Operation.erase_134_add_2 => 
      { ones := state.ones - 1, twos := state.twos + 2, 
        threes := state.threes - 1, fours := state.fours - 1 }
  | Operation.erase_234_add_1 => 
      { ones := state.ones + 2, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours - 1 }

/-- Checks if the board state is in a final state (only three numbers remain) -/
def isFinalState (state : BoardState) : Bool :=
  (state.ones + state.twos + state.threes + state.fours) = 3

/-- Calculates the product of the remaining numbers -/
def productOfRemaining (state : BoardState) : ℕ :=
  (if state.ones > 0 then 1^state.ones else 1) *
  (if state.twos > 0 then 2^state.twos else 1) *
  (if state.threes > 0 then 3^state.threes else 1) *
  (if state.fours > 0 then 4^state.fours else 1)

/-- The main theorem to prove -/
theorem blackboard_problem :
  ∃ (operations : List Operation),
    let initialState : BoardState := { ones := 11, twos := 22, threes := 33, fours := 44 }
    let finalState := operations.foldl applyOperation initialState
    isFinalState finalState ∧ productOfRemaining finalState = 12 := by
  sorry


end NUMINAMATH_CALUDE_blackboard_problem_l3548_354864


namespace NUMINAMATH_CALUDE_pelt_costs_l3548_354807

/-- Proof of pelt costs given total cost and individual profits -/
theorem pelt_costs (total_cost : ℝ) (total_profit_percent : ℝ) 
  (profit_percent_1 : ℝ) (profit_percent_2 : ℝ) 
  (h1 : total_cost = 22500)
  (h2 : total_profit_percent = 40)
  (h3 : profit_percent_1 = 25)
  (h4 : profit_percent_2 = 50) :
  ∃ (cost_1 cost_2 : ℝ),
    cost_1 + cost_2 = total_cost ∧
    cost_1 * (1 + profit_percent_1 / 100) + cost_2 * (1 + profit_percent_2 / 100) 
      = total_cost * (1 + total_profit_percent / 100) ∧
    cost_1 = 9000 ∧
    cost_2 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_pelt_costs_l3548_354807


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_greater_than_one_l3548_354849

/-- A linear function y = mx + b decreases if and only if its slope m is negative -/
axiom decreasing_linear_function (m b : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0

/-- For the function y = (1-a)x + 2, if it decreases as x increases, then a > 1 -/
theorem decreasing_function_implies_a_greater_than_one (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (1 - a) * x₁ + 2 > (1 - a) * x₂ + 2) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_greater_than_one_l3548_354849


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_25_l3548_354874

theorem arithmetic_square_root_of_25 : Real.sqrt 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_25_l3548_354874


namespace NUMINAMATH_CALUDE_product_is_rational_l3548_354814

def primes : List Nat := [3, 5, 7, 11, 13, 17]

def product : ℚ :=
  primes.foldl (fun acc p => acc * (1 - 1 / (p * p : ℚ))) 1

theorem product_is_rational : ∃ (a b : ℕ), product = a / b :=
  sorry

end NUMINAMATH_CALUDE_product_is_rational_l3548_354814


namespace NUMINAMATH_CALUDE_min_roots_symmetric_function_l3548_354823

/-- A function with specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f (2 + x)) ∧
  (∀ x, f (7 - x) = f (7 + x)) ∧
  f 0 = 0

/-- The set of roots of f in the interval [-1000, 1000] -/
def RootSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc (-1000) 1000 ∧ f x = 0}

/-- The theorem stating the minimum number of roots -/
theorem min_roots_symmetric_function (f : ℝ → ℝ) (h : SymmetricFunction f) :
    401 ≤ (RootSet f).ncard := by
  sorry

end NUMINAMATH_CALUDE_min_roots_symmetric_function_l3548_354823


namespace NUMINAMATH_CALUDE_cafeteria_pie_problem_l3548_354880

/-- Given a cafeteria with initial apples, some handed out, and a number of pies made,
    calculate the number of apples used per pie. -/
def apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (pies_made : ℕ) : ℕ :=
  (initial_apples - handed_out) / pies_made

/-- Theorem: In the specific case of 50 initial apples, 5 handed out, and 9 pies made,
    the number of apples per pie is 5. -/
theorem cafeteria_pie_problem :
  apples_per_pie 50 5 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pie_problem_l3548_354880


namespace NUMINAMATH_CALUDE_circle_symmetry_l3548_354840

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 1 = 0

-- Define symmetry with respect to origin
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-2)^2 + y^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, original_circle x y ↔ symmetric_circle (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3548_354840


namespace NUMINAMATH_CALUDE_number_of_selection_schemes_l3548_354863

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of cities to visit -/
def total_cities : ℕ := 4

/-- The number of people who cannot visit a specific city -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for cities with restrictions -/
def selection_schemes (n m r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - 2 * ((n - 1).factorial / (n - m).factorial)

/-- The main theorem stating the number of selection schemes -/
theorem number_of_selection_schemes :
  selection_schemes total_people total_cities restricted_people = 240 := by
  sorry

end NUMINAMATH_CALUDE_number_of_selection_schemes_l3548_354863


namespace NUMINAMATH_CALUDE_abigail_spending_l3548_354806

theorem abigail_spending (initial_amount : ℝ) : 
  let food_expense := 0.6 * initial_amount
  let remainder_after_food := initial_amount - food_expense
  let phone_bill := 0.25 * remainder_after_food
  let remainder_after_phone := remainder_after_food - phone_bill
  let entertainment_expense := 20
  let final_amount := remainder_after_phone - entertainment_expense
  (final_amount = 40) → (initial_amount = 200) :=
by
  sorry

end NUMINAMATH_CALUDE_abigail_spending_l3548_354806


namespace NUMINAMATH_CALUDE_smallest_n_properties_count_non_14_divisors_l3548_354845

def is_perfect_power (x : ℕ) (k : ℕ) : Prop :=
  ∃ y : ℕ, x = y^k

def smallest_n : ℕ :=
  sorry

theorem smallest_n_properties (n : ℕ) (hn : n = smallest_n) :
  is_perfect_power (n / 2) 2 ∧
  is_perfect_power (n / 3) 3 ∧
  is_perfect_power (n / 5) 5 ∧
  is_perfect_power (n / 7) 7 :=
  sorry

theorem count_non_14_divisors (n : ℕ) (hn : n = smallest_n) :
  (Finset.filter (fun d => ¬(14 ∣ d)) (Nat.divisors n)).card = 240 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_properties_count_non_14_divisors_l3548_354845


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3548_354866

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - (x - 2)/3 > 1) ↔ (x < 1/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3548_354866


namespace NUMINAMATH_CALUDE_cyros_population_growth_l3548_354899

/-- The number of years it takes for the population to meet or exceed the island's capacity -/
def years_to_capacity (island_size : ℕ) (land_per_person : ℕ) (initial_population : ℕ) (doubling_period : ℕ) : ℕ :=
  sorry

theorem cyros_population_growth :
  years_to_capacity 32000 2 500 30 = 150 :=
sorry

end NUMINAMATH_CALUDE_cyros_population_growth_l3548_354899


namespace NUMINAMATH_CALUDE_bakery_storage_l3548_354812

theorem bakery_storage (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 5 / 4)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 3000 := by
sorry

end NUMINAMATH_CALUDE_bakery_storage_l3548_354812


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3548_354881

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3548_354881


namespace NUMINAMATH_CALUDE_concentric_circles_chord_count_l3548_354889

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle ABC is 80 degrees, then the number of segments needed to return to the starting point is 18. -/
theorem concentric_circles_chord_count (angle_ABC : ℝ) (n : ℕ) : 
  angle_ABC = 80 → n * 100 = 360 * (n / 18) → n = 18 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_count_l3548_354889


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l3548_354870

def total_dogs : ℕ := 15
def group_1_size : ℕ := 6
def group_2_size : ℕ := 5
def group_3_size : ℕ := 4

theorem dog_grouping_combinations :
  (total_dogs = group_1_size + group_2_size + group_3_size) →
  (Nat.choose (total_dogs - 2) (group_1_size - 1) * Nat.choose (total_dogs - group_1_size - 1) group_2_size = 72072) := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l3548_354870


namespace NUMINAMATH_CALUDE_three_dice_probability_l3548_354830

theorem three_dice_probability : 
  let dice := 6
  let prob_first := (3 : ℚ) / dice  -- Probability of rolling less than 4 on first die
  let prob_second := (3 : ℚ) / dice -- Probability of rolling an even number on second die
  let prob_third := (2 : ℚ) / dice  -- Probability of rolling greater than 4 on third die
  prob_first * prob_second * prob_third = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_three_dice_probability_l3548_354830


namespace NUMINAMATH_CALUDE_melissa_games_l3548_354816

theorem melissa_games (total_points : ℕ) (points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) :
  total_points / points_per_game = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_melissa_games_l3548_354816


namespace NUMINAMATH_CALUDE_book_pricing_theorem_l3548_354800

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

-- Define the production cost
def production_cost : ℕ := 5

-- Define the theorem
theorem book_pricing_theorem :
  -- Part 1: Exactly 6 values of n where C(n+1) < C(n)
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ C (n + 1) < C n) ∧
  -- Part 2: Profit range for two individuals buying 60 books
  (∀ a b : ℕ, a + b = 60 → a ≥ 1 → b ≥ 1 →
    302 ≤ C a + C b - 60 * production_cost ∧
    C a + C b - 60 * production_cost ≤ 384) :=
by sorry

end NUMINAMATH_CALUDE_book_pricing_theorem_l3548_354800


namespace NUMINAMATH_CALUDE_sqrt_two_subset_P_l3548_354828

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by sorry

end NUMINAMATH_CALUDE_sqrt_two_subset_P_l3548_354828


namespace NUMINAMATH_CALUDE_berts_spending_l3548_354884

theorem berts_spending (initial_amount : ℚ) : 
  initial_amount = 44 →
  let hardware_spent := (1 / 4) * initial_amount
  let after_hardware := initial_amount - hardware_spent
  let after_drycleaner := after_hardware - 9
  let grocery_spent := (1 / 2) * after_drycleaner
  let final_amount := after_drycleaner - grocery_spent
  final_amount = 12 := by sorry

end NUMINAMATH_CALUDE_berts_spending_l3548_354884


namespace NUMINAMATH_CALUDE_last_two_digits_square_l3548_354890

theorem last_two_digits_square (n : ℕ) : 
  (n % 100 = n^2 % 100) ↔ (n % 100 = 0 ∨ n % 100 = 1 ∨ n % 100 = 25 ∨ n % 100 = 76) := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_square_l3548_354890


namespace NUMINAMATH_CALUDE_solve_for_m_l3548_354876

theorem solve_for_m : ∀ m : ℝ, (∃ x : ℝ, x = 3 ∧ 3 * m - 2 * x = 6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3548_354876


namespace NUMINAMATH_CALUDE_base_r_square_property_l3548_354803

/-- A natural number x is representable as a two-digit number with identical digits in base r -/
def is_two_digit_identical (x r : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < r ∧ x = a * (r + 1)

/-- A natural number y is representable as a four-digit number in base r with form b00b -/
def is_four_digit_b00b (y r : ℕ) : Prop :=
  ∃ b : ℕ, 0 < b ∧ b < r ∧ y = b * (r^3 + 1)

/-- The main theorem -/
theorem base_r_square_property (r : ℕ) (hr : r ≤ 100) :
  (∃ x : ℕ, is_two_digit_identical x r ∧ is_four_digit_b00b (x^2) r) →
  r = 2 ∨ r = 23 :=
by sorry

end NUMINAMATH_CALUDE_base_r_square_property_l3548_354803


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3548_354886

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3548_354886


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l3548_354842

/-- In a triangle ABC, if sin(A)/a = cos(B)/b, then B = 45° --/
theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) :
  (0 < a) → (0 < b) → (0 < A) → (A < π) → (0 < B) → (B < π) →
  (Real.sin A / a = Real.cos B / b) →
  B = π/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l3548_354842


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3548_354879

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (3 * a^4 - 6 * a^3 + 11 * a^2 + 15 * a - 7 = 0) →
  (3 * b^4 - 6 * b^3 + 11 * b^2 + 15 * b - 7 = 0) →
  (3 * c^4 - 6 * c^3 + 11 * c^2 + 15 * c - 7 = 0) →
  (3 * d^4 - 6 * d^3 + 11 * d^2 + 15 * d - 7 = 0) →
  a^2 + b^2 + c^2 + d^2 = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3548_354879


namespace NUMINAMATH_CALUDE_white_wins_iff_n_gt_3_l3548_354873

/-- Represents the outcome of the game -/
inductive GameOutcome
  | WhiteWins
  | BlackWins

/-- Represents the game state -/
structure GameState where
  board_size : Nat
  white_position : Nat
  black_position : Nat

/-- Determines the winner of the game given the initial state -/
def determine_winner (initial_state : GameState) : GameOutcome :=
  if initial_state.board_size > 3 then GameOutcome.WhiteWins
  else GameOutcome.BlackWins

/-- Theorem stating the winning condition for the game -/
theorem white_wins_iff_n_gt_3 (n : Nat) (h : n > 2) :
  determine_winner {board_size := n, white_position := 1, black_position := n} = GameOutcome.WhiteWins ↔ n > 3 := by
  sorry


end NUMINAMATH_CALUDE_white_wins_iff_n_gt_3_l3548_354873


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3548_354859

/-- The minimum distance between two points P and Q, where P lies on the curve y = x^2 - ln(x) 
    and Q lies on the line y = x - 2, and both P and Q have the same y-coordinate, is 2. -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ → |x₂ - x₁| ≥ min_dist) ∧
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ ∧ |x₂ - x₁| = min_dist) ∧
  min_dist = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3548_354859


namespace NUMINAMATH_CALUDE_product_real_iff_condition_l3548_354820

/-- For complex numbers z₁ = a + bi and z₂ = c + di, where a, b, c, and d are real numbers,
    the product z₁ * z₂ is real if and only if ad + bc = 0. -/
theorem product_real_iff_condition (a b c d : ℝ) :
  (Complex.I * Complex.I = -1) →
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk c d
  (z₁ * z₂).im = 0 ↔ a * d + b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_real_iff_condition_l3548_354820


namespace NUMINAMATH_CALUDE_xiaohong_fruit_money_l3548_354829

/-- The price difference between 500g of apples and 500g of pears in yuan -/
def price_difference : ℚ := 55 / 100

/-- The amount saved when buying 5 kg of apples in yuan -/
def apple_savings : ℚ := 4

/-- The amount saved when buying 6 kg of pears in yuan -/
def pear_savings : ℚ := 3

/-- The price of 1 kg of pears in yuan -/
def pear_price : ℚ := 45 / 10

theorem xiaohong_fruit_money : 
  ∃ (total : ℚ), 
    total = 6 * pear_price - pear_savings ∧ 
    total = 5 * (pear_price + 2 * price_difference) - apple_savings ∧
    total = 24 := by
  sorry

end NUMINAMATH_CALUDE_xiaohong_fruit_money_l3548_354829


namespace NUMINAMATH_CALUDE_age_difference_of_parents_l3548_354875

theorem age_difference_of_parents (albert_age brother_age father_age mother_age : ℕ) :
  father_age = albert_age + 48 →
  mother_age = brother_age + 46 →
  brother_age = albert_age - 2 →
  father_age - mother_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_difference_of_parents_l3548_354875


namespace NUMINAMATH_CALUDE_mod_equiv_problem_l3548_354878

theorem mod_equiv_problem (m : ℕ) : 
  197 * 879 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equiv_problem_l3548_354878


namespace NUMINAMATH_CALUDE_monotonic_increasing_iff_a_in_range_l3548_354847

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem monotonic_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (a ∈ Set.Icc (3/2) 3) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_iff_a_in_range_l3548_354847


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3548_354831

theorem election_votes_theorem (total_votes : ℕ) : 
  (total_votes : ℚ) * (60 / 100) - (total_votes : ℚ) * (40 / 100) = 280 → 
  total_votes = 1400 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3548_354831


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3548_354805

/-- A geometric sequence with fifth term 243 and sixth term 729 has first term 3 -/
theorem geometric_sequence_first_term : ∀ (a : ℝ) (r : ℝ),
  a * r^4 = 243 →
  a * r^5 = 729 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3548_354805


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3548_354887

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 2 * x^2 + x^2 * Real.cos (1 / x) else 0

theorem f_derivative_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3548_354887


namespace NUMINAMATH_CALUDE_cost_per_bag_of_chips_l3548_354843

/-- Given three friends buying chips, prove the cost per bag --/
theorem cost_per_bag_of_chips (num_friends : ℕ) (num_bags : ℕ) (payment_per_friend : ℚ) : 
  num_friends = 3 → num_bags = 5 → payment_per_friend = 5 →
  (num_friends * payment_per_friend) / num_bags = 3 := by
sorry

end NUMINAMATH_CALUDE_cost_per_bag_of_chips_l3548_354843


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3548_354851

theorem purely_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 2*x - 3) (x + 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3548_354851


namespace NUMINAMATH_CALUDE_repair_time_for_14_people_l3548_354872

/-- Represents the time needed for a given number of people to repair the dam -/
structure RepairTime where
  people : ℕ
  minutes : ℕ

/-- The dam repair scenario -/
structure DamRepair where
  repair1 : RepairTime
  repair2 : RepairTime

/-- Calculates the time needed for a given number of people to repair the dam -/
def calculateRepairTime (d : DamRepair) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 14 people need 30 minutes to repair the dam -/
theorem repair_time_for_14_people (d : DamRepair) 
  (h1 : d.repair1 = ⟨10, 45⟩) 
  (h2 : d.repair2 = ⟨20, 20⟩) : 
  calculateRepairTime d 14 = 30 :=
sorry

end NUMINAMATH_CALUDE_repair_time_for_14_people_l3548_354872


namespace NUMINAMATH_CALUDE_sin_675_degrees_l3548_354827

theorem sin_675_degrees : Real.sin (675 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_675_degrees_l3548_354827


namespace NUMINAMATH_CALUDE_group_collection_theorem_l3548_354837

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members. -/
def totalCollectionInRupees (numberOfMembers : ℕ) : ℚ :=
  (numberOfMembers * numberOfMembers : ℚ) / 100

/-- Proves that for a group of 88 members, where each member contributes as many
    paise as there are members, the total collection amount is 77.44 rupees. -/
theorem group_collection_theorem :
  totalCollectionInRupees 88 = 77.44 := by
  sorry

#eval totalCollectionInRupees 88

end NUMINAMATH_CALUDE_group_collection_theorem_l3548_354837


namespace NUMINAMATH_CALUDE_parameterization_validity_l3548_354802

def is_valid_parameterization (x₀ y₀ dx dy : ℝ) : Prop :=
  y₀ = -3 * x₀ + 5 ∧ dy / dx = -3

theorem parameterization_validity 
  (x₀ y₀ dx dy : ℝ) (dx_nonzero : dx ≠ 0) :
  is_valid_parameterization x₀ y₀ dx dy ↔
  (∀ t : ℝ, -3 * (x₀ + t * dx) + 5 = y₀ + t * dy) :=
sorry

end NUMINAMATH_CALUDE_parameterization_validity_l3548_354802


namespace NUMINAMATH_CALUDE_sum_recurring_thirds_equals_one_l3548_354818

-- Define the recurring decimal 0.333...
def recurring_third : ℚ := 1 / 3

-- Define the recurring decimal 0.666...
def recurring_two_thirds : ℚ := 2 / 3

-- Theorem: The sum of 0.333... and 0.666... is equal to 1
theorem sum_recurring_thirds_equals_one : 
  recurring_third + recurring_two_thirds = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_recurring_thirds_equals_one_l3548_354818


namespace NUMINAMATH_CALUDE_inequality_proof_l3548_354858

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3548_354858


namespace NUMINAMATH_CALUDE_exists_72_degree_angle_l3548_354819

/-- Represents a hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  a : ℝ  -- First angle in the progression
  d : ℝ  -- Common difference

/-- The sum of angles in a hexagon is 720° -/
axiom hexagon_angle_sum (h : ArithmeticHexagon) : 
  h.a + (h.a + h.d) + (h.a + 2*h.d) + (h.a + 3*h.d) + (h.a + 4*h.d) + (h.a + 5*h.d) = 720

/-- Theorem: There exists a hexagon with angles in arithmetic progression that has a 72° angle -/
theorem exists_72_degree_angle : ∃ h : ArithmeticHexagon, 
  h.a = 72 ∨ (h.a + h.d) = 72 ∨ (h.a + 2*h.d) = 72 ∨ 
  (h.a + 3*h.d) = 72 ∨ (h.a + 4*h.d) = 72 ∨ (h.a + 5*h.d) = 72 :=
sorry

end NUMINAMATH_CALUDE_exists_72_degree_angle_l3548_354819


namespace NUMINAMATH_CALUDE_carries_remaining_money_l3548_354871

/-- The amount of money Carrie has left after shopping -/
def money_left (initial_amount sweater_price tshirt_price shoes_price jeans_original_price jeans_discount : ℚ) : ℚ :=
  initial_amount - (sweater_price + tshirt_price + shoes_price + (jeans_original_price * (1 - jeans_discount)))

/-- Proof that Carrie has $27.50 left after shopping -/
theorem carries_remaining_money :
  money_left 91 24 6 11 30 (25/100) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_carries_remaining_money_l3548_354871


namespace NUMINAMATH_CALUDE_integer_expression_l3548_354856

theorem integer_expression (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℕ, k * m = 3) ↔ 
  ∃ z : ℤ, z = (((n + 1)^2 - 3*k) / k^2) * (n.factorial / (k.factorial * (n - k).factorial)) :=
sorry

end NUMINAMATH_CALUDE_integer_expression_l3548_354856


namespace NUMINAMATH_CALUDE_unique_solution_l3548_354885

theorem unique_solution : ∀ x y : ℕ+, 
  (x : ℝ) ^ (y : ℝ) - 1 = (y : ℝ) ^ (x : ℝ) → 
  2 * (x : ℝ) ^ (y : ℝ) = (y : ℝ) ^ (x : ℝ) + 5 → 
  x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3548_354885


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3548_354892

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 25 * (x + 3)^2 + 4 * y^2 = 100

-- Define the center of the ellipse
def center : ℝ × ℝ := (-3, 0)

-- Define the semi-major and semi-minor axis lengths
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 2

-- Define the endpoints of the major and minor axes
def major_axis_endpoint : ℝ × ℝ := (-3, 5)
def minor_axis_endpoint : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem ellipse_axis_endpoints_distance :
  let C := major_axis_endpoint
  let D := minor_axis_endpoint
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoints_distance_l3548_354892


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3548_354832

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x = (81 : ℝ)^3 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3548_354832


namespace NUMINAMATH_CALUDE_distance_ratio_theorem_l3548_354853

theorem distance_ratio_theorem (x : ℝ) (h1 : x^2 + (-9)^2 = 18^2) :
  |(-9)| / 18 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_theorem_l3548_354853


namespace NUMINAMATH_CALUDE_first_day_exceeding_200_chocolates_l3548_354898

def chocolate_count (n : ℕ) : ℕ := 3 * 3^(n - 1)

theorem first_day_exceeding_200_chocolates :
  (∃ n : ℕ, n > 0 ∧ chocolate_count n > 200) ∧
  (∀ m : ℕ, m > 0 ∧ m < 6 → chocolate_count m ≤ 200) ∧
  chocolate_count 6 > 200 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_200_chocolates_l3548_354898


namespace NUMINAMATH_CALUDE_ninth_term_value_l3548_354897

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem ninth_term_value (seq : ArithmeticSequence) 
    (h6 : seq.S 6 = 3) 
    (h11 : seq.S 11 = 18) : 
  seq.a 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3548_354897


namespace NUMINAMATH_CALUDE_lucas_age_l3548_354848

theorem lucas_age (noah_age mia_age lucas_age : ℕ) : 
  noah_age = 12 →
  mia_age = noah_age + 5 →
  lucas_age = mia_age - 6 →
  lucas_age = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_age_l3548_354848


namespace NUMINAMATH_CALUDE_three_card_selections_count_l3548_354835

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of ways to select and order three different cards from a standard deck -/
def ThreeCardSelections : ℕ := StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)

/-- Theorem: The number of ways to select and order three different cards from a standard 52-card deck is 132600 -/
theorem three_card_selections_count : ThreeCardSelections = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_selections_count_l3548_354835


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l3548_354822

/-- Proves that the salary of the new supervisor is $510 given the conditions of the problem -/
theorem new_supervisor_salary
  (initial_people : ℕ)
  (initial_average_salary : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average_salary : ℚ)
  (h_initial_people : initial_people = 9)
  (h_initial_average : initial_average_salary = 430)
  (h_old_supervisor : old_supervisor_salary = 870)
  (h_new_average : new_average_salary = 390)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 510 ∧
    (initial_people - 1) * (initial_average_salary * initial_people - old_supervisor_salary) / (initial_people - 1) +
    new_supervisor_salary = new_average_salary * initial_people :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l3548_354822


namespace NUMINAMATH_CALUDE_max_value_abc_l3548_354869

theorem max_value_abc (a b c : ℝ) (h : 2 * a + 3 * b + c = 6) :
  ∃ (max : ℝ), max = 9/2 ∧ ∀ (x y z : ℝ), 2 * x + 3 * y + z = 6 → x * y + x * z + y * z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l3548_354869


namespace NUMINAMATH_CALUDE_dvd_rack_sequence_l3548_354841

theorem dvd_rack_sequence (rack : Fin 6 → ℕ) 
  (h1 : rack 0 = 2)
  (h2 : rack 1 = 4)
  (h4 : rack 3 = 16)
  (h5 : rack 4 = 32)
  (h6 : rack 5 = 64)
  (h_double : ∀ i : Fin 5, rack (i.succ) = 2 * rack i) :
  rack 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rack_sequence_l3548_354841


namespace NUMINAMATH_CALUDE_largest_prime_with_square_conditions_l3548_354893

theorem largest_prime_with_square_conditions : 
  ∀ p : ℕ, 
    p.Prime → 
    (∃ x : ℕ, (p + 1) / 2 = x^2) → 
    (∃ y : ℕ, (p^2 + 1) / 2 = y^2) → 
    p ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_with_square_conditions_l3548_354893
